#!/usr/bin/env python3
"""High-quality LEFT/CENTER/RIGHT Veo renderer for the new detection pipeline.

Detector-agnostic first stage:
    stacked 2048x2048 dual-lens frame + camera_alignment.veo
        -> three rectilinear perspective views

The rendered views overlap visually so players are not cut at boundaries.
Logical ownership is still not applied to detections here, but V2 computes
and exports exact non-overlapping ownership masks for visual validation.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.reproduce_veo_panorama import (  # noqa: E402
    BLEND_BUBBLE_RADIUS,
    BLEND_DISTANCE,
    PROJECTION_BUBBLE_RADIUS,
    project_world_target_to_lens,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402

MATCH_ID = "match_629890153a11"
DEFAULT_VIDEO = ROOT / "person_detect" / "clips" / MATCH_ID / "clip_12m36s_90s.mp4"
DEFAULT_CALIBRATION = ROOT / "person_detect" / "clips" / MATCH_ID / "camera_alignment.veo"
DEFAULT_OUT = ROOT / "person_detect" / "outputs" / "interactive_view" / "diagnostics" / "interactive_renderer"

SOURCE_W = 2048
SOURCE_H = 2048
OUTPUT_W = 2048
OUTPUT_H = 1152
OVERLAP = 0.15          # fraction of one logical sector added on EACH side
H_MARGIN_DEG = 1.5
V_MARGIN_DEG = 2.0
CONTEXT_HEIGHT_M = 2.5  # include complete players around pitch limits
EDGE_INSET_DEG = 0.35   # keep outer views just inside Veo presentation limits


@dataclass(frozen=True)
class ViewSpec:
    name: str
    longitude: float
    latitude: float
    vfov: float
    hfov: float
    owner_min: float
    owner_max: float
    rendered_min: float
    rendered_max: float

    def json(self) -> dict:
        d = asdict(self)
        d.update({
            "longitude_deg": math.degrees(self.longitude),
            "latitude_deg": math.degrees(self.latitude),
            "vfov_deg": math.degrees(self.vfov),
            "hfov_deg": math.degrees(self.hfov),
            "owner_min_deg": math.degrees(self.owner_min),
            "owner_max_deg": math.degrees(self.owner_max),
            "rendered_min_deg": math.degrees(self.rendered_min),
            "rendered_max_deg": math.degrees(self.rendered_max),
        })
        return d


@dataclass(frozen=True)
class ViewMap:
    spec: ViewSpec
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    blend: np.ndarray


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)


def presentation_to_world(metadata) -> np.ndarray:
    # Same validated convention as geometry/reproduce_veo_panorama.py.
    return metadata.camera2world @ np.linalg.inv(metadata.camera2presentation)


def camera_world(metadata) -> np.ndarray:
    return (metadata.camera2world @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]


def unwrap(values: np.ndarray, ref: float) -> np.ndarray:
    return ref + np.arctan2(np.sin(values - ref), np.cos(values - ref))


def perimeter(length: float, width: float, n: int = 96) -> np.ndarray:
    hl, hw = length / 2.0, width / 2.0
    t = np.linspace(0.0, 1.0, n)
    return np.concatenate([
        np.column_stack((-hl + 2 * hl * t, np.full_like(t, hw))),
        np.column_stack((np.full_like(t, hl), hw - 2 * hw * t)),
        np.column_stack((hl - 2 * hl * t, np.full_like(t, -hw))),
        np.column_stack((np.full_like(t, -hl), -hw + 2 * hw * t)),
    ])


def angles_for_world_points(metadata, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dirs = points - camera_world(metadata)[None, :]
    dirs_h = np.column_stack((dirs, np.zeros(len(dirs))))
    world_to_pres = np.linalg.inv(presentation_to_world(metadata))
    p = dirs_h @ world_to_pres.T
    lon = np.arctan2(p[:, 0], p[:, 2])
    lat = np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2]))
    return lon, lat


def derive_views(metadata, width: int, height: int, overlap: float) -> tuple[list[ViewSpec], dict]:
    if not 0.0 <= overlap < 0.5:
        raise ValueError("overlap must be in [0, 0.5)")

    xz = perimeter(metadata.field_length, metadata.field_width)
    cam = camera_world(metadata)
    # Ground Y is 0. Above-ground has the same sign as the camera height.
    top_y = math.copysign(CONTEXT_HEIGHT_M, cam[1])
    ground = np.column_stack((xz[:, 0], np.zeros(len(xz)), xz[:, 1]))
    tops = ground.copy()
    tops[:, 1] = top_y
    lon, lat = angles_for_world_points(metadata, np.concatenate((ground, tops)))

    va = metadata.va_presentation
    ref = 0.5 * (float(va["left"]) + float(va["right"]))
    lon = unwrap(lon, ref)

    lon_min = float(lon.min()) - math.radians(H_MARGIN_DEG)
    lon_max = float(lon.max()) + math.radians(H_MARGIN_DEG)
    lat_min = float(lat.min()) - math.radians(V_MARGIN_DEG)
    lat_max = float(lat.max()) + math.radians(V_MARGIN_DEG)

    va_lon = unwrap(np.array([float(va["left"]), float(va["right"])]), ref)
    lon_min = max(lon_min, float(va_lon.min()))
    lon_max = min(lon_max, float(va_lon.max()))
    lat_min = max(lat_min, min(float(va["bottom"]), float(va["top"])))
    lat_max = min(lat_max, max(float(va["bottom"]), float(va["top"])))

    if lon_max <= lon_min or lat_max <= lat_min:
        raise RuntimeError("Could not derive a valid field angular envelope")

    sector = (lon_max - lon_min) / 3.0
    requested_hfov = sector * (1.0 + 2.0 * overlap)
    aspect = width / height
    vfov = max(
        2.0 * math.atan(math.tan(requested_hfov / 2.0) / aspect),
        lat_max - lat_min,
    )
    if vfov >= math.radians(100.0):
        raise RuntimeError(f"Derived VFOV is too wide: {math.degrees(vfov):.2f} deg")
    hfov = 2.0 * math.atan(math.tan(vfov / 2.0) * aspect)
    pitch = 0.5 * (lat_min + lat_max)

    # Keep the rendered FOV inside Veo's declared presentation range.
    # LEFT/RIGHT may otherwise waste detector pixels on invalid black/blue areas.
    edge_inset = math.radians(EDGE_INSET_DEG)
    safe_va_min = float(va_lon.min()) + edge_inset
    safe_va_max = float(va_lon.max()) - edge_inset
    half_hfov = hfov / 2.0

    specs = []
    for i, name in enumerate(("left", "center", "right")):
        owner_min = lon_min + i * sector
        owner_max = owner_min + sector
        nominal_center = 0.5 * (owner_min + owner_max)

        # A valid center must both:
        #   1) keep the full logical owner sector visible, and
        #   2) keep the rendered FOV inside Veo's usable presentation limits.
        min_center = max(
            owner_max - half_hfov,
            safe_va_min + half_hfov,
        )
        max_center = min(
            owner_min + half_hfov,
            safe_va_max - half_hfov,
        )

        if min_center <= max_center:
            longitude = min(max(nominal_center, min_center), max_center)
        else:
            # If the two constraints cannot both be met exactly, preserve the
            # whole owner sector first and move as far inward as possible.
            owner_min_center = owner_max - half_hfov
            owner_max_center = owner_min + half_hfov
            inward_target = (
                safe_va_min + half_hfov
                if name == "left"
                else safe_va_max - half_hfov
                if name == "right"
                else nominal_center
            )
            longitude = min(
                max(inward_target, owner_min_center),
                owner_max_center,
            )

        specs.append(ViewSpec(
            name=name,
            longitude=longitude,
            latitude=pitch,
            vfov=vfov,
            hfov=hfov,
            owner_min=owner_min,
            owner_max=owner_max,
            rendered_min=longitude - half_hfov,
            rendered_max=longitude + half_hfov,
        ))

    info = {
        "field_lon_min_deg": math.degrees(lon_min),
        "field_lon_max_deg": math.degrees(lon_max),
        "field_lat_min_deg": math.degrees(lat_min),
        "field_lat_max_deg": math.degrees(lat_max),
        "logical_sector_deg": math.degrees(sector),
        "visual_overlap_fraction_each_side": overlap,
        "edge_inset_deg": EDGE_INSET_DEG,
        "veo_safe_lon_min_deg": math.degrees(safe_va_min),
        "veo_safe_lon_max_deg": math.degrees(safe_va_max),
    }
    return specs, info



def ownership_mask(spec: ViewSpec, width: int, height: int) -> np.ndarray:
    """Exact per-pixel logical ownership for the rendered perspective view.

    The rendered views overlap visually, but these masks do not overlap in
    presentation longitude. They are diagnostic in V2; the future detection
    stage will apply ownership after projecting a bbox foot point.
    """
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    u, v = np.meshgrid(x, y)

    tan_v = math.tan(spec.vfov / 2.0)
    aspect = width / height
    rays = np.stack((
        (2.0 * u - 1.0) * tan_v * aspect,
        (2.0 * v - 1.0) * tan_v,
        np.ones_like(u),
    ), axis=-1).reshape(-1, 3)

    transform = (rot_y(spec.longitude) @ rot_x(spec.latitude))[:3, :3]
    p = rays @ transform.T
    lon = np.arctan2(p[:, 0], p[:, 2])
    lon = spec.longitude + np.arctan2(
        np.sin(lon - spec.longitude),
        np.cos(lon - spec.longitude),
    )

    # LEFT/CENTER use a half-open upper edge. RIGHT includes the final edge.
    if spec.name == "right":
        owned = (lon >= spec.owner_min) & (lon <= spec.owner_max)
    else:
        owned = (lon >= spec.owner_min) & (lon < spec.owner_max)
    return owned.reshape(height, width)


def draw_ownership_boundaries(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Draw the non-overlapping ownership limits without hiding the image."""
    out = image.copy()
    m = (mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, contours, -1, (255, 0, 255), 3, cv2.LINE_AA)
    return out


def virtual_geometry(metadata, spec: ViewSpec, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(width, dtype=np.float64) + 0.5) / width
    y = (np.arange(height, dtype=np.float64) + 0.5) / height
    u, v = np.meshgrid(x, y)

    tan_v = math.tan(spec.vfov / 2.0)
    aspect = width / height
    rays = np.column_stack((
        ((2.0 * u - 1.0) * tan_v * aspect).ravel(),
        ((2.0 * v - 1.0) * tan_v).ravel(),
        np.ones(width * height),
        np.zeros(width * height),
    ))

    transform = presentation_to_world(metadata) @ rot_y(spec.longitude) @ rot_x(spec.latitude)
    dirs_h = rays @ transform.T
    dirs = dirs_h[:, :3]
    cam = camera_world(metadata)

    dy = dirs[:, 1]
    up_down = (np.sign(dy) + 1.0) / 2.0
    safe_dy = np.where(np.abs(dy) > 1e-12, dy, 1e-12)
    ground_distance = -cam[1] / safe_dy
    distance = up_down * ground_distance + (1.0 - up_down) * PROJECTION_BUBBLE_RADIUS

    projection = dirs * distance[:, None]
    target = np.column_stack((cam[None, :] + projection, np.ones(len(projection))))

    # Preserve Veo's original left/right lens blending rule.
    projection_h = np.column_stack((projection, np.zeros(len(projection))))
    world_to_pres = np.linalg.inv(presentation_to_world(metadata))
    blend_target = projection_h @ world_to_pres.T
    blend_target[:, 3] = 1.0
    projection_len = np.linalg.norm(projection_h, axis=1)
    mask = (projection_len > BLEND_BUBBLE_RADIUS) | (projection[:, 1] < 0.0)
    if np.any(mask):
        lengths = np.maximum(np.linalg.norm(blend_target[mask], axis=1), 1e-12)
        blend_target[mask] = blend_target[mask] / lengths[:, None] * BLEND_BUBBLE_RADIUS
    blend = np.clip(blend_target[:, 0] / BLEND_DISTANCE + 0.5, 0.0, 1.0)
    return target, blend.reshape(height, width)


def build_map(metadata, spec: ViewSpec, sw: int, sh: int, ow: int, oh: int) -> ViewMap:
    target, blend = virtual_geometry(metadata, spec, ow, oh)
    lx, ly, lv = project_world_target_to_lens(target, metadata.left, metadata.camera2world, sw, sh, True)
    rx, ry, rv = project_world_target_to_lens(target, metadata.right, metadata.camera2world, sw, sh, False)
    shape = (oh, ow)
    return ViewMap(spec, lx.reshape(shape), ly.reshape(shape), rx.reshape(shape), ry.reshape(shape),
                   lv.reshape(shape), rv.reshape(shape), blend)


class InteractiveRenderer:
    def __init__(self, calibration: Path, width: int = OUTPUT_W, height: int = OUTPUT_H,
                 overlap: float = OVERLAP):
        self.metadata = load_veo_metadata(calibration)
        self.width, self.height = width, height
        self.specs, self.envelope = derive_views(self.metadata, width, height, overlap)
        self.maps = {
            s.name: build_map(self.metadata, s, SOURCE_W, SOURCE_H, width, height)
            for s in self.specs
        }
        self.ownership_masks = {
            s.name: ownership_mask(s, width, height)
            for s in self.specs
        }

    @staticmethod
    def _one(frame: np.ndarray, m: ViewMap) -> np.ndarray:
        left = cv2.remap(frame, m.left_x, m.left_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        right = cv2.remap(frame, m.right_x, m.right_y, cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        b = m.blend[..., None].astype(np.float32)
        out = left.astype(np.float32) * (1.0 - b) + right.astype(np.float32) * b

        # Prevent black seams where only one physical lens has a valid sample.
        only_left = m.left_valid & ~m.right_valid
        only_right = m.right_valid & ~m.left_valid
        neither = ~m.left_valid & ~m.right_valid
        out[only_left] = left[only_left]
        out[only_right] = right[only_right]
        out[neither] = 0
        return np.clip(out, 0, 255).astype(np.uint8)

    def render(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        h, w = frame.shape[:2]
        if (w, h) != (SOURCE_W, SOURCE_H):
            raise ValueError(f"Expected {SOURCE_W}x{SOURCE_H}, got {w}x{h}")
        return {name: self._one(frame, m) for name, m in self.maps.items()}

    def summary(self) -> dict:
        views = []
        for spec in self.specs:
            m = self.maps[spec.name]
            info = spec.json()
            info.update({
                "left_valid_fraction": float(m.left_valid.mean()),
                "right_valid_fraction": float(m.right_valid.mean()),
                "either_lens_valid_fraction": float((m.left_valid | m.right_valid).mean()),
                "both_lenses_valid_fraction": float((m.left_valid & m.right_valid).mean()),
            })
            views.append(info)
        return {
            "source_size": [SOURCE_W, SOURCE_H],
            "output_size_per_view": [self.width, self.height],
            "field_length_m": float(self.metadata.field_length),
            "field_width_m": float(self.metadata.field_width),
            "field_envelope": self.envelope,
            "views": views,
        }


def read_frame(video: Path, time_s: float) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target = max(0, int(round(time_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, frame = cap.read()
    actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame near {time_s:.3f}s")
    if (w, h) != (SOURCE_W, SOURCE_H):
        raise RuntimeError(f"V1 expects {SOURCE_W}x{SOURCE_H}; video is {w}x{h}")
    return frame, {"fps": fps, "requested_time_s": time_s, "actual_frame": actual,
                   "actual_time_s": actual / fps if fps > 0 else None}


def preview(views: dict[str, np.ndarray], panel_width: int = 640,
            ownership_masks: dict[str, np.ndarray] | None = None) -> np.ndarray:
    panels = []
    for name in ("left", "center", "right"):
        img = views[name]
        if ownership_masks is not None:
            img = draw_ownership_boundaries(img, ownership_masks[name])
        scale = panel_width / img.shape[1]
        p = cv2.resize(img, (panel_width, int(round(img.shape[0] * scale))), interpolation=cv2.INTER_AREA)
        cv2.putText(p, name.upper(), (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(p)
    return np.concatenate(panels, axis=1)


def render_seconds(renderer: InteractiveRenderer, video: Path, out: Path,
                   start_s: float, duration_s: float) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        raise RuntimeError("Invalid source FPS")
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(start_s * fps))))
    total = max(1, int(round(duration_s * fps)))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writers = {
        name: cv2.VideoWriter(str(out / f"{name}.mp4"), fourcc, fps,
                              (renderer.width, renderer.height))
        for name in ("left", "center", "right")
    }
    if not all(w.isOpened() for w in writers.values()):
        cap.release()
        for w in writers.values():
            w.release()
        raise RuntimeError("Could not open one or more output video writers")

    done = 0
    try:
        while done < total:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            views = renderer.render(frame)
            for name, writer in writers.items():
                writer.write(views[name])
            done += 1
            if done == 1 or done % max(1, int(round(fps))) == 0:
                print(f"Rendered {done}/{total} frames", flush=True)
    finally:
        cap.release()
        for w in writers.values():
            w.release()
    return {"fps": fps, "requested_frames": total, "rendered_frames": done,
            "rendered_duration_s": done / fps}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Render Veo LEFT/CENTER/RIGHT perspective views")
    p.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    p.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    p.add_argument("--time", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=0.0,
                   help="If >0, render that many seconds to three MP4 files")
    p.add_argument("--width", type=int, default=OUTPUT_W)
    p.add_argument("--height", type=int, default=OUTPUT_H)
    p.add_argument("--overlap", type=float, default=OVERLAP)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.calibration.exists():
        raise FileNotFoundError(args.calibration)

    print("Building static perspective remap grids...")
    renderer = InteractiveRenderer(args.calibration, args.width, args.height, args.overlap)
    summary = renderer.summary()
    for v in summary["views"]:
        print(f"{v['name'].upper():>6}: yaw={v['longitude_deg']:.2f} deg, "
              f"pitch={v['latitude_deg']:.2f} deg, HFOV={v['hfov_deg']:.2f} deg, "
              f"VFOV={v['vfov_deg']:.2f} deg, valid={100*v['either_lens_valid_fraction']:.2f}%")

    run = {"video": str(args.video), "calibration": str(args.calibration), "renderer": summary}
    if args.duration > 0:
        run["clip"] = render_seconds(renderer, args.video, args.out_dir, args.time, args.duration)
        print(f"Saved LEFT/CENTER/RIGHT videos in {args.out_dir}")
    else:
        frame, frame_info = read_frame(args.video, args.time)
        views = renderer.render(frame)
        for name, img in views.items():
            cv2.imwrite(str(args.out_dir / f"{name}.png"), img)
        cv2.imwrite(str(args.out_dir / "preview_left_center_right.jpg"), preview(views))
        cv2.imwrite(
            str(args.out_dir / "preview_ownership.jpg"),
            preview(views, ownership_masks=renderer.ownership_masks),
        )
        for name, mask in renderer.ownership_masks.items():
            cv2.imwrite(
                str(args.out_dir / f"{name}_ownership_mask.png"),
                (mask.astype(np.uint8) * 255),
            )
        run["frame"] = frame_info
        print(f"Saved diagnostic images in {args.out_dir}")

    with (args.out_dir / "renderer_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run, f, indent=2)


if __name__ == "__main__":
    main()
