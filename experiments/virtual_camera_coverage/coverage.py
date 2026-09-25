#!/usr/bin/env python3
"""Render a six-view field coverage experiment directly from stacked Veo frames.

This module intentionally imports only the established Veo projection/remap
primitives.  It does not create or read a panorama.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.reproduce_veo_panorama import (  # noqa: E402
    BLEND_BUBBLE_RADIUS,
    BLEND_DISTANCE,
    PROJECTION_BUBBLE_RADIUS,
    project_world_target_to_lens,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.views.interactive_view.renderer import (  # noqa: E402
    SOURCE_H,
    SOURCE_W,
    camera_world,
    presentation_to_world,
    rot_x,
    rot_y,
)


PLAYER_CONTEXT_M = 2.5
DEFAULT_GUARD_M = 0.75
DEFAULT_NEAR_CONTEXT_M = 1.0
DEFAULT_OUT = Path("experiments/virtual_camera_coverage/output")
EPS = 1e-10


@dataclass(frozen=True)
class FieldBounds:
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    def corners(self, y: float = 0.0) -> np.ndarray:
        return np.array([
            [self.x_min, y, self.z_min], [self.x_max, y, self.z_min],
            [self.x_max, y, self.z_max], [self.x_min, y, self.z_max],
        ], dtype=np.float64)


@dataclass(frozen=True)
class CameraSpec:
    name: str
    column: int
    row: int
    ownership: FieldBounds
    requested: FieldBounds
    rendered: FieldBounds
    longitude: float
    latitude: float
    hfov: float
    vfov: float
    context_clamped: bool
    context_clamp_reason: str | None

    def json(self) -> dict:
        d = asdict(self)
        for key in ("longitude", "latitude", "hfov", "vfov"):
            d[f"{key}_deg"] = math.degrees(getattr(self, key))
        return d


@dataclass
class CameraMap:
    spec: CameraSpec
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    blend: np.ndarray


def logical_cells(length: float, width: float) -> list[tuple[str, int, int, FieldBounds]]:
    """Return six half-open physical ownership cells, in row-major order."""
    if length <= 0 or width <= 0:
        raise ValueError("field dimensions must be positive")
    x_edges = np.linspace(-length / 2, length / 2, 4)
    z_edges = np.linspace(-width / 2, width / 2, 3)
    result = []
    for row in range(2):
        for col in range(3):
            result.append((
                f"r{row + 1}c{col + 1}", col, row,
                FieldBounds(float(x_edges[col]), float(x_edges[col + 1]),
                            float(z_edges[row]), float(z_edges[row + 1])),
            ))
    return result


def _near_and_far_z(metadata) -> tuple[float, float]:
    cam_z = float(camera_world(metadata)[2])
    half = float(metadata.field_width) / 2
    if abs(cam_z - half) < abs(cam_z + half):
        return half, -half
    return -half, half


def _expand(bounds: FieldBounds, field_length: float, field_width: float,
            guard: float, near_z: float, near_context: float,
            outer_context: float) -> FieldBounds:
    """Expand a cell only at the relevant physical outer edges.

    Internal edges receive guard only.  This keeps each camera local to its
    cell instead of accidentally giving every camera the whole pitch width.
    """
    half_l, half_w = field_length / 2, field_width / 2
    x_left = outer_context if abs(bounds.x_min + half_l) < EPS else guard
    x_right = outer_context if abs(bounds.x_max - half_l) < EPS else guard
    near_is_min = near_z < 0
    z_left_outer = (near_is_min and abs(bounds.z_min + half_w) < EPS) or (not near_is_min and abs(bounds.z_min - half_w) < EPS)
    z_right_outer = (near_is_min and abs(bounds.z_max - half_w) < EPS) or (not near_is_min and abs(bounds.z_max + half_w) < EPS)
    z_min_extra = near_context if z_left_outer and near_is_min else outer_context if z_left_outer else guard
    z_max_extra = near_context if z_right_outer and not near_is_min else outer_context if z_right_outer else guard
    if not near_is_min:
        z_min_extra = outer_context if z_left_outer else guard
        z_max_extra = near_context if z_right_outer else guard
    return FieldBounds(bounds.x_min - x_left, bounds.x_max + x_right,
                       bounds.z_min - z_min_extra, bounds.z_max + z_max_extra)


def _region_points(bounds: FieldBounds) -> np.ndarray:
    ground = bounds.corners(0.0)
    top = bounds.corners(PLAYER_CONTEXT_M)
    # Corners plus edge samples make the angular envelope stable for the
    # perspective transform even when a cell is close to the camera.
    points = [ground, top]
    for y in (0.0, PLAYER_CONTEXT_M):
        t = np.linspace(0.0, 1.0, 17)
        points.append(np.column_stack((
            np.r_[bounds.x_min + (bounds.x_max - bounds.x_min) * t,
                  np.full_like(t, bounds.x_max),
                  bounds.x_max - (bounds.x_max - bounds.x_min) * t,
                  np.full_like(t, bounds.x_min)],
            np.r_[np.full_like(t, bounds.z_min),
                  bounds.z_min + (bounds.z_max - bounds.z_min) * t,
                  np.full_like(t, bounds.z_max),
                  bounds.z_max - (bounds.z_max - bounds.z_min) * t],
            np.full(68, y),
        )))
    return np.concatenate(points)


def _angles(metadata, points: np.ndarray, reference: float) -> tuple[np.ndarray, np.ndarray]:
    cam = camera_world(metadata)
    dirs = points - cam[None, :]
    p = np.column_stack((dirs, np.zeros(len(dirs)))) @ np.linalg.inv(presentation_to_world(metadata)).T
    lon = reference + np.arctan2(np.sin(np.arctan2(p[:, 0], p[:, 2]) - reference),
                                 np.cos(np.arctan2(p[:, 0], p[:, 2]) - reference))
    lat = np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2]))
    return lon, lat


def _safe_rendered_bounds(metadata, requested: FieldBounds, reference: float,
                          safe_lon: tuple[float, float], safe_lat: tuple[float, float]) -> tuple[FieldBounds, bool, str | None]:
    """Reduce only the context expansion when it exceeds calibrated VA limits."""
    points = _region_points(requested)
    lon, lat = _angles(metadata, points, reference)
    if (lon.min() >= safe_lon[0] and lon.max() <= safe_lon[1] and
            lat.min() >= safe_lat[0] and lat.max() <= safe_lat[1]):
        return requested, False, None
    # The physical field itself is expected to be valid. Keep its exact bounds
    # and binary-search a scale for the extra context until all samples fit.
    field = FieldBounds(
        max(-metadata.field_length / 2, requested.x_min),
        min(metadata.field_length / 2, requested.x_max),
        max(-metadata.field_width / 2, requested.z_min),
        min(metadata.field_width / 2, requested.z_max),
    )
    lo, hi = 0.0, 1.0
    for _ in range(45):
        scale = (lo + hi) / 2
        candidate = FieldBounds(
            field.x_min - (field.x_min - requested.x_min) * scale,
            field.x_max + (requested.x_max - field.x_max) * scale,
            field.z_min - (field.z_min - requested.z_min) * scale,
            field.z_max + (requested.z_max - field.z_max) * scale,
        )
        c_lon, c_lat = _angles(metadata, _region_points(candidate), reference)
        if (c_lon.min() >= safe_lon[0] and c_lon.max() <= safe_lon[1] and
                c_lat.min() >= safe_lat[0] and c_lat.max() <= safe_lat[1]):
            lo = scale
        else:
            hi = scale
    rendered = FieldBounds(
        field.x_min - (field.x_min - requested.x_min) * lo,
        field.x_max + (requested.x_max - field.x_max) * lo,
        field.z_min - (field.z_min - requested.z_min) * lo,
        field.z_max + (requested.z_max - field.z_max) * lo,
    )
    return rendered, True, "requested context exceeded calibrated presentation validity"


def _derive_spec(metadata, name: str, col: int, row: int, ownership: FieldBounds,
                 requested: FieldBounds, rendered: FieldBounds, clamped: bool,
                 reason: str | None, width: int, height: int) -> CameraSpec:
    va = metadata.va_presentation
    ref = 0.5 * (float(va["left"]) + float(va["right"]))
    va_lon = np.unwrap(np.array([float(va["left"]), float(va["right"])]))
    # Unwrap relative to the same reference used for points.
    va_lon = ref + np.arctan2(np.sin(va_lon - ref), np.cos(va_lon - ref))
    safe_lon = (float(va_lon.min()) + math.radians(0.35), float(va_lon.max()) - math.radians(0.35))
    safe_lat = (min(float(va["bottom"]), float(va["top"])) + math.radians(0.35),
                max(float(va["bottom"]), float(va["top"])) - math.radians(0.35))
    lon, lat = _angles(metadata, _region_points(rendered), ref)
    lon_min, lon_max = max(lon.min(), safe_lon[0]), min(lon.max(), safe_lon[1])
    lat_min, lat_max = max(lat.min(), safe_lat[0]), min(lat.max(), safe_lat[1])
    if lon_max <= lon_min or lat_max <= lat_min:
        raise RuntimeError(f"{name}: region has no valid calibrated presentation envelope")
    center_lon, center_lat = (lon_min + lon_max) / 2, (lat_min + lat_max) / 2
    aspect = width / height
    hfov = max(lon_max - lon_min, 2 * math.atan(math.tan((lat_max - lat_min) / 2) * aspect))
    vfov = 2 * math.atan(math.tan(hfov / 2) / aspect)
    if not all(np.isfinite((hfov, vfov))) or not (0 < hfov < math.pi and 0 < vfov < math.pi):
        raise RuntimeError(f"{name}: invalid derived FOV")
    return CameraSpec(name, col, row, ownership, requested, rendered, center_lon,
                      center_lat, hfov, vfov, clamped, reason)


def derive_specs(metadata, width: int, height: int, guard_margin_m: float,
                 near_context_m: float) -> list[CameraSpec]:
    if guard_margin_m < 0 or near_context_m < 0:
        raise ValueError("guard-margin-m and near-context-m must be non-negative")
    near_z, far_z = _near_and_far_z(metadata)
    specs = []
    for name, col, row, ownership in logical_cells(metadata.field_length, metadata.field_width):
        tight = FieldBounds(ownership.x_min - guard_margin_m, ownership.x_max + guard_margin_m,
                            ownership.z_min - guard_margin_m, ownership.z_max + guard_margin_m)
        specs.append(_derive_spec(metadata, f"tight_{name}", col, row, ownership, tight,
                                  tight, False, None, width, height))
        requested = _expand(ownership, metadata.field_length, metadata.field_width,
                            guard_margin_m, near_z, near_context_m, 0.75)
        va = metadata.va_presentation
        ref = 0.5 * (float(va["left"]) + float(va["right"]))
        va_lon = ref + np.arctan2(np.sin(np.array([va["left"], va["right"]]) - ref),
                                  np.cos(np.array([va["left"], va["right"]]) - ref))
        safe_lon = (float(va_lon.min()) + math.radians(0.35), float(va_lon.max()) - math.radians(0.35))
        safe_lat = (min(va["bottom"], va["top"]) + math.radians(0.35), max(va["bottom"], va["top"]) - math.radians(0.35))
        rendered, clamped, reason = _safe_rendered_bounds(metadata, requested, ref, safe_lon, safe_lat)
        specs.append(_derive_spec(metadata, f"context_{name}", col, row, ownership, requested,
                                  rendered, clamped, reason, width, height))
    return specs


def _virtual_geometry(metadata, spec: CameraSpec, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    x = (np.arange(width) + 0.5) / width
    y = (np.arange(height) + 0.5) / height
    u, v = np.meshgrid(x, y)
    tan_v = math.tan(spec.vfov / 2)
    rays = np.column_stack((((2 * u - 1) * tan_v * width / height).ravel(),
                            ((2 * v - 1) * tan_v).ravel(),
                            np.ones(width * height), np.zeros(width * height)))
    transform = presentation_to_world(metadata) @ rot_y(spec.longitude) @ rot_x(spec.latitude)
    dirs = (rays @ transform.T)[:, :3]
    cam = camera_world(metadata)
    dy = dirs[:, 1]
    ground = -cam[1] / np.where(np.abs(dy) > EPS, dy, EPS)
    distance = np.where(dy < 0, ground, PROJECTION_BUBBLE_RADIUS)
    projection = dirs * distance[:, None]
    target = np.column_stack((cam[None, :] + projection, np.ones(len(projection))))
    ph = np.column_stack((projection, np.zeros(len(projection))))
    blend_target = ph @ np.linalg.inv(presentation_to_world(metadata)).T
    blend_target[:, 3] = 1
    mask = (np.linalg.norm(projection, axis=1) > BLEND_BUBBLE_RADIUS) | (projection[:, 1] < 0)
    if np.any(mask):
        lengths = np.maximum(np.linalg.norm(blend_target[mask], axis=1), EPS)
        blend_target[mask] *= (BLEND_BUBBLE_RADIUS / lengths)[:, None]
    return target, np.clip(blend_target[:, 0] / BLEND_DISTANCE + 0.5, 0, 1).reshape(height, width)


def build_map(metadata, spec: CameraSpec, width: int, height: int) -> CameraMap:
    target, blend = _virtual_geometry(metadata, spec, width, height)
    lx, ly, lv = project_world_target_to_lens(target, metadata.left, metadata.camera2world, SOURCE_W, SOURCE_H, True)
    rx, ry, rv = project_world_target_to_lens(target, metadata.right, metadata.camera2world, SOURCE_W, SOURCE_H, False)
    shape = (height, width)
    return CameraMap(spec, lx.reshape(shape), ly.reshape(shape), rx.reshape(shape), ry.reshape(shape),
                     lv.reshape(shape), rv.reshape(shape), blend)


def render_map(frame: np.ndarray, m: CameraMap) -> np.ndarray:
    left = cv2.remap(frame, m.left_x, m.left_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right = cv2.remap(frame, m.right_x, m.right_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    b = m.blend[..., None].astype(np.float32)
    out = left.astype(np.float32) * (1 - b) + right.astype(np.float32) * b
    only_left, only_right = m.left_valid & ~m.right_valid, m.right_valid & ~m.left_valid
    out[only_left], out[only_right] = left[only_left], right[only_right]
    out[~(m.left_valid | m.right_valid)] = 0
    return np.clip(out, 0, 255).astype(np.uint8)


def read_frame(video: Path, time_s: float) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    requested = max(0, int(round(time_s * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, requested)
    ok, frame = cap.read()
    actual = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame near {time_s:.3f}s")
    if frame.shape[1:] != (SOURCE_W, SOURCE_H, 3):
        raise RuntimeError(f"Expected {SOURCE_W}x{SOURCE_H} source frame, got {frame.shape[1]}x{frame.shape[0]}")
    return frame, {"fps": fps, "requested_time_s": time_s, "requested_frame": requested,
                   "actual_frame": actual, "actual_time_s": actual / fps if fps else None}


def _label(panel: np.ndarray, text: str) -> np.ndarray:
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(panel, text, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def contact_sheet(images: dict[str, np.ndarray], maps: dict[str, CameraMap], width: int) -> np.ndarray:
    sections = []
    for mode in ("tight", "context"):
        rows = []
        for row in range(2):
            row_panels = []
            for col in range(3):
                name = f"{mode}_r{row + 1}c{col + 1}"
                m = maps[name]
                panel = cv2.resize(images[name], (width, round(images[name].shape[0] * width / images[name].shape[1])), interpolation=cv2.INTER_AREA)
                either = (m.left_valid | m.right_valid).mean()
                row_panels.append(_label(panel, f"{name}  HFOV {math.degrees(m.spec.hfov):.1f}  VFOV {math.degrees(m.spec.vfov):.1f}  valid {100*either:.1f}%"))
            rows.append(np.concatenate(row_panels, axis=1))
        title = np.zeros((38, width * 3, 3), dtype=np.uint8)
        cv2.putText(title, mode.upper(), (12, 27), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2, cv2.LINE_AA)
        sections.extend([title, *rows])
    return np.concatenate(sections, axis=0)


def _bounds_json(bounds: FieldBounds) -> dict:
    return asdict(bounds)


def run(args: argparse.Namespace) -> None:
    metadata = load_veo_metadata(args.calibration)
    specs = derive_specs(metadata, args.width, args.height, args.guard_margin_m, args.near_context_m)
    maps = {spec.name: build_map(metadata, spec, args.width, args.height) for spec in specs}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"video": str(args.video), "calibration": str(args.calibration),
               "field_length_m": metadata.field_length, "field_width_m": metadata.field_width,
               "guard_margin_m": args.guard_margin_m, "near_context_m": args.near_context_m,
               "player_context_m": PLAYER_CONTEXT_M, "frames": []}
    for time_s in args.times:
        frame, frame_info = read_frame(args.video, time_s)
        images = {name: render_map(frame, m) for name, m in maps.items()}
        frame_dir = args.out_dir / f"frame_{frame_info['actual_frame']:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        for name, image in images.items():
            cv2.imwrite(str(frame_dir / f"{name}.png"), image)
        cv2.imwrite(str(frame_dir / "comparison.jpg"), contact_sheet(images, maps, args.panel_width))
        virtual = []
        for spec in specs:
            m = maps[spec.name]
            virtual.append({**spec.json(), "ownership": _bounds_json(spec.ownership),
                            "requested": _bounds_json(spec.requested), "rendered": _bounds_json(spec.rendered),
                            "left_valid_fraction": float(m.left_valid.mean()),
                            "right_valid_fraction": float(m.right_valid.mean()),
                            "either_lens_valid_fraction": float((m.left_valid | m.right_valid).mean()),
                            "both_lenses_valid_fraction": float((m.left_valid & m.right_valid).mean())})
        summary["frames"].append({"source": frame_info, "cameras": virtual,
                                   "comparison": str(frame_dir / "comparison.jpg")})
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--times", type=float, nargs="+", required=True, help="Exactly five source timestamps in seconds")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--guard-margin-m", type=float, default=DEFAULT_GUARD_M)
    p.add_argument("--near-context-m", type=float, default=DEFAULT_NEAR_CONTEXT_M)
    p.add_argument("--panel-width", type=int, default=480)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    if len(args.times) != 5:
        p.error("--times must contain exactly five timestamps")
    if args.width <= 0 or args.height <= 0:
        p.error("--width and --height must be positive")
    return args


if __name__ == "__main__":
    run(parse_args())
