#!/usr/bin/env python3
"""Visual validation for the common LEFT/CENTER/RIGHT field geometry.

This diagnostic intentionally contains no detector code.

It renders one source frame and overlays:
- the strict football-pitch boundary,
- the +0.50 m extension ONLY on the far touchline,
- the non-overlapping logical ownership region of each virtual view,
- a sparse ground grid inside the strict pitch.

The purpose is to validate, with the same image the detector will see, that
renderer + projection + field filter + ownership all agree before any model is
connected.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect.interactive_renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_VIDEO,
    OUTPUT_H,
    OUTPUT_W,
    OVERLAP,
    InteractiveRenderer,
    ViewSpec,
    read_frame,
    rot_x,
    rot_y,
)
from person_detect.projection import (  # noqa: E402
    DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    FieldProjector,
)


DEFAULT_OUT_DIR = ROOT / "person_detect" / "outputs" / "projection_debug"


def world_to_pixel(
    projector: FieldProjector,
    spec: ViewSpec,
    world_x: float,
    world_y: float,
    world_z: float,
) -> tuple[float, float] | None:
    """Project a world point into one virtual perspective view."""
    point = np.array([world_x, world_y, world_z], dtype=np.float64)
    direction_world = point - projector.camera

    transform = (
        projector.presentation_to_world_matrix
        @ rot_y(spec.longitude)
        @ rot_x(spec.latitude)
    )
    world_to_view = np.linalg.inv(transform)

    direction_h = np.array(
        [
            direction_world[0],
            direction_world[1],
            direction_world[2],
            0.0,
        ],
        dtype=np.float64,
    )
    view = (world_to_view @ direction_h)[:3]

    depth = float(view[2])
    if not np.isfinite(view).all() or depth <= 1e-9:
        return None

    tan_v = math.tan(spec.vfov / 2.0)
    aspect = projector.width / projector.height

    normalized_x = float(view[0]) / depth
    normalized_y = float(view[1]) / depth

    u = 0.5 + normalized_x / (2.0 * tan_v * aspect)
    v = 0.5 + normalized_y / (2.0 * tan_v)

    x = u * projector.width - 0.5
    y = v * projector.height - 0.5

    if not np.isfinite([x, y]).all():
        return None

    return float(x), float(y)


def _visible_pixel(
    pixel: tuple[float, float] | None,
    width: int,
    height: int,
    pad: float = 8.0,
) -> tuple[int, int] | None:
    if pixel is None:
        return None

    x, y = pixel
    if x < -pad or x > width - 1 + pad:
        return None
    if y < -pad or y > height - 1 + pad:
        return None

    return (
        int(round(min(max(x, 0.0), width - 1.0))),
        int(round(min(max(y, 0.0), height - 1.0))),
    )


def project_polyline(
    projector: FieldProjector,
    spec: ViewSpec,
    points_xyz: np.ndarray,
) -> list[list[tuple[int, int]]]:
    """Project a world polyline and split it at invisible segments."""
    segments: list[list[tuple[int, int]]] = []
    current: list[tuple[int, int]] = []

    for point in points_xyz:
        pixel = world_to_pixel(
            projector,
            spec,
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )
        visible = _visible_pixel(
            pixel,
            projector.width,
            projector.height,
        )

        if visible is None:
            if len(current) >= 2:
                segments.append(current)
            current = []
            continue

        current.append(visible)

    if len(current) >= 2:
        segments.append(current)

    return segments


def line_points(
    x1: float,
    z1: float,
    x2: float,
    z2: float,
    samples: int = 300,
) -> np.ndarray:
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    return np.column_stack(
        (
            x1 + (x2 - x1) * t,
            np.zeros_like(t),
            z1 + (z2 - z1) * t,
        )
    )


def strict_field_lines(projector: FieldProjector) -> list[np.ndarray]:
    hl = projector.half_length
    hw = projector.half_width
    return [
        line_points(-hl, -hw, hl, -hw),
        line_points(hl, -hw, hl, hw),
        line_points(hl, hw, -hl, hw),
        line_points(-hl, hw, -hl, -hw),
    ]


def far_margin_line(projector: FieldProjector) -> np.ndarray:
    z = projector.far_touchline_z
    if z > 0.0:
        z += projector.far_touchline_margin_m
    else:
        z -= projector.far_touchline_margin_m

    return line_points(
        -projector.half_length,
        z,
        projector.half_length,
        z,
    )


def ground_grid(projector: FieldProjector) -> np.ndarray:
    """Sparse exact world points for an easy visual sanity check."""
    xs = np.linspace(
        -projector.half_length,
        projector.half_length,
        11,
        dtype=np.float64,
    )
    zs = np.linspace(
        -projector.half_width,
        projector.half_width,
        7,
        dtype=np.float64,
    )

    return np.array(
        [[x, 0.0, z] for z in zs for x in xs],
        dtype=np.float64,
    )


def draw_ownership_contour(
    image: np.ndarray,
    ownership_mask: np.ndarray,
) -> None:
    mask = ownership_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    cv2.drawContours(
        image,
        contours,
        -1,
        (255, 0, 255),
        3,
        cv2.LINE_AA,
    )


def draw_world_polyline(
    image: np.ndarray,
    projector: FieldProjector,
    spec: ViewSpec,
    points_xyz: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    for segment in project_polyline(projector, spec, points_xyz):
        cv2.polylines(
            image,
            [np.asarray(segment, dtype=np.int32)],
            isClosed=False,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )


def annotate_view(
    image: np.ndarray,
    view_name: str,
    renderer: InteractiveRenderer,
    projector: FieldProjector,
) -> np.ndarray:
    spec = projector.spec_by_name[view_name]
    out = image.copy()

    # Ownership is visual-only here. Detection filtering will use the projected
    # foot point and FieldProjector.owner_for_world_point().
    draw_ownership_contour(
        out,
        renderer.ownership_masks[view_name],
    )

    # Strict field boundary.
    for line in strict_field_lines(projector):
        draw_world_polyline(
            out,
            projector,
            spec,
            line,
            (255, 255, 0),
            3,
        )

    # The ONLY expanded field edge.
    draw_world_polyline(
        out,
        projector,
        spec,
        far_margin_line(projector),
        (0, 255, 255),
        3,
    )

    # Ground reference points. A point is drawn only when it is visible in the
    # current rendered view; its owner is evaluated from world geometry.
    for world in ground_grid(projector):
        pixel = _visible_pixel(
            world_to_pixel(
                projector,
                spec,
                float(world[0]),
                float(world[1]),
                float(world[2]),
            ),
            projector.width,
            projector.height,
        )
        if pixel is None:
            continue

        owner, _ = projector.owner_for_world_point(
            float(world[0]),
            float(world[2]),
        )
        owned_here = owner == view_name

        cv2.circle(
            out,
            pixel,
            5 if owned_here else 3,
            (0, 255, 0) if owned_here else (96, 96, 96),
            -1,
            cv2.LINE_AA,
        )

    cv2.putText(
        out,
        view_name.upper(),
        (32, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "strict pitch",
        (32, 104),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 0),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        f"far sideline +{projector.far_touchline_margin_m:.2f} m",
        (32, 138),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        out,
        "ownership",
        (32, 172),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )

    return out


def make_preview(
    annotated: dict[str, np.ndarray],
    panel_width: int = 640,
) -> np.ndarray:
    panels = []

    for name in ("left", "center", "right"):
        image = annotated[name]
        scale = panel_width / image.shape[1]
        panel = cv2.resize(
            image,
            (
                panel_width,
                int(round(image.shape[0] * scale)),
            ),
            interpolation=cv2.INTER_AREA,
        )
        panels.append(panel)

    return np.concatenate(panels, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate field projection and ownership visually."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--time",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--width",
        type=int,
        default=OUTPUT_W,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=OUTPUT_H,
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=OVERLAP,
    )
    parser.add_argument(
        "--far-margin",
        type=float,
        default=DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.calibration.exists():
        raise FileNotFoundError(args.calibration)

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    renderer = InteractiveRenderer(
        args.calibration,
        width=args.width,
        height=args.height,
        overlap=args.overlap,
    )
    projector = FieldProjector(
        args.calibration,
        width=args.width,
        height=args.height,
        overlap=args.overlap,
        far_touchline_margin_m=args.far_margin,
    )

    # Strong sanity check: the two independent users of derive_views must agree.
    for renderer_spec, projector_spec in zip(
        renderer.specs,
        projector.specs,
        strict=True,
    ):
        fields = (
            "longitude",
            "latitude",
            "vfov",
            "hfov",
            "owner_min",
            "owner_max",
        )
        for field in fields:
            if not math.isclose(
                getattr(renderer_spec, field),
                getattr(projector_spec, field),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise RuntimeError(
                    f"Renderer/projector disagree for "
                    f"{renderer_spec.name}.{field}"
                )

    frame, frame_info = read_frame(
        args.video,
        args.time,
    )
    views = renderer.render(frame)

    annotated = {
        name: annotate_view(
            image,
            name,
            renderer,
            projector,
        )
        for name, image in views.items()
    }

    for name, image in annotated.items():
        cv2.imwrite(
            str(args.out_dir / f"{name}_geometry.png"),
            image,
        )

    preview = make_preview(annotated)
    preview_path = args.out_dir / "projection_validation.jpg"
    cv2.imwrite(str(preview_path), preview)

    summary = {
        "video": str(args.video),
        "frame": frame_info,
        "renderer": renderer.summary(),
        "projector": projector.summary(),
        "diagnostic": {
            "strict_field_boundary": True,
            "far_touchline_margin_m": args.far_margin,
            "ownership_masks": True,
            "ground_grid": {
                "x_samples": 11,
                "z_samples": 7,
            },
        },
    }

    summary_path = args.out_dir / "projection_validation.json"
    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print(f"Saved: {preview_path}")
    print(f"Saved: {summary_path}")
    print(
        "Far touchline: "
        f"{projector.far_touchline_name} "
        f"(z={projector.far_touchline_z:.3f} m), "
        f"margin={projector.far_touchline_margin_m:.2f} m"
    )


if __name__ == "__main__":
    main()
