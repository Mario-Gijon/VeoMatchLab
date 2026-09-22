#!/usr/bin/env python3
"""Visual validation for the physical two-lens Veo geometry.

No detector is used.

Outputs:
    left_geometry.png
    right_geometry.png
    ownership_map.png
    two_lens_validation.jpg
    projection_validation.json

The overlays answer four questions before any detector is connected:
  1. Do the calibrated pitch lines land on the real raw-lens field lines?
  2. Does the +0.50 m far-touchline margin land just outside the far line?
  3. Do both lenses together cover the whole pitch?
  4. Is ownership unique and sensible through the overlap?
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.reproduce_veo_panorama import (  # noqa: E402
    project_world_target_to_lens,
)
from person_detect.views.two_lens_view.renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_OUT,
    DEFAULT_VIDEO,
    LENS_H,
    LENS_W,
    STACKED_H,
    STACKED_W,
    TwoLensRenderer,
    read_frame,
)
from person_detect.views.two_lens_view.projection import (  # noqa: E402
    DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    TwoLensProjector,
)


def world_to_local_pixels(
    projector: TwoLensProjector,
    lens_name: str,
    points_xz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lens = projector.metadata.lens(lens_name)

    targets = np.column_stack(
        [
            points_xz[:, 0],
            np.zeros(len(points_xz), dtype=np.float64),
            points_xz[:, 1],
            np.ones(len(points_xz), dtype=np.float64),
        ]
    )

    map_x, map_y, valid = project_world_target_to_lens(
        targets,
        lens,
        projector.metadata.camera2world,
        STACKED_W,
        STACKED_H,
        lens_name == "left",
    )

    local_y = (
        map_y
        if lens_name == "left"
        else map_y - LENS_H
    )

    valid = (
        valid
        & (map_x >= 0)
        & (map_x < LENS_W)
        & (local_y >= 0)
        & (local_y < LENS_H)
    )

    pixels = np.column_stack([map_x, local_y])
    return pixels, valid


def segment(
    x1: float,
    z1: float,
    x2: float,
    z2: float,
    n: int = 240,
) -> np.ndarray:
    return np.column_stack(
        [
            np.linspace(x1, x2, n),
            np.linspace(z1, z2, n),
        ]
    )


def draw_world_polyline(
    image: np.ndarray,
    projector: TwoLensProjector,
    lens_name: str,
    points_xz: np.ndarray,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    pixels, valid = world_to_local_pixels(
        projector,
        lens_name,
        points_xz,
    )

    for i in range(len(points_xz) - 1):
        if not valid[i] or not valid[i + 1]:
            continue

        p1 = tuple(np.round(pixels[i]).astype(int))
        p2 = tuple(np.round(pixels[i + 1]).astype(int))

        if np.linalg.norm(pixels[i + 1] - pixels[i]) > 80:
            continue

        cv2.line(
            image,
            p1,
            p2,
            color,
            thickness,
            cv2.LINE_AA,
        )


def draw_geometry(
    image: np.ndarray,
    projector: TwoLensProjector,
    lens_name: str,
) -> np.ndarray:
    result = image.copy()

    hl = projector.half_length
    hw = projector.half_width

    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    gray = (150, 150, 150)
    magenta = (255, 0, 255)

    # Strict pitch boundary.
    strict_lines = [
        segment(-hl, -hw, hl, -hw),
        segment(hl, -hw, hl, hw),
        segment(hl, hw, -hl, hw),
        segment(-hl, hw, -hl, -hw),
        segment(0.0, -hw, 0.0, hw),
    ]

    for line in strict_lines:
        draw_world_polyline(
            result,
            projector,
            lens_name,
            line,
            cyan,
            2,
        )

    # +0.50m only outside the far touchline.
    far_z = (
        projector.far_touchline_z
        + (
            projector.far_touchline_margin_m
            if projector.far_touchline_z > 0
            else -projector.far_touchline_margin_m
        )
    )
    draw_world_polyline(
        result,
        projector,
        lens_name,
        segment(-hl, far_z, hl, far_z),
        yellow,
        2,
    )

    # Sparse ground samples. Green = this lens owns the world point.
    # Gray = visible in this lens but owned by the other lens.
    xs = np.linspace(-hl, hl, 22)
    zs = np.linspace(-hw, hw, 14)

    seam_points = []

    for z in zs:
        previous_owner = None
        previous_point = None

        for x in xs:
            owner, left, right = projector.owner_for_world_point(
                float(x),
                float(z),
            )
            visibility = left if lens_name == "left" else right

            if visibility.visible:
                center = (
                    int(round(visibility.pixel_x)),
                    int(round(visibility.pixel_y)),
                )
                cv2.circle(
                    result,
                    center,
                    4,
                    green if owner == lens_name else gray,
                    -1,
                    cv2.LINE_AA,
                )

            if (
                previous_owner is not None
                and owner is not None
                and owner != previous_owner
                and previous_point is not None
            ):
                seam_points.append(
                    (
                        0.5 * (previous_point[0] + x),
                        0.5 * (previous_point[1] + z),
                    )
                )

            previous_owner = owner
            previous_point = (x, z)

    if seam_points:
        seam = np.asarray(seam_points, dtype=np.float64)
        seam = seam[np.argsort(seam[:, 1])]
        draw_world_polyline(
            result,
            projector,
            lens_name,
            seam,
            magenta,
            3,
        )

    cv2.putText(
        result,
        f"PHYSICAL {lens_name.upper()} LENS",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return result


def ownership_map(
    projector: TwoLensProjector,
    *,
    width: int = 1050,
) -> np.ndarray:
    height = int(
        round(
            width
            * projector.metadata.field_width
            / projector.metadata.field_length
        )
    )

    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    # Background colors deliberately subtle; white/magenta overlays carry the
    # useful information.
    left_color = np.array([70, 70, 35], dtype=np.uint8)
    right_color = np.array([35, 70, 70], dtype=np.uint8)
    gap_color = np.array([25, 25, 25], dtype=np.uint8)

    for py in range(height):
        z = (
            projector.half_width
            - (py + 0.5) / height
            * projector.metadata.field_width
        )

        for px in range(width):
            x = (
                -projector.half_length
                + (px + 0.5) / width
                * projector.metadata.field_length
            )

            owner, _, _ = projector.owner_for_world_point(
                float(x),
                float(z),
            )

            if owner == "left":
                canvas[py, px] = left_color
            elif owner == "right":
                canvas[py, px] = right_color
            else:
                canvas[py, px] = gap_color

    cv2.rectangle(
        canvas,
        (0, 0),
        (width - 1, height - 1),
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    cv2.line(
        canvas,
        (width // 2, 0),
        (width // 2, height - 1),
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Highlight ownership boundary.
    owner_left = np.all(canvas == left_color, axis=2).astype(np.uint8)
    owner_right = np.all(canvas == right_color, axis=2).astype(np.uint8)
    boundary = cv2.morphologyEx(
        owner_left,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    boundary |= cv2.morphologyEx(
        owner_right,
        cv2.MORPH_GRADIENT,
        np.ones((3, 3), dtype=np.uint8),
    )
    canvas[boundary.astype(bool)] = (255, 0, 255)

    cv2.putText(
        canvas,
        "LEFT OWNER",
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "RIGHT OWNER",
        (width - 220, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return canvas


def compose(
    left: np.ndarray,
    right: np.ndarray,
    owner_map: np.ndarray,
) -> np.ndarray:
    panel_w = 960
    panel_h = 480

    left_panel = cv2.resize(
        left,
        (panel_w, panel_h),
        interpolation=cv2.INTER_AREA,
    )
    right_panel = cv2.resize(
        right,
        (panel_w, panel_h),
        interpolation=cv2.INTER_AREA,
    )

    top = np.concatenate([left_panel, right_panel], axis=1)

    owner_h = 500
    scale = owner_h / owner_map.shape[0]
    owner_w = int(round(owner_map.shape[1] * scale))
    owner_resized = cv2.resize(
        owner_map,
        (owner_w, owner_h),
        interpolation=cv2.INTER_NEAREST,
    )

    bottom = np.zeros(
        (owner_h + 70, top.shape[1], 3),
        dtype=np.uint8,
    )
    x0 = (bottom.shape[1] - owner_w) // 2
    bottom[50 : 50 + owner_h, x0 : x0 + owner_w] = owner_resized

    cv2.putText(
        bottom,
        "OWNERSHIP ON THE REAL PITCH (magenta = seam)",
        (30, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return np.concatenate([top, bottom], axis=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate physical two-lens geometry and ownership."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument(
        "--far-margin",
        type=float,
        default=DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT / "projection_debug",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame, frame_info = read_frame(args.video, args.time)

    renderer = TwoLensRenderer(args.calibration)
    views = renderer.render(frame)

    projector = TwoLensProjector(
        args.calibration,
        far_touchline_margin_m=args.far_margin,
    )

    left = draw_geometry(
        views["left"],
        projector,
        "left",
    )
    right = draw_geometry(
        views["right"],
        projector,
        "right",
    )
    owner_map = ownership_map(projector)
    validation = compose(left, right, owner_map)

    left_path = args.out_dir / "left_geometry.png"
    right_path = args.out_dir / "right_geometry.png"
    ownership_path = args.out_dir / "ownership_map.png"
    validation_path = args.out_dir / "two_lens_validation.jpg"
    summary_path = args.out_dir / "projection_validation.json"

    cv2.imwrite(str(left_path), left)
    cv2.imwrite(str(right_path), right)
    cv2.imwrite(str(ownership_path), owner_map)
    cv2.imwrite(str(validation_path), validation)

    summary = projector.summary()
    summary.update(
        {
            "video": str(args.video),
            "frame": frame_info,
            "colors": {
                "strict_pitch_boundary": "cyan",
                "far_touchline_margin": "yellow",
                "owned_sample": "green",
                "visible_but_other_owner_sample": "gray",
                "ownership_seam": "magenta",
            },
        }
    )

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print()
    print(f"Saved: {validation_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
