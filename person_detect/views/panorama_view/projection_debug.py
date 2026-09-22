"""Render panorama projection diagnostics without running detector inference."""

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

from person_detect.views.panorama_view.projection import (  # noqa: E402
    DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    PanoramaProjector,
)
from person_detect.views.panorama_view.renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_OUT,
    DEFAULT_VIDEO,
    PanoramaRenderer,
    read_frame,
)


def _draw_world_line(
    image: np.ndarray,
    projector: PanoramaProjector,
    points: list[tuple[float, float]],
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    pixels = []
    for world_x, world_z in points:
        pixel = projector.pixel_for_world(world_x, world_z)
        if pixel is None:
            continue
        x, y = pixel
        if np.isfinite([x, y]).all():
            pixels.append((int(round(x)), int(round(y))))
    if len(pixels) >= 2:
        cv2.polylines(image, [np.asarray(pixels, dtype=np.int32)], False, color, thickness, cv2.LINE_AA)


def _pitch_perimeter(length: float, width: float, *, z_far: float) -> list[tuple[float, float]]:
    half_length = length / 2.0
    half_width = width / 2.0
    points = [
        (-half_length, -half_width),
        (half_length, -half_width),
        (half_length, half_width),
        (-half_length, half_width),
        (-half_length, -half_width),
    ]
    # Keep the argument explicit in the call site so the far-touchline
    # convention is visible in the diagnostic implementation.
    _ = z_far
    return points


def draw_projection_geometry(
    panorama: np.ndarray,
    projector: PanoramaProjector,
) -> np.ndarray:
    output = panorama.copy()
    field_length = float(projector.metadata.field_length)
    field_width = float(projector.metadata.field_width)
    half_length = field_length / 2.0
    half_width = field_width / 2.0

    _draw_world_line(
        output,
        projector,
        _pitch_perimeter(field_length, field_width, z_far=projector.far_touchline_z),
        (255, 255, 255),
        5,
    )

    # Midfield and a useful 10 m / 5 m pitch grid.
    _draw_world_line(
        output,
        projector,
        [(0.0, -half_width), (0.0, half_width)],
        (220, 220, 220),
        3,
    )
    for x in np.arange(-half_length + 10.0, half_length, 10.0):
        _draw_world_line(output, projector, [(float(x), -half_width), (float(x), half_width)], (90, 150, 90), 2)
    for z in np.arange(-half_width + 5.0, half_width, 5.0):
        _draw_world_line(output, projector, [(-half_length, float(z)), (half_length, float(z))], (90, 150, 90), 2)

    margin = float(projector.far_touchline_margin_m)
    if projector.far_touchline_z > 0.0:
        margin_line = [(x, half_width + margin) for x in np.linspace(-half_length, half_length, 96)]
    else:
        margin_line = [(x, -half_width - margin) for x in np.linspace(-half_length, half_length, 96)]
    _draw_world_line(output, projector, margin_line, (0, 165, 255), 5)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate calibrated panorama pitch projection.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--far-margin", type=float, default=DEFAULT_FAR_TOUCHLINE_MARGIN_M)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame, frame_info = read_frame(args.video, args.time)
    renderer = PanoramaRenderer(args.calibration, width=args.width, height=args.height)
    projector = PanoramaProjector(
        args.calibration,
        width=renderer.width,
        height=renderer.height,
        far_touchline_margin_m=args.far_margin,
    )
    panorama = renderer.render(frame)["panorama"]
    diagnostic = draw_projection_geometry(panorama, projector)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.out_dir / "panorama_projection.png"
    summary_path = args.out_dir / "panorama_projection.json"
    cv2.imwrite(str(image_path), diagnostic)

    va = renderer.metadata.va_presentation
    summary = {
        "calibration": str(args.calibration),
        "video": str(args.video),
        "frame": frame_info,
        "panorama_width": renderer.width,
        "panorama_height": renderer.height,
        "field_length_m": float(renderer.metadata.field_length),
        "field_width_m": float(renderer.metadata.field_width),
        "camera_world_xyz": [float(value) for value in projector.camera],
        "far_touchline": {
            "name": projector.far_touchline_name,
            "z": projector.far_touchline_z,
            "margin_outward_m": projector.far_touchline_margin_m,
        },
        "angular_presentation_range": {key: float(value) for key, value in va.items()},
        "left_valid_fraction": float(np.mean(renderer.left_valid)),
        "right_valid_fraction": float(np.mean(renderer.right_valid)),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {image_path}")
    print(f"Saved: {summary_path}")


if __name__ == "__main__":
    main()
