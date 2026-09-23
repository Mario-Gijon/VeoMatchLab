"""Projection diagnostics for the fixed 3840×2160 panorama."""

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
    DEFAULT_HEIGHT,
    DEFAULT_OUT,
    DEFAULT_VIDEO,
    DEFAULT_WIDTH,
    PanoramaRenderer,
    read_frame,
)
from person_detect.views.ultra_panorama_view.projection_debug import (  # noqa: E402
    draw_projection_geometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate fixed panorama pitch projection.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
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
