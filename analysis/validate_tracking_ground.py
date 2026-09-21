#!/usr/bin/env python3
"""Small CLI for validating analysis.tracking_ground against a Veo match.

Examples:

  python analysis/validate_tracking_ground.py \
    --calibration data/camera_alignment-....veo \
    --detections detections_constrained.det \
    --time 3154.251 \
    --pixel 960 800

  python analysis/validate_tracking_ground.py \
    --calibration data/camera_alignment-....veo \
    --detections detections_constrained.det \
    --time 3154.251 \
    --field 41.5 0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.tracking_ground import (  # noqa: E402
    LegacyCameraDirections,
    camera_state_at_time,
    field_point_to_tracking_pixel,
    tracking_ground_for_pixel,
)
from analysis.veo_metadata import load_veo_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Veo Tracking pixel <-> field geometry."
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--pixel",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Tracking-view pixel in browser/top-left coordinates.",
    )
    group.add_argument(
        "--field",
        nargs=2,
        type=float,
        metavar=("X", "Z"),
        help="Canonical field point in metres (Y=0).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = load_veo_metadata(args.calibration)
    directions = LegacyCameraDirections.load(args.detections)

    state = camera_state_at_time(metadata, directions, args.time)
    if state is None:
        raise SystemExit(
            f"No camera-direction sample exists at t={args.time:.6f}s"
        )

    result: dict = {
        "directions": {
            "source": str(args.detections),
            "sample_count": directions.sample_count,
            "earliest_frame": directions.earliest_frame,
            "latest_frame": directions.latest_frame,
        },
        "camera_state": state.to_dict(include_matrices=False),
    }

    if args.pixel is not None:
        pixel_x, pixel_y = args.pixel
        intersection = tracking_ground_for_pixel(
            metadata,
            directions,
            args.time,
            pixel_x,
            pixel_y,
            args.width,
            args.height,
        )
        result["tracking_to_field"] = intersection.to_dict()

        # Useful validation invariant: if the click hits Y=0, project the
        # recovered X/Z back into Tracking with the same camera state.
        if intersection.world_x is not None and intersection.world_z is not None:
            reprojection = field_point_to_tracking_pixel(
                metadata,
                directions,
                args.time,
                intersection.world_x,
                intersection.world_z,
                args.width,
                args.height,
            )
            result["roundtrip_reprojection"] = reprojection.to_dict()
            if reprojection.pixel_x is not None and reprojection.pixel_y is not None:
                result["roundtrip_pixel_error"] = {
                    "dx": reprojection.pixel_x - pixel_x,
                    "dy": reprojection.pixel_y - pixel_y,
                }

    else:
        world_x, world_z = args.field
        projection = field_point_to_tracking_pixel(
            metadata,
            directions,
            args.time,
            world_x,
            world_z,
            args.width,
            args.height,
        )
        result["field_to_tracking"] = projection.to_dict()

        if projection.visible and projection.pixel_x is not None and projection.pixel_y is not None:
            intersection = tracking_ground_for_pixel(
                metadata,
                directions,
                args.time,
                projection.pixel_x,
                projection.pixel_y,
                args.width,
                args.height,
            )
            result["roundtrip_intersection"] = intersection.to_dict()
            if intersection.world_x is not None and intersection.world_z is not None:
                result["roundtrip_field_error_m"] = (
                    (intersection.world_x - world_x) ** 2
                    + (intersection.world_z - world_z) ** 2
                ) ** 0.5

    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=True))


if __name__ == "__main__":
    main()
