#!/usr/bin/env python3
"""Render known pitch landmarks on a Veo Tracking frame.

This is a visual validation tool for analysis.tracking_ground.  It does not
change calibration or fit any parameters: it simply projects canonical field
points through the reconstructed dynamic Tracking camera.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.tracking_ground import (  # noqa: E402
    LegacyCameraDirections,
    field_point_to_tracking_pixel,
)
from analysis.veo_metadata import load_veo_metadata  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overlay canonical football landmarks on a Veo Tracking frame."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/tracking_geometry_validation.png"),
    )
    return parser.parse_args()


def extract_frame(video_path: Path, time_s: float):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    target_frame = int(round(time_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame near t={time_s:.6f}s")

    return frame, {
        "video_fps": fps,
        "requested_time_s": float(time_s),
        "requested_frame": target_frame,
        "actual_frame": actual_frame,
        "actual_time_approx_s": actual_frame / fps if fps > 0 else None,
    }


def pitch_landmarks(field_length: float, field_width: float):
    half_l = field_length / 2.0
    half_w = field_width / 2.0

    penalty_x = half_l - 11.0
    penalty_area_x = half_l - 16.5
    penalty_area_z = 20.16
    goal_area_x = half_l - 5.5
    goal_area_z = 9.16

    return [
        ("Center", 0.0, 0.0),
        ("Left penalty", -penalty_x, 0.0),
        ("Right penalty", penalty_x, 0.0),
        ("Halfway near", 0.0, -half_w),
        ("Halfway far", 0.0, half_w),
        ("Left near corner", -half_l, -half_w),
        ("Left far corner", -half_l, half_w),
        ("Right near corner", half_l, -half_w),
        ("Right far corner", half_l, half_w),
        ("Left PA near", -penalty_area_x, -penalty_area_z),
        ("Left PA far", -penalty_area_x, penalty_area_z),
        ("Right PA near", penalty_area_x, -penalty_area_z),
        ("Right PA far", penalty_area_x, penalty_area_z),
        ("Left GA near", -goal_area_x, -goal_area_z),
        ("Left GA far", -goal_area_x, goal_area_z),
        ("Right GA near", goal_area_x, -goal_area_z),
        ("Right GA far", goal_area_x, goal_area_z),
    ]


def main() -> None:
    args = parse_args()
    metadata = load_veo_metadata(args.calibration)
    directions = LegacyCameraDirections.load(args.detections)
    frame, frame_info = extract_frame(args.video, args.time)
    height, width = frame.shape[:2]

    projected = []
    for name, world_x, world_z in pitch_landmarks(
        metadata.field_length,
        metadata.field_width,
    ):
        point = field_point_to_tracking_pixel(
            metadata,
            directions,
            args.time,
            world_x,
            world_z,
            width,
            height,
        )
        item = {
            "name": name,
            "world_x": world_x,
            "world_z": world_z,
            **point.to_dict(),
        }
        projected.append(item)

        if not point.visible or point.pixel_x is None or point.pixel_y is None:
            continue

        x = int(round(point.pixel_x))
        y = int(round(point.pixel_y))

        cv2.drawMarker(
            frame,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.circle(frame, (x, y), 7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            name,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), frame):
        raise RuntimeError(f"Could not write output image: {args.out}")

    summary = {
        "video": str(args.video),
        "calibration": str(args.calibration),
        "detections": str(args.detections),
        "frame": frame_info,
        "image_size": [width, height],
        "output": str(args.out),
        "visible_landmarks": sum(1 for item in projected if item["visible"]),
        "landmarks": projected,
    }

    summary_path = args.out.with_suffix(".json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved overlay: {args.out}")
    print(f"Saved details: {summary_path}")


if __name__ == "__main__":
    main()
