#!/usr/bin/env python3
"""Visual tuning validator for analysis.tracking_ground_v3.

Use zero values to reproduce the Veo-derived model exactly, or pass tiny
yaw/screen offsets while we determine the residual systematic error.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.tracking_ground_v3 import (
    LegacyCameraDirections,
    TrackingTuning,
    field_point_to_tracking_pixel,
)
from analysis.veo_metadata import load_veo_metadata


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--calibration", type=Path, required=True)
    p.add_argument("--detections", type=Path, required=True)
    p.add_argument("--time", type=float, required=True)
    p.add_argument("--screen-offset-x", type=float, default=0.0)
    p.add_argument("--screen-offset-y", type=float, default=0.0)
    p.add_argument("--yaw-bias-deg", type=float, default=0.0)
    p.add_argument("--pitch-bias-deg", type=float, default=0.0)
    p.add_argument("--fov-bias-deg", type=float, default=0.0)
    p.add_argument("--out", type=Path, default=Path("output/tracking_geometry_v3.png"))
    return p.parse_args()


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
    hl = field_length / 2.0
    hw = field_width / 2.0
    px = hl - 11.0
    pax = hl - 16.5
    paz = 20.16
    gax = hl - 5.5
    gaz = 9.16
    return [
        ("Center", 0.0, 0.0),
        ("Left penalty", -px, 0.0),
        ("Right penalty", px, 0.0),
        ("Halfway near", 0.0, -hw),
        ("Halfway far", 0.0, hw),
        ("Left near corner", -hl, -hw),
        ("Left far corner", -hl, hw),
        ("Right near corner", hl, -hw),
        ("Right far corner", hl, hw),
        ("Left PA near", -pax, -paz),
        ("Left PA far", -pax, paz),
        ("Right PA near", pax, -paz),
        ("Right PA far", pax, paz),
        ("Left GA near", -gax, -gaz),
        ("Left GA far", -gax, gaz),
        ("Right GA near", gax, -gaz),
        ("Right GA far", gax, gaz),
    ]


def main() -> None:
    args = parse_args()
    metadata = load_veo_metadata(args.calibration)
    directions = LegacyCameraDirections.load(args.detections)
    frame, frame_info = extract_frame(args.video, args.time)
    height, width = frame.shape[:2]

    tuning = TrackingTuning(
        yaw_bias_rad=math.radians(args.yaw_bias_deg),
        pitch_bias_rad=math.radians(args.pitch_bias_deg),
        fov_bias_rad=math.radians(args.fov_bias_deg),
        screen_offset_x_px=args.screen_offset_x,
        screen_offset_y_px=args.screen_offset_y,
    )

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
            tuning=tuning,
        )
        projected.append(
            {
                "name": name,
                "world_x": world_x,
                "world_z": world_z,
                **point.to_dict(),
            }
        )

        if not point.visible or point.pixel_x is None or point.pixel_y is None:
            continue

        x = int(round(point.pixel_x))
        y = int(round(point.pixel_y))
        cv2.drawMarker(
            frame, (x, y), (0, 0, 255),
            markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2,
        )
        cv2.circle(frame, (x, y), 7, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(
            frame, name, (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA,
        )

    # Put tuning values in the image itself so screenshots cannot be confused.
    label = (
        f"offsetX={args.screen_offset_x:+.1f}px  "
        f"yaw={args.yaw_bias_deg:+.4f}deg  "
        f"offsetY={args.screen_offset_y:+.1f}px"
    )
    cv2.rectangle(frame, (10, 10), (760, 48), (0, 0, 0), -1)
    cv2.putText(
        frame, label, (20, 37),
        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA,
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
        "tuning": {
            "screen_offset_x_px": args.screen_offset_x,
            "screen_offset_y_px": args.screen_offset_y,
            "yaw_bias_deg": args.yaw_bias_deg,
            "pitch_bias_deg": args.pitch_bias_deg,
            "fov_bias_deg": args.fov_bias_deg,
        },
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
