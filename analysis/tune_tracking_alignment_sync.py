#!/usr/bin/env python3
"""Interactive Veo Tracking alignment + DET sync tuner.

This tool keeps the real Standard/Tracking video frame fixed at ``video_time``
while allowing the camera-direction time used from
``detections_constrained.det`` to move independently.

That lets us test whether the residual alignment error is caused by a constant
time offset between ``video.mp4`` and the .det camera-direction stream.

Sliders
-------
Offset X px      residual horizontal screen translation
Offset Y px      residual vertical screen translation
Yaw x0.01 deg    residual yaw bias, range -5..+5 degrees
DET sync frames  camera-direction offset, range -30..+30 frames
Time x0.1 s      video frame time

The projection uses:
    det_time = video_time + det_sync_frames / TRACKING_FPS

Controls
--------
S           Save current JSON + screenshot
R           Reset X/Y/Yaw/DET sync to command-line defaults
0           Zero X/Y/Yaw/DET sync
Left/Right  Previous/next video frame
A/D         -/+ 1 second
Q or Esc    Quit
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.tracking_ground_v3 import (  # noqa: E402
    TRACKING_FPS,
    LegacyCameraDirections,
    TrackingTuning,
    field_point_to_tracking_pixel,
)
from analysis.veo_metadata import load_veo_metadata  # noqa: E402


WINDOW_NAME = "Veo Tracking + DET sync tuner"

OFFSET_X_MIN = -250
OFFSET_X_MAX = 250
OFFSET_Y_MIN = -150
OFFSET_Y_MAX = 150

YAW_MIN_DEG = -5.0
YAW_MAX_DEG = 5.0
YAW_STEP_DEG = 0.01

DET_SYNC_MIN_FRAMES = -30
DET_SYNC_MAX_FRAMES = 30

TIME_STEP_S = 0.1


@dataclass
class VideoInfo:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune Veo Tracking residual alignment and DET time sync."
    )
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--time", type=float, default=0.0)

    parser.add_argument("--offset-x", type=float, default=0.0)
    parser.add_argument("--offset-y", type=float, default=0.0)
    parser.add_argument("--yaw-bias-deg", type=float, default=0.0)
    parser.add_argument("--det-sync-frames", type=int, default=0)

    parser.add_argument(
        "--save",
        type=Path,
        default=Path("output/tracking_alignment_sync_tuning.json"),
    )
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("output/tracking_alignment_sync_tuning.png"),
    )
    parser.add_argument("--window-width", type=int, default=1400)
    parser.add_argument("--window-height", type=int, default=900)
    return parser.parse_args()


def read_video_info(capture: cv2.VideoCapture) -> VideoInfo:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0.0:
        raise RuntimeError("Video reports an invalid FPS")
    if frame_count <= 0:
        raise RuntimeError("Video reports an invalid frame count")
    if width <= 0 or height <= 0:
        raise RuntimeError("Video reports invalid image dimensions")

    return VideoInfo(
        fps=fps,
        frame_count=frame_count,
        duration_s=frame_count / fps,
        width=width,
        height=height,
    )


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


def clamp(value, minimum, maximum):
    return max(minimum, min(maximum, value))


def x_to_slider(value: float) -> int:
    return int(round(clamp(value, OFFSET_X_MIN, OFFSET_X_MAX) - OFFSET_X_MIN))


def slider_to_x(position: int) -> float:
    return float(position + OFFSET_X_MIN)


def y_to_slider(value: float) -> int:
    return int(round(clamp(value, OFFSET_Y_MIN, OFFSET_Y_MAX) - OFFSET_Y_MIN))


def slider_to_y(position: int) -> float:
    return float(position + OFFSET_Y_MIN)


def yaw_to_slider(value_deg: float) -> int:
    value = clamp(value_deg, YAW_MIN_DEG, YAW_MAX_DEG)
    return int(round((value - YAW_MIN_DEG) / YAW_STEP_DEG))


def slider_to_yaw(position: int) -> float:
    return YAW_MIN_DEG + float(position) * YAW_STEP_DEG


def sync_to_slider(frames: int) -> int:
    frames = int(clamp(frames, DET_SYNC_MIN_FRAMES, DET_SYNC_MAX_FRAMES))
    return frames - DET_SYNC_MIN_FRAMES


def slider_to_sync(position: int) -> int:
    return int(position + DET_SYNC_MIN_FRAMES)


def time_to_slider(value_s: float, duration_s: float) -> int:
    value = clamp(value_s, 0.0, max(duration_s, 0.0))
    return int(round(value / TIME_STEP_S))


def slider_to_time(position: int) -> float:
    return float(position) * TIME_STEP_S


def extract_frame(
    capture: cv2.VideoCapture,
    info: VideoInfo,
    time_s: float,
) -> tuple[np.ndarray, int, float]:
    target_frame = int(round(time_s * info.fps))
    target_frame = int(clamp(target_frame, 0, info.frame_count - 1))

    capture.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = capture.read()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read tracking frame {target_frame} near t={time_s:.3f}s"
        )

    actual_frame = int(capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    actual_time = actual_frame / info.fps

    return frame, actual_frame, actual_time


def overlay_landmarks(
    frame: np.ndarray,
    metadata,
    directions: LegacyCameraDirections,
    *,
    det_time_s: float,
    tuning: TrackingTuning,
):
    result = frame.copy()
    height, width = result.shape[:2]

    rows = []
    visible_count = 0

    for name, world_x, world_z in pitch_landmarks(
        metadata.field_length,
        metadata.field_width,
    ):
        projection = field_point_to_tracking_pixel(
            metadata,
            directions,
            det_time_s,
            world_x,
            world_z,
            width,
            height,
            tuning=tuning,
        )

        rows.append(
            {
                "name": name,
                "world_x": world_x,
                "world_z": world_z,
                **projection.to_dict(),
            }
        )

        if (
            not projection.visible
            or projection.pixel_x is None
            or projection.pixel_y is None
        ):
            continue

        visible_count += 1
        x = int(round(projection.pixel_x))
        y = int(round(projection.pixel_y))

        cv2.drawMarker(
            result,
            (x, y),
            (0, 0, 255),
            markerType=cv2.MARKER_CROSS,
            markerSize=18,
            thickness=2,
        )
        cv2.circle(
            result,
            (x, y),
            7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            result,
            name,
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return result, rows, visible_count


def draw_status(
    frame: np.ndarray,
    *,
    offset_x: float,
    offset_y: float,
    yaw_deg: float,
    det_sync_frames: int,
    video_time_s: float,
    det_time_s: float,
    decoded_time_s: float,
    frame_index: int,
    visible_count: int,
):
    sync_seconds = det_sync_frames / TRACKING_FPS

    lines = [
        (
            f"X={offset_x:+.0f}px   "
            f"Y={offset_y:+.0f}px   "
            f"Yaw={yaw_deg:+.2f} deg   "
            f"DET sync={det_sync_frames:+d} fr ({sync_seconds:+.4f}s)"
        ),
        (
            f"Video t={video_time_s:.3f}s   "
            f"DET t={det_time_s:.3f}s   "
            f"decoded={decoded_time_s:.3f}s   "
            f"frame={frame_index}   "
            f"landmarks={visible_count}"
        ),
        "S save | R reset initial | 0 zero tuning | arrows frame | A/D +/-1s | Q quit",
    ]

    x0 = 12
    y0 = 12
    line_height = 30
    box_width = min(frame.shape[1] - 24, 1450)
    box_height = 12 + line_height * len(lines)

    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + box_width, y0 + box_height),
        (0, 0, 0),
        -1,
    )

    for index, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (x0 + 12, y0 + 25 + index * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62 if index < 2 else 0.50,
            (255, 255, 255),
            2 if index < 2 else 1,
            cv2.LINE_AA,
        )


def save_state(
    *,
    args: argparse.Namespace,
    info: VideoInfo,
    video_time_s: float,
    det_time_s: float,
    decoded_time_s: float,
    frame_index: int,
    det_sync_frames: int,
    tuning: TrackingTuning,
    yaw_deg: float,
    landmarks: list[dict],
    screenshot: np.ndarray,
):
    args.save.parent.mkdir(parents=True, exist_ok=True)
    args.screenshot.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": 2,
        "kind": "veo_tracking_alignment_and_det_sync_tuning",
        "video": str(args.video),
        "calibration": str(args.calibration),
        "detections": str(args.detections),
        "video_info": {
            "fps": info.fps,
            "frame_count": info.frame_count,
            "duration_s": info.duration_s,
            "width": info.width,
            "height": info.height,
        },
        "validation_frame": {
            "video_time_s": video_time_s,
            "det_time_s": det_time_s,
            "decoded_video_time_s": decoded_time_s,
            "video_frame_index": frame_index,
        },
        "tuning": {
            "screen_offset_x_px": tuning.screen_offset_x_px,
            "screen_offset_y_px": tuning.screen_offset_y_px,
            "screen_offset_x_normalized": tuning.screen_offset_x_px / info.width,
            "screen_offset_y_normalized": tuning.screen_offset_y_px / info.height,
            "yaw_bias_deg": yaw_deg,
            "yaw_bias_rad": tuning.yaw_bias_rad,
            "det_sync_frames": det_sync_frames,
            "det_sync_seconds": det_sync_frames / TRACKING_FPS,
        },
        "landmarks": landmarks,
    }

    args.save.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if not cv2.imwrite(str(args.screenshot), screenshot):
        raise RuntimeError(f"Could not write screenshot: {args.screenshot}")

    print()
    print("[saved]")
    print(f"  JSON:       {args.save}")
    print(f"  Screenshot: {args.screenshot}")
    print(
        "  Tuning:     "
        f"X={tuning.screen_offset_x_px:+.0f}px, "
        f"Y={tuning.screen_offset_y_px:+.0f}px, "
        f"Yaw={yaw_deg:+.2f}°, "
        f"DET sync={det_sync_frames:+d} frames "
        f"({det_sync_frames / TRACKING_FPS:+.4f}s)"
    )
    print()


def main() -> None:
    args = parse_args()

    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.calibration.exists():
        raise FileNotFoundError(args.calibration)
    if not args.detections.exists():
        raise FileNotFoundError(args.detections)

    metadata = load_veo_metadata(args.calibration)
    directions = LegacyCameraDirections.load(args.detections)

    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    info = read_video_info(capture)

    print("Veo Tracking + DET sync tuner")
    print(f"  Video:      {args.video}")
    print(f"  Resolution: {info.width}x{info.height}")
    print(f"  Video FPS:  {info.fps:.6f}")
    print(f"  DET FPS:    {TRACKING_FPS:.6f}")
    print(f"  Duration:   {info.duration_s:.3f}s")
    print(f"  DET rows:   {directions.sample_count}")
    print()
    print("Keep the video frame fixed and move DET sync to test temporal alignment.")
    print("Press S when a combination looks useful.")

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        args.window_width,
        args.window_height,
    )

    cv2.createTrackbar(
        "Offset X px",
        WINDOW_NAME,
        x_to_slider(args.offset_x),
        OFFSET_X_MAX - OFFSET_X_MIN,
        lambda _value: None,
    )
    cv2.createTrackbar(
        "Offset Y px",
        WINDOW_NAME,
        y_to_slider(args.offset_y),
        OFFSET_Y_MAX - OFFSET_Y_MIN,
        lambda _value: None,
    )

    yaw_max = int(round((YAW_MAX_DEG - YAW_MIN_DEG) / YAW_STEP_DEG))
    cv2.createTrackbar(
        "Yaw x0.01 deg",
        WINDOW_NAME,
        yaw_to_slider(args.yaw_bias_deg),
        yaw_max,
        lambda _value: None,
    )

    cv2.createTrackbar(
        "DET sync frames",
        WINDOW_NAME,
        sync_to_slider(args.det_sync_frames),
        DET_SYNC_MAX_FRAMES - DET_SYNC_MIN_FRAMES,
        lambda _value: None,
    )

    time_max = max(1, int(math.ceil(info.duration_s / TIME_STEP_S)))
    cv2.createTrackbar(
        "Time x0.1 s",
        WINDOW_NAME,
        time_to_slider(args.time, info.duration_s),
        time_max,
        lambda _value: None,
    )

    initial_x = args.offset_x
    initial_y = args.offset_y
    initial_yaw = args.yaw_bias_deg
    initial_sync = args.det_sync_frames

    cached_time_position = None
    cached_frame = None
    cached_frame_index = None
    cached_decoded_time = None

    last_state = None
    rendered = None
    rendered_landmarks = []

    pending_time_position = None
    pending_since = 0.0
    time_debounce_s = 0.08

    try:
        while True:
            x_pos = cv2.getTrackbarPos("Offset X px", WINDOW_NAME)
            y_pos = cv2.getTrackbarPos("Offset Y px", WINDOW_NAME)
            yaw_pos = cv2.getTrackbarPos("Yaw x0.01 deg", WINDOW_NAME)
            sync_pos = cv2.getTrackbarPos("DET sync frames", WINDOW_NAME)
            time_pos = cv2.getTrackbarPos("Time x0.1 s", WINDOW_NAME)

            offset_x = slider_to_x(x_pos)
            offset_y = slider_to_y(y_pos)
            yaw_deg = slider_to_yaw(yaw_pos)
            det_sync_frames = slider_to_sync(sync_pos)

            requested_video_time = clamp(
                slider_to_time(time_pos),
                0.0,
                max(0.0, info.duration_s - 1.0 / info.fps),
            )

            if time_pos != pending_time_position:
                pending_time_position = time_pos
                pending_since = time.monotonic()

            should_decode_time = (
                cached_frame is None
                or cached_time_position == time_pos
                or (time.monotonic() - pending_since) >= time_debounce_s
            )

            if (
                should_decode_time
                and (
                    cached_frame is None
                    or cached_time_position != time_pos
                )
            ):
                (
                    cached_frame,
                    cached_frame_index,
                    cached_decoded_time,
                ) = extract_frame(
                    capture,
                    info,
                    requested_video_time,
                )
                cached_time_position = time_pos

            if cached_frame is not None:
                # Important: synchronize against the ACTUALLY decoded video frame,
                # not merely the requested slider time.
                det_time = (
                    cached_decoded_time
                    + det_sync_frames / TRACKING_FPS
                )

                state = (
                    offset_x,
                    offset_y,
                    yaw_deg,
                    det_sync_frames,
                    cached_time_position,
                    cached_frame_index,
                )

                if state != last_state:
                    tuning = TrackingTuning(
                        yaw_bias_rad=math.radians(yaw_deg),
                        screen_offset_x_px=offset_x,
                        screen_offset_y_px=offset_y,
                    )

                    rendered, rendered_landmarks, visible_count = overlay_landmarks(
                        cached_frame,
                        metadata,
                        directions,
                        det_time_s=det_time,
                        tuning=tuning,
                    )

                    draw_status(
                        rendered,
                        offset_x=offset_x,
                        offset_y=offset_y,
                        yaw_deg=yaw_deg,
                        det_sync_frames=det_sync_frames,
                        video_time_s=cached_decoded_time,
                        det_time_s=det_time,
                        decoded_time_s=cached_decoded_time,
                        frame_index=cached_frame_index,
                        visible_count=visible_count,
                    )

                    last_state = state

                cv2.imshow(WINDOW_NAME, rendered)

            key = cv2.waitKeyEx(15)

            if key in (ord("q"), ord("Q"), 27):
                break

            if key in (ord("s"), ord("S")) and rendered is not None:
                tuning = TrackingTuning(
                    yaw_bias_rad=math.radians(yaw_deg),
                    screen_offset_x_px=offset_x,
                    screen_offset_y_px=offset_y,
                )
                det_time = (
                    cached_decoded_time
                    + det_sync_frames / TRACKING_FPS
                )
                save_state(
                    args=args,
                    info=info,
                    video_time_s=cached_decoded_time,
                    det_time_s=det_time,
                    decoded_time_s=cached_decoded_time,
                    frame_index=cached_frame_index,
                    det_sync_frames=det_sync_frames,
                    tuning=tuning,
                    yaw_deg=yaw_deg,
                    landmarks=rendered_landmarks,
                    screenshot=rendered,
                )

            elif key in (ord("r"), ord("R")):
                cv2.setTrackbarPos("Offset X px", WINDOW_NAME, x_to_slider(initial_x))
                cv2.setTrackbarPos("Offset Y px", WINDOW_NAME, y_to_slider(initial_y))
                cv2.setTrackbarPos(
                    "Yaw x0.01 deg",
                    WINDOW_NAME,
                    yaw_to_slider(initial_yaw),
                )
                cv2.setTrackbarPos(
                    "DET sync frames",
                    WINDOW_NAME,
                    sync_to_slider(initial_sync),
                )

            elif key == ord("0"):
                cv2.setTrackbarPos("Offset X px", WINDOW_NAME, x_to_slider(0.0))
                cv2.setTrackbarPos("Offset Y px", WINDOW_NAME, y_to_slider(0.0))
                cv2.setTrackbarPos(
                    "Yaw x0.01 deg",
                    WINDOW_NAME,
                    yaw_to_slider(0.0),
                )
                cv2.setTrackbarPos(
                    "DET sync frames",
                    WINDOW_NAME,
                    sync_to_slider(0),
                )

            elif key in (2424832, 65361):
                new_time = max(
                    0.0,
                    cached_decoded_time - 1.0 / info.fps,
                )
                cv2.setTrackbarPos(
                    "Time x0.1 s",
                    WINDOW_NAME,
                    time_to_slider(new_time, info.duration_s),
                )

            elif key in (2555904, 65363):
                new_time = min(
                    info.duration_s,
                    cached_decoded_time + 1.0 / info.fps,
                )
                cv2.setTrackbarPos(
                    "Time x0.1 s",
                    WINDOW_NAME,
                    time_to_slider(new_time, info.duration_s),
                )

            elif key in (ord("a"), ord("A")):
                new_time = max(0.0, requested_video_time - 1.0)
                cv2.setTrackbarPos(
                    "Time x0.1 s",
                    WINDOW_NAME,
                    time_to_slider(new_time, info.duration_s),
                )

            elif key in (ord("d"), ord("D")):
                new_time = min(info.duration_s, requested_video_time + 1.0)
                cv2.setTrackbarPos(
                    "Time x0.1 s",
                    WINDOW_NAME,
                    time_to_slider(new_time, info.duration_s),
                )

    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
