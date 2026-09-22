#!/usr/bin/env python3
"""Raw two-lens view for Veo stacked 2048x2048 footage.

The source transcode stores:
    top half    -> physical LEFT lens  (2048x1024)
    bottom half -> physical RIGHT lens (2048x1024)

This view intentionally performs NO rectification, perspective rendering,
resizing, blending or resampling before detection. The detector receives the
original lens pixels at the maximum detail available in the source file.
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

MATCH_ID = "match_629890153a11"
DEFAULT_VIDEO = (
    ROOT
    / "person_detect"
    / "clips"
    / MATCH_ID
    / "clip_12m36s_90s.mp4"
)
DEFAULT_CALIBRATION = (
    ROOT
    / "person_detect"
    / "clips"
    / MATCH_ID
    / "camera_alignment.veo"
)
DEFAULT_OUT = (
    ROOT
    / "person_detect"
    / "outputs"
    / "two_lens_view"
)

STACKED_W = 2048
STACKED_H = 2048
LENS_W = 2048
LENS_H = 1024


class TwoLensRenderer:
    """Return the two physical Veo lens images without resampling."""

    view_names = ("left", "right")

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
    ) -> None:
        # Keep the calibration path in the renderer interface so both view
        # implementations are initialized from the same run configuration.
        self.calibration = Path(calibration)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a valid BGR source frame")

        height, width = frame.shape[:2]
        if width != STACKED_W or height != STACKED_H:
            raise ValueError(
                "The validated Veo two-lens source is expected to be "
                f"{STACKED_W}x{STACKED_H}, got {width}x{height}."
            )

    def render(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Split a stacked source frame into independent physical lenses."""
        self._validate_frame(frame)

        # copy() makes each returned view contiguous and independent while
        # preserving the source pixels exactly.
        return {
            "left": frame[:LENS_H, :].copy(),
            "right": frame[LENS_H:, :].copy(),
        }


def read_frame(
    video: str | Path,
    time_s: float,
) -> tuple[np.ndarray, dict]:
    video = Path(video)
    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0.0:
        cap.release()
        raise RuntimeError("Video reports invalid FPS")

    requested_frame = max(0, int(round(float(time_s) * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, requested_frame)

    ok, frame = cap.read()
    actual_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame near t={time_s:.6f}s from {video}"
        )

    return frame, {
        "fps": fps,
        "frame_count": frame_count,
        "requested_time_s": float(time_s),
        "requested_frame": requested_frame,
        "actual_frame": actual_frame,
        "actual_time_s": actual_frame / fps,
    }


def preview(
    views: dict[str, np.ndarray],
    *,
    panel_width: int = 960,
) -> np.ndarray:
    panels = []

    for name in ("left", "right"):
        image = views[name]
        scale = panel_width / image.shape[1]
        panel_height = int(round(image.shape[0] * scale))

        panel = cv2.resize(
            image,
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )

        cv2.putText(
            panel,
            f"PHYSICAL {name.upper()} LENS - ORIGINAL PIXELS",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        panels.append(panel)

    return np.concatenate(panels, axis=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview the two original physical Veo lenses."
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame, frame_info = read_frame(args.video, args.time)
    renderer = TwoLensRenderer(args.calibration)
    views = renderer.render(frame)

    left_path = args.out_dir / "left_original.png"
    right_path = args.out_dir / "right_original.png"
    preview_path = args.out_dir / "preview_left_right.jpg"
    summary_path = args.out_dir / "renderer_summary.json"

    cv2.imwrite(str(left_path), views["left"])
    cv2.imwrite(str(right_path), views["right"])
    cv2.imwrite(str(preview_path), preview(views))

    summary = {
        "video": str(args.video),
        "calibration": str(args.calibration),
        "frame": frame_info,
        "stacked_source_size": [STACKED_W, STACKED_H],
        "lens_size": [LENS_W, LENS_H],
        "views": ["left", "right"],
        "resampling_before_detection": False,
    }

    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))
    print(f"Saved: {preview_path}")


if __name__ == "__main__":
    main()
