"""High-resolution calibrated panorama renderer for Veo stacked footage.

The maps in this renderer are the same maps used by the validated panorama
reproduction in ``geometry.reproduce_veo_panorama``.  They are built once per
renderer, then reused for every decoded video frame.
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
    build_output_world_geometry,
    project_world_target_to_lens,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402

MATCH_ID = "match_629890153a11"
DEFAULT_VIDEO = ROOT / "person_detect" / "clips" / MATCH_ID / "clip_12m36s_90s.mp4"
DEFAULT_CALIBRATION = ROOT / "person_detect" / "clips" / MATCH_ID / "camera_alignment.veo"
DEFAULT_OUT = ROOT / "person_detect" / "outputs" / "ultra_panorama_view" / "diagnostics"

SOURCE_W = 2048
SOURCE_H = 2048
DEFAULT_HEIGHT = 1024
MAX_WIDTH = 4096


def default_panorama_size(metadata, *, height: int = DEFAULT_HEIGHT) -> tuple[int, int]:
    """Return the source-faithful panorama dimensions for calibration."""
    if height <= 0:
        raise ValueError("height must be positive")
    va = metadata.va_presentation
    horizontal_span = float(va["right"] - va["left"])
    vertical_span = float(va["top"] - va["bottom"])
    if horizontal_span <= 0.0 or vertical_span <= 0.0:
        raise ValueError("Calibration presentation range must have positive spans")
    width = round(height * horizontal_span / vertical_span)
    return min(MAX_WIDTH, max(1, width)), int(height)


class PanoramaRenderer:
    """Render one calibrated panorama from each stacked source frame."""

    view_names = ("panorama",)

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
        *,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        self.calibration = Path(calibration)
        self.metadata = load_veo_metadata(self.calibration)

        if width is None and height is None:
            self.width, self.height = default_panorama_size(self.metadata)
        else:
            self.height = int(DEFAULT_HEIGHT if height is None else height)
            self.width = int(
                default_panorama_size(self.metadata, height=self.height)[0]
                if width is None
                else width
            )
            if self.width <= 0 or self.height <= 0:
                raise ValueError("Panorama width and height must be positive")

        target_world, self.blend = build_output_world_geometry(
            self.metadata,
            self.width,
            self.height,
        )

        left_x, left_y, left_valid = project_world_target_to_lens(
            target_world,
            self.metadata.left,
            self.metadata.camera2world,
            SOURCE_W,
            SOURCE_H,
            True,
        )
        right_x, right_y, right_valid = project_world_target_to_lens(
            target_world,
            self.metadata.right,
            self.metadata.camera2world,
            SOURCE_W,
            SOURCE_H,
            False,
        )

        self.left_map_x = left_x.reshape(self.height, self.width)
        self.left_map_y = left_y.reshape(self.height, self.width)
        self.right_map_x = right_x.reshape(self.height, self.width)
        self.right_map_y = right_y.reshape(self.height, self.width)
        self.left_valid = left_valid.reshape(self.height, self.width)
        self.right_valid = right_valid.reshape(self.height, self.width)

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Expected a valid BGR source frame")
        height, width = frame.shape[:2]
        if width != SOURCE_W or height != SOURCE_H:
            raise ValueError(
                "The validated Veo panorama source is expected to be "
                f"{SOURCE_W}x{SOURCE_H}, got {width}x{height}."
            )

    def render(self, frame: np.ndarray) -> dict[str, np.ndarray]:
        """Remap and blend one stacked BGR source frame."""
        self._validate_frame(frame)

        left = cv2.remap(
            frame,
            self.left_map_x,
            self.left_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        right = cv2.remap(
            frame,
            self.right_map_x,
            self.right_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        blend = self.blend[..., None].astype(np.float32)
        panorama = np.clip(
            left.astype(np.float32) * (1.0 - blend)
            + right.astype(np.float32) * blend,
            0.0,
            255.0,
        ).astype(np.uint8)
        return {"panorama": panorama}

    def summary(self) -> dict:
        va = self.metadata.va_presentation
        return {
            "calibration": str(self.calibration),
            "panorama_size": [self.width, self.height],
            "source_stacked_size": [SOURCE_W, SOURCE_H],
            "presentation_range": {
                "left": float(va["left"]),
                "right": float(va["right"]),
                "bottom": float(va["bottom"]),
                "top": float(va["top"]),
            },
            "left_valid_fraction": float(np.mean(self.left_valid)),
            "right_valid_fraction": float(np.mean(self.right_valid)),
        }


def read_frame(video: str | Path, time_s: float) -> tuple[np.ndarray, dict]:
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
        raise RuntimeError(f"Could not read frame near t={time_s:.6f}s from {video}")
    return frame, {
        "fps": fps,
        "frame_count": frame_count,
        "requested_time_s": float(time_s),
        "requested_frame": requested_frame,
        "actual_frame": actual_frame,
        "actual_time_s": actual_frame / fps,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a calibrated Veo panorama.")
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--time", type=float, default=10.0)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame, frame_info = read_frame(args.video, args.time)
    renderer = PanoramaRenderer(
        args.calibration,
        width=args.width,
        height=args.height,
    )
    panorama = renderer.render(frame)["panorama"]
    output = args.out_dir / "panorama.png"
    summary = {
        "video": str(args.video),
        "frame": frame_info,
        **renderer.summary(),
    }
    cv2.imwrite(str(output), panorama)
    (args.out_dir / "renderer_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"Saved: {output}")


UltraPanoramaRenderer = PanoramaRenderer


if __name__ == "__main__":
    main()
