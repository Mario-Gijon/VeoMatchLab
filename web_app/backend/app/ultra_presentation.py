"""Render the visual Ultra presentation at source-video cadence only."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from experiments.virtual_camera_coverage.coverage import SequentialFrameReader
from experiments.virtual_camera_coverage.ultra_panorama_overlay import (
    build_ultra_panorama_maps,
    render_ultra_panorama,
)
from geometry.veo_metadata import load_veo_metadata


@dataclass(frozen=True)
class UltraPresentationStats:
    fps: float
    frames: int
    render_seconds: float
    encode_seconds: float


def source_frame_indices(start_s: float, duration_s: float, fps: float, frame_count: int | None = None) -> range:
    """Physical source-frame indices in the half-open presentation interval."""
    if not (fps > 0.0 and duration_s > 0.0):
        return range(0)
    first = max(0, int(math.ceil(start_s * fps - 1e-9)))
    end = max(first, int(math.ceil((start_s + duration_s) * fps - 1e-9)))
    if frame_count is not None:
        end = min(end, max(0, frame_count))
    return range(first, end)


def render_ultra_presentation(
    *,
    video: Path,
    calibration: Path,
    output: Path,
    start_s: float,
    duration_s: float,
    check_cancelled: Callable[[], None] | None = None,
) -> UltraPresentationStats:
    """Render every available source frame; this never invokes detection/tracking."""
    metadata = load_veo_metadata(calibration)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    maps = None
    render_seconds = 0.0
    encode_seconds = 0.0
    frames = 0
    fps = 0.0
    try:
        with SequentialFrameReader(video) as reader:
            fps = reader.fps
            frame_count = int(reader.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if reader.cap is not None else None
            for frame_index in source_frame_indices(start_s, duration_s, fps, frame_count):
                if check_cancelled is not None:
                    check_cancelled()
                frame, _ = reader.read(frame_index / fps)
                started = time.perf_counter()
                if maps is None:
                    maps = build_ultra_panorama_maps(frame, metadata)
                panorama = render_ultra_panorama(frame, metadata, maps)
                render_seconds += time.perf_counter() - started
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                        (panorama.shape[1], panorama.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not open ultra panorama video writer")
                started = time.perf_counter()
                writer.write(panorama)
                encode_seconds += time.perf_counter() - started
                frames += 1
    finally:
        if writer is not None:
            writer.release()
    if frames == 0:
        raise RuntimeError("Ultra presentation contained no source frames")
    return UltraPresentationStats(fps=fps, frames=frames, render_seconds=render_seconds, encode_seconds=encode_seconds)
