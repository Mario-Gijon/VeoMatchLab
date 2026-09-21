#!/usr/bin/env python3
"""Create a short RF-DETR player-on-pitch tracking diagnostic video.

This is deliberately an experiment: it reads the existing Dual Lens video and
validated camera alignment, but never changes them or application storage.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.player_detection.pitch_tracking import (
    PitchObservation,
    PitchTracker,
    inside_field_detections,
    merge_cross_camera,
)
from experiments.player_detection.run_interval import (
    build_sampling_plan,
    clear_model,
    data_root,
    load_model,
    map_detection,
    match_inputs,
    person_class_ids,
    person_predictions,
    raw_detection_record,
    split_dual_lens,
    write_jsonl,
)
from geometry.veo_metadata import load_veo_metadata


CAMERA_WIDTH = 960
CAMERA_HEIGHT = 480
PITCH_WIDTH = CAMERA_WIDTH * 2
PITCH_HEIGHT = 600


def color_for_id(global_track_id: int | None) -> tuple[int, int, int]:
    """Return a stable, bright BGR colour for one global track ID."""
    if global_track_id is None:
        return (100, 100, 100)
    # A short fixed palette keeps labels readable while avoiding any identity claim.
    palette = (
        (45, 210, 245), (100, 235, 90), (235, 120, 80), (210, 80, 235),
        (65, 180, 255), (255, 190, 120), (170, 120, 255), (125, 240, 210),
    )
    return palette[(global_track_id - 1) % len(palette)]


def annotate_camera(image: np.ndarray, records: Iterable[dict[str, Any]], lens: str) -> np.ndarray:
    """Draw native-lens boxes and global IDs without changing source pixels."""
    import cv2

    annotated = image.copy()
    for record in records:
        x1, y1, x2, y2 = (round(value) for value in record["bbox_xyxy"])
        global_track_id = record.get("global_track_id")
        inside = record.get("inside_field") is True
        color = color_for_id(global_track_id) if inside else (80, 155, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        foot_x, foot_y = (round(value) for value in record["footpoint_native"])
        cv2.circle(annotated, (foot_x, foot_y), 4, color, -1)
        label = f"G{global_track_id} {record['confidence']:.2f}" if global_track_id else f"{record['confidence']:.2f} not projected"
        cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 2, cv2.LINE_AA)
    cv2.putText(annotated, lens.upper(), (18, 32), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.resize(annotated, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)


def pitch_point(metadata: Any, x: float, z: float, *, width: int, height: int, margin: int) -> tuple[int, int]:
    px = round(margin + (x + metadata.field_length / 2) / metadata.field_length * (width - 2 * margin))
    py = round(margin + (metadata.field_width / 2 - z) / metadata.field_width * (height - 2 * margin))
    return px, py


def render_pitch(metadata: Any, timestamp_s: float, observations: Iterable[PitchObservation], counters: dict[str, int]) -> np.ndarray:
    """Render canonical field X/Z observations and only their global IDs."""
    import cv2

    canvas = np.full((PITCH_HEIGHT, PITCH_WIDTH, 3), (35, 120, 55), dtype=np.uint8)
    margin = 48
    white = (242, 242, 242)
    cv2.rectangle(canvas, (margin, margin), (PITCH_WIDTH - margin, PITCH_HEIGHT - margin), white, 2)
    centre_x = PITCH_WIDTH // 2
    centre_y = PITCH_HEIGHT // 2
    cv2.line(canvas, (centre_x, margin), (centre_x, PITCH_HEIGHT - margin), white, 2)
    radius = round((PITCH_HEIGHT - 2 * margin) * 9.15 / metadata.field_width)
    cv2.circle(canvas, (centre_x, centre_y), radius, white, 2)
    for observation in observations:
        global_track_id = observation.global_track_id
        px, py = pitch_point(metadata, observation.x, observation.z, width=PITCH_WIDTH, height=PITCH_HEIGHT, margin=margin)
        color = color_for_id(global_track_id)
        cv2.circle(canvas, (px, py), 9, color, -1)
        cv2.putText(canvas, f"G{global_track_id}", (px + 11, py - 8), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 2, cv2.LINE_AA)
    label = (
        f"t={timestamp_s:.3f}s   left={counters['left']}  right={counters['right']}  "
        f"projected={counters['projected']}  merges={counters['merges']}  active tracks={counters['active_tracks']}"
    )
    cv2.putText(canvas, label, (48, 30), cv2.FONT_HERSHEY_SIMPLEX, .62, white, 2, cv2.LINE_AA)
    return canvas


def render_frame(metadata: Any, timestamp_s: float, left: np.ndarray, right: np.ndarray, records: list[dict[str, Any]], observations: list[PitchObservation], counters: dict[str, int]) -> np.ndarray:
    import cv2

    left_view = annotate_camera(left, (item for item in records if item["lens"] == "left"), "left")
    right_view = annotate_camera(right, (item for item in records if item["lens"] == "right"), "right")
    cameras = cv2.hconcat((left_view, right_view))
    pitch = render_pitch(metadata, timestamp_s, observations, counters)
    return cv2.vconcat((cameras, pitch))


def observation_record(observation: PitchObservation, *, frame_index: int, timestamp_s: float) -> dict[str, Any]:
    return {
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "global_track_id": observation.global_track_id,
        "source": observation.source,
        "world_x": observation.x,
        "world_z": observation.z,
        "source_detections": [
            {
                "lens": detection.lens,
                "confidence": detection.source_record["confidence"],
                "bbox_xyxy": detection.source_record["bbox_xyxy"],
                "raw_footpoint": detection.source_record["raw_footpoint"],
            }
            for detection in observation.detections
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match", default="match_fc5f185c7710", help="Stored match identifier")
    parser.add_argument("--start", type=float, default=750.0, help="Start time in source-video seconds")
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument("--sample-fps", type=float, default=2.0, help="Diagnostic frames per source-video second")
    parser.add_argument("--threshold", type=float, default=.25)
    parser.add_argument("--cross-camera-distance", type=float, default=1.25, help="Maximum left/right merge distance in field metres")
    parser.add_argument("--track-distance", type=float, default=2.0, help="Maximum pitch association distance in field metres")
    parser.add_argument("--max-missed", type=int, default=2, help="Number of sampled frames a track may miss")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start < 0 or args.duration <= 0 or args.sample_fps <= 0:
        raise ValueError("--start must be non-negative and duration/sample-fps must be positive")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if args.cross_camera_distance < 0 or args.track_distance < 0 or args.max_missed < 0:
        raise ValueError("distance and --max-missed values must be non-negative")

    _match_dir, video_path, calibration_path = match_inputs(args.match)
    import cv2

    source = cv2.VideoCapture(str(video_path))
    source_fps = float(source.get(cv2.CAP_PROP_FPS) or 0.0)
    source.release()
    plan = build_sampling_plan(args.start, args.duration, args.sample_fps, source_fps)
    if not plan:
        raise RuntimeError("Sampling plan is empty")
    metadata = load_veo_metadata(calibration_path)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.match}_pitch-debug_start-{args.start:g}_duration-{args.duration:g}_fps-{args.sample_fps:g}_{timestamp}"
    output_dir = data_root() / "experiments" / "player_detection" / run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "frames").mkdir()

    manifest = {
        "run_id": run_id,
        "match_id": args.match,
        "video": str(video_path),
        "calibration": str(calibration_path),
        "model": "large",
        "interval": {"start_s": args.start, "duration_s": args.duration, "source_fps": source_fps},
        "sample_fps": args.sample_fps,
        "threshold": args.threshold,
        "cross_camera_distance_m": args.cross_camera_distance,
        "track_distance_m": args.track_distance,
        "max_missed_sampled_frames": args.max_missed,
        "sample_plan": [{"timestamp_s": sample.timestamp_s, "frame_index": sample.frame_index} for sample in plan],
    }
    (output_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    model = load_model("large")
    person_ids = person_class_ids(model)
    tracker = PitchTracker(args.track_distance, args.max_missed)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        clear_model(model)
        raise RuntimeError(f"Could not open Dual Lens video: {video_path}")
    writer: cv2.VideoWriter | None = None
    all_records: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    total_merges = 0
    inference_seconds = 0.0
    try:
        for sample in plan:
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            decoded, stacked = capture.read()
            if not decoded or stacked is None:
                raise RuntimeError(f"Could not decode source frame {sample.frame_index}")
            left, right = split_dual_lens(stacked)
            frame_records: list[dict[str, Any]] = []
            for lens, image in (("left", left), ("right", right)):
                started = time.perf_counter()
                predictions = person_predictions(model, image, args.threshold, person_ids)
                inference_seconds += time.perf_counter() - started
                for bbox, confidence, class_id, class_name in predictions:
                    record = raw_detection_record(
                        timestamp_s=sample.timestamp_s, frame_index=sample.frame_index, lens=lens,
                        confidence=confidence, bbox_xyxy=bbox, model="large", class_id=class_id, class_name=class_name,
                    )
                    map_detection(record, metadata)
                    frame_records.append(record)
                    all_records.append(record)

            projected = inside_field_detections(frame_records)
            observations = merge_cross_camera(projected, args.cross_camera_distance)
            tracker.update(observations, frame_index=sample.frame_index, timestamp_s=sample.timestamp_s)
            for observation in observations:
                for detection in observation.detections:
                    detection.source_record["global_track_id"] = observation.global_track_id
                    detection.source_record["observation_source"] = observation.source
                all_observations.append(observation_record(observation, frame_index=sample.frame_index, timestamp_s=sample.timestamp_s))
            merges = sum(observation.source == "both" for observation in observations)
            total_merges += merges
            counters = {
                "left": sum(record["lens"] == "left" for record in frame_records),
                "right": sum(record["lens"] == "right" for record in frame_records),
                "projected": len(projected),
                "merges": merges,
                "active_tracks": len(tracker.tracks),
            }
            combined = render_frame(metadata, sample.timestamp_s, left, right, frame_records, observations, counters)
            if writer is None:
                writer = cv2.VideoWriter(str(output_dir / "pitch_debug.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), args.sample_fps, (combined.shape[1], combined.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Could not create pitch_debug.mp4")
            writer.write(combined)
            frame_path = output_dir / "frames" / f"frame_{sample.frame_index:08d}_{sample.timestamp_s:010.3f}.jpg"
            if not cv2.imwrite(str(frame_path), combined):
                raise RuntimeError(f"Could not write debug frame {frame_path}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        clear_model(model)

    write_jsonl(output_dir / "detections.jsonl", all_records)
    write_jsonl(output_dir / "observations.jsonl", all_observations)
    summary = {
        **manifest,
        "sampled_frames": len(plan),
        "total_raw_person_detections": len(all_records),
        "total_projected_detections": sum(item.get("inside_field") is True for item in all_records),
        "total_cross_camera_merges": total_merges,
        "active_tracks_at_end": len(tracker.tracks),
        "inference_elapsed_time_s": inference_seconds,
        "average_inference_ms_per_lens_image": inference_seconds * 1000 / (len(plan) * 2),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output_dir)


if __name__ == "__main__":
    main()
