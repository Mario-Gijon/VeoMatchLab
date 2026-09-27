#!/usr/bin/env python3
"""Run Tracking V1 over six-view, same-frame-associated person measurements."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tracking.person_measurement_adapter import (  # noqa: E402
    to_tracking_measurement,
)
from experiments.virtual_camera_coverage.association import (  # noqa: E402
    DEFAULT_FIELD_MARGIN_M,
    DEFAULT_MAX_CENTER_X_NORM,
    DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
    DEFAULT_MAX_SOURCE_ANCHOR_NORM,
    DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
    DEFAULT_MAX_WORLD_DISTANCE_M,
    DEFAULT_MIN_HORIZONTAL_IOM,
    DEFAULT_MIN_SOURCE_IOM,
    DEFAULT_MIN_SOURCE_IOU,
    DEFAULT_MIN_VERTICAL_IOM,
    Observation,
    associate_observations,
    build_observation,
)
from experiments.virtual_camera_coverage.coverage import (  # noqa: E402
    DEFAULT_FAR_CAMERAS,
    DEFAULT_GRID_STEP_M,
    DEFAULT_GUARD_M,
    DEFAULT_NEAR_CAMERAS,
    DEFAULT_SPLIT_FRACTION,
    build_map,
    derive_specs,
    render_map,
    SequentialFrameReader,
)
from experiments.virtual_camera_coverage.detect import (  # noqa: E402
    DEFAULT_C2_WEIGHTS,
    DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M,
    confirm_deferred_rescues_from_owner_images,
    ordered_tight_specs,
    predict_persons,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.models.rfdetr_2xl import RFDETR2XLBackend  # noqa: E402
from tracking import GlobalTracker, PositionKind, TrackLifecycle  # noqa: E402


DEFAULT_OUT_DIR = Path("experiments/tracking/output")
# Tracking V1 diagnostics use the agreed C2 operating threshold.  Keep this
# local to the tracking experiment: detect.py remains an independent
# same-frame-association experiment with its own CLI defaults.
DEFAULT_TRACKING_THRESHOLD = 0.40


@dataclass
class PipelineProfile:
    frame_read_seconds: float = 0.0
    virtual_camera_render_seconds: float = 0.0
    detector_base_seconds: float = 0.0
    rescue_detector_seconds: float = 0.0
    association_seconds: float = 0.0
    tracker_seconds: float = 0.0
    ultra_render_seconds: float = 0.0
    video_encode_seconds: float = 0.0
    persist_seconds: float = 0.0
    total_seconds: float = 0.0
    samples: int = 0
    base_detector_calls: int = 0
    rescue_detector_calls: int = 0
    measurements: int = 0
    tracks: int = 0

    def as_dict(self) -> dict[str, float | int]:
        return {
            "frame_read_seconds": self.frame_read_seconds,
            "virtual_camera_render_seconds": self.virtual_camera_render_seconds,
            "detector_base_seconds": self.detector_base_seconds,
            "rescue_detector_seconds": self.rescue_detector_seconds,
            "rescue_detector_mean_seconds": self.rescue_detector_seconds / self.rescue_detector_calls if self.rescue_detector_calls else 0.0,
            "association_seconds": self.association_seconds,
            "tracker_seconds": self.tracker_seconds,
            "ultra_render_seconds": self.ultra_render_seconds,
            "video_encode_seconds": self.video_encode_seconds,
            "persist_seconds": self.persist_seconds,
            "job_total_seconds": self.total_seconds,
            "samples": self.samples,
            "base_detector_calls": self.base_detector_calls,
            "rescue_detector_calls": self.rescue_detector_calls,
            "measurements": self.measurements,
            "tracks": self.tracks,
        }

    def print_compact(self) -> None:
        print(
            "[clip-profile] "
            f"samples={self.samples} frame_read={self.frame_read_seconds:.3f} "
            f"virtual_cameras={self.virtual_camera_render_seconds:.3f} "
            f"detector_base={self.detector_base_seconds:.3f} detector_base_calls={self.base_detector_calls} "
            f"detector_rescue={self.rescue_detector_seconds:.3f} detector_rescue_calls={self.rescue_detector_calls} "
            f"rescue_mean={self.rescue_detector_seconds / self.rescue_detector_calls if self.rescue_detector_calls else 0.0:.3f} "
            f"association={self.association_seconds:.3f} tracker={self.tracker_seconds:.3f} "
            f"ultra={self.ultra_render_seconds:.3f} encoding={self.video_encode_seconds:.3f} "
            f"persist={self.persist_seconds:.3f} measurements={self.measurements} tracks={self.tracks} total={self.total_seconds:.3f}"
        )


def _snapshot_dict(snapshot: Any) -> dict[str, Any]:
    return {
        "track_id": snapshot.track_id,
        "x": snapshot.x,
        "z": snapshot.z,
        "vx": snapshot.vx,
        "vz": snapshot.vz,
        "lifecycle": snapshot.lifecycle.value,
        "position_kind": snapshot.position_kind.value,
        "measurement_id": snapshot.measurement_id,
        "missed_seconds": snapshot.missed_seconds,
        "hits": snapshot.hits,
    }


def sample_timestamps(start_s: float, duration_s: float, period_s: float) -> list[float]:
    """Return samples in the half-open interval [start_s, start_s + duration_s)."""
    if duration_s <= 0.0 or period_s <= 0.0:
        return []
    count = int(math.ceil(duration_s / period_s - 1e-12))
    return [start_s + index * period_s for index in range(count)]


# Private compatibility name for existing experiment callers.
_timestamps = sample_timestamps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--weights", type=Path, default=DEFAULT_C2_WEIGHTS)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=25.0)
    parser.add_argument("--sample-period-s", type=float, default=1.0)
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_TRACKING_THRESHOLD,
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--split-fraction", type=float, default=DEFAULT_SPLIT_FRACTION)
    parser.add_argument("--guard-margin-m", type=float, default=DEFAULT_GUARD_M)
    parser.add_argument("--grid-step-m", type=float, default=DEFAULT_GRID_STEP_M)
    parser.add_argument("--field-margin-m", type=float, default=DEFAULT_FIELD_MARGIN_M)
    parser.add_argument(
        "--rescue-confirm-threshold",
        type=float,
        default=DEFAULT_TRACKING_THRESHOLD,
    )
    parser.add_argument("--rescue-confirm-max-world-distance-m", type=float, default=DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    if not args.video.is_file() or not args.calibration.is_file() or not args.weights.is_file():
        parser.error("--video, --calibration, and --weights must name existing files")
    if args.start_s < 0 or args.duration_s <= 0 or args.sample_period_s <= 0:
        parser.error("--start-s must be >= 0; --duration-s and --sample-period-s must be > 0")
    if not 0.0 <= args.threshold <= 1.0 or not 0.0 <= args.rescue_confirm_threshold <= 1.0:
        parser.error("detection thresholds must be in [0, 1]")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    return args


def run(
    args: argparse.Namespace,
    *,
    frame_callback: Callable[..., None] | None = None,
    measurement_callback: Callable[..., None] | None = None,
    analysis_callback: Callable[..., None] | None = None,
    profile: PipelineProfile | None = None,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    profile = profile or PipelineProfile()
    metadata = load_veo_metadata(args.calibration)
    specs, layout, _, _ = derive_specs(
        metadata, args.width, args.height, split_fraction=args.split_fraction,
        near_count=DEFAULT_NEAR_CAMERAS, far_count=DEFAULT_FAR_CAMERAS,
        guard_margin_m=args.guard_margin_m, grid_step_m=args.grid_step_m,
    )
    tight_specs = ordered_tight_specs(specs)
    spec_by_name = {spec.name: spec for spec in tight_specs}
    maps = {spec.name: build_map(metadata, spec, args.width, args.height) for spec in tight_specs}
    backend = RFDETR2XLBackend(args.weights, device=args.device)
    print(f"[clip-analysis] detector={backend.name} device={backend.device} weights={args.weights}")
    tracker = GlobalTracker()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    frames_path = args.out_dir / "tracking_frames.jsonl"
    total_measurements = 0
    observed_samples = 0
    predicted_samples = 0
    max_confirmed = 0
    confirmed_ids: set[int] = set()
    skipped_no_position = 0

    with frames_path.open("w", encoding="utf-8") as output, SequentialFrameReader(args.video) as frame_reader:
        for requested_time in _timestamps(args.start_s, args.duration_s, args.sample_period_s):
            started = time.perf_counter()
            frame, frame_info = frame_reader.read(requested_time)
            profile.frame_read_seconds += time.perf_counter() - started
            frame_index = int(frame_info["actual_frame"])
            timestamp = float(frame_info["actual_time_s"])
            rendered: dict[str, np.ndarray] = {}
            observations: list[Observation] = []
            observation_id = 1

            for spec in tight_specs:
                started = time.perf_counter()
                image = render_map(frame, maps[spec.name])
                profile.virtual_camera_render_seconds += time.perf_counter() - started
                rendered[spec.name] = image
                started = time.perf_counter()
                detections = predict_persons(backend, image, threshold=args.threshold)
                profile.detector_base_seconds += time.perf_counter() - started
                profile.base_detector_calls += 1
                for detection in detections:
                    observations.append(build_observation(
                        observation_id=observation_id, frame_number=frame_index,
                        time_s=timestamp, metadata=metadata, layout=layout, spec=spec,
                        camera_map=maps[spec.name], xyxy=detection.xyxy,
                        confidence=detection.confidence, field_margin_m=args.field_margin_m,
                    ))
                    observation_id += 1

            started = time.perf_counter()
            association = associate_observations(
                observations, layout,
                min_source_iou=DEFAULT_MIN_SOURCE_IOU,
                max_source_anchor_norm=DEFAULT_MAX_SOURCE_ANCHOR_NORM,
                max_world_distance_m=DEFAULT_MAX_WORLD_DISTANCE_M,
                min_source_iom=DEFAULT_MIN_SOURCE_IOM,
                min_horizontal_iom=DEFAULT_MIN_HORIZONTAL_IOM,
                min_vertical_iom=DEFAULT_MIN_VERTICAL_IOM,
                max_center_x_norm=DEFAULT_MAX_CENTER_X_NORM,
                max_containment_world_distance_m=DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
                max_truncated_world_distance_m=DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
            )
            profile.association_seconds += time.perf_counter() - started

            def predict_rescue(model: Any, image: np.ndarray) -> list[Any]:
                profile.rescue_detector_calls += 1
                started_prediction = time.perf_counter()
                try:
                    return predict_persons(model, image, threshold=args.rescue_confirm_threshold)
                finally:
                    profile.rescue_detector_seconds += time.perf_counter() - started_prediction

            rescue_detector_before = profile.rescue_detector_seconds
            started = time.perf_counter()
            association, _, _ = confirm_deferred_rescues_from_owner_images(
                backend=backend, metadata=metadata, layout=layout, rendered=rendered,
                maps=maps, spec_by_name=spec_by_name, association=association,
                frame_number=frame_index, time_s=timestamp,
                field_margin_m=args.field_margin_m,
                threshold=args.rescue_confirm_threshold,
                max_world_distance_m=args.rescue_confirm_max_world_distance_m,
                predict=predict_rescue,
            )
            profile.association_seconds += max(0.0, time.perf_counter() - started - (profile.rescue_detector_seconds - rescue_detector_before))
            observations_by_id = {item.observation_id: item for item in association.observations}
            measurements = []
            for person in association.measurements:
                measurement = to_tracking_measurement(
                    person, frame_index=frame_index, timestamp=timestamp,
                    observations_by_id=observations_by_id,
                )
                if measurement is None:
                    skipped_no_position += 1
                else:
                    measurements.append(measurement)

            if measurement_callback is not None:
                measurement_callback(
                    frame_index=frame_index,
                    timestamp=timestamp,
                    measurements=tuple(measurements),
                )

            started = time.perf_counter()
            tracking_frame = tracker.update(measurements, timestamp=timestamp, frame_index=frame_index)
            profile.tracker_seconds += time.perf_counter() - started
            if analysis_callback is not None:
                analysis_callback(
                    frame_index=frame_index, timestamp=timestamp, association=association,
                    measurements=tuple(measurements), tracking_frame=tracking_frame,
                )
            total_measurements += len(measurements)
            profile.samples += 1
            profile.measurements += len(measurements)
            profile.tracks += len(tracking_frame.tracks)
            observed_samples += sum(sample.position_kind is PositionKind.OBSERVED for sample in tracking_frame.tracks)
            predicted_samples += sum(sample.position_kind is PositionKind.PREDICTED for sample in tracking_frame.tracks)
            confirmed = [sample for sample in tracking_frame.tracks if sample.is_confirmed]
            confirmed_ids.update(sample.track_id for sample in confirmed)
            max_confirmed = max(max_confirmed, len(confirmed))
            record = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "measurement_count": len(measurements),
                "skipped_measurements_without_position": len(association.measurements) - len(measurements),
                "tracks": [_snapshot_dict(sample) for sample in tracking_frame.tracks],
            }
            if frame_callback is not None:
                frame_callback(
                    frame=frame,
                    metadata=metadata,
                    spec_by_name=spec_by_name,
                    tracking_frame=tracking_frame,
                    measurements=tuple(measurements),
                    record=record,
                )
            started = time.perf_counter()
            output.write(json.dumps(record, allow_nan=False) + "\n")
            profile.persist_seconds += time.perf_counter() - started
            print(f"frame={frame_index} t={timestamp:.3f}s measurements={len(measurements)} tracks={len(tracking_frame.tracks)}")

    lifecycles = [tracker.track_lifecycle(track_id) for track_id in tracker.trajectories()]
    summary = {
        "frames_processed": len(_timestamps(args.start_s, args.duration_s, args.sample_period_s)),
        "measurements_processed": total_measurements,
        "tracks_created": tracker.next_track_id - 1,
        "tracks_confirmed": len(confirmed_ids),
        "tracks_lost": sum(item is TrackLifecycle.LOST for item in lifecycles),
        "maximum_simultaneous_confirmed_tracks": max_confirmed,
        "observed_samples": observed_samples,
        "predicted_samples": predicted_samples,
        "skipped_measurements_without_position": skipped_no_position,
        "detector_inference_seconds": profile.detector_base_seconds,
        "sample_period_seconds": args.sample_period_s,
    }
    profile.total_seconds = time.perf_counter() - started_total
    summary["profile"] = profile.as_dict()
    summary_path = args.out_dir / "tracking_summary.json"
    started = time.perf_counter()
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    profile.persist_seconds += time.perf_counter() - started
    print(json.dumps(summary, indent=2))
    profile.print_compact()
    print(f"frames: {frames_path}")
    print(f"summary: {summary_path}")
    return summary


if __name__ == "__main__":
    run(parse_args())
