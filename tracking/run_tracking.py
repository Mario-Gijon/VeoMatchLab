#!/usr/bin/env python3
"""Run the validated tracking V1 modules on an existing detector run."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from tracking.association import AssociationConfig
from tracking.kalman import KalmanConfig
from tracking.observations import load_run_observations
from tracking.renderer import PitchRenderer
from tracking.track_manager import TrackManager, TrackManagerConfig


ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-visible", type=int, default=25)
    return parser.parse_args(argv)


def _json_default(value):
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _open_writer(path: Path, fps: float):
    command = [
        "ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "1920x1080", "-r", f"{fps:.9f}", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _fps(frames) -> float:
    deltas = [b.time_s - a.time_s for a, b in zip(frames, frames[1:]) if b.time_s > a.time_s]
    if not deltas:
        return 30.0
    return 1.0 / statistics.median(deltas)


def _raw_record(observation):
    return {
        "detection_index": observation.detection_index,
        "confidence": observation.confidence,
        "world_x": observation.world_x,
        "world_z": observation.world_z,
        "foot_u": observation.foot_u,
        "foot_v": observation.foot_v,
        "measurement_std_x_m": observation.quality.std_x_m,
        "measurement_std_z_m": observation.quality.std_z_m,
    }


def _track_record(track):
    return {
        "track_id": track.track_id,
        "lifecycle": track.lifecycle,
        "visible": track.visible,
        "reliability": track.reliability,
        "frame_state": track.frame_state,
        "x": track.x,
        "z": track.z,
        "vx": track.vx,
        "vz": track.vz,
        "speed_mps": track.speed_mps,
        "hits": track.hits,
        "misses": track.misses,
        "consecutive_misses": track.consecutive_misses,
        "seconds_since_observation": track.seconds_since_observation,
        "confidence_ema": track.confidence_ema,
    }


def _output_dir(args, source_config):
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    view = source_config.get("view", "unknown_view")
    model = source_config.get("model", {}).get("name", "unknown_model")
    clip_id = source_config.get("clip_id", "clip")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return ROOT / "tracking" / "outputs" / view / model / clip_id / stamp


def run_tracking(args) -> Path:
    if args.max_visible < 1:
        raise ValueError("--max-visible must be >= 1")
    observation_run = load_run_observations(args.run_dir)
    output_dir = _output_dir(args, observation_run.config)
    output_dir.mkdir(parents=True, exist_ok=False)

    manager_config = TrackManagerConfig(max_visible_tracks=args.max_visible)
    kalman_config = KalmanConfig()
    association_config = AssociationConfig()
    manager = TrackManager(
        config=manager_config,
        kalman_config=kalman_config,
        association_config=association_config,
    )
    renderer = PitchRenderer(observation_run.projector)
    fps = _fps(observation_run.frames)
    result_writer = _open_writer(output_dir / "result_2d.mp4", fps)
    debug_writer = _open_writer(output_dir / "debug_raw_vs_tracked.mp4", fps)
    tracks_path = output_dir / "tracks.jsonl"
    frame_records = []
    visible_counts = []
    max_speed = 0.0
    totals = {key: 0 for key in (
        "input_observations", "total_assignments", "total_unmatched_observations",
        "total_tracks_created", "total_confirmations", "total_reactivations",
        "total_lost_transitions", "total_retirements",
    )}

    try:
        with tracks_path.open("w", encoding="utf-8") as handle:
            for frame in observation_run.frames:
                result = manager.process_frame(
                    frame=frame.frame,
                    time_s=frame.time_s,
                    observations=frame.observations,
                )
                record = {
                    "frame": frame.frame,
                    "time_s": frame.time_s,
                    "observation_count": len(frame.observations),
                    "visible_track_count": len(result.visible_tracks),
                    "assignment_count": len(result.assignments),
                    "unmatched_observation_indices": list(result.unmatched_observation_indices),
                    "new_track_ids": list(result.new_track_ids),
                    "newly_confirmed_track_ids": list(result.newly_confirmed_track_ids),
                    "reactivated_track_ids": list(result.reactivated_track_ids),
                    "newly_lost_track_ids": list(result.newly_lost_track_ids),
                    "retired_track_ids": list(result.retired_track_ids),
                    "raw_observations": [_raw_record(item) for item in frame.observations],
                    "tracks": [_track_record(item) for item in result.all_active_tracks],
                }
                handle.write(json.dumps(record, default=_json_default) + "\n")
                frame_records.append(record)
                visible_counts.append(len(result.visible_tracks))
                totals["input_observations"] += len(frame.observations)
                totals["total_assignments"] += len(result.assignments)
                totals["total_unmatched_observations"] += len(result.unmatched_observation_indices)
                totals["total_tracks_created"] += len(result.new_track_ids)
                totals["total_confirmations"] += len(result.newly_confirmed_track_ids)
                totals["total_reactivations"] += len(result.reactivated_track_ids)
                totals["total_lost_transitions"] += len(result.newly_lost_track_ids)
                totals["total_retirements"] += len(result.retired_track_ids)
                speeds = [item.speed_mps for item in result.all_active_tracks]
                if speeds:
                    max_speed = max(max_speed, *speeds)
                result_writer.stdin.write(renderer.render_result(result.visible_tracks, frame=frame.frame, time_s=frame.time_s).tobytes())
                debug_writer.stdin.write(renderer.render_debug(frame.observations, result.visible_tracks, frame=frame.frame, time_s=frame.time_s).tobytes())
    finally:
        for writer in (result_writer, debug_writer):
            if writer.stdin is not None:
                writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError("ffmpeg failed while writing tracking output")

    source_config_path = observation_run.run_dir / "config.json"
    tracking_config = {
        "source_run_dir": str(observation_run.run_dir),
        "source_detection_config": observation_run.config,
        "track_manager_config": asdict(manager_config),
        "association_config": asdict(association_config),
        "kalman_config": asdict(kalman_config),
        "output_paths": {name: str(output_dir / name) for name in (
            "tracks.jsonl", "summary.json", "result_2d.mp4", "debug_raw_vs_tracked.mp4",
        )},
    }
    _write_json(output_dir / "tracking_config.json", tracking_config)
    duration = observation_run.frames[-1].time_s - observation_run.frames[0].time_s if observation_run.frames else 0.0
    visible_total = sum(visible_counts)
    summary = {
        "processed_frames": len(observation_run.frames),
        "processed_duration_s": duration,
        **totals,
        "visible_count_min": min(visible_counts) if visible_counts else 0,
        "visible_count_max": max(visible_counts) if visible_counts else 0,
        "visible_count_mean": statistics.mean(visible_counts) if visible_counts else 0.0,
        "visible_count_median": statistics.median(visible_counts) if visible_counts else 0.0,
        "maximum_visible_count_observed": max(visible_counts) if visible_counts else 0,
        "observed_visible_track_frames": sum(
            sum(item["frame_state"] == "observed" for item in record["tracks"] if item["visible"])
            for record in frame_records
        ),
        "predicted_visible_track_frames": sum(
            sum(item["frame_state"] == "predicted" for item in record["tracks"] if item["visible"])
            for record in frame_records
        ),
        "maximum_reported_track_speed": max_speed,
        "output_paths": tracking_config["output_paths"],
        "source_config_path": str(source_config_path),
        "fps": fps,
    }
    _write_json(output_dir / "summary.json", summary)
    return output_dir


def main(argv=None) -> int:
    output = run_tracking(parse_args(argv))
    print(f"Tracking output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
