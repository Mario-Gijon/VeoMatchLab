#!/usr/bin/env python3
"""Run conservative clip-global offline tracking V2.

Copy this revision over tracking/run_offline_context.py after promoting
offline_context_V2.py to the stable tracking/offline_context.py.

No detector is executed.  The entire detections.jsonl run is loaded first and
the final frame sequence is produced only after clip-global analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess

from tracking.association import AssociationConfig
from tracking.kalman import KalmanConfig
from tracking.observations import load_run_observations
from tracking.offline_context import (
    OfflineContextConfig,
    process_offline_context,
)
from tracking.offline_renderer import OfflinePitchRenderer
from tracking.track_manager import TrackManagerConfig


ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-visible", type=int, default=25)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def _fps(frames) -> float:
    deltas = [
        b.time_s - a.time_s
        for a, b in zip(frames, frames[1:])
        if b.time_s > a.time_s
    ]
    if not deltas:
        return 30.0
    return 1.0 / statistics.median(deltas)


def _output_dir(args, source_config):
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    view = source_config.get("view", "unknown_view")
    model = source_config.get("model", {}).get("name", "unknown_model")
    clip_id = source_config.get("clip_id", "clip")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return (
        ROOT / "tracking" / "offline_outputs"
        / view / model / clip_id / stamp
    )


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _open_writer(path: Path, fps: float):
    command = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24", "-s", "1920x1080",
        "-r", f"{fps:.9f}", "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _tracklet_record(item):
    return {
        "tracklet_id": item.tracklet_id,
        "source_track_id": item.source_track_id,
        "source_confirmed": item.source_confirmed,
        "anchor": item.anchor,
        "start_frame": item.start_frame,
        "end_frame": item.end_frame,
        "start_time_s": item.start_time_s,
        "end_time_s": item.end_time_s,
        "observed_count": item.observed_count,
        "observed_span_s": item.observed_span_s,
        "frame_span": item.frame_span,
        "coverage": item.coverage,
        "mean_confidence": item.mean_confidence,
        "max_endpoint_speed_mps": item.max_endpoint_speed_mps,
        "start_position": [float(v) for v in item.start_position],
        "end_position": [float(v) for v in item.end_position],
        "start_velocity": (
            None if item.start_velocity is None
            else [float(v) for v in item.start_velocity]
        ),
        "end_velocity": (
            None if item.end_velocity is None
            else [float(v) for v in item.end_velocity]
        ),
    }


def _trajectory_record(item):
    return {
        "trajectory_id": item.trajectory_id,
        "tracklet_ids": list(item.tracklet_ids),
        "source_track_ids": list(item.source_track_ids),
        "mean_confidence": item.mean_confidence,
        "observed_count": item.observed_count,
        "observed_span_s": item.observed_span_s,
        "fragment_count": item.fragment_count,
        "max_observed_gap_s": item.max_observed_gap_s,
        "render_eligible": item.render_eligible,
    }


def _frame_track_record(item):
    return {
        "trajectory_id": item.trajectory_id,
        "state": item.state,
        "x": item.x,
        "z": item.z,
        "vx": item.vx,
        "vz": item.vz,
        "speed_mps": (item.vx ** 2 + item.vz ** 2) ** 0.5,
        "support_score": item.support_score,
        "observed_count": item.observed_count,
        "fragment_count": item.fragment_count,
        "mean_confidence": item.mean_confidence,
    }


def main(argv=None) -> int:
    args = _parse_args(argv)
    if args.max_visible < 1:
        raise ValueError("--max-visible must be >= 1")

    observation_run = load_run_observations(args.run_dir)
    output_dir = _output_dir(args, observation_run.config)
    output_dir.mkdir(parents=True, exist_ok=False)

    context_config = OfflineContextConfig(
        max_visible_tracks=args.max_visible,
    )
    manager_config = TrackManagerConfig(
        max_visible_tracks=args.max_visible,
    )
    association_config = AssociationConfig()
    kalman_config = KalmanConfig()

    result = process_offline_context(
        observation_run.frames,
        context_config=context_config,
        manager_config=manager_config,
        association_config=association_config,
        kalman_config=kalman_config,
    )

    tracklets_path = output_dir / "tracklets.json"
    trajectories_path = output_dir / "trajectories.json"
    tracks_path = output_dir / "offline_tracks.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "context_config.json"
    calibration_path = output_dir / "calibration.json"

    _write_json(calibration_path, asdict(result.calibration))
    _write_json(
        tracklets_path,
        [_tracklet_record(item) for item in result.tracklets],
    )
    _write_json(
        trajectories_path,
        [_trajectory_record(item) for item in result.trajectories],
    )

    visible_counts = []
    observed_track_frames = 0
    interpolated_track_frames = 0
    extrapolated_track_frames = 0
    max_rendered_speed = 0.0

    with tracks_path.open("w", encoding="utf-8") as handle:
        for frame in observation_run.frames:
            tracks = result.frame_tracks.get(frame.frame, ())
            visible_counts.append(len(tracks))

            for item in tracks:
                if item.state == "observed":
                    observed_track_frames += 1
                elif item.state == "interpolated":
                    interpolated_track_frames += 1
                elif item.state == "extrapolated":
                    extrapolated_track_frames += 1

                speed = (item.vx ** 2 + item.vz ** 2) ** 0.5
                max_rendered_speed = max(max_rendered_speed, speed)

            handle.write(
                json.dumps(
                    {
                        "frame": frame.frame,
                        "time_s": frame.time_s,
                        "raw_observation_count": len(frame.observations),
                        "visible_trajectory_count": len(tracks),
                        "tracks": [
                            _frame_track_record(item)
                            for item in tracks
                        ],
                    }
                )
                + "\n"
            )

    output_paths = {
        "calibration.json": str(calibration_path),
        "tracklets.json": str(tracklets_path),
        "trajectories.json": str(trajectories_path),
        "offline_tracks.jsonl": str(tracks_path),
        "summary.json": str(summary_path),
        "context_config.json": str(config_path),
    }

    if not args.no_video:
        fps = _fps(observation_run.frames)
        renderer = OfflinePitchRenderer(observation_run.projector)
        result_video = output_dir / "result_offline_2d.mp4"
        debug_video = output_dir / "debug_raw_vs_offline.mp4"

        result_writer = _open_writer(result_video, fps)
        debug_writer = _open_writer(debug_video, fps)

        try:
            for frame in observation_run.frames:
                tracks = result.frame_tracks.get(frame.frame, ())
                result_writer.stdin.write(
                    renderer.render_result(
                        tracks,
                        frame=frame.frame,
                        time_s=frame.time_s,
                    ).tobytes()
                )
                debug_writer.stdin.write(
                    renderer.render_debug(
                        frame.observations,
                        tracks,
                        frame=frame.frame,
                        time_s=frame.time_s,
                    ).tobytes()
                )
        finally:
            for writer in (result_writer, debug_writer):
                if writer.stdin is not None:
                    writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError("ffmpeg failed")

        output_paths["result_offline_2d.mp4"] = str(result_video)
        output_paths["debug_raw_vs_offline.mp4"] = str(debug_video)

    duration = (
        observation_run.frames[-1].time_s
        - observation_run.frames[0].time_s
        if observation_run.frames
        else 0.0
    )

    eligible = [
        item for item in result.trajectories
        if item.render_eligible
    ]
    linked = [
        item for item in result.trajectories
        if item.fragment_count > 1
    ]

    summary = {
        "processed_frames": len(observation_run.frames),
        "processed_duration_s": duration,
        "input_observations": observation_run.observation_count,
        "covariance_calibration": asdict(result.calibration),
        "preliminary_forward": asdict(result.preliminary.stats),
        "calibrated_forward": asdict(result.calibrated.stats),
        "tracklet_count": len(result.tracklets),
        "anchor_tracklet_count": len(result.anchor_tracklets),
        "weak_tracklet_count": len(result.weak_tracklets),
        "anchor_observation_count": sum(
            item.observed_count for item in result.anchor_tracklets
        ),
        "weak_observation_count": sum(
            item.observed_count for item in result.weak_tracklets
        ),
        "link_candidate_count": result.link_result.candidate_count,
        "accepted_tracklet_links": len(result.link_result.links),
        "trajectory_count": len(result.trajectories),
        "linked_trajectory_count": len(linked),
        "render_eligible_trajectory_count": len(eligible),
        "visible_count_min": min(visible_counts) if visible_counts else 0,
        "visible_count_max": max(visible_counts) if visible_counts else 0,
        "visible_count_mean": (
            statistics.fmean(visible_counts) if visible_counts else 0.0
        ),
        "visible_count_median": (
            statistics.median(visible_counts) if visible_counts else 0.0
        ),
        "frames_at_visible_cap": sum(
            value >= args.max_visible
            for value in visible_counts
        ),
        "observed_track_frames": observed_track_frames,
        "interpolated_track_frames": interpolated_track_frames,
        "extrapolated_track_frames": extrapolated_track_frames,
        "maximum_rendered_speed_mps": max_rendered_speed,
        "output_paths": output_paths,
    }

    _write_json(
        config_path,
        {
            "source_run_dir": str(observation_run.run_dir),
            "source_detection_config": observation_run.config,
            "offline_context_config": asdict(context_config),
            "track_manager_config": asdict(manager_config),
            "association_config": asdict(association_config),
            "kalman_config": asdict(kalman_config),
            "output_paths": output_paths,
        },
    )
    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"Offline context V2 output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
