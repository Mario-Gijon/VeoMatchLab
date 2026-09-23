#!/usr/bin/env python3
"""Run VeoMatchLab Offline Tracking V3 with clip-global identity resolution.

This command does NOT run RF-DETR.  It reuses an existing detector run:

1. Offline Context V2 proposal generation.
2. V3 duplicate-branch resolution using independent raw-detection evidence.
3. V3 global path-cover linking with boundary-aware birth/death costs.
4. Final 2D reconstruction after the full clip has been analysed.

The stable project modules are imported:
    tracking.offline_context
    tracking.offline_renderer
    tracking.global_identity

Promote ``global_identity.py`` as a new stable module, keep this runner as the
versioned V3 entry point.
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
from tracking.global_identity import (
    DuplicateEvidence,
    GlobalIdentityConfig,
    IdentityLinkCandidate,
    solve_global_identities,
)
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
    parser.add_argument(
        "--evaluation-tail-s",
        type=float,
        default=20.0,
        help="Report a separate clean-window summary for the final N seconds.",
    )
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
        ROOT
        / "tracking"
        / "offline_outputs_v3"
        / view
        / model
        / clip_id
        / stamp
    )


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _open_writer(path: Path, fps: float):
    command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        "1920x1080",
        "-r",
        f"{fps:.9f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def _duplicate_record(item: DuplicateEvidence) -> dict:
    return asdict(item)


def _link_record(item: IdentityLinkCandidate) -> dict:
    return asdict(item)


def _identity_record(identity) -> dict:
    return {
        "identity_id": identity.identity_id,
        "fragment_ids": list(identity.fragment_ids),
        "member_trajectory_ids": list(identity.member_trajectory_ids),
        "source_track_ids": list(identity.source_track_ids),
        "support_score": identity.support_score,
        "start_time_s": identity.start_time_s,
        "end_time_s": identity.end_time_s,
        "boundary_birth": identity.boundary_birth,
        "boundary_death": identity.boundary_death,
        "rendered_frame_count": len(identity.states),
    }


def _track_record(track) -> dict:
    return {
        "identity_id": track.trajectory_id,
        "state": track.state,
        "x": track.x,
        "z": track.z,
        "vx": track.vx,
        "vz": track.vz,
        "speed_mps": track.speed_mps,
        "support_score": track.support_score,
        "member_trajectory_count": track.member_trajectory_count,
        "source_track_count": track.source_track_count,
    }


def _window_metrics(
    frames,
    frame_tracks,
    *,
    start_time_s: float,
    max_visible: int,
):
    selected_frames = [
        frame
        for frame in frames
        if float(frame.time_s) >= start_time_s
    ]

    visible_counts = []
    unique_ids = set()
    observed = 0
    interpolated = 0
    max_speed = 0.0

    for frame in selected_frames:
        tracks = frame_tracks.get(int(frame.frame), ())
        visible_counts.append(len(tracks))

        for track in tracks:
            unique_ids.add(int(track.trajectory_id))
            if track.state == "observed":
                observed += 1
            else:
                interpolated += 1
            max_speed = max(max_speed, float(track.speed_mps))

    return {
        "frames": len(selected_frames),
        "start_time_s": start_time_s,
        "end_time_s": (
            float(selected_frames[-1].time_s)
            if selected_frames
            else None
        ),
        "unique_identity_ids": len(unique_ids),
        "visible_count_min": min(visible_counts) if visible_counts else 0,
        "visible_count_max": max(visible_counts) if visible_counts else 0,
        "visible_count_mean": (
            statistics.fmean(visible_counts)
            if visible_counts
            else 0.0
        ),
        "visible_count_median": (
            statistics.median(visible_counts)
            if visible_counts
            else 0.0
        ),
        "frames_at_visible_cap": sum(
            count >= max_visible
            for count in visible_counts
        ),
        "observed_track_frames": observed,
        "interpolated_track_frames": interpolated,
        "maximum_rendered_speed_mps": max_speed,
    }


def main(argv=None) -> int:
    args = _parse_args(argv)

    if args.max_visible < 1:
        raise ValueError("--max-visible must be >= 1")
    if args.evaluation_tail_s <= 0.0:
        raise ValueError("--evaluation-tail-s must be > 0")

    observation_run = load_run_observations(args.run_dir)
    output_dir = _output_dir(args, observation_run.config)
    output_dir.mkdir(parents=True, exist_ok=False)

    offline_config = OfflineContextConfig(
        max_visible_tracks=args.max_visible,
    )
    manager_config = TrackManagerConfig(
        max_visible_tracks=args.max_visible,
    )
    association_config = AssociationConfig()
    kalman_config = KalmanConfig()

    # V2 remains the proposal generator.
    v2_result = process_offline_context(
        observation_run.frames,
        context_config=offline_config,
        manager_config=manager_config,
        association_config=association_config,
        kalman_config=kalman_config,
    )

    identity_config = GlobalIdentityConfig(
        max_visible_tracks=args.max_visible,
        state_interpolation_gap_s=offline_config.render_gap_s,
        render_gap_s=offline_config.render_gap_s,
    )

    metadata = observation_run.projector.metadata
    field_length_m = float(metadata.field_length)
    field_width_m = float(metadata.field_width)

    v3_result = solve_global_identities(
        observation_run.frames,
        v2_result.trajectories,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=identity_config,
    )

    duplicate_evidence_path = output_dir / "duplicate_evidence.json"
    identity_fragments_path = output_dir / "identity_fragments.json"
    identity_links_path = output_dir / "identity_links.json"
    identities_path = output_dir / "global_identities.json"
    tracks_path = output_dir / "global_tracks.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "global_identity_config.json"

    interesting_duplicate_evidence = [
        item
        for item in v3_result.duplicate_evidence
        if (
            item.duplicate
            or (
                item.independent_evidence_ratio is not None
                and item.independent_evidence_ratio >= 0.50
                and item.median_distance_m <= 1.25
            )
        )
    ]

    _write_json(
        duplicate_evidence_path,
        [_duplicate_record(item) for item in interesting_duplicate_evidence],
    )

    _write_json(
        identity_fragments_path,
        [
            {
                "fragment_id": fragment.fragment_id,
                "member_trajectory_ids": list(fragment.member_trajectory_ids),
                "source_track_ids": list(fragment.source_track_ids),
                "start_frame": fragment.start_frame,
                "end_frame": fragment.end_frame,
                "start_time_s": fragment.start_time_s,
                "end_time_s": fragment.end_time_s,
                "support_score": fragment.support_score,
                "rendered_frame_count": len(fragment.states),
            }
            for fragment in v3_result.fragments
        ],
    )

    accepted_link_set = set(v3_result.accepted_links)
    _write_json(
        identity_links_path,
        [
            {
                **_link_record(item),
                "accepted": (
                    item.predecessor_fragment_id,
                    item.successor_fragment_id,
                )
                in accepted_link_set,
            }
            for item in v3_result.link_candidates
        ],
    )

    _write_json(
        identities_path,
        [_identity_record(item) for item in v3_result.identities],
    )

    with tracks_path.open("w", encoding="utf-8") as handle:
        for frame in observation_run.frames:
            tracks = v3_result.frame_tracks.get(int(frame.frame), ())
            handle.write(
                json.dumps(
                    {
                        "frame": int(frame.frame),
                        "time_s": float(frame.time_s),
                        "raw_observation_count": len(frame.observations),
                        "visible_identity_count": len(tracks),
                        "tracks": [_track_record(item) for item in tracks],
                    }
                )
                + "\n"
            )

    output_paths = {
        "duplicate_evidence.json": str(duplicate_evidence_path),
        "identity_fragments.json": str(identity_fragments_path),
        "identity_links.json": str(identity_links_path),
        "global_identities.json": str(identities_path),
        "global_tracks.jsonl": str(tracks_path),
        "summary.json": str(summary_path),
        "global_identity_config.json": str(config_path),
    }

    if not args.no_video:
        fps = _fps(observation_run.frames)
        renderer = OfflinePitchRenderer(observation_run.projector)

        result_video = output_dir / "result_global_identity_2d.mp4"
        debug_video = output_dir / "debug_raw_vs_global_identity.mp4"

        result_writer = _open_writer(result_video, fps)
        debug_writer = _open_writer(debug_video, fps)

        try:
            for frame in observation_run.frames:
                tracks = v3_result.frame_tracks.get(int(frame.frame), ())

                result_writer.stdin.write(
                    renderer.render_result(
                        tracks,
                        frame=int(frame.frame),
                        time_s=float(frame.time_s),
                    ).tobytes()
                )
                debug_writer.stdin.write(
                    renderer.render_debug(
                        frame.observations,
                        tracks,
                        frame=int(frame.frame),
                        time_s=float(frame.time_s),
                    ).tobytes()
                )
        finally:
            for writer in (result_writer, debug_writer):
                if writer.stdin is not None:
                    writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError("ffmpeg failed while writing V3 video")

        output_paths["result_global_identity_2d.mp4"] = str(result_video)
        output_paths["debug_raw_vs_global_identity.mp4"] = str(debug_video)

    clip_start = float(observation_run.frames[0].time_s)
    clip_end = float(observation_run.frames[-1].time_s)
    tail_start = max(
        clip_start,
        clip_end - args.evaluation_tail_s,
    )

    full_metrics = _window_metrics(
        observation_run.frames,
        v3_result.frame_tracks,
        start_time_s=clip_start,
        max_visible=args.max_visible,
    )
    tail_metrics = _window_metrics(
        observation_run.frames,
        v3_result.frame_tracks,
        start_time_s=tail_start,
        max_visible=args.max_visible,
    )

    duplicate_pairs = [
        item
        for item in v3_result.duplicate_evidence
        if item.duplicate
    ]
    multi_member_components = [
        component
        for component in v3_result.duplicate_components
        if len(component) > 1
    ]

    summary = {
        "processed_frames": len(observation_run.frames),
        "processed_duration_s": clip_end - clip_start,
        "input_observations": observation_run.observation_count,
        "field_length_m": field_length_m,
        "field_width_m": field_width_m,
        "v2_proposal": {
            "trajectory_count": len(v2_result.trajectories),
            "tracklet_count": len(v2_result.tracklets),
            "anchor_tracklet_count": len(v2_result.anchor_tracklets),
            "weak_tracklet_count": len(v2_result.weak_tracklets),
            "accepted_tracklet_links": len(v2_result.link_result.links),
        },
        "v3_identity_resolution": {
            "duplicate_pairs": len(duplicate_pairs),
            "duplicate_components": len(multi_member_components),
            "blocked_duplicate_merges": len(
                v3_result.blocked_duplicate_merges
            ),
            "identity_fragments_after_duplicate_resolution": len(
                v3_result.fragments
            ),
            "global_link_candidates": len(v3_result.link_candidates),
            "accepted_global_links": len(v3_result.accepted_links),
            "final_identity_count": len(v3_result.identities),
            "boundary_birth_identities": sum(
                identity.boundary_birth
                for identity in v3_result.identities
            ),
            "boundary_death_identities": sum(
                identity.boundary_death
                for identity in v3_result.identities
            ),
        },
        "full_clip": full_metrics,
        "evaluation_tail": {
            "requested_tail_s": args.evaluation_tail_s,
            **tail_metrics,
        },
        "output_paths": output_paths,
    }

    _write_json(
        config_path,
        {
            "source_run_dir": str(observation_run.run_dir),
            "source_detection_config": observation_run.config,
            "offline_context_config": asdict(offline_config),
            "track_manager_config": asdict(manager_config),
            "association_config": asdict(association_config),
            "kalman_config": asdict(kalman_config),
            "global_identity_config": asdict(identity_config),
            "output_paths": output_paths,
        },
    )
    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"Offline global identity V3 output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
