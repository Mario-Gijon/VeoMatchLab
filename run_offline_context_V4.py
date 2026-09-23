#!/usr/bin/env python3
"""Run VeoMatchLab Offline Tracking V4 on an existing detector run.

No RF-DETR inference is executed.  The command reuses detections.jsonl, runs the
stable Offline Context V2 proposal generator, then applies the versioned V4
global identity solver.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess

from tracking.association import AssociationConfig
from tracking.global_identity_V4 import (
    GlobalIdentityV4Config,
    make_config,
    solve_global_identities,
)
from tracking.kalman import KalmanConfig
from tracking.observations import load_run_observations
from tracking.offline_context import OfflineContextConfig, process_offline_context
from tracking.offline_renderer import OfflinePitchRenderer
from tracking.track_manager import TrackManagerConfig


ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-visible", type=int, default=25)
    parser.add_argument("--evaluation-tail-s", type=float, default=20.0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


def _fps(frames) -> float:
    deltas = [
        b.time_s - a.time_s
        for a, b in zip(frames, frames[1:])
        if b.time_s > a.time_s
    ]
    return 1.0 / statistics.median(deltas) if deltas else 30.0


def _output_dir(args, source_config):
    if args.output_dir:
        return args.output_dir.expanduser().resolve()
    view = source_config.get("view", "unknown_view")
    model = source_config.get("model", {}).get("name", "unknown_model")
    clip_id = source_config.get("clip_id", "clip")
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return ROOT / "tracking" / "offline_outputs_v4" / view / model / clip_id / stamp


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


def _track_record(track):
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


def _window_metrics(frames, frame_tracks, *, start_time_s, max_visible):
    selected = [frame for frame in frames if float(frame.time_s) >= start_time_s]
    visible_counts = []
    unique_ids = set()
    observed = 0
    interpolated = 0
    max_speed = 0.0

    for frame in selected:
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
        "frames": len(selected),
        "start_time_s": start_time_s,
        "end_time_s": float(selected[-1].time_s) if selected else None,
        "unique_identity_ids": len(unique_ids),
        "visible_count_min": min(visible_counts) if visible_counts else 0,
        "visible_count_max": max(visible_counts) if visible_counts else 0,
        "visible_count_mean": statistics.fmean(visible_counts) if visible_counts else 0.0,
        "visible_count_median": statistics.median(visible_counts) if visible_counts else 0.0,
        "frames_at_visible_cap": sum(value >= max_visible for value in visible_counts),
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

    run = load_run_observations(args.run_dir)
    output_dir = _output_dir(args, run.config)
    output_dir.mkdir(parents=True, exist_ok=False)

    offline_cfg = OfflineContextConfig(max_visible_tracks=args.max_visible)
    manager_cfg = TrackManagerConfig(max_visible_tracks=args.max_visible)
    association_cfg = AssociationConfig()
    kalman_cfg = KalmanConfig()

    proposal = process_offline_context(
        run.frames,
        context_config=offline_cfg,
        manager_config=manager_cfg,
        association_config=association_cfg,
        kalman_config=kalman_cfg,
    )

    v4_cfg = make_config(
        state_interpolation_gap_s=offline_cfg.render_gap_s,
        render_gap_s=offline_cfg.render_gap_s,
        max_visible_tracks=args.max_visible,
    )

    metadata = run.projector.metadata
    field_length_m = float(metadata.field_length)
    field_width_m = float(metadata.field_width)

    result = solve_global_identities(
        run.frames,
        proposal.trajectories,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=v4_cfg,
    )

    duplicate_path = output_dir / "duplicate_evidence_V4.json"
    fragments_path = output_dir / "identity_fragments_V4.json"
    links_path = output_dir / "identity_links_V4.json"
    identities_path = output_dir / "global_identities_V4.json"
    tracks_path = output_dir / "global_tracks_V4.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "global_identity_V4_config.json"

    interesting_duplicates = [
        item
        for item in result.duplicate_evidence
        if (
            item.duplicate
            or (
                item.independent_evidence_ratio is not None
                and item.independent_evidence_ratio >= 0.50
                and item.median_distance_m <= 1.25
            )
        )
    ]
    _write_json(duplicate_path, [asdict(item) for item in interesting_duplicates])

    bridge_ids = set(result.bridge_fragment_ids)
    ephemeral_ids = set(result.ephemeral_fragment_ids)
    _write_json(
        fragments_path,
        [
            {
                "fragment_id": fragment.fragment_id,
                "member_trajectory_ids": list(fragment.member_trajectory_ids),
                "source_track_ids": list(fragment.source_track_ids),
                "initial_role": fragment.role,
                "final_role": (
                    "bridge"
                    if fragment.fragment_id in bridge_ids
                    else "ephemeral"
                    if fragment.fragment_id in ephemeral_ids
                    else "core"
                ),
                "start_time_s": fragment.start_time_s,
                "end_time_s": fragment.end_time_s,
                "observed_start_time_s": fragment.observed_start_time_s,
                "observed_end_time_s": fragment.observed_end_time_s,
                "observed_count": fragment.observed_count,
                "observed_span_s": fragment.observed_span_s,
                "support_score": fragment.support_score,
                "rendered_frame_count": len(fragment.states),
            }
            for fragment in result.fragments
        ],
    )

    accepted = set(result.accepted_links)
    _write_json(
        links_path,
        [
            {
                **asdict(item),
                "accepted": (
                    item.predecessor_fragment_id,
                    item.successor_fragment_id,
                ) in accepted,
            }
            for item in result.link_candidates
        ],
    )

    _write_json(
        identities_path,
        [
            {
                "identity_id": identity.identity_id,
                "fragment_ids": list(identity.fragment_ids),
                "member_trajectory_ids": list(identity.member_trajectory_ids),
                "source_track_ids": list(identity.source_track_ids),
                "support_score": identity.support_score,
                "start_time_s": identity.start_time_s,
                "end_time_s": identity.end_time_s,
                "boundary_birth": identity.boundary_birth,
                "boundary_death": identity.boundary_death,
                "core_fragment_count": identity.core_fragment_count,
                "bridge_fragment_count": identity.bridge_fragment_count,
                "rendered_frame_count": len(identity.states),
            }
            for identity in result.identities
        ],
    )

    with tracks_path.open("w", encoding="utf-8") as handle:
        for frame in run.frames:
            tracks = result.frame_tracks.get(int(frame.frame), ())
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
        "duplicate_evidence_V4.json": str(duplicate_path),
        "identity_fragments_V4.json": str(fragments_path),
        "identity_links_V4.json": str(links_path),
        "global_identities_V4.json": str(identities_path),
        "global_tracks_V4.jsonl": str(tracks_path),
        "summary.json": str(summary_path),
        "global_identity_V4_config.json": str(config_path),
    }

    if not args.no_video:
        fps = _fps(run.frames)
        renderer = OfflinePitchRenderer(run.projector)
        result_video = output_dir / "result_global_identity_V4_2d.mp4"
        debug_video = output_dir / "debug_raw_vs_global_identity_V4.mp4"
        result_writer = _open_writer(result_video, fps)
        debug_writer = _open_writer(debug_video, fps)
        try:
            for frame in run.frames:
                tracks = result.frame_tracks.get(int(frame.frame), ())
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
                    raise RuntimeError("ffmpeg failed while writing V4 output")
        output_paths["result_global_identity_V4_2d.mp4"] = str(result_video)
        output_paths["debug_raw_vs_global_identity_V4.mp4"] = str(debug_video)

    clip_start = float(run.frames[0].time_s)
    clip_end = float(run.frames[-1].time_s)
    tail_start = max(clip_start, clip_end - args.evaluation_tail_s)

    full_metrics = _window_metrics(
        run.frames,
        result.frame_tracks,
        start_time_s=clip_start,
        max_visible=args.max_visible,
    )
    tail_metrics = _window_metrics(
        run.frames,
        result.frame_tracks,
        start_time_s=tail_start,
        max_visible=args.max_visible,
    )

    candidate_modes = Counter(item.handoff_mode for item in result.link_candidates)
    accepted_candidate_by_pair = {
        (item.predecessor_fragment_id, item.successor_fragment_id): item
        for item in result.link_candidates
    }
    accepted_modes = Counter(
        accepted_candidate_by_pair[pair].handoff_mode
        for pair in result.accepted_links
        if pair in accepted_candidate_by_pair
    )

    summary = {
        "processed_frames": len(run.frames),
        "processed_duration_s": clip_end - clip_start,
        "input_observations": run.observation_count,
        "field_length_m": field_length_m,
        "field_width_m": field_width_m,
        "v2_proposal": {
            "trajectory_count": len(proposal.trajectories),
            "tracklet_count": len(proposal.tracklets),
            "anchor_tracklet_count": len(proposal.anchor_tracklets),
            "weak_tracklet_count": len(proposal.weak_tracklets),
            "accepted_tracklet_links": len(proposal.link_result.links),
        },
        "v4_identity_resolution": {
            "duplicate_pairs": sum(item.duplicate for item in result.duplicate_evidence),
            "duplicate_components": sum(len(component) > 1 for component in result.duplicate_components),
            "blocked_duplicate_merges": len(result.blocked_duplicate_merges),
            "fragment_count": len(result.fragments),
            "core_fragments": len(result.core_fragment_ids),
            "bridge_fragments": len(result.bridge_fragment_ids),
            "ephemeral_fragments": len(result.ephemeral_fragment_ids),
            "raw_global_link_candidates": result.raw_candidate_count,
            "retained_topk_link_candidates": len(result.link_candidates),
            "candidate_handoff_modes": dict(candidate_modes),
            "accepted_global_links": len(result.accepted_links),
            "accepted_handoff_modes": dict(accepted_modes),
            "final_identity_count": len(result.identities),
            "boundary_birth_identities": sum(identity.boundary_birth for identity in result.identities),
            "boundary_death_identities": sum(identity.boundary_death for identity in result.identities),
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
            "source_run_dir": str(run.run_dir),
            "source_detection_config": run.config,
            "offline_context_config": asdict(offline_cfg),
            "track_manager_config": asdict(manager_cfg),
            "association_config": asdict(association_cfg),
            "kalman_config": asdict(kalman_cfg),
            "global_identity_V4_config": asdict(v4_cfg),
            "output_paths": output_paths,
        },
    )
    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"Offline global identity V4 output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
