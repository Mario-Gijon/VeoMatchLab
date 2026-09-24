#!/usr/bin/env python3
"""Run VeoMatchLab Offline Tracking V5.1 on an existing detector run.

V5.1 keeps V5 unchanged and adds one conservative full-clip post-link pass:
short-lived duplicate identity branches may be suppressed only when motion,
containment and STRONG raw evidence all agree that they are not two independent
footballers.

Detector evidence contract is unchanged from V5:
- source detector threshold <= weak threshold (default 0.20)
- STRONG: confidence >= 0.40
- WEAK:   0.20 <= confidence < 0.40
- local proposals are created from STRONG observations only
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import statistics

from tracking.association import AssociationConfig
from tracking import global_identity_V5 as v5
from tracking.global_identity_V5_1 import make_config, solve_global_identities
from tracking.kalman import KalmanConfig
from tracking.observations import load_run_observations
from tracking.offline_context import OfflineContextConfig, process_offline_context
from tracking.offline_renderer_V5 import OfflinePitchRendererV5
from tracking.track_manager import TrackManagerConfig
from tracking.run_offline_context_V5 import (
    _analytical_record,
    _compose_panorama_vs_2d,
    _filter_frames,
    _fps,
    _open_writer,
    _source_result_path,
    _window_metrics,
    _write_json,
)


ROOT = Path(__file__).resolve().parents[1]


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--weak-threshold", type=float, default=0.20)
    parser.add_argument("--strong-threshold", type=float, default=0.40)
    parser.add_argument("--evaluation-tail-s", type=float, default=20.0)
    parser.add_argument("--proposal-max-visible", type=int, default=25)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args(argv)


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
        / "offline_outputs_v5_1"
        / view
        / model
        / clip_id
        / stamp
    )


def _population_excess_metrics(frames, frame_tracks, *, expected_on_field: int):
    counts = [
        len(frame_tracks.get(int(frame.frame), ()))
        for frame in frames
    ]
    if not counts:
        return {
            "mean_excess_over_expected": 0.0,
            "maximum_excess_over_expected": 0,
        }
    return {
        "mean_excess_over_expected": float(
            statistics.fmean(max(0, count - expected_on_field) for count in counts)
        ),
        "maximum_excess_over_expected": int(
            max(max(0, count - expected_on_field) for count in counts)
        ),
    }


def _identity_record(identity):
    return {
        "identity_id": identity.identity_id,
        "fragment_ids": list(identity.fragment_ids),
        "member_trajectory_ids": list(identity.member_trajectory_ids),
        "source_track_ids": list(identity.source_track_ids),
        "support_score": identity.support_score,
        "observed_start_time_s": identity.observed_start_time_s,
        "observed_end_time_s": identity.observed_end_time_s,
        "boundary_birth": identity.boundary_birth,
        "boundary_death": identity.boundary_death,
        "core_fragment_count": identity.core_fragment_count,
        "bridge_fragment_count": identity.bridge_fragment_count,
        "analytical_frame_count": len(identity.states),
    }


def main(argv=None) -> int:
    args = _parse_args(argv)

    if not 0.0 <= args.weak_threshold <= args.strong_threshold <= 1.0:
        raise ValueError(
            "thresholds must satisfy 0 <= weak <= strong <= 1"
        )
    if args.evaluation_tail_s <= 0.0:
        raise ValueError("--evaluation-tail-s must be > 0")
    if args.proposal_max_visible < 1:
        raise ValueError("--proposal-max-visible must be >= 1")

    run = load_run_observations(args.run_dir)

    source_threshold = run.config.get("threshold")
    if source_threshold is None:
        raise ValueError(
            "source detector config has no threshold; V5.1 cannot prove "
            "WEAK evidence was preserved"
        )
    source_threshold = float(source_threshold)
    if source_threshold > args.weak_threshold + 1e-9:
        raise ValueError(
            "V5.1 requires a detector run at or below the weak threshold. "
            f"Source threshold={source_threshold:.3f}, "
            f"weak threshold={args.weak_threshold:.3f}."
        )

    (
        full_frames,
        strong_frames,
        strong_count,
        weak_count,
        discarded_below_weak,
    ) = _filter_frames(
        run.frames,
        weak_threshold=args.weak_threshold,
        strong_threshold=args.strong_threshold,
    )

    output_dir = _output_dir(args, run.config)
    output_dir.mkdir(parents=True, exist_ok=False)

    offline_cfg = OfflineContextConfig(
        max_visible_tracks=args.proposal_max_visible,
    )
    manager_cfg = TrackManagerConfig(
        max_visible_tracks=args.proposal_max_visible,
    )
    association_cfg = AssociationConfig()
    kalman_cfg = KalmanConfig()

    # Same proposal contract as V5: WEAK observations cannot birth a track.
    proposal = process_offline_context(
        strong_frames,
        context_config=offline_cfg,
        manager_config=manager_cfg,
        association_config=association_cfg,
        kalman_config=kalman_cfg,
    )

    v51_cfg = make_config(
        weak_threshold=args.weak_threshold,
        strong_threshold=args.strong_threshold,
    )

    metadata = run.projector.metadata
    field_length_m = float(metadata.field_length)
    field_width_m = float(metadata.field_width)

    result = solve_global_identities(
        full_frames,
        strong_frames,
        proposal.trajectories,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=v51_cfg,
    )

    # Reconstruct V5-before-refinement frame tracks only for A/B diagnostics.
    pre_frame_tracks = v5.build_frame_tracks(
        full_frames,
        result.pre_refinement_identities,
    )

    duplicate_path = output_dir / "duplicate_evidence_V5_1.json"
    fragments_path = output_dir / "identity_fragments_V5_1.json"
    links_path = output_dir / "identity_links_V5_1.json"
    branch_path = output_dir / "branch_competitions_V5_1.json"
    suppressed_path = output_dir / "suppressed_identities_V5_1.json"
    identities_path = output_dir / "final_identities_V5_1.json"
    tracks_path = output_dir / "analytical_tracks_V5_1.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "global_identity_V5_1_config.json"

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
    _write_json(
        duplicate_path,
        [asdict(item) for item in interesting_duplicates],
    )

    _write_json(
        fragments_path,
        [
            {
                "fragment_id": fragment.fragment_id,
                "member_trajectory_ids": list(fragment.member_trajectory_ids),
                "source_track_ids": list(fragment.source_track_ids),
                "role": fragment.role,
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
        branch_path,
        [asdict(item) for item in result.branch_competitions],
    )
    _write_json(
        suppressed_path,
        {
            "suppressed_identity_ids": list(result.suppressed_identity_ids),
            "count": len(result.suppressed_identity_ids),
        },
    )
    _write_json(
        identities_path,
        [_identity_record(identity) for identity in result.identities],
    )

    with tracks_path.open("w", encoding="utf-8") as handle:
        for frame in full_frames:
            tracks = result.frame_tracks.get(int(frame.frame), ())
            handle.write(
                json.dumps(
                    {
                        "frame": int(frame.frame),
                        "time_s": float(frame.time_s),
                        "raw_observation_count": len(frame.observations),
                        "visible_identity_count": len(tracks),
                        "tracks": [
                            _analytical_record(item)
                            for item in tracks
                        ],
                    }
                )
                + "\n"
            )

    output_paths = {
        "duplicate_evidence_V5_1.json": str(duplicate_path),
        "identity_fragments_V5_1.json": str(fragments_path),
        "identity_links_V5_1.json": str(links_path),
        "branch_competitions_V5_1.json": str(branch_path),
        "suppressed_identities_V5_1.json": str(suppressed_path),
        "final_identities_V5_1.json": str(identities_path),
        "analytical_tracks_V5_1.jsonl": str(tracks_path),
        "summary.json": str(summary_path),
        "global_identity_V5_1_config.json": str(config_path),
    }

    if not args.no_video:
        fps = _fps(full_frames)
        renderer = OfflinePitchRendererV5(
            run.projector,
            weak_threshold=args.weak_threshold,
            strong_threshold=args.strong_threshold,
        )

        analytical_video = output_dir / "result_analytical_V5_1_2d.mp4"
        debug_video = output_dir / "debug_raw_evidence_vs_analytical_V5_1.mp4"

        analytical_writer = _open_writer(analytical_video, fps)
        debug_writer = _open_writer(debug_video, fps)

        try:
            for frame in full_frames:
                tracks = result.frame_tracks.get(int(frame.frame), ())

                if analytical_writer.stdin is None:
                    raise RuntimeError("analytical ffmpeg stdin unavailable")
                analytical_writer.stdin.write(
                    renderer.render_result(
                        tracks,
                        frame=int(frame.frame),
                        time_s=float(frame.time_s),
                    ).tobytes()
                )

                if debug_writer.stdin is None:
                    raise RuntimeError("debug ffmpeg stdin unavailable")
                debug_writer.stdin.write(
                    renderer.render_debug(
                        frame.observations,
                        tracks,
                        frame=int(frame.frame),
                        time_s=float(frame.time_s),
                    ).tobytes()
                )
        finally:
            for writer in (analytical_writer, debug_writer):
                if writer.stdin is not None:
                    writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError(
                        "ffmpeg failed while writing V5.1 output"
                    )

        output_paths["result_analytical_V5_1_2d.mp4"] = str(analytical_video)
        output_paths[
            "debug_raw_evidence_vs_analytical_V5_1.mp4"
        ] = str(debug_video)

        comparison_video = (
            output_dir / "comparison_panorama_vs_analytical_V5_1.mp4"
        )
        (
            comparison_frames,
            comparison_width,
            comparison_height,
        ) = _compose_panorama_vs_2d(
            _source_result_path(run),
            analytical_video,
            comparison_video,
            fps=fps,
            frame_labels=[
                (int(frame.frame), float(frame.time_s))
                for frame in full_frames
            ],
        )
        output_paths[
            "comparison_panorama_vs_analytical_V5_1.mp4"
        ] = str(comparison_video)

    clip_start = float(full_frames[0].time_s)
    clip_end = float(full_frames[-1].time_s)
    tail_start = max(clip_start, clip_end - args.evaluation_tail_s)
    expected = v51_cfg.base.expected_on_field

    pre_full_metrics = _window_metrics(
        full_frames,
        pre_frame_tracks,
        start_time_s=clip_start,
        expected_on_field=expected,
    )
    full_metrics = _window_metrics(
        full_frames,
        result.frame_tracks,
        start_time_s=clip_start,
        expected_on_field=expected,
    )
    tail_metrics = _window_metrics(
        full_frames,
        result.frame_tracks,
        start_time_s=tail_start,
        expected_on_field=expected,
    )

    pre_full_metrics.update(
        _population_excess_metrics(
            full_frames,
            pre_frame_tracks,
            expected_on_field=expected,
        )
    )
    full_metrics.update(
        _population_excess_metrics(
            full_frames,
            result.frame_tracks,
            expected_on_field=expected,
        )
    )

    candidate_tiers = Counter(item.tier for item in result.link_candidates)
    candidate_by_pair = {
        (
            item.predecessor_fragment_id,
            item.successor_fragment_id,
        ): item
        for item in result.link_candidates
    }
    accepted_tiers = Counter(
        candidate_by_pair[pair].tier
        for pair in result.accepted_links
        if pair in candidate_by_pair
    )

    duplicate_branches = [
        item
        for item in result.branch_competitions
        if item.duplicate_branch
    ]
    rejected_competition_reasons = Counter(
        item.reason
        for item in result.branch_competitions
        if not item.duplicate_branch
    )

    suppressed_set = set(result.suppressed_identity_ids)
    suppressed_track_frames = sum(
        len(identity.states)
        for identity in result.pre_refinement_identities
        if identity.identity_id in suppressed_set
    )
    frames_changed = sum(
        len(pre_frame_tracks.get(int(frame.frame), ()))
        != len(result.frame_tracks.get(int(frame.frame), ()))
        for frame in full_frames
    )

    clip_edge_window_s = 1.0
    interior_births = sum(
        identity.observed_start_time_s > clip_start + clip_edge_window_s
        and not identity.boundary_birth
        for identity in result.identities
    )
    interior_deaths = sum(
        identity.observed_end_time_s < clip_end - clip_edge_window_s
        and not identity.boundary_death
        for identity in result.identities
    )

    summary = {
        "processed_frames": len(full_frames),
        "processed_duration_s": clip_end - clip_start,
        "field_length_m": field_length_m,
        "field_width_m": field_width_m,
        "thresholds": {
            "source_detector_threshold": source_threshold,
            "weak_threshold": args.weak_threshold,
            "strong_threshold": args.strong_threshold,
        },
        "input_evidence": {
            "total_preserved_observations": strong_count + weak_count,
            "strong_observations": strong_count,
            "weak_observations": weak_count,
            "discarded_below_weak": discarded_below_weak,
        },
        "strong_only_v2_proposal": {
            "trajectory_count": len(proposal.trajectories),
            "tracklet_count": len(proposal.tracklets),
            "anchor_tracklet_count": len(proposal.anchor_tracklets),
            "weak_tracklet_count": len(proposal.weak_tracklets),
            "accepted_tracklet_links": len(proposal.link_result.links),
        },
        "v5_linking_before_refinement": {
            "fragment_count": len(result.fragments),
            "raw_global_link_candidates": result.raw_candidate_count,
            "retained_topk_link_candidates": len(result.link_candidates),
            "candidate_tiers": dict(candidate_tiers),
            "accepted_global_links": len(result.accepted_links),
            "accepted_tiers": dict(accepted_tiers),
            "identity_count": len(result.pre_refinement_identities),
        },
        "v5_1_branch_refinement": {
            "evaluated_competitions": len(result.branch_competitions),
            "duplicate_branch_candidates": len(duplicate_branches),
            "suppressed_identity_count": len(result.suppressed_identity_ids),
            "suppressed_identity_ids": list(result.suppressed_identity_ids),
            "suppressed_track_frames": int(suppressed_track_frames),
            "frames_with_population_change": int(frames_changed),
            "rejected_reasons": dict(rejected_competition_reasons),
            "final_identity_count": len(result.identities),
            "interior_births": int(interior_births),
            "interior_deaths": int(interior_deaths),
        },
        "before_refinement_full_clip": pre_full_metrics,
        "full_clip": full_metrics,
        "evaluation_tail": {
            "requested_tail_s": args.evaluation_tail_s,
            **tail_metrics,
        },
        "output_paths": output_paths,
    }

    if not args.no_video:
        summary["comparison_video"] = {
            "frames": comparison_frames,
            "width": comparison_width,
            "height": comparison_height,
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
            "global_identity_V5_1_config": asdict(v51_cfg),
            "output_paths": output_paths,
        },
    )

    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(f"Offline global identity V5.1 output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
