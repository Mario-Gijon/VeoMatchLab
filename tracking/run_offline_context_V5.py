#!/usr/bin/env python3
"""Run VeoMatchLab Offline Tracking V5 on an existing detector run.

V5 expects a detector run produced at or below the configured weak threshold
(default 0.20). It then separates evidence into:

- STRONG: confidence >= 0.40
- WEAK:   0.20 <= confidence < 0.40

Only STRONG observations are allowed to create local proposal tracks.
STRONG + WEAK observations are available to the clip-global V5 linker.
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

import cv2
import numpy as np

from tracking.association import AssociationConfig
from tracking.global_identity_V5 import (
    make_config,
    solve_global_identities,
)
from tracking.kalman import KalmanConfig
from tracking.observations import ObservationFrame, load_run_observations
from tracking.offline_context import OfflineContextConfig, process_offline_context
from tracking.offline_renderer_V5 import OfflinePitchRendererV5
from tracking.track_manager import TrackManagerConfig


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

    return (
        ROOT
        / "tracking"
        / "offline_outputs_v5"
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


def _open_writer(
    path: Path,
    fps: float,
    *,
    width: int = 1920,
    height: int = 1080,
):
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
        f"{width}x{height}",
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


def _fit_frame(
    image: np.ndarray,
    *,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    source_h, source_w = image.shape[:2]
    scale = min(
        max_width / source_w,
        max_height / source_h,
    )
    width = max(2, int(round(source_w * scale)))
    height = max(2, int(round(source_h * scale)))

    width -= width % 2
    height -= height % 2

    return cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_AREA,
    )


def _compose_panorama_vs_2d(
    panorama_path: Path,
    tracking_2d_path: Path,
    output_path: Path,
    *,
    fps: float,
    frame_labels: list[tuple[int, float]] | None = None,
) -> tuple[int, int, int]:
    panorama = cv2.VideoCapture(str(panorama_path))
    tracking_2d = cv2.VideoCapture(str(tracking_2d_path))

    if not panorama.isOpened():
        raise RuntimeError(
            f"Could not open panorama result video: {panorama_path}"
        )
    if not tracking_2d.isOpened():
        panorama.release()
        raise RuntimeError(
            f"Could not open V5 analytical video: {tracking_2d_path}"
        )

    top_probe = _fit_frame(
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        max_width=1920,
        max_height=1080,
    )
    bottom_probe = _fit_frame(
        np.zeros((1080, 1920, 3), dtype=np.uint8),
        max_width=1920,
        max_height=540,
    )

    top_height = top_probe.shape[0]
    bottom_height = bottom_probe.shape[0]
    separator_height = 40

    output_width = 1920
    output_height = (
        top_height
        + separator_height
        + bottom_height
    )

    writer = _open_writer(
        output_path,
        fps,
        width=output_width,
        height=output_height,
    )

    written = 0

    try:
        while True:
            ok_top, top = panorama.read()
            ok_bottom, bottom = tracking_2d.read()

            if (
                not ok_top
                or top is None
                or not ok_bottom
                or bottom is None
            ):
                break

            top = _fit_frame(
                top,
                max_width=output_width,
                max_height=1080,
            )
            bottom = _fit_frame(
                bottom,
                max_width=output_width,
                max_height=540,
            )

            canvas = np.zeros(
                (output_height, output_width, 3),
                dtype=np.uint8,
            )

            top_x = (output_width - top.shape[1]) // 2
            bottom_x = (output_width - bottom.shape[1]) // 2

            canvas[
                : top.shape[0],
                top_x : top_x + top.shape[1],
            ] = top

            bottom_y = top_height + separator_height
            canvas[
                bottom_y : bottom_y + bottom.shape[0],
                bottom_x : bottom_x + bottom.shape[1],
            ] = bottom

            # Hide source headers/footers so frame/time appears once.
            cv2.rectangle(
                canvas,
                (0, 0),
                (output_width, min(78, top_height)),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                canvas,
                (0, max(0, top_height - 58)),
                (output_width, top_height),
                (0, 0, 0),
                -1,
            )
            cv2.rectangle(
                canvas,
                (0, bottom_y),
                (
                    output_width,
                    bottom_y + min(78, bottom.shape[0]),
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                canvas,
                "PANORAMA + DETECTIONS (threshold source)",
                (22, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "V5 ANALYTICAL GLOBAL TRACKING",
                (bottom_x + 22, bottom_y + 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.86,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

            if frame_labels is not None and written < len(frame_labels):
                frame_number, time_s = frame_labels[written]
            else:
                frame_number, time_s = written, written / fps

            cv2.putText(
                canvas,
                f"frame={frame_number}  t={time_s:.2f}s",
                (22, top_height + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (220, 220, 220),
                2,
                cv2.LINE_AA,
            )

            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")

            writer.stdin.write(canvas.tobytes())
            written += 1
    finally:
        panorama.release()
        tracking_2d.release()

        if writer.stdin is not None:
            writer.stdin.close()

        if writer.wait() != 0:
            raise RuntimeError(
                "ffmpeg failed while writing V5 comparison video"
            )

    return written, output_width, output_height


def _filter_frames(
    frames,
    *,
    weak_threshold: float,
    strong_threshold: float,
):
    full_frames: list[ObservationFrame] = []
    strong_frames: list[ObservationFrame] = []

    weak_count = 0
    strong_count = 0
    discarded_below_weak = 0

    for frame in frames:
        accepted = []
        strong = []

        for observation in frame.observations:
            confidence = float(observation.confidence)

            if confidence < weak_threshold:
                discarded_below_weak += 1
                continue

            accepted.append(observation)

            if confidence >= strong_threshold:
                strong.append(observation)
                strong_count += 1
            else:
                weak_count += 1

        full_frames.append(
            ObservationFrame(
                frame=int(frame.frame),
                time_s=float(frame.time_s),
                observations=tuple(accepted),
            )
        )
        strong_frames.append(
            ObservationFrame(
                frame=int(frame.frame),
                time_s=float(frame.time_s),
                observations=tuple(strong),
            )
        )

    return (
        full_frames,
        strong_frames,
        strong_count,
        weak_count,
        discarded_below_weak,
    )


def _analytical_record(track):
    return {
        "identity_id": track.identity_id,
        "evidence": track.evidence,
        "x": track.x,
        "z": track.z,
        "vx": track.vx,
        "vz": track.vz,
        "speed_mps": track.speed_mps,
        "support_score": track.support_score,
        "member_trajectory_count": track.member_trajectory_count,
        "source_track_count": track.source_track_count,
        "supporting_confidence": track.supporting_confidence,
        "supporting_detection_index": track.supporting_detection_index,
    }


def _window_metrics(
    frames,
    frame_tracks,
    *,
    start_time_s: float,
    expected_on_field: int,
):
    selected = [
        frame
        for frame in frames
        if float(frame.time_s) >= start_time_s
    ]

    visible_counts: list[int] = []
    unique_ids: set[int] = set()

    evidence_counts = Counter()
    maximum_speed = 0.0

    for frame in selected:
        tracks = frame_tracks.get(int(frame.frame), ())
        visible_counts.append(len(tracks))

        for track in tracks:
            unique_ids.add(int(track.identity_id))
            evidence_counts[track.evidence] += 1
            maximum_speed = max(
                maximum_speed,
                float(track.speed_mps),
            )

    if visible_counts:
        mean_missing = statistics.fmean(
            max(0, expected_on_field - count)
            for count in visible_counts
        )
    else:
        mean_missing = 0.0

    return {
        "frames": len(selected),
        "start_time_s": start_time_s,
        "end_time_s": (
            float(selected[-1].time_s)
            if selected
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
        "frames_below_expected": sum(
            count < expected_on_field
            for count in visible_counts
        ),
        "frames_at_expected": sum(
            count == expected_on_field
            for count in visible_counts
        ),
        "frames_above_expected": sum(
            count > expected_on_field
            for count in visible_counts
        ),
        "mean_missing_from_expected": float(mean_missing),
        "strong_observed_track_frames": evidence_counts["strong_observed"],
        "strong_bridge_track_frames": evidence_counts["strong_bridge"],
        "weak_bridge_track_frames": evidence_counts["weak_bridge"],
        "inferred_track_frames": evidence_counts["inferred"],
        "maximum_analytical_speed_mps": maximum_speed,
    }


def _source_result_path(run):
    source_result = Path(
        run.config.get(
            "output_video",
            run.run_dir / "result.mp4",
        )
    )

    if source_result.is_absolute():
        return source_result

    run_relative = (run.run_dir / source_result).resolve()
    if run_relative.exists():
        return run_relative

    return (ROOT / source_result).resolve()


def main(argv=None) -> int:
    args = _parse_args(argv)

    if not 0.0 <= args.weak_threshold <= args.strong_threshold <= 1.0:
        raise ValueError(
            "thresholds must satisfy "
            "0 <= weak <= strong <= 1"
        )

    if args.evaluation_tail_s <= 0.0:
        raise ValueError(
            "--evaluation-tail-s must be > 0"
        )

    if args.proposal_max_visible < 1:
        raise ValueError(
            "--proposal-max-visible must be >= 1"
        )

    run = load_run_observations(args.run_dir)

    source_threshold = run.config.get("threshold")
    if source_threshold is None:
        raise ValueError(
            "source detector config has no threshold; "
            "V5 cannot prove WEAK evidence was preserved"
        )

    source_threshold = float(source_threshold)
    if source_threshold > args.weak_threshold + 1e-9:
        raise ValueError(
            "V5 requires the detector run to preserve WEAK evidence. "
            f"Source threshold is {source_threshold:.3f}, "
            f"but V5 weak threshold is {args.weak_threshold:.3f}. "
            "Re-run detection at or below the weak threshold."
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

    # Critical V5 rule:
    # local proposals are generated from STRONG observations only.
    proposal = process_offline_context(
        strong_frames,
        context_config=offline_cfg,
        manager_config=manager_cfg,
        association_config=association_cfg,
        kalman_config=kalman_cfg,
    )

    v5_cfg = make_config(
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
        config=v5_cfg,
    )

    duplicate_path = output_dir / "duplicate_evidence_V5.json"
    fragments_path = output_dir / "identity_fragments_V5.json"
    links_path = output_dir / "identity_links_V5.json"
    identities_path = output_dir / "final_identities_V5.json"
    tracks_path = output_dir / "analytical_tracks_V5.jsonl"
    summary_path = output_dir / "summary.json"
    config_path = output_dir / "global_identity_V5_config.json"

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
                "member_trajectory_ids": list(
                    fragment.member_trajectory_ids
                ),
                "source_track_ids": list(
                    fragment.source_track_ids
                ),
                "role": fragment.role,
                "start_time_s": fragment.start_time_s,
                "end_time_s": fragment.end_time_s,
                "observed_start_time_s": (
                    fragment.observed_start_time_s
                ),
                "observed_end_time_s": (
                    fragment.observed_end_time_s
                ),
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
                "member_trajectory_ids": list(
                    identity.member_trajectory_ids
                ),
                "source_track_ids": list(
                    identity.source_track_ids
                ),
                "support_score": identity.support_score,
                "observed_start_time_s": (
                    identity.observed_start_time_s
                ),
                "observed_end_time_s": (
                    identity.observed_end_time_s
                ),
                "boundary_birth": identity.boundary_birth,
                "boundary_death": identity.boundary_death,
                "core_fragment_count": (
                    identity.core_fragment_count
                ),
                "bridge_fragment_count": (
                    identity.bridge_fragment_count
                ),
                "analytical_frame_count": len(identity.states),
            }
            for identity in result.identities
        ],
    )

    with tracks_path.open("w", encoding="utf-8") as handle:
        for frame in full_frames:
            tracks = result.frame_tracks.get(
                int(frame.frame),
                (),
            )
            handle.write(
                json.dumps(
                    {
                        "frame": int(frame.frame),
                        "time_s": float(frame.time_s),
                        "raw_observation_count": len(
                            frame.observations
                        ),
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
        "duplicate_evidence_V5.json": str(duplicate_path),
        "identity_fragments_V5.json": str(fragments_path),
        "identity_links_V5.json": str(links_path),
        "final_identities_V5.json": str(identities_path),
        "analytical_tracks_V5.jsonl": str(tracks_path),
        "summary.json": str(summary_path),
        "global_identity_V5_config.json": str(config_path),
    }

    if not args.no_video:
        fps = _fps(full_frames)
        renderer = OfflinePitchRendererV5(
            run.projector,
            weak_threshold=args.weak_threshold,
            strong_threshold=args.strong_threshold,
        )

        analytical_video = (
            output_dir / "result_analytical_V5_2d.mp4"
        )
        debug_video = (
            output_dir
            / "debug_raw_evidence_vs_analytical_V5.mp4"
        )

        analytical_writer = _open_writer(
            analytical_video,
            fps,
        )
        debug_writer = _open_writer(
            debug_video,
            fps,
        )

        try:
            for frame in full_frames:
                tracks = result.frame_tracks.get(
                    int(frame.frame),
                    (),
                )

                if analytical_writer.stdin is None:
                    raise RuntimeError(
                        "analytical ffmpeg stdin unavailable"
                    )
                analytical_writer.stdin.write(
                    renderer.render_result(
                        tracks,
                        frame=int(frame.frame),
                        time_s=float(frame.time_s),
                    ).tobytes()
                )

                if debug_writer.stdin is None:
                    raise RuntimeError(
                        "debug ffmpeg stdin unavailable"
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
            for writer in (
                analytical_writer,
                debug_writer,
            ):
                if writer.stdin is not None:
                    writer.stdin.close()
                if writer.wait() != 0:
                    raise RuntimeError(
                        "ffmpeg failed while writing V5 output"
                    )

        output_paths[
            "result_analytical_V5_2d.mp4"
        ] = str(analytical_video)
        output_paths[
            "debug_raw_evidence_vs_analytical_V5.mp4"
        ] = str(debug_video)

        source_result = _source_result_path(run)
        comparison_video = (
            output_dir
            / "comparison_panorama_vs_analytical_V5.mp4"
        )

        (
            comparison_frames,
            comparison_width,
            comparison_height,
        ) = _compose_panorama_vs_2d(
            source_result,
            analytical_video,
            comparison_video,
            fps=fps,
            frame_labels=[
                (
                    int(frame.frame),
                    float(frame.time_s),
                )
                for frame in full_frames
            ],
        )

        output_paths[
            "comparison_panorama_vs_analytical_V5.mp4"
        ] = str(comparison_video)

    clip_start = float(full_frames[0].time_s)
    clip_end = float(full_frames[-1].time_s)
    tail_start = max(
        clip_start,
        clip_end - args.evaluation_tail_s,
    )

    full_metrics = _window_metrics(
        full_frames,
        result.frame_tracks,
        start_time_s=clip_start,
        expected_on_field=v5_cfg.expected_on_field,
    )
    tail_metrics = _window_metrics(
        full_frames,
        result.frame_tracks,
        start_time_s=tail_start,
        expected_on_field=v5_cfg.expected_on_field,
    )

    candidate_tiers = Counter(
        item.tier
        for item in result.link_candidates
    )

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

    clip_edge_window_s = 1.0
    interior_births = sum(
        identity.observed_start_time_s
        > clip_start + clip_edge_window_s
        and not identity.boundary_birth
        for identity in result.identities
    )
    interior_deaths = sum(
        identity.observed_end_time_s
        < clip_end - clip_edge_window_s
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
            "total_preserved_observations": (
                strong_count + weak_count
            ),
            "strong_observations": strong_count,
            "weak_observations": weak_count,
            "discarded_below_weak": (
                discarded_below_weak
            ),
        },
        "strong_only_v2_proposal": {
            "trajectory_count": len(
                proposal.trajectories
            ),
            "tracklet_count": len(
                proposal.tracklets
            ),
            "anchor_tracklet_count": len(
                proposal.anchor_tracklets
            ),
            "weak_tracklet_count": len(
                proposal.weak_tracklets
            ),
            "accepted_tracklet_links": len(
                proposal.link_result.links
            ),
        },
        "v5_identity_resolution": {
            "duplicate_pairs": sum(
                item.duplicate
                for item in result.duplicate_evidence
            ),
            "duplicate_components": sum(
                len(component) > 1
                for component in result.duplicate_components
            ),
            "blocked_duplicate_merges": len(
                result.blocked_duplicate_merges
            ),
            "fragment_count": len(
                result.fragments
            ),
            "core_fragments": sum(
                fragment.role == "core"
                for fragment in result.fragments
            ),
            "bridge_candidate_fragments": sum(
                fragment.role == "bridge_candidate"
                for fragment in result.fragments
            ),
            "ephemeral_fragments": sum(
                fragment.role == "ephemeral"
                for fragment in result.fragments
            ),
            "raw_global_link_candidates": (
                result.raw_candidate_count
            ),
            "retained_topk_link_candidates": len(
                result.link_candidates
            ),
            "candidate_tiers": dict(
                candidate_tiers
            ),
            "accepted_global_links": len(
                result.accepted_links
            ),
            "accepted_tiers": dict(
                accepted_tiers
            ),
            "final_identity_count": len(
                result.identities
            ),
            "interior_births": int(
                interior_births
            ),
            "interior_deaths": int(
                interior_deaths
            ),
        },
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
            "offline_context_config": asdict(
                offline_cfg
            ),
            "track_manager_config": asdict(
                manager_cfg
            ),
            "association_config": asdict(
                association_cfg
            ),
            "kalman_config": asdict(
                kalman_cfg
            ),
            "global_identity_V5_config": asdict(
                v5_cfg
            ),
            "output_paths": output_paths,
        },
    )

    _write_json(summary_path, summary)

    print(json.dumps(summary, indent=2))
    print(
        f"Offline global identity V5 output: "
        f"{output_dir}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
