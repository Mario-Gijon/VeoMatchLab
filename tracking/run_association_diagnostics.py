#!/usr/bin/env python3
"""Run Tracking V2.1 diagnostics over an existing detector run.

This command performs no model inference and renders no video.  It reuses the
stored observations and the stable tracking modules, then writes diagnostics
that explain where association candidates were rejected.

Example:

python -m tracking.run_association_diagnostics \
  --run-dir person_detect/outputs/panorama_view/rfdetr_2xl/.../20260923T024752
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

from tracking.association import AssociationConfig
from tracking.kalman import KalmanConfig
from tracking.observations import load_run_observations
from tracking.track_manager import TrackManager, TrackManagerConfig


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explain V2.1 association rejections without detector inference."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--max-visible", type=int, default=25)
    return parser.parse_args()


def _nearest_to_dict(value):
    return None if value is None else asdict(value)


def _frame_to_dict(frame_result):
    return {
        "frame": frame_result.frame,
        "time_s": frame_result.time_s,
        "assignment_count": len(frame_result.assignments),
        "unmatched_observation_indices": list(
            frame_result.unmatched_observation_indices
        ),
        "new_track_ids": list(frame_result.new_track_ids),
        "newly_confirmed_track_ids": list(
            frame_result.newly_confirmed_track_ids
        ),
        "reactivated_track_ids": list(frame_result.reactivated_track_ids),
        "newly_lost_track_ids": list(frame_result.newly_lost_track_ids),
        "retired_track_ids": list(frame_result.retired_track_ids),
        "visible_track_count": len(frame_result.visible_track_ids),
        "stages": [
            asdict(stage)
            for stage in frame_result.association_stage_diagnostics
        ],
        "unmatched": [
            {
                "observation_index": item.observation_index,
                "detection_index": item.detection_index,
                "confidence": item.confidence,
                "nearest_confirmed": _nearest_to_dict(
                    item.nearest_confirmed
                ),
                "nearest_lost": _nearest_to_dict(item.nearest_lost),
                "nearest_tentative": _nearest_to_dict(
                    item.nearest_tentative
                ),
            }
            for item in frame_result.unmatched_observation_diagnostics
        ],
    }


def _safe_stats(values):
    values = [float(v) for v in values]
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "count": len(values),
        "min": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[index],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def main() -> None:
    args = _parse_args()

    run = load_run_observations(Path(args.run_dir))

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.run_dir) / "tracking_v2_1_diagnostics"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    manager_config = replace(
        TrackManagerConfig(),
        max_visible_tracks=args.max_visible,
    )
    association_config = AssociationConfig()
    kalman_config = KalmanConfig()

    manager = TrackManager(
        config=manager_config,
        association_config=association_config,
        kalman_config=kalman_config,
    )

    frame_records = []

    stage_aggregate = defaultdict(Counter)
    unmatched_reason_counts = {
        "nearest_confirmed": Counter(),
        "nearest_lost": Counter(),
        "nearest_tentative": Counter(),
    }

    unmatched_nearest_confirmed_distances = []
    unmatched_nearest_confirmed_physical_radii = []
    unmatched_nearest_confirmed_uncertainty_slack = []
    unmatched_nearest_confirmed_mahalanobis = []

    total_observations = 0
    total_assignments = 0
    total_births = 0
    total_unmatched_after_first_frame = 0

    for frame_number, observation_frame in enumerate(run.frames):
        observations = tuple(observation_frame.observations)
        total_observations += len(observations)

        result = manager.process_frame(
            frame=observation_frame.frame,
            time_s=observation_frame.time_s,
            observations=observations,
        )

        total_assignments += len(result.assignments)
        total_births += len(result.new_track_ids)

        if frame_number > 0:
            total_unmatched_after_first_frame += len(
                result.unmatched_observation_indices
            )

        for stage in result.association_stage_diagnostics:
            counter = stage_aggregate[stage.lifecycle]
            counter["frames"] += 1
            counter["track_count_sum"] += stage.track_count
            counter["observation_count_sum"] += stage.observation_count
            counter["pairs_evaluated"] += stage.pairs_evaluated
            counter["allowed_pairs"] += stage.allowed_pairs
            counter["physical_gate_rejections"] += (
                stage.physical_gate_rejections
            )
            counter["mahalanobis_gate_rejections"] += (
                stage.mahalanobis_gate_rejections
            )
            counter["non_finite_mahalanobis_rejections"] += (
                stage.non_finite_mahalanobis_rejections
            )
            counter["other_rejections"] += stage.other_rejections
            counter["matches"] += stage.matches

        for item in result.unmatched_observation_diagnostics:
            for key, nearest in (
                ("nearest_confirmed", item.nearest_confirmed),
                ("nearest_lost", item.nearest_lost),
                ("nearest_tentative", item.nearest_tentative),
            ):
                if nearest is None:
                    unmatched_reason_counts[key]["no_candidate"] += 1
                else:
                    unmatched_reason_counts[key][nearest.reason] += 1

            nearest = item.nearest_confirmed
            if nearest is not None:
                unmatched_nearest_confirmed_distances.append(
                    nearest.predicted_distance_m
                )
                unmatched_nearest_confirmed_physical_radii.append(
                    nearest.physical_radius_m
                )
                unmatched_nearest_confirmed_uncertainty_slack.append(
                    nearest.uncertainty_slack_m
                )
                if nearest.mahalanobis_squared is not None:
                    unmatched_nearest_confirmed_mahalanobis.append(
                        nearest.mahalanobis_squared
                    )

        frame_records.append(_frame_to_dict(result))

    jsonl_path = output_dir / "association_diagnostics.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in frame_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    stage_summary = {}
    for lifecycle, counter in stage_aggregate.items():
        pairs = counter["pairs_evaluated"]
        rejected = (
            counter["physical_gate_rejections"]
            + counter["mahalanobis_gate_rejections"]
            + counter["non_finite_mahalanobis_rejections"]
            + counter["other_rejections"]
        )
        stage_summary[lifecycle] = {
            **dict(counter),
            "rejected_pairs": rejected,
            "physical_rejection_fraction": (
                counter["physical_gate_rejections"] / pairs
                if pairs else None
            ),
            "mahalanobis_rejection_fraction": (
                counter["mahalanobis_gate_rejections"] / pairs
                if pairs else None
            ),
            "allowed_pair_fraction": (
                counter["allowed_pairs"] / pairs
                if pairs else None
            ),
        }

    summary = {
        "source_run_dir": str(Path(args.run_dir)),
        "processed_frames": len(run.frames),
        "input_observations": total_observations,
        "total_assignments": total_assignments,
        "total_tracks_created": total_births,
        "unmatched_observations_after_first_frame": (
            total_unmatched_after_first_frame
        ),
        "configs": {
            "track_manager": asdict(manager_config),
            "association": asdict(association_config),
            "kalman": asdict(kalman_config),
        },
        "stage_summary": stage_summary,
        "unmatched_reason_counts": {
            key: dict(counter)
            for key, counter in unmatched_reason_counts.items()
        },
        "unmatched_nearest_confirmed": {
            "predicted_distance_m": _safe_stats(
                unmatched_nearest_confirmed_distances
            ),
            "physical_radius_m": _safe_stats(
                unmatched_nearest_confirmed_physical_radii
            ),
            "uncertainty_slack_m": _safe_stats(
                unmatched_nearest_confirmed_uncertainty_slack
            ),
            "mahalanobis_squared": _safe_stats(
                unmatched_nearest_confirmed_mahalanobis
            ),
        },
        "outputs": {
            "association_diagnostics_jsonl": str(jsonl_path),
            "summary_json": str(output_dir / "summary.json"),
        },
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
