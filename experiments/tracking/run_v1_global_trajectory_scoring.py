#!/usr/bin/env python3
"""Run Tracking V1 Hito 7G.3 centered global trajectory scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_trajectory_scoring import (  # noqa: E402
    GLOBAL_TRAJECTORY_SCORING_VERSION,
    GlobalTrajectoryScoringConfig,
    identity_profiles_from_rows,
    score_global_trajectory_candidates,
)
from tracking.actors.identity_expansion import (  # noqa: E402
    build_hard_identity_pairs,
    build_identity_family_evidence,
    fit_identity_calibration,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain an object"
                )
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ) + "\n"
            )
            count += 1
    return count


def _stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {
            "count": 0, "min": None, "p10": None, "median": None,
            "p90": None, "p95": None, "p99": None, "max": None,
            "mean": None,
        }

    def pct(p):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    return {
        "count": len(values),
        "min": values[0],
        "p10": pct(0.10),
        "median": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--candidate-dir")
    parser.add_argument("--identity-dir")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    candidate_dir = (
        Path(args.candidate_dir)
        if args.candidate_dir
        else root / "diagnostics"
        / "global_trajectory_candidates"
        / "global-trajectory-candidates-v1"
    )
    identity_dir = (
        Path(args.identity_dir)
        if args.identity_dir
        else root / "diagnostics"
        / "global_identity_consolidation"
        / "global-identity-consolidation-v2-family-guard"
    )

    append_path = candidate_dir / "append_candidates.jsonl"
    insert_path = candidate_dir / "insert_candidates.jsonl"
    direct_profiles_path = identity_dir / "profiles.jsonl"
    legacy_profiles_path = (
        identity_dir / "round_06" / "profiles.jsonl"
    )
    profiles_path = (
        direct_profiles_path
        if direct_profiles_path.is_file()
        else legacy_profiles_path
    )
    family_path = identity_dir / "final_family_assignments.jsonl"
    assignments_path = identity_dir / "final_tracklet_assignments.jsonl"
    hard_conflicts_path = (
        root / "diagnostics" / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )
    trusted_positive_path = (
        root / "diagnostics" / "reid_calibration"
        / "reid-overlap-calibration-v1"
        / "split_positive.jsonl"
    )

    for path in (
        append_path,
        insert_path,
        profiles_path,
        family_path,
        assignments_path,
        hard_conflicts_path,
        trusted_positive_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    append_rows = _iter_jsonl(append_path)
    insert_rows = _iter_jsonl(insert_path)
    profile_rows = _iter_jsonl(profiles_path)
    family_rows = _iter_jsonl(family_path)
    assignments = _iter_jsonl(assignments_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)
    trusted_positive = [
        float(row["cosine"])
        for row in _iter_jsonl(trusted_positive_path)
    ]

    profile_map = identity_profiles_from_rows(profile_rows)
    family_evidence = build_identity_family_evidence(family_rows)
    hard_pairs = build_hard_identity_pairs(
        tracklet_assignment_rows=assignments,
        hard_conflicts=hard_conflicts,
    )
    calibration = fit_identity_calibration(
        profiles=tuple(profile_map.values()),
        hard_identity_pairs=hard_pairs,
        trusted_positive_cosines=trusted_positive,
        identity_family_evidence=family_evidence,
    )

    config = GlobalTrajectoryScoringConfig()
    result = score_global_trajectory_candidates(
        append_candidates=append_rows,
        insert_candidates=insert_rows,
        profile_rows=profile_rows,
        calibration=calibration,
        family_rows=family_rows,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "global_trajectory_scoring"
        / GLOBAL_TRAJECTORY_SCORING_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    scored_append_path = output_dir / "append_scored.jsonl"
    scored_insert_path = output_dir / "insert_scored.jsonl"

    _write_jsonl(
        scored_append_path,
        (row.to_dict() for row in result.append_candidates),
    )
    _write_jsonl(
        scored_insert_path,
        (row.to_dict() for row in result.insert_candidates),
    )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "profiled_identity_count": len(profile_rows),
        "unprofiled_identity_count": (
            798 - len(profile_rows)
        ),
        "hard_identity_pair_count": len(hard_pairs),
        "trusted_positive_count": len(trusted_positive),
        "append_trajectory_score": _stats([
            row.trajectory_score
            for row in result.append_candidates
            if row.trajectory_score is not None
        ]),
        "insert_trajectory_score": _stats([
            row.trajectory_score
            for row in result.insert_candidates
            if row.trajectory_score is not None
        ]),
        "append_appearance_score": _stats([
            row.appearance_score
            for row in result.append_candidates
            if row.appearance_score is not None
        ]),
        "insert_appearance_score": _stats([
            row.appearance_score
            for row in result.insert_candidates
            if row.appearance_score is not None
        ]),
        "append_geometry_support": _stats([
            row.geometry_support
            for row in result.append_candidates
        ]),
        "insert_geometry_support": _stats([
            row.geometry_support
            for row in result.insert_candidates
        ]),
        "append_velocity_support_available_count": sum(
            row.velocity_support is not None
            for row in result.append_candidates
        ),
        "insert_any_velocity_support_available_count": sum(
            (
                row.left_velocity_support is not None
                or row.right_velocity_support is not None
            )
            for row in result.insert_candidates
        ),
        "calibration": {
            "trusted_positive": _stats(calibration.positive),
            "same_kit_negative": _stats(
                calibration.same_kit_negative
            ),
            "different_kit_negative": _stats(
                calibration.different_kit_negative
            ),
            "unknown_kit_negative": _stats(
                calibration.unknown_kit_negative
            ),
        },
        "source_append_candidates": str(append_path),
        "source_insert_candidates": str(insert_path),
        "source_identity_profiles": str(profiles_path),
        "source_family_assignments": str(family_path),
        "selection_applied": False,
        "provisional": True,
        "canonical": False,
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "scoring_version": GLOBAL_TRAJECTORY_SCORING_VERSION,
            "append_scored": scored_append_path.name,
            "insert_scored": scored_insert_path.name,
            "summary": summary_path.name,
            "selection_applied": False,
            "identity_decisions_changed": False,
        },
    )

    print(
        "[global-trajectory-scoring-7G.3] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[global-trajectory-scoring-7G.3] "
        f"append={len(result.append_candidates)} "
        f"appearance={summary['append_appearance_available_count']} "
        f"mutual_best={summary['append_mutual_best_count']} "
        f"mutual_top3={summary['append_mutual_top3_count']}"
    )
    print(
        "[global-trajectory-scoring-7G.3] "
        f"insert={len(result.insert_candidates)} "
        f"appearance={summary['insert_appearance_available_count']} "
        f"mutual_best={summary['insert_mutual_best_count']} "
        f"mutual_top3={summary['insert_mutual_top3_count']}"
    )
    print(
        "[global-trajectory-scoring-7G.3] append_bands="
        + json.dumps(summary["append_score_bands"], sort_keys=True)
    )
    print(
        "[global-trajectory-scoring-7G.3] insert_bands="
        + json.dumps(summary["insert_score_bands"], sort_keys=True)
    )
    print(
        "[global-trajectory-scoring-7G.3] append_score="
        + json.dumps(summary["append_trajectory_score"], sort_keys=True)
    )
    print(
        "[global-trajectory-scoring-7G.3] insert_score="
        + json.dumps(summary["insert_trajectory_score"], sort_keys=True)
    )
    print(
        "[global-trajectory-scoring-7G.3] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
