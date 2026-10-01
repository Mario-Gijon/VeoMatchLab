#!/usr/bin/env python3
"""Run Tracking V1 Hito 7G.4 conservative trajectory proposal solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_trajectory_solver import (  # noqa: E402
    GLOBAL_TRAJECTORY_PROPOSAL_SOLVER_VERSION,
    GlobalTrajectoryProposalSolverConfig,
    solve_global_trajectory_proposals,
)
from tracking.actors.identity_expansion import (  # noqa: E402
    build_hard_identity_pairs,
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
            "p90": None, "p95": None, "max": None, "mean": None,
        }

    def pct(p):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return values[lo] * (1 - frac) + values[hi] * frac

    return {
        "count": len(values),
        "min": values[0],
        "p10": pct(0.10),
        "median": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": values[-1],
        "mean": statistics.fmean(values),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--scoring-dir")
    parser.add_argument("--identity-dir")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.70,
        help="Proposal floor; score is not a probability.",
    )
    parser.add_argument(
        "--competition-mode",
        choices=("mutual_best", "mutual_top3"),
        default="mutual_best",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    scoring_dir = (
        Path(args.scoring_dir)
        if args.scoring_dir
        else root / "diagnostics"
        / "global_trajectory_scoring"
        / "global-trajectory-scoring-v1"
    )
    identity_dir = (
        Path(args.identity_dir)
        if args.identity_dir
        else root / "diagnostics"
        / "global_identity_consolidation"
        / "global-identity-consolidation-v2-family-guard"
    )

    append_path = scoring_dir / "append_scored.jsonl"
    insert_path = scoring_dir / "insert_scored.jsonl"
    identities_path = identity_dir / "final_identities.jsonl"
    assignments_path = (
        identity_dir / "final_tracklet_assignments.jsonl"
    )
    hard_conflicts_path = (
        root / "diagnostics" / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )

    for path in (
        append_path,
        insert_path,
        identities_path,
        assignments_path,
        hard_conflicts_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    append_rows = _iter_jsonl(append_path)
    insert_rows = _iter_jsonl(insert_path)
    identities = _iter_jsonl(identities_path)
    assignments = _iter_jsonl(assignments_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)

    hard_pairs = build_hard_identity_pairs(
        tracklet_assignment_rows=assignments,
        hard_conflicts=hard_conflicts,
    )

    config = GlobalTrajectoryProposalSolverConfig(
        min_trajectory_score=args.min_score,
        competition_mode=args.competition_mode,
        require_identity_disjoint=True,
    )
    result = solve_global_trajectory_proposals(
        append_rows=append_rows,
        insert_rows=insert_rows,
        hard_identity_pairs=hard_pairs,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "global_trajectory_solver"
        / GLOBAL_TRAJECTORY_PROPOSAL_SOLVER_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    proposals_path = output_dir / "selected_proposals.jsonl"
    _write_jsonl(
        proposals_path,
        (row.to_dict() for row in result.selected),
    )

    summary = result.summary(
        source_identity_count=len(identities)
    )
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "hard_identity_pair_count": len(hard_pairs),
        "selected_score": _stats([
            row.trajectory_score for row in result.selected
        ]),
        "selected_profile_confidence": _stats([
            row.profile_confidence
            for row in result.selected
            if row.profile_confidence is not None
        ]),
        "selected_geometry_support": _stats([
            row.geometry_support for row in result.selected
        ]),
        "selected_append_gap_s": _stats([
            row.source["gap_s"]
            for row in result.selected
            if row.candidate_type == "append"
        ]),
        "selected_insert_total_bridge_gap_s": _stats([
            float(row.source["left_gap_s"])
            + float(row.source["right_gap_s"])
            for row in result.selected
            if row.candidate_type == "insert"
        ]),
        "source_append_scored": str(append_path),
        "source_insert_scored": str(insert_path),
        "source_identities": str(identities_path),
        "source_tracklet_assignments": str(assignments_path),
        "source_hard_conflicts": str(hard_conflicts_path),
        "audit_required_before_merge": True,
        "provisional": True,
        "canonical": False,
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "proposal_solver_version": (
                GLOBAL_TRAJECTORY_PROPOSAL_SOLVER_VERSION
            ),
            "selected_proposals": proposals_path.name,
            "summary": summary_path.name,
            "selected_proposal_count": len(result.selected),
            "identity_partition_changed": False,
            "audit_required_before_merge": True,
        },
    )

    print(
        "[global-trajectory-solver-7G.4] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[global-trajectory-solver-7G.4] "
        f"append_input={len(append_rows)} "
        f"insert_input={len(insert_rows)} "
        f"eligible={result.eligible_count} "
        f"selected={len(result.selected)}"
    )
    print(
        "[global-trajectory-solver-7G.4] selected_types="
        + json.dumps(
            summary["selected_type_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-solver-7G.4] selected_family="
        + json.dumps(
            summary["selected_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-solver-7G.4] rejections="
        + json.dumps(
            summary["rejection_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-solver-7G.4] selected_score="
        + json.dumps(summary["selected_score"], sort_keys=True)
    )
    print(
        "[global-trajectory-solver-7G.4] "
        f"projected_identities_if_accepted="
        f"{summary['projected_identity_count_if_all_accepted']}"
    )
    print(
        "[global-trajectory-solver-7G.4] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
