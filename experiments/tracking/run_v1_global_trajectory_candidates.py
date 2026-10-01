#!/usr/bin/env python3
"""Run Tracking V1 Hito 7G.2 geometry-first global trajectory candidates."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_trajectory_candidates import (  # noqa: E402
    GLOBAL_TRAJECTORY_CANDIDATE_VERSION,
    GlobalTrajectoryCandidateConfig,
    build_global_trajectory_candidates,
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
        "mean": sum(values) / len(values),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--trajectory-dir")
    parser.add_argument("--identity-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-append-gap-s", type=float, default=15.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    trajectory_dir = (
        Path(args.trajectory_dir)
        if args.trajectory_dir
        else root / "diagnostics"
        / "identity_trajectory_endpoints"
        / "identity-trajectory-endpoint-v2-segments"
    )
    identity_dir = (
        Path(args.identity_dir)
        if args.identity_dir
        else root / "diagnostics"
        / "global_identity_consolidation"
        / "global-identity-consolidation-v2-family-guard"
    )

    profiles_path = trajectory_dir / "profiles.jsonl"
    family_path = identity_dir / "final_family_assignments.jsonl"

    for path in (profiles_path, family_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    profiles = _iter_jsonl(profiles_path)
    family_rows = _iter_jsonl(family_path)

    config = GlobalTrajectoryCandidateConfig(
        max_append_gap_s=args.max_append_gap_s,
    )
    result = build_global_trajectory_candidates(
        trajectory_profiles=profiles,
        family_rows=family_rows,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "global_trajectory_candidates"
        / GLOBAL_TRAJECTORY_CANDIDATE_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    append_path = output_dir / "append_candidates.jsonl"
    insert_path = output_dir / "insert_candidates.jsonl"
    _write_jsonl(
        append_path,
        (row.to_dict() for row in result.append_candidates),
    )
    _write_jsonl(
        insert_path,
        (row.to_dict() for row in result.insert_candidates),
    )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "identity_count": len(profiles),
        "append_gap_s": _stats([
            row.geometry.gap_s for row in result.append_candidates
        ]),
        "append_distance_m": _stats([
            row.geometry.endpoint_distance_m
            for row in result.append_candidates
        ]),
        "append_required_speed_mps": _stats([
            row.geometry.required_speed_after_slack_mps
            for row in result.append_candidates
            if row.geometry.required_speed_after_slack_mps is not None
        ]),
        "append_velocity_prediction_available_count": sum(
            (
                row.geometry.forward_prediction_error_m is not None
                or row.geometry.backward_prediction_error_m is not None
            )
            for row in result.append_candidates
        ),
        "insert_left_gap_s": _stats([
            row.left_geometry.gap_s
            for row in result.insert_candidates
        ]),
        "insert_right_gap_s": _stats([
            row.right_geometry.gap_s
            for row in result.insert_candidates
        ]),
        "insert_total_bridge_gap_s": _stats([
            row.left_geometry.gap_s + row.right_geometry.gap_s
            for row in result.insert_candidates
        ]),
        "insert_max_required_speed_mps": _stats([
            max(
                row.left_geometry.required_speed_after_slack_mps or 0.0,
                row.right_geometry.required_speed_after_slack_mps or 0.0,
            )
            for row in result.insert_candidates
        ]),
        "insert_two_sided_velocity_prediction_count": sum(
            (
                (
                    row.left_geometry.forward_prediction_error_m is not None
                    or row.left_geometry.backward_prediction_error_m is not None
                )
                and (
                    row.right_geometry.forward_prediction_error_m is not None
                    or row.right_geometry.backward_prediction_error_m is not None
                )
            )
            for row in result.insert_candidates
        ),
        "source_trajectory_profiles": str(profiles_path),
        "source_family_assignments": str(family_path),
        "reid_scoring_applied": False,
        "competition_solver_applied": False,
        "provisional": True,
        "canonical": False,
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "candidate_version": GLOBAL_TRAJECTORY_CANDIDATE_VERSION,
            "append_candidates": append_path.name,
            "insert_candidates": insert_path.name,
            "summary": summary_path.name,
            "candidate_count": summary["candidate_count"],
            "identity_decisions_changed": False,
        },
    )

    print(
        "[global-trajectory-candidates-7G.2] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        f"identities={len(profiles)} "
        f"append={len(result.append_candidates)} "
        f"insert={len(result.insert_candidates)} "
        f"total={summary['candidate_count']}"
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        f"append_evaluated={result.append_pair_evaluated_count} "
        f"append_physics_rejected={result.append_physically_rejected_count}"
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        f"insert_evaluated={result.insert_pair_evaluated_count} "
        f"insert_physics_rejected={result.insert_physically_rejected_count} "
        f"host_gaps_with_candidate={result.host_gap_with_candidate_count}"
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        "append_family="
        + json.dumps(
            summary["append_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        "insert_family="
        + json.dumps(
            summary["insert_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        "append_gap="
        + json.dumps(summary["append_gap_s"], sort_keys=True)
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        "insert_bridge="
        + json.dumps(
            summary["insert_total_bridge_gap_s"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-candidates-7G.2] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
