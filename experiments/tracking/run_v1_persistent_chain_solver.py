#!/usr/bin/env python3
"""Run Tracking V1 Hito 7E.3B conservative persistent-chain solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.chain_solver import (  # noqa: E402
    PERSISTENT_CHAIN_SOLVER_VERSION,
    PersistentChainSolverConfig,
    solve_persistent_chain_paths,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        )
        + "\n",
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
                )
                + "\n"
            )
            count += 1
    return count


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--min-rank-score",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--competition-mode",
        choices=("mutual_best", "mutual_top3"),
        default="mutual_best",
    )
    parser.add_argument(
        "--allow-different-kit",
        action="store_true",
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

    backbone_root = (
        root / "diagnostics" / "actor_temporal_backbone"
        / "temporal-actor-backbone-v1"
    )
    actors_path = backbone_root / "actors.jsonl"
    assignments_path = backbone_root / "assignments.jsonl"

    chain_reid_root = (
        root / "diagnostics" / "actor_chain_reid"
        / "actor-chain-reid-v1"
    )
    candidates_path = chain_reid_root / "candidates.jsonl"

    conflicts_path = (
        root / "diagnostics" / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )

    for path in (
        actors_path,
        assignments_path,
        candidates_path,
        conflicts_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    actors = _iter_jsonl(actors_path)
    assignments = _iter_jsonl(assignments_path)
    candidates = _iter_jsonl(candidates_path)
    conflicts = _iter_jsonl(conflicts_path)

    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=candidates,
        hard_conflicts=conflicts,
        config=PersistentChainSolverConfig(
            min_rank_score=args.min_rank_score,
            competition_mode=args.competition_mode,
            allow_different_kit=args.allow_different_kit,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "persistent_chain_solver"
        / PERSISTENT_CHAIN_SOLVER_VERSION
    )

    identities_path = output_dir / "identities.jsonl"
    actor_assignments_path = output_dir / "actor_assignments.jsonl"
    tracklet_assignments_path = output_dir / "tracklet_assignments.jsonl"
    selected_links_path = output_dir / "selected_links.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    identity_count = _write_jsonl(
        identities_path,
        (identity.to_dict() for identity in result.identities),
    )
    actor_assignment_count = _write_jsonl(
        actor_assignments_path,
        result.actor_assignments,
    )
    tracklet_assignment_count = _write_jsonl(
        tracklet_assignments_path,
        result.tracklet_assignments,
    )
    selected_link_count = _write_jsonl(
        selected_links_path,
        (link.to_dict() for link in result.selected_links),
    )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_actor_count": len(actors),
        "source_tracklet_assignment_count": len(assignments),
        "source_chain_candidate_count": len(candidates),
        "source_hard_conflict_count": len(conflicts),
        "source_backbone_actors": str(actors_path),
        "source_backbone_assignments": str(assignments_path),
        "source_chain_candidates": str(candidates_path),
        "source_hard_conflicts": str(conflicts_path),
    })
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "solver_version": PERSISTENT_CHAIN_SOLVER_VERSION,
            "identities": identities_path.name,
            "actor_assignments": actor_assignments_path.name,
            "tracklet_assignments": tracklet_assignments_path.name,
            "selected_links": selected_links_path.name,
            "summary": summary_path.name,
            "identity_count": identity_count,
            "actor_assignment_count": actor_assignment_count,
            "tracklet_assignment_count": tracklet_assignment_count,
            "selected_link_count": selected_link_count,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[persistent-chain-solver-7E.3B] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"source_actors={len(actors)} "
        f"candidates={len(candidates)} "
        f"eligible={summary['eligible_candidate_count']} "
        f"selected={summary['selected_link_count']}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"identities={summary['identity_count']} "
        f"non_singleton={summary['non_singleton_identity_count']} "
        f"singletons={summary['singleton_identity_count']} "
        f"largest_actors={summary['largest_identity_actor_count']} "
        f"largest_tracklets={summary['largest_identity_tracklet_count']}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"identity_actor_sizes="
        f"{json.dumps(summary['identity_actor_count'], sort_keys=True)}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"selected_scores="
        f"{json.dumps(summary['selected_score'], sort_keys=True)}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"selected_gaps="
        f"{json.dumps(summary['selected_gap_s'], sort_keys=True)}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"selected_kit="
        f"{json.dumps(summary['selected_kit_family_counts'], sort_keys=True)}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"rejections="
        f"{json.dumps(summary['rejection_counts'], sort_keys=True)}"
    )
    print(
        "[persistent-chain-solver-7E.3B] "
        f"artifact={index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
