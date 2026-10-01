#!/usr/bin/env python3
"""Run Tracking V1 Hito 7E.2 conservative temporal actor backbone.

Inputs:
* canonical V1 tracklets;
* Hito 7B sparse hard conflicts;
* Hito 7E.1 scored links.

Only strong local sequential continuations are allowed into this provisional
backbone. Long-gap appearance rescue remains deferred to the later actor-chain
re-identification stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.temporal_backbone import (  # noqa: E402
    TEMPORAL_BACKBONE_VERSION,
    TemporalBackboneConfig,
    solve_temporal_backbone,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402
from tracking.tracklets.models import Tracklet  # noqa: E402


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


def _load_tracklets(path: Path) -> tuple[Tracklet, ...]:
    return tuple(
        Tracklet.from_dict(row)
        for row in _iter_jsonl(path)
    )


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
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
        "--max-endpoint-rank",
        type=int,
        default=3,
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

    tracklets_path = (
        root / "tracklets" / "tracklets.jsonl"
    )
    conflicts_path = (
        root
        / "diagnostics"
        / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )
    scores_path = (
        root
        / "diagnostics"
        / "actor_link_scoring"
        / "actor-link-evidence-v1"
        / "scored_links.jsonl"
    )

    for path in (
        tracklets_path,
        conflicts_path,
        scores_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    tracklets = _load_tracklets(tracklets_path)
    conflicts = _iter_jsonl(conflicts_path)
    scores = _iter_jsonl(scores_path)

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=scores,
        hard_conflicts=conflicts,
        config=TemporalBackboneConfig(
            min_rank_score=args.min_rank_score,
            max_endpoint_rank=args.max_endpoint_rank,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            root
            / "diagnostics"
            / "actor_temporal_backbone"
            / TEMPORAL_BACKBONE_VERSION
        )
    )

    actors_path = output_dir / "actors.jsonl"
    assignments_path = output_dir / "assignments.jsonl"
    edges_path = output_dir / "selected_edges.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    actor_count = _write_jsonl(
        actors_path,
        (item.to_dict() for item in result.paths),
    )
    assignment_count = _write_jsonl(
        assignments_path,
        result.assignments,
    )
    selected_edge_count = _write_jsonl(
        edges_path,
        (
            item.to_dict()
            for item in result.accepted_edges
        ),
    )

    summary = result.summary()
    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "tracklet_count": len(tracklets),
            "hard_conflict_count": len(conflicts),
            "scored_link_count": len(scores),
            "source_tracklets": str(tracklets_path),
            "source_hard_conflicts": str(conflicts_path),
            "source_scored_links": str(scores_path),
        }
    )
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "backbone_version": TEMPORAL_BACKBONE_VERSION,
            "actors": actors_path.name,
            "assignments": assignments_path.name,
            "selected_edges": edges_path.name,
            "summary": summary_path.name,
            "actor_count": actor_count,
            "assignment_count": assignment_count,
            "selected_edge_count": selected_edge_count,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[actor-temporal-backbone-7E.2] "
        f"match={context.match_id} "
        f"clip={context.clip_id}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"tracklets={len(tracklets)} "
        f"eligible={summary['eligible_edge_count']} "
        f"selected={summary['accepted_edge_count']}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"paths={summary['path_count']} "
        f"non_singleton={summary['non_singleton_path_count']} "
        f"singletons={summary['singleton_path_count']} "
        f"largest={summary['largest_path_tracklet_count']}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"path_sizes="
        f"{json.dumps(summary['path_tracklet_count'], sort_keys=True)}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"selected_scores="
        f"{json.dumps(summary['accepted_score'], sort_keys=True)}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"selected_gaps="
        f"{json.dumps(summary['accepted_gap_s'], sort_keys=True)}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"rejections="
        f"{json.dumps(summary['rejection_counts'], sort_keys=True)}"
    )
    print(
        "[actor-temporal-backbone-7E.2] "
        f"artifact={index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
