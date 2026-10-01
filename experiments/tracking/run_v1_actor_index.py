#!/usr/bin/env python3
"""Run Hito 7B.1 sparse temporal/physical tracklet indexing.

This runner reads already-completed Tracking V1 tracklets for one materialized
clip. It does NOT run RF-DETR, crops, ReID, identity solving or persistent
tracking.

Output is diagnostic only:

tracking/v1/diagnostics/actor_index/sparse-tracklet-index-v1/
    nodes.jsonl
    coobserved_pairs.jsonl
    hard_conflicts.jsonl
    local_transitions.jsonl
    summary.json
    index.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.index import (  # noqa: E402
    ACTOR_INDEX_VERSION,
    SparseTrackletIndexConfig,
    build_sparse_tracklet_index,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402
from tracking.tracklets.models import Tracklet  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            yield payload


def _load_tracklets(path: Path) -> tuple[Tracklet, ...]:
    return tuple(
        Tracklet.from_dict(payload)
        for payload in _iter_jsonl(path)
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
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


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    ordered = sorted(float(value) for value in values)

    def percentile(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = probability * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        fraction = position - low
        return (
            ordered[low] * (1.0 - fraction)
            + ordered[high] * fraction
        )

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--local-gap-horizon-s",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--max-transition-speed-mps",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--transition-position-slack-m",
        type=float,
        default=1.5,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    tracklets_path = (
        context.tracking_root
        / "tracklets"
        / "tracklets.jsonl"
    )
    if not tracklets_path.is_file():
        raise FileNotFoundError(
            f"Tracking V1 tracklets do not exist: {tracklets_path}"
        )

    tracklets = _load_tracklets(tracklets_path)
    config = SparseTrackletIndexConfig(
        local_gap_horizon_s=args.local_gap_horizon_s,
        max_transition_speed_mps=args.max_transition_speed_mps,
        transition_position_slack_m=(
            args.transition_position_slack_m
        ),
    )
    index = build_sparse_tracklet_index(tracklets, config)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else context.tracking_root
        / "diagnostics"
        / "actor_index"
        / ACTOR_INDEX_VERSION
    )

    hard_conflicts = index.hard_coexistence_conflicts()
    close_pairs = index.close_coobserved_pairs()
    transitions = tuple(index.iter_local_transition_features())
    reachable = tuple(
        item for item in transitions
        if item.physically_reachable
    )
    unreachable = tuple(
        item for item in transitions
        if not item.physically_reachable
    )

    nodes_path = output_dir / "nodes.jsonl"
    coobserved_path = output_dir / "coobserved_pairs.jsonl"
    conflicts_path = output_dir / "hard_conflicts.jsonl"
    transitions_path = output_dir / "local_transitions.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    node_count = _write_jsonl(
        nodes_path,
        (item.to_dict() for item in index.nodes),
    )
    coobserved_count = _write_jsonl(
        coobserved_path,
        (item.to_dict() for item in index.coobserved_pairs),
    )
    conflict_count = _write_jsonl(
        conflicts_path,
        (item.to_dict() for item in hard_conflicts),
    )
    transition_count = _write_jsonl(
        transitions_path,
        (item.to_dict() for item in transitions),
    )

    possible_pair_count = (
        index.tracklet_count * (index.tracklet_count - 1) // 2
    )

    successor_degree = Counter(
        item.predecessor_tracklet_id
        for item in transitions
    )
    reachable_successor_degree = Counter(
        item.predecessor_tracklet_id
        for item in reachable
    )

    summary = {
        "schema_version": 1,
        "index_version": ACTOR_INDEX_VERSION,
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_tracklets": str(tracklets_path),
        "tracklet_count": index.tracklet_count,
        "observation_count": index.observation_count,
        "observed_frame_count": index.observed_frame_count,
        "total_possible_pair_count": possible_pair_count,
        "exact_coobserved_pair_count": len(index.coobserved_pairs),
        "hard_coexistence_conflict_count": len(hard_conflicts),
        "close_coobserved_pair_count": len(close_pairs),
        "local_transition_pair_count": len(transitions),
        "local_reachable_transition_count": len(reachable),
        "local_unreachable_transition_count": len(unreachable),
        "local_transition_fraction_of_all_pairs": (
            len(transitions) / possible_pair_count
            if possible_pair_count
            else None
        ),
        "local_reachable_fraction_of_all_pairs": (
            len(reachable) / possible_pair_count
            if possible_pair_count
            else None
        ),
        "successor_query_degree": _stats(
            [
                float(successor_degree.get(node.tracklet_id, 0))
                for node in index.nodes
            ]
        ),
        "reachable_successor_degree": _stats(
            [
                float(
                    reachable_successor_degree.get(
                        node.tracklet_id,
                        0,
                    )
                )
                for node in index.nodes
            ]
        ),
        "config": config.to_dict(),
        "identity_decisions": False,
        "appearance_used": False,
        "population_forced": False,
    }
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "index_version": ACTOR_INDEX_VERSION,
            "nodes": nodes_path.name,
            "coobserved_pairs": coobserved_path.name,
            "hard_conflicts": conflicts_path.name,
            "local_transitions": transitions_path.name,
            "summary": summary_path.name,
            "node_count": node_count,
            "coobserved_pair_count": coobserved_count,
            "hard_conflict_count": conflict_count,
            "local_transition_count": transition_count,
        },
    )

    print(
        "[actor-index-7B.1] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[actor-index-7B.1] "
        f"tracklets={index.tracklet_count} "
        f"observations={index.observation_count} "
        f"frames={index.observed_frame_count}"
    )
    print(
        "[actor-index-7B.1] "
        f"possible_pairs={possible_pair_count} "
        f"coobserved_pairs={len(index.coobserved_pairs)} "
        f"hard_conflicts={len(hard_conflicts)} "
        f"close_pairs={len(close_pairs)}"
    )
    print(
        "[actor-index-7B.1] "
        f"local_pairs={len(transitions)} "
        f"reachable={len(reachable)} "
        f"unreachable={len(unreachable)} "
        f"reachable_fraction="
        f"{summary['local_reachable_fraction_of_all_pairs']:.6f}"
    )
    print(
        "[actor-index-7B.1] "
        f"reachable_degree="
        f"{json.dumps(summary['reachable_successor_degree'], sort_keys=True)}"
    )
    print(f"[actor-index-7B.1] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
