#!/usr/bin/env python3
"""Run Hito 7B.3 unified sparse actor-candidate graph.

Consumes existing Tracking V1 tracklets + existing PRTReID embeddings.
Does NOT run detector, crop extraction, ReID inference, solver or persistent
tracking.
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


from tracking.actors.candidates import (  # noqa: E402
    UNIFIED_CANDIDATE_VERSION,
    UnifiedCandidateConfig,
    build_unified_candidate_graph,
)
from tracking.actors.index import (  # noqa: E402
    SparseTrackletIndexConfig,
    build_sparse_tracklet_index,
)
from tracking.actors.reid_retrieval import (  # noqa: E402
    SparseReIDRetrievalConfig,
    build_sparse_reid_retrieval,
)
from tracking.appearance.evidence import (  # noqa: E402
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
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
        "--local-gap-horizon-s",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--top-k-per-tracklet",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--cosine-block-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--skip-rich-appearance",
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

    tracklets_path = root / "tracklets" / "tracklets.jsonl"
    embeddings_path = (
        root
        / "diagnostics"
        / "reid_benchmark"
        / "prtreid-soccernet-baseline-global-v1"
        / "embeddings.jsonl"
    )
    if not tracklets_path.is_file():
        raise FileNotFoundError(tracklets_path)
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)

    tracklets = _load_tracklets(tracklets_path)

    evidence_config = AppearanceEvidenceConfig()
    samples = tuple(
        iter_embedding_jsonl(
            embeddings_path,
            evidence_config,
        )
    )
    profiles = tuple(
        build_tracklet_profiles(
            samples,
            evidence_config,
        )
    )

    physical_index = build_sparse_tracklet_index(
        tracklets,
        SparseTrackletIndexConfig(
            local_gap_horizon_s=args.local_gap_horizon_s,
        ),
    )

    # ReID retrieval must exclude every hard coexistence conflict before top-K.
    coexistence_conflicts = (
        physical_index.hard_coexistence_conflicts()
    )
    retrieval = build_sparse_reid_retrieval(
        profiles,
        hard_conflicts=coexistence_conflicts,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=args.top_k_per_tracklet,
            cosine_block_size=args.cosine_block_size,
        ),
    )

    graph = build_unified_candidate_graph(
        physical_index=physical_index,
        reid_retrieval=retrieval,
        profiles=profiles,
        config=UnifiedCandidateConfig(
            local_gap_horizon_s=args.local_gap_horizon_s,
            compute_rich_appearance=(
                not args.skip_rich_appearance
            ),
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "actor_candidates"
        / UNIFIED_CANDIDATE_VERSION
    )

    candidates_path = output_dir / "candidates.jsonl"
    conflicts_path = output_dir / "hard_conflicts.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    candidate_count = _write_jsonl(
        candidates_path,
        (
            candidate.to_dict()
            for candidate in graph.candidates
        ),
    )
    conflict_count = _write_jsonl(
        conflicts_path,
        (
            conflict.to_dict()
            for conflict in graph.hard_conflicts
        ),
    )

    summary = graph.summary()
    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "sample_count": len(samples),
            "source_tracklets": str(tracklets_path),
            "source_embeddings": str(embeddings_path),
            "reid_retrieval_candidate_count": len(
                retrieval.candidates
            ),
            "reid_directed_neighbor_count": len(
                retrieval.neighbors
            ),
        }
    )
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "candidate_version": UNIFIED_CANDIDATE_VERSION,
            "candidates": candidates_path.name,
            "hard_conflicts": conflicts_path.name,
            "summary": summary_path.name,
            "candidate_count": candidate_count,
            "hard_conflict_count": conflict_count,
            "identity_decisions": False,
        },
    )

    print(
        "[actor-candidates-7B.3] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"tracklets={summary['tracklet_count']} "
        f"profiles={summary['profiled_tracklet_count']} "
        f"unprofiled={summary['unprofiled_tracklet_count']}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"possible_pairs={summary['total_possible_pair_count']} "
        f"hard_conflicts={summary['hard_conflict_count']} "
        f"candidates={summary['candidate_count']}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"candidate_fraction="
        f"{summary['candidate_fraction_of_all_pairs']:.6f} "
        f"reduction={summary['pair_reduction_fraction']:.6f}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"reasons={json.dumps(summary['candidate_reason_counts'], sort_keys=True)}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"reason_combinations="
        f"{json.dumps(summary['candidate_reason_combinations'], sort_keys=True)}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"hard_reasons="
        f"{json.dumps(summary['hard_conflict_reason_counts'], sort_keys=True)}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"rich_appearance="
        f"{summary['candidate_with_rich_appearance_count']} "
        f"without_rich="
        f"{summary['candidate_without_rich_appearance_count']}"
    )
    print(
        "[actor-candidates-7B.3] "
        f"degree={json.dumps(summary['candidate_degree'], sort_keys=True)}"
    )
    print(f"[actor-candidates-7B.3] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
