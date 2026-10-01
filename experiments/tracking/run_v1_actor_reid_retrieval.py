#!/usr/bin/env python3
"""Run Hito 7B.2 global sparse ReID retrieval on an existing Tracking V1 clip.

This runner reuses already-computed PRTReID embeddings.  It does NOT execute:
RF-DETR, crop extraction, ReID inference, identity solving or persistent
tracking.

The retrieval is global over the clip.  Exact physical coexistence conflicts
are rebuilt from the tracklets and excluded before top-K selection.
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


from tracking.actors.index import (  # noqa: E402
    SparseTrackletIndexConfig,
    build_sparse_tracklet_index,
)
from tracking.actors.reid_retrieval import (  # noqa: E402
    REID_RETRIEVAL_VERSION,
    SparseReIDRetrievalConfig,
    build_sparse_reid_retrieval,
)
from tracking.appearance.evidence import (  # noqa: E402
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
)
from tracking.clip_context import (  # noqa: E402
    resolve_tracking_clip_context,
)
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
        "--top-k-per-tracklet",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--cosine-block-size",
        type=int,
        default=256,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    tracking_root = context.tracking_root

    tracklets_path = (
        tracking_root
        / "tracklets"
        / "tracklets.jsonl"
    )
    embeddings_path = (
        tracking_root
        / "diagnostics"
        / "reid_benchmark"
        / "prtreid-soccernet-baseline-global-v1"
        / "embeddings.jsonl"
    )

    if not tracklets_path.is_file():
        raise FileNotFoundError(
            f"Tracking V1 tracklets do not exist: {tracklets_path}"
        )
    if not embeddings_path.is_file():
        raise FileNotFoundError(
            f"PRTReID embeddings do not exist: {embeddings_path}"
        )

    tracklets = _load_tracklets(tracklets_path)
    tracklet_ids = {
        tracklet.tracklet_id
        for tracklet in tracklets
    }

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

    unknown_profile_ids = sorted(
        profile.tracklet_id
        for profile in profiles
        if profile.tracklet_id not in tracklet_ids
    )
    if unknown_profile_ids:
        raise ValueError(
            "appearance profiles reference unknown tracklets: "
            f"{unknown_profile_ids[:10]}"
        )

    # Hito 7B.2 only needs exact coexistence hard conflicts from the physical
    # index.  The 1 s local horizon reflects the 7B.1 measurement, but it does
    # not affect coexistence conflict construction.
    physical_index = build_sparse_tracklet_index(
        tracklets,
        SparseTrackletIndexConfig(
            local_gap_horizon_s=1.0,
        ),
    )
    hard_conflicts = (
        physical_index.hard_coexistence_conflicts()
    )

    retrieval = build_sparse_reid_retrieval(
        profiles,
        hard_conflicts=hard_conflicts,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=args.top_k_per_tracklet,
            cosine_block_size=args.cosine_block_size,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else tracking_root
        / "diagnostics"
        / "actor_reid_retrieval"
        / REID_RETRIEVAL_VERSION
    )

    profiles_path = output_dir / "profiles.jsonl"
    neighbors_path = output_dir / "neighbors.jsonl"
    candidates_path = output_dir / "candidate_pairs.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    profile_count = _write_jsonl(
        profiles_path,
        (
            profile.to_dict()
            for profile in retrieval.profiles
        ),
    )
    neighbor_count = _write_jsonl(
        neighbors_path,
        (
            neighbor.to_dict()
            for neighbor in retrieval.neighbors
        ),
    )
    candidate_count = _write_jsonl(
        candidates_path,
        (
            candidate.to_dict()
            for candidate in retrieval.candidates
        ),
    )

    summary = retrieval.summary(
        total_tracklet_count=len(tracklets),
    )
    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "source_tracklets": str(tracklets_path),
            "source_embeddings": str(embeddings_path),
            "sample_count": len(samples),
            "physical_hard_conflict_count_all_tracklets": (
                len(hard_conflicts)
            ),
        }
    )
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "retrieval_version": REID_RETRIEVAL_VERSION,
            "profiles": profiles_path.name,
            "neighbors": neighbors_path.name,
            "candidate_pairs": candidates_path.name,
            "summary": summary_path.name,
            "profile_count": profile_count,
            "directed_neighbor_count": neighbor_count,
            "candidate_pair_count": candidate_count,
            "identity_decisions": False,
        },
    )

    print(
        "[actor-reid-7B.2] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[actor-reid-7B.2] "
        f"tracklets={len(tracklets)} "
        f"samples={len(samples)} "
        f"profiles={len(profiles)} "
        f"unprofiled={len(tracklets)-len(profiles)}"
    )
    print(
        "[actor-reid-7B.2] "
        f"hard_conflicts_all={len(hard_conflicts)} "
        f"hard_conflicts_profiled="
        f"{summary['hard_conflict_pair_count']}"
    )
    print(
        "[actor-reid-7B.2] "
        f"profile_possible_pairs="
        f"{summary['possible_profile_pair_count']} "
        f"directed_neighbors="
        f"{summary['directed_neighbor_count']} "
        f"candidate_pairs="
        f"{summary['candidate_pair_count']}"
    )
    print(
        "[actor-reid-7B.2] "
        f"candidate_fraction="
        f"{summary['candidate_fraction_of_profile_pairs']:.6f} "
        f"reduction="
        f"{summary['pair_reduction_fraction']:.6f} "
        f"mutual={summary['mutual_candidate_count']} "
        f"one_sided={summary['one_sided_candidate_count']}"
    )
    print(
        "[actor-reid-7B.2] "
        f"candidate_degree="
        f"{json.dumps(summary['candidate_degree'], sort_keys=True)}"
    )
    print(
        "[actor-reid-7B.2] "
        f"candidate_cosine="
        f"{json.dumps(summary['candidate_aggregate_cosine'], sort_keys=True)}"
    )
    print(f"[actor-reid-7B.2] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
