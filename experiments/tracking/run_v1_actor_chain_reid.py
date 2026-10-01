#!/usr/bin/env python3
"""Run Tracking V1 Hito 7E.3A chain-level ReID evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.chain_reid import (  # noqa: E402
    CHAIN_REID_VERSION,
    ChainReIDConfig,
    build_chain_candidates,
    build_chain_profiles,
    fit_chain_calibration,
    summarize_chain_reid,
)
from tracking.appearance.evidence import (  # noqa: E402
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    result = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain an object"
                )
            result.append(dict(payload))
    return result


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
    parser.add_argument("--top-k-successors", type=int, default=16)
    parser.add_argument("--top-k-predecessors", type=int, default=16)
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
    actor_path = backbone_root / "actors.jsonl"
    assignment_path = backbone_root / "assignments.jsonl"
    hard_path = (
        root / "diagnostics" / "actor_candidates"
        / "sparse-actor-candidates-v1" / "hard_conflicts.jsonl"
    )
    kit_path = (
        root / "diagnostics" / "kit_clustering"
        / "kit-clustering-v1" / "assignments.jsonl"
    )
    embeddings_path = (
        root / "diagnostics" / "reid_benchmark"
        / "prtreid-soccernet-baseline-global-v1"
        / "embeddings.jsonl"
    )
    trusted_positive_path = (
        root / "diagnostics" / "reid_calibration"
        / "reid-overlap-calibration-v1"
        / "split_positive.jsonl"
    )

    for path in (
        actor_path,
        assignment_path,
        hard_path,
        kit_path,
        embeddings_path,
        trusted_positive_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    actors = _iter_jsonl(actor_path)
    assignments = _iter_jsonl(assignment_path)
    hard_conflicts = _iter_jsonl(hard_path)
    kit_assignments = _iter_jsonl(kit_path)
    trusted_positive_rows = _iter_jsonl(trusted_positive_path)
    trusted_positive_cosines = tuple(
        float(row["cosine"])
        for row in trusted_positive_rows
    )

    evidence_config = AppearanceEvidenceConfig()
    samples = tuple(
        iter_embedding_jsonl(
            embeddings_path,
            evidence_config,
        )
    )
    tracklet_profiles = tuple(
        build_tracklet_profiles(
            samples,
            evidence_config,
        )
    )

    chain_profiles = build_chain_profiles(
        actor_rows=actors,
        assignments=assignments,
        tracklet_profiles=tracklet_profiles,
        kit_assignments=kit_assignments,
    )

    calibration, positive_rows, negative_rows = fit_chain_calibration(
        chain_profiles=chain_profiles,
        tracklet_profiles=tracklet_profiles,
        assignments=assignments,
        hard_conflicts=hard_conflicts,
        trusted_positive_cosines=trusted_positive_cosines,
    )

    actor_by_tracklet = {
        str(row["tracklet_id"]): str(row["actor_id"])
        for row in assignments
    }
    hard_actor_pairs = set()
    for row in hard_conflicts:
        a = actor_by_tracklet.get(str(row["tracklet_a"]))
        b = actor_by_tracklet.get(str(row["tracklet_b"]))
        if a is None or b is None:
            continue
        if a == b:
            raise RuntimeError(
                f"hard conflict inside temporal actor {a}"
            )
        hard_actor_pairs.add(
            (a, b) if a < b else (b, a)
        )

    config = ChainReIDConfig(
        top_k_successors=args.top_k_successors,
        top_k_predecessors=args.top_k_predecessors,
    )
    candidates = build_chain_candidates(
        chain_profiles=chain_profiles,
        calibration=calibration,
        hard_actor_pairs=hard_actor_pairs,
        config=config,
    )

    summary = summarize_chain_reid(
        chain_profiles=chain_profiles,
        candidates=candidates,
        calibration=calibration,
        total_actor_count=len(actors),
        config=config,
    )
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "tracklet_profile_count": len(tracklet_profiles),
        "appearance_sample_count": len(samples),
        "hard_actor_pair_count": len(hard_actor_pairs),
        "source_backbone_actors": str(actor_path),
        "source_backbone_assignments": str(assignment_path),
        "source_hard_conflicts": str(hard_path),
        "source_kit_assignments": str(kit_path),
        "source_embeddings": str(embeddings_path),
        "source_trusted_positive": str(trusted_positive_path),
        "trusted_positive_count": len(trusted_positive_cosines),
        "chain_split_diagnostic_count": len(positive_rows),
        "chain_split_diagnostic": {
            "count": len(positive_rows),
            "min": (
                min(float(row["cosine"]) for row in positive_rows)
                if positive_rows else None
            ),
            "max": (
                max(float(row["cosine"]) for row in positive_rows)
                if positive_rows else None
            ),
        },
    })

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "actor_chain_reid"
        / CHAIN_REID_VERSION
    )

    profiles_path = output_dir / "profiles.jsonl"
    candidates_path = output_dir / "candidates.jsonl"
    positives_path = output_dir / "chain_split_diagnostic.jsonl"
    negatives_path = output_dir / "calibration_negative.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    profile_count = _write_jsonl(
        profiles_path,
        (row.to_dict() for row in chain_profiles),
    )
    candidate_count = _write_jsonl(
        candidates_path,
        (row.to_dict() for row in candidates),
    )
    positive_count = _write_jsonl(positives_path, positive_rows)
    negative_count = _write_jsonl(negatives_path, negative_rows)
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "chain_reid_version": CHAIN_REID_VERSION,
            "profiles": profiles_path.name,
            "candidates": candidates_path.name,
            "chain_split_diagnostic": positives_path.name,
            "calibration_negative": negatives_path.name,
            "summary": summary_path.name,
            "profile_count": profile_count,
            "candidate_count": candidate_count,
            "trusted_positive_count": len(trusted_positive_cosines),
            "chain_split_diagnostic_count": positive_count,
            "negative_calibration_count": negative_count,
            "identity_decisions": False,
        },
    )

    print(
        "[actor-chain-reid-7E.3A] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"actors={len(actors)} profiles={profile_count} "
        f"unprofiled={len(actors)-profile_count}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"trusted_positive={len(trusted_positive_cosines)} "
        f"chain_split_diagnostic={positive_count} "
        f"calibration_negative={negative_count} "
        f"hard_actor_pairs={len(hard_actor_pairs)}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"calibration={json.dumps(summary['calibration'], sort_keys=True)}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"candidates={candidate_count} "
        f"mutual_best={summary['mutual_best_count']} "
        f"mutual_top3={summary['mutual_top3_count']}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"score={json.dumps(summary['rank_score'], sort_keys=True)}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"cosine={json.dumps(summary['aggregate_cosine'], sort_keys=True)}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"gap={json.dumps(summary['gap_s'], sort_keys=True)}"
    )
    print(
        "[actor-chain-reid-7E.3A] "
        f"bands={json.dumps(summary['score_band_counts'], sort_keys=True)}"
    )
    print(f"[actor-chain-reid-7E.3A] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
