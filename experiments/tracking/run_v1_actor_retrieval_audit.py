#!/usr/bin/env python3
"""Run Hito 7B.4 sparse-candidate recall diagnostics on an existing clip.

No detector, crop extraction or PRTReID inference is executed.  The runner
reuses Tracking V1 tracklets and the already-generated PRTReID embeddings.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.retrieval_audit import (  # noqa: E402
    RETRIEVAL_AUDIT_VERSION,
    RetrievalAuditConfig,
    run_sparse_candidate_recall_audit,
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
        "--min-split-samples",
        type=int,
        default=4,
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

    audit = run_sparse_candidate_recall_audit(
        tracklets=tracklets,
        samples=samples,
        full_profiles=profiles,
        evidence_config=evidence_config,
        audit_config=RetrievalAuditConfig(
            local_gap_horizon_s=args.local_gap_horizon_s,
            min_split_samples=args.min_split_samples,
        ),
    )
    payload = audit.to_dict()
    payload.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "source_tracklets": str(tracklets_path),
            "source_embeddings": str(embeddings_path),
            "tracklet_count": len(tracklets),
            "sample_count": len(samples),
            "profile_count": len(profiles),
        }
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "actor_retrieval_audit"
        / RETRIEVAL_AUDIT_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "summary.json"
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    local = payload["local_physics"]
    reid = payload["reid_split"]
    print(
        "[actor-retrieval-audit-7B.4] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[actor-retrieval-audit-7B.4] "
        f"local_proxy={local['eligible_adjacent_transition_count']} "
        f"recovered={local['recovered_transition_count']} "
        f"missed={local['missed_transition_count']} "
        f"recall={local['recall']}"
    )
    print(
        "[actor-retrieval-audit-7B.4] "
        f"reid_split_tracklets={reid['eligible_tracklet_count']} "
        f"queries={reid['query_count']} "
        f"mrr={reid['mean_reciprocal_rank']}"
    )
    print(
        "[actor-retrieval-audit-7B.4] "
        f"recall_at={json.dumps(reid['recall_at'], sort_keys=True)}"
    )
    print(
        "[actor-retrieval-audit-7B.4] "
        f"rank={json.dumps(reid['rank'], sort_keys=True)}"
    )
    print(
        "[actor-retrieval-audit-7B.4] "
        f"mate_cosine={json.dumps(reid['mate_cosine'], sort_keys=True)}"
    )
    print(
        f"[actor-retrieval-audit-7B.4] artifact={output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
