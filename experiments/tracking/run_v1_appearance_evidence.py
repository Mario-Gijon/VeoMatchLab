#!/usr/bin/env python3
"""Publish Tracking V1 Hito 2F: tracklet appearance evidence.

Input:
    tracking/v1/diagnostics/reid_benchmark/
        prtreid-soccernet-baseline-global-v1/embeddings.jsonl

Published APPEARANCE stage:
    tracking/v1/appearance/
        index.json
        profiles.jsonl
        pairwise_evidence.jsonl
        summary.json

The stage contains appearance evidence only.  It does not merge tracklets or
assign physical identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tracking.appearance.evidence import (
    DEFAULT_MODEL_ID,
    AppearanceEvidenceConfig,
    build_evidence_summary,
    build_pairwise_evidence,
    build_tracklet_profiles,
    iter_embedding_jsonl,
)
from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import TrackingStage


STAGE_VERSION = "2f-prtreid-evidence-v1"


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    match_id = payload.get("match_id")
    clip_id = payload.get("clip_id")
    if not isinstance(match_id, str) or not match_id:
        raise ValueError(f"{path} does not contain match_id")
    if not isinstance(clip_id, str) or not clip_id:
        raise ValueError(f"{path} does not contain clip_id")
    return match_id, clip_id


def _tracklet_ids_from_completed_stage(store: TrackingArtifactStore) -> set[str]:
    manifest = store.load_manifest()
    stage = manifest.stage(TrackingStage.TRACKLETS)
    if stage.artifact is None:
        raise RuntimeError("completed tracklets stage has no artifact reference")

    ids: set[str] = set()
    row_count = 0
    for row in store.iter_jsonl(stage.artifact):
        row_count += 1
        tracklet_id = row.get("tracklet_id")
        if isinstance(tracklet_id, str) and tracklet_id:
            ids.add(tracklet_id)

    # Some old/experimental tracklet serializers may omit the id in a shape
    # this helper recognizes.  Coverage can still use the row count.
    if ids and len(ids) != row_count:
        raise ValueError("tracklet artifact contains duplicate/missing tracklet_id values")
    if ids:
        return ids
    return {f"__row_{index:08d}" for index in range(row_count)}


def publish_appearance_stage(
    *,
    store: TrackingArtifactStore,
    embeddings_path: Path,
    config: AppearanceEvidenceConfig,
    restart: bool = False,
) -> dict[str, Any]:
    if not store.stage_completed(TrackingStage.TRACKLETS):
        raise RuntimeError("Hito 2F requires a completed tracklets stage")
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)

    tracklet_ids = _tracklet_ids_from_completed_stage(store)
    samples = list(iter_embedding_jsonl(embeddings_path, config))
    profiles = build_tracklet_profiles(samples, config)
    pairs = build_pairwise_evidence(profiles)

    if tracklet_ids and not next(iter(tracklet_ids)).startswith("__row_"):
        unknown = sorted({profile.tracklet_id for profile in profiles} - tracklet_ids)
        if unknown:
            raise ValueError(
                "appearance samples reference tracklets absent from the completed "
                f"tracklet artifact: {unknown[:10]}"
            )

    summary = build_evidence_summary(
        profiles,
        pairs,
        total_tracklet_count=len(tracklet_ids),
    )
    summary["stage_version"] = STAGE_VERSION
    summary["source_embeddings"] = str(embeddings_path)
    summary["config"] = config.to_dict()

    begun = False
    try:
        store.begin_stage(
            TrackingStage.APPEARANCE,
            version=STAGE_VERSION,
            config=config.to_dict(),
            metadata={
                "source_model_id": config.model_id,
                "source_embeddings": str(embeddings_path),
            },
            restart=restart,
        )
        begun = True

        profiles_file = store.write_jsonl(
            TrackingStage.APPEARANCE,
            "profiles.jsonl",
            (profile.to_dict() for profile in profiles),
        )
        pairs_file = store.write_jsonl(
            TrackingStage.APPEARANCE,
            "pairwise_evidence.jsonl",
            (pair.to_dict() for pair in pairs),
        )
        summary_file = store.write_json(
            TrackingStage.APPEARANCE,
            "summary.json",
            summary,
        )
        index_file = store.write_json(
            TrackingStage.APPEARANCE,
            "index.json",
            {
                "schema_version": 1,
                "stage_version": STAGE_VERSION,
                "model_id": config.model_id,
                "descriptor_name": config.descriptor_name,
                "profiles": profiles_file.ref,
                "pairwise_evidence": pairs_file.ref,
                "summary": summary_file.ref,
                "identity_decisions": False,
            },
        )

        store.complete_stage(
            TrackingStage.APPEARANCE,
            artifact=index_file,
            summary=summary_file,
            metadata={
                "profile_count": len(profiles),
                "pair_count": len(pairs),
                "sample_count": len(samples),
            },
        )
    except Exception as exc:
        if begun:
            try:
                store.fail_stage(TrackingStage.APPEARANCE, exc)
            except Exception:
                pass
        raise

    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/tracking/dev_clip.json",
        help="dev clip JSON used only to resolve match_id and clip_id",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--descriptor-name", default="reid")
    parser.add_argument(
        "--embeddings-jsonl",
        help="override the default PRTReID benchmark embeddings.jsonl",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="restart an appearance stage left in RUNNING state",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    match_id = args.match_id
    clip_id = args.clip_id
    if not match_id or not clip_id:
        cfg_match, cfg_clip = _load_dev_clip(Path(args.config))
        match_id = match_id or cfg_match
        clip_id = clip_id or cfg_clip

    store = TrackingArtifactStore.for_clip(
        args.data_root,
        match_id,
        clip_id,
    )

    embeddings_path = (
        Path(args.embeddings_jsonl)
        if args.embeddings_jsonl
        else store.root
        / "diagnostics"
        / "reid_benchmark"
        / args.model_id
        / "embeddings.jsonl"
    )

    config = AppearanceEvidenceConfig(
        descriptor_name=args.descriptor_name,
        model_id=args.model_id,
    )

    print(
        "[appearance-evidence] starting "
        f"match={match_id} clip={clip_id} model={args.model_id}"
    )
    print(f"[appearance-evidence] embeddings={embeddings_path}")

    summary = publish_appearance_stage(
        store=store,
        embeddings_path=embeddings_path,
        config=config,
        restart=args.restart,
    )

    print(
        "[appearance-evidence] completed "
        f"samples={summary['sample_count']} "
        f"profiles={summary['profile_count']} "
        f"pairs={summary['pair_count']} "
        f"coverage={summary.get('profile_coverage')}"
    )
    print(f"[appearance-evidence] artifact={store.root / 'appearance' / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
