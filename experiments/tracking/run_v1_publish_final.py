#!/usr/bin/env python3
"""Publish Tracking V1 Hito 4F as the canonical TrackingStage.FINAL.

This runner performs all validation before mutating the stage manifest. It then
writes:

tracking/v1/final/
    trajectories.jsonl
    segment_breaks.jsonl
    review_queue.json
    summary.json
    index.json

and marks TrackingStage.FINAL as COMPLETED.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracking.artifacts import (
    TrackingArtifactStore,
)
from tracking.contracts import (
    StageStatus,
    TrackingStage,
)
from tracking.final.publish import (
    FINAL_DIAGNOSTICS_VERSION,
    FINAL_PUBLISH_VERSION,
    GAP_RECONSTRUCTION_VERSION,
    IDENTITY_PUBLISH_VERSION,
    OBSERVATION_FUSION_VERSION,
    SMOOTHING_VERSION,
    FinalPublishInputs,
    iter_jsonl,
    load_json,
    prepare_final_publication,
)


def _load_dev_clip(
    path: Path,
) -> tuple[str, str]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    return (
        str(payload["match_id"]),
        str(payload["clip_id"]),
    )


def _source_paths(
    *,
    store: TrackingArtifactStore,
) -> dict[str, Path]:
    root = store.root
    smoothing_dir = (
        root
        / "diagnostics"
        / "final_smoothing"
        / SMOOTHING_VERSION
    )
    diagnostics_dir = (
        root
        / "diagnostics"
        / "final_trajectory_diagnostics"
        / FINAL_DIAGNOSTICS_VERSION
    )
    identities_dir = (
        root / "identities"
    )

    return {
        "trajectories": (
            smoothing_dir
            / "trajectories.jsonl"
        ),
        "segment_breaks": (
            smoothing_dir
            / "segment_breaks.jsonl"
        ),
        "smoothing_summary": (
            smoothing_dir
            / "summary.json"
        ),
        "diagnostics_summary": (
            diagnostics_dir
            / "summary.json"
        ),
        "review_queue": (
            diagnostics_dir
            / "review_queue.json"
        ),
        "identities": (
            identities_dir
            / "identities.jsonl"
        ),
        "identity_summary": (
            identities_dir
            / "summary.json"
        ),
    }


def run_publish_final(
    *,
    store: TrackingArtifactStore,
    paths: dict[str, Path],
    restart: bool = False,
) -> dict:
    """Validate first, then atomically publish TrackingStage.FINAL."""

    manifest_before = store.load_manifest()
    identities_stage = (
        manifest_before.stage(
            TrackingStage.IDENTITIES
        )
    )
    if (
        identities_stage.status
        is not StageStatus.COMPLETED
    ):
        raise RuntimeError(
            "TrackingStage.IDENTITIES must be completed "
            "before publishing FINAL"
        )
    if (
        identities_stage.version
        != IDENTITY_PUBLISH_VERSION
    ):
        raise RuntimeError(
            "FINAL requires canonical identities from "
            f"{IDENTITY_PUBLISH_VERSION!r}; got "
            f"{identities_stage.version!r}"
        )

    inputs = FinalPublishInputs(
        trajectories=tuple(
            iter_jsonl(
                paths["trajectories"]
            )
        ),
        segment_breaks=tuple(
            iter_jsonl(
                paths["segment_breaks"]
            )
        ),
        review_queue=load_json(
            paths["review_queue"]
        ),
        smoothing_summary=load_json(
            paths["smoothing_summary"]
        ),
        diagnostics_summary=load_json(
            paths[
                "diagnostics_summary"
            ]
        ),
        identities=tuple(
            iter_jsonl(
                paths["identities"]
            )
        ),
        identity_summary=load_json(
            paths["identity_summary"]
        ),
    )

    # All guardrails run before begin_stage: a failed validation cannot destroy
    # or invalidate any previously published FINAL artifact.
    result = prepare_final_publication(
        inputs
    )

    store.begin_stage(
        TrackingStage.FINAL,
        version=FINAL_PUBLISH_VERSION,
        config={
            "identity_publisher_version": (
                IDENTITY_PUBLISH_VERSION
            ),
            "observation_fusion_version": (
                OBSERVATION_FUSION_VERSION
            ),
            "gap_reconstruction_version": (
                GAP_RECONSTRUCTION_VERSION
            ),
            "smoothing_version": (
                SMOOTHING_VERSION
            ),
            "diagnostics_version": (
                FINAL_DIAGNOSTICS_VERSION
            ),
        },
        metadata={
            "publisher_version": (
                FINAL_PUBLISH_VERSION
            ),
            "validated_before_stage_begin": True,
        },
        invalidate_downstream=True,
        clean_outputs=True,
        restart=restart,
    )

    try:
        trajectories_file = (
            store.write_jsonl(
                TrackingStage.FINAL,
                "trajectories.jsonl",
                result.trajectories,
            )
        )
        breaks_file = (
            store.write_jsonl(
                TrackingStage.FINAL,
                "segment_breaks.jsonl",
                result.segment_breaks,
            )
        )
        queue_file = store.write_json(
            TrackingStage.FINAL,
            "review_queue.json",
            result.review_queue,
        )
        summary_file = store.write_json(
            TrackingStage.FINAL,
            "summary.json",
            result.summary,
        )

        index_payload = {
            "schema_version": 1,
            "publisher_version": (
                FINAL_PUBLISH_VERSION
            ),
            "canonical": True,
            "final_stage_completed": True,
            "trajectories": {
                "ref": trajectories_file.ref,
                "record_count": (
                    trajectories_file.record_count
                ),
                "sha256": (
                    trajectories_file.sha256
                ),
            },
            "segment_breaks": {
                "ref": breaks_file.ref,
                "record_count": (
                    breaks_file.record_count
                ),
                "sha256": breaks_file.sha256,
            },
            "review_queue": {
                "ref": queue_file.ref,
                "sha256": queue_file.sha256,
            },
            "summary": {
                "ref": summary_file.ref,
                "sha256": summary_file.sha256,
            },
            "provenance": {
                **result.provenance,
                "sources": {
                    key: str(path)
                    for key, path
                    in sorted(
                        paths.items()
                    )
                },
            },
        }
        index_file = store.write_json(
            TrackingStage.FINAL,
            "index.json",
            index_payload,
        )

        manifest = store.complete_stage(
            TrackingStage.FINAL,
            artifact=index_file,
            summary=summary_file,
            metadata={
                "canonical": True,
                "trajectory_count": (
                    result.summary[
                        "trajectory_count"
                    ]
                ),
                "point_count": (
                    result.summary[
                        "point_count"
                    ]
                ),
                "segment_count": (
                    result.summary[
                        "segment_count"
                    ]
                ),
                "segment_break_count": (
                    result.summary[
                        "segment_break_count"
                    ]
                ),
                "review_point_count": (
                    result.summary[
                        "review_point_count"
                    ]
                ),
                "blocker_count": 0,
                "provenance": (
                    result.provenance
                ),
            },
        )
    except Exception as exc:
        store.fail_stage(
            TrackingStage.FINAL,
            exc,
            metadata={
                "publisher_version": (
                    FINAL_PUBLISH_VERSION
                ),
            },
        )
        raise

    published = dict(result.summary)
    final_stage = manifest.stage(
        TrackingStage.FINAL
    )
    published["artifact"] = (
        index_file.ref
    )
    published["manifest_status"] = (
        final_stage.status.value
    )
    published["manifest_version"] = (
        final_stage.version
    )
    return published


def parse_args(
    argv=None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        default=(
            "experiments/tracking/"
            "dev_clip.json"
        ),
    )
    parser.add_argument(
        "--data-root",
        default="data",
    )
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "restart a stale RUNNING FINAL stage; "
            "not needed for normal publication"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    match_id = args.match_id
    clip_id = args.clip_id
    if not match_id or not clip_id:
        cfg_match, cfg_clip = (
            _load_dev_clip(
                Path(args.config)
            )
        )
        match_id = (
            match_id or cfg_match
        )
        clip_id = (
            clip_id or cfg_clip
        )

    store = (
        TrackingArtifactStore.for_clip(
            args.data_root,
            match_id,
            clip_id,
        )
    )
    paths = _source_paths(
        store=store
    )

    print(
        "[final-publish] validating "
        f"match={match_id} "
        f"clip={clip_id}"
    )
    summary = run_publish_final(
        store=store,
        paths=paths,
        restart=args.restart,
    )
    print(
        "[final-publish] completed "
        f"trajectories="
        f"{summary['trajectory_count']} "
        f"points="
        f"{summary['point_count']} "
        f"segments="
        f"{summary['segment_count']} "
        f"gap_breaks="
        f"{summary['segment_break_count']}"
    )
    print(
        "[final-publish] review "
        f"points="
        f"{summary['review_point_count']} "
        f"identities="
        f"{summary['review_identity_count']} "
        f"blockers="
        f"{summary['blocker_count']}"
    )
    print(
        "[final-publish] stage "
        f"status="
        f"{summary['manifest_status']} "
        f"version="
        f"{summary['manifest_version']}"
    )
    print(
        "[final-publish] artifact="
        f"{summary['artifact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
