#!/usr/bin/env python3
"""Publish validated Tracking V1 identities as TrackingStage.IDENTITIES.

Hito 3F consumes the already validated `global-backbone-v1` solution and its
3C/3D/3E audits. It performs no solving. After all guardrails pass, it writes:

tracking/v1/identities/
    identities.jsonl
    assignments.jsonl
    identity_links.jsonl
    summary.json
    index.json

and marks TrackingStage.IDENTITIES as COMPLETED.

Starting this stage invalidates TrackingStage.FINAL, as intended: Hito 4 must
be recomputed from the newly canonical identities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import TrackingStage
from tracking.identities.publish import (
    CANDIDATE_GRAPH_VERSION,
    DIAGNOSTICS_VERSION,
    FRAGMENTATION_AUDIT_VERSION,
    PUBLISH_VERSION,
    SCORING_VERSION,
    SOLVER_VERSION,
    VISUAL_AUDIT_VERSION,
    IdentityPublishInputs,
    iter_jsonl,
    load_json,
    prepare_identity_publication,
)


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["match_id"]), str(payload["clip_id"])


def _source_paths(
    *,
    store: TrackingArtifactStore,
) -> dict[str, Path]:
    root = store.root
    solver_dir = (
        root
        / "diagnostics"
        / "identity_solver"
        / SOLVER_VERSION
    )
    return {
        "identities": solver_dir / "identities.jsonl",
        "assignments": solver_dir / "assignments.jsonl",
        "intra_identity_links": (
            solver_dir / "intra_identity_links.jsonl"
        ),
        "selected_backbone_links": (
            solver_dir / "selected_backbone_links.jsonl"
        ),
        "solver_summary": solver_dir / "summary.json",
        "tracklets": root / "tracklets" / "tracklets.jsonl",
        "candidate_edges": (
            root
            / "diagnostics"
            / "identity_candidates"
            / CANDIDATE_GRAPH_VERSION
            / "edges.jsonl"
        ),
        "diagnostics_summary": (
            root
            / "diagnostics"
            / "identity_diagnostics"
            / DIAGNOSTICS_VERSION
            / "summary.json"
        ),
        "visual_summary": (
            root
            / "diagnostics"
            / "identity_visual_audit"
            / VISUAL_AUDIT_VERSION
            / "summary.json"
        ),
        "fragmentation_summary": (
            root
            / "diagnostics"
            / "identity_fragmentation"
            / FRAGMENTATION_AUDIT_VERSION
            / "summary.json"
        ),
    }


def run_publish_identities(
    *,
    store: TrackingArtifactStore,
    paths: dict[str, Path],
    restart: bool = False,
) -> dict:
    """Validate first, then atomically publish the canonical stage."""

    inputs = IdentityPublishInputs(
        identities=tuple(iter_jsonl(paths["identities"])),
        assignments=tuple(iter_jsonl(paths["assignments"])),
        intra_identity_links=tuple(
            iter_jsonl(paths["intra_identity_links"])
        ),
        selected_backbone_links=tuple(
            iter_jsonl(paths["selected_backbone_links"])
        ),
        tracklets=tuple(iter_jsonl(paths["tracklets"])),
        candidate_edges=tuple(
            iter_jsonl(paths["candidate_edges"])
        ),
        solver_summary=load_json(paths["solver_summary"]),
        diagnostics_summary=load_json(
            paths["diagnostics_summary"]
        ),
        visual_summary=load_json(paths["visual_summary"]),
        fragmentation_summary=load_json(
            paths["fragmentation_summary"]
        ),
    )

    # Crucial: perform every guardrail before mutating stage state.
    result = prepare_identity_publication(inputs)

    begin_metadata = {
        "source_solver_version": SOLVER_VERSION,
        "publisher_version": PUBLISH_VERSION,
        "validated_before_stage_begin": True,
    }
    store.begin_stage(
        TrackingStage.IDENTITIES,
        version=PUBLISH_VERSION,
        config={
            "solver_version": SOLVER_VERSION,
            "candidate_graph_version": CANDIDATE_GRAPH_VERSION,
            "scoring_version": SCORING_VERSION,
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "visual_audit_version": VISUAL_AUDIT_VERSION,
            "fragmentation_audit_version": (
                FRAGMENTATION_AUDIT_VERSION
            ),
        },
        metadata=begin_metadata,
        invalidate_downstream=True,
        clean_outputs=True,
        restart=restart,
    )

    try:
        identities_file = store.write_jsonl(
            TrackingStage.IDENTITIES,
            "identities.jsonl",
            result.identities,
        )
        assignments_file = store.write_jsonl(
            TrackingStage.IDENTITIES,
            "assignments.jsonl",
            result.assignments,
        )
        links_file = store.write_jsonl(
            TrackingStage.IDENTITIES,
            "identity_links.jsonl",
            result.identity_links,
        )
        summary_file = store.write_json(
            TrackingStage.IDENTITIES,
            "summary.json",
            result.summary,
        )

        index_payload = {
            "schema_version": 1,
            "publisher_version": PUBLISH_VERSION,
            "canonical": True,
            "identity_stage_completed": True,
            "identities": {
                "ref": identities_file.ref,
                "record_count": identities_file.record_count,
                "sha256": identities_file.sha256,
            },
            "assignments": {
                "ref": assignments_file.ref,
                "record_count": assignments_file.record_count,
                "sha256": assignments_file.sha256,
            },
            "identity_links": {
                "ref": links_file.ref,
                "record_count": links_file.record_count,
                "sha256": links_file.sha256,
            },
            "summary": {
                "ref": summary_file.ref,
                "sha256": summary_file.sha256,
            },
            "provenance": {
                **result.provenance,
                "sources": {
                    key: str(path)
                    for key, path in sorted(paths.items())
                },
            },
        }
        index_file = store.write_json(
            TrackingStage.IDENTITIES,
            "index.json",
            index_payload,
        )

        manifest = store.complete_stage(
            TrackingStage.IDENTITIES,
            artifact=index_file,
            summary=summary_file,
            metadata={
                "canonical": True,
                "identity_count": result.summary[
                    "identity_count"
                ],
                "tracklet_count": result.summary[
                    "tracklet_count"
                ],
                "selected_backbone_bridge_count": (
                    result.summary[
                        "selected_backbone_bridge_count"
                    ]
                ),
                "provenance": result.provenance,
            },
        )
    except Exception as exc:
        store.fail_stage(
            TrackingStage.IDENTITIES,
            exc,
            metadata={
                "publisher_version": PUBLISH_VERSION,
            },
        )
        raise

    published_summary = dict(result.summary)
    published_summary["artifact"] = index_file.ref
    published_summary["manifest_status"] = (
        manifest.stage(TrackingStage.IDENTITIES).status.value
    )
    published_summary["manifest_version"] = (
        manifest.stage(TrackingStage.IDENTITIES).version
    )
    return published_summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/tracking/dev_clip.json",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument(
        "--restart",
        action="store_true",
        help=(
            "restart a stale RUNNING identities stage; not needed for "
            "normal first publication or republishing a completed stage"
        ),
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
    paths = _source_paths(store=store)

    print(
        "[identity-publish] validating "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_publish_identities(
        store=store,
        paths=paths,
        restart=args.restart,
    )
    print(
        "[identity-publish] completed "
        f"identities={summary['identity_count']} "
        f"tracklets={summary['tracklet_count']} "
        f"singletons={summary['singleton_identity_count']} "
        f"links={summary['identity_link_count']} "
        f"backbone_bridges="
        f"{summary['selected_backbone_bridge_count']}"
    )
    print(
        "[identity-publish] validation "
        f"hard_violations="
        f"{summary['hard_constraint_violation_count']} "
        f"status={summary['manifest_status']} "
        f"version={summary['manifest_version']}"
    )
    print(
        "[identity-publish] artifact="
        f"{summary['artifact']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
