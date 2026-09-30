#!/usr/bin/env python3
"""Run Tracking V1 Hito 3E V1: fragmentation audit on global-backbone-v1.

This audit does not merge identities. It searches for strong cross-identity
trajectory handoffs, especially around singleton identities, and highlights
cases where a path-like handoff conflicts with the current clique objective.

Outputs:
tracking/v1/diagnostics/identity_fragmentation/fragmentation-audit-v1/
    boundary_links.jsonl
    identity_fragmentation.jsonl
    review_queue.json
    summary.json
    index.json
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.identities.fragmentation_audit import (
    FRAGMENTATION_AUDIT_VERSION,
    FragmentationAuditConfig,
    build_cross_identity_continuations,
    build_fragmentation_summary,
    build_identity_fragmentation_records,
    build_review_queue,
    iter_jsonl,
    load_candidate_graph,
    load_identity_meta,
    load_scored_links,
    load_tracklet_meta,
)


SOLVER_VERSION = "global-backbone-v1"
SCORING_VERSION = "soft-link-v1"
CANDIDATE_GRAPH_VERSION = "physical-compatibility-v1"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    count = 0
    try:
        with temp.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
            destination.flush()
            os.fsync(destination.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return count


def run_fragmentation_audit(
    *,
    identities_path: Path,
    tracklets_path: Path,
    candidate_edges_path: Path,
    scored_links_path: Path,
    output_dir: Path,
    config: FragmentationAuditConfig,
) -> dict[str, Any]:
    tracklets = load_tracklet_meta(
        iter_jsonl(tracklets_path)
    )
    identities = load_identity_meta(
        iter_jsonl(identities_path),
        tracklets,
    )
    candidate_edges, hard = load_candidate_graph(
        iter_jsonl(candidate_edges_path)
    )
    scored_links = load_scored_links(
        iter_jsonl(scored_links_path)
    )

    boundary_links, strong_threshold = (
        build_cross_identity_continuations(
            identities=identities,
            tracklets=tracklets,
            candidate_edges=candidate_edges,
            hard=hard,
            scored_links=scored_links,
            config=config,
        )
    )

    clip_start_s = min(
        tracklet.start_time_s
        for tracklet in tracklets.values()
    )
    clip_end_s = max(
        tracklet.end_time_s
        for tracklet in tracklets.values()
    )

    identity_records = (
        build_identity_fragmentation_records(
            identities=identities,
            boundary_links=boundary_links,
            clip_start_s=clip_start_s,
            clip_end_s=clip_end_s,
            config=config,
        )
    )
    summary = build_fragmentation_summary(
        identities=identities,
        boundary_links=boundary_links,
        identity_records=identity_records,
        strong_threshold=strong_threshold,
        config=config,
    )
    review_queue = build_review_queue(
        boundary_links=boundary_links,
        identity_records=identity_records,
    )

    summary["source_identities"] = str(identities_path)
    summary["source_tracklets"] = str(tracklets_path)
    summary["source_candidate_edges"] = str(
        candidate_edges_path
    )
    summary["source_scored_links"] = str(
        scored_links_path
    )

    boundary_path = output_dir / "boundary_links.jsonl"
    identities_out = (
        output_dir / "identity_fragmentation.jsonl"
    )
    review_path = output_dir / "review_queue.json"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    boundary_count = _write_jsonl(
        boundary_path,
        boundary_links,
    )
    identity_count = _write_jsonl(
        identities_out,
        identity_records,
    )
    _write_json(review_path, review_queue)
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "fragmentation_audit_version": (
                FRAGMENTATION_AUDIT_VERSION
            ),
            "provisional": True,
            "identity_stage_completed": False,
            "identity_decisions": False,
            "boundary_links": boundary_path.name,
            "identity_fragmentation": (
                identities_out.name
            ),
            "review_queue": review_path.name,
            "summary": summary_path.name,
            "boundary_link_count": boundary_count,
            "identity_count": identity_count,
        },
    )
    return summary


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["match_id"]), str(payload["clip_id"])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/tracking/dev_clip.json",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--identities")
    parser.add_argument("--tracklets")
    parser.add_argument("--candidate-edges")
    parser.add_argument("--scored-links")
    parser.add_argument("--solver-summary")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--strong-positive-quantile",
        type=float,
        default=0.95,
    )
    parser.add_argument(
        "--strong-utility-floor",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--boundary-tolerance-s",
        type=float,
        default=0.25,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    match_id = args.match_id
    clip_id = args.clip_id
    if not match_id or not clip_id:
        cfg_match, cfg_clip = _load_dev_clip(
            Path(args.config)
        )
        match_id = match_id or cfg_match
        clip_id = clip_id or cfg_clip

    store = TrackingArtifactStore.for_clip(
        args.data_root,
        match_id,
        clip_id,
    )

    identities_path = (
        Path(args.identities)
        if args.identities
        else store.root
        / "diagnostics"
        / "identity_solver"
        / SOLVER_VERSION
        / "identities.jsonl"
    )
    tracklets_path = (
        Path(args.tracklets)
        if args.tracklets
        else store.root
        / "tracklets"
        / "tracklets.jsonl"
    )
    candidate_edges_path = (
        Path(args.candidate_edges)
        if args.candidate_edges
        else store.root
        / "diagnostics"
        / "identity_candidates"
        / CANDIDATE_GRAPH_VERSION
        / "edges.jsonl"
    )
    scored_links_path = (
        Path(args.scored_links)
        if args.scored_links
        else store.root
        / "diagnostics"
        / "identity_link_scoring"
        / SCORING_VERSION
        / "scored_links.jsonl"
    )
    solver_summary_path = (
        Path(args.solver_summary)
        if args.solver_summary
        else store.root
        / "diagnostics"
        / "identity_solver"
        / SOLVER_VERSION
        / "summary.json"
    )
    solver_summary = json.loads(
        solver_summary_path.read_text(
            encoding="utf-8"
        )
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "identity_fragmentation"
        / FRAGMENTATION_AUDIT_VERSION
    )

    config = FragmentationAuditConfig(
        strong_positive_quantile=(
            args.strong_positive_quantile
        ),
        strong_utility_floor=args.strong_utility_floor,
        boundary_tolerance_s=args.boundary_tolerance_s,
        strong_utility_threshold_override=float(
            solver_summary["strong_utility_threshold"]
        ),
    )

    print(
        "[identity-fragmentation] starting "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_fragmentation_audit(
        identities_path=identities_path,
        tracklets_path=tracklets_path,
        candidate_edges_path=candidate_edges_path,
        scored_links_path=scored_links_path,
        output_dir=output_dir,
        config=config,
    )
    print(
        "[identity-fragmentation] completed "
        f"identities={summary['identity_count']} "
        f"singletons={summary['singleton_identity_count']} "
        f"positive_cross="
        f"{summary['positive_cross_identity_continuation_count']} "
        f"strong_boundary="
        f"{summary['strong_boundary_handoff_count']} "
        f"trajectory_clique_conflicts="
        f"{summary['trajectory_vs_clique_conflict_count']} "
        f"remaining_reciprocal="
        f"{summary['remaining_reciprocal_strong_boundary_handoff_count']}"
    )
    print(
        "[identity-fragmentation] strong_utility_threshold="
        f"{summary['strong_utility_threshold']:.6f}"
    )
    print(
        "[identity-fragmentation] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
