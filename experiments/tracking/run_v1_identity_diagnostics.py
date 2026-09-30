#!/usr/bin/env python3
"""Run Tracking V1 Hito 3C V1: backbone-aware identity diagnostics.

Inputs:
- final global-backbone-v1 identities;
- baseline global-partition-v1 identities;
- selected backbone links;
- global-backbone-v1 solver summary;
- 3B-A scored links;
- 3A V1 candidate graph;
- canonical tracklets.

Outputs:
tracking/v1/diagnostics/identity_diagnostics/identity-backbone-audit-v1/
    index.json
    identity_audits.jsonl
    bridge_audits.jsonl
    baseline_identity_audits.jsonl
    review_queue.json
    summary.json

The old all-pairs clique split is retained only as a diagnostic.  A split that
exactly removes a selected backbone bridge is EXPECTED and is not treated as an
identity error.

This runner does NOT modify identities and does NOT complete
TrackingStage.IDENTITIES.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.identities.diagnostics import (
    DIAGNOSTICS_VERSION,
    audit_backbone_partition,
    iter_jsonl,
    load_candidate_edges_for_diagnostics,
    load_hard_cannot_links,
    load_selected_backbone_links,
    load_tracklet_spans,
    load_utilities,
)


FINAL_SOLVER_VERSION = "global-backbone-v1"
BASELINE_SOLVER_VERSION = "global-partition-v1"
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


def run_identity_diagnostics(
    *,
    identities_path: Path,
    baseline_identities_path: Path,
    selected_backbone_links_path: Path,
    solver_summary_path: Path,
    tracklets_path: Path,
    scored_links_path: Path,
    candidate_edges_path: Path,
    output_dir: Path,
    max_exact_split_size: int = 16,
    boundary_tolerance_s: float = 0.25,
) -> dict[str, Any]:
    final_identities = list(
        iter_jsonl(identities_path)
    )
    baseline_identities = list(
        iter_jsonl(baseline_identities_path)
    )
    selected_links = load_selected_backbone_links(
        iter_jsonl(
            selected_backbone_links_path
        )
    )
    solver_summary = json.loads(
        solver_summary_path.read_text(
            encoding="utf-8"
        )
    )
    spans = load_tracklet_spans(
        iter_jsonl(tracklets_path)
    )
    utilities = load_utilities(
        iter_jsonl(scored_links_path)
    )
    candidate_rows = list(
        iter_jsonl(candidate_edges_path)
    )
    hard = load_hard_cannot_links(
        candidate_rows
    )
    candidate_edges = (
        load_candidate_edges_for_diagnostics(
            candidate_rows
        )
    )

    (
        identity_audits,
        bridge_audits,
        baseline_identity_audits,
        summary,
        review_queue,
    ) = audit_backbone_partition(
        final_identities=final_identities,
        baseline_identities=baseline_identities,
        selected_backbone_links=selected_links,
        spans=spans,
        utilities=utilities,
        hard=hard,
        candidate_edges=candidate_edges,
        solver_summary=solver_summary,
        max_exact_split_size=max_exact_split_size,
        boundary_tolerance_s=boundary_tolerance_s,
    )

    summary["source_identities"] = str(
        identities_path
    )
    summary["source_baseline_identities"] = str(
        baseline_identities_path
    )
    summary["source_selected_backbone_links"] = str(
        selected_backbone_links_path
    )
    summary["source_solver_summary"] = str(
        solver_summary_path
    )
    summary["source_tracklets"] = str(
        tracklets_path
    )
    summary["source_scored_links"] = str(
        scored_links_path
    )
    summary["source_candidate_edges"] = str(
        candidate_edges_path
    )
    summary["max_exact_split_size"] = (
        max_exact_split_size
    )
    summary["boundary_tolerance_s"] = (
        boundary_tolerance_s
    )

    identity_path = (
        output_dir / "identity_audits.jsonl"
    )
    bridge_path = (
        output_dir / "bridge_audits.jsonl"
    )
    baseline_path = (
        output_dir
        / "baseline_identity_audits.jsonl"
    )
    review_path = (
        output_dir / "review_queue.json"
    )
    summary_path = (
        output_dir / "summary.json"
    )
    index_path = output_dir / "index.json"

    identity_count = _write_jsonl(
        identity_path,
        identity_audits,
    )
    bridge_count = _write_jsonl(
        bridge_path,
        bridge_audits,
    )
    baseline_count = _write_jsonl(
        baseline_path,
        baseline_identity_audits,
    )
    _write_json(
        review_path,
        review_queue,
    )
    _write_json(
        summary_path,
        summary,
    )
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "diagnostics_version": (
                DIAGNOSTICS_VERSION
            ),
            "provisional": True,
            "identity_stage_completed": False,
            "identity_decisions": False,
            "identity_audits": (
                identity_path.name
            ),
            "bridge_audits": (
                bridge_path.name
            ),
            "baseline_identity_audits": (
                baseline_path.name
            ),
            "review_queue": review_path.name,
            "summary": summary_path.name,
            "identity_audit_count": (
                identity_count
            ),
            "bridge_audit_count": (
                bridge_count
            ),
            "baseline_identity_audit_count": (
                baseline_count
            ),
        },
    )
    return summary


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(
        path.read_text(encoding="utf-8")
    )
    return (
        str(payload["match_id"]),
        str(payload["clip_id"]),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--config",
        default="experiments/tracking/dev_clip.json",
    )
    parser.add_argument(
        "--data-root",
        default="data",
    )
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--identities")
    parser.add_argument(
        "--baseline-identities"
    )
    parser.add_argument(
        "--selected-backbone-links"
    )
    parser.add_argument(
        "--solver-summary"
    )
    parser.add_argument("--tracklets")
    parser.add_argument("--scored-links")
    parser.add_argument("--candidate-edges")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--max-exact-split-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--boundary-tolerance-s",
        type=float,
        default=0.25,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.max_exact_split_size < 2:
        raise ValueError(
            "--max-exact-split-size must be >= 2"
        )
    if args.boundary_tolerance_s < 0:
        raise ValueError(
            "--boundary-tolerance-s must be >= 0"
        )

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
    solver_root = (
        store.root
        / "diagnostics"
        / "identity_solver"
    )
    final_solver_dir = (
        solver_root
        / FINAL_SOLVER_VERSION
    )
    baseline_solver_dir = (
        solver_root
        / BASELINE_SOLVER_VERSION
    )

    identities_path = (
        Path(args.identities)
        if args.identities
        else final_solver_dir
        / "identities.jsonl"
    )
    baseline_identities_path = (
        Path(args.baseline_identities)
        if args.baseline_identities
        else baseline_solver_dir
        / "identities.jsonl"
    )
    selected_backbone_links_path = (
        Path(args.selected_backbone_links)
        if args.selected_backbone_links
        else final_solver_dir
        / "selected_backbone_links.jsonl"
    )
    solver_summary_path = (
        Path(args.solver_summary)
        if args.solver_summary
        else final_solver_dir
        / "summary.json"
    )
    tracklets_path = (
        Path(args.tracklets)
        if args.tracklets
        else store.root
        / "tracklets"
        / "tracklets.jsonl"
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
    candidate_edges_path = (
        Path(args.candidate_edges)
        if args.candidate_edges
        else store.root
        / "diagnostics"
        / "identity_candidates"
        / CANDIDATE_GRAPH_VERSION
        / "edges.jsonl"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "identity_diagnostics"
        / DIAGNOSTICS_VERSION
    )

    print(
        "[identity-diagnostics] starting "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_identity_diagnostics(
        identities_path=identities_path,
        baseline_identities_path=(
            baseline_identities_path
        ),
        selected_backbone_links_path=(
            selected_backbone_links_path
        ),
        solver_summary_path=(
            solver_summary_path
        ),
        tracklets_path=tracklets_path,
        scored_links_path=scored_links_path,
        candidate_edges_path=(
            candidate_edges_path
        ),
        output_dir=output_dir,
        max_exact_split_size=(
            args.max_exact_split_size
        ),
        boundary_tolerance_s=(
            args.boundary_tolerance_s
        ),
    )

    print(
        "[identity-diagnostics] completed "
        f"baseline={summary['baseline_identity_count']} "
        f"final={summary['final_identity_count']} "
        f"bridges={summary['selected_backbone_bridge_count']} "
        f"bridge_failures="
        f"{summary['bridge_validation_failure_count']} "
        f"hard_violations="
        f"{summary['final_hard_constraint_violation_count']} "
        f"unselected_reciprocal="
        f"{summary['unselected_reciprocal_eligible_boundary_handoff_count']}"
    )
    print(
        "[identity-diagnostics] legacy_split_pressure "
        f"expected="
        f"{summary['expected_legacy_clique_split_pressure_count']} "
        f"unexpected="
        f"{summary['unexpected_legacy_clique_split_pressure_count']}"
    )
    print(
        "[identity-diagnostics] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
