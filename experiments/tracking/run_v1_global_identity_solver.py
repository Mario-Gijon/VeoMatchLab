#!/usr/bin/env python3
"""Run Tracking V1 Hito 3B-B: global physical-identity solver.

This runner intentionally publishes a PROVISIONAL solver artifact under
diagnostics.  It does not complete TrackingStage.IDENTITIES yet; Hito 3C will
audit the partition before it becomes the canonical identities stage.

Inputs:
- Hito 3A physical candidate graph;
- Hito 3B-A soft scored links.

Outputs:
tracking/v1/diagnostics/identity_solver/global-backbone-v1/
    index.json
    identities.jsonl
    assignments.jsonl
    intra_identity_links.jsonl
    cut_positive_links.jsonl
    selected_backbone_links.jsonl
    baseline_comparison.json
    solver_trace.json
    summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.identities.global_solver import (
    BACKBONE_SOLVER_VERSION,
    BackboneAugmentationConfig,
    assignment_records,
    backbone_solution_diagnostics,
    build_solver_input,
    cut_positive_link_records,
    identity_records,
    intra_identity_link_records,
    iter_jsonl,
    load_tracklet_windows,
    solve_global_backbone_partition,
    validate_backbone_partition,
)


CANDIDATE_GRAPH_VERSION = "physical-compatibility-v1"
SCORING_VERSION = "soft-link-v1"


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


def run_global_identity_solver(
    *,
    candidate_edges_path: Path,
    scored_links_path: Path,
    nodes_path: Path,
    output_dir: Path,
    config: BackboneAugmentationConfig | None = None,
) -> dict[str, Any]:
    candidate_edges = list(iter_jsonl(candidate_edges_path))
    scored_links = list(iter_jsonl(scored_links_path))
    tracklet_windows = load_tracklet_windows(
        iter_jsonl(nodes_path)
    )

    solver_input = build_solver_input(
        candidate_edges,
        scored_links,
    )

    (
        result,
        baseline,
        selected_links,
        backbone_diagnostics,
        start_results,
    ) = solve_global_backbone_partition(
        solver_input,
        candidate_edges=candidate_edges,
        tracklet_windows=tracklet_windows,
        config=config,
    )

    validation = validate_backbone_partition(
        final_result=result,
        baseline_result=baseline,
        selected_links=selected_links,
        solver_input=solver_input,
    )
    if validation["hard_constraint_violation_count"] != 0:
        raise RuntimeError(
            "backbone solver produced hard cannot-link violations"
        )
    if not validation["objective_matches"]:
        raise RuntimeError(
            "final clique diagnostic objective failed consistency check"
        )
    if not validation["baseline_cluster_preserved"]:
        raise RuntimeError(
            "backbone augmentation unexpectedly split a baseline cluster"
        )
    if validation[
        "selected_backbone_link_violation_count"
    ] != 0:
        raise RuntimeError(
            "selected backbone bridge was not preserved"
        )

    identities = identity_records(
        result,
        solver_input,
        solver_version=BACKBONE_SOLVER_VERSION,
    )
    assignments = assignment_records(
        identities,
        solver_version=BACKBONE_SOLVER_VERSION,
    )
    intra_links = intra_identity_link_records(
        result,
        solver_input,
        solver_version=BACKBONE_SOLVER_VERSION,
    )
    cut_positive = cut_positive_link_records(
        result,
        solver_input,
        solver_version=BACKBONE_SOLVER_VERSION,
    )
    selected_rows = [
        link.to_dict()
        for link in selected_links
    ]

    comparison = backbone_solution_diagnostics(
        final_result=result,
        baseline_result=baseline,
        selected_links=selected_links,
        solver_input=solver_input,
        backbone_diagnostics=backbone_diagnostics,
    )

    summary = {
        "schema_version": 1,
        "solver_version": BACKBONE_SOLVER_VERSION,
        "provisional": True,
        "identity_stage_completed": False,
        "tracklet_count": len(
            solver_input.tracklet_ids
        ),
        "hard_cannot_link_count": len(
            solver_input.hard_cannot_links
        ),
        "soft_link_count": len(
            solver_input.utilities
        ),
        "selected_start": result.start_name,
        "selected_iterations": result.iterations,
        **comparison,
        "validation": validation,
        "baseline_start_results": list(start_results),
        "source_candidate_edges": str(
            candidate_edges_path
        ),
        "source_scored_links": str(
            scored_links_path
        ),
        "source_nodes": str(nodes_path),
    }

    identities_path = output_dir / "identities.jsonl"
    assignments_path = output_dir / "assignments.jsonl"
    intra_path = output_dir / "intra_identity_links.jsonl"
    cut_path = output_dir / "cut_positive_links.jsonl"
    selected_path = output_dir / "selected_backbone_links.jsonl"
    comparison_path = output_dir / "baseline_comparison.json"
    trace_path = output_dir / "solver_trace.json"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    identity_count = _write_jsonl(
        identities_path,
        identities,
    )
    assignment_count = _write_jsonl(
        assignments_path,
        assignments,
    )
    intra_count = _write_jsonl(
        intra_path,
        intra_links,
    )
    cut_count = _write_jsonl(
        cut_path,
        cut_positive,
    )
    selected_count = _write_jsonl(
        selected_path,
        selected_rows,
    )
    _write_json(
        comparison_path,
        comparison,
    )
    _write_json(
        trace_path,
        {
            "schema_version": 1,
            "solver_version": BACKBONE_SOLVER_VERSION,
            "baseline_start": baseline.start_name,
            "baseline_trace": list(baseline.trace),
            "baseline_starts": list(start_results),
            "backbone_trace": list(result.trace),
            "selected_backbone_links": selected_rows,
        },
    )
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "solver_version": BACKBONE_SOLVER_VERSION,
            "provisional": True,
            "identity_stage_completed": False,
            "identities": identities_path.name,
            "assignments": assignments_path.name,
            "intra_identity_links": intra_path.name,
            "cut_positive_links": cut_path.name,
            "selected_backbone_links": selected_path.name,
            "baseline_comparison": comparison_path.name,
            "solver_trace": trace_path.name,
            "summary": summary_path.name,
            "identity_count": identity_count,
            "assignment_count": assignment_count,
            "intra_identity_link_count": intra_count,
            "cut_positive_link_count": cut_count,
            "selected_backbone_link_count": selected_count,
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
    parser.add_argument("--candidate-edges")
    parser.add_argument("--scored-links")
    parser.add_argument("--nodes")
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
        config_match, config_clip = _load_dev_clip(
            Path(args.config)
        )
        match_id = match_id or config_match
        clip_id = clip_id or config_clip

    store = TrackingArtifactStore.for_clip(
        args.data_root,
        match_id,
        clip_id,
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
    nodes_path = (
        Path(args.nodes)
        if args.nodes
        else store.root
        / "diagnostics"
        / "identity_candidates"
        / CANDIDATE_GRAPH_VERSION
        / "nodes.jsonl"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "identity_solver"
        / BACKBONE_SOLVER_VERSION
    )

    print(
        "[global-identity-solver] starting "
        f"match={match_id} clip={clip_id}"
    )
    print(
        f"[global-identity-solver] candidates={candidate_edges_path}"
    )
    print(
        f"[global-identity-solver] scored={scored_links_path}"
    )

    summary = run_global_identity_solver(
        candidate_edges_path=candidate_edges_path,
        scored_links_path=scored_links_path,
        nodes_path=nodes_path,
        output_dir=output_dir,
        config=BackboneAugmentationConfig(
            strong_positive_quantile=(
                args.strong_positive_quantile
            ),
            strong_utility_floor=(
                args.strong_utility_floor
            ),
            boundary_tolerance_s=(
                args.boundary_tolerance_s
            ),
        ),
    )

    print(
        "[global-identity-solver] completed "
        f"tracklets={summary['tracklet_count']} "
        f"baseline_identities={summary['baseline_identity_count']} "
        f"identities={summary['final_identity_count']} "
        f"selected_backbone="
        f"{summary['selected_backbone_link_count']}"
    )
    print(
        "[global-identity-solver] validation "
        f"cannot_link_violations="
        f"{summary['validation']['hard_constraint_violation_count']} "
        f"baseline_preserved="
        f"{summary['validation']['baseline_cluster_preserved']} "
        f"final_positive_capture="
        f"{summary['final_positive_utility_capture_fraction']}"
    )
    print(
        "[global-identity-solver] provisional_artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
