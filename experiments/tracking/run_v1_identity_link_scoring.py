#!/usr/bin/env python3
"""Run Tracking V1 Hito 3B-A: explainable soft link scoring.

Inputs:
- Hito 3A candidate graph
- Hito 3A node summaries
- PRTReID benchmark summary

Outputs:
tracking/v1/diagnostics/identity_link_scoring/soft-link-v1/
    index.json
    calibration.json
    scored_links.jsonl
    summary.json
    top_links.json

This runner does not start or complete TrackingStage.IDENTITIES.  It produces
solver input only.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.identities.link_scoring import (
    SCORING_VERSION,
    build_scoring_summary,
    fit_soft_link_calibration,
    iter_jsonl,
    load_appearance_calibration,
    load_json,
    load_node_confidence,
    score_candidate_edges,
)


CANDIDATE_GRAPH_VERSION = "physical-compatibility-v1"
PRTREID_MODEL_ID = "prtreid-soccernet-baseline-global-v1"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
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


def run_soft_link_scoring(
    *,
    candidate_edges_path: Path,
    nodes_path: Path,
    benchmark_summary_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_edges = list(iter_jsonl(candidate_edges_path))
    node_confidence = load_node_confidence(nodes_path)
    benchmark_summary = load_json(benchmark_summary_path)

    appearance_calibration = load_appearance_calibration(
        benchmark_summary
    )
    candidate_edges = [
        edge
        for edge in source_edges
        if edge.get("status") == "candidate"
    ]
    calibration = fit_soft_link_calibration(
        candidate_edges,
        appearance_calibration,
    )
    scored_links = score_candidate_edges(
        source_edges,
        calibration,
        node_confidence=node_confidence,
    )
    summary = build_scoring_summary(
        source_edges,
        scored_links,
        calibration,
    )
    summary["source_candidate_edges"] = str(candidate_edges_path)
    summary["source_nodes"] = str(nodes_path)
    summary["source_reid_benchmark_summary"] = str(
        benchmark_summary_path
    )

    calibration_path = output_dir / "calibration.json"
    links_path = output_dir / "scored_links.jsonl"
    summary_path = output_dir / "summary.json"
    top_links_path = output_dir / "top_links.json"
    index_path = output_dir / "index.json"

    _write_json(calibration_path, calibration.to_dict())
    link_count = _write_jsonl(
        links_path,
        (link.to_dict() for link in scored_links),
    )

    top_continuations = sorted(
        (
            link
            for link in scored_links
            if link.link_kind == "continuation"
        ),
        key=lambda link: (
            link.rank_score,
            link.evidence_confidence,
        ),
        reverse=True,
    )[:25]
    top_overlaps = sorted(
        (
            link
            for link in scored_links
            if link.link_kind == "overlap_duplicate"
        ),
        key=lambda link: (
            link.rank_score,
            link.evidence_confidence,
        ),
        reverse=True,
    )[:25]
    _write_json(
        top_links_path,
        {
            "schema_version": 1,
            "scoring_version": SCORING_VERSION,
            "identity_decisions": False,
            "continuation": [
                link.to_dict()
                for link in top_continuations
            ],
            "overlap_duplicate": [
                link.to_dict()
                for link in top_overlaps
            ],
        },
    )

    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "scoring_version": SCORING_VERSION,
            "identity_decisions": False,
            "hard_constraints_modified": False,
            "rank_score_is_probability": False,
            "calibration": calibration_path.name,
            "scored_links": links_path.name,
            "summary": summary_path.name,
            "top_links": top_links_path.name,
            "scored_link_count": link_count,
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
    parser.add_argument("--nodes")
    parser.add_argument("--benchmark-summary")
    parser.add_argument("--output-dir")
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

    candidate_dir = (
        store.root
        / "diagnostics"
        / "identity_candidates"
        / CANDIDATE_GRAPH_VERSION
    )
    candidate_edges_path = (
        Path(args.candidate_edges)
        if args.candidate_edges
        else candidate_dir / "edges.jsonl"
    )
    nodes_path = (
        Path(args.nodes)
        if args.nodes
        else candidate_dir / "nodes.jsonl"
    )
    benchmark_summary_path = (
        Path(args.benchmark_summary)
        if args.benchmark_summary
        else store.root
        / "diagnostics"
        / "reid_benchmark"
        / PRTREID_MODEL_ID
        / "summary.json"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "identity_link_scoring"
        / SCORING_VERSION
    )

    print(
        "[identity-link-scoring] starting "
        f"match={match_id} clip={clip_id}"
    )
    print(
        f"[identity-link-scoring] edges={candidate_edges_path}"
    )
    print(
        f"[identity-link-scoring] benchmark={benchmark_summary_path}"
    )

    summary = run_soft_link_scoring(
        candidate_edges_path=candidate_edges_path,
        nodes_path=nodes_path,
        benchmark_summary_path=benchmark_summary_path,
        output_dir=output_dir,
    )

    print(
        "[identity-link-scoring] completed "
        f"scored={summary['scored_link_count']} "
        f"continuation="
        f"{summary['link_kind_counts'].get('continuation', 0)} "
        f"overlap="
        f"{summary['link_kind_counts'].get('overlap_duplicate', 0)} "
        f"appearance={summary['appearance_available_count']}"
    )
    print(
        "[identity-link-scoring] rank_score="
        + json.dumps(summary["rank_score"], sort_keys=True)
    )
    print(
        "[identity-link-scoring] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
