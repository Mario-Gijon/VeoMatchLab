#!/usr/bin/env python3
"""Run Tracking V1 Hito 3A: physical compatibility / candidate graph.

This is a diagnostics/intermediate artifact for the future IDENTITIES solver.
It deliberately does NOT begin or complete the TrackingStage.IDENTITIES stage.

Inputs:
- completed tracklets stage
- completed appearance stage (Hito 2F pairwise evidence)

Outputs:
tracking/v1/diagnostics/identity_candidates/physical-compatibility-v1/
    index.json
    nodes.jsonl
    edges.jsonl
    summary.json

All unordered tracklet pairs are represented.  A pair is either a candidate or
a hard physics-only cannot-link constraint.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.appearance.evidence import DEFAULT_MODEL_ID
from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import TrackingStage
from tracking.identities.candidates import (
    GRAPH_VERSION,
    PhysicalCompatibilityConfig,
    build_candidate_graph,
    build_candidate_graph_summary,
    iter_tracklet_nodes_jsonl,
    load_appearance_pairwise_jsonl,
)


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


def _write_jsonl(path: Path, records: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temp.open("w", encoding="utf-8") as destination:
            for record in records:
                destination.write(
                    json.dumps(
                        dict(record),
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


def run_candidate_diagnostics(
    *,
    tracklets_path: Path,
    appearance_edges_path: Path,
    output_dir: Path,
    config: PhysicalCompatibilityConfig,
) -> dict[str, Any]:
    nodes = list(iter_tracklet_nodes_jsonl(tracklets_path, config))
    appearance = load_appearance_pairwise_jsonl(appearance_edges_path)
    graph = build_candidate_graph(
        nodes,
        config,
        appearance_by_pair=appearance,
    )
    summary = build_candidate_graph_summary(graph, config)
    summary["source_tracklets"] = str(tracklets_path)
    summary["source_appearance_pairwise"] = str(appearance_edges_path)

    nodes_path = output_dir / "nodes.jsonl"
    edges_path = output_dir / "edges.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    node_count = _write_jsonl(
        nodes_path,
        (node.to_dict() for node in graph.nodes),
    )
    edge_count = _write_jsonl(
        edges_path,
        (edge.to_dict() for edge in graph.edges),
    )
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "graph_version": GRAPH_VERSION,
            "identity_decisions": False,
            "hard_constraints_only_from_physics": True,
            "nodes": nodes_path.name,
            "edges": edges_path.name,
            "summary": summary_path.name,
            "node_count": node_count,
            "edge_count": edge_count,
        },
    )
    return summary


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["match_id"]), str(payload["clip_id"])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/tracking/dev_clip.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument(
        "--output-dir",
        help="override diagnostics output directory",
    )
    parser.add_argument(
        "--max-transition-speed-mps",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--transition-position-slack-m",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--overlap-close-median-separation-m",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--overlap-hard-median-separation-m",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--single-frame-hard-separation-m",
        "--single-support-hard-separation-m",
        dest="single_frame_hard_separation_m",
        type=float,
        default=2.75,
        help=(
            "hard separation threshold for exactly one supported "
            "contemporaneous comparison; legacy option name retained"
        ),
    )
    parser.add_argument(
        "--overlap-interpolation-max-gap-s",
        type=float,
        default=0.35,
        help=(
            "maximum bracketing observation span allowed for local "
            "overlap-only interpolation"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    match_id = args.match_id
    clip_id = args.clip_id
    if not match_id or not clip_id:
        config_match, config_clip = _load_dev_clip(Path(args.config))
        match_id = match_id or config_match
        clip_id = clip_id or config_clip

    store = TrackingArtifactStore.for_clip(
        args.data_root,
        match_id,
        clip_id,
    )

    if not store.stage_completed(TrackingStage.TRACKLETS):
        raise RuntimeError("Hito 3A requires completed tracklets stage")
    if not store.stage_completed(TrackingStage.APPEARANCE):
        raise RuntimeError("Hito 3A requires completed appearance stage")

    manifest = store.load_manifest()
    tracklets_stage = manifest.stage(TrackingStage.TRACKLETS)
    appearance_stage = manifest.stage(TrackingStage.APPEARANCE)

    if tracklets_stage.artifact is None:
        raise RuntimeError("completed tracklets stage has no artifact")
    if appearance_stage.artifact is None:
        raise RuntimeError("completed appearance stage has no artifact")

    tracklets_path = store.resolve_artifact(tracklets_stage.artifact)
    appearance_index = store.read_json(appearance_stage.artifact)
    pairwise_ref = appearance_index.get("pairwise_evidence")
    if not isinstance(pairwise_ref, str) or not pairwise_ref:
        raise RuntimeError("appearance index lacks pairwise_evidence")
    appearance_edges_path = store.resolve_artifact(pairwise_ref)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "identity_candidates"
        / GRAPH_VERSION
    )

    compatibility_config = PhysicalCompatibilityConfig(
        max_transition_speed_mps=args.max_transition_speed_mps,
        transition_position_slack_m=args.transition_position_slack_m,
        overlap_close_median_separation_m=(
            args.overlap_close_median_separation_m
        ),
        overlap_hard_median_separation_m=(
            args.overlap_hard_median_separation_m
        ),
        single_frame_hard_separation_m=(
            args.single_frame_hard_separation_m
        ),
        overlap_interpolation_max_gap_s=(
            args.overlap_interpolation_max_gap_s
        ),
    )

    print(
        "[identity-candidates] starting "
        f"match={match_id} clip={clip_id}"
    )
    print(f"[identity-candidates] tracklets={tracklets_path}")
    print(f"[identity-candidates] appearance={appearance_edges_path}")

    summary = run_candidate_diagnostics(
        tracklets_path=tracklets_path,
        appearance_edges_path=appearance_edges_path,
        output_dir=output_dir,
        config=compatibility_config,
    )

    print(
        "[identity-candidates] completed "
        f"nodes={summary['node_count']} "
        f"pairs={summary['pair_count']} "
        f"candidates={summary['candidate_count']} "
        f"cannot_link={summary['cannot_link_count']} "
        f"appearance={summary['appearance_attached_count']}"
    )
    print(
        "[identity-candidates] reasons="
        + json.dumps(summary["reason_counts"], sort_keys=True)
    )
    print(f"[identity-candidates] artifact={output_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
