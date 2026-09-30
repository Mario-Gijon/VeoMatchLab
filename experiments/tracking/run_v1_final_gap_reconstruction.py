#!/usr/bin/env python3
"""Run Tracking V1 Hito 4C: conservative safe-gap reconstruction.

Consumes:
- Hito 4B `observation-fusion-v1/trajectories.jsonl`;
- Hito 3A V1 `physical-compatibility-v1/edges.jsonl`.

Produces:
tracking/v1/diagnostics/final_gap_reconstruction/safe-gap-v1/
    trajectories.jsonl
    gap_events.jsonl
    summary.json
    index.json

This stage does not smooth trajectories and does not complete
TrackingStage.FINAL.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import StageStatus, TrackingStage
from tracking.final import (
    GAP_RECONSTRUCTION_VERSION,
    GapReconstructionConfig,
    load_candidate_edges,
    load_trajectories,
    reconstruct_safe_gaps,
)
from tracking.final.reconstruction import iter_jsonl


FUSION_VERSION = "observation-fusion-v1"
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


def run_gap_reconstruction(
    *,
    trajectories_path: Path,
    candidate_edges_path: Path,
    output_dir: Path,
    config: GapReconstructionConfig,
) -> dict[str, Any]:
    trajectories = load_trajectories(
        iter_jsonl(trajectories_path)
    )
    candidate_edges = load_candidate_edges(
        iter_jsonl(candidate_edges_path)
    )

    rebuilt, events, summary = (
        reconstruct_safe_gaps(
            trajectories=trajectories,
            candidate_edges=candidate_edges,
            config=config,
        )
    )
    summary["source_trajectories"] = str(
        trajectories_path
    )
    summary["source_candidate_edges"] = str(
        candidate_edges_path
    )

    trajectories_out = (
        output_dir / "trajectories.jsonl"
    )
    events_out = output_dir / "gap_events.jsonl"
    summary_out = output_dir / "summary.json"
    index_out = output_dir / "index.json"

    trajectory_count = _write_jsonl(
        trajectories_out,
        (
            trajectory.to_dict()
            for trajectory in rebuilt
        ),
    )
    event_count = _write_jsonl(
        events_out,
        events,
    )
    _write_json(summary_out, summary)
    _write_json(
        index_out,
        {
            "schema_version": 1,
            "reconstruction_version": (
                GAP_RECONSTRUCTION_VERSION
            ),
            "provisional": True,
            "final_stage_completed": False,
            "trajectories": trajectories_out.name,
            "gap_events": events_out.name,
            "summary": summary_out.name,
            "trajectory_count": trajectory_count,
            "gap_event_count": event_count,
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
    parser.add_argument("--trajectories")
    parser.add_argument("--candidate-edges")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--max-same-tracklet-gap-s",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--max-cross-tracklet-gap-s",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--max-endpoint-speed-mps",
        type=float,
        default=12.0,
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
    manifest = store.load_manifest()
    identity_stage = manifest.stage(
        TrackingStage.IDENTITIES
    )
    if identity_stage.status is not StageStatus.COMPLETED:
        raise RuntimeError(
            "TrackingStage.IDENTITIES must be completed "
            "before Hito 4C"
        )

    trajectories_path = (
        Path(args.trajectories)
        if args.trajectories
        else store.root
        / "diagnostics"
        / "final_observation_fusion"
        / FUSION_VERSION
        / "trajectories.jsonl"
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
        / "final_gap_reconstruction"
        / GAP_RECONSTRUCTION_VERSION
    )

    config = GapReconstructionConfig(
        max_same_tracklet_gap_s=(
            args.max_same_tracklet_gap_s
        ),
        max_cross_tracklet_gap_s=(
            args.max_cross_tracklet_gap_s
        ),
        max_endpoint_speed_mps=(
            args.max_endpoint_speed_mps
        ),
    )

    print(
        "[final-gap-reconstruction] starting "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_gap_reconstruction(
        trajectories_path=trajectories_path,
        candidate_edges_path=(
            candidate_edges_path
        ),
        output_dir=output_dir,
        config=config,
    )
    print(
        "[final-gap-reconstruction] completed "
        f"trajectories={summary['trajectory_count']} "
        f"input_points={summary['input_point_count']} "
        f"output_points={summary['output_point_count']} "
        f"interpolated="
        f"{summary['interpolated_point_count']}"
    )
    print(
        "[final-gap-reconstruction] gaps "
        f"detected={summary['gap_event_count']} "
        f"filled={summary['filled_gap_count']} "
        f"preserved={summary['preserved_gap_count']} "
        f"max_filled_s="
        f"{summary['max_filled_gap_duration_s']:.6f} "
        f"max_speed="
        f"{summary['max_filled_endpoint_speed_mps']:.6f}"
    )
    print(
        "[final-gap-reconstruction] cadence "
        f"frames={summary['sampling_cadence']['frame_step']} "
        f"dt={summary['sampling_cadence']['time_step_s']:.6f}"
    )
    print(
        "[final-gap-reconstruction] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
