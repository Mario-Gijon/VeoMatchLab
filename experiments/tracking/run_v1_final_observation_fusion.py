#!/usr/bin/env python3
"""Run Tracking V1 Hito 4A/4B: contracts + same-frame observation fusion.

Consumes the canonical COMPLETED identities stage and canonical tracklets.
Produces an intermediate diagnostic artifact; it deliberately does NOT begin
or complete TrackingStage.FINAL.

Outputs:
tracking/v1/diagnostics/final_observation_fusion/observation-fusion-v1/
    trajectories.jsonl
    fusion_events.jsonl
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
from tracking.contracts import StageStatus, TrackingStage
from tracking.final import (
    OBSERVATION_FUSION_VERSION,
    ObservationFusionConfig,
    build_observation_fused_trajectories,
)
from tracking.final.fusion import iter_jsonl


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


def run_observation_fusion(
    *,
    identities_path: Path,
    tracklets_path: Path,
    output_dir: Path,
    config: ObservationFusionConfig,
) -> dict[str, Any]:
    trajectories, events, summary = (
        build_observation_fused_trajectories(
            identity_rows=iter_jsonl(identities_path),
            tracklet_rows=iter_jsonl(tracklets_path),
            config=config,
        )
    )
    summary["source_identities"] = str(
        identities_path
    )
    summary["source_tracklets"] = str(
        tracklets_path
    )

    trajectories_path = (
        output_dir / "trajectories.jsonl"
    )
    events_path = (
        output_dir / "fusion_events.jsonl"
    )
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    trajectory_count = _write_jsonl(
        trajectories_path,
        (
            trajectory.to_dict()
            for trajectory in trajectories
        ),
    )
    event_count = _write_jsonl(
        events_path,
        events,
    )
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "fusion_version": (
                OBSERVATION_FUSION_VERSION
            ),
            "provisional": True,
            "final_stage_completed": False,
            "trajectories": trajectories_path.name,
            "fusion_events": events_path.name,
            "summary": summary_path.name,
            "trajectory_count": trajectory_count,
            "fusion_event_count": event_count,
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
    parser.add_argument("--tracklets")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--max-fusion-diameter-m",
        type=float,
        default=1.5,
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
            "before Hito 4A/4B"
        )

    identities_path = (
        Path(args.identities)
        if args.identities
        else store.root
        / "identities"
        / "identities.jsonl"
    )
    tracklets_path = (
        Path(args.tracklets)
        if args.tracklets
        else store.root
        / "tracklets"
        / "tracklets.jsonl"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "final_observation_fusion"
        / OBSERVATION_FUSION_VERSION
    )

    config = ObservationFusionConfig(
        max_fusion_diameter_m=(
            args.max_fusion_diameter_m
        ),
    )

    print(
        "[final-observation-fusion] starting "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_observation_fusion(
        identities_path=identities_path,
        tracklets_path=tracklets_path,
        output_dir=output_dir,
        config=config,
    )
    print(
        "[final-observation-fusion] completed "
        f"trajectories={summary['trajectory_count']} "
        f"tracklets={summary['tracklet_count']} "
        f"input_obs={summary['input_observation_count']} "
        f"points={summary['output_point_count']} "
        f"collapsed={summary['collapsed_observation_count']}"
    )
    print(
        "[final-observation-fusion] fusion "
        f"events={summary['same_frame_fusion_event_count']} "
        f"weighted="
        f"{summary['weighted_mean_fusion_count']} "
        f"conflicts="
        f"{summary['medoid_selected_conflict_count']} "
        f"max_diameter_m="
        f"{summary['max_fusion_input_diameter_m']:.6f}"
    )
    print(
        "[final-observation-fusion] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
