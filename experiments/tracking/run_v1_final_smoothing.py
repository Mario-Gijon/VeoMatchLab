#!/usr/bin/env python3
"""Run Tracking V1 Hito 4D: offline temporal smoothing + kinematics.

Consumes:
tracking/v1/diagnostics/final_gap_reconstruction/safe-gap-v1/
    trajectories.jsonl

Produces:
tracking/v1/diagnostics/final_smoothing/rts-cv-v1/
    trajectories.jsonl
    segment_breaks.jsonl
    summary.json
    index.json

This is still provisional and does NOT complete TrackingStage.FINAL.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import (
    StageStatus,
    TrackingStage,
)
from tracking.final import (
    SMOOTHING_VERSION,
    TemporalSmoothingConfig,
    smooth_trajectories,
)
from tracking.final.smoothing import (
    iter_jsonl,
    load_trajectories,
)


RECONSTRUCTION_VERSION = "safe-gap-v1"


def _atomic_write(
    path: Path,
    content: bytes,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
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


def _write_json(
    path: Path,
    payload: Any,
) -> None:
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
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temp = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    count = 0
    try:
        with temp.open(
            "w",
            encoding="utf-8",
        ) as destination:
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
            os.fsync(
                destination.fileno()
            )
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return count


def run_temporal_smoothing(
    *,
    trajectories_path: Path,
    output_dir: Path,
    config: TemporalSmoothingConfig,
) -> dict[str, Any]:
    trajectories = load_trajectories(
        iter_jsonl(trajectories_path)
    )
    smoothed, breaks, summary = (
        smooth_trajectories(
            trajectories=trajectories,
            config=config,
        )
    )
    summary[
        "source_trajectories"
    ] = str(trajectories_path)

    trajectories_out = (
        output_dir / "trajectories.jsonl"
    )
    breaks_out = (
        output_dir / "segment_breaks.jsonl"
    )
    summary_out = (
        output_dir / "summary.json"
    )
    index_out = (
        output_dir / "index.json"
    )

    trajectory_count = _write_jsonl(
        trajectories_out,
        (
            trajectory.to_dict()
            for trajectory in smoothed
        ),
    )
    break_count = _write_jsonl(
        breaks_out,
        breaks,
    )
    _write_json(
        summary_out,
        summary,
    )
    _write_json(
        index_out,
        {
            "schema_version": 1,
            "smoothing_version": (
                SMOOTHING_VERSION
            ),
            "provisional": True,
            "final_stage_completed": False,
            "trajectories": (
                trajectories_out.name
            ),
            "segment_breaks": (
                breaks_out.name
            ),
            "summary": summary_out.name,
            "trajectory_count": (
                trajectory_count
            ),
            "segment_break_count": (
                break_count
            ),
        },
    )
    return summary


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
    parser.add_argument("--trajectories")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--process-acceleration-sigma-mps2",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--observed-position-sigma-m",
        type=float,
        default=0.30,
    )
    parser.add_argument(
        "--fused-position-sigma-m",
        type=float,
        default=0.35,
    )
    parser.add_argument(
        "--interpolated-position-sigma-m",
        type=float,
        default=0.90,
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
    manifest = store.load_manifest()
    identity_stage = manifest.stage(
        TrackingStage.IDENTITIES
    )
    if (
        identity_stage.status
        is not StageStatus.COMPLETED
    ):
        raise RuntimeError(
            "TrackingStage.IDENTITIES "
            "must be completed before 4D"
        )

    trajectories_path = (
        Path(args.trajectories)
        if args.trajectories
        else store.root
        / "diagnostics"
        / "final_gap_reconstruction"
        / RECONSTRUCTION_VERSION
        / "trajectories.jsonl"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "final_smoothing"
        / SMOOTHING_VERSION
    )

    config = TemporalSmoothingConfig(
        process_acceleration_sigma_mps2=(
            args.process_acceleration_sigma_mps2
        ),
        observed_position_sigma_m=(
            args.observed_position_sigma_m
        ),
        fused_position_sigma_m=(
            args.fused_position_sigma_m
        ),
        interpolated_position_sigma_m=(
            args.interpolated_position_sigma_m
        ),
    )

    print(
        "[final-smoothing] starting "
        f"match={match_id} "
        f"clip={clip_id}"
    )
    summary = run_temporal_smoothing(
        trajectories_path=(
            trajectories_path
        ),
        output_dir=output_dir,
        config=config,
    )
    print(
        "[final-smoothing] completed "
        f"trajectories="
        f"{summary['trajectory_count']} "
        f"points="
        f"{summary['output_point_count']} "
        f"segments="
        f"{summary['segment_count']} "
        f"gap_breaks="
        f"{summary['preserved_gap_break_count']}"
    )
    print(
        "[final-smoothing] adjustment "
        f"p95="
        f"{summary['position_adjustment_m']['p95']:.6f} "
        f"max="
        f"{summary['position_adjustment_m']['max']:.6f} "
        f"review_count="
        f"{summary['adjustment_review_count']}"
    )
    print(
        "[final-smoothing] acceleration "
        f"raw_p95="
        f"{summary['raw_finite_difference_acceleration_mps2']['p95']:.6f} "
        f"smoothed_p95="
        f"{summary['smoothed_finite_difference_acceleration_mps2']['p95']:.6f}"
    )
    print(
        "[final-smoothing] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
