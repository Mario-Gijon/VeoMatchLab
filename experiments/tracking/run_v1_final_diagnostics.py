#!/usr/bin/env python3
"""Run Tracking V1 Hito 4E: final trajectory diagnostics.

Consumes `rts-cv-v1` trajectories + segment breaks + smoothing summary.
Produces audit artifacts only; TrackingStage.FINAL remains NOT_RUN.
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
from tracking.final.diagnostics import (
    FINAL_DIAGNOSTICS_VERSION,
    FinalDiagnosticsConfig,
    audit_final_trajectories,
    iter_jsonl,
    load_json,
)


SMOOTHING_VERSION = "rts-cv-v1"


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


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
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


def run_final_diagnostics(
    *,
    trajectories_path: Path,
    segment_breaks_path: Path,
    smoothing_summary_path: Path,
    output_dir: Path,
    config: FinalDiagnosticsConfig,
) -> dict[str, Any]:
    trajectory_rows = tuple(iter_jsonl(trajectories_path))
    segment_break_rows = tuple(iter_jsonl(segment_breaks_path))
    smoothing_summary = load_json(smoothing_summary_path)

    (
        identity_audits,
        segment_audits,
        point_reviews,
        blockers,
        review_queue,
        summary,
    ) = audit_final_trajectories(
        trajectory_rows=trajectory_rows,
        segment_break_rows=segment_break_rows,
        source_summary=smoothing_summary,
        config=config,
    )

    summary["source_trajectories"] = str(trajectories_path)
    summary["source_segment_breaks"] = str(segment_breaks_path)
    summary["source_smoothing_summary"] = str(smoothing_summary_path)

    identity_path = output_dir / "identity_audits.jsonl"
    segment_path = output_dir / "segment_audits.jsonl"
    point_path = output_dir / "point_reviews.jsonl"
    blocker_path = output_dir / "blockers.jsonl"
    queue_path = output_dir / "review_queue.json"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    identity_count = _write_jsonl(identity_path, identity_audits)
    segment_count = _write_jsonl(segment_path, segment_audits)
    review_count = _write_jsonl(point_path, point_reviews)
    blocker_count = _write_jsonl(blocker_path, blockers)
    _write_json(queue_path, review_queue)
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "diagnostics_version": FINAL_DIAGNOSTICS_VERSION,
            "provisional": True,
            "final_stage_completed": False,
            "publishable": summary["publishable"],
            "identity_audits": identity_path.name,
            "segment_audits": segment_path.name,
            "point_reviews": point_path.name,
            "blockers": blocker_path.name,
            "review_queue": queue_path.name,
            "summary": summary_path.name,
            "identity_audit_count": identity_count,
            "segment_audit_count": segment_count,
            "point_review_count": review_count,
            "blocker_count": blocker_count,
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
    parser.add_argument("--trajectories")
    parser.add_argument("--segment-breaks")
    parser.add_argument("--smoothing-summary")
    parser.add_argument("--output-dir")
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
    manifest = store.load_manifest()
    identity_stage = manifest.stage(TrackingStage.IDENTITIES)
    if identity_stage.status is not StageStatus.COMPLETED:
        raise RuntimeError(
            "TrackingStage.IDENTITIES must be completed before 4E"
        )

    smoothing_dir = (
        store.root
        / "diagnostics"
        / "final_smoothing"
        / SMOOTHING_VERSION
    )
    trajectories_path = (
        Path(args.trajectories)
        if args.trajectories
        else smoothing_dir / "trajectories.jsonl"
    )
    segment_breaks_path = (
        Path(args.segment_breaks)
        if args.segment_breaks
        else smoothing_dir / "segment_breaks.jsonl"
    )
    smoothing_summary_path = (
        Path(args.smoothing_summary)
        if args.smoothing_summary
        else smoothing_dir / "summary.json"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "final_trajectory_diagnostics"
        / FINAL_DIAGNOSTICS_VERSION
    )

    print(
        "[final-diagnostics] starting "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_final_diagnostics(
        trajectories_path=trajectories_path,
        segment_breaks_path=segment_breaks_path,
        smoothing_summary_path=smoothing_summary_path,
        output_dir=output_dir,
        config=FinalDiagnosticsConfig(),
    )
    print(
        "[final-diagnostics] completed "
        f"trajectories={summary['trajectory_count']} "
        f"points={summary['point_count']} "
        f"segments={summary['segment_count']} "
        f"gap_breaks={summary['segment_break_count']}"
    )
    print(
        "[final-diagnostics] gate "
        f"publishable={summary['publishable']} "
        f"blockers={summary['blocker_count']} "
        f"review_points={summary['review_point_count']} "
        f"review_segments={summary['review_segment_count']}"
    )
    print(
        "[final-diagnostics] flags "
        f"{json.dumps(summary['review_flag_counts'], sort_keys=True)}"
    )
    print(
        "[final-diagnostics] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
