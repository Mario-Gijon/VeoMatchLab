#!/usr/bin/env python3
"""Run Hito 6C: persistent actor trajectory reconstruction.

Consumes:
- canonical Tracking V1 tracklets;
- Hito 6B V1 ``anchored-actor-assignment-v1`` actors/assignments/summary.

Produces diagnostic/provisional artifacts under:
tracking/v1/diagnostics/persistent_actor_trajectories/persistent-trajectory-v1/
    trajectories.jsonl
    occupancy.jsonl
    gaps.jsonl
    summary.json
    index.json

Tracking V1 FINAL is never overwritten.  Hito 6E visual validation is required
before promotion of persistent trajectories to any canonical downstream stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.artifacts import TrackingArtifactStore  # noqa: E402
from tracking.contracts import TrackingStage  # noqa: E402
from tracking.persistent.trajectories import (  # noqa: E402
    ASSIGNMENT_SOLVER_VERSION,
    TRAJECTORY_VERSION,
    PersistentTrajectoryConfig,
    build_persistent_actor_trajectories,
)


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return str(payload["match_id"]), str(payload["clip_id"])


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _iter_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(payload)
    return tuple(rows)


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        (
            json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2)
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as destination:
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
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _source_paths(store: TrackingArtifactStore) -> dict[str, Path]:
    if not store.stage_completed(TrackingStage.TRACKLETS):
        raise RuntimeError("Hito 6C requires completed tracklets stage")

    root = store.root
    assignment_dir = (
        root
        / "diagnostics"
        / "persistent_actor_solver"
        / ASSIGNMENT_SOLVER_VERSION
    )
    paths = {
        "tracklets": root / "tracklets" / "tracklets.jsonl",
        "assignment_index": assignment_dir / "index.json",
        "assignment_summary": assignment_dir / "summary.json",
        "actors": assignment_dir / "actors.jsonl",
        "assignments": assignment_dir / "assignments.jsonl",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Hito 6C source artifacts are missing: " + ", ".join(missing)
        )

    index = _load_json(paths["assignment_index"])
    if index.get("solver_version") != ASSIGNMENT_SOLVER_VERSION:
        raise RuntimeError("Hito 6C requires anchored-actor-assignment-v1")
    if index.get("canonical") is True:
        raise RuntimeError("unexpected canonical flag on provisional 6B artifact")

    summary = _load_json(paths["assignment_summary"])
    if summary.get("solver_version") != ASSIGNMENT_SOLVER_VERSION:
        raise RuntimeError("Hito 6C assignment summary version mismatch")
    if int(summary.get("hard_constraint_violation_count", -1)) != 0:
        raise RuntimeError("Hito 6C refuses a 6B solution with hard violations")
    return paths


def run_trajectory_builder(
    *,
    store: TrackingArtifactStore,
    output_dir: Path,
    config: PersistentTrajectoryConfig,
) -> dict[str, Any]:
    paths = _source_paths(store)
    result = build_persistent_actor_trajectories(
        tracklet_rows=_iter_jsonl(paths["tracklets"]),
        actor_rows=_iter_jsonl(paths["actors"]),
        assignment_rows=_iter_jsonl(paths["assignments"]),
        assignment_summary=_load_json(paths["assignment_summary"]),
        config=config,
    )

    trajectories_path = output_dir / "trajectories.jsonl"
    occupancy_path = output_dir / "occupancy.jsonl"
    gaps_path = output_dir / "gaps.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    trajectory_count = _write_jsonl(trajectories_path, result["trajectories"])
    occupancy_count = _write_jsonl(occupancy_path, result["occupancy"])
    gap_count = _write_jsonl(gaps_path, result["gaps"])

    summary = dict(result["summary"])
    summary["sources"] = {key: str(value) for key, value in sorted(paths.items())}
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "trajectory_version": TRAJECTORY_VERSION,
            "assignment_solver_version": ASSIGNMENT_SOLVER_VERSION,
            "canonical": False,
            "provisional": True,
            "tracking_v1_unchanged": True,
            "visual_validation_required": True,
            "trajectories": trajectories_path.name,
            "occupancy": occupancy_path.name,
            "gaps": gaps_path.name,
            "summary": summary_path.name,
            "actor_count": trajectory_count,
            "frame_count": occupancy_count,
            "gap_count": gap_count,
        },
    )
    summary["artifact"] = str(index_path)
    return summary


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/tracking/dev_clip.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-internal-gap-s", type=float, default=1.50)
    parser.add_argument("--max-edge-prediction-s", type=float, default=0.50)
    parser.add_argument("--max-internal-endpoint-speed-mps", type=float, default=12.0)
    parser.add_argument("--max-position-std-m", type=float, default=2.50)
    parser.add_argument("--process-acceleration-sigma-mps2", type=float, default=10.0)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    match_id = args.match_id
    clip_id = args.clip_id
    if not match_id or not clip_id:
        cfg_match, cfg_clip = _load_dev_clip(Path(args.config))
        match_id = match_id or cfg_match
        clip_id = clip_id or cfg_clip

    store = TrackingArtifactStore.for_clip(args.data_root, match_id, clip_id)
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "persistent_actor_trajectories"
        / TRAJECTORY_VERSION
    )
    config = PersistentTrajectoryConfig(
        max_internal_gap_s=args.max_internal_gap_s,
        max_edge_prediction_s=args.max_edge_prediction_s,
        max_internal_endpoint_speed_mps=args.max_internal_endpoint_speed_mps,
        max_position_std_m=args.max_position_std_m,
        process_acceleration_sigma_mps2=args.process_acceleration_sigma_mps2,
    )

    print(f"[persistent-actors-6C] starting match={match_id} clip={clip_id}")
    summary = run_trajectory_builder(store=store, output_dir=output_dir, config=config)

    print(
        "[persistent-actors-6C] grid "
        f"actors={summary['actor_count']} frames={summary['frame_count']} "
        f"frame_step={summary['cadence']['frame_step']} "
        f"dt={summary['cadence']['time_step_s']:.6f}s"
    )
    print(
        "[persistent-actors-6C] coverage "
        f"direct={summary['direct_actor_frame_count']}/"
        f"{summary['total_possible_actor_frames']} "
        f"({summary['direct_coverage_fraction']:.6f}) "
        f"represented={summary['represented_actor_frame_count']}/"
        f"{summary['total_possible_actor_frames']} "
        f"({summary['represented_coverage_fraction']:.6f})"
    )
    print(
        "[persistent-actors-6C] origins "
        f"observed={summary['observed_point_count']} "
        f"fused={summary['fused_point_count']} "
        f"interpolated={summary['interpolated_point_count']} "
        f"predicted={summary['predicted_point_count']}"
    )
    print(
        "[persistent-actors-6C] full-population frames "
        f"before={summary['full_population_frame_count_before']}/"
        f"{summary['frame_count']} "
        f"({summary['full_population_frame_fraction_before']:.6f}) "
        f"after={summary['full_population_frame_count_after']}/"
        f"{summary['frame_count']} "
        f"({summary['full_population_frame_fraction_after']:.6f})"
    )
    print(
        "[persistent-actors-6C] disappearance events "
        f"before={summary['disappearance_event_count_before']} "
        f"after={summary['disappearance_event_count_after']}"
    )
    print(
        "[persistent-actors-6C] gaps "
        f"total={summary['gap_count']} "
        f"preserved={summary['preserved_gap_count']} "
        f"actions={json.dumps(summary['gap_action_counts'], sort_keys=True)}"
    )
    print(
        "[persistent-actors-6C] uncertainty "
        f"position_std_p95={summary['position_std_m']['p95']} "
        f"position_std_max={summary['position_std_m']['max']} "
        f"prediction_age_max={summary['prediction_age_s']['max']}"
    )

    print("[persistent-actors-6C] actors requiring review")
    trajectories = _iter_jsonl(output_dir / "trajectories.jsonl")
    for row in trajectories:
        if int(row["segment_count"]) <= 1 and float(row["represented_frame_fraction"]) >= 0.999999:
            continue
        print(
            "  "
            f"{row['actor_id']} "
            f"coverage={float(row['represented_frame_fraction']):.4f} "
            f"segments={row['segment_count']} "
            f"direct={row['direct_point_count']} "
            f"interpolated={row['interpolated_point_count']} "
            f"predicted={row['predicted_point_count']} "
            f"max_std={float(row['max_position_std_m']):.3f}m"
        )

    print(f"[persistent-actors-6C] artifact={summary['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
