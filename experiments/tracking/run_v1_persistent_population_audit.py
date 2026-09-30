#!/usr/bin/env python3
"""Run Hito 6D: persistent population audit.

Consumes the provisional Hito 6B V1 actor assignment, Hito 6C persistent
trajectories, and the original tracklet-level physical compatibility graph.
Produces a read-only audit artifact.  No tracking or trajectory source is
modified.
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
from tracking.persistent.audit import (  # noqa: E402
    ASSIGNMENT_SOLVER_VERSION,
    AUDIT_VERSION,
    TRAJECTORY_VERSION,
    PersistentPopulationAuditConfig,
    audit_persistent_population,
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
    root = store.root
    assignment_dir = (
        root
        / "diagnostics"
        / "persistent_actor_solver"
        / ASSIGNMENT_SOLVER_VERSION
    )
    trajectory_dir = (
        root
        / "diagnostics"
        / "persistent_actor_trajectories"
        / TRAJECTORY_VERSION
    )
    paths = {
        "assignment_summary": assignment_dir / "summary.json",
        "assignments": assignment_dir / "assignments.jsonl",
        "trajectory_summary": trajectory_dir / "summary.json",
        "trajectories": trajectory_dir / "trajectories.jsonl",
        "occupancy": trajectory_dir / "occupancy.jsonl",
        "gaps": trajectory_dir / "gaps.jsonl",
        "hard_pairs": (
            root
            / "diagnostics"
            / "identity_candidates"
            / "physical-compatibility-v1"
            / "edges.jsonl"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Hito 6D source artifacts are missing: " + ", ".join(missing)
        )

    assignment_summary = _load_json(paths["assignment_summary"])
    trajectory_summary = _load_json(paths["trajectory_summary"])
    if assignment_summary.get("solver_version") != ASSIGNMENT_SOLVER_VERSION:
        raise RuntimeError("Hito 6D requires anchored-actor-assignment-v1")
    if trajectory_summary.get("trajectory_version") != TRAJECTORY_VERSION:
        raise RuntimeError("Hito 6D requires persistent-trajectory-v1")
    return paths


def run_population_audit(
    *,
    store: TrackingArtifactStore,
    output_dir: Path,
    config: PersistentPopulationAuditConfig,
) -> dict[str, Any]:
    paths = _source_paths(store)
    result = audit_persistent_population(
        trajectory_rows=_iter_jsonl(paths["trajectories"]),
        occupancy_rows=_iter_jsonl(paths["occupancy"]),
        gap_rows=_iter_jsonl(paths["gaps"]),
        trajectory_summary=_load_json(paths["trajectory_summary"]),
        assignment_rows=_iter_jsonl(paths["assignments"]),
        assignment_summary=_load_json(paths["assignment_summary"]),
        hard_pair_rows=_iter_jsonl(paths["hard_pairs"]),
        config=config,
    )

    summary_path = output_dir / "summary.json"
    issues_path = output_dir / "issues.jsonl"
    missing_path = output_dir / "missing_actor_frames.jsonl"
    actor_reviews_path = output_dir / "actor_reviews.jsonl"
    collisions_path = output_dir / "collision_reviews.jsonl"
    gaps_path = output_dir / "preserved_gaps.jsonl"
    index_path = output_dir / "index.json"

    summary = dict(result["summary"])
    summary["sources"] = {key: str(value) for key, value in sorted(paths.items())}
    _write_json(summary_path, summary)
    issue_count = _write_jsonl(issues_path, result["issues"])
    missing_count = _write_jsonl(missing_path, result["missing_actor_frames"])
    actor_review_count = _write_jsonl(actor_reviews_path, result["actor_reviews"])
    collision_count = _write_jsonl(collisions_path, result["collision_reviews"])
    preserved_gap_count = _write_jsonl(gaps_path, result["preserved_gaps"])

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "audit_version": AUDIT_VERSION,
            "trajectory_version": TRAJECTORY_VERSION,
            "assignment_solver_version": ASSIGNMENT_SOLVER_VERSION,
            "canonical": False,
            "provisional": True,
            "tracking_v1_unchanged": True,
            "status": summary["status"],
            "eligible_for_visual_validation": summary["eligible_for_visual_validation"],
            "summary": summary_path.name,
            "issues": issues_path.name,
            "missing_actor_frames": missing_path.name,
            "actor_reviews": actor_reviews_path.name,
            "collision_reviews": collisions_path.name,
            "preserved_gaps": gaps_path.name,
            "issue_count": issue_count,
            "missing_actor_frame_count": missing_count,
            "actor_review_count": actor_review_count,
            "collision_review_count": collision_count,
            "preserved_gap_count": preserved_gap_count,
        },
    )
    summary["artifact"] = str(index_path)
    return {
        "summary": summary,
        "issues": result["issues"],
        "actor_reviews": result["actor_reviews"],
        "missing_actor_frames": result["missing_actor_frames"],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/tracking/dev_clip.json")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-represented-coverage", type=float, default=0.98)
    parser.add_argument("--min-full-population-frames", type=float, default=0.95)
    parser.add_argument("--position-std-review-m", type=float, default=1.00)
    parser.add_argument("--speed-review-mps", type=float, default=12.0)
    parser.add_argument("--acceleration-review-mps2", type=float, default=20.0)
    parser.add_argument("--close-actor-review-m", type=float, default=0.35)
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
        / "persistent_population_audit"
        / AUDIT_VERSION
    )
    config = PersistentPopulationAuditConfig(
        min_represented_coverage_fraction=args.min_represented_coverage,
        min_full_population_frame_fraction=args.min_full_population_frames,
        position_std_review_m=args.position_std_review_m,
        speed_review_mps=args.speed_review_mps,
        acceleration_review_mps2=args.acceleration_review_mps2,
        close_actor_review_m=args.close_actor_review_m,
    )

    print(f"[persistent-audit-6D] starting match={match_id} clip={clip_id}")
    result = run_population_audit(store=store, output_dir=output_dir, config=config)
    summary = result["summary"]

    print(
        "[persistent-audit-6D] population "
        f"expected={summary['expected_actor_count']} actors={summary['actor_count']} "
        f"frames={summary['frame_count']} minimum={summary['minimum_represented_actor_count']}"
    )
    print(
        "[persistent-audit-6D] physics "
        f"cannot_links={summary['hard_cannot_link_count']} "
        f"violations={summary['hard_constraint_violation_count']}"
    )
    print(
        "[persistent-audit-6D] coverage "
        f"direct={summary['direct_coverage_fraction']:.6f} "
        f"represented={summary['represented_coverage_fraction']:.6f} "
        f"full_population={summary['full_population_frame_count']}/"
        f"{summary['frame_count']} ({summary['full_population_frame_fraction']:.6f})"
    )
    print(
        "[persistent-audit-6D] continuity "
        f"missing_actor_frames={summary['missing_actor_frame_count']} "
        f"internal_missing={summary['internal_missing_actor_frame_count']} "
        f"disappearance_events={summary['disappearance_event_count']}"
    )
    print(
        "[persistent-audit-6D] prediction "
        f"points={summary['predicted_point_count']} "
        f"max_age={summary['prediction_age_s']['max']} "
        f"age_violations={summary['prediction_age_violation_count']} "
        f"position_std_p95={summary['position_std_m']['p95']} "
        f"position_std_max={summary['position_std_m']['max']}"
    )
    print(
        "[persistent-audit-6D] kinematics "
        f"speed_p95={summary['speed_mps']['p95']} "
        f"speed_max={summary['speed_mps']['max']} "
        f"high_speed_points={summary['high_speed_point_count']} "
        f"high_acceleration_transitions={summary['high_acceleration_transition_count']}"
    )
    print(
        "[persistent-audit-6D] spatial-review "
        f"close_pairs={summary['close_actor_pair_count']} "
        f"coordinate_reviews={summary['coordinate_sanity_review_count']} "
        f"preserved_gaps={summary['preserved_gap_count']}"
    )

    if result["actor_reviews"]:
        print("[persistent-audit-6D] actors requiring review")
        for row in result["actor_reviews"]:
            print(
                "  "
                f"{row['actor_id']} missing={row['missing_frame_count']} "
                f"internal={row['internal_missing_frame_count']} "
                f"frames={row['missing_frame_indices']} "
                f"coverage={row['represented_frame_fraction']:.6f}"
            )

    print(
        "[persistent-audit-6D] status="
        f"{summary['status']} blockers={summary['blocking_issue_count']} "
        f"reviews={summary['review_issue_count']} "
        f"visual_validation={summary['eligible_for_visual_validation']}"
    )
    print(f"[persistent-audit-6D] artifact={summary['artifact']}")
    return 0 if summary["blocking_issue_count"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
