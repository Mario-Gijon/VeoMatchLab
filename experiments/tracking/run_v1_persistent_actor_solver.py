#!/usr/bin/env python3
"""Run Hito 6B V1: anchored persistent-actor assignment with noise.

Inputs (all immutable Tracking V1 evidence):
- canonical tracklets
- original physical-compatibility-v1 hard cannot-links
- canonical FINAL trajectories (anchor discovery only)
- canonical identity assignments + identity links
- soft-link-v1 scored candidate links
- Hito 6A population estimate

Outputs:
tracking/v1/diagnostics/persistent_actor_solver/anchored-actor-assignment-v1/
    actors.jsonl
    assignments.jsonl
    noise_tracklets.jsonl
    identity_membership.jsonl
    reward_pairs.jsonl
    anchor_candidates.jsonl
    summary.json
    index.json

This runner never modifies tracking/v1/final or any canonical TrackingStage.
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
from tracking.persistent.solver import (  # noqa: E402
    FINAL_PUBLISH_VERSION,
    IDENTITY_PUBLISH_VERSION,
    PERSISTENT_CANDIDATE_VERSION,
    PHYSICAL_GRAPH_VERSION,
    SOFT_LINK_VERSION,
    SOLVER_VERSION,
    PersistentActorAssignmentConfig,
    solve_anchored_actor_assignment,
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
    for stage in (
        TrackingStage.TRACKLETS,
        TrackingStage.IDENTITIES,
        TrackingStage.FINAL,
    ):
        if not store.stage_completed(stage):
            raise RuntimeError(f"Hito 6B requires completed {stage.value} stage")

    root = store.root
    paths = {
        "tracklets": root / "tracklets" / "tracklets.jsonl",
        "physical_edges": (
            root
            / "diagnostics"
            / "identity_candidates"
            / PHYSICAL_GRAPH_VERSION
            / "edges.jsonl"
        ),
        "final_index": root / "final" / "index.json",
        "trajectories": root / "final" / "trajectories.jsonl",
        "identity_index": root / "identities" / "index.json",
        "identity_assignments": root / "identities" / "assignments.jsonl",
        "identity_links": root / "identities" / "identity_links.jsonl",
        "scored_links": (
            root
            / "diagnostics"
            / "identity_link_scoring"
            / SOFT_LINK_VERSION
            / "scored_links.jsonl"
        ),
        "population": (
            root
            / "diagnostics"
            / "persistent_actor_candidates"
            / PERSISTENT_CANDIDATE_VERSION
            / "population.json"
        ),
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Hito 6B source artifacts are missing: " + ", ".join(missing)
        )

    final_index = _load_json(paths["final_index"])
    if (
        final_index.get("publisher_version") != FINAL_PUBLISH_VERSION
        or final_index.get("canonical") is not True
        or final_index.get("final_stage_completed") is not True
    ):
        raise RuntimeError("Hito 6B requires canonical final-publish-v1")

    identity_index = _load_json(paths["identity_index"])
    if (
        identity_index.get("publisher_version") != IDENTITY_PUBLISH_VERSION
        or identity_index.get("canonical") is not True
    ):
        raise RuntimeError("Hito 6B requires canonical identity-publish-v1")

    population = _load_json(paths["population"])
    if population.get("candidate_version") not in {None, PERSISTENT_CANDIDATE_VERSION}:
        raise RuntimeError("Hito 6B population candidate version mismatch")
    return paths


def run_solver(
    *,
    store: TrackingArtifactStore,
    output_dir: Path,
    config: PersistentActorAssignmentConfig,
) -> dict[str, Any]:
    paths = _source_paths(store)
    result = solve_anchored_actor_assignment(
        tracklet_rows=_iter_jsonl(paths["tracklets"]),
        physical_candidate_rows=_iter_jsonl(paths["physical_edges"]),
        trajectory_rows=_iter_jsonl(paths["trajectories"]),
        identity_assignment_rows=_iter_jsonl(paths["identity_assignments"]),
        identity_link_rows=_iter_jsonl(paths["identity_links"]),
        scored_link_rows=_iter_jsonl(paths["scored_links"]),
        population=_load_json(paths["population"]),
        config=config,
    )

    actors_path = output_dir / "actors.jsonl"
    assignments_path = output_dir / "assignments.jsonl"
    noise_path = output_dir / "noise_tracklets.jsonl"
    identities_path = output_dir / "identity_membership.jsonl"
    rewards_path = output_dir / "reward_pairs.jsonl"
    anchors_path = output_dir / "anchor_candidates.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    actor_count = _write_jsonl(actors_path, result["actors"])
    assignment_count = _write_jsonl(assignments_path, result["assignments"])
    noise_count = _write_jsonl(noise_path, result["noise_tracklets"])
    identity_count = _write_jsonl(identities_path, result["identity_membership"])
    reward_count = _write_jsonl(rewards_path, result["reward_pairs"])
    anchor_count = _write_jsonl(anchors_path, result["anchor_candidates"])

    summary = dict(result["summary"])
    summary["sources"] = {key: str(path) for key, path in sorted(paths.items())}
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "solver_version": SOLVER_VERSION,
            "canonical": False,
            "provisional": True,
            "tracking_v1_unchanged": True,
            "actors": actors_path.name,
            "assignments": assignments_path.name,
            "noise_tracklets": noise_path.name,
            "identity_membership": identities_path.name,
            "reward_pairs": rewards_path.name,
            "anchor_candidates": anchors_path.name,
            "summary": summary_path.name,
            "actor_count": actor_count,
            "assignment_count": assignment_count,
            "noise_tracklet_count": noise_count,
            "identity_membership_count": identity_count,
            "reward_pair_count": reward_count,
            "anchor_candidate_count": anchor_count,
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
    parser.add_argument("--max-anchor-configurations", type=int, default=12)
    parser.add_argument("--max-window-s", type=float, default=120.0)
    parser.add_argument("--time-limit-s", type=float, default=60.0)
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
        / "persistent_actor_solver"
        / SOLVER_VERSION
    )
    config = PersistentActorAssignmentConfig(
        max_anchor_configurations=args.max_anchor_configurations,
        max_window_s=args.max_window_s,
        time_limit_s=args.time_limit_s,
    )

    print(f"[persistent-actors-6B-v1] starting match={match_id} clip={clip_id}")
    summary = run_solver(store=store, output_dir=output_dir, config=config)

    print(
        "[persistent-actors-6B-v1] population "
        f"estimate={summary['population_estimate']} "
        f"confidence={summary['population_confidence']} "
        f"window={summary['window_duration_s']:.3f}s"
    )
    print(
        "[persistent-actors-6B-v1] anchors "
        f"tested={summary['anchor_configuration_count']} "
        f"minimum_noise_ties={summary['minimum_noise_anchor_count']} "
        f"chosen_frame={summary['chosen_anchor_frame_index']} "
        f"uncertainty_sum={summary['chosen_anchor_uncertainty_sum_m']:.4f}"
    )
    print(
        "[persistent-actors-6B-v1] assignment "
        f"actors={summary['actor_count']} "
        f"tracklets={summary['tracklet_count']} "
        f"assigned={summary['assigned_tracklet_count']} "
        f"noise={summary['noise_tracklet_count']} "
        f"dropped_observations={summary['dropped_observation_count']} "
        f"dropped_fraction={summary['dropped_observation_fraction']:.6f}"
    )
    print(
        "[persistent-actors-6B-v1] physics "
        f"cannot_links={summary['hard_cannot_link_count']} "
        f"violations={summary['hard_constraint_violation_count']}"
    )
    print(
        "[persistent-actors-6B-v1] evidence "
        f"v1_links={summary['v1_identity_link_count']} "
        f"preserved={summary['preserved_v1_identity_link_count']} "
        f"preserved_fraction={summary['preserved_v1_identity_link_fraction']} "
        f"soft_sum={summary['soft_evidence_sum']:.4f}"
    )
    print(
        "[persistent-actors-6B-v1] identity_status="
        + json.dumps(summary["identity_status_counts"], sort_keys=True)
    )

    print("[persistent-actors-6B-v1] noise tracklets")
    for row in _iter_jsonl(output_dir / "noise_tracklets.jsonl"):
        print(
            "  "
            f"{row['tracklet_id']} "
            f"identity={row['source_identity_id']} "
            f"observations={row['observation_count']} "
            f"time={row['start_timestamp_s']}..{row['end_timestamp_s']} "
            f"confidence={row['mean_confidence']}"
        )

    print("[persistent-actors-6B-v1] actors with >1 source physical identity")
    for row in _iter_jsonl(output_dir / "actors.jsonl"):
        if int(row["source_identity_count"]) <= 1:
            continue
        print(
            "  "
            f"{row['actor_id']} anchor={row['anchor_identity_id']} "
            f"tracklets={row['tracklet_count']} "
            f"source_identities={row['source_identity_ids']}"
        )

    print("[persistent-actors-6B-v1] split/partial V1 identities")
    for row in _iter_jsonl(output_dir / "identity_membership.jsonl"):
        if row["status"] == "preserved":
            continue
        print(
            "  "
            f"{row['identity_id']} status={row['status']} "
            f"actors={row['actor_ids']} "
            f"noise={row['noise_tracklet_ids']}"
        )

    print(f"[persistent-actors-6B-v1] artifact={summary['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
