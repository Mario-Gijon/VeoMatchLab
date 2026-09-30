#!/usr/bin/env python3
"""Run Hito 6A: persistent-actor candidate diagnostics over canonical V1.

This runner is intentionally read-only with respect to TrackingStage.*.  It
consumes the already completed APPEARANCE, IDENTITIES and FINAL stages and
writes only a diagnostics artifact:

tracking/v1/diagnostics/persistent_actor_candidates/
    persistent-actor-candidates-v1.1/
        identity_nodes.jsonl
        pairs.jsonl
        population.json
        appearance_calibration.json
        summary.json
        index.json

No physical identity is merged and Tracking V1 is never rewritten.
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

from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import TrackingStage
from tracking.persistent.candidates import (
    CANDIDATE_VERSION,
    FINAL_PUBLISH_VERSION,
    IDENTITY_PUBLISH_VERSION,
    PHYSICAL_GRAPH_VERSION,
    PersistentActorCandidateConfig,
    build_candidate_diagnostics,
)


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["match_id"]), str(payload["clip_id"])


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


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
        (json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2) + "\n").encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n")
                count += 1
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _validate_sources(store: TrackingArtifactStore) -> dict[str, Path]:
    required = (TrackingStage.APPEARANCE, TrackingStage.IDENTITIES, TrackingStage.FINAL)
    for stage in required:
        if not store.stage_completed(stage):
            raise RuntimeError(f"Hito 6A requires completed {stage.value} stage")

    identity_index_path = store.root / "identities" / "index.json"
    final_index_path = store.root / "final" / "index.json"
    appearance_index_path = store.root / "appearance" / "index.json"
    physical_edges_path = (
        store.root
        / "diagnostics"
        / "identity_candidates"
        / PHYSICAL_GRAPH_VERSION
        / "edges.jsonl"
    )
    soft_link_calibration_path = (
        store.root
        / "diagnostics"
        / "identity_link_scoring"
        / "soft-link-v1"
        / "calibration.json"
    )
    paths = {
        "identity_index": identity_index_path,
        "identities": store.root / "identities" / "identities.jsonl",
        "final_index": final_index_path,
        "trajectories": store.root / "final" / "trajectories.jsonl",
        "appearance_index": appearance_index_path,
        "appearance_pairs": store.root / "appearance" / "pairwise_evidence.jsonl",
        "physical_edges": physical_edges_path,
        "soft_link_calibration": soft_link_calibration_path,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Hito 6A source artifacts are missing: " + ", ".join(missing))

    identity_index = _load_json(identity_index_path)
    if identity_index.get("publisher_version") != IDENTITY_PUBLISH_VERSION or identity_index.get("canonical") is not True:
        raise RuntimeError("Hito 6A requires canonical identity-publish-v1")
    final_index = _load_json(final_index_path)
    if final_index.get("publisher_version") != FINAL_PUBLISH_VERSION or final_index.get("canonical") is not True or final_index.get("final_stage_completed") is not True:
        raise RuntimeError("Hito 6A requires completed canonical final-publish-v1")
    appearance_index = _load_json(appearance_index_path)
    if not isinstance(appearance_index.get("pairwise_evidence"), str):
        raise RuntimeError("appearance index lacks pairwise_evidence")
    return paths


def run_persistent_actor_candidates(
    *,
    store: TrackingArtifactStore,
    output_dir: Path,
    config: PersistentActorCandidateConfig,
) -> dict[str, Any]:
    paths = _validate_sources(store)
    result = build_candidate_diagnostics(
        identity_rows=_iter_jsonl(paths["identities"]),
        trajectory_rows=_iter_jsonl(paths["trajectories"]),
        appearance_rows=_iter_jsonl(paths["appearance_pairs"]),
        physical_candidate_rows=_iter_jsonl(paths["physical_edges"]),
        trusted_appearance_calibration=_load_json(paths["soft_link_calibration"]),
        config=config,
    )

    nodes_path = output_dir / "identity_nodes.jsonl"
    pairs_path = output_dir / "pairs.jsonl"
    population_path = output_dir / "population.json"
    calibration_path = output_dir / "appearance_calibration.json"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    node_count = _write_jsonl(nodes_path, result["nodes"])
    pair_count = _write_jsonl(pairs_path, result["pairs"])
    _write_json(population_path, result["population"])
    _write_json(calibration_path, result["appearance_calibration"])

    summary = dict(result["summary"])
    summary["sources"] = {key: str(path) for key, path in sorted(paths.items())}
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "candidate_version": CANDIDATE_VERSION,
            "identity_decisions": False,
            "canonical_tracking_v1_unchanged": True,
            "identity_nodes": nodes_path.name,
            "pairs": pairs_path.name,
            "population": population_path.name,
            "appearance_calibration": calibration_path.name,
            "summary": summary_path.name,
            "identity_count": node_count,
            "pair_count": pair_count,
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
    parser.add_argument("--max-transition-speed-mps", type=float, default=12.0)
    parser.add_argument("--transition-position-slack-m", type=float, default=1.5)
    parser.add_argument("--stable-population-support-fraction", type=float, default=0.25)
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
        / "persistent_actor_candidates"
        / CANDIDATE_VERSION
    )
    config = PersistentActorCandidateConfig(
        max_transition_speed_mps=args.max_transition_speed_mps,
        transition_position_slack_m=args.transition_position_slack_m,
        stable_population_support_fraction=args.stable_population_support_fraction,
    )

    print(f"[persistent-actors-6A] starting match={match_id} clip={clip_id}")
    summary = run_persistent_actor_candidates(
        store=store,
        output_dir=output_dir,
        config=config,
    )
    population = summary["population"]
    calibration = summary["appearance_calibration"]
    print(
        "[persistent-actors-6A] population "
        f"estimate={population['estimated_population']} "
        f"confidence={population['confidence']} "
        f"raw_mode={population['raw_modal_population']} "
        f"raw_max={population['raw_max_population']} "
        f"distinct_mode={population['distinct_modal_population']} "
        f"distinct_stable={population['distinct_stable_high_population']} "
        f"distinct_max={population['distinct_max_population']} "
        f"raw_support={population['raw_at_least_support_fraction']:.3f}"
    )
    print(
        "[persistent-actors-6A] graph "
        f"identities={summary['identity_count']} "
        f"pairs={summary['identity_pair_count']} "
        f"candidates={summary['candidate_pair_count']} "
        f"solver_eligible={summary['solver_eligible_sequential_count']} "
        f"cannot_link={summary['cannot_link_pair_count']}"
    )
    print(
        "[persistent-actors-6A] appearance "
        f"available={calibration['available']} "
        f"source={calibration['source']} "
        f"diagnostic_positive_pairs={calibration['positive_pair_count']} "
        f"diagnostic_negative_pairs={calibration['negative_pair_count']} "
        f"neg_p95={calibration['negative_p95']} "
        f"pos_p10={calibration['positive_p10']}"
    )
    print("[persistent-actors-6A] top sequential candidates")
    for row in summary["top_solver_candidates"][:15]:
        print(
            "  "
            f"{row['predecessor_identity_id']} -> {row['successor_identity_id']} "
            f"score={row['link_score']:.4f} "
            f"gap={row['gap_s']:.3f}s "
            f"dist={row['distance_m']:.2f}m "
            f"raw_speed={row['raw_transition_speed_mps']:.2f}m/s "
            f"hard_speed={row['hard_required_speed_mps']:.2f}m/s "
            f"reid={row['appearance_cosine']} "
            f"rank={row['predecessor_rank']}"
        )
    print(f"[persistent-actors-6A] artifact={summary['artifact']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
