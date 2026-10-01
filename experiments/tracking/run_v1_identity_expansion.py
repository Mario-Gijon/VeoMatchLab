#!/usr/bin/env python3
"""Run Tracking V1 Hito 7E.5 family-aware identity expansion.

This reruns the seeded identity expansion from the conservative 7E.3B source
identities.  Family evidence is rebuilt for those SOURCE identities directly
from 7C.1E tracklet consensus, avoiding any circular dependency on a previous
7E.4 expansion result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.identity_expansion import (  # noqa: E402
    IDENTITY_EXPANSION_VERSION,
    IdentityExpansionConfig,
    run_identity_expansion,
)
from tracking.appearance.kit_entity_consensus import (  # noqa: E402
    KitEntityConsensusConfig,
    aggregate_entity_family_consensus,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain an object"
                )
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def _link_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row["predecessor_identity_id"]),
        str(row["successor_identity_id"]),
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--top-k-successors", type=int, default=16)
    parser.add_argument("--top-k-predecessors", type=int, default=16)
    parser.add_argument("--min-rank-score", type=float, default=0.60)
    parser.add_argument(
        "--competition-mode",
        choices=("mutual_best", "mutual_top3"),
        default="mutual_best",
    )
    parser.add_argument(
        "--min-seed-profiled-tracklets",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--same-primary-bonus",
        type=float,
        default=0.035,
    )
    parser.add_argument(
        "--different-primary-penalty",
        type=float,
        default=0.150,
    )
    parser.add_argument(
        "--baseline-dir",
        help=(
            "Optional baseline expansion directory for selected-link "
            "comparison. Defaults to seeded-identity-expansion-v1-score060."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    persistent_root = (
        root / "diagnostics" / "persistent_chain_solver"
        / "persistent-chain-path-v1"
    )
    identities_path = persistent_root / "identities.jsonl"
    actor_assignments_path = (
        persistent_root / "actor_assignments.jsonl"
    )
    tracklet_assignments_path = (
        persistent_root / "tracklet_assignments.jsonl"
    )

    chain_profile_path = (
        root / "diagnostics" / "actor_chain_reid"
        / "actor-chain-reid-v1" / "profiles.jsonl"
    )
    hard_conflicts_path = (
        root / "diagnostics" / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )
    trusted_positive_path = (
        root / "diagnostics" / "reid_calibration"
        / "reid-overlap-calibration-v1"
        / "split_positive.jsonl"
    )
    tracklet_family_path = (
        root / "diagnostics" / "kit_family_consensus"
        / "kit-family-consensus-v1" / "assignments.jsonl"
    )

    for path in (
        identities_path,
        actor_assignments_path,
        tracklet_assignments_path,
        chain_profile_path,
        hard_conflicts_path,
        trusted_positive_path,
        tracklet_family_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    identities = _iter_jsonl(identities_path)
    actor_assignments = _iter_jsonl(actor_assignments_path)
    tracklet_assignments = _iter_jsonl(tracklet_assignments_path)
    chain_profiles = _iter_jsonl(chain_profile_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)
    trusted_positive_rows = _iter_jsonl(trusted_positive_path)
    trusted_positive_cosines = tuple(
        float(row["cosine"])
        for row in trusted_positive_rows
    )
    tracklet_family_rows = _iter_jsonl(tracklet_family_path)

    # IMPORTANT: build family evidence on the 1079 SOURCE persistent
    # identities, not on a previous expansion result. This avoids circular
    # self-reinforcement when we compare family-aware association to baseline.
    source_family_result = aggregate_entity_family_consensus(
        tracklet_consensus_rows=tracklet_family_rows,
        tracklet_entity_rows=tracklet_assignments,
        entity_id_field="identity_id",
        entity_level="source_identity",
        config=KitEntityConsensusConfig(),
    )
    source_family_rows = tuple(
        row.to_dict() for row in source_family_result.entities
    )

    config = IdentityExpansionConfig(
        top_k_successors=args.top_k_successors,
        top_k_predecessors=args.top_k_predecessors,
        min_rank_score=args.min_rank_score,
        competition_mode=args.competition_mode,
        min_seed_profiled_tracklets=(
            args.min_seed_profiled_tracklets
        ),
        # Legacy different-kit rejection is intentionally irrelevant when
        # source identity family evidence is supplied. Different primary is a
        # strong SOFT penalty, never a hard rejection.
        allow_different_kit=True,
        same_primary_bonus=args.same_primary_bonus,
        different_primary_penalty=args.different_primary_penalty,
    )

    result = run_identity_expansion(
        identity_rows=identities,
        persistent_actor_assignments=actor_assignments,
        persistent_tracklet_assignments=tracklet_assignments,
        chain_profile_rows=chain_profiles,
        hard_conflicts=hard_conflicts,
        trusted_positive_cosines=trusted_positive_cosines,
        identity_family_rows=source_family_rows,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "identity_expansion"
        / IDENTITY_EXPANSION_VERSION
    )

    profiles_path = output_dir / "profiles.jsonl"
    source_family_out = (
        output_dir / "source_identity_family_assignments.jsonl"
    )
    candidates_path = output_dir / "candidates.jsonl"
    selected_path = output_dir / "selected_links.jsonl"
    expanded_path = output_dir / "identities.jsonl"
    identity_assignments_out = (
        output_dir / "identity_assignments.jsonl"
    )
    actor_assignments_out = (
        output_dir / "actor_assignments.jsonl"
    )
    tracklet_assignments_out = (
        output_dir / "tracklet_assignments.jsonl"
    )
    comparison_path = output_dir / "comparison.json"
    comparison_links_path = output_dir / "comparison_links.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    profile_count = _write_jsonl(
        profiles_path,
        (row.to_dict() for row in result.profiles),
    )
    source_family_count = _write_jsonl(
        source_family_out,
        source_family_rows,
    )
    candidate_count = _write_jsonl(
        candidates_path,
        (row.to_dict() for row in result.candidates),
    )
    selected_count = _write_jsonl(
        selected_path,
        (row.to_dict() for row in result.selected_candidates),
    )
    identity_count = _write_jsonl(
        expanded_path,
        (row.to_dict() for row in result.expanded_identities),
    )
    identity_assignment_count = _write_jsonl(
        identity_assignments_out,
        result.identity_assignments,
    )
    actor_assignment_count = _write_jsonl(
        actor_assignments_out,
        result.actor_assignments,
    )
    tracklet_assignment_count = _write_jsonl(
        tracklet_assignments_out,
        result.tracklet_assignments,
    )

    baseline_dir = (
        Path(args.baseline_dir)
        if args.baseline_dir
        else root / "diagnostics" / "identity_expansion"
        / "seeded-identity-expansion-v1-score060"
    )
    baseline_selected_path = baseline_dir / "selected_links.jsonl"

    comparison: dict[str, Any] = {
        "baseline_available": baseline_selected_path.is_file(),
        "baseline_dir": str(baseline_dir),
        "baseline_selected_link_count": None,
        "family_aware_selected_link_count": selected_count,
        "common": None,
        "lost": None,
        "new": None,
    }
    comparison_rows: list[dict[str, Any]] = []

    if baseline_selected_path.is_file():
        baseline_rows = _iter_jsonl(baseline_selected_path)
        baseline_by_key = {_link_key(row): row for row in baseline_rows}
        aware_by_key = {
            _link_key(row.to_dict()): row.to_dict()
            for row in result.selected_candidates
        }

        baseline_keys = set(baseline_by_key)
        aware_keys = set(aware_by_key)
        common = baseline_keys & aware_keys
        lost = baseline_keys - aware_keys
        new = aware_keys - baseline_keys

        comparison.update({
            "baseline_selected_link_count": len(baseline_keys),
            "common": len(common),
            "lost": len(lost),
            "new": len(new),
        })

        for status, keys in (
            ("common", common),
            ("lost", lost),
            ("new", new),
        ):
            for key in sorted(keys):
                baseline = baseline_by_key.get(key)
                aware = aware_by_key.get(key)
                comparison_rows.append({
                    "status": status,
                    "predecessor_identity_id": key[0],
                    "successor_identity_id": key[1],
                    "baseline_rank_score": (
                        None
                        if baseline is None
                        else float(baseline["rank_score"])
                    ),
                    "family_aware_rank_score": (
                        None
                        if aware is None
                        else float(aware["rank_score"])
                    ),
                    "base_rank_score": (
                        None
                        if aware is None
                        else float(aware["base_rank_score"])
                    ),
                    "family_relation": (
                        None
                        if aware is None
                        else aware["family_relation"]
                    ),
                    "family_pair_strength": (
                        None
                        if aware is None
                        else float(aware["family_pair_strength"])
                    ),
                    "family_adjustment": (
                        None
                        if aware is None
                        else float(aware["family_adjustment"])
                    ),
                })

    _write_json(comparison_path, comparison)
    _write_jsonl(comparison_links_path, comparison_rows)

    summary = result.summary()
    source_family_summary = source_family_result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_identity_count": len(identities),
        "source_actor_assignment_count": len(actor_assignments),
        "source_tracklet_assignment_count": len(tracklet_assignments),
        "source_chain_profile_count": len(chain_profiles),
        "source_hard_conflict_count": len(hard_conflicts),
        "trusted_positive_count": len(trusted_positive_cosines),
        "unprofiled_identity_count": len(identities) - len(result.profiles),
        "source_family_consensus": source_family_summary,
        "source_family_assignment_count": source_family_count,
        "source_identities": str(identities_path),
        "source_actor_assignments": str(actor_assignments_path),
        "source_tracklet_assignments": str(tracklet_assignments_path),
        "source_chain_profiles": str(chain_profile_path),
        "source_hard_conflicts": str(hard_conflicts_path),
        "source_trusted_positive": str(trusted_positive_path),
        "source_tracklet_family_consensus": str(tracklet_family_path),
        "baseline_comparison": comparison,
        "family_evidence_circular_dependency": False,
    })
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "expansion_version": IDENTITY_EXPANSION_VERSION,
            "profiles": profiles_path.name,
            "source_identity_family_assignments": source_family_out.name,
            "candidates": candidates_path.name,
            "selected_links": selected_path.name,
            "identities": expanded_path.name,
            "identity_assignments": identity_assignments_out.name,
            "actor_assignments": actor_assignments_out.name,
            "tracklet_assignments": tracklet_assignments_out.name,
            "comparison": comparison_path.name,
            "comparison_links": comparison_links_path.name,
            "summary": summary_path.name,
            "profile_count": profile_count,
            "candidate_count": candidate_count,
            "selected_link_count": selected_count,
            "identity_count": identity_count,
            "identity_assignment_count": identity_assignment_count,
            "actor_assignment_count": actor_assignment_count,
            "tracklet_assignment_count": tracklet_assignment_count,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[identity-expansion-7E.5] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[identity-expansion-7E.5] "
        f"source_identities={len(identities)} "
        f"profiles={profile_count} "
        f"unprofiled={summary['unprofiled_identity_count']} "
        f"seeds={summary['initial_seed_count']}"
    )
    print(
        "[identity-expansion-7E.5] source_family="
        + json.dumps(
            {
                "usable": source_family_summary["usable_for_tracking_count"],
                "states": source_family_summary["state_counts"],
                "families": source_family_summary["family_counts"],
            },
            sort_keys=True,
        )
    )
    print(
        "[identity-expansion-7E.5] "
        f"candidates={candidate_count} "
        f"mutual_best={summary['mutual_best_count']} "
        f"mutual_top3={summary['mutual_top3_count']}"
    )
    print(
        "[identity-expansion-7E.5] family_relations="
        + json.dumps(summary["family_relation_counts"], sort_keys=True)
    )
    print(
        "[identity-expansion-7E.5] bands="
        + json.dumps(summary["score_band_counts"], sort_keys=True)
    )
    print(
        "[identity-expansion-7E.5] "
        f"selected={selected_count} identities={identity_count} "
        f"non_singleton={summary['non_singleton_expanded_identity_count']} "
        f"largest_source_ids={summary['largest_source_identity_count']}"
    )
    print(
        "[identity-expansion-7E.5] selected_family_relations="
        + json.dumps(
            summary["selected_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[identity-expansion-7E.5] selected_scores="
        + json.dumps(summary["selected_score"], sort_keys=True)
    )
    print(
        "[identity-expansion-7E.5] rejections="
        + json.dumps(summary["rejection_counts"], sort_keys=True)
    )
    print(
        "[identity-expansion-7E.5] comparison="
        + json.dumps(comparison, sort_keys=True)
    )
    print(
        "[identity-expansion-7E.5] "
        f"artifact={index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
