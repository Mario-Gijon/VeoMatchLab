#!/usr/bin/env python3
"""Run Tracking V1 Hito 7F iterative global identity consolidation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_identity_consolidation import (  # noqa: E402
    GLOBAL_IDENTITY_CONSOLIDATION_VERSION,
    GlobalIdentityConsolidationConfig,
    consolidate_until_convergence,
)
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
    rows = []
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
        ) + "\n",
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
                ) + "\n"
            )
            count += 1
    return count


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-rounds", type=int, default=8)
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
        default=0.300,
    )
    parser.add_argument(
        "--probable-different-primary-penalty",
        type=float,
        default=0.200,
    )
    parser.add_argument(
        "--probable-primary-min-dominance",
        type=float,
        default=0.60,
    )
    parser.add_argument(
        "--probable-primary-min-confidence",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--probable-primary-min-evidence-coverage",
        type=float,
        default=0.15,
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

    initial_identities = _iter_jsonl(identities_path)
    initial_actor_assignments = _iter_jsonl(actor_assignments_path)
    initial_tracklet_assignments = _iter_jsonl(
        tracklet_assignments_path
    )
    chain_profiles = _iter_jsonl(chain_profile_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)
    trusted_positive = tuple(
        float(row["cosine"])
        for row in _iter_jsonl(trusted_positive_path)
    )
    tracklet_family_rows = _iter_jsonl(tracklet_family_path)

    expansion_config = IdentityExpansionConfig(
        top_k_successors=args.top_k_successors,
        top_k_predecessors=args.top_k_predecessors,
        min_rank_score=args.min_rank_score,
        competition_mode=args.competition_mode,
        min_seed_profiled_tracklets=(
            args.min_seed_profiled_tracklets
        ),
        allow_different_kit=True,
        same_primary_bonus=args.same_primary_bonus,
        different_primary_penalty=args.different_primary_penalty,
        probable_different_primary_penalty=(
            args.probable_different_primary_penalty
        ),
        probable_primary_min_dominance=(
            args.probable_primary_min_dominance
        ),
        probable_primary_min_confidence=(
            args.probable_primary_min_confidence
        ),
        probable_primary_min_evidence_coverage=(
            args.probable_primary_min_evidence_coverage
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "global_identity_consolidation"
        / GLOBAL_IDENTITY_CONSOLIDATION_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    round_artifacts: dict[int, dict[str, str]] = {}

    def family_step(
        tracklet_assignments: list[Mapping[str, Any]]
        | tuple[Mapping[str, Any], ...],
    ) -> tuple[Mapping[str, Any], ...]:
        result = aggregate_entity_family_consensus(
            tracklet_consensus_rows=tracklet_family_rows,
            tracklet_entity_rows=tracklet_assignments,
            entity_id_field="identity_id",
            entity_level="consolidated_identity",
            config=KitEntityConsensusConfig(),
        )
        return tuple(row.to_dict() for row in result.entities)

    def expansion_step(
        identities,
        actor_assignments,
        tracklet_assignments,
        family_rows,
        round_index,
    ):
        result = run_identity_expansion(
            identity_rows=identities,
            persistent_actor_assignments=actor_assignments,
            persistent_tracklet_assignments=tracklet_assignments,
            chain_profile_rows=chain_profiles,
            hard_conflicts=hard_conflicts,
            trusted_positive_cosines=trusted_positive,
            identity_family_rows=family_rows,
            config=expansion_config,
        )

        round_dir = output_dir / f"round_{round_index:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)

        _write_jsonl(
            round_dir / "source_family_assignments.jsonl",
            family_rows,
        )
        _write_jsonl(
            round_dir / "profiles.jsonl",
            (row.to_dict() for row in result.profiles),
        )
        _write_jsonl(
            round_dir / "candidates.jsonl",
            (row.to_dict() for row in result.candidates),
        )
        _write_jsonl(
            round_dir / "selected_links.jsonl",
            (row.to_dict() for row in result.selected_candidates),
        )
        _write_jsonl(
            round_dir / "identities.jsonl",
            (row.to_dict() for row in result.expanded_identities),
        )
        _write_jsonl(
            round_dir / "actor_assignments.jsonl",
            result.actor_assignments,
        )
        _write_jsonl(
            round_dir / "tracklet_assignments.jsonl",
            result.tracklet_assignments,
        )
        _write_json(
            round_dir / "summary.json",
            result.summary(),
        )

        round_artifacts[round_index] = {
            "dir": str(round_dir),
            "summary": str(round_dir / "summary.json"),
            "selected_links": str(
                round_dir / "selected_links.jsonl"
            ),
        }
        return result

    result = consolidate_until_convergence(
        initial_identities=initial_identities,
        initial_actor_assignments=initial_actor_assignments,
        initial_tracklet_assignments=initial_tracklet_assignments,
        family_step=family_step,
        expansion_step=expansion_step,
        config=GlobalIdentityConsolidationConfig(
            max_rounds=args.max_rounds,
        ),
    )

    final_identities_path = output_dir / "final_identities.jsonl"
    final_actor_assignments_path = (
        output_dir / "final_actor_assignments.jsonl"
    )
    final_tracklet_assignments_path = (
        output_dir / "final_tracklet_assignments.jsonl"
    )
    final_family_path = (
        output_dir / "final_family_assignments.jsonl"
    )
    lineage_path = output_dir / "lineage.jsonl"
    all_links_path = (
        output_dir / "selected_links_all_rounds.jsonl"
    )

    _write_jsonl(final_identities_path, result.final_identities)
    _write_jsonl(
        final_actor_assignments_path,
        result.final_actor_assignments,
    )
    _write_jsonl(
        final_tracklet_assignments_path,
        result.final_tracklet_assignments,
    )
    _write_jsonl(final_family_path, result.final_family_rows)
    _write_jsonl(lineage_path, result.lineage_rows)
    _write_jsonl(
        all_links_path,
        result.selected_links_all_rounds,
    )

    # Pre-guard comparison.  7F.B is intentionally allowed to change the
    # previous 7E.5 link set: weak-primary contradictions were previously
    # collapsed to neutral.  We therefore report common/lost/new instead of
    # requiring exact equality.
    validated_7e5 = (
        root / "diagnostics" / "identity_expansion"
        / "seeded-identity-expansion-v2-family-aware"
        / "selected_links.jsonl"
    )
    round1_path = output_dir / "round_01" / "selected_links.jsonl"
    round1_comparison: dict[str, Any] = {
        "reference_available": validated_7e5.is_file(),
        "exact_selected_link_set_match": None,
        "reference_count": None,
        "round_01_count": None,
        "common": None,
        "lost": None,
        "new": None,
    }
    if validated_7e5.is_file() and round1_path.is_file():
        def keys(path: Path) -> set[tuple[str, str]]:
            return {
                (
                    str(row["predecessor_identity_id"]),
                    str(row["successor_identity_id"]),
                )
                for row in _iter_jsonl(path)
            }

        reference_keys = keys(validated_7e5)
        round_keys = keys(round1_path)
        round1_comparison.update({
            "exact_selected_link_set_match": (
                reference_keys == round_keys
            ),
            "reference_count": len(reference_keys),
            "round_01_count": len(round_keys),
            "common": len(reference_keys & round_keys),
            "lost": len(reference_keys - round_keys),
            "new": len(round_keys - reference_keys),
        })

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "identity_expansion_version": IDENTITY_EXPANSION_VERSION,
        "family_contradiction_guard": {
            "enabled": True,
            "weak_agreement_positive_bonus": False,
            "weak_opposition_relation": "probable_different_primary",
            "hard_cannot_link_created": False,
        },
        "expansion_config": expansion_config.to_dict(),
        "round_artifacts": {
            str(key): value
            for key, value in sorted(round_artifacts.items())
        },
        "round_01_vs_pre_guard_7e5": round1_comparison,
        "source_initial_identities": str(identities_path),
        "source_initial_actor_assignments": str(
            actor_assignments_path
        ),
        "source_initial_tracklet_assignments": str(
            tracklet_assignments_path
        ),
        "source_tracklet_family_consensus": str(
            tracklet_family_path
        ),
        "final_identities": str(final_identities_path),
        "final_actor_assignments": str(
            final_actor_assignments_path
        ),
        "final_tracklet_assignments": str(
            final_tracklet_assignments_path
        ),
        "final_family_assignments": str(final_family_path),
        "lineage": str(lineage_path),
        "selected_links_all_rounds": str(all_links_path),
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "consolidation_version": (
                GLOBAL_IDENTITY_CONSOLIDATION_VERSION
            ),
            "summary": "summary.json",
            "final_identities": final_identities_path.name,
            "final_actor_assignments": (
                final_actor_assignments_path.name
            ),
            "final_tracklet_assignments": (
                final_tracklet_assignments_path.name
            ),
            "final_family_assignments": final_family_path.name,
            "lineage": lineage_path.name,
            "selected_links_all_rounds": all_links_path.name,
            "canonical_candidate": True,
            "canonical": False,
        },
    )

    print(
        "[global-identity-consolidation-7F.B] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    for row in result.rounds:
        print(
            "[global-identity-consolidation-7F.B] "
            f"round={row.round_index} "
            f"source={row.source_identity_count} "
            f"profiles={row.profiled_identity_count} "
            f"family_usable={row.family_usable_count} "
            f"candidates={row.candidate_count} "
            f"selected={row.selected_link_count} "
            f"output={row.output_identity_count} "
            f"relations="
            f"{json.dumps(dict(row.selected_family_relation_counts), sort_keys=True)} "
            f"converged={row.converged}"
        )
    print(
        "[global-identity-consolidation-7F.B] "
        f"final_identities={len(result.final_identities)} "
        f"total_links={len(result.selected_links_all_rounds)} "
        f"converged={result.converged} "
        f"reason={result.stopped_reason}"
    )
    print(
        "[global-identity-consolidation-7F.B] "
        "round_01_vs_pre_guard_7E5="
        + json.dumps(round1_comparison, sort_keys=True)
    )
    print(
        "[global-identity-consolidation-7F.B] "
        f"artifact={summary_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
