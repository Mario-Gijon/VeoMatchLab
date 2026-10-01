"""Tracking V1 Hito 7F: iterative global identity consolidation.

7F.B keeps the iterative 7F controller but upgrades the family evidence
feeding each round with a weak-primary contradiction guard.  Conservative
ReID + physical hard conflicts + soft kit-family evidence can recover
additional long-gap links without treating weak agreement as positive proof.  One expansion pass is still
not enough, though: after a safe merge we have a richer identity profile and a
better family consensus, so the candidate graph should be rebuilt and scored
again.

This module provides the *iteration controller*.  It deliberately does not
invent a target population (22, 23, etc.) and it never relaxes the configured
association threshold between rounds.

Round loop
----------
current identities
    -> recompute entity family consensus from 7C.1E tracklet evidence
    -> rebuild aggregate ReID profiles
    -> refit hard-negative calibration
    -> rebuild global sequential candidates
    -> solve conservative family-aware expansion
    -> repeat until no selected links

Hard tracklet cannot-links remain inviolable because each expansion round
rebuilds component-level hard identity pairs from the original physical
conflicts.

The result of 7F is a *canonical candidate*, not yet canonical.  Hito 7G still
has to render/audit the complete clip before promotion.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence


GLOBAL_IDENTITY_CONSOLIDATION_VERSION = (
    "global-identity-consolidation-v2-family-guard"
)


def _identity_id(row: Mapping[str, Any]) -> str:
    return str(row["identity_id"])


def _source_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("source_identity_ids")
    if raw is None:
        return (_identity_id(row),)
    return tuple(str(value) for value in raw)


@dataclass(frozen=True, slots=True)
class GlobalIdentityConsolidationConfig:
    max_rounds: int = 8
    require_strict_count_decrease: bool = True

    def __post_init__(self) -> None:
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_rounds": self.max_rounds,
            "require_strict_count_decrease": (
                self.require_strict_count_decrease
            ),
        }


@dataclass(frozen=True, slots=True)
class ConsolidationRound:
    round_index: int
    source_identity_count: int
    profiled_identity_count: int
    family_usable_count: int
    candidate_count: int
    selected_link_count: int
    output_identity_count: int
    selected_family_relation_counts: Mapping[str, int]
    selected_score_min: float | None
    selected_score_median: float | None
    selected_score_max: float | None
    selected_gap_max_s: float | None
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "consolidation_version": (
                GLOBAL_IDENTITY_CONSOLIDATION_VERSION
            ),
            "round_index": self.round_index,
            "source_identity_count": self.source_identity_count,
            "profiled_identity_count": self.profiled_identity_count,
            "family_usable_count": self.family_usable_count,
            "candidate_count": self.candidate_count,
            "selected_link_count": self.selected_link_count,
            "output_identity_count": self.output_identity_count,
            "selected_family_relation_counts": dict(
                self.selected_family_relation_counts
            ),
            "selected_score_min": self.selected_score_min,
            "selected_score_median": self.selected_score_median,
            "selected_score_max": self.selected_score_max,
            "selected_gap_max_s": self.selected_gap_max_s,
            "converged": self.converged,
        }


@dataclass(frozen=True, slots=True)
class GlobalIdentityConsolidationResult:
    rounds: tuple[ConsolidationRound, ...]
    final_identities: tuple[Mapping[str, Any], ...]
    final_actor_assignments: tuple[Mapping[str, Any], ...]
    final_tracklet_assignments: tuple[Mapping[str, Any], ...]
    final_family_rows: tuple[Mapping[str, Any], ...]
    selected_links_all_rounds: tuple[Mapping[str, Any], ...]
    lineage_rows: tuple[Mapping[str, Any], ...]
    converged: bool
    stopped_reason: str
    config: GlobalIdentityConsolidationConfig

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "consolidation_version": (
                GLOBAL_IDENTITY_CONSOLIDATION_VERSION
            ),
            "round_count": len(self.rounds),
            "converged": self.converged,
            "stopped_reason": self.stopped_reason,
            "initial_identity_count": (
                self.rounds[0].source_identity_count
                if self.rounds else len(self.final_identities)
            ),
            "final_identity_count": len(self.final_identities),
            "total_selected_link_count": sum(
                row.selected_link_count for row in self.rounds
            ),
            "final_family_usable_count": sum(
                bool(row.get("usable_for_tracking", False))
                for row in self.final_family_rows
            ),
            "rounds": [row.to_dict() for row in self.rounds],
            "config": self.config.to_dict(),
            "population_forced": False,
            "threshold_relaxed_between_rounds": False,
            "hard_constraints_inviolable": True,
            "missing_family_evidence_is_negative": False,
            "provisional": True,
            "canonical_candidate": True,
            "canonical": False,
        }


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


def _build_lineage_seed(
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for row in identities:
        identity_id = _identity_id(row)
        if identity_id in result:
            raise ValueError(f"duplicate identity {identity_id}")
        result[identity_id] = _source_ids(row)
    return result


def _compose_lineage(
    *,
    output_identities: Sequence[Mapping[str, Any]],
    previous_lineage: Mapping[str, tuple[str, ...]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}

    for row in output_identities:
        output_id = _identity_id(row)
        source_ids = tuple(
            str(value)
            for value in row.get("source_identity_ids", ())
        )
        if not source_ids:
            raise ValueError(
                f"output identity {output_id} has no source_identity_ids"
            )

        initial: list[str] = []
        seen: set[str] = set()
        for source_id in source_ids:
            if source_id not in previous_lineage:
                raise ValueError(
                    f"unknown source identity in lineage: {source_id}"
                )
            for initial_id in previous_lineage[source_id]:
                if initial_id not in seen:
                    seen.add(initial_id)
                    initial.append(initial_id)

        result[output_id] = tuple(initial)

    return result


def _validate_assignment_coverage(
    *,
    identities: Sequence[Mapping[str, Any]],
    actor_assignments: Sequence[Mapping[str, Any]],
    tracklet_assignments: Sequence[Mapping[str, Any]],
) -> None:
    identity_ids = {_identity_id(row) for row in identities}

    actor_seen: set[str] = set()
    for row in actor_assignments:
        identity_id = str(row["identity_id"])
        actor_id = str(row["actor_id"])
        if identity_id not in identity_ids:
            raise ValueError(
                f"actor assignment references unknown identity {identity_id}"
            )
        if actor_id in actor_seen:
            raise ValueError(f"duplicate actor assignment {actor_id}")
        actor_seen.add(actor_id)

    tracklet_seen: set[str] = set()
    for row in tracklet_assignments:
        identity_id = str(row["identity_id"])
        tracklet_id = str(row["tracklet_id"])
        if identity_id not in identity_ids:
            raise ValueError(
                "tracklet assignment references unknown identity "
                f"{identity_id}"
            )
        if tracklet_id in tracklet_seen:
            raise ValueError(
                f"duplicate tracklet assignment {tracklet_id}"
            )
        tracklet_seen.add(tracklet_id)


def consolidate_until_convergence(
    *,
    initial_identities: Sequence[Mapping[str, Any]],
    initial_actor_assignments: Sequence[Mapping[str, Any]],
    initial_tracklet_assignments: Sequence[Mapping[str, Any]],
    family_step: Callable[
        [Sequence[Mapping[str, Any]]],
        Sequence[Mapping[str, Any]],
    ],
    expansion_step: Callable[
        [
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
            Sequence[Mapping[str, Any]],
            int,
        ],
        Any,
    ],
    config: GlobalIdentityConsolidationConfig | None = None,
) -> GlobalIdentityConsolidationResult:
    """Iteratively run the conservative expansion until convergence.

    ``expansion_step`` must return an object exposing the same public fields as
    ``IdentityExpansionResult``:
      profiles, candidates, selected_candidates, expanded_identities,
      actor_assignments, tracklet_assignments.

    This dependency injection keeps the controller independently testable.
    """

    selected = config or GlobalIdentityConsolidationConfig()

    current_identities = tuple(dict(row) for row in initial_identities)
    current_actor_assignments = tuple(
        dict(row) for row in initial_actor_assignments
    )
    current_tracklet_assignments = tuple(
        dict(row) for row in initial_tracklet_assignments
    )

    _validate_assignment_coverage(
        identities=current_identities,
        actor_assignments=current_actor_assignments,
        tracklet_assignments=current_tracklet_assignments,
    )

    lineage = _build_lineage_seed(current_identities)
    round_rows: list[ConsolidationRound] = []
    selected_links_all: list[dict[str, Any]] = []

    converged = False
    stopped_reason = "max_rounds"

    for round_index in range(1, selected.max_rounds + 1):
        family_rows = tuple(
            dict(row)
            for row in family_step(current_tracklet_assignments)
        )

        family_ids = {
            str(row.get("identity_id", row.get("entity_id")))
            for row in family_rows
        }
        missing_family_entities = (
            {_identity_id(row) for row in current_identities}
            - family_ids
        )
        if missing_family_entities:
            raise ValueError(
                "family step omitted identities: "
                + ", ".join(sorted(missing_family_entities)[:5])
            )

        expansion = expansion_step(
            current_identities,
            current_actor_assignments,
            current_tracklet_assignments,
            family_rows,
            round_index,
        )

        output_identities = tuple(
            row.to_dict() if hasattr(row, "to_dict") else dict(row)
            for row in expansion.expanded_identities
        )
        output_actor_assignments = tuple(
            dict(row) for row in expansion.actor_assignments
        )
        output_tracklet_assignments = tuple(
            dict(row) for row in expansion.tracklet_assignments
        )

        _validate_assignment_coverage(
            identities=output_identities,
            actor_assignments=output_actor_assignments,
            tracklet_assignments=output_tracklet_assignments,
        )

        selected_candidates = tuple(expansion.selected_candidates)
        scores = [
            float(row.rank_score)
            for row in selected_candidates
        ]
        gaps = [
            float(row.gap_s)
            for row in selected_candidates
        ]
        relation_counts = Counter(
            str(row.family_relation)
            for row in selected_candidates
        )

        converged_this_round = len(selected_candidates) == 0

        round_rows.append(
            ConsolidationRound(
                round_index=round_index,
                source_identity_count=len(current_identities),
                profiled_identity_count=len(expansion.profiles),
                family_usable_count=sum(
                    bool(row.get("usable_for_tracking", False))
                    for row in family_rows
                ),
                candidate_count=len(expansion.candidates),
                selected_link_count=len(selected_candidates),
                output_identity_count=len(output_identities),
                selected_family_relation_counts=dict(
                    sorted(relation_counts.items())
                ),
                selected_score_min=(
                    min(scores) if scores else None
                ),
                selected_score_median=_median(scores),
                selected_score_max=(
                    max(scores) if scores else None
                ),
                selected_gap_max_s=(
                    max(gaps) if gaps else None
                ),
                converged=converged_this_round,
            )
        )

        for candidate in selected_candidates:
            row = candidate.to_dict()
            predecessor = str(
                row["predecessor_identity_id"]
            )
            successor = str(
                row["successor_identity_id"]
            )
            row.update({
                "consolidation_version": (
                    GLOBAL_IDENTITY_CONSOLIDATION_VERSION
                ),
                "round_index": round_index,
                "predecessor_initial_identity_ids": list(
                    lineage[predecessor]
                ),
                "successor_initial_identity_ids": list(
                    lineage[successor]
                ),
            })
            selected_links_all.append(row)

        if converged_this_round:
            converged = True
            stopped_reason = "no_selected_links"
            # Do not replace current identities with a no-op renumbering round.
            break

        if (
            selected.require_strict_count_decrease
            and len(output_identities) >= len(current_identities)
        ):
            raise RuntimeError(
                "selected links did not strictly reduce identity count"
            )

        next_lineage = _compose_lineage(
            output_identities=output_identities,
            previous_lineage=lineage,
        )

        current_identities = output_identities
        current_actor_assignments = output_actor_assignments
        current_tracklet_assignments = output_tracklet_assignments
        lineage = next_lineage

    final_family_rows = tuple(
        dict(row)
        for row in family_step(current_tracklet_assignments)
    )

    lineage_rows = tuple(
        {
            "schema_version": 1,
            "consolidation_version": (
                GLOBAL_IDENTITY_CONSOLIDATION_VERSION
            ),
            "identity_id": identity_id,
            "initial_identity_ids": list(initial_ids),
            "initial_identity_count": len(initial_ids),
        }
        for identity_id, initial_ids in sorted(lineage.items())
    )

    return GlobalIdentityConsolidationResult(
        rounds=tuple(round_rows),
        final_identities=tuple(current_identities),
        final_actor_assignments=tuple(current_actor_assignments),
        final_tracklet_assignments=tuple(current_tracklet_assignments),
        final_family_rows=final_family_rows,
        selected_links_all_rounds=tuple(selected_links_all),
        lineage_rows=lineage_rows,
        converged=converged,
        stopped_reason=stopped_reason,
        config=selected,
    )
