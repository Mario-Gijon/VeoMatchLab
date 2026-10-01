"""Tracking V1 Hito 7G.4: conservative global trajectory proposal solver.

7G.3 scores a broad APPEND/INSERT hypothesis graph.  This stage selects a
small, non-conflicting set of *merge proposals* for audit.

It deliberately does NOT mutate identities yet.

Why proposals instead of immediately merging?
---------------------------------------------
The global trajectory score is new evidence and is not a calibrated
probability.  We therefore first select only mutually competitive, high-score,
hard-compatible proposals, audit them visually, and only then allow them to
change the identity partition.

Conservative first-pass rules
-----------------------------
* score must exist and be >= min_trajectory_score;
* candidate must be mutual-best (configurable, default mutual_best);
* hard physical cannot-links remain inviolable;
* family evidence remains soft -- it can reduce the score but never creates a
  hard prohibition here;
* selected proposals are pairwise identity-disjoint in this pass:
  one source identity can participate in at most one proposal;
* APPEND and INSERT compete in the same global queue;
* no target population is assumed.

Pairwise-disjoint selection is intentional.  It prevents a single bad global
edge from propagating through a long chain in the first trajectory round.
After audit/acceptance, a later iteration can rebuild trajectories and expose
new opportunities.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


GLOBAL_TRAJECTORY_PROPOSAL_SOLVER_VERSION = (
    "global-trajectory-proposal-solver-v1"
)


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryProposalSolverConfig:
    min_trajectory_score: float = 0.70
    competition_mode: str = "mutual_best"
    require_identity_disjoint: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_trajectory_score <= 1.0:
            raise ValueError("min_trajectory_score must be in [0,1]")
        if self.competition_mode not in {
            "mutual_best",
            "mutual_top3",
        }:
            raise ValueError(
                "competition_mode must be mutual_best or mutual_top3"
            )


@dataclass(frozen=True, slots=True)
class TrajectoryMergeProposal:
    proposal_rank: int
    candidate_type: str
    first_identity_id: str
    second_identity_id: str
    trajectory_score: float
    profile_confidence: float | None
    geometry_support: float
    family_relation: str
    source: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.source)
        payload.update({
            "schema_version": 1,
            "proposal_solver_version": (
                GLOBAL_TRAJECTORY_PROPOSAL_SOLVER_VERSION
            ),
            "proposal_rank": self.proposal_rank,
            "candidate_type": self.candidate_type,
            "first_identity_id": self.first_identity_id,
            "second_identity_id": self.second_identity_id,
            "trajectory_score": self.trajectory_score,
            "profile_confidence": self.profile_confidence,
            "geometry_support": self.geometry_support,
            "family_relation": self.family_relation,
            "proposal_selected": True,
            "identity_partition_changed": False,
        })
        return payload


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryProposalSolverResult:
    selected: tuple[TrajectoryMergeProposal, ...]
    eligible_count: int
    rejection_counts: Mapping[str, int]
    input_append_count: int
    input_insert_count: int
    config: GlobalTrajectoryProposalSolverConfig

    def summary(self, *, source_identity_count: int) -> dict[str, Any]:
        type_counts = Counter(
            row.candidate_type for row in self.selected
        )
        family_counts = Counter(
            row.family_relation for row in self.selected
        )
        selected_scores = [
            row.trajectory_score for row in self.selected
        ]
        return {
            "schema_version": 1,
            "proposal_solver_version": (
                GLOBAL_TRAJECTORY_PROPOSAL_SOLVER_VERSION
            ),
            "source_identity_count": source_identity_count,
            "input_append_count": self.input_append_count,
            "input_insert_count": self.input_insert_count,
            "eligible_count": self.eligible_count,
            "selected_proposal_count": len(self.selected),
            "selected_identity_count": 2 * len(self.selected),
            "projected_identity_count_if_all_accepted": (
                source_identity_count - len(self.selected)
            ),
            "selected_type_counts": dict(sorted(type_counts.items())),
            "selected_family_relation_counts": dict(
                sorted(family_counts.items())
            ),
            "selected_score_min": (
                min(selected_scores) if selected_scores else None
            ),
            "selected_score_median": (
                sorted(selected_scores)[len(selected_scores) // 2]
                if selected_scores else None
            ),
            "selected_score_max": (
                max(selected_scores) if selected_scores else None
            ),
            "rejection_counts": dict(
                sorted(self.rejection_counts.items())
            ),
            "config": {
                "min_trajectory_score": (
                    self.config.min_trajectory_score
                ),
                "competition_mode": self.config.competition_mode,
                "require_identity_disjoint": (
                    self.config.require_identity_disjoint
                ),
            },
            "hard_constraints_inviolable": True,
            "family_is_hard_constraint": False,
            "population_forced": False,
            "identity_partition_changed": False,
            "selection_is_proposal_only": True,
        }


def _competition_passes(
    row: Mapping[str, Any],
    mode: str,
) -> bool:
    if mode == "mutual_best":
        return bool(row.get("mutual_best", False))
    if mode == "mutual_top3":
        return bool(row.get("mutual_top3", False))
    raise ValueError(f"unsupported competition mode {mode!r}")


def _proposal_identity_pair(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    candidate_type = str(row["candidate_type"])
    if candidate_type == "append":
        return (
            str(row["predecessor_identity_id"]),
            str(row["successor_identity_id"]),
            candidate_type,
        )
    if candidate_type == "insert":
        return (
            str(row["host_identity_id"]),
            str(row["inserted_identity_id"]),
            candidate_type,
        )
    raise ValueError(
        f"unsupported candidate_type {candidate_type!r}"
    )


def _stable_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    first, second, candidate_type = _proposal_identity_pair(row)
    if candidate_type == "insert":
        detail = (
            int(row.get("host_gap_index", -1)),
            int(row.get("host_preceding_segment_index", -1)),
            int(row.get("host_following_segment_index", -1)),
        )
    else:
        detail = ()
    return (
        candidate_type,
        first,
        second,
        detail,
    )


def solve_global_trajectory_proposals(
    *,
    append_rows: Sequence[Mapping[str, Any]],
    insert_rows: Sequence[Mapping[str, Any]],
    hard_identity_pairs: set[tuple[str, str]],
    config: GlobalTrajectoryProposalSolverConfig | None = None,
) -> GlobalTrajectoryProposalSolverResult:
    selected_config = (
        config or GlobalTrajectoryProposalSolverConfig()
    )

    rejection = Counter()
    eligible: list[dict[str, Any]] = []

    for raw in [*append_rows, *insert_rows]:
        row = dict(raw)
        score = row.get("trajectory_score")
        if score is None:
            rejection["missing_score"] += 1
            continue

        score = float(score)
        if score < selected_config.min_trajectory_score:
            rejection["score_below_floor"] += 1
            continue

        if not _competition_passes(
            row,
            selected_config.competition_mode,
        ):
            rejection["competition_filter"] += 1
            continue

        first, second, _ = _proposal_identity_pair(row)
        if first == second:
            rejection["self_pair"] += 1
            continue

        if _canonical_pair(first, second) in hard_identity_pairs:
            rejection["hard_identity_conflict"] += 1
            continue

        eligible.append(row)

    # APPEND and INSERT compete globally.  Score comes first; confidence and
    # geometry only break ties.  The final stable key makes the result
    # deterministic.
    eligible.sort(
        key=lambda row: (
            -float(row["trajectory_score"]),
            -float(row.get("profile_confidence") or 0.0),
            -float(row.get("geometry_support") or 0.0),
            _stable_key(row),
        )
    )

    used_identities: set[str] = set()
    selected_rows: list[TrajectoryMergeProposal] = []

    for row in eligible:
        first, second, candidate_type = _proposal_identity_pair(row)

        if selected_config.require_identity_disjoint and (
            first in used_identities
            or second in used_identities
        ):
            rejection["identity_already_used"] += 1
            continue

        if selected_config.require_identity_disjoint:
            used_identities.add(first)
            used_identities.add(second)

        selected_rows.append(
            TrajectoryMergeProposal(
                proposal_rank=len(selected_rows) + 1,
                candidate_type=candidate_type,
                first_identity_id=first,
                second_identity_id=second,
                trajectory_score=float(row["trajectory_score"]),
                profile_confidence=(
                    None
                    if row.get("profile_confidence") is None
                    else float(row["profile_confidence"])
                ),
                geometry_support=float(
                    row.get("geometry_support", 0.0)
                ),
                family_relation=str(
                    row.get("family_relation", "neutral")
                ),
                source=row,
            )
        )

    # Assert the safety property instead of trusting the loop.
    if selected_config.require_identity_disjoint:
        flattened = [
            identity_id
            for proposal in selected_rows
            for identity_id in (
                proposal.first_identity_id,
                proposal.second_identity_id,
            )
        ]
        if len(flattened) != len(set(flattened)):
            raise AssertionError(
                "selected proposal identities are not disjoint"
            )

    return GlobalTrajectoryProposalSolverResult(
        selected=tuple(selected_rows),
        eligible_count=len(eligible),
        rejection_counts=dict(rejection),
        input_append_count=len(append_rows),
        input_insert_count=len(insert_rows),
        config=selected_config,
    )
