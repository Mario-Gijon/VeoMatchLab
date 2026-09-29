from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import exp, hypot, isfinite, log
from statistics import median
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .models import PositionKind, TrackSample


class FixedRosterInfeasibleError(RuntimeError):
    """No admissible exact-K identity forest exists."""


class PairEvidenceMode(str, Enum):
    HANDOVER = "handover"
    TEMPORAL = "temporal"
    CLOSE_DUPLICATE = "close_duplicate"
    NONE = "none"


@dataclass(slots=True, frozen=True)
class FixedRosterConfig:
    """
    Conservative pair-evidence model for V2.7.1 fixed-roster assignment.

    V2.6.2 deliberately allows identity fragments to overlap in time. That is
    necessary because the forward tracker can keep an old ID alive while a new
    ID is already receiving real observations for the same player.

    The optimizer builds exactly `roster_size` identity trees. Every non-root
    fragment selects one earlier fragment as parent. Pairwise hard conflicts
    prevent two clearly different simultaneously-observed people from ending
    in the same tree, including through a transitive chain.

    Pair scores are deterministic confidence scores in [0, 1], not calibrated
    probabilities.
    """

    roster_size: int = 23

    # Definite-different evidence.
    hard_conflict_min_frames: int = 3
    hard_conflict_min_overlap_fraction: float = 0.20
    hard_conflict_median_distance_m: float = 6.0

    # Handover: later fragment OBSERVED while earlier identity is often
    # PREDICTED/INTERPOLATED over the same frames.
    handover_min_compared_frames: int = 5
    handover_min_coverage: float = 0.70
    handover_reference_coverage: float = 0.70
    handover_min_fraction: float = 0.70
    handover_max_median_distance_m: float = 6.0
    handover_max_p90_distance_m: float = 12.0
    handover_position_sigma_m: float = 3.25
    handover_velocity_sigma_m_s: float = 5.0

    # Simultaneous duplicate observations of the same player.
    duplicate_max_median_distance_m: float = 3.5
    duplicate_max_p90_distance_m: float = 6.0
    duplicate_position_sigma_m: float = 1.75
    duplicate_velocity_sigma_m_s: float = 4.5

    # Non-overlapping continuation.
    temporal_min_gap_seconds: float = 1.0 / 30.0
    temporal_max_gap_seconds: float = 8.0
    temporal_endpoint_slack_m: float = 2.5
    max_speed_m_s: float = 12.5
    endpoint_fit_window_seconds: float = 0.80
    endpoint_fit_max_samples: int = 10
    temporal_position_sigma_base_m: float = 1.75
    temporal_position_sigma_growth_m_per_s: float = 1.50
    temporal_velocity_sigma_m_s: float = 5.0
    temporal_gap_tau_seconds: float = 4.0
    temporal_prediction_gate_base_m: float = 3.5
    temporal_prediction_gate_growth_m_per_s: float = 4.0
    temporal_prediction_gate_max_m: float = 18.0

    minimum_parent_score: float = 0.03

    high_confidence_score: float = 0.75
    medium_confidence_score: float = 0.50
    ranking_size: int = 5

    def __post_init__(self) -> None:
        if self.roster_size < 1:
            raise ValueError("roster_size must be >= 1")
        if self.hard_conflict_min_frames < 1:
            raise ValueError("hard_conflict_min_frames must be >= 1")
        fractions = (
            self.hard_conflict_min_overlap_fraction,
            self.handover_min_coverage,
            self.handover_reference_coverage,
            self.handover_min_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("fraction parameters must be in [0, 1]")

        positive = (
            self.hard_conflict_median_distance_m,
            self.handover_max_median_distance_m,
            self.handover_max_p90_distance_m,
            self.handover_position_sigma_m,
            self.handover_velocity_sigma_m_s,
            self.duplicate_max_median_distance_m,
            self.duplicate_max_p90_distance_m,
            self.duplicate_position_sigma_m,
            self.duplicate_velocity_sigma_m_s,
            self.temporal_min_gap_seconds,
            self.temporal_max_gap_seconds,
            self.max_speed_m_s,
            self.endpoint_fit_window_seconds,
            self.temporal_position_sigma_base_m,
            self.temporal_position_sigma_growth_m_per_s,
            self.temporal_velocity_sigma_m_s,
            self.temporal_gap_tau_seconds,
            self.temporal_prediction_gate_base_m,
            self.temporal_prediction_gate_growth_m_per_s,
            self.temporal_prediction_gate_max_m,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive FixedRosterConfig parameters must be > 0")

        if self.temporal_endpoint_slack_m < 0.0:
            raise ValueError("temporal_endpoint_slack_m must be >= 0")
        if self.temporal_max_gap_seconds < self.temporal_min_gap_seconds:
            raise ValueError(
                "temporal_max_gap_seconds must be >= temporal_min_gap_seconds"
            )
        if self.endpoint_fit_max_samples < 2:
            raise ValueError("endpoint_fit_max_samples must be >= 2")
        if self.handover_min_compared_frames < 1:
            raise ValueError("handover_min_compared_frames must be >= 1")

        for value in (
            self.minimum_parent_score,
            self.high_confidence_score,
            self.medium_confidence_score,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("score thresholds must be in [0, 1]")
        if self.medium_confidence_score > self.high_confidence_score:
            raise ValueError(
                "medium_confidence_score must be <= high_confidence_score"
            )
        if self.ranking_size < 1:
            raise ValueError("ranking_size must be >= 1")


@dataclass(slots=True, frozen=True)
class IdentityPairEvidence:
    earlier_id: int
    later_id: int

    score: float
    mode: PairEvidenceMode

    hard_conflict: bool
    hard_conflict_reason: str | None

    simultaneous_observed_frames: int
    simultaneous_observed_fraction: float
    simultaneous_median_distance_m: float | None
    simultaneous_p90_distance_m: float | None
    simultaneous_median_velocity_difference_m_s: float | None

    handover_compared_frames: int
    handover_coverage: float
    handover_fraction: float
    handover_median_distance_m: float | None
    handover_p90_distance_m: float | None
    handover_median_velocity_difference_m_s: float | None
    handover_score: float

    temporal_gap_seconds: float | None
    temporal_endpoint_distance_m: float | None
    temporal_implied_speed_m_s: float | None
    temporal_forward_error_m: float | None
    temporal_backward_error_m: float | None
    temporal_velocity_difference_m_s: float | None
    temporal_score: float

    duplicate_score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["mode"] = self.mode.value
        return payload


@dataclass(slots=True, frozen=True)
class FixedRosterParentLink:
    parent_id: int
    child_id: int
    score: float
    confidence: str
    evidence: IdentityPairEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "score": self.score,
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class FixedRosterReport:
    input_track_count: int
    roster_size: int
    candidate_parent_links: int
    hard_conflict_pairs: int
    forced_roots_without_parent_candidate: tuple[int, ...]

    selected_parent_links: tuple[FixedRosterParentLink, ...]
    components: tuple[tuple[int, ...], ...]
    representative_by_track: Mapping[int, int]
    candidate_rankings: Mapping[int, tuple[IdentityPairEvidence, ...]]

    @property
    def final_identity_count(self) -> int:
        return len(self.components)

    @property
    def high_confidence_links(self) -> tuple[FixedRosterParentLink, ...]:
        return tuple(
            link
            for link in self.selected_parent_links
            if link.confidence == "high"
        )

    @property
    def medium_confidence_links(self) -> tuple[FixedRosterParentLink, ...]:
        return tuple(
            link
            for link in self.selected_parent_links
            if link.confidence == "medium"
        )

    @property
    def low_confidence_links(self) -> tuple[FixedRosterParentLink, ...]:
        return tuple(
            link
            for link in self.selected_parent_links
            if link.confidence == "low"
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "input_tracks": self.input_track_count,
                "roster_size": self.roster_size,
                "candidate_parent_links": self.candidate_parent_links,
                "hard_conflict_pairs": self.hard_conflict_pairs,
                "forced_roots_without_parent_candidate": len(
                    self.forced_roots_without_parent_candidate
                ),
                "selected_parent_links": len(self.selected_parent_links),
                "final_identities": self.final_identity_count,
                "high_confidence_links": len(self.high_confidence_links),
                "medium_confidence_links": len(self.medium_confidence_links),
                "low_confidence_links": len(self.low_confidence_links),
            },
            "forced_root_track_ids": list(
                self.forced_roots_without_parent_candidate
            ),
            "selected_parent_links": [
                link.to_dict()
                for link in self.selected_parent_links
            ],
            "components": [
                list(component)
                for component in self.components
            ],
            "representative_by_track": {
                str(track_id): representative
                for track_id, representative in sorted(
                    self.representative_by_track.items()
                )
            },
            "candidate_rankings": {
                str(track_id): [
                    candidate.to_dict()
                    for candidate in ranking
                ]
                for track_id, ranking in sorted(
                    self.candidate_rankings.items()
                )
            },
        }


@dataclass(slots=True, frozen=True)
class _EndpointState:
    timestamp: float
    x: float
    z: float
    vx: float
    vz: float


def identity_pair_evidence(
    first: Sequence[TrackSample],
    second: Sequence[TrackSample],
    *,
    config: FixedRosterConfig | None = None,
) -> IdentityPairEvidence:
    """Return overlap-aware same-identity evidence for two trajectories."""
    cfg = config or FixedRosterConfig()

    first_id = _single_track_id(first)
    second_id = _single_track_id(second)
    if first_id == second_id:
        raise ValueError("identity_pair_evidence requires different track IDs")

    first_samples = _validated_samples(first_id, first)
    second_samples = _validated_samples(second_id, second)

    first_observed = _observed(first_samples)
    second_observed = _observed(second_samples)
    if not first_observed or not second_observed:
        return _empty_evidence(first_id, second_id)

    first_key = (
        first_observed[0].timestamp,
        first_id,
    )
    second_key = (
        second_observed[0].timestamp,
        second_id,
    )

    if first_key <= second_key:
        earlier_id, earlier_samples, earlier_observed = (
            first_id,
            first_samples,
            first_observed,
        )
        later_id, later_samples, later_observed = (
            second_id,
            second_samples,
            second_observed,
        )
    else:
        earlier_id, earlier_samples, earlier_observed = (
            second_id,
            second_samples,
            second_observed,
        )
        later_id, later_samples, later_observed = (
            first_id,
            first_samples,
            first_observed,
        )

    (
        simultaneous_frames,
        simultaneous_fraction,
        simultaneous_median_distance,
        simultaneous_p90_distance,
        simultaneous_median_velocity_difference,
    ) = _simultaneous_metrics(
        earlier_observed,
        later_observed,
    )

    handover = _handover_evidence(
        later_observed,
        earlier_samples,
        config=cfg,
    )
    temporal = _temporal_evidence(
        earlier_observed,
        later_observed,
        config=cfg,
    )

    # V2.7.2 nuance:
    #
    # Simultaneous OBSERVED support is normally evidence of two different
    # people, BUT an ID handover can briefly contain real observations on both
    # IDs while the tracker changes ownership. The diagnostic 9<->38 case is
    # exactly this pattern: appreciable overlap, yet >70% of the supported
    # transition is a proper OBSERVED-vs-PREDICTED handover.
    #
    # Therefore a *strong gated handover* is allowed to override the generic
    # simultaneous-observation conflict. Nearby players such as 3<->4,
    # 11<->25, 27<->28 remain conflicts because their handover fraction is low.
    strong_handover = (
        handover["score"] >= cfg.minimum_parent_score
        and handover["compared_frames"] >= cfg.handover_min_compared_frames
        and handover["coverage"] >= cfg.handover_min_coverage
        and handover["handover_fraction"] >= cfg.handover_min_fraction
    )

    hard_conflict = (
        simultaneous_frames >= cfg.hard_conflict_min_frames
        and simultaneous_fraction
        >= cfg.hard_conflict_min_overlap_fraction
        and not strong_handover
    )

    hard_reason = None
    if hard_conflict:
        median_text = (
            f"{simultaneous_median_distance:.3f}"
            if simultaneous_median_distance is not None
            else "null"
        )
        hard_reason = (
            "simultaneous_observed_overlap_without_strong_handover:"
            f"frames={simultaneous_frames},"
            f"fraction={simultaneous_fraction:.3f},"
            f"median_distance_m={median_text},"
            f"handover_fraction={handover['handover_fraction']:.3f}"
        )
    # CLOSE_DUPLICATE is intentionally disabled in V2.7.1. Geometry alone
    # cannot prove that two simultaneous footballers are the same person. Keep
    # the report field for backwards-compatible diagnostics, but it is always 0.
    duplicate_score = 0.0

    choices = [
        (handover["score"], PairEvidenceMode.HANDOVER),
        (temporal["score"], PairEvidenceMode.TEMPORAL),
    ]
    score, mode = max(
        choices,
        key=lambda item: (
            item[0],
            -_mode_priority(item[1]),
        ),
    )

    if hard_conflict:
        score = 0.0
        mode = PairEvidenceMode.NONE
    elif score < cfg.minimum_parent_score:
        score = 0.0
        mode = PairEvidenceMode.NONE

    return IdentityPairEvidence(
        earlier_id=earlier_id,
        later_id=later_id,
        score=float(_clamp01(score)),
        mode=mode,
        hard_conflict=hard_conflict,
        hard_conflict_reason=hard_reason,
        simultaneous_observed_frames=simultaneous_frames,
        simultaneous_observed_fraction=float(simultaneous_fraction),
        simultaneous_median_distance_m=simultaneous_median_distance,
        simultaneous_p90_distance_m=simultaneous_p90_distance,
        simultaneous_median_velocity_difference_m_s=(
            simultaneous_median_velocity_difference
        ),
        handover_compared_frames=int(handover["compared_frames"]),
        handover_coverage=float(handover["coverage"]),
        handover_fraction=float(handover["handover_fraction"]),
        handover_median_distance_m=handover["median_distance_m"],
        handover_p90_distance_m=handover["p90_distance_m"],
        handover_median_velocity_difference_m_s=(
            handover["median_velocity_difference_m_s"]
        ),
        handover_score=float(handover["score"]),
        temporal_gap_seconds=temporal["gap_seconds"],
        temporal_endpoint_distance_m=temporal["endpoint_distance_m"],
        temporal_implied_speed_m_s=temporal["implied_speed_m_s"],
        temporal_forward_error_m=temporal["forward_error_m"],
        temporal_backward_error_m=temporal["backward_error_m"],
        temporal_velocity_difference_m_s=(
            temporal["velocity_difference_m_s"]
        ),
        temporal_score=float(temporal["score"]),
        duplicate_score=float(duplicate_score),
    )


def resolve_fixed_roster(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    config: FixedRosterConfig | None = None,
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    FixedRosterReport,
]:
    """
    Globally partition trajectories into exactly K overlap-aware identities.

    The MILP has two kinds of binary variables:

      assignment[i, r]
          track i belongs to identity represented by earlier/equal track r.

      parent[e]
          a positive-evidence pair is selected as the parent edge connecting
          a non-root fragment to an earlier fragment in the same identity.

    Every non-root has exactly one selected parent. The parent graph is acyclic
    because pair edges are oriented by first real observation time (then ID).
    Exactly K roots therefore produce exactly K identity trees.

    Hard-conflict pairs are forbidden from sharing any representative, so a
    locally plausible chain cannot transitively collapse two people that are
    clearly observed far apart at the same time.
    """
    cfg = config or FixedRosterConfig()
    normalized = _validated_trajectories(trajectories)

    if not normalized:
        raise FixedRosterInfeasibleError("No trajectories were supplied")
    if len(normalized) < cfg.roster_size:
        raise FixedRosterInfeasibleError(
            f"Cannot form {cfg.roster_size} identities from "
            f"{len(normalized)} tracks"
        )

    ordered_ids = tuple(
        sorted(
            normalized,
            key=lambda track_id: (
                _observed(normalized[track_id])[0].timestamp,
                track_id,
            ),
        )
    )
    n = len(ordered_ids)
    index_by_id = {
        track_id: index
        for index, track_id in enumerate(ordered_ids)
    }

    evidence_by_pair: dict[
        tuple[int, int],
        IdentityPairEvidence,
    ] = {}
    hard_conflicts: set[tuple[int, int]] = set()
    parent_evidence: list[IdentityPairEvidence] = []
    rankings: dict[int, list[IdentityPairEvidence]] = {
        track_id: []
        for track_id in ordered_ids
    }

    for left_index, left_id in enumerate(ordered_ids):
        for right_id in ordered_ids[left_index + 1:]:
            evidence = identity_pair_evidence(
                normalized[left_id],
                normalized[right_id],
                config=cfg,
            )
            pair = (
                min(left_id, right_id),
                max(left_id, right_id),
            )
            evidence_by_pair[pair] = evidence
            rankings[left_id].append(evidence)
            rankings[right_id].append(evidence)

            if evidence.hard_conflict:
                hard_conflicts.add(pair)
            elif evidence.score >= cfg.minimum_parent_score:
                parent_evidence.append(evidence)

    ranking_report = {
        track_id: tuple(
            sorted(
                rows,
                key=lambda evidence: (
                    -evidence.score,
                    evidence.hard_conflict,
                    evidence.earlier_id,
                    evidence.later_id,
                ),
            )[:cfg.ranking_size]
        )
        for track_id, rows in rankings.items()
    }

    incoming_by_child: dict[int, list[int]] = {
        track_id: []
        for track_id in ordered_ids
    }
    for edge_index, evidence in enumerate(parent_evidence):
        incoming_by_child[evidence.later_id].append(edge_index)

    forced_roots = tuple(
        track_id
        for track_id in ordered_ids
        if not incoming_by_child[track_id]
    )
    if len(forced_roots) > cfg.roster_size:
        raise FixedRosterInfeasibleError(
            "Too many trajectories have no admissible earlier identity parent: "
            f"forced_roots={len(forced_roots)}, "
            f"target_roster={cfg.roster_size}, "
            f"track_ids={list(forced_roots)}"
        )

    selected_edge_indices, representative_index = _solve_identity_forest_milp(
        ordered_ids,
        parent_evidence,
        hard_conflicts,
        roster_size=cfg.roster_size,
    )

    representative_by_track = {
        track_id: ordered_ids[representative_index[index_by_id[track_id]]]
        for track_id in ordered_ids
    }

    grouped: dict[int, list[int]] = {}
    for track_id in ordered_ids:
        grouped.setdefault(
            representative_by_track[track_id],
            [],
        ).append(track_id)

    components = tuple(
        tuple(
            sorted(
                members,
                key=lambda track_id: (
                    index_by_id[track_id],
                    track_id,
                ),
            )
        )
        for _, members in sorted(
            grouped.items(),
            key=lambda item: index_by_id[item[0]],
        )
    )

    if len(components) != cfg.roster_size:
        raise RuntimeError(
            "Fixed-roster MILP invariant failed: "
            f"expected {cfg.roster_size} components, got {len(components)}"
        )

    selected_links = tuple(
        FixedRosterParentLink(
            parent_id=parent_evidence[edge_index].earlier_id,
            child_id=parent_evidence[edge_index].later_id,
            score=parent_evidence[edge_index].score,
            confidence=_confidence_label(
                parent_evidence[edge_index].score,
                cfg,
            ),
            evidence=parent_evidence[edge_index],
        )
        for edge_index in sorted(
            selected_edge_indices,
            key=lambda edge_index: (
                index_by_id[parent_evidence[edge_index].later_id],
                index_by_id[parent_evidence[edge_index].earlier_id],
            ),
        )
    )

    merged: dict[int, tuple[TrackSample, ...]] = {}
    for component in components:
        representative = representative_by_track[component[0]]
        merged[representative] = _merge_component(
            representative,
            component,
            normalized,
        )

    report = FixedRosterReport(
        input_track_count=n,
        roster_size=cfg.roster_size,
        candidate_parent_links=len(parent_evidence),
        hard_conflict_pairs=len(hard_conflicts),
        forced_roots_without_parent_candidate=forced_roots,
        selected_parent_links=selected_links,
        components=components,
        representative_by_track=representative_by_track,
        candidate_rankings=ranking_report,
    )
    return merged, report


def _solve_identity_forest_milp(
    ordered_ids: tuple[int, ...],
    parent_evidence: Sequence[IdentityPairEvidence],
    hard_conflicts: set[tuple[int, int]],
    *,
    roster_size: int,
) -> tuple[set[int], list[int]]:
    n = len(ordered_ids)
    edge_count = len(parent_evidence)
    index_by_id = {
        track_id: index
        for index, track_id in enumerate(ordered_ids)
    }

    assignment_count = n * n
    variable_count = assignment_count + edge_count

    def a_var(item_index: int, representative_index: int) -> int:
        return item_index * n + representative_index

    def p_var(edge_index: int) -> int:
        return assignment_count + edge_index

    objective = np.zeros(variable_count, dtype=np.float64)
    epsilon = 1e-12
    for edge_index, evidence in enumerate(parent_evidence):
        objective[p_var(edge_index)] = -log(
            max(evidence.score, epsilon)
        )

    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)

    # A track can only use itself or an earlier-ordered track as representative.
    for item_index in range(n):
        for representative_index in range(item_index + 1, n):
            upper_bounds[
                a_var(item_index, representative_index)
            ] = 0.0

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float],
        lower: float,
        upper: float,
    ) -> None:
        rows.append(dict(coefficients))
        row_lower.append(lower)
        row_upper.append(upper)

    # Every track belongs to exactly one representative.
    for item_index in range(n):
        add_constraint(
            {
                a_var(item_index, representative_index): 1.0
                for representative_index in range(item_index + 1)
            },
            1.0,
            1.0,
        )

    # A representative must represent itself before others may use it.
    for item_index in range(n):
        for representative_index in range(item_index):
            add_constraint(
                {
                    a_var(item_index, representative_index): 1.0,
                    a_var(
                        representative_index,
                        representative_index,
                    ): -1.0,
                },
                -np.inf,
                0.0,
            )

    # Exactly K roots / identity representatives.
    add_constraint(
        {
            a_var(item_index, item_index): 1.0
            for item_index in range(n)
        },
        float(roster_size),
        float(roster_size),
    )

    incoming_edge_indices: dict[int, list[int]] = {
        item_index: []
        for item_index in range(n)
    }
    for edge_index, evidence in enumerate(parent_evidence):
        child_index = index_by_id[evidence.later_id]
        incoming_edge_indices[child_index].append(edge_index)

    # Root OR exactly one positive-evidence parent.
    for item_index in range(n):
        coefficients = {
            a_var(item_index, item_index): 1.0,
        }
        for edge_index in incoming_edge_indices[item_index]:
            coefficients[p_var(edge_index)] = 1.0
        add_constraint(
            coefficients,
            1.0,
            1.0,
        )

    # A selected parent edge requires parent and child to share the same
    # representative. Equality is enforced over the full assignment vector.
    for edge_index, evidence in enumerate(parent_evidence):
        parent_index = index_by_id[evidence.earlier_id]
        child_index = index_by_id[evidence.later_id]
        parent_variable = p_var(edge_index)

        for representative_index in range(n):
            add_constraint(
                {
                    parent_variable: 1.0,
                    a_var(parent_index, representative_index): 1.0,
                    a_var(child_index, representative_index): -1.0,
                },
                -np.inf,
                1.0,
            )
            add_constraint(
                {
                    parent_variable: 1.0,
                    a_var(parent_index, representative_index): -1.0,
                    a_var(child_index, representative_index): 1.0,
                },
                -np.inf,
                1.0,
            )

    # Definite-different tracks may never share a representative, even through
    # a transitive chain of individually plausible parent links.
    for left_id, right_id in hard_conflicts:
        left_index = index_by_id[left_id]
        right_index = index_by_id[right_id]
        common_limit = min(left_index, right_index)
        for representative_index in range(common_limit + 1):
            add_constraint(
                {
                    a_var(left_index, representative_index): 1.0,
                    a_var(right_index, representative_index): 1.0,
                },
                -np.inf,
                1.0,
            )

    data: list[float] = []
    row_indices: list[int] = []
    column_indices: list[int] = []

    for row_index, coefficients in enumerate(rows):
        for column_index, coefficient in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            data.append(coefficient)

    matrix = coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(rows), variable_count),
    ).tocsr()

    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int32),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix,
            np.asarray(row_lower, dtype=np.float64),
            np.asarray(row_upper, dtype=np.float64),
        ),
        options={"presolve": True},
    )

    if not result.success or result.x is None:
        raise FixedRosterInfeasibleError(
            "Global exact-K overlap-aware identity MILP is infeasible "
            f"(status={result.status}, message={result.message!r}). "
            "Inspect forced roots, candidate rankings and hard conflicts "
            "before changing thresholds."
        )

    selected_edges = {
        edge_index
        for edge_index in range(edge_count)
        if result.x[p_var(edge_index)] >= 0.5
    }

    representative_index: list[int] = []
    for item_index in range(n):
        active = [
            rep_index
            for rep_index in range(item_index + 1)
            if result.x[a_var(item_index, rep_index)] >= 0.5
        ]
        if len(active) != 1:
            raise RuntimeError(
                "MILP assignment invariant failed for "
                f"track index {item_index}: {active}"
            )
        representative_index.append(active[0])

    expected_parent_edges = n - roster_size
    if len(selected_edges) != expected_parent_edges:
        raise RuntimeError(
            "MILP parent-edge invariant failed: "
            f"expected {expected_parent_edges}, got {len(selected_edges)}"
        )

    return selected_edges, representative_index


def _handover_evidence(
    later_observed: Sequence[TrackSample],
    earlier_trajectory: Sequence[TrackSample],
    *,
    config: FixedRosterConfig,
) -> dict[str, object]:
    """
    Score the temporal overlap in which the earlier raw ID is still alive.

    V2.7.1 divided by the *entire* lifetime of the later fragment. That was
    wrong for long-lived replacement IDs: e.g. raw 29 can continue for twenty
    seconds after raw 1 has fully expired, even though the first several seconds
    form an almost perfect handover.

    Coverage is now local to the time support where the earlier trajectory
    exists. This asks the correct question:

        "While both raw IDs could be compared, how complete and handover-like
         was the transition?"

    It does not penalize the successor merely for surviving much longer.
    """
    if not later_observed or not earlier_trajectory:
        return _empty_handover()

    earlier_by_frame = {
        sample.frame_index: sample
        for sample in earlier_trajectory
    }
    first_supported_frame = earlier_trajectory[0].frame_index
    last_supported_frame = earlier_trajectory[-1].frame_index

    supported_later = [
        fragment
        for fragment in later_observed
        if (
            first_supported_frame
            <= fragment.frame_index
            <= last_supported_frame
        )
    ]
    if not supported_later:
        return _empty_handover()

    distances: list[float] = []
    velocity_differences: list[float] = []
    handover_frames = 0

    for fragment in supported_later:
        earlier = earlier_by_frame.get(fragment.frame_index)
        if earlier is None:
            continue

        distance = hypot(
            fragment.x - earlier.x,
            fragment.z - earlier.z,
        )
        velocity_difference = hypot(
            fragment.vx - earlier.vx,
            fragment.vz - earlier.vz,
        )
        if not isfinite(distance) or not isfinite(velocity_difference):
            continue

        distances.append(distance)
        velocity_differences.append(velocity_difference)

        if earlier.position_kind is not PositionKind.OBSERVED:
            handover_frames += 1

    if not distances:
        return _empty_handover()

    compared = len(distances)
    coverage = compared / len(supported_later)
    handover_fraction = handover_frames / compared
    sorted_distances = sorted(distances)
    median_distance = float(median(sorted_distances))
    p90_distance = _percentile(sorted_distances, 0.90)
    median_velocity_difference = float(median(velocity_differences))

    if (
        compared < config.handover_min_compared_frames
        or coverage < config.handover_min_coverage
        or handover_fraction < config.handover_min_fraction
        or median_distance > config.handover_max_median_distance_m
        or p90_distance > config.handover_max_p90_distance_m
    ):
        score = 0.0
    else:
        coverage_term = max(
            1e-6,
            min(
                1.0,
                coverage / config.handover_reference_coverage,
            ),
        )
        handover_term = max(1e-6, handover_fraction)
        position_term = max(
            1e-12,
            _gaussian_score(
                median_distance,
                config.handover_position_sigma_m,
            ),
        )
        velocity_term = max(
            1e-12,
            _gaussian_score(
                median_velocity_difference,
                config.handover_velocity_sigma_m_s,
            ),
        )

        score = exp(
            0.22 * log(coverage_term)
            + 0.28 * log(handover_term)
            + 0.35 * log(position_term)
            + 0.15 * log(velocity_term)
        )

    return {
        "compared_frames": compared,
        "coverage": coverage,
        "handover_fraction": handover_fraction,
        "median_distance_m": median_distance,
        "p90_distance_m": p90_distance,
        "median_velocity_difference_m_s": median_velocity_difference,
        "score": _clamp01(score),
    }


def _duplicate_score(
    *,
    simultaneous_frames: int,
    simultaneous_fraction: float,
    median_distance_m: float | None,
    p90_distance_m: float | None,
    median_velocity_difference_m_s: float | None,
    config: FixedRosterConfig,
) -> float:
    if (
        simultaneous_frames < config.hard_conflict_min_frames
        or median_distance_m is None
        or p90_distance_m is None
        or median_velocity_difference_m_s is None
        or median_distance_m > config.duplicate_max_median_distance_m
        or p90_distance_m > config.duplicate_max_p90_distance_m
    ):
        return 0.0

    overlap_term = max(1e-6, simultaneous_fraction)
    position_term = max(
        1e-12,
        _gaussian_score(
            median_distance_m,
            config.duplicate_position_sigma_m,
        ),
    )
    velocity_term = max(
        1e-12,
        _gaussian_score(
            median_velocity_difference_m_s,
            config.duplicate_velocity_sigma_m_s,
        ),
    )

    return _clamp01(
        exp(
            0.25 * log(overlap_term)
            + 0.55 * log(position_term)
            + 0.20 * log(velocity_term)
        )
    )


def _temporal_evidence(
    earlier_observed: Sequence[TrackSample],
    later_observed: Sequence[TrackSample],
    *,
    config: FixedRosterConfig,
) -> dict[str, object]:
    if not earlier_observed or not later_observed:
        return _empty_temporal()

    if earlier_observed[-1].timestamp >= later_observed[0].timestamp:
        return _empty_temporal()

    earlier = _endpoint_state(
        earlier_observed,
        at_start=False,
        config=config,
    )
    later = _endpoint_state(
        later_observed,
        at_start=True,
        config=config,
    )

    gap = later.timestamp - earlier.timestamp
    if (
        gap < config.temporal_min_gap_seconds
        or gap > config.temporal_max_gap_seconds
    ):
        return _empty_temporal()

    endpoint_distance = hypot(
        later.x - earlier.x,
        later.z - earlier.z,
    )
    implied_speed = endpoint_distance / gap
    reachable = (
        config.temporal_endpoint_slack_m
        + config.max_speed_m_s * gap
    )
    if endpoint_distance > reachable:
        return _empty_temporal()

    forward_x = earlier.x + earlier.vx * gap
    forward_z = earlier.z + earlier.vz * gap
    backward_x = later.x - later.vx * gap
    backward_z = later.z - later.vz * gap

    forward_error = hypot(
        later.x - forward_x,
        later.z - forward_z,
    )
    backward_error = hypot(
        earlier.x - backward_x,
        earlier.z - backward_z,
    )
    velocity_difference = hypot(
        earlier.vx - later.vx,
        earlier.vz - later.vz,
    )

    prediction_gate = min(
        config.temporal_prediction_gate_max_m,
        config.temporal_prediction_gate_base_m
        + config.temporal_prediction_gate_growth_m_per_s * gap,
    )
    if (
        forward_error > prediction_gate
        or backward_error > prediction_gate
    ):
        return _empty_temporal()

    sigma_position = (
        config.temporal_position_sigma_base_m
        + config.temporal_position_sigma_growth_m_per_s * gap
    )

    forward_term = max(
        1e-12,
        _gaussian_score(
            forward_error,
            sigma_position,
        ),
    )
    backward_term = max(
        1e-12,
        _gaussian_score(
            backward_error,
            sigma_position,
        ),
    )
    velocity_term = max(
        1e-12,
        _gaussian_score(
            velocity_difference,
            config.temporal_velocity_sigma_m_s,
        ),
    )
    gap_term = max(
        1e-12,
        exp(-gap / config.temporal_gap_tau_seconds),
    )

    reachable_motion = max(
        config.max_speed_m_s * gap,
        1e-9,
    )
    excess_after_slack = max(
        0.0,
        endpoint_distance - config.temporal_endpoint_slack_m,
    )
    reach_fraction = min(
        1.0,
        excess_after_slack / reachable_motion,
    )
    reach_term = max(
        1e-6,
        1.0 - reach_fraction * reach_fraction,
    )

    score = exp(
        0.30 * log(forward_term)
        + 0.22 * log(backward_term)
        + 0.18 * log(velocity_term)
        + 0.15 * log(gap_term)
        + 0.15 * log(reach_term)
    )

    return {
        "gap_seconds": float(gap),
        "endpoint_distance_m": float(endpoint_distance),
        "implied_speed_m_s": float(implied_speed),
        "forward_error_m": float(forward_error),
        "backward_error_m": float(backward_error),
        "velocity_difference_m_s": float(velocity_difference),
        "score": _clamp01(score),
    }


def _simultaneous_metrics(
    first_observed: Sequence[TrackSample],
    second_observed: Sequence[TrackSample],
) -> tuple[int, float, float | None, float | None, float | None]:
    second_by_frame = {
        sample.frame_index: sample
        for sample in second_observed
    }

    distances: list[float] = []
    velocity_differences: list[float] = []

    for first in first_observed:
        second = second_by_frame.get(first.frame_index)
        if second is None:
            continue

        distance = hypot(
            first.x - second.x,
            first.z - second.z,
        )
        velocity_difference = hypot(
            first.vx - second.vx,
            first.vz - second.vz,
        )
        if not isfinite(distance) or not isfinite(velocity_difference):
            continue

        distances.append(distance)
        velocity_differences.append(velocity_difference)

    frames = len(distances)
    shorter = min(
        len(first_observed),
        len(second_observed),
    )
    fraction = frames / shorter if shorter else 0.0

    if not distances:
        return 0, 0.0, None, None, None

    sorted_distances = sorted(distances)
    return (
        frames,
        float(fraction),
        float(median(sorted_distances)),
        float(_percentile(sorted_distances, 0.90)),
        float(median(velocity_differences)),
    )


def _endpoint_state(
    observed: Sequence[TrackSample],
    *,
    at_start: bool,
    config: FixedRosterConfig,
) -> _EndpointState:
    anchor = observed[0] if at_start else observed[-1]
    anchor_time = anchor.timestamp

    if at_start:
        candidates = [
            sample
            for sample in observed
            if (
                0.0
                <= sample.timestamp - anchor_time
                <= config.endpoint_fit_window_seconds
            )
        ][:config.endpoint_fit_max_samples]
    else:
        candidates = [
            sample
            for sample in observed
            if (
                0.0
                <= anchor_time - sample.timestamp
                <= config.endpoint_fit_window_seconds
            )
        ][-config.endpoint_fit_max_samples:]

    if len(candidates) < 2:
        return _EndpointState(
            timestamp=anchor_time,
            x=anchor.x,
            z=anchor.z,
            vx=anchor.vx,
            vz=anchor.vz,
        )

    times = np.asarray(
        [
            sample.timestamp - anchor_time
            for sample in candidates
        ],
        dtype=np.float64,
    )
    design = np.column_stack(
        (
            np.ones_like(times),
            times,
        )
    )
    xs = np.asarray(
        [sample.x for sample in candidates],
        dtype=np.float64,
    )
    zs = np.asarray(
        [sample.z for sample in candidates],
        dtype=np.float64,
    )

    try:
        x_coeff, *_ = np.linalg.lstsq(
            design,
            xs,
            rcond=None,
        )
        z_coeff, *_ = np.linalg.lstsq(
            design,
            zs,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return _EndpointState(
            timestamp=anchor_time,
            x=anchor.x,
            z=anchor.z,
            vx=anchor.vx,
            vz=anchor.vz,
        )

    x, vx = float(x_coeff[0]), float(x_coeff[1])
    z, vz = float(z_coeff[0]), float(z_coeff[1])

    speed = hypot(vx, vz)
    if speed > config.max_speed_m_s:
        scale = config.max_speed_m_s / speed
        vx *= scale
        vz *= scale

    if not all(
        isfinite(value)
        for value in (x, z, vx, vz)
    ):
        return _EndpointState(
            timestamp=anchor_time,
            x=anchor.x,
            z=anchor.z,
            vx=anchor.vx,
            vz=anchor.vz,
        )

    return _EndpointState(
        timestamp=anchor_time,
        x=x,
        z=z,
        vx=vx,
        vz=vz,
    )


def _merge_component(
    representative: int,
    component: Sequence[int],
    trajectories: Mapping[int, tuple[TrackSample, ...]],
) -> tuple[TrackSample, ...]:
    by_frame: dict[int, tuple[int, TrackSample]] = {}

    ordered_sources = [representative] + [
        track_id
        for track_id in component
        if track_id != representative
    ]

    for source_id in ordered_sources:
        for sample in trajectories[source_id]:
            rewritten = replace(
                sample,
                track_id=representative,
            )
            existing = by_frame.get(rewritten.frame_index)

            if existing is None:
                by_frame[rewritten.frame_index] = (
                    source_id,
                    rewritten,
                )
                continue

            existing_source, existing_sample = existing
            if _prefer_sample(
                new=rewritten,
                new_source=source_id,
                existing=existing_sample,
                existing_source=existing_source,
                representative=representative,
            ):
                by_frame[rewritten.frame_index] = (
                    source_id,
                    rewritten,
                )

    return tuple(
        sample
        for _, sample in sorted(
            by_frame.values(),
            key=lambda pair: (
                pair[1].frame_index,
                pair[1].timestamp,
            ),
        )
    )


def _prefer_sample(
    *,
    new: TrackSample,
    new_source: int,
    existing: TrackSample,
    existing_source: int,
    representative: int,
) -> bool:
    new_priority = _sample_priority(new)
    existing_priority = _sample_priority(existing)

    if new_priority != existing_priority:
        return new_priority > existing_priority

    if new.position_kind is PositionKind.OBSERVED:
        if (
            new.confidence is not None
            and existing.confidence is not None
        ):
            if new.confidence != existing.confidence:
                return new.confidence > existing.confidence
        elif new.confidence is not None:
            return True
        elif existing.confidence is not None:
            return False

    if existing_source == representative:
        return False
    if new_source == representative:
        return True

    return new_source < existing_source


def _sample_priority(sample: TrackSample) -> int:
    return {
        PositionKind.OBSERVED: 3,
        PositionKind.INTERPOLATED: 2,
        PositionKind.PREDICTED: 1,
    }[sample.position_kind]


def _observed(
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    return tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.OBSERVED
    )


def _validated_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
) -> dict[int, tuple[TrackSample, ...]]:
    result: dict[int, tuple[TrackSample, ...]] = {}

    for track_id, samples in sorted(trajectories.items()):
        normalized = _validated_samples(track_id, samples)
        if normalized and _observed(normalized):
            result[track_id] = normalized

    return result


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.frame_index,
                sample.timestamp,
            ),
        )
    )

    previous_frame: int | None = None
    previous_timestamp: float | None = None

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match "
                f"sample track_id {sample.track_id}"
            )
        if sample.frame_index < 0:
            raise ValueError("Track sample frame_index must be >= 0")
        if not isfinite(sample.timestamp):
            raise ValueError("Track sample timestamp must be finite")
        if not all(
            isfinite(value)
            for value in (
                sample.x,
                sample.z,
                sample.vx,
                sample.vz,
            )
        ):
            raise ValueError(
                f"Track {track_id} contains non-finite state"
            )
        if (
            previous_frame is not None
            and sample.frame_index == previous_frame
        ):
            raise ValueError(
                f"Track {track_id} contains duplicate frame_index "
                f"{sample.frame_index}"
            )
        if (
            previous_timestamp is not None
            and sample.timestamp <= previous_timestamp
        ):
            raise ValueError(
                f"Track {track_id} timestamps must be strictly increasing"
            )

        previous_frame = sample.frame_index
        previous_timestamp = sample.timestamp

    return ordered


def _single_track_id(samples: Sequence[TrackSample]) -> int:
    if not samples:
        raise ValueError("Track sequence must not be empty")
    track_id = samples[0].track_id
    if any(sample.track_id != track_id for sample in samples):
        raise ValueError("Track sequence contains multiple track IDs")
    return track_id


def _empty_evidence(
    first_id: int,
    second_id: int,
) -> IdentityPairEvidence:
    earlier_id, later_id = sorted((first_id, second_id))
    return IdentityPairEvidence(
        earlier_id=earlier_id,
        later_id=later_id,
        score=0.0,
        mode=PairEvidenceMode.NONE,
        hard_conflict=False,
        hard_conflict_reason=None,
        simultaneous_observed_frames=0,
        simultaneous_observed_fraction=0.0,
        simultaneous_median_distance_m=None,
        simultaneous_p90_distance_m=None,
        simultaneous_median_velocity_difference_m_s=None,
        handover_compared_frames=0,
        handover_coverage=0.0,
        handover_fraction=0.0,
        handover_median_distance_m=None,
        handover_p90_distance_m=None,
        handover_median_velocity_difference_m_s=None,
        handover_score=0.0,
        temporal_gap_seconds=None,
        temporal_endpoint_distance_m=None,
        temporal_implied_speed_m_s=None,
        temporal_forward_error_m=None,
        temporal_backward_error_m=None,
        temporal_velocity_difference_m_s=None,
        temporal_score=0.0,
        duplicate_score=0.0,
    )


def _empty_handover() -> dict[str, object]:
    return {
        "compared_frames": 0,
        "coverage": 0.0,
        "handover_fraction": 0.0,
        "median_distance_m": None,
        "p90_distance_m": None,
        "median_velocity_difference_m_s": None,
        "score": 0.0,
    }


def _empty_temporal() -> dict[str, object]:
    return {
        "gap_seconds": None,
        "endpoint_distance_m": None,
        "implied_speed_m_s": None,
        "forward_error_m": None,
        "backward_error_m": None,
        "velocity_difference_m_s": None,
        "score": 0.0,
    }


def _confidence_label(
    score: float,
    config: FixedRosterConfig,
) -> str:
    if score >= config.high_confidence_score:
        return "high"
    if score >= config.medium_confidence_score:
        return "medium"
    return "low"


def _mode_priority(mode: PairEvidenceMode) -> int:
    return {
        PairEvidenceMode.HANDOVER: 0,
        PairEvidenceMode.TEMPORAL: 1,
        PairEvidenceMode.CLOSE_DUPLICATE: 2,
        PairEvidenceMode.NONE: 3,
    }[mode]


def _gaussian_score(value: float, sigma: float) -> float:
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0")
    return exp(-0.5 * (value / sigma) ** 2)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _percentile(
    sorted_values: Sequence[float],
    q: float,
) -> float:
    if not sorted_values:
        raise ValueError("percentile requires data")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )
