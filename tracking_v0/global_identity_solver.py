"""V2.8A-V3: candidate identity clusters + set-partition MILP.

This audit solver removes the ROOT-star limitation from V1/V2.

Instead of deciding:
    ROOT <- FRAGMENT
directly, it first generates *candidate canonical identity clusters*.

A valid cluster:
  * contains no HARD/STRONG conflicting pair,
  * is connected by positive identity evidence,
  * may contain A-B-C even when A-C is weak, as long as A-B and B-C provide a
    valid support path,
  * may contain conservative appearance segments from the same raw tracker ID;
    those visual splits pay only a soft same-identity penalty.

The MILP then chooses:
  * exactly K candidate clusters (23 in the known no-substitution clip),
  * each identity unit belongs to at most one chosen cluster,
  * every unselected unit becomes JUNK.

Only after the partition is solved is the best canonical root selected inside
each chosen cluster.  The root therefore no longer constrains cluster formation.

This is still AUDIT-ONLY.  It does not rewrite tracking frames.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import itertools
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .team_appearance import AppearanceLabel
from .team_similarity import RelativeTeamLabel


class PairConflict(str, Enum):
    NONE = "none"
    STRONG = "strong"
    HARD = "hard"


@dataclass(frozen=True)
class IdentityUnit:
    key: str
    track_id: int
    segment_index: int
    start_frame: int
    end_frame: int

    observed_count: int
    observed_span_s: float

    appearance_label: AppearanceLabel
    team_label: RelativeTeamLabel

    appearance_coherence: float
    appearance_q10: float
    embedding_count: int

    quality: float
    role_anchor: bool = False

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("unit key must not be empty")
        if self.end_frame < self.start_frame:
            raise ValueError("invalid frame range")
        if self.observed_count < 0:
            raise ValueError("observed_count must be >= 0")
        if self.observed_span_s < 0.0:
            raise ValueError("observed_span_s must be >= 0")
        if self.embedding_count < 1:
            raise ValueError("embedding_count must be >= 1")
        for name in (
            "appearance_coherence",
            "appearance_q10",
            "quality",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["appearance_label"] = self.appearance_label.value
        payload["team_label"] = self.team_label.value
        return payload


@dataclass(frozen=True)
class UnitPairEvidence:
    unit_a: str
    unit_b: str

    reid_score: float
    geometry_score: float
    team_score: float
    fused_support_score: float

    team_compatibility: str

    simultaneous_count: int
    simultaneous_fraction: float
    simultaneous_median_distance_m: float | None

    conflict: PairConflict
    reason: str

    # Applied only if both units are placed in the same canonical identity.
    soft_same_identity_penalty: float = 0.0

    def __post_init__(self) -> None:
        if self.unit_a == self.unit_b:
            raise ValueError("pair requires different units")
        for name in (
            "reid_score",
            "geometry_score",
            "team_score",
            "fused_support_score",
            "simultaneous_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.simultaneous_count < 0:
            raise ValueError("simultaneous_count must be >= 0")
        if self.soft_same_identity_penalty < 0.0:
            raise ValueError(
                "soft_same_identity_penalty must be >= 0"
            )

    @property
    def pair_key(self) -> tuple[str, str]:
        return _pair_key(self.unit_a, self.unit_b)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["conflict"] = self.conflict.value
        return payload


@dataclass(frozen=True)
class ClusterSolverConfig:
    identity_count: int = 23

    # Candidate graph.
    min_connectivity_support: float = 0.60
    preferred_connectivity_support: float = 0.67
    max_cluster_size: int = 5
    max_candidates: int = 25000

    # Cluster scoring.
    preserve_realness_weight: float = 0.80
    mst_support_weight: float = 0.42
    strong_edge_bonus_weight: float = 0.12
    weak_edge_penalty_weight: float = 0.14
    cluster_size_penalty: float = 0.025

    # Junk scoring.
    junk_low_quality_reward: float = 0.16
    junk_bias: float = 0.03
    junk_realness_penalty: float = 0.78
    junk_realness_power: float = 1.55

    # Root choice is post-hoc only.
    root_role_anchor_bonus: float = 0.05

    mip_rel_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.identity_count < 1:
            raise ValueError("identity_count must be >= 1")
        if not (
            0.0
            <= self.min_connectivity_support
            <= self.preferred_connectivity_support
            <= 1.0
        ):
            raise ValueError("invalid connectivity support interval")
        if self.max_cluster_size < 1:
            raise ValueError("max_cluster_size must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.junk_realness_power <= 0.0:
            raise ValueError("junk_realness_power must be > 0")
        if self.mip_rel_gap < 0.0:
            raise ValueError("mip_rel_gap must be >= 0")


@dataclass(frozen=True)
class CandidateCluster:
    candidate_id: int
    member_keys: tuple[str, ...]

    utility: float

    mst_edges: tuple[tuple[str, str, float], ...]
    mst_mean_support: float
    mst_min_support: float

    internal_strong_edge_count: int
    soft_identity_penalty: float

    def __post_init__(self) -> None:
        if not self.member_keys:
            raise ValueError("candidate cluster must not be empty")

    @property
    def size(self) -> int:
        return len(self.member_keys)

    def to_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "member_keys": list(self.member_keys),
            "size": self.size,
            "utility": self.utility,
            "mst_edges": [
                {
                    "unit_a": a,
                    "unit_b": b,
                    "support": support,
                }
                for a, b, support in self.mst_edges
            ],
            "mst_mean_support": self.mst_mean_support,
            "mst_min_support": self.mst_min_support,
            "internal_strong_edge_count": self.internal_strong_edge_count,
            "soft_identity_penalty": self.soft_identity_penalty,
        }


@dataclass(frozen=True)
class CanonicalIdentity:
    canonical_id: int
    root_key: str
    member_keys: tuple[str, ...]

    team_label: RelativeTeamLabel

    root_quality: float
    role_anchor: bool

    cluster_utility: float
    mst_mean_support: float
    mst_min_support: float
    soft_identity_penalty: float

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "root_key": self.root_key,
            "member_keys": list(self.member_keys),
            "team_label": self.team_label.value,
            "root_quality": self.root_quality,
            "role_anchor": self.role_anchor,
            "cluster_utility": self.cluster_utility,
            "mst_mean_support": self.mst_mean_support,
            "mst_min_support": self.mst_min_support,
            "soft_identity_penalty": self.soft_identity_penalty,
        }


@dataclass(frozen=True)
class ClusterIdentitySolution:
    identity_count: int
    objective_utility: float

    identities: tuple[CanonicalIdentity, ...]
    junk_keys: tuple[str, ...]

    selected_candidate_ids: tuple[int, ...]
    candidate_count: int

    solver_status: int
    solver_message: str

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_count": self.identity_count,
            "objective_utility": self.objective_utility,
            "identities": [
                identity.to_dict()
                for identity in self.identities
            ],
            "junk_keys": list(self.junk_keys),
            "selected_candidate_ids": list(
                self.selected_candidate_ids
            ),
            "candidate_count": self.candidate_count,
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
        }


def generate_candidate_clusters(
    units: Sequence[IdentityUnit],
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    *,
    config: ClusterSolverConfig | None = None,
) -> tuple[CandidateCluster, ...]:
    cfg = config or ClusterSolverConfig()

    unit_by_key = {unit.key: unit for unit in units}
    if len(unit_by_key) != len(units):
        raise ValueError("unit keys must be unique")

    pairs = _normalize_pairs(
        pair_evidence,
        valid_keys=set(unit_by_key),
    )

    # Positive-support graph used only for cluster connectivity.
    adjacency: dict[str, set[str]] = {
        key: set()
        for key in unit_by_key
    }

    for relation in pairs.values():
        if relation.conflict is not PairConflict.NONE:
            continue
        if (
            relation.fused_support_score
            < cfg.min_connectivity_support
        ):
            continue

        adjacency[relation.unit_a].add(
            relation.unit_b
        )
        adjacency[relation.unit_b].add(
            relation.unit_a
        )

    candidate_members: set[tuple[str, ...]] = {
        (key,)
        for key in unit_by_key
    }

    # Enumerate connected induced subsets by monotonic expansion.  We only
    # expand from the lexicographically smallest member, which removes duplicate
    # generation without requiring all C(n,k) subsets.
    sorted_keys = sorted(unit_by_key)

    for start in sorted_keys:
        frontier = {
            neighbor
            for neighbor in adjacency[start]
            if neighbor > start
        }
        _expand_connected_subset(
            current=(start,),
            frontier=frontier,
            start=start,
            adjacency=adjacency,
            pair_evidence=pairs,
            config=cfg,
            output=candidate_members,
        )

        if len(candidate_members) > cfg.max_candidates:
            raise RuntimeError(
                "candidate cluster generation exceeded "
                f"max_candidates={cfg.max_candidates}"
            )

    candidates = []

    for candidate_id, members in enumerate(
        sorted(
            candidate_members,
            key=lambda item: (
                len(item),
                item,
            ),
        )
    ):
        candidate = _score_candidate(
            candidate_id,
            members,
            unit_by_key=unit_by_key,
            pair_evidence=pairs,
            config=cfg,
        )
        if candidate is not None:
            candidates.append(candidate)

    return tuple(candidates)


def solve_identity_clusters(
    units: Sequence[IdentityUnit],
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    *,
    config: ClusterSolverConfig | None = None,
) -> ClusterIdentitySolution:
    cfg = config or ClusterSolverConfig()

    if not units:
        raise ValueError("solver requires units")
    if cfg.identity_count > len(units):
        raise ValueError("identity_count exceeds unit count")

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    candidates = generate_candidate_clusters(
        units,
        pair_evidence,
        config=cfg,
    )

    candidate_var = {
        candidate.candidate_id: index
        for index, candidate in enumerate(candidates)
    }

    junk_offset = len(candidates)
    junk_var = {
        unit.key: junk_offset + index
        for index, unit in enumerate(units)
    }

    variable_count = len(candidates) + len(units)

    objective = np.zeros(
        variable_count,
        dtype=np.float64,
    )

    for candidate in candidates:
        objective[
            candidate_var[candidate.candidate_id]
        ] = -candidate.utility

    for unit in units:
        junk_utility = _junk_utility(
            unit,
            config=cfg,
        )
        objective[
            junk_var[unit.key]
        ] = -junk_utility

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(
        coefficients: Mapping[int, float],
        lb: float,
        ub: float,
    ) -> None:
        rows.append(dict(coefficients))
        lower.append(float(lb))
        upper.append(float(ub))

    # Each unit is covered by exactly one selected identity cluster OR junk.
    candidates_by_unit: dict[str, list[CandidateCluster]] = {
        unit.key: []
        for unit in units
    }
    for candidate in candidates:
        for member in candidate.member_keys:
            candidates_by_unit[member].append(candidate)

    for unit in units:
        coeff = {
            junk_var[unit.key]: 1.0,
        }
        for candidate in candidates_by_unit[unit.key]:
            coeff[
                candidate_var[candidate.candidate_id]
            ] = 1.0

        add(coeff, 1.0, 1.0)

    # Exactly K canonical identities.
    add(
        {
            candidate_var[candidate.candidate_id]: 1.0
            for candidate in candidates
        },
        cfg.identity_count,
        cfg.identity_count,
    )

    matrix = _rows_to_sparse(
        rows,
        variable_count=variable_count,
    )

    result = milp(
        c=objective,
        integrality=np.ones(
            variable_count,
            dtype=np.int8,
        ),
        bounds=Bounds(
            lb=np.zeros(
                variable_count,
                dtype=np.float64,
            ),
            ub=np.ones(
                variable_count,
                dtype=np.float64,
            ),
        ),
        constraints=LinearConstraint(
            matrix,
            lb=np.asarray(
                lower,
                dtype=np.float64,
            ),
            ub=np.asarray(
                upper,
                dtype=np.float64,
            ),
        ),
        options={
            "mip_rel_gap": cfg.mip_rel_gap,
        },
    )

    if result.x is None or not result.success:
        raise RuntimeError(
            "cluster identity MILP failed: "
            f"status={result.status}, message={result.message}"
        )

    chosen = result.x >= 0.5

    selected = [
        candidate
        for candidate in candidates
        if chosen[
            candidate_var[candidate.candidate_id]
        ]
    ]

    if len(selected) != cfg.identity_count:
        raise RuntimeError(
            f"solver selected {len(selected)} identities, "
            f"expected {cfg.identity_count}"
        )

    junk_keys = tuple(
        sorted(
            unit.key
            for unit in units
            if chosen[
                junk_var[unit.key]
            ]
        )
    )

    identities = _build_canonical_identities(
        selected,
        unit_by_key=unit_by_key,
        config=cfg,
    )

    _validate_partition(
        units=units,
        identities=identities,
        junk_keys=junk_keys,
        pair_evidence=pair_evidence,
        identity_count=cfg.identity_count,
    )

    return ClusterIdentitySolution(
        identity_count=cfg.identity_count,
        objective_utility=float(-result.fun),
        identities=identities,
        junk_keys=junk_keys,
        selected_candidate_ids=tuple(
            sorted(
                candidate.candidate_id
                for candidate in selected
            )
        ),
        candidate_count=len(candidates),
        solver_status=int(result.status),
        solver_message=str(result.message),
    )


def compute_unit_quality(
    *,
    observed_count: int,
    observed_span_s: float,
    appearance_coherence: float,
    appearance_q10: float,
    team_label: RelativeTeamLabel,
    clip_observed_count_reference: int,
    clip_span_s: float,
    role_anchor: bool = False,
) -> float:
    """Clip-aware realness retained from V2.8A-V2."""

    if clip_observed_count_reference < 1:
        raise ValueError(
            "clip_observed_count_reference must be >= 1"
        )
    if clip_span_s <= 0.0:
        raise ValueError("clip_span_s must be > 0")

    count_ratio = min(
        1.0,
        max(
            0.0,
            observed_count
            / clip_observed_count_reference,
        ),
    )
    span_ratio = min(
        1.0,
        max(
            0.0,
            observed_span_s / clip_span_s,
        ),
    )
    joint = math.sqrt(
        count_ratio * span_ratio
    )

    observed_support = (
        0.45 * span_ratio
        + 0.35 * count_ratio
        + 0.20 * joint
    )

    coherence_score = _rescale(
        appearance_coherence,
        0.50,
        0.86,
    )
    q10_score = _rescale(
        appearance_q10,
        0.42,
        0.78,
    )

    team_score = (
        1.0
        if team_label
        in {
            RelativeTeamLabel.TEAM_A,
            RelativeTeamLabel.TEAM_B,
        }
        else 0.65
    )

    value = (
        0.70 * observed_support
        + 0.18 * coherence_score
        + 0.07 * q10_score
        + 0.05 * team_score
    )

    if role_anchor:
        value += 0.02

    return float(
        min(1.0, max(0.0, value))
    )


def _expand_connected_subset(
    *,
    current: tuple[str, ...],
    frontier: set[str],
    start: str,
    adjacency: Mapping[str, set[str]],
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    config: ClusterSolverConfig,
    output: set[tuple[str, ...]],
) -> None:
    if len(current) >= config.max_cluster_size:
        return

    for node in sorted(frontier):
        proposed = tuple(
            sorted(
                (*current, node)
            )
        )

        # Hard/strong conflicts invalidate the entire cluster.
        if _has_internal_conflict(
            proposed,
            pair_evidence=pair_evidence,
        ):
            continue

        output.add(proposed)

        if len(proposed) >= config.max_cluster_size:
            continue

        new_frontier = set(frontier)
        new_frontier.discard(node)

        for member in proposed:
            for neighbor in adjacency[member]:
                if neighbor <= start:
                    continue
                if neighbor in proposed:
                    continue
                new_frontier.add(neighbor)

        _expand_connected_subset(
            current=proposed,
            frontier=new_frontier,
            start=start,
            adjacency=adjacency,
            pair_evidence=pair_evidence,
            config=config,
            output=output,
        )


def _score_candidate(
    candidate_id: int,
    members: tuple[str, ...],
    *,
    unit_by_key: Mapping[str, IdentityUnit],
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    config: ClusterSolverConfig,
) -> CandidateCluster | None:
    if _has_internal_conflict(
        members,
        pair_evidence=pair_evidence,
    ):
        return None

    realness = sum(
        unit_by_key[key].quality
        for key in members
    )

    if len(members) == 1:
        utility = (
            config.preserve_realness_weight
            * realness
        )
        return CandidateCluster(
            candidate_id=candidate_id,
            member_keys=members,
            utility=float(utility),
            mst_edges=(),
            mst_mean_support=1.0,
            mst_min_support=1.0,
            internal_strong_edge_count=0,
            soft_identity_penalty=0.0,
        )

    mst = _maximum_spanning_tree(
        members,
        pair_evidence=pair_evidence,
        min_support=config.min_connectivity_support,
    )
    if mst is None:
        return None

    supports = [
        support
        for _, _, support in mst
    ]

    weak_penalty = sum(
        _weak_connectivity_penalty(
            support,
            config=config,
        )
        for support in supports
    )

    soft_penalty = 0.0
    strong_edge_count = 0

    for left, right in itertools.combinations(
        members,
        2,
    ):
        relation = pair_evidence.get(
            _pair_key(left, right)
        )
        if relation is None:
            continue

        soft_penalty += (
            relation.soft_same_identity_penalty
        )

        if (
            relation.conflict
            is PairConflict.NONE
            and relation.fused_support_score
            >= config.preferred_connectivity_support
        ):
            strong_edge_count += 1

    mst_mean = float(
        np.mean(
            np.asarray(
                supports,
                dtype=np.float64,
            )
        )
    )
    mst_min = float(
        min(supports)
    )

    utility = (
        config.preserve_realness_weight
        * realness
        + config.mst_support_weight
        * sum(supports)
        + config.strong_edge_bonus_weight
        * strong_edge_count
        - config.weak_edge_penalty_weight
        * weak_penalty
        - soft_penalty
        - config.cluster_size_penalty
        * max(
            0,
            len(members) - 1,
        )
    )

    return CandidateCluster(
        candidate_id=candidate_id,
        member_keys=members,
        utility=float(utility),
        mst_edges=tuple(mst),
        mst_mean_support=mst_mean,
        mst_min_support=mst_min,
        internal_strong_edge_count=strong_edge_count,
        soft_identity_penalty=float(soft_penalty),
    )


def _maximum_spanning_tree(
    members: Sequence[str],
    *,
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    min_support: float,
) -> list[tuple[str, str, float]] | None:
    """Kruskal maximum spanning tree over positive compatible edges."""

    if len(members) <= 1:
        return []

    edges = []

    for left, right in itertools.combinations(
        members,
        2,
    ):
        relation = pair_evidence.get(
            _pair_key(left, right)
        )
        if relation is None:
            continue
        if relation.conflict is not PairConflict.NONE:
            continue
        if (
            relation.fused_support_score
            < min_support
        ):
            continue

        edges.append(
            (
                float(
                    relation.fused_support_score
                ),
                left,
                right,
            )
        )

    edges.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        )
    )

    parent = {
        key: key
        for key in members
    }

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(a: str, b: str) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    chosen = []

    for support, left, right in edges:
        if union(left, right):
            chosen.append(
                (left, right, support)
            )
            if len(chosen) == len(members) - 1:
                break

    if len(chosen) != len(members) - 1:
        return None

    return chosen


def _build_canonical_identities(
    selected: Sequence[CandidateCluster],
    *,
    unit_by_key: Mapping[str, IdentityUnit],
    config: ClusterSolverConfig,
) -> tuple[CanonicalIdentity, ...]:
    rows = []

    ordered = sorted(
        selected,
        key=lambda candidate: (
            min(
                unit_by_key[key].start_frame
                for key in candidate.member_keys
            ),
            candidate.member_keys,
        ),
    )

    for canonical_id, candidate in enumerate(
        ordered,
        start=1,
    ):
        root_key = max(
            candidate.member_keys,
            key=lambda key: (
                unit_by_key[key].quality
                + (
                    config.root_role_anchor_bonus
                    if unit_by_key[key].role_anchor
                    else 0.0
                ),
                unit_by_key[key].observed_span_s,
                unit_by_key[key].observed_count,
                -unit_by_key[key].segment_index,
                key,
            ),
        )

        known_teams = {
            unit_by_key[key].team_label
            for key in candidate.member_keys
            if unit_by_key[key].team_label
            in {
                RelativeTeamLabel.TEAM_A,
                RelativeTeamLabel.TEAM_B,
            }
        }

        team_label = (
            next(iter(known_teams))
            if len(known_teams) == 1
            else RelativeTeamLabel.UNKNOWN
        )

        rows.append(
            CanonicalIdentity(
                canonical_id=canonical_id,
                root_key=root_key,
                member_keys=candidate.member_keys,
                team_label=team_label,
                root_quality=unit_by_key[root_key].quality,
                role_anchor=unit_by_key[root_key].role_anchor,
                cluster_utility=candidate.utility,
                mst_mean_support=candidate.mst_mean_support,
                mst_min_support=candidate.mst_min_support,
                soft_identity_penalty=candidate.soft_identity_penalty,
            )
        )

    return tuple(rows)


def _validate_partition(
    *,
    units: Sequence[IdentityUnit],
    identities: Sequence[CanonicalIdentity],
    junk_keys: Sequence[str],
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    identity_count: int,
) -> None:
    if len(identities) != identity_count:
        raise RuntimeError("wrong number of identities")

    seen = set()

    for identity in identities:
        for key in identity.member_keys:
            if key in seen:
                raise RuntimeError(
                    f"unit {key} appears in multiple identities"
                )
            seen.add(key)

        for left, right in itertools.combinations(
            identity.member_keys,
            2,
        ):
            relation = pair_evidence.get(
                _pair_key(left, right)
            )
            if (
                relation is not None
                and relation.conflict
                is not PairConflict.NONE
            ):
                raise RuntimeError(
                    "conflicting pair inside selected identity: "
                    f"{left}, {right}"
                )

    junk = set(junk_keys)

    if seen & junk:
        raise RuntimeError(
            "unit is both selected and junk"
        )

    all_units = {
        unit.key
        for unit in units
    }

    if seen | junk != all_units:
        missing = all_units - (seen | junk)
        raise RuntimeError(
            f"partition does not cover all units: {sorted(missing)}"
        )


def _has_internal_conflict(
    members: Sequence[str],
    *,
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
) -> bool:
    for left, right in itertools.combinations(
        members,
        2,
    ):
        relation = pair_evidence.get(
            _pair_key(left, right)
        )
        if (
            relation is not None
            and relation.conflict
            is not PairConflict.NONE
        ):
            return True

    return False


def _weak_connectivity_penalty(
    support: float,
    *,
    config: ClusterSolverConfig,
) -> float:
    if support >= config.preferred_connectivity_support:
        return 0.0

    width = (
        config.preferred_connectivity_support
        - config.min_connectivity_support
    )
    if width <= 0.0:
        return 0.0

    return float(
        min(
            1.0,
            max(
                0.0,
                (
                    config.preferred_connectivity_support
                    - support
                )
                / width,
            ),
        )
    )


def _junk_utility(
    unit: IdentityUnit,
    *,
    config: ClusterSolverConfig,
) -> float:
    return float(
        config.junk_low_quality_reward
        * (1.0 - unit.quality)
        + config.junk_bias
        - config.junk_realness_penalty
        * (
            unit.quality
            ** config.junk_realness_power
        )
    )


def _normalize_pairs(
    pair_evidence: Mapping[tuple[str, str], UnitPairEvidence],
    *,
    valid_keys: set[str],
) -> dict[tuple[str, str], UnitPairEvidence]:
    normalized = {}

    for relation in pair_evidence.values():
        if (
            relation.unit_a not in valid_keys
            or relation.unit_b not in valid_keys
        ):
            raise ValueError(
                "pair evidence references unknown unit"
            )

        key = relation.pair_key
        if key in normalized:
            raise ValueError(
                f"duplicate pair evidence: {key}"
            )
        normalized[key] = relation

    return normalized


def _rows_to_sparse(
    rows: Sequence[Mapping[int, float]],
    *,
    variable_count: int,
):
    data = []
    row_indices = []
    col_indices = []

    for row_index, coefficients in enumerate(
        rows
    ):
        for col_index, value in coefficients.items():
            if value == 0.0:
                continue
            row_indices.append(row_index)
            col_indices.append(col_index)
            data.append(float(value))

    return coo_matrix(
        (
            data,
            (row_indices, col_indices),
        ),
        shape=(len(rows), variable_count),
        dtype=np.float64,
    ).tocsr()


def _pair_key(
    left: str,
    right: str,
) -> tuple[str, str]:
    return tuple(sorted((left, right)))


def _rescale(
    value: float,
    low: float,
    high: float,
) -> float:
    if high <= low:
        raise ValueError("invalid rescale interval")

    return float(
        min(
            1.0,
            max(
                0.0,
                (value - low) / (high - low),
            ),
        )
    )
