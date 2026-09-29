"""V2.8A-V5: hybrid identity graph solver.

This is the final conceptual fusion of the two evidence families developed so
far:

    identity evidence
        ReID + team + same-frame conflict + appearance segmentation

    temporal evidence
        before/after motion + overlap continuity

A canonical player is NOT forced to be a ROOT-star and is NOT forced to be a
strict linear path.

Why a graph?
------------
A real tracker can duplicate one physical player into two raw IDs for a short
period.  It can also create an ID switch where one raw ID branches into a later
fragment while another duplicate still exists.  Therefore the latent physical
identity may look like:

        A ---- B
         \
          C

even though the person is of course one physical trajectory.

The correct rule is:

  * HARD/STRONG identity conflict -> units can never share a canonical identity.
  * temporal CONTRADICTION -> units can never share a canonical identity.
  * temporal SUPPORT -> positive evidence.
  * temporal NEUTRAL -> does not punish strong identity evidence.
  * a multi-unit identity must be connected by admissible hybrid evidence.

The solver enumerates small connected candidate identity graphs and uses a
set-partition MILP to choose exactly K canonical identities.  Uncovered units
are JUNK.

Audit-only: no tracking frame rewrite.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
import math
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .global_identity_solver import (
    IdentityUnit,
    PairConflict,
    UnitPairEvidence,
)
from .team_similarity import RelativeTeamLabel
from .temporal_identity import TemporalTransitionEvidence


@dataclass(frozen=True)
class IdentityGraphSolverConfig:
    identity_count: int = 23

    # Edge admission.
    min_support_identity: float = 0.45
    min_neutral_identity: float = 0.65
    min_hybrid_edge_score: float = 0.58

    support_temporal_weight: float = 0.62
    support_identity_weight: float = 0.38

    neutral_temporal_weight: float = 0.35
    neutral_identity_weight: float = 0.65
    neutral_temporal_score: float = 0.50

    preferred_edge_score: float = 0.70
    weak_edge_penalty: float = 0.10

    max_cluster_size: int = 6
    max_candidates: int = 30000

    # Cluster utility.
    preserve_realness_weight: float = 0.82
    mst_edge_weight: float = 0.48
    extra_strong_pair_bonus_weight: float = 0.08
    cluster_size_penalty: float = 0.018

    # Junk.
    junk_low_quality_reward: float = 0.14
    junk_bias: float = 0.02
    junk_realness_penalty: float = 0.88
    junk_realness_power: float = 1.55

    root_role_anchor_bonus: float = 0.05

    mip_rel_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.identity_count < 1:
            raise ValueError("identity_count must be >= 1")

        for name in (
            "min_support_identity",
            "min_neutral_identity",
            "min_hybrid_edge_score",
            "support_temporal_weight",
            "support_identity_weight",
            "neutral_temporal_weight",
            "neutral_identity_weight",
            "neutral_temporal_score",
            "preferred_edge_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if abs(
            self.support_temporal_weight
            + self.support_identity_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "support temporal + identity weights must equal 1"
            )

        if abs(
            self.neutral_temporal_weight
            + self.neutral_identity_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "neutral temporal + identity weights must equal 1"
            )

        if self.max_cluster_size < 1:
            raise ValueError("max_cluster_size must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.junk_realness_power <= 0.0:
            raise ValueError("junk_realness_power must be > 0")
        if self.mip_rel_gap < 0.0:
            raise ValueError("mip_rel_gap must be >= 0")


@dataclass(frozen=True)
class HybridIdentityEdge:
    unit_a: str
    unit_b: str

    identity_support: float
    temporal_score: float
    temporal_verdict: str

    hybrid_score: float
    weak_penalty: float

    reason: str

    @property
    def pair_key(self) -> tuple[str, str]:
        return _pair_key(
            self.unit_a,
            self.unit_b,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateIdentityGraph:
    candidate_id: int
    member_keys: tuple[str, ...]

    utility: float

    mst_edges: tuple[HybridIdentityEdge, ...]
    mst_min_score: float
    mst_mean_score: float

    extra_strong_pair_bonus: float
    soft_identity_penalty: float

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
                edge.to_dict()
                for edge in self.mst_edges
            ],
            "mst_min_score": self.mst_min_score,
            "mst_mean_score": self.mst_mean_score,
            "extra_strong_pair_bonus": self.extra_strong_pair_bonus,
            "soft_identity_penalty": self.soft_identity_penalty,
        }


@dataclass(frozen=True)
class CanonicalIdentityGraph:
    canonical_id: int
    root_key: str
    member_keys: tuple[str, ...]

    team_label: RelativeTeamLabel

    root_quality: float
    role_anchor: bool

    graph_utility: float
    mst_min_score: float
    mst_mean_score: float
    soft_identity_penalty: float

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "root_key": self.root_key,
            "member_keys": list(self.member_keys),
            "team_label": self.team_label.value,
            "root_quality": self.root_quality,
            "role_anchor": self.role_anchor,
            "graph_utility": self.graph_utility,
            "mst_min_score": self.mst_min_score,
            "mst_mean_score": self.mst_mean_score,
            "soft_identity_penalty": self.soft_identity_penalty,
        }


@dataclass(frozen=True)
class IdentityGraphSolution:
    identity_count: int
    objective_utility: float

    identities: tuple[CanonicalIdentityGraph, ...]
    junk_keys: tuple[str, ...]

    candidate_count: int
    hybrid_edge_count: int
    selected_candidate_ids: tuple[int, ...]

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
            "candidate_count": self.candidate_count,
            "hybrid_edge_count": self.hybrid_edge_count,
            "selected_candidate_ids": list(
                self.selected_candidate_ids
            ),
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
        }


def build_hybrid_identity_edges(
    units: Sequence[IdentityUnit],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
    *,
    config: IdentityGraphSolverConfig | None = None,
) -> dict[
    tuple[str, str],
    HybridIdentityEdge,
]:
    cfg = config or IdentityGraphSolverConfig()

    unit_keys = {
        unit.key
        for unit in units
    }

    result = {}

    for relation in pair_evidence.values():
        a = relation.unit_a
        b = relation.unit_b

        if a not in unit_keys or b not in unit_keys:
            continue

        if relation.conflict is not PairConflict.NONE:
            continue

        transition = _find_transition(
            a,
            b,
            transitions=transitions,
        )

        verdict = (
            getattr(
                transition,
                "verdict",
                "neutral",
            )
            if transition is not None
            else "neutral"
        )

        if verdict == "contradiction":
            continue

        if verdict == "support":
            if (
                relation.fused_support_score
                < cfg.min_support_identity
            ):
                continue

            temporal_score = float(
                transition.temporal_score
            )

            hybrid_score = (
                cfg.support_temporal_weight
                * temporal_score
                + cfg.support_identity_weight
                * relation.fused_support_score
            )

            reason = "temporal_support_plus_identity"

        elif verdict == "neutral":
            if (
                relation.fused_support_score
                < cfg.min_neutral_identity
            ):
                continue

            temporal_score = (
                cfg.neutral_temporal_score
            )

            hybrid_score = (
                cfg.neutral_temporal_weight
                * temporal_score
                + cfg.neutral_identity_weight
                * relation.fused_support_score
            )

            reason = "identity_supported_temporal_neutral"

        else:
            raise ValueError(
                f"unknown temporal verdict {verdict}"
            )

        if hybrid_score < cfg.min_hybrid_edge_score:
            continue

        edge = HybridIdentityEdge(
            unit_a=a,
            unit_b=b,
            identity_support=float(
                relation.fused_support_score
            ),
            temporal_score=float(
                temporal_score
            ),
            temporal_verdict=verdict,
            hybrid_score=float(
                hybrid_score
            ),
            weak_penalty=_weak_edge_penalty(
                hybrid_score,
                config=cfg,
            ),
            reason=reason,
        )

        result[
            edge.pair_key
        ] = edge

    return result


def generate_candidate_identity_graphs(
    units: Sequence[IdentityUnit],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
    *,
    eligible_anchor_keys: set[str] | frozenset[str] | None = None,
    config: IdentityGraphSolverConfig | None = None,
) -> tuple[CandidateIdentityGraph, ...]:
    cfg = config or IdentityGraphSolverConfig()

    eligible_anchors = (
        frozenset(
            unit.key
            for unit in units
        )
        if eligible_anchor_keys is None
        else frozenset(
            eligible_anchor_keys
        )
    )

    unknown_anchors = (
        eligible_anchors
        - {
            unit.key
            for unit in units
        }
    )
    if unknown_anchors:
        raise ValueError(
            "eligible_anchor_keys contains unknown units: "
            + ", ".join(
                sorted(
                    unknown_anchors
                )
            )
        )

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    edges = build_hybrid_identity_edges(
        units,
        pair_evidence,
        transitions,
        config=cfg,
    )

    adjacency = {
        key: set()
        for key in unit_by_key
    }

    for edge in edges.values():
        adjacency[
            edge.unit_a
        ].add(edge.unit_b)
        adjacency[
            edge.unit_b
        ].add(edge.unit_a)

    members_set: set[
        tuple[str, ...]
    ] = {
        (key,)
        for key in unit_by_key
    }

    for start in sorted(
        unit_by_key
    ):
        frontier = {
            neighbor
            for neighbor in adjacency[
                start
            ]
            if neighbor > start
        }

        _expand_connected_graph(
            current=(start,),
            frontier=frontier,
            start=start,
            adjacency=adjacency,
            pair_evidence=pair_evidence,
            transitions=transitions,
            config=cfg,
            output=members_set,
        )

        if len(members_set) > cfg.max_candidates:
            raise RuntimeError(
                "candidate graph generation exceeded "
                f"max_candidates={cfg.max_candidates}"
            )

    candidates = []

    for candidate_id, members in enumerate(
        sorted(
            members_set,
            key=lambda item: (
                len(item),
                item,
            ),
        )
    ):
        candidate = _score_candidate_graph(
            candidate_id,
            members,
            unit_by_key=unit_by_key,
            pair_evidence=pair_evidence,
            transitions=transitions,
            edges=edges,
            eligible_anchor_keys=eligible_anchors,
            config=cfg,
        )

        if candidate is not None:
            candidates.append(
                candidate
            )

    return tuple(candidates)


def solve_identity_graphs(
    units: Sequence[IdentityUnit],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
    *,
    eligible_anchor_keys: set[str] | frozenset[str] | None = None,
    config: IdentityGraphSolverConfig | None = None,
) -> IdentityGraphSolution:
    cfg = config or IdentityGraphSolverConfig()

    eligible_anchors = (
        frozenset(
            unit.key
            for unit in units
        )
        if eligible_anchor_keys is None
        else frozenset(
            eligible_anchor_keys
        )
    )

    if not units:
        raise ValueError(
            "identity graph solver requires units"
        )
    if cfg.identity_count > len(units):
        raise ValueError(
            "identity_count exceeds unit count"
        )

    unknown_anchors = (
        eligible_anchors
        - {
            unit.key
            for unit in units
        }
    )
    if unknown_anchors:
        raise ValueError(
            "eligible_anchor_keys contains unknown units: "
            + ", ".join(
                sorted(
                    unknown_anchors
                )
            )
        )

    if len(
        eligible_anchors
    ) < cfg.identity_count:
        raise ValueError(
            "not enough inside-eligible anchor units to build "
            f"{cfg.identity_count} canonical identities"
        )

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    edges = build_hybrid_identity_edges(
        units,
        pair_evidence,
        transitions,
        config=cfg,
    )

    candidates = generate_candidate_identity_graphs(
        units,
        pair_evidence,
        transitions,
        eligible_anchor_keys=eligible_anchors,
        config=cfg,
    )

    candidate_var = {
        candidate.candidate_id: index
        for index, candidate in enumerate(
            candidates
        )
    }

    junk_offset = len(candidates)

    junk_var = {
        unit.key: junk_offset + index
        for index, unit in enumerate(
            units
        )
    }

    variable_count = (
        len(candidates)
        + len(units)
    )

    objective = np.zeros(
        variable_count,
        dtype=np.float64,
    )

    for candidate in candidates:
        objective[
            candidate_var[
                candidate.candidate_id
            ]
        ] = -candidate.utility

    for unit in units:
        objective[
            junk_var[
                unit.key
            ]
        ] = -_junk_utility(
            unit,
            config=cfg,
        )

    rows = []
    lower = []
    upper = []

    def add(
        coefficients,
        lb,
        ub,
    ):
        rows.append(
            dict(
                coefficients
            )
        )
        lower.append(
            float(lb)
        )
        upper.append(
            float(ub)
        )

    candidates_by_unit = {
        unit.key: []
        for unit in units
    }

    for candidate in candidates:
        for member in candidate.member_keys:
            candidates_by_unit[
                member
            ].append(
                candidate
            )

    # Every unit is exactly one selected identity graph OR junk.
    for unit in units:
        coeff = {
            junk_var[
                unit.key
            ]: 1.0,
        }

        for candidate in candidates_by_unit[
            unit.key
        ]:
            coeff[
                candidate_var[
                    candidate.candidate_id
                ]
            ] = 1.0

        add(
            coeff,
            1.0,
            1.0,
        )

    # Fixed short-clip roster.
    add(
        {
            candidate_var[
                candidate.candidate_id
            ]: 1.0
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

    if (
        result.x is None
        or not result.success
    ):
        raise RuntimeError(
            "identity graph MILP failed: "
            f"status={result.status}, "
            f"message={result.message}"
        )

    chosen = (
        result.x >= 0.5
    )

    selected = [
        candidate
        for candidate in candidates
        if chosen[
            candidate_var[
                candidate.candidate_id
            ]
        ]
    ]

    junk_keys = tuple(
        sorted(
            unit.key
            for unit in units
            if chosen[
                junk_var[
                    unit.key
                ]
            ]
        )
    )

    identities = _build_canonical_graphs(
        selected,
        unit_by_key=unit_by_key,
        config=cfg,
    )

    _validate_solution(
        units=units,
        identities=identities,
        junk_keys=junk_keys,
        pair_evidence=pair_evidence,
        transitions=transitions,
        identity_count=cfg.identity_count,
    )

    return IdentityGraphSolution(
        identity_count=cfg.identity_count,
        objective_utility=float(
            -result.fun
        ),
        identities=identities,
        junk_keys=junk_keys,
        candidate_count=len(
            candidates
        ),
        hybrid_edge_count=len(
            edges
        ),
        selected_candidate_ids=tuple(
            sorted(
                candidate.candidate_id
                for candidate in selected
            )
        ),
        solver_status=int(
            result.status
        ),
        solver_message=str(
            result.message
        ),
    )


def _expand_connected_graph(
    *,
    current: tuple[str, ...],
    frontier: set[str],
    start: str,
    adjacency: Mapping[
        str,
        set[str],
    ],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
    config: IdentityGraphSolverConfig,
    output: set[tuple[str, ...]],
) -> None:
    if len(current) >= config.max_cluster_size:
        return

    for node in sorted(
        frontier
    ):
        proposed = tuple(
            sorted(
                (
                    *current,
                    node,
                )
            )
        )

        if _has_internal_veto(
            proposed,
            pair_evidence=pair_evidence,
            transitions=transitions,
        ):
            continue

        output.add(
            proposed
        )

        if len(proposed) >= config.max_cluster_size:
            continue

        new_frontier = set(
            frontier
        )
        new_frontier.discard(
            node
        )

        for member in proposed:
            for neighbor in adjacency[
                member
            ]:
                if neighbor <= start:
                    continue
                if neighbor in proposed:
                    continue
                new_frontier.add(
                    neighbor
                )

        _expand_connected_graph(
            current=proposed,
            frontier=new_frontier,
            start=start,
            adjacency=adjacency,
            pair_evidence=pair_evidence,
            transitions=transitions,
            config=config,
            output=output,
        )


def _score_candidate_graph(
    candidate_id: int,
    members: tuple[str, ...],
    *,
    unit_by_key: Mapping[
        str,
        IdentityUnit,
    ],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
    edges: Mapping[
        tuple[str, str],
        HybridIdentityEdge,
    ],
    eligible_anchor_keys: frozenset[str],
    config: IdentityGraphSolverConfig,
) -> CandidateIdentityGraph | None:
    if _has_internal_veto(
        members,
        pair_evidence=pair_evidence,
        transitions=transitions,
    ):
        return None

    # An outside-only person/fragment can never ESTABLISH a canonical roster
    # identity.  It may still be attached to an identity graph when at least
    # one other member has genuine OBSERVED history inside the strict pitch.
    if not any(
        key in eligible_anchor_keys
        for key in members
    ):
        return None

    realness = sum(
        unit_by_key[
            key
        ].quality
        for key in members
    )

    if len(members) == 1:
        return CandidateIdentityGraph(
            candidate_id=candidate_id,
            member_keys=members,
            utility=float(
                config.preserve_realness_weight
                * realness
            ),
            mst_edges=(),
            mst_min_score=1.0,
            mst_mean_score=1.0,
            extra_strong_pair_bonus=0.0,
            soft_identity_penalty=0.0,
        )

    mst = _maximum_spanning_tree(
        members,
        edges=edges,
    )

    if mst is None:
        return None

    mst_scores = [
        edge.hybrid_score
        for edge in mst
    ]

    weak_penalty = sum(
        edge.weak_penalty
        for edge in mst
    )

    soft_penalty = 0.0
    extra_bonus = 0.0

    mst_pairs = {
        edge.pair_key
        for edge in mst
    }

    for left, right in itertools.combinations(
        members,
        2,
    ):
        relation = pair_evidence.get(
            _pair_key(
                left,
                right,
            )
        )

        if relation is None:
            continue

        soft_penalty += (
            relation.soft_same_identity_penalty
        )

        pair_key = _pair_key(
            left,
            right,
        )

        edge = edges.get(
            pair_key
        )

        if (
            edge is not None
            and pair_key not in mst_pairs
            and edge.hybrid_score
            >= config.preferred_edge_score
        ):
            extra_bonus += (
                edge.hybrid_score
                - config.preferred_edge_score
            )

    utility = (
        config.preserve_realness_weight
        * realness
        + config.mst_edge_weight
        * sum(
            mst_scores
        )
        + config.extra_strong_pair_bonus_weight
        * extra_bonus
        - weak_penalty
        - soft_penalty
        - config.cluster_size_penalty
        * (
            len(members)
            - 1
        )
    )

    return CandidateIdentityGraph(
        candidate_id=candidate_id,
        member_keys=members,
        utility=float(
            utility
        ),
        mst_edges=tuple(
            mst
        ),
        mst_min_score=float(
            min(
                mst_scores
            )
        ),
        mst_mean_score=float(
            np.mean(
                np.asarray(
                    mst_scores,
                    dtype=np.float64,
                )
            )
        ),
        extra_strong_pair_bonus=float(
            extra_bonus
        ),
        soft_identity_penalty=float(
            soft_penalty
        ),
    )


def _maximum_spanning_tree(
    members: Sequence[str],
    *,
    edges: Mapping[
        tuple[str, str],
        HybridIdentityEdge,
    ],
) -> list[
    HybridIdentityEdge
] | None:
    rows = []

    for left, right in itertools.combinations(
        members,
        2,
    ):
        edge = edges.get(
            _pair_key(
                left,
                right,
            )
        )

        if edge is not None:
            rows.append(
                edge
            )

    rows.sort(
        key=lambda edge: (
            -edge.hybrid_score,
            edge.unit_a,
            edge.unit_b,
        )
    )

    parent = {
        key: key
        for key in members
    }

    def find(key):
        while parent[key] != key:
            parent[key] = parent[
                parent[key]
            ]
            key = parent[key]
        return key

    def union(a, b):
        ra = find(a)
        rb = find(b)

        if ra == rb:
            return False

        parent[rb] = ra
        return True

    chosen = []

    for edge in rows:
        if union(
            edge.unit_a,
            edge.unit_b,
        ):
            chosen.append(
                edge
            )

            if len(chosen) == len(
                members
            ) - 1:
                break

    if len(chosen) != len(
        members
    ) - 1:
        return None

    return chosen


def _build_canonical_graphs(
    selected: Sequence[
        CandidateIdentityGraph
    ],
    *,
    unit_by_key: Mapping[
        str,
        IdentityUnit,
    ],
    config: IdentityGraphSolverConfig,
) -> tuple[
    CanonicalIdentityGraph,
    ...,
]:
    ordered = sorted(
        selected,
        key=lambda candidate: (
            min(
                unit_by_key[
                    key
                ].start_frame
                for key in candidate.member_keys
            ),
            candidate.member_keys,
        ),
    )

    identities = []

    for canonical_id, candidate in enumerate(
        ordered,
        start=1,
    ):
        root_key = max(
            candidate.member_keys,
            key=lambda key: (
                unit_by_key[
                    key
                ].quality
                + (
                    config.root_role_anchor_bonus
                    if unit_by_key[
                        key
                    ].role_anchor
                    else 0.0
                ),
                unit_by_key[
                    key
                ].observed_span_s,
                unit_by_key[
                    key
                ].observed_count,
                key,
            ),
        )

        known_teams = {
            unit_by_key[
                key
            ].team_label
            for key in candidate.member_keys
            if unit_by_key[
                key
            ].team_label
            in {
                RelativeTeamLabel.TEAM_A,
                RelativeTeamLabel.TEAM_B,
            }
        }

        team_label = (
            next(
                iter(
                    known_teams
                )
            )
            if len(
                known_teams
            ) == 1
            else RelativeTeamLabel.UNKNOWN
        )

        identities.append(
            CanonicalIdentityGraph(
                canonical_id=canonical_id,
                root_key=root_key,
                member_keys=candidate.member_keys,
                team_label=team_label,
                root_quality=unit_by_key[
                    root_key
                ].quality,
                role_anchor=unit_by_key[
                    root_key
                ].role_anchor,
                graph_utility=candidate.utility,
                mst_min_score=candidate.mst_min_score,
                mst_mean_score=candidate.mst_mean_score,
                soft_identity_penalty=(
                    candidate.soft_identity_penalty
                ),
            )
        )

    return tuple(
        identities
    )


def _validate_solution(
    *,
    units: Sequence[IdentityUnit],
    identities: Sequence[
        CanonicalIdentityGraph
    ],
    junk_keys: Sequence[str],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
    identity_count: int,
) -> None:
    if len(identities) != identity_count:
        raise RuntimeError(
            "wrong number of canonical identities"
        )

    seen = set()

    for identity in identities:
        if _has_internal_veto(
            identity.member_keys,
            pair_evidence=pair_evidence,
            transitions=transitions,
        ):
            raise RuntimeError(
                "selected identity contains a veto pair"
            )

        for key in identity.member_keys:
            if key in seen:
                raise RuntimeError(
                    f"unit {key} appears in multiple identities"
                )
            seen.add(
                key
            )

    junk = set(
        junk_keys
    )

    if seen & junk:
        raise RuntimeError(
            "unit is both canonical and junk"
        )

    all_units = {
        unit.key
        for unit in units
    }

    if (
        seen | junk
        != all_units
    ):
        raise RuntimeError(
            "identity graph partition does not cover all units"
        )


def _has_internal_veto(
    members: Sequence[str],
    *,
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
) -> bool:
    for left, right in itertools.combinations(
        members,
        2,
    ):
        relation = pair_evidence.get(
            _pair_key(
                left,
                right,
            )
        )

        if (
            relation is not None
            and relation.conflict
            is not PairConflict.NONE
        ):
            return True

        transition = _find_transition(
            left,
            right,
            transitions=transitions,
        )

        if (
            transition is not None
            and getattr(
                transition,
                "verdict",
                "neutral",
            )
            == "contradiction"
        ):
            return True

    return False


def _find_transition(
    left: str,
    right: str,
    *,
    transitions: Mapping[
        tuple[str, str],
        TemporalTransitionEvidence,
    ],
) -> TemporalTransitionEvidence | None:
    return (
        transitions.get(
            (
                left,
                right,
            )
        )
        or transitions.get(
            (
                right,
                left,
            )
        )
    )


def _weak_edge_penalty(
    score: float,
    *,
    config: IdentityGraphSolverConfig,
) -> float:
    if score >= config.preferred_edge_score:
        return 0.0

    width = (
        config.preferred_edge_score
        - config.min_hybrid_edge_score
    )

    if width <= 0.0:
        return 0.0

    severity = (
        config.preferred_edge_score
        - score
    ) / width

    return float(
        config.weak_edge_penalty
        * min(
            1.0,
            max(
                0.0,
                severity,
            ),
        )
    )


def _junk_utility(
    unit: IdentityUnit,
    *,
    config: IdentityGraphSolverConfig,
) -> float:
    return float(
        config.junk_low_quality_reward
        * (
            1.0
            - unit.quality
        )
        + config.junk_bias
        - config.junk_realness_penalty
        * (
            unit.quality
            ** config.junk_realness_power
        )
    )


def _rows_to_sparse(
    rows,
    *,
    variable_count,
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

            row_indices.append(
                row_index
            )
            col_indices.append(
                col_index
            )
            data.append(
                float(
                    value
                )
            )

    return coo_matrix(
        (
            data,
            (
                row_indices,
                col_indices,
            ),
        ),
        shape=(
            len(rows),
            variable_count,
        ),
        dtype=np.float64,
    ).tocsr()


def _pair_key(
    left: str,
    right: str,
) -> tuple[str, str]:
    return tuple(
        sorted(
            (
                left,
                right,
            )
        )
    )
