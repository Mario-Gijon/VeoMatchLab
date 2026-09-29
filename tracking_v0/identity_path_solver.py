"""V2.8A-V4: temporal identity paths + set-partition MILP.

This module combines the two evidence families we now have:

  identity evidence:
      ReID + relative team + pair conflicts + visual segmentation

  temporal evidence:
      before/after motion + bidirectional extrapolation + short-gap/overlap
      continuity

A canonical player is represented as an ORDERED PATH of identity units rather
than a bag/cluster or ROOT-star.

Example:
    10:0 -> 10:1 -> 41:0

A path is valid only when:
  * every adjacent temporal handover is plausible,
  * no pair anywhere inside the path has HARD/STRONG conflict,
  * each edge has at least some independent identity support,
  * the full path is selected globally as one of exactly K identities.

The MILP chooses exactly K paths and marks uncovered units as JUNK.  High
realness units are expensive to discard.

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
class IdentityPathSolverConfig:
    identity_count: int = 23

    # Directed edge admission.
    min_temporal_score: float = 0.48
    min_identity_support: float = 0.45
    min_neutral_identity_support: float = 0.67
    neutral_temporal_score: float = 0.50
    min_combined_edge_score: float = 0.58

    temporal_weight: float = 0.62
    identity_weight: float = 0.38

    preferred_edge_score: float = 0.70
    weak_edge_penalty: float = 0.10

    max_path_size: int = 6
    max_candidates: int = 30000

    # Candidate-path objective.
    preserve_realness_weight: float = 0.84
    edge_score_weight: float = 0.52
    nonlocal_identity_bonus_weight: float = 0.08
    path_size_penalty: float = 0.018

    # Junk.
    junk_low_quality_reward: float = 0.14
    junk_bias: float = 0.02
    junk_realness_penalty: float = 0.86
    junk_realness_power: float = 1.55

    root_role_anchor_bonus: float = 0.05

    mip_rel_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.identity_count < 1:
            raise ValueError("identity_count must be >= 1")

        for name in (
            "min_temporal_score",
            "min_identity_support",
            "min_neutral_identity_support",
            "neutral_temporal_score",
            "min_combined_edge_score",
            "temporal_weight",
            "identity_weight",
            "preferred_edge_score",
            "weak_edge_penalty",
            "preserve_realness_weight",
            "edge_score_weight",
            "nonlocal_identity_bonus_weight",
            "path_size_penalty",
            "junk_low_quality_reward",
            "junk_bias",
            "junk_realness_penalty",
            "root_role_anchor_bonus",
            "mip_rel_gap",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")

        if abs(
            self.temporal_weight
            + self.identity_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "temporal_weight + identity_weight must equal 1"
            )

        for name in (
            "min_temporal_score",
            "min_identity_support",
            "min_neutral_identity_support",
            "neutral_temporal_score",
            "min_combined_edge_score",
            "preferred_edge_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.max_path_size < 1:
            raise ValueError("max_path_size must be >= 1")
        if self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1")
        if self.junk_realness_power <= 0.0:
            raise ValueError("junk_realness_power must be > 0")
        if self.mip_rel_gap < 0.0:
            raise ValueError("mip_rel_gap must be >= 0")


@dataclass(frozen=True)
class TemporalIdentityEdge:
    source_key: str
    target_key: str

    temporal_score: float
    identity_support: float
    combined_score: float

    regime: str
    temporal_verdict: str
    gap_s: float
    overlap_s: float

    weak_penalty: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateIdentityPath:
    candidate_id: int
    member_keys: tuple[str, ...]

    utility: float

    edges: tuple[TemporalIdentityEdge, ...]
    mean_edge_score: float
    minimum_edge_score: float

    nonlocal_identity_bonus: float
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
            "edges": [
                edge.to_dict()
                for edge in self.edges
            ],
            "mean_edge_score": self.mean_edge_score,
            "minimum_edge_score": self.minimum_edge_score,
            "nonlocal_identity_bonus": self.nonlocal_identity_bonus,
            "soft_identity_penalty": self.soft_identity_penalty,
        }


@dataclass(frozen=True)
class CanonicalIdentityPath:
    canonical_id: int
    root_key: str
    member_keys: tuple[str, ...]

    team_label: RelativeTeamLabel

    root_quality: float
    role_anchor: bool

    path_utility: float
    mean_edge_score: float
    minimum_edge_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "team_label": self.team_label.value,
            "member_keys": list(self.member_keys),
        }


@dataclass(frozen=True)
class IdentityPathSolution:
    identity_count: int
    objective_utility: float

    identities: tuple[CanonicalIdentityPath, ...]
    junk_keys: tuple[str, ...]

    candidate_count: int
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
            "selected_candidate_ids": list(
                self.selected_candidate_ids
            ),
            "solver_status": self.solver_status,
            "solver_message": self.solver_message,
        }


def build_temporal_identity_edges(
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
    config: IdentityPathSolverConfig | None = None,
) -> dict[
    tuple[str, str],
    TemporalIdentityEdge,
]:
    """Fuse identity and temporal evidence without making time a blind gate.

    Precedence:
      1. HARD/STRONG identity conflict -> reject.
      2. temporal CONTRADICTION -> reject.
      3. temporal SUPPORT -> normal weighted fusion.
      4. temporal NEUTRAL -> allowed only with strong identity evidence.

    This is crucial for interleaved raw-ID handovers. A pair can have strong
    ReID/team/geometry identity evidence while the temporal motion model is
    simply unable to measure a clean before/after transition. NEUTRAL means
    "don't know", not "different person".
    """

    cfg = config or IdentityPathSolverConfig()

    unit_keys = {
        unit.key
        for unit in units
    }

    result = {}

    for transition in transitions.values():
        source = transition.source_key
        target = transition.target_key

        if (
            source not in unit_keys
            or target not in unit_keys
            or source == target
        ):
            continue

        relation = pair_evidence.get(
            _pair_key(
                source,
                target,
            )
        )
        if relation is None:
            continue

        if relation.conflict is not PairConflict.NONE:
            continue

        verdict = getattr(
            transition,
            "verdict",
            "neutral",
        )

        if verdict == "contradiction":
            continue

        if verdict == "support":
            if not transition.valid:
                continue
            if (
                transition.temporal_score
                < cfg.min_temporal_score
            ):
                continue
            if (
                relation.fused_support_score
                < cfg.min_identity_support
            ):
                continue

            temporal_component = (
                transition.temporal_score
            )

        elif verdict == "neutral":
            if (
                relation.fused_support_score
                < cfg.min_neutral_identity_support
            ):
                continue

            # Neutral is deliberately mapped to a non-punitive midpoint.
            temporal_component = (
                cfg.neutral_temporal_score
            )

        else:
            raise ValueError(
                f"unknown temporal verdict: {verdict}"
            )

        combined = (
            cfg.temporal_weight
            * temporal_component
            + cfg.identity_weight
            * relation.fused_support_score
        )

        if combined < cfg.min_combined_edge_score:
            continue

        weak_penalty = _weak_edge_penalty(
            combined,
            config=cfg,
        )

        key = (
            source,
            target,
        )

        edge = TemporalIdentityEdge(
            source_key=source,
            target_key=target,
            temporal_score=float(
                temporal_component
            ),
            identity_support=float(
                relation.fused_support_score
            ),
            combined_score=float(
                combined
            ),
            regime=transition.regime,
            temporal_verdict=verdict,
            gap_s=float(
                transition.gap_s
            ),
            overlap_s=float(
                transition.overlap_s
            ),
            weak_penalty=float(
                weak_penalty
            ),
        )

        current = result.get(key)

        if (
            current is None
            or edge.combined_score
            > current.combined_score
        ):
            result[key] = edge

    return result

def generate_candidate_identity_paths(
    units: Sequence[IdentityUnit],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    edges: Mapping[
        tuple[str, str],
        TemporalIdentityEdge,
    ],
    *,
    config: IdentityPathSolverConfig | None = None,
) -> tuple[CandidateIdentityPath, ...]:
    cfg = config or IdentityPathSolverConfig()

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    adjacency: dict[
        str,
        list[TemporalIdentityEdge],
    ] = {
        key: []
        for key in unit_by_key
    }

    for edge in edges.values():
        if (
            edge.source_key in adjacency
            and edge.target_key in unit_by_key
        ):
            adjacency[
                edge.source_key
            ].append(edge)

    for rows in adjacency.values():
        rows.sort(
            key=lambda edge: (
                -edge.combined_score,
                edge.target_key,
            )
        )

    path_members: set[
        tuple[str, ...]
    ] = {
        (key,)
        for key in unit_by_key
    }

    for start in sorted(unit_by_key):
        _expand_path(
            current=(start,),
            adjacency=adjacency,
            pair_evidence=pair_evidence,
            config=cfg,
            output=path_members,
        )

        if len(path_members) > cfg.max_candidates:
            raise RuntimeError(
                "candidate path generation exceeded "
                f"max_candidates={cfg.max_candidates}"
            )

    candidates = []

    for candidate_id, members in enumerate(
        sorted(
            path_members,
            key=lambda item: (
                len(item),
                item,
            ),
        )
    ):
        candidate = _score_path(
            candidate_id,
            members,
            unit_by_key=unit_by_key,
            pair_evidence=pair_evidence,
            edges=edges,
            config=cfg,
        )
        if candidate is not None:
            candidates.append(
                candidate
            )

    return tuple(candidates)


def solve_identity_paths(
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
    config: IdentityPathSolverConfig | None = None,
) -> IdentityPathSolution:
    cfg = config or IdentityPathSolverConfig()

    if not units:
        raise ValueError(
            "identity path solver requires units"
        )
    if cfg.identity_count > len(units):
        raise ValueError(
            "identity_count exceeds unit count"
        )

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    edges = build_temporal_identity_edges(
        units,
        pair_evidence,
        transitions,
        config=cfg,
    )

    candidates = generate_candidate_identity_paths(
        units,
        pair_evidence,
        edges,
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
        for index, unit in enumerate(units)
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
            junk_var[unit.key]
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
            dict(coefficients)
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
            ].append(candidate)

    # Exactly one selected path or JUNK per unit.
    for unit in units:
        coeff = {
            junk_var[unit.key]: 1.0,
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

    # Exactly K identities.
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
            "identity path MILP failed: "
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

    identities = _build_canonical_paths(
        selected,
        unit_by_key=unit_by_key,
        config=cfg,
    )

    _validate_solution(
        units=units,
        identities=identities,
        junk_keys=junk_keys,
        pair_evidence=pair_evidence,
        identity_count=cfg.identity_count,
    )

    return IdentityPathSolution(
        identity_count=cfg.identity_count,
        objective_utility=float(
            -result.fun
        ),
        identities=identities,
        junk_keys=junk_keys,
        candidate_count=len(
            candidates
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


def _expand_path(
    *,
    current: tuple[str, ...],
    adjacency: Mapping[
        str,
        Sequence[TemporalIdentityEdge],
    ],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    config: IdentityPathSolverConfig,
    output: set[tuple[str, ...]],
) -> None:
    if len(current) >= config.max_path_size:
        return

    tail = current[-1]

    for edge in adjacency.get(
        tail,
        (),
    ):
        target = edge.target_key

        if target in current:
            continue

        proposed = (
            *current,
            target,
        )

        if _has_internal_conflict(
            proposed,
            pair_evidence=pair_evidence,
        ):
            continue

        output.add(
            proposed
        )

        _expand_path(
            current=proposed,
            adjacency=adjacency,
            pair_evidence=pair_evidence,
            config=config,
            output=output,
        )


def _score_path(
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
    edges: Mapping[
        tuple[str, str],
        TemporalIdentityEdge,
    ],
    config: IdentityPathSolverConfig,
) -> CandidateIdentityPath | None:
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
        return CandidateIdentityPath(
            candidate_id=candidate_id,
            member_keys=members,
            utility=float(
                config.preserve_realness_weight
                * realness
            ),
            edges=(),
            mean_edge_score=1.0,
            minimum_edge_score=1.0,
            nonlocal_identity_bonus=0.0,
            soft_identity_penalty=0.0,
        )

    path_edges = []

    for source, target in zip(
        members,
        members[1:],
    ):
        edge = edges.get(
            (
                source,
                target,
            )
        )
        if edge is None:
            return None

        path_edges.append(
            edge
        )

    edge_scores = [
        edge.combined_score
        for edge in path_edges
    ]

    weak_penalty = sum(
        edge.weak_penalty
        for edge in path_edges
    )

    soft_penalty = 0.0
    nonlocal_bonus = 0.0

    adjacent_pairs = {
        _pair_key(
            edge.source_key,
            edge.target_key,
        )
        for edge in path_edges
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

        if (
            _pair_key(
                left,
                right,
            )
            not in adjacent_pairs
            and relation.conflict
            is PairConflict.NONE
            and relation.fused_support_score
            >= 0.67
        ):
            nonlocal_bonus += (
                relation.fused_support_score
                - 0.67
            )

    utility = (
        config.preserve_realness_weight
        * realness
        + config.edge_score_weight
        * sum(
            edge_scores
        )
        + config.nonlocal_identity_bonus_weight
        * nonlocal_bonus
        - weak_penalty
        - soft_penalty
        - config.path_size_penalty
        * (
            len(members)
            - 1
        )
    )

    return CandidateIdentityPath(
        candidate_id=candidate_id,
        member_keys=members,
        utility=float(
            utility
        ),
        edges=tuple(
            path_edges
        ),
        mean_edge_score=float(
            np.mean(
                np.asarray(
                    edge_scores,
                    dtype=np.float64,
                )
            )
        ),
        minimum_edge_score=float(
            min(
                edge_scores
            )
        ),
        nonlocal_identity_bonus=float(
            nonlocal_bonus
        ),
        soft_identity_penalty=float(
            soft_penalty
        ),
    )


def _build_canonical_paths(
    selected: Sequence[
        CandidateIdentityPath
    ],
    *,
    unit_by_key: Mapping[
        str,
        IdentityUnit,
    ],
    config: IdentityPathSolverConfig,
) -> tuple[
    CanonicalIdentityPath,
    ...,
]:
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

    identities = []

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
            next(
                iter(
                    known_teams
                )
            )
            if len(known_teams) == 1
            else RelativeTeamLabel.UNKNOWN
        )

        identities.append(
            CanonicalIdentityPath(
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
                path_utility=candidate.utility,
                mean_edge_score=candidate.mean_edge_score,
                minimum_edge_score=candidate.minimum_edge_score,
            )
        )

    return tuple(
        identities
    )


def _validate_solution(
    *,
    units: Sequence[IdentityUnit],
    identities: Sequence[
        CanonicalIdentityPath
    ],
    junk_keys: Sequence[str],
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
    ],
    identity_count: int,
) -> None:
    if len(identities) != identity_count:
        raise RuntimeError(
            "wrong number of identity paths"
        )

    seen = set()

    for identity in identities:
        for key in identity.member_keys:
            if key in seen:
                raise RuntimeError(
                    f"unit {key} appears in multiple paths"
                )
            seen.add(
                key
            )

        if _has_internal_conflict(
            identity.member_keys,
            pair_evidence=pair_evidence,
        ):
            raise RuntimeError(
                "selected path contains identity conflict"
            )

    junk = set(
        junk_keys
    )

    if seen & junk:
        raise RuntimeError(
            "unit is both selected and junk"
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
            "identity path partition does not cover all units"
        )


def _has_internal_conflict(
    members: Sequence[str],
    *,
    pair_evidence: Mapping[
        tuple[str, str],
        UnitPairEvidence,
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

    return False


def _weak_edge_penalty(
    score: float,
    *,
    config: IdentityPathSolverConfig,
) -> float:
    if score >= config.preferred_edge_score:
        return 0.0

    width = (
        config.preferred_edge_score
        - config.min_combined_edge_score
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
    config: IdentityPathSolverConfig,
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
