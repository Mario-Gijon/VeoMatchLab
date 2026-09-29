"""Counterfactual diagnostics for the V2.8A-V5 identity-graph solver.

This module does not replace the production/audit solver.  It answers questions
such as:

    "What would the globally optimal 23-identity solution look like if 11:0
     were NOT allowed to be JUNK?"

    "What is the objective cost of forcing 13:0 and 13:1 to be the same
     identity?"

The same candidate graphs and the same set-partition objective are reused.
Only explicit audit constraints are added.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .global_identity_solver import IdentityUnit
from .identity_graph_solver import (
    CandidateIdentityGraph,
    IdentityGraphSolverConfig,
)


@dataclass(frozen=True)
class CounterfactualConstraints:
    required_nonjunk_keys: tuple[str, ...] = ()
    required_same_pairs: tuple[
        tuple[str, str],
        ...,
    ] = ()
    required_separate_pairs: tuple[
        tuple[str, str],
        ...,
    ] = ()


@dataclass(frozen=True)
class CounterfactualSolution:
    feasible: bool
    objective_utility: float | None

    selected_candidate_ids: tuple[int, ...]
    selected_groups: tuple[
        tuple[str, ...],
        ...,
    ]
    junk_keys: tuple[str, ...]

    solver_status: int | None
    solver_message: str

    def identity_of(
        self,
        unit_key: str,
    ) -> tuple[str, ...] | None:
        for group in self.selected_groups:
            if unit_key in group:
                return group
        return None


def solve_counterfactual_partition(
    units: Sequence[IdentityUnit],
    candidates: Sequence[CandidateIdentityGraph],
    *,
    config: IdentityGraphSolverConfig,
    constraints: CounterfactualConstraints | None = None,
) -> CounterfactualSolution:
    audit = constraints or CounterfactualConstraints()

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    if len(unit_by_key) != len(units):
        raise ValueError("unit keys must be unique")

    for key in audit.required_nonjunk_keys:
        if key not in unit_by_key:
            raise ValueError(
                f"unknown required_nonjunk key {key}"
            )

    for pair in (
        *audit.required_same_pairs,
        *audit.required_separate_pairs,
    ):
        if len(pair) != 2:
            raise ValueError(
                "counterfactual pair constraints require 2 keys"
            )
        if (
            pair[0] not in unit_by_key
            or pair[1] not in unit_by_key
        ):
            raise ValueError(
                f"unknown pair constraint {pair}"
            )

    candidate_var = {
        candidate.candidate_id: index
        for index, candidate in enumerate(
            candidates
        )
    }

    junk_offset = len(
        candidates
    )
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
            config=config,
        )

    rows: list[dict[int, float]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(
        coefficients: Mapping[int, float],
        lb: float,
        ub: float,
    ) -> None:
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

    # Exact partition: one identity graph OR JUNK.
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

    # Exactly the fixed short-clip roster count.
    add(
        {
            candidate_var[
                candidate.candidate_id
            ]: 1.0
            for candidate in candidates
        },
        config.identity_count,
        config.identity_count,
    )

    # Force selected realness hypotheses.
    for key in audit.required_nonjunk_keys:
        add(
            {
                junk_var[key]: 1.0,
            },
            0.0,
            0.0,
        )

    # Force two units to be covered by one common candidate.
    for left, right in audit.required_same_pairs:
        common = {
            candidate_var[
                candidate.candidate_id
            ]: 1.0
            for candidate in candidates
            if (
                left in candidate.member_keys
                and right in candidate.member_keys
            )
        }

        if not common:
            return CounterfactualSolution(
                feasible=False,
                objective_utility=None,
                selected_candidate_ids=(),
                selected_groups=(),
                junk_keys=(),
                solver_status=None,
                solver_message=(
                    f"no candidate graph contains required pair "
                    f"{left}<->{right}"
                ),
            )

        add(
            common,
            1.0,
            1.0,
        )

    # Forbid any candidate graph containing both units.
    for left, right in audit.required_separate_pairs:
        common = {
            candidate_var[
                candidate.candidate_id
            ]: 1.0
            for candidate in candidates
            if (
                left in candidate.member_keys
                and right in candidate.member_keys
            )
        }

        if common:
            add(
                common,
                0.0,
                0.0,
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
            "mip_rel_gap": config.mip_rel_gap,
        },
    )

    if (
        result.x is None
        or not result.success
    ):
        return CounterfactualSolution(
            feasible=False,
            objective_utility=None,
            selected_candidate_ids=(),
            selected_groups=(),
            junk_keys=(),
            solver_status=int(
                result.status
            ),
            solver_message=str(
                result.message
            ),
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

    return CounterfactualSolution(
        feasible=True,
        objective_utility=float(
            -result.fun
        ),
        selected_candidate_ids=tuple(
            sorted(
                candidate.candidate_id
                for candidate in selected
            )
        ),
        selected_groups=tuple(
            sorted(
                (
                    tuple(
                        candidate.member_keys
                    )
                    for candidate in selected
                ),
                key=lambda group: (
                    group[0],
                    len(group),
                    group,
                ),
            )
        ),
        junk_keys=junk_keys,
        solver_status=int(
            result.status
        ),
        solver_message=str(
            result.message
        ),
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
