from __future__ import annotations

from tracking_v0.global_identity_solver import (
    IdentityUnit,
)
from tracking_v0.identity_graph_counterfactual import (
    CounterfactualConstraints,
    solve_counterfactual_partition,
)
from tracking_v0.identity_graph_solver import (
    CandidateIdentityGraph,
    HybridIdentityEdge,
    IdentityGraphSolverConfig,
)
from tracking_v0.team_appearance import AppearanceLabel
from tracking_v0.team_similarity import RelativeTeamLabel


def unit(key, quality):
    track, segment = key.split(":")
    return IdentityUnit(
        key=key,
        track_id=int(track),
        segment_index=int(segment),
        start_frame=0,
        end_frame=10,
        observed_count=10,
        observed_span_s=1.0,
        appearance_label=AppearanceLabel.TEAM_A,
        team_label=RelativeTeamLabel.TEAM_A,
        appearance_coherence=0.7,
        appearance_q10=0.6,
        embedding_count=4,
        quality=quality,
    )


def candidate(cid, members, utility):
    return CandidateIdentityGraph(
        candidate_id=cid,
        member_keys=tuple(members),
        utility=utility,
        mst_edges=(),
        mst_min_score=1.0,
        mst_mean_score=1.0,
        extra_strong_pair_bonus=0.0,
        soft_identity_penalty=0.0,
    )


def test_force_nonjunk_changes_partition():
    units = [
        unit("1:0", 0.9),
        unit("2:0", 0.8),
        unit("3:0", 0.7),
    ]
    candidates = [
        candidate(0, ("1:0",), 0.90),
        candidate(1, ("2:0",), 0.80),
        candidate(2, ("3:0",), 0.70),
        candidate(
            3,
            ("1:0", "2:0"),
            1.85,
        ),
    ]
    config = IdentityGraphSolverConfig(
        identity_count=2,
    )

    baseline = solve_counterfactual_partition(
        units,
        candidates,
        config=config,
    )
    forced = solve_counterfactual_partition(
        units,
        candidates,
        config=config,
        constraints=CounterfactualConstraints(
            required_nonjunk_keys=(
                "3:0",
            ),
        ),
    )

    assert baseline.feasible
    assert forced.feasible
    assert "3:0" not in forced.junk_keys


def test_force_same_pair_requires_common_candidate():
    units = [
        unit("1:0", 0.9),
        unit("2:0", 0.8),
        unit("3:0", 0.7),
    ]
    candidates = [
        candidate(0, ("1:0",), 0.90),
        candidate(1, ("2:0",), 0.80),
        candidate(2, ("3:0",), 0.70),
        candidate(
            3,
            ("1:0", "2:0"),
            1.85,
        ),
    ]
    config = IdentityGraphSolverConfig(
        identity_count=2,
    )

    forced = solve_counterfactual_partition(
        units,
        candidates,
        config=config,
        constraints=CounterfactualConstraints(
            required_same_pairs=(
                ("1:0", "2:0"),
            ),
        ),
    )

    assert forced.feasible
    assert (
        forced.identity_of("1:0")
        == forced.identity_of("2:0")
    )


def test_impossible_same_pair_is_reported_infeasible():
    units = [
        unit("1:0", 0.9),
        unit("2:0", 0.8),
    ]
    candidates = [
        candidate(0, ("1:0",), 0.90),
        candidate(1, ("2:0",), 0.80),
    ]
    config = IdentityGraphSolverConfig(
        identity_count=1,
    )

    forced = solve_counterfactual_partition(
        units,
        candidates,
        config=config,
        constraints=CounterfactualConstraints(
            required_same_pairs=(
                ("1:0", "2:0"),
            ),
        ),
    )

    assert not forced.feasible
