from __future__ import annotations

from tracking.global_identity_solver import (
    IdentityUnit,
    PairConflict,
    UnitPairEvidence,
)
from tracking.identity_graph_solver import (
    IdentityGraphSolverConfig,
    build_hybrid_identity_edges,
    solve_identity_graphs,
)
from tracking.team_appearance import AppearanceLabel
from tracking.team_similarity import RelativeTeamLabel
from tracking.temporal_identity import TemporalTransitionEvidence


def unit(
    key,
    *,
    quality,
    team=RelativeTeamLabel.TEAM_A,
):
    track, segment = key.split(":")
    return IdentityUnit(
        key=key,
        track_id=int(track),
        segment_index=int(segment),
        start_frame=0,
        end_frame=100,
        observed_count=100,
        observed_span_s=8.0,
        appearance_label=AppearanceLabel.TEAM_A,
        team_label=team,
        appearance_coherence=0.75,
        appearance_q10=0.65,
        embedding_count=5,
        quality=quality,
    )


def pair(
    a,
    b,
    *,
    support,
    conflict=PairConflict.NONE,
):
    row = UnitPairEvidence(
        unit_a=a,
        unit_b=b,
        reid_score=support,
        geometry_score=support,
        team_score=1.0,
        fused_support_score=support,
        team_compatibility="same_team",
        simultaneous_count=0,
        simultaneous_fraction=0.0,
        simultaneous_median_distance_m=None,
        conflict=conflict,
        reason="test",
        soft_same_identity_penalty=0.0,
    )
    return row.pair_key, row


def transition(
    a,
    b,
    *,
    score,
    verdict,
):
    row = TemporalTransitionEvidence(
        source_key=a,
        target_key=b,
        source_track_id=int(a.split(":")[0]),
        target_track_id=int(b.split(":")[0]),
        gap_s=0.5,
        overlap_s=0.0,
        forward_error_m=0.2,
        backward_error_m=0.2,
        symmetric_error_m=0.2,
        overlap_median_distance_m=None,
        overlap_q90_distance_m=None,
        implied_speed_mps=2.0,
        velocity_cosine=1.0,
        velocity_difference_mps=0.1,
        position_score=score,
        velocity_score=score,
        gap_score=score,
        temporal_score=score,
        valid=(verdict != "contradiction"),
        regime="gap",
        reason="test",
        verdict=verdict,
    )
    return (a, b), row


def identity_for(solution, key):
    return next(
        identity
        for identity in solution.identities
        if key in identity.member_keys
    )


def test_support_edges_connect_branching_identity_graph():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.84),
        unit("3:0", quality=0.82),
        unit("4:0", quality=0.92),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.80),
            pair("1:0", "3:0", support=0.78),
        ]
    )
    transitions = dict(
        [
            transition(
                "1:0",
                "2:0",
                score=0.88,
                verdict="support",
            ),
            transition(
                "1:0",
                "3:0",
                score=0.84,
                verdict="support",
            ),
        ]
    )

    solution = solve_identity_graphs(
        units,
        pairs,
        transitions,
        config=IdentityGraphSolverConfig(
            identity_count=2,
        ),
    )

    group = identity_for(
        solution,
        "1:0",
    )

    assert {
        "1:0",
        "2:0",
        "3:0",
    }.issubset(
        set(group.member_keys)
    )


def test_neutral_temporal_allows_strong_identity_edge():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.86),
        unit("3:0", quality=0.91),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.72),
        ]
    )
    transitions = dict(
        [
            transition(
                "1:0",
                "2:0",
                score=0.50,
                verdict="neutral",
            ),
        ]
    )

    edges = build_hybrid_identity_edges(
        units,
        pairs,
        transitions,
    )

    assert ("1:0", "2:0") in edges

    solution = solve_identity_graphs(
        units,
        pairs,
        transitions,
        config=IdentityGraphSolverConfig(
            identity_count=2,
        ),
    )

    assert (
        identity_for(solution, "1:0").canonical_id
        == identity_for(solution, "2:0").canonical_id
    )


def test_temporal_contradiction_vetoes_strong_identity():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.86),
        unit("3:0", quality=0.91),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.95),
        ]
    )
    transitions = dict(
        [
            transition(
                "1:0",
                "2:0",
                score=0.0,
                verdict="contradiction",
            ),
        ]
    )

    solution = solve_identity_graphs(
        units,
        pairs,
        transitions,
        config=IdentityGraphSolverConfig(
            identity_count=2,
        ),
    )

    selected = {
        member: identity.canonical_id
        for identity in solution.identities
        for member in identity.member_keys
    }

    assert not (
        "1:0" in selected
        and "2:0" in selected
        and selected["1:0"]
        == selected["2:0"]
    )


def test_identity_conflict_vetoes_even_good_temporal_support():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.86),
        unit("3:0", quality=0.91),
    ]
    pairs = dict(
        [
            pair(
                "1:0",
                "2:0",
                support=0.95,
                conflict=PairConflict.STRONG,
            ),
        ]
    )
    transitions = dict(
        [
            transition(
                "1:0",
                "2:0",
                score=0.95,
                verdict="support",
            ),
        ]
    )

    solution = solve_identity_graphs(
        units,
        pairs,
        transitions,
        config=IdentityGraphSolverConfig(
            identity_count=2,
        ),
    )

    selected = {
        member: identity.canonical_id
        for identity in solution.identities
        for member in identity.member_keys
    }

    assert not (
        "1:0" in selected
        and "2:0" in selected
        and selected["1:0"]
        == selected["2:0"]
    )


def test_high_realness_survives_before_low_quality_junk():
    units = [
        unit("1:0", quality=0.95),
        unit("2:0", quality=0.90),
        unit("3:0", quality=0.88),
        unit("4:0", quality=0.15),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.80),
        ]
    )
    transitions = dict(
        [
            transition(
                "1:0",
                "2:0",
                score=0.90,
                verdict="support",
            ),
        ]
    )

    solution = solve_identity_graphs(
        units,
        pairs,
        transitions,
        config=IdentityGraphSolverConfig(
            identity_count=2,
        ),
    )

    assert "4:0" in solution.junk_keys
    assert "3:0" not in solution.junk_keys
