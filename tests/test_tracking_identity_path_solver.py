from __future__ import annotations

from tracking.global_identity_solver import (
    IdentityUnit,
    PairConflict,
    UnitPairEvidence,
)
from tracking.identity_path_solver import (
    IdentityPathSolverConfig,
    solve_identity_paths,
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
    source,
    target,
    *,
    score,
):
    return (
        (source, target),
        TemporalTransitionEvidence(
            source_key=source,
            target_key=target,
            source_track_id=int(source.split(":")[0]),
            target_track_id=int(target.split(":")[0]),
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
            position_score=0.95,
            velocity_score=0.95,
            gap_score=0.90,
            temporal_score=score,
            valid=True,
            regime="gap",
            reason="test",
            verdict="support",
        ),
    )


def identity_for(solution, key):
    return next(
        identity
        for identity in solution.identities
        if key in identity.member_keys
    )


def test_temporal_chain_forms_one_identity_path():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.82),
        unit("4:0", quality=0.90),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.80),
            pair("2:0", "3:0", support=0.82),
        ]
    )
    transitions = dict(
        [
            transition("1:0", "2:0", score=0.92),
            transition("2:0", "3:0", score=0.90),
        ]
    )

    solution = solve_identity_paths(
        units,
        pairs,
        transitions,
        config=IdentityPathSolverConfig(
            identity_count=2,
        ),
    )

    path = identity_for(
        solution,
        "1:0",
    )

    assert path.member_keys == (
        "1:0",
        "2:0",
        "3:0",
    )


def test_conflicting_nonadjacent_pair_cannot_share_path():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.82),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.85),
            pair("2:0", "3:0", support=0.85),
            pair(
                "1:0",
                "3:0",
                support=0.90,
                conflict=PairConflict.STRONG,
            ),
        ]
    )
    transitions = dict(
        [
            transition("1:0", "2:0", score=0.92),
            transition("2:0", "3:0", score=0.92),
        ]
    )

    solution = solve_identity_paths(
        units,
        pairs,
        transitions,
        config=IdentityPathSolverConfig(
            identity_count=2,
        ),
    )

    assert not any(
        {
            "1:0",
            "2:0",
            "3:0",
        }.issubset(
            set(identity.member_keys)
        )
        for identity in solution.identities
    )


def test_high_realness_unit_survives_before_low_quality_junk():
    units = [
        unit("1:0", quality=0.95),
        unit("2:0", quality=0.90),
        unit("3:0", quality=0.88),
        unit("4:0", quality=0.15),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.82),
        ]
    )
    transitions = dict(
        [
            transition("1:0", "2:0", score=0.90),
        ]
    )

    solution = solve_identity_paths(
        units,
        pairs,
        transitions,
        config=IdentityPathSolverConfig(
            identity_count=2,
        ),
    )

    assert "4:0" in solution.junk_keys
    assert "3:0" not in solution.junk_keys


def test_visual_similarity_without_temporal_continuity_does_not_merge():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.88),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.95),
        ]
    )
    transitions = {}

    solution = solve_identity_paths(
        units,
        pairs,
        transitions,
        config=IdentityPathSolverConfig(
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
        and selected["1:0"] == selected["2:0"]
    )


def test_temporal_continuity_still_needs_identity_support():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.88),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.30),
        ]
    )
    transitions = dict(
        [
            transition("1:0", "2:0", score=0.98),
        ]
    )

    solution = solve_identity_paths(
        units,
        pairs,
        transitions,
        config=IdentityPathSolverConfig(
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
        and selected["1:0"] == selected["2:0"]
    )


def test_strong_identity_can_use_neutral_temporal_evidence():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.88),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.93),
        ]
    )

    neutral = TemporalTransitionEvidence(
        source_key="1:0",
        target_key="2:0",
        source_track_id=1,
        target_track_id=2,
        gap_s=-3.0,
        overlap_s=3.0,
        forward_error_m=None,
        backward_error_m=None,
        symmetric_error_m=None,
        overlap_median_distance_m=None,
        overlap_q90_distance_m=None,
        implied_speed_mps=None,
        velocity_cosine=None,
        velocity_difference_mps=0.0,
        position_score=0.5,
        velocity_score=0.5,
        gap_score=0.5,
        temporal_score=0.5,
        valid=True,
        regime="interleaved_overlap",
        reason="no_simultaneous_observations_neutral",
        verdict="neutral",
    )

    solution = solve_identity_paths(
        units,
        pairs,
        {("1:0", "2:0"): neutral},
        config=IdentityPathSolverConfig(
            identity_count=2,
        ),
    )

    assert (
        identity_for(solution, "1:0").canonical_id
        == identity_for(solution, "2:0").canonical_id
    )


def test_temporal_contradiction_vetoes_even_strong_identity():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.88),
    ]
    pairs = dict(
        [
            pair("1:0", "2:0", support=0.95),
        ]
    )

    contradiction = TemporalTransitionEvidence(
        source_key="1:0",
        target_key="2:0",
        source_track_id=1,
        target_track_id=2,
        gap_s=0.5,
        overlap_s=0.0,
        forward_error_m=20.0,
        backward_error_m=20.0,
        symmetric_error_m=20.0,
        overlap_median_distance_m=None,
        overlap_q90_distance_m=None,
        implied_speed_mps=40.0,
        velocity_cosine=1.0,
        velocity_difference_mps=0.0,
        position_score=0.0,
        velocity_score=1.0,
        gap_score=0.8,
        temporal_score=0.0,
        valid=False,
        regime="gap",
        reason="implied_speed_too_large",
        verdict="contradiction",
    )

    solution = solve_identity_paths(
        units,
        pairs,
        {("1:0", "2:0"): contradiction},
        config=IdentityPathSolverConfig(
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
        and selected["1:0"] == selected["2:0"]
    )
