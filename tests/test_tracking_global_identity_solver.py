from __future__ import annotations

from tracking_v0.global_identity_solver import (
    ClusterSolverConfig,
    IdentityUnit,
    PairConflict,
    UnitPairEvidence,
    solve_identity_clusters,
)
from tracking_v0.team_appearance import AppearanceLabel
from tracking_v0.team_similarity import RelativeTeamLabel


def unit(
    key: str,
    *,
    quality: float,
    team: RelativeTeamLabel = RelativeTeamLabel.TEAM_A,
) -> IdentityUnit:
    track_id, segment = key.split(":")
    return IdentityUnit(
        key=key,
        track_id=int(track_id),
        segment_index=int(segment),
        start_frame=0,
        end_frame=100,
        observed_count=90,
        observed_span_s=6.0,
        appearance_label=AppearanceLabel.TEAM_A,
        team_label=team,
        appearance_coherence=0.78,
        appearance_q10=0.68,
        embedding_count=6,
        quality=quality,
    )


def pair(
    a: str,
    b: str,
    *,
    support: float,
    conflict: PairConflict = PairConflict.NONE,
    team: str = "same_team",
    soft_penalty: float = 0.0,
) -> UnitPairEvidence:
    return UnitPairEvidence(
        unit_a=a,
        unit_b=b,
        reid_score=support,
        geometry_score=support,
        team_score=1.0 if team == "same_team" else 0.0,
        fused_support_score=support,
        team_compatibility=team,
        simultaneous_count=0,
        simultaneous_fraction=0.0,
        simultaneous_median_distance_m=None,
        conflict=conflict,
        reason="test",
        soft_same_identity_penalty=soft_penalty,
    )


def mapping(*rows):
    return {
        row.pair_key: row
        for row in rows
    }


def identity_for(solution, key):
    return next(
        identity
        for identity in solution.identities
        if key in identity.member_keys
    )


def test_chain_support_can_form_one_identity_without_direct_ac_edge():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.80),
        unit("4:0", quality=0.90),
    ]
    pairs = mapping(
        pair("1:0", "2:0", support=0.84),
        pair("2:0", "3:0", support=0.82),
        # 1<->3 intentionally missing/weak.
    )

    solution = solve_identity_clusters(
        units,
        pairs,
        config=ClusterSolverConfig(
            identity_count=2,
            max_cluster_size=4,
        ),
    )

    group = identity_for(solution, "1:0")
    assert {
        "1:0",
        "2:0",
        "3:0",
    }.issubset(
        set(group.member_keys)
    )


def test_conflict_invalidates_cluster_even_if_connected_by_third_member():
    units = [
        unit("1:0", quality=0.90),
        unit("2:0", quality=0.85),
        unit("3:0", quality=0.80),
    ]
    pairs = mapping(
        pair("1:0", "2:0", support=0.90),
        pair("2:0", "3:0", support=0.90),
        pair(
            "1:0",
            "3:0",
            support=0.90,
            conflict=PairConflict.STRONG,
        ),
    )

    solution = solve_identity_clusters(
        units,
        pairs,
        config=ClusterSolverConfig(
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


def test_visual_split_soft_penalty_still_allows_same_identity():
    units = [
        unit("1:0", quality=0.88),
        unit("1:1", quality=0.82),
        unit("2:0", quality=0.90),
    ]
    pairs = mapping(
        pair(
            "1:0",
            "1:1",
            support=0.80,
            soft_penalty=0.07,
        ),
    )

    solution = solve_identity_clusters(
        units,
        pairs,
        config=ClusterSolverConfig(
            identity_count=2,
        ),
    )

    assert (
        identity_for(solution, "1:0").canonical_id
        == identity_for(solution, "1:1").canonical_id
    )


def test_high_realness_unit_is_preserved_before_low_quality_junk():
    units = [
        unit("1:0", quality=0.95),
        unit("2:0", quality=0.90),
        unit("3:0", quality=0.88),
        unit("4:0", quality=0.18),
    ]
    pairs = mapping(
        pair("1:0", "2:0", support=0.80),
    )

    solution = solve_identity_clusters(
        units,
        pairs,
        config=ClusterSolverConfig(
            identity_count=2,
        ),
    )

    assert "4:0" in solution.junk_keys
    assert "3:0" not in solution.junk_keys


def test_hard_team_conflict_never_same_identity():
    units = [
        unit(
            "1:0",
            quality=0.90,
            team=RelativeTeamLabel.TEAM_A,
        ),
        unit(
            "2:0",
            quality=0.85,
            team=RelativeTeamLabel.TEAM_B,
        ),
        unit(
            "3:0",
            quality=0.80,
            team=RelativeTeamLabel.TEAM_B,
        ),
    ]
    pairs = mapping(
        pair(
            "1:0",
            "2:0",
            support=0.95,
            conflict=PairConflict.HARD,
            team="team_conflict",
        ),
        pair("2:0", "3:0", support=0.82),
    )

    solution = solve_identity_clusters(
        units,
        pairs,
        config=ClusterSolverConfig(
            identity_count=2,
        ),
    )

    assert (
        identity_for(solution, "1:0").canonical_id
        != identity_for(solution, "2:0").canonical_id
    )


def test_posthoc_root_is_highest_quality_member():
    units = [
        unit("1:0", quality=0.60),
        unit("2:0", quality=0.91),
        unit("3:0", quality=0.20),
    ]
    pairs = mapping(
        pair("1:0", "2:0", support=0.90),
    )

    solution = solve_identity_clusters(
        units,
        pairs,
        config=ClusterSolverConfig(
            identity_count=2,
        ),
    )

    group = identity_for(solution, "1:0")
    assert group.root_key == "2:0"
