from __future__ import annotations

from tracking.roster_recovery import (
    RawTracklet,
    RosterClass,
    RosterObservation,
    RosterRecoveryConfig,
    recover_roster_identities,
)
from tracking.team_similarity import RelativeTeamLabel


def obs(
    raw_id: int,
    frame: int,
    t: float,
    x: float,
    z: float = 0.0,
):
    return RosterObservation(
        raw_track_id=raw_id,
        frame_index=frame,
        timestamp=t,
        x=x,
        z=z,
        measurement_id=f"{frame}:{raw_id}",
        confidence=0.9,
    )


def tracklet(
    raw_id: int,
    rows,
    *,
    team=RelativeTeamLabel.UNKNOWN,
    inside=None,
):
    rows = tuple(rows)
    return RawTracklet(
        raw_track_id=raw_id,
        observations=rows,
        team_label=team,
        inside_observed_count=(
            len(rows)
            if inside is None
            else inside
        ),
        continuation_observed_count=len(rows),
    )


def cfg(**kwargs):
    values = dict(
        roster_size=2,
        seed_search_seconds=0.2,
        recovery_batch_seconds=0.15,
        min_recovery_score=0.58,
        min_outside_only_recovery_score=0.64,
        min_assignment_margin=0.03,
    )
    values.update(kwargs)
    return RosterRecoveryConfig(**values)


def test_stable_seed_tracks_need_no_reid_queries():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
                obs(1, 2, 0.2, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.5),
                obs(2, 2, 0.2, 11.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
    ]

    calls = []

    def appearance(a, b):
        calls.append((a, b))
        return 0.9

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=cfg(),
    )

    assert result.raw_to_canonical == {
        1: 1,
        2: 2,
    }
    assert result.appearance_query_count == 0
    assert calls == []


def test_fragment_is_recovered_only_after_raw_track_break():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 20.0),
                obs(2, 1, 0.1, 20.0),
                obs(2, 2, 0.2, 20.0),
                obs(2, 3, 0.3, 20.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 2.2),
                obs(3, 4, 0.4, 3.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    calls = []

    def appearance(a, b):
        calls.append((a, b))
        return 0.92 if {a, b} == {1, 3} else 0.2

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=cfg(),
    )

    assert result.raw_to_canonical[3] == result.raw_to_canonical[1]
    assert result.raw_to_canonical[3] != result.raw_to_canonical[2]
    assert result.appearance_query_count > 0
    assert calls


def test_known_team_conflict_is_hard_veto():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 20.0),
                obs(2, 1, 0.1, 20.0),
                obs(2, 2, 0.2, 20.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 1.0),
                obs(3, 4, 0.4, 1.5),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
    ]

    def appearance(a, b):
        # Deliberately misleading visual evidence.
        return 0.99

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=cfg(),
    )

    if 1 in result.raw_to_canonical and 3 in result.raw_to_canonical:
        assert (
            result.raw_to_canonical[3]
            != result.raw_to_canonical[1]
        )


def test_global_assignment_prevents_two_fragments_claiming_same_id():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 11.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.30, 2.2),
                obs(3, 4, 0.40, 3.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            4,
            [
                obs(4, 3, 0.31, 11.8),
                obs(4, 4, 0.41, 12.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    similarities = {
        frozenset((1, 3)): 0.94,
        frozenset((2, 3)): 0.60,
        frozenset((1, 4)): 0.62,
        frozenset((2, 4)): 0.95,
    }

    def appearance(a, b):
        return similarities.get(frozenset((a, b)), 0.5)

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=cfg(),
    )

    assert result.raw_to_canonical[3] == result.raw_to_canonical[1]
    assert result.raw_to_canonical[4] == result.raw_to_canonical[2]
    assert result.raw_to_canonical[3] != result.raw_to_canonical[4]


def test_outside_only_fragment_may_continue_established_player():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0, 33.5),
                obs(1, 1, 0.1, 0.0, 33.8),
            ],
            team=RelativeTeamLabel.TEAM_A,
            inside=2,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 20.0, 0.0),
                obs(2, 1, 0.1, 20.0, 0.0),
                obs(2, 2, 0.2, 20.0, 0.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 0.0, 34.5),
                obs(3, 4, 0.4, 0.0, 35.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
            inside=0,
        ),
    ]

    def appearance(a, b):
        return 0.93 if {a, b} == {1, 3} else 0.4

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=cfg(),
    )

    assert result.raw_to_canonical[3] == result.raw_to_canonical[1]


def test_outside_only_coach_never_establishes_roster_slot():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.5),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 0, 0.0, 2.0, 35.0),
                obs(3, 1, 0.1, 2.0, 35.0),
                obs(3, 2, 0.2, 2.0, 35.0),
            ],
            team=RelativeTeamLabel.UNKNOWN,
            inside=0,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.5,
        config=cfg(),
    )

    assert len(result.identities) == 2
    assert 3 not in {
        identity.seed_raw_track_id
        for identity in result.identities
    }


def test_long_simultaneous_overlap_is_not_merged_even_with_high_reid():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.0),
                obs(1, 2, 0.2, 0.0),
                obs(1, 3, 0.3, 0.0),
                obs(1, 4, 0.4, 0.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 20.0),
                obs(2, 1, 0.1, 20.0),
                obs(2, 2, 0.2, 20.0),
                obs(2, 3, 0.3, 20.0),
                obs(2, 4, 0.4, 20.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 2, 0.2, 8.0),
                obs(3, 3, 0.3, 8.0),
                obs(3, 4, 0.4, 8.0),
                obs(3, 5, 0.5, 8.0),
                obs(3, 6, 0.6, 8.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.99,
        config=cfg(
            max_overlap_seconds=0.15,
        ),
    )

    if 2 in result.raw_to_canonical and 3 in result.raw_to_canonical:
        assert (
            result.raw_to_canonical[2]
            != result.raw_to_canonical[3]
        )


def test_seed_allows_boundary_player_if_early_window_proves_inside_history():
    tracks = [
        tracklet(
            1,
            [
                # At the first frame the player is 0.8 m beyond the touchline,
                # but still inside the 1.5 m continuation corridor.
                obs(1, 0, 0.0, 0.0, 34.8),
                # Shortly afterwards the same raw track is strictly inside.
                obs(1, 1, 0.1, 0.0, 33.8),
                obs(1, 2, 0.2, 0.0, 33.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
            inside=2,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0, 0.0),
                obs(2, 1, 0.1, 10.0, 0.0),
                obs(2, 2, 0.2, 10.0, 0.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        config=cfg(
            seed_search_seconds=0.2,
        ),
    )

    assert result.seed_frame_index == 0
    assert {
        identity.seed_raw_track_id
        for identity in result.identities
    } == {1, 2}


def test_outside_only_coach_cannot_help_fill_seed_even_inside_corridor():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.0, 0.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                # Coach sits inside the 1.5 m continuation corridor but never
                # has a strict-inside observation.
                obs(2, 0, 0.0, 2.0, 34.8),
                obs(2, 1, 0.1, 2.0, 34.9),
            ],
            team=RelativeTeamLabel.UNKNOWN,
            inside=0,
        ),
    ]

    import pytest

    with pytest.raises(
        RuntimeError,
        match="V6 full-match roster MILP found no feasible solution",
    ):
        recover_roster_identities(
            tracks,
            config=cfg(
                seed_search_seconds=0.2,
            ),
        )


def test_progressive_bootstrap_does_not_require_full_roster_same_frame():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, -10.0),
                obs(1, 1, 0.1, -9.5),
                obs(1, 2, 0.2, -9.0),
                obs(1, 3, 0.3, -8.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.5),
                obs(2, 2, 0.2, 11.0),
                obs(2, 3, 0.3, 11.5),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        # Missing participant only becomes visible after the initial frame.
        tracklet(
            3,
            [
                obs(3, 2, 0.2, 35.0),
                obs(3, 3, 0.3, 35.5),
            ],
            team=RelativeTeamLabel.UNKNOWN,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        config=RosterRecoveryConfig(
            roster_size=3,
            seed_search_seconds=0.5,
            initial_seed_min_fraction=0.66,
            min_recovery_score=0.58,
            min_outside_only_recovery_score=0.64,
            min_assignment_margin=0.03,
        ),
    )

    assert result.seed_frame_index == 0
    assert result.roster_completed_frame_index == 2
    assert len(result.identities) == 3
    assert set(result.raw_to_canonical) == {1, 2, 3}


def test_bootstrap_recovers_fragment_before_opening_new_roster_slot():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 20.0),
                obs(2, 1, 0.1, 20.0),
                obs(2, 2, 0.2, 20.0),
                obs(2, 3, 0.3, 20.0),
                obs(2, 4, 0.4, 20.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        # Fragment of raw 1.
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 2.0),
                obs(3, 4, 0.4, 3.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        # Truly missing third participant.
        tracklet(
            4,
            [
                obs(4, 5, 0.5, -30.0),
                obs(4, 6, 0.6, -29.5),
            ],
            team=RelativeTeamLabel.UNKNOWN,
        ),
    ]

    def appearance(a, b):
        if {a, b} == {1, 3}:
            return 0.95
        return 0.30

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=RosterRecoveryConfig(
            roster_size=3,
            seed_search_seconds=0.8,
            initial_seed_min_fraction=0.66,
            min_recovery_score=0.58,
            min_outside_only_recovery_score=0.64,
            min_assignment_margin=0.03,
        ),
    )

    assert result.raw_to_canonical[3] == result.raw_to_canonical[1]
    assert result.raw_to_canonical[4] not in {
        result.raw_to_canonical[1],
        result.raw_to_canonical[2],
    }
    assert len(result.identities) == 3


def test_v3_exact_roster_composition_is_assigned_globally():
    tracks = [
        RawTracklet(
            raw_track_id=1,
            observations=(
                obs(1, 0, 0.0, -10.0),
                obs(1, 1, 0.1, -9.5),
            ),
            team_label=RelativeTeamLabel.TEAM_A,
            team_a_similarity=0.96,
            team_b_similarity=0.62,
            referee_score=0.05,
            inside_observed_count=2,
            continuation_observed_count=2,
        ),
        RawTracklet(
            raw_track_id=2,
            observations=(
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.5),
            ),
            team_label=RelativeTeamLabel.TEAM_B,
            team_a_similarity=0.63,
            team_b_similarity=0.97,
            referee_score=0.10,
            inside_observed_count=2,
            continuation_observed_count=2,
        ),
        RawTracklet(
            raw_track_id=3,
            observations=(
                obs(3, 0, 0.0, 0.0),
                obs(3, 1, 0.1, 0.5),
            ),
            team_label=RelativeTeamLabel.UNKNOWN,
            team_a_similarity=0.78,
            team_b_similarity=0.79,
            referee_score=0.95,
            inside_observed_count=2,
            continuation_observed_count=2,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        config=RosterRecoveryConfig(
            roster_size=3,
            team_a_count=1,
            team_b_count=1,
            referee_count=1,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=0.66,
        ),
    )

    by_seed = {
        identity.seed_raw_track_id: identity.roster_class
        for identity in result.identities
    }

    assert by_seed[1] is RosterClass.TEAM_A
    assert by_seed[2] is RosterClass.TEAM_B
    assert by_seed[3] is RosterClass.REFEREE
    assert result.composition_score is not None


def test_v3_bootstrap_decision_uses_full_window_but_backfills_root_history():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, -10.0),
                obs(1, 1, 0.1, -9.5),
                obs(1, 2, 0.2, -9.0),
                obs(1, 3, 0.3, -8.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.5),
                obs(2, 2, 0.2, 11.0),
                obs(2, 3, 0.3, 11.5),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 2, 0.2, 35.0),
                obs(3, 3, 0.3, 35.5),
                obs(3, 4, 0.4, 36.0),
            ],
            team=RelativeTeamLabel.UNKNOWN,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        config=RosterRecoveryConfig(
            roster_size=3,
            seed_search_seconds=0.8,
            initial_seed_min_fraction=0.66,
            bootstrap_root_recovery_guard=0.70,
        ),
    )

    assert result.seed_frame_index == 0
    # V6 has no bootstrap authority. The final decision deliberately uses all
    # available context through the end of this synthetic clip.
    assert result.bootstrap_decision_timestamp == 0.4
    assert result.global_context_end_timestamp == 0.4
    assert result.bootstrap_added_root_ids == ()
    assert set(result.raw_to_canonical) == {1, 2, 3}


def test_v4_occupied_canonical_is_pruned_before_reid():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.0),
                obs(2, 2, 0.2, 10.0),
                obs(2, 3, 0.3, 10.0),
                obs(2, 4, 0.4, 10.0),
                obs(2, 5, 0.5, 10.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 2.0),
                obs(3, 4, 0.4, 3.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    calls = []

    def appearance(a, b):
        calls.append((a, b))
        # Deliberately make the occupied ID look visually better. It still
        # must not be queried/selected because it is physically occupied by
        # another observed person.
        if {a, b} == {2, 3}:
            return 0.99
        if {a, b} == {1, 3}:
            return 0.70
        return 0.5

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=RosterRecoveryConfig(
            roster_size=2,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=1.0,
            min_recovery_score=0.60,
            min_unique_candidate_score=0.48,
            min_assignment_margin=0.03,
        ),
    )

    assert result.raw_to_canonical[3] == result.raw_to_canonical[1]
    assert (2, 3) not in calls
    assert (3, 2) not in calls

    # V5 prunes impossible simultaneous transitions before ReID/optimization.
    # The important invariant is that the occupied canonical cannot receive raw 3.
    assert result.raw_to_canonical[3] != result.raw_to_canonical[2]


def test_v4_unique_free_identity_can_recover_below_normal_threshold():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 10.0),
                obs(2, 1, 0.1, 10.0),
                obs(2, 2, 0.2, 10.0),
                obs(2, 3, 0.3, 10.0),
                obs(2, 4, 0.4, 10.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 3.2),
                obs(3, 4, 0.4, 4.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.45,
        config=RosterRecoveryConfig(
            roster_size=2,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=1.0,
            # Intentionally too high for ordinary score+margin acceptance.
            min_recovery_score=0.80,
            min_unique_candidate_score=0.48,
            min_structural_candidate_score=0.35,
            min_assignment_margin=0.10,
        ),
    )

    assert result.raw_to_canonical[3] == result.raw_to_canonical[1]

    accepted = [
        row
        for row in result.accepted_recoveries
        if row.fragment_raw_track_id == 3
    ]
    assert accepted
    assert accepted[-1].reason == "accepted_full_match_path"
    assert accepted[-1].total_score < 0.80


def test_v4_trusted_team_conflict_prunes_before_reid():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 20.0),
                obs(2, 1, 0.1, 20.0),
                obs(2, 2, 0.2, 20.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            3,
            [
                obs(3, 3, 0.3, 1.0),
                obs(3, 4, 0.4, 1.5),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
    ]

    calls = []

    def appearance(a, b):
        calls.append((a, b))
        return 0.99

    result = recover_roster_identities(
        tracks,
        appearance_similarity=appearance,
        config=cfg(),
    )

    # TEAM_B fragment must never ask ReID whether it is canonical TEAM_A.
    assert (1, 3) not in calls
    assert (3, 1) not in calls
    if 1 in result.raw_to_canonical and 3 in result.raw_to_canonical:
        assert (
            result.raw_to_canonical[3]
            != result.raw_to_canonical[1]
        )


def test_v5_global_paths_explain_two_long_simultaneous_fragments_with_different_ids():
    """Greedy recovery could let one fragment consume the only locally good ID.

    V5 sees both long fragments at once. Since they overlap for several seconds
    while being tens of metres apart, they must occupy two different canonical
    paths if both are worth explaining.
    """
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
                obs(1, 2, 0.2, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            2,
            [
                obs(2, 0, 0.0, 28.0),
                obs(2, 1, 0.1, 28.5),
                obs(2, 2, 0.2, 29.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        # Both start after a realistic 5.8 s observation gap and then coexist
        # for ~4 s, far apart.
        tracklet(
            3,
            [
                obs(3, 60, 6.0, 6.0),
                obs(3, 70, 7.0, 8.0),
                obs(3, 80, 8.0, 10.0),
                obs(3, 90, 9.0, 12.0),
                obs(3, 100, 10.0, 14.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
        tracklet(
            4,
            [
                obs(4, 60, 6.0, 35.0),
                obs(4, 70, 7.0, 36.0),
                obs(4, 80, 8.0, 37.0),
                obs(4, 90, 9.0, 38.0),
                obs(4, 100, 10.0, 39.0),
            ],
            team=RelativeTeamLabel.TEAM_B,
        ),
    ]

    similarities = {
        frozenset((1, 3)): 0.80,
        frozenset((1, 4)): 0.74,
        frozenset((2, 3)): 0.65,
        frozenset((2, 4)): 0.72,
    }

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda a, b: similarities.get(
            frozenset((a, b)),
            0.5,
        ),
        config=RosterRecoveryConfig(
            roster_size=2,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=1.0,
            max_recovery_gap_s=15.0,
            global_mip_time_limit_s=5.0,
        ),
    )

    assert 3 in result.raw_to_canonical
    assert 4 in result.raw_to_canonical
    assert result.raw_to_canonical[3] != result.raw_to_canonical[4]
    assert result.global_solver_status == "optimal"


def test_v5_six_second_disappearance_is_not_identity_break():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
                obs(1, 2, 0.2, 1.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 62, 6.2, 15.0),
                obs(2, 63, 6.3, 15.5),
                obs(2, 64, 6.4, 16.0),
                obs(2, 65, 6.5, 16.5),
                obs(2, 66, 6.6, 17.0),
                obs(2, 67, 6.7, 17.5),
                obs(2, 68, 6.8, 18.0),
                obs(2, 69, 6.9, 18.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.82,
        config=RosterRecoveryConfig(
            roster_size=1,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=1.0,
            max_recovery_gap_s=15.0,
            global_mip_time_limit_s=5.0,
        ),
    )

    assert result.raw_to_canonical[2] == result.raw_to_canonical[1]
    accepted = [
        row
        for row in result.accepted_recoveries
        if row.fragment_raw_track_id == 2
    ]
    assert accepted
    assert accepted[-1].gap_s >= 5.9
    assert accepted[-1].reason == "accepted_full_match_path"


def test_v5_long_simultaneous_tracks_cannot_share_one_canonical_path():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.2),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 20, 2.0, 5.0),
                obs(2, 30, 3.0, 6.0),
                obs(2, 40, 4.0, 7.0),
                obs(2, 50, 5.0, 8.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            3,
            [
                obs(3, 20, 2.0, 30.0),
                obs(3, 30, 3.0, 31.0),
                obs(3, 40, 4.0, 32.0),
                obs(3, 50, 5.0, 33.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.75,
        config=RosterRecoveryConfig(
            roster_size=1,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=1.0,
            max_recovery_gap_s=15.0,
            global_mip_time_limit_s=5.0,
        ),
    )

    # Only one of the two simultaneous people may continue the one-player path.
    assigned = [
        raw_id
        for raw_id in (2, 3)
        if raw_id in result.raw_to_canonical
    ]
    assert len(assigned) == 1


def test_v5_zero_corridor_outsider_has_no_global_coverage_reward():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        RawTracklet(
            raw_track_id=2,
            observations=(
                obs(2, 20, 2.0, 60.0, 40.0),
                obs(2, 30, 3.0, 60.5, 40.0),
                obs(2, 40, 4.0, 61.0, 40.0),
                obs(2, 50, 5.0, 61.5, 40.0),
                obs(2, 60, 6.0, 62.0, 40.0),
                obs(2, 70, 7.0, 62.5, 40.0),
                obs(2, 80, 8.0, 63.0, 40.0),
                obs(2, 90, 9.0, 63.5, 40.0),
            ),
            team_label=RelativeTeamLabel.UNKNOWN,
            inside_observed_count=0,
            continuation_observed_count=0,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.99,
        config=RosterRecoveryConfig(
            roster_size=1,
            seed_search_seconds=0.2,
            initial_seed_min_fraction=1.0,
            global_mip_time_limit_s=5.0,
        ),
    )

    assert 2 not in result.raw_to_canonical
    assert 2 in result.unassigned_fragment_ids


def test_v6_late_long_participant_can_replace_early_short_candidate():
    """The first frame is evidence, not a mandatory roster anchor."""
    tracks = [
        # Short early TEAM_A candidate.
        RawTracklet(
            raw_track_id=1,
            observations=(
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.2),
            ),
            team_label=RelativeTeamLabel.TEAM_A,
            team_a_similarity=0.96,
            team_b_similarity=0.60,
            inside_observed_count=2,
            continuation_observed_count=2,
        ),
        # Long TEAM_A participant.
        RawTracklet(
            raw_track_id=2,
            observations=tuple(
                obs(2, frame, frame / 10.0, 20.0 + frame * 0.05)
                for frame in range(0, 101, 10)
            ),
            team_label=RelativeTeamLabel.TEAM_A,
            team_a_similarity=0.97,
            team_b_similarity=0.61,
            inside_observed_count=11,
            continuation_observed_count=11,
        ),
        # TEAM_B first appears later, but is clearly a real long participant.
        RawTracklet(
            raw_track_id=3,
            observations=tuple(
                obs(3, frame, frame / 10.0, -20.0 + frame * 0.04)
                for frame in range(10, 101, 10)
            ),
            team_label=RelativeTeamLabel.TEAM_B,
            team_a_similarity=0.60,
            team_b_similarity=0.97,
            inside_observed_count=10,
            continuation_observed_count=10,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.5,
        config=RosterRecoveryConfig(
            roster_size=2,
            team_a_count=1,
            team_b_count=1,
            referee_count=0,
            global_mip_time_limit_s=5.0,
        ),
    )

    assert 2 in result.raw_to_canonical
    assert 3 in result.raw_to_canonical
    assert 1 not in result.raw_to_canonical
    assert result.bootstrap_added_root_ids == ()


def test_v6_exact_1_1_1_composition_participates_in_path_selection():
    tracks = [
        RawTracklet(
            raw_track_id=1,
            observations=(
                obs(1, 0, 0.0, -15.0),
                obs(1, 1, 0.1, -14.5),
                obs(1, 2, 0.2, -14.0),
            ),
            team_label=RelativeTeamLabel.TEAM_A,
            team_a_similarity=0.98,
            team_b_similarity=0.55,
            referee_score=0.02,
            inside_observed_count=3,
            continuation_observed_count=3,
        ),
        RawTracklet(
            raw_track_id=2,
            observations=(
                obs(2, 0, 0.0, 15.0),
                obs(2, 1, 0.1, 15.5),
                obs(2, 2, 0.2, 16.0),
            ),
            team_label=RelativeTeamLabel.TEAM_B,
            team_a_similarity=0.55,
            team_b_similarity=0.98,
            referee_score=0.01,
            inside_observed_count=3,
            continuation_observed_count=3,
        ),
        RawTracklet(
            raw_track_id=3,
            observations=(
                obs(3, 0, 0.0, 0.0),
                obs(3, 1, 0.1, 0.2),
                obs(3, 2, 0.2, 0.4),
            ),
            team_label=RelativeTeamLabel.UNKNOWN,
            team_a_similarity=0.77,
            team_b_similarity=0.78,
            referee_score=0.99,
            inside_observed_count=3,
            continuation_observed_count=3,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        config=RosterRecoveryConfig(
            roster_size=3,
            team_a_count=1,
            team_b_count=1,
            referee_count=1,
            global_mip_time_limit_s=5.0,
        ),
    )

    identity_by_raw = {
        raw_id: identity
        for identity in result.identities
        for raw_id in identity.member_raw_track_ids
    }

    assert identity_by_raw[1].roster_class is RosterClass.TEAM_A
    assert identity_by_raw[2].roster_class is RosterClass.TEAM_B
    assert identity_by_raw[3].roster_class is RosterClass.REFEREE


def test_v6_full_context_timestamp_is_clip_end_not_seed_window():
    tracks = [
        tracklet(
            1,
            [
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.5),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
        tracklet(
            2,
            [
                obs(2, 50, 5.0, 4.0),
                obs(2, 60, 6.0, 5.0),
                obs(2, 70, 7.0, 6.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        appearance_similarity=lambda _a, _b: 0.9,
        config=RosterRecoveryConfig(
            roster_size=1,
            seed_search_seconds=0.2,
            max_recovery_gap_s=15.0,
            global_mip_time_limit_s=5.0,
        ),
    )

    assert result.global_context_end_timestamp == 7.0
    assert result.bootstrap_decision_timestamp == 7.0
    assert result.raw_to_canonical[1] == result.raw_to_canonical[2]


def test_v6_ambiguous_kit_is_soft_not_opposite_team_veto():
    # Delta is 0.16 < hard margin 0.18, mirroring the kind of ambiguity seen
    # in the real raw35 evidence.
    tracks = [
        RawTracklet(
            raw_track_id=1,
            observations=(
                obs(1, 0, 0.0, 0.0),
                obs(1, 1, 0.1, 0.3),
                obs(1, 2, 0.2, 0.6),
            ),
            team_label=RelativeTeamLabel.TEAM_A,
            team_a_similarity=0.93,
            team_b_similarity=0.77,
            inside_observed_count=3,
            continuation_observed_count=3,
        ),
    ]

    result = recover_roster_identities(
        tracks,
        config=RosterRecoveryConfig(
            roster_size=1,
            team_a_count=0,
            team_b_count=1,
            referee_count=0,
            global_hard_team_margin=0.18,
            global_mip_time_limit_s=5.0,
        ),
    )

    assert 1 in result.raw_to_canonical
    assert result.identities[0].roster_class is RosterClass.TEAM_B
