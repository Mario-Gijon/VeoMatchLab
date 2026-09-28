from __future__ import annotations

import numpy as np

from tracking.physical_identity_solver import (
    PhysicalIdentitySolverConfig,
    PhysicalObservation,
    PhysicalRelationKind,
    PhysicalSegmentQuality,
    PhysicalTrackSegment,
    solve_physical_identities,
)
from tracking.team_similarity import RelativeTeamLabel


def obs(frame, t, x, z=0.0):
    return PhysicalObservation(
        frame_index=frame,
        timestamp=t,
        x=x,
        z=z,
        measurement_id=f"{frame}",
        confidence=0.9,
    )


def proto(x, y):
    value = np.asarray([x, y], dtype=np.float32)
    return value / np.linalg.norm(value)


def segment(
    segment_id,
    raw_id,
    rows,
    *,
    team=RelativeTeamLabel.UNKNOWN,
    team_conf=0.0,
    reid=None,
    inside=None,
    allowed=None,
    segment_index=0,
    quality=PhysicalSegmentQuality.CLEAN,
    quality_score=1.0,
):
    rows = tuple(rows)
    if inside is None:
        inside = len(rows)
    if allowed is None:
        allowed = len(rows)
    return PhysicalTrackSegment(
        segment_id=segment_id,
        raw_track_id=raw_id,
        segment_index=segment_index,
        observations=rows,
        team_label=team,
        team_confidence=team_conf,
        reid_prototype=reid,
        identity_quality=quality,
        identity_quality_score=quality_score,
        inside_observed_count=inside,
        continuation_observed_count=allowed,
    )


def test_opposite_confident_teams_cannot_be_same_physical_identity():
    tracks = [
        segment(
            1, 1,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.5)],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(1, 0),
        ),
        segment(
            2, 2,
            [obs(20, 2.0, 1.0), obs(21, 2.1, 1.5)],
            team=RelativeTeamLabel.TEAM_B,
            team_conf=0.95,
            reid=proto(1, 0),
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(roster_size=1, mip_time_limit_s=5.0),
    )

    assigned = set(result.segment_to_identity)
    assert not ({1, 2} <= assigned)
    assert result.hard_team_prunes >= 1


def test_six_second_gap_can_continue_same_person():
    tracks = [
        segment(
            1, 1,
            [obs(0, 0.0, 0.0), obs(2, 0.2, 1.0)],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(1, 0),
        ),
        segment(
            2, 2,
            [
                obs(62, 6.2, 15.0), obs(64, 6.4, 16.0),
                obs(66, 6.6, 17.0), obs(68, 6.8, 18.0),
                obs(70, 7.0, 19.0), obs(72, 7.2, 20.0),
            ],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(0.98, 0.05),
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(roster_size=1, mip_time_limit_s=5.0),
    )

    assert result.segment_to_identity[1] == result.segment_to_identity[2]
    selected = [
        edge for edge in result.selected_continuations
        if edge.right_segment_id == 2
    ]
    assert selected
    assert selected[0].gap_s >= 5.9


def test_solver_has_no_referee_class_and_keeps_unlabeled_person_valid():
    track = segment(
        1, 1,
        [obs(0, 0.0, 0.0), obs(1, 0.1, 0.2)],
        team=RelativeTeamLabel.UNKNOWN,
        team_conf=0.0,
    )

    config = PhysicalIdentitySolverConfig(roster_size=1, mip_time_limit_s=5.0)
    result = solve_physical_identities([track], config=config)

    assert not hasattr(config, "referee_count")
    assert not hasattr(result.identities[0], "roster_class")
    assert result.identities[0].team_label is RelativeTeamLabel.UNKNOWN


def test_exactly_23_physical_identities_without_forcing_11_11_1():
    tracks = []
    for i in range(24):
        # All simultaneous: they cannot be chained as continuations, so the
        # exact 23-person roster must select 23 distinct physical people.
        tracks.append(
            segment(
                i + 1,
                i + 1,
                [
                    obs(0, 0.0, float(i * 3), 0.0),
                    obs(10, 1.0, float(i * 3), 0.5),
                ],
                team=(
                    RelativeTeamLabel.TEAM_A
                    if i < 8
                    else RelativeTeamLabel.TEAM_B
                    if i < 16
                    else RelativeTeamLabel.UNKNOWN
                ),
                team_conf=(0.95 if i < 16 else 0.0),
            )
        )

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(roster_size=23, mip_time_limit_s=10.0),
    )

    assert result.solver_status == "optimal"
    assert len(result.identities) == 23
    assert len(result.segment_to_identity) == 23
    assert len(result.unassigned_segment_ids) == 1


def test_zero_corridor_outsider_cannot_establish_identity():
    tracks = [
        segment(
            1, 1,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.5)],
        ),
        segment(
            2, 2,
            [obs(20, 2.0, 50.0), obs(30, 3.0, 51.0)],
            inside=0,
            allowed=0,
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(roster_size=1, mip_time_limit_s=5.0),
    )

    assert 1 in result.segment_to_identity
    assert 2 not in result.segment_to_identity


def test_simultaneous_same_team_motion_can_be_reported_as_duplicate_hypothesis():
    left_rows = []
    right_rows = []
    for frame in range(0, 20):
        t = frame / 10.0
        left_rows.append(obs(frame, t, frame * 0.30, frame * 0.10))
        right_rows.append(obs(frame, t, frame * 0.30 + 4.0, frame * 0.10 + 3.0))

    tracks = [
        segment(
            1, 1, left_rows,
            team=RelativeTeamLabel.TEAM_B,
            team_conf=0.95,
            reid=proto(1.0, 0.1),
        ),
        segment(
            2, 2, right_rows,
            team=RelativeTeamLabel.TEAM_B,
            team_conf=0.95,
            reid=proto(0.99, 0.1),
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(
            roster_size=1,
            duplicate_min_shared_frames=8,
            duplicate_max_median_distance_m=8.0,
            duplicate_min_movement_cosine=0.8,
            mip_time_limit_s=5.0,
        ),
    )

    assert result.duplicate_candidates
    duplicate = result.duplicate_candidates[0]
    assert duplicate.kind is PhysicalRelationKind.DUPLICATE
    assert duplicate.movement_cosine_median is not None
    assert duplicate.movement_cosine_median > 0.95


def test_segments_created_from_same_raw_switch_are_not_glued_back_together():
    tracks = [
        segment(
            1, 27,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.5)],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(1.0, 0.0),
            segment_index=0,
        ),
        segment(
            2, 27,
            [obs(20, 2.0, 1.0), obs(21, 2.1, 1.5)],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(1.0, 0.0),
            segment_index=1,
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(roster_size=1, mip_time_limit_s=5.0),
    )

    assert not ({1, 2} <= set(result.segment_to_identity))


def test_contaminated_segment_cannot_establish_physical_identity():
    tracks = [
        segment(
            1,
            1,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.5)],
            quality=PhysicalSegmentQuality.CLEAN,
        ),
        segment(
            2,
            2,
            [
                obs(0, 0.0, 10.0),
                obs(1, 0.1, 10.2),
                obs(2, 0.2, 10.4),
                obs(3, 0.3, 10.6),
            ],
            quality=PhysicalSegmentQuality.CONTAMINATED,
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(
            roster_size=1,
            mip_time_limit_s=5.0,
        ),
    )

    assert 1 in result.segment_to_identity
    assert 2 not in result.segment_to_identity


def test_ambiguous_segment_cannot_establish_identity_by_itself():
    tracks = [
        segment(
            1,
            1,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.4)],
            quality=PhysicalSegmentQuality.CLEAN,
        ),
        segment(
            2,
            2,
            [obs(0, 0.0, 20.0), obs(1, 0.1, 20.4)],
            quality=PhysicalSegmentQuality.AMBIGUOUS,
            quality_score=0.6,
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(
            roster_size=1,
            mip_time_limit_s=5.0,
        ),
    )

    # The one physical path must start from CLEAN evidence.
    assert 1 in result.segment_to_identity
    assert result.identities[0].member_segment_ids[0] == 1


def test_ambiguous_segment_may_attach_at_tail_but_cannot_bridge():
    tracks = [
        segment(
            1,
            1,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.4)],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(1, 0),
            quality=PhysicalSegmentQuality.CLEAN,
        ),
        segment(
            2,
            2,
            [obs(20, 2.0, 1.0), obs(21, 2.1, 1.4)],
            team=RelativeTeamLabel.UNKNOWN,
            reid=proto(1, 0.02),
            quality=PhysicalSegmentQuality.AMBIGUOUS,
            quality_score=0.7,
        ),
        segment(
            3,
            3,
            [obs(40, 4.0, 2.0), obs(41, 4.1, 2.4)],
            team=RelativeTeamLabel.TEAM_A,
            team_conf=0.95,
            reid=proto(1, 0.01),
            quality=PhysicalSegmentQuality.CLEAN,
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(
            roster_size=1,
            mip_time_limit_s=5.0,
        ),
    )

    # There must never be an outgoing continuation from ambiguous segment 2.
    assert all(
        edge.left_segment_id != 2
        for edge in result.continuation_candidates
    )


def test_contaminated_segment_never_creates_or_bridges_relation():
    tracks = [
        segment(
            1,
            1,
            [obs(0, 0.0, 0.0), obs(1, 0.1, 0.5)],
            quality=PhysicalSegmentQuality.CLEAN,
        ),
        segment(
            2,
            2,
            [obs(20, 2.0, 1.0), obs(21, 2.1, 1.5)],
            quality=PhysicalSegmentQuality.CONTAMINATED,
        ),
        segment(
            3,
            3,
            [obs(40, 4.0, 2.0), obs(41, 4.1, 2.5)],
            quality=PhysicalSegmentQuality.CLEAN,
        ),
    ]

    result = solve_physical_identities(
        tracks,
        config=PhysicalIdentitySolverConfig(
            roster_size=1,
            mip_time_limit_s=5.0,
        ),
    )

    assert all(
        edge.left_segment_id != 2 and edge.right_segment_id != 2
        for edge in result.continuation_candidates
    )
    assert 2 not in result.segment_to_identity
