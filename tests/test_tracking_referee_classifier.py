import numpy as np

from tracking_v0.physical_identity_solver import (
    PhysicalIdentity,
    PhysicalObservation,
    PhysicalSegmentQuality,
    PhysicalTrackSegment,
)
from tracking_v0.referee_classifier import (
    RefereeClassifierConfig,
    classify_referee_identity,
)
from tracking_v0.team_similarity import RelativeTeamLabel


def unit(x, y):
    vector = np.asarray([x, y], dtype=np.float32)
    return vector / np.linalg.norm(vector)


def make_segment(
    segment_id,
    raw_id,
    *,
    x,
    z=0.0,
    frames=8,
    team=RelativeTeamLabel.UNKNOWN,
):
    observations = tuple(
        PhysicalObservation(
            frame_index=i,
            timestamp=i / 10.0,
            x=x,
            z=z + 0.05 * i,
        )
        for i in range(frames)
    )
    return PhysicalTrackSegment(
        segment_id=segment_id,
        raw_track_id=raw_id,
        segment_index=0,
        observations=observations,
        team_label=team,
        team_confidence=(1.0 if team is not RelativeTeamLabel.UNKNOWN else 0.0),
        identity_quality=PhysicalSegmentQuality.CLEAN,
        identity_quality_score=0.95,
        inside_observed_count=frames,
        continuation_observed_count=frames,
    )


def make_identity(identity_id, segment_ids, team):
    return PhysicalIdentity(
        identity_id=identity_id,
        member_segment_ids=tuple(segment_ids),
        member_keys=tuple(f"{segment_id}:0" for segment_id in segment_ids),
        raw_track_ids=tuple(segment_ids),
        team_label=team,
    )


def test_overrepresented_team_outlier_is_selected_as_referee():
    segments = {}
    identities = []
    kits = {}

    next_segment = 1

    # Eleven ordinary TEAM_A players.
    for identity_id in range(1, 12):
        segment = make_segment(
            next_segment,
            next_segment,
            x=-10.0 + identity_id,
            team=RelativeTeamLabel.TEAM_A,
        )
        segments[next_segment] = segment
        identities.append(
            make_identity(
                identity_id,
                [next_segment],
                RelativeTeamLabel.TEAM_A,
            )
        )
        kits[next_segment] = unit(1.0, 0.03 * (identity_id % 3))
        next_segment += 1

    # Twelfth TEAM_A-labelled person is actually the referee: same coarse team
    # label, but full-path kit appearance is an outlier.
    referee_identity_id = 12
    segment = make_segment(
        next_segment,
        next_segment,
        x=0.0,
        team=RelativeTeamLabel.TEAM_A,
        frames=12,
    )
    segments[next_segment] = segment
    identities.append(
        make_identity(
            referee_identity_id,
            [next_segment],
            RelativeTeamLabel.TEAM_A,
        )
    )
    kits[next_segment] = unit(0.68, 0.73)
    next_segment += 1

    # Ten ordinary TEAM_B players.
    for identity_id in range(13, 23):
        segment = make_segment(
            next_segment,
            next_segment,
            x=5.0,
            team=RelativeTeamLabel.TEAM_B,
        )
        segments[next_segment] = segment
        identities.append(
            make_identity(
                identity_id,
                [next_segment],
                RelativeTeamLabel.TEAM_B,
            )
        )
        kits[next_segment] = unit(0.02 * (identity_id % 3), 1.0)
        next_segment += 1

    # One UNKNOWN special-kit goalkeeper close to the goal.
    segment = make_segment(
        next_segment,
        next_segment,
        x=50.5,
        team=RelativeTeamLabel.UNKNOWN,
        frames=12,
    )
    segments[next_segment] = segment
    identities.append(
        make_identity(
            23,
            [next_segment],
            RelativeTeamLabel.UNKNOWN,
        )
    )
    kits[next_segment] = unit(-1.0, 0.1)

    result = classify_referee_identity(
        identities,
        segment_by_id=segments,
        kit_prototype_by_segment_id=kits,
        config=RefereeClassifierConfig(
            expected_team_size=11,
        ),
    )

    assert result.referee_identity_id == referee_identity_id
    assert len(result.player_identity_ids) == 22

    winner = result.candidates[0]
    assert winner.roster_structure_feasible
    assert winner.team_a_after_removal == 11
    assert winner.team_b_after_removal == 10
    assert winner.unknown_after_removal == 1


def test_unknown_referee_beats_unknown_goalkeeper():
    segments = {}
    identities = []
    kits = {}
    next_segment = 1

    # 11 A + 10 B.
    for identity_id in range(1, 12):
        segment = make_segment(
            next_segment,
            next_segment,
            x=-5.0,
            team=RelativeTeamLabel.TEAM_A,
        )
        segments[next_segment] = segment
        identities.append(
            make_identity(
                identity_id,
                [next_segment],
                RelativeTeamLabel.TEAM_A,
            )
        )
        kits[next_segment] = unit(1.0, 0.02)
        next_segment += 1

    for identity_id in range(12, 22):
        segment = make_segment(
            next_segment,
            next_segment,
            x=5.0,
            team=RelativeTeamLabel.TEAM_B,
        )
        segments[next_segment] = segment
        identities.append(
            make_identity(
                identity_id,
                [next_segment],
                RelativeTeamLabel.TEAM_B,
            )
        )
        kits[next_segment] = unit(0.02, 1.0)
        next_segment += 1

    # UNKNOWN referee in central field.
    referee_segment = next_segment
    segment = make_segment(
        referee_segment,
        referee_segment,
        x=0.0,
        z=4.0,
        frames=12,
        team=RelativeTeamLabel.UNKNOWN,
    )
    segments[referee_segment] = segment
    identities.append(
        make_identity(
            22,
            [referee_segment],
            RelativeTeamLabel.UNKNOWN,
        )
    )
    kits[referee_segment] = unit(0.67, 0.74)
    next_segment += 1

    # UNKNOWN goalkeeper near goal.
    keeper_segment = next_segment
    segment = make_segment(
        keeper_segment,
        keeper_segment,
        x=51.0,
        z=0.0,
        frames=12,
        team=RelativeTeamLabel.UNKNOWN,
    )
    segments[keeper_segment] = segment
    identities.append(
        make_identity(
            23,
            [keeper_segment],
            RelativeTeamLabel.UNKNOWN,
        )
    )
    kits[keeper_segment] = unit(-0.8, 0.6)

    result = classify_referee_identity(
        identities,
        segment_by_id=segments,
        kit_prototype_by_segment_id=kits,
    )

    assert result.referee_identity_id == 22

    by_id = {
        row.identity_id: row
        for row in result.candidates
    }
    assert by_id[22].non_goalkeeper_score > by_id[23].non_goalkeeper_score
    assert by_id[23].goal_zone_fraction > 0.9


def test_referee_may_be_multi_segment_identity():
    segments = {}
    kits = {}

    # Two ordinary A players, two ordinary B players, plus one TEAM_A-labelled
    # referee made of two physical segments.  expected_team_size=2 means that
    # removing the referee restores a structurally valid 2+2 player roster.
    s1 = make_segment(1, 1, x=-12, team=RelativeTeamLabel.TEAM_A)
    s2 = make_segment(2, 2, x=-7, team=RelativeTeamLabel.TEAM_A)
    s3 = make_segment(3, 3, x=8, team=RelativeTeamLabel.TEAM_B)
    s4 = make_segment(4, 4, x=13, team=RelativeTeamLabel.TEAM_B)
    s5 = make_segment(5, 5, x=0, team=RelativeTeamLabel.TEAM_A, frames=5)
    s6 = make_segment(6, 6, x=3, team=RelativeTeamLabel.TEAM_A, frames=5)

    for segment in (s1, s2, s3, s4, s5, s6):
        segments[segment.segment_id] = segment

    kits[1] = unit(1.0, 0.00)
    kits[2] = unit(1.0, 0.04)
    kits[3] = unit(0.00, 1.0)
    kits[4] = unit(0.04, 1.0)
    kits[5] = unit(-0.65, -0.76)
    kits[6] = unit(-0.70, -0.71)

    identities = [
        make_identity(1, [1], RelativeTeamLabel.TEAM_A),
        make_identity(2, [2], RelativeTeamLabel.TEAM_A),
        make_identity(3, [3], RelativeTeamLabel.TEAM_B),
        make_identity(4, [4], RelativeTeamLabel.TEAM_B),
        PhysicalIdentity(
            identity_id=5,
            member_segment_ids=(5, 6),
            member_keys=("5:0", "6:0"),
            raw_track_ids=(5, 6),
            team_label=RelativeTeamLabel.TEAM_A,
        ),
    ]

    result = classify_referee_identity(
        identities,
        segment_by_id=segments,
        kit_prototype_by_segment_id=kits,
        config=RefereeClassifierConfig(
            expected_team_size=2,
            full_support_inside_observations=5,
        ),
    )

    assert result.referee_identity_id == 5
    assert result.candidates[0].member_segment_ids == (5, 6)



def test_result_never_changes_physical_identity_membership():
    segments = {
        1: make_segment(1, 1, x=-5, team=RelativeTeamLabel.TEAM_A),
        2: make_segment(2, 2, x=5, team=RelativeTeamLabel.TEAM_B),
        3: make_segment(3, 3, x=0, team=RelativeTeamLabel.TEAM_A),
    }
    identities = [
        make_identity(1, [1], RelativeTeamLabel.TEAM_A),
        make_identity(2, [2], RelativeTeamLabel.TEAM_B),
        make_identity(3, [3], RelativeTeamLabel.TEAM_A),
    ]
    kits = {
        1: unit(1, 0),
        2: unit(0, 1),
        3: unit(0.7, 0.7),
    }

    original = {
        identity.identity_id: identity.member_segment_ids
        for identity in identities
    }

    result = classify_referee_identity(
        identities,
        segment_by_id=segments,
        kit_prototype_by_segment_id=kits,
        config=RefereeClassifierConfig(
            expected_team_size=1,
        ),
    )

    assert len(result.player_identity_ids) == 2
    assert set(result.player_identity_ids) | {result.referee_identity_id} == {
        1, 2, 3
    }
    assert original == {
        identity.identity_id: identity.member_segment_ids
        for identity in identities
    }
