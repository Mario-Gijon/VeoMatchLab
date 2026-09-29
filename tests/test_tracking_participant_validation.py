from __future__ import annotations

from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample
from tracking_v0.participant_validation import (
    ParticipantDecision,
    filter_participants,
    validate_participants,
)


DT = 1.0 / 15.0
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def sample(
    track_id: int,
    frame_index: int,
    *,
    x: float,
    z: float,
    kind: PositionKind = PositionKind.OBSERVED,
    confidence: float | None = 0.95,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=frame_index * DT,
        x=x,
        z=z,
        vx=0.0,
        vz=0.0,
        lifecycle=TrackLifecycle.ACTIVE,
        position_kind=kind,
        measurement_id=(
            f"{frame_index}:{track_id}"
            if kind is PositionKind.OBSERVED
            else None
        ),
        confidence=(
            confidence
            if kind is PositionKind.OBSERVED
            else None
        ),
    )


def persistent_track(
    track_id: int,
    *,
    frames: int,
    x: float,
    z: float,
) -> list[TrackSample]:
    return [
        sample(
            track_id,
            frame,
            x=x,
            z=z,
            kind=PositionKind.OBSERVED,
        )
        for frame in range(frames)
    ]


def test_stable_player_is_kept() -> None:
    trajectories = {
        1: persistent_track(
            1,
            frames=150,
            x=0.0,
            z=0.0,
        ),
    }

    report = validate_participants(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[1]

    assert result.decision is ParticipantDecision.KEEP
    assert result.accepted
    assert result.reasons == ("stable_resolved_participant",)
    assert result.metrics.observed_count == 150
    assert result.metrics.strict_observed_fraction == 1.0


def test_touchline_player_is_not_rejected() -> None:
    trajectories = {
        7: persistent_track(
            7,
            frames=150,
            x=0.0,
            z=FIELD_WIDTH / 2.0 - 0.20,
        ),
    }

    report = validate_participants(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[7]

    assert result.decision is ParticipantDecision.KEEP
    assert result.accepted
    assert result.metrics.strict_observed_fraction == 1.0
    assert result.metrics.max_field_depth_m <= 0.20 + 1e-9


def test_goalkeeper_near_goal_line_is_not_rejected() -> None:
    trajectories = {
        13: persistent_track(
            13,
            frames=150,
            x=FIELD_LENGTH / 2.0 - 0.35,
            z=0.0,
        ),
    }

    report = validate_participants(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[13]

    assert result.decision is ParticipantDecision.KEEP
    assert result.accepted
    assert result.metrics.strict_observed_fraction == 1.0


def test_persistent_person_clearly_outside_field_is_rejected() -> None:
    trajectories = {
        32: persistent_track(
            32,
            frames=55,
            x=0.0,
            z=FIELD_WIDTH / 2.0 + 2.25,
        ),
    }

    report = validate_participants(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[32]

    assert result.decision is ParticipantDecision.REJECT
    assert not result.accepted
    assert "persistent_outside_field" in result.reasons
    assert result.metrics.observed_count == 55
    assert result.metrics.strict_observed_fraction == 0.0
    assert result.metrics.max_field_depth_m <= -2.0


def test_sparse_orphan_with_two_real_observations_is_rejected() -> None:
    samples: list[TrackSample] = []

    for frame in range(15):
        kind = (
            PositionKind.OBSERVED
            if frame in {3, 10}
            else PositionKind.PREDICTED
        )
        samples.append(
            sample(
                28,
                frame,
                x=8.0,
                z=5.0,
                kind=kind,
            )
        )

    report = validate_participants(
        {28: samples},
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[28]

    assert result.decision is ParticipantDecision.REJECT
    assert not result.accepted
    assert "ephemeral_sparse_orphan" in result.reasons
    assert result.metrics.observed_count == 2
    assert result.metrics.observed_fraction < 0.35
    assert result.metrics.longest_observed_run_seconds <= 0.10


def test_three_consecutive_real_observations_with_long_coasting_are_rejected() -> None:
    samples: list[TrackSample] = []

    for frame in range(18):
        kind = (
            PositionKind.OBSERVED
            if frame in {6, 7, 8}
            else PositionKind.PREDICTED
        )
        samples.append(
            sample(
                33,
                frame,
                x=6.0,
                z=4.0,
                kind=kind,
            )
        )

    report = validate_participants(
        {33: samples},
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[33]

    assert result.decision is ParticipantDecision.REJECT
    assert not result.accepted
    assert result.metrics.observed_count == 3
    assert result.metrics.observed_fraction < 0.35
    assert result.metrics.longest_observed_run_seconds <= 0.30


def test_short_but_well_observed_inside_track_remains_uncertain() -> None:
    # A real person may only enter the clip near the end. Five consecutive
    # observations clearly inside the field are not enough for KEEP, but are
    # too much real support for the sparse-orphan rejection rule.
    trajectories = {
        40: persistent_track(
            40,
            frames=5,
            x=12.0,
            z=3.0,
        ),
    }

    report = validate_participants(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[40]

    assert result.decision is ParticipantDecision.UNCERTAIN
    assert result.accepted


def test_resolved_identity_with_fragment_replacement_is_kept() -> None:
    # Simulate a canonical identity after resolution: most of the clip exists,
    # with a short predicted stretch that has already been partially replaced
    # by real observations from a fragment.
    samples: list[TrackSample] = []

    for frame in range(150):
        if 60 <= frame <= 64:
            kind = PositionKind.PREDICTED
        else:
            kind = PositionKind.OBSERVED

        samples.append(
            sample(
                2,
                frame,
                x=10.0,
                z=5.0,
                kind=kind,
            )
        )

    report = validate_participants(
        {2: samples},
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    result = report.results[2]

    assert result.decision is ParticipantDecision.KEEP
    assert result.accepted
    assert result.metrics.observed_count == 145
    assert result.metrics.predicted_count == 5


def test_filter_removes_only_rejected_identities() -> None:
    player = persistent_track(
        1,
        frames=150,
        x=0.0,
        z=0.0,
    )

    outside = persistent_track(
        32,
        frames=55,
        x=0.0,
        z=FIELD_WIDTH / 2.0 + 2.25,
    )

    uncertain = persistent_track(
        40,
        frames=5,
        x=12.0,
        z=3.0,
    )

    trajectories = {
        1: player,
        32: outside,
        40: uncertain,
    }

    report = validate_participants(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )
    filtered = filter_participants(
        trajectories,
        report,
    )

    assert report.kept_track_ids == (1,)
    assert report.rejected_track_ids == (32,)
    assert report.uncertain_track_ids == (40,)
    assert report.accepted_track_ids == (1, 40)
    assert set(filtered) == {1, 40}
