from __future__ import annotations

from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample
from tracking_v0.quality import (
    TrackQualityDecision,
    analyze_track_quality,
    analyze_trajectories,
    filter_trajectories,
    signed_field_depth,
)


FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0
DT = 1.0 / 15.0


def sample(
    track_id: int,
    index: int,
    *,
    x: float,
    z: float,
    kind: PositionKind = PositionKind.OBSERVED,
    confidence: float | None = 0.95,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=index,
        timestamp=index * DT,
        x=x,
        z=z,
        vx=0.0,
        vz=0.0,
        lifecycle=TrackLifecycle.ACTIVE,
        position_kind=kind,
        measurement_id=f"{index}:1" if kind is PositionKind.OBSERVED else None,
        confidence=confidence if kind is PositionKind.OBSERVED else None,
    )


def observed_track(
    track_id: int,
    *,
    count: int,
    x: float,
    z: float,
    start_index: int = 0,
) -> list[TrackSample]:
    return [
        sample(track_id, start_index + index, x=x, z=z)
        for index in range(count)
    ]


def test_signed_field_depth_uses_nearest_field_boundary() -> None:
    assert signed_field_depth(
        0.0,
        0.0,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    ) == 34.0

    assert signed_field_depth(
        FIELD_LENGTH / 2.0,
        0.0,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    ) == 0.0

    assert signed_field_depth(
        FIELD_LENGTH / 2.0 + 0.4,
        0.0,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    ) == -0.3999999999999986


def test_persistent_player_is_kept() -> None:
    samples = observed_track(
        1,
        count=150,
        x=0.0,
        z=0.0,
    )

    result = analyze_track_quality(
        samples,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert result.decision is TrackQualityDecision.KEEP
    assert result.accepted
    assert "convincing_deep_field_presence" in result.reasons
    assert result.metrics.observed_fraction == 1.0
    assert result.metrics.max_field_depth_m > 20.0


def test_touchline_player_is_not_rejected() -> None:
    # A winger can spend a long period only 30 cm inside the touchline.
    # Long, stable strict-field presence must protect this trajectory.
    z = FIELD_WIDTH / 2.0 - 0.30
    samples = observed_track(
        2,
        count=180,
        x=5.0,
        z=z,
    )

    result = analyze_track_quality(
        samples,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert result.decision is TrackQualityDecision.KEEP
    assert result.accepted
    assert "long_stable_strict_field_presence" in result.reasons
    assert result.metrics.edge_observed_fraction == 1.0
    assert result.metrics.strict_observed_fraction == 1.0


def test_goalkeeper_near_end_line_is_not_rejected() -> None:
    # A goalkeeper can legitimately stay close to the end line for a long time.
    x = FIELD_LENGTH / 2.0 - 0.50
    samples = observed_track(
        3,
        count=180,
        x=x,
        z=0.0,
    )

    result = analyze_track_quality(
        samples,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert result.decision is TrackQualityDecision.KEEP
    assert result.accepted
    assert result.metrics.edge_observed_fraction == 1.0
    assert result.metrics.strict_observed_fraction == 1.0


def test_photographer_grazing_field_for_three_frames_is_rejected() -> None:
    # Most of the temporal track is prediction/coasting. The person is only
    # observed for three frames and the estimated footpoint barely enters.
    samples: list[TrackSample] = []

    for index in range(30):
        if index in {10, 11, 12}:
            samples.append(
                sample(
                    24,
                    index,
                    x=0.0,
                    z=FIELD_WIDTH / 2.0 - 0.10,
                    kind=PositionKind.OBSERVED,
                )
            )
        else:
            samples.append(
                sample(
                    24,
                    index,
                    x=0.0,
                    z=FIELD_WIDTH / 2.0 + 0.15,
                    kind=PositionKind.PREDICTED,
                )
            )

    result = analyze_track_quality(
        samples,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert result.decision is TrackQualityDecision.REJECT
    assert not result.accepted
    assert "edge_confined" in result.reasons
    assert "sparse_observations" in result.reasons
    assert "never_convincingly_inside" in result.reasons
    assert result.metrics.observed_count == 3
    assert result.metrics.max_field_depth_m <= 0.10 + 1e-9


def test_coach_briefly_stepping_near_touchline_is_rejected() -> None:
    # Twenty real observations over ~1.3 s, embedded in a longer coasting track.
    # The trajectory never gets more than 20 cm inside the strict field.
    samples: list[TrackSample] = []

    for index in range(50):
        if 10 <= index < 30:
            samples.append(
                sample(
                    25,
                    index,
                    x=8.0,
                    z=FIELD_WIDTH / 2.0 - 0.20,
                    kind=PositionKind.OBSERVED,
                )
            )
        else:
            samples.append(
                sample(
                    25,
                    index,
                    x=8.0,
                    z=FIELD_WIDTH / 2.0 + 0.10,
                    kind=PositionKind.PREDICTED,
                )
            )

    result = analyze_track_quality(
        samples,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert result.decision is TrackQualityDecision.REJECT
    assert not result.accepted
    assert "temporary_sideline_presence" in result.reasons
    assert result.metrics.observed_count == 20
    assert result.metrics.observed_fraction < 0.55
    assert result.metrics.max_field_depth_m <= 0.20 + 1e-9


def test_short_ambiguous_track_is_preserved_as_uncertain() -> None:
    # This person is short-lived, but is a full metre inside the field.
    # That is neither strong enough for KEEP nor safe enough for REJECT.
    samples = observed_track(
        26,
        count=5,
        x=0.0,
        z=FIELD_WIDTH / 2.0 - 1.0,
    )

    result = analyze_track_quality(
        samples,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert result.decision is TrackQualityDecision.UNCERTAIN
    assert result.accepted
    assert "no_deep_field_evidence" in result.reasons


def test_filter_removes_only_rejected_tracks() -> None:
    player = observed_track(1, count=150, x=0.0, z=0.0)
    uncertain = observed_track(
        26,
        count=5,
        x=0.0,
        z=FIELD_WIDTH / 2.0 - 1.0,
    )

    photographer: list[TrackSample] = []
    for index in range(30):
        photographer.append(
            sample(
                24,
                index,
                x=0.0,
                z=FIELD_WIDTH / 2.0 - 0.10,
                kind=(
                    PositionKind.OBSERVED
                    if index in {10, 11, 12}
                    else PositionKind.PREDICTED
                ),
            )
        )

    trajectories = {
        1: player,
        24: photographer,
        26: uncertain,
    }

    report = analyze_trajectories(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )
    filtered = filter_trajectories(trajectories, report)

    assert report.kept_track_ids == (1,)
    assert report.rejected_track_ids == (24,)
    assert report.uncertain_track_ids == (26,)
    assert report.accepted_track_ids == (1, 26)
    assert set(filtered) == {1, 26}
