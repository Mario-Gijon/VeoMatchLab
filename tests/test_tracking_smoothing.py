from __future__ import annotations

from math import hypot, isfinite

import pytest

from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample
from tracking_v0.smoothing import (
    SmoothingConfig,
    smooth_trajectories,
    smooth_trajectory,
)


DT = 1.0 / 15.0


def sample(
    track_id: int,
    frame_index: int,
    *,
    timestamp: float | None = None,
    x: float,
    z: float = 0.0,
    vx: float = 0.0,
    vz: float = 0.0,
    kind: PositionKind = PositionKind.OBSERVED,
    confidence: float | None = 0.95,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=(
            frame_index * DT
            if timestamp is None
            else timestamp
        ),
        x=x,
        z=z,
        vx=vx,
        vz=vz,
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


def test_single_sample_is_preserved() -> None:
    raw = (
        sample(
            7,
            0,
            x=12.0,
            z=-4.0,
            vx=1.5,
            vz=-0.5,
        ),
    )

    result = smooth_trajectory(raw)

    assert result == raw


def test_rts_uses_future_observation_to_correct_predicted_gap() -> None:
    """
    The forward tracker emitted nonsense positions in the coast gap.

    Because PREDICTED samples are not measurements, the offline smoother should
    ignore those raw x/z values and reconstruct the gap using the observations
    before and after it.
    """
    raw: list[TrackSample] = []

    # Stable observations before the gap: true motion x ~= t.
    for frame in range(0, 6):
        t = frame * DT
        raw.append(
            sample(
                2,
                frame,
                x=t,
                vx=1.0,
                kind=PositionKind.OBSERVED,
            )
        )

    # Deliberately absurd forward predictions.
    for frame in range(6, 12):
        raw.append(
            sample(
                2,
                frame,
                x=40.0 + frame,
                vx=12.0,
                kind=PositionKind.PREDICTED,
            )
        )

    # Real observations resume on the original constant-velocity path.
    for frame in range(12, 19):
        t = frame * DT
        raw.append(
            sample(
                2,
                frame,
                x=t,
                vx=1.0,
                kind=PositionKind.OBSERVED,
            )
        )

    smoothed = smooth_trajectory(raw)

    by_frame = {
        item.frame_index: item
        for item in smoothed
    }

    for frame in range(6, 12):
        truth_x = frame * DT
        raw_error = abs((40.0 + frame) - truth_x)
        smooth_error = abs(by_frame[frame].x - truth_x)

        assert smooth_error < 1.0
        assert smooth_error < raw_error * 0.05
        assert by_frame[frame].position_kind is PositionKind.PREDICTED


def test_future_information_changes_earlier_state_not_just_last_frame() -> None:
    """
    Verify this is genuinely a backward RTS pass rather than only another
    forward filter.
    """
    raw: list[TrackSample] = []

    # Person appears stationary initially.
    for frame in range(0, 5):
        raw.append(
            sample(
                10,
                frame,
                x=0.0,
                vx=0.0,
            )
        )

    # Coast gap.
    for frame in range(5, 10):
        raw.append(
            sample(
                10,
                frame,
                x=0.0,
                vx=0.0,
                kind=PositionKind.PREDICTED,
            )
        )

    # Future observations reveal that the player had moved.
    for frame in range(10, 15):
        raw.append(
            sample(
                10,
                frame,
                x=3.0 + (frame - 10) * 0.15,
                vx=2.25,
            )
        )

    smoothed = smooth_trajectory(
        raw,
        config=SmoothingConfig(
            process_acceleration_std_m_s2=4.0,
        ),
    )

    by_frame = {
        item.frame_index: item
        for item in smoothed
    }

    # A state before the future observations should be revised away from the
    # original zero-only forward estimate.
    assert by_frame[8].x > 0.05
    assert by_frame[8].vx > 0.05


def test_metadata_and_sample_identity_are_preserved() -> None:
    raw = [
        sample(
            12,
            0,
            x=0.0,
            z=1.0,
            kind=PositionKind.OBSERVED,
            confidence=0.87,
        ),
        sample(
            12,
            1,
            x=0.1,
            z=1.0,
            kind=PositionKind.PREDICTED,
        ),
        sample(
            12,
            2,
            x=0.2,
            z=1.0,
            kind=PositionKind.OBSERVED,
            confidence=0.91,
        ),
    ]

    smoothed = smooth_trajectory(raw)

    assert len(smoothed) == len(raw)

    for before, after in zip(raw, smoothed):
        assert after.track_id == before.track_id
        assert after.frame_index == before.frame_index
        assert after.timestamp == before.timestamp
        assert after.position_kind is before.position_kind
        assert after.lifecycle is before.lifecycle
        assert after.measurement_id == before.measurement_id
        assert after.confidence == before.confidence

        assert all(
            isfinite(value)
            for value in (
                after.x,
                after.z,
                after.vx,
                after.vz,
            )
        )


def test_variable_dt_sequence_is_supported() -> None:
    raw = [
        sample(
            4,
            0,
            timestamp=0.0,
            x=0.0,
            vx=1.0,
        ),
        sample(
            4,
            1,
            timestamp=0.10,
            x=0.10,
            vx=1.0,
        ),
        sample(
            4,
            2,
            timestamp=0.35,
            x=9.0,
            vx=5.0,
            kind=PositionKind.PREDICTED,
        ),
        sample(
            4,
            3,
            timestamp=0.70,
            x=0.70,
            vx=1.0,
        ),
    ]

    smoothed = smooth_trajectory(raw)

    assert len(smoothed) == 4
    assert smoothed[2].timestamp == 0.35
    assert abs(smoothed[2].x - 0.35) < 0.75


def test_predicted_samples_are_not_used_as_measurements() -> None:
    """
    Two trajectories differ only in the raw x/z of their PREDICTED sample.
    Their smoothed states must be identical because predictions are never fed
    back as observations.
    """
    common_start = sample(
        5,
        0,
        x=0.0,
        vx=1.0,
    )
    common_end = sample(
        5,
        2,
        x=2.0 * DT,
        vx=1.0,
    )

    a = (
        common_start,
        sample(
            5,
            1,
            x=500.0,
            z=-500.0,
            vx=100.0,
            vz=-100.0,
            kind=PositionKind.PREDICTED,
        ),
        common_end,
    )

    b = (
        common_start,
        sample(
            5,
            1,
            x=-800.0,
            z=900.0,
            vx=-120.0,
            vz=70.0,
            kind=PositionKind.PREDICTED,
        ),
        common_end,
    )

    smooth_a = smooth_trajectory(a)
    smooth_b = smooth_trajectory(b)

    for left, right in zip(smooth_a, smooth_b):
        assert left.x == pytest.approx(right.x, abs=1e-12)
        assert left.z == pytest.approx(right.z, abs=1e-12)
        assert left.vx == pytest.approx(right.vx, abs=1e-12)
        assert left.vz == pytest.approx(right.vz, abs=1e-12)


def test_batch_smoothing_preserves_track_ids_and_reports_metrics() -> None:
    trajectories = {
        1: tuple(
            sample(
                1,
                frame,
                x=frame * DT,
                vx=1.0,
            )
            for frame in range(20)
        ),
        2: tuple(
            sample(
                2,
                frame,
                x=10.0 + frame * DT,
                z=4.0,
                vx=1.0,
            )
            for frame in range(20)
        ),
    }

    smoothed, report = smooth_trajectories(trajectories)

    assert set(smoothed) == {1, 2}
    assert report.track_ids == (1, 2)

    for track_id in (1, 2):
        metrics = report.tracks[track_id]

        assert metrics.track_id == track_id
        assert metrics.sample_count == 20
        assert metrics.observed_count == 20
        assert metrics.predicted_count == 0
        assert metrics.interpolated_count == 0
        assert metrics.duration_s > 0.0
        assert metrics.median_sample_period_s == pytest.approx(DT)
        assert metrics.longest_sample_gap_s == pytest.approx(DT)

        assert metrics.observed_rms_correction_m >= 0.0
        assert metrics.observed_max_correction_m >= 0.0
        assert metrics.raw_max_speed_m_s >= 0.0
        assert metrics.smoothed_max_speed_m_s >= 0.0


def test_smoothing_reduces_extreme_predicted_speed_artifact() -> None:
    raw = [
        sample(
            8,
            0,
            x=0.0,
            vx=1.0,
        ),
        sample(
            8,
            1,
            x=100.0,
            vx=50.0,
            kind=PositionKind.PREDICTED,
        ),
        sample(
            8,
            2,
            x=2.0 * DT,
            vx=1.0,
        ),
        sample(
            8,
            3,
            x=3.0 * DT,
            vx=1.0,
        ),
    ]

    smoothed, report = smooth_trajectories(
        {8: raw},
    )

    metrics = report.tracks[8]

    assert metrics.raw_max_speed_m_s >= 50.0
    assert metrics.smoothed_max_speed_m_s < metrics.raw_max_speed_m_s

    assert max(
        hypot(item.vx, item.vz)
        for item in smoothed[8]
    ) < 10.0


def test_duplicate_timestamps_are_rejected() -> None:
    raw = [
        sample(
            9,
            0,
            timestamp=1.0,
            x=0.0,
        ),
        sample(
            9,
            1,
            timestamp=1.0,
            x=0.1,
        ),
    ]

    with pytest.raises(
        ValueError,
        match="timestamps must be strictly increasing",
    ):
        smooth_trajectory(raw)
