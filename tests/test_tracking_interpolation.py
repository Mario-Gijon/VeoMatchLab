from __future__ import annotations

from math import hypot, isfinite

import pytest

from tracking.interpolation import (
    InterpolationConfig,
    interpolate_trajectories,
    interpolate_trajectory,
)
from tracking.models import PositionKind, TrackLifecycle, TrackSample


FPS = 30.0
DT = 1.0 / FPS


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
    measurement_id: str | None = None,
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
            measurement_id
            if measurement_id is not None
            else (
                f"{frame_index}:{track_id}"
                if kind is PositionKind.OBSERVED
                else None
            )
        ),
        confidence=(
            confidence
            if kind is PositionKind.OBSERVED
            else None
        ),
    )


def test_15hz_source_samples_become_contiguous_30fps_frames() -> None:
    raw = (
        sample(
            2,
            100,
            x=0.0,
            vx=3.0,
        ),
        sample(
            2,
            102,
            x=0.2,
            vx=3.0,
        ),
        sample(
            2,
            104,
            x=0.4,
            vx=3.0,
        ),
    )

    output, metrics = interpolate_trajectory(raw)

    assert [item.frame_index for item in output] == [
        100,
        101,
        102,
        103,
        104,
    ]
    assert metrics.input_samples == 3
    assert metrics.output_samples == 5
    assert metrics.inserted_samples == 2
    assert metrics.filled_gaps == 2
    assert metrics.skipped_large_gaps == 0


def test_original_samples_are_preserved_exactly() -> None:
    raw = (
        sample(
            7,
            10,
            x=1.0,
            z=2.0,
            vx=1.5,
            vz=-0.5,
            kind=PositionKind.OBSERVED,
            confidence=0.88,
            measurement_id="10:measurement-a",
        ),
        sample(
            7,
            12,
            x=1.1,
            z=1.95,
            vx=1.6,
            vz=-0.4,
            kind=PositionKind.PREDICTED,
        ),
        sample(
            7,
            14,
            x=1.25,
            z=1.90,
            vx=1.7,
            vz=-0.3,
            kind=PositionKind.OBSERVED,
            confidence=0.92,
            measurement_id="14:measurement-b",
        ),
    )

    output, _ = interpolate_trajectory(raw)
    by_frame = {
        item.frame_index: item
        for item in output
    }

    assert by_frame[10] == raw[0]
    assert by_frame[12] == raw[1]
    assert by_frame[14] == raw[2]


def test_inserted_samples_are_marked_interpolated() -> None:
    raw = (
        sample(
            3,
            20,
            x=5.0,
            z=2.0,
            vx=2.0,
        ),
        sample(
            3,
            22,
            x=5.2,
            z=2.0,
            vx=2.0,
        ),
    )

    output, _ = interpolate_trajectory(raw)

    middle = next(
        item
        for item in output
        if item.frame_index == 21
    )

    assert middle.track_id == 3
    assert middle.position_kind is PositionKind.INTERPOLATED
    assert middle.measurement_id is None
    assert middle.confidence is None
    assert middle.lifecycle is raw[0].lifecycle


def test_constant_velocity_hermite_hits_exact_midpoint() -> None:
    left = sample(
        5,
        0,
        timestamp=0.0,
        x=0.0,
        z=0.0,
        vx=3.0,
        vz=1.5,
    )
    right = sample(
        5,
        2,
        timestamp=2.0 / 30.0,
        x=0.2,
        z=0.1,
        vx=3.0,
        vz=1.5,
    )

    output, metrics = interpolate_trajectory(
        (left, right),
    )

    middle = output[1]

    assert middle.frame_index == 1
    assert middle.timestamp == pytest.approx(1.0 / 30.0)
    assert middle.x == pytest.approx(0.1, abs=1e-12)
    assert middle.z == pytest.approx(0.05, abs=1e-12)
    assert middle.vx == pytest.approx(3.0, abs=1e-12)
    assert middle.vz == pytest.approx(1.5, abs=1e-12)
    assert metrics.max_hermite_vs_linear_deviation_m == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_large_gap_is_not_invented() -> None:
    left = sample(
        8,
        0,
        timestamp=0.0,
        x=0.0,
        vx=1.0,
    )
    right = sample(
        8,
        120,
        timestamp=4.0,
        x=4.0,
        vx=1.0,
    )

    output, metrics = interpolate_trajectory(
        (left, right),
        config=InterpolationConfig(
            max_gap_seconds=3.0,
        ),
    )

    assert output == (left, right)
    assert metrics.inserted_samples == 0
    assert metrics.filled_gaps == 0
    assert metrics.skipped_large_gaps == 1
    assert metrics.max_skipped_gap_seconds == pytest.approx(4.0)


def test_tangent_guard_prevents_extreme_loop_and_speed() -> None:
    raw = (
        sample(
            11,
            0,
            timestamp=0.0,
            x=0.0,
            z=0.0,
            vx=100.0,
            vz=100.0,
        ),
        sample(
            11,
            2,
            timestamp=2.0 / 30.0,
            x=0.10,
            z=0.0,
            vx=-100.0,
            vz=-100.0,
        ),
    )

    output, metrics = interpolate_trajectory(raw)

    middle = output[1]

    assert all(
        isfinite(value)
        for value in (
            middle.x,
            middle.z,
            middle.vx,
            middle.vz,
        )
    )
    assert hypot(middle.vx, middle.vz) <= 12.5 + 1e-9
    assert metrics.max_interpolated_speed_m_s <= 12.5 + 1e-9

    # The cubic is allowed to curve, but the tangent guard must stop the
    # absurd 100 m/s endpoint velocities from producing a giant excursion.
    assert abs(middle.x) < 1.0
    assert abs(middle.z) < 1.0


def test_identical_endpoints_do_not_create_velocity_loop() -> None:
    raw = (
        sample(
            12,
            30,
            x=4.0,
            z=-2.0,
            vx=10.0,
            vz=5.0,
        ),
        sample(
            12,
            32,
            x=4.0,
            z=-2.0,
            vx=-10.0,
            vz=-5.0,
        ),
    )

    output, metrics = interpolate_trajectory(raw)
    middle = output[1]

    assert middle.x == pytest.approx(4.0)
    assert middle.z == pytest.approx(-2.0)
    assert middle.vx == pytest.approx(0.0)
    assert middle.vz == pytest.approx(0.0)
    assert metrics.max_hermite_vs_linear_deviation_m == pytest.approx(0.0)


def test_batch_interpolation_reports_inserted_samples_per_track() -> None:
    trajectories = {
        1: (
            sample(
                1,
                0,
                x=0.0,
                vx=1.0,
            ),
            sample(
                1,
                2,
                x=2.0 / 30.0,
                vx=1.0,
            ),
            sample(
                1,
                4,
                x=4.0 / 30.0,
                vx=1.0,
            ),
        ),
        2: (
            sample(
                2,
                10,
                x=5.0,
                z=2.0,
                vx=0.0,
            ),
            sample(
                2,
                12,
                x=5.0,
                z=2.0,
                vx=0.0,
            ),
        ),
    }

    output, report = interpolate_trajectories(
        trajectories,
    )

    assert report.track_ids == (1, 2)
    assert report.tracks[1].inserted_samples == 2
    assert report.tracks[2].inserted_samples == 1
    assert report.total_inserted_samples == 3

    assert [sample.frame_index for sample in output[1]] == [
        0,
        1,
        2,
        3,
        4,
    ]
    assert [sample.frame_index for sample in output[2]] == [
        10,
        11,
        12,
    ]


def test_non_contiguous_but_valid_multi_frame_gap_is_filled() -> None:
    raw = (
        sample(
            20,
            100,
            timestamp=100.0 / 30.0,
            x=0.0,
            vx=1.0,
        ),
        sample(
            20,
            106,
            timestamp=106.0 / 30.0,
            x=0.2,
            vx=1.0,
        ),
    )

    output, metrics = interpolate_trajectory(raw)

    assert [item.frame_index for item in output] == [
        100,
        101,
        102,
        103,
        104,
        105,
        106,
    ]
    assert metrics.inserted_samples == 5
    assert metrics.filled_gaps == 1
    assert metrics.max_filled_gap_seconds == pytest.approx(0.2)


def test_duplicate_or_decreasing_frame_indices_are_rejected() -> None:
    duplicate = (
        sample(
            9,
            10,
            timestamp=0.0,
            x=0.0,
        ),
        sample(
            9,
            10,
            timestamp=0.1,
            x=0.1,
        ),
    )

    with pytest.raises(
        ValueError,
        match="frame indices must be strictly increasing",
    ):
        interpolate_trajectory(duplicate)


def test_duplicate_or_decreasing_timestamps_are_rejected() -> None:
    bad_time = (
        sample(
            9,
            10,
            timestamp=1.0,
            x=0.0,
        ),
        sample(
            9,
            12,
            timestamp=1.0,
            x=0.1,
        ),
    )

    with pytest.raises(
        ValueError,
        match="timestamps must be strictly increasing",
    ):
        interpolate_trajectory(bad_time)
