from __future__ import annotations

import math

import pytest

from tracking.final.models import (
    FusionMode,
    IdentityTrajectory,
    TrajectoryPoint,
    TrajectoryPointOrigin,
)
from tracking.final.reconstruction import (
    SamplingCadence,
)
from tracking.final.smoothing import (
    TemporalSmoothingConfig,
    smooth_trajectories,
    split_continuous_segments,
)


def _point(
    frame,
    time_s,
    x,
    z=0.0,
    *,
    origin=TrajectoryPointOrigin.OBSERVED,
    identity="identity-001",
):
    if (
        origin
        is TrajectoryPointOrigin.INTERPOLATED
    ):
        fusion_mode = (
            FusionMode.NOT_APPLICABLE
        )
        tracklets = (
            "trk-a",
        )
        detections = (
            f"left-{frame}",
            f"right-{frame}",
        )
        source_count = 2
    else:
        fusion_mode = FusionMode.SINGLE
        tracklets = ("trk-a",)
        detections = (f"det-{frame}",)
        source_count = 1

    return TrajectoryPoint(
        identity_id=identity,
        frame_index=frame,
        timestamp_s=time_s,
        x=x,
        z=z,
        confidence=0.9,
        origin=origin,
        fusion_mode=fusion_mode,
        source_tracklet_ids=tracklets,
        source_detection_ids=detections,
        source_count=source_count,
        position_spread_m=0.0,
        metadata={
            "fusion_conflict": False
        },
    )


def _trajectory(points):
    return IdentityTrajectory(
        identity_id="identity-001",
        tracklet_ids=("trk-a",),
        points=tuple(points),
        trajectory_version="safe-gap-v1",
    )


def test_preserved_gap_splits_hard_segments():
    trajectory = _trajectory(
        [
            _point(0, 0.0, 0.0),
            _point(3, 0.1001, 0.1),
            _point(30, 1.001, 10.0),
            _point(33, 1.1011, 10.1),
        ]
    )
    segments, breaks = (
        split_continuous_segments(
            trajectory,
            SamplingCadence(
                frame_step=3,
                time_step_s=0.1001,
            ),
            tolerance_s=0.02,
        )
    )

    assert [len(row) for row in segments] == [
        2, 2
    ]
    assert len(breaks) == 1
    assert breaks[0]["left_frame_index"] == 3
    assert breaks[0]["right_frame_index"] == 30


def test_rts_reduces_constant_velocity_jitter():
    trajectory = _trajectory(
        [
            _point(0, 0.0, 0.00),
            _point(3, 0.1001, 0.12),
            _point(6, 0.2002, 0.75),
            _point(9, 0.3003, 0.32),
            _point(12, 0.4004, 0.40),
            _point(15, 0.5005, 0.51),
        ]
    )
    smoothed, _, summary = (
        smooth_trajectories(
            trajectories=[trajectory],
            cadence=SamplingCadence(
                frame_step=3,
                time_step_s=0.1001,
            ),
        )
    )
    points = smoothed[0].points

    raw_error = abs(
        trajectory.points[2].x - 0.20
    )
    smooth_error = abs(
        points[2].smoothed_x - 0.20
    )
    assert smooth_error < raw_error
    assert (
        summary[
            "smoothed_finite_difference_acceleration_mps2"
        ]["p95"]
        <
        summary[
            "raw_finite_difference_acceleration_mps2"
        ]["p95"]
    )


def test_interpolated_measurement_is_weaker_than_observed():
    trajectory = _trajectory(
        [
            _point(0, 0.0, 0.0),
            _point(
                3,
                0.1001,
                0.1,
                origin=(
                    TrajectoryPointOrigin.INTERPOLATED
                ),
            ),
            _point(6, 0.2002, 0.2),
        ]
    )
    smoothed, _, _ = (
        smooth_trajectories(
            trajectories=[trajectory],
            cadence=SamplingCadence(
                frame_step=3,
                time_step_s=0.1001,
            ),
        )
    )
    points = smoothed[0].points
    assert (
        points[1].measurement_sigma_m
        > points[0].measurement_sigma_m
    )


def test_two_point_segment_keeps_measurements_and_velocity_only():
    trajectory = _trajectory(
        [
            _point(0, 0.0, 0.0),
            _point(3, 0.1001, 0.2),
        ]
    )
    smoothed, _, _ = (
        smooth_trajectories(
            trajectories=[trajectory],
            cadence=SamplingCadence(
                frame_step=3,
                time_step_s=0.1001,
            ),
        )
    )
    points = smoothed[0].points

    assert points[0].smoothed_x == pytest.approx(
        0.0
    )
    assert points[1].smoothed_x == pytest.approx(
        0.2
    )
    assert points[0].speed_mps == pytest.approx(
        0.2 / 0.1001
    )
    assert points[0].acceleration_mps2 is None
    assert (
        points[0].kinematics_quality
        == "two_point_velocity_only"
    )


def test_smoothing_never_changes_frames_or_point_count():
    trajectory = _trajectory(
        [
            _point(0, 0.0, 0.0),
            _point(3, 0.1001, 0.1),
            _point(6, 0.2002, 0.2),
            _point(30, 1.001, 5.0),
        ]
    )
    smoothed, breaks, summary = (
        smooth_trajectories(
            trajectories=[trajectory],
            cadence=SamplingCadence(
                frame_step=3,
                time_step_s=0.1001,
            ),
        )
    )

    assert [
        point.frame_index
        for point in smoothed[0].points
    ] == [0, 3, 6, 30]
    assert summary["input_point_count"] == 4
    assert summary["output_point_count"] == 4
    assert len(breaks) == 1
