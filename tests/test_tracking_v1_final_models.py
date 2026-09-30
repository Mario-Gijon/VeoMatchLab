from __future__ import annotations

import pytest

from tracking.final.models import (
    FusionMode,
    IdentityTrajectory,
    TrajectoryPoint,
    TrajectoryPointOrigin,
)


def _point(
    frame,
    time_s,
    *,
    identity="identity-001",
):
    return TrajectoryPoint(
        identity_id=identity,
        frame_index=frame,
        timestamp_s=time_s,
        x=float(frame),
        z=0.0,
        confidence=0.9,
        origin=TrajectoryPointOrigin.OBSERVED,
        fusion_mode=FusionMode.SINGLE,
        source_tracklet_ids=("trk-a",),
        source_detection_ids=(f"det-{frame}",),
        source_count=1,
        position_spread_m=0.0,
    )


def test_point_round_trip():
    point = _point(3, 0.3)
    restored = TrajectoryPoint.from_dict(
        point.to_dict()
    )
    assert restored == point


def test_trajectory_requires_one_point_per_frame():
    with pytest.raises(
        ValueError,
        match="strictly increasing|multiple points",
    ):
        IdentityTrajectory(
            identity_id="identity-001",
            tracklet_ids=("trk-a",),
            points=(
                _point(3, 0.3),
                _point(3, 0.4),
            ),
        )


def test_fused_point_requires_multiple_sources():
    with pytest.raises(
        ValueError,
        match="at least two sources",
    ):
        TrajectoryPoint(
            identity_id="identity-001",
            frame_index=0,
            timestamp_s=0.0,
            x=0.0,
            z=0.0,
            confidence=0.9,
            origin=TrajectoryPointOrigin.FUSED,
            fusion_mode=FusionMode.WEIGHTED_MEAN,
            source_tracklet_ids=("trk-a",),
            source_detection_ids=("det-a",),
            source_count=1,
            position_spread_m=0.0,
        )
