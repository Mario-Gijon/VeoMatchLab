from __future__ import annotations

import pytest

from tracking.final.fusion import (
    ObservationFusionConfig,
    build_observation_fused_trajectories,
)
from tracking.final.models import (
    FusionMode,
    TrajectoryPointOrigin,
)


def _tracklet(tracklet_id, observations):
    return {
        "tracklet_id": tracklet_id,
        "observations": [
            {
                "detection_id": det,
                "frame_index": frame,
                "timestamp_s": time_s,
                "x": x,
                "z": z,
                "confidence": confidence,
            }
            for (
                det,
                frame,
                time_s,
                x,
                z,
                confidence,
            ) in observations
        ],
    }


def _identity(identity_id, tracklets):
    return {
        "identity_id": identity_id,
        "tracklet_ids": list(tracklets),
        "canonical": True,
        "provisional": False,
    }


def test_single_observation_is_preserved_exactly():
    trajectories, events, summary = (
        build_observation_fused_trajectories(
            identity_rows=[
                _identity("identity-001", ["trk-a"])
            ],
            tracklet_rows=[
                _tracklet(
                    "trk-a",
                    [
                        (
                            "det-a",
                            3,
                            0.3,
                            1.25,
                            -2.5,
                            0.91,
                        )
                    ],
                )
            ],
        )
    )
    point = trajectories[0].points[0]
    assert point.origin is TrajectoryPointOrigin.OBSERVED
    assert point.x == pytest.approx(1.25)
    assert point.z == pytest.approx(-2.5)
    assert point.confidence == pytest.approx(0.91)
    assert events == ()
    assert summary["collapsed_observation_count"] == 0


def test_close_same_frame_sources_are_weighted_mean_fused():
    trajectories, events, summary = (
        build_observation_fused_trajectories(
            identity_rows=[
                _identity(
                    "identity-001",
                    ["trk-a", "trk-b"],
                )
            ],
            tracklet_rows=[
                _tracklet(
                    "trk-a",
                    [("det-a", 10, 1.0, 0.0, 0.0, 0.9)],
                ),
                _tracklet(
                    "trk-b",
                    [("det-b", 10, 1.0, 1.0, 0.0, 0.8)],
                ),
            ],
            config=ObservationFusionConfig(
                max_fusion_diameter_m=1.5
            ),
        )
    )
    point = trajectories[0].points[0]
    assert point.origin is TrajectoryPointOrigin.FUSED
    assert point.fusion_mode is FusionMode.WEIGHTED_MEAN
    assert point.x == pytest.approx(
        (0.9 * 0.0 + 0.8 * 1.0) / 1.7
    )
    assert point.source_count == 2
    assert len(events) == 1
    assert events[0]["fusion_conflict"] is False
    assert summary["weighted_mean_fusion_count"] == 1


def test_far_same_frame_sources_select_medoid_not_midpoint():
    trajectories, events, summary = (
        build_observation_fused_trajectories(
            identity_rows=[
                _identity(
                    "identity-001",
                    ["trk-a", "trk-b", "trk-c"],
                )
            ],
            tracklet_rows=[
                _tracklet(
                    "trk-a",
                    [("det-a", 10, 1.0, 0.0, 0.0, 0.95)],
                ),
                _tracklet(
                    "trk-b",
                    [("det-b", 10, 1.0, 0.2, 0.0, 0.90)],
                ),
                _tracklet(
                    "trk-c",
                    [("det-c", 10, 1.0, 5.0, 0.0, 0.60)],
                ),
            ],
            config=ObservationFusionConfig(
                max_fusion_diameter_m=1.5
            ),
        )
    )
    point = trajectories[0].points[0]
    assert point.fusion_mode is FusionMode.MEDOID_SELECTED
    assert min(abs(point.x - 0.0), abs(point.x - 0.2)) < 1e-9
    assert events[0]["fusion_conflict"] is True
    assert summary["medoid_selected_conflict_count"] == 1


def test_identity_partition_must_cover_all_tracklets():
    with pytest.raises(
        ValueError,
        match="do not cover tracklets exactly",
    ):
        build_observation_fused_trajectories(
            identity_rows=[
                _identity("identity-001", ["trk-a"])
            ],
            tracklet_rows=[
                _tracklet(
                    "trk-a",
                    [("det-a", 0, 0.0, 0.0, 0.0, 0.9)],
                ),
                _tracklet(
                    "trk-b",
                    [("det-b", 1, 0.1, 1.0, 0.0, 0.9)],
                ),
            ],
        )
