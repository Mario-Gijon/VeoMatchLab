from __future__ import annotations

import math

import pytest

from tracking.actors.identity_trajectory_endpoints import (
    IdentityTrajectoryEndpointConfig,
    build_identity_trajectory_profiles,
)


def _obs(det, frame, t, x, z, confidence=1.0):
    return {
        "detection_id": det,
        "frame_index": frame,
        "timestamp_s": t,
        "x": x,
        "z": z,
        "confidence": confidence,
    }


def _tracklet(tracklet_id, observations):
    return {
        "tracklet_id": tracklet_id,
        "observations": observations,
        "metadata": {},
    }


def _identity(identity_id, tracklets):
    return {
        "identity_id": identity_id,
        "tracklet_ids": tracklets,
    }


def test_exact_first_and_last_observations_define_outer_endpoints():
    rows = [
        _tracklet(
            "T1",
            [
                _obs("d1", 10, 1.0, 2.0, 3.0),
                _obs("d2", 11, 1.1, 2.5, 3.5),
            ],
        ),
        _tracklet(
            "T2",
            [
                _obs("d3", 20, 2.0, 9.0, 8.0),
                _obs("d4", 21, 2.1, 9.5, 8.5),
            ],
        ),
    ]
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T1", "T2"])],
        tracklet_rows=rows,
    )[0]

    assert profile.start.timestamp_s == 1.0
    assert profile.end.timestamp_s == 2.1
    assert len(profile.segments) == 2
    assert len(profile.internal_gaps) == 1


def test_linear_motion_velocity_is_recovered():
    observations = [
        _obs(
            f"d{i}",
            i,
            i * 0.1,
            2.0 * (i * 0.1),
            -1.0 * (i * 0.1),
            0.9,
        )
        for i in range(6)
    ]
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T"])],
        tracklet_rows=[_tracklet("T", observations)],
    )[0]

    assert len(profile.segments) == 1
    assert profile.end.velocity is not None
    assert profile.end.velocity.vx_mps == pytest.approx(2.0)
    assert profile.end.velocity.vz_mps == pytest.approx(-1.0)
    assert profile.end.velocity.speed_mps == pytest.approx(math.sqrt(5.0))
    assert profile.end.velocity.residual_rmse_m < 1e-9


def test_large_gap_creates_two_segments_and_explicit_gap():
    rows = [
        _tracklet(
            "T1",
            [
                _obs("d1", 0, 0.0, 0.0, 0.0),
                _obs("d2", 1, 0.1, 0.1, 0.0),
            ],
        ),
        _tracklet(
            "T2",
            [
                _obs("d3", 100, 10.0, 50.0, 0.0),
                _obs("d4", 101, 10.1, 50.1, 0.0),
            ],
        ),
    ]
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T1", "T2"])],
        tracklet_rows=rows,
        config=IdentityTrajectoryEndpointConfig(
            max_contiguous_gap_s=0.45,
        ),
    )[0]

    assert len(profile.segments) == 2
    assert profile.internal_gap_count == 1
    gap = profile.internal_gaps[0]
    assert gap.gap_s == pytest.approx(9.9)
    assert gap.endpoint_distance_m == pytest.approx(49.9)
    assert gap.straight_line_speed_mps == pytest.approx(49.9 / 9.9)


def test_velocity_never_bridges_large_internal_gap():
    rows = [
        _tracklet(
            "T1",
            [
                _obs("d1", 0, 0.0, 0.0, 0.0),
                _obs("d2", 1, 0.1, 0.1, 0.0),
            ],
        ),
        _tracklet(
            "T2",
            [
                _obs("d3", 100, 10.0, 50.0, 0.0),
                _obs("d4", 101, 10.1, 50.1, 0.0),
            ],
        ),
    ]
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T1", "T2"])],
        tracklet_rows=rows,
    )[0]

    assert profile.segments[0].end.velocity is not None
    assert profile.segments[0].end.velocity.vx_mps == pytest.approx(1.0)
    assert profile.segments[1].start.velocity is not None
    assert profile.segments[1].start.velocity.vx_mps == pytest.approx(1.0)


def test_short_gap_stays_inside_one_segment():
    rows = [
        _tracklet(
            "T1",
            [
                _obs("d1", 0, 0.0, 0.0, 0.0),
                _obs("d2", 1, 0.1, 0.1, 0.0),
            ],
        ),
        _tracklet(
            "T2",
            [
                _obs("d3", 3, 0.4, 0.4, 0.0),
                _obs("d4", 4, 0.5, 0.5, 0.0),
            ],
        ),
    ]
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T1", "T2"])],
        tracklet_rows=rows,
    )[0]

    assert len(profile.segments) == 1
    assert profile.internal_gap_count == 0


def test_singleton_segment_has_no_velocity():
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T"])],
        tracklet_rows=[
            _tracklet("T", [_obs("d1", 0, 0.0, 1.0, 2.0)])
        ],
    )[0]
    assert profile.start.velocity is None
    assert profile.end.velocity is None


def test_unknown_tracklet_is_rejected():
    with pytest.raises(ValueError):
        build_identity_trajectory_profiles(
            identity_rows=[_identity("P", ["missing"])],
            tracklet_rows=[],
        )


def test_duplicate_identity_is_rejected():
    rows = [
        _tracklet(
            "T",
            [
                _obs("d1", 0, 0.0, 0.0, 0.0),
                _obs("d2", 1, 0.1, 0.1, 0.0),
            ],
        )
    ]
    with pytest.raises(ValueError):
        build_identity_trajectory_profiles(
            identity_rows=[
                _identity("P", ["T"]),
                _identity("P", ["T"]),
            ],
            tracklet_rows=rows,
        )


def test_stage_changes_no_identity_decision():
    profile = build_identity_trajectory_profiles(
        identity_rows=[_identity("P", ["T"])],
        tracklet_rows=[
            _tracklet(
                "T",
                [
                    _obs("d1", 0, 0.0, 0.0, 0.0),
                    _obs("d2", 1, 0.1, 0.1, 0.0),
                ],
            )
        ],
    )[0]
    payload = profile.to_dict()
    assert payload["identity_id"] == "P"
    assert payload["identity_decision"] is None
    assert payload["continuity_segment_count"] == 1
