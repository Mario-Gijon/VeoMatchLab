from __future__ import annotations

import copy

from tracking.final.diagnostics import (
    audit_final_trajectories,
)


def _point(
    frame,
    time_s,
    *,
    segment="identity-001:segment-001",
    speed=1.0,
    acceleration=1.0,
    adjustment=0.1,
    quality="rts_cv",
):
    if quality == "rts_cv":
        vx = speed
        vz = 0.0
        ax = acceleration
        az = 0.0
        velocity_std = 0.9
        smoothing_applied = True
    elif quality == "two_point_velocity_only":
        vx = speed
        vz = 0.0
        ax = None
        az = None
        acceleration = None
        velocity_std = None
        smoothing_applied = False
    else:
        vx = vz = speed = None
        ax = az = acceleration = None
        velocity_std = None
        smoothing_applied = False

    measurement_x = float(frame)
    return {
        "smoothing_version": "rts-cv-v1",
        "identity_id": "identity-001",
        "segment_id": segment,
        "segment_index": 1,
        "frame_index": frame,
        "timestamp_s": time_s,
        "measurement_x": measurement_x,
        "measurement_z": 0.0,
        "smoothed_x": measurement_x + adjustment,
        "smoothed_z": 0.0,
        "measurement_confidence": 0.9,
        "measurement_sigma_m": 0.3,
        "origin": "observed",
        "fusion_mode": "single",
        "source_tracklet_ids": ["trk-a"],
        "source_detection_ids": [f"det-{frame}"],
        "source_count": 1,
        "position_spread_m": 0.0,
        "position_adjustment_m": adjustment,
        "position_std_m": 0.15,
        "vx_mps": vx,
        "vz_mps": vz,
        "speed_mps": speed,
        "ax_mps2": ax,
        "az_mps2": az,
        "acceleration_mps2": acceleration,
        "velocity_std_mps": velocity_std,
        "kinematics_quality": quality,
        "metadata": {"smoothing_applied": smoothing_applied},
    }


def _trajectory(points, segment_count=1):
    return {
        "trajectory_version": "rts-cv-v1",
        "identity_id": "identity-001",
        "tracklet_ids": ["trk-a"],
        "tracklet_count": 1,
        "point_count": len(points),
        "segment_count": segment_count,
        "points": points,
        "metadata": {},
    }


def test_clean_long_segment_has_no_blockers():
    rows = [
        _trajectory(
            [
                _point(0, 0.0),
                _point(3, 0.1001),
                _point(6, 0.2002),
            ]
        )
    ]
    result = audit_final_trajectories(
        trajectory_rows=rows,
        source_summary={
            "smoothing_version": "rts-cv-v1",
            "trajectory_count": 1,
            "output_point_count": 3,
            "segment_count": 1,
            "preserved_gap_break_count": 0,
        },
    )
    assert result[3] == ()
    assert result[2] == ()
    assert result[-1]["publishable"] is True


def test_short_segment_is_review_not_blocker():
    rows = [
        _trajectory(
            [
                _point(
                    0,
                    0.0,
                    quality="two_point_velocity_only",
                ),
                _point(
                    3,
                    0.1001,
                    quality="two_point_velocity_only",
                ),
            ]
        )
    ]
    result = audit_final_trajectories(
        trajectory_rows=rows,
    )
    assert result[3] == ()
    assert "short_segment" in result[1][0]["review_flags"]
    assert result[-1]["publishable"] is True


def test_physical_outlier_is_review_not_structural_blocker():
    rows = [
        _trajectory(
            [
                _point(
                    0,
                    0.0,
                    speed=15.0,
                    acceleration=20.0,
                    adjustment=1.2,
                ),
                _point(
                    3,
                    0.1001,
                    speed=15.0,
                    acceleration=20.0,
                    adjustment=1.2,
                ),
                _point(
                    6,
                    0.2002,
                    speed=15.0,
                    acceleration=20.0,
                    adjustment=1.2,
                ),
            ]
        )
    ]
    result = audit_final_trajectories(
        trajectory_rows=rows,
    )
    assert len(result[2]) == 3
    assert result[3] == ()
    assert result[-1]["publishable"] is True
    assert result[-1]["review_flag_counts"]["speed_gt_12mps"] == 3


def test_inconsistent_speed_vector_is_blocker():
    rows = [
        _trajectory(
            [
                _point(0, 0.0),
                _point(3, 0.1001),
                _point(6, 0.2002),
            ]
        )
    ]
    broken = copy.deepcopy(rows)
    broken[0]["points"][1]["speed_mps"] = 9.0

    result = audit_final_trajectories(
        trajectory_rows=broken,
    )
    assert any(
        row["flag"] == "speed_vector_inconsistent"
        for row in result[3]
    )
    assert result[-1]["publishable"] is False


def test_segment_break_artifact_must_match_partition():
    rows = [
        _trajectory(
            [
                _point(0, 0.0),
                _point(3, 0.1001),
                _point(
                    30,
                    1.001,
                    segment="identity-001:segment-002",
                ),
                _point(
                    33,
                    1.1011,
                    segment="identity-001:segment-002",
                ),
            ],
            segment_count=2,
        )
    ]

    clean_break = {
        "identity_id": "identity-001",
        "left_frame_index": 3,
        "right_frame_index": 30,
    }
    assert audit_final_trajectories(
        trajectory_rows=rows,
        segment_break_rows=[clean_break],
    )[3] == ()

    result = audit_final_trajectories(
        trajectory_rows=rows,
        segment_break_rows=[],
    )
    assert any(
        row["flag"] == "missing_segment_break_artifact"
        for row in result[3]
    )
