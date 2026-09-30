from __future__ import annotations

import pytest

from tracking.persistent.trajectories import (
    PersistentTrajectoryConfig,
    build_persistent_actor_trajectories,
    infer_sampling_cadence,
    _load_tracklets,
)


def obs(frame: int, x: float, z: float = 0.0, confidence: float = 0.9, det: str | None = None):
    return {
        "detection_id": det or f"det-{frame}-{x}",
        "frame_index": frame,
        "timestamp_s": frame / 30.0,
        "x": x,
        "z": z,
        "confidence": confidence,
    }


def tracklet(tracklet_id: str, observations):
    return {
        "tracklet_id": tracklet_id,
        "observations": list(observations),
        "metadata": {},
    }


def actor(actor_id: str, tracklets, identity: str | None = None):
    return {
        "solver_version": "anchored-actor-assignment-v1",
        "actor_id": actor_id,
        "anchor_frame_index": 3,
        "anchor_identity_id": identity or f"id-{actor_id}",
        "tracklet_ids": list(tracklets),
        "source_identity_ids": [identity or f"id-{actor_id}"],
    }


def assignment(tracklet_id: str, actor_id: str | None):
    return {
        "solver_version": "anchored-actor-assignment-v1",
        "tracklet_id": tracklet_id,
        "actor_id": actor_id,
        "status": "noise" if actor_id is None else "assigned",
    }


def build(tracklets, actors, assignments, **config_kwargs):
    return build_persistent_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        assignment_summary={
            "solver_version": "anchored-actor-assignment-v1",
            "actor_count": len(actors),
            "hard_constraint_violation_count": 0,
        },
        config=PersistentTrajectoryConfig(**config_kwargs),
    )


def test_cadence_is_inferred_from_global_sampled_frames():
    rows = [
        tracklet("A", [obs(0, 0), obs(3, 1), obs(6, 2)]),
        tracklet("B", [obs(0, 10), obs(6, 12)]),
    ]
    cadence = infer_sampling_cadence(_load_tracklets(rows))
    assert cadence.frame_step == 3
    assert cadence.time_step_s == pytest.approx(0.1)
    assert cadence.frames == (0, 3, 6)


def test_short_internal_gap_is_interpolated_with_two_sided_support():
    rows = [tracklet("A", [obs(0, 0), obs(3, 1), obs(9, 3), obs(12, 4)])]
    result = build(
        rows,
        [actor("actor-001", ["A"])],
        [assignment("A", "actor-001")],
        max_internal_gap_s=0.5,
    )
    trajectory = result["trajectories"][0]
    by_frame = {point["frame_index"]: point for point in trajectory["points"]}
    assert by_frame[6]["origin"] == "interpolated"
    assert by_frame[6]["support_before_frame_index"] == 3
    assert by_frame[6]["support_after_frame_index"] == 9
    assert result["summary"]["interpolated_point_count"] == 1
    assert result["summary"]["disappearance_event_count_after"] == 0


def test_long_internal_gap_is_preserved_not_blindly_filled():
    rows = [tracklet("A", [obs(0, 0), obs(3, 1), obs(30, 3), obs(33, 4)])]
    result = build(
        rows,
        [actor("actor-001", ["A"])],
        [assignment("A", "actor-001")],
        max_internal_gap_s=0.5,
        max_edge_prediction_s=0.1,
    )
    trajectory = result["trajectories"][0]
    frames = {point["frame_index"] for point in trajectory["points"]}
    assert 6 not in frames
    internal = next(row for row in result["gaps"] if row["gap_kind"] == "internal")
    assert internal["action"] == "preserved"
    assert internal["reason"] == "internal_gap_too_long"
    assert trajectory["segment_count"] == 2


def test_internal_gap_with_impossible_endpoint_speed_is_preserved():
    rows = [tracklet("A", [obs(0, 0), obs(3, 0), obs(9, 20), obs(12, 20)])]
    result = build(
        rows,
        [actor("actor-001", ["A"])],
        [assignment("A", "actor-001")],
        max_internal_gap_s=1.0,
        max_internal_endpoint_speed_mps=12.0,
    )
    internal = next(row for row in result["gaps"] if row["gap_kind"] == "internal")
    assert internal["action"] == "preserved"
    assert internal["reason"] == "endpoint_speed_too_high"


def test_short_leading_and_trailing_edges_are_predicted_when_supported():
    # B extends the global cadence to frames 0..15; A is observed only 3..12.
    rows = [
        tracklet("A", [obs(3, 1), obs(6, 2), obs(9, 3), obs(12, 4)]),
        tracklet("B", [obs(0, 20), obs(3, 20), obs(6, 20), obs(9, 20), obs(12, 20), obs(15, 20)]),
    ]
    result = build(
        rows,
        [actor("actor-001", ["A"]), actor("actor-002", ["B"])],
        [assignment("A", "actor-001"), assignment("B", "actor-002")],
        max_edge_prediction_s=0.11,
    )
    traj = next(row for row in result["trajectories"] if row["actor_id"] == "actor-001")
    by_frame = {point["frame_index"]: point for point in traj["points"]}
    assert by_frame[0]["origin"] == "predicted"
    assert by_frame[15]["origin"] == "predicted"
    assert by_frame[0]["prediction_age_s"] == pytest.approx(0.1)
    assert by_frame[15]["prediction_age_s"] == pytest.approx(0.1)


def test_edge_prediction_does_not_extend_beyond_configured_age():
    rows = [
        tracklet("A", [obs(6, 2), obs(9, 3), obs(12, 4)]),
        tracklet("B", [obs(0, 20), obs(3, 20), obs(6, 20), obs(9, 20), obs(12, 20), obs(15, 20), obs(18, 20)]),
    ]
    result = build(
        rows,
        [actor("actor-001", ["A"]), actor("actor-002", ["B"])],
        [assignment("A", "actor-001"), assignment("B", "actor-002")],
        max_edge_prediction_s=0.11,
    )
    traj = next(row for row in result["trajectories"] if row["actor_id"] == "actor-001")
    frames = {point["frame_index"] for point in traj["points"]}
    assert 3 in frames
    assert 0 not in frames
    assert 15 in frames
    assert 18 not in frames


def test_same_actor_same_frame_observations_are_fused():
    rows = [
        tracklet("A", [obs(0, 0, det="a0"), obs(3, 1, det="a3"), obs(6, 2, det="a6")]),
        tracklet("B", [obs(3, 1.2, confidence=0.8, det="b3")]),
    ]
    result = build(
        rows,
        [actor("actor-001", ["A", "B"])],
        [assignment("A", "actor-001"), assignment("B", "actor-001")],
    )
    point = next(
        point for point in result["trajectories"][0]["points"] if point["frame_index"] == 3
    )
    assert point["origin"] == "fused"
    assert point["fusion_mode"] == "weighted_mean"
    assert set(point["source_tracklet_ids"]) == {"A", "B"}
    assert point["source_count"] == 2


def test_spatially_inconsistent_duplicate_uses_medoid_not_mean():
    rows = [
        tracklet("A", [obs(0, 0), obs(3, 1, confidence=0.95, det="a3"), obs(6, 2)]),
        tracklet("B", [obs(3, 10, confidence=0.2, det="b3")]),
    ]
    result = build(
        rows,
        [actor("actor-001", ["A", "B"])],
        [assignment("A", "actor-001"), assignment("B", "actor-001")],
        weighted_fusion_max_spread_m=1.0,
    )
    point = next(
        point for point in result["trajectories"][0]["points"] if point["frame_index"] == 3
    )
    assert point["origin"] == "fused"
    assert point["fusion_mode"] == "medoid_selected"
    assert point["fusion_conflict"] is True
    assert point["measurement_x"] == pytest.approx(1.0)


def test_noise_tracklets_are_not_used_in_actor_trajectory():
    rows = [
        tracklet("A", [obs(0, 0), obs(3, 1), obs(6, 2)]),
        tracklet("N", [obs(3, 100, det="noise")]),
    ]
    result = build(
        rows,
        [actor("actor-001", ["A"])],
        [assignment("A", "actor-001"), assignment("N", None)],
    )
    trajectory = result["trajectories"][0]
    assert all("N" not in point["source_tracklet_ids"] for point in trajectory["points"])
    assert trajectory["tracklet_ids"] == ["A"]


def test_occupancy_reports_before_and_after_reconstruction():
    rows = [
        tracklet("A", [obs(0, 0), obs(3, 1), obs(9, 3), obs(12, 4)]),
        tracklet("B", [obs(0, 10), obs(3, 10), obs(6, 10), obs(9, 10), obs(12, 10)]),
    ]
    result = build(
        rows,
        [actor("actor-001", ["A"]), actor("actor-002", ["B"])],
        [assignment("A", "actor-001"), assignment("B", "actor-002")],
        max_internal_gap_s=0.5,
    )
    frame6 = next(row for row in result["occupancy"] if row["frame_index"] == 6)
    assert frame6["direct_actor_count"] == 1
    assert frame6["interpolated_actor_count"] == 1
    assert frame6["represented_actor_count"] == 2
    assert result["summary"]["full_population_frame_count_before"] == 4
    assert result["summary"]["full_population_frame_count_after"] == 5


def test_actor_membership_mismatch_is_rejected():
    rows = [tracklet("A", [obs(0, 0), obs(3, 1)])]
    with pytest.raises(ValueError, match="membership differs"):
        build(
            rows,
            [actor("actor-001", ["WRONG"])],
            [assignment("A", "actor-001")],
        )


def test_summary_with_hard_violation_is_rejected():
    rows = [tracklet("A", [obs(0, 0), obs(3, 1)])]
    with pytest.raises(ValueError, match="hard constraint violations"):
        build_persistent_actor_trajectories(
            tracklet_rows=rows,
            actor_rows=[actor("actor-001", ["A"])],
            assignment_rows=[assignment("A", "actor-001")],
            assignment_summary={
                "solver_version": "anchored-actor-assignment-v1",
                "actor_count": 1,
                "hard_constraint_violation_count": 1,
            },
        )
