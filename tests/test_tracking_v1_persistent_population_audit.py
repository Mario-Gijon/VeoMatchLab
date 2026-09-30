from __future__ import annotations

from copy import deepcopy

from tracking.persistent.audit import (
    PersistentPopulationAuditConfig,
    audit_persistent_population,
)


def _point(actor: str, frame: int, *, origin: str = "observed", x: float = 0.0):
    return {
        "actor_id": actor,
        "frame_index": frame,
        "timestamp_s": frame * 0.1,
        "x": x,
        "z": 0.0,
        "origin": origin,
        "position_std_m": 0.1,
        "speed_mps": 1.0,
        "vx_mps": 1.0,
        "vz_mps": 0.0,
        "prediction_age_s": 0.1 if origin == "predicted" else 0.0,
    }


def _fixture():
    actors = ["actor-001", "actor-002"]
    trajectories = [
        {
            "actor_id": actor,
            "points": [_point(actor, frame, x=float(index) * 2.0) for frame in (0, 1, 2)],
        }
        for index, actor in enumerate(actors)
    ]
    occupancy = []
    for frame in (0, 1, 2):
        occupancy.append(
            {
                "frame_index": frame,
                "timestamp_s": frame * 0.1,
                "represented_actor_count": 2,
                "direct_actor_count": 2,
                "actor_ids": actors,
                "missing_actor_ids": [],
            }
        )
    assignment_rows = [
        {"tracklet_id": "trk-1", "actor_id": "actor-001", "status": "assigned"},
        {"tracklet_id": "trk-2", "actor_id": "actor-002", "status": "assigned"},
    ]
    assignment_summary = {
        "solver_version": "anchored-actor-assignment-v1",
        "population_estimate": 2,
        "actor_count": 2,
        "hard_cannot_link_count": 1,
    }
    trajectory_summary = {
        "trajectory_version": "persistent-trajectory-v1",
        "actor_count": 2,
        "frame_count": 3,
        "represented_actor_frame_count": 6,
        "full_population_frame_count_after": 3,
        "cadence": {"frame_step": 1, "time_step_s": 0.1},
        "config": {"max_edge_prediction_s": 0.5},
    }
    hard_rows = [
        {"status": "cannot_link", "tracklet_a": "trk-1", "tracklet_b": "trk-2"}
    ]
    return {
        "trajectory_rows": trajectories,
        "occupancy_rows": occupancy,
        "gap_rows": [],
        "trajectory_summary": trajectory_summary,
        "assignment_rows": assignment_rows,
        "assignment_summary": assignment_summary,
        "hard_pair_rows": hard_rows,
    }


def _run(data, **config_kwargs):
    return audit_persistent_population(
        **data,
        config=PersistentPopulationAuditConfig(**config_kwargs),
    )


def test_clean_fixture_passes():
    result = _run(_fixture())
    assert result["summary"]["status"] == "PASS"
    assert result["summary"]["blocking_issue_count"] == 0
    assert result["summary"]["hard_constraint_violation_count"] == 0


def test_hard_cannot_link_violation_is_blocker():
    data = _fixture()
    data["assignment_rows"][1]["actor_id"] = "actor-001"
    result = _run(data)
    assert result["summary"]["status"] == "FAIL"
    assert result["summary"]["hard_constraint_violation_count"] == 1


def test_actor_count_mismatch_is_blocker():
    data = _fixture()
    data["assignment_summary"]["population_estimate"] = 3
    result = _run(data)
    assert any(issue["code"] == "actor_count_mismatch" for issue in result["issues"])


def test_edge_missing_frame_is_review_not_disappearance():
    data = _fixture()
    data["trajectory_rows"][0]["points"] = data["trajectory_rows"][0]["points"][1:]
    data["occupancy_rows"][0]["represented_actor_count"] = 1
    data["occupancy_rows"][0]["direct_actor_count"] = 1
    data["occupancy_rows"][0]["actor_ids"] = ["actor-002"]
    data["occupancy_rows"][0]["missing_actor_ids"] = ["actor-001"]
    data["trajectory_summary"]["represented_actor_frame_count"] = 5
    data["trajectory_summary"]["full_population_frame_count_after"] = 2
    result = _run(
        data,
        min_represented_coverage_fraction=0.0,
        min_full_population_frame_fraction=0.0,
    )
    assert result["summary"]["blocking_issue_count"] == 0
    assert result["summary"]["status"] == "PASS_WITH_REVIEW"
    assert result["summary"]["internal_missing_actor_frame_count"] == 0


def test_internal_missing_frame_is_blocker():
    data = _fixture()
    points = data["trajectory_rows"][0]["points"]
    data["trajectory_rows"][0]["points"] = [points[0], points[2]]
    data["occupancy_rows"][1]["represented_actor_count"] = 1
    data["occupancy_rows"][1]["direct_actor_count"] = 1
    data["occupancy_rows"][1]["actor_ids"] = ["actor-002"]
    data["occupancy_rows"][1]["missing_actor_ids"] = ["actor-001"]
    data["trajectory_summary"]["represented_actor_frame_count"] = 5
    data["trajectory_summary"]["full_population_frame_count_after"] = 2
    result = _run(
        data,
        min_represented_coverage_fraction=0.0,
        min_full_population_frame_fraction=0.0,
    )
    assert result["summary"]["status"] == "FAIL"
    assert result["summary"]["internal_missing_actor_frame_count"] == 1


def test_prediction_beyond_source_policy_is_blocker():
    data = _fixture()
    point = data["trajectory_rows"][0]["points"][-1]
    point["origin"] = "predicted"
    point["prediction_age_s"] = 0.8
    data["occupancy_rows"][-1]["direct_actor_count"] = 1
    result = _run(data)
    assert result["summary"]["prediction_age_violation_count"] == 1
    assert result["summary"]["status"] == "FAIL"


def test_high_uncertainty_is_review():
    data = _fixture()
    data["trajectory_rows"][0]["points"][0]["position_std_m"] = 1.2
    result = _run(data)
    assert result["summary"]["blocking_issue_count"] == 0
    assert result["summary"]["high_uncertainty_point_count"] == 1
    assert result["summary"]["status"] == "PASS_WITH_REVIEW"


def test_close_actors_are_review_only():
    data = _fixture()
    for trajectory in data["trajectory_rows"]:
        for point in trajectory["points"]:
            point["x"] = 0.0
    result = _run(data)
    assert result["summary"]["blocking_issue_count"] == 0
    assert result["summary"]["close_actor_pair_count"] == 3
    assert result["summary"]["status"] == "PASS_WITH_REVIEW"


def test_occupancy_mismatch_is_blocker():
    data = _fixture()
    data["occupancy_rows"][0]["actor_ids"] = ["actor-001"]
    result = _run(data)
    assert any(issue["code"] == "occupancy_trajectory_mismatch" for issue in result["issues"])


def test_summary_mismatch_is_blocker():
    data = _fixture()
    data["trajectory_summary"]["represented_actor_frame_count"] = 999
    result = _run(data)
    assert any(issue["code"] == "trajectory_summary_mismatch" for issue in result["issues"])


def test_preserved_gap_is_reported_without_becoming_blocker():
    data = _fixture()
    data["gap_rows"] = [
        {
            "actor_id": "actor-001",
            "action": "preserved",
            "gap_kind": "leading",
        }
    ]
    result = _run(data)
    assert result["summary"]["preserved_gap_count"] == 1
    assert result["summary"]["blocking_issue_count"] == 0
