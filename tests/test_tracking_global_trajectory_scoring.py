from __future__ import annotations

import pytest

from tracking.actors.global_trajectory_scoring import (
    GlobalTrajectoryScoringConfig,
    score_global_trajectory_candidates,
)
from tracking.actors.identity_expansion import IdentityCalibration


def _profile(identity_id, vector, reliability=0.8):
    return {
        "identity_id": identity_id,
        "source_actor_ids": [],
        "tracklet_ids": [f"T-{identity_id}"],
        "profiled_actor_ids": [],
        "profiled_tracklet_count": 2,
        "aggregate_embedding": vector,
        "reliability": reliability,
        "start_time_s": 0.0,
        "end_time_s": 1.0,
        "kit_cluster_id": None,
        "kit_dominance": None,
        "initial_seed": True,
    }


def _geometry(speed=1.0, error=0.5, reliability=0.8):
    return {
        "gap_s": 1.0,
        "endpoint_distance_m": 2.0,
        "reachability_limit_m": 13.5,
        "reachability_margin_m": 11.5,
        "required_speed_after_slack_mps": speed,
        "forward_prediction_error_m": error,
        "backward_prediction_error_m": error,
        "velocity_direction_cosine": 0.9,
        "predecessor_velocity_reliability": reliability,
        "successor_velocity_reliability": reliability,
        "velocity_prediction_used": True,
        "physically_reachable": True,
    }


def _append(a, b, geometry=None):
    return {
        "candidate_type": "append",
        "predecessor_identity_id": a,
        "successor_identity_id": b,
        "geometry": geometry or _geometry(),
        "gap_s": 1.0,
        "family_relation": "neutral",
    }


def _insert(host, inserted, gap=1, left=None, right=None):
    return {
        "candidate_type": "insert",
        "host_identity_id": host,
        "inserted_identity_id": inserted,
        "host_gap_index": gap,
        "host_preceding_segment_index": gap,
        "host_following_segment_index": gap + 1,
        "left_geometry": left or _geometry(),
        "right_geometry": right or _geometry(),
        "left_gap_s": 1.0,
        "right_gap_s": 1.0,
        "family_relation": "neutral",
    }


def _calibration():
    return IdentityCalibration(
        positive=(0.75, 0.85, 0.90, 0.95),
        same_kit_negative=(0.10, 0.20, 0.40, 0.60),
        different_kit_negative=(0.05, 0.10, 0.20),
        unknown_kit_negative=(0.10, 0.30, 0.50),
    )


def _family(identity_id, family_id):
    return {
        "identity_id": identity_id,
        "state": "confident_primary",
        "family_type": "primary",
        "family_id": family_id,
        "candidate_family_type": "primary",
        "candidate_family_id": family_id,
        "confidence": 0.9,
        "dominance": 1.0,
        "evidence_coverage": 0.9,
        "usable_for_tracking": True,
        "family_distribution": {f"primary-{family_id}": 1.0},
    }


def test_better_reid_wins_same_append_competition():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.10]),
        _profile("C", [0.60, 0.80]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[_append("A", "B"), _append("A", "C")],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    by_successor = {
        row.source["successor_identity_id"]: row
        for row in result.append_candidates
    }
    assert by_successor["B"].outgoing_reid_rank == 1
    assert (
        by_successor["B"].trajectory_score
        > by_successor["C"].trajectory_score
    )


def test_physical_effort_penalizes_required_sprint():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.1]),
        _profile("C", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[
            _append("A", "B", _geometry(speed=1.0)),
            _append("A", "C", _geometry(speed=11.0)),
        ],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    rows = result.append_candidates
    assert rows[0].physical_effort_support != rows[1].physical_effort_support
    low = next(r for r in rows if r.source["successor_identity_id"] == "B")
    high = next(r for r in rows if r.source["successor_identity_id"] == "C")
    assert low.physical_effort_support > high.physical_effort_support


def test_good_velocity_prediction_beats_bad_one():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.1]),
        _profile("C", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[
            _append("A", "B", _geometry(error=0.2)),
            _append("A", "C", _geometry(error=8.0)),
        ],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    b = next(r for r in result.append_candidates if r.source["successor_identity_id"] == "B")
    c = next(r for r in result.append_candidates if r.source["successor_identity_id"] == "C")
    assert b.velocity_support > c.velocity_support
    assert b.geometry_support > c.geometry_support


def test_different_primary_applies_negative_adjustment():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[_append("A", "B")],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
        family_rows=[_family("A", 0), _family("B", 1)],
    )
    row = result.append_candidates[0]
    assert row.family_relation == "different_primary"
    assert row.family_adjustment < 0.0
    assert row.trajectory_score < row.trajectory_base_score


def test_insert_uses_both_boundaries():
    profiles = [
        _profile("H", [1.0, 0.0]),
        _profile("I1", [0.99, 0.1]),
        _profile("I2", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[],
        insert_candidates=[
            _insert(
                "H", "I1",
                left=_geometry(speed=1.0),
                right=_geometry(speed=1.0),
            ),
            _insert(
                "H", "I2",
                left=_geometry(speed=1.0),
                right=_geometry(speed=11.0),
            ),
        ],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    first = next(r for r in result.insert_candidates if r.source["inserted_identity_id"] == "I1")
    second = next(r for r in result.insert_candidates if r.source["inserted_identity_id"] == "I2")
    assert first.geometry_support > second.geometry_support


def test_insert_competition_is_ranked_per_host_gap_and_inserted_identity():
    profiles = [
        _profile("H1", [1.0, 0.0]),
        _profile("H2", [0.5, 0.866]),
        _profile("I", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[],
        insert_candidates=[
            _insert("H1", "I", gap=1),
            _insert("H2", "I", gap=1),
        ],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    best = max(
        result.insert_candidates,
        key=lambda row: row.aggregate_cosine,
    )
    assert best.inserted_reid_rank == 1


def test_missing_profile_leaves_score_unknown_not_negative():
    profiles = [_profile("A", [1.0, 0.0])]
    result = score_global_trajectory_candidates(
        append_candidates=[_append("A", "B")],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    row = result.append_candidates[0]
    assert row.appearance_available is False
    assert row.appearance_score is None
    assert row.trajectory_score is None
    assert row.geometry_support is not None


def test_mutual_best_is_based_on_final_score():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[_append("A", "B")],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    row = result.append_candidates[0]
    assert row.mutual_best is True
    assert row.mutual_top3 is True


def test_stage_never_selects_identity():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.1]),
    ]
    result = score_global_trajectory_candidates(
        append_candidates=[_append("A", "B")],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    assert result.append_candidates[0].to_dict()["identity_decision"] is None
    assert result.summary()["identity_decisions_changed"] is False


def test_neutral_appearance_plus_perfect_geometry_does_not_cross_point_70():
    """Regression: v1 sqrt(0.5 * 1.0) ~= 0.707 inflated neutral evidence."""
    profiles = [
        _profile("A", [1.0, 0.0], reliability=0.0),
        _profile("B", [1.0, 0.0], reliability=0.0),
    ]
    geometry = _geometry(speed=0.0, error=0.0, reliability=1.0)
    geometry["gap_s"] = 0.0
    result = score_global_trajectory_candidates(
        append_candidates=[_append("A", "B", geometry)],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    row = result.append_candidates[0]
    assert row.appearance_score == pytest.approx(0.5)
    assert row.trajectory_score < 0.70


def test_long_easy_gap_geometry_decays_toward_neutral():
    profiles = [
        _profile("A", [1.0, 0.0]),
        _profile("B", [0.99, 0.1]),
    ]
    short = _geometry(speed=0.0, error=0.0)
    short["gap_s"] = 0.1
    short["velocity_prediction_used"] = False
    long = _geometry(speed=0.0, error=0.0)
    long["gap_s"] = 12.0
    long["velocity_prediction_used"] = False

    result = score_global_trajectory_candidates(
        append_candidates=[
            _append("A", "B", short),
            _append("A", "B", long),
        ],
        insert_candidates=[],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    supports = sorted(
        [row.geometry_support for row in result.append_candidates],
        reverse=True,
    )
    assert supports[0] > supports[1]
    assert abs(supports[1] - 0.5) < 0.01


def test_insert_geometry_is_limited_by_weaker_boundary():
    profiles = [
        _profile("H", [1.0, 0.0]),
        _profile("I", [0.99, 0.1]),
    ]
    strong = _geometry(speed=0.0, error=0.0)
    strong["gap_s"] = 0.1
    strong["velocity_prediction_used"] = False
    weak = _geometry(speed=11.0, error=0.0)
    weak["gap_s"] = 0.1
    weak["velocity_prediction_used"] = False

    result = score_global_trajectory_candidates(
        append_candidates=[],
        insert_candidates=[_insert("H", "I", left=strong, right=weak)],
        profile_rows=profiles,
        calibration=_calibration(),
    )
    row = result.insert_candidates[0]
    assert row.geometry_support == pytest.approx(
        min(
            row.left_physical_effort_support,
            row.right_physical_effort_support,
        )
    )
