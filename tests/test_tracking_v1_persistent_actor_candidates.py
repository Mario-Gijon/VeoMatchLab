from __future__ import annotations

import pytest

from tracking.persistent.candidates import (
    PersistentActorCandidateConfig,
    build_candidate_diagnostics,
    estimate_population,
)


def _identity(identity_id: str, *tracklets: str):
    return {
        "identity_id": identity_id,
        "tracklet_ids": list(tracklets),
        "canonical": True,
        "provisional": False,
    }


def _point(identity_id: str, frame: int, time_s: float, x: float, z: float, *, segment: str = "s1", vx: float = 0.0, vz: float = 0.0):
    return {
        "identity_id": identity_id,
        "frame_index": frame,
        "timestamp_s": time_s,
        "smoothed_x": x,
        "smoothed_z": z,
        "measurement_x": x,
        "measurement_z": z,
        "measurement_confidence": 0.9,
        "position_std_m": 0.2,
        "vx_mps": vx,
        "vz_mps": vz,
        "segment_id": f"{identity_id}:{segment}",
        "origin": "observed",
    }


def _trajectory(identity_id: str, tracklet: str, points):
    return {
        "identity_id": identity_id,
        "tracklet_ids": [tracklet],
        "canonical": True,
        "provisional": False,
        "points": list(points),
    }


def _physical(a: str, b: str, status: str = "candidate", reason: str = "transition_reachable"):
    first, second = sorted((a, b))
    return {
        "tracklet_a": first,
        "tracklet_b": second,
        "status": status,
        "reason": reason,
    }


def _appearance(a: str, b: str, cosine: float, reliability: float = 0.9):
    first, second = sorted((a, b))
    return {
        "tracklet_a": first,
        "tracklet_b": second,
        "aggregate_cosine": cosine,
        "reliability": reliability,
        "support_quality": 0.9,
    }


def test_population_estimator_is_not_hardcoded_to_23():
    trajectories = []
    for index in range(5):
        identity_id = f"identity-{index:03d}"
        trajectories.append(
            _trajectory(
                identity_id,
                f"trk-{index:03d}",
                [_point(identity_id, frame, frame / 10, index, 0) for frame in range(10)],
            )
        )
    estimate = estimate_population(trajectories)
    assert estimate["estimated_population"] == 5
    assert estimate["confidence"] == "high"


def test_population_ignores_a_rare_high_spike():
    trajectories = []
    # Four actors across every frame.
    for index in range(4):
        identity_id = f"identity-{index:03d}"
        trajectories.append(
            _trajectory(
                identity_id,
                f"trk-{index:03d}",
                [_point(identity_id, frame, frame / 10, index, 0) for frame in range(20)],
            )
        )
    # A fifth false/transient identity exists in only two frames (<25%).
    trajectories.append(
        _trajectory(
            "identity-999",
            "trk-999",
            [_point("identity-999", 5, 0.5, 9, 0), _point("identity-999", 6, 0.6, 9, 0)],
        )
    )
    estimate = estimate_population(trajectories)
    assert estimate["estimated_population"] == 4
    assert estimate["max_population"] == 5


def test_v1_hard_constraint_can_never_be_overridden_by_appearance():
    identities = [_identity("identity-A", "trk-A"), _identity("identity-B", "trk-B")]
    trajectories = [
        _trajectory("identity-A", "trk-A", [_point("identity-A", 0, 0.0, 0, 0), _point("identity-A", 3, 0.1, 0.2, 0)]),
        _trajectory("identity-B", "trk-B", [_point("identity-B", 30, 1.0, 0.4, 0), _point("identity-B", 33, 1.1, 0.6, 0)]),
    ]
    result = build_candidate_diagnostics(
        identity_rows=identities,
        trajectory_rows=trajectories,
        appearance_rows=[_appearance("trk-A", "trk-B", 0.999)],
        physical_candidate_rows=[_physical("trk-A", "trk-B", "cannot_link", "transition_unreachable")],
    )
    pair = result["pairs"][0]
    assert pair["status"] == "cannot_link"
    assert pair["link_score"] is None
    assert pair["hard_block_reason"] == "v1_hard_constraint"


def test_sequential_candidate_contains_motion_evidence_without_forcing_a_merge():
    identities = [_identity("identity-A", "trk-A"), _identity("identity-B", "trk-B")]
    trajectories = [
        _trajectory("identity-A", "trk-A", [_point("identity-A", 0, 0.0, 0, 0, vx=1), _point("identity-A", 3, 0.1, 0.1, 0, vx=1)]),
        _trajectory("identity-B", "trk-B", [_point("identity-B", 15, 0.5, 0.5, 0, vx=1), _point("identity-B", 18, 0.6, 0.6, 0, vx=1)]),
    ]
    result = build_candidate_diagnostics(
        identity_rows=identities,
        trajectory_rows=trajectories,
        appearance_rows=[],
        physical_candidate_rows=[_physical("trk-A", "trk-B")],
    )
    pair = result["pairs"][0]
    assert pair["status"] == "candidate"
    assert pair["relation"] == "sequential"
    assert pair["solver_eligible"] is True
    assert pair["predecessor_identity_id"] == "identity-A"
    assert pair["successor_identity_id"] == "identity-B"
    assert pair["motion"]["hard_required_speed_mps"] < 12.0
    assert pair["motion"]["raw_transition_speed_mps"] > pair["motion"]["hard_required_speed_mps"]
    assert pair["appearance"]["available"] is False
    assert pair["score_components"]["appearance"] is None
    assert pair["link_score"] is not None
    assert pair["identity_decision"] is None


def test_impossible_endpoint_transition_is_hard_blocked_even_if_3a_row_is_candidate():
    identities = [_identity("identity-A", "trk-A"), _identity("identity-B", "trk-B")]
    trajectories = [
        _trajectory("identity-A", "trk-A", [_point("identity-A", 0, 0.0, 0, 0), _point("identity-A", 3, 0.1, 0, 0)]),
        _trajectory("identity-B", "trk-B", [_point("identity-B", 6, 0.2, 20, 0), _point("identity-B", 9, 0.3, 20, 0)]),
    ]
    result = build_candidate_diagnostics(
        identity_rows=identities,
        trajectory_rows=trajectories,
        appearance_rows=[],
        physical_candidate_rows=[_physical("trk-A", "trk-B")],
        config=PersistentActorCandidateConfig(max_transition_speed_mps=12.0, transition_position_slack_m=1.5),
    )
    pair = result["pairs"][0]
    assert pair["status"] == "cannot_link"
    assert pair["hard_block_reason"] == "endpoint_transition_impossible"
    assert pair["link_score"] is None


def test_overlap_is_preserved_for_later_duplicate_handling_not_path_cover():
    identities = [_identity("identity-A", "trk-A"), _identity("identity-B", "trk-B")]
    trajectories = [
        _trajectory("identity-A", "trk-A", [_point("identity-A", 0, 0.0, 0, 0), _point("identity-A", 3, 0.1, 0.1, 0)]),
        _trajectory("identity-B", "trk-B", [_point("identity-B", 3, 0.1, 0.15, 0), _point("identity-B", 6, 0.2, 0.2, 0)]),
    ]
    result = build_candidate_diagnostics(
        identity_rows=identities,
        trajectory_rows=trajectories,
        appearance_rows=[],
        physical_candidate_rows=[_physical("trk-A", "trk-B")],
    )
    pair = result["pairs"][0]
    assert pair["status"] == "candidate"
    assert pair["relation"] == "overlap"
    assert pair["solver_eligible"] is False
    assert pair["overlap"]["common_frame_count"] == 1



def test_soft_speed_uses_raw_displacement_not_hard_slack():
    identities = [_identity("identity-A", "trk-A"), _identity("identity-B", "trk-B")]
    trajectories = [
        _trajectory("identity-A", "trk-A", [_point("identity-A", 0, 0.0, 0, 0), _point("identity-A", 3, 0.1, 0, 0)]),
        _trajectory("identity-B", "trk-B", [_point("identity-B", 6, 0.2, 0.73, 0), _point("identity-B", 9, 0.3, 0.8, 0)]),
    ]
    result = build_candidate_diagnostics(
        identity_rows=identities,
        trajectory_rows=trajectories,
        appearance_rows=[],
        physical_candidate_rows=[_physical("trk-A", "trk-B")],
    )
    motion = result["pairs"][0]["motion"]
    assert motion["raw_transition_speed_mps"] == pytest.approx(7.3)
    assert motion["hard_required_speed_mps"] == 0.0
    assert result["pairs"][0]["score_components"]["speed"] < 1.0


def test_population_uses_mode_not_stable_raw_spike_and_reports_hard_distinct_bound():
    identities = [_identity(f"identity-{index:03d}", f"trk-{index:03d}") for index in range(5)]
    trajectories = []
    # Four identities are present for 20 frames. A duplicate fifth fragment is
    # present for six frames (30%), which used to push stable-high occupancy to 5.
    for index in range(4):
        identity_id = f"identity-{index:03d}"
        trajectories.append(
            _trajectory(
                identity_id,
                f"trk-{index:03d}",
                [_point(identity_id, frame, frame / 10, index * 5.0, 0) for frame in range(20)],
            )
        )
    trajectories.append(
        _trajectory(
            "identity-004",
            "trk-004",
            [_point("identity-004", frame, frame / 10, 0.1, 0) for frame in range(6)],
        )
    )

    physical = []
    # The first four are pairwise hard-distinct. The fifth duplicate is not hard
    # constrained against identity-000.
    for a in range(4):
        for b in range(a + 1, 4):
            physical.append(_physical(f"trk-{a:03d}", f"trk-{b:03d}", "cannot_link", "simultaneous_spatial_separation"))
    for b in range(1, 4):
        physical.append(_physical("trk-004", f"trk-{b:03d}", "cannot_link", "simultaneous_spatial_separation"))
    physical.append(_physical("trk-000", "trk-004", "candidate", "overlap_close"))

    estimate = estimate_population(
        trajectories,
        identity_rows=identities,
        physical_candidate_rows=physical,
    )
    assert estimate["raw_modal_population"] == 4
    assert estimate["raw_max_population"] == 5
    assert estimate["estimated_population"] == 4
    assert estimate["distinct_max_population"] == 4
    assert estimate["confidence"] == "high"


def test_trusted_v1_appearance_calibration_overrides_bad_empirical_separation():
    identities = [
        _identity("identity-A", "trk-A1", "trk-A2"),
        _identity("identity-B", "trk-B"),
    ]
    trajectory_a = _trajectory("identity-A", "trk-A1", [_point("identity-A", 0, 0.0, 0, 0), _point("identity-A", 3, 0.1, 0.1, 0)])
    trajectory_a["tracklet_ids"] = ["trk-A1", "trk-A2"]
    trajectories = [
        trajectory_a,
        _trajectory("identity-B", "trk-B", [_point("identity-B", 30, 1.0, 1, 0), _point("identity-B", 33, 1.1, 1.1, 0)]),
    ]
    # Deliberately poor empirical internal positive; trusted V1 calibration is clean.
    appearance = [
        _appearance("trk-A1", "trk-A2", 0.70),
        _appearance("trk-A1", "trk-B", 0.90),
        _appearance("trk-A2", "trk-B", 0.89),
    ]
    physical = [
        _physical("trk-A1", "trk-A2"),
        _physical("trk-A1", "trk-B", "cannot_link", "simultaneous_spatial_separation"),
        _physical("trk-A2", "trk-B", "cannot_link", "simultaneous_spatial_separation"),
    ]
    trusted = {
        "appearance": {
            "negative_anchor": {"value": 0.905},
            "positive_anchor": {"value": 0.935},
        }
    }
    result = build_candidate_diagnostics(
        identity_rows=identities,
        trajectory_rows=trajectories,
        appearance_rows=appearance,
        physical_candidate_rows=physical,
        trusted_appearance_calibration=trusted,
    )
    calibration = result["appearance_calibration"]
    assert calibration["available"] is True
    assert calibration["source"] == "tracking-v1-soft-link-calibration"
    assert calibration["negative_p95"] == pytest.approx(0.905)
    assert calibration["positive_p10"] == pytest.approx(0.935)
