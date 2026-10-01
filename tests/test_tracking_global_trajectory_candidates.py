from __future__ import annotations

from tracking.actors.global_trajectory_candidates import (
    GlobalTrajectoryCandidateConfig,
    build_global_trajectory_candidates,
)


def _endpoint(t, x, z, vx=None, vz=None, reliability=0.8):
    velocity = None
    if vx is not None and vz is not None:
        velocity = {
            "vx_mps": vx,
            "vz_mps": vz,
            "reliability": reliability,
        }
    return {
        "frame_index": int(round(t * 10)),
        "timestamp_s": t,
        "x": x,
        "z": z,
        "observation_confidence": 0.9,
        "tracklet_id": f"T-{t}-{x}-{z}",
        "velocity": velocity,
    }


def _profile(identity_id, start, end, segments=None, gaps=None):
    if segments is None:
        segments = [{
            "segment_index": 1,
            "start": start,
            "end": end,
        }]
    return {
        "identity_id": identity_id,
        "start": start,
        "end": end,
        "segments": segments,
        "internal_gaps": list(gaps or []),
    }


def _family(
    identity_id,
    *,
    state="confident_primary",
    family_key="primary-0",
    usable=True,
    candidate_family_key=None,
    candidate_family_type=None,
    dominance=1.0,
    confidence=0.9,
    coverage=0.9,
):
    return {
        "identity_id": identity_id,
        "state": state,
        "family_type": "primary" if usable else "unknown",
        "family_key": family_key if usable else None,
        "usable_for_tracking": usable,
        "candidate_family_type": candidate_family_type,
        "candidate_family_key": candidate_family_key,
        "dominance": dominance,
        "confidence": confidence,
        "evidence_coverage": coverage,
    }


def test_reachable_append_is_emitted():
    a = _profile("A", _endpoint(0, 0, 0), _endpoint(1, 1, 0))
    b = _profile("B", _endpoint(2, 5, 0), _endpoint(3, 6, 0))
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
    )
    assert len(result.append_candidates) == 1
    row = result.append_candidates[0]
    assert row.predecessor_identity_id == "A"
    assert row.successor_identity_id == "B"
    assert row.geometry.physically_reachable is True


def test_physically_impossible_append_is_rejected():
    a = _profile("A", _endpoint(0, 0, 0), _endpoint(1, 0, 0))
    b = _profile("B", _endpoint(1.1, 20, 0), _endpoint(2, 20, 0))
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
    )
    assert len(result.append_candidates) == 0
    assert result.append_physically_rejected_count == 1


def test_append_horizon_limits_retrieval_not_identity():
    a = _profile("A", _endpoint(0, 0, 0), _endpoint(1, 0, 0))
    b = _profile("B", _endpoint(30, 1, 0), _endpoint(31, 2, 0))
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
        config=GlobalTrajectoryCandidateConfig(max_append_gap_s=5.0),
    )
    assert len(result.append_candidates) == 0
    assert result.append_pair_evaluated_count == 0


def test_insert_requires_entire_identity_inside_host_gap():
    host = _profile(
        "H",
        _endpoint(0, 0, 0),
        _endpoint(10, 10, 0),
        segments=[
            {
                "segment_index": 1,
                "start": _endpoint(0, 0, 0),
                "end": _endpoint(2, 2, 0),
            },
            {
                "segment_index": 2,
                "start": _endpoint(8, 8, 0),
                "end": _endpoint(10, 10, 0),
            },
        ],
        gaps=[{
            "gap_index": 1,
            "preceding_segment_index": 1,
            "following_segment_index": 2,
            "start_time_s": 2.0,
            "end_time_s": 8.0,
        }],
    )
    inside = _profile(
        "I",
        _endpoint(3, 3, 0),
        _endpoint(7, 7, 0),
    )
    crossing = _profile(
        "X",
        _endpoint(3, 3, 0),
        _endpoint(9, 9, 0),
    )

    result = build_global_trajectory_candidates(
        trajectory_profiles=[host, inside, crossing],
    )
    ids = {
        row.inserted_identity_id
        for row in result.insert_candidates
        if row.host_identity_id == "H"
    }
    assert "I" in ids
    assert "X" not in ids


def test_insert_requires_both_sides_physically_reachable():
    host = _profile(
        "H",
        _endpoint(0, 0, 0),
        _endpoint(4, 0, 0),
        segments=[
            {
                "segment_index": 1,
                "start": _endpoint(0, 0, 0),
                "end": _endpoint(1, 0, 0),
            },
            {
                "segment_index": 2,
                "start": _endpoint(3, 0, 0),
                "end": _endpoint(4, 0, 0),
            },
        ],
        gaps=[{
            "gap_index": 1,
            "preceding_segment_index": 1,
            "following_segment_index": 2,
            "start_time_s": 1.0,
            "end_time_s": 3.0,
        }],
    )
    bad = _profile(
        "B",
        _endpoint(1.1, 20, 0),
        _endpoint(2.9, 0, 0),
    )
    result = build_global_trajectory_candidates(
        trajectory_profiles=[host, bad],
    )
    assert not [
        row for row in result.insert_candidates
        if row.host_identity_id == "H"
        and row.inserted_identity_id == "B"
    ]


def test_velocity_prediction_is_not_used_beyond_horizon():
    a = _profile(
        "A",
        _endpoint(0, 0, 0),
        _endpoint(1, 1, 0, vx=1, vz=0),
    )
    b = _profile(
        "B",
        _endpoint(6, 6, 0, vx=1, vz=0),
        _endpoint(7, 7, 0),
    )
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
        config=GlobalTrajectoryCandidateConfig(
            velocity_use_horizon_s=2.0,
        ),
    )
    row = result.append_candidates[0]
    assert row.geometry.velocity_prediction_used is False
    assert row.geometry.forward_prediction_error_m is None


def test_confident_primary_relations_are_attached():
    a = _profile("A", _endpoint(0, 0, 0), _endpoint(1, 0, 0))
    b = _profile("B", _endpoint(2, 1, 0), _endpoint(3, 2, 0))
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
        family_rows=[
            _family("A", family_key="primary-0"),
            _family("B", family_key="primary-1"),
        ],
    )
    assert result.append_candidates[0].family_relation == "different_primary"


def test_weak_opposite_primary_is_probable_not_hard():
    a = _profile("A", _endpoint(0, 0, 0), _endpoint(1, 0, 0))
    b = _profile("B", _endpoint(2, 1, 0), _endpoint(3, 2, 0))
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
        family_rows=[
            _family("A", family_key="primary-0"),
            _family(
                "B",
                state="weak_primary",
                usable=False,
                candidate_family_type="primary",
                candidate_family_key="primary-1",
                dominance=0.7,
                confidence=0.4,
                coverage=0.4,
            ),
        ],
    )
    row = result.append_candidates[0]
    assert row.family_relation == "probable_different_primary"
    assert row.geometry.physically_reachable is True


def test_stage_never_changes_identity_decision():
    a = _profile("A", _endpoint(0, 0, 0), _endpoint(1, 0, 0))
    b = _profile("B", _endpoint(2, 1, 0), _endpoint(3, 2, 0))
    result = build_global_trajectory_candidates(
        trajectory_profiles=[a, b],
    )
    assert result.append_candidates[0].to_dict()["identity_decision"] is None
    assert result.summary()["identity_decisions_changed"] is False
