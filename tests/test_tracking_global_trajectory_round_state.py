import pytest

from tracking.actors.global_trajectory_round_state import (
    build_global_trajectory_round_state,
)


def _identity(identity_id, actors, tracklets, start=0.0, end=1.0):
    return {
        "identity_id": identity_id,
        "actor_ids": actors,
        "tracklet_ids": tracklets,
        "start_time_s": start,
        "end_time_s": end,
    }


def _chain(actor_id, vector, tracklet_count=2):
    return {
        "actor_id": actor_id,
        "aggregate_embedding": vector,
        "reliability": 0.8,
        "profiled_tracklet_count": tracklet_count,
        "kit_cluster_id": None,
        "kit_dominance": None,
    }


def _assignment(tracklet_id, identity_id):
    return {
        "tracklet_id": tracklet_id,
        "identity_id": identity_id,
    }


def _family(tracklet_id, family_id=0):
    return {
        "tracklet_id": tracklet_id,
        "candidate_family_type": "primary",
        "candidate_family_id": family_id,
        "state": "confident_primary",
        "dominance": 1.0,
        "known_fraction": 1.0,
        "total_quality_mass": 4.0,
        "total_samples": 3,
        "support_samples": 3,
    }


def test_refresh_builds_one_profile_and_family_row_per_supported_identity():
    state = build_global_trajectory_round_state(
        identity_rows=[
            _identity("P", ["A"], ["T"]),
        ],
        tracklet_assignment_rows=[
            _assignment("T", "P"),
        ],
        chain_profile_rows=[
            _chain("A", [1.0, 0.0]),
        ],
        tracklet_family_rows=[
            _family("T"),
        ],
    )
    assert state.identity_count == 1
    assert len(state.profiles) == 1
    assert len(state.family_rows) == 1
    assert state.profiles[0].identity_id == "P"
    assert state.family_rows[0].entity_id == "P"


def test_merged_identity_reaggregates_multiple_actor_profiles():
    state = build_global_trajectory_round_state(
        identity_rows=[
            _identity("P", ["A", "B"], ["T1", "T2"]),
        ],
        tracklet_assignment_rows=[
            _assignment("T1", "P"),
            _assignment("T2", "P"),
        ],
        chain_profile_rows=[
            _chain("A", [1.0, 0.0]),
            _chain("B", [0.8, 0.2]),
        ],
        tracklet_family_rows=[
            _family("T1"),
            _family("T2"),
        ],
    )
    profile = state.profiles[0]
    assert profile.profiled_actor_ids == ("A", "B")
    assert profile.profiled_tracklet_count == 4


def test_identity_without_actor_profile_remains_unprofiled_not_error():
    state = build_global_trajectory_round_state(
        identity_rows=[
            _identity("P", ["A"], ["T"]),
        ],
        tracklet_assignment_rows=[
            _assignment("T", "P"),
        ],
        chain_profile_rows=[],
        tracklet_family_rows=[
            _family("T"),
        ],
    )
    assert len(state.profiles) == 0
    assert state.summary()["unprofiled_identity_count"] == 1


def test_tracklet_coverage_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_global_trajectory_round_state(
            identity_rows=[
                _identity("P", ["A"], ["T"]),
            ],
            tracklet_assignment_rows=[],
            chain_profile_rows=[
                _chain("A", [1.0, 0.0]),
            ],
            tracklet_family_rows=[
                _family("T"),
            ],
        )


def test_tracklet_owner_mismatch_is_rejected():
    with pytest.raises(ValueError):
        build_global_trajectory_round_state(
            identity_rows=[
                _identity("P", ["A"], ["T"]),
                _identity("Q", ["B"], ["U"]),
            ],
            tracklet_assignment_rows=[
                _assignment("T", "Q"),
                _assignment("U", "Q"),
            ],
            chain_profile_rows=[
                _chain("A", [1.0, 0.0]),
                _chain("B", [0.0, 1.0]),
            ],
            tracklet_family_rows=[
                _family("T"),
                _family("U"),
            ],
        )


def test_refresh_does_not_change_identity_decisions():
    state = build_global_trajectory_round_state(
        identity_rows=[
            _identity("P", ["A"], ["T"]),
        ],
        tracklet_assignment_rows=[
            _assignment("T", "P"),
        ],
        chain_profile_rows=[
            _chain("A", [1.0, 0.0]),
        ],
        tracklet_family_rows=[
            _family("T"),
        ],
    )
    summary = state.summary()
    assert summary["identity_decisions_changed"] is False
    assert summary["population_forced"] is False
