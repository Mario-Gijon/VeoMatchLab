import pytest

from tracking.actors.global_trajectory_apply import (
    apply_global_trajectory_proposals,
)


def _identity(identity_id, actors, tracklets, start, end, sources=None):
    return {
        "identity_id": identity_id,
        "source_identity_ids": sources or [identity_id],
        "actor_ids": actors,
        "tracklet_ids": tracklets,
        "start_time_s": start,
        "end_time_s": end,
        "seeded": True,
    }


def _actor(actor_id, identity_id):
    return {"actor_id": actor_id, "identity_id": identity_id}


def _tracklet(tracklet_id, identity_id):
    return {"tracklet_id": tracklet_id, "identity_id": identity_id}


def _append(a, b, score=0.72):
    return {
        "candidate_type": "append",
        "predecessor_identity_id": a,
        "successor_identity_id": b,
        "trajectory_score": score,
        "family_relation": "same_primary",
    }


def _insert(host, inserted, score=0.71):
    return {
        "candidate_type": "insert",
        "host_identity_id": host,
        "inserted_identity_id": inserted,
        "trajectory_score": score,
        "family_relation": "neutral",
    }


def _base():
    identities = [
        _identity("A", ["a1"], ["t1"], 0.0, 1.0, ["S1"]),
        _identity("B", ["a2"], ["t2"], 2.0, 3.0, ["S2"]),
        _identity("C", ["a3"], ["t3"], 0.0, 5.0, ["S3"]),
        _identity("D", ["a4"], ["t4"], 2.0, 3.0, ["S4"]),
        _identity("E", ["a5"], ["t5"], 6.0, 7.0, ["S5"]),
    ]
    actors = [
        _actor("a1", "A"),
        _actor("a2", "B"),
        _actor("a3", "C"),
        _actor("a4", "D"),
        _actor("a5", "E"),
    ]
    tracklets = [
        _tracklet("t1", "A"),
        _tracklet("t2", "B"),
        _tracklet("t3", "C"),
        _tracklet("t4", "D"),
        _tracklet("t5", "E"),
    ]
    return identities, actors, tracklets


def test_append_keeps_predecessor_as_stable_anchor():
    identities, actors, tracklets = _base()
    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actors,
        tracklet_assignment_rows=tracklets,
        proposal_rows=[_append("A", "B")],
        hard_identity_pairs=set(),
    )
    by_id = {row["identity_id"]: row for row in result.identities}
    assert "A" in by_id
    assert "B" not in by_id
    assert by_id["A"]["trajectory_immediate_source_identity_ids"] == [
        "A", "B"
    ]
    assert set(by_id["A"]["tracklet_ids"]) == {"t1", "t2"}


def test_insert_keeps_host_as_stable_anchor():
    identities, actors, tracklets = _base()
    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actors,
        tracklet_assignment_rows=tracklets,
        proposal_rows=[_insert("C", "D")],
        hard_identity_pairs=set(),
    )
    ids = {row["identity_id"] for row in result.identities}
    assert "C" in ids
    assert "D" not in ids


def test_unaffected_identity_ids_are_preserved():
    identities, actors, tracklets = _base()
    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actors,
        tracklet_assignment_rows=tracklets,
        proposal_rows=[_append("A", "B")],
        hard_identity_pairs=set(),
    )
    ids = {row["identity_id"] for row in result.identities}
    assert {"C", "D", "E"}.issubset(ids)


def test_actor_and_tracklet_assignments_are_remapped_and_conserved():
    identities, actors, tracklets = _base()
    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actors,
        tracklet_assignment_rows=tracklets,
        proposal_rows=[_append("A", "B"), _insert("C", "D")],
        hard_identity_pairs=set(),
    )
    actor_map = {
        row["actor_id"]: row["identity_id"]
        for row in result.actor_assignments
    }
    tracklet_map = {
        row["tracklet_id"]: row["identity_id"]
        for row in result.tracklet_assignments
    }
    assert actor_map["a2"] == "A"
    assert tracklet_map["t2"] == "A"
    assert actor_map["a4"] == "C"
    assert tracklet_map["t4"] == "C"
    assert len(actor_map) == 5
    assert len(tracklet_map) == 5


def test_overlapping_proposals_are_rejected():
    identities, actors, tracklets = _base()
    with pytest.raises(ValueError):
        apply_global_trajectory_proposals(
            identity_rows=identities,
            actor_assignment_rows=actors,
            tracklet_assignment_rows=tracklets,
            proposal_rows=[
                _append("A", "B"),
                _append("A", "E"),
            ],
            hard_identity_pairs=set(),
        )


def test_hard_conflicting_merge_is_rejected():
    identities, actors, tracklets = _base()
    with pytest.raises(ValueError):
        apply_global_trajectory_proposals(
            identity_rows=identities,
            actor_assignment_rows=actors,
            tracklet_assignment_rows=tracklets,
            proposal_rows=[_append("A", "B")],
            hard_identity_pairs={("A", "B")},
        )


def test_flattened_lineage_is_preserved():
    identities, actors, tracklets = _base()
    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actors,
        tracklet_assignment_rows=tracklets,
        proposal_rows=[_append("A", "B")],
        hard_identity_pairs=set(),
    )
    row = next(
        row for row in result.identities
        if row["identity_id"] == "A"
    )
    assert row["source_identity_ids"] == ["S1", "S2"]


def test_count_reduction_equals_number_of_disjoint_proposals():
    identities, actors, tracklets = _base()
    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actors,
        tracklet_assignment_rows=tracklets,
        proposal_rows=[_append("A", "B"), _insert("C", "D")],
        hard_identity_pairs=set(),
    )
    assert len(result.identities) == 3
    summary = result.summary()
    assert summary["source_identity_count"] == 5
    assert summary["accepted_proposal_count"] == 2
    assert summary["output_identity_count"] == 3
    assert summary["identity_count_reduction"] == 2
    assert summary["population_forced"] is False
