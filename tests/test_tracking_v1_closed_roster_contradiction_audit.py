from tracking.actors.closed_roster_contradiction_audit import (
    ContradictionAuditConfig,
    build_contradiction_audit,
)


def _tracklet(tid, points):
    return {
        "tracklet_id": tid,
        "observations": [
            {
                "detection_id": f"{tid}-{i}",
                "frame_index": frame,
                "timestamp_s": frame / 10.0,
                "x": x,
                "z": z,
                "confidence": .95,
            }
            for i, (frame, x, z) in enumerate(points)
        ],
    }


def _family(tid, key):
    return {
        "tracklet_id": tid,
        "family_key": key,
        "candidate_family_key": key,
        "state": "confident_primary",
    }


def _config(n=2):
    return ContradictionAuditConfig(expected_actor_slots=n)


def test_audit_targets_only_two_suspicious_rules():
    tracklets = [
        _tracklet("a", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("b", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, .2, 0)]),
    ]
    actors = [{"actor_id": "actor-0001"}, {"actor_id": "actor-0002"}]
    assignments = [
        {"tracklet_id": "a", "actor_id": "actor-0001", "initial_assignment": True},
        {"tracklet_id": "b", "actor_id": "actor-0002", "initial_assignment": True},
        {"tracklet_id": "x", "actor_id": "actor-0001", "initial_assignment": False},
    ]
    decisions = [
        {"tracklet_id": "x", "status": "assigned",
         "rule": "closed_world_best_viable_actor",
         "best_actor_score": .4, "actor_margin": .01},
        {"tracklet_id": "a", "status": "assigned",
         "rule": "bidirectional_bridge"},
    ]

    targets, pairs, summary = build_contradiction_audit(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        decision_rows=decisions,
        config=_config(),
    )

    assert len(targets) == 1
    assert targets[0]["tracklet_id"] == "x"
    assert len(pairs) == 2
    assert summary["target_count"] == 1


def test_forced_target_is_removed_from_own_actor_during_audit():
    tracklets = [
        _tracklet("a", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("b", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, .2, 0), (3, .3, 0)]),
    ]
    actors = [{"actor_id": "actor-0001"}, {"actor_id": "actor-0002"}]
    assignments = [
        {"tracklet_id": "a", "actor_id": "actor-0001", "initial_assignment": True},
        {"tracklet_id": "b", "actor_id": "actor-0002", "initial_assignment": True},
        {"tracklet_id": "x", "actor_id": "actor-0001", "initial_assignment": False},
    ]
    decisions = [{
        "tracklet_id": "x",
        "status": "assigned",
        "rule": "closed_world_best_viable_actor",
        "best_actor_score": .4,
        "actor_margin": .01,
    }]

    targets, pairs, _ = build_contradiction_audit(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        decision_rows=decisions,
        config=_config(),
    )

    own = next(
        row for row in pairs
        if row["tracklet_id"] == "x"
        and row["actor_id"] == "actor-0001"
    )
    assert "exact_frame_coexistence" not in own["blockers"]
    assert targets[0]["chosen_actor_audit"]["actor_id"] == "actor-0001"


def test_initial_coexistence_pressure_is_reported():
    tracklets = [
        _tracklet("a", [(0, 0, 0), (2, .2, 0)]),
        _tracklet("b", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, .3, 0)]),
    ]
    actors = [{"actor_id": "actor-0001"}, {"actor_id": "actor-0002"}]
    assignments = [
        {"tracklet_id": "a", "actor_id": "actor-0001", "initial_assignment": True},
        {"tracklet_id": "b", "actor_id": "actor-0002", "initial_assignment": True},
    ]
    decisions = [{
        "tracklet_id": "x",
        "status": "non_roster",
        "rule": "no_physically_viable_actor",
    }]

    targets, pairs, summary = build_contradiction_audit(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        decision_rows=decisions,
        config=_config(),
    )

    actor1 = next(
        row for row in pairs if row["actor_id"] == "actor-0001"
    )
    assert "exact_frame_coexistence" in actor1["blockers"]
    assert actor1["initial_only_partition_pressure"] is True
    assert targets[0]["initial_assignment_pressure_signal"] is True
    assert summary[
        "no_viable_initial_assignment_pressure_signal_count"
    ] == 1


def test_confident_family_mismatch_is_explicit_blocker():
    tracklets = [
        _tracklet("a", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("b", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, .2, 0)]),
    ]
    actors = [{"actor_id": "actor-0001"}, {"actor_id": "actor-0002"}]
    assignments = [
        {"tracklet_id": "a", "actor_id": "actor-0001", "initial_assignment": True},
        {"tracklet_id": "b", "actor_id": "actor-0002", "initial_assignment": True},
    ]
    decisions = [{
        "tracklet_id": "x",
        "status": "non_roster",
        "rule": "no_physically_viable_actor",
    }]
    families = [
        _family("a", "primary-0"),
        _family("b", "primary-1"),
        _family("x", "primary-1"),
    ]

    _, pairs, _ = build_contradiction_audit(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        decision_rows=decisions,
        family_rows=families,
        config=_config(),
    )

    actor1 = next(
        row for row in pairs if row["actor_id"] == "actor-0001"
    )
    assert "confident_family_mismatch" in actor1["blockers"]


def test_before_geometry_blocker_records_source_tracklet():
    tracklets = [
        _tracklet("a", [(0, 0, 0), (1, 0, 0)]),
        _tracklet("b", [(0, 10, 0), (1, 10, 0)]),
        _tracklet("x", [(2, 50, 0)]),
    ]
    actors = [{"actor_id": "actor-0001"}, {"actor_id": "actor-0002"}]
    assignments = [
        {"tracklet_id": "a", "actor_id": "actor-0001", "initial_assignment": True},
        {"tracklet_id": "b", "actor_id": "actor-0002", "initial_assignment": True},
    ]
    decisions = [{
        "tracklet_id": "x",
        "status": "non_roster",
        "rule": "no_physically_viable_actor",
    }]

    _, pairs, _ = build_contradiction_audit(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        decision_rows=decisions,
        config=_config(),
    )

    actor1 = next(
        row for row in pairs if row["actor_id"] == "actor-0001"
    )
    assert "before_geometry" in actor1["blockers"]
    assert actor1["before"]["tracklet_id"] == "a"
    assert actor1["before"]["initial_assignment"] is True


def test_short_off_field_tracklet_gets_strong_non_roster_signal():
    tracklets = [
        _tracklet("a", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("b", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("coach", [(2, 60, 40), (3, 60.1, 40)]),
    ]
    actors = [{"actor_id": "actor-0001"}, {"actor_id": "actor-0002"}]
    assignments = [
        {"tracklet_id": "a", "actor_id": "actor-0001", "initial_assignment": True},
        {"tracklet_id": "b", "actor_id": "actor-0002", "initial_assignment": True},
    ]
    decisions = [{
        "tracklet_id": "coach",
        "status": "non_roster",
        "rule": "no_physically_viable_actor",
    }]

    targets, _, _ = build_contradiction_audit(
        tracklet_rows=tracklets,
        actor_rows=actors,
        assignment_rows=assignments,
        decision_rows=decisions,
        config=_config(),
    )

    assert targets[0]["off_field_ratio"] == 1.0
    assert targets[0]["strong_non_roster_signal"] is True
