from tracking.actors.fixed_actor_roster import (
    FixedActorRosterConfig,
    build_fixed_actor_roster,
)


def _tracklet(identifier, start_frame, points):
    observations = []
    for offset, (frame, x, z) in enumerate(points):
        observations.append({
            "detection_id": f"{identifier}-{offset}",
            "frame_index": frame,
            "timestamp_s": frame / 10.0,
            "x": x,
            "z": z,
            "confidence": 0.95,
        })
    return {
        "tracklet_id": identifier,
        "observations": observations,
        "metadata": {},
    }


def _family(tracklet_id, key, confident=True):
    kind, raw = key.split("-")
    return {
        "tracklet_id": tracklet_id,
        "family_type": kind if confident else "unknown",
        "family_id": int(raw) if confident else None,
        "family_key": key if confident else None,
        "candidate_family_type": kind,
        "candidate_family_id": int(raw),
        "candidate_family_key": key,
        "state": (
            "confident_primary"
            if confident and kind == "primary"
            else "confident_special"
            if confident
            else "weak_primary"
            if kind == "primary"
            else "weak_special"
        ),
    }


def _profile(tracklet_id, vector):
    return {
        "tracklet_id": tracklet_id,
        "aggregate_embedding": list(vector),
        "reliability": 1.0,
        "effective_sample_count": 3.0,
    }


def _assignments(result):
    return {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }


def _config(actor_slots=2, **overrides):
    values = {
        "actor_slots": actor_slots,
        "bootstrap_window_s": 1.0,
        "bootstrap_fill_window_s": 1.0,
        "local_gap_s": 1.25,
        "recovery_gap_s": 5.0,
    }
    values.update(overrides)
    return FixedActorRosterConfig(**values)


def test_roster_has_fixed_number_of_actor_slots():
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.1, 0.0)]),
        _tracklet("b", 0, [(0, 10.0, 0.0), (1, 10.1, 0.0)]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        config=_config(actor_slots=2),
    )
    assert len(result.actors) == 2
    assert result.summary()["actor_count_is_hard_capped"] is True
    assert result.summary()["new_actor_creation_after_bootstrap"] is False


def test_recent_new_tracklet_reuses_existing_slot():
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.2, 0.0)]),
        _tracklet("b", 0, [(0, 10.0, 0.0), (1, 10.2, 0.0)]),
        _tracklet("a2", 2, [(2, 0.4, 0.0), (3, 0.6, 0.0)]),
    ]
    families = [
        _family("a", "primary-0"),
        _family("b", "primary-1"),
        _family("a2", "primary-0"),
    ]
    profiles = [
        _profile("a", [1, 0]),
        _profile("b", [0, 1]),
        _profile("a2", [1, 0]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        family_rows=families,
        appearance_rows=profiles,
        config=_config(actor_slots=2),
    )
    ids = _assignments(result)
    assert ids["a2"] == ids["a"]
    assert len(result.actors) == 2


def test_exact_frame_coexistence_blocks_same_slot():
    # Seed a is observed again at frame 3. New x begins frame 2 and also has
    # an observation at frame 3, so they cannot be one physical actor.
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.1, 0.0), (3, 0.3, 0.0)]),
        _tracklet("b", 0, [(0, 10.0, 0.0), (1, 10.1, 0.0)]),
        _tracklet("x", 2, [(2, 0.2, 0.0), (3, 0.3, 0.0)]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        config=_config(actor_slots=2),
    )
    ids = _assignments(result)
    assert ids.get("x") != ids["a"]
    assert result.exact_coexistence_rejection_count >= 1


def test_confident_family_contradiction_blocks_handoff():
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.1, 0.0)]),
        _tracklet("b", 0, [(0, 10.0, 0.0), (1, 10.1, 0.0)]),
        _tracklet("x", 2, [(2, 0.2, 0.0), (3, 0.3, 0.0)]),
    ]
    families = [
        _family("a", "primary-0"),
        _family("b", "primary-1"),
        _family("x", "primary-1"),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        family_rows=families,
        config=_config(actor_slots=2),
    )
    ids = _assignments(result)
    assert ids.get("x") != ids["a"]
    assert result.hard_family_rejection_count >= 1


def test_global_batch_assignment_uses_two_different_slots():
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.2, 0.0)]),
        _tracklet("b", 0, [(0, 2.0, 0.0), (1, 1.8, 0.0)]),
        _tracklet("x", 2, [(2, 0.5, 0.0), (3, 0.7, 0.0)]),
        _tracklet("y", 2, [(2, 1.5, 0.0), (3, 1.3, 0.0)]),
    ]
    profiles = [
        _profile("a", [1, 0]),
        _profile("b", [0, 1]),
        _profile("x", [1, 0]),
        _profile("y", [0, 1]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        appearance_rows=profiles,
        config=_config(
            actor_slots=2,
            geometry_weight=0.20,
            reid_weight=0.55,
        ),
    )
    ids = _assignments(result)
    assert ids["x"] != ids["y"]
    assert ids["x"] == ids["a"]
    assert ids["y"] == ids["b"]


def test_recovery_requires_reid_support():
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.2, 0.0)]),
        _tracklet("b", 0, [(0, 10.0, 0.0), (1, 10.2, 0.0)]),
        _tracklet("late", 20, [(20, 1.0, 0.0), (21, 1.2, 0.0)]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        appearance_rows=[],
        config=_config(actor_slots=2, local_gap_s=0.5, recovery_gap_s=3.0),
    )
    ids = _assignments(result)
    assert "late" not in ids


def test_unmatched_tracklet_never_creates_actor():
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.1, 0.0)]),
        _tracklet("b", 0, [(0, 10.0, 0.0), (1, 10.1, 0.0)]),
        _tracklet("far", 2, [(2, 500.0, 0.0), (3, 500.1, 0.0)]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        config=_config(actor_slots=2),
    )
    assert len(result.actors) == 2
    assert any(
        row["tracklet_id"] == "far"
        for row in result.unassigned_tracklets
    )


def test_bootstrap_can_leave_fixed_slot_empty_then_fill_without_new_actor():
    # Bootstrap best frame has one real observation although two slots exist.
    # A second actor appears shortly afterwards and initializes the empty slot.
    tracklets = [
        _tracklet("a", 0, [(0, 0.0, 0.0), (1, 0.1, 0.0)]),
        _tracklet("new", 2, [(2, 20.0, 0.0), (3, 20.1, 0.0)]),
    ]
    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        config=_config(
            actor_slots=2,
            bootstrap_window_s=0.15,
            bootstrap_fill_window_s=1.0,
        ),
    )
    assert len(result.actors) == 2
    assert sum(row["initialized"] for row in result.actors) == 2
    assert result.summary()["new_actor_creation_after_bootstrap"] is False
