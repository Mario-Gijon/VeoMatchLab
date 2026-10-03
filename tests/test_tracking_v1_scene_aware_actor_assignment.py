from tracking.actors.scene_aware_actor_assignment import (
    SceneAwareActorConfig,
    build_scene_aware_actor_assignment,
)


def _tracklet(identifier, frames, xs, z=0.0):
    observations = []
    for i, (frame, x) in enumerate(zip(frames, xs)):
        observations.append({
            "detection_id": f"{identifier}-{i}",
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
    family_type, raw_id = key.split("-")
    return {
        "tracklet_id": tracklet_id,
        "family_type": family_type if confident else "unknown",
        "family_id": int(raw_id) if confident else None,
        "family_key": key if confident else None,
        "candidate_family_type": family_type,
        "candidate_family_id": int(raw_id),
        "candidate_family_key": key,
        "state": (
            "confident_primary"
            if confident and family_type == "primary"
            else "confident_special"
            if confident
            else "weak_primary"
            if family_type == "primary"
            else "weak_special"
        ),
    }


def _profile(tracklet_id, values):
    return {
        "tracklet_id": tracklet_id,
        "aggregate_embedding": list(values),
        "reliability": 1.0,
        "effective_sample_count": 3.0,
    }


def _assignments(result):
    return {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }


def _config(**overrides):
    values = {
        "local_lost_gap_s": 1.25,
        "recovery_gap_s": 5.0,
        "soft_visible_actor_limit": 24,
    }
    values.update(overrides)
    return SceneAwareActorConfig(**values)


def test_recent_lost_actor_gets_near_successor():
    tracklets = [
        _tracklet("a", [0, 1, 2], [0.0, 0.2, 0.4]),
        _tracklet("b", [3, 4, 5], [0.6, 0.8, 1.0]),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        family_rows=[
            _family("a", "primary-0"),
            _family("b", "primary-0"),
        ],
        appearance_rows=[
            _profile("a", [1, 0, 0]),
            _profile("b", [1, 0, 0]),
        ],
        config=_config(),
    )
    ids = _assignments(result)
    assert ids["a"] == ids["b"]


def test_actor_observed_in_current_frame_cannot_be_reused():
    tracklets = [
        _tracklet("visible", [0, 1, 2, 3, 4], [0, .1, .2, .3, .4]),
        _tracklet("new", [3, 4], [.35, .45]),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        appearance_rows=[
            _profile("visible", [1, 0, 0]),
            _profile("new", [1, 0, 0]),
        ],
        config=_config(),
    )
    ids = _assignments(result)
    assert ids["visible"] != ids["new"]
    assert result.visible_exclusion_count > 0


def test_internal_hole_is_not_treated_as_tracklet_loss():
    tracklets = [
        # Still the same running tracklet: observation at frame 0 and later 4.
        _tracklet("a", [0, 4], [0.0, 0.4]),
        _tracklet("new", [2, 3], [0.2, 0.3]),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        appearance_rows=[
            _profile("a", [1, 0, 0]),
            _profile("new", [1, 0, 0]),
        ],
        config=_config(),
    )
    ids = _assignments(result)
    assert ids["a"] != ids["new"]
    assert result.not_ended_exclusion_count > 0


def test_confident_family_contradiction_blocks_handoff():
    tracklets = [
        _tracklet("a", [0, 1], [0.0, 0.2]),
        _tracklet("b", [2, 3], [0.3, 0.5]),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        family_rows=[
            _family("a", "primary-0"),
            _family("b", "primary-1"),
        ],
        appearance_rows=[
            _profile("a", [1, 0, 0]),
            _profile("b", [1, 0, 0]),
        ],
        config=_config(),
    )
    ids = _assignments(result)
    assert ids["a"] != ids["b"]
    assert result.hard_family_rejection_count >= 1


def test_hard_conflict_overrides_identical_reid():
    tracklets = [
        _tracklet("a", [0, 1], [0.0, 0.2]),
        _tracklet("b", [2, 3], [0.3, 0.5]),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        appearance_rows=[
            _profile("a", [1, 0, 0]),
            _profile("b", [1, 0, 0]),
        ],
        hard_conflicts=[{"tracklet_a": "a", "tracklet_b": "b"}],
        config=_config(),
    )
    ids = _assignments(result)
    assert ids["a"] != ids["b"]
    assert result.hard_conflict_rejection_count >= 1


def test_global_assignment_solves_two_lost_two_new_jointly():
    # At frame 2 both previous actors have ended. Two new tracklets start
    # together. Geometry + ReID should produce a one-to-one cross-scene match.
    tracklets = [
        _tracklet("a", [0, 1], [0.0, 0.0]),
        _tracklet("b", [0, 1], [4.0, 4.0]),
        _tracklet("n1", [2, 3], [1.5, 1.6]),
        _tracklet("n2", [2, 3], [2.5, 2.4]),
    ]
    profiles = [
        _profile("a", [1, 0, 0]),
        _profile("b", [0, 1, 0]),
        _profile("n1", [1, 0, 0]),
        _profile("n2", [0, 1, 0]),
    ]
    families = [
        _family("a", "primary-0"),
        _family("b", "primary-0"),
        _family("n1", "primary-0"),
        _family("n2", "primary-0"),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        family_rows=families,
        appearance_rows=profiles,
        config=_config(
            local_shortlist_extra_m=10.0,
            geometry_weight=.20,
            reid_weight=.50,
        ),
    )
    ids = _assignments(result)
    assert ids["n1"] == ids["a"]
    assert ids["n2"] == ids["b"]
    assert ids["n1"] != ids["n2"]


def test_long_recovery_requires_reid():
    tracklets = [
        _tracklet("a", [0, 1], [0.0, 0.2]),
        _tracklet("b", [25, 26], [1.0, 1.2]),
    ]
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        appearance_rows=[],
        config=_config(),
    )
    ids = _assignments(result)
    assert ids["a"] != ids["b"]


def test_soft_population_never_forces_impossible_merge():
    tracklets = [
        _tracklet(f"p{i}", [0, 1], [i * 4.0, i * 4.0 + .1])
        for i in range(24)
    ]
    tracklets.append(
        _tracklet("impossible", [2, 3], [500.0, 500.1])
    )
    result = build_scene_aware_actor_assignment(
        tracklet_rows=tracklets,
        config=_config(),
    )
    assert len(result.actors) == 25
    assert result.summary()["population_forced_merge_count"] == 0
