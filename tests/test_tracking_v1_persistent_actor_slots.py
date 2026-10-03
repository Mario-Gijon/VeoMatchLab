from tracking.actors.persistent_actor_slots import (
    PersistentActorSlotConfig,
    build_persistent_actor_slots,
)


def _tracklet(identifier, start_t, end_t, x0, x1, z=0.0):
    start_frame = int(round(start_t * 100))
    end_frame = max(start_frame + 1, int(round(end_t * 100)))
    return {
        "tracklet_id": identifier,
        "observations": [
            {
                "detection_id": identifier + "-a",
                "frame_index": start_frame,
                "timestamp_s": start_t,
                "x": x0,
                "z": z,
                "confidence": 0.95,
            },
            {
                "detection_id": identifier + "-b",
                "frame_index": end_frame,
                "timestamp_s": end_t,
                "x": x1,
                "z": z,
                "confidence": 0.95,
            },
        ],
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


def _profile(tracklet_id, vector, reliability=1.0):
    return {
        "tracklet_id": tracklet_id,
        "aggregate_embedding": list(vector),
        "reliability": reliability,
        "effective_sample_count": 3.0,
    }


def _assignments(result):
    return {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }


def _config(**overrides):
    values = {
        "local_gap_s": 1.25,
        "recovery_gap_s": 5.0,
        "soft_visible_actor_limit": 24,
    }
    values.update(overrides)
    return PersistentActorSlotConfig(**values)


def test_near_recent_lost_same_kit_reid_handoffs():
    tracklets = [
        _tracklet("a", 0.0, 0.5, 0.0, 1.0),
        _tracklet("b", 0.6, 1.0, 1.15, 1.8),
    ]
    families = [_family("a", "primary-0"), _family("b", "primary-0")]
    profiles = [_profile("a", [1, 0, 0]), _profile("b", [1, 0, 0])]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        family_rows=families,
        appearance_rows=profiles,
        config=_config(),
    )

    ids = _assignments(result)
    assert ids["a"] == ids["b"]
    assert result.accepted_handoffs[0].candidate.tier == "local"


def test_actor_still_visible_is_excluded_even_if_spatially_closest():
    # "visible" is physically closest to c, but it is still being observed.
    # "lost" ended just before c starts and must be the only legal handoff.
    tracklets = [
        _tracklet("visible", 0.0, 2.0, 0.0, 0.5),
        _tracklet("lost", 0.0, 0.9, 3.0, 3.4),
        _tracklet("c", 1.0, 1.4, 0.55, 1.0),
    ]
    profiles = [
        _profile("visible", [1, 0, 0]),
        _profile("lost", [1, 0, 0]),
        _profile("c", [1, 0, 0]),
    ]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        appearance_rows=profiles,
        config=_config(position_slack_m=4.0),
    )
    ids = _assignments(result)
    assert ids["c"] != ids["visible"]
    assert result.temporal_exclusion_count > 0


def test_confident_family_contradiction_is_hard_rejection():
    tracklets = [
        _tracklet("a", 0.0, 0.5, 0.0, 1.0),
        _tracklet("b", 0.6, 1.0, 1.1, 1.8),
    ]
    families = [_family("a", "primary-0"), _family("b", "primary-1")]
    profiles = [_profile("a", [1, 0, 0]), _profile("b", [1, 0, 0])]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        family_rows=families,
        appearance_rows=profiles,
        config=_config(),
    )

    ids = _assignments(result)
    assert ids["a"] != ids["b"]
    assert result.hard_family_rejection_count >= 1


def test_recovery_tier_requires_reid():
    tracklets = [
        _tracklet("a", 0.0, 0.5, 0.0, 1.0),
        _tracklet("b", 2.5, 3.0, 2.0, 2.5),
    ]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        appearance_rows=[],
        config=_config(),
    )

    ids = _assignments(result)
    assert ids["a"] != ids["b"]


def test_recovery_tier_can_restore_actor_with_matching_reid():
    tracklets = [
        _tracklet("a", 0.0, 0.5, 0.0, 1.0),
        _tracklet("b", 2.5, 3.0, 2.0, 2.5),
    ]
    profiles = [_profile("a", [1, 0, 0]), _profile("b", [1, 0, 0])]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        appearance_rows=profiles,
        config=_config(),
    )

    ids = _assignments(result)
    assert ids["a"] == ids["b"]
    assert result.accepted_handoffs[0].candidate.tier == "recovery"


def test_reid_disambiguates_two_physically_plausible_lost_actors():
    tracklets = [
        _tracklet("a", 0.0, 0.8, 0.0, 0.5),
        _tracklet("b", 0.0, 0.8, 2.0, 1.5),
        _tracklet("new", 0.9, 1.2, 1.0, 1.4),
    ]
    families = [
        _family("a", "primary-0"),
        _family("b", "primary-0"),
        _family("new", "primary-0"),
    ]
    profiles = [
        _profile("a", [1, 0, 0]),
        _profile("b", [0, 1, 0]),
        _profile("new", [1, 0, 0]),
    ]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        family_rows=families,
        appearance_rows=profiles,
        config=_config(
            min_best_margin=0.01,
            geometry_weight=0.15,
            reid_weight=0.55,
        ),
    )

    ids = _assignments(result)
    assert ids["new"] == ids["a"]
    assert ids["new"] != ids["b"]


def test_hard_conflict_never_overridden_by_reid():
    tracklets = [
        _tracklet("a", 0.0, 0.5, 0.0, 1.0),
        _tracklet("b", 0.6, 1.0, 1.1, 1.8),
    ]
    profiles = [_profile("a", [1, 0, 0]), _profile("b", [1, 0, 0])]
    conflicts = [{"tracklet_a": "a", "tracklet_b": "b"}]

    result = build_persistent_actor_slots(
        tracklet_rows=tracklets,
        appearance_rows=profiles,
        hard_conflicts=conflicts,
        config=_config(),
    )

    ids = _assignments(result)
    assert ids["a"] != ids["b"]
    assert result.hard_conflict_rejection_count >= 1


def test_population_prior_never_forces_impossible_merge():
    first = [
        _tracklet(
            f"t{i}",
            0.0,
            1.0,
            float(i) * 4.0,
            float(i) * 4.0 + 0.2,
        )
        for i in range(24)
    ]
    impossible = _tracklet("new", 1.1, 1.4, 500.0, 500.2)

    result = build_persistent_actor_slots(
        tracklet_rows=[*first, impossible],
        config=_config(),
    )

    assert len(result.actors) == 25
    assert result.summary()["population_forced_merge_count"] == 0
