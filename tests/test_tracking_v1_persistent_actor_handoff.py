from tracking.actors.persistent_actor_handoff import (
    PersistentActorHandoffConfig,
    build_persistent_actor_handoffs,
)


def _tracklet(identifier, start_t, x0, z0, *, dx=0.20):
    frame = int(round(start_t * 10))
    return {
        "tracklet_id": identifier,
        "observations": [
            {
                "detection_id": identifier + "-a",
                "frame_index": frame,
                "timestamp_s": start_t,
                "x": x0,
                "z": z0,
                "confidence": 0.9,
            },
            {
                "detection_id": identifier + "-b",
                "frame_index": frame + 1,
                "timestamp_s": start_t + 0.10,
                "x": x0 + dx,
                "z": z0,
                "confidence": 0.9,
            },
        ],
        "metadata": {},
    }


def _family(tracklet_id, key, *, confident=True):
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


def _config(**overrides):
    values = {
        "max_handoff_gap_s": 1.5,
        "soft_live_actor_limit": 24,
    }
    values.update(overrides)
    return PersistentActorHandoffConfig(**values)


def test_short_near_same_family_fragments_become_one_actor():
    tracklets = [
        _tracklet("trk-1", 0.0, 0.0, 0.0),
        _tracklet("trk-2", 0.2, 0.45, 0.0),
        _tracklet("trk-3", 0.4, 0.90, 0.0),
    ]
    families = [
        _family("trk-1", "special-0"),
        _family("trk-2", "special-0"),
        _family("trk-3", "special-0"),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=families,
        config=_config(),
    )
    assert len(result.actors) == 1
    assert len(result.accepted_handoffs) == 2
    assert result.actors[0]["tracklet_ids"] == [
        "trk-1", "trk-2", "trk-3"
    ]


def test_confident_family_contradiction_is_never_handed_off():
    tracklets = [
        _tracklet("trk-red", 0.0, 0.0, 0.0),
        _tracklet("trk-white", 0.2, 0.30, 0.0),
    ]
    families = [
        _family("trk-red", "primary-0"),
        _family("trk-white", "primary-1"),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=families,
        config=_config(),
    )
    assert len(result.actors) == 2
    assert not result.accepted_handoffs
    assert result.hard_family_rejection_count >= 1


def test_far_successor_fails_physical_gate():
    tracklets = [
        _tracklet("trk-1", 0.0, 0.0, 0.0),
        _tracklet("trk-2", 0.2, 20.0, 0.0),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=[],
        config=_config(),
    )
    assert len(result.actors) == 2
    assert not result.accepted_handoffs
    assert result.hard_geometry_rejection_count >= 1


def test_missing_family_is_neutral_not_negative():
    tracklets = [
        _tracklet("trk-1", 0.0, 0.0, 0.0),
        _tracklet("trk-2", 0.2, 0.35, 0.0),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=[],
        config=_config(),
    )
    assert len(result.actors) == 1
    assert result.accepted_handoffs[0].candidate.family_relation == "neutral"


def test_simultaneous_successors_cannot_share_one_actor():
    tracklets = [
        _tracklet("trk-a", 0.0, 0.0, 0.0),
        _tracklet("trk-b", 0.2, 0.35, 0.0),
        _tracklet("trk-c", 0.2, 0.40, 0.05),
    ]
    families = [
        _family("trk-a", "primary-0"),
        _family("trk-b", "primary-0"),
        _family("trk-c", "primary-0"),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=families,
        config=_config(),
    )
    assignments = {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }
    assert assignments["trk-b"] != assignments["trk-c"]


def test_ambiguous_two_actor_competition_creates_new_actor():
    tracklets = [
        _tracklet("trk-a", 0.0, 0.0, 0.0, dx=0.0),
        _tracklet("trk-b", 0.0, 1.0, 0.0, dx=0.0),
        _tracklet("trk-new", 0.2, 0.50, 0.0, dx=0.0),
    ]
    families = [
        _family("trk-a", "primary-0"),
        _family("trk-b", "primary-0"),
        _family("trk-new", "primary-0"),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=families,
        config=_config(min_best_margin=0.20),
    )
    assignments = {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }
    assert assignments["trk-new"] not in {
        assignments["trk-a"],
        assignments["trk-b"],
    }


def test_long_gap_is_not_local_handoff():
    tracklets = [
        _tracklet("trk-1", 0.0, 0.0, 0.0),
        _tracklet("trk-2", 2.0, 0.4, 0.0),
    ]
    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=[],
        config=_config(max_handoff_gap_s=1.0),
    )
    assert len(result.actors) == 2
    assert not result.accepted_handoffs


def test_population_limit_is_soft_and_never_forces_impossible_merge():
    first = [
        _tracklet(f"trk-{index:02d}", 0.0, index * 3.0, 0.0)
        for index in range(24)
    ]
    impossible = _tracklet("trk-new", 0.2, 500.0, 0.0)
    result = build_persistent_actor_handoffs(
        tracklet_rows=[*first, impossible],
        family_rows=[],
        config=_config(soft_live_actor_limit=24),
    )
    assert len(result.actors) == 25
    assert result.summary()["population_forced_merge_count"] == 0
