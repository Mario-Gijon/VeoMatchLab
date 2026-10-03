import pytest

from tracking.actors.global_closed_roster_reassignment import (
    GlobalClosedRosterConfig,
    NON_ROSTER_ID,
    optimize_global_closed_roster,
)


def _tracklet(tid, points, confidence=.95):
    return {
        "tracklet_id": tid,
        "observations": [
            {
                "detection_id": f"{tid}-{i}",
                "frame_index": frame,
                "timestamp_s": frame / 10.0,
                "x": x,
                "z": z,
                "confidence": confidence,
            }
            for i, (frame, x, z) in enumerate(points)
        ],
    }


def _family(tid, key, confident=True):
    kind, _ = key.split("-")
    return {
        "tracklet_id": tid,
        "family_key": key if confident else None,
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


def _profile(tid, values):
    return {
        "tracklet_id": tid,
        "aggregate_embedding": list(values),
        "reliability": 1.0,
        "effective_sample_count": 3.0,
    }


def _roster(n=2):
    actors = []
    fixed_assignments = []
    for i in range(1, n + 1):
        aid = f"actor-{i:04d}"
        seed = f"seed-{i}"
        actors.append({
            "actor_id": aid,
            "tracklet_ids": [seed],
        })
        fixed_assignments.append({
            "tracklet_id": seed,
            "actor_id": aid,
            "actor_sequence_index": 0,
        })
    return actors, fixed_assignments


def _config(n=2, **kwargs):
    values = {
        "expected_actor_slots": n,
        "max_rounds": 5,
        "direct_move_margin": .01,
        "swap_move_margin": .01,
        "eviction_move_margin": .01,
        "non_roster_move_margin": .02,
    }
    values.update(kwargs)
    return GlobalClosedRosterConfig(**values)


def _assigned(result):
    return {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }


def _non_roster(result):
    return {
        row["tracklet_id"]
        for row in result.non_roster_tracklets
    }


def test_seeds_are_immutable_and_actor_count_never_changes():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, .2, 0), (3, .3, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
        {"tracklet_id": "x", "actor_id": "actor-0001"},
    ]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        config=_config(),
    )

    ids = _assigned(result)
    assert ids["seed-1"] == "actor-0001"
    assert ids["seed-2"] == "actor-0002"
    assert len(result.actors) == 2
    assert result.summary()["new_actor_creation"] is False


def test_wrong_nonseed_assignment_can_move_to_better_actor():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0), (5, .5, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0), (5, 10.5, 0)]),
        _tracklet("x", [(2, 10.2, 0), (3, 10.3, 0), (4, 10.4, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
        # Intentionally wrong.
        {"tracklet_id": "x", "actor_id": "actor-0001"},
    ]

    # The wrong initial partition itself is geometrically bad but has no exact
    # frame collision, so 7I.6 is allowed to repair it.
    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        config=_config(),
    )

    assert _assigned(result)["x"] == "actor-0002"
    assert result.summary()["actor_to_actor_reassignment_count"] >= 1


def test_non_roster_infield_fragment_can_be_rescued():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0), (5, .5, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0), (5, 10.5, 0)]),
        _tracklet("x", [(2, .2, 0), (3, .3, 0), (4, .4, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
    ]
    non_roster = [{"tracklet_id": "x"}]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        initial_non_roster_rows=non_roster,
        config=_config(),
    )

    assert _assigned(result)["x"] == "actor-0001"
    assert result.summary()["rescued_from_non_roster_count"] >= 1


def test_short_offfield_extra_remains_non_roster():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0), (2, .2, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0), (2, 10.2, 0)]),
        _tracklet("coach", [(1, 60, 40), (2, 60.1, 40)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
    ]
    non_roster = [{"tracklet_id": "coach"}]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        initial_non_roster_rows=non_roster,
        config=_config(),
    )

    assert "coach" in _non_roster(result)


def test_reid_cannot_override_impossible_geometry():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, 0, 0), (5, .5, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10, 0), (5, 10.5, 0)]),
        _tracklet("x", [(2, 10.1, 0), (3, 10.2, 0), (4, 10.4, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
    ]
    non_roster = [{"tracklet_id": "x"}]

    profiles = [
        _profile("seed-1", [1, 0]),
        _profile("seed-2", [0, 1]),
        _profile("x", [1, 0]),  # ReID incorrectly loves actor-1.
    ]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        initial_non_roster_rows=non_roster,
        appearance_rows=profiles,
        config=_config(),
    )

    assert _assigned(result)["x"] == "actor-0002"


def test_family_is_supporting_not_absolute_global_lock():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0), (5, .5, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0), (5, 10.5, 0)]),
        _tracklet("x", [(2, .2, 0), (3, .3, 0), (4, .4, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
    ]
    non_roster = [{"tracklet_id": "x"}]
    families = [
        _family("seed-1", "primary-0"),
        _family("seed-2", "primary-1"),
        # Deliberately wrong confident family label.
        _family("x", "primary-1"),
    ]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        initial_non_roster_rows=non_roster,
        family_rows=families,
        config=_config(
            geometry_weight=.72,
            temporal_weight=.16,
            family_weight=.08,
            reid_weight=.04,
        ),
    )

    assert _assigned(result)["x"] == "actor-0001"


def test_exact_frame_partition_is_clean_after_optimization():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, .2, 0), (3, .3, 0)]),
        _tracklet("y", [(2, 10.2, 0), (3, 10.3, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
        {"tracklet_id": "x", "actor_id": "actor-0001"},
        {"tracklet_id": "y", "actor_id": "actor-0002"},
    ]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        config=_config(),
    )

    ids = _assigned(result)
    assert ids["x"] != ids["y"]


def test_every_tracklet_has_terminal_status():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0)]),
        _tracklet("seed-2", [(0, 10, 0)]),
        _tracklet("x", [(2, .2, 0)]),
        _tracklet("coach", [(2, 60, 40)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
    ]
    non_roster = [
        {"tracklet_id": "x"},
        {"tracklet_id": "coach"},
    ]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        initial_non_roster_rows=non_roster,
        config=_config(),
    )

    terminal = set(_assigned(result)) | _non_roster(result)
    assert terminal == {"seed-1", "seed-2", "x", "coach"}
    assert result.summary()["unresolved_tracklet_count"] == 0


def test_trajectory_marks_short_gap_for_future_interpolation():
    actors, fixed = _roster(1)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("x", [(3, .3, 0), (4, .4, 0)]),
    ]
    initial = [
        {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
        {"tracklet_id": "x", "actor_id": "actor-0001"},
    ]

    result = optimize_global_closed_roster(
        tracklet_rows=tracklets,
        fixed_actor_rows=actors,
        fixed_assignment_rows=fixed,
        initial_assignment_rows=initial,
        config=_config(1),
    )

    trajectory = result.actor_trajectories[0]
    point = next(
        row for row in trajectory["points"]
        if row["frame_index"] == 1
    )
    assert point["interpolate_to_next"] is True


def test_seed_count_must_match_actor_count():
    actors, fixed = _roster(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0)]),
        _tracklet("seed-2", [(0, 10, 0)]),
    ]
    with pytest.raises(ValueError):
        optimize_global_closed_roster(
            tracklet_rows=tracklets,
            fixed_actor_rows=actors,
            fixed_assignment_rows=fixed[:1],
            initial_assignment_rows=[
                {"tracklet_id": "seed-1", "actor_id": "actor-0001"},
                {"tracklet_id": "seed-2", "actor_id": "actor-0002"},
            ],
            config=_config(2),
        )
