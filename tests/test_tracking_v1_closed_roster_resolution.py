import pytest

from tracking.actors.closed_roster_resolution import (
    ClosedRosterConfig,
    NON_ROSTER_ID,
    resolve_closed_roster,
)


def _tracklet(identifier, points):
    return {
        "tracklet_id": identifier,
        "observations": [
            {
                "detection_id": f"{identifier}-{i}",
                "frame_index": frame,
                "timestamp_s": frame / 10.0,
                "x": x,
                "z": z,
                "confidence": .95,
            }
            for i, (frame, x, z) in enumerate(points)
        ],
        "metadata": {},
    }


def _family(tracklet_id, key, confident=True):
    kind, raw = key.split("-")
    return {
        "tracklet_id": tracklet_id,
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


def _profile(tracklet_id, values):
    return {
        "tracklet_id": tracklet_id,
        "aggregate_embedding": list(values),
        "reliability": 1.0,
        "effective_sample_count": 3.0,
    }


def _base(actor_count=2):
    actors = [
        {
            "actor_id": f"actor-{i:04d}",
            "tracklet_ids": [f"seed-{i}"],
        }
        for i in range(1, actor_count + 1)
    ]
    assignments = [
        {
            "tracklet_id": f"seed-{i}",
            "actor_id": f"actor-{i:04d}",
        }
        for i in range(1, actor_count + 1)
    ]
    return actors, assignments


def _config(actor_count=2, **kwargs):
    values = {
        "expected_actor_slots": actor_count,
        "field_length_m": 105.0,
        "field_width_m": 68.0,
        "hard_geometry_window_s": 2.5,
        "context_window_s": 6.0,
        "max_iterations": 6,
    }
    values.update(kwargs)
    return ClosedRosterConfig(**values)


def _assignment_map(result):
    return {
        row["tracklet_id"]: row["actor_id"]
        for row in result.tracklet_assignments
    }


def test_keeps_exact_closed_actor_count_and_no_unresolved():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .2, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.2, 0)]),
        _tracklet("x", [(2, .4, 0), (3, .6, 0)]),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        config=_config(2),
    )
    assert len(result.actors) == 2
    assert result.summary()["unresolved_tracklet_count"] == 0
    assert result.summary()["new_actor_creation"] is False


def test_bidirectional_geometry_recovers_middle_fragment():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .2, 0), (5, 1.0, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.2, 0), (5, 11.0, 0)]),
        _tracklet("middle", [(2, .4, 0), (3, .6, 0), (4, .8, 0)]),
    ]
    families = [
        _family("seed-1", "primary-0"),
        _family("seed-2", "primary-1"),
        _family("middle", "primary-0"),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        family_rows=families,
        config=_config(2),
    )
    ids = _assignment_map(result)
    assert ids["middle"] == ids["seed-1"]
    decision = next(
        row for row in result.decisions
        if row.tracklet_id == "middle"
    )
    assert decision.rule in {
        "bidirectional_bridge",
        "unique_physically_viable_actor",
    }


def test_exact_coexistence_prevents_wrong_actor():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (2, .2, 0), (4, .4, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0)]),
        _tracklet("x", [(2, 10.2, 0), (3, 10.3, 0)]),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        config=_config(2),
    )
    ids = _assignment_map(result)
    assert ids["x"] == "actor-0002"


def test_confident_kit_mismatch_is_not_rescued_by_reid():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .2, 0)]),
        _tracklet("seed-2", [(0, 1.2, 0), (1, 1.4, 0)]),
        _tracklet("x", [(2, .4, 0), (3, .6, 0)]),
    ]
    families = [
        _family("seed-1", "primary-0"),
        _family("seed-2", "primary-1"),
        _family("x", "primary-1"),
    ]
    profiles = [
        _profile("seed-1", [1, 0]),
        _profile("seed-2", [0, 1]),
        _profile("x", [1, 0]),  # ReID would incorrectly prefer actor-1.
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        family_rows=families,
        appearance_rows=profiles,
        config=_config(2),
    )
    ids = _assignment_map(result)
    assert ids["x"] == "actor-0002"


def test_short_off_field_extra_becomes_non_roster():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0), (2, .2, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.1, 0), (2, 10.2, 0)]),
        _tracklet("coach", [(1, 60, 40), (2, 60.2, 40)]),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        config=_config(2),
    )
    ids = _assignment_map(result)
    assert "coach" not in ids
    row = next(
        row for row in result.non_roster_tracklets
        if row["tracklet_id"] == "coach"
    )
    assert row["identity_id"] == NON_ROSTER_ID


def test_infield_fragment_is_not_thrown_away_merely_for_missing_reid():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .2, 0), (5, 1, 0)]),
        _tracklet("seed-2", [(0, 10, 0), (1, 10.2, 0), (5, 11, 0)]),
        _tracklet("x", [(2, .4, 0), (3, .6, 0), (4, .8, 0)]),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        appearance_rows=[],
        config=_config(2),
    )
    assert _assignment_map(result)["x"] == "actor-0001"


def test_closed_world_competition_does_not_put_overlapping_fragments_on_same_actor():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .2, 0)]),
        _tracklet("seed-2", [(0, 3, 0), (1, 3.2, 0)]),
        _tracklet("x", [(2, .5, 0), (3, .7, 0)]),
        _tracklet("y", [(2, 2.7, 0), (3, 2.9, 0)]),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        config=_config(2),
    )
    ids = _assignment_map(result)
    assert ids["x"] != ids["y"]


def test_trajectory_marks_small_gaps_interpolatable_and_large_gaps_lost():
    actors, assignments = _base(1)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0), (1, .1, 0)]),
        _tracklet("short", [(3, .3, 0), (4, .4, 0)]),
        _tracklet("late", [(20, 2, 0), (21, 2.1, 0)]),
    ]
    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        config=_config(
            1,
            visual_interpolation_max_gap_s=.5,
        ),
    )
    trajectory = result.actor_trajectories[0]
    by_frame = {
        point["frame_index"]: point
        for point in trajectory["points"]
    }
    assert by_frame[1]["interpolate_to_next"] is True
    assert by_frame[4]["interpolate_to_next"] is False


def test_actor_count_mismatch_is_rejected():
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0, 0)]),
        _tracklet("seed-2", [(0, 10, 0)]),
    ]
    with pytest.raises(ValueError):
        resolve_closed_roster(
            tracklet_rows=tracklets,
            actor_rows=actors,
            initial_assignment_rows=assignments,
            config=_config(3),
        )


def test_closed_world_fallback_assigns_ambiguous_tracklet_exactly_once():
    """Regression: final closed-world fallback must not add the winner twice."""
    actors, assignments = _base(2)
    tracklets = [
        _tracklet("seed-1", [(0, 0.0, 0), (1, 0.1, 0)]),
        _tracklet("seed-2", [(0, 1.0, 0), (1, 0.9, 0)]),
        # Exactly between both actors: geometry is deliberately ambiguous, so
        # normal iterative resolution leaves it pending and the final closed
        # world fallback must choose one existing actor.
        _tracklet("ambiguous", [(2, 0.5, 0), (3, 0.5, 0)]),
    ]

    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        config=_config(2),
    )

    assignments_by_tracklet = _assignment_map(result)
    assert assignments_by_tracklet["ambiguous"] in {
        "actor-0001",
        "actor-0002",
    }

    owner = assignments_by_tracklet["ambiguous"]
    actor_row = next(
        row for row in result.actors
        if row["actor_id"] == owner
    )
    assert actor_row["tracklet_ids"].count("ambiguous") == 1

    decision = next(
        row for row in result.decisions
        if row.tracklet_id == "ambiguous"
    )
    assert decision.rule == "closed_world_best_viable_actor"
    assert decision.needs_review is True
