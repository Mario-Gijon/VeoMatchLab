import math

from tracking.actors.actor_trajectory_reconstruction import (
    ActorTrajectoryConfig,
    build_actor_trajectories,
    build_raw_actor_trajectories,
    reconstruct_short_actor_gaps,
)


def _tracklet(tid, rows):
    return {
        "tracklet_id": tid,
        "observations": [
            {
                "detection_id": f"{tid}-{i}",
                "frame_index": frame,
                "timestamp_s": timestamp,
                "x": x,
                "z": z,
                "confidence": .95,
            }
            for i, (frame, timestamp, x, z) in enumerate(rows)
        ],
    }


def _detection_frame(frame, timestamp, detections):
    return {
        "frame_index": frame,
        "timestamp_s": timestamp,
        "detections": [
            {
                "detection_id": detection_id,
                "frame_index": frame,
                "timestamp_s": timestamp,
                "confidence": .95,
                "x": x,
                "z": z,
                "source_lens": source_lens,
                "source_bbox_xyxy": source_bbox,
                "source_footpoint_xy": None,
                "representative_view": view,
                "virtual_bbox_xyxy": virtual_bbox,
                "contributing_views": [view] if view else [],
                "metadata": {},
            }
            for (
                detection_id,
                x,
                z,
                view,
                virtual_bbox,
                source_lens,
                source_bbox,
            ) in detections
        ],
    }


def _actors(n):
    return [{"actor_id": f"actor-{i:04d}"} for i in range(1, n + 1)]


def _assignment(tid, actor):
    return {"tracklet_id": tid, "actor_id": actor}


def _config(n=1, **kwargs):
    values = {
        "expected_actor_slots": n,
        "max_interpolation_gap_s": .5,
        "max_endpoint_speed_mps": 12.5,
    }
    values.update(kwargs)
    return ActorTrajectoryConfig(**values)


def test_builds_exact_fixed_actor_count_and_excludes_non_roster():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(0, 0.0, 10, 0)]),
        _tracklet("noise", [(0, 0.0, 60, 40)]),
    ]
    result = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(2),
        assignment_rows=[
            _assignment("a", "actor-0001"),
            _assignment("b", "actor-0002"),
        ],
        detection_frame_rows=[],
        expected_actor_slots=2,
    )
    assert len(result) == 2
    assert {
        tid
        for trajectory in result
        for tid in trajectory.tracklet_ids
    } == {"a", "b"}


def test_fills_every_missing_source_frame_between_10hz_anchors():
    tracklets = [
        _tracklet("a", [
            (0, 0.0, 0, 0),
            (3, 0.1, .3, 0),
        ]),
    ]
    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
        ],
        detection_frame_rows=[],
        expected_actor_slots=1,
    )
    rebuilt, events = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=_config(),
    )
    assert [p.frame_index for p in rebuilt[0].points] == [0, 1, 2, 3]
    assert sum(p.origin.value == "interpolated" for p in rebuilt[0].points) == 2
    assert events[0].accepted is True
    assert events[0].inserted_point_count == 2


def test_cross_tracklet_gap_is_allowed_for_same_actor():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(3, 0.1, .3, 0)]),
    ]
    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
            _assignment("b", "actor-0001"),
        ],
        detection_frame_rows=[],
        expected_actor_slots=1,
    )
    rebuilt, events = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=_config(),
    )
    assert len(rebuilt[0].points) == 4
    assert events[0].cross_tracklet is True
    assert events[0].accepted is True


def test_long_gap_remains_lost_and_has_no_carry_forward():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(30, 1.0, 1, 0)]),
    ]
    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
            _assignment("b", "actor-0001"),
        ],
        detection_frame_rows=[],
        expected_actor_slots=1,
    )
    rebuilt, events = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=_config(),
    )
    assert [p.frame_index for p in rebuilt[0].points] == [0, 30]
    assert events[0].accepted is False
    assert events[0].reason == "gap_too_long"


def test_implausible_speed_gap_is_preserved():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(3, 0.1, 10, 0)]),
    ]
    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
            _assignment("b", "actor-0001"),
        ],
        detection_frame_rows=[],
        expected_actor_slots=1,
    )
    rebuilt, events = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=_config(),
    )
    assert len(rebuilt[0].points) == 2
    assert events[0].reason == "endpoint_speed_too_high"


def test_interpolated_visual_point_keeps_two_real_visual_anchors():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(3, 0.1, .3, 0)]),
    ]
    detections = [
        _detection_frame(0, 0.0, [
            ("a-0", 0, 0, "tight_far_c1",
             [10, 20, 30, 60], "left", [100, 100, 120, 160]),
        ]),
        _detection_frame(3, 0.1, [
            ("b-0", .3, 0, "tight_far_c2",
             [12, 22, 32, 62], "right", [200, 100, 220, 160]),
        ]),
    ]
    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
            _assignment("b", "actor-0001"),
        ],
        detection_frame_rows=detections,
        expected_actor_slots=1,
    )
    rebuilt, events = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=_config(),
    )
    inserted = next(
        point for point in rebuilt[0].points
        if point.frame_index == 1
    )
    visual = inserted.metadata["visual_interpolation"]
    assert visual["visual_interpolation_allowed"] is True
    assert visual["left"]["representative_view"] == "tight_far_c1"
    assert visual["right"]["representative_view"] == "tight_far_c2"
    # Different physical lens: never interpolate a source bbox between them.
    assert visual["source_bbox_interpolation_allowed"] is False
    assert events[0].visual_interpolation_allowed is True


def test_same_lens_source_bbox_can_be_interpolated_later_by_ui():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(3, 0.1, .3, 0)]),
    ]
    detections = [
        _detection_frame(0, 0.0, [
            ("a-0", 0, 0, "tight_far_c1",
             [10, 20, 30, 60], "left", [100, 100, 120, 160]),
        ]),
        _detection_frame(3, 0.1, [
            ("b-0", .3, 0, "tight_far_c2",
             [12, 22, 32, 62], "left", [104, 100, 124, 160]),
        ]),
    ]
    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
            _assignment("b", "actor-0001"),
        ],
        detection_frame_rows=detections,
        expected_actor_slots=1,
    )
    rebuilt, _ = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=_config(),
    )
    inserted = next(
        point for point in rebuilt[0].points
        if point.frame_index == 1
    )
    assert inserted.metadata["visual_interpolation"][
        "source_bbox_interpolation_allowed"
    ] is True


def test_full_pipeline_outputs_smoothed_source_frame_cadence():
    tracklets = [
        _tracklet("a", [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, .35, .02),
            (6, 0.2, .58, -.03),
            (9, 0.3, .92, .01),
        ]),
    ]
    result = build_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
        ],
        detection_frame_rows=[],
        config=_config(),
    )

    points = result.display_trajectories[0]["points"]
    assert [p["frame_index"] for p in points] == list(range(10))
    assert result.display_trajectories[0]["segment_count"] == 1
    assert any(p["origin"] == "interpolated" for p in points)
    assert all(math.isfinite(p["x"]) and math.isfinite(p["z"]) for p in points)


def test_smoothed_positions_are_separate_from_measurements():
    tracklets = [
        _tracklet("a", [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, .3, .0),
            (6, 0.2, .6, 1.0),  # deliberate spatial jitter
            (9, 0.3, .9, .0),
            (12, 0.4, 1.2, .0),
        ]),
    ]
    result = build_actor_trajectories(
        tracklet_rows=tracklets,
        actor_rows=_actors(1),
        assignment_rows=[
            _assignment("a", "actor-0001"),
        ],
        detection_frame_rows=[],
        config=_config(),
    )
    observed = [
        p for p in result.display_trajectories[0]["points"]
        if p["origin"] == "observed"
    ]
    assert any(
        abs(p["z"] - p["measurement_z"]) > 1e-6
        for p in observed
    )


def test_exact_frame_coexistence_in_actor_partition_is_rejected():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0, 0)]),
        _tracklet("b", [(0, 0.0, 1, 0)]),
    ]
    try:
        build_raw_actor_trajectories(
            tracklet_rows=tracklets,
            actor_rows=_actors(1),
            assignment_rows=[
                _assignment("a", "actor-0001"),
                _assignment("b", "actor-0001"),
            ],
            detection_frame_rows=[],
            expected_actor_slots=1,
        )
    except ValueError as exc:
        assert "exact-frame coexistence" in str(exc)
    else:
        raise AssertionError("expected exact-frame coexistence failure")
