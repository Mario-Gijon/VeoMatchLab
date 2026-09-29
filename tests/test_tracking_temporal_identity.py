from __future__ import annotations

from tracking_v0.identity_evidence import ObservedTrackPoint
from tracking_v0.temporal_identity import (
    FieldBounds,
    compute_temporal_transition,
    evaluate_temporary_outside_candidate,
)


def point(
    track,
    frame,
    timestamp,
    x,
    z=0.0,
):
    return ObservedTrackPoint(
        track_id=track,
        frame_index=frame,
        timestamp=timestamp,
        x=x,
        z=z,
    )


def test_clean_constant_velocity_gap_scores_high():
    a = tuple(
        point(
            1,
            frame,
            frame / 10.0,
            frame / 10.0 * 2.0,
        )
        for frame in range(0, 11)
    )
    b = tuple(
        point(
            2,
            frame,
            frame / 10.0,
            frame / 10.0 * 2.0,
        )
        for frame in range(14, 25)
    )

    evidence = compute_temporal_transition(
        "1:0",
        "2:0",
        rows_a=a,
        rows_b=b,
    )

    assert evidence.valid
    assert evidence.regime == "gap"
    assert evidence.verdict == "support"
    assert evidence.temporal_score > 0.85
    assert evidence.symmetric_error_m < 0.15


def test_impossible_jump_is_rejected_or_scores_near_zero():
    a = tuple(
        point(
            1,
            frame,
            frame / 10.0,
            frame / 10.0,
        )
        for frame in range(0, 11)
    )
    b = tuple(
        point(
            2,
            frame,
            frame / 10.0,
            40.0 + frame / 10.0,
        )
        for frame in range(14, 25)
    )

    evidence = compute_temporal_transition(
        "1:0",
        "2:0",
        rows_a=a,
        rows_b=b,
    )

    assert not evidence.valid
    assert evidence.verdict == "contradiction"
    assert evidence.temporal_score == 0.0


def test_short_close_duplicate_overlap_scores_high():
    a = tuple(
        point(
            1,
            frame,
            frame / 10.0,
            frame / 10.0,
        )
        for frame in range(0, 21)
    )
    b = tuple(
        point(
            2,
            frame,
            frame / 10.0,
            frame / 10.0 + 0.15,
        )
        for frame in range(17, 31)
    )

    evidence = compute_temporal_transition(
        "1:0",
        "2:0",
        rows_a=a,
        rows_b=b,
    )

    assert evidence.valid
    assert evidence.regime == "observed_overlap"
    assert evidence.verdict == "support"
    assert evidence.overlap_median_distance_m < 0.3
    assert evidence.temporal_score > 0.80


def test_outside_candidate_requires_known_boundary_exit():
    field = FieldBounds(
        min_x=-52.5,
        max_x=52.5,
        min_z=-34.0,
        max_z=34.0,
    )

    history = tuple(
        point(
            7,
            frame,
            frame / 10.0,
            50.0 + 0.2 * frame,
            0.0,
        )
        for frame in range(0, 11)
    )

    candidate = point(
        7,
        13,
        1.3,
        53.1,
        0.0,
    )

    evidence = evaluate_temporary_outside_candidate(
        history,
        candidate,
        field=field,
    )

    assert evidence.allowed
    assert evidence.reason == "admissible_known_exit_continuation"


def test_arbitrary_outside_person_is_not_admitted():
    field = FieldBounds(
        min_x=-52.5,
        max_x=52.5,
        min_z=-34.0,
        max_z=34.0,
    )

    history = tuple(
        point(
            7,
            frame,
            frame / 10.0,
            0.1 * frame,
            0.0,
        )
        for frame in range(0, 11)
    )

    candidate = point(
        7,
        13,
        1.3,
        53.1,
        0.0,
    )

    evidence = evaluate_temporary_outside_candidate(
        history,
        candidate,
        field=field,
    )

    assert not evidence.allowed
    assert evidence.reason == "last_inside_not_near_boundary"


def test_long_support_overlap_without_same_frame_observations_is_neutral():
    # Two raw IDs interleave over a long support interval but are never observed
    # in the same frame. Support-span overlap must not be treated as a temporal
    # contradiction.
    a = tuple(
        point(
            1,
            frame,
            frame / 10.0,
            frame / 10.0,
        )
        for frame in range(0, 80, 2)
    )
    b = tuple(
        point(
            2,
            frame,
            frame / 10.0,
            frame / 10.0 + 0.1,
        )
        for frame in range(21, 101, 2)
    )

    evidence = compute_temporal_transition(
        "1:0",
        "2:0",
        rows_a=a,
        rows_b=b,
    )

    assert evidence.valid
    assert evidence.verdict == "neutral"
    assert evidence.temporal_score == 0.50
    assert evidence.regime == "interleaved_overlap"
