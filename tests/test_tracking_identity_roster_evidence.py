from __future__ import annotations

from tracking.identity_evidence import ObservedTrackPoint
from tracking.identity_roster_evidence import (
    assign_observations_to_units,
    build_canonical_membership,
    build_frame_canonical_points,
    compute_unit_occupancy_evidence,
)


def point(track, frame, x, z=0.0):
    return ObservedTrackPoint(
        track_id=track,
        frame_index=frame,
        timestamp=frame / 10.0,
        x=x,
        z=z,
    )


def test_assign_observations_to_temporal_units():
    raw = {
        1: tuple(
            point(1, frame, float(frame))
            for frame in range(10)
        )
    }
    units = {
        "1:0": {
            "track_id": 1,
            "start_frame": 0,
            "end_frame": 4,
        },
        "1:1": {
            "track_id": 1,
            "start_frame": 5,
            "end_frame": 9,
        },
    }

    result = assign_observations_to_units(
        raw,
        units,
    )

    assert len(result["1:0"]) == 5
    assert len(result["1:1"]) == 5


def test_shadowed_unit_is_detected_as_represented():
    units = {
        "1:0": tuple(
            point(1, frame, float(frame))
            for frame in range(10)
        ),
        "2:0": tuple(
            point(
                2,
                frame,
                float(frame) + 0.2,
            )
            for frame in range(10)
        ),
    }

    unit_to_canonical = {
        "1:0": 1,
    }

    frame_points = build_frame_canonical_points(
        unit_points=units,
        unit_to_canonical=unit_to_canonical,
    )

    pair_evidence = {
        ("1:0", "2:0"): {
            "conflict": "none",
            "fused_support_score": 0.9,
        }
    }

    evidence = compute_unit_occupancy_evidence(
        "2:0",
        unit_points=units,
        frame_canonical_points=frame_points,
        unit_to_canonical=unit_to_canonical,
        pair_evidence=pair_evidence,
        hybrid_edges={},
        roster_size=23,
    )

    assert evidence.represented_fraction == 1.0
    assert evidence.unique_fraction == 0.0
    assert evidence.best_shadow_identity == 1


def test_conflicting_nearby_unit_is_not_treated_as_shadow():
    units = {
        "1:0": tuple(
            point(1, frame, float(frame))
            for frame in range(10)
        ),
        "2:0": tuple(
            point(
                2,
                frame,
                float(frame) + 0.2,
            )
            for frame in range(10)
        ),
    }

    unit_to_canonical = {
        "1:0": 1,
    }

    frame_points = build_frame_canonical_points(
        unit_points=units,
        unit_to_canonical=unit_to_canonical,
    )

    pair_evidence = {
        ("1:0", "2:0"): {
            "conflict": "strong",
            "fused_support_score": 0.9,
        }
    }

    evidence = compute_unit_occupancy_evidence(
        "2:0",
        unit_points=units,
        frame_canonical_points=frame_points,
        unit_to_canonical=unit_to_canonical,
        pair_evidence=pair_evidence,
        hybrid_edges={},
        roster_size=23,
    )

    assert evidence.represented_fraction == 0.0
    assert evidence.unique_fraction == 1.0
