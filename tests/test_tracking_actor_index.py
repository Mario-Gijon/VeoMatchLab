from __future__ import annotations

import pytest

from tracking.actors.index import (
    SparseTrackletIndexConfig,
    build_sparse_tracklet_index,
)
from tracking.tracklets.models import Tracklet, TrackletObservation


def _tracklet(
    tracklet_id: str,
    samples: list[tuple[int, float, float, float]],
    *,
    confidence: float = 0.95,
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=tuple(
            TrackletObservation(
                detection_id=f"{tracklet_id}-det-{index}",
                frame_index=frame,
                timestamp_s=timestamp,
                x=x,
                z=z,
                confidence=confidence,
            )
            for index, (frame, timestamp, x, z) in enumerate(samples)
        ),
    )


def test_exact_frame_index_is_deterministic():
    first = _tracklet(
        "trk-a",
        [(0, 0.0, 0.0, 0.0), (3, 0.1, 0.2, 0.0)],
    )
    second = _tracklet(
        "trk-b",
        [(0, 0.0, 10.0, 0.0), (6, 0.2, 9.8, 0.0)],
    )

    index = build_sparse_tracklet_index([second, first])

    assert index.frame_tracklet_ids(0) == ("trk-a", "trk-b")
    assert index.frame_tracklet_ids(3) == ("trk-a",)
    assert index.frame_tracklet_ids(6) == ("trk-b",)
    assert index.observed_frame_count == 3


def test_local_successor_query_uses_time_horizon_not_all_pairs():
    predecessor = _tracklet(
        "trk-a",
        [(0, 0.0, 0.0, 0.0), (3, 0.1, 0.0, 0.0)],
    )
    near = _tracklet("trk-b", [(6, 0.2, 0.5, 0.0)])
    far = _tracklet("trk-c", [(300, 10.0, 0.5, 0.0)])

    index = build_sparse_tracklet_index(
        [predecessor, near, far],
        SparseTrackletIndexConfig(local_gap_horizon_s=1.0),
    )

    assert [
        item.tracklet_id
        for item in index.query_local_successors("trk-a")
    ] == ["trk-b"]


def test_physically_plausible_transition_is_reachable():
    first = _tracklet(
        "trk-a",
        [(0, 0.0, 0.0, 0.0), (3, 0.1, 0.5, 0.0)],
    )
    second = _tracklet("trk-b", [(15, 0.5, 2.0, 0.0)])

    index = build_sparse_tracklet_index([first, second])
    features = index.transition_features("trk-a", "trk-b")

    assert features.gap_s == pytest.approx(0.4)
    assert features.endpoint_distance_m == pytest.approx(1.5)
    assert features.physically_reachable is True


def test_short_impossible_transition_is_not_reachable():
    first = _tracklet(
        "trk-a",
        [(0, 0.0, 0.0, 0.0), (3, 0.1, 0.0, 0.0)],
    )
    second = _tracklet("trk-b", [(6, 0.2, 10.0, 0.0)])

    index = build_sparse_tracklet_index([first, second])
    features = index.transition_features("trk-a", "trk-b")

    assert features.reachability_limit_m == pytest.approx(2.7)
    assert features.physically_reachable is False


def test_one_exact_shared_frame_far_apart_is_hard_conflict():
    first = _tracklet("trk-a", [(0, 0.0, 0.0, 0.0)])
    second = _tracklet("trk-b", [(0, 0.0, 5.0, 0.0)])

    index = build_sparse_tracklet_index([first, second])
    conflicts = index.hard_coexistence_conflicts()

    assert len(conflicts) == 1
    assert conflicts[0].reason == "simultaneous_spatial_separation"


def test_multiple_shared_frames_use_robust_median():
    first = _tracklet(
        "trk-a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 0.0, 0.0),
            (6, 0.2, 0.0, 0.0),
        ],
    )
    second = _tracklet(
        "trk-b",
        [
            (0, 0.0, 4.0, 0.0),
            (3, 0.1, 4.2, 0.0),
            (6, 0.2, 0.4, 0.0),
        ],
    )

    index = build_sparse_tracklet_index([first, second])
    evidence = index.coobserved_pairs[0]

    assert evidence.shared_frame_count == 3
    assert evidence.median_separation_m == pytest.approx(4.0)
    assert len(index.hard_coexistence_conflicts()) == 1


def test_close_exact_overlap_is_not_hard_and_is_exposed():
    first = _tracklet(
        "trk-a",
        [(0, 0.0, 0.0, 0.0), (3, 0.1, 0.2, 0.0)],
    )
    second = _tracklet(
        "trk-b",
        [(0, 0.0, 0.4, 0.1), (3, 0.1, 0.6, 0.1)],
    )

    index = build_sparse_tracklet_index([first, second])

    assert index.hard_coexistence_conflicts() == ()
    assert len(index.close_coobserved_pairs()) == 1


def test_endpoint_velocity_uses_multiple_samples():
    first = _tracklet(
        "trk-a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 0.5, 0.0),
            (6, 0.2, 1.0, 0.0),
            (9, 0.3, 1.5, 0.0),
        ],
    )

    index = build_sparse_tracklet_index([first])
    velocity = index.node("trk-a").end_velocity

    assert velocity is not None
    assert velocity.vx_mps == pytest.approx(5.0)
    assert abs(velocity.vz_mps) < 1e-9


def test_prediction_error_is_soft_feature_only():
    first = _tracklet(
        "trk-a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 0.5, 0.0),
            (6, 0.2, 1.0, 0.0),
        ],
    )
    second = _tracklet(
        "trk-b",
        [
            (15, 0.5, 1.2, 1.0),
            (18, 0.6, 1.3, 1.0),
            (21, 0.7, 1.4, 1.0),
        ],
    )

    index = build_sparse_tracklet_index([first, second])
    features = index.transition_features("trk-a", "trk-b")

    assert features.forward_prediction_error_m is not None
    assert features.backward_prediction_error_m is not None
    assert features.physically_reachable is True


def test_pair_ordering_is_canonical():
    first = _tracklet("trk-z", [(0, 0.0, 0.0, 0.0)])
    second = _tracklet("trk-a", [(0, 0.0, 5.0, 0.0)])

    index = build_sparse_tracklet_index([first, second])
    evidence = index.coobserved_pairs[0]
    conflict = index.hard_coexistence_conflicts()[0]

    assert (evidence.tracklet_a, evidence.tracklet_b) == (
        "trk-a",
        "trk-z",
    )
    assert (conflict.tracklet_a, conflict.tracklet_b) == (
        "trk-a",
        "trk-z",
    )
