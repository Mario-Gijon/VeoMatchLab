from __future__ import annotations

import pytest

from tracking.identities.candidates import (
    PhysicalCompatibilityConfig,
    TrackletObservation,
    build_candidate_edge,
    build_candidate_graph,
    tracklet_node_from_row,
)


CFG = PhysicalCompatibilityConfig()


def _row(tracklet_id, points):
    return {
        "tracklet_id": tracklet_id,
        "observations": [
            {
                "detection_id": f"det-{tracklet_id}-{frame}",
                "frame_index": frame,
                "timestamp_s": timestamp,
                "x": x,
                "z": z,
                "confidence": confidence,
            }
            for frame, timestamp, x, z, confidence in points
        ],
        "metadata": {"builder": "test"},
    }


def _node(tracklet_id, points):
    return tracklet_node_from_row(_row(tracklet_id, points), CFG)


def test_simultaneous_far_tracklets_are_hard_cannot_link():
    a = _node(
        "trk-a",
        [(0, 0.0, 0.0, 0.0, 0.9), (3, 0.1, 0.1, 0.0, 0.9)],
    )
    b = _node(
        "trk-b",
        [(0, 0.0, 10.0, 0.0, 0.9), (3, 0.1, 10.1, 0.0, 0.9)],
    )
    edge = build_candidate_edge(a, b, CFG)
    assert edge.status == "cannot_link"
    assert edge.reason == "simultaneous_spatial_separation"
    assert edge.hard_constraint is True


def test_simultaneous_close_tracklets_remain_candidates():
    a = _node(
        "trk-a",
        [(0, 0.0, 0.0, 0.0, 0.9), (3, 0.1, 0.1, 0.0, 0.9)],
    )
    b = _node(
        "trk-b",
        [(0, 0.0, 0.6, 0.0, 0.9), (3, 0.1, 0.7, 0.0, 0.9)],
    )
    edge = build_candidate_edge(a, b, CFG)
    assert edge.status == "candidate"
    assert edge.reason == "overlap_close"


def test_simultaneous_mid_distance_is_ambiguous_not_hard_banned():
    a = _node(
        "trk-a",
        [(0, 0.0, 0.0, 0.0, 0.9), (3, 0.1, 0.0, 0.0, 0.9)],
    )
    b = _node(
        "trk-b",
        [(0, 0.0, 2.0, 0.0, 0.9), (3, 0.1, 2.0, 0.0, 0.9)],
    )
    edge = build_candidate_edge(a, b, CFG)
    assert edge.status == "candidate"
    assert edge.reason == "overlap_ambiguous"


def test_reachable_temporal_transition_is_candidate():
    a = _node(
        "trk-a",
        [(0, 0.0, 0.0, 0.0, 0.9), (3, 0.1, 0.2, 0.0, 0.9)],
    )
    b = _node(
        "trk-b",
        [(6, 0.2, 1.8, 0.0, 0.9), (9, 0.3, 2.0, 0.0, 0.9)],
    )
    edge = build_candidate_edge(a, b, CFG)
    assert edge.status == "candidate"
    assert edge.reason == "transition_reachable"
    assert edge.physics["gap_s"] == pytest.approx(0.1)


def test_unreachable_temporal_transition_is_hard_cannot_link():
    a = _node(
        "trk-a",
        [(0, 0.0, 0.0, 0.0, 0.9), (3, 0.1, 0.0, 0.0, 0.9)],
    )
    b = _node(
        "trk-b",
        [(6, 0.2, 20.0, 0.0, 0.9), (9, 0.3, 20.0, 0.0, 0.9)],
    )
    edge = build_candidate_edge(a, b, CFG)
    assert edge.status == "cannot_link"
    assert edge.reason == "transition_unreachable"


def test_appearance_never_overrides_physical_cannot_link():
    a = _node(
        "trk-a",
        [(0, 0.0, 0.0, 0.0, 0.9), (3, 0.1, 0.0, 0.0, 0.9)],
    )
    b = _node(
        "trk-b",
        [(0, 0.0, 15.0, 0.0, 0.9), (3, 0.1, 15.0, 0.0, 0.9)],
    )
    appearance = {
        "tracklet_a": "trk-a",
        "tracklet_b": "trk-b",
        "aggregate_cosine": 0.9999,
        "reliability": 0.99,
    }
    edge = build_candidate_edge(a, b, CFG, appearance=appearance)
    assert edge.status == "cannot_link"
    assert edge.appearance["aggregate_cosine"] == 0.9999


def test_singleton_tracklet_has_no_velocity_but_is_valid_node():
    node = _node("trk-a", [(3, 0.1, 1.0, 2.0, 0.8)])
    assert node.start_velocity_mps is None
    assert node.end_velocity_mps is None


def test_graph_contains_every_unordered_pair():
    nodes = [
        _node("trk-a", [(0, 0.0, 0.0, 0.0, 0.9)]),
        _node("trk-b", [(3, 0.1, 0.0, 0.0, 0.9)]),
        _node("trk-c", [(6, 0.2, 0.0, 0.0, 0.9)]),
        _node("trk-d", [(9, 0.3, 0.0, 0.0, 0.9)]),
    ]
    graph = build_candidate_graph(nodes, CFG)
    assert len(graph.nodes) == 4
    assert len(graph.edges) == 6
