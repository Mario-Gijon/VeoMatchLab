from __future__ import annotations

from tracking.actors.temporal_backbone import (
    TemporalBackboneConfig,
    solve_temporal_backbone,
)
from tracking.tracklets.models import (
    Tracklet,
    TrackletObservation,
)


def _tracklet(
    tracklet_id: str,
    start: float,
    end: float,
    x: float = 0.0,
) -> Tracklet:
    if end == start:
        rows = [(0, start)]
    else:
        rows = [(0, start), (1, end)]
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=tuple(
            TrackletObservation(
                detection_id=f"{tracklet_id}-{index}",
                frame_index=index,
                timestamp_s=time,
                x=x,
                z=0.0,
                confidence=0.95,
            )
            for index, (_, time) in enumerate(rows)
        ),
    )


def _link(
    a: str,
    b: str,
    *,
    score: float,
    rank_a: int = 1,
    rank_b: int = 1,
    family: str = "local_continuation",
):
    return {
        "tracklet_a": min(a, b),
        "tracklet_b": max(a, b),
        "link_family": family,
        "rank_score": score,
        "evidence_confidence": 0.9,
        "endpoint_rank_a": (
            rank_a if min(a, b) == a else rank_b
        ),
        "endpoint_rank_b": (
            rank_b if max(a, b) == b else rank_a
        ),
    }


def _pairs(result):
    return {
        (
            edge.predecessor_tracklet_id,
            edge.successor_tracklet_id,
        )
        for edge in result.accepted_edges
    }


def test_builds_multi_tracklet_path():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 0.3, 0.5),
        _tracklet("c", 0.6, 0.8),
    ]
    links = [
        _link("a", "b", score=0.9),
        _link("b", "c", score=0.85, rank_a=2),
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=[],
    )

    assert len(result.paths) == 1
    assert result.paths[0].tracklet_ids == (
        "a",
        "b",
        "c",
    )


def test_one_successor_limit_keeps_stronger_edge():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 0.3, 0.5),
        _tracklet("c", 0.4, 0.6),
    ]
    links = [
        _link("a", "b", score=0.90),
        _link("a", "c", score=0.80, rank_b=2),
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=[],
    )

    assert ("a", "b") in _pairs(result)
    assert ("a", "c") not in _pairs(result)


def test_one_predecessor_limit_keeps_stronger_edge():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 0.0, 0.2),
        _tracklet("c", 0.4, 0.6),
    ]
    links = [
        _link("a", "c", score=0.90),
        _link("b", "c", score=0.80, rank_b=2),
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=[],
    )

    assert ("a", "c") in _pairs(result)
    assert ("b", "c") not in _pairs(result)


def test_transitive_component_hard_conflict_blocks_extension():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 0.3, 0.5),
        _tracklet("c", 0.6, 0.8),
    ]
    links = [
        _link("a", "b", score=0.90),
        _link("b", "c", score=0.85, rank_a=2),
    ]
    conflicts = [
        {
            "tracklet_a": "a",
            "tracklet_b": "c",
        }
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=conflicts,
    )

    assert ("a", "b") in _pairs(result)
    assert ("b", "c") not in _pairs(result)
    assert (
        result.rejection_counts[
            "component_hard_conflict"
        ]
        == 1
    )


def test_appearance_rescue_is_deferred():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 5.0, 5.2),
    ]
    links = [
        _link(
            "a",
            "b",
            score=0.95,
            family="appearance_rescue",
        )
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=[],
    )

    assert result.accepted_edges == ()
    assert len(result.paths) == 2
    assert (
        result.rejection_counts["non_local_family"]
        == 1
    )


def test_endpoint_rank_limit_is_conservative():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 0.3, 0.5),
    ]
    links = [
        _link(
            "a",
            "b",
            score=0.95,
            rank_a=4,
            rank_b=1,
        )
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=[],
        config=TemporalBackboneConfig(
            min_rank_score=0.70,
            max_endpoint_rank=3,
        ),
    )

    assert result.accepted_edges == ()
    assert (
        result.rejection_counts[
            "endpoint_rank_above_limit"
        ]
        == 1
    )


def test_low_score_is_rejected_and_singletons_preserved():
    tracklets = [
        _tracklet("a", 0.0, 0.2),
        _tracklet("b", 0.3, 0.5),
        _tracklet("c", 1.0, 1.1),
    ]
    links = [
        _link("a", "b", score=0.69),
    ]

    result = solve_temporal_backbone(
        tracklets=tracklets,
        scored_links=links,
        hard_conflicts=[],
    )

    assert len(result.paths) == 3
    assert result.accepted_edges == ()
    assert {
        row["tracklet_id"]
        for row in result.assignments
    } == {"a", "b", "c"}
