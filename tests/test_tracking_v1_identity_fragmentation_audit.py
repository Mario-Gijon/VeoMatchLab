from __future__ import annotations

import pytest

from tracking.identities.fragmentation_audit import (
    FragmentationAuditConfig,
    build_cross_identity_continuations,
    build_identity_fragmentation_records,
    effective_utility,
    load_candidate_graph,
    load_identity_meta,
    load_scored_links,
    load_tracklet_meta,
)


def _tracklet(tracklet_id, start, end, n=3):
    if n == 1:
        times = [start]
    else:
        step = (end - start) / (n - 1)
        times = [start + i * step for i in range(n)]
    return {
        "tracklet_id": tracklet_id,
        "observations": [
            {
                "frame_index": i,
                "timestamp_s": time_s,
            }
            for i, time_s in enumerate(times)
        ],
    }


def _candidate(a, b, *, pred, succ, gap, status="candidate"):
    if status == "cannot_link":
        return {
            "tracklet_a": a,
            "tracklet_b": b,
            "status": "cannot_link",
            "reason": "transition_unreachable",
            "physics": {},
        }
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": "candidate",
        "reason": "transition_reachable",
        "physics": {
            "predecessor": pred,
            "successor": succ,
            "gap_s": gap,
            "endpoint_distance_m": 1.0,
            "required_speed_after_slack_mps": 0.0,
            "prediction_error_mean_m": 1.0,
            "velocity_direction_cosine": 1.0,
        },
    }


def _score(a, b, score, confidence=1.0):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "rank_score": score,
        "evidence_confidence": confidence,
        "link_kind": "continuation",
        "identity_decision": None,
    }


def test_effective_utility_keeps_confidence_separate():
    assert effective_utility(
        _score("a", "b", 1.0, 1.0)
    ) == pytest.approx(1.0)
    assert effective_utility(
        _score("a", "b", 1.0, 0.2)
    ) == pytest.approx(0.2)


def test_strong_boundary_can_expose_trajectory_vs_clique_conflict():
    tracklets = load_tracklet_meta(
        [
            _tracklet("a", 0.0, 1.0),
            _tracklet("b", 1.1, 2.0),
            _tracklet("c", 2.1, 3.0),
        ]
    )
    identities = load_identity_meta(
        [
            {
                "identity_id": "identity-a",
                "tracklet_ids": ["a"],
            },
            {
                "identity_id": "identity-b",
                "tracklet_ids": ["b", "c"],
            },
        ],
        tracklets,
    )
    candidate_edges, hard = load_candidate_graph(
        [
            _candidate(
                "a", "b",
                pred="a", succ="b", gap=0.1,
            ),
            _candidate(
                "a", "c",
                pred="a", succ="c", gap=1.1,
            ),
            _candidate(
                "b", "c",
                pred="b", succ="c", gap=0.1,
            ),
        ]
    )
    scored = load_scored_links(
        [
            _score("a", "b", 0.95, 1.0),
            _score("a", "c", 0.05, 1.0),
            _score("b", "c", 0.90, 1.0),
        ]
    )

    links, threshold = build_cross_identity_continuations(
        identities=identities,
        tracklets=tracklets,
        candidate_edges=candidate_edges,
        hard=hard,
        scored_links=scored,
        config=FragmentationAuditConfig(
            strong_positive_quantile=0.95,
            strong_utility_floor=0.35,
        ),
    )

    ab = next(
        row
        for row in links
        if row["predecessor_tracklet_id"] == "a"
        and row["successor_tracklet_id"] == "b"
    )
    assert threshold >= 0.35
    assert ab["boundary_aligned"] is True
    assert ab["strong_boundary_handoff"] is True
    assert ab["whole_identity_merge_feasible"] is True
    assert ab["whole_identity_merge_delta"] < 0
    assert ab["trajectory_vs_clique_conflict"] is True


def test_hard_conflict_prevents_whole_identity_merge_signal():
    tracklets = load_tracklet_meta(
        [
            _tracklet("a", 0.0, 1.0),
            _tracklet("b", 1.1, 2.0),
            _tracklet("c", 1.1, 2.0),
        ]
    )
    identities = load_identity_meta(
        [
            {
                "identity_id": "identity-a",
                "tracklet_ids": ["a"],
            },
            {
                "identity_id": "identity-b",
                "tracklet_ids": ["b", "c"],
            },
        ],
        tracklets,
    )
    candidate_edges, hard = load_candidate_graph(
        [
            _candidate(
                "a", "b",
                pred="a", succ="b", gap=0.1,
            ),
            _candidate(
                "a", "c",
                pred="a", succ="c", gap=0.1,
                status="cannot_link",
            ),
            _candidate(
                "b", "c",
                pred="b", succ="c", gap=0.1,
            ),
        ]
    )
    scored = load_scored_links(
        [
            _score("a", "b", 0.95, 1.0),
            _score("b", "c", 0.90, 1.0),
        ]
    )

    links, _ = build_cross_identity_continuations(
        identities=identities,
        tracklets=tracklets,
        candidate_edges=candidate_edges,
        hard=hard,
        scored_links=scored,
        config=FragmentationAuditConfig(),
    )
    ab = links[0]
    assert ab["strong_positive"] is True
    assert ab["whole_identity_merge_feasible"] is False
    assert ab["trajectory_vs_clique_conflict"] is False
    assert (
        "strong_link_but_hard_merge_conflict"
        in ab["review_flags"]
    )


def test_long_singleton_is_not_called_fragmentation_candidate():
    tracklets = load_tracklet_meta(
        [
            _tracklet("a", 0.0, 9.0, n=10),
            _tracklet("b", 9.1, 10.0, n=3),
        ]
    )
    identities = load_identity_meta(
        [
            {
                "identity_id": "identity-a",
                "tracklet_ids": ["a"],
            },
            {
                "identity_id": "identity-b",
                "tracklet_ids": ["b"],
            },
        ],
        tracklets,
    )
    candidate_edges, hard = load_candidate_graph(
        [
            _candidate(
                "a", "b",
                pred="a", succ="b", gap=0.1,
            ),
        ]
    )
    scored = load_scored_links(
        [_score("a", "b", 0.55, 0.4)]
    )
    links, _ = build_cross_identity_continuations(
        identities=identities,
        tracklets=tracklets,
        candidate_edges=candidate_edges,
        hard=hard,
        scored_links=scored,
        config=FragmentationAuditConfig(),
    )
    records = build_identity_fragmentation_records(
        identities=identities,
        boundary_links=links,
        clip_start_s=0.0,
        clip_end_s=10.0,
        config=FragmentationAuditConfig(),
    )
    identity_a = next(
        row
        for row in records
        if row["identity_id"] == "identity-a"
    )
    assert identity_a["review_class"] == "long_singleton"
    assert identity_a["requires_review"] is False



def test_threshold_override_matches_solver_contract():
    config = FragmentationAuditConfig(
        strong_utility_threshold_override=0.7,
    )
    tracklets = load_tracklet_meta(
        [
            _tracklet("a", 0.0, 1.0),
            _tracklet("b", 1.1, 2.0),
            _tracklet("c", 2.1, 3.0),
        ]
    )
    identities = load_identity_meta(
        [
            {"identity_id": "ia", "tracklet_ids": ["a"]},
            {"identity_id": "ib", "tracklet_ids": ["b", "c"]},
        ],
        tracklets,
    )
    candidate_edges, hard = load_candidate_graph(
        [
            _candidate("a", "b", pred="a", succ="b", gap=0.1),
            _candidate("a", "c", pred="a", succ="c", gap=1.1),
            _candidate("b", "c", pred="b", succ="c", gap=0.1),
        ]
    )
    scored = load_scored_links(
        [
            _score("a", "b", 0.82, 1.0),
            _score("a", "c", 0.05, 1.0),
            _score("b", "c", 0.95, 1.0),
        ]
    )
    links, threshold = build_cross_identity_continuations(
        identities=identities,
        tracklets=tracklets,
        candidate_edges=candidate_edges,
        hard=hard,
        scored_links=scored,
        config=config,
    )
    assert threshold == pytest.approx(0.7)
