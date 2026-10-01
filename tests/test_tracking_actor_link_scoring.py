from __future__ import annotations

import pytest

from tracking.actors.link_scoring import (
    ActorLinkScoringConfig,
    EmpiricalReIDCalibration,
    KitAssignment,
    _kit_family,
    _rank_component,
    build_actor_link_scores,
)


def _candidate(
    a: str,
    b: str,
    *,
    cosine: float = 0.95,
    local: bool = False,
    rank_a: int | None = 1,
    rank_b: int | None = 1,
):
    reasons = ["appearance_topk"]
    local_row = None
    if local:
        reasons.append("local_physics")
        local_row = {
            "gap_s": 0.1,
            "endpoint_distance_m": 0.7,
            "required_speed_after_slack_mps": 0.0,
            "forward_prediction_error_m": 0.4,
            "backward_prediction_error_m": 0.5,
            "velocity_direction_cosine": 0.9,
            "predecessor_end_confidence": 0.95,
            "successor_start_confidence": 0.95,
        }

    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "reasons": reasons,
        "local_transition": local_row,
        "close_overlap": None,
        "appearance_retrieval": {
            "aggregate_cosine": cosine,
            "rank_a_to_b": rank_a,
            "rank_b_to_a": rank_b,
            "mutual_topk": (
                rank_a is not None and rank_b is not None
            ),
        },
        "appearance_evidence": {
            "aggregate_cosine": cosine,
            "reliability": 0.9,
            "support_quality": 0.9,
            "sample_count_a": 2,
            "sample_count_b": 2,
            "effective_sample_count_a": 1.8,
            "effective_sample_count_b": 1.8,
            "cross_similarity": {"median": cosine},
            "bidirectional_best_similarity": {
                "median": cosine
            },
        },
    }


def _kit(tracklet: str, cluster: int):
    return {
        "tracklet_id": tracklet,
        "cluster_id": cluster,
        "best_cosine": 0.95,
        "second_best_cosine": 0.7,
        "margin": 0.25,
        "profile_reliability": 0.9,
        "sample_count": 2,
        "anchor_used_for_fit": True,
    }


def _positive(cosine):
    return {
        "tracklet_a": "x",
        "tracklet_b": "x",
        "cosine": cosine,
    }


def _negative(a, b, cosine):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "cosine": cosine,
    }


def test_kit_family():
    a = KitAssignment.from_row(_kit("a", 0))
    b = KitAssignment.from_row(_kit("b", 0))
    c = KitAssignment.from_row(_kit("c", 1))

    assert _kit_family(a, b) == "same_kit"
    assert _kit_family(a, c) == "different_kit"
    assert _kit_family(a, None) == "unknown_kit"


def test_empirical_reid_high_cosine_scores_above_low():
    calibration = EmpiricalReIDCalibration(
        split_positive=(0.85, 0.90, 0.95, 0.98),
        same_kit_negative=(0.50, 0.70, 0.85, 0.92),
        different_kit_negative=(0.30, 0.50, 0.70),
        unknown_kit_negative=(0.40, 0.60, 0.80),
    )

    low = calibration.reid_support(
        0.75,
        kit_family="same_kit",
    )
    high = calibration.reid_support(
        0.96,
        kit_family="same_kit",
    )

    assert high["support"] > low["support"]
    assert (
        high["negative_tail_fraction"]
        < low["negative_tail_fraction"]
    )


def test_rank_component_strongly_prefers_mutual_top1():
    config = ActorLinkScoringConfig()

    top1 = _rank_component(
        {
            "rank_a_to_b": 1,
            "rank_b_to_a": 1,
        },
        config,
    )
    weak = _rank_component(
        {
            "rank_a_to_b": 30,
            "rank_b_to_a": None,
        },
        config,
    )

    assert top1["support"] > weak["support"]


def test_build_scores_does_not_assign_identity():
    candidates = [
        _candidate("a", "b", local=True),
    ]
    kits = [_kit("a", 0), _kit("b", 0)]
    positives = [
        _positive(0.90),
        _positive(0.95),
    ]
    negatives = [
        _negative("a", "b", 0.70),
        _negative("a", "b", 0.80),
    ]

    links, summary = build_actor_link_scores(
        candidates,
        kit_assignments=kits,
        split_positive_rows=positives,
        simultaneous_negative_rows=negatives,
    )

    assert len(links) == 1
    assert 0.0 <= links[0].rank_score <= 1.0
    assert links[0].mutual_best is True
    assert links[0].to_dict()["identity_decision"] is None
    assert summary["identity_decisions"] is False


def test_competition_marks_only_endpoint_best():
    candidates = [
        _candidate("a", "b", cosine=0.98),
        _candidate("a", "c", cosine=0.90),
        _candidate("b", "d", cosine=0.80),
    ]
    kits = [
        _kit("a", 0),
        _kit("b", 0),
        _kit("c", 0),
        _kit("d", 0),
    ]
    positives = [
        _positive(0.88),
        _positive(0.93),
        _positive(0.98),
    ]
    negatives = [
        _negative("a", "b", 0.60),
        _negative("a", "c", 0.75),
        _negative("b", "d", 0.82),
    ]

    links, _ = build_actor_link_scores(
        candidates,
        kit_assignments=kits,
        split_positive_rows=positives,
        simultaneous_negative_rows=negatives,
    )

    by_pair = {
        (link.tracklet_a, link.tracklet_b): link
        for link in links
    }

    assert by_pair[("a", "b")].endpoint_rank_a == 1
    assert by_pair[("a", "c")].endpoint_rank_a > 1


def test_different_kit_softly_reduces_support():
    same_candidate = _candidate("a", "b", cosine=0.94)
    diff_candidate = _candidate("a", "c", cosine=0.94)

    kits = [
        _kit("a", 0),
        _kit("b", 0),
        _kit("c", 1),
    ]
    positives = [
        _positive(0.90),
        _positive(0.95),
        _positive(0.98),
    ]
    negatives = [
        _negative("a", "b", 0.80),
        _negative("a", "c", 0.60),
        _negative("b", "c", 0.65),
    ]

    links, _ = build_actor_link_scores(
        [same_candidate, diff_candidate],
        kit_assignments=kits,
        split_positive_rows=positives,
        simultaneous_negative_rows=negatives,
    )

    rows = {
        (link.tracklet_a, link.tracklet_b): link
        for link in links
    }

    same_kit_support = rows[
        ("a", "b")
    ].components["kit"]["support"]
    different_kit_support = rows[
        ("a", "c")
    ].components["kit"]["support"]

    assert same_kit_support > different_kit_support
