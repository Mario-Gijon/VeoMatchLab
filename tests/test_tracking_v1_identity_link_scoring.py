from __future__ import annotations

import pytest

from tracking.identities.link_scoring import (
    AppearanceCalibration,
    fit_soft_link_calibration,
    load_appearance_calibration,
    score_candidate_edge,
)


def _benchmark():
    return {
        "tracklet_level": {
            "split_positive": {
                "p10": 0.9344,
            },
            "concurrent_negative": {
                "p95": 0.9095,
            },
            "roc_auc": 0.984,
            "p10_positive_minus_p95_negative": 0.0249,
        }
    }


def _continuation(
    a,
    b,
    *,
    gap,
    speed,
    prediction_error,
    direction,
    cosine=None,
    reliability=0.8,
):
    appearance = None
    if cosine is not None:
        appearance = {
            "aggregate_cosine": cosine,
            "bidirectional_best_similarity": {
                "median": cosine,
            },
            "reliability": reliability,
            "support_quality": 0.9,
            "sample_count_a": 3,
            "sample_count_b": 3,
        }
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": "candidate",
        "reason": "transition_reachable",
        "temporal_relation": "non_overlapping",
        "physics": {
            "gap_s": gap,
            "required_speed_after_slack_mps": speed,
            "prediction_error_mean_m": prediction_error,
            "velocity_direction_cosine": direction,
        },
        "appearance": appearance,
    }


def _overlap(a, b, *, median, frames, cosine=None):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": "candidate",
        "reason": (
            "overlap_close"
            if median <= 1.5
            else "overlap_ambiguous"
        ),
        "temporal_relation": "simultaneous_samples",
        "physics": {
            "shared_frame_count": frames,
            "shared_frame_separation_m": {
                "median": median,
            },
            "hard_median_threshold_m": 3.0,
        },
        "appearance": (
            {
                "aggregate_cosine": cosine,
                "bidirectional_best_similarity": {
                    "median": cosine,
                },
                "reliability": 0.8,
                "support_quality": 0.9,
            }
            if cosine is not None
            else None
        ),
    }


def _calibration(edges):
    return fit_soft_link_calibration(
        edges,
        load_appearance_calibration(_benchmark()),
    )


def test_appearance_calibration_uses_benchmark_anchors():
    calibration = load_appearance_calibration(_benchmark())
    assert calibration.support(0.9095) == pytest.approx(0.0)
    assert calibration.support(0.9344) == pytest.approx(1.0)
    midpoint = (0.9095 + 0.9344) / 2
    assert calibration.support(midpoint) == pytest.approx(0.5)


def test_appearance_calibration_accepts_prtreid_report_wrapper():
    wrapped = {
        "backend": {
            "model_id": "prtreid-soccernet-baseline-global-v1",
            "embedding_dim": 256,
        },
        "report": _benchmark(),
    }
    calibration = load_appearance_calibration(wrapped)
    assert calibration.negative_p95 == pytest.approx(0.9095)
    assert calibration.positive_p10 == pytest.approx(0.9344)
    assert calibration.benchmark_roc_auc == pytest.approx(0.984)


def test_continuation_rank_rewards_consistent_short_link():
    good = _continuation(
        "trk-a",
        "trk-b",
        gap=0.1,
        speed=1.5,
        prediction_error=1.0,
        direction=0.95,
        cosine=0.97,
    )
    bad = _continuation(
        "trk-c",
        "trk-d",
        gap=8.0,
        speed=10.0,
        prediction_error=30.0,
        direction=-0.8,
        cosine=0.75,
    )
    middle = _continuation(
        "trk-e",
        "trk-f",
        gap=2.0,
        speed=4.0,
        prediction_error=5.0,
        direction=0.2,
        cosine=0.92,
    )
    calibration = _calibration([good, bad, middle])

    good_score = score_candidate_edge(
        good, calibration
    ).rank_score
    bad_score = score_candidate_edge(
        bad, calibration
    ).rank_score

    assert good_score > bad_score


def test_missing_appearance_is_unknown_not_zero_evidence():
    edges = [
        _continuation(
            "trk-a",
            "trk-b",
            gap=0.1,
            speed=1.0,
            prediction_error=1.0,
            direction=1.0,
            cosine=None,
        ),
        _continuation(
            "trk-c",
            "trk-d",
            gap=1.0,
            speed=3.0,
            prediction_error=3.0,
            direction=0.5,
            cosine=0.95,
        ),
    ]
    calibration = _calibration(edges)
    link = score_candidate_edge(edges[0], calibration)

    assert link.family_scores["appearance"] is None
    assert link.rank_score > 0.0
    assert link.components["appearance"]["available"] is False


def test_low_reliability_reid_is_shrunk_toward_neutral():
    edges = [
        _continuation(
            "trk-a",
            "trk-b",
            gap=0.1,
            speed=1.0,
            prediction_error=1.0,
            direction=1.0,
            cosine=0.97,
            reliability=0.01,
        ),
        _continuation(
            "trk-c",
            "trk-d",
            gap=2.0,
            speed=5.0,
            prediction_error=5.0,
            direction=0.0,
            cosine=0.8,
        ),
    ]
    calibration = _calibration(edges)
    link = score_candidate_edge(edges[0], calibration)
    app = link.components["appearance"]

    assert app["raw_support"] == pytest.approx(1.0)
    assert 0.5 < app["effective_support"] < 0.6


def test_overlap_close_scores_higher_geometry_than_ambiguous():
    close = _overlap(
        "trk-a",
        "trk-b",
        median=0.5,
        frames=10,
        cosine=0.96,
    )
    ambiguous = _overlap(
        "trk-c",
        "trk-d",
        median=2.5,
        frames=10,
        cosine=0.96,
    )
    # Need at least continuation edges to fit continuation scales.
    continuation_a = _continuation(
        "trk-e",
        "trk-f",
        gap=0.1,
        speed=1.0,
        prediction_error=1.0,
        direction=1.0,
        cosine=0.95,
    )
    continuation_b = _continuation(
        "trk-g",
        "trk-h",
        gap=2.0,
        speed=4.0,
        prediction_error=5.0,
        direction=0.0,
        cosine=0.8,
    )
    calibration = _calibration(
        [close, ambiguous, continuation_a, continuation_b]
    )

    close_link = score_candidate_edge(close, calibration)
    ambiguous_link = score_candidate_edge(
        ambiguous, calibration
    )

    assert (
        close_link.family_scores["overlap_geometry"]
        > ambiguous_link.family_scores["overlap_geometry"]
    )


def test_rank_score_is_not_identity_decision():
    edge = _continuation(
        "trk-a",
        "trk-b",
        gap=0.1,
        speed=1.0,
        prediction_error=1.0,
        direction=1.0,
        cosine=0.98,
    )
    other = _continuation(
        "trk-c",
        "trk-d",
        gap=2.0,
        speed=5.0,
        prediction_error=5.0,
        direction=0.0,
        cosine=0.8,
    )
    calibration = _calibration([edge, other])
    payload = score_candidate_edge(edge, calibration).to_dict()

    assert payload["identity_decision"] is None
    assert payload["rank_score_semantics"] == (
        "relative_evidence_ranking_not_probability"
    )
