from __future__ import annotations

from tracking.appearance.jersey_number import (
    JerseyNumberAggregationConfig,
    JerseyNumberObservation,
    aggregate_all_tracklets,
    aggregate_tracklet_jersey_evidence,
    parse_jersey_number,
)


def _obs(
    *,
    tracklet="a",
    detection="d1",
    number=17,
    confidence=0.9,
    readiness=0.9,
    rank=1,
    time=0.0,
):
    return JerseyNumberObservation(
        tracklet_id=tracklet,
        detection_id=detection,
        frame_index=int(round(time * 10)),
        timestamp_s=time,
        raw_text=(None if number is None else str(number)),
        parsed_number=number,
        ocr_confidence=confidence,
        readiness_score=readiness,
        rank_within_tracklet=rank,
        backend="test",
    )


def test_parse_jersey_number_is_conservative():
    assert parse_jersey_number("17") == 17
    assert parse_jersey_number("O9") == 9
    assert parse_jersey_number("7x") == 7
    assert parse_jersey_number("177") is None
    assert parse_jersey_number("") is None
    assert parse_jersey_number(None) is None


def test_no_read_is_unknown_not_negative():
    profile = aggregate_tracklet_jersey_evidence(
        [
            _obs(number=None, confidence=0.0),
            _obs(
                detection="d2",
                number=None,
                confidence=0.0,
                time=1.0,
            ),
        ]
    )

    assert profile.best_number is None
    assert profile.strength == "unknown"
    assert profile.total_evidence_weight == 0.0


def test_two_strong_agreeing_views_produce_strong_evidence():
    profile = aggregate_tracklet_jersey_evidence(
        [
            _obs(
                detection="d1",
                number=17,
                confidence=0.95,
                time=0.0,
            ),
            _obs(
                detection="d2",
                number=17,
                confidence=0.90,
                rank=2,
                time=1.0,
            ),
            _obs(
                detection="d3",
                number=None,
                confidence=0.0,
                rank=3,
                time=2.0,
            ),
        ]
    )

    assert profile.best_number == 17
    assert profile.strength == "strong"
    assert profile.supports[0].support_view_count == 2
    assert profile.unknown_view_count == 1


def test_conflicting_weak_read_does_not_override_strong_consensus():
    profile = aggregate_tracklet_jersey_evidence(
        [
            _obs(
                detection="d1",
                number=17,
                confidence=0.95,
                time=0.0,
            ),
            _obs(
                detection="d2",
                number=17,
                confidence=0.92,
                rank=2,
                time=1.0,
            ),
            _obs(
                detection="d3",
                number=11,
                confidence=0.25,
                readiness=0.5,
                rank=3,
                time=2.0,
            ),
        ]
    )

    assert profile.best_number == 17
    assert profile.supports[0].number == 17
    assert profile.supports[0].normalized_support > 0.9


def test_one_read_can_be_evidence_but_not_strong():
    profile = aggregate_tracklet_jersey_evidence(
        [
            _obs(
                number=10,
                confidence=0.95,
            )
        ]
    )

    assert profile.best_number == 10
    assert profile.strength != "strong"


def test_equal_conflicting_reads_preserve_ambiguity():
    profile = aggregate_tracklet_jersey_evidence(
        [
            _obs(
                detection="d1",
                number=7,
                confidence=0.8,
                readiness=0.9,
                time=0.0,
            ),
            _obs(
                detection="d2",
                number=17,
                confidence=0.8,
                readiness=0.9,
                time=1.0,
            ),
        ]
    )

    assert len(profile.supports) == 2
    assert profile.margin is not None
    assert profile.margin < 0.2
    assert profile.strength == "weak"


def test_tracklets_are_aggregated_independently():
    profiles = aggregate_all_tracklets(
        [
            _obs(tracklet="a", number=9),
            _obs(
                tracklet="b",
                detection="d2",
                number=10,
                time=1.0,
            ),
        ]
    )

    by_id = {item.tracklet_id: item for item in profiles}
    assert by_id["a"].best_number == 9
    assert by_id["b"].best_number == 10


def test_output_never_claims_hard_identity():
    profile = aggregate_tracklet_jersey_evidence(
        [
            _obs(detection="d1", number=17),
            _obs(
                detection="d2",
                number=17,
                time=1.0,
            ),
        ]
    )

    assert profile.to_dict()["hard_identity_fact"] is False
