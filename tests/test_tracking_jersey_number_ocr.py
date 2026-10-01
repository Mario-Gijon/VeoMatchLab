from __future__ import annotations

import numpy as np

from experiments.tracking.run_v1_jersey_number_ocr import (
    _build_preprocess_variants,
    _choose_variant_consensus,
    _select_rows,
)


def test_strongest_selection_sorts_before_limit():
    rows = [
        {
            "tracklet_id": "a",
            "frame_index": 1,
            "readiness_score": 0.4,
            "rank_within_tracklet": 1,
        },
        {
            "tracklet_id": "b",
            "frame_index": 2,
            "readiness_score": 0.9,
            "rank_within_tracklet": 1,
        },
        {
            "tracklet_id": "c",
            "frame_index": 3,
            "readiness_score": 0.7,
            "rank_within_tracklet": 1,
        },
    ]

    selected = _select_rows(
        rows,
        selection="strongest",
        limit=2,
    )

    assert [
        row["tracklet_id"]
        for row in selected
    ] == ["b", "c"]


def test_input_selection_preserves_input_order():
    rows = [
        {
            "tracklet_id": "a",
            "readiness_score": 0.1,
        },
        {
            "tracklet_id": "b",
            "readiness_score": 0.9,
        },
    ]

    selected = _select_rows(
        rows,
        selection="input",
        limit=None,
    )

    assert [
        row["tracklet_id"]
        for row in selected
    ] == ["a", "b"]


def test_jersey_preprocessing_builds_expected_variants():
    image = np.zeros(
        (20, 10, 3),
        dtype=np.uint8,
    )
    variants = _build_preprocess_variants(
        image,
        profile="jersey",
    )

    names = [
        name for name, _ in variants
    ]
    assert names == [
        "original",
        "x2",
        "x4",
        "x4_clahe",
        "x4_sharpen",
    ]
    assert variants[1][1].shape[:2] == (
        40,
        20,
    )
    assert variants[2][1].shape[:2] == (
        80,
        40,
    )


def test_variant_consensus_prefers_agreement_over_small_conf_gap():
    predictions = [
        {
            "variant": "original",
            "raw_text": "17",
            "parsed_number": 17,
            "ocr_confidence": 0.72,
        },
        {
            "variant": "x2",
            "raw_text": "17",
            "parsed_number": 17,
            "ocr_confidence": 0.70,
        },
        {
            "variant": "x4",
            "raw_text": "71",
            "parsed_number": 71,
            "ocr_confidence": 0.75,
        },
    ]

    chosen = _choose_variant_consensus(
        predictions
    )

    assert chosen["parsed_number"] == 17
    assert (
        chosen["variant_support_count"]
        == 2
    )


def test_variant_consensus_keeps_single_clear_prediction():
    predictions = [
        {
            "variant": "original",
            "raw_text": None,
            "parsed_number": None,
            "ocr_confidence": 0.0,
        },
        {
            "variant": "x4",
            "raw_text": "9",
            "parsed_number": 9,
            "ocr_confidence": 0.88,
        },
    ]

    chosen = _choose_variant_consensus(
        predictions
    )

    assert chosen["parsed_number"] == 9
    assert chosen["ocr_confidence"] == 0.88
    assert (
        chosen["variant_support_count"]
        == 1
    )


def test_variant_consensus_unknown_when_nothing_reads():
    predictions = [
        {
            "variant": "original",
            "raw_text": None,
            "parsed_number": None,
            "ocr_confidence": 0.0,
        }
    ]

    chosen = _choose_variant_consensus(
        predictions
    )

    assert chosen["parsed_number"] is None
    assert chosen["ocr_confidence"] == 0.0
