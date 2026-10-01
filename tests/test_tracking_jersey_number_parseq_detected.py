from __future__ import annotations

import numpy as np

from experiments.tracking.run_v1_jersey_number_parseq_detected import (
    _best_parseq_candidate,
    _detected_crops,
    _horizontal_crop,
)


def test_horizontal_crop_extracts_expected_region():
    image = np.zeros(
        (100, 80, 3),
        dtype=np.uint8,
    )

    crop = _horizontal_crop(
        image,
        [10, 30, 20, 60],
    )

    assert crop is not None
    assert crop.shape == (
        40,
        20,
        3,
    )


def test_horizontal_crop_clamps_to_image():
    image = np.zeros(
        (50, 40, 3),
        dtype=np.uint8,
    )

    crop = _horizontal_crop(
        image,
        [-10, 100, -5, 70],
    )

    assert crop is not None
    assert crop.shape == (
        50,
        40,
        3,
    )


def test_detected_crops_supports_horizontal_boxes():
    image = np.zeros(
        (100, 100, 3),
        dtype=np.uint8,
    )

    crops = _detected_crops(
        image,
        horizontal=[
            [10, 40, 20, 60],
            [50, 80, 30, 70],
        ],
        free=[],
    )

    assert len(crops) == 2


def test_best_parseq_candidate_prefers_valid_digit():
    result = _best_parseq_candidate(
        [
            ("S9", None, 0.99),
            ("9", 9, 0.82),
            ("7", 7, 0.60),
        ]
    )

    raw, number, confidence, index = result

    assert raw == "9"
    assert number == 9
    assert confidence == 0.82
    assert index == 1


def test_best_parseq_candidate_uses_highest_valid_confidence():
    result = _best_parseq_candidate(
        [
            ("10", 10, 0.72),
            ("9", 9, 0.91),
        ]
    )

    assert result[1] == 9
    assert result[2] == 0.91


def test_best_parseq_candidate_preserves_invalid_raw_for_audit():
    result = _best_parseq_candidate(
        [
            ("S9", None, 0.80),
            ("ABC", None, 0.30),
        ]
    )

    assert result[0] == "S9"
    assert result[1] is None
    assert result[2] == 0.80


def test_best_parseq_candidate_empty_is_unknown():
    assert _best_parseq_candidate([]) == (
        None,
        None,
        0.0,
        None,
    )
