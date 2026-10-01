from __future__ import annotations

import numpy as np

from experiments.tracking.run_v1_jersey_numberbox_parseq import (
    _crop_candidate,
    _expanded_xyxy,
    _sequence_confidence,
    _strict_parse_number,
)


def test_strict_parse_number_accepts_only_one_or_two_digits():
    assert _strict_parse_number("9") == 9
    assert _strict_parse_number("10") == 10
    assert _strict_parse_number(" 15 ") == 15


def test_strict_parse_number_rejects_noise_and_zero():
    assert _strict_parse_number("S9") is None
    assert _strict_parse_number("9a") is None
    assert _strict_parse_number("100") is None
    assert _strict_parse_number("0") is None
    assert _strict_parse_number("") is None


def test_sequence_confidence_scalar():
    assert _sequence_confidence(0.8) == 0.8


def test_sequence_confidence_multiplies_token_scores():
    result = _sequence_confidence([0.9, 0.8])
    assert abs(result - 0.72) < 1e-9


def test_sequence_confidence_flattens_nested_values():
    result = _sequence_confidence([[0.5], [0.8]])
    assert abs(result - 0.4) < 1e-9


def test_expanded_xyxy_adds_padding_and_clamps():
    bounds = _expanded_xyxy(
        [10, 10, 30, 30],
        image_width=100,
        image_height=80,
        pad_x=0.25,
        pad_y=0.50,
    )
    assert bounds == (5, 0, 35, 40)


def test_expanded_xyxy_rejects_invalid_box():
    assert _expanded_xyxy(
        [20, 20, 10, 30],
        image_width=100,
        image_height=100,
        pad_x=0.1,
        pad_y=0.1,
    ) is None


def test_crop_candidate_returns_expected_region():
    image = np.zeros((50, 60, 3), dtype=np.uint8)
    image[10:30, 20:40] = 255

    result = _crop_candidate(
        image,
        [20, 10, 40, 30],
        pad_x=0.0,
        pad_y=0.0,
    )

    assert result is not None
    crop, bounds = result
    assert bounds == (20, 10, 40, 30)
    assert crop.shape == (20, 20, 3)
    assert int(crop.mean()) == 255
