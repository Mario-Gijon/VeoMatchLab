from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracking.appearance.quality import (
    ReIDCropQualityConfig,
    grayscale_metrics,
    score_reid_crop,
)


def test_default_weights_sum_to_one():
    config = ReIDCropQualityConfig()
    assert (
        config.resolution_weight
        + config.sharpness_weight
        + config.contrast_weight
        + config.selection_weight
        + config.clean_border_weight
    ) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hard_min_bbox_height_px": -1},
        {"min_reid_quality": 1.1},
        {"resolution_height_good_px": 20},
        {"sharpness_good": 5},
        {"contrast_good": 5},
        {"black_fraction_bad": 0.01},
        {"resolution_weight": 0.50},
        {"good_threshold": 0.60},
    ],
)
def test_invalid_config_rejected(kwargs):
    with pytest.raises(ValueError):
        ReIDCropQualityConfig(**kwargs)


def test_grayscale_metrics_detect_flat_image():
    crop = np.full((80, 40, 3), 120, dtype=np.uint8)
    lap, std, black = grayscale_metrics(crop)

    assert lap == pytest.approx(0.0)
    assert std == pytest.approx(0.0)
    assert black == pytest.approx(0.0)


def test_grayscale_metrics_detect_black_border_fraction():
    crop = np.full((20, 20, 3), 100, dtype=np.uint8)
    crop[:, :2] = 0

    _, _, black = grayscale_metrics(crop)

    assert black > 0.20
    assert black < 0.35


def checkerboard(height=120, width=60):
    yy, xx = np.indices((height, width))
    gray = np.where((xx + yy) % 2 == 0, 40, 220).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def test_large_sharp_crop_scores_high():
    result = score_reid_crop(
        crop=checkerboard(),
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.95,
        crop_clipped=False,
    )

    assert result.reid_eligible
    assert result.reid_quality > 0.85
    assert result.tier == "good"
    assert result.resolution_score == pytest.approx(1.0)


def test_small_bbox_is_rejected_even_with_good_pixels():
    result = score_reid_crop(
        crop=checkerboard(50, 20),
        virtual_bbox_xyxy=(0, 0, 20, 40),
        selection_quality=1.0,
        crop_clipped=False,
    )

    assert not result.reid_eligible
    assert "bbox_height_below_min" in result.rejection_reasons


def test_narrow_bbox_is_rejected():
    result = score_reid_crop(
        crop=checkerboard(120, 10),
        virtual_bbox_xyxy=(0, 0, 9, 120),
        selection_quality=1.0,
        crop_clipped=False,
    )

    assert not result.reid_eligible
    assert "bbox_width_below_min" in result.rejection_reasons


def test_flat_blurry_crop_scores_below_sharp_equivalent():
    sharp = checkerboard(120, 60)
    flat = np.full_like(sharp, 120)

    sharp_result = score_reid_crop(
        crop=sharp,
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.9,
        crop_clipped=False,
    )
    flat_result = score_reid_crop(
        crop=flat,
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.9,
        crop_clipped=False,
    )

    assert sharp_result.reid_quality > flat_result.reid_quality
    assert sharp_result.sharpness_score > flat_result.sharpness_score


def test_black_border_penalizes_quality():
    clean = checkerboard(120, 60)
    dirty = clean.copy()
    dirty[:, :2] = 0

    clean_result = score_reid_crop(
        crop=clean,
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.9,
        crop_clipped=False,
    )
    dirty_result = score_reid_crop(
        crop=dirty,
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.9,
        crop_clipped=False,
    )

    assert clean_result.clean_border_score > dirty_result.clean_border_score


def test_clipped_crop_reduces_clean_border_component():
    crop = checkerboard(120, 60)

    unclipped = score_reid_crop(
        crop=crop,
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.9,
        crop_clipped=False,
    )
    clipped = score_reid_crop(
        crop=crop,
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.9,
        crop_clipped=True,
    )

    assert clipped.clean_border_score == pytest.approx(
        0.5 * unclipped.clean_border_score
    )
    assert clipped.reid_quality < unclipped.reid_quality


def test_result_dict_keeps_selection_and_reid_quality_separate():
    result = score_reid_crop(
        crop=checkerboard(),
        virtual_bbox_xyxy=(0, 0, 60, 120),
        selection_quality=0.77,
        crop_clipped=False,
    )
    payload = result.to_dict()

    assert payload["selection_quality"] == pytest.approx(0.77)
    assert 0.0 <= payload["reid_quality"] <= 1.0
    assert "player_id" not in payload
