from __future__ import annotations

import numpy as np
import pytest

from tracking.appearance.crops import (
    AppearanceCropConfig,
    crop_virtual_person,
    normalize_bbox_xyxy,
    padded_bbox_xyxy,
    safe_filename_token,
)


def test_default_crop_config_is_valid():
    config = AppearanceCropConfig()

    assert config.padding_x_fraction == pytest.approx(0.08)
    assert config.padding_y_fraction == pytest.approx(0.06)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"padding_x_fraction": -0.1},
        {"padding_y_fraction": -0.1},
        {"min_crop_width_px": 0},
        {"min_crop_height_px": 0},
    ],
)
def test_crop_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AppearanceCropConfig(**kwargs)


def test_normalize_bbox_requires_positive_area():
    assert normalize_bbox_xyxy((1, 2, 5, 9)) == (
        1.0,
        2.0,
        5.0,
        9.0,
    )

    with pytest.raises(ValueError):
        normalize_bbox_xyxy((1, 2, 1, 9))

    with pytest.raises(ValueError):
        normalize_bbox_xyxy((1, 2, 5, 2))


def test_padded_bbox_rounds_outward():
    config = AppearanceCropConfig(
        padding_x_fraction=0.1,
        padding_y_fraction=0.1,
    )

    bbox, clipped = padded_bbox_xyxy(
        (10.2, 20.4, 30.6, 60.8),
        image_width=100,
        image_height=100,
        config=config,
    )

    assert bbox == (8, 16, 33, 65)
    assert not clipped


def test_padded_bbox_clips_at_image_edge():
    config = AppearanceCropConfig(
        padding_x_fraction=0.2,
        padding_y_fraction=0.2,
    )

    bbox, clipped = padded_bbox_xyxy(
        (1.0, 2.0, 20.0, 30.0),
        image_width=50,
        image_height=50,
        config=config,
    )

    assert bbox[0] == 0
    assert bbox[1] == 0
    assert clipped


def test_crop_virtual_person_returns_copy_and_metadata():
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[20:80, 30:70] = 255

    result = crop_virtual_person(
        image,
        (30, 20, 70, 80),
        config=AppearanceCropConfig(
            padding_x_fraction=0.0,
            padding_y_fraction=0.0,
        ),
    )

    assert result.crop.shape == (60, 40, 3)
    assert result.bbox_xyxy_int == (30, 20, 70, 80)
    assert not result.clipped

    result.crop[:] = 0
    assert image[30, 40].tolist() == [255, 255, 255]


def test_crop_virtual_person_rejects_too_small_crop():
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="below"):
        crop_virtual_person(
            image,
            (10, 10, 13, 15),
            config=AppearanceCropConfig(
                padding_x_fraction=0.0,
                padding_y_fraction=0.0,
                min_crop_width_px=6,
                min_crop_height_px=12,
            ),
        )


def test_crop_virtual_person_rejects_non_array():
    with pytest.raises(TypeError):
        crop_virtual_person(
            [[0]],
            (0, 0, 1, 1),
        )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("trk-000001", "trk-000001"),
        ("det:1/2", "det_1_2"),
        (" hello world ", "hello_world"),
    ],
)
def test_safe_filename_token(raw, expected):
    assert safe_filename_token(raw) == expected


def test_safe_filename_token_rejects_empty():
    with pytest.raises(ValueError):
        safe_filename_token("  ")
