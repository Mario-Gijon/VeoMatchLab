from __future__ import annotations

import cv2
import numpy as np
import pytest

from tracking.appearance.kit import (
    KIT_FEATURE_DIM,
    KitEvidenceConfig,
    KitRejectedSample,
    KitSample,
    build_kit_tracklet_profiles,
    compare_kit_profiles,
    extract_kit_sample,
)


def _crop(
    *,
    torso_bgr=(0, 0, 255),
    shorts_bgr=(255, 0, 0),
    background_bgr=(0, 150, 0),
    height=120,
    width=50,
):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = np.asarray(background_bgr, dtype=np.uint8)

    # Fill generous player regions so the configured torso/shorts ROIs contain
    # kit pixels plus some grass at their edges.
    cv2.rectangle(
        image,
        (int(width * 0.18), int(height * 0.10)),
        (int(width * 0.82), int(height * 0.60)),
        torso_bgr,
        thickness=-1,
    )
    cv2.rectangle(
        image,
        (int(width * 0.24), int(height * 0.54)),
        (int(width * 0.76), int(height * 0.88)),
        shorts_bgr,
        thickness=-1,
    )
    return image


def _sample(
    tracklet_id: str,
    detection_id: str,
    image,
):
    result = extract_kit_sample(
        image,
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        frame_index=0,
        timestamp_s=0.0,
        crop_ref=f"{detection_id}.jpg",
        selection_quality=0.9,
        reid_quality=0.9,
    )
    assert isinstance(result, KitSample)
    return result


def test_signature_has_expected_dimension_and_is_unit_norm():
    sample = _sample("a", "a1", _crop())

    assert len(sample.signature) == KIT_FEATURE_DIM
    assert np.linalg.norm(sample.signature) == pytest.approx(1.0)


def test_same_kit_is_more_similar_than_different_kit():
    red_blue_a = _sample(
        "a",
        "a1",
        _crop(
            torso_bgr=(0, 0, 255),
            shorts_bgr=(255, 0, 0),
        ),
    )
    red_blue_b = _sample(
        "b",
        "b1",
        _crop(
            torso_bgr=(0, 0, 240),
            shorts_bgr=(240, 0, 0),
        ),
    )
    yellow_black = _sample(
        "c",
        "c1",
        _crop(
            torso_bgr=(0, 255, 255),
            shorts_bgr=(20, 20, 20),
        ),
    )

    profiles = build_kit_tracklet_profiles(
        [red_blue_a, red_blue_b, yellow_black]
    )
    by_id = {item.tracklet_id: item for item in profiles}

    same = compare_kit_profiles(by_id["a"], by_id["b"])
    different = compare_kit_profiles(by_id["a"], by_id["c"])

    assert same.kit_cosine > different.kit_cosine


def test_green_background_is_suppressed():
    sample = _sample(
        "a",
        "a1",
        _crop(background_bgr=(0, 180, 0)),
    )

    assert sample.torso_non_grass_fraction < 1.0
    assert sample.shorts_non_grass_fraction < 1.0
    assert sample.torso_non_grass_fraction > 0.1


def test_tiny_crop_is_rejected_not_forced_into_profile():
    result = extract_kit_sample(
        _crop(height=20, width=6),
        tracklet_id="a",
        detection_id="a1",
        frame_index=0,
        timestamp_s=0.0,
        crop_ref="a1.jpg",
        selection_quality=0.9,
    )

    assert isinstance(result, KitRejectedSample)
    assert "crop_height_below_min" in result.reasons
    assert "crop_width_below_min" in result.reasons


def test_multiple_samples_are_aggregated_per_tracklet():
    first = _sample(
        "a",
        "a1",
        _crop(torso_bgr=(0, 0, 255)),
    )
    second_raw = extract_kit_sample(
        _crop(torso_bgr=(0, 0, 245)),
        tracklet_id="a",
        detection_id="a2",
        frame_index=10,
        timestamp_s=1.0,
        crop_ref="a2.jpg",
        selection_quality=0.8,
        reid_quality=0.8,
    )
    assert isinstance(second_raw, KitSample)

    profiles = build_kit_tracklet_profiles([first, second_raw])

    assert len(profiles) == 1
    assert profiles[0].sample_count == 2
    assert profiles[0].effective_sample_count > 1.0
    assert profiles[0].internal_similarity["mean"] > 0.9


def test_pair_evidence_is_soft_and_assigns_no_team():
    first = _sample("a", "a1", _crop())
    second = _sample("b", "b1", _crop())
    profiles = build_kit_tracklet_profiles([first, second])

    evidence = compare_kit_profiles(profiles[0], profiles[1])
    payload = evidence.to_dict()

    assert payload["same_team_decision"] is None
    assert payload["identity_decision"] is None
    assert 0.0 <= evidence.reliability <= 1.0


def test_black_kit_is_not_rejected_as_low_saturation():
    black = _sample(
        "a",
        "a1",
        _crop(
            torso_bgr=(20, 20, 20),
            shorts_bgr=(30, 30, 30),
        ),
    )

    assert isinstance(black, KitSample)
