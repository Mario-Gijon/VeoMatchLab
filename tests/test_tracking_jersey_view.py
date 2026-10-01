from __future__ import annotations

import cv2
import numpy as np

from tracking.appearance.jersey_view import (
    JerseyViewConfig,
    JerseyViewObservation,
    RejectedJerseyView,
    crop_jersey_roi,
    score_jersey_view,
    select_jersey_views,
)


def _jersey_crop(
    *,
    text="17",
    height=140,
    width=70,
    shirt_bgr=(30, 30, 210),
    text_bgr=(245, 245, 245),
):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (0, 130, 0)

    cv2.rectangle(
        image,
        (int(width * 0.15), int(height * 0.07)),
        (int(width * 0.85), int(height * 0.70)),
        shirt_bgr,
        thickness=-1,
    )

    if text:
        scale = max(0.45, height / 145.0)
        cv2.putText(
            image,
            text,
            (int(width * 0.25), int(height * 0.48)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            text_bgr,
            max(1, int(round(height / 75))),
            cv2.LINE_AA,
        )
    return image


def _score(image, *, tracklet="a", detection="a1", time=0.0):
    return score_jersey_view(
        image,
        tracklet_id=tracklet,
        detection_id=detection,
        frame_index=int(round(time * 10)),
        timestamp_s=time,
        crop_ref=f"{detection}.jpg",
        selection_quality=0.9,
        reid_quality=0.9,
    )


def test_number_like_high_resolution_crop_scores_above_blurred_crop():
    clear = _score(_jersey_crop())
    blurred_image = cv2.GaussianBlur(
        _jersey_crop(),
        (19, 19),
        7.0,
    )
    blurred = _score(
        blurred_image,
        detection="a2",
    )

    assert isinstance(clear, JerseyViewObservation)
    assert isinstance(blurred, JerseyViewObservation)
    assert clear.readiness_score > blurred.readiness_score
    assert clear.sharpness_score > blurred.sharpness_score


def test_flat_shirt_has_less_structure_than_printed_number():
    printed = _score(_jersey_crop(text="17"))
    flat = _score(
        _jersey_crop(text=""),
        detection="a2",
    )

    assert isinstance(printed, JerseyViewObservation)
    assert isinstance(flat, JerseyViewObservation)
    assert printed.edge_structure_score >= flat.edge_structure_score
    assert printed.central_structure_score >= flat.central_structure_score


def test_tiny_crop_is_rejected():
    result = _score(
        _jersey_crop(height=20, width=6),
    )

    assert isinstance(result, RejectedJerseyView)
    assert "crop_height_below_min" in result.reasons
    assert "crop_width_below_min" in result.reasons


def test_green_goalkeeper_is_not_removed_by_colour_rule():
    result = _score(
        _jersey_crop(
            shirt_bgr=(20, 180, 20),
            text_bgr=(250, 250, 250),
        )
    )

    assert isinstance(result, JerseyViewObservation)
    payload = result.to_dict()
    assert payload["jersey_number"] is None
    assert payload["orientation"] == "unknown"


def test_roi_is_central_shirt_region():
    image = _jersey_crop()
    result = _score(image)

    assert isinstance(result, JerseyViewObservation)
    roi = crop_jersey_roi(image, result.roi_xyxy)

    assert roi.shape[0] < image.shape[0]
    assert roi.shape[1] < image.shape[1]
    assert roi.shape[0] >= 16
    assert roi.shape[1] >= 8


def test_selection_is_temporally_diverse_and_limited():
    config = JerseyViewConfig(
        max_selected_per_tracklet=2,
        min_selected_score=0.0,
        min_temporal_spacing_s=0.5,
    )
    observations = []
    for index, time in enumerate((0.0, 0.1, 0.8, 1.5)):
        result = score_jersey_view(
            _jersey_crop(),
            tracklet_id="a",
            detection_id=f"a{index}",
            frame_index=index,
            timestamp_s=time,
            crop_ref=f"a{index}.jpg",
            selection_quality=0.9 - 0.05 * index,
            reid_quality=0.9,
            config=config,
        )
        assert isinstance(result, JerseyViewObservation)
        observations.append(result)

    selected = select_jersey_views(
        observations,
        config,
    )

    assert len(selected) == 2
    times = [
        item.observation.timestamp_s
        for item in selected
    ]
    assert abs(times[0] - times[1]) >= 0.5


def test_view_stage_never_assigns_number_or_orientation():
    result = _score(_jersey_crop(text="99"))

    assert isinstance(result, JerseyViewObservation)
    payload = result.to_dict()
    assert payload["jersey_number"] is None
    assert payload["recognition_confidence"] is None
    assert payload["orientation"] == "unknown"
