from __future__ import annotations

from experiments.tracking.run_v1_jersey_numberbox_roboflow import (
    _extract_boxes,
    _prediction_box,
)


def test_prediction_box_converts_center_xywh():
    raw = {
        "x": 50,
        "y": 40,
        "width": 20,
        "height": 10,
        "confidence": 0.9,
        "class": "numberBox",
        "class_id": 0,
    }

    box = _prediction_box(
        raw,
        image_width=100,
        image_height=80,
    )

    assert box is not None
    assert box["xyxy"] == [40.0, 35.0, 60.0, 45.0]
    assert box["confidence"] == 0.9


def test_prediction_box_clamps_to_image():
    raw = {
        "x": 5,
        "y": 5,
        "width": 20,
        "height": 20,
        "confidence": 0.8,
        "class": "numberBox",
    }

    box = _prediction_box(
        raw,
        image_width=100,
        image_height=80,
    )

    assert box is not None
    assert box["xyxy"] == [0.0, 0.0, 15.0, 15.0]


def test_prediction_box_rejects_invalid_geometry():
    assert _prediction_box(
        {
            "x": 1,
            "y": 1,
            "width": 0,
            "height": 4,
            "confidence": 0.5,
        },
        image_width=10,
        image_height=10,
    ) is None


def test_extract_boxes_filters_class_and_sorts():
    result = {
        "predictions": [
            {
                "x": 20,
                "y": 20,
                "width": 10,
                "height": 10,
                "confidence": 0.4,
                "class": "numberBox",
            },
            {
                "x": 30,
                "y": 30,
                "width": 10,
                "height": 10,
                "confidence": 0.9,
                "class": "other",
            },
            {
                "x": 40,
                "y": 40,
                "width": 10,
                "height": 10,
                "confidence": 0.8,
                "class": "numberBox",
            },
        ]
    }

    boxes = _extract_boxes(
        result,
        image_width=100,
        image_height=100,
        class_name="numberBox",
    )

    assert len(boxes) == 2
    assert [box["confidence"] for box in boxes] == [0.8, 0.4]


def test_extract_boxes_handles_missing_predictions():
    assert _extract_boxes(
        {},
        image_width=100,
        image_height=100,
        class_name="numberBox",
    ) == []
