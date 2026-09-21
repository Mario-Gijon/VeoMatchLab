import numpy as np

from experiments.player_detection.run_interval import (
    Sample,
    bbox_bottom_center,
    build_sampling_plan,
    field_people,
    lens_footpoint_to_raw,
    person_predictions,
    raw_detection_record,
    split_dual_lens,
)


class FakeDetections:
    def __init__(self, class_ids, class_names):
        self.xyxy = np.asarray([[1, 2, 101, 202]] * len(class_ids), dtype=float)
        self.confidence = np.asarray([.9] * len(class_ids), dtype=float)
        self.class_id = np.asarray(class_ids)
        self.data = {"class_name": np.asarray(class_names, dtype=object)}


class FakeModel:
    def __init__(self, detections):
        self.detections = detections
        self.seen_image = None

    def predict(self, image, *, threshold):
        self.seen_image = image
        return self.detections


def test_split_stacked_dual_lens_preserves_native_lens_shapes():
    frame = np.zeros((2048, 2048, 3), dtype=np.uint8)
    frame[:1024] = 11
    frame[1024:] = 22
    left, right = split_dual_lens(frame)
    assert left.shape == right.shape == (1024, 2048, 3)
    assert left[0, 0, 0] == 11
    assert right[0, 0, 0] == 22


def test_bbox_bottom_center_and_lens_to_raw_coordinates():
    assert bbox_bottom_center([10, 20, 30, 50]) == (20.0, 50.0)
    assert lens_footpoint_to_raw("left", 20, 50) == (20, 50)
    assert lens_footpoint_to_raw("right", 20, 50) == (20, 1074)


def test_raw_detection_preserves_native_bbox_coordinates():
    record = raw_detection_record(
        timestamp_s=12.5,
        frame_index=375,
        lens="right",
        confidence=.9,
        bbox_xyxy=[1, 2, 101, 202],
        model="large",
    )
    assert record["bbox_xyxy"] == [1.0, 2.0, 101.0, 202.0]
    assert record["footpoint_native"] == [51.0, 202.0]
    assert record["raw_footpoint"] == [51.0, 1226.0]


def test_rfdetr_class_name_metadata_filters_person_and_preserves_class_id():
    image_bgr = np.zeros((8, 12, 3), dtype=np.uint8)
    detections = FakeDetections([1, 2, 3, 4], ["person", "tennis racket", "sports ball", "kite"])
    model = FakeModel(detections)
    accepted = person_predictions(model, image_bgr, .1, {99})
    assert len(accepted) == 1
    assert accepted[0][2:] == (1, "person")


def test_detector_input_is_rgb_with_original_dimensions():
    image_bgr = np.asarray([[[10, 20, 30]]], dtype=np.uint8)
    model = FakeModel(FakeDetections([1], ["person"]))
    person_predictions(model, image_bgr, .1, set())
    assert model.seen_image.shape == image_bgr.shape
    assert model.seen_image[0, 0].tolist() == [30, 20, 10]


def test_field_people_filters_only_strict_inside_field_results():
    records = [{"inside_field": True}, {"inside_field": False}, {"inside_field": None}]
    assert field_people(records) == [{"inside_field": True}]


def test_sampling_plan_is_deterministic_and_uses_source_frame_indices():
    plan = build_sampling_plan(10, 3, 1, 30)
    assert plan == [Sample(10.0, 300), Sample(11.0, 330), Sample(12.0, 360)]
    assert build_sampling_plan(10, 3, 1, 30) == plan
