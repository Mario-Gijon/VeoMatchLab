"""View-agnostic detector invocation for the common benchmark runner.

This adapter deliberately owns no geometry, filtering, ownership, rendering,
or temporal behavior. It only turns an ordered dictionary of BGR view images
into the backend's existing ``ModelDetection`` lists.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from person_detect.models.rfdetr_2xl import ModelDetection as RFDETR2XLModelDetection
from person_detect.models.rfdetr_large import ModelDetection


INTERACTIVE_VIEWS = ("left", "center", "right")
TWO_LENS_VIEWS = ("left", "right")
PANORAMA_VIEWS = ("panorama",)


def _validate_images(views_bgr: Mapping[str, np.ndarray]) -> None:
    for name, image in views_bgr.items():
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"{name} is not a valid BGR image")


def _predict_two_lens_rfdetr_large(backend, views_bgr, *, threshold: float):
    """Use the loaded RF-DETR Large model for the physical lens pair."""
    if backend.name != "rfdetr_large":
        raise ValueError(
            "two_lens_view has only been validated with rfdetr_large so far; "
            f"model {backend.name!r} is not supported for this view."
        )

    images = [
        cv2.cvtColor(views_bgr[name], cv2.COLOR_BGR2RGB)
        for name in TWO_LENS_VIEWS
    ]
    try:
        predictions = backend.model.predict(
            images,
            threshold=threshold,
            include_source_image=False,
        )
    except TypeError:
        predictions = backend.model.predict(images, threshold=threshold)

    if not isinstance(predictions, (list, tuple)) or len(predictions) != len(images):
        raise RuntimeError(
            f"Expected {len(images)} RF-DETR prediction sets, "
            f"got {len(predictions) if isinstance(predictions, (list, tuple)) else 1}."
        )

    result = {}
    for view_name, detections in zip(TWO_LENS_VIEWS, predictions, strict=True):
        names = backend._class_names(detections)
        xyxy = np.asarray(detections.xyxy, dtype=np.float64)
        confidence = np.asarray(detections.confidence, dtype=np.float64)
        if len(names) != len(xyxy) or len(confidence) != len(xyxy):
            raise RuntimeError(f"Inconsistent RF-DETR output lengths for {view_name}")

        persons = []
        for box, score, class_name in zip(xyxy, confidence, names, strict=True):
            if class_name.lower() != "person":
                continue
            values = [float(value) for value in box]
            if not np.isfinite([*values, score]).all():
                continue
            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                continue
            persons.append(
                ModelDetection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=float(score),
                )
            )
        result[view_name] = persons
    return result


def _predict_two_lens_rfdetr_2xl(backend, views_bgr, *, threshold: float):
    result = {}
    for view_name in TWO_LENS_VIEWS:
        image = views_bgr[view_name]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = backend._predict_one(rgb, threshold=threshold)
        names = backend._class_names(detections)
        xyxy = np.asarray(detections.xyxy, dtype=np.float64)
        confidence = np.asarray(detections.confidence, dtype=np.float64)
        if len(names) != len(xyxy) or len(confidence) != len(xyxy):
            raise RuntimeError(f"Inconsistent RF-DETR 2XL output lengths for {view_name}")
        persons = []
        for box, score, class_name in zip(xyxy, confidence, names, strict=True):
            if class_name.lower() != "person":
                continue
            values = [float(value) for value in box]
            if not np.isfinite([*values, score]).all():
                continue
            x1, y1, x2, y2 = values
            if x2 <= x1 or y2 <= y1:
                continue
            persons.append(RFDETR2XLModelDetection((x1, y1, x2, y2), float(score)))
        result[view_name] = persons
    return result


def _predict_two_lens_dfine(backend, views_bgr, *, threshold: float):
    return {
        view_name: backend._predict_one(views_bgr[view_name], threshold=threshold)
        for view_name in TWO_LENS_VIEWS
    }


def _predict_two_lens_dfine_x_coco(backend, views_bgr, *, threshold: float):
    return _predict_two_lens_dfine(backend, views_bgr, threshold=threshold)


def _predict_two_lens_dfine_x_obj2coco(backend, views_bgr, *, threshold: float):
    return _predict_two_lens_dfine(backend, views_bgr, threshold=threshold)


def _predict_two_lens_yolo(backend, views_bgr, *, threshold: float):
    return {
        view_name: backend._predict_one(views_bgr[view_name], threshold=threshold)
        for view_name in TWO_LENS_VIEWS
    }


def _filter_rfdetr_persons(backend, detections, detection_type):
    names = backend._class_names(detections)
    xyxy = np.asarray(detections.xyxy, dtype=np.float64)
    confidence = np.asarray(detections.confidence, dtype=np.float64)
    if len(names) != len(xyxy) or len(confidence) != len(xyxy):
        raise RuntimeError("Inconsistent RF-DETR panorama output lengths")

    persons = []
    for box, score, class_name in zip(xyxy, confidence, names, strict=True):
        if class_name.lower() != "person":
            continue
        values = [float(value) for value in box]
        if not np.isfinite([*values, score]).all():
            continue
        x1, y1, x2, y2 = values
        if x2 <= x1 or y2 <= y1:
            continue
        persons.append(detection_type((x1, y1, x2, y2), float(score)))
    return persons


def _predict_panorama_rfdetr_large(backend, image_bgr, *, threshold: float):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    try:
        prediction = backend.model.predict(
            [image_rgb],
            threshold=threshold,
            include_source_image=False,
        )
    except TypeError:
        prediction = backend.model.predict([image_rgb], threshold=threshold)
    if isinstance(prediction, (list, tuple)):
        if len(prediction) != 1:
            raise RuntimeError(
                "Expected exactly one RF-DETR panorama prediction set, "
                f"got {len(prediction)}"
            )
        prediction = prediction[0]
    return _filter_rfdetr_persons(backend, prediction, ModelDetection)


def _predict_panorama_rfdetr_2xl(backend, image_bgr, *, threshold: float):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    prediction = backend._predict_one(image_rgb, threshold=threshold)
    return _filter_rfdetr_persons(backend, prediction, RFDETR2XLModelDetection)


def _predict_panorama(backend, views_bgr, *, threshold: float):
    image = views_bgr["panorama"]
    if backend.name == "rfdetr_large":
        persons = _predict_panorama_rfdetr_large(
            backend,
            image,
            threshold=threshold,
        )
    elif backend.name == "rfdetr_2xl":
        persons = _predict_panorama_rfdetr_2xl(
            backend,
            image,
            threshold=threshold,
        )
    elif backend.name in {"dfine_x_coco", "dfine_x_obj2coco", "yolo26x"}:
        persons = backend._predict_one(image, threshold=threshold)
    else:
        raise ValueError(f"Unsupported panorama_view backend: {backend.name!r}")
    return {"panorama": persons}


def predict_views(backend, views_bgr: Mapping[str, np.ndarray], *, threshold: float):
    """Run one already-loaded backend over an ordered view dictionary."""
    _validate_images(views_bgr)
    names = tuple(views_bgr)
    if names == INTERACTIVE_VIEWS:
        return backend.predict_views(dict(views_bgr), threshold=threshold)
    if names == PANORAMA_VIEWS:
        return _predict_panorama(backend, views_bgr, threshold=threshold)
    if names == TWO_LENS_VIEWS:
        if backend.name == "rfdetr_large":
            return _predict_two_lens_rfdetr_large(backend, views_bgr, threshold=threshold)
        if backend.name == "rfdetr_2xl":
            return _predict_two_lens_rfdetr_2xl(backend, views_bgr, threshold=threshold)
        if backend.name == "dfine_x_coco":
            return _predict_two_lens_dfine_x_coco(
                backend, views_bgr, threshold=threshold,
            )
        if backend.name == "dfine_x_obj2coco":
            return _predict_two_lens_dfine_x_obj2coco(
                backend, views_bgr, threshold=threshold,
            )
        if backend.name == "yolo26x":
            return _predict_two_lens_yolo(backend, views_bgr, threshold=threshold)
        raise ValueError(f"Unsupported two_lens_view backend: {backend.name!r}")
    raise ValueError(f"Unsupported detector view ordering: {names!r}")
