"""View-agnostic detector invocation for the common benchmark runner.

This adapter deliberately owns no geometry, filtering, ownership, rendering,
or temporal behavior. It only turns an ordered dictionary of BGR view images
into the backend's existing ``ModelDetection`` lists.
"""

from __future__ import annotations

from collections.abc import Mapping

import cv2
import numpy as np

from person_detect.backends.rfdetr_large import ModelDetection


INTERACTIVE_VIEWS = ("left", "center", "right")
TWO_LENS_VIEWS = ("left", "right")


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


def predict_views(backend, views_bgr: Mapping[str, np.ndarray], *, threshold: float):
    """Run one already-loaded backend over an ordered view dictionary."""
    _validate_images(views_bgr)
    names = tuple(views_bgr)
    if names == INTERACTIVE_VIEWS:
        return backend.predict_views(dict(views_bgr), threshold=threshold)
    if names == TWO_LENS_VIEWS:
        return _predict_two_lens_rfdetr_large(
            backend,
            views_bgr,
            threshold=threshold,
        )
    raise ValueError(f"Unsupported detector view ordering: {names!r}")
