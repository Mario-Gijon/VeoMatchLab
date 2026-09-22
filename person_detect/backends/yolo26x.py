"""YOLO26x backend for the clean Veo player-detection benchmark.

Model-specific responsibility only:
    LEFT/CENTER/RIGHT BGR images -> person boxes + confidence

This backend uses the official Ultralytics Python API with the preserved local
`yolo26x.pt` checkpoint.

Geometry, field filtering, ownership, result rendering and run organization
remain in the common `person_detect/run_detection.py` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ModelDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"


class YOLO26XBackend:
    """Local-checkpoint Ultralytics YOLO26x inference wrapper.

    We keep the official predict path:
      - detection checkpoint loaded with `YOLO(...)`
      - COCO person class only (`classes=[0]`)
      - confidence threshold supplied by the common runner
      - default one-to-many detection head + Ultralytics NMS
      - 640-pixel inference target (official default)

    LEFT/CENTER/RIGHT are processed sequentially to keep VRAM usage predictable.
    """

    PERSON_CLASS_ID = 0
    INPUT_SIZE = 640
    IOU_THRESHOLD = 0.70

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | None = None,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()

        if not self.weights.exists():
            raise FileNotFoundError(self.weights)

        try:
            import torch
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "Ultralytics is not available in the active environment. "
                "Activate .venv-detection and install/update it with "
                "`pip install -U ultralytics`."
            ) from exc

        self.device = (
            device
            or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )

        self.model = YOLO(str(self.weights))

    @property
    def name(self) -> str:
        return "yolo26x"

    def info(self) -> dict:
        return {
            "backend": "ultralytics_yolo",
            "variant": "yolo26x",
            "name": self.name,
            "weights": str(self.weights),
            "device": self.device,
            "native_resolution": self.INPUT_SIZE,
            "inference_mode": "sequential_views",
            "person_class_id": self.PERSON_CLASS_ID,
            "iou_threshold": self.IOU_THRESHOLD,
            "head": "default_one_to_many_with_nms",
        }

    def _predict_one(
        self,
        image_bgr: np.ndarray,
        *,
        threshold: float,
    ) -> list[ModelDetection]:
        if (
            image_bgr is None
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError("Expected a valid BGR image")

        results = self.model.predict(
            source=image_bgr,
            conf=threshold,
            iou=self.IOU_THRESHOLD,
            imgsz=self.INPUT_SIZE,
            classes=[self.PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
            save=False,
            stream=False,
        )

        if not isinstance(results, (list, tuple)):
            results = list(results)

        if len(results) != 1:
            raise RuntimeError(
                "Expected exactly one Ultralytics Results object "
                f"for one view, got {len(results)}."
            )

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        class_ids = boxes.cls.detach().cpu().numpy()

        if (
            len(xyxy) != len(confidence)
            or len(xyxy) != len(class_ids)
        ):
            raise RuntimeError(
                "Inconsistent YOLO26x result lengths."
            )

        persons: list[ModelDetection] = []

        for box, score, class_id in zip(
            xyxy,
            confidence,
            class_ids,
            strict=True,
        ):
            # `classes=[0]` already requests only person, but keep this check
            # so backend output remains explicit and defensive.
            if int(class_id) != self.PERSON_CLASS_ID:
                continue

            x1, y1, x2, y2 = [
                float(value)
                for value in box
            ]
            score = float(score)

            if not np.isfinite(
                [x1, y1, x2, y2, score]
            ).all():
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            persons.append(
                ModelDetection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=score,
                )
            )

        return persons

    def predict_views(
        self,
        views_bgr: dict[str, np.ndarray],
        *,
        threshold: float,
    ) -> dict[str, list[ModelDetection]]:
        """Run LEFT/CENTER/RIGHT sequentially through YOLO26x."""
        order = ("left", "center", "right")

        missing = [
            name
            for name in order
            if name not in views_bgr
        ]
        if missing:
            raise ValueError(
                f"Missing rendered views: {missing}"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        return {
            view_name: self._predict_one(
                views_bgr[view_name],
                threshold=threshold,
            )
            for view_name in order
        }
