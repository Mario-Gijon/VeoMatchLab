"""RF-DETR 2XL backend for the clean Veo player-detection benchmark.

Model-specific responsibility only:
    LEFT/CENTER/RIGHT BGR images -> person boxes + confidence

Geometry, field filtering, ownership, radar rendering and clip organization
remain in the common pipeline (`person_detect/run_detection.py`).

RF-DETR 2XL is deliberately kept in its own backend file so the already
validated RF-DETR Large implementation remains reproducible and untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"


class RFDETR2XLBackend:
    """Local-checkpoint RF-DETR 2XL inference wrapper.

    The three virtual views are processed sequentially. 2XL is much heavier
    than Large (880x880 native inference), so this avoids making the benchmark
    depend on whether a three-image batch fits in GPU memory.
    """

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
            from rfdetr import RFDETR2XLarge
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR 2XL is not available in the active environment. "
                'Activate .venv-detection and install the Plus extension with '
                '`pip install "rfdetr[plus]"`.'
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = RFDETR2XLarge(
            pretrain_weights=str(self.weights),
            device=self.device,
        )

        self.model_resolution = int(
            getattr(
                getattr(self.model, "model_config", None),
                "resolution",
                880,
            )
        )

    @property
    def name(self) -> str:
        return "rfdetr_2xl"

    def info(self) -> dict:
        return {
            "backend": "rfdetr",
            "variant": "2xl",
            "name": self.name,
            "weights": str(self.weights),
            "device": self.device,
            "native_resolution": self.model_resolution,
            "inference_mode": "sequential_views",
        }

    @staticmethod
    def _class_names(detections) -> list[str]:
        """Resolve class names without hard-coding RF-DETR COCO IDs."""
        data = getattr(detections, "data", {}) or {}
        names = data.get("class_name")

        if names is not None:
            return [str(name) for name in names]

        # Compatibility fallback for older RF-DETR releases.
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES

            class_ids = np.asarray(detections.class_id).astype(int).tolist()

            if isinstance(COCO_CLASSES, dict):
                resolved = [
                    str(COCO_CLASSES.get(class_id, ""))
                    for class_id in class_ids
                ]
                if class_ids and not any(resolved):
                    ordered = [
                        name
                        for _, name in sorted(COCO_CLASSES.items())
                    ]
                    resolved = [
                        (
                            str(ordered[class_id])
                            if 0 <= class_id < len(ordered)
                            else ""
                        )
                        for class_id in class_ids
                    ]
                return resolved

            return [
                (
                    str(COCO_CLASSES[class_id])
                    if 0 <= class_id < len(COCO_CLASSES)
                    else ""
                )
                for class_id in class_ids
            ]
        except Exception as exc:
            raise RuntimeError(
                "RF-DETR predictions do not expose `data['class_name']`, "
                "and the COCO class-name fallback could not be resolved."
            ) from exc

    def _predict_one(
        self,
        image_rgb: np.ndarray,
        *,
        threshold: float,
    ):
        # Recent RF-DETR releases support include_source_image=False; retry
        # without it for compatibility with older installed versions.
        try:
            prediction = self.model.predict(
                image_rgb,
                threshold=threshold,
                include_source_image=False,
            )
        except TypeError:
            prediction = self.model.predict(
                image_rgb,
                threshold=threshold,
            )

        if isinstance(prediction, (list, tuple)):
            if len(prediction) != 1:
                raise RuntimeError(
                    "Expected exactly one RF-DETR 2XL prediction set "
                    f"for one view, got {len(prediction)}."
                )
            prediction = prediction[0]

        return prediction

    def predict_views(
        self,
        views_bgr: dict[str, np.ndarray],
        *,
        threshold: float,
    ) -> dict[str, list[ModelDetection]]:
        """Run LEFT/CENTER/RIGHT sequentially through RF-DETR 2XL."""
        order = ("left", "center", "right")

        missing = [name for name in order if name not in views_bgr]
        if missing:
            raise ValueError(f"Missing rendered views: {missing}")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        result: dict[str, list[ModelDetection]] = {}

        for view_name in order:
            image = views_bgr[view_name]

            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{view_name} is not a valid BGR image")

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            detections = self._predict_one(
                rgb,
                threshold=threshold,
            )

            names = self._class_names(detections)
            xyxy = np.asarray(
                detections.xyxy,
                dtype=np.float64,
            )
            confidence = np.asarray(
                detections.confidence,
                dtype=np.float64,
            )

            if len(names) != len(xyxy) or len(confidence) != len(xyxy):
                raise RuntimeError(
                    "Inconsistent RF-DETR 2XL output lengths for "
                    f"{view_name}"
                )

            persons: list[ModelDetection] = []

            for box, score, class_name in zip(
                xyxy,
                confidence,
                names,
                strict=True,
            ):
                if class_name.lower() != "person":
                    continue

                x1, y1, x2, y2 = [
                    float(value)
                    for value in box
                ]

                if not np.isfinite(
                    [x1, y1, x2, y2, score]
                ).all():
                    continue

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
