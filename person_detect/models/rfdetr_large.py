"""RF-DETR backend for the clean Veo player-detection benchmark.

The backend has one job:
    LEFT/CENTER/RIGHT RGB images -> person boxes + confidence

It does NOT know about field geometry, ownership, radar rendering, or tracking.
Those remain common project code so every model is compared on the same basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"


class RFDETRBackend:
    """Local-checkpoint RF-DETR inference wrapper.

    V1 supports RF-DETR Large. The interface is intentionally already generic
    enough to add 2XL without changing the common runner.
    """

    def __init__(
        self,
        weights: str | Path,
        *,
        variant: str = "large",
        device: str | None = None,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        self.variant = variant.lower().strip()

        if not self.weights.exists():
            raise FileNotFoundError(self.weights)

        if self.variant != "large":
            raise ValueError(
                "This first backend version supports variant='large' only. "
                "2XL will be added after Large is validated end-to-end."
            )

        try:
            import torch
            from rfdetr import RFDETRLarge
        except ImportError as exc:
            raise RuntimeError(
                "RF-DETR is not available in the active environment. "
                "Activate .venv-detection and ensure the `rfdetr` package is installed."
            ) from exc

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Explicit path means RF-DETR uses our preserved local checkpoint rather
        # than downloading another copy into its model cache.
        self.model = RFDETRLarge(
            pretrain_weights=str(self.weights),
            device=self.device,
        )

        self.model_resolution = int(
            getattr(getattr(self.model, "model_config", None), "resolution", 704)
        )

    @property
    def name(self) -> str:
        return "rfdetr_large"

    def info(self) -> dict:
        return {
            "backend": "rfdetr",
            "variant": self.variant,
            "name": self.name,
            "weights": str(self.weights),
            "device": self.device,
            "native_resolution": self.model_resolution,
        }

    @staticmethod
    def _class_names(detections) -> list[str]:
        """Resolve names without hard-coding RF-DETR class-id conventions."""
        data = getattr(detections, "data", {}) or {}
        names = data.get("class_name")

        if names is not None:
            return [str(name) for name in names]

        # Compatibility fallback for older RF-DETR releases.
        try:
            from rfdetr.assets.coco_classes import COCO_CLASSES

            class_ids = np.asarray(detections.class_id).astype(int).tolist()

            if isinstance(COCO_CLASSES, dict):
                resolved = [str(COCO_CLASSES.get(cid, "")) for cid in class_ids]
                # Some older releases emitted active-first 0-based ids.
                if class_ids and not any(resolved):
                    ordered = [name for _, name in sorted(COCO_CLASSES.items())]
                    resolved = [
                        str(ordered[cid]) if 0 <= cid < len(ordered) else ""
                        for cid in class_ids
                    ]
                return resolved

            return [
                str(COCO_CLASSES[cid]) if 0 <= cid < len(COCO_CLASSES) else ""
                for cid in class_ids
            ]
        except Exception as exc:
            raise RuntimeError(
                "RF-DETR predictions do not expose `data['class_name']`, "
                "and COCO class-name fallback could not be resolved."
            ) from exc

    def predict_views(
        self,
        views_bgr: dict[str, np.ndarray],
        *,
        threshold: float,
    ) -> dict[str, list[ModelDetection]]:
        """Run the three virtual views as one RF-DETR batch."""
        order = ("left", "center", "right")

        missing = [name for name in order if name not in views_bgr]
        if missing:
            raise ValueError(f"Missing rendered views: {missing}")

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        rgb_images = []
        for name in order:
            image = views_bgr[name]
            if image is None or image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"{name} is not a valid BGR image")
            rgb_images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # include_source_image=False avoids retaining three 2048x1152 images
        # inside Supervision metadata for every frame. Older releases may not
        # expose this keyword, so retry compatibly if necessary.
        try:
            predictions = self.model.predict(
                rgb_images,
                threshold=threshold,
                include_source_image=False,
            )
        except TypeError:
            predictions = self.model.predict(
                rgb_images,
                threshold=threshold,
            )

        if not isinstance(predictions, (list, tuple)):
            predictions = [predictions]

        if len(predictions) != 3:
            raise RuntimeError(
                f"Expected 3 RF-DETR prediction sets, got {len(predictions)}"
            )

        result: dict[str, list[ModelDetection]] = {}

        for view_name, detections in zip(order, predictions, strict=True):
            names = self._class_names(detections)
            xyxy = np.asarray(detections.xyxy, dtype=np.float64)
            confidence = np.asarray(detections.confidence, dtype=np.float64)

            if len(names) != len(xyxy) or len(confidence) != len(xyxy):
                raise RuntimeError(
                    f"Inconsistent RF-DETR output lengths for {view_name}"
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

                x1, y1, x2, y2 = [float(value) for value in box]
                if not np.isfinite([x1, y1, x2, y2, score]).all():
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
