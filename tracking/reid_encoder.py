"""Sprint C.1: OSNet-based person ReID feature extraction.

This module is deliberately small and model-agnostic at the call boundary:
it accepts rectified BGR crops from Sprint A and returns L2-normalized feature
vectors.  No identity fusion happens here.

The default backend is Torchreid's official FeatureExtractor.  A valid ReID
checkpoint path is REQUIRED; we never silently fall back to ImageNet-only or
random weights for this experiment.

Important:
  * Sprint A crops are OpenCV BGR.
  * Torchreid's numpy input path expects image arrays that become PIL images,
    i.e. RGB channel order.
  * We therefore convert BGR -> RGB explicitly before inference.
  * Returned features are normalized here because Sprint C uses cosine
    similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import cv2
import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ReIDEncoderConfig:
    model_name: str = "osnet_ain_x1_0"
    device: str = "cuda"
    batch_size: int = 64
    image_height: int = 256
    image_width: int = 128
    verbose: bool = False

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must not be empty")
        if not self.device:
            raise ValueError("device must not be empty")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.image_height <= 0 or self.image_width <= 0:
            raise ValueError("image size must be positive")


class TorchreidEncoder:
    """Thin, testable wrapper around torchreid.utils.FeatureExtractor."""

    def __init__(
        self,
        model_path: str | Path | None,
        *,
        config: ReIDEncoderConfig | None = None,
        extractor: Callable[[Any], Any] | None = None,
    ) -> None:
        self.config = config or ReIDEncoderConfig()

        self.model_path = (
            Path(model_path).expanduser().resolve()
            if model_path is not None
            else None
        )

        if extractor is None:
            if self.model_path is None or not self.model_path.is_file():
                raise FileNotFoundError(
                    "A valid ReID checkpoint is required. "
                    f"Got: {self.model_path}"
                )

            try:
                from torchreid.utils import FeatureExtractor
            except Exception as exc:
                raise RuntimeError(
                    "Torchreid is not importable. Install the official "
                    "KaiyangZhou/deep-person-reid package in this environment."
                ) from exc

            extractor = FeatureExtractor(
                model_name=self.config.model_name,
                model_path=str(self.model_path),
                image_size=(
                    self.config.image_height,
                    self.config.image_width,
                ),
                device=self.config.device,
                verbose=self.config.verbose,
            )

        self._extractor = extractor

    def encode_bgr(
        self,
        images_bgr: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Encode OpenCV BGR crops into L2-normalized ReID embeddings."""

        if not images_bgr:
            return np.empty((0, 0), dtype=np.float32)

        prepared: list[np.ndarray] = []
        for image in images_bgr:
            prepared.append(
                self._prepare_bgr(image)
            )

        batches: list[np.ndarray] = []

        for start in range(
            0,
            len(prepared),
            self.config.batch_size,
        ):
            batch = prepared[
                start:start + self.config.batch_size
            ]
            raw = self._extractor(batch)
            array = _to_numpy_2d(raw)
            batches.append(array)

        features = np.concatenate(
            batches,
            axis=0,
        ).astype(np.float32, copy=False)

        if features.shape[0] != len(prepared):
            raise RuntimeError(
                "ReID backend returned a different number of features "
                f"({features.shape[0]}) than input images ({len(prepared)})"
            )

        return l2_normalize_rows(features)

    def encode_one_bgr(
        self,
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        rows = self.encode_bgr([image_bgr])
        if rows.shape[0] != 1:
            raise RuntimeError("expected exactly one feature row")
        return rows[0]

    @staticmethod
    def _prepare_bgr(
        image_bgr: np.ndarray,
    ) -> np.ndarray:
        if not isinstance(image_bgr, np.ndarray):
            raise TypeError("image must be a numpy.ndarray")
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError(
                "ReID image must have shape HxWx3"
            )
        if image_bgr.size == 0:
            raise ValueError("ReID image must not be empty")
        if image_bgr.dtype != np.uint8:
            if np.issubdtype(image_bgr.dtype, np.floating):
                image_bgr = np.clip(
                    image_bgr,
                    0,
                    255,
                ).astype(np.uint8)
            else:
                image_bgr = image_bgr.astype(np.uint8)

        return cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )


def l2_normalize_rows(
    features: np.ndarray,
) -> np.ndarray:
    array = np.asarray(
        features,
        dtype=np.float32,
    )

    if array.ndim != 2:
        raise ValueError(
            "features must have shape (N, D)"
        )
    if not np.isfinite(array).all():
        raise ValueError(
            "features contain non-finite values"
        )

    norms = np.linalg.norm(
        array,
        axis=1,
        keepdims=True,
    )

    if np.any(norms <= EPS):
        raise ValueError(
            "ReID backend produced a zero-norm embedding"
        )

    return array / norms


def _to_numpy_2d(
    value: Any,
) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    array = np.asarray(
        value,
        dtype=np.float32,
    )

    if array.ndim == 1:
        array = array[None, :]

    if array.ndim != 2:
        raise RuntimeError(
            "ReID backend output must have shape (N, D); "
            f"got {array.shape}"
        )

    return array
