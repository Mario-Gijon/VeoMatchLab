"""ReID backend contract and model-independent embedding helpers.

The clean Tracking V1 appearance stage must not depend on one concrete ReID
implementation.  A backend only has to turn BGR player crops into a fixed-size
numeric embedding.  This module validates and L2-normalizes that output before
it is admitted into the appearance contracts.

Concrete neural backends (PRTReID, KPR, etc.) can therefore live behind the
same interface without changing the benchmark, appearance artifact, or global
identity solver.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, Sequence, runtime_checkable

import cv2
import numpy as np


@runtime_checkable
class ReIDBackend(Protocol):
    """Minimal interface implemented by every ReID embedding backend."""

    @property
    def model_id(self) -> str:
        """Stable identifier written into appearance descriptors."""

    @property
    def embedding_dim(self) -> int:
        """Number of scalar values produced for one crop."""

    def embed_batch(
        self,
        images_bgr: Sequence[np.ndarray],
    ) -> np.ndarray:
        """Return an ``N x D`` floating-point matrix."""


def _validate_model_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("model_id must be a non-empty string")
    return value.strip()


def l2_normalize_vector(
    values: Sequence[float] | np.ndarray,
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("embedding must be a non-empty 1D vector")
    if not np.isfinite(array).all():
        raise ValueError("embedding contains non-finite values")

    norm = float(np.linalg.norm(array))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding has zero or invalid L2 norm")

    return array / norm


def validate_embedding_batch(
    values: np.ndarray,
    *,
    expected_count: int,
    expected_dim: int,
) -> np.ndarray:
    """Validate a backend matrix and return L2-normalized float32 rows."""
    if expected_count < 0:
        raise ValueError("expected_count must be >= 0")
    if expected_dim <= 0:
        raise ValueError("expected_dim must be > 0")

    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("backend output must have shape N x D")
    if matrix.shape != (expected_count, expected_dim):
        raise ValueError(
            "backend output shape mismatch: "
            f"got {matrix.shape}, expected {(expected_count, expected_dim)}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("backend output contains non-finite values")

    if expected_count == 0:
        return matrix.copy()

    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms <= 1e-12):
        raise ValueError("backend output contains zero-norm embeddings")

    return matrix / norms[:, None]


def embed_validated(
    backend: ReIDBackend,
    images_bgr: Sequence[np.ndarray],
) -> np.ndarray:
    """Run a backend and enforce the Tracking V1 embedding contract."""
    _validate_model_id(backend.model_id)

    dimension = int(backend.embedding_dim)
    if dimension <= 0:
        raise ValueError("backend embedding_dim must be > 0")

    images = tuple(images_bgr)
    for index, image in enumerate(images):
        if not isinstance(image, np.ndarray):
            raise TypeError(f"images_bgr[{index}] must be a numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"images_bgr[{index}] must have shape HxWx3"
            )
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(
                f"images_bgr[{index}] must be non-empty"
            )

    raw = backend.embed_batch(images)
    return validate_embedding_batch(
        raw,
        expected_count=len(images),
        expected_dim=dimension,
    )


def cosine_similarity(
    first: Sequence[float] | np.ndarray,
    second: Sequence[float] | np.ndarray,
) -> float:
    a = l2_normalize_vector(first)
    b = l2_normalize_vector(second)
    if a.shape != b.shape:
        raise ValueError(
            f"embedding dimensions differ: {a.shape} vs {b.shape}"
        )
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def quality_weighted_mean(
    embeddings: Sequence[Sequence[float] | np.ndarray],
    qualities: Sequence[float],
) -> np.ndarray:
    """Quality-weighted aggregate followed by L2 normalization."""
    if len(embeddings) == 0:
        raise ValueError("at least one embedding is required")
    if len(embeddings) != len(qualities):
        raise ValueError(
            "embeddings and qualities must have the same length"
        )

    normalized = [
        l2_normalize_vector(item)
        for item in embeddings
    ]
    dimension = normalized[0].shape
    if any(item.shape != dimension for item in normalized):
        raise ValueError("all embeddings must have the same dimension")

    weights = np.asarray(qualities, dtype=np.float32)
    if not np.isfinite(weights).all():
        raise ValueError("qualities contain non-finite values")
    if np.any(weights < 0.0):
        raise ValueError("qualities must be >= 0")
    if float(weights.sum()) <= 1e-12:
        raise ValueError("at least one quality must be > 0")

    matrix = np.stack(normalized, axis=0)
    aggregate = np.sum(
        matrix * weights[:, None],
        axis=0,
    ) / float(weights.sum())

    return l2_normalize_vector(aggregate)


@dataclass(frozen=True, slots=True)
class HSVHistogramReIDBackend:
    """Cheap diagnostic baseline, not the final football ReID model.

    It describes the central torso region with a 3D HSV histogram after
    suppressing likely pitch-green pixels.  The backend is useful because it
    exercises the exact same benchmark API as a neural ReID model while making
    no extra ML dependency assumptions.

    Expected failure mode is informative: different players from the same team
    should often look very similar.  PRTReID/KPR must improve on this baseline.
    """

    h_bins: int = 8
    s_bins: int = 4
    v_bins: int = 4

    @property
    def model_id(self) -> str:
        return "hsv-torso-hist-v1"

    @property
    def embedding_dim(self) -> int:
        return self.h_bins * self.s_bins * self.v_bins

    def __post_init__(self) -> None:
        for name, value in (
            ("h_bins", self.h_bins),
            ("s_bins", self.s_bins),
            ("v_bins", self.v_bins),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
            ):
                raise ValueError(f"{name} must be an integer > 0")

    @staticmethod
    def _torso(image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]

        x1 = int(round(width * 0.12))
        x2 = int(round(width * 0.88))
        y1 = int(round(height * 0.08))
        y2 = int(round(height * 0.72))

        x1 = max(0, min(width - 1, x1))
        x2 = max(x1 + 1, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(y1 + 1, min(height, y2))

        return image[y1:y2, x1:x2]

    def _embedding(self, image: np.ndarray) -> np.ndarray:
        torso = self._torso(image)
        hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

        # OpenCV hue is [0, 179].  Suppress saturated green pixels so the
        # baseline mostly measures the player rather than the pitch.
        hue = hsv[..., 0]
        saturation = hsv[..., 1]
        likely_grass = (
            (hue >= 30)
            & (hue <= 95)
            & (saturation >= 45)
        )
        mask = (~likely_grass).astype(np.uint8) * 255

        usable_fraction = float(np.mean(mask > 0))
        if usable_fraction < 0.15:
            mask = None

        hist = cv2.calcHist(
            [hsv],
            [0, 1, 2],
            mask,
            [self.h_bins, self.s_bins, self.v_bins],
            [0, 180, 0, 256, 0, 256],
        )
        vector = hist.reshape(-1).astype(np.float32)

        # A pathological all-zero masked histogram should never escape the
        # backend. Fallback to the unmasked torso before failing.
        if float(np.linalg.norm(vector)) <= 1e-12:
            hist = cv2.calcHist(
                [hsv],
                [0, 1, 2],
                None,
                [self.h_bins, self.s_bins, self.v_bins],
                [0, 180, 0, 256, 0, 256],
            )
            vector = hist.reshape(-1).astype(np.float32)

        return l2_normalize_vector(vector)

    def embed_batch(
        self,
        images_bgr: Sequence[np.ndarray],
    ) -> np.ndarray:
        if not images_bgr:
            return np.empty(
                (0, self.embedding_dim),
                dtype=np.float32,
            )

        return np.stack(
            [self._embedding(image) for image in images_bgr],
            axis=0,
        )


def create_reid_backend(name: str) -> ReIDBackend:
    """Small registry intentionally ready for future neural backends."""
    normalized = name.strip().lower().replace("_", "-")

    if normalized in {
        "hsv",
        "hsv-hist",
        "hsv-torso-hist",
        "hsv-torso-hist-v1",
    }:
        return HSVHistogramReIDBackend()

    raise ValueError(
        f"unknown ReID backend {name!r}; available: hsv-hist"
    )
