"""Actual crop-quality scoring for Tracking V1 ReID evidence.

This module intentionally runs *after* a visual crop exists.  It answers a
different question than appearance moment selection:

    selection_quality:
        "Did this observation look promising before reading its pixels?"

    reid_quality:
        "Are these actual crop pixels suitable as identity evidence?"

A crop can therefore be excellent for geometry/tracking while being rejected
for ReID.  Nothing here assigns a player identity, team or shirt number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import cv2
import numpy as np


def _finite(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _non_negative(name: str, value: float) -> float:
    parsed = _finite(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _unit_interval(name: str, value: float) -> float:
    parsed = _finite(name, value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ramp(value: float, *, bad: float, good: float) -> float:
    if not good > bad:
        raise ValueError("good threshold must be > bad threshold")
    return _clamp01((float(value) - bad) / (good - bad))


def _inverse_ramp(value: float, *, good: float, bad: float) -> float:
    if not bad > good:
        raise ValueError("bad threshold must be > good threshold")
    return 1.0 - _clamp01((float(value) - good) / (bad - good))


def _bbox_size(
    bbox_xyxy: Sequence[float],
) -> tuple[float, float]:
    if isinstance(bbox_xyxy, (str, bytes)) or len(bbox_xyxy) != 4:
        raise ValueError("bbox_xyxy must contain four values")

    x1, y1, x2, y2 = (
        _finite(f"bbox_xyxy[{index}]", value)
        for index, value in enumerate(bbox_xyxy)
    )
    width = x2 - x1
    height = y2 - y1
    if width <= 0.0 or height <= 0.0:
        raise ValueError("bbox_xyxy must have positive area")
    return width, height


@dataclass(frozen=True, slots=True)
class ReIDCropQualityConfig:
    """First empirical quality policy for the current Veo crop scale.

    Thresholds are diagnostic starting points, not model truth.  They are
    intentionally exposed in the output so they can be recalibrated after
    visual inspection or once a concrete ReID model is selected.
    """

    hard_min_bbox_height_px: float = 48.0
    hard_min_bbox_width_px: float = 12.0
    min_reid_quality: float = 0.55

    resolution_height_bad_px: float = 40.0
    resolution_height_good_px: float = 96.0
    resolution_width_bad_px: float = 10.0
    resolution_width_good_px: float = 40.0

    sharpness_bad: float = 12.0
    sharpness_good: float = 80.0

    contrast_bad: float = 12.0
    contrast_good: float = 45.0

    black_fraction_good: float = 0.02
    black_fraction_bad: float = 0.30

    resolution_weight: float = 0.45
    sharpness_weight: float = 0.25
    contrast_weight: float = 0.10
    selection_weight: float = 0.15
    clean_border_weight: float = 0.05

    weak_threshold: float = 0.70
    good_threshold: float = 0.85

    def __post_init__(self) -> None:
        for name in (
            "hard_min_bbox_height_px",
            "hard_min_bbox_width_px",
            "resolution_height_bad_px",
            "resolution_height_good_px",
            "resolution_width_bad_px",
            "resolution_width_good_px",
            "sharpness_bad",
            "sharpness_good",
            "contrast_bad",
            "contrast_good",
            "black_fraction_good",
            "black_fraction_bad",
        ):
            _non_negative(name, getattr(self, name))

        for name in (
            "min_reid_quality",
            "weak_threshold",
            "good_threshold",
        ):
            _unit_interval(name, getattr(self, name))

        if not self.resolution_height_good_px > self.resolution_height_bad_px:
            raise ValueError(
                "resolution_height_good_px must exceed bad threshold"
            )
        if not self.resolution_width_good_px > self.resolution_width_bad_px:
            raise ValueError(
                "resolution_width_good_px must exceed bad threshold"
            )
        if not self.sharpness_good > self.sharpness_bad:
            raise ValueError("sharpness_good must exceed sharpness_bad")
        if not self.contrast_good > self.contrast_bad:
            raise ValueError("contrast_good must exceed contrast_bad")
        if not self.black_fraction_bad > self.black_fraction_good:
            raise ValueError(
                "black_fraction_bad must exceed black_fraction_good"
            )
        if not self.good_threshold > self.weak_threshold:
            raise ValueError(
                "good_threshold must exceed weak_threshold"
            )

        weights = (
            self.resolution_weight,
            self.sharpness_weight,
            self.contrast_weight,
            self.selection_weight,
            self.clean_border_weight,
        )
        for index, value in enumerate(weights):
            _non_negative(f"weight[{index}]", value)
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("ReID crop-quality weights must sum to 1.0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReIDCropQuality:
    selection_quality: float
    reid_quality: float
    reid_eligible: bool
    tier: str

    bbox_width_px: float
    bbox_height_px: float
    crop_width_px: int
    crop_height_px: int

    laplacian_variance: float
    grayscale_std: float
    black_fraction: float

    resolution_score: float
    sharpness_score: float
    contrast_score: float
    clean_border_score: float

    rejection_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "selection_quality": self.selection_quality,
            "reid_quality": self.reid_quality,
            "reid_eligible": self.reid_eligible,
            "tier": self.tier,
            "bbox_width_px": self.bbox_width_px,
            "bbox_height_px": self.bbox_height_px,
            "crop_width_px": self.crop_width_px,
            "crop_height_px": self.crop_height_px,
            "laplacian_variance": self.laplacian_variance,
            "grayscale_std": self.grayscale_std,
            "black_fraction": self.black_fraction,
            "resolution_score": self.resolution_score,
            "sharpness_score": self.sharpness_score,
            "contrast_score": self.contrast_score,
            "clean_border_score": self.clean_border_score,
            "rejection_reasons": list(self.rejection_reasons),
        }


def grayscale_metrics(
    crop: np.ndarray,
) -> tuple[float, float, float]:
    """Return (Laplacian variance, grayscale std, near-black border fraction).

    Only the outer two-pixel border is inspected for near-black content.  This
    targets invalid/remapped image borders without penalizing a genuinely dark
    or black football kit in the middle of the crop.
    """
    if not isinstance(crop, np.ndarray):
        raise TypeError("crop must be a numpy array")
    if crop.ndim not in (2, 3):
        raise ValueError("crop must have shape HxW or HxWxC")
    if crop.shape[0] <= 0 or crop.shape[1] <= 0:
        raise ValueError("crop must be non-empty")

    if crop.ndim == 3:
        if crop.shape[2] != 3:
            raise ValueError("3D crop must have exactly 3 channels")
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    gray_f = gray.astype(np.float64)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_variance = float(laplacian.var())
    grayscale_std = float(gray_f.std())
    border_width = min(2, gray.shape[0], gray.shape[1])
    border_mask = np.zeros(gray.shape, dtype=bool)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True

    black_fraction = float(
        np.mean(gray[border_mask] <= 2)
    )

    return laplacian_variance, grayscale_std, black_fraction


def score_reid_crop(
    *,
    crop: np.ndarray,
    virtual_bbox_xyxy: Sequence[float],
    selection_quality: float,
    crop_clipped: bool,
    config: ReIDCropQualityConfig | None = None,
) -> ReIDCropQuality:
    """Score actual crop pixels as ReID evidence."""
    selected = config or ReIDCropQualityConfig()

    selection_score = _unit_interval(
        "selection_quality",
        selection_quality,
    )
    bbox_width, bbox_height = _bbox_size(virtual_bbox_xyxy)

    lap_var, gray_std, black_fraction = grayscale_metrics(crop)

    height_score = _ramp(
        bbox_height,
        bad=selected.resolution_height_bad_px,
        good=selected.resolution_height_good_px,
    )
    width_score = _ramp(
        bbox_width,
        bad=selected.resolution_width_bad_px,
        good=selected.resolution_width_good_px,
    )
    resolution_score = (
        0.80 * height_score
        + 0.20 * width_score
    )

    sharpness_score = _ramp(
        lap_var,
        bad=selected.sharpness_bad,
        good=selected.sharpness_good,
    )
    contrast_score = _ramp(
        gray_std,
        bad=selected.contrast_bad,
        good=selected.contrast_good,
    )
    clean_border_score = _inverse_ramp(
        black_fraction,
        good=selected.black_fraction_good,
        bad=selected.black_fraction_bad,
    )
    if crop_clipped:
        clean_border_score *= 0.50

    reid_quality = (
        selected.resolution_weight * resolution_score
        + selected.sharpness_weight * sharpness_score
        + selected.contrast_weight * contrast_score
        + selected.selection_weight * selection_score
        + selected.clean_border_weight * clean_border_score
    )
    reid_quality = _clamp01(reid_quality)

    reasons: list[str] = []
    if bbox_height < selected.hard_min_bbox_height_px:
        reasons.append("bbox_height_below_min")
    if bbox_width < selected.hard_min_bbox_width_px:
        reasons.append("bbox_width_below_min")
    if reid_quality < selected.min_reid_quality:
        reasons.append("reid_quality_below_min")

    eligible = not reasons

    if not eligible:
        tier = "reject"
    elif reid_quality >= selected.good_threshold:
        tier = "good"
    elif reid_quality >= selected.weak_threshold:
        tier = "usable"
    else:
        tier = "weak"

    return ReIDCropQuality(
        selection_quality=selection_score,
        reid_quality=reid_quality,
        reid_eligible=eligible,
        tier=tier,
        bbox_width_px=bbox_width,
        bbox_height_px=bbox_height,
        crop_width_px=int(crop.shape[1]),
        crop_height_px=int(crop.shape[0]),
        laplacian_variance=lap_var,
        grayscale_std=gray_std,
        black_fraction=black_fraction,
        resolution_score=resolution_score,
        sharpness_score=sharpness_score,
        contrast_score=contrast_score,
        clean_border_score=clean_border_score,
        rejection_reasons=tuple(reasons),
    )
