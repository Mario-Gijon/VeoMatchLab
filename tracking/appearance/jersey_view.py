"""Hito 7C.2A: jersey-number view/readiness evidence.

This stage does NOT recognize shirt numbers.  It answers a narrower question:

    "Which already-saved player crops are the best candidates for a future
     jersey-number recognizer?"

The score is deliberately colour-independent.  In particular, no pitch-green
suppression is used here because a goalkeeper may legitimately wear green and
must not be erased by a grass heuristic.

Signals:
* crop/torso resolution;
* local sharpness;
* local contrast;
* stroke/edge structure compatible with printed digits;
* central structure inside the shirt ROI;
* existing selection/ReID quality as weak priors.

Important semantics
-------------------
* high readiness != a number is definitely visible;
* low readiness != the player has no readable number anywhere else;
* no front/back orientation is inferred;
* no OCR result, team label, role or player identity is produced;
* the runner keeps several temporally diverse views per tracklet so the later
  recognizer can aggregate evidence instead of trusting one frame.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import cv2
import numpy as np


JERSEY_VIEW_VERSION = "jersey-view-v1"


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ramp(value: float, *, bad: float, good: float) -> float:
    if not good > bad:
        raise ValueError("good must be > bad")
    return _clamp01((float(value) - bad) / (good - bad))


def _band_score(
    value: float,
    *,
    bad_low: float,
    good_low: float,
    good_high: float,
    bad_high: float,
) -> float:
    if not (bad_low < good_low <= good_high < bad_high):
        raise ValueError("invalid band-score bounds")
    value = float(value)
    if value <= bad_low or value >= bad_high:
        return 0.0
    if good_low <= value <= good_high:
        return 1.0
    if value < good_low:
        return (value - bad_low) / (good_low - bad_low)
    return (bad_high - value) / (bad_high - good_high)


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


@dataclass(frozen=True, slots=True)
class JerseyViewConfig:
    # Central shirt region.  It intentionally covers both front and back
    # number locations and avoids most shorts/legs/background.
    roi_x1: float = 0.14
    roi_y1: float = 0.08
    roi_x2: float = 0.86
    roi_y2: float = 0.66

    min_crop_height_px: int = 28
    min_crop_width_px: int = 8
    min_roi_height_px: int = 16
    min_roi_width_px: int = 8

    resolution_height_bad_px: float = 36.0
    resolution_height_good_px: float = 120.0
    resolution_width_bad_px: float = 10.0
    resolution_width_good_px: float = 44.0

    sharpness_bad: float = 18.0
    sharpness_good: float = 130.0
    contrast_bad: float = 10.0
    contrast_good: float = 42.0

    # Digits tend to create a moderate amount of stroke/edge structure.
    edge_bad_low: float = 0.005
    edge_good_low: float = 0.035
    edge_good_high: float = 0.22
    edge_bad_high: float = 0.48

    central_edge_fraction_bad: float = 0.20
    central_edge_fraction_good: float = 0.42

    resolution_weight: float = 0.34
    sharpness_weight: float = 0.23
    contrast_weight: float = 0.13
    edge_structure_weight: float = 0.16
    central_structure_weight: float = 0.08
    prior_quality_weight: float = 0.06

    max_selected_per_tracklet: int = 4
    min_selected_score: float = 0.28
    min_temporal_spacing_s: float = 0.35

    def __post_init__(self) -> None:
        if not (0.0 <= self.roi_x1 < self.roi_x2 <= 1.0):
            raise ValueError("invalid ROI x bounds")
        if not (0.0 <= self.roi_y1 < self.roi_y2 <= 1.0):
            raise ValueError("invalid ROI y bounds")
        for value in (
            self.min_crop_height_px,
            self.min_crop_width_px,
            self.min_roi_height_px,
            self.min_roi_width_px,
            self.max_selected_per_tracklet,
        ):
            if value < 1:
                raise ValueError("integer thresholds must be >= 1")
        if not 0.0 <= self.min_selected_score <= 1.0:
            raise ValueError("min_selected_score must be in [0,1]")
        if self.min_temporal_spacing_s < 0.0:
            raise ValueError("min_temporal_spacing_s must be >= 0")

        weights = (
            self.resolution_weight,
            self.sharpness_weight,
            self.contrast_weight,
            self.edge_structure_weight,
            self.central_structure_weight,
            self.prior_quality_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("score weights must be non-negative")
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("score weights must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JerseyViewObservation:
    sample_id: str
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    crop_ref: str

    roi_xyxy: tuple[int, int, int, int]
    crop_width_px: int
    crop_height_px: int
    roi_width_px: int
    roi_height_px: int

    readiness_score: float
    resolution_score: float
    sharpness_score: float
    contrast_score: float
    edge_structure_score: float
    central_structure_score: float
    prior_quality_score: float

    laplacian_variance: float
    grayscale_std: float
    edge_density: float
    central_edge_fraction: float

    selection_quality: float
    reid_quality: float | None

    def __post_init__(self) -> None:
        if not self.sample_id or not self.tracklet_id or not self.detection_id:
            raise ValueError("ids must be non-empty")
        if not 0.0 <= self.readiness_score <= 1.0:
            raise ValueError("readiness_score must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "jersey_view_version": JERSEY_VIEW_VERSION,
            "sample_id": self.sample_id,
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "crop_ref": self.crop_ref,
            "roi_xyxy": list(self.roi_xyxy),
            "crop_width_px": self.crop_width_px,
            "crop_height_px": self.crop_height_px,
            "roi_width_px": self.roi_width_px,
            "roi_height_px": self.roi_height_px,
            "readiness_score": self.readiness_score,
            "resolution_score": self.resolution_score,
            "sharpness_score": self.sharpness_score,
            "contrast_score": self.contrast_score,
            "edge_structure_score": self.edge_structure_score,
            "central_structure_score": self.central_structure_score,
            "prior_quality_score": self.prior_quality_score,
            "laplacian_variance": self.laplacian_variance,
            "grayscale_std": self.grayscale_std,
            "edge_density": self.edge_density,
            "central_edge_fraction": self.central_edge_fraction,
            "selection_quality": self.selection_quality,
            "reid_quality": self.reid_quality,
            "orientation": "unknown",
            "jersey_number": None,
            "recognition_confidence": None,
        }


@dataclass(frozen=True, slots=True)
class RejectedJerseyView:
    tracklet_id: str
    detection_id: str
    frame_index: int
    crop_ref: str
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "crop_ref": self.crop_ref,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True, slots=True)
class SelectedJerseyView:
    observation: JerseyViewObservation
    rank_within_tracklet: int

    def to_dict(self) -> dict[str, Any]:
        payload = self.observation.to_dict()
        payload["rank_within_tracklet"] = self.rank_within_tracklet
        payload["selected_for_recognition"] = True
        return payload


def jersey_roi_xyxy(
    image_bgr: np.ndarray,
    config: JerseyViewConfig | None = None,
) -> tuple[int, int, int, int]:
    selected = config or JerseyViewConfig()
    height, width = image_bgr.shape[:2]

    x1 = max(0, min(width - 1, int(math.floor(width * selected.roi_x1))))
    x2 = max(x1 + 1, min(width, int(math.ceil(width * selected.roi_x2))))
    y1 = max(0, min(height - 1, int(math.floor(height * selected.roi_y1))))
    y2 = max(y1 + 1, min(height, int(math.ceil(height * selected.roi_y2))))
    return x1, y1, x2, y2


def crop_jersey_roi(
    image_bgr: np.ndarray,
    roi_xyxy: Sequence[int],
) -> np.ndarray:
    if len(roi_xyxy) != 4:
        raise ValueError("roi_xyxy must have four values")
    x1, y1, x2, y2 = [int(value) for value in roi_xyxy]
    if not (0 <= x1 < x2 <= image_bgr.shape[1]):
        raise ValueError("invalid ROI x bounds")
    if not (0 <= y1 < y2 <= image_bgr.shape[0]):
        raise ValueError("invalid ROI y bounds")
    return image_bgr[y1:y2, x1:x2].copy()


def _central_edge_fraction(edges: np.ndarray) -> float:
    height, width = edges.shape
    x1 = int(round(width * 0.20))
    x2 = int(round(width * 0.80))
    y1 = int(round(height * 0.10))
    y2 = int(round(height * 0.90))

    total_edges = int(np.count_nonzero(edges))
    if total_edges == 0:
        return 0.0
    central_edges = int(
        np.count_nonzero(edges[y1:y2, x1:x2])
    )
    return central_edges / total_edges


def score_jersey_view(
    image_bgr: np.ndarray,
    *,
    tracklet_id: str,
    detection_id: str,
    frame_index: int,
    timestamp_s: float,
    crop_ref: str,
    selection_quality: float,
    reid_quality: float | None = None,
    config: JerseyViewConfig | None = None,
) -> JerseyViewObservation | RejectedJerseyView:
    selected = config or JerseyViewConfig()

    if not isinstance(image_bgr, np.ndarray):
        raise TypeError("image_bgr must be a numpy array")
    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.shape[0] <= 0
        or image_bgr.shape[1] <= 0
    ):
        raise ValueError("image_bgr must have shape HxWx3")

    height, width = image_bgr.shape[:2]
    reasons: list[str] = []
    if height < selected.min_crop_height_px:
        reasons.append("crop_height_below_min")
    if width < selected.min_crop_width_px:
        reasons.append("crop_width_below_min")

    roi_xyxy = jersey_roi_xyxy(image_bgr, selected)
    roi = crop_jersey_roi(image_bgr, roi_xyxy)
    roi_height, roi_width = roi.shape[:2]

    if roi_height < selected.min_roi_height_px:
        reasons.append("roi_height_below_min")
    if roi_width < selected.min_roi_width_px:
        reasons.append("roi_width_below_min")

    if reasons:
        return RejectedJerseyView(
            tracklet_id=str(tracklet_id),
            detection_id=str(detection_id),
            frame_index=int(frame_index),
            crop_ref=str(crop_ref),
            reasons=tuple(sorted(set(reasons))),
        )

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # CLAHE makes the structure score less sensitive to sun/shadow while raw
    # gray is kept for actual contrast statistics.
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    ).apply(gray)

    laplacian_variance = float(
        cv2.Laplacian(clahe, cv2.CV_64F).var()
    )
    grayscale_std = float(gray.astype(np.float64).std())

    median_value = float(np.median(clahe))
    lower = int(max(0.0, 0.66 * median_value))
    upper = int(min(255.0, max(lower + 1.0, 1.33 * median_value)))
    edges = cv2.Canny(
        clahe,
        lower,
        upper,
        L2gradient=True,
    )
    edge_density = float(np.mean(edges > 0))
    central_edge_fraction = _central_edge_fraction(edges)

    height_score = _ramp(
        height,
        bad=selected.resolution_height_bad_px,
        good=selected.resolution_height_good_px,
    )
    width_score = _ramp(
        width,
        bad=selected.resolution_width_bad_px,
        good=selected.resolution_width_good_px,
    )
    resolution_score = 0.78 * height_score + 0.22 * width_score

    sharpness_score = _ramp(
        laplacian_variance,
        bad=selected.sharpness_bad,
        good=selected.sharpness_good,
    )
    contrast_score = _ramp(
        grayscale_std,
        bad=selected.contrast_bad,
        good=selected.contrast_good,
    )
    edge_structure_score = _band_score(
        edge_density,
        bad_low=selected.edge_bad_low,
        good_low=selected.edge_good_low,
        good_high=selected.edge_good_high,
        bad_high=selected.edge_bad_high,
    )
    central_structure_score = _ramp(
        central_edge_fraction,
        bad=selected.central_edge_fraction_bad,
        good=selected.central_edge_fraction_good,
    )

    selection_q = _clamp01(
        _finite("selection_quality", selection_quality)
    )
    reid_q = (
        None
        if reid_quality is None
        else _clamp01(_finite("reid_quality", reid_quality))
    )
    prior_quality_score = (
        selection_q
        if reid_q is None
        else 0.40 * selection_q + 0.60 * reid_q
    )

    readiness = _clamp01(
        selected.resolution_weight * resolution_score
        + selected.sharpness_weight * sharpness_score
        + selected.contrast_weight * contrast_score
        + selected.edge_structure_weight * edge_structure_score
        + selected.central_structure_weight * central_structure_score
        + selected.prior_quality_weight * prior_quality_score
    )

    return JerseyViewObservation(
        sample_id=f"jersey-view:{detection_id}",
        tracklet_id=str(tracklet_id),
        detection_id=str(detection_id),
        frame_index=int(frame_index),
        timestamp_s=_finite("timestamp_s", timestamp_s),
        crop_ref=str(crop_ref),
        roi_xyxy=roi_xyxy,
        crop_width_px=int(width),
        crop_height_px=int(height),
        roi_width_px=int(roi_width),
        roi_height_px=int(roi_height),
        readiness_score=readiness,
        resolution_score=resolution_score,
        sharpness_score=sharpness_score,
        contrast_score=contrast_score,
        edge_structure_score=edge_structure_score,
        central_structure_score=central_structure_score,
        prior_quality_score=prior_quality_score,
        laplacian_variance=laplacian_variance,
        grayscale_std=grayscale_std,
        edge_density=edge_density,
        central_edge_fraction=central_edge_fraction,
        selection_quality=selection_q,
        reid_quality=reid_q,
    )


def select_jersey_views(
    observations: Iterable[JerseyViewObservation],
    config: JerseyViewConfig | None = None,
) -> tuple[SelectedJerseyView, ...]:
    selected = config or JerseyViewConfig()
    grouped: dict[str, list[JerseyViewObservation]] = {}

    for observation in observations:
        grouped.setdefault(
            observation.tracklet_id,
            [],
        ).append(observation)

    result: list[SelectedJerseyView] = []

    for tracklet_id in sorted(grouped):
        ranked = sorted(
            grouped[tracklet_id],
            key=lambda item: (
                -item.readiness_score,
                -item.crop_height_px,
                item.timestamp_s,
                item.sample_id,
            ),
        )

        chosen: list[JerseyViewObservation] = []
        for observation in ranked:
            if observation.readiness_score < selected.min_selected_score:
                continue
            if any(
                abs(
                    observation.timestamp_s
                    - existing.timestamp_s
                ) < selected.min_temporal_spacing_s
                for existing in chosen
            ):
                continue

            chosen.append(observation)
            if len(chosen) >= selected.max_selected_per_tracklet:
                break

        for rank, observation in enumerate(chosen, start=1):
            result.append(
                SelectedJerseyView(
                    observation=observation,
                    rank_within_tracklet=rank,
                )
            )

    result.sort(
        key=lambda item: (
            item.observation.tracklet_id,
            item.rank_within_tracklet,
        )
    )
    return tuple(result)


def jersey_view_summary(
    observations: Sequence[JerseyViewObservation],
    rejected: Sequence[RejectedJerseyView],
    selected: Sequence[SelectedJerseyView],
    *,
    total_tracklet_count: int | None = None,
) -> dict[str, Any]:
    observed_tracklets = {
        item.tracklet_id for item in observations
    }
    selected_tracklets = {
        item.observation.tracklet_id for item in selected
    }

    payload = {
        "schema_version": 1,
        "jersey_view_version": JERSEY_VIEW_VERSION,
        "observation_count": len(observations),
        "rejected_count": len(rejected),
        "selected_count": len(selected),
        "observed_tracklet_count": len(observed_tracklets),
        "selected_tracklet_count": len(selected_tracklets),
        "readiness_score": _stats(
            [item.readiness_score for item in observations]
        ),
        "selected_readiness_score": _stats(
            [
                item.observation.readiness_score
                for item in selected
            ]
        ),
        "selected_per_tracklet": _stats(
            [
                float(
                    sum(
                        1
                        for item in selected
                        if item.observation.tracklet_id == tracklet_id
                    )
                )
                for tracklet_id in sorted(selected_tracklets)
            ]
        ),
        "number_recognition_run": False,
        "jersey_numbers_assigned": False,
        "orientation_inferred": False,
        "colour_used_for_scoring": False,
    }

    if total_tracklet_count is not None:
        payload["total_tracklet_count"] = total_tracklet_count
        payload["selected_tracklet_coverage"] = (
            len(selected_tracklets) / total_tracklet_count
            if total_tracklet_count
            else None
        )

    return payload
