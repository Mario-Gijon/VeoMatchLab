"""Hito 7C.1: model-free football kit appearance evidence.

This module extracts a compact colour/appearance signature from already-saved
player crops.  It deliberately does NOT assign a team, role, player identity
or shirt number.

Why this exists
---------------
PRTReID is useful but football is unusually ambiguous: many players share the
same kit.  That ambiguity is also useful structure.  A stable kit descriptor
can later tell the offline actor solver that two fragments are visually
compatible with the same team/role, or that they are strongly inconsistent.

The descriptor is intentionally simple and inspectable:
* upper-torso and shorts regions are handled separately;
* likely pitch-green pixels are suppressed;
* colourful hue, saturation, brightness and achromatic fractions are retained;
* multiple observations of one local tracklet are quality-weighted and
  aggregated;
* pairwise kit evidence is continuous/soft.  No threshold here means
  "same team" or "different team".

Later milestones may cluster these profiles into team/goalkeeper/official
groups and may add jersey-number evidence.  This module only provides the
pixel-level evidence required to do that safely.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


KIT_EVIDENCE_VERSION = "kit-appearance-v1"

# One ROI contains:
# 12 hue + 4 saturation + 4 value + 4 scalar fractions = 24 dimensions.
ROI_FEATURE_DIM = 24
KIT_FEATURE_DIM = ROI_FEATURE_DIM * 2


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
        raise ValueError("good must be greater than bad")
    return _clamp01((float(value) - bad) / (good - bad))


def _unit(values: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} has zero norm")
    return vector / norm


def _unit_or_zero(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def cosine(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    a = _unit(first, label="first kit signature")
    b = _unit(second, label="second kit signature")
    if a.shape != b.shape:
        raise ValueError("kit signature dimensions differ")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


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
class KitEvidenceConfig:
    # Relative crop regions.  The appearance crop already has modest padding.
    torso_x1: float = 0.12
    torso_y1: float = 0.08
    torso_x2: float = 0.88
    torso_y2: float = 0.62

    shorts_x1: float = 0.18
    shorts_y1: float = 0.52
    shorts_x2: float = 0.82
    shorts_y2: float = 0.90

    min_crop_height_px: int = 28
    min_crop_width_px: int = 8
    min_non_grass_pixels_per_roi: int = 32

    grass_hue_min: int = 30
    grass_hue_max: int = 95
    grass_saturation_min: int = 45

    colourful_saturation_min: int = 32
    colourful_value_min: int = 20

    torso_weight: float = 0.72
    shorts_weight: float = 0.28

    resolution_height_bad_px: float = 28.0
    resolution_height_good_px: float = 90.0
    resolution_width_bad_px: float = 8.0
    resolution_width_good_px: float = 36.0
    coverage_bad: float = 0.20
    coverage_good: float = 0.55

    def __post_init__(self) -> None:
        for prefix in ("torso", "shorts"):
            x1 = getattr(self, f"{prefix}_x1")
            y1 = getattr(self, f"{prefix}_y1")
            x2 = getattr(self, f"{prefix}_x2")
            y2 = getattr(self, f"{prefix}_y2")
            if not (0.0 <= x1 < x2 <= 1.0):
                raise ValueError(f"{prefix} x bounds are invalid")
            if not (0.0 <= y1 < y2 <= 1.0):
                raise ValueError(f"{prefix} y bounds are invalid")

        if self.min_crop_height_px < 1 or self.min_crop_width_px < 1:
            raise ValueError("minimum crop size must be positive")
        if self.min_non_grass_pixels_per_roi < 1:
            raise ValueError("min_non_grass_pixels_per_roi must be positive")
        if not math.isclose(
            self.torso_weight + self.shorts_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("torso_weight + shorts_weight must equal 1")
        if self.torso_weight <= 0.0 or self.shorts_weight <= 0.0:
            raise ValueError("ROI weights must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KitSample:
    sample_id: str
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    crop_ref: str
    quality: float
    signature: tuple[float, ...]
    torso_non_grass_fraction: float
    shorts_non_grass_fraction: float
    torso_colourful_fraction: float
    shorts_colourful_fraction: float
    crop_width_px: int
    crop_height_px: int

    def __post_init__(self) -> None:
        if not self.sample_id or not self.tracklet_id or not self.detection_id:
            raise ValueError("sample/tracklet/detection ids must be non-empty")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0,1]")
        if len(self.signature) != KIT_FEATURE_DIM:
            raise ValueError(
                f"kit signature must have {KIT_FEATURE_DIM} values"
            )
        _unit(self.signature, label=self.sample_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kit_evidence_version": KIT_EVIDENCE_VERSION,
            "sample_id": self.sample_id,
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "crop_ref": self.crop_ref,
            "quality": self.quality,
            "signature": list(self.signature),
            "torso_non_grass_fraction": self.torso_non_grass_fraction,
            "shorts_non_grass_fraction": self.shorts_non_grass_fraction,
            "torso_colourful_fraction": self.torso_colourful_fraction,
            "shorts_colourful_fraction": self.shorts_colourful_fraction,
            "crop_width_px": self.crop_width_px,
            "crop_height_px": self.crop_height_px,
            "team_label": None,
            "role_label": None,
            "jersey_number": None,
        }


@dataclass(frozen=True, slots=True)
class KitRejectedSample:
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
class KitTrackletProfile:
    tracklet_id: str
    aggregate_signature: tuple[float, ...]
    sample_count: int
    effective_sample_count: float
    reliability: float
    mean_quality: float
    internal_similarity: Mapping[str, Any]
    first_timestamp_s: float
    last_timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kit_evidence_version": KIT_EVIDENCE_VERSION,
            "tracklet_id": self.tracklet_id,
            "aggregate_signature": list(self.aggregate_signature),
            "sample_count": self.sample_count,
            "effective_sample_count": self.effective_sample_count,
            "reliability": self.reliability,
            "mean_quality": self.mean_quality,
            "internal_similarity": dict(self.internal_similarity),
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
            "team_label": None,
            "role_label": None,
        }


@dataclass(frozen=True, slots=True)
class KitPairEvidence:
    tracklet_a: str
    tracklet_b: str
    kit_cosine: float
    torso_cosine: float
    shorts_cosine: float
    reliability: float
    support_quality: float

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("pair must use canonical tracklet ordering")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kit_evidence_version": KIT_EVIDENCE_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "kit_cosine": self.kit_cosine,
            "torso_cosine": self.torso_cosine,
            "shorts_cosine": self.shorts_cosine,
            "reliability": self.reliability,
            "support_quality": self.support_quality,
            "same_team_decision": None,
            "identity_decision": None,
        }


def _roi(
    image: np.ndarray,
    *,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    height, width = image.shape[:2]
    left = max(0, min(width - 1, int(math.floor(width * x1))))
    right = max(left + 1, min(width, int(math.ceil(width * x2))))
    top = max(0, min(height - 1, int(math.floor(height * y1))))
    bottom = max(top + 1, min(height, int(math.ceil(height * y2))))
    return image[top:bottom, left:right]


def _histogram(
    values: np.ndarray,
    *,
    bins: int,
    value_range: tuple[float, float],
) -> np.ndarray:
    if values.size == 0:
        return np.zeros(bins, dtype=np.float64)
    hist, _ = np.histogram(
        values,
        bins=bins,
        range=value_range,
    )
    return _unit_or_zero(hist.astype(np.float64))


def _roi_signature(
    roi_bgr: np.ndarray,
    config: KitEvidenceConfig,
) -> tuple[np.ndarray, dict[str, float | int]]:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[..., 0]
    saturation = hsv[..., 1]
    value = hsv[..., 2]

    grass = (
        (hue >= config.grass_hue_min)
        & (hue <= config.grass_hue_max)
        & (saturation >= config.grass_saturation_min)
    )
    usable = ~grass
    usable_count = int(np.count_nonzero(usable))
    pixel_count = int(usable.size)
    non_grass_fraction = (
        usable_count / pixel_count
        if pixel_count
        else 0.0
    )

    colourful = (
        usable
        & (saturation >= config.colourful_saturation_min)
        & (value >= config.colourful_value_min)
    )
    colourful_count = int(np.count_nonzero(colourful))
    colourful_fraction = (
        colourful_count / usable_count
        if usable_count
        else 0.0
    )

    hue_hist = _histogram(
        hue[colourful],
        bins=12,
        value_range=(0.0, 180.0),
    )
    sat_hist = _histogram(
        saturation[usable],
        bins=4,
        value_range=(0.0, 256.0),
    )
    val_hist = _histogram(
        value[usable],
        bins=4,
        value_range=(0.0, 256.0),
    )

    if usable_count:
        low_sat_fraction = float(
            np.mean(saturation[usable] < 40)
        )
        dark_fraction = float(
            np.mean(value[usable] < 55)
        )
        bright_fraction = float(
            np.mean(value[usable] > 205)
        )
    else:
        low_sat_fraction = 0.0
        dark_fraction = 0.0
        bright_fraction = 0.0

    scalar = _unit_or_zero(
        np.asarray(
            [
                low_sat_fraction,
                dark_fraction,
                bright_fraction,
                colourful_fraction,
            ],
            dtype=np.float64,
        )
    )

    # Equalise the influence of feature families before torso/shorts weighting.
    vector = np.concatenate(
        [
            hue_hist * math.sqrt(0.55),
            sat_hist * math.sqrt(0.15),
            val_hist * math.sqrt(0.15),
            scalar * math.sqrt(0.15),
        ]
    )
    vector = _unit_or_zero(vector)

    return vector, {
        "pixel_count": pixel_count,
        "usable_count": usable_count,
        "non_grass_fraction": non_grass_fraction,
        "colourful_fraction": colourful_fraction,
    }


def extract_kit_sample(
    image_bgr: np.ndarray,
    *,
    tracklet_id: str,
    detection_id: str,
    frame_index: int,
    timestamp_s: float,
    crop_ref: str,
    selection_quality: float,
    reid_quality: float | None = None,
    config: KitEvidenceConfig | None = None,
) -> KitSample | KitRejectedSample:
    selected = config or KitEvidenceConfig()

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

    torso = _roi(
        image_bgr,
        x1=selected.torso_x1,
        y1=selected.torso_y1,
        x2=selected.torso_x2,
        y2=selected.torso_y2,
    )
    shorts = _roi(
        image_bgr,
        x1=selected.shorts_x1,
        y1=selected.shorts_y1,
        x2=selected.shorts_x2,
        y2=selected.shorts_y2,
    )

    torso_vector, torso_info = _roi_signature(torso, selected)
    shorts_vector, shorts_info = _roi_signature(shorts, selected)

    if (
        int(torso_info["usable_count"])
        < selected.min_non_grass_pixels_per_roi
    ):
        reasons.append("torso_non_grass_pixels_below_min")
    if (
        int(shorts_info["usable_count"])
        < selected.min_non_grass_pixels_per_roi
    ):
        reasons.append("shorts_non_grass_pixels_below_min")

    signature = np.concatenate(
        [
            torso_vector * math.sqrt(selected.torso_weight),
            shorts_vector * math.sqrt(selected.shorts_weight),
        ]
    )
    signature_norm = float(np.linalg.norm(signature))
    if signature_norm <= 1e-12:
        reasons.append("empty_kit_signature")
    else:
        signature = signature / signature_norm

    if reasons:
        return KitRejectedSample(
            tracklet_id=str(tracklet_id),
            detection_id=str(detection_id),
            frame_index=int(frame_index),
            crop_ref=str(crop_ref),
            reasons=tuple(sorted(set(reasons))),
        )

    selection_q = _clamp01(_finite("selection_quality", selection_quality))
    reid_q = (
        selection_q
        if reid_quality is None
        else _clamp01(_finite("reid_quality", reid_quality))
    )
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
    resolution_score = 0.80 * height_score + 0.20 * width_score
    coverage = 0.70 * float(
        torso_info["non_grass_fraction"]
    ) + 0.30 * float(shorts_info["non_grass_fraction"])
    coverage_score = _ramp(
        coverage,
        bad=selected.coverage_bad,
        good=selected.coverage_good,
    )

    quality = _clamp01(
        0.45 * resolution_score
        + 0.30 * coverage_score
        + 0.15 * selection_q
        + 0.10 * reid_q
    )

    return KitSample(
        sample_id=f"kit:{detection_id}",
        tracklet_id=str(tracklet_id),
        detection_id=str(detection_id),
        frame_index=int(frame_index),
        timestamp_s=_finite("timestamp_s", timestamp_s),
        crop_ref=str(crop_ref),
        quality=quality,
        signature=tuple(float(value) for value in signature),
        torso_non_grass_fraction=float(
            torso_info["non_grass_fraction"]
        ),
        shorts_non_grass_fraction=float(
            shorts_info["non_grass_fraction"]
        ),
        torso_colourful_fraction=float(
            torso_info["colourful_fraction"]
        ),
        shorts_colourful_fraction=float(
            shorts_info["colourful_fraction"]
        ),
        crop_width_px=int(width),
        crop_height_px=int(height),
    )


def build_kit_tracklet_profile(
    samples: Sequence[KitSample],
) -> KitTrackletProfile:
    if not samples:
        raise ValueError("cannot build kit profile without samples")

    ordered = sorted(
        samples,
        key=lambda item: (
            item.timestamp_s,
            item.frame_index,
            item.sample_id,
        ),
    )
    tracklet_ids = {item.tracklet_id for item in ordered}
    if len(tracklet_ids) != 1:
        raise ValueError("kit profile samples must share one tracklet")

    vectors = np.vstack(
        [
            _unit(item.signature, label=item.sample_id)
            for item in ordered
        ]
    )
    qualities = np.asarray(
        [max(item.quality, 1e-3) for item in ordered],
        dtype=np.float64,
    )
    weights = qualities / float(np.sum(qualities))
    centroid = _unit(
        np.sum(vectors * weights[:, None], axis=0),
        label="kit centroid",
    )
    similarities = np.clip(
        vectors @ centroid,
        -1.0,
        1.0,
    )

    # Second pass softly discounts samples that disagree with the initial
    # centroid, without deleting them.
    agreement = np.clip(
        (similarities + 1.0) * 0.5,
        0.0,
        1.0,
    )
    weights = qualities * (0.20 + 0.80 * np.square(agreement))
    weights = weights / float(np.sum(weights))
    centroid = _unit(
        np.sum(vectors * weights[:, None], axis=0),
        label="refined kit centroid",
    )
    similarities = np.clip(vectors @ centroid, -1.0, 1.0)

    effective_count = float(1.0 / np.sum(np.square(weights)))
    mean_quality = float(np.mean(qualities))
    mean_similarity = float(np.mean(similarities))
    internal_consistency = _clamp01(
        (mean_similarity + 1.0) * 0.5
    )
    support_factor = float(
        1.0 - math.exp(-effective_count / 2.0)
    )
    reliability = _clamp01(
        mean_quality * internal_consistency * support_factor
    )

    return KitTrackletProfile(
        tracklet_id=ordered[0].tracklet_id,
        aggregate_signature=tuple(
            float(value) for value in centroid
        ),
        sample_count=len(ordered),
        effective_sample_count=effective_count,
        reliability=reliability,
        mean_quality=mean_quality,
        internal_similarity=_stats(
            [float(value) for value in similarities]
        ),
        first_timestamp_s=float(ordered[0].timestamp_s),
        last_timestamp_s=float(ordered[-1].timestamp_s),
    )


def build_kit_tracklet_profiles(
    samples: Iterable[KitSample],
) -> tuple[KitTrackletProfile, ...]:
    grouped: dict[str, list[KitSample]] = {}
    seen_sample_ids: set[str] = set()

    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate kit sample_id: {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)
        grouped.setdefault(sample.tracklet_id, []).append(sample)

    return tuple(
        build_kit_tracklet_profile(grouped[tracklet_id])
        for tracklet_id in sorted(grouped)
    )


def compare_kit_profiles(
    first: KitTrackletProfile,
    second: KitTrackletProfile,
) -> KitPairEvidence:
    if first.tracklet_id == second.tracklet_id:
        raise ValueError("kit pair requires two different tracklets")
    if second.tracklet_id < first.tracklet_id:
        first, second = second, first

    vector_a = _unit(
        first.aggregate_signature,
        label=first.tracklet_id,
    )
    vector_b = _unit(
        second.aggregate_signature,
        label=second.tracklet_id,
    )

    torso_a = vector_a[:ROI_FEATURE_DIM]
    torso_b = vector_b[:ROI_FEATURE_DIM]
    shorts_a = vector_a[ROI_FEATURE_DIM:]
    shorts_b = vector_b[ROI_FEATURE_DIM:]

    return KitPairEvidence(
        tracklet_a=first.tracklet_id,
        tracklet_b=second.tracklet_id,
        kit_cosine=float(
            np.clip(np.dot(vector_a, vector_b), -1.0, 1.0)
        ),
        torso_cosine=cosine(torso_a, torso_b),
        shorts_cosine=cosine(shorts_a, shorts_b),
        reliability=float(
            math.sqrt(
                max(0.0, first.reliability * second.reliability)
            )
        ),
        support_quality=float(
            math.sqrt(
                max(0.0, first.mean_quality * second.mean_quality)
            )
        ),
    )


def kit_profile_summary(
    samples: Sequence[KitSample],
    profiles: Sequence[KitTrackletProfile],
    *,
    total_tracklet_count: int | None = None,
) -> dict[str, Any]:
    internal_means = [
        float(profile.internal_similarity["mean"])
        for profile in profiles
        if profile.internal_similarity.get("mean") is not None
    ]
    payload = {
        "schema_version": 1,
        "kit_evidence_version": KIT_EVIDENCE_VERSION,
        "sample_count": len(samples),
        "profile_count": len(profiles),
        "sample_quality": _stats(
            [sample.quality for sample in samples]
        ),
        "profile_sample_count": _stats(
            [float(profile.sample_count) for profile in profiles]
        ),
        "profile_reliability": _stats(
            [profile.reliability for profile in profiles]
        ),
        "profile_internal_similarity_mean": _stats(
            internal_means
        ),
        "team_labels_assigned": False,
        "role_labels_assigned": False,
        "jersey_numbers_used": False,
    }
    if total_tracklet_count is not None:
        payload["total_tracklet_count"] = total_tracklet_count
        payload["unprofiled_tracklet_count"] = (
            total_tracklet_count - len(profiles)
        )
        payload["profile_coverage"] = (
            len(profiles) / total_tracklet_count
            if total_tracklet_count
            else None
        )
    return payload
