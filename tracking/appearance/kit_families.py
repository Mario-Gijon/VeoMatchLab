"""Hito 7C.1D: football-constrained, illumination-robust kit families.

Football gives us useful structure that generic free-K clustering ignores:
there are exactly two dominant outfield-team kit families in a match, while
goalkeepers and officials are minority visual families.  This module uses that
structure without hard-coding any colour, club, goalkeeper colour or referee
colour.

Pipeline
--------
1. Extract an illumination-robust colour-family signature from each saved crop.
   Absolute brightness is deliberately de-emphasised.  Chromaticity, hue,
   saturation, Lab chroma and brightness relative to grass in the SAME crop
   carry most of the signal.
2. Aggregate samples per tracklet.
3. Robustly fit exactly two dominant primary families using trimmed weighted
   spherical k-means.  Minority appearances do not drag the primary centroids.
4. Profiles that do not confidently fit either primary family form a residual
   pool.
5. Discover 0..3 coherent special families only inside that residual pool.
6. Anything weak/ambiguous remains unknown.

The output is still appearance evidence, not identity truth.  Family
differences are SOFT evidence only; physical cannot-links remain the only hard
identity constraints.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


KIT_FAMILY_VERSION = "kit-family-v1"

# Per ROI:
# rgb chromaticity median (3)
# rgb chromaticity IQR (3)
# Lab a/b median (2)
# saturation median (1)
# weighted hue sin/cos (2)
# colourful fraction (1)
# achromatic fraction (1)
# brightness relative to local grass (1)
ROI_DIM = 14
SIGNATURE_DIM = ROI_DIM * 2


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _unit(values: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be non-empty 1-D")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} has non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} has zero norm")
    return vector / norm


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "min": None, "p10": None, "median": None,
            "p90": None, "p95": None, "max": None, "mean": None,
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
    }


@dataclass(frozen=True, slots=True)
class KitFamilyConfig:
    torso_x1: float = 0.14
    torso_y1: float = 0.10
    torso_x2: float = 0.86
    torso_y2: float = 0.60

    shorts_x1: float = 0.20
    shorts_y1: float = 0.52
    shorts_x2: float = 0.80
    shorts_y2: float = 0.88

    min_crop_height_px: int = 28
    min_crop_width_px: int = 8
    min_usable_pixels: int = 28

    grass_hue_min: int = 30
    grass_hue_max: int = 95
    grass_saturation_min: int = 40

    colourful_saturation_min: int = 40
    colourful_value_min: int = 25

    torso_weight: float = 0.72
    shorts_weight: float = 0.28

    # Primary-family fit.
    primary_core_fraction: float = 0.72
    primary_max_iterations: int = 80
    min_primary_core_count: int = 20
    primary_similarity_quantile: float = 0.05
    primary_similarity_scale: float = 0.94
    primary_margin_quantile: float = 0.05
    primary_margin_scale: float = 0.75

    # Special families are minority residual groups, never a free-K fit over
    # the whole clip.
    max_special_families: int = 3
    min_special_profiles: int = 5
    min_special_mass_fraction: float = 0.01
    max_special_primary_cosine: float = 0.94
    min_special_silhouette: float = 0.05

    min_profile_reliability_for_family: float = 0.20

    def __post_init__(self) -> None:
        if not math.isclose(
            self.torso_weight + self.shorts_weight,
            1.0,
            abs_tol=1e-9,
        ):
            raise ValueError("torso_weight + shorts_weight must equal 1")
        if not 0.5 <= self.primary_core_fraction <= 1.0:
            raise ValueError("primary_core_fraction must be in [0.5,1]")
        if self.max_special_families < 0 or self.max_special_families > 3:
            raise ValueError("max_special_families must be in [0,3]")
        if not -1.0 <= self.max_special_primary_cosine <= 1.0:
            raise ValueError("invalid max_special_primary_cosine")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KitFamilySample:
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    crop_ref: str
    quality: float
    signature: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kit_family_version": KIT_FAMILY_VERSION,
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "crop_ref": self.crop_ref,
            "quality": self.quality,
            "signature": list(self.signature),
        }


@dataclass(frozen=True, slots=True)
class KitFamilyProfile:
    tracklet_id: str
    signature: tuple[float, ...]
    sample_count: int
    reliability: float
    first_timestamp_s: float
    last_timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kit_family_version": KIT_FAMILY_VERSION,
            "tracklet_id": self.tracklet_id,
            "signature": list(self.signature),
            "sample_count": self.sample_count,
            "reliability": self.reliability,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
        }


@dataclass(frozen=True, slots=True)
class KitFamilyAssignment:
    tracklet_id: str
    family_type: str
    family_id: int | None
    state: str
    best_cosine: float
    second_best_cosine: float | None
    margin: float | None
    profile_reliability: float

    def __post_init__(self) -> None:
        if self.family_type not in {"primary", "special", "unknown"}:
            raise ValueError("invalid family_type")
        if self.state not in {
            "confident_primary",
            "confident_special",
            "ambiguous_primary",
            "residual_unknown",
            "low_reliability",
        }:
            raise ValueError("invalid state")

    @property
    def usable_for_tracking(self) -> bool:
        return self.state in {"confident_primary", "confident_special"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "family_type": self.family_type,
            "family_id": self.family_id,
            "family_key": (
                None
                if self.family_id is None
                else f"{self.family_type}-{self.family_id}"
            ),
            "state": self.state,
            "usable_for_tracking": self.usable_for_tracking,
            "best_cosine": self.best_cosine,
            "second_best_cosine": self.second_best_cosine,
            "margin": self.margin,
            "profile_reliability": self.profile_reliability,
            "team_label": None,
            "role_label": None,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class KitFamilyCluster:
    family_type: str
    family_id: int
    centroid: tuple[float, ...]
    profile_count: int
    weighted_mass: float
    mean_cosine: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_type": self.family_type,
            "family_id": self.family_id,
            "family_key": f"{self.family_type}-{self.family_id}",
            "centroid": list(self.centroid),
            "profile_count": self.profile_count,
            "weighted_mass": self.weighted_mass,
            "mean_cosine": self.mean_cosine,
            "team_label": None,
            "role_label": None,
        }


@dataclass(frozen=True, slots=True)
class KitFamilyResult:
    assignments: tuple[KitFamilyAssignment, ...]
    clusters: tuple[KitFamilyCluster, ...]
    primary_similarity_thresholds: tuple[float, float]
    primary_margin_thresholds: tuple[float, float]
    special_family_count: int
    special_model_scores: tuple[Mapping[str, Any], ...]
    config: KitFamilyConfig

    def summary(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        types: dict[str, int] = {}
        for row in self.assignments:
            states[row.state] = states.get(row.state, 0) + 1
            types[row.family_type] = types.get(row.family_type, 0) + 1
        return {
            "schema_version": 1,
            "kit_family_version": KIT_FAMILY_VERSION,
            "profile_count": len(self.assignments),
            "primary_family_count": 2,
            "special_family_count": self.special_family_count,
            "assignment_state_counts": dict(sorted(states.items())),
            "assignment_type_counts": dict(sorted(types.items())),
            "usable_for_tracking_count": sum(
                row.usable_for_tracking for row in self.assignments
            ),
            "primary_similarity_thresholds": list(
                self.primary_similarity_thresholds
            ),
            "primary_margin_thresholds": list(
                self.primary_margin_thresholds
            ),
            "special_model_scores": [
                dict(row) for row in self.special_model_scores
            ],
            "clusters": [row.to_dict() for row in self.clusters],
            "config": self.config.to_dict(),
            "semantic_roles_assigned": False,
            "identity_hard_constraints_created": False,
            "illumination_robust_descriptor": True,
            "football_structure": {
                "dominant_primary_families": 2,
                "special_families_max": 3,
            },
        }


def _roi(image: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    h, w = image.shape[:2]
    left = max(0, min(w - 1, int(round(w * x1))))
    right = max(left + 1, min(w, int(round(w * x2))))
    top = max(0, min(h - 1, int(round(h * y1))))
    bottom = max(top + 1, min(h, int(round(h * y2))))
    return image[top:bottom, left:right]


def _grass_mask(hsv: np.ndarray, config: KitFamilyConfig) -> np.ndarray:
    h = hsv[..., 0]
    s = hsv[..., 1]
    return (
        (h >= config.grass_hue_min)
        & (h <= config.grass_hue_max)
        & (s >= config.grass_saturation_min)
    )


def _relative_grass_value(
    image_bgr: np.ndarray,
    config: KitFamilyConfig,
) -> float:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = _grass_mask(hsv, config)
    if int(np.count_nonzero(mask)) < 12:
        return float(np.median(hsv[..., 2]))
    return float(np.median(hsv[..., 2][mask]))


def _roi_descriptor(
    roi_bgr: np.ndarray,
    *,
    grass_value: float,
    config: KitFamilyConfig,
) -> np.ndarray:
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB).astype(np.float64)

    grass = _grass_mask(hsv, config)
    usable = ~grass
    if int(np.count_nonzero(usable)) < config.min_usable_pixels:
        raise ValueError("not enough non-grass pixels")

    pixels = rgb[usable]
    total = np.sum(pixels, axis=1, keepdims=True)
    chroma = pixels / np.maximum(total, 1.0)

    chroma_median = np.median(chroma, axis=0)
    chroma_iqr = (
        np.percentile(chroma, 75, axis=0)
        - np.percentile(chroma, 25, axis=0)
    )

    a = (lab[..., 1][usable].astype(np.float64) - 128.0) / 128.0
    b = (lab[..., 2][usable].astype(np.float64) - 128.0) / 128.0
    lab_chroma = np.asarray(
        [np.median(a), np.median(b)],
        dtype=np.float64,
    )

    saturation = hsv[..., 1][usable].astype(np.float64) / 255.0
    value = hsv[..., 2][usable].astype(np.float64)
    sat_median = float(np.median(saturation))

    colourful = (
        (hsv[..., 1][usable] >= config.colourful_saturation_min)
        & (hsv[..., 2][usable] >= config.colourful_value_min)
    )
    colourful_fraction = float(np.mean(colourful))
    achromatic_fraction = float(
        np.mean(hsv[..., 1][usable] < config.colourful_saturation_min)
    )

    hue = hsv[..., 0][usable].astype(np.float64)
    angle = hue * (2.0 * math.pi / 180.0)
    hue_weight = np.maximum(saturation, 0.05)
    hue_sin = float(
        np.sum(np.sin(angle) * hue_weight) / np.sum(hue_weight)
    )
    hue_cos = float(
        np.sum(np.cos(angle) * hue_weight) / np.sum(hue_weight)
    )

    median_value = float(np.median(value))
    # Ratio against the grass in the SAME crop is much less sensitive to
    # exposure/shadow than absolute V.  tanh keeps it bounded.
    rel_log = math.log(
        (median_value + 8.0) / (float(grass_value) + 8.0)
    )
    relative_brightness = math.tanh(rel_log)

    vector = np.concatenate([
        chroma_median * math.sqrt(0.34),
        chroma_iqr * math.sqrt(0.08),
        lab_chroma * math.sqrt(0.16),
        np.asarray([sat_median]) * math.sqrt(0.08),
        np.asarray([hue_sin, hue_cos]) * math.sqrt(0.16),
        np.asarray([
            colourful_fraction,
            achromatic_fraction,
        ]) * math.sqrt(0.10),
        np.asarray([relative_brightness]) * math.sqrt(0.08),
    ])
    return _unit(vector, label="ROI colour-family descriptor")


def extract_kit_family_sample(
    image_bgr: np.ndarray,
    *,
    tracklet_id: str,
    detection_id: str,
    frame_index: int,
    timestamp_s: float,
    crop_ref: str,
    quality: float,
    config: KitFamilyConfig | None = None,
) -> KitFamilySample | None:
    selected = config or KitFamilyConfig()
    if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError("image must be HxWx3")
    h, w = image_bgr.shape[:2]
    if h < selected.min_crop_height_px or w < selected.min_crop_width_px:
        return None

    grass_value = _relative_grass_value(image_bgr, selected)
    torso = _roi(
        image_bgr,
        selected.torso_x1, selected.torso_y1,
        selected.torso_x2, selected.torso_y2,
    )
    shorts = _roi(
        image_bgr,
        selected.shorts_x1, selected.shorts_y1,
        selected.shorts_x2, selected.shorts_y2,
    )
    try:
        torso_vec = _roi_descriptor(
            torso,
            grass_value=grass_value,
            config=selected,
        )
        shorts_vec = _roi_descriptor(
            shorts,
            grass_value=grass_value,
            config=selected,
        )
    except ValueError:
        return None

    signature = np.concatenate([
        torso_vec * math.sqrt(selected.torso_weight),
        shorts_vec * math.sqrt(selected.shorts_weight),
    ])
    signature = _unit(signature, label="kit-family signature")

    return KitFamilySample(
        tracklet_id=str(tracklet_id),
        detection_id=str(detection_id),
        frame_index=int(frame_index),
        timestamp_s=float(timestamp_s),
        crop_ref=str(crop_ref),
        quality=_clamp01(float(quality)),
        signature=tuple(float(v) for v in signature),
    )


def build_kit_family_profiles(
    samples: Iterable[KitFamilySample],
) -> tuple[KitFamilyProfile, ...]:
    by_tracklet: dict[str, list[KitFamilySample]] = {}
    for sample in samples:
        by_tracklet.setdefault(sample.tracklet_id, []).append(sample)

    result: list[KitFamilyProfile] = []
    for tracklet_id in sorted(by_tracklet):
        rows = sorted(
            by_tracklet[tracklet_id],
            key=lambda item: (
                item.timestamp_s,
                item.frame_index,
                item.detection_id,
            ),
        )
        vectors = np.vstack([
            _unit(row.signature, label=row.detection_id)
            for row in rows
        ])
        quality = np.asarray(
            [max(0.05, row.quality) for row in rows],
            dtype=np.float64,
        )
        weights = quality / np.sum(quality)
        centroid = _unit(
            np.sum(vectors * weights[:, None], axis=0),
            label=f"{tracklet_id} family profile",
        )
        similarity = vectors @ centroid
        consistency = _clamp01(
            (float(np.mean(similarity)) + 1.0) * 0.5
        )
        support = 1.0 - math.exp(-len(rows) / 2.0)
        reliability = _clamp01(
            float(np.mean(quality)) * consistency * support
        )
        result.append(
            KitFamilyProfile(
                tracklet_id=tracklet_id,
                signature=tuple(float(v) for v in centroid),
                sample_count=len(rows),
                reliability=reliability,
                first_timestamp_s=float(rows[0].timestamp_s),
                last_timestamp_s=float(rows[-1].timestamp_s),
            )
        )
    return tuple(result)


def _profile_weight(profile: KitFamilyProfile) -> float:
    return max(
        1e-6,
        profile.reliability * math.sqrt(max(1, profile.sample_count)),
    )


def _initial_two_centroids(
    vectors: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    first = int(np.argmax(weights))
    similarity = vectors @ vectors[first]
    normalized_weight = weights / max(float(np.max(weights)), 1e-12)
    score = (1.0 - similarity) * (0.25 + 0.75 * normalized_weight)
    score[first] = -1.0
    second = int(np.argmax(score))
    return vectors[[first, second]].copy()


def _trimmed_two_primary_fit(
    vectors: np.ndarray,
    weights: np.ndarray,
    config: KitFamilyConfig,
) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    centroids = _initial_two_centroids(vectors, weights)
    labels = np.zeros(vectors.shape[0], dtype=np.int64)
    cores: tuple[np.ndarray, np.ndarray] = (
        np.asarray([], dtype=np.int64),
        np.asarray([], dtype=np.int64),
    )

    for _ in range(config.primary_max_iterations):
        similarity = vectors @ centroids.T
        new_labels = np.argmax(similarity, axis=1).astype(np.int64)
        new_centroids = centroids.copy()
        new_cores: list[np.ndarray] = []

        for cluster_id in range(2):
            members = np.flatnonzero(new_labels == cluster_id)
            if members.size == 0:
                raise ValueError("empty primary family during robust fit")

            member_similarity = similarity[members, cluster_id]
            order = members[
                np.argsort(member_similarity)[::-1]
            ]
            take = max(
                min(config.min_primary_core_count, members.size),
                int(math.ceil(
                    members.size * config.primary_core_fraction
                )),
            )
            take = min(take, members.size)
            core = np.sort(order[:take])
            core_weights = weights[core]
            centroid = np.sum(
                vectors[core] * core_weights[:, None],
                axis=0,
            )
            new_centroids[cluster_id] = _unit(
                centroid,
                label=f"primary-{cluster_id}",
            )
            new_cores.append(core)

        movement = float(
            np.max(np.linalg.norm(new_centroids - centroids, axis=1))
        )
        stable = np.array_equal(labels, new_labels)
        centroids = new_centroids
        labels = new_labels
        cores = (new_cores[0], new_cores[1])
        if stable and movement <= 1e-7:
            break

    masses = [
        float(np.sum(weights[cores[i]]))
        for i in range(2)
    ]
    order = sorted(range(2), key=lambda i: -masses[i])
    if order != [0, 1]:
        centroids = centroids[order]
        remap = {old: new for new, old in enumerate(order)}
        labels = np.asarray(
            [remap[int(v)] for v in labels],
            dtype=np.int64,
        )
        cores = tuple(cores[old] for old in order)  # type: ignore[assignment]

    return centroids, labels, cores


def _primary_thresholds(
    vectors: np.ndarray,
    centroids: np.ndarray,
    cores: tuple[np.ndarray, np.ndarray],
    config: KitFamilyConfig,
) -> tuple[tuple[float, float], tuple[float, float]]:
    sim_thresholds = []
    margin_thresholds = []
    similarity = vectors @ centroids.T
    for cluster_id in range(2):
        core = cores[cluster_id]
        own = similarity[core, cluster_id]
        other = similarity[core, 1 - cluster_id]
        margin = own - other
        sim_thresholds.append(float(
            np.quantile(own, config.primary_similarity_quantile)
            * config.primary_similarity_scale
        ))
        margin_thresholds.append(max(
            0.0,
            float(
                np.quantile(
                    margin,
                    config.primary_margin_quantile,
                )
                * config.primary_margin_scale
            ),
        ))
    return (
        (sim_thresholds[0], sim_thresholds[1]),
        (margin_thresholds[0], margin_thresholds[1]),
    )


def _spherical_kmeans(
    vectors: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    chosen = [int(np.argmax(weights))]
    while len(chosen) < k:
        similarity = vectors @ vectors[chosen].T
        best = np.max(similarity, axis=1)
        score = (1.0 - best) * (
            0.25 + 0.75 * weights / max(float(np.max(weights)), 1e-12)
        )
        score[np.asarray(chosen)] = -1.0
        chosen.append(int(np.argmax(score)))
    centroids = vectors[chosen].copy()
    labels = np.full(vectors.shape[0], -1, dtype=np.int64)

    for _ in range(80):
        similarity = vectors @ centroids.T
        new_labels = np.argmax(similarity, axis=1).astype(np.int64)
        new_centroids = centroids.copy()
        for cluster_id in range(k):
            members = np.flatnonzero(new_labels == cluster_id)
            if members.size == 0:
                return centroids, labels
            centroid = np.sum(
                vectors[members] * weights[members, None],
                axis=0,
            )
            new_centroids[cluster_id] = _unit(
                centroid,
                label=f"special-{cluster_id}",
            )
        stable = np.array_equal(labels, new_labels)
        centroids = new_centroids
        labels = new_labels
        if stable:
            break
    return centroids, labels


def _silhouette(
    vectors: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> float:
    if vectors.shape[0] <= k:
        return -1.0
    distance = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
    scores = np.zeros(vectors.shape[0], dtype=np.float64)
    groups = [np.flatnonzero(labels == i) for i in range(k)]
    for index in range(vectors.shape[0]):
        own = int(labels[index])
        same = groups[own]
        same = same[same != index]
        if same.size == 0:
            continue
        a = float(np.average(distance[index, same], weights=weights[same]))
        b = math.inf
        for group_id, members in enumerate(groups):
            if group_id == own or members.size == 0:
                continue
            b = min(
                b,
                float(np.average(
                    distance[index, members],
                    weights=weights[members],
                )),
            )
        denom = max(a, b)
        if math.isfinite(b) and denom > 1e-12:
            scores[index] = (b - a) / denom
    return float(np.average(scores, weights=weights))


def cluster_kit_families(
    profiles: Iterable[KitFamilyProfile],
    config: KitFamilyConfig | None = None,
) -> KitFamilyResult:
    selected = config or KitFamilyConfig()
    ordered = tuple(sorted(profiles, key=lambda p: p.tracklet_id))
    if len(ordered) < 4:
        raise ValueError("not enough profiles")

    vectors = np.vstack([
        _unit(p.signature, label=p.tracklet_id)
        for p in ordered
    ])
    weights = np.asarray(
        [_profile_weight(p) for p in ordered],
        dtype=np.float64,
    )

    primary_centroids, _, primary_cores = _trimmed_two_primary_fit(
        vectors,
        weights,
        selected,
    )
    sim_thresholds, margin_thresholds = _primary_thresholds(
        vectors,
        primary_centroids,
        primary_cores,
        selected,
    )

    primary_similarity = vectors @ primary_centroids.T
    primary_best_id = np.argmax(primary_similarity, axis=1)
    primary_best = np.max(primary_similarity, axis=1)
    primary_second = np.min(primary_similarity, axis=1)
    primary_margin = primary_best - primary_second

    state: list[str] = []
    family_type: list[str] = []
    family_id: list[int | None] = []

    residual_indices: list[int] = []
    for index, profile in enumerate(ordered):
        best_id = int(primary_best_id[index])
        if profile.reliability < selected.min_profile_reliability_for_family:
            state.append("low_reliability")
            family_type.append("unknown")
            family_id.append(None)
            continue

        sim_ok = (
            float(primary_best[index])
            >= sim_thresholds[best_id]
        )
        margin_ok = (
            float(primary_margin[index])
            >= margin_thresholds[best_id]
        )
        if sim_ok and margin_ok:
            state.append("confident_primary")
            family_type.append("primary")
            family_id.append(best_id)
        else:
            state.append("ambiguous_primary")
            family_type.append("unknown")
            family_id.append(None)
            residual_indices.append(index)

    special_scores: list[dict[str, Any]] = []
    special_centroids = np.empty((0, vectors.shape[1]), dtype=np.float64)
    special_labels = np.asarray([], dtype=np.int64)
    special_source_indices = np.asarray(residual_indices, dtype=np.int64)

    eligible_residual = np.asarray([
        i for i in residual_indices
        if ordered[i].reliability
        >= selected.min_profile_reliability_for_family
    ], dtype=np.int64)
    special_source_indices = eligible_residual

    best_special = None
    if (
        selected.max_special_families > 0
        and eligible_residual.size >= selected.min_special_profiles
    ):
        rv = vectors[eligible_residual]
        rw = weights[eligible_residual]
        total_mass = float(np.sum(weights))

        for k in range(1, selected.max_special_families + 1):
            if rv.shape[0] < k * selected.min_special_profiles:
                continue
            centroids, labels = _spherical_kmeans(rv, rw, k)
            if np.any(labels < 0):
                continue

            counts = [
                int(np.sum(labels == cluster_id))
                for cluster_id in range(k)
            ]
            masses = [
                float(np.sum(rw[labels == cluster_id]))
                for cluster_id in range(k)
            ]
            primary_cos = centroids @ primary_centroids.T
            max_primary_cos = [
                float(np.max(primary_cos[cluster_id]))
                for cluster_id in range(k)
            ]

            valid = all(
                counts[c] >= selected.min_special_profiles
                and masses[c] / max(total_mass, 1e-12)
                >= selected.min_special_mass_fraction
                and max_primary_cos[c]
                <= selected.max_special_primary_cosine
                for c in range(k)
            )
            silhouette = (
                0.0
                if k == 1
                else _silhouette(rv, labels, rw, k)
            )
            if k > 1 and silhouette < selected.min_special_silhouette:
                valid = False

            residual = float(np.average(
                1.0 - np.max(rv @ centroids.T, axis=1),
                weights=rw,
            ))
            row = {
                "k": k,
                "valid": bool(valid),
                "silhouette": silhouette,
                "weighted_residual": residual,
                "cluster_counts": counts,
                "cluster_mass_fractions": [
                    mass / max(total_mass, 1e-12)
                    for mass in masses
                ],
                "max_primary_cosines": max_primary_cos,
            }
            special_scores.append(row)

            if valid:
                key = (
                    silhouette if k > 1 else -residual,
                    -residual,
                    -k,
                )
                if best_special is None or key > best_special[0]:
                    best_special = (
                        key, k, centroids, labels,
                    )

    if best_special is not None:
        _, special_k, special_centroids, special_labels = best_special

        # Canonical special IDs by weighted mass.
        rw = weights[special_source_indices]
        masses = [
            float(np.sum(rw[special_labels == i]))
            for i in range(special_k)
        ]
        order = sorted(range(special_k), key=lambda i: -masses[i])
        remap = {old: new for new, old in enumerate(order)}
        special_centroids = special_centroids[order]
        special_labels = np.asarray(
            [remap[int(v)] for v in special_labels],
            dtype=np.int64,
        )

        for local_index, source_index in enumerate(special_source_indices):
            cluster_id = int(special_labels[local_index])
            special_similarity = vectors[source_index] @ special_centroids.T
            best = float(special_similarity[cluster_id])
            if special_centroids.shape[0] > 1:
                ordered_sim = np.sort(special_similarity)
                second = float(ordered_sim[-2])
                margin = best - second
            else:
                second = None
                margin = None

            # A special assignment must actually prefer the special centroid
            # over both primary families.
            primary_max = float(np.max(
                primary_similarity[source_index]
            ))
            if best > primary_max:
                state[source_index] = "confident_special"
                family_type[source_index] = "special"
                family_id[source_index] = cluster_id
            else:
                state[source_index] = "residual_unknown"

    special_count = (
        int(special_centroids.shape[0])
        if best_special is not None
        else 0
    )

    assignments: list[KitFamilyAssignment] = []
    for index, profile in enumerate(ordered):
        if family_type[index] == "special":
            sid = int(family_id[index])
            sims = vectors[index] @ special_centroids.T
            best = float(sims[sid])
            if special_count > 1:
                sorted_sim = np.sort(sims)
                second = float(sorted_sim[-2])
                margin = best - second
            else:
                second = None
                margin = None
        else:
            best = float(primary_best[index])
            second = float(primary_second[index])
            margin = float(primary_margin[index])

        assignments.append(
            KitFamilyAssignment(
                tracklet_id=profile.tracklet_id,
                family_type=family_type[index],
                family_id=family_id[index],
                state=state[index],
                best_cosine=best,
                second_best_cosine=second,
                margin=margin,
                profile_reliability=profile.reliability,
            )
        )

    clusters: list[KitFamilyCluster] = []
    for cluster_id in range(2):
        members = [
            i for i, row in enumerate(assignments)
            if row.family_type == "primary"
            and row.family_id == cluster_id
        ]
        mass = float(np.sum(weights[members])) if members else 0.0
        cos = [
            float(vectors[i] @ primary_centroids[cluster_id])
            for i in members
        ]
        clusters.append(
            KitFamilyCluster(
                family_type="primary",
                family_id=cluster_id,
                centroid=tuple(
                    float(v) for v in primary_centroids[cluster_id]
                ),
                profile_count=len(members),
                weighted_mass=mass,
                mean_cosine=float(np.mean(cos)) if cos else 0.0,
            )
        )

    for cluster_id in range(special_count):
        members = [
            i for i, row in enumerate(assignments)
            if row.family_type == "special"
            and row.family_id == cluster_id
        ]
        mass = float(np.sum(weights[members])) if members else 0.0
        cos = [
            float(vectors[i] @ special_centroids[cluster_id])
            for i in members
        ]
        clusters.append(
            KitFamilyCluster(
                family_type="special",
                family_id=cluster_id,
                centroid=tuple(
                    float(v) for v in special_centroids[cluster_id]
                ),
                profile_count=len(members),
                weighted_mass=mass,
                mean_cosine=float(np.mean(cos)) if cos else 0.0,
            )
        )

    return KitFamilyResult(
        assignments=tuple(assignments),
        clusters=tuple(clusters),
        primary_similarity_thresholds=sim_thresholds,
        primary_margin_thresholds=margin_thresholds,
        special_family_count=special_count,
        special_model_scores=tuple(special_scores),
        config=selected,
    )
