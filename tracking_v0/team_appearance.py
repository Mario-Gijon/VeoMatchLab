"""Sprint B.1: conservative visual kit grouping from rectified ReID crops.

This module intentionally does NOT perform person ReID yet.

It solves a smaller, safer problem first:

    rectified player crops
        -> colour/kit descriptor per crop
        -> unsupervised visual clusters
        -> two dominant outfield-team groups
        -> SPECIAL / MIXED / UNKNOWN for everything uncertain

Why crop-level clustering instead of one descriptor per raw track?
Because raw GlobalTracker IDs can already contain identity switches.  If track 12
contains red-player crops first and white-player crops later, averaging the whole
track would hide exactly the bug we want to detect.  We therefore cluster crops
first and only then vote at track level.

HOME/AWAY semantics are deliberately NOT guessed from pixels.  The two dominant
outfield groups are named TEAM_A / TEAM_B; a later match-metadata/manual mapping
can map those to HOME / AWAY without changing the vision pipeline.

Likewise, SPECIAL is not automatically called goalkeeper or referee.  That
semantic role is a later conservative layer that can combine appearance with
2D position/behaviour.  UNKNOWN and MIXED are first-class outputs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

import cv2
import numpy as np


EPS = 1e-9


class AppearanceLabel(str, Enum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    SPECIAL = "special"
    MIXED = "mixed"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class KitAppearanceConfig:
    """Descriptor, clustering and conservative track-vote policy."""

    torso_x1: float = 0.18
    torso_x2: float = 0.82
    torso_y1: float = 0.12
    torso_y2: float = 0.60

    lower_x1: float = 0.18
    lower_x2: float = 0.82
    lower_y1: float = 0.52
    lower_y2: float = 0.92

    hue_bins: int = 12
    saturation_bins: int = 6
    value_bins: int = 6

    torso_weight: float = 0.72
    lower_weight: float = 0.28

    # Green-pitch suppression in OpenCV HSV units (H in [0, 179]).
    green_h_min: int = 28
    green_h_max: int = 95
    green_s_min: int = 45
    green_v_min: int = 35
    min_non_grass_pixels: int = 24

    # Automatic crop-cluster selection.
    min_clusters: int = 2
    max_clusters: int = 5
    kmeans_iterations: int = 80
    kmeans_restarts: int = 6
    tiny_cluster_crops: int = 3
    tiny_cluster_penalty: float = 0.08

    # Track-level decision.  A track must have enough crop evidence and one
    # appearance group must dominate.  Otherwise we explicitly keep uncertainty.
    min_track_crops: int = 3
    min_track_purity: float = 0.70
    mixed_second_share: float = 0.22

    # Only clusters with meaningful support across tracks may become one of the
    # two outfield-team groups.
    min_team_distinct_tracks: int = 2

    def __post_init__(self) -> None:
        for name in (
            "torso_x1",
            "torso_x2",
            "torso_y1",
            "torso_y2",
            "lower_x1",
            "lower_x2",
            "lower_y1",
            "lower_y2",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if not self.torso_x1 < self.torso_x2:
            raise ValueError("torso_x1 must be < torso_x2")
        if not self.torso_y1 < self.torso_y2:
            raise ValueError("torso_y1 must be < torso_y2")
        if not self.lower_x1 < self.lower_x2:
            raise ValueError("lower_x1 must be < lower_x2")
        if not self.lower_y1 < self.lower_y2:
            raise ValueError("lower_y1 must be < lower_y2")

        if min(self.hue_bins, self.saturation_bins, self.value_bins) < 2:
            raise ValueError("histogram bin counts must be >= 2")

        if self.torso_weight < 0.0 or self.lower_weight < 0.0:
            raise ValueError("region weights must be >= 0")
        if abs(self.torso_weight + self.lower_weight - 1.0) > 1e-9:
            raise ValueError("torso_weight + lower_weight must equal 1")

        if self.min_clusters < 2:
            raise ValueError("min_clusters must be >= 2")
        if self.max_clusters < self.min_clusters:
            raise ValueError("max_clusters must be >= min_clusters")
        if self.kmeans_iterations < 1 or self.kmeans_restarts < 1:
            raise ValueError("kmeans iterations/restarts must be >= 1")
        if self.tiny_cluster_crops < 1:
            raise ValueError("tiny_cluster_crops must be >= 1")
        if self.tiny_cluster_penalty < 0.0:
            raise ValueError("tiny_cluster_penalty must be >= 0")
        if self.min_track_crops < 1:
            raise ValueError("min_track_crops must be >= 1")
        if not 0.0 < self.min_track_purity <= 1.0:
            raise ValueError("min_track_purity must be in (0, 1]")
        if not 0.0 <= self.mixed_second_share <= 1.0:
            raise ValueError("mixed_second_share must be in [0, 1]")
        if self.min_team_distinct_tracks < 1:
            raise ValueError("min_team_distinct_tracks must be >= 1")


@dataclass(frozen=True)
class CropAppearance:
    track_id: int
    frame_index: int
    quality: float

    descriptor: np.ndarray
    representative_rgb: tuple[int, int, int]

    torso_valid_fraction: float
    lower_valid_fraction: float

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "quality": self.quality,
            "representative_rgb": list(self.representative_rgb),
            "torso_valid_fraction": self.torso_valid_fraction,
            "lower_valid_fraction": self.lower_valid_fraction,
        }


@dataclass(frozen=True)
class AppearanceCluster:
    cluster_id: int
    crop_count: int
    distinct_track_count: int
    support_weight: float
    representative_rgb: tuple[int, int, int]
    ui_color_hex: str
    semantic_label: AppearanceLabel

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["semantic_label"] = self.semantic_label.value
        payload["representative_rgb"] = list(self.representative_rgb)
        return payload


@dataclass(frozen=True)
class TrackAppearanceDecision:
    track_id: int
    label: AppearanceLabel
    cluster_id: int | None

    crop_count: int
    support_weight: float
    top_share: float
    second_share: float

    representative_rgb: tuple[int, int, int] | None
    ui_color_hex: str | None

    cluster_shares: Mapping[int, float]

    @property
    def confident(self) -> bool:
        return self.label not in {
            AppearanceLabel.UNKNOWN,
            AppearanceLabel.MIXED,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "label": self.label.value,
            "cluster_id": self.cluster_id,
            "crop_count": self.crop_count,
            "support_weight": self.support_weight,
            "top_share": self.top_share,
            "second_share": self.second_share,
            "representative_rgb": (
                list(self.representative_rgb)
                if self.representative_rgb is not None
                else None
            ),
            "ui_color_hex": self.ui_color_hex,
            "cluster_shares": {
                str(cluster_id): share
                for cluster_id, share in sorted(self.cluster_shares.items())
            },
            "confident": self.confident,
        }


@dataclass(frozen=True)
class KitAppearanceResult:
    chosen_k: int
    silhouette_score: float

    crop_cluster_ids: tuple[int, ...]
    clusters: tuple[AppearanceCluster, ...]
    track_decisions: Mapping[int, TrackAppearanceDecision]

    def to_dict(self) -> dict[str, object]:
        return {
            "chosen_k": self.chosen_k,
            "silhouette_score": self.silhouette_score,
            "clusters": [
                cluster.to_dict()
                for cluster in self.clusters
            ],
            "track_decisions": {
                str(track_id): decision.to_dict()
                for track_id, decision in sorted(
                    self.track_decisions.items()
                )
            },
        }


def extract_crop_appearance(
    image_bgr: np.ndarray,
    *,
    track_id: int,
    frame_index: int,
    quality: float = 1.0,
    config: KitAppearanceConfig | None = None,
) -> CropAppearance:
    """Extract a colour/kit descriptor from one rectified person crop."""

    cfg = config or KitAppearanceConfig()

    if (
        image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
        or image_bgr.size == 0
    ):
        raise ValueError("image_bgr must be a non-empty HxWx3 image")
    if not 0.0 <= quality <= 1.0:
        raise ValueError("quality must be in [0, 1]")

    torso = _relative_roi(
        image_bgr,
        cfg.torso_x1,
        cfg.torso_y1,
        cfg.torso_x2,
        cfg.torso_y2,
    )
    lower = _relative_roi(
        image_bgr,
        cfg.lower_x1,
        cfg.lower_y1,
        cfg.lower_x2,
        cfg.lower_y2,
    )

    torso_descriptor, torso_rgb, torso_valid = _region_descriptor(
        torso,
        config=cfg,
    )
    lower_descriptor, lower_rgb, lower_valid = _region_descriptor(
        lower,
        config=cfg,
    )

    descriptor = np.concatenate(
        (
            math.sqrt(cfg.torso_weight) * torso_descriptor,
            math.sqrt(cfg.lower_weight) * lower_descriptor,
        )
    ).astype(np.float64, copy=False)

    norm = float(np.linalg.norm(descriptor))
    if not np.isfinite(descriptor).all() or norm <= EPS:
        raise ValueError("appearance descriptor is invalid")
    descriptor = descriptor / norm

    # The torso is a much better proxy for kit/UI colour than shorts/boots.
    # Fall back to the lower body only if the torso contained almost no usable
    # non-grass pixels.
    representative_rgb = (
        torso_rgb
        if torso_valid >= 0.05
        else lower_rgb
    )

    return CropAppearance(
        track_id=int(track_id),
        frame_index=int(frame_index),
        quality=float(quality),
        descriptor=descriptor,
        representative_rgb=representative_rgb,
        torso_valid_fraction=float(torso_valid),
        lower_valid_fraction=float(lower_valid),
    )


def appearance_similarity(
    left: CropAppearance | np.ndarray,
    right: CropAppearance | np.ndarray,
) -> float:
    """Cosine similarity in [0, 1] for normalized kit descriptors."""

    a = left.descriptor if isinstance(left, CropAppearance) else np.asarray(left)
    b = right.descriptor if isinstance(right, CropAppearance) else np.asarray(right)

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("appearance vectors must be equal-length 1D arrays")

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na <= EPS or nb <= EPS:
        raise ValueError("appearance vector norm must be > 0")

    cosine = float(np.dot(a / na, b / nb))
    # Histograms are nonnegative, so negative cosine is not meaningful here.
    return float(min(1.0, max(0.0, cosine)))


def fit_kit_appearance(
    crops: Sequence[CropAppearance],
    *,
    config: KitAppearanceConfig | None = None,
) -> KitAppearanceResult:
    """Cluster crop-level kit appearance and vote conservatively per raw track."""

    cfg = config or KitAppearanceConfig()
    if len(crops) < cfg.min_clusters:
        raise ValueError(
            f"need at least {cfg.min_clusters} crops, got {len(crops)}"
        )

    descriptors = np.stack(
        [crop.descriptor for crop in crops],
        axis=0,
    ).astype(np.float64)

    k_upper = min(
        cfg.max_clusters,
        len(crops),
    )
    if k_upper < cfg.min_clusters:
        raise ValueError("not enough crops for requested clustering range")

    best: tuple[
        float,
        int,
        np.ndarray,
        np.ndarray,
        float,
    ] | None = None

    for k in range(cfg.min_clusters, k_upper + 1):
        labels, centres, inertia = _deterministic_kmeans(
            descriptors,
            k=k,
            iterations=cfg.kmeans_iterations,
            restarts=cfg.kmeans_restarts,
        )
        silhouette = _silhouette_score(
            descriptors,
            labels,
        )

        sizes = np.bincount(
            labels,
            minlength=k,
        )
        tiny = int(
            np.sum(
                sizes < cfg.tiny_cluster_crops
            )
        )
        adjusted = silhouette - cfg.tiny_cluster_penalty * tiny

        candidate = (
            adjusted,
            -k,  # prefer fewer clusters on an exact tie
            labels,
            centres,
            inertia,
        )
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    assert best is not None
    _, negative_k, labels, centres, _ = best
    chosen_k = -negative_k
    silhouette = _silhouette_score(
        descriptors,
        labels,
    )

    cluster_stats = _build_cluster_stats(
        crops,
        labels,
        chosen_k,
    )

    team_cluster_ids = _choose_two_team_clusters(
        cluster_stats,
        config=cfg,
    )

    semantic_by_cluster: dict[int, AppearanceLabel] = {}
    if len(team_cluster_ids) >= 1:
        semantic_by_cluster[team_cluster_ids[0]] = AppearanceLabel.TEAM_A
    if len(team_cluster_ids) >= 2:
        semantic_by_cluster[team_cluster_ids[1]] = AppearanceLabel.TEAM_B

    clusters: list[AppearanceCluster] = []
    for cluster_id in range(chosen_k):
        stats = cluster_stats[cluster_id]
        label = semantic_by_cluster.get(
            cluster_id,
            AppearanceLabel.SPECIAL,
        )
        rgb = stats["representative_rgb"]

        clusters.append(
            AppearanceCluster(
                cluster_id=cluster_id,
                crop_count=stats["crop_count"],
                distinct_track_count=stats["distinct_track_count"],
                support_weight=stats["support_weight"],
                representative_rgb=rgb,
                ui_color_hex=rgb_to_hex(rgb),
                semantic_label=label,
            )
        )

    cluster_by_id = {
        cluster.cluster_id: cluster
        for cluster in clusters
    }

    decisions = _track_decisions(
        crops,
        labels,
        cluster_by_id,
        config=cfg,
    )

    return KitAppearanceResult(
        chosen_k=chosen_k,
        silhouette_score=float(silhouette),
        crop_cluster_ids=tuple(int(value) for value in labels),
        clusters=tuple(clusters),
        track_decisions=decisions,
    )


def rgb_to_hex(rgb: Sequence[int]) -> str:
    if len(rgb) != 3:
        raise ValueError("rgb must contain exactly three channels")
    values = [
        max(0, min(255, int(round(value))))
        for value in rgb
    ]
    return "#" + "".join(f"{value:02X}" for value in values)


def _relative_roi(
    image: np.ndarray,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> np.ndarray:
    h, w = image.shape[:2]

    ix1 = max(0, min(w - 1, int(math.floor(x1 * w))))
    iy1 = max(0, min(h - 1, int(math.floor(y1 * h))))
    ix2 = max(ix1 + 1, min(w, int(math.ceil(x2 * w))))
    iy2 = max(iy1 + 1, min(h, int(math.ceil(y2 * h))))

    return image[iy1:iy2, ix1:ix2]


def _region_descriptor(
    roi_bgr: np.ndarray,
    *,
    config: KitAppearanceConfig,
) -> tuple[np.ndarray, tuple[int, int, int], float]:
    hsv = cv2.cvtColor(
        roi_bgr,
        cv2.COLOR_BGR2HSV,
    )

    pixels_hsv = hsv.reshape(-1, 3)
    pixels_bgr = roi_bgr.reshape(-1, 3)

    h = pixels_hsv[:, 0]
    s = pixels_hsv[:, 1]
    v = pixels_hsv[:, 2]

    green = (
        (h >= config.green_h_min)
        & (h <= config.green_h_max)
        & (s >= config.green_s_min)
        & (v >= config.green_v_min)
    )

    # Black letterbox pixels from aspect-preserving crop normalization are
    # uninformative, while true black kit pixels in the body interior usually
    # have some texture / nonzero value.  Only exact-ish black is discarded.
    padding_black = (
        (pixels_bgr[:, 0] <= 3)
        & (pixels_bgr[:, 1] <= 3)
        & (pixels_bgr[:, 2] <= 3)
    )

    usable = ~(green | padding_black)

    if int(np.count_nonzero(usable)) < config.min_non_grass_pixels:
        usable = ~padding_black

    if int(np.count_nonzero(usable)) < config.min_non_grass_pixels:
        usable = np.ones(
            len(pixels_hsv),
            dtype=bool,
        )

    selected_hsv = pixels_hsv[usable]
    selected_bgr = pixels_bgr[usable]

    hue_hist = np.histogram(
        selected_hsv[:, 0],
        bins=config.hue_bins,
        range=(0, 180),
    )[0].astype(np.float64)
    saturation_hist = np.histogram(
        selected_hsv[:, 1],
        bins=config.saturation_bins,
        range=(0, 256),
    )[0].astype(np.float64)
    value_hist = np.histogram(
        selected_hsv[:, 2],
        bins=config.value_bins,
        range=(0, 256),
    )[0].astype(np.float64)

    hue_hist = _l1_normalize(hue_hist)
    saturation_hist = _l1_normalize(saturation_hist)
    value_hist = _l1_normalize(value_hist)

    # Hellinger / RootSIFT-style transform makes histogram Euclidean distance
    # less dominated by one large bin.
    hist = np.concatenate(
        (
            np.sqrt(hue_hist),
            np.sqrt(saturation_hist),
            np.sqrt(value_hist),
        )
    )

    selected_rgb = selected_bgr[:, ::-1].astype(np.float64)
    median_rgb = np.median(
        selected_rgb,
        axis=0,
    )
    representative_rgb = tuple(
        int(round(value))
        for value in median_rgb
    )

    valid_fraction = float(
        np.count_nonzero(usable)
        / max(1, len(usable))
    )

    norm = float(np.linalg.norm(hist))
    if norm <= EPS:
        raise ValueError("empty region appearance histogram")

    return (
        hist / norm,
        representative_rgb,
        valid_fraction,
    )


def _l1_normalize(vector: np.ndarray) -> np.ndarray:
    total = float(np.sum(vector))
    if total <= EPS:
        return np.full_like(
            vector,
            1.0 / len(vector),
            dtype=np.float64,
        )
    return vector / total


def _deterministic_kmeans(
    data: np.ndarray,
    *,
    k: int,
    iterations: int,
    restarts: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    if data.ndim != 2:
        raise ValueError("data must be a 2D matrix")
    if not 1 < k <= len(data):
        raise ValueError("invalid k")

    best: tuple[float, np.ndarray, np.ndarray] | None = None

    for restart in range(restarts):
        centres = _farthest_point_initialisation(
            data,
            k,
            offset=restart,
        )

        labels = np.zeros(
            len(data),
            dtype=np.int32,
        )

        for _ in range(iterations):
            distances = _squared_distances(
                data,
                centres,
            )
            new_labels = np.argmin(
                distances,
                axis=1,
            ).astype(np.int32)

            new_centres = centres.copy()
            for cluster_id in range(k):
                members = data[
                    new_labels == cluster_id
                ]
                if len(members) == 0:
                    # Re-seed empty cluster at the currently worst represented
                    # point.  Deterministic tie-breaking comes from argmax.
                    closest = np.min(
                        distances,
                        axis=1,
                    )
                    new_centres[cluster_id] = data[
                        int(np.argmax(closest))
                    ]
                else:
                    centre = np.mean(
                        members,
                        axis=0,
                    )
                    norm = float(
                        np.linalg.norm(centre)
                    )
                    if norm > EPS:
                        centre = centre / norm
                    new_centres[cluster_id] = centre

            if np.array_equal(
                labels,
                new_labels,
            ) and np.allclose(
                centres,
                new_centres,
                atol=1e-10,
            ):
                labels = new_labels
                centres = new_centres
                break

            labels = new_labels
            centres = new_centres

        final_distances = _squared_distances(
            data,
            centres,
        )
        inertia = float(
            np.sum(
                final_distances[
                    np.arange(len(data)),
                    labels,
                ]
            )
        )

        if best is None or inertia < best[0]:
            best = (
                inertia,
                labels.copy(),
                centres.copy(),
            )

    assert best is not None
    return best[1], best[2], best[0]


def _farthest_point_initialisation(
    data: np.ndarray,
    k: int,
    *,
    offset: int,
) -> np.ndarray:
    # First seed cycles through high-norm rows across restarts.  Descriptors are
    # normally normalized, so the lexicographic projection provides a stable
    # deterministic spread instead of randomness.
    projection = data @ np.linspace(
        1.0,
        2.0,
        data.shape[1],
        dtype=np.float64,
    )
    order = np.argsort(
        projection,
        kind="mergesort",
    )
    first = int(
        order[
            (offset * max(1, len(order) // max(1, k)))
            % len(order)
        ]
    )

    indices = [first]

    while len(indices) < k:
        centres = data[
            np.asarray(indices)
        ]
        distances = _squared_distances(
            data,
            centres,
        )
        closest = np.min(
            distances,
            axis=1,
        )
        closest[
            np.asarray(indices)
        ] = -1.0
        indices.append(
            int(np.argmax(closest))
        )

    return data[
        np.asarray(indices)
    ].copy()


def _squared_distances(
    data: np.ndarray,
    centres: np.ndarray,
) -> np.ndarray:
    diff = (
        data[:, None, :]
        - centres[None, :, :]
    )
    return np.einsum(
        "nkd,nkd->nk",
        diff,
        diff,
    )


def _silhouette_score(
    data: np.ndarray,
    labels: np.ndarray,
) -> float:
    n = len(data)
    if n < 3:
        return 0.0

    unique = np.unique(labels)
    if len(unique) < 2:
        return 0.0

    distances = np.sqrt(
        np.maximum(
            0.0,
            _pairwise_squared_distances(data),
        )
    )

    values: list[float] = []

    for index in range(n):
        own = labels[index]
        own_indices = np.flatnonzero(
            labels == own
        )
        own_indices = own_indices[
            own_indices != index
        ]

        if len(own_indices) == 0:
            values.append(0.0)
            continue

        a = float(
            np.mean(
                distances[
                    index,
                    own_indices,
                ]
            )
        )

        b = math.inf
        for other in unique:
            if other == own:
                continue
            other_indices = np.flatnonzero(
                labels == other
            )
            if len(other_indices) == 0:
                continue
            b = min(
                b,
                float(
                    np.mean(
                        distances[
                            index,
                            other_indices,
                        ]
                    )
                ),
            )

        if not math.isfinite(b):
            values.append(0.0)
            continue

        denominator = max(a, b)
        values.append(
            0.0
            if denominator <= EPS
            else (b - a) / denominator
        )

    return float(
        np.mean(values)
    )


def _pairwise_squared_distances(
    data: np.ndarray,
) -> np.ndarray:
    diff = (
        data[:, None, :]
        - data[None, :, :]
    )
    return np.einsum(
        "ijd,ijd->ij",
        diff,
        diff,
    )


def _build_cluster_stats(
    crops: Sequence[CropAppearance],
    labels: np.ndarray,
    k: int,
) -> dict[int, dict[str, object]]:
    result: dict[int, dict[str, object]] = {}

    for cluster_id in range(k):
        indices = np.flatnonzero(
            labels == cluster_id
        )
        members = [
            crops[int(index)]
            for index in indices
        ]

        qualities = np.asarray(
            [
                max(EPS, crop.quality)
                for crop in members
            ],
            dtype=np.float64,
        )
        rgbs = np.asarray(
            [
                crop.representative_rgb
                for crop in members
            ],
            dtype=np.float64,
        )

        if len(members) == 0:
            rgb = (127, 127, 127)
            distinct_tracks = 0
            support_weight = 0.0
        else:
            # Weighted coordinate-wise median would be overkill for these tiny
            # arrays; the ordinary median is robust enough to one bad crop.
            median_rgb = np.median(
                rgbs,
                axis=0,
            )
            rgb = tuple(
                int(round(value))
                for value in median_rgb
            )
            distinct_tracks = len(
                {
                    crop.track_id
                    for crop in members
                }
            )
            support_weight = float(
                np.sum(qualities)
            )

        result[cluster_id] = {
            "crop_count": len(members),
            "distinct_track_count": distinct_tracks,
            "support_weight": support_weight,
            "representative_rgb": rgb,
        }

    return result


def _choose_two_team_clusters(
    stats: Mapping[int, Mapping[str, object]],
    *,
    config: KitAppearanceConfig,
) -> tuple[int, ...]:
    eligible = [
        cluster_id
        for cluster_id, row in stats.items()
        if int(row["distinct_track_count"])
        >= config.min_team_distinct_tracks
    ]

    ordered = sorted(
        eligible,
        key=lambda cluster_id: (
            -int(
                stats[cluster_id][
                    "distinct_track_count"
                ]
            ),
            -float(
                stats[cluster_id][
                    "support_weight"
                ]
            ),
            -int(
                stats[cluster_id][
                    "crop_count"
                ]
            ),
            cluster_id,
        ),
    )

    return tuple(
        ordered[:2]
    )


def _track_decisions(
    crops: Sequence[CropAppearance],
    labels: np.ndarray,
    cluster_by_id: Mapping[int, AppearanceCluster],
    *,
    config: KitAppearanceConfig,
) -> dict[int, TrackAppearanceDecision]:
    grouped: dict[int, list[int]] = {}

    for index, crop in enumerate(crops):
        grouped.setdefault(
            crop.track_id,
            [],
        ).append(index)

    decisions: dict[int, TrackAppearanceDecision] = {}

    for track_id, indices in sorted(grouped.items()):
        weights: dict[int, float] = {}
        total_weight = 0.0

        for index in indices:
            cluster_id = int(labels[index])
            weight = max(
                EPS,
                crops[index].quality,
            )
            weights[cluster_id] = (
                weights.get(cluster_id, 0.0)
                + weight
            )
            total_weight += weight

        shares = {
            cluster_id: (
                weight / total_weight
                if total_weight > EPS
                else 0.0
            )
            for cluster_id, weight in weights.items()
        }

        ordered = sorted(
            shares.items(),
            key=lambda item: (
                -item[1],
                item[0],
            ),
        )

        top_cluster = (
            ordered[0][0]
            if ordered
            else None
        )
        top_share = (
            ordered[0][1]
            if ordered
            else 0.0
        )
        second_share = (
            ordered[1][1]
            if len(ordered) > 1
            else 0.0
        )

        if (
            len(indices) < config.min_track_crops
            or top_cluster is None
        ):
            label = AppearanceLabel.UNKNOWN
            chosen_cluster = None
        elif (
            top_share < config.min_track_purity
            or second_share >= config.mixed_second_share
        ):
            label = AppearanceLabel.MIXED
            chosen_cluster = top_cluster
        else:
            cluster = cluster_by_id[
                top_cluster
            ]
            label = cluster.semantic_label
            chosen_cluster = top_cluster

        if chosen_cluster is None:
            representative_rgb = None
            ui_color = None
        else:
            cluster = cluster_by_id[
                chosen_cluster
            ]
            representative_rgb = (
                cluster.representative_rgb
            )
            ui_color = cluster.ui_color_hex

        decisions[track_id] = TrackAppearanceDecision(
            track_id=track_id,
            label=label,
            cluster_id=chosen_cluster,
            crop_count=len(indices),
            support_weight=float(total_weight),
            top_share=float(top_share),
            second_share=float(second_share),
            representative_rgb=representative_rgb,
            ui_color_hex=ui_color,
            cluster_shares=shares,
        )

    return decisions
