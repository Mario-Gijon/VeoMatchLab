"""Tracking V1 Hito 7C.1C: adaptive anonymous kit-group discovery.

This diagnostic stage improves on the original K=2-dominated kit clustering
without hard-coding team colours, referee colours, goalkeeper colours, team
count, or role count.

It remains deliberately semantic-free:
* groups are anonymous visual kit groups;
* K is selected per clip from a bounded candidate range;
* minority groups are allowed when they are coherent and improve fit;
* weak / ambiguous assignments may be marked unusable for identity evidence;
* no kit decision is a hard identity cannot-link.

The stage consumes the existing tracklet-level ``KitTrackletProfile`` objects
from Hito 7C.1 and uses deterministic weighted spherical k-means.  Model
selection starts from the strongest silhouette model and may preserve larger K
models when they provide a meaningful residual-fit improvement without
collapsing silhouette, cluster support, or centroid separation.

This is a diagnostic V2.  Downstream identity stages should only consume it
after visual/statistical validation on the clip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from tracking.appearance.kit import KitTrackletProfile


ADAPTIVE_KIT_CLUSTERING_VERSION = "kit-clustering-adaptive-v2"


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


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


@dataclass(frozen=True, slots=True)
class AdaptiveKitClusteringConfig:
    min_k: int = 2
    max_k: int = 8
    max_iterations: int = 100
    convergence_tol: float = 1e-7

    min_anchor_sample_count: int = 2
    min_anchor_reliability: float = 0.45
    min_anchor_count: int = 40

    # Structural safeguards for a candidate K.
    min_cluster_anchor_count: int = 3
    min_cluster_mass_fraction: float = 0.015
    max_centroid_cosine: float = 0.985

    # A larger K may be kept even when pure silhouette prefers a smaller K,
    # but only when it materially improves residual fit and stays close to the
    # best silhouette.
    min_relative_residual_gain: float = 0.08
    max_silhouette_drop: float = 0.08
    min_margin_ratio: float = 0.65

    # Assignment confidence is calibrated from the selected model's own
    # anchor assignments, per anonymous group.  These quantiles are not colour
    # or role assumptions.
    anchor_best_cosine_quantile: float = 0.05
    anchor_margin_quantile: float = 0.10
    assignment_threshold_scale: float = 0.90
    min_assignment_reliability: float = 0.20

    def __post_init__(self) -> None:
        if self.min_k < 2:
            raise ValueError("min_k must be >= 2")
        if self.max_k < self.min_k:
            raise ValueError("max_k must be >= min_k")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.convergence_tol <= 0.0:
            raise ValueError("convergence_tol must be > 0")
        if self.min_anchor_sample_count < 1:
            raise ValueError("min_anchor_sample_count must be >= 1")
        if not 0.0 <= self.min_anchor_reliability <= 1.0:
            raise ValueError("min_anchor_reliability must be in [0,1]")
        if self.min_anchor_count < self.max_k:
            raise ValueError("min_anchor_count must be >= max_k")
        if self.min_cluster_anchor_count < 1:
            raise ValueError("min_cluster_anchor_count must be >= 1")
        if not 0.0 <= self.min_cluster_mass_fraction <= 1.0:
            raise ValueError("min_cluster_mass_fraction must be in [0,1]")
        if not -1.0 <= self.max_centroid_cosine <= 1.0:
            raise ValueError("max_centroid_cosine must be in [-1,1]")
        if self.min_relative_residual_gain < 0.0:
            raise ValueError("min_relative_residual_gain must be >= 0")
        if self.max_silhouette_drop < 0.0:
            raise ValueError("max_silhouette_drop must be >= 0")
        if not 0.0 <= self.min_margin_ratio <= 1.0:
            raise ValueError("min_margin_ratio must be in [0,1]")
        for name in (
            "anchor_best_cosine_quantile",
            "anchor_margin_quantile",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if not 0.0 < self.assignment_threshold_scale <= 1.0:
            raise ValueError(
                "assignment_threshold_scale must be in (0,1]"
            )
        if not 0.0 <= self.min_assignment_reliability <= 1.0:
            raise ValueError(
                "min_assignment_reliability must be in [0,1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveKitModelScore:
    k: int
    weighted_silhouette: float
    mean_best_cosine: float
    weighted_residual: float
    mean_margin: float
    min_anchor_cluster_size: int
    max_anchor_cluster_size: int
    min_cluster_mass_fraction: float
    max_pairwise_centroid_cosine: float
    structurally_valid: bool
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AdaptiveKitSelectionStep:
    from_k: int
    candidate_k: int
    relative_residual_gain: float
    silhouette_drop_from_best: float
    margin_ratio: float
    structurally_valid: bool
    accepted: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reasons"] = list(self.reasons)
        return payload


@dataclass(frozen=True, slots=True)
class AdaptiveKitCluster:
    cluster_id: int
    centroid: tuple[float, ...]
    anchor_profile_count: int
    assigned_profile_count: int
    usable_assignment_count: int
    weighted_anchor_mass: float
    anchor_mass_fraction: float
    mean_best_cosine: float
    median_best_cosine: float
    mean_margin: float
    best_cosine_threshold: float
    margin_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": list(self.centroid),
            "anchor_profile_count": self.anchor_profile_count,
            "assigned_profile_count": self.assigned_profile_count,
            "usable_assignment_count": self.usable_assignment_count,
            "weighted_anchor_mass": self.weighted_anchor_mass,
            "anchor_mass_fraction": self.anchor_mass_fraction,
            "mean_best_cosine": self.mean_best_cosine,
            "median_best_cosine": self.median_best_cosine,
            "mean_margin": self.mean_margin,
            "best_cosine_threshold": self.best_cosine_threshold,
            "margin_threshold": self.margin_threshold,
            "semantic_label": None,
            "team_label": None,
            "role_label": None,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveKitAssignment:
    tracklet_id: str
    cluster_id: int
    best_cosine: float
    second_best_cosine: float
    margin: float
    profile_reliability: float
    sample_count: int
    anchor_used_for_fit: bool
    assignment_state: str
    usable_for_identity: bool
    best_cosine_threshold: float
    margin_threshold: float

    def __post_init__(self) -> None:
        if self.assignment_state not in {
            "confident",
            "ambiguous",
            "outlier",
            "low_reliability",
        }:
            raise ValueError("invalid assignment_state")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "cluster_id": self.cluster_id,
            "best_cosine": self.best_cosine,
            "second_best_cosine": self.second_best_cosine,
            "margin": self.margin,
            "profile_reliability": self.profile_reliability,
            "sample_count": self.sample_count,
            "anchor_used_for_fit": self.anchor_used_for_fit,
            "assignment_state": self.assignment_state,
            "usable_for_identity": self.usable_for_identity,
            "best_cosine_threshold": self.best_cosine_threshold,
            "margin_threshold": self.margin_threshold,
            "semantic_label": None,
            "team_label": None,
            "role_label": None,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveKitClusteringResult:
    selected_k: int
    silhouette_best_k: int
    model_scores: tuple[AdaptiveKitModelScore, ...]
    selection_trace: tuple[AdaptiveKitSelectionStep, ...]
    clusters: tuple[AdaptiveKitCluster, ...]
    assignments: tuple[AdaptiveKitAssignment, ...]
    anchor_tracklet_ids: tuple[str, ...]
    config: AdaptiveKitClusteringConfig

    def summary(self) -> dict[str, Any]:
        state_counts: dict[str, int] = {}
        for item in self.assignments:
            state_counts[item.assignment_state] = (
                state_counts.get(item.assignment_state, 0) + 1
            )
        return {
            "schema_version": 1,
            "adaptive_kit_clustering_version": (
                ADAPTIVE_KIT_CLUSTERING_VERSION
            ),
            "selected_k": self.selected_k,
            "silhouette_best_k": self.silhouette_best_k,
            "profile_count": len(self.assignments),
            "anchor_profile_count": len(self.anchor_tracklet_ids),
            "usable_assignment_count": sum(
                item.usable_for_identity for item in self.assignments
            ),
            "assignment_state_counts": dict(sorted(state_counts.items())),
            "model_scores": [
                item.to_dict() for item in self.model_scores
            ],
            "selection_trace": [
                item.to_dict() for item in self.selection_trace
            ],
            "clusters": [
                item.to_dict() for item in self.clusters
            ],
            "assignment_best_cosine": _stats(
                [item.best_cosine for item in self.assignments]
            ),
            "assignment_margin": _stats(
                [item.margin for item in self.assignments]
            ),
            "config": self.config.to_dict(),
            "semantic_labels_assigned": False,
            "team_labels_assigned": False,
            "role_labels_assigned": False,
            "identity_hard_constraints_created": False,
            "clip_adaptive": True,
        }


def _profile_weight(profile: KitTrackletProfile) -> float:
    return max(
        1e-6,
        float(profile.reliability)
        * math.sqrt(max(1, int(profile.sample_count))),
    )


def _select_anchor_indices(
    profiles: Sequence[KitTrackletProfile],
    config: AdaptiveKitClusteringConfig,
) -> np.ndarray:
    strong = [
        index
        for index, profile in enumerate(profiles)
        if (
            profile.sample_count >= config.min_anchor_sample_count
            and profile.reliability >= config.min_anchor_reliability
        )
    ]
    if len(strong) >= config.min_anchor_count:
        return np.asarray(strong, dtype=np.int64)

    ranked = sorted(
        range(len(profiles)),
        key=lambda index: (
            -profiles[index].reliability,
            -profiles[index].sample_count,
            profiles[index].tracklet_id,
        ),
    )
    take = min(
        len(ranked),
        max(config.min_anchor_count, config.max_k),
    )
    return np.asarray(sorted(ranked[:take]), dtype=np.int64)


def _initial_centroids(
    vectors: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> np.ndarray:
    if k > vectors.shape[0]:
        raise ValueError("k cannot exceed anchor count")

    chosen = [int(np.argmax(weights))]
    while len(chosen) < k:
        similarities = vectors @ vectors[chosen].T
        nearest_similarity = np.max(similarities, axis=1)
        distance = np.clip(1.0 - nearest_similarity, 0.0, 2.0)
        normalized_weight = weights / max(float(np.max(weights)), 1e-12)
        score = distance * (0.25 + 0.75 * normalized_weight)
        score[np.asarray(chosen, dtype=np.int64)] = -1.0
        chosen.append(int(np.argmax(score)))

    return vectors[np.asarray(chosen, dtype=np.int64)].copy()


def _fit_spherical_kmeans(
    vectors: np.ndarray,
    weights: np.ndarray,
    *,
    k: int,
    config: AdaptiveKitClusteringConfig,
) -> tuple[np.ndarray, np.ndarray, int]:
    centroids = _initial_centroids(vectors, weights, k)
    labels = np.full(vectors.shape[0], -1, dtype=np.int64)

    for iteration in range(1, config.max_iterations + 1):
        similarities = vectors @ centroids.T
        new_labels = np.argmax(similarities, axis=1).astype(np.int64)

        new_centroids = np.empty_like(centroids)
        for cluster_id in range(k):
            members = np.flatnonzero(new_labels == cluster_id)
            if members.size == 0:
                best_similarity = np.max(similarities, axis=1)
                candidate_order = np.argsort(best_similarity)
                picked = next(
                    int(index)
                    for index in candidate_order
                    if not any(
                        np.allclose(
                            vectors[index],
                            new_centroids[prior],
                            atol=1e-12,
                            rtol=0.0,
                        )
                        for prior in range(cluster_id)
                    )
                )
                new_centroids[cluster_id] = vectors[picked]
                continue

            cluster_weights = weights[members]
            centroid = np.sum(
                vectors[members] * cluster_weights[:, None],
                axis=0,
            )
            norm = float(np.linalg.norm(centroid))
            new_centroids[cluster_id] = (
                vectors[members[0]]
                if norm <= 1e-12
                else centroid / norm
            )

        movement = float(
            np.max(
                np.linalg.norm(
                    new_centroids - centroids,
                    axis=1,
                )
            )
        )
        stable_labels = np.array_equal(labels, new_labels)
        centroids = new_centroids
        labels = new_labels
        if stable_labels or movement <= config.convergence_tol:
            return centroids, labels, iteration

    return centroids, labels, config.max_iterations


def _weighted_cosine_silhouette(
    vectors: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    k: int,
) -> float:
    distances = np.clip(1.0 - vectors @ vectors.T, 0.0, 2.0)
    scores = np.zeros(vectors.shape[0], dtype=np.float64)
    members_by_cluster = [
        np.flatnonzero(labels == cluster_id)
        for cluster_id in range(k)
    ]

    for index in range(vectors.shape[0]):
        own_cluster = int(labels[index])
        own_members = members_by_cluster[own_cluster]
        own_others = own_members[own_members != index]
        if own_others.size == 0:
            scores[index] = 0.0
            continue

        own_weights = weights[own_others]
        a = float(
            np.sum(distances[index, own_others] * own_weights)
            / np.sum(own_weights)
        )

        b = math.inf
        for cluster_id, members in enumerate(members_by_cluster):
            if cluster_id == own_cluster or members.size == 0:
                continue
            cluster_weights = weights[members]
            mean_distance = float(
                np.sum(distances[index, members] * cluster_weights)
                / np.sum(cluster_weights)
            )
            b = min(b, mean_distance)

        denominator = max(a, b)
        scores[index] = (
            (b - a) / denominator
            if math.isfinite(b) and denominator > 1e-12
            else 0.0
        )

    return float(np.sum(scores * weights) / np.sum(weights))


def _canonicalize_clusters(
    centroids: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masses = [
        float(np.sum(weights[labels == cluster_id]))
        for cluster_id in range(centroids.shape[0])
    ]
    order = sorted(
        range(centroids.shape[0]),
        key=lambda cluster_id: (
            -masses[cluster_id],
            tuple(
                -round(float(value), 12)
                for value in centroids[cluster_id][:8]
            ),
        ),
    )
    old_to_new = {old: new for new, old in enumerate(order)}
    remapped = np.asarray(
        [old_to_new[int(value)] for value in labels],
        dtype=np.int64,
    )
    return centroids[np.asarray(order, dtype=np.int64)], remapped


def _model_score(
    *,
    k: int,
    vectors: np.ndarray,
    weights: np.ndarray,
    centroids: np.ndarray,
    labels: np.ndarray,
    iterations: int,
    config: AdaptiveKitClusteringConfig,
) -> AdaptiveKitModelScore:
    similarities = vectors @ centroids.T
    ordered_scores = np.sort(similarities, axis=1)
    best = ordered_scores[:, -1]
    second = ordered_scores[:, -2]

    cluster_sizes = np.asarray(
        [
            int(np.sum(labels == cluster_id))
            for cluster_id in range(k)
        ],
        dtype=np.int64,
    )
    cluster_mass = np.asarray(
        [
            float(np.sum(weights[labels == cluster_id]))
            for cluster_id in range(k)
        ],
        dtype=np.float64,
    )
    total_mass = float(np.sum(cluster_mass))
    mass_fraction = cluster_mass / max(total_mass, 1e-12)

    centroid_similarity = centroids @ centroids.T
    np.fill_diagonal(centroid_similarity, -1.0)
    max_centroid_cosine = float(np.max(centroid_similarity))

    mean_best = float(np.average(best, weights=weights))
    residual = float(
        np.average(1.0 - best, weights=weights)
    )
    mean_margin = float(
        np.average(best - second, weights=weights)
    )

    structurally_valid = bool(
        int(np.min(cluster_sizes))
        >= config.min_cluster_anchor_count
        and float(np.min(mass_fraction))
        >= config.min_cluster_mass_fraction
        and max_centroid_cosine <= config.max_centroid_cosine
    )

    return AdaptiveKitModelScore(
        k=k,
        weighted_silhouette=_weighted_cosine_silhouette(
            vectors,
            labels,
            weights,
            k,
        ),
        mean_best_cosine=mean_best,
        weighted_residual=residual,
        mean_margin=mean_margin,
        min_anchor_cluster_size=int(np.min(cluster_sizes)),
        max_anchor_cluster_size=int(np.max(cluster_sizes)),
        min_cluster_mass_fraction=float(np.min(mass_fraction)),
        max_pairwise_centroid_cosine=max_centroid_cosine,
        structurally_valid=structurally_valid,
        iterations=iterations,
    )


def _select_model(
    rows: Sequence[
        tuple[
            AdaptiveKitModelScore,
            np.ndarray,
            np.ndarray,
        ]
    ],
    config: AdaptiveKitClusteringConfig,
) -> tuple[
    int,
    int,
    tuple[AdaptiveKitSelectionStep, ...],
]:
    valid_rows = [
        row for row in rows if row[0].structurally_valid
    ]
    pool = valid_rows if valid_rows else list(rows)

    silhouette_best = max(
        pool,
        key=lambda row: (
            row[0].weighted_silhouette,
            -row[0].k,
        ),
    )[0]
    selected = silhouette_best
    trace: list[AdaptiveKitSelectionStep] = []

    for score, _centroids, _labels in sorted(
        rows,
        key=lambda row: row[0].k,
    ):
        if score.k <= selected.k:
            continue

        current_residual = max(
            selected.weighted_residual,
            1e-12,
        )
        relative_gain = max(
            0.0,
            (
                selected.weighted_residual
                - score.weighted_residual
            )
            / current_residual,
        )
        silhouette_drop = max(
            0.0,
            silhouette_best.weighted_silhouette
            - score.weighted_silhouette,
        )
        margin_ratio = (
            score.mean_margin / max(selected.mean_margin, 1e-12)
        )

        reasons: list[str] = []
        if not score.structurally_valid:
            reasons.append("structurally_invalid")
        if relative_gain < config.min_relative_residual_gain:
            reasons.append("residual_gain_below_min")
        if silhouette_drop > config.max_silhouette_drop:
            reasons.append("silhouette_drop_too_large")
        if margin_ratio < config.min_margin_ratio:
            reasons.append("margin_ratio_below_min")

        accepted = not reasons
        trace.append(
            AdaptiveKitSelectionStep(
                from_k=selected.k,
                candidate_k=score.k,
                relative_residual_gain=relative_gain,
                silhouette_drop_from_best=silhouette_drop,
                margin_ratio=margin_ratio,
                structurally_valid=score.structurally_valid,
                accepted=accepted,
                reasons=tuple(reasons),
            )
        )
        if accepted:
            selected = score

    return selected.k, silhouette_best.k, tuple(trace)


def _cluster_thresholds(
    *,
    labels: np.ndarray,
    best: np.ndarray,
    margin: np.ndarray,
    k: int,
    config: AdaptiveKitClusteringConfig,
) -> tuple[tuple[float, float], ...]:
    result: list[tuple[float, float]] = []
    for cluster_id in range(k):
        members = np.flatnonzero(labels == cluster_id)
        cluster_best = best[members]
        cluster_margin = margin[members]
        if members.size == 0:
            result.append((-1.0, math.inf))
            continue

        best_threshold = float(
            np.quantile(
                cluster_best,
                config.anchor_best_cosine_quantile,
            )
            * config.assignment_threshold_scale
        )
        margin_threshold = float(
            np.quantile(
                cluster_margin,
                config.anchor_margin_quantile,
            )
            * config.assignment_threshold_scale
        )
        result.append((best_threshold, margin_threshold))
    return tuple(result)


def cluster_kit_profiles_adaptive(
    profiles: Iterable[KitTrackletProfile],
    config: AdaptiveKitClusteringConfig | None = None,
) -> AdaptiveKitClusteringResult:
    selected = config or AdaptiveKitClusteringConfig()
    ordered = tuple(
        sorted(profiles, key=lambda item: item.tracklet_id)
    )
    if len(ordered) < selected.min_k:
        raise ValueError("not enough kit profiles to cluster")

    ids = [profile.tracklet_id for profile in ordered]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate kit tracklet profile")

    vectors = np.vstack(
        [
            _unit(
                profile.aggregate_signature,
                label=profile.tracklet_id,
            )
            for profile in ordered
        ]
    )
    weights = np.asarray(
        [_profile_weight(profile) for profile in ordered],
        dtype=np.float64,
    )

    anchor_indices = _select_anchor_indices(ordered, selected)
    anchor_vectors = vectors[anchor_indices]
    anchor_weights = weights[anchor_indices]

    max_k = min(selected.max_k, len(anchor_indices))
    if max_k < selected.min_k:
        raise ValueError("not enough anchor profiles for requested K range")

    model_rows: list[
        tuple[
            AdaptiveKitModelScore,
            np.ndarray,
            np.ndarray,
        ]
    ] = []

    for k in range(selected.min_k, max_k + 1):
        centroids, labels, iterations = _fit_spherical_kmeans(
            anchor_vectors,
            anchor_weights,
            k=k,
            config=selected,
        )
        centroids, labels = _canonicalize_clusters(
            centroids,
            labels,
            anchor_weights,
        )
        model_rows.append(
            (
                _model_score(
                    k=k,
                    vectors=anchor_vectors,
                    weights=anchor_weights,
                    centroids=centroids,
                    labels=labels,
                    iterations=iterations,
                    config=selected,
                ),
                centroids,
                labels,
            )
        )

    selected_k, silhouette_best_k, trace = _select_model(
        model_rows,
        selected,
    )

    selected_row = next(
        row for row in model_rows if row[0].k == selected_k
    )
    selected_score, centroids, anchor_labels = selected_row

    anchor_similarity = anchor_vectors @ centroids.T
    anchor_sorted = np.sort(anchor_similarity, axis=1)
    anchor_best = anchor_sorted[:, -1]
    anchor_second = anchor_sorted[:, -2]
    anchor_margin = anchor_best - anchor_second

    thresholds = _cluster_thresholds(
        labels=anchor_labels,
        best=anchor_best,
        margin=anchor_margin,
        k=selected_k,
        config=selected,
    )

    all_similarity = vectors @ centroids.T
    all_labels = np.argmax(all_similarity, axis=1).astype(np.int64)
    all_sorted = np.sort(all_similarity, axis=1)
    all_best = all_sorted[:, -1]
    all_second = all_sorted[:, -2]
    all_margin = all_best - all_second

    anchor_id_set = {
        ordered[int(index)].tracklet_id
        for index in anchor_indices
    }

    assignments: list[AdaptiveKitAssignment] = []
    for index, profile in enumerate(ordered):
        cluster_id = int(all_labels[index])
        best_threshold, margin_threshold = thresholds[cluster_id]
        best_value = float(all_best[index])
        margin_value = float(all_margin[index])

        if profile.reliability < selected.min_assignment_reliability:
            state = "low_reliability"
        elif best_value < best_threshold:
            state = "outlier"
        elif margin_value < margin_threshold:
            state = "ambiguous"
        else:
            state = "confident"

        assignments.append(
            AdaptiveKitAssignment(
                tracklet_id=profile.tracklet_id,
                cluster_id=cluster_id,
                best_cosine=best_value,
                second_best_cosine=float(all_second[index]),
                margin=margin_value,
                profile_reliability=float(profile.reliability),
                sample_count=int(profile.sample_count),
                anchor_used_for_fit=(
                    profile.tracklet_id in anchor_id_set
                ),
                assignment_state=state,
                usable_for_identity=(state == "confident"),
                best_cosine_threshold=best_threshold,
                margin_threshold=margin_threshold,
            )
        )

    total_anchor_mass = float(np.sum(anchor_weights))
    clusters: list[AdaptiveKitCluster] = []
    for cluster_id in range(selected_k):
        anchor_positions = np.flatnonzero(
            anchor_labels == cluster_id
        )
        anchor_original_indices = anchor_indices[anchor_positions]
        anchor_mass = float(
            np.sum(weights[anchor_original_indices])
        )
        assigned = [
            item
            for item in assignments
            if item.cluster_id == cluster_id
        ]
        best_values = [item.best_cosine for item in assigned]
        margin_values = [item.margin for item in assigned]
        best_threshold, margin_threshold = thresholds[cluster_id]

        clusters.append(
            AdaptiveKitCluster(
                cluster_id=cluster_id,
                centroid=tuple(
                    float(value)
                    for value in centroids[cluster_id]
                ),
                anchor_profile_count=int(anchor_positions.size),
                assigned_profile_count=len(assigned),
                usable_assignment_count=sum(
                    item.usable_for_identity
                    for item in assigned
                ),
                weighted_anchor_mass=anchor_mass,
                anchor_mass_fraction=(
                    anchor_mass / max(total_anchor_mass, 1e-12)
                ),
                mean_best_cosine=(
                    float(np.mean(best_values))
                    if best_values else 0.0
                ),
                median_best_cosine=(
                    float(np.median(best_values))
                    if best_values else 0.0
                ),
                mean_margin=(
                    float(np.mean(margin_values))
                    if margin_values else 0.0
                ),
                best_cosine_threshold=best_threshold,
                margin_threshold=margin_threshold,
            )
        )

    return AdaptiveKitClusteringResult(
        selected_k=selected_k,
        silhouette_best_k=silhouette_best_k,
        model_scores=tuple(row[0] for row in model_rows),
        selection_trace=trace,
        clusters=tuple(clusters),
        assignments=tuple(assignments),
        anchor_tracklet_ids=tuple(sorted(anchor_id_set)),
        config=selected,
    )

