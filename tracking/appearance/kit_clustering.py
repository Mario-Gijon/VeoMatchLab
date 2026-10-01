"""Hito 7C.1B: unsupervised clustering of football kit evidence.

This stage groups tracklet-level kit signatures into visual kit clusters.  It
does NOT name those clusters as team A, team B, goalkeeper, referee, etc.

The fit intentionally uses stronger multi-sample tracklet profiles as anchors
when enough are available.  Singleton/weak profiles are still assigned after
the centroids are learned, so evidence coverage remains high without letting
the noisiest fragments determine the cluster geometry.

Clustering uses deterministic weighted spherical k-means with cosine distance.
Several K values are audited; the K with the best weighted cosine silhouette on
anchor profiles is reported as ``suggested_k``.  This is a diagnostic model
selection result, not a semantic team decision.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np

from tracking.appearance.kit import (
    KitTrackletProfile,
)


KIT_CLUSTERING_VERSION = "kit-clustering-v1"


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
class KitClusteringConfig:
    min_k: int = 2
    max_k: int = 8
    max_iterations: int = 100
    convergence_tol: float = 1e-7

    min_anchor_sample_count: int = 2
    min_anchor_reliability: float = 0.45
    min_anchor_count: int = 40

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KitKModelScore:
    k: int
    weighted_silhouette: float
    mean_best_cosine: float
    mean_margin: float
    min_anchor_cluster_size: int
    max_anchor_cluster_size: int
    iterations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KitCluster:
    cluster_id: int
    centroid: tuple[float, ...]
    anchor_profile_count: int
    assigned_profile_count: int
    weighted_anchor_mass: float
    mean_best_cosine: float
    median_best_cosine: float
    mean_margin: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "centroid": list(self.centroid),
            "anchor_profile_count": self.anchor_profile_count,
            "assigned_profile_count": self.assigned_profile_count,
            "weighted_anchor_mass": self.weighted_anchor_mass,
            "mean_best_cosine": self.mean_best_cosine,
            "median_best_cosine": self.median_best_cosine,
            "mean_margin": self.mean_margin,
            "semantic_label": None,
            "team_label": None,
            "role_label": None,
        }


@dataclass(frozen=True, slots=True)
class KitClusterAssignment:
    tracklet_id: str
    cluster_id: int
    best_cosine: float
    second_best_cosine: float
    margin: float
    profile_reliability: float
    sample_count: int
    anchor_used_for_fit: bool

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
            "semantic_label": None,
            "team_label": None,
            "role_label": None,
        }


@dataclass(frozen=True, slots=True)
class KitClusteringResult:
    suggested_k: int
    model_scores: tuple[KitKModelScore, ...]
    clusters: tuple[KitCluster, ...]
    assignments: tuple[KitClusterAssignment, ...]
    anchor_tracklet_ids: tuple[str, ...]
    config: KitClusteringConfig

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kit_clustering_version": KIT_CLUSTERING_VERSION,
            "suggested_k": self.suggested_k,
            "profile_count": len(self.assignments),
            "anchor_profile_count": len(self.anchor_tracklet_ids),
            "model_scores": [
                item.to_dict() for item in self.model_scores
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
        }


def _profile_weight(profile: KitTrackletProfile) -> float:
    # Reliability already contains quality, support and internal consistency.
    # sqrt(sample_count) provides modest extra support without allowing long
    # fragments to dominate centroids.
    return max(
        1e-6,
        float(profile.reliability)
        * math.sqrt(max(1, int(profile.sample_count))),
    )


def _select_anchor_indices(
    profiles: Sequence[KitTrackletProfile],
    config: KitClusteringConfig,
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

    # Deterministic fallback: strongest profiles overall.  This keeps the stage
    # generic for shorter clips while still avoiding singleton/noisy dominance.
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
    count = vectors.shape[0]
    if k > count:
        raise ValueError("k cannot exceed anchor count")

    chosen: list[int] = [int(np.argmax(weights))]
    while len(chosen) < k:
        similarities = vectors @ vectors[chosen].T
        nearest_similarity = np.max(similarities, axis=1)
        distance = np.clip(1.0 - nearest_similarity, 0.0, 2.0)

        # Prefer a distant point with meaningful support.  Existing centroids
        # are forbidden explicitly.
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
    config: KitClusteringConfig,
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
                # Re-seed with the point least represented by its current best
                # centroid.  This is deterministic and prevents dead clusters.
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
            if norm <= 1e-12:
                new_centroids[cluster_id] = vectors[members[0]]
            else:
                new_centroids[cluster_id] = centroid / norm

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
    # N is the anchor subset, not all tracklets.  A dense cosine matrix is
    # small enough here and keeps the metric exact/deterministic.
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
            np.sum(
                distances[index, own_others] * own_weights
            )
            / np.sum(own_weights)
        )

        b = math.inf
        for cluster_id, members in enumerate(members_by_cluster):
            if cluster_id == own_cluster or members.size == 0:
                continue
            cluster_weights = weights[members]
            mean_distance = float(
                np.sum(
                    distances[index, members] * cluster_weights
                )
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
    anchor_labels: np.ndarray,
    anchor_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masses = [
        float(np.sum(anchor_weights[anchor_labels == cluster_id]))
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
    old_to_new = {
        old: new
        for new, old in enumerate(order)
    }
    remapped = np.asarray(
        [old_to_new[int(value)] for value in anchor_labels],
        dtype=np.int64,
    )
    return centroids[np.asarray(order, dtype=np.int64)], remapped


def cluster_kit_profiles(
    profiles: Iterable[KitTrackletProfile],
    config: KitClusteringConfig | None = None,
) -> KitClusteringResult:
    selected = config or KitClusteringConfig()
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
    all_weights = np.asarray(
        [_profile_weight(profile) for profile in ordered],
        dtype=np.float64,
    )

    anchor_indices = _select_anchor_indices(ordered, selected)
    anchor_vectors = vectors[anchor_indices]
    anchor_weights = all_weights[anchor_indices]

    max_k = min(selected.max_k, len(anchor_indices))
    if max_k < selected.min_k:
        raise ValueError("not enough anchor profiles for requested K range")

    model_rows: list[
        tuple[KitKModelScore, np.ndarray, np.ndarray]
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
        similarities = anchor_vectors @ centroids.T
        sorted_scores = np.sort(similarities, axis=1)
        best = sorted_scores[:, -1]
        second = sorted_scores[:, -2]
        sizes = [
            int(np.sum(labels == cluster_id))
            for cluster_id in range(k)
        ]

        score = KitKModelScore(
            k=k,
            weighted_silhouette=_weighted_cosine_silhouette(
                anchor_vectors,
                labels,
                anchor_weights,
                k,
            ),
            mean_best_cosine=float(
                np.average(best, weights=anchor_weights)
            ),
            mean_margin=float(
                np.average(best - second, weights=anchor_weights)
            ),
            min_anchor_cluster_size=min(sizes),
            max_anchor_cluster_size=max(sizes),
            iterations=iterations,
        )
        model_rows.append((score, centroids, labels))

    # Primary criterion: silhouette.  Ties prefer the smaller K so a semantic
    # role is never invented merely because an extra centroid fits noise.
    best_row = max(
        model_rows,
        key=lambda row: (
            row[0].weighted_silhouette,
            -row[0].k,
        ),
    )
    best_score, centroids, anchor_labels = best_row
    suggested_k = best_score.k

    all_similarities = vectors @ centroids.T
    all_labels = np.argmax(all_similarities, axis=1).astype(np.int64)
    sorted_scores = np.sort(all_similarities, axis=1)
    all_best = sorted_scores[:, -1]
    all_second = sorted_scores[:, -2]
    all_margin = all_best - all_second

    anchor_id_set = {
        ordered[int(index)].tracklet_id
        for index in anchor_indices
    }

    assignments = tuple(
        KitClusterAssignment(
            tracklet_id=profile.tracklet_id,
            cluster_id=int(all_labels[index]),
            best_cosine=float(all_best[index]),
            second_best_cosine=float(all_second[index]),
            margin=float(all_margin[index]),
            profile_reliability=float(profile.reliability),
            sample_count=int(profile.sample_count),
            anchor_used_for_fit=(
                profile.tracklet_id in anchor_id_set
            ),
        )
        for index, profile in enumerate(ordered)
    )

    clusters: list[KitCluster] = []
    anchor_index_to_position = {
        int(original_index): anchor_position
        for anchor_position, original_index in enumerate(anchor_indices)
    }

    for cluster_id in range(suggested_k):
        assigned_indices = np.flatnonzero(all_labels == cluster_id)
        anchor_positions = np.flatnonzero(
            anchor_labels == cluster_id
        )
        anchor_original_indices = anchor_indices[anchor_positions]
        anchor_mass = float(
            np.sum(all_weights[anchor_original_indices])
        )
        best_values = [
            float(all_best[index])
            for index in assigned_indices
        ]
        margin_values = [
            float(all_margin[index])
            for index in assigned_indices
        ]
        clusters.append(
            KitCluster(
                cluster_id=cluster_id,
                centroid=tuple(
                    float(value) for value in centroids[cluster_id]
                ),
                anchor_profile_count=int(anchor_positions.size),
                assigned_profile_count=int(assigned_indices.size),
                weighted_anchor_mass=anchor_mass,
                mean_best_cosine=(
                    float(np.mean(best_values))
                    if best_values
                    else 0.0
                ),
                median_best_cosine=(
                    float(np.median(best_values))
                    if best_values
                    else 0.0
                ),
                mean_margin=(
                    float(np.mean(margin_values))
                    if margin_values
                    else 0.0
                ),
            )
        )

    return KitClusteringResult(
        suggested_k=suggested_k,
        model_scores=tuple(
            row[0] for row in model_rows
        ),
        clusters=tuple(clusters),
        assignments=assignments,
        anchor_tracklet_ids=tuple(sorted(anchor_id_set)),
        config=selected,
    )
