"""Tracklet-level appearance evidence for Tracking V1.

This module consumes already-selected ReID samples (normally the 256-D global
embedding from the SoccerNet PRTReID checkpoint) and converts them into two
stable, solver-facing contracts:

1. ``TrackletAppearanceProfile``: robust appearance summary for one tracklet.
2. ``TrackletPairAppearanceEvidence``: raw, continuous evidence between two
   tracklets.

Important: this module never decides that two tracklets are the same identity.
There is deliberately no merge threshold and no identity assignment here.
Geometry, coexistence, motion and global optimization belong to the future
identity solver.

Aggregation is robust without hard-rejecting outliers:
- every sample embedding is L2-normalized;
- a quality-weighted medoid is selected;
- samples that disagree with the medoid are smoothly down-weighted;
- a second centroid pass refines the weights;
- individual embeddings, quality, centroid agreement and final aggregation
  weight are preserved in the published profile.

All similarities are cosine similarities.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


DEFAULT_MODEL_ID = "prtreid-soccernet-baseline-global-v1"
DEFAULT_DESCRIPTOR_NAME = "reid"


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_vector(values: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} has zero norm")
    return vector / norm


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    value = float(np.dot(a, b))
    return float(np.clip(value, -1.0, 1.0))


@dataclass(frozen=True, slots=True)
class DistributionStats:
    count: int
    min: float | None = None
    p10: float | None = None
    median: float | None = None
    p90: float | None = None
    p95: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None

    @classmethod
    def from_values(cls, values: Iterable[float]) -> "DistributionStats":
        array = np.asarray(list(values), dtype=np.float64)
        if array.size == 0:
            return cls(count=0)
        if array.ndim != 1 or not np.all(np.isfinite(array)):
            raise ValueError("distribution values must be a finite 1-D sequence")
        return cls(
            count=int(array.size),
            min=float(np.min(array)),
            p10=float(np.percentile(array, 10)),
            median=float(np.median(array)),
            p90=float(np.percentile(array, 90)),
            p95=float(np.percentile(array, 95)),
            max=float(np.max(array)),
            mean=float(np.mean(array)),
            std=float(np.std(array)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "min": self.min,
            "p10": self.p10,
            "median": self.median,
            "p90": self.p90,
            "p95": self.p95,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
        }


@dataclass(frozen=True, slots=True)
class AppearanceEvidenceConfig:
    descriptor_name: str = DEFAULT_DESCRIPTOR_NAME
    model_id: str = DEFAULT_MODEL_ID
    medoid_agreement_floor: float = 0.15
    agreement_power: float = 2.0

    def __post_init__(self) -> None:
        if not self.descriptor_name.strip():
            raise ValueError("descriptor_name must not be empty")
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not (0.0 <= self.medoid_agreement_floor <= 1.0):
            raise ValueError("medoid_agreement_floor must be in [0, 1]")
        if self.agreement_power <= 0:
            raise ValueError("agreement_power must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "descriptor_name": self.descriptor_name,
            "model_id": self.model_id,
            "medoid_agreement_floor": self.medoid_agreement_floor,
            "agreement_power": self.agreement_power,
        }


@dataclass(frozen=True, slots=True)
class AppearanceEvidenceSample:
    sample_id: str
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    quality: float
    embedding: tuple[float, ...]
    crop_ref: str | None = None
    view_name: str | None = None

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id must not be empty")
        if not self.tracklet_id:
            raise ValueError("tracklet_id must not be empty")
        if not self.detection_id:
            raise ValueError("detection_id must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be >= 0")
        if not (0.0 <= self.quality <= 1.0):
            raise ValueError("quality must be in [0, 1]")
        _unit_vector(self.embedding, label=f"embedding for {self.sample_id}")

    @property
    def embedding_dim(self) -> int:
        return len(self.embedding)


@dataclass(frozen=True, slots=True)
class ProfileSample:
    sample_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    quality: float
    similarity_to_centroid: float
    aggregation_weight: float
    embedding: tuple[float, ...]
    crop_ref: str | None = None
    view_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "quality": self.quality,
            "similarity_to_centroid": self.similarity_to_centroid,
            "aggregation_weight": self.aggregation_weight,
            "embedding": list(self.embedding),
            "crop_ref": self.crop_ref,
            "view_name": self.view_name,
        }


@dataclass(frozen=True, slots=True)
class TrackletAppearanceProfile:
    tracklet_id: str
    model_id: str
    descriptor_name: str
    embedding_dim: int
    aggregate_embedding: tuple[float, ...]
    medoid_sample_id: str
    samples: tuple[ProfileSample, ...]
    quality_stats: DistributionStats
    centroid_similarity: DistributionStats
    internal_pair_similarity: DistributionStats
    effective_sample_count: float
    dispersion: float
    reliability: float
    first_timestamp_s: float
    last_timestamp_s: float

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tracklet_id": self.tracklet_id,
            "model_id": self.model_id,
            "descriptor_name": self.descriptor_name,
            "embedding_dim": self.embedding_dim,
            "sample_count": self.sample_count,
            "effective_sample_count": self.effective_sample_count,
            "aggregate_embedding": list(self.aggregate_embedding),
            "medoid_sample_id": self.medoid_sample_id,
            "quality": self.quality_stats.to_dict(),
            "centroid_similarity": self.centroid_similarity.to_dict(),
            "internal_pair_similarity": self.internal_pair_similarity.to_dict(),
            "dispersion": self.dispersion,
            "reliability": self.reliability,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
            "samples": [sample.to_dict() for sample in self.samples],
        }


@dataclass(frozen=True, slots=True)
class TrackletPairAppearanceEvidence:
    tracklet_a: str
    tracklet_b: str
    model_id: str
    descriptor_name: str
    sample_count_a: int
    sample_count_b: int
    effective_sample_count_a: float
    effective_sample_count_b: float
    aggregate_cosine: float
    cross_similarity: DistributionStats
    bidirectional_best_similarity: DistributionStats
    support_quality: float
    reliability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "model_id": self.model_id,
            "descriptor_name": self.descriptor_name,
            "sample_count_a": self.sample_count_a,
            "sample_count_b": self.sample_count_b,
            "effective_sample_count_a": self.effective_sample_count_a,
            "effective_sample_count_b": self.effective_sample_count_b,
            "cross_pair_count": self.cross_similarity.count,
            "aggregate_cosine": self.aggregate_cosine,
            "cross_similarity": self.cross_similarity.to_dict(),
            "bidirectional_best_similarity": self.bidirectional_best_similarity.to_dict(),
            "support_quality": self.support_quality,
            "reliability": self.reliability,
        }


def _descriptor_from_row(
    row: Mapping[str, Any],
    *,
    descriptor_name: str,
    model_id: str,
) -> Mapping[str, Any]:
    descriptors = row.get("descriptors")
    if not isinstance(descriptors, list):
        raise ValueError("embedding row must contain a descriptors list")
    matches = [
        descriptor
        for descriptor in descriptors
        if isinstance(descriptor, Mapping)
        and descriptor.get("name") == descriptor_name
        and descriptor.get("model_id") == model_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one descriptor name={descriptor_name!r} "
            f"model_id={model_id!r}, got {len(matches)}"
        )
    return matches[0]


def sample_from_embedding_row(
    row: Mapping[str, Any],
    config: AppearanceEvidenceConfig,
) -> AppearanceEvidenceSample:
    descriptor = _descriptor_from_row(
        row,
        descriptor_name=config.descriptor_name,
        model_id=config.model_id,
    )
    unit = _unit_vector(
        descriptor.get("values", []),
        label=f"descriptor {row.get('sample_id', '<unknown>')}",
    )
    return AppearanceEvidenceSample(
        sample_id=str(row["sample_id"]),
        tracklet_id=str(row["tracklet_id"]),
        detection_id=str(row["detection_id"]),
        frame_index=int(row["frame_index"]),
        timestamp_s=_finite_float("timestamp_s", row["timestamp_s"]),
        quality=_finite_float("quality", row["quality"]),
        embedding=tuple(float(value) for value in unit),
        crop_ref=(str(row["crop_ref"]) if row.get("crop_ref") is not None else None),
        view_name=(str(row["view_name"]) if row.get("view_name") is not None else None),
    )


def iter_embedding_jsonl(
    path: str | Path,
    config: AppearanceEvidenceConfig,
) -> Iterator[AppearanceEvidenceSample]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source_path}:{line_number} is not a JSON object")
            try:
                yield sample_from_embedding_row(payload, config)
            except Exception as exc:
                raise ValueError(f"{source_path}:{line_number}: {exc}") from exc


def _quality_weighted_medoid(vectors: np.ndarray, qualities: np.ndarray) -> int:
    if vectors.shape[0] == 1:
        return 0
    similarities = vectors @ vectors.T
    scores = []
    for index in range(vectors.shape[0]):
        mask = np.ones(vectors.shape[0], dtype=bool)
        mask[index] = False
        denominator = float(np.sum(qualities[mask]))
        if denominator <= 1e-12:
            score = float(np.mean(similarities[index, mask]))
        else:
            score = float(np.sum(similarities[index, mask] * qualities[mask]) / denominator)
        scores.append(score)
    # deterministic: np.argmax chooses the first sample on ties
    return int(np.argmax(np.asarray(scores, dtype=np.float64)))


def _smooth_agreement_weights(
    similarities: np.ndarray,
    qualities: np.ndarray,
    config: AppearanceEvidenceConfig,
) -> np.ndarray:
    # Convert cosine [-1,1] to agreement [0,1].  Keep a non-zero floor so a
    # visually unusual sample is down-weighted, never hard-deleted.
    agreement = np.clip((similarities + 1.0) * 0.5, 0.0, 1.0)
    shaped = np.power(agreement, config.agreement_power)
    factor = config.medoid_agreement_floor + (1.0 - config.medoid_agreement_floor) * shaped
    weights = qualities * factor
    total = float(np.sum(weights))
    if total <= 1e-12:
        return np.full_like(weights, 1.0 / len(weights))
    return weights / total


def build_tracklet_profile(
    samples: Sequence[AppearanceEvidenceSample],
    config: AppearanceEvidenceConfig,
) -> TrackletAppearanceProfile:
    if not samples:
        raise ValueError("cannot build a profile without samples")

    ordered = sorted(
        samples,
        key=lambda sample: (sample.timestamp_s, sample.frame_index, sample.sample_id),
    )
    tracklet_ids = {sample.tracklet_id for sample in ordered}
    if len(tracklet_ids) != 1:
        raise ValueError("all samples in a profile must belong to one tracklet")

    dims = {sample.embedding_dim for sample in ordered}
    if len(dims) != 1:
        raise ValueError("all sample embeddings in a profile must share a dimension")
    embedding_dim = next(iter(dims))

    vectors = np.vstack(
        [_unit_vector(sample.embedding, label=sample.sample_id) for sample in ordered]
    )
    qualities = np.asarray([sample.quality for sample in ordered], dtype=np.float64)

    medoid_index = _quality_weighted_medoid(vectors, qualities)
    medoid = vectors[medoid_index]
    medoid_similarities = np.clip(vectors @ medoid, -1.0, 1.0)
    weights = _smooth_agreement_weights(medoid_similarities, qualities, config)

    centroid = np.sum(vectors * weights[:, None], axis=0)
    centroid = _unit_vector(centroid, label="first-pass aggregate embedding")

    # A second pass makes the result less dependent on the chosen medoid while
    # preserving the same smooth/no-hard-reject policy.
    centroid_similarities = np.clip(vectors @ centroid, -1.0, 1.0)
    final_weights = _smooth_agreement_weights(centroid_similarities, qualities, config)
    centroid = np.sum(vectors * final_weights[:, None], axis=0)
    centroid = _unit_vector(centroid, label="aggregate embedding")
    centroid_similarities = np.clip(vectors @ centroid, -1.0, 1.0)

    pair_values: list[float] = []
    for i, j in itertools.combinations(range(len(ordered)), 2):
        pair_values.append(_cosine(vectors[i], vectors[j]))

    effective_sample_count = float(1.0 / np.sum(np.square(final_weights)))
    centroid_stats = DistributionStats.from_values(centroid_similarities.tolist())
    internal_stats = DistributionStats.from_values(pair_values)
    quality_stats = DistributionStats.from_values(qualities.tolist())

    centroid_mean = float(centroid_stats.mean if centroid_stats.mean is not None else 1.0)
    dispersion = float(max(0.0, 1.0 - centroid_mean))
    internal_consistency = float(np.clip((centroid_mean + 1.0) * 0.5, 0.0, 1.0))
    support_factor = float(1.0 - math.exp(-effective_sample_count / 2.0))
    mean_quality = float(quality_stats.mean if quality_stats.mean is not None else 0.0)
    reliability = float(np.clip(mean_quality * internal_consistency * support_factor, 0.0, 1.0))

    profile_samples = tuple(
        ProfileSample(
            sample_id=sample.sample_id,
            detection_id=sample.detection_id,
            frame_index=sample.frame_index,
            timestamp_s=sample.timestamp_s,
            quality=sample.quality,
            similarity_to_centroid=float(centroid_similarities[index]),
            aggregation_weight=float(final_weights[index]),
            embedding=tuple(float(value) for value in vectors[index]),
            crop_ref=sample.crop_ref,
            view_name=sample.view_name,
        )
        for index, sample in enumerate(ordered)
    )

    return TrackletAppearanceProfile(
        tracklet_id=ordered[0].tracklet_id,
        model_id=config.model_id,
        descriptor_name=config.descriptor_name,
        embedding_dim=embedding_dim,
        aggregate_embedding=tuple(float(value) for value in centroid),
        medoid_sample_id=ordered[medoid_index].sample_id,
        samples=profile_samples,
        quality_stats=quality_stats,
        centroid_similarity=centroid_stats,
        internal_pair_similarity=internal_stats,
        effective_sample_count=effective_sample_count,
        dispersion=dispersion,
        reliability=reliability,
        first_timestamp_s=float(ordered[0].timestamp_s),
        last_timestamp_s=float(ordered[-1].timestamp_s),
    )


def build_tracklet_profiles(
    samples: Iterable[AppearanceEvidenceSample],
    config: AppearanceEvidenceConfig,
) -> list[TrackletAppearanceProfile]:
    grouped: dict[str, list[AppearanceEvidenceSample]] = {}
    seen_sample_ids: set[str] = set()

    for sample in samples:
        if sample.sample_id in seen_sample_ids:
            raise ValueError(f"duplicate sample_id: {sample.sample_id}")
        seen_sample_ids.add(sample.sample_id)
        grouped.setdefault(sample.tracklet_id, []).append(sample)

    return [
        build_tracklet_profile(grouped[tracklet_id], config)
        for tracklet_id in sorted(grouped)
    ]


def compare_tracklet_profiles(
    profile_a: TrackletAppearanceProfile,
    profile_b: TrackletAppearanceProfile,
) -> TrackletPairAppearanceEvidence:
    if profile_a.tracklet_id == profile_b.tracklet_id:
        raise ValueError("pair evidence requires two different tracklets")
    if profile_a.model_id != profile_b.model_id:
        raise ValueError("cannot compare profiles from different models")
    if profile_a.descriptor_name != profile_b.descriptor_name:
        raise ValueError("cannot compare profiles from different descriptors")
    if profile_a.embedding_dim != profile_b.embedding_dim:
        raise ValueError("cannot compare profiles with different dimensions")

    # Canonical pair ordering makes the serialized edge deterministic and
    # ensures compare(A,B) == compare(B,A).
    if profile_b.tracklet_id < profile_a.tracklet_id:
        profile_a, profile_b = profile_b, profile_a

    aggregate_a = _unit_vector(profile_a.aggregate_embedding, label=profile_a.tracklet_id)
    aggregate_b = _unit_vector(profile_b.aggregate_embedding, label=profile_b.tracklet_id)
    aggregate_cosine = _cosine(aggregate_a, aggregate_b)

    vectors_a = np.vstack(
        [_unit_vector(sample.embedding, label=sample.sample_id) for sample in profile_a.samples]
    )
    vectors_b = np.vstack(
        [_unit_vector(sample.embedding, label=sample.sample_id) for sample in profile_b.samples]
    )
    cross = np.clip(vectors_a @ vectors_b.T, -1.0, 1.0)
    cross_stats = DistributionStats.from_values(cross.reshape(-1).tolist())

    # Each sample asks "what is my best-supported counterpart in the other
    # tracklet?".  Combining A->B and B->A avoids a one-sided lucky crop from
    # dominating the evidence.
    best_a_to_b = np.max(cross, axis=1)
    best_b_to_a = np.max(cross, axis=0)
    bidirectional_best = np.concatenate([best_a_to_b, best_b_to_a])
    best_stats = DistributionStats.from_values(bidirectional_best.tolist())

    mean_quality_a = float(profile_a.quality_stats.mean or 0.0)
    mean_quality_b = float(profile_b.quality_stats.mean or 0.0)
    support_quality = float(math.sqrt(max(0.0, mean_quality_a * mean_quality_b)))
    reliability = float(math.sqrt(max(0.0, profile_a.reliability * profile_b.reliability)))

    return TrackletPairAppearanceEvidence(
        tracklet_a=profile_a.tracklet_id,
        tracklet_b=profile_b.tracklet_id,
        model_id=profile_a.model_id,
        descriptor_name=profile_a.descriptor_name,
        sample_count_a=profile_a.sample_count,
        sample_count_b=profile_b.sample_count,
        effective_sample_count_a=profile_a.effective_sample_count,
        effective_sample_count_b=profile_b.effective_sample_count,
        aggregate_cosine=aggregate_cosine,
        cross_similarity=cross_stats,
        bidirectional_best_similarity=best_stats,
        support_quality=support_quality,
        reliability=reliability,
    )


def build_pairwise_evidence(
    profiles: Sequence[TrackletAppearanceProfile],
) -> list[TrackletPairAppearanceEvidence]:
    ordered = sorted(profiles, key=lambda profile: profile.tracklet_id)
    seen: set[str] = set()
    for profile in ordered:
        if profile.tracklet_id in seen:
            raise ValueError(f"duplicate tracklet profile: {profile.tracklet_id}")
        seen.add(profile.tracklet_id)

    return [
        compare_tracklet_profiles(profile_a, profile_b)
        for profile_a, profile_b in itertools.combinations(ordered, 2)
    ]


def build_evidence_summary(
    profiles: Sequence[TrackletAppearanceProfile],
    pairs: Sequence[TrackletPairAppearanceEvidence],
    *,
    total_tracklet_count: int | None = None,
) -> dict[str, Any]:
    profile_ids = {profile.tracklet_id for profile in profiles}
    sample_count = sum(profile.sample_count for profile in profiles)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "model_id": profiles[0].model_id if profiles else None,
        "descriptor_name": profiles[0].descriptor_name if profiles else None,
        "embedding_dim": profiles[0].embedding_dim if profiles else None,
        "sample_count": sample_count,
        "profile_count": len(profiles),
        "pair_count": len(pairs),
        "profile_sample_count": DistributionStats.from_values(
            [float(profile.sample_count) for profile in profiles]
        ).to_dict(),
        "profile_effective_sample_count": DistributionStats.from_values(
            [profile.effective_sample_count for profile in profiles]
        ).to_dict(),
        "profile_reliability": DistributionStats.from_values(
            [profile.reliability for profile in profiles]
        ).to_dict(),
        "profile_dispersion": DistributionStats.from_values(
            [profile.dispersion for profile in profiles]
        ).to_dict(),
        "pair_aggregate_cosine": DistributionStats.from_values(
            [pair.aggregate_cosine for pair in pairs]
        ).to_dict(),
        "pair_reliability": DistributionStats.from_values(
            [pair.reliability for pair in pairs]
        ).to_dict(),
    }

    if total_tracklet_count is not None:
        if total_tracklet_count < len(profile_ids):
            raise ValueError("total_tracklet_count cannot be smaller than profile_count")
        summary["total_tracklet_count"] = int(total_tracklet_count)
        summary["unprofiled_tracklet_count"] = int(total_tracklet_count - len(profile_ids))
        summary["profile_coverage"] = (
            float(len(profile_ids) / total_tracklet_count)
            if total_tracklet_count
            else None
        )

    return summary
