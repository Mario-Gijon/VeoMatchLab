"""Hito 7D: robust ReID calibration for Tracking V1.

This module calibrates the existing PRTReID aggregate cosine using proxies
available inside one Veo clip. It does NOT assign identities.

Proxy families
--------------
split_positive
    A conservative tracklet is split chronologically into two halves. The two
    halves are treated as a same-actor proxy. This relies on the V1 principle
    that fragmentation is preferable to identity contamination.

simultaneous_negative
    Two tracklets observed at the same time and separated enough in field
    coordinates to trigger a Hito 7B hard coexistence conflict. These are
    physically impossible to be one actor.

close_overlap
    Co-observed tracklets that remain spatially close. These are diagnostic
    only: they may be duplicate observations of one actor OR two nearby
    footballers. They are never used as positive labels.

The resulting p95 negative / p10 positive anchors are empirical calibration
points for the later solver. They are not probabilities and not merge
thresholds.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tracking.actors.index import (
    SparseTrackletIndexConfig,
    build_sparse_tracklet_index,
)
from tracking.appearance.evidence import (
    AppearanceEvidenceConfig,
    AppearanceEvidenceSample,
    DistributionStats,
    TrackletAppearanceProfile,
    build_tracklet_profile,
    build_tracklet_profiles,
)
from tracking.tracklets.models import Tracklet


REID_CALIBRATION_VERSION = "reid-overlap-calibration-v1"


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding has zero norm")
    return vector / norm


def _cosine(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    a = _unit(first)
    b = _unit(second)
    if a.shape != b.shape:
        raise ValueError("embedding dimensions differ")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _roc_auc(
    positives: Sequence[float],
    negatives: Sequence[float],
) -> float | None:
    """Exact Mann-Whitney AUC without sklearn.

    AUC here means the probability that a randomly chosen positive cosine is
    greater than a randomly chosen negative cosine, with ties worth 0.5.
    """
    if not positives or not negatives:
        return None

    ordered_negatives = sorted(float(v) for v in negatives)
    wins = 0.0

    for value in positives:
        value = float(value)
        lower = bisect_left(ordered_negatives, value)
        upper = bisect_right(ordered_negatives, value)
        wins += lower
        wins += 0.5 * (upper - lower)

    return float(
        wins / (len(positives) * len(ordered_negatives))
    )


@dataclass(frozen=True, slots=True)
class ReIDCalibrationConfig:
    min_split_samples: int = 4

    def __post_init__(self) -> None:
        if self.min_split_samples < 4:
            raise ValueError("min_split_samples must be >= 4")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_split_samples": self.min_split_samples,
        }


@dataclass(frozen=True, slots=True)
class ReIDCalibrationExample:
    family: str
    tracklet_a: str
    tracklet_b: str
    cosine: float
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.family not in {
            "split_positive",
            "simultaneous_negative",
            "close_overlap",
        }:
            raise ValueError("unsupported calibration family")
        if not -1.0 <= self.cosine <= 1.0:
            raise ValueError("cosine must be in [-1, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "calibration_version": REID_CALIBRATION_VERSION,
            "family": self.family,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "cosine": self.cosine,
            "evidence": dict(self.evidence),
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class ReIDOverlapCalibration:
    split_positive: tuple[ReIDCalibrationExample, ...]
    simultaneous_negative: tuple[ReIDCalibrationExample, ...]
    close_overlap: tuple[ReIDCalibrationExample, ...]
    profiled_tracklet_count: int
    unprofiled_tracklet_count: int
    config: ReIDCalibrationConfig

    def summary(self) -> dict[str, Any]:
        positive_values = [
            item.cosine for item in self.split_positive
        ]
        negative_values = [
            item.cosine for item in self.simultaneous_negative
        ]
        close_values = [
            item.cosine for item in self.close_overlap
        ]

        positive_stats = DistributionStats.from_values(
            positive_values
        )
        negative_stats = DistributionStats.from_values(
            negative_values
        )
        close_stats = DistributionStats.from_values(
            close_values
        )

        positive_p10 = positive_stats.p10
        negative_p95 = negative_stats.p95
        margin = (
            float(positive_p10 - negative_p95)
            if positive_p10 is not None
            and negative_p95 is not None
            else None
        )

        return {
            "schema_version": 1,
            "calibration_version": REID_CALIBRATION_VERSION,
            "profiled_tracklet_count": self.profiled_tracklet_count,
            "unprofiled_tracklet_count": self.unprofiled_tracklet_count,
            "split_positive": positive_stats.to_dict(),
            "simultaneous_negative": negative_stats.to_dict(),
            "close_overlap_diagnostic": close_stats.to_dict(),
            "roc_auc": _roc_auc(
                positive_values,
                negative_values,
            ),
            "anchors": {
                "negative_p95": negative_p95,
                "positive_p10": positive_p10,
                "p10_positive_minus_p95_negative": margin,
                "ordered": (
                    bool(margin > 0.0)
                    if margin is not None
                    else None
                ),
                "semantics": (
                    "empirical calibration anchors only; "
                    "not probability and not merge thresholds"
                ),
            },
            "close_overlap_used_as_positive": False,
            "identity_decisions": False,
            "population_forced": False,
            "config": self.config.to_dict(),
        }


def _split_positive_examples(
    samples: Iterable[AppearanceEvidenceSample],
    *,
    evidence_config: AppearanceEvidenceConfig,
    config: ReIDCalibrationConfig,
) -> tuple[ReIDCalibrationExample, ...]:
    grouped: dict[str, list[AppearanceEvidenceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.tracklet_id].append(sample)

    result: list[ReIDCalibrationExample] = []

    for tracklet_id in sorted(grouped):
        ordered = sorted(
            grouped[tracklet_id],
            key=lambda item: (
                item.timestamp_s,
                item.frame_index,
                item.sample_id,
            ),
        )
        if len(ordered) < config.min_split_samples:
            continue

        midpoint = len(ordered) // 2
        left = ordered[:midpoint]
        right = ordered[midpoint:]

        if len(left) < 2 or len(right) < 2:
            continue

        left_profile = build_tracklet_profile(
            left,
            evidence_config,
        )
        right_profile = build_tracklet_profile(
            right,
            evidence_config,
        )

        cosine = _cosine(
            left_profile.aggregate_embedding,
            right_profile.aggregate_embedding,
        )

        result.append(
            ReIDCalibrationExample(
                family="split_positive",
                tracklet_a=tracklet_id,
                tracklet_b=tracklet_id,
                cosine=cosine,
                evidence={
                    "left_sample_count": len(left),
                    "right_sample_count": len(right),
                    "left_first_timestamp_s": left[0].timestamp_s,
                    "left_last_timestamp_s": left[-1].timestamp_s,
                    "right_first_timestamp_s": right[0].timestamp_s,
                    "right_last_timestamp_s": right[-1].timestamp_s,
                    "proxy": "chronological_same_tracklet_split",
                },
            )
        )

    return tuple(result)


def build_reid_overlap_calibration(
    *,
    tracklets: Sequence[Tracklet],
    samples: Sequence[AppearanceEvidenceSample],
    evidence_config: AppearanceEvidenceConfig | None = None,
    physical_config: SparseTrackletIndexConfig | None = None,
    calibration_config: ReIDCalibrationConfig | None = None,
) -> ReIDOverlapCalibration:
    evidence_config = (
        evidence_config or AppearanceEvidenceConfig()
    )
    physical_config = (
        physical_config or SparseTrackletIndexConfig()
    )
    calibration_config = (
        calibration_config or ReIDCalibrationConfig()
    )

    profiles = tuple(
        build_tracklet_profiles(
            samples,
            evidence_config,
        )
    )
    profiles_by_id: dict[str, TrackletAppearanceProfile] = {
        profile.tracklet_id: profile
        for profile in profiles
    }

    known_tracklet_ids = {
        tracklet.tracklet_id
        for tracklet in tracklets
    }
    unknown_profiles = (
        set(profiles_by_id) - known_tracklet_ids
    )
    if unknown_profiles:
        raise ValueError(
            "appearance profiles reference unknown tracklets: "
            f"{sorted(unknown_profiles)[:10]}"
        )

    split_positive = _split_positive_examples(
        samples,
        evidence_config=evidence_config,
        config=calibration_config,
    )

    physical = build_sparse_tracklet_index(
        tracklets,
        physical_config,
    )

    negative_examples: list[ReIDCalibrationExample] = []
    for conflict in physical.hard_coexistence_conflicts():
        profile_a = profiles_by_id.get(conflict.tracklet_a)
        profile_b = profiles_by_id.get(conflict.tracklet_b)
        if profile_a is None or profile_b is None:
            continue

        negative_examples.append(
            ReIDCalibrationExample(
                family="simultaneous_negative",
                tracklet_a=conflict.tracklet_a,
                tracklet_b=conflict.tracklet_b,
                cosine=_cosine(
                    profile_a.aggregate_embedding,
                    profile_b.aggregate_embedding,
                ),
                evidence={
                    "reason": conflict.reason,
                    "physics": dict(conflict.evidence),
                    "profile_reliability_a": profile_a.reliability,
                    "profile_reliability_b": profile_b.reliability,
                },
            )
        )

    close_examples: list[ReIDCalibrationExample] = []
    for pair in physical.close_coobserved_pairs():
        profile_a = profiles_by_id.get(pair.tracklet_a)
        profile_b = profiles_by_id.get(pair.tracklet_b)
        if profile_a is None or profile_b is None:
            continue

        close_examples.append(
            ReIDCalibrationExample(
                family="close_overlap",
                tracklet_a=pair.tracklet_a,
                tracklet_b=pair.tracklet_b,
                cosine=_cosine(
                    profile_a.aggregate_embedding,
                    profile_b.aggregate_embedding,
                ),
                evidence={
                    "shared_frame_count": pair.shared_frame_count,
                    "first_shared_frame": pair.first_shared_frame,
                    "last_shared_frame": pair.last_shared_frame,
                    "min_separation_m": pair.min_separation_m,
                    "median_separation_m": pair.median_separation_m,
                    "max_separation_m": pair.max_separation_m,
                    "mean_separation_m": pair.mean_separation_m,
                    "profile_reliability_a": profile_a.reliability,
                    "profile_reliability_b": profile_b.reliability,
                    "used_as_positive": False,
                },
            )
        )

    return ReIDOverlapCalibration(
        split_positive=split_positive,
        simultaneous_negative=tuple(negative_examples),
        close_overlap=tuple(close_examples),
        profiled_tracklet_count=len(profiles),
        unprofiled_tracklet_count=(
            len(tracklets) - len(profiles)
        ),
        config=calibration_config,
    )
