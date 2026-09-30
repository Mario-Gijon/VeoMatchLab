"""Proxy benchmark for football ReID embeddings in Tracking V1.

The benchmark deliberately uses relationships already implied by conservative
local tracking instead of pretending that player ground-truth labels exist.

Proxy positives
---------------
1. Two accepted crops from the SAME tracklet.
2. Early-half aggregate vs late-half aggregate of the SAME sufficiently
   sampled tracklet.  This is the closest available simulation of reconnecting
   one player after fragmentation.

Safe-ish proxy negatives
------------------------
1. Accepted samples from DIFFERENT tracklets at the exact same source frame,
   provided their field positions are sufficiently separated.
2. Aggregate embeddings of DIFFERENT tracklets that coexist at a source frame
   with sufficient field-space separation.

These are diagnostics, not formal benchmark ground truth.  A local tracklet can
still contain an identity error, and same-frame association can still miss a
duplicate.  The separation requirement reduces the latter risk.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tracking.appearance.models import AppearanceSample
from tracking.appearance.reid import (
    cosine_similarity,
    quality_weighted_mean,
)
from tracking.tracklets import Tracklet, TrackletSet


@dataclass(frozen=True, slots=True)
class ReIDBenchmarkConfig:
    descriptor_name: str = "reid"
    min_split_samples: int = 4
    negative_min_separation_m: float = 1.50
    hard_example_count: int = 20

    def __post_init__(self) -> None:
        if not self.descriptor_name.strip():
            raise ValueError("descriptor_name must be non-empty")
        if self.min_split_samples < 2:
            raise ValueError("min_split_samples must be >= 2")
        if self.negative_min_separation_m < 0.0:
            raise ValueError(
                "negative_min_separation_m must be >= 0"
            )
        if self.hard_example_count < 1:
            raise ValueError("hard_example_count must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
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
        "min": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "median": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def roc_auc(
    positives: Sequence[float],
    negatives: Sequence[float],
) -> float | None:
    """Probability that a random positive scores above a random negative.

    Ties contribute 0.5.  The implementation is dependency-free and adequate
    for the small diagnostic pair counts used here.
    """
    if not positives or not negatives:
        return None

    wins = 0.0
    total = len(positives) * len(negatives)

    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5

    return wins / total


def _sample_embedding(
    sample: AppearanceSample,
    *,
    descriptor_name: str,
    model_id: str,
) -> np.ndarray:
    descriptor = sample.descriptor(
        descriptor_name,
        model_id,
    )
    if descriptor is None:
        raise ValueError(
            f"sample {sample.sample_id!r} lacks "
            f"{descriptor_name!r}/{model_id!r}"
        )
    return np.asarray(
        descriptor.values,
        dtype=np.float32,
    )


def _normalize_samples(
    samples: Iterable[
        AppearanceSample | Mapping[str, Any]
    ],
) -> tuple[AppearanceSample, ...]:
    result = tuple(
        item
        if isinstance(item, AppearanceSample)
        else AppearanceSample.from_dict(item)
        for item in samples
    )

    sample_ids: set[str] = set()
    detection_ids: set[str] = set()

    for item in result:
        if item.sample_id in sample_ids:
            raise ValueError(
                f"duplicate sample_id: {item.sample_id!r}"
            )
        sample_ids.add(item.sample_id)

        if item.detection_id in detection_ids:
            raise ValueError(
                "benchmark expects at most one ReID sample per "
                f"detection_id, duplicate {item.detection_id!r}"
            )
        detection_ids.add(item.detection_id)

    return result


def _tracklet_lookup(
    tracklets: TrackletSet | Iterable[
        Tracklet | Mapping[str, Any]
    ],
) -> dict[str, Tracklet]:
    if isinstance(tracklets, TrackletSet):
        normalized = tracklets
    else:
        normalized = TrackletSet(
            tracklets=tuple(
                item
                if isinstance(item, Tracklet)
                else Tracklet.from_dict(item)
                for item in tracklets
            )
        )
    return {
        item.tracklet_id: item
        for item in normalized.tracklets
    }


def _observation_maps(
    tracklet: Tracklet,
) -> tuple[
    dict[int, Any],
    set[int],
]:
    by_frame = {
        item.frame_index: item
        for item in tracklet.observations
    }
    return by_frame, set(by_frame)


def _max_shared_frame_separation(
    first: Tracklet,
    second: Tracklet,
) -> tuple[float | None, int | None]:
    first_by_frame, first_frames = _observation_maps(first)
    second_by_frame, second_frames = _observation_maps(second)

    shared = sorted(first_frames & second_frames)
    if not shared:
        return None, None

    best_distance = -1.0
    best_frame: int | None = None

    for frame in shared:
        a = first_by_frame[frame]
        b = second_by_frame[frame]
        distance = math.hypot(
            a.x - b.x,
            a.z - b.z,
        )
        if distance > best_distance:
            best_distance = distance
            best_frame = frame

    return best_distance, best_frame


def _aggregate(
    samples: Sequence[AppearanceSample],
    *,
    descriptor_name: str,
    model_id: str,
) -> np.ndarray:
    return quality_weighted_mean(
        [
            _sample_embedding(
                item,
                descriptor_name=descriptor_name,
                model_id=model_id,
            )
            for item in samples
        ],
        [item.quality for item in samples],
    )


def benchmark_reid(
    samples: Iterable[
        AppearanceSample | Mapping[str, Any]
    ],
    tracklets: TrackletSet | Iterable[
        Tracklet | Mapping[str, Any]
    ],
    *,
    model_id: str,
    config: ReIDBenchmarkConfig | None = None,
) -> dict[str, Any]:
    selected = config or ReIDBenchmarkConfig()
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model_id must be non-empty")
    model_id = model_id.strip()

    normalized_samples = _normalize_samples(samples)
    tracklet_by_id = _tracklet_lookup(tracklets)

    samples_by_tracklet: dict[str, list[AppearanceSample]] = {}
    samples_by_frame: dict[int, list[AppearanceSample]] = {}

    for sample in normalized_samples:
        if sample.tracklet_id not in tracklet_by_id:
            raise ValueError(
                f"sample references missing tracklet "
                f"{sample.tracklet_id!r}"
            )

        tracklet = tracklet_by_id[sample.tracklet_id]
        observations = {
            item.detection_id: item
            for item in tracklet.observations
        }
        observation = observations.get(sample.detection_id)
        if observation is None:
            raise ValueError(
                f"sample {sample.sample_id!r} references detection "
                f"outside tracklet {sample.tracklet_id!r}"
            )
        if observation.frame_index != sample.frame_index:
            raise ValueError(
                f"sample {sample.sample_id!r} frame does not match tracklet"
            )

        # Validate descriptor existence/dimension eagerly.
        _sample_embedding(
            sample,
            descriptor_name=selected.descriptor_name,
            model_id=model_id,
        )

        samples_by_tracklet.setdefault(
            sample.tracklet_id,
            [],
        ).append(sample)
        samples_by_frame.setdefault(
            sample.frame_index,
            [],
        ).append(sample)

    for values in samples_by_tracklet.values():
        values.sort(
            key=lambda item: (
                item.timestamp_s,
                item.sample_id,
            )
        )

    # 1) Same-tracklet sample positives.
    sample_positive_scores: list[float] = []
    sample_positive_examples: list[dict[str, Any]] = []

    for tracklet_id, values in samples_by_tracklet.items():
        for left_index in range(len(values)):
            for right_index in range(left_index + 1, len(values)):
                first = values[left_index]
                second = values[right_index]
                score = cosine_similarity(
                    _sample_embedding(
                        first,
                        descriptor_name=selected.descriptor_name,
                        model_id=model_id,
                    ),
                    _sample_embedding(
                        second,
                        descriptor_name=selected.descriptor_name,
                        model_id=model_id,
                    ),
                )
                sample_positive_scores.append(score)
                sample_positive_examples.append(
                    {
                        "score": score,
                        "tracklet_id": tracklet_id,
                        "sample_a": first.sample_id,
                        "sample_b": second.sample_id,
                        "time_gap_s": (
                            second.timestamp_s - first.timestamp_s
                        ),
                    }
                )

    # 2) Same-frame, spatially-separated sample negatives.
    sample_negative_scores: list[float] = []
    sample_negative_examples: list[dict[str, Any]] = []

    observation_by_detection = {
        observation.detection_id: observation
        for tracklet in tracklet_by_id.values()
        for observation in tracklet.observations
    }

    for frame, values in samples_by_frame.items():
        ordered = sorted(
            values,
            key=lambda item: (
                item.tracklet_id,
                item.sample_id,
            ),
        )
        for left_index in range(len(ordered)):
            for right_index in range(left_index + 1, len(ordered)):
                first = ordered[left_index]
                second = ordered[right_index]
                if first.tracklet_id == second.tracklet_id:
                    continue

                first_obs = observation_by_detection[
                    first.detection_id
                ]
                second_obs = observation_by_detection[
                    second.detection_id
                ]
                separation = math.hypot(
                    first_obs.x - second_obs.x,
                    first_obs.z - second_obs.z,
                )
                if separation < selected.negative_min_separation_m:
                    continue

                score = cosine_similarity(
                    _sample_embedding(
                        first,
                        descriptor_name=selected.descriptor_name,
                        model_id=model_id,
                    ),
                    _sample_embedding(
                        second,
                        descriptor_name=selected.descriptor_name,
                        model_id=model_id,
                    ),
                )
                sample_negative_scores.append(score)
                sample_negative_examples.append(
                    {
                        "score": score,
                        "frame_index": frame,
                        "tracklet_a": first.tracklet_id,
                        "tracklet_b": second.tracklet_id,
                        "sample_a": first.sample_id,
                        "sample_b": second.sample_id,
                        "separation_m": separation,
                    }
                )

    # Aggregate every tracklet that owns accepted samples.
    aggregate_by_tracklet = {
        tracklet_id: _aggregate(
            values,
            descriptor_name=selected.descriptor_name,
            model_id=model_id,
        )
        for tracklet_id, values in samples_by_tracklet.items()
    }

    # 3) Early-half vs late-half aggregate positives.
    split_positive_scores: list[float] = []
    split_positive_examples: list[dict[str, Any]] = []

    for tracklet_id, values in samples_by_tracklet.items():
        if len(values) < selected.min_split_samples:
            continue

        midpoint = len(values) // 2
        early = values[:midpoint]
        late = values[midpoint:]

        if not early or not late:
            continue

        early_embedding = _aggregate(
            early,
            descriptor_name=selected.descriptor_name,
            model_id=model_id,
        )
        late_embedding = _aggregate(
            late,
            descriptor_name=selected.descriptor_name,
            model_id=model_id,
        )
        score = cosine_similarity(
            early_embedding,
            late_embedding,
        )
        split_positive_scores.append(score)
        split_positive_examples.append(
            {
                "score": score,
                "tracklet_id": tracklet_id,
                "early_count": len(early),
                "late_count": len(late),
                "early_end_s": early[-1].timestamp_s,
                "late_start_s": late[0].timestamp_s,
            }
        )

    # 4) Concurrent tracklet aggregate negatives.
    tracklet_negative_scores: list[float] = []
    tracklet_negative_examples: list[dict[str, Any]] = []

    tracklet_ids = sorted(aggregate_by_tracklet)
    for left_index in range(len(tracklet_ids)):
        for right_index in range(left_index + 1, len(tracklet_ids)):
            first_id = tracklet_ids[left_index]
            second_id = tracklet_ids[right_index]

            separation, shared_frame = _max_shared_frame_separation(
                tracklet_by_id[first_id],
                tracklet_by_id[second_id],
            )
            if (
                separation is None
                or separation < selected.negative_min_separation_m
            ):
                continue

            score = cosine_similarity(
                aggregate_by_tracklet[first_id],
                aggregate_by_tracklet[second_id],
            )
            tracklet_negative_scores.append(score)
            tracklet_negative_examples.append(
                {
                    "score": score,
                    "tracklet_a": first_id,
                    "tracklet_b": second_id,
                    "shared_frame": shared_frame,
                    "separation_m": separation,
                }
            )

    sample_auc = roc_auc(
        sample_positive_scores,
        sample_negative_scores,
    )
    tracklet_auc = roc_auc(
        split_positive_scores,
        tracklet_negative_scores,
    )

    split_stats = _stats(split_positive_scores)
    tracklet_negative_stats = _stats(
        tracklet_negative_scores
    )

    separation_margin = None
    if (
        split_stats["p10"] is not None
        and tracklet_negative_stats["p95"] is not None
    ):
        separation_margin = (
            float(split_stats["p10"])
            - float(tracklet_negative_stats["p95"])
        )

    hard_count = selected.hard_example_count

    hardest_sample_negatives = sorted(
        sample_negative_examples,
        key=lambda item: (
            -float(item["score"]),
            item["tracklet_a"],
            item["tracklet_b"],
        ),
    )[:hard_count]

    weakest_sample_positives = sorted(
        sample_positive_examples,
        key=lambda item: (
            float(item["score"]),
            item["tracklet_id"],
        ),
    )[:hard_count]

    hardest_tracklet_negatives = sorted(
        tracklet_negative_examples,
        key=lambda item: (
            -float(item["score"]),
            item["tracklet_a"],
            item["tracklet_b"],
        ),
    )[:hard_count]

    weakest_split_positives = sorted(
        split_positive_examples,
        key=lambda item: (
            float(item["score"]),
            item["tracklet_id"],
        ),
    )[:hard_count]

    return {
        "model_id": model_id,
        "sample_count": len(normalized_samples),
        "tracklet_with_sample_count": len(samples_by_tracklet),
        "config": selected.to_dict(),
        "sample_level": {
            "same_tracklet_positive": _stats(
                sample_positive_scores
            ),
            "same_frame_negative": _stats(
                sample_negative_scores
            ),
            "roc_auc": sample_auc,
            "weakest_positive_examples": (
                weakest_sample_positives
            ),
            "hardest_negative_examples": (
                hardest_sample_negatives
            ),
        },
        "tracklet_level": {
            "split_positive": split_stats,
            "concurrent_negative": tracklet_negative_stats,
            "roc_auc": tracklet_auc,
            "p10_positive_minus_p95_negative": separation_margin,
            "weakest_positive_examples": (
                weakest_split_positives
            ),
            "hardest_negative_examples": (
                hardest_tracklet_negatives
            ),
        },
    }
