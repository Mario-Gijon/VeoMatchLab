"""Hito 7C.2B: soft jersey-number evidence and tracklet aggregation.

This module is OCR-backend agnostic.  It consumes per-view OCR predictions and
turns them into conservative tracklet-level evidence.

Important semantics
-------------------
* one OCR read is never a hard identity fact;
* "no number read" is UNKNOWN, not evidence against a player/number;
* multiple agreeing views accumulate support;
* conflicting readings are preserved in the support distribution;
* readiness from Hito 7C.2A is only a weak quality weight, not a probability;
* the output may expose a best-supported number, but it never assigns a player
  identity or creates a hard same-number constraint.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import re
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


JERSEY_NUMBER_VERSION = "jersey-number-evidence-v1"


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


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


def parse_jersey_number(
    text: Any,
    *,
    min_number: int = 0,
    max_number: int = 99,
) -> int | None:
    """Extract one conservative 1-2 digit jersey number from OCR text.

    Non-digits are ignored, which tolerates outputs such as ``"O9"`` -> 9.
    More than two detected digits are rejected rather than arbitrarily
    truncating them.
    """

    if text is None:
        return None
    digits = "".join(re.findall(r"\d", str(text)))
    if not 1 <= len(digits) <= 2:
        return None

    number = int(digits)
    if not min_number <= number <= max_number:
        return None
    return number


@dataclass(frozen=True, slots=True)
class JerseyNumberAggregationConfig:
    min_number: int = 0
    max_number: int = 99

    # Recognition weight = confidence^power * readiness prior * rank prior.
    confidence_power: float = 2.0
    readiness_floor: float = 0.45
    rank_decay: float = 0.12

    strong_min_support_views: int = 2
    strong_min_best_fraction: float = 0.67
    strong_min_margin: float = 0.20
    strong_min_max_confidence: float = 0.50

    moderate_min_best_fraction: float = 0.55
    moderate_min_margin: float = 0.10
    moderate_min_max_confidence: float = 0.35

    def __post_init__(self) -> None:
        if self.min_number < 0:
            raise ValueError("min_number must be >= 0")
        if self.max_number < self.min_number:
            raise ValueError("max_number must be >= min_number")
        if self.confidence_power <= 0.0:
            raise ValueError("confidence_power must be > 0")
        if not 0.0 <= self.readiness_floor <= 1.0:
            raise ValueError("readiness_floor must be in [0,1]")
        if self.rank_decay < 0.0:
            raise ValueError("rank_decay must be >= 0")
        if self.strong_min_support_views < 1:
            raise ValueError("strong_min_support_views must be >= 1")

        for name in (
            "strong_min_best_fraction",
            "strong_min_margin",
            "strong_min_max_confidence",
            "moderate_min_best_fraction",
            "moderate_min_margin",
            "moderate_min_max_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class JerseyNumberObservation:
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    raw_text: str | None
    parsed_number: int | None
    ocr_confidence: float
    readiness_score: float
    rank_within_tracklet: int
    backend: str

    def __post_init__(self) -> None:
        if not self.tracklet_id or not self.detection_id:
            raise ValueError("tracklet_id/detection_id must be non-empty")
        if not 0.0 <= self.ocr_confidence <= 1.0:
            raise ValueError("ocr_confidence must be in [0,1]")
        if not 0.0 <= self.readiness_score <= 1.0:
            raise ValueError("readiness_score must be in [0,1]")
        if self.rank_within_tracklet < 1:
            raise ValueError("rank_within_tracklet must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "jersey_number_version": JERSEY_NUMBER_VERSION,
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "raw_text": self.raw_text,
            "parsed_number": self.parsed_number,
            "ocr_confidence": self.ocr_confidence,
            "readiness_score": self.readiness_score,
            "rank_within_tracklet": self.rank_within_tracklet,
            "backend": self.backend,
        }


@dataclass(frozen=True, slots=True)
class JerseyNumberSupport:
    number: int
    normalized_support: float
    evidence_weight: float
    support_view_count: int
    max_ocr_confidence: float
    mean_ocr_confidence: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackletJerseyEvidence:
    tracklet_id: str
    selected_view_count: int
    recognized_view_count: int
    unknown_view_count: int
    best_number: int | None
    strength: str
    best_fraction: float | None
    margin: float | None
    total_evidence_weight: float
    supports: tuple[JerseyNumberSupport, ...]
    first_timestamp_s: float
    last_timestamp_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "jersey_number_version": JERSEY_NUMBER_VERSION,
            "tracklet_id": self.tracklet_id,
            "selected_view_count": self.selected_view_count,
            "recognized_view_count": self.recognized_view_count,
            "unknown_view_count": self.unknown_view_count,
            "best_number": self.best_number,
            "strength": self.strength,
            "best_fraction": self.best_fraction,
            "margin": self.margin,
            "total_evidence_weight": self.total_evidence_weight,
            "supports": [item.to_dict() for item in self.supports],
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
            "hard_identity_fact": False,
        }


def observation_evidence_weight(
    observation: JerseyNumberObservation,
    config: JerseyNumberAggregationConfig,
) -> float:
    if observation.parsed_number is None:
        return 0.0

    confidence = _clamp01(observation.ocr_confidence)
    readiness_prior = (
        config.readiness_floor
        + (1.0 - config.readiness_floor)
        * _clamp01(observation.readiness_score)
    )
    rank_prior = math.exp(
        -config.rank_decay
        * max(0, observation.rank_within_tracklet - 1)
    )
    return (
        confidence ** config.confidence_power
        * readiness_prior
        * rank_prior
    )


def aggregate_tracklet_jersey_evidence(
    observations: Sequence[JerseyNumberObservation],
    config: JerseyNumberAggregationConfig | None = None,
) -> TrackletJerseyEvidence:
    selected = config or JerseyNumberAggregationConfig()
    if not observations:
        raise ValueError("cannot aggregate empty observations")

    ordered = tuple(
        sorted(
            observations,
            key=lambda item: (
                item.timestamp_s,
                item.frame_index,
                item.detection_id,
            ),
        )
    )
    tracklet_ids = {item.tracklet_id for item in ordered}
    if len(tracklet_ids) != 1:
        raise ValueError("observations must share one tracklet_id")

    weight_by_number: dict[int, float] = defaultdict(float)
    confidences_by_number: dict[int, list[float]] = defaultdict(list)
    view_count_by_number: dict[int, int] = defaultdict(int)

    recognized = 0
    for observation in ordered:
        if observation.parsed_number is None:
            continue
        recognized += 1
        number = int(observation.parsed_number)
        weight_by_number[number] += observation_evidence_weight(
            observation,
            selected,
        )
        confidences_by_number[number].append(
            float(observation.ocr_confidence)
        )
        view_count_by_number[number] += 1

    total_weight = float(sum(weight_by_number.values()))
    if not weight_by_number or total_weight <= 1e-12:
        return TrackletJerseyEvidence(
            tracklet_id=ordered[0].tracklet_id,
            selected_view_count=len(ordered),
            recognized_view_count=recognized,
            unknown_view_count=len(ordered) - recognized,
            best_number=None,
            strength="unknown",
            best_fraction=None,
            margin=None,
            total_evidence_weight=0.0,
            supports=(),
            first_timestamp_s=float(ordered[0].timestamp_s),
            last_timestamp_s=float(ordered[-1].timestamp_s),
        )

    ranked_numbers = sorted(
        weight_by_number,
        key=lambda number: (
            -weight_by_number[number],
            number,
        ),
    )

    support_rows: list[JerseyNumberSupport] = []
    for number in ranked_numbers:
        confidences = confidences_by_number[number]
        support_rows.append(
            JerseyNumberSupport(
                number=number,
                normalized_support=float(
                    weight_by_number[number] / total_weight
                ),
                evidence_weight=float(weight_by_number[number]),
                support_view_count=int(view_count_by_number[number]),
                max_ocr_confidence=float(max(confidences)),
                mean_ocr_confidence=float(np.mean(confidences)),
            )
        )

    best = support_rows[0]
    second_fraction = (
        support_rows[1].normalized_support
        if len(support_rows) > 1
        else 0.0
    )
    margin = float(
        best.normalized_support - second_fraction
    )

    if (
        best.support_view_count >= selected.strong_min_support_views
        and best.normalized_support >= selected.strong_min_best_fraction
        and margin >= selected.strong_min_margin
        and best.max_ocr_confidence
        >= selected.strong_min_max_confidence
    ):
        strength = "strong"
    elif (
        best.normalized_support >= selected.moderate_min_best_fraction
        and margin >= selected.moderate_min_margin
        and best.max_ocr_confidence
        >= selected.moderate_min_max_confidence
    ):
        strength = "moderate"
    else:
        strength = "weak"

    return TrackletJerseyEvidence(
        tracklet_id=ordered[0].tracklet_id,
        selected_view_count=len(ordered),
        recognized_view_count=recognized,
        unknown_view_count=len(ordered) - recognized,
        best_number=best.number,
        strength=strength,
        best_fraction=best.normalized_support,
        margin=margin,
        total_evidence_weight=total_weight,
        supports=tuple(support_rows),
        first_timestamp_s=float(ordered[0].timestamp_s),
        last_timestamp_s=float(ordered[-1].timestamp_s),
    )


def aggregate_all_tracklets(
    observations: Iterable[JerseyNumberObservation],
    config: JerseyNumberAggregationConfig | None = None,
) -> tuple[TrackletJerseyEvidence, ...]:
    selected = config or JerseyNumberAggregationConfig()
    grouped: dict[str, list[JerseyNumberObservation]] = defaultdict(list)
    for observation in observations:
        grouped[observation.tracklet_id].append(observation)

    return tuple(
        aggregate_tracklet_jersey_evidence(
            grouped[tracklet_id],
            selected,
        )
        for tracklet_id in sorted(grouped)
    )


def jersey_number_summary(
    observations: Sequence[JerseyNumberObservation],
    profiles: Sequence[TrackletJerseyEvidence],
    *,
    total_tracklet_count: int | None = None,
) -> dict[str, Any]:
    recognized = [
        item for item in observations
        if item.parsed_number is not None
    ]
    strength_counts: dict[str, int] = defaultdict(int)
    for profile in profiles:
        strength_counts[profile.strength] += 1

    payload: dict[str, Any] = {
        "schema_version": 1,
        "jersey_number_version": JERSEY_NUMBER_VERSION,
        "observation_count": len(observations),
        "recognized_observation_count": len(recognized),
        "recognized_observation_fraction": (
            len(recognized) / len(observations)
            if observations
            else None
        ),
        "tracklet_profile_count": len(profiles),
        "tracklet_strength_counts": dict(
            sorted(strength_counts.items())
        ),
        "ocr_confidence_recognized": _stats(
            [item.ocr_confidence for item in recognized]
        ),
        "best_fraction": _stats(
            [
                profile.best_fraction
                for profile in profiles
                if profile.best_fraction is not None
            ]
        ),
        "margin": _stats(
            [
                profile.margin
                for profile in profiles
                if profile.margin is not None
            ]
        ),
        "hard_identity_facts_created": False,
        "same_number_hard_link_created": False,
    }

    if total_tracklet_count is not None:
        payload["total_tracklet_count"] = int(total_tracklet_count)
        payload["strong_tracklet_coverage"] = (
            strength_counts.get("strong", 0) / total_tracklet_count
            if total_tracklet_count
            else None
        )
        payload["known_number_tracklet_coverage"] = (
            sum(
                1 for profile in profiles
                if profile.best_number is not None
            )
            / total_tracklet_count
            if total_tracklet_count
            else None
        )

    return payload


def observation_from_prediction_row(
    row: Mapping[str, Any],
    config: JerseyNumberAggregationConfig | None = None,
) -> JerseyNumberObservation:
    selected = config or JerseyNumberAggregationConfig()
    raw_text = (
        None
        if row.get("raw_text") is None
        else str(row.get("raw_text"))
    )
    parsed = row.get("parsed_number")
    if parsed is None:
        parsed_number = parse_jersey_number(
            raw_text,
            min_number=selected.min_number,
            max_number=selected.max_number,
        )
    else:
        parsed_number = int(parsed)
        if not selected.min_number <= parsed_number <= selected.max_number:
            parsed_number = None

    return JerseyNumberObservation(
        tracklet_id=str(row["tracklet_id"]),
        detection_id=str(row["detection_id"]),
        frame_index=int(row["frame_index"]),
        timestamp_s=_finite("timestamp_s", row["timestamp_s"]),
        raw_text=raw_text,
        parsed_number=parsed_number,
        ocr_confidence=_clamp01(
            _finite(
                "ocr_confidence",
                row.get("ocr_confidence", 0.0),
            )
        ),
        readiness_score=_clamp01(
            _finite(
                "readiness_score",
                row.get("readiness_score", 0.0),
            )
        ),
        rank_within_tracklet=int(
            row.get("rank_within_tracklet", 1)
        ),
        backend=str(row.get("backend", "unknown")),
    )
