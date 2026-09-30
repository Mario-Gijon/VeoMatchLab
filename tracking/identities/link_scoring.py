"""Soft link scoring for Tracking V1 identity candidates.

Hito 3B-A consumes the Hito 3A candidate graph and produces explainable,
continuous evidence for the future global identity solver.

Design goals
------------
* Only ``candidate`` edges are scored. Hard ``cannot_link`` constraints remain
  untouched in Hito 3A.
* No identity assignment or merge decision is made here.
* Appearance never becomes a hard constraint.
* Missing appearance is treated as unknown, not as negative evidence.
* Continuations and overlap/duplicate candidates are scored with different
  evidence families.
* Feature scales for continuation timing/motion are learned empirically from
  the current candidate graph (robust p10/p90 anchors), rather than hidden
  hand-tuned linear weights.
* PRTReID appearance is calibrated from the benchmark itself:
  concurrent-negative p95 -> support 0
  split-positive p10      -> support 1
* Family scores are combined by an unweighted geometric mean.  This is a
  ranking heuristic, NOT a probability.

The complete raw source evidence and every normalized component are retained so
the future global solver can use the vector directly instead of relying only
on ``rank_score``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


SCORING_VERSION = "soft-link-v1"


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _smoothstep01(value: float) -> float:
    x = _clamp01(value)
    return x * x * (3.0 - 2.0 * x)


def _map_increasing(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.5
    return _smoothstep01((value - low) / (high - low))


def _map_decreasing(value: float, low: float, high: float) -> float:
    return 1.0 - _map_increasing(value, low, high)


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError("geometric mean received non-finite values")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("geometric mean inputs must be in [0,1]")
    if np.any(array == 0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _arithmetic_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError("arithmetic mean received non-finite values")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError("arithmetic mean inputs must be in [0,1]")
    return float(np.mean(array))


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
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
class EmpiricalScale:
    """Robust p10/p90 scale used for a monotonic feature mapping."""

    p10: float
    p90: float
    higher_is_better: bool

    def __post_init__(self) -> None:
        if not math.isfinite(self.p10) or not math.isfinite(self.p90):
            raise ValueError("empirical scale anchors must be finite")
        if self.p90 < self.p10:
            raise ValueError("p90 must be >= p10")

    def map(self, value: float) -> float:
        value = _finite_float("scale value", value)
        if self.higher_is_better:
            return _map_increasing(value, self.p10, self.p90)
        return _map_decreasing(value, self.p10, self.p90)

    def to_dict(self) -> dict[str, Any]:
        return {
            "p10": self.p10,
            "p90": self.p90,
            "higher_is_better": self.higher_is_better,
        }


@dataclass(frozen=True, slots=True)
class AppearanceCalibration:
    """Benchmark-derived aggregate-cosine calibration."""

    negative_p95: float
    positive_p10: float
    benchmark_roc_auc: float | None = None
    benchmark_margin: float | None = None

    def __post_init__(self) -> None:
        if self.positive_p10 <= self.negative_p95:
            raise ValueError(
                "appearance calibration requires positive_p10 > negative_p95"
            )

    def support(self, cosine: float) -> float:
        return _map_increasing(
            _finite_float("appearance cosine", cosine),
            self.negative_p95,
            self.positive_p10,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "negative_anchor": {
                "source": "tracklet_level.concurrent_negative.p95",
                "value": self.negative_p95,
                "support": 0.0,
            },
            "positive_anchor": {
                "source": "tracklet_level.split_positive.p10",
                "value": self.positive_p10,
                "support": 1.0,
            },
            "benchmark_roc_auc": self.benchmark_roc_auc,
            "benchmark_margin": self.benchmark_margin,
        }


@dataclass(frozen=True, slots=True)
class SoftLinkCalibration:
    appearance: AppearanceCalibration
    continuation_gap_s: EmpiricalScale
    continuation_required_speed_mps: EmpiricalScale
    continuation_prediction_error_m: EmpiricalScale | None
    overlap_shared_frame_count: EmpiricalScale | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scoring_version": SCORING_VERSION,
            "appearance": self.appearance.to_dict(),
            "continuation_gap_s": self.continuation_gap_s.to_dict(),
            "continuation_required_speed_mps": (
                self.continuation_required_speed_mps.to_dict()
            ),
            "continuation_prediction_error_m": (
                self.continuation_prediction_error_m.to_dict()
                if self.continuation_prediction_error_m is not None
                else None
            ),
            "overlap_shared_frame_count": (
                self.overlap_shared_frame_count.to_dict()
                if self.overlap_shared_frame_count is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ScoredLink:
    tracklet_a: str
    tracklet_b: str
    link_kind: str
    rank_score: float
    evidence_confidence: float
    family_scores: Mapping[str, float | None]
    components: Mapping[str, Any]
    source_reason: str
    source_temporal_relation: str

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("scored link requires canonical tracklet order")
        if self.link_kind not in {
            "continuation",
            "overlap_duplicate",
        }:
            raise ValueError("unsupported link_kind")
        if not (0.0 <= self.rank_score <= 1.0):
            raise ValueError("rank_score must be in [0,1]")
        if not (0.0 <= self.evidence_confidence <= 1.0):
            raise ValueError("evidence_confidence must be in [0,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scoring_version": SCORING_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "link_kind": self.link_kind,
            "rank_score": self.rank_score,
            "rank_score_semantics": "relative_evidence_ranking_not_probability",
            "evidence_confidence": self.evidence_confidence,
            "family_scores": dict(self.family_scores),
            "components": dict(self.components),
            "source_reason": self.source_reason,
            "source_temporal_relation": self.source_temporal_relation,
            "identity_decision": None,
        }


def load_appearance_calibration(
    benchmark_summary: Mapping[str, Any],
) -> AppearanceCalibration:
    # The generic ReID benchmark report may be stored either directly at the
    # root or wrapped by the backend runner under ``report``.  The PRTReID
    # runner uses the wrapped shape:
    #
    # {
    #   "backend": {...},
    #   "report": {
    #       "sample_level": {...},
    #       "tracklet_level": {...},
    #   }
    # }
    #
    # Accept both layouts so 3B-A remains compatible with the generic
    # benchmark and backend-specific wrappers.
    report = benchmark_summary.get("report")
    if isinstance(report, Mapping):
        benchmark_report = report
    else:
        benchmark_report = benchmark_summary

    tracklet_level = benchmark_report.get("tracklet_level")
    if not isinstance(tracklet_level, Mapping):
        raise ValueError(
            "benchmark summary lacks report.tracklet_level "
            "(or top-level tracklet_level)"
        )

    split_positive = tracklet_level.get("split_positive")
    concurrent_negative = tracklet_level.get("concurrent_negative")
    if not isinstance(split_positive, Mapping):
        raise ValueError("benchmark summary lacks split_positive stats")
    if not isinstance(concurrent_negative, Mapping):
        raise ValueError("benchmark summary lacks concurrent_negative stats")

    return AppearanceCalibration(
        negative_p95=_finite_float(
            "concurrent_negative.p95",
            concurrent_negative["p95"],
        ),
        positive_p10=_finite_float(
            "split_positive.p10",
            split_positive["p10"],
        ),
        benchmark_roc_auc=(
            _finite_float("tracklet roc_auc", tracklet_level["roc_auc"])
            if tracklet_level.get("roc_auc") is not None
            else None
        ),
        benchmark_margin=(
            _finite_float(
                "tracklet margin",
                tracklet_level["p10_positive_minus_p95_negative"],
            )
            if tracklet_level.get("p10_positive_minus_p95_negative") is not None
            else None
        ),
    )


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source_path}:{line_number} is not a JSON object"
                )
            yield payload


def _candidate_edges(
    edges: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for edge in edges:
        status = edge.get("status")
        if status == "candidate":
            result.append(dict(edge))
        elif status != "cannot_link":
            raise ValueError(f"unexpected candidate-graph status: {status!r}")
    return result


def _robust_scale(
    values: Sequence[float],
    *,
    higher_is_better: bool,
) -> EmpiricalScale:
    if not values:
        raise ValueError("cannot fit empirical scale from no values")
    array = np.asarray(values, dtype=np.float64)
    return EmpiricalScale(
        p10=float(np.percentile(array, 10)),
        p90=float(np.percentile(array, 90)),
        higher_is_better=higher_is_better,
    )


def fit_soft_link_calibration(
    candidate_edges: Sequence[Mapping[str, Any]],
    appearance_calibration: AppearanceCalibration,
) -> SoftLinkCalibration:
    continuations = [
        edge
        for edge in candidate_edges
        if edge.get("reason") == "transition_reachable"
    ]
    if not continuations:
        raise ValueError("candidate graph contains no continuation edges")

    gaps: list[float] = []
    speeds: list[float] = []
    prediction_errors: list[float] = []
    shared_counts: list[float] = []

    for edge in continuations:
        physics = edge.get("physics")
        if not isinstance(physics, Mapping):
            raise ValueError("continuation edge lacks physics payload")
        gaps.append(_finite_float("gap_s", physics["gap_s"]))
        required_speed = physics.get("required_speed_after_slack_mps")
        if required_speed is None:
            # Zero-gap candidates within positional slack require no movement.
            required_speed = 0.0
        speeds.append(
            _finite_float(
                "required_speed_after_slack_mps",
                required_speed,
            )
        )
        prediction_error = physics.get("prediction_error_mean_m")
        if prediction_error is not None:
            prediction_errors.append(
                _finite_float("prediction_error_mean_m", prediction_error)
            )

    for edge in candidate_edges:
        if edge.get("reason") not in {
            "overlap_close",
            "overlap_ambiguous",
        }:
            continue
        physics = edge.get("physics")
        if not isinstance(physics, Mapping):
            continue
        count = physics.get("shared_frame_count")
        if count is not None:
            shared_counts.append(float(count))

    return SoftLinkCalibration(
        appearance=appearance_calibration,
        continuation_gap_s=_robust_scale(
            gaps,
            higher_is_better=False,
        ),
        continuation_required_speed_mps=_robust_scale(
            speeds,
            higher_is_better=False,
        ),
        continuation_prediction_error_m=(
            _robust_scale(
                prediction_errors,
                higher_is_better=False,
            )
            if prediction_errors
            else None
        ),
        overlap_shared_frame_count=(
            _robust_scale(
                shared_counts,
                higher_is_better=True,
            )
            if shared_counts
            else None
        ),
    )


def _appearance_components(
    appearance: Mapping[str, Any] | None,
    calibration: AppearanceCalibration,
) -> tuple[dict[str, Any], float | None, float | None]:
    if appearance is None:
        return (
            {
                "available": False,
                "aggregate_cosine": None,
                "bidirectional_best_median": None,
                "aggregate_support": None,
                "bidirectional_support": None,
                "raw_support": None,
                "reliability": None,
                "support_quality": None,
                "effective_support": None,
            },
            None,
            None,
        )

    aggregate_cosine = _finite_float(
        "appearance.aggregate_cosine",
        appearance["aggregate_cosine"],
    )
    aggregate_support = calibration.support(aggregate_cosine)

    bidirectional = appearance.get("bidirectional_best_similarity")
    bidirectional_median = None
    bidirectional_support = None
    if isinstance(bidirectional, Mapping) and bidirectional.get("median") is not None:
        bidirectional_median = _finite_float(
            "appearance.bidirectional_best_similarity.median",
            bidirectional["median"],
        )
        bidirectional_support = calibration.support(
            bidirectional_median
        )

    raw_support_inputs = [aggregate_support]
    if bidirectional_support is not None:
        raw_support_inputs.append(bidirectional_support)
    raw_support = _geometric_mean(raw_support_inputs)
    assert raw_support is not None

    reliability = _clamp01(
        _finite_float(
            "appearance.reliability",
            appearance.get("reliability", 0.0),
        )
    )
    support_quality = _clamp01(
        _finite_float(
            "appearance.support_quality",
            appearance.get("support_quality", 0.0),
        )
    )
    appearance_confidence = _geometric_mean(
        [reliability, support_quality]
    )
    assert appearance_confidence is not None

    # Confidence-gate appearance around neutral support=0.5.  Low-quality ReID
    # cannot dominate the ranking in either direction.
    effective_support = (
        0.5
        + appearance_confidence * (raw_support - 0.5)
    )
    effective_support = _clamp01(effective_support)

    return (
        {
            "available": True,
            "aggregate_cosine": aggregate_cosine,
            "bidirectional_best_median": bidirectional_median,
            "aggregate_support": aggregate_support,
            "bidirectional_support": bidirectional_support,
            "raw_support": raw_support,
            "reliability": reliability,
            "support_quality": support_quality,
            "appearance_confidence": appearance_confidence,
            "effective_support": effective_support,
            "sample_count_a": appearance.get("sample_count_a"),
            "sample_count_b": appearance.get("sample_count_b"),
            "effective_sample_count_a": appearance.get(
                "effective_sample_count_a"
            ),
            "effective_sample_count_b": appearance.get(
                "effective_sample_count_b"
            ),
        },
        effective_support,
        appearance_confidence,
    )


def _node_pair_confidence(
    tracklet_a: str,
    tracklet_b: str,
    node_confidence: Mapping[str, float] | None,
) -> float | None:
    if node_confidence is None:
        return None
    if tracklet_a not in node_confidence or tracklet_b not in node_confidence:
        return None
    a = _clamp01(
        _finite_float("node confidence", node_confidence[tracklet_a])
    )
    b = _clamp01(
        _finite_float("node confidence", node_confidence[tracklet_b])
    )
    return math.sqrt(a * b)


def _score_continuation(
    edge: Mapping[str, Any],
    calibration: SoftLinkCalibration,
    node_confidence: Mapping[str, float] | None,
) -> ScoredLink:
    physics = edge["physics"]
    if not isinstance(physics, Mapping):
        raise ValueError("continuation edge lacks physics")

    gap_s = _finite_float("gap_s", physics["gap_s"])
    required_speed = physics.get("required_speed_after_slack_mps")
    required_speed_mps = (
        0.0
        if required_speed is None
        else _finite_float("required speed", required_speed)
    )

    temporal_support = calibration.continuation_gap_s.map(gap_s)
    speed_support = calibration.continuation_required_speed_mps.map(
        required_speed_mps
    )

    prediction_error = physics.get("prediction_error_mean_m")
    prediction_support = None
    if (
        prediction_error is not None
        and calibration.continuation_prediction_error_m is not None
    ):
        prediction_support = (
            calibration.continuation_prediction_error_m.map(
                _finite_float(
                    "prediction_error_mean_m",
                    prediction_error,
                )
            )
        )

    direction_cosine = physics.get("velocity_direction_cosine")
    direction_support = None
    if direction_cosine is not None:
        direction_cosine = float(
            np.clip(
                _finite_float(
                    "velocity_direction_cosine",
                    direction_cosine,
                ),
                -1.0,
                1.0,
            )
        )
        direction_support = (direction_cosine + 1.0) * 0.5

    motion_values = [speed_support]
    if prediction_support is not None:
        motion_values.append(prediction_support)
    if direction_support is not None:
        motion_values.append(direction_support)
    motion_support = _geometric_mean(motion_values)
    assert motion_support is not None

    appearance_components, appearance_support, appearance_confidence = (
        _appearance_components(
            edge.get("appearance"),
            calibration.appearance,
        )
    )

    family_values = [temporal_support, motion_support]
    if appearance_support is not None:
        family_values.append(appearance_support)
    rank_score = _geometric_mean(family_values)
    assert rank_score is not None

    motion_availability = len(motion_values) / 3.0
    node_pair_confidence = _node_pair_confidence(
        str(edge["tracklet_a"]),
        str(edge["tracklet_b"]),
        node_confidence,
    )
    confidence_values = [motion_availability]
    if node_pair_confidence is not None:
        confidence_values.append(node_pair_confidence)
    if appearance_confidence is not None:
        confidence_values.append(appearance_confidence)

    # Confidence measures how much trustworthy evidence is available, not
    # whether that evidence supports the link. Missing ReID reduces coverage
    # but is never converted into negative appearance evidence.
    source_availability = (
        2.0 + (1.0 if appearance_confidence is not None else 0.0)
    ) / 3.0
    base_confidence = _arithmetic_mean(confidence_values)
    assert base_confidence is not None
    evidence_confidence = _clamp01(
        source_availability * base_confidence
    )

    return ScoredLink(
        tracklet_a=str(edge["tracklet_a"]),
        tracklet_b=str(edge["tracklet_b"]),
        link_kind="continuation",
        rank_score=rank_score,
        evidence_confidence=evidence_confidence,
        family_scores={
            "appearance": appearance_support,
            "temporal": temporal_support,
            "motion": motion_support,
        },
        components={
            "appearance": appearance_components,
            "temporal": {
                "gap_s": gap_s,
                "support": temporal_support,
            },
            "motion": {
                "required_speed_after_slack_mps": required_speed_mps,
                "required_speed_support": speed_support,
                "prediction_error_mean_m": prediction_error,
                "prediction_error_support": prediction_support,
                "velocity_direction_cosine": direction_cosine,
                "velocity_direction_support": direction_support,
                "support": motion_support,
                "available_component_count": len(motion_values),
                "component_capacity": 3,
            },
            "source_confidence": {
                "node_pair_mean_confidence": node_pair_confidence,
            },
            "raw_physics": dict(physics),
        },
        source_reason=str(edge["reason"]),
        source_temporal_relation=str(edge["temporal_relation"]),
    )


def _score_overlap(
    edge: Mapping[str, Any],
    calibration: SoftLinkCalibration,
    node_confidence: Mapping[str, float] | None,
) -> ScoredLink:
    physics = edge["physics"]
    if not isinstance(physics, Mapping):
        raise ValueError("overlap edge lacks physics")

    separation_stats = physics.get("shared_frame_separation_m")
    median_separation_m = None
    spatial_support = None
    if isinstance(separation_stats, Mapping):
        median_value = separation_stats.get("median")
        hard_threshold = physics.get("hard_median_threshold_m")
        if median_value is not None and hard_threshold is not None:
            median_separation_m = _finite_float(
                "median overlap separation",
                median_value,
            )
            hard_threshold = _finite_float(
                "hard overlap separation threshold",
                hard_threshold,
            )
            if hard_threshold <= 0:
                raise ValueError("hard overlap threshold must be > 0")
            spatial_support = _map_decreasing(
                median_separation_m,
                0.0,
                hard_threshold,
            )

    shared_frame_count = int(physics.get("shared_frame_count", 0) or 0)
    observation_support = None
    if (
        shared_frame_count > 0
        and calibration.overlap_shared_frame_count is not None
    ):
        observation_support = calibration.overlap_shared_frame_count.map(
            float(shared_frame_count)
        )

    appearance_components, appearance_support, appearance_confidence = (
        _appearance_components(
            edge.get("appearance"),
            calibration.appearance,
        )
    )

    # Spatial separation is link support. Shared-frame count is instead a
    # confidence/coverage signal: one close shared frame can be meaningful,
    # but should be trusted less than twenty close shared frames.
    geometry_support = spatial_support

    family_values = []
    if geometry_support is not None:
        family_values.append(geometry_support)
    if appearance_support is not None:
        family_values.append(appearance_support)

    # An unsynchronized overlap with no geometry and no ReID is still a 3A
    # candidate, but 3B-A has no positive evidence with which to rank it.
    rank_score = (
        _geometric_mean(family_values)
        if family_values
        else 0.5
    )
    assert rank_score is not None

    node_pair_confidence = _node_pair_confidence(
        str(edge["tracklet_a"]),
        str(edge["tracklet_b"]),
        node_confidence,
    )
    confidence_values = []
    available_link_sources = 0

    if geometry_support is not None:
        available_link_sources += 1
        if observation_support is not None:
            confidence_values.append(observation_support)

    if appearance_confidence is not None:
        available_link_sources += 1
        confidence_values.append(appearance_confidence)

    if node_pair_confidence is not None:
        confidence_values.append(node_pair_confidence)

    source_availability = available_link_sources / 2.0
    base_confidence = (
        _arithmetic_mean(confidence_values)
        if confidence_values
        else 0.0
    )
    assert base_confidence is not None
    evidence_confidence = _clamp01(
        source_availability * base_confidence
    )

    return ScoredLink(
        tracklet_a=str(edge["tracklet_a"]),
        tracklet_b=str(edge["tracklet_b"]),
        link_kind="overlap_duplicate",
        rank_score=rank_score,
        evidence_confidence=evidence_confidence,
        family_scores={
            "appearance": appearance_support,
            "overlap_geometry": geometry_support,
        },
        components={
            "appearance": appearance_components,
            "overlap_geometry": {
                "median_separation_m": median_separation_m,
                "spatial_support": spatial_support,
                "shared_frame_count": shared_frame_count,
                "shared_frame_support": observation_support,
                "interval_overlap_s": physics.get("interval_overlap_s"),
                "support": geometry_support,
            },
            "source_confidence": {
                "node_pair_mean_confidence": node_pair_confidence,
            },
            "raw_physics": dict(physics),
        },
        source_reason=str(edge["reason"]),
        source_temporal_relation=str(edge["temporal_relation"]),
    )


def score_candidate_edge(
    edge: Mapping[str, Any],
    calibration: SoftLinkCalibration,
    *,
    node_confidence: Mapping[str, float] | None = None,
) -> ScoredLink:
    if edge.get("status") != "candidate":
        raise ValueError("3B-A only scores candidate edges")

    reason = edge.get("reason")
    if reason == "transition_reachable":
        return _score_continuation(
            edge,
            calibration,
            node_confidence,
        )
    if reason in {
        "overlap_close",
        "overlap_ambiguous",
        "overlap_unsynchronized",
    }:
        return _score_overlap(
            edge,
            calibration,
            node_confidence,
        )
    raise ValueError(f"unsupported candidate reason: {reason!r}")


def score_candidate_edges(
    edges: Sequence[Mapping[str, Any]],
    calibration: SoftLinkCalibration,
    *,
    node_confidence: Mapping[str, float] | None = None,
) -> list[ScoredLink]:
    candidates = _candidate_edges(edges)
    return [
        score_candidate_edge(
            edge,
            calibration,
            node_confidence=node_confidence,
        )
        for edge in candidates
    ]


def load_node_confidence(
    nodes_jsonl: str | Path,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for row in iter_jsonl(nodes_jsonl):
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id in result:
            raise ValueError(f"duplicate node: {tracklet_id}")
        result[tracklet_id] = _clamp01(
            _finite_float(
                "mean_confidence",
                row["mean_confidence"],
            )
        )
    return result


def build_scoring_summary(
    source_edges: Sequence[Mapping[str, Any]],
    scored_links: Sequence[ScoredLink],
    calibration: SoftLinkCalibration,
) -> dict[str, Any]:
    source_candidate_count = sum(
        1 for edge in source_edges if edge.get("status") == "candidate"
    )
    source_cannot_link_count = sum(
        1 for edge in source_edges if edge.get("status") == "cannot_link"
    )

    by_kind: dict[str, list[ScoredLink]] = {}
    for link in scored_links:
        by_kind.setdefault(link.link_kind, []).append(link)

    appearance_available_count = sum(
        1
        for link in scored_links
        if link.components["appearance"]["available"]
    )

    return {
        "schema_version": 1,
        "scoring_version": SCORING_VERSION,
        "identity_decisions": False,
        "rank_score_is_probability": False,
        "hard_constraints_modified": False,
        "source_pair_count": len(source_edges),
        "source_candidate_count": source_candidate_count,
        "source_cannot_link_count": source_cannot_link_count,
        "scored_link_count": len(scored_links),
        "appearance_available_count": appearance_available_count,
        "appearance_missing_count": (
            len(scored_links) - appearance_available_count
        ),
        "link_kind_counts": {
            kind: len(links)
            for kind, links in sorted(by_kind.items())
        },
        "rank_score": _stats(
            [link.rank_score for link in scored_links]
        ),
        "evidence_confidence": _stats(
            [link.evidence_confidence for link in scored_links]
        ),
        "rank_score_by_kind": {
            kind: _stats([link.rank_score for link in links])
            for kind, links in sorted(by_kind.items())
        },
        "evidence_confidence_by_kind": {
            kind: _stats(
                [link.evidence_confidence for link in links]
            )
            for kind, links in sorted(by_kind.items())
        },
        "calibration": calibration.to_dict(),
    }
