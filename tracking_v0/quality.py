from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from statistics import median
from typing import Mapping, Sequence

from .models import PositionKind, TrackSample


class TrackQualityDecision(str, Enum):
    """Offline decision about whether a temporal track should reach final output."""

    KEEP = "keep"
    UNCERTAIN = "uncertain"
    REJECT = "reject"


@dataclass(slots=True, frozen=True)
class TrackQualityConfig:
    """
    Conservative thresholds for Tracking V2.1.

    The classifier is deliberately asymmetric:
    - KEEP may be granted from strong evidence that a person participates.
    - REJECT requires strong temporal evidence of a sideline/transient ghost.
    - Everything else is UNCERTAIN and is preserved in final output.

    No rule uses a target count such as 23 players.
    """

    edge_band_m: float = 0.75
    deep_inside_m: float = 1.50

    keep_min_deep_observed_seconds: float = 0.40
    keep_min_deep_observed_samples: int = 4
    keep_min_long_observed_span_seconds: float = 8.0
    keep_min_long_observed_fraction: float = 0.55
    keep_min_long_strict_fraction: float = 0.85
    keep_min_long_observed_samples: int = 30

    reject_max_field_depth_m: float = 0.75
    reject_min_edge_fraction: float = 0.90
    reject_max_observed_span_seconds: float = 3.0
    reject_max_observed_fraction: float = 0.55
    reject_max_observed_samples: int = 24

    reject_sparse_max_observed_samples: int = 8
    reject_sparse_max_observed_span_seconds: float = 1.0
    reject_sparse_max_observed_fraction: float = 0.35

    contiguous_gap_multiplier: float = 1.60
    contiguous_gap_floor_seconds: float = 0.10

    def __post_init__(self) -> None:
        non_negative = (
            self.edge_band_m,
            self.deep_inside_m,
            self.keep_min_deep_observed_seconds,
            self.keep_min_long_observed_span_seconds,
            self.reject_max_field_depth_m,
            self.reject_max_observed_span_seconds,
            self.reject_sparse_max_observed_span_seconds,
            self.contiguous_gap_floor_seconds,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError("TrackQualityConfig distance/time thresholds must be >= 0")

        fractions = (
            self.keep_min_long_observed_fraction,
            self.keep_min_long_strict_fraction,
            self.reject_min_edge_fraction,
            self.reject_max_observed_fraction,
            self.reject_sparse_max_observed_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("TrackQualityConfig fractions must be in [0, 1]")

        counts = (
            self.keep_min_deep_observed_samples,
            self.keep_min_long_observed_samples,
            self.reject_max_observed_samples,
            self.reject_sparse_max_observed_samples,
        )
        if any(value < 1 for value in counts):
            raise ValueError("TrackQualityConfig sample-count thresholds must be >= 1")

        if self.contiguous_gap_multiplier <= 0.0:
            raise ValueError("contiguous_gap_multiplier must be > 0")


@dataclass(slots=True, frozen=True)
class TrackQualityMetrics:
    track_id: int
    sample_count: int
    observed_count: int
    predicted_count: int
    interpolated_count: int

    lifetime_seconds: float
    observed_span_seconds: float
    observed_fraction: float

    strict_observed_count: int
    strict_observed_fraction: float
    edge_observed_count: int
    edge_observed_fraction: float
    deep_observed_count: int
    deep_observed_fraction: float
    deep_observed_seconds: float

    min_field_depth_m: float
    median_field_depth_m: float
    p90_field_depth_m: float
    max_field_depth_m: float

    longest_observed_run_seconds: float
    longest_observation_gap_seconds: float
    median_sample_period_seconds: float

    mean_observed_confidence: float | None
    min_observed_confidence: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class TrackQualityResult:
    track_id: int
    decision: TrackQualityDecision
    reasons: tuple[str, ...]
    metrics: TrackQualityMetrics

    @property
    def accepted(self) -> bool:
        """UNCERTAIN is intentionally preserved; only REJECT is removed."""
        return self.decision is not TrackQualityDecision.REJECT

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "decision": self.decision.value,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class TrackQualityReport:
    results: Mapping[int, TrackQualityResult]

    @property
    def kept_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.decision is TrackQualityDecision.KEEP
        )

    @property
    def uncertain_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.decision is TrackQualityDecision.UNCERTAIN
        )

    @property
    def rejected_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.decision is TrackQualityDecision.REJECT
        )

    @property
    def accepted_track_ids(self) -> tuple[int, ...]:
        """Tracks allowed to reach final output: KEEP + UNCERTAIN."""
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.accepted
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "kept_track_ids": list(self.kept_track_ids),
            "uncertain_track_ids": list(self.uncertain_track_ids),
            "rejected_track_ids": list(self.rejected_track_ids),
            "accepted_track_ids": list(self.accepted_track_ids),
            "tracks": {
                str(track_id): result.to_dict()
                for track_id, result in sorted(self.results.items())
            },
        }


def signed_field_depth(
    x: float,
    z: float,
    *,
    field_length: float,
    field_width: float,
) -> float:
    """
    Signed distance from a point to the nearest rectangular field boundary.

    Positive values are strictly inside the field.
    Zero lies on a field boundary.
    Negative values are outside the strict field.
    """
    if not all(isfinite(value) for value in (x, z, field_length, field_width)):
        raise ValueError("field coordinates and dimensions must be finite")
    if field_length <= 0.0 or field_width <= 0.0:
        raise ValueError("field dimensions must be > 0")

    return min(
        field_length / 2.0 - abs(float(x)),
        field_width / 2.0 - abs(float(z)),
    )


def analyze_track_quality(
    samples: Sequence[TrackSample],
    *,
    field_length: float,
    field_width: float,
    config: TrackQualityConfig | None = None,
) -> TrackQualityResult:
    """
    Analyze one complete forward trajectory using temporal evidence.

    This function does not modify, smooth, stitch, or interpolate the track.
    It only decides whether there is sufficiently strong evidence to reject
    the whole trajectory as a sideline/transient ghost.
    """
    cfg = config or TrackQualityConfig()
    ordered = _validated_samples(samples)
    track_id = ordered[0].track_id

    metrics = _metrics_for_track(
        ordered,
        field_length=field_length,
        field_width=field_width,
        config=cfg,
    )
    decision, reasons = _classify(metrics, cfg)

    return TrackQualityResult(
        track_id=track_id,
        decision=decision,
        reasons=tuple(reasons),
        metrics=metrics,
    )


def analyze_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    field_length: float,
    field_width: float,
    config: TrackQualityConfig | None = None,
) -> TrackQualityReport:
    """Analyze every trajectory independently; no target player count is used."""
    cfg = config or TrackQualityConfig()
    results: dict[int, TrackQualityResult] = {}

    for track_id, samples in sorted(trajectories.items()):
        if not samples:
            continue
        result = analyze_track_quality(
            samples,
            field_length=field_length,
            field_width=field_width,
            config=cfg,
        )
        if result.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match sample track_id "
                f"{result.track_id}"
            )
        results[track_id] = result

    return TrackQualityReport(results=results)


def filter_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
    report: TrackQualityReport,
) -> dict[int, tuple[TrackSample, ...]]:
    """
    Return only tracks allowed into final output.

    KEEP and UNCERTAIN are retained.
    Only tracks classified REJECT are removed.
    """
    accepted = set(report.accepted_track_ids)
    return {
        track_id: tuple(samples)
        for track_id, samples in trajectories.items()
        if track_id in accepted
    }


def _validated_samples(samples: Sequence[TrackSample]) -> tuple[TrackSample, ...]:
    if not samples:
        raise ValueError("Track quality analysis requires at least one sample")

    ordered = tuple(sorted(samples, key=lambda item: (item.timestamp, item.frame_index)))
    track_id = ordered[0].track_id
    previous_timestamp: float | None = None
    previous_frame_index: int | None = None

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError("All samples must belong to the same track_id")
        if not isfinite(sample.timestamp):
            raise ValueError("Track sample timestamp must be finite")
        if not all(isfinite(value) for value in (sample.x, sample.z, sample.vx, sample.vz)):
            raise ValueError("Track sample position/velocity must be finite")

        if previous_timestamp is not None and sample.timestamp < previous_timestamp:
            raise ValueError("Track sample timestamps must be non-decreasing")
        if previous_frame_index is not None and sample.frame_index < previous_frame_index:
            raise ValueError("Track sample frame indices must be non-decreasing")

        previous_timestamp = sample.timestamp
        previous_frame_index = sample.frame_index

    return ordered


def _metrics_for_track(
    samples: tuple[TrackSample, ...],
    *,
    field_length: float,
    field_width: float,
    config: TrackQualityConfig,
) -> TrackQualityMetrics:
    observed = tuple(
        sample for sample in samples
        if sample.position_kind is PositionKind.OBSERVED
    )
    predicted = tuple(
        sample for sample in samples
        if sample.position_kind is PositionKind.PREDICTED
    )
    interpolated = tuple(
        sample for sample in samples
        if sample.position_kind is PositionKind.INTERPOLATED
    )

    sample_count = len(samples)
    observed_count = len(observed)

    lifetime_seconds = max(
        0.0,
        float(samples[-1].timestamp - samples[0].timestamp),
    )

    if observed:
        observed_span_seconds = max(
            0.0,
            float(observed[-1].timestamp - observed[0].timestamp),
        )
    else:
        observed_span_seconds = 0.0

    observed_fraction = observed_count / sample_count if sample_count else 0.0

    depths = [
        signed_field_depth(
            sample.x,
            sample.z,
            field_length=field_length,
            field_width=field_width,
        )
        for sample in observed
    ]

    strict_count = sum(depth >= 0.0 for depth in depths)
    edge_count = sum(depth <= config.edge_band_m for depth in depths)
    deep_count = sum(depth >= config.deep_inside_m for depth in depths)

    strict_fraction = strict_count / observed_count if observed_count else 0.0
    edge_fraction = edge_count / observed_count if observed_count else 0.0
    deep_fraction = deep_count / observed_count if observed_count else 0.0

    sample_period = _median_positive_delta(samples)
    deep_seconds = deep_count * sample_period

    if depths:
        sorted_depths = sorted(depths)
        min_depth = float(sorted_depths[0])
        median_depth = float(median(sorted_depths))
        p90_depth = float(_percentile(sorted_depths, 0.90))
        max_depth = float(sorted_depths[-1])
    else:
        min_depth = median_depth = p90_depth = max_depth = float("-inf")

    observed_runs, observed_gaps = _observed_run_metrics(
        observed,
        nominal_period=sample_period,
        config=config,
    )

    confidences = [
        float(sample.confidence)
        for sample in observed
        if sample.confidence is not None and isfinite(float(sample.confidence))
    ]

    return TrackQualityMetrics(
        track_id=samples[0].track_id,
        sample_count=sample_count,
        observed_count=observed_count,
        predicted_count=len(predicted),
        interpolated_count=len(interpolated),
        lifetime_seconds=lifetime_seconds,
        observed_span_seconds=observed_span_seconds,
        observed_fraction=observed_fraction,
        strict_observed_count=strict_count,
        strict_observed_fraction=strict_fraction,
        edge_observed_count=edge_count,
        edge_observed_fraction=edge_fraction,
        deep_observed_count=deep_count,
        deep_observed_fraction=deep_fraction,
        deep_observed_seconds=deep_seconds,
        min_field_depth_m=min_depth,
        median_field_depth_m=median_depth,
        p90_field_depth_m=p90_depth,
        max_field_depth_m=max_depth,
        longest_observed_run_seconds=observed_runs,
        longest_observation_gap_seconds=observed_gaps,
        median_sample_period_seconds=sample_period,
        mean_observed_confidence=(
            sum(confidences) / len(confidences)
            if confidences else None
        ),
        min_observed_confidence=min(confidences) if confidences else None,
    )


def _classify(
    metrics: TrackQualityMetrics,
    config: TrackQualityConfig,
) -> tuple[TrackQualityDecision, list[str]]:
    """
    Conservative rule set.

    Ordering is deliberate:
    1. Strong positive participation evidence wins first.
    2. Strong edge/transient ghost evidence may reject.
    3. Everything else remains UNCERTAIN and therefore visible.
    """
    reasons: list[str] = []

    strong_deep_evidence = (
        metrics.deep_observed_count >= config.keep_min_deep_observed_samples
        and metrics.deep_observed_seconds
        >= config.keep_min_deep_observed_seconds
    )

    strong_long_evidence = (
        metrics.observed_count >= config.keep_min_long_observed_samples
        and metrics.observed_span_seconds
        >= config.keep_min_long_observed_span_seconds
        and metrics.observed_fraction
        >= config.keep_min_long_observed_fraction
        and metrics.strict_observed_fraction
        >= config.keep_min_long_strict_fraction
    )

    if strong_deep_evidence:
        reasons.append("convincing_deep_field_presence")
    if strong_long_evidence:
        reasons.append("long_stable_strict_field_presence")

    if strong_deep_evidence or strong_long_evidence:
        return TrackQualityDecision.KEEP, reasons

    edge_confined = (
        metrics.observed_count > 0
        and metrics.edge_observed_fraction >= config.reject_min_edge_fraction
        and metrics.max_field_depth_m <= config.reject_max_field_depth_m
    )

    sparse_edge_transient = (
        edge_confined
        and metrics.observed_count <= config.reject_sparse_max_observed_samples
        and metrics.observed_span_seconds
        <= config.reject_sparse_max_observed_span_seconds
        and metrics.observed_fraction
        <= config.reject_sparse_max_observed_fraction
    )

    sideline_transient = (
        edge_confined
        and metrics.observed_count <= config.reject_max_observed_samples
        and metrics.observed_span_seconds
        <= config.reject_max_observed_span_seconds
        and metrics.observed_fraction
        <= config.reject_max_observed_fraction
    )

    if sparse_edge_transient:
        return TrackQualityDecision.REJECT, [
            "edge_confined",
            "sparse_observations",
            "brief_observed_presence",
            "mostly_unobserved_track",
            "never_convincingly_inside",
        ]

    if sideline_transient:
        return TrackQualityDecision.REJECT, [
            "edge_confined",
            "temporary_sideline_presence",
            "never_convincingly_inside",
        ]

    if metrics.observed_count == 0:
        reasons.append("no_observed_samples")
    if edge_confined:
        reasons.append("edge_confined_but_not_safe_to_reject")
    if metrics.max_field_depth_m < config.deep_inside_m:
        reasons.append("no_deep_field_evidence")
    if metrics.observed_fraction < config.reject_max_observed_fraction:
        reasons.append("low_observed_fraction")
    if not reasons:
        reasons.append("insufficient_evidence_for_keep_or_reject")

    return TrackQualityDecision.UNCERTAIN, reasons


def _median_positive_delta(samples: Sequence[TrackSample]) -> float:
    deltas = [
        float(b.timestamp - a.timestamp)
        for a, b in zip(samples, samples[1:])
        if b.timestamp > a.timestamp
    ]
    if not deltas:
        return 0.0
    return float(median(deltas))


def _observed_run_metrics(
    observed: Sequence[TrackSample],
    *,
    nominal_period: float,
    config: TrackQualityConfig,
) -> tuple[float, float]:
    if not observed:
        return 0.0, 0.0
    if len(observed) == 1:
        return max(0.0, nominal_period), 0.0

    threshold = max(
        config.contiguous_gap_floor_seconds,
        nominal_period * config.contiguous_gap_multiplier,
    )

    longest_run = max(0.0, nominal_period)
    current_run_start = observed[0].timestamp
    previous_timestamp = observed[0].timestamp
    longest_gap = 0.0

    for sample in observed[1:]:
        gap = float(sample.timestamp - previous_timestamp)

        if gap <= threshold:
            longest_run = max(
                longest_run,
                float(sample.timestamp - current_run_start) + nominal_period,
            )
        else:
            longest_gap = max(longest_gap, gap)
            current_run_start = sample.timestamp
            longest_run = max(longest_run, nominal_period)

        previous_timestamp = sample.timestamp

    return float(longest_run), float(longest_gap)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires at least one value")
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )
