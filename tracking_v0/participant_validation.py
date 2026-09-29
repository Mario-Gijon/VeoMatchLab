from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from math import isfinite
from statistics import median
from typing import Mapping, Sequence

from .models import PositionKind, TrackSample
from .quality import signed_field_depth


class ParticipantDecision(str, Enum):
    """Final offline decision for a resolved identity component."""

    KEEP = "keep"
    UNCERTAIN = "uncertain"
    REJECT = "reject"


@dataclass(slots=True, frozen=True)
class ParticipantValidationConfig:
    """
    Conservative final participant validation for Tracking V2.3.

    This stage runs AFTER global identity resolution.

    It deliberately does NOT force a target count such as 23. Instead, it
    rejects only two high-confidence non-participant patterns that remain
    after identity consolidation:

    1) persistent_outside_field:
       a well-observed identity that remains clearly outside the strict field;

    2) ephemeral_sparse_orphan:
       a tiny identity with almost no real observational support and no stable
       temporal presence.

    Everything else is KEEP or UNCERTAIN and remains available for output.
    """

    # Strong positive participant evidence.
    keep_min_observed_samples: int = 20
    keep_min_observed_fraction: float = 0.50
    keep_min_strict_fraction: float = 0.80
    keep_min_observed_span_seconds: float = 1.50

    # Persistent person clearly outside the playing field.
    outside_min_observed_samples: int = 15
    outside_min_observed_span_seconds: float = 1.50
    outside_max_strict_fraction: float = 0.05
    outside_max_field_depth_m: float = -1.00

    # Tiny orphan identity. This is intentionally based on real OBSERVED
    # support, not total lifetime, because a forward tracker may coast for
    # seconds after only one or two detections.
    sparse_max_observed_samples: int = 3
    sparse_max_observed_fraction: float = 0.35
    sparse_max_longest_observed_run_seconds: float = 0.30

    contiguous_gap_multiplier: float = 1.60
    contiguous_gap_floor_seconds: float = 0.10

    def __post_init__(self) -> None:
        counts = (
            self.keep_min_observed_samples,
            self.outside_min_observed_samples,
            self.sparse_max_observed_samples,
        )
        if any(value < 1 for value in counts):
            raise ValueError("ParticipantValidationConfig counts must be >= 1")

        fractions = (
            self.keep_min_observed_fraction,
            self.keep_min_strict_fraction,
            self.outside_max_strict_fraction,
            self.sparse_max_observed_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError(
                "ParticipantValidationConfig fractions must be in [0, 1]"
            )

        non_negative = (
            self.keep_min_observed_span_seconds,
            self.outside_min_observed_span_seconds,
            self.sparse_max_longest_observed_run_seconds,
            self.contiguous_gap_floor_seconds,
        )
        if any(value < 0.0 for value in non_negative):
            raise ValueError(
                "ParticipantValidationConfig time thresholds must be >= 0"
            )

        if self.contiguous_gap_multiplier <= 0.0:
            raise ValueError("contiguous_gap_multiplier must be > 0")


@dataclass(slots=True, frozen=True)
class ParticipantMetrics:
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

    min_field_depth_m: float
    median_field_depth_m: float
    p90_field_depth_m: float
    max_field_depth_m: float

    longest_observed_run_seconds: float
    median_sample_period_seconds: float

    mean_observed_confidence: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class ParticipantValidationResult:
    track_id: int
    decision: ParticipantDecision
    reasons: tuple[str, ...]
    metrics: ParticipantMetrics

    @property
    def accepted(self) -> bool:
        return self.decision is not ParticipantDecision.REJECT

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "decision": self.decision.value,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "metrics": self.metrics.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class ParticipantValidationReport:
    results: Mapping[int, ParticipantValidationResult]

    @property
    def kept_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.decision is ParticipantDecision.KEEP
        )

    @property
    def uncertain_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.decision is ParticipantDecision.UNCERTAIN
        )

    @property
    def rejected_track_ids(self) -> tuple[int, ...]:
        return tuple(
            track_id
            for track_id, result in sorted(self.results.items())
            if result.decision is ParticipantDecision.REJECT
        )

    @property
    def accepted_track_ids(self) -> tuple[int, ...]:
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


def validate_participants(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    field_length: float,
    field_width: float,
    config: ParticipantValidationConfig | None = None,
) -> ParticipantValidationReport:
    """
    Validate globally resolved identities as final participant candidates.

    No expected roster size is used. The classifier works only from temporal
    support and field occupancy of each already-resolved identity.
    """
    cfg = config or ParticipantValidationConfig()
    results: dict[int, ParticipantValidationResult] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(track_id, raw_samples)
        if not samples:
            continue

        metrics = _metrics(
            samples,
            field_length=field_length,
            field_width=field_width,
            config=cfg,
        )
        decision, reasons = _classify(metrics, cfg)

        results[track_id] = ParticipantValidationResult(
            track_id=track_id,
            decision=decision,
            reasons=tuple(reasons),
            metrics=metrics,
        )

    return ParticipantValidationReport(results=results)


def filter_participants(
    trajectories: Mapping[int, Sequence[TrackSample]],
    report: ParticipantValidationReport,
) -> dict[int, tuple[TrackSample, ...]]:
    """
    Preserve KEEP + UNCERTAIN and remove only high-confidence REJECT identities.
    """
    accepted = set(report.accepted_track_ids)

    return {
        track_id: tuple(samples)
        for track_id, samples in trajectories.items()
        if track_id in accepted
    }


def _metrics(
    samples: tuple[TrackSample, ...],
    *,
    field_length: float,
    field_width: float,
    config: ParticipantValidationConfig,
) -> ParticipantMetrics:
    observed = tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.OBSERVED
    )
    predicted = tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.PREDICTED
    )
    interpolated = tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.INTERPOLATED
    )

    sample_count = len(samples)
    observed_count = len(observed)

    lifetime_seconds = max(
        0.0,
        float(samples[-1].timestamp - samples[0].timestamp),
    )

    observed_span_seconds = (
        max(
            0.0,
            float(observed[-1].timestamp - observed[0].timestamp),
        )
        if observed
        else 0.0
    )

    observed_fraction = (
        observed_count / sample_count
        if sample_count
        else 0.0
    )

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
    strict_fraction = (
        strict_count / observed_count
        if observed_count
        else 0.0
    )

    if depths:
        sorted_depths = sorted(depths)
        min_depth = float(sorted_depths[0])
        median_depth = float(median(sorted_depths))
        p90_depth = float(_percentile(sorted_depths, 0.90))
        max_depth = float(sorted_depths[-1])
    else:
        min_depth = median_depth = p90_depth = max_depth = float("-inf")

    sample_period = _median_positive_delta(samples)
    longest_run = _longest_observed_run(
        observed,
        nominal_period=sample_period,
        config=config,
    )

    confidences = [
        float(sample.confidence)
        for sample in observed
        if sample.confidence is not None
        and isfinite(float(sample.confidence))
    ]

    return ParticipantMetrics(
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
        min_field_depth_m=min_depth,
        median_field_depth_m=median_depth,
        p90_field_depth_m=p90_depth,
        max_field_depth_m=max_depth,
        longest_observed_run_seconds=longest_run,
        median_sample_period_seconds=sample_period,
        mean_observed_confidence=(
            sum(confidences) / len(confidences)
            if confidences
            else None
        ),
    )


def _classify(
    metrics: ParticipantMetrics,
    config: ParticipantValidationConfig,
) -> tuple[ParticipantDecision, list[str]]:
    """
    Final conservative participant decision.

    REJECT is checked only for patterns that are incompatible with stable
    participation. Otherwise strong positive evidence gives KEEP and ambiguous
    identities stay UNCERTAIN.
    """
    persistent_outside = (
        metrics.observed_count >= config.outside_min_observed_samples
        and metrics.observed_span_seconds
        >= config.outside_min_observed_span_seconds
        and metrics.strict_observed_fraction
        <= config.outside_max_strict_fraction
        and metrics.max_field_depth_m
        <= config.outside_max_field_depth_m
    )

    if persistent_outside:
        return ParticipantDecision.REJECT, [
            "persistent_outside_field",
            "well_observed_but_never_enters_strict_field",
        ]

    ephemeral_sparse_orphan = (
        metrics.observed_count <= config.sparse_max_observed_samples
        and metrics.observed_fraction
        <= config.sparse_max_observed_fraction
        and metrics.longest_observed_run_seconds
        <= config.sparse_max_longest_observed_run_seconds
    )

    if ephemeral_sparse_orphan:
        return ParticipantDecision.REJECT, [
            "ephemeral_sparse_orphan",
            "insufficient_real_temporal_support",
        ]

    stable_participant = (
        metrics.observed_count >= config.keep_min_observed_samples
        and metrics.observed_fraction >= config.keep_min_observed_fraction
        and metrics.strict_observed_fraction >= config.keep_min_strict_fraction
        and metrics.observed_span_seconds
        >= config.keep_min_observed_span_seconds
    )

    if stable_participant:
        return ParticipantDecision.KEEP, [
            "stable_resolved_participant",
        ]

    return ParticipantDecision.UNCERTAIN, [
        "insufficient_evidence_for_keep_or_reject",
    ]


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.frame_index,
                sample.timestamp,
            ),
        )
    )

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match "
                f"sample track_id {sample.track_id}"
            )

        if not isfinite(sample.timestamp):
            raise ValueError("Track sample timestamp must be finite")

        if not all(
            isfinite(value)
            for value in (
                sample.x,
                sample.z,
                sample.vx,
                sample.vz,
            )
        ):
            raise ValueError(
                f"Track {track_id} contains non-finite state"
            )

    return ordered


def _median_positive_delta(
    samples: Sequence[TrackSample],
) -> float:
    deltas = [
        float(b.timestamp - a.timestamp)
        for a, b in zip(samples, samples[1:])
        if b.timestamp > a.timestamp
    ]
    return float(median(deltas)) if deltas else 0.0


def _longest_observed_run(
    observed: Sequence[TrackSample],
    *,
    nominal_period: float,
    config: ParticipantValidationConfig,
) -> float:
    if not observed:
        return 0.0

    if len(observed) == 1:
        return max(0.0, nominal_period)

    threshold = max(
        config.contiguous_gap_floor_seconds,
        nominal_period * config.contiguous_gap_multiplier,
    )

    longest = max(0.0, nominal_period)
    run_start = observed[0].timestamp
    previous = observed[0].timestamp

    for sample in observed[1:]:
        gap = float(sample.timestamp - previous)

        if gap <= threshold:
            longest = max(
                longest,
                float(sample.timestamp - run_start) + nominal_period,
            )
        else:
            run_start = sample.timestamp
            longest = max(longest, nominal_period)

        previous = sample.timestamp

    return float(longest)


def _percentile(
    sorted_values: Sequence[float],
    q: float,
) -> float:
    if not sorted_values:
        raise ValueError("percentile requires data")

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
