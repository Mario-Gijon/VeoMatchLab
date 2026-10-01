"""Tracking V1 Hito 7G.1: trajectory segments/endpoints for consolidated identities.

Revision V1
-----------
The first 7G.1 artifact exposed only the outer start/end of each identity.  The
real clip shows that many already-consolidated identities contain internal
observation gaps.  Global reconstruction must therefore reason about *every
contiguous observed segment and every internal gap*, otherwise an orphan
fragment that belongs inside a known identity's gap could never be recovered.

This stage remains diagnostic only:
* X/Z come directly from canonical TrackletObservation field coordinates.
* observations are split into contiguous segments using max_contiguous_gap_s;
* endpoint velocity is estimated only inside one contiguous segment;
* velocity is soft evidence, never a hard constraint;
* missing velocity is unknown, not negative;
* no identity decisions are changed and no population count is forced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

from tracking.tracklets.models import Tracklet, TrackletObservation


IDENTITY_TRAJECTORY_ENDPOINT_VERSION = "identity-trajectory-endpoint-v2-segments"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


@dataclass(frozen=True, slots=True)
class IdentityTrajectoryEndpointConfig:
    velocity_window_s: float = 1.00
    max_contiguous_gap_s: float = 0.45
    velocity_min_samples: int = 2
    velocity_min_span_s: float = 0.08
    velocity_residual_scale_m: float = 0.75
    velocity_reliable_span_s: float = 0.50
    velocity_reliable_samples: int = 5
    soft_speed_reference_mps: float = 12.0

    def __post_init__(self) -> None:
        for name in (
            "velocity_window_s",
            "max_contiguous_gap_s",
            "velocity_min_span_s",
            "velocity_residual_scale_m",
            "velocity_reliable_span_s",
            "soft_speed_reference_mps",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.velocity_min_samples < 2:
            raise ValueError("velocity_min_samples must be >= 2")
        if self.velocity_reliable_samples < 2:
            raise ValueError("velocity_reliable_samples must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EndpointVelocityEstimate:
    vx_mps: float
    vz_mps: float
    speed_mps: float
    sample_count: int
    span_s: float
    mean_observation_confidence: float
    residual_rmse_m: float
    reliability: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "vx_mps": self.vx_mps,
            "vz_mps": self.vz_mps,
            "speed_mps": self.speed_mps,
            "sample_count": self.sample_count,
            "span_s": self.span_s,
            "mean_observation_confidence": self.mean_observation_confidence,
            "residual_rmse_m": self.residual_rmse_m,
            "reliability": self.reliability,
        }


@dataclass(frozen=True, slots=True)
class TrajectoryEndpoint:
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    observation_confidence: float
    tracklet_id: str
    velocity: EndpointVelocityEstimate | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "x": self.x,
            "z": self.z,
            "observation_confidence": self.observation_confidence,
            "tracklet_id": self.tracklet_id,
            "velocity": self.velocity.to_dict() if self.velocity else None,
        }


@dataclass(frozen=True, slots=True)
class IdentityTrajectorySegment:
    segment_index: int
    start: TrajectoryEndpoint
    end: TrajectoryEndpoint
    tracklet_ids: tuple[str, ...]
    observation_count: int
    unique_frame_count: int
    mean_observation_confidence: float

    @property
    def duration_s(self) -> float:
        return self.end.timestamp_s - self.start.timestamp_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_index": self.segment_index,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": len(self.tracklet_ids),
            "observation_count": self.observation_count,
            "unique_frame_count": self.unique_frame_count,
            "duration_s": self.duration_s,
            "mean_observation_confidence": self.mean_observation_confidence,
        }


@dataclass(frozen=True, slots=True)
class IdentityInternalGap:
    gap_index: int
    preceding_segment_index: int
    following_segment_index: int
    start_time_s: float
    end_time_s: float
    gap_s: float
    endpoint_distance_m: float
    straight_line_speed_mps: float | None
    preceding_tracklet_id: str
    following_tracklet_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_index": self.gap_index,
            "preceding_segment_index": self.preceding_segment_index,
            "following_segment_index": self.following_segment_index,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "gap_s": self.gap_s,
            "endpoint_distance_m": self.endpoint_distance_m,
            "straight_line_speed_mps": self.straight_line_speed_mps,
            "preceding_tracklet_id": self.preceding_tracklet_id,
            "following_tracklet_id": self.following_tracklet_id,
        }


@dataclass(frozen=True, slots=True)
class IdentityTrajectoryProfile:
    identity_id: str
    start: TrajectoryEndpoint
    end: TrajectoryEndpoint
    segments: tuple[IdentityTrajectorySegment, ...]
    internal_gaps: tuple[IdentityInternalGap, ...]
    tracklet_ids: tuple[str, ...]
    observation_count: int
    unique_frame_count: int
    duration_s: float
    observed_span_s: float
    mean_observation_confidence: float

    @property
    def internal_gap_count(self) -> int:
        return len(self.internal_gaps)

    @property
    def max_internal_gap_s(self) -> float:
        return max(
            (gap.gap_s for gap in self.internal_gaps),
            default=0.0,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 2,
            "trajectory_endpoint_version": IDENTITY_TRAJECTORY_ENDPOINT_VERSION,
            "identity_id": self.identity_id,
            "start": self.start.to_dict(),
            "end": self.end.to_dict(),
            "segments": [row.to_dict() for row in self.segments],
            "continuity_segment_count": len(self.segments),
            "internal_gaps": [row.to_dict() for row in self.internal_gaps],
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": len(self.tracklet_ids),
            "observation_count": self.observation_count,
            "unique_frame_count": self.unique_frame_count,
            "duration_s": self.duration_s,
            "observed_span_s": self.observed_span_s,
            "internal_gap_count": self.internal_gap_count,
            "max_internal_gap_s": self.max_internal_gap_s,
            "mean_observation_confidence": self.mean_observation_confidence,
            "start_velocity_available": self.start.velocity is not None,
            "end_velocity_available": self.end.velocity is not None,
            "identity_decision": None,
        }


def _ordered_identity_observations(
    *,
    tracklet_ids: Sequence[str],
    tracklets_by_id: Mapping[str, Tracklet],
) -> list[tuple[str, TrackletObservation]]:
    observations: list[tuple[str, TrackletObservation]] = []
    seen_detection_ids: set[str] = set()

    for tracklet_id in tracklet_ids:
        try:
            tracklet = tracklets_by_id[tracklet_id]
        except KeyError as exc:
            raise ValueError(
                f"identity references unknown tracklet {tracklet_id}"
            ) from exc

        for observation in tracklet.observations:
            if observation.detection_id in seen_detection_ids:
                raise ValueError(
                    "identity contains duplicate detection "
                    f"{observation.detection_id}"
                )
            seen_detection_ids.add(observation.detection_id)
            observations.append((tracklet_id, observation))

    observations.sort(
        key=lambda item: (
            item[1].timestamp_s,
            item[1].frame_index,
            item[0],
            item[1].detection_id,
        )
    )
    return observations


def _split_contiguous_segments(
    observations: Sequence[tuple[str, TrackletObservation]],
    *,
    config: IdentityTrajectoryEndpointConfig,
) -> list[list[tuple[str, TrackletObservation]]]:
    if not observations:
        return []

    segments: list[list[tuple[str, TrackletObservation]]] = [
        [observations[0]]
    ]
    for item in observations[1:]:
        previous = segments[-1][-1][1]
        current = item[1]
        gap_s = current.timestamp_s - previous.timestamp_s
        if gap_s < -1e-12:
            raise ValueError("observations are not temporally ordered")
        if gap_s > config.max_contiguous_gap_s + 1e-12:
            segments.append([item])
        else:
            segments[-1].append(item)
    return segments


def _endpoint_window(
    observations: Sequence[tuple[str, TrackletObservation]],
    *,
    side: str,
    config: IdentityTrajectoryEndpointConfig,
) -> list[tuple[str, TrackletObservation]]:
    if not observations:
        return []

    if side == "start":
        anchor_t = observations[0][1].timestamp_s
        return [
            row for row in observations
            if row[1].timestamp_s - anchor_t
            <= config.velocity_window_s + 1e-12
        ]

    if side == "end":
        anchor_t = observations[-1][1].timestamp_s
        return [
            row for row in observations
            if anchor_t - row[1].timestamp_s
            <= config.velocity_window_s + 1e-12
        ]

    raise ValueError("side must be 'start' or 'end'")


def _weighted_linear_velocity(
    rows: Sequence[tuple[str, TrackletObservation]],
    *,
    config: IdentityTrajectoryEndpointConfig,
) -> EndpointVelocityEstimate | None:
    if len(rows) < config.velocity_min_samples:
        return None

    observations = [item[1] for item in rows]
    span_s = observations[-1].timestamp_s - observations[0].timestamp_s
    if span_s < config.velocity_min_span_s:
        return None

    weights = [max(1e-3, float(item.confidence)) for item in observations]
    total_weight = sum(weights)

    mean_t = sum(
        w * item.timestamp_s for w, item in zip(weights, observations)
    ) / total_weight
    mean_x = sum(
        w * item.x for w, item in zip(weights, observations)
    ) / total_weight
    mean_z = sum(
        w * item.z for w, item in zip(weights, observations)
    ) / total_weight

    denominator = sum(
        w * (item.timestamp_s - mean_t) ** 2
        for w, item in zip(weights, observations)
    )
    if denominator <= 1e-12:
        return None

    vx = sum(
        w * (item.timestamp_s - mean_t) * (item.x - mean_x)
        for w, item in zip(weights, observations)
    ) / denominator
    vz = sum(
        w * (item.timestamp_s - mean_t) * (item.z - mean_z)
        for w, item in zip(weights, observations)
    ) / denominator

    residual_sq = 0.0
    for w, item in zip(weights, observations):
        dt = item.timestamp_s - mean_t
        px = mean_x + vx * dt
        pz = mean_z + vz * dt
        residual_sq += w * (
            (item.x - px) ** 2 + (item.z - pz) ** 2
        )
    residual_rmse = math.sqrt(residual_sq / total_weight)

    mean_confidence = total_weight / len(observations)
    speed = math.hypot(vx, vz)
    sample_factor = min(
        1.0,
        (len(observations) - 1)
        / max(1, config.velocity_reliable_samples - 1),
    )
    span_factor = min(1.0, span_s / config.velocity_reliable_span_s)
    residual_factor = math.exp(
        -residual_rmse / config.velocity_residual_scale_m
    )
    speed_factor = 1.0 / (
        1.0
        + max(
            0.0,
            speed - config.soft_speed_reference_mps,
        ) / config.soft_speed_reference_mps
    )
    reliability = _clamp01(
        mean_confidence
        * math.sqrt(sample_factor * span_factor)
        * residual_factor
        * speed_factor
    )

    return EndpointVelocityEstimate(
        vx_mps=float(vx),
        vz_mps=float(vz),
        speed_mps=float(speed),
        sample_count=len(observations),
        span_s=float(span_s),
        mean_observation_confidence=float(mean_confidence),
        residual_rmse_m=float(residual_rmse),
        reliability=float(reliability),
    )


def _make_segment(
    *,
    segment_index: int,
    rows: Sequence[tuple[str, TrackletObservation]],
    config: IdentityTrajectoryEndpointConfig,
) -> IdentityTrajectorySegment:
    first_tracklet, first = rows[0]
    last_tracklet, last = rows[-1]

    start_velocity = _weighted_linear_velocity(
        _endpoint_window(rows, side="start", config=config),
        config=config,
    )
    end_velocity = _weighted_linear_velocity(
        _endpoint_window(rows, side="end", config=config),
        config=config,
    )

    confidences = [item.confidence for _, item in rows]
    return IdentityTrajectorySegment(
        segment_index=segment_index,
        start=TrajectoryEndpoint(
            frame_index=first.frame_index,
            timestamp_s=first.timestamp_s,
            x=first.x,
            z=first.z,
            observation_confidence=first.confidence,
            tracklet_id=first_tracklet,
            velocity=start_velocity,
        ),
        end=TrajectoryEndpoint(
            frame_index=last.frame_index,
            timestamp_s=last.timestamp_s,
            x=last.x,
            z=last.z,
            observation_confidence=last.confidence,
            tracklet_id=last_tracklet,
            velocity=end_velocity,
        ),
        tracklet_ids=tuple(sorted({tracklet_id for tracklet_id, _ in rows})),
        observation_count=len(rows),
        unique_frame_count=len({item.frame_index for _, item in rows}),
        mean_observation_confidence=float(
            sum(confidences) / len(confidences)
        ),
    )


def _make_internal_gaps(
    segments: Sequence[IdentityTrajectorySegment],
) -> tuple[IdentityInternalGap, ...]:
    result: list[IdentityInternalGap] = []
    for gap_index, (left, right) in enumerate(
        zip(segments, segments[1:]),
        start=1,
    ):
        gap_s = right.start.timestamp_s - left.end.timestamp_s
        distance = math.hypot(
            right.start.x - left.end.x,
            right.start.z - left.end.z,
        )
        straight_speed = (
            distance / gap_s if gap_s > 1e-12 else None
        )
        result.append(
            IdentityInternalGap(
                gap_index=gap_index,
                preceding_segment_index=left.segment_index,
                following_segment_index=right.segment_index,
                start_time_s=left.end.timestamp_s,
                end_time_s=right.start.timestamp_s,
                gap_s=float(gap_s),
                endpoint_distance_m=float(distance),
                straight_line_speed_mps=(
                    float(straight_speed)
                    if straight_speed is not None
                    else None
                ),
                preceding_tracklet_id=left.end.tracklet_id,
                following_tracklet_id=right.start.tracklet_id,
            )
        )
    return tuple(result)


def build_identity_trajectory_profiles(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    tracklet_rows: Sequence[Mapping[str, Any] | Tracklet],
    config: IdentityTrajectoryEndpointConfig | None = None,
) -> tuple[IdentityTrajectoryProfile, ...]:
    selected = config or IdentityTrajectoryEndpointConfig()

    tracklets_by_id: dict[str, Tracklet] = {}
    for raw in tracklet_rows:
        tracklet = raw if isinstance(raw, Tracklet) else Tracklet.from_dict(raw)
        if tracklet.tracklet_id in tracklets_by_id:
            raise ValueError(f"duplicate tracklet {tracklet.tracklet_id}")
        tracklets_by_id[tracklet.tracklet_id] = tracklet

    result: list[IdentityTrajectoryProfile] = []
    seen_identity_ids: set[str] = set()

    for identity_row in identity_rows:
        identity_id = str(identity_row["identity_id"])
        if identity_id in seen_identity_ids:
            raise ValueError(f"duplicate identity {identity_id}")
        seen_identity_ids.add(identity_id)

        tracklet_ids = tuple(
            str(value) for value in identity_row.get("tracklet_ids", ())
        )
        if not tracklet_ids:
            raise ValueError(f"identity {identity_id} has no tracklet_ids")

        ordered = _ordered_identity_observations(
            tracklet_ids=tracklet_ids,
            tracklets_by_id=tracklets_by_id,
        )
        if not ordered:
            raise ValueError(f"identity {identity_id} has no observations")

        raw_segments = _split_contiguous_segments(
            ordered,
            config=selected,
        )
        segments = tuple(
            _make_segment(
                segment_index=index,
                rows=rows,
                config=selected,
            )
            for index, rows in enumerate(raw_segments, start=1)
        )
        internal_gaps = _make_internal_gaps(segments)

        confidences = [item.confidence for _, item in ordered]
        unique_frames = {item.frame_index for _, item in ordered}

        result.append(
            IdentityTrajectoryProfile(
                identity_id=identity_id,
                start=segments[0].start,
                end=segments[-1].end,
                segments=segments,
                internal_gaps=internal_gaps,
                tracklet_ids=tuple(sorted(set(tracklet_ids))),
                observation_count=len(ordered),
                unique_frame_count=len(unique_frames),
                duration_s=float(
                    segments[-1].end.timestamp_s
                    - segments[0].start.timestamp_s
                ),
                observed_span_s=float(
                    sum(segment.duration_s for segment in segments)
                ),
                mean_observation_confidence=float(
                    sum(confidences) / len(confidences)
                ),
            )
        )

    return tuple(sorted(result, key=lambda row: row.identity_id))
