"""Conservative gap reconstruction for Tracking V1 physical trajectories.

Hito 4C takes the one-point-per-identity/frame output of 4B and fills only
short, physically supported holes. It never smooths observed data and never
bridges long disappearances.

Policy
------
Same-tracklet gaps:
- endpoint duration <= 0.35 s;
- endpoint speed <= 12 m/s;
- neither endpoint is a fusion conflict;
- gap respects the inferred sampling cadence.

Cross-tracklet gaps:
- endpoint duration <= 0.75 s;
- endpoint speed <= 12 m/s;
- direct Hito 3A `transition_reachable` support from a left endpoint tracklet
  to a right endpoint tracklet;
- the 3A transition's required speed remains <= 12 m/s;
- neither endpoint is a fusion conflict;
- gap respects the inferred sampling cadence.

Accepted gaps are linearly interpolated at the clip's actual sampling cadence.
Long or unsupported gaps are explicitly preserved and recorded.
"""

from __future__ import annotations

import collections
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .models import (
    FusionMode,
    IdentityTrajectory,
    TrajectoryPoint,
    TrajectoryPointOrigin,
)


GAP_RECONSTRUCTION_VERSION = "safe-gap-v1"


@dataclass(frozen=True, slots=True)
class GapReconstructionConfig:
    max_same_tracklet_gap_s: float = 0.35
    max_cross_tracklet_gap_s: float = 0.75
    max_endpoint_speed_mps: float = 12.0
    timestamp_tolerance_s: float = 0.02
    physics_gap_tolerance_s: float = 0.05

    def __post_init__(self) -> None:
        if self.max_same_tracklet_gap_s <= 0:
            raise ValueError(
                "max_same_tracklet_gap_s must be > 0"
            )
        if self.max_cross_tracklet_gap_s <= 0:
            raise ValueError(
                "max_cross_tracklet_gap_s must be > 0"
            )
        if self.max_endpoint_speed_mps <= 0:
            raise ValueError(
                "max_endpoint_speed_mps must be > 0"
            )
        if self.timestamp_tolerance_s < 0:
            raise ValueError(
                "timestamp_tolerance_s must be >= 0"
            )
        if self.physics_gap_tolerance_s < 0:
            raise ValueError(
                "physics_gap_tolerance_s must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class SamplingCadence:
    frame_step: int
    time_step_s: float

    def __post_init__(self) -> None:
        if self.frame_step <= 0:
            raise ValueError("frame_step must be > 0")
        if self.time_step_s <= 0:
            raise ValueError("time_step_s must be > 0")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source}:{line_number} is not an object"
                )
            yield payload


def load_trajectories(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[IdentityTrajectory, ...]:
    trajectories = tuple(
        IdentityTrajectory.from_dict(row)
        for row in rows
    )
    if not trajectories:
        raise ValueError("no trajectories supplied")
    seen: set[str] = set()
    for trajectory in trajectories:
        if trajectory.identity_id in seen:
            raise ValueError(
                f"duplicate trajectory: {trajectory.identity_id}"
            )
        seen.add(trajectory.identity_id)
    return trajectories


def load_candidate_edges(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        a = str(row["tracklet_a"])
        b = str(row["tracklet_b"])
        if a == b:
            raise ValueError("candidate edge cannot self-link")
        pair = (a, b) if a < b else (b, a)
        if pair in result:
            raise ValueError(
                f"duplicate candidate edge: {pair}"
            )
        result[pair] = dict(row)
    return result


def infer_sampling_cadence(
    trajectories: Sequence[IdentityTrajectory],
) -> SamplingCadence:
    frame_deltas: list[int] = []
    paired_time_deltas: list[tuple[int, float]] = []

    for trajectory in trajectories:
        for left, right in zip(
            trajectory.points,
            trajectory.points[1:],
        ):
            frame_delta = (
                right.frame_index - left.frame_index
            )
            time_delta = (
                right.timestamp_s - left.timestamp_s
            )
            if frame_delta <= 0 or time_delta <= 0:
                raise ValueError(
                    "trajectory cadence requires increasing points"
                )
            frame_deltas.append(frame_delta)
            paired_time_deltas.append(
                (frame_delta, time_delta)
            )

    if not frame_deltas:
        raise ValueError(
            "cannot infer cadence from one-point trajectories only"
        )

    counts = collections.Counter(frame_deltas)
    highest_count = max(counts.values())
    frame_step = min(
        delta
        for delta, count in counts.items()
        if count == highest_count
    )

    time_steps = [
        time_delta
        for frame_delta, time_delta
        in paired_time_deltas
        if frame_delta == frame_step
    ]
    if not time_steps:
        raise ValueError(
            "could not infer nominal timestamp step"
        )

    return SamplingCadence(
        frame_step=frame_step,
        time_step_s=float(
            statistics.median(time_steps)
        ),
    )


def _distance(
    left: TrajectoryPoint,
    right: TrajectoryPoint,
) -> float:
    return math.hypot(
        right.x - left.x,
        right.z - left.z,
    )


def _endpoint_speed(
    left: TrajectoryPoint,
    right: TrajectoryPoint,
) -> float:
    duration = right.timestamp_s - left.timestamp_s
    if duration <= 0:
        raise ValueError(
            "gap endpoints must increase in time"
        )
    return _distance(left, right) / duration


def _has_fusion_conflict(
    point: TrajectoryPoint,
) -> bool:
    return bool(
        point.metadata.get("fusion_conflict", False)
    )


def _candidate_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _direct_transition_support(
    *,
    left: TrajectoryPoint,
    right: TrajectoryPoint,
    candidate_edges: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
    config: GapReconstructionConfig,
) -> dict[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    left_tracklets = set(left.source_tracklet_ids)
    right_tracklets = set(right.source_tracklet_ids)
    gap_s = right.timestamp_s - left.timestamp_s

    for left_tracklet in sorted(left_tracklets):
        for right_tracklet in sorted(right_tracklets):
            if left_tracklet == right_tracklet:
                continue
            edge = candidate_edges.get(
                _candidate_pair(
                    left_tracklet,
                    right_tracklet,
                )
            )
            if edge is None:
                continue
            if edge.get("status") != "candidate":
                continue
            if edge.get("reason") != "transition_reachable":
                continue

            physics = edge.get("physics")
            if not isinstance(physics, Mapping):
                continue
            if str(physics.get("predecessor")) != left_tracklet:
                continue
            if str(physics.get("successor")) != right_tracklet:
                continue

            physics_gap = physics.get("gap_s")
            if physics_gap is None:
                continue
            try:
                physics_gap = float(physics_gap)
            except (TypeError, ValueError):
                continue
            if (
                abs(physics_gap - gap_s)
                > config.physics_gap_tolerance_s
            ):
                continue

            required_speed = physics.get(
                "required_speed_after_slack_mps"
            )
            if required_speed is None:
                required_speed = 0.0
            required_speed = float(required_speed)
            if (
                not math.isfinite(required_speed)
                or required_speed
                > config.max_endpoint_speed_mps
            ):
                continue

            candidates.append(
                {
                    "tracklet_a": str(
                        edge["tracklet_a"]
                    ),
                    "tracklet_b": str(
                        edge["tracklet_b"]
                    ),
                    "reason": edge["reason"],
                    "predecessor": left_tracklet,
                    "successor": right_tracklet,
                    "gap_s": physics_gap,
                    "endpoint_distance_m": (
                        physics.get(
                            "endpoint_distance_m"
                        )
                    ),
                    "required_speed_after_slack_mps": (
                        required_speed
                    ),
                    "prediction_error_mean_m": (
                        physics.get(
                            "prediction_error_mean_m"
                        )
                    ),
                    "velocity_direction_cosine": (
                        physics.get(
                            "velocity_direction_cosine"
                        )
                    ),
                }
            )

    if not candidates:
        return None

    def _key(row: Mapping[str, Any]) -> tuple[float, float, str, str]:
        prediction_error = row.get(
            "prediction_error_mean_m"
        )
        if prediction_error is None:
            prediction_error = math.inf
        return (
            float(
                row[
                    "required_speed_after_slack_mps"
                ]
            ),
            float(prediction_error),
            str(row["predecessor"]),
            str(row["successor"]),
        )

    return min(candidates, key=_key)


def _classify_gap(
    *,
    left: TrajectoryPoint,
    right: TrajectoryPoint,
    cadence: SamplingCadence,
    candidate_edges: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
    config: GapReconstructionConfig,
) -> dict[str, Any]:
    frame_delta = (
        right.frame_index - left.frame_index
    )
    duration_s = (
        right.timestamp_s - left.timestamp_s
    )
    distance_m = _distance(left, right)
    endpoint_speed_mps = (
        distance_m / duration_s
    )

    shared_tracklets = sorted(
        set(left.source_tracklet_ids)
        & set(right.source_tracklet_ids)
    )
    gap_kind = (
        "same_tracklet"
        if shared_tracklets
        else "cross_tracklet"
    )

    event = {
        "gap_kind": gap_kind,
        "left_frame_index": left.frame_index,
        "right_frame_index": right.frame_index,
        "left_timestamp_s": left.timestamp_s,
        "right_timestamp_s": right.timestamp_s,
        "frame_delta": frame_delta,
        "duration_s": duration_s,
        "distance_m": distance_m,
        "endpoint_speed_mps": (
            endpoint_speed_mps
        ),
        "left_source_tracklet_ids": list(
            left.source_tracklet_ids
        ),
        "right_source_tracklet_ids": list(
            right.source_tracklet_ids
        ),
        "shared_source_tracklet_ids": (
            shared_tracklets
        ),
        "accepted": False,
        "decision_reason": None,
        "transition_support": None,
        "missing_point_count": 0,
    }

    if frame_delta <= cadence.frame_step:
        event["decision_reason"] = "not_a_gap"
        return event

    if frame_delta % cadence.frame_step != 0:
        event["decision_reason"] = "cadence_mismatch"
        return event

    interval_count = (
        frame_delta // cadence.frame_step
    )
    missing_point_count = interval_count - 1
    event["missing_point_count"] = (
        missing_point_count
    )

    expected_duration = (
        interval_count * cadence.time_step_s
    )
    if (
        abs(duration_s - expected_duration)
        > config.timestamp_tolerance_s
    ):
        event["decision_reason"] = (
            "timestamp_cadence_mismatch"
        )
        return event

    if (
        _has_fusion_conflict(left)
        or _has_fusion_conflict(right)
    ):
        event["decision_reason"] = (
            "fusion_conflict_endpoint"
        )
        return event

    if (
        endpoint_speed_mps
        > config.max_endpoint_speed_mps
    ):
        event["decision_reason"] = (
            "endpoint_speed_too_high"
        )
        return event

    if gap_kind == "same_tracklet":
        if (
            duration_s
            > config.max_same_tracklet_gap_s
        ):
            event["decision_reason"] = (
                "same_tracklet_gap_too_long"
            )
            return event
        event["accepted"] = True
        event["decision_reason"] = (
            "safe_same_tracklet_gap"
        )
        return event

    if (
        duration_s
        > config.max_cross_tracklet_gap_s
    ):
        event["decision_reason"] = (
            "cross_tracklet_gap_too_long"
        )
        return event

    support = _direct_transition_support(
        left=left,
        right=right,
        candidate_edges=candidate_edges,
        config=config,
    )
    if support is None:
        event["decision_reason"] = (
            "no_direct_transition_support"
        )
        return event

    event["transition_support"] = support
    event["accepted"] = True
    event["decision_reason"] = (
        "safe_cross_tracklet_transition"
    )
    return event


def _endpoint_provenance(
    left: TrajectoryPoint,
    right: TrajectoryPoint,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:
    tracklets = tuple(
        sorted(
            set(left.source_tracklet_ids)
            | set(right.source_tracklet_ids)
        )
    )
    detections = tuple(
        sorted(
            set(left.source_detection_ids)
            | set(right.source_detection_ids)
        )
    )
    return tracklets, detections


def _interpolate_gap(
    *,
    identity_id: str,
    left: TrajectoryPoint,
    right: TrajectoryPoint,
    cadence: SamplingCadence,
    event: Mapping[str, Any],
    gap_id: str,
) -> tuple[TrajectoryPoint, ...]:
    if not event["accepted"]:
        return ()

    frame_delta = (
        right.frame_index - left.frame_index
    )
    interval_count = (
        frame_delta // cadence.frame_step
    )
    tracklets, detections = _endpoint_provenance(
        left, right
    )
    confidence = min(
        left.confidence,
        right.confidence,
    )
    spread = max(
        left.position_spread_m,
        right.position_spread_m,
    )

    result = []
    for interval_index in range(
        1, interval_count
    ):
        alpha = (
            interval_index / interval_count
        )
        frame_index = (
            left.frame_index
            + interval_index
            * cadence.frame_step
        )
        timestamp_s = (
            left.timestamp_s
            + alpha
            * (
                right.timestamp_s
                - left.timestamp_s
            )
        )
        x = left.x + alpha * (right.x - left.x)
        z = left.z + alpha * (right.z - left.z)

        result.append(
            TrajectoryPoint(
                identity_id=identity_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                x=x,
                z=z,
                confidence=confidence,
                origin=(
                    TrajectoryPointOrigin.INTERPOLATED
                ),
                fusion_mode=(
                    FusionMode.NOT_APPLICABLE
                ),
                source_tracklet_ids=tracklets,
                source_detection_ids=detections,
                source_count=len(detections),
                position_spread_m=spread,
                metadata={
                    "interpolation_version": (
                        GAP_RECONSTRUCTION_VERSION
                    ),
                    "interpolation_method": "linear",
                    "interpolation_alpha": alpha,
                    "gap_id": gap_id,
                    "gap_kind": event["gap_kind"],
                    "gap_duration_s": (
                        event["duration_s"]
                    ),
                    "gap_distance_m": (
                        event["distance_m"]
                    ),
                    "endpoint_speed_mps": (
                        event[
                            "endpoint_speed_mps"
                        ]
                    ),
                    "left_frame_index": (
                        left.frame_index
                    ),
                    "right_frame_index": (
                        right.frame_index
                    ),
                    "left_timestamp_s": (
                        left.timestamp_s
                    ),
                    "right_timestamp_s": (
                        right.timestamp_s
                    ),
                    "left_origin": (
                        left.origin.value
                    ),
                    "right_origin": (
                        right.origin.value
                    ),
                    "confidence_policy": (
                        "min_endpoint"
                    ),
                    "transition_support": (
                        event[
                            "transition_support"
                        ]
                    ),
                },
            )
        )

    return tuple(result)


def reconstruct_safe_gaps(
    *,
    trajectories: Sequence[
        IdentityTrajectory
    ],
    candidate_edges: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
    config: GapReconstructionConfig | None = None,
    cadence: SamplingCadence | None = None,
) -> tuple[
    tuple[IdentityTrajectory, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    if config is None:
        config = GapReconstructionConfig()
    if cadence is None:
        cadence = infer_sampling_cadence(
            trajectories
        )

    rebuilt: list[IdentityTrajectory] = []
    events: list[dict[str, Any]] = []
    input_point_count = 0
    output_point_count = 0
    interpolated_point_count = 0

    for trajectory in sorted(
        trajectories,
        key=lambda item: item.identity_id,
    ):
        new_points: list[TrajectoryPoint] = []
        identity_events = []
        points = trajectory.points
        input_point_count += len(points)

        if len(points) == 1:
            new_points.extend(points)
        else:
            new_points.append(points[0])
            for gap_index, (left, right) in enumerate(
                zip(points, points[1:]),
                start=1,
            ):
                if (
                    right.frame_index
                    - left.frame_index
                    > cadence.frame_step
                ):
                    event = _classify_gap(
                        left=left,
                        right=right,
                        cadence=cadence,
                        candidate_edges=(
                            candidate_edges
                        ),
                        config=config,
                    )
                    gap_id = (
                        f"{trajectory.identity_id}"
                        f":gap-{gap_index:03d}"
                    )
                    event = {
                        "schema_version": 1,
                        "reconstruction_version": (
                            GAP_RECONSTRUCTION_VERSION
                        ),
                        "gap_id": gap_id,
                        "identity_id": (
                            trajectory.identity_id
                        ),
                        **event,
                    }
                    inserted = _interpolate_gap(
                        identity_id=(
                            trajectory.identity_id
                        ),
                        left=left,
                        right=right,
                        cadence=cadence,
                        event=event,
                        gap_id=gap_id,
                    )
                    event[
                        "interpolated_point_count"
                    ] = len(inserted)
                    events.append(event)
                    identity_events.append(event)
                    new_points.extend(inserted)
                    interpolated_point_count += (
                        len(inserted)
                    )
                new_points.append(right)

        filled = sum(
            event["accepted"]
            for event in identity_events
        )
        preserved = (
            len(identity_events) - filled
        )

        metadata = dict(
            trajectory.metadata
        )
        metadata.update(
            {
                "builder": (
                    GAP_RECONSTRUCTION_VERSION
                ),
                "source_trajectory_version": (
                    trajectory.trajectory_version
                ),
                "gaps_filled": True,
                "smoothed": False,
                "gap_event_count": len(
                    identity_events
                ),
                "filled_gap_count": filled,
                "preserved_gap_count": (
                    preserved
                ),
                "interpolated_point_count": sum(
                    event[
                        "interpolated_point_count"
                    ]
                    for event in identity_events
                ),
            }
        )

        rebuilt_trajectory = (
            IdentityTrajectory(
                identity_id=(
                    trajectory.identity_id
                ),
                tracklet_ids=(
                    trajectory.tracklet_ids
                ),
                points=tuple(new_points),
                trajectory_version=(
                    GAP_RECONSTRUCTION_VERSION
                ),
                metadata=metadata,
            )
        )
        rebuilt.append(rebuilt_trajectory)
        output_point_count += (
            rebuilt_trajectory.point_count
        )

    filled_events = [
        event
        for event in events
        if event["accepted"]
    ]
    preserved_events = [
        event
        for event in events
        if not event["accepted"]
    ]

    reason_counts = collections.Counter(
        event["decision_reason"]
        for event in events
    )
    kind_counts = collections.Counter(
        event["gap_kind"]
        for event in events
    )
    filled_kind_counts = collections.Counter(
        event["gap_kind"]
        for event in filled_events
    )

    summary = {
        "schema_version": 1,
        "reconstruction_version": (
            GAP_RECONSTRUCTION_VERSION
        ),
        "provisional": True,
        "final_stage_completed": False,
        "trajectory_count": len(rebuilt),
        "input_point_count": (
            input_point_count
        ),
        "output_point_count": (
            output_point_count
        ),
        "interpolated_point_count": (
            interpolated_point_count
        ),
        "gap_event_count": len(events),
        "filled_gap_count": len(
            filled_events
        ),
        "preserved_gap_count": len(
            preserved_events
        ),
        "gap_kind_counts": dict(
            sorted(kind_counts.items())
        ),
        "filled_gap_kind_counts": dict(
            sorted(
                filled_kind_counts.items()
            )
        ),
        "decision_reason_counts": dict(
            sorted(reason_counts.items())
        ),
        "max_filled_gap_duration_s": (
            max(
                (
                    event["duration_s"]
                    for event in filled_events
                ),
                default=0.0,
            )
        ),
        "max_preserved_gap_duration_s": (
            max(
                (
                    event["duration_s"]
                    for event in preserved_events
                ),
                default=0.0,
            )
        ),
        "max_filled_endpoint_speed_mps": (
            max(
                (
                    event[
                        "endpoint_speed_mps"
                    ]
                    for event in filled_events
                ),
                default=0.0,
            )
        ),
        "sampling_cadence": {
            "frame_step": cadence.frame_step,
            "time_step_s": cadence.time_step_s,
        },
        "config": {
            "max_same_tracklet_gap_s": (
                config.max_same_tracklet_gap_s
            ),
            "max_cross_tracklet_gap_s": (
                config.max_cross_tracklet_gap_s
            ),
            "max_endpoint_speed_mps": (
                config.max_endpoint_speed_mps
            ),
            "timestamp_tolerance_s": (
                config.timestamp_tolerance_s
            ),
            "physics_gap_tolerance_s": (
                config.physics_gap_tolerance_s
            ),
        },
    }

    return (
        tuple(rebuilt),
        tuple(events),
        summary,
    )
