"""Same-frame observation fusion for canonical Tracking V1 identities.

Hito 4B collapses multiple tracklet observations that belong to the same
physical identity and frame into exactly one trajectory point.

This stage NEVER fills temporal gaps and NEVER smooths trajectories.

Policy
------
1. One source observation -> copy it verbatim (`origin=observed`).
2. Multiple sources with physically compact support -> confidence-weighted
   field-space mean (`origin=fused`, `fusion_mode=weighted_mean`).
3. Multiple sources whose pairwise diameter exceeds the conservative fusion
   gate -> do not average across the disagreement. Choose the weighted medoid
   and mark the point as `fusion_conflict` for later diagnostics.

The conflict fallback guarantees one point per identity/frame without silently
inventing a midpoint between spatially inconsistent observations.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .models import (
    OBSERVATION_FUSION_VERSION,
    FusionMode,
    IdentityTrajectory,
    TrajectoryPoint,
    TrajectoryPointOrigin,
)


@dataclass(frozen=True, slots=True)
class ObservationFusionConfig:
    max_fusion_diameter_m: float = 1.5
    timestamp_tolerance_s: float = 1e-6
    minimum_weight: float = 1e-3

    def __post_init__(self) -> None:
        if self.max_fusion_diameter_m <= 0:
            raise ValueError(
                "max_fusion_diameter_m must be > 0"
            )
        if self.timestamp_tolerance_s < 0:
            raise ValueError(
                "timestamp_tolerance_s must be >= 0"
            )
        if not 0 < self.minimum_weight <= 1:
            raise ValueError(
                "minimum_weight must be in (0,1]"
            )


@dataclass(frozen=True, slots=True)
class SourceObservation:
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float


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


def _distance(
    a: SourceObservation,
    b: SourceObservation,
) -> float:
    return math.hypot(a.x - b.x, a.z - b.z)


def _pairwise_diameter(
    observations: Sequence[SourceObservation],
) -> float:
    diameter = 0.0
    for index, left in enumerate(observations):
        for right in observations[index + 1 :]:
            diameter = max(
                diameter,
                _distance(left, right),
            )
    return diameter


def _weighted_medoid(
    observations: Sequence[SourceObservation],
    config: ObservationFusionConfig,
) -> SourceObservation:
    def cost(candidate: SourceObservation) -> tuple[float, float, str]:
        total = 0.0
        for other in observations:
            weight = max(
                other.confidence,
                config.minimum_weight,
            )
            total += weight * _distance(
                candidate, other
            )
        # Prefer higher-confidence source on geometric ties.
        return (
            total,
            -candidate.confidence,
            candidate.detection_id,
        )

    return min(observations, key=cost)


def _weighted_mean(
    observations: Sequence[SourceObservation],
    config: ObservationFusionConfig,
) -> tuple[float, float, float]:
    weights = [
        max(obs.confidence, config.minimum_weight)
        for obs in observations
    ]
    denominator = sum(weights)
    x = sum(
        weight * obs.x
        for weight, obs in zip(weights, observations)
    ) / denominator
    z = sum(
        weight * obs.z
        for weight, obs in zip(weights, observations)
    ) / denominator

    # Weighted source confidence; deliberately does not inflate simply because
    # two detectors saw the same person.
    confidence = sum(
        weight * obs.confidence
        for weight, obs in zip(weights, observations)
    ) / denominator
    return x, z, confidence


def _spread_from_point(
    observations: Sequence[SourceObservation],
    x: float,
    z: float,
) -> float:
    return max(
        math.hypot(obs.x - x, obs.z - z)
        for obs in observations
    )


def _fuse_frame(
    *,
    identity_id: str,
    observations: Sequence[SourceObservation],
    config: ObservationFusionConfig,
) -> tuple[TrajectoryPoint, dict[str, Any] | None]:
    ordered = tuple(
        sorted(
            observations,
            key=lambda obs: (
                obs.tracklet_id,
                obs.detection_id,
            ),
        )
    )
    if not ordered:
        raise ValueError("cannot fuse an empty frame")

    frame_index = ordered[0].frame_index
    timestamp_s = ordered[0].timestamp_s
    for obs in ordered:
        if obs.frame_index != frame_index:
            raise ValueError(
                "same-frame fusion received different frame indices"
            )
        if (
            abs(obs.timestamp_s - timestamp_s)
            > config.timestamp_tolerance_s
        ):
            raise ValueError(
                f"frame {frame_index} has inconsistent timestamps"
            )

    source_tracklets = tuple(
        sorted({obs.tracklet_id for obs in ordered})
    )
    source_detections = tuple(
        obs.detection_id for obs in ordered
    )
    if len(set(source_detections)) != len(source_detections):
        raise ValueError(
            f"frame {frame_index} repeats a detection source"
        )

    if len(ordered) == 1:
        obs = ordered[0]
        return (
            TrajectoryPoint(
                identity_id=identity_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                x=obs.x,
                z=obs.z,
                confidence=obs.confidence,
                origin=TrajectoryPointOrigin.OBSERVED,
                fusion_mode=FusionMode.SINGLE,
                source_tracklet_ids=source_tracklets,
                source_detection_ids=source_detections,
                source_count=1,
                position_spread_m=0.0,
                metadata={
                    "fusion_conflict": False,
                },
            ),
            None,
        )

    diameter = _pairwise_diameter(ordered)
    medoid = _weighted_medoid(ordered, config)

    if diameter <= config.max_fusion_diameter_m:
        x, z, confidence = _weighted_mean(
            ordered, config
        )
        spread = _spread_from_point(
            ordered, x, z
        )
        mode = FusionMode.WEIGHTED_MEAN
        fusion_conflict = False
        selected_detection_id = None
    else:
        x = medoid.x
        z = medoid.z
        confidence = medoid.confidence
        spread = _spread_from_point(
            ordered, x, z
        )
        mode = FusionMode.MEDOID_SELECTED
        fusion_conflict = True
        selected_detection_id = medoid.detection_id

    point = TrajectoryPoint(
        identity_id=identity_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        x=x,
        z=z,
        confidence=confidence,
        origin=TrajectoryPointOrigin.FUSED,
        fusion_mode=mode,
        source_tracklet_ids=source_tracklets,
        source_detection_ids=source_detections,
        source_count=len(ordered),
        position_spread_m=spread,
        metadata={
            "fusion_conflict": fusion_conflict,
            "input_pairwise_diameter_m": diameter,
            "selected_detection_id": (
                selected_detection_id
            ),
        },
    )

    event = {
        "schema_version": 1,
        "fusion_version": OBSERVATION_FUSION_VERSION,
        "identity_id": identity_id,
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "source_count": len(ordered),
        "source_tracklet_ids": list(source_tracklets),
        "source_detection_ids": list(source_detections),
        "input_positions": [
            {
                "tracklet_id": obs.tracklet_id,
                "detection_id": obs.detection_id,
                "x": obs.x,
                "z": obs.z,
                "confidence": obs.confidence,
            }
            for obs in ordered
        ],
        "pairwise_diameter_m": diameter,
        "fusion_mode": mode.value,
        "fusion_conflict": fusion_conflict,
        "selected_detection_id": (
            selected_detection_id
        ),
        "output": {
            "x": x,
            "z": z,
            "confidence": confidence,
            "position_spread_m": spread,
        },
    }
    return point, event


def _load_tracklets(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, tuple[SourceObservation, ...]]:
    result: dict[str, tuple[SourceObservation, ...]] = {}
    global_detection_owner: dict[str, str] = {}

    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id in result:
            raise ValueError(
                f"duplicate tracklet: {tracklet_id}"
            )

        raw_observations = row.get("observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise ValueError(
                f"{tracklet_id} has no observations"
            )

        observations = []
        previous_frame = None
        previous_time = None
        for raw in raw_observations:
            detection_id = str(raw["detection_id"])
            previous_owner = global_detection_owner.get(
                detection_id
            )
            if previous_owner is not None:
                raise ValueError(
                    f"detection {detection_id} belongs to "
                    f"{previous_owner} and {tracklet_id}"
                )
            global_detection_owner[
                detection_id
            ] = tracklet_id

            frame_index = int(raw["frame_index"])
            timestamp_s = float(raw["timestamp_s"])
            if (
                previous_frame is not None
                and frame_index <= previous_frame
            ):
                raise ValueError(
                    f"{tracklet_id} frames are not strictly increasing"
                )
            if (
                previous_time is not None
                and timestamp_s <= previous_time
            ):
                raise ValueError(
                    f"{tracklet_id} timestamps are not strictly increasing"
                )
            previous_frame = frame_index
            previous_time = timestamp_s

            observations.append(
                SourceObservation(
                    tracklet_id=tracklet_id,
                    detection_id=detection_id,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    x=float(raw["x"]),
                    z=float(raw["z"]),
                    confidence=float(raw["confidence"]),
                )
            )

        result[tracklet_id] = tuple(observations)

    return result


def build_observation_fused_trajectories(
    *,
    identity_rows: Iterable[Mapping[str, Any]],
    tracklet_rows: Iterable[Mapping[str, Any]],
    config: ObservationFusionConfig | None = None,
) -> tuple[
    tuple[IdentityTrajectory, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    if config is None:
        config = ObservationFusionConfig()

    tracklets = _load_tracklets(tracklet_rows)
    identity_rows = tuple(identity_rows)

    assigned: set[str] = set()
    trajectories = []
    fusion_events = []

    input_observation_count = 0
    output_point_count = 0

    for identity_row in sorted(
        identity_rows,
        key=lambda row: str(row["identity_id"]),
    ):
        identity_id = str(identity_row["identity_id"])
        if identity_row.get("canonical") is not True:
            raise ValueError(
                f"{identity_id} is not a canonical published identity"
            )

        tracklet_ids = tuple(
            sorted(
                str(value)
                for value in identity_row["tracklet_ids"]
            )
        )
        if not tracklet_ids:
            raise ValueError(
                f"{identity_id} has no tracklets"
            )

        by_frame: dict[
            int, list[SourceObservation]
        ] = {}
        identity_input_count = 0

        for tracklet_id in tracklet_ids:
            if tracklet_id in assigned:
                raise ValueError(
                    f"{tracklet_id} belongs to multiple identities"
                )
            assigned.add(tracklet_id)

            observations = tracklets.get(tracklet_id)
            if observations is None:
                raise ValueError(
                    f"{identity_id} references missing {tracklet_id}"
                )
            identity_input_count += len(observations)
            for observation in observations:
                by_frame.setdefault(
                    observation.frame_index, []
                ).append(observation)

        points = []
        identity_fusion_events = 0
        identity_conflicts = 0
        for frame_index in sorted(by_frame):
            point, event = _fuse_frame(
                identity_id=identity_id,
                observations=by_frame[frame_index],
                config=config,
            )
            points.append(point)
            if event is not None:
                fusion_events.append(event)
                identity_fusion_events += 1
                identity_conflicts += int(
                    event["fusion_conflict"]
                )

        trajectories.append(
            IdentityTrajectory(
                identity_id=identity_id,
                tracklet_ids=tracklet_ids,
                points=tuple(points),
                metadata={
                    "builder": (
                        OBSERVATION_FUSION_VERSION
                    ),
                    "input_observation_count": (
                        identity_input_count
                    ),
                    "same_frame_fusion_count": (
                        identity_fusion_events
                    ),
                    "fusion_conflict_count": (
                        identity_conflicts
                    ),
                    "gaps_filled": False,
                    "smoothed": False,
                },
            )
        )

        input_observation_count += identity_input_count
        output_point_count += len(points)

    if assigned != set(tracklets):
        unassigned = sorted(set(tracklets) - assigned)
        extra = sorted(assigned - set(tracklets))
        raise ValueError(
            "canonical identities do not cover tracklets exactly; "
            f"unassigned={unassigned[:10]} extra={extra[:10]}"
        )

    event_diameters = [
        float(event["pairwise_diameter_m"])
        for event in fusion_events
    ]
    summary = {
        "schema_version": 1,
        "fusion_version": OBSERVATION_FUSION_VERSION,
        "provisional": True,
        "final_stage_completed": False,
        "trajectory_count": len(trajectories),
        "tracklet_count": len(tracklets),
        "input_observation_count": (
            input_observation_count
        ),
        "output_point_count": output_point_count,
        "collapsed_observation_count": (
            input_observation_count
            - output_point_count
        ),
        "same_frame_fusion_event_count": len(
            fusion_events
        ),
        "weighted_mean_fusion_count": sum(
            event["fusion_mode"]
            == FusionMode.WEIGHTED_MEAN.value
            for event in fusion_events
        ),
        "medoid_selected_conflict_count": sum(
            event["fusion_mode"]
            == FusionMode.MEDOID_SELECTED.value
            for event in fusion_events
        ),
        "max_fusion_input_diameter_m": (
            max(event_diameters)
            if event_diameters
            else 0.0
        ),
        "config": {
            "max_fusion_diameter_m": (
                config.max_fusion_diameter_m
            ),
            "timestamp_tolerance_s": (
                config.timestamp_tolerance_s
            ),
            "minimum_weight": (
                config.minimum_weight
            ),
        },
    }
    return (
        tuple(trajectories),
        tuple(fusion_events),
        summary,
    )
