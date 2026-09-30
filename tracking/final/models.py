"""Stable contracts for Tracking V1 final physical trajectories.

Field coordinates (X, Z) remain the primary state.

Point provenance:
- observed: one real canonical observation;
- fused: multiple real same-frame observations fused by 4B;
- interpolated: a synthetic point inserted by 4C between two validated
  observed/fused endpoints.

`fusion_mode=not_applicable` is mandatory for interpolated points so synthetic
trajectory reconstruction is never confused with same-frame observation
fusion.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence


FINAL_TRAJECTORY_SCHEMA_VERSION = 1
OBSERVATION_FUSION_VERSION = "observation-fusion-v1"


class TrajectoryPointOrigin(str, Enum):
    OBSERVED = "observed"
    FUSED = "fused"
    INTERPOLATED = "interpolated"


class FusionMode(str, Enum):
    SINGLE = "single"
    WEIGHTED_MEAN = "weighted_mean"
    MEDOID_SELECTED = "medoid_selected"
    NOT_APPLICABLE = "not_applicable"


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_empty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(
    name: str,
    values: Sequence[Any],
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(
        _non_empty_string(name, value)
        for value in values
    )
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _json_safe_metadata(value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "metadata must be JSON-serializable"
        ) from exc
    return result


@dataclass(frozen=True, slots=True)
class TrajectoryPoint:
    identity_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float
    origin: TrajectoryPointOrigin
    fusion_mode: FusionMode
    source_tracklet_ids: tuple[str, ...]
    source_detection_ids: tuple[str, ...]
    source_count: int
    position_spread_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "identity_id",
            _non_empty_string("identity_id", self.identity_id),
        )
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError(
                "frame_index must be an integer >= 0"
            )

        timestamp_s = _finite_float(
            "timestamp_s", self.timestamp_s
        )
        if timestamp_s < 0:
            raise ValueError("timestamp_s must be >= 0")
        object.__setattr__(self, "timestamp_s", timestamp_s)

        object.__setattr__(self, "x", _finite_float("x", self.x))
        object.__setattr__(self, "z", _finite_float("z", self.z))

        confidence = _finite_float(
            "confidence", self.confidence
        )
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        object.__setattr__(self, "confidence", confidence)

        if not isinstance(self.origin, TrajectoryPointOrigin):
            try:
                object.__setattr__(
                    self,
                    "origin",
                    TrajectoryPointOrigin(self.origin),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid trajectory origin: {self.origin!r}"
                ) from exc

        if not isinstance(self.fusion_mode, FusionMode):
            try:
                object.__setattr__(
                    self,
                    "fusion_mode",
                    FusionMode(self.fusion_mode),
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"invalid fusion_mode: {self.fusion_mode!r}"
                ) from exc

        tracklets = _string_tuple(
            "source_tracklet_ids",
            self.source_tracklet_ids,
        )
        detections = _string_tuple(
            "source_detection_ids",
            self.source_detection_ids,
        )
        object.__setattr__(
            self, "source_tracklet_ids", tracklets
        )
        object.__setattr__(
            self, "source_detection_ids", detections
        )

        if (
            isinstance(self.source_count, bool)
            or not isinstance(self.source_count, int)
            or self.source_count < 1
        ):
            raise ValueError(
                "source_count must be an integer >= 1"
            )
        if self.source_count != len(detections):
            raise ValueError(
                "source_count must match source_detection_ids"
            )

        spread = _finite_float(
            "position_spread_m",
            self.position_spread_m,
        )
        if spread < 0:
            raise ValueError(
                "position_spread_m must be >= 0"
            )
        object.__setattr__(
            self, "position_spread_m", spread
        )

        if self.origin is TrajectoryPointOrigin.OBSERVED:
            if self.source_count != 1:
                raise ValueError(
                    "observed points require exactly one source"
                )
            if self.fusion_mode is not FusionMode.SINGLE:
                raise ValueError(
                    "observed points require fusion_mode=single"
                )

        elif self.origin is TrajectoryPointOrigin.FUSED:
            if self.source_count < 2:
                raise ValueError(
                    "fused points require at least two sources"
                )
            if self.fusion_mode not in {
                FusionMode.WEIGHTED_MEAN,
                FusionMode.MEDOID_SELECTED,
            }:
                raise ValueError(
                    "fused points require a fusion mode"
                )

        elif self.origin is TrajectoryPointOrigin.INTERPOLATED:
            if self.fusion_mode is not FusionMode.NOT_APPLICABLE:
                raise ValueError(
                    "interpolated points require "
                    "fusion_mode=not_applicable"
                )
            if self.source_count < 2:
                raise ValueError(
                    "interpolated points require endpoint provenance"
                )

        object.__setattr__(
            self,
            "metadata",
            _json_safe_metadata(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                FINAL_TRAJECTORY_SCHEMA_VERSION
            ),
            "identity_id": self.identity_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "x": self.x,
            "z": self.z,
            "confidence": self.confidence,
            "origin": self.origin.value,
            "fusion_mode": self.fusion_mode.value,
            "source_tracklet_ids": list(
                self.source_tracklet_ids
            ),
            "source_detection_ids": list(
                self.source_detection_ids
            ),
            "source_count": self.source_count,
            "position_spread_m": self.position_spread_m,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "TrajectoryPoint":
        return cls(
            identity_id=payload["identity_id"],
            frame_index=payload["frame_index"],
            timestamp_s=payload["timestamp_s"],
            x=payload["x"],
            z=payload["z"],
            confidence=payload["confidence"],
            origin=payload["origin"],
            fusion_mode=payload["fusion_mode"],
            source_tracklet_ids=tuple(
                payload["source_tracklet_ids"]
            ),
            source_detection_ids=tuple(
                payload["source_detection_ids"]
            ),
            source_count=payload["source_count"],
            position_spread_m=payload[
                "position_spread_m"
            ],
            metadata=payload.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class IdentityTrajectory:
    identity_id: str
    tracklet_ids: tuple[str, ...]
    points: tuple[TrajectoryPoint, ...]
    trajectory_version: str = OBSERVATION_FUSION_VERSION
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identity_id = _non_empty_string(
            "identity_id", self.identity_id
        )
        object.__setattr__(self, "identity_id", identity_id)

        tracklet_ids = _string_tuple(
            "tracklet_ids", self.tracklet_ids
        )
        object.__setattr__(
            self, "tracklet_ids", tracklet_ids
        )

        if isinstance(self.points, (str, bytes)) or not isinstance(
            self.points, Sequence
        ):
            raise ValueError("points must be a sequence")
        if not self.points:
            raise ValueError(
                "an identity trajectory must contain points"
            )

        normalized = tuple(
            point
            if isinstance(point, TrajectoryPoint)
            else TrajectoryPoint.from_dict(point)
            for point in self.points
        )

        previous_frame = None
        previous_time = None
        seen_frames: set[int] = set()
        for point in normalized:
            if point.identity_id != identity_id:
                raise ValueError(
                    "trajectory point identity_id mismatch"
                )
            if point.frame_index in seen_frames:
                raise ValueError(
                    "trajectory contains multiple points "
                    f"for frame {point.frame_index}"
                )
            if (
                previous_frame is not None
                and point.frame_index <= previous_frame
            ):
                raise ValueError(
                    "trajectory frames must be strictly increasing"
                )
            if (
                previous_time is not None
                and point.timestamp_s <= previous_time
            ):
                raise ValueError(
                    "trajectory timestamps must be strictly increasing"
                )
            seen_frames.add(point.frame_index)
            previous_frame = point.frame_index
            previous_time = point.timestamp_s

        object.__setattr__(self, "points", normalized)
        object.__setattr__(
            self,
            "trajectory_version",
            _non_empty_string(
                "trajectory_version",
                self.trajectory_version,
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            _json_safe_metadata(self.metadata),
        )

    @property
    def point_count(self) -> int:
        return len(self.points)

    @property
    def observed_point_count(self) -> int:
        return sum(
            point.origin is TrajectoryPointOrigin.OBSERVED
            for point in self.points
        )

    @property
    def fused_point_count(self) -> int:
        return sum(
            point.origin is TrajectoryPointOrigin.FUSED
            for point in self.points
        )

    @property
    def interpolated_point_count(self) -> int:
        return sum(
            point.origin is TrajectoryPointOrigin.INTERPOLATED
            for point in self.points
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": (
                FINAL_TRAJECTORY_SCHEMA_VERSION
            ),
            "trajectory_version": self.trajectory_version,
            "identity_id": self.identity_id,
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": len(self.tracklet_ids),
            "point_count": self.point_count,
            "observed_point_count": (
                self.observed_point_count
            ),
            "fused_point_count": self.fused_point_count,
            "interpolated_point_count": (
                self.interpolated_point_count
            ),
            "points": [
                point.to_dict()
                for point in self.points
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "IdentityTrajectory":
        return cls(
            identity_id=payload["identity_id"],
            tracklet_ids=tuple(
                payload["tracklet_ids"]
            ),
            points=tuple(
                TrajectoryPoint.from_dict(item)
                for item in payload["points"]
            ),
            trajectory_version=payload.get(
                "trajectory_version",
                OBSERVATION_FUSION_VERSION,
            ),
            metadata=payload.get("metadata", {}),
        )
