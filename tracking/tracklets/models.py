"""Tracklet contracts for the clean Tracking V1 pipeline.

A tracklet is a short, conservative temporal fragment. It is deliberately not
an identity: one physical player may produce several tracklets and global
reconstruction is responsible for joining them later.

The contract is intentionally self-contained enough for downstream stages to
reason about motion without reopening the detections artifact for every
operation. Visual/source details remain owned by CanonicalDetection and can be
resolved through ``detection_id`` when needed.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")


def _finite_float(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return parsed


def _non_empty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _json_safe_dict(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values") from exc
    return result


@dataclass(slots=True)
class TrackletObservation:
    """One canonical detection as used by a short temporal tracklet.

    ``detection_id`` links back to the detections-stage artifact. Geometry is
    repeated intentionally: tracklet construction and later identity solving
    should be able to inspect the field-space trajectory without repeatedly
    joining the full detection stream.
    """

    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float

    def __post_init__(self) -> None:
        self.detection_id = _non_empty_string("detection_id", self.detection_id)
        _require_non_negative_int("frame_index", self.frame_index)

        self.timestamp_s = _finite_float("timestamp_s", self.timestamp_s)
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be >= 0")

        self.x = _finite_float("x", self.x)
        self.z = _finite_float("z", self.z)

        self.confidence = _finite_float("confidence", self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "x": self.x,
            "z": self.z,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackletObservation":
        return cls(
            detection_id=payload["detection_id"],
            frame_index=payload["frame_index"],
            timestamp_s=payload["timestamp_s"],
            x=payload["x"],
            z=payload["z"],
            confidence=payload["confidence"],
        )


@dataclass(slots=True)
class Tracklet:
    """A conservative, temporally ordered fragment of one possible person.

    This object has no player identity, team, shirt number or ReID embedding.
    Those belong to later stages.

    Contract invariants:
    * observations are non-empty;
    * frame indices and timestamps are strictly increasing;
    * detection ids are unique;
    * gaps are allowed;
    * no assumption is made that different tracklets belong to different
      physical people.
    """

    tracklet_id: str
    observations: tuple[TrackletObservation, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tracklet_id = _non_empty_string("tracklet_id", self.tracklet_id)

        if not isinstance(self.observations, Sequence) or isinstance(
            self.observations, (str, bytes)
        ):
            raise ValueError("observations must be a sequence")
        if not self.observations:
            raise ValueError("a tracklet must contain at least one observation")

        normalized: list[TrackletObservation] = []
        seen_detection_ids: set[str] = set()
        previous_frame: int | None = None
        previous_timestamp: float | None = None

        for raw in self.observations:
            observation = (
                raw
                if isinstance(raw, TrackletObservation)
                else TrackletObservation.from_dict(raw)
            )

            if observation.detection_id in seen_detection_ids:
                raise ValueError(
                    f"duplicate detection_id in tracklet: {observation.detection_id!r}"
                )
            seen_detection_ids.add(observation.detection_id)

            if previous_frame is not None and observation.frame_index <= previous_frame:
                raise ValueError("tracklet frame indices must be strictly increasing")
            if (
                previous_timestamp is not None
                and observation.timestamp_s <= previous_timestamp
            ):
                raise ValueError("tracklet timestamps must be strictly increasing")

            normalized.append(observation)
            previous_frame = observation.frame_index
            previous_timestamp = observation.timestamp_s

        self.observations = tuple(normalized)
        self.metadata = _json_safe_dict("metadata", self.metadata)

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def start_timestamp_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_timestamp_s(self) -> float:
        return self.observations[-1].timestamp_s

    @property
    def duration_s(self) -> float:
        return self.end_timestamp_s - self.start_timestamp_s

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def mean_confidence(self) -> float:
        return sum(item.confidence for item in self.observations) / len(
            self.observations
        )

    @property
    def displacement_m(self) -> float:
        first = self.observations[0]
        last = self.observations[-1]
        return math.hypot(last.x - first.x, last.z - first.z)

    @property
    def path_length_m(self) -> float:
        total = 0.0
        for first, second in zip(self.observations, self.observations[1:]):
            total += math.hypot(second.x - first.x, second.z - first.z)
        return total

    @property
    def max_frame_gap(self) -> int:
        if len(self.observations) < 2:
            return 0
        return max(
            second.frame_index - first.frame_index - 1
            for first, second in zip(self.observations, self.observations[1:])
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "observations": [item.to_dict() for item in self.observations],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Tracklet":
        raw_observations = payload.get("observations")
        if not isinstance(raw_observations, (list, tuple)):
            raise ValueError("observations must be a list or tuple")
        return cls(
            tracklet_id=payload["tracklet_id"],
            observations=tuple(
                item
                if isinstance(item, TrackletObservation)
                else TrackletObservation.from_dict(item)
                for item in raw_observations
            ),
            metadata=payload.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "Tracklet":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("Tracklet JSON must encode an object")
        return cls.from_dict(payload)


@dataclass(slots=True)
class TrackletSet:
    """All short tracklets produced for one detections artifact."""

    tracklets: tuple[Tracklet, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tracklets, Sequence) or isinstance(
            self.tracklets, (str, bytes)
        ):
            raise ValueError("tracklets must be a sequence")

        normalized: list[Tracklet] = []
        ids: set[str] = set()
        detection_owners: dict[str, str] = {}

        for raw in self.tracklets:
            tracklet = raw if isinstance(raw, Tracklet) else Tracklet.from_dict(raw)

            if tracklet.tracklet_id in ids:
                raise ValueError(f"duplicate tracklet_id: {tracklet.tracklet_id!r}")
            ids.add(tracklet.tracklet_id)

            for observation in tracklet.observations:
                previous_owner = detection_owners.get(observation.detection_id)
                if previous_owner is not None:
                    raise ValueError(
                        f"detection {observation.detection_id!r} belongs to both "
                        f"{previous_owner!r} and {tracklet.tracklet_id!r}"
                    )
                detection_owners[observation.detection_id] = tracklet.tracklet_id

            normalized.append(tracklet)

        self.tracklets = tuple(normalized)
        self.metadata = _json_safe_dict("metadata", self.metadata)

    @property
    def tracklet_count(self) -> int:
        return len(self.tracklets)

    @property
    def observation_count(self) -> int:
        return sum(tracklet.observation_count for tracklet in self.tracklets)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklets": [tracklet.to_dict() for tracklet in self.tracklets],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackletSet":
        raw_tracklets = payload.get("tracklets", ())
        if not isinstance(raw_tracklets, (list, tuple)):
            raise ValueError("tracklets must be a list or tuple")
        return cls(
            tracklets=tuple(
                item if isinstance(item, Tracklet) else Tracklet.from_dict(item)
                for item in raw_tracklets
            ),
            metadata=payload.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "TrackletSet":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("TrackletSet JSON must encode an object")
        return cls.from_dict(payload)
