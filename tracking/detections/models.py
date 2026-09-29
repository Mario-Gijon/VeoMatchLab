"""Detection contracts for the clean Tracking V1 pipeline.

A :class:`CanonicalDetection` represents one physical person observed at one
instant.  It intentionally contains no temporal identity such as ``track_id``
or ``player_id``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Sequence, TypeAlias


BBoxXYXY: TypeAlias = tuple[float, float, float, float]
PointXY: TypeAlias = tuple[float, float]
SourceLens: TypeAlias = Literal["left", "right"]


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


def _normalize_bbox(name: str, value: Sequence[float] | None) -> BBoxXYXY | None:
    if value is None:
        return None
    if len(value) != 4:
        raise ValueError(f"{name} must contain exactly 4 values")
    x1, y1, x2, y2 = (_finite_float(f"{name}[{index}]", item) for index, item in enumerate(value))
    if x2 < x1 or y2 < y1:
        raise ValueError(f"{name} must use xyxy ordering with x2>=x1 and y2>=y1")
    return (x1, y1, x2, y2)


def _normalize_point(name: str, value: Sequence[float] | None) -> PointXY | None:
    if value is None:
        return None
    if len(value) != 2:
        raise ValueError(f"{name} must contain exactly 2 values")
    x, y = (_finite_float(f"{name}[{index}]", item) for index, item in enumerate(value))
    return (x, y)


def _normalize_views(value: Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in value:
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError("contributing_views entries must be non-empty strings")
        view = raw.strip()
        if view not in seen:
            normalized.append(view)
            seen.add(view)
    return tuple(normalized)


def _json_safe_dict(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values") from exc
    return result


@dataclass(slots=True)
class CanonicalDetection:
    """One physical person detected in one frame, before temporal tracking."""

    detection_id: str
    frame_index: int
    timestamp_s: float
    confidence: float
    x: float
    z: float

    source_lens: SourceLens | None = None
    source_bbox_xyxy: BBoxXYXY | None = None
    source_footpoint_xy: PointXY | None = None
    representative_view: str | None = None
    virtual_bbox_xyxy: BBoxXYXY | None = None
    contributing_views: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.detection_id, str) or not self.detection_id.strip():
            raise ValueError("detection_id must be a non-empty string")
        self.detection_id = self.detection_id.strip()

        _require_non_negative_int("frame_index", self.frame_index)
        self.timestamp_s = _finite_float("timestamp_s", self.timestamp_s)
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be >= 0")

        self.confidence = _finite_float("confidence", self.confidence)
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        self.x = _finite_float("x", self.x)
        self.z = _finite_float("z", self.z)

        if self.source_lens not in (None, "left", "right"):
            raise ValueError("source_lens must be 'left', 'right', or None")

        self.source_bbox_xyxy = _normalize_bbox("source_bbox_xyxy", self.source_bbox_xyxy)
        self.source_footpoint_xy = _normalize_point("source_footpoint_xy", self.source_footpoint_xy)
        self.virtual_bbox_xyxy = _normalize_bbox("virtual_bbox_xyxy", self.virtual_bbox_xyxy)

        if self.representative_view is not None:
            if not isinstance(self.representative_view, str) or not self.representative_view.strip():
                raise ValueError("representative_view must be a non-empty string or None")
            self.representative_view = self.representative_view.strip()

        self.contributing_views = _normalize_views(self.contributing_views)
        self.metadata = _json_safe_dict("metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "confidence": self.confidence,
            "x": self.x,
            "z": self.z,
            "source_lens": self.source_lens,
            "source_bbox_xyxy": list(self.source_bbox_xyxy) if self.source_bbox_xyxy is not None else None,
            "source_footpoint_xy": list(self.source_footpoint_xy) if self.source_footpoint_xy is not None else None,
            "representative_view": self.representative_view,
            "virtual_bbox_xyxy": list(self.virtual_bbox_xyxy) if self.virtual_bbox_xyxy is not None else None,
            "contributing_views": list(self.contributing_views),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalDetection":
        return cls(
            detection_id=payload["detection_id"],
            frame_index=payload["frame_index"],
            timestamp_s=payload["timestamp_s"],
            confidence=payload["confidence"],
            x=payload["x"],
            z=payload["z"],
            source_lens=payload.get("source_lens"),
            source_bbox_xyxy=payload.get("source_bbox_xyxy"),
            source_footpoint_xy=payload.get("source_footpoint_xy"),
            representative_view=payload.get("representative_view"),
            virtual_bbox_xyxy=payload.get("virtual_bbox_xyxy"),
            contributing_views=payload.get("contributing_views", ()),
            metadata=payload.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "CanonicalDetection":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("CanonicalDetection JSON must encode an object")
        return cls.from_dict(payload)


@dataclass(slots=True)
class CanonicalDetectionFrame:
    """All canonical physical-person detections for one video frame."""

    frame_index: int
    timestamp_s: float
    detections: tuple[CanonicalDetection, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative_int("frame_index", self.frame_index)
        self.timestamp_s = _finite_float("timestamp_s", self.timestamp_s)
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be >= 0")

        normalized: list[CanonicalDetection] = []
        ids: set[str] = set()

        for item in self.detections:
            detection = item if isinstance(item, CanonicalDetection) else CanonicalDetection.from_dict(item)
            if detection.frame_index != self.frame_index:
                raise ValueError(
                    f"detection {detection.detection_id!r} belongs to frame "
                    f"{detection.frame_index}, expected {self.frame_index}"
                )
            if not math.isclose(detection.timestamp_s, self.timestamp_s, rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(
                    f"detection {detection.detection_id!r} timestamp "
                    f"{detection.timestamp_s} does not match frame timestamp {self.timestamp_s}"
                )
            if detection.detection_id in ids:
                raise ValueError(f"duplicate detection_id in frame: {detection.detection_id!r}")
            ids.add(detection.detection_id)
            normalized.append(detection)

        self.detections = tuple(normalized)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "detections": [detection.to_dict() for detection in self.detections],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CanonicalDetectionFrame":
        raw_detections = payload.get("detections", ())
        if not isinstance(raw_detections, (list, tuple)):
            raise ValueError("detections must be a list or tuple")
        return cls(
            frame_index=payload["frame_index"],
            timestamp_s=payload["timestamp_s"],
            detections=tuple(
                item if isinstance(item, CanonicalDetection) else CanonicalDetection.from_dict(item)
                for item in raw_detections
            ),
        )

    def to_json(self) -> str:
        """Serialize as one compact JSON object, suitable for one JSONL line."""
        return json.dumps(self.to_dict(), ensure_ascii=False, allow_nan=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, raw: str) -> "CanonicalDetectionFrame":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("CanonicalDetectionFrame JSON must encode an object")
        return cls.from_dict(payload)
