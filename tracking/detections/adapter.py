"""Adapter from same-frame person association to canonical V1 detections.

This module is deliberately independent from ``tracking_v0`` and does not
import the experimental association implementation at runtime.  It consumes a
small structural interface (objects exposing the documented attributes) and
publishes stable :class:`CanonicalDetection` objects.

One canonical detection represents one physical-person hypothesis in one
frame, *after* same-frame duplicate association across virtual views and
*before* any temporal tracking.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Protocol, Sequence

from .models import CanonicalDetection, CanonicalDetectionFrame


class _SourceLensEvidenceLike(Protocol):
    lens: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    anchor_x: float
    anchor_y: float
    sample_count: int

    def to_dict(self) -> Mapping[str, Any]: ...


class _ObservationLike(Protocol):
    observation_id: int
    frame_number: int
    time_s: float
    view_name: str
    band: str
    xyxy: Sequence[float]
    confidence: float
    owner_cell: str | None
    ownership_role: str
    source_lenses: Sequence[_SourceLensEvidenceLike]


class _PersonMeasurementLike(Protocol):
    measurement_id: int
    observation_ids: Sequence[int]
    primary_observation_id: int
    representative_world_x: float | None
    representative_world_z: float | None
    fused_world_x: float | None
    fused_world_z: float | None
    position_spread_m: float | None
    confidence: float
    owner_supported: bool
    multi_view: bool
    creation_source: str


class _AssociationResultLike(Protocol):
    observations: Sequence[_ObservationLike]
    measurements: Sequence[_PersonMeasurementLike]


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _measurement_position(
    measurement: _PersonMeasurementLike,
) -> tuple[float, float] | None:
    """Prefer fused X/Z and fall back to the representative observation."""
    fused_x = _finite_or_none(measurement.fused_world_x)
    fused_z = _finite_or_none(measurement.fused_world_z)
    if fused_x is not None and fused_z is not None:
        return fused_x, fused_z

    representative_x = _finite_or_none(measurement.representative_world_x)
    representative_z = _finite_or_none(measurement.representative_world_z)
    if representative_x is not None and representative_z is not None:
        return representative_x, representative_z

    return None


def _source_lens_payload(
    primary: _ObservationLike | None,
) -> tuple[
    str | None,
    tuple[float, float, float, float] | None,
    tuple[float, float] | None,
]:
    """Expose one unambiguous original-lens footprint in the top-level contract.

    A virtual observation can occasionally sample both physical Veo lenses.
    ``CanonicalDetection`` only has one compact top-level source-lens slot, so
    we populate it only when exactly one valid left/right lens contributed.
    Full source-lens provenance is always preserved in ``metadata``.
    """
    if primary is None:
        return None, None, None

    valid = [
        item
        for item in primary.source_lenses
        if getattr(item, "lens", None) in ("left", "right")
    ]
    if len(valid) != 1:
        return None, None, None

    source = valid[0]
    return (
        source.lens,
        (
            float(source.x_min),
            float(source.y_min),
            float(source.x_max),
            float(source.y_max),
        ),
        (float(source.anchor_x), float(source.anchor_y)),
    )


def _source_lenses_metadata(
    observation: _ObservationLike,
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in observation.source_lenses:
        to_dict = getattr(item, "to_dict", None)
        if callable(to_dict):
            payload = dict(to_dict())
        else:
            payload = {
                "lens": item.lens,
                "x_min": float(item.x_min),
                "y_min": float(item.y_min),
                "x_max": float(item.x_max),
                "y_max": float(item.y_max),
                "anchor_x": float(item.anchor_x),
                "anchor_y": float(item.anchor_y),
                "sample_count": int(item.sample_count),
            }
        result.append(payload)
    return result


def canonical_detection_from_measurement(
    measurement: _PersonMeasurementLike,
    *,
    frame_index: int,
    timestamp_s: float,
    observations_by_id: Mapping[int, _ObservationLike],
) -> CanonicalDetection | None:
    """Convert one same-frame physical-person hypothesis to V1.

    Returns ``None`` only when neither fused nor representative world position
    is available. Broken provenance (for example a missing primary observation)
    is considered an error instead of being silently hidden.
    """
    position = _measurement_position(measurement)
    if position is None:
        return None

    primary = observations_by_id.get(measurement.primary_observation_id)
    if primary is None:
        raise ValueError(
            "primary observation "
            f"{measurement.primary_observation_id!r} is missing"
        )

    contributing: list[_ObservationLike] = []
    seen_observation_ids: set[int] = set()
    for raw_id in measurement.observation_ids:
        observation_id = int(raw_id)
        if observation_id in seen_observation_ids:
            raise ValueError(
                f"duplicate observation_id in measurement: {observation_id}"
            )
        seen_observation_ids.add(observation_id)

        observation = observations_by_id.get(observation_id)
        if observation is None:
            raise ValueError(
                f"contributing observation {observation_id!r} is missing"
            )
        if observation.frame_number != frame_index:
            raise ValueError(
                f"observation {observation_id!r} belongs to frame "
                f"{observation.frame_number}, expected {frame_index}"
            )
        if not math.isclose(
            float(observation.time_s),
            float(timestamp_s),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"observation {observation_id!r} timestamp "
                f"{observation.time_s} does not match {timestamp_s}"
            )
        contributing.append(observation)

    if measurement.primary_observation_id not in seen_observation_ids:
        raise ValueError("primary observation is not part of observation_ids")

    source_lens, source_bbox, source_footpoint = _source_lens_payload(primary)

    contributing_views: list[str] = []
    seen_views: set[str] = set()
    for observation in contributing:
        view_name = str(observation.view_name).strip()
        if not view_name:
            raise ValueError("observation view_name must be non-empty")
        if view_name not in seen_views:
            seen_views.add(view_name)
            contributing_views.append(view_name)

    metadata: dict[str, Any] = {
        "same_frame_measurement_id": int(measurement.measurement_id),
        "observation_ids": [int(item) for item in measurement.observation_ids],
        "primary_observation_id": int(measurement.primary_observation_id),
        "owner_supported": bool(measurement.owner_supported),
        "multi_view": bool(measurement.multi_view),
        "creation_source": str(measurement.creation_source),
        "position_spread_m": (
            None
            if measurement.position_spread_m is None
            else float(measurement.position_spread_m)
        ),
        "representative_band": str(primary.band),
        "representative_owner_cell": primary.owner_cell,
        "representative_ownership_role": str(primary.ownership_role),
        "source_pixels": [
            {
                "observation_id": int(observation.observation_id),
                "view_name": str(observation.view_name),
                "lenses": _source_lenses_metadata(observation),
            }
            for observation in contributing
        ],
    }

    return CanonicalDetection(
        detection_id=(
            f"det-{frame_index:08d}-{int(measurement.measurement_id):04d}"
        ),
        frame_index=frame_index,
        timestamp_s=float(timestamp_s),
        confidence=float(measurement.confidence),
        x=position[0],
        z=position[1],
        source_lens=source_lens,
        source_bbox_xyxy=source_bbox,
        source_footpoint_xy=source_footpoint,
        representative_view=str(primary.view_name),
        virtual_bbox_xyxy=tuple(float(value) for value in primary.xyxy),
        contributing_views=tuple(contributing_views),
        metadata=metadata,
    )


def canonical_frame_from_association(
    association: _AssociationResultLike,
    *,
    frame_index: int,
    timestamp_s: float,
) -> CanonicalDetectionFrame:
    """Convert a complete same-frame association result to the V1 contract."""
    observations_by_id: dict[int, _ObservationLike] = {}
    for observation in association.observations:
        observation_id = int(observation.observation_id)
        if observation_id in observations_by_id:
            raise ValueError(
                f"duplicate observation_id in association: {observation_id}"
            )
        observations_by_id[observation_id] = observation

    detections: list[CanonicalDetection] = []
    seen_measurement_ids: set[int] = set()

    for measurement in sorted(
        association.measurements,
        key=lambda item: int(item.measurement_id),
    ):
        measurement_id = int(measurement.measurement_id)
        if measurement_id in seen_measurement_ids:
            raise ValueError(
                f"duplicate measurement_id in association: {measurement_id}"
            )
        seen_measurement_ids.add(measurement_id)

        detection = canonical_detection_from_measurement(
            measurement,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            observations_by_id=observations_by_id,
        )
        if detection is not None:
            detections.append(detection)

    return CanonicalDetectionFrame(
        frame_index=frame_index,
        timestamp_s=float(timestamp_s),
        detections=tuple(detections),
    )


def canonical_frames_from_associations(
    rows: Iterable[
        tuple[int, float, _AssociationResultLike]
    ],
) -> Iterable[CanonicalDetectionFrame]:
    """Lazily convert ``(frame_index, timestamp_s, association)`` rows."""
    previous_frame: int | None = None
    previous_timestamp: float | None = None

    for frame_index, timestamp_s, association in rows:
        if previous_frame is not None and frame_index <= previous_frame:
            raise ValueError("association rows must use increasing frame_index")
        if (
            previous_timestamp is not None
            and float(timestamp_s) <= previous_timestamp
        ):
            raise ValueError("association rows must use increasing timestamp_s")

        frame = canonical_frame_from_association(
            association,
            frame_index=frame_index,
            timestamp_s=timestamp_s,
        )
        previous_frame = frame.frame_index
        previous_timestamp = frame.timestamp_s
        yield frame
