"""Adapter from same-frame association output to Tracking V1 input.

This module deliberately contains no temporal association or tracker policy.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from experiments.virtual_camera_coverage.association import (
    Observation,
    PersonMeasurement,
)
from tracking_v0 import TrackingMeasurement


def to_tracking_measurement(
    person: PersonMeasurement,
    *,
    frame_index: int,
    timestamp: float,
    observations_by_id: Mapping[int, Observation],
) -> TrackingMeasurement | None:
    """Convert one associated person to a tracker input, or skip no-position data.

    Association IDs are scoped to one source frame, so the tracker-facing ID is
    made unique by pairing it with the actual source-frame index.
    """
    x = person.fused_world_x
    z = person.fused_world_z
    if x is None or z is None:
        x = person.representative_world_x
        z = person.representative_world_z

    if x is None or z is None:
        return None

    primary = observations_by_id.get(person.primary_observation_id)
    contributing = [
        observations_by_id[observation_id]
        for observation_id in person.observation_ids
        if observation_id in observations_by_id
    ]

    metadata: dict[str, Any] = {
        "same_frame_measurement_id": person.measurement_id,
        "observation_ids": list(person.observation_ids),
        "primary_observation_id": person.primary_observation_id,
        "owner_supported": person.owner_supported,
        "multi_view": person.multi_view,
        "creation_source": person.creation_source,
        "position_spread_m": person.position_spread_m,
        "representative_camera": primary.view_name if primary else None,
        "representative_band": primary.band if primary else None,
        "representative_bbox_xyxy": list(primary.xyxy) if primary else None,
        "representative_detection_confidence": (
            primary.confidence if primary else None
        ),
        "representative_owner_cell": primary.owner_cell if primary else None,
        "representative_ownership_role": (
            primary.ownership_role if primary else None
        ),
        "source_pixels": [
            {
                "observation_id": observation.observation_id,
                "view_name": observation.view_name,
                "lenses": [item.to_dict() for item in observation.source_lenses],
            }
            for observation in contributing
        ],
    }

    return TrackingMeasurement(
        measurement_id=f"{frame_index}:{person.measurement_id}",
        frame_index=frame_index,
        timestamp=timestamp,
        x=x,
        z=z,
        confidence=person.confidence,
        position_covariance=None,
        metadata=metadata,
    )
