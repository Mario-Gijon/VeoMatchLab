"""Veo-specific observation quality for metric field tracking.

V1 models two effects:

1. Real local geometric sensitivity: how much X/Z changes when the detector
   footpoint moves by one panorama pixel. This comes directly from the
   validated pixel->ground projector.

2. Panorama seam prior: Veo blends the two physical lenses. The exact blend
   value at one output pixel is reconstructed from the same presentation
   geometry/constants used by ``geometry.reproduce_veo_panorama``.

Detector confidence and bbox size are recorded but intentionally do not modify
covariance in V1. Their relationship to actual X/Z jitter should be calibrated
empirically before introducing extra heuristic weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from geometry.reproduce_veo_panorama import (
    BLEND_BUBBLE_RADIUS,
    BLEND_DISTANCE,
    PROJECTION_BUBBLE_RADIUS,
)
from tracking.uncertainty import (
    ProjectionSensitivity,
    covariance_summary,
    metric_covariance,
    projection_sensitivity,
)


@dataclass(frozen=True)
class VeoQualityConfig:
    """Transparent, tunable assumptions for observation covariance."""

    derivative_step_px: float = 1.0
    sigma_u_px: float = 1.0
    sigma_v_px: float = 1.0

    # seam_score is 0 outside the blend and 1 at a 50/50 blend.
    # seam_strength=1 means variance is at most doubled at the seam center.
    seam_strength: float = 1.0

    # Small isotropic floor to avoid numerically overconfident measurements.
    minimum_metric_std_m: float = 0.01

    def __post_init__(self) -> None:
        if self.derivative_step_px <= 0.0:
            raise ValueError("derivative_step_px must be > 0")
        if self.sigma_u_px <= 0.0 or self.sigma_v_px <= 0.0:
            raise ValueError("pixel standard deviations must be > 0")
        if self.seam_strength < 0.0:
            raise ValueError("seam_strength must be >= 0")
        if self.minimum_metric_std_m < 0.0:
            raise ValueError("minimum_metric_std_m must be >= 0")


@dataclass(frozen=True)
class ObservationQuality:
    """Per-observation uncertainty components consumed by the future tracker."""

    confidence: float
    bbox_width: float
    bbox_height: float

    sensitivity: ProjectionSensitivity

    blend_value: float
    seam_score: float
    seam_variance_multiplier: float

    geometric_covariance: np.ndarray
    measurement_covariance: np.ndarray

    std_x_m: float
    std_z_m: float
    correlation_xz: float
    ellipse_std_minor_m: float
    ellipse_std_major_m: float

    def to_dict(self) -> dict:
        return {
            "confidence": self.confidence,
            "bbox_width": self.bbox_width,
            "bbox_height": self.bbox_height,
            "sensitivity": self.sensitivity.to_dict(),
            "blend_value": self.blend_value,
            "seam_score": self.seam_score,
            "seam_variance_multiplier": self.seam_variance_multiplier,
            "geometric_covariance": self.geometric_covariance.tolist(),
            "measurement_covariance": self.measurement_covariance.tolist(),
            "std_x_m": self.std_x_m,
            "std_z_m": self.std_z_m,
            "correlation_xz": self.correlation_xz,
            "ellipse_std_minor_m": self.ellipse_std_minor_m,
            "ellipse_std_major_m": self.ellipse_std_major_m,
        }


def panorama_blend_value(
    metadata: Any,
    *,
    width: int,
    height: int,
    pixel_x: float,
    pixel_y: float,
) -> float:
    """Reproduce Veo's panorama blend value for one output pixel.

    This mirrors the blend portion of
    ``geometry.reproduce_veo_panorama.build_output_world_geometry`` without
    allocating a full panorama-sized map.
    """

    width = int(width)
    height = int(height)
    pixel_x = float(pixel_x)
    pixel_y = float(pixel_y)

    if width <= 0 or height <= 0:
        raise ValueError("panorama dimensions must be positive")
    if not 0.0 <= pixel_x < width or not 0.0 <= pixel_y < height:
        raise ValueError("pixel is outside panorama bounds")

    va = metadata.va_presentation

    u = (pixel_x + 0.5) / width
    v = 1.0 - (pixel_y + 0.5) / height

    lon = u * (float(va["right"]) - float(va["left"])) + float(va["left"])
    lat = v * (float(va["top"]) - float(va["bottom"])) + float(va["bottom"])

    cos_lat = math.cos(lat)
    direction_presentation = np.asarray(
        [
            math.sin(lon) * cos_lat,
            -math.sin(lat),
            math.cos(lon) * cos_lat,
            0.0,
        ],
        dtype=np.float64,
    )

    camera_to_world_presentation = (
        metadata.camera2world @ np.linalg.inv(metadata.camera2presentation)
    )
    direction_world_h = camera_to_world_presentation @ direction_presentation
    direction_world = direction_world_h[:3]

    camera_world = (
        metadata.camera2world
        @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )

    dy = float(direction_world[1])
    up_down = (float(np.sign(dy)) + 1.0) / 2.0
    safe_dy = dy if abs(dy) > 1e-12 else 1e-12

    ground_distance = -float(camera_world[1]) / safe_dy
    distance = (
        up_down * ground_distance
        + (1.0 - up_down) * PROJECTION_BUBBLE_RADIUS
    )

    projection_direction = direction_world * distance
    projection_direction_h = np.asarray(
        [
            projection_direction[0],
            projection_direction[1],
            projection_direction[2],
            0.0,
        ],
        dtype=np.float64,
    )

    world_to_presentation = np.linalg.inv(camera_to_world_presentation)
    blend_target = world_to_presentation @ projection_direction_h
    blend_target = blend_target.copy()
    blend_target[3] = 1.0

    projection_length = float(np.linalg.norm(projection_direction_h))
    normalize_target = (
        projection_length > BLEND_BUBBLE_RADIUS
        or float(projection_direction[1]) < 0.0
    )

    if normalize_target:
        length = max(float(np.linalg.norm(blend_target)), 1e-12)
        blend_target = blend_target / length * BLEND_BUBBLE_RADIUS

    return float(
        np.clip(
            float(blend_target[0]) / BLEND_DISTANCE + 0.5,
            0.0,
            1.0,
        )
    )


def seam_score_from_blend(blend_value: float) -> float:
    """Smooth 0..1 seam score; peak at an equal left/right blend."""

    blend = float(blend_value)
    if not 0.0 <= blend <= 1.0:
        raise ValueError("blend_value must be in [0, 1]")

    # 4*b*(1-b): 0 at pure-left/pure-right, 1 at b=0.5.
    return float(4.0 * blend * (1.0 - blend))


class VeoObservationQualityModel:
    """Compute observation covariance using actual Veo geometry."""

    def __init__(
        self,
        projector: Any,
        *,
        config: VeoQualityConfig | None = None,
    ) -> None:
        self.projector = projector
        self.config = config or VeoQualityConfig()

        if not hasattr(projector, "metadata"):
            raise TypeError("projector must expose Veo metadata")
        if not hasattr(projector, "width") or not hasattr(projector, "height"):
            raise TypeError("projector must expose width and height")

    def evaluate(
        self,
        *,
        pixel_x: float,
        pixel_y: float,
        bbox_width: float,
        bbox_height: float,
        confidence: float,
    ) -> ObservationQuality:
        confidence = float(confidence)
        bbox_width = float(bbox_width)
        bbox_height = float(bbox_height)

        if not np.isfinite([confidence, bbox_width, bbox_height]).all():
            raise ValueError("quality inputs must be finite")
        if bbox_width <= 0.0 or bbox_height <= 0.0:
            raise ValueError("bbox dimensions must be > 0")

        sensitivity = projection_sensitivity(
            self.projector,
            pixel_x,
            pixel_y,
            step_px=self.config.derivative_step_px,
        )

        geometric = metric_covariance(
            sensitivity,
            sigma_u_px=self.config.sigma_u_px,
            sigma_v_px=self.config.sigma_v_px,
        )

        blend_value = panorama_blend_value(
            self.projector.metadata,
            width=int(self.projector.width),
            height=int(self.projector.height),
            pixel_x=pixel_x,
            pixel_y=pixel_y,
        )
        seam_score = seam_score_from_blend(blend_value)
        seam_multiplier = 1.0 + self.config.seam_strength * seam_score

        measurement = geometric * seam_multiplier

        floor_variance = self.config.minimum_metric_std_m ** 2
        if floor_variance > 0.0:
            measurement = measurement + np.eye(2, dtype=np.float64) * floor_variance

        measurement = 0.5 * (measurement + measurement.T)
        summary = covariance_summary(measurement)

        return ObservationQuality(
            confidence=confidence,
            bbox_width=bbox_width,
            bbox_height=bbox_height,
            sensitivity=sensitivity,
            blend_value=blend_value,
            seam_score=seam_score,
            seam_variance_multiplier=float(seam_multiplier),
            geometric_covariance=geometric,
            measurement_covariance=measurement,
            std_x_m=summary["std_x_m"],
            std_z_m=summary["std_z_m"],
            correlation_xz=summary["correlation_xz"],
            ellipse_std_minor_m=summary["ellipse_std_minor_m"],
            ellipse_std_major_m=summary["ellipse_std_major_m"],
        )


__all__ = [
    "ObservationQuality",
    "VeoObservationQualityModel",
    "VeoQualityConfig",
    "panorama_blend_value",
    "seam_score_from_blend",
]
