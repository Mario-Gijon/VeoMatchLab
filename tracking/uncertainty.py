"""Metric uncertainty induced by panorama pixel uncertainty.

The central idea is simple: the validated Veo projector already maps a
panorama pixel to a ground point (X, Z). We numerically differentiate that
mapping around the bbox footpoint.

The resulting Jacobian J has units of metres per pixel::

    [ dX/du  dX/dv ]
J = [               ]
    [ dZ/du  dZ/dv ]

Given a pixel covariance P_px, metric covariance is then::

    R_geo = J @ P_px @ J.T

This module is camera-agnostic. Veo-specific seam handling lives in
``tracking.veo_quality``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ProjectionSensitivity:
    """Local pixel->metre differential around one projected footpoint."""

    pixel_x: float
    pixel_y: float
    step_px: float
    jacobian: np.ndarray
    metres_per_pixel_u: float
    metres_per_pixel_v: float
    singular_values: np.ndarray
    anisotropy: float
    derivative_u_mode: str
    derivative_v_mode: str

    def to_dict(self) -> dict:
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "step_px": self.step_px,
            "jacobian": self.jacobian.tolist(),
            "metres_per_pixel_u": self.metres_per_pixel_u,
            "metres_per_pixel_v": self.metres_per_pixel_v,
            "singular_values": self.singular_values.tolist(),
            "anisotropy": self.anisotropy,
            "derivative_u_mode": self.derivative_u_mode,
            "derivative_v_mode": self.derivative_v_mode,
        }


def _ground_point(projector: Any, pixel_x: float, pixel_y: float) -> np.ndarray:
    result = projector.project_pixel(float(pixel_x), float(pixel_y))
    if result.world_x is None or result.world_z is None:
        raise ValueError(
            f"pixel ({pixel_x:.3f}, {pixel_y:.3f}) has no ground intersection"
        )

    point = np.asarray(
        [float(result.world_x), float(result.world_z)],
        dtype=np.float64,
    )
    if not np.isfinite(point).all():
        raise ValueError("projector returned a non-finite ground point")
    return point


def _axis_derivative(
    projector: Any,
    *,
    pixel_x: float,
    pixel_y: float,
    axis: int,
    step_px: float,
    center: np.ndarray,
) -> tuple[np.ndarray, str]:
    width = float(projector.width)
    height = float(projector.height)

    if axis == 0:
        minus_ok = pixel_x - step_px >= 0.0
        plus_ok = pixel_x + step_px < width

        def sample(delta: float) -> np.ndarray:
            return _ground_point(projector, pixel_x + delta, pixel_y)

    elif axis == 1:
        minus_ok = pixel_y - step_px >= 0.0
        plus_ok = pixel_y + step_px < height

        def sample(delta: float) -> np.ndarray:
            return _ground_point(projector, pixel_x, pixel_y + delta)

    else:
        raise ValueError("axis must be 0 (u) or 1 (v)")

    minus = None
    plus = None

    if minus_ok:
        try:
            minus = sample(-step_px)
        except ValueError:
            minus = None

    if plus_ok:
        try:
            plus = sample(step_px)
        except ValueError:
            plus = None

    if minus is not None and plus is not None:
        return (plus - minus) / (2.0 * step_px), "central"

    if plus is not None:
        return (plus - center) / step_px, "forward"

    if minus is not None:
        return (center - minus) / step_px, "backward"

    raise ValueError(
        "could not estimate local projection derivative on either side "
        f"of pixel ({pixel_x:.3f}, {pixel_y:.3f})"
    )


def projection_sensitivity(
    projector: Any,
    pixel_x: float,
    pixel_y: float,
    *,
    step_px: float = 1.0,
) -> ProjectionSensitivity:
    """Numerically estimate local metric sensitivity at one panorama pixel."""

    pixel_x = float(pixel_x)
    pixel_y = float(pixel_y)
    step_px = float(step_px)

    if step_px <= 0.0 or not math.isfinite(step_px):
        raise ValueError("step_px must be finite and > 0")
    if not np.isfinite([pixel_x, pixel_y]).all():
        raise ValueError("pixel coordinates must be finite")
    if not 0.0 <= pixel_x < float(projector.width):
        raise ValueError("pixel_x is outside projector bounds")
    if not 0.0 <= pixel_y < float(projector.height):
        raise ValueError("pixel_y is outside projector bounds")

    center = _ground_point(projector, pixel_x, pixel_y)

    du, mode_u = _axis_derivative(
        projector,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        axis=0,
        step_px=step_px,
        center=center,
    )
    dv, mode_v = _axis_derivative(
        projector,
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        axis=1,
        step_px=step_px,
        center=center,
    )

    jacobian = np.column_stack((du, dv))
    if jacobian.shape != (2, 2) or not np.isfinite(jacobian).all():
        raise ValueError("projection Jacobian is invalid")

    metres_per_pixel_u = float(np.linalg.norm(du))
    metres_per_pixel_v = float(np.linalg.norm(dv))

    singular_values = np.linalg.svd(jacobian, compute_uv=False).astype(np.float64)

    smallest = float(np.min(singular_values))
    largest = float(np.max(singular_values))
    anisotropy = largest / smallest if smallest > 1e-12 else float("inf")

    return ProjectionSensitivity(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        step_px=step_px,
        jacobian=jacobian,
        metres_per_pixel_u=metres_per_pixel_u,
        metres_per_pixel_v=metres_per_pixel_v,
        singular_values=singular_values,
        anisotropy=float(anisotropy),
        derivative_u_mode=mode_u,
        derivative_v_mode=mode_v,
    )


def metric_covariance(
    sensitivity: ProjectionSensitivity,
    *,
    sigma_u_px: float = 1.0,
    sigma_v_px: float = 1.0,
    correlation_uv: float = 0.0,
) -> np.ndarray:
    """Propagate a pixel covariance through the local projection Jacobian."""

    sigma_u_px = float(sigma_u_px)
    sigma_v_px = float(sigma_v_px)
    correlation_uv = float(correlation_uv)

    if sigma_u_px <= 0.0 or sigma_v_px <= 0.0:
        raise ValueError("pixel standard deviations must be > 0")
    if not -1.0 < correlation_uv < 1.0:
        raise ValueError("correlation_uv must be strictly between -1 and 1")

    covariance_uv = correlation_uv * sigma_u_px * sigma_v_px
    pixel_covariance = np.asarray(
        [
            [sigma_u_px * sigma_u_px, covariance_uv],
            [covariance_uv, sigma_v_px * sigma_v_px],
        ],
        dtype=np.float64,
    )

    jacobian = sensitivity.jacobian
    result = jacobian @ pixel_covariance @ jacobian.T
    result = 0.5 * (result + result.T)

    if not np.isfinite(result).all():
        raise ValueError("metric covariance is non-finite")

    eigenvalues = np.linalg.eigvalsh(result)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError("metric covariance is not positive semidefinite")

    return result


def covariance_summary(covariance: np.ndarray) -> dict[str, float]:
    """Convenient diagnostics for a 2x2 X/Z covariance matrix."""

    covariance = np.asarray(covariance, dtype=np.float64)
    if covariance.shape != (2, 2):
        raise ValueError("covariance must be 2x2")

    variance_x = max(0.0, float(covariance[0, 0]))
    variance_z = max(0.0, float(covariance[1, 1]))
    std_x = math.sqrt(variance_x)
    std_z = math.sqrt(variance_z)

    denominator = std_x * std_z
    correlation = (
        float(covariance[0, 1]) / denominator
        if denominator > 1e-12
        else 0.0
    )
    correlation = max(-1.0, min(1.0, correlation))

    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    ellipse_std_minor = math.sqrt(float(eigenvalues[0]))
    ellipse_std_major = math.sqrt(float(eigenvalues[1]))

    return {
        "std_x_m": std_x,
        "std_z_m": std_z,
        "correlation_xz": correlation,
        "ellipse_std_minor_m": ellipse_std_minor,
        "ellipse_std_major_m": ellipse_std_major,
    }


__all__ = [
    "ProjectionSensitivity",
    "projection_sensitivity",
    "metric_covariance",
    "covariance_summary",
]
