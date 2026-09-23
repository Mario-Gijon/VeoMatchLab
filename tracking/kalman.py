"""Constant-velocity Kalman filter for VeoMatchLab field tracking.

State
-----
    x = [X, Z, Vx, Vz]

where X/Z are field coordinates in metres and Vx/Vz are metres/second.

Measurement
-----------
    z = [Xobs, Zobs]

Each observation supplies its own 2x2 measurement covariance R.  That is the
important link with tracking.veo_quality: observations in geometrically noisy
parts of the Veo panorama can be trusted less than observations in favourable
regions.

This module deliberately contains no track association, lifecycle management,
team classification or detector code.  It is the motion-estimation primitive
used by future Track objects.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np


_EPS = 1e-12


def _as_vec2(value: Iterable[float], *, name: str) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=np.float64)
    if result.shape != (2,):
        raise ValueError(f"{name} must contain exactly 2 values")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _as_cov2(value: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 2):
        raise ValueError(f"{name} must be a 2x2 matrix")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")

    # Numerical callers may produce tiny asymmetries.
    result = 0.5 * (result + result.T)

    eigenvalues = np.linalg.eigvalsh(result)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError(f"{name} must be positive semidefinite")

    return result


def _symmetrize_psd(matrix: np.ndarray, *, floor: float = 0.0) -> np.ndarray:
    """Return a symmetric covariance, clipping only tiny negative eigenvalues."""
    matrix = np.asarray(matrix, dtype=np.float64)
    matrix = 0.5 * (matrix + matrix.T)

    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, float(floor))
    repaired = (vectors * values) @ vectors.T
    return 0.5 * (repaired + repaired.T)


@dataclass(frozen=True)
class KalmanConfig:
    """Tunable motion assumptions.

    The defaults are intentionally permissive starting values, not football
    ground truth.  They should later be calibrated from real tracked clips.

    acceleration_std_mps2:
        White-acceleration process-noise scale.  Higher values let the filter
        react faster to changes of direction but reduce smoothing.

    initial_velocity_std_mps:
        Velocity uncertainty when a track is born from a position-only
        observation.

    covariance_floor:
        Small numerical floor applied to state covariance eigenvalues.
    """

    acceleration_std_mps2: float = 4.0
    initial_velocity_std_mps: float = 6.0
    covariance_floor: float = 1e-9

    def __post_init__(self) -> None:
        if not math.isfinite(self.acceleration_std_mps2):
            raise ValueError("acceleration_std_mps2 must be finite")
        if self.acceleration_std_mps2 < 0.0:
            raise ValueError("acceleration_std_mps2 must be >= 0")

        if not math.isfinite(self.initial_velocity_std_mps):
            raise ValueError("initial_velocity_std_mps must be finite")
        if self.initial_velocity_std_mps <= 0.0:
            raise ValueError("initial_velocity_std_mps must be > 0")

        if not math.isfinite(self.covariance_floor):
            raise ValueError("covariance_floor must be finite")
        if self.covariance_floor < 0.0:
            raise ValueError("covariance_floor must be >= 0")


@dataclass(frozen=True)
class KalmanPrediction:
    """State immediately after a predict() call."""

    dt: float
    position: np.ndarray
    velocity: np.ndarray
    state: np.ndarray
    covariance: np.ndarray


@dataclass(frozen=True)
class KalmanUpdate:
    """Diagnostics from one measurement update."""

    measurement: np.ndarray
    measurement_covariance: np.ndarray
    predicted_position: np.ndarray
    filtered_position: np.ndarray
    filtered_velocity: np.ndarray

    innovation: np.ndarray
    innovation_covariance: np.ndarray
    mahalanobis_squared: float

    state: np.ndarray
    covariance: np.ndarray


class FieldKalmanFilter:
    """2D constant-velocity Kalman filter in metric pitch coordinates."""

    measurement_matrix = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    def __init__(
        self,
        *,
        state: np.ndarray,
        covariance: np.ndarray,
        config: KalmanConfig | None = None,
    ) -> None:
        self.config = config or KalmanConfig()

        state = np.asarray(state, dtype=np.float64)
        covariance = np.asarray(covariance, dtype=np.float64)

        if state.shape != (4,):
            raise ValueError("state must have shape (4,)")
        if covariance.shape != (4, 4):
            raise ValueError("covariance must have shape (4,4)")
        if not np.isfinite(state).all() or not np.isfinite(covariance).all():
            raise ValueError("state and covariance must be finite")

        self._state = state.copy()
        self._covariance = _symmetrize_psd(
            covariance,
            floor=self.config.covariance_floor,
        )

    @classmethod
    def from_measurement(
        cls,
        position: Iterable[float],
        measurement_covariance: np.ndarray,
        *,
        initial_velocity: Iterable[float] = (0.0, 0.0),
        config: KalmanConfig | None = None,
    ) -> "FieldKalmanFilter":
        """Create a new track from one position observation."""

        cfg = config or KalmanConfig()
        position = _as_vec2(position, name="position")
        velocity = _as_vec2(initial_velocity, name="initial_velocity")
        measurement_covariance = _as_cov2(
            measurement_covariance,
            name="measurement_covariance",
        )

        state = np.asarray(
            [position[0], position[1], velocity[0], velocity[1]],
            dtype=np.float64,
        )

        covariance = np.zeros((4, 4), dtype=np.float64)
        covariance[:2, :2] = measurement_covariance

        velocity_variance = cfg.initial_velocity_std_mps ** 2
        covariance[2, 2] = velocity_variance
        covariance[3, 3] = velocity_variance

        return cls(
            state=state,
            covariance=covariance,
            config=cfg,
        )

    @property
    def state(self) -> np.ndarray:
        return self._state.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._covariance.copy()

    @property
    def position(self) -> np.ndarray:
        return self._state[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self._state[2:].copy()

    @property
    def speed_mps(self) -> float:
        return float(np.linalg.norm(self._state[2:]))

    @staticmethod
    def transition_matrix(dt: float) -> np.ndarray:
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")

        return np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def process_covariance(self, dt: float) -> np.ndarray:
        """White-acceleration process covariance for [X,Z,Vx,Vz].

        For each axis:
            position += velocity * dt
            acceleration is unmodelled white noise

        The standard constant-velocity discretisation gives the familiar
        dt^4/4, dt^3/2 and dt^2 terms.
        """

        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and > 0")

        q = self.config.acceleration_std_mps2 ** 2

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        return q * np.asarray(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

    def predict(self, dt: float) -> KalmanPrediction:
        """Advance the filter without a measurement."""

        F = self.transition_matrix(dt)
        Q = self.process_covariance(dt)

        self._state = F @ self._state
        self._covariance = (
            F @ self._covariance @ F.T + Q
        )
        self._covariance = _symmetrize_psd(
            self._covariance,
            floor=self.config.covariance_floor,
        )

        return KalmanPrediction(
            dt=float(dt),
            position=self.position,
            velocity=self.velocity,
            state=self.state,
            covariance=self.covariance,
        )

    def innovation(
        self,
        measurement: Iterable[float],
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return residual y and innovation covariance S for current state."""

        z = _as_vec2(measurement, name="measurement")
        R = _as_cov2(
            measurement_covariance,
            name="measurement_covariance",
        )

        H = self.measurement_matrix
        residual = z - H @ self._state
        innovation_covariance = H @ self._covariance @ H.T + R
        innovation_covariance = _symmetrize_psd(
            innovation_covariance,
            floor=self.config.covariance_floor,
        )
        return residual, innovation_covariance

    def mahalanobis_squared(
        self,
        measurement: Iterable[float],
        measurement_covariance: np.ndarray,
    ) -> float:
        """Statistical distance useful later for association gating."""

        residual, S = self.innovation(
            measurement,
            measurement_covariance,
        )

        try:
            solved = np.linalg.solve(S, residual)
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "innovation covariance is singular"
            ) from exc

        value = float(residual.T @ solved)
        return max(0.0, value)

    def update(
        self,
        measurement: Iterable[float],
        measurement_covariance: np.ndarray,
    ) -> KalmanUpdate:
        """Correct the current predicted state with one X/Z observation."""

        z = _as_vec2(measurement, name="measurement")
        R = _as_cov2(
            measurement_covariance,
            name="measurement_covariance",
        )

        H = self.measurement_matrix
        predicted_position = self.position

        residual, S = self.innovation(z, R)

        try:
            # K = P H^T S^-1, evaluated without explicitly inverting S.
            PHt = self._covariance @ H.T
            gain = np.linalg.solve(S.T, PHt.T).T
        except np.linalg.LinAlgError as exc:
            raise RuntimeError(
                "innovation covariance is singular"
            ) from exc

        mahalanobis_squared = self.mahalanobis_squared(z, R)

        self._state = self._state + gain @ residual

        # Joseph form is more numerically stable than (I-KH)P.
        identity = np.eye(4, dtype=np.float64)
        correction = identity - gain @ H
        self._covariance = (
            correction @ self._covariance @ correction.T
            + gain @ R @ gain.T
        )
        self._covariance = _symmetrize_psd(
            self._covariance,
            floor=self.config.covariance_floor,
        )

        return KalmanUpdate(
            measurement=z.copy(),
            measurement_covariance=R.copy(),
            predicted_position=predicted_position,
            filtered_position=self.position,
            filtered_velocity=self.velocity,
            innovation=residual.copy(),
            innovation_covariance=S.copy(),
            mahalanobis_squared=mahalanobis_squared,
            state=self.state,
            covariance=self.covariance,
        )

    def predict_and_update(
        self,
        *,
        dt: float,
        measurement: Iterable[float] | None,
        measurement_covariance: np.ndarray | None = None,
    ) -> KalmanPrediction | KalmanUpdate:
        """Convenience step.

        If measurement is None, prediction is kept and returned.
        If a measurement is supplied, R is required and the update result is
        returned.
        """

        prediction = self.predict(dt)

        if measurement is None:
            if measurement_covariance is not None:
                raise ValueError(
                    "measurement_covariance must be None when measurement is None"
                )
            return prediction

        if measurement_covariance is None:
            raise ValueError(
                "measurement_covariance is required when measurement is provided"
            )

        return self.update(
            measurement,
            measurement_covariance,
        )


__all__ = [
    "FieldKalmanFilter",
    "KalmanConfig",
    "KalmanPrediction",
    "KalmanUpdate",
]
