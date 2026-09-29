from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """
    Linear Kalman filter over [X, Z, Vx, Vz].

    The process model is constant velocity with white acceleration noise.
    Measurement observations contain only field position [X, Z].
    """

    _H = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    _I = np.eye(4, dtype=np.float64)

    def __init__(
        self,
        *,
        x: float,
        z: float,
        process_acceleration_std: float,
        initial_position_std: float,
        initial_velocity_std: float,
    ) -> None:
        self._state = np.array([x, z, 0.0, 0.0], dtype=np.float64)
        self._covariance = np.diag(
            [
                initial_position_std**2,
                initial_position_std**2,
                initial_velocity_std**2,
                initial_velocity_std**2,
            ]
        ).astype(np.float64)
        self._process_acceleration_std = float(process_acceleration_std)

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
    def position_covariance(self) -> np.ndarray:
        return self._covariance[:2, :2].copy()

    def predict(self, dt: float) -> None:
        if dt < 0:
            raise ValueError("Kalman prediction dt must be >= 0")
        if dt == 0:
            return

        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        transition = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        q = self._process_acceleration_std**2
        process_noise = q * np.array(
            [
                [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
                [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
                [dt3 / 2.0, 0.0, dt2, 0.0],
                [0.0, dt3 / 2.0, 0.0, dt2],
            ],
            dtype=np.float64,
        )

        self._state = transition @ self._state
        self._covariance = (
            transition @ self._covariance @ transition.T + process_noise
        )
        self._symmetrize_covariance()

    def innovation(
        self,
        measurement_position: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(measurement_position, dtype=np.float64).reshape(2)
        measurement_covariance = np.asarray(
            measurement_covariance, dtype=np.float64
        ).reshape(2, 2)

        residual = position - self._H @ self._state
        innovation_covariance = (
            self._H @ self._covariance @ self._H.T + measurement_covariance
        )
        return residual, innovation_covariance

    def mahalanobis_squared(
        self,
        measurement_position: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> float:
        residual, innovation_covariance = self.innovation(
            measurement_position,
            measurement_covariance,
        )
        try:
            solution = np.linalg.solve(innovation_covariance, residual)
        except np.linalg.LinAlgError:
            solution = np.linalg.pinv(innovation_covariance) @ residual
        return float(residual.T @ solution)

    def update(
        self,
        measurement_position: np.ndarray,
        measurement_covariance: np.ndarray,
    ) -> None:
        position = np.asarray(measurement_position, dtype=np.float64).reshape(2)
        measurement_covariance = np.asarray(
            measurement_covariance, dtype=np.float64
        ).reshape(2, 2)

        residual, innovation_covariance = self.innovation(
            position,
            measurement_covariance,
        )

        ph_t = self._covariance @ self._H.T
        try:
            kalman_gain = np.linalg.solve(
                innovation_covariance.T,
                ph_t.T,
            ).T
        except np.linalg.LinAlgError:
            kalman_gain = ph_t @ np.linalg.pinv(innovation_covariance)

        self._state = self._state + kalman_gain @ residual

        # Joseph form is more numerically stable than (I - KH)P.
        identity_minus_kh = self._I - kalman_gain @ self._H
        self._covariance = (
            identity_minus_kh
            @ self._covariance
            @ identity_minus_kh.T
            + kalman_gain @ measurement_covariance @ kalman_gain.T
        )
        self._symmetrize_covariance()

    def clamp_speed(self, max_speed_m_s: float) -> None:
        velocity = self._state[2:4]
        speed = float(np.linalg.norm(velocity))
        if speed <= max_speed_m_s or speed == 0.0:
            return
        self._state[2:4] = velocity * (max_speed_m_s / speed)

    def _symmetrize_covariance(self) -> None:
        self._covariance = 0.5 * (self._covariance + self._covariance.T)
