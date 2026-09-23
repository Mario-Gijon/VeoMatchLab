#!/usr/bin/env python3
"""Deterministic synthetic validation for tracking.kalman.

The test deliberately includes:
- low-noise and high-noise measurement regions,
- anisotropic X/Z covariance,
- a short observation gap,
- per-frame variable R,
- prediction-only frames.

It does not test multi-object association.  Its purpose is to validate the
single-track motion primitive before Hungarian/lifecycle logic is added.
"""

from __future__ import annotations

import math

import numpy as np

from tracking.kalman import FieldKalmanFilter, KalmanConfig


FPS = 30.0
DT = 1.0 / FPS
FRAME_COUNT = 300
GAP_START = 135
GAP_END = 149  # inclusive


def true_state(frame: int) -> tuple[np.ndarray, np.ndarray]:
    """Smooth synthetic football-like path with gentle direction changes."""

    t = frame * DT

    # Ground-truth position in metres.
    x = 4.0 + 3.2 * t + 0.45 * math.sin(0.65 * t)
    z = -7.0 + 1.25 * t + 0.75 * math.sin(0.38 * t + 0.4)

    # Analytical velocity.
    vx = 3.2 + 0.45 * 0.65 * math.cos(0.65 * t)
    vz = 1.25 + 0.75 * 0.38 * math.cos(0.38 * t + 0.4)

    return (
        np.asarray([x, z], dtype=np.float64),
        np.asarray([vx, vz], dtype=np.float64),
    )


def measurement_covariance(frame: int) -> np.ndarray:
    """Pretend the second half is a geometrically difficult Veo region."""

    if frame < FRAME_COUNT // 2:
        sigma_x = 0.06
        sigma_z = 0.11
        correlation = 0.15
    else:
        sigma_x = 0.18
        sigma_z = 0.38
        correlation = 0.30

    covariance_xz = correlation * sigma_x * sigma_z

    return np.asarray(
        [
            [sigma_x**2, covariance_xz],
            [covariance_xz, sigma_z**2],
        ],
        dtype=np.float64,
    )


def main() -> None:
    rng = np.random.default_rng(20260923)

    truth = []
    measurements = []
    filtered = []
    statuses = []

    first_truth, _ = true_state(0)
    first_R = measurement_covariance(0)
    first_measurement = rng.multivariate_normal(first_truth, first_R)

    kf = FieldKalmanFilter.from_measurement(
        first_measurement,
        first_R,
        config=KalmanConfig(
            acceleration_std_mps2=4.0,
            initial_velocity_std_mps=6.0,
        ),
    )

    truth.append(first_truth)
    measurements.append(first_measurement)
    filtered.append(kf.position)
    statuses.append("observed")

    for frame in range(1, FRAME_COUNT):
        position_true, _ = true_state(frame)
        R = measurement_covariance(frame)

        kf.predict(DT)

        if GAP_START <= frame <= GAP_END:
            measurement = None
            statuses.append("predicted")
        else:
            measurement = rng.multivariate_normal(position_true, R)
            kf.update(measurement, R)
            statuses.append("observed")

        truth.append(position_true)
        measurements.append(
            np.asarray([np.nan, np.nan])
            if measurement is None
            else measurement
        )
        filtered.append(kf.position)

    truth_arr = np.asarray(truth)
    measurements_arr = np.asarray(measurements)
    filtered_arr = np.asarray(filtered)

    observed_mask = np.isfinite(measurements_arr[:, 0])

    raw_error = np.linalg.norm(
        measurements_arr[observed_mask] - truth_arr[observed_mask],
        axis=1,
    )
    filtered_error = np.linalg.norm(
        filtered_arr[observed_mask] - truth_arr[observed_mask],
        axis=1,
    )

    raw_rmse = float(np.sqrt(np.mean(raw_error**2)))
    filtered_rmse = float(np.sqrt(np.mean(filtered_error**2)))

    gap_error = np.linalg.norm(
        filtered_arr[GAP_START : GAP_END + 1]
        - truth_arr[GAP_START : GAP_END + 1],
        axis=1,
    )
    gap_max_error = float(np.max(gap_error))

    # A simple jitter proxy: frame-to-frame acceleration of the displayed path.
    filtered_step = np.diff(filtered_arr, axis=0)
    filtered_second_difference = np.diff(filtered_step, axis=0)
    max_second_difference = float(
        np.max(np.linalg.norm(filtered_second_difference, axis=1))
    )

    print("Synthetic Kalman validation")
    print(f"frames:                 {FRAME_COUNT}")
    print(f"gap:                    {GAP_START}-{GAP_END} "
          f"({GAP_END - GAP_START + 1} frames)")
    print(f"raw observed RMSE:      {raw_rmse:.4f} m")
    print(f"filtered observed RMSE: {filtered_rmse:.4f} m")
    print(f"gap max error:          {gap_max_error:.4f} m")
    print(f"max 2nd-difference:     {max_second_difference:.4f} m/frame²")
    print(f"final speed estimate:   {kf.speed_mps:.4f} m/s")

    if not np.isfinite(filtered_arr).all():
        raise SystemExit("FAIL: filtered trajectory contains non-finite values")

    if filtered_rmse >= raw_rmse:
        raise SystemExit(
            "FAIL: filtered RMSE did not improve over raw measurements"
        )

    if gap_max_error > 1.5:
        raise SystemExit(
            "FAIL: short prediction-only gap diverged excessively"
        )

    # Validate statistical gating behaviour: a nearby point should be much
    # cheaper than an impossible teleport.
    current_R = measurement_covariance(FRAME_COUNT - 1)
    near = kf.position + np.asarray([0.15, -0.10])
    far = kf.position + np.asarray([20.0, 15.0])

    near_d2 = kf.mahalanobis_squared(near, current_R)
    far_d2 = kf.mahalanobis_squared(far, current_R)

    print(f"near Mahalanobis²:      {near_d2:.4f}")
    print(f"far Mahalanobis²:       {far_d2:.4f}")

    if not far_d2 > near_d2 * 100.0:
        raise SystemExit(
            "FAIL: Mahalanobis gating does not separate near/far candidates"
        )

    print("PASS")


if __name__ == "__main__":
    main()
