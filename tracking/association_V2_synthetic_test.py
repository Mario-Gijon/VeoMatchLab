#!/usr/bin/env python3
"""V2 association tests.

Run after promoting association_V2.py to the stable tracking/association.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tracking.association import (
    AssociationConfig,
    evaluate_pair,
    physical_gate_components,
)
from tracking.kalman import KalmanConfig
from tracking.track import Track


@dataclass(frozen=True)
class Obs:
    frame: int
    time_s: float
    world_x: float
    world_z: float
    confidence: float
    measurement_covariance: np.ndarray

    @property
    def world_position(self):
        return np.asarray([self.world_x, self.world_z], dtype=np.float64)


def make_obs(frame, x, z, sigma):
    return Obs(
        frame=frame,
        time_s=frame / 30.0,
        world_x=x,
        world_z=z,
        confidence=0.7,
        measurement_covariance=np.diag([sigma**2, sigma**2]),
    )


def main():
    cfg = AssociationConfig(
        mahalanobis_gate_squared=1000.0,  # isolate physical gate in this test
        max_speed_mps=12.5,
        position_slack_m=0.10,
        uncertainty_sigma_multiplier=3.0,
    )

    initial = make_obs(0, 0.0, 0.0, 0.05)

    near_track = Track(
        track_id=1,
        initial_observation=initial,
        kalman_config=KalmanConfig(),
    )
    near_track.predict_to(frame=1, time_s=1 / 30.0)

    # Same displacement, but one observation is precise and the other is from
    # a deliberately noisy geometry region.
    precise = make_obs(1, 1.15, 0.0, 0.05)
    noisy = make_obs(1, 1.15, 0.0, 0.40)

    precise_pair = evaluate_pair(
        near_track, precise,
        track_index=0, observation_index=0, config=cfg
    )
    noisy_pair = evaluate_pair(
        near_track, noisy,
        track_index=0, observation_index=1, config=cfg
    )

    _, precise_uncertainty, precise_radius = physical_gate_components(
        near_track, precise, config=cfg
    )
    _, noisy_uncertainty, noisy_radius = physical_gate_components(
        near_track, noisy, config=cfg
    )

    # A truly impossible jump must remain rejected even with noisy geometry.
    impossible = make_obs(1, 5.0, 0.0, 0.40)
    impossible_pair = evaluate_pair(
        near_track, impossible,
        track_index=0, observation_index=2, config=cfg
    )

    print("V2 uncertainty-aware physical gate")
    print(f"precise uncertainty slack: {precise_uncertainty:.4f} m")
    print(f"noisy uncertainty slack:   {noisy_uncertainty:.4f} m")
    print(f"precise total radius:      {precise_radius:.4f} m")
    print(f"noisy total radius:        {noisy_radius:.4f} m")
    print(f"precise 1.15 m allowed:    {precise_pair.allowed}")
    print(f"noisy 1.15 m allowed:      {noisy_pair.allowed}")
    print(f"noisy 5.00 m allowed:      {impossible_pair.allowed}")

    if not noisy_uncertainty > precise_uncertainty * 4.0:
        raise SystemExit("FAIL: noisy Veo observation did not receive wider slack")
    if precise_pair.allowed:
        raise SystemExit("FAIL: precise observation received excessive physical slack")
    if not noisy_pair.allowed:
        raise SystemExit("FAIL: noisy observation did not benefit from adaptive slack")
    if impossible_pair.allowed:
        raise SystemExit("FAIL: impossible jump passed the physical gate")

    print("PASS")


if __name__ == "__main__":
    main()
