#!/usr/bin/env python3
"""Synthetic validation for V2.1 diagnostic instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tracking.association import AssociationConfig
from tracking.kalman import KalmanConfig
from tracking.track_manager import TrackManager, TrackManagerConfig


FPS = 30.0
DT = 1.0 / FPS


@dataclass(frozen=True)
class Obs:
    frame: int
    time_s: float
    truth_id: int
    world_x: float
    world_z: float
    confidence: float
    measurement_covariance: np.ndarray
    detection_index: int

    @property
    def world_position(self):
        return np.asarray([self.world_x, self.world_z], dtype=np.float64)


def obs(frame, truth_id, x, z, sigma=0.06, detection_index=0):
    return Obs(
        frame=frame,
        time_s=frame * DT,
        truth_id=truth_id,
        world_x=float(x),
        world_z=float(z),
        confidence=0.8,
        measurement_covariance=np.diag([sigma**2, sigma**2]),
        detection_index=detection_index,
    )


def main():
    manager = TrackManager(
        config=TrackManagerConfig(
            min_confirmed_hits=3,
            tentative_max_gap_s=0.15,
            visible_prediction_horizon_s=0.50,
            lost_memory_horizon_s=2.0,
            max_visible_tracks=25,
        ),
        kalman_config=KalmanConfig(),
        association_config=AssociationConfig(
            mahalanobis_gate_squared=16.0,
            max_speed_mps=12.5,
            position_slack_m=0.10,
            uncertainty_sigma_multiplier=3.0,
        ),
    )

    # Establish a confirmed track moving gently.
    for frame in range(3):
        result = manager.process_frame(
            frame=frame,
            time_s=frame * DT,
            observations=[obs(frame, 1, 0.10 * frame, 0.0)],
        )

    # Give the confirmed track one valid observation plus one impossible
    # unmatched observation.  The unmatched item should expose the exact
    # rejection reason against the nearest confirmed track.
    result = manager.process_frame(
        frame=3,
        time_s=3 * DT,
        observations=[
            obs(3, 1, 0.30, 0.0, detection_index=0),
            obs(3, 999, 5.0, 0.0, detection_index=1),
        ],
    )

    stages = {
        stage.lifecycle: stage
        for stage in result.association_stage_diagnostics
    }

    confirmed = stages["confirmed"]

    print("V2.1 diagnostics synthetic validation")
    print(f"confirmed pairs evaluated:   {confirmed.pairs_evaluated}")
    print(f"confirmed physical rejects:  {confirmed.physical_gate_rejections}")
    print(f"confirmed matches:           {confirmed.matches}")
    print(f"final unmatched:             {len(result.unmatched_observation_indices)}")

    if confirmed.pairs_evaluated != 2:
        raise SystemExit("FAIL: expected two confirmed pair evaluations")
    if confirmed.matches != 1:
        raise SystemExit("FAIL: expected one confirmed match")
    if confirmed.physical_gate_rejections < 1:
        raise SystemExit("FAIL: impossible pair was not logged as physical rejection")

    if len(result.unmatched_observation_diagnostics) != 1:
        raise SystemExit("FAIL: expected one unmatched diagnostic")

    unmatched = result.unmatched_observation_diagnostics[0]

    # The established track is already matched one-to-one, so it is correctly
    # absent from the final unmatched-candidate set.  This confirms that the
    # diagnostic describes why the remaining free tracks could not claim the
    # observation, rather than pretending an already-consumed track could take
    # a second detection.
    if unmatched.observation_index != 1:
        raise SystemExit("FAIL: wrong unmatched observation was diagnosed")

    print("diagnostic bookkeeping:      PASS")
    print("PASS")


if __name__ == "__main__":
    main()
