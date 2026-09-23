#!/usr/bin/env python3
"""Deterministic multi-object validation for Track + Hungarian association.

The synthetic scene contains four anonymous players.  Two pass close to one
another, one disappears for a short gap, and distant false detections are
injected.  The test checks that:

- IDs remain attached to the correct synthetic trajectories,
- the missing player is recovered with the same track after the gap,
- far false detections remain unmatched,
- prediction frames are produced instead of deleting the track,
- no implausible track jump appears.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tracking.association import AssociationConfig, associate
from tracking.kalman import KalmanConfig
from tracking.track import Track


FPS = 30.0
DT = 1.0 / FPS
FRAME_COUNT = 240

# Track truth 2 disappears for 8 frames and then returns.
GAP_TRUTH_ID = 2
GAP_START = 102
GAP_END = 109


@dataclass(frozen=True)
class SyntheticObservation:
    frame: int
    time_s: float
    truth_id: int
    world_x: float
    world_z: float
    measurement_covariance: np.ndarray

    @property
    def world_position(self) -> np.ndarray:
        return np.asarray([self.world_x, self.world_z], dtype=np.float64)


def truth_position(truth_id: int, frame: int) -> np.ndarray:
    t = frame * DT

    if truth_id == 0:
        # Left -> right through the central area.
        x = -18.0 + 4.6 * t
        z = -1.4 + 0.20 * math.sin(0.8 * t)
    elif truth_id == 1:
        # Right -> left, passes close to truth 0 with opposite velocity.
        x = 18.0 - 4.6 * t
        z = 1.4 + 0.20 * math.sin(0.8 * t + 0.5)
    elif truth_id == 2:
        x = -6.0 + 2.2 * t
        z = -14.0 + 1.3 * t + 0.35 * math.sin(0.5 * t)
    elif truth_id == 3:
        x = 7.0 + 0.6 * t
        z = 13.0 - 1.6 * t
    else:
        raise ValueError(truth_id)

    return np.asarray([x, z], dtype=np.float64)


def observation_covariance(truth_id: int, frame: int) -> np.ndarray:
    # Truth 2 simulates a less favourable Veo region.
    if truth_id == 2:
        sigma_x, sigma_z, rho = 0.15, 0.34, 0.20
    else:
        sigma_x, sigma_z, rho = 0.07, 0.13, 0.10

    cov_xz = rho * sigma_x * sigma_z
    return np.asarray(
        [
            [sigma_x**2, cov_xz],
            [cov_xz, sigma_z**2],
        ],
        dtype=np.float64,
    )


def make_observations(frame: int, rng: np.random.Generator) -> list[SyntheticObservation]:
    time_s = frame * DT
    observations: list[SyntheticObservation] = []

    for truth_id in range(4):
        if (
            truth_id == GAP_TRUTH_ID
            and GAP_START <= frame <= GAP_END
        ):
            continue

        truth = truth_position(truth_id, frame)
        R = observation_covariance(truth_id, frame)
        measured = rng.multivariate_normal(truth, R)

        observations.append(
            SyntheticObservation(
                frame=frame,
                time_s=time_s,
                truth_id=truth_id,
                world_x=float(measured[0]),
                world_z=float(measured[1]),
                measurement_covariance=R,
            )
        )

    # Far, physically impossible clutter.  It must not steal any established
    # track.  It is intentionally not turned into a new track in this test.
    if frame in {40, 80, 120, 160, 200}:
        observations.append(
            SyntheticObservation(
                frame=frame,
                time_s=time_s,
                truth_id=999,
                world_x=45.0,
                world_z=-28.0,
                measurement_covariance=np.diag([0.08**2, 0.12**2]),
            )
        )

    # Shuffle observation order so Hungarian cannot rely on list position.
    rng.shuffle(observations)
    return observations


def main() -> None:
    rng = np.random.default_rng(20260923)

    first_observations = make_observations(0, rng)
    by_truth = {
        observation.truth_id: observation
        for observation in first_observations
        if observation.truth_id != 999
    }

    tracks = [
        Track(
            track_id=truth_id,
            initial_observation=by_truth[truth_id],
            kalman_config=KalmanConfig(
                acceleration_std_mps2=4.0,
                initial_velocity_std_mps=6.0,
            ),
        )
        for truth_id in range(4)
    ]

    association_config = AssociationConfig(
        mahalanobis_gate_squared=16.0,
        max_speed_mps=12.5,
        position_slack_m=1.50,
        euclidean_cost_weight=0.05,
    )

    wrong_identity_matches = 0
    false_positive_matches = 0
    unmatched_false_positives = 0
    gap_prediction_frames = 0
    recovered_after_gap = False
    total_matches = 0
    max_filtered_step_m = 0.0

    previous_positions = {
        track.track_id: track.position.copy()
        for track in tracks
    }

    for frame in range(1, FRAME_COUNT):
        time_s = frame * DT

        for track in tracks:
            track.predict_to(frame=frame, time_s=time_s)

        observations = make_observations(frame, rng)

        result = associate(
            tracks,
            observations,
            config=association_config,
        )

        matched_by_track = {
            match.track_index: match
            for match in result.matches
        }

        for track_index, track in enumerate(tracks):
            match = matched_by_track.get(track_index)

            if match is None:
                track.mark_missed()
                if (
                    track.track_id == GAP_TRUTH_ID
                    and GAP_START <= frame <= GAP_END
                ):
                    gap_prediction_frames += 1
            else:
                observation = observations[match.observation_index]

                if observation.truth_id == 999:
                    false_positive_matches += 1
                elif observation.truth_id != track.track_id:
                    wrong_identity_matches += 1

                if (
                    track.track_id == GAP_TRUTH_ID
                    and frame == GAP_END + 1
                    and observation.truth_id == GAP_TRUTH_ID
                ):
                    recovered_after_gap = True

                track.update(observation)
                total_matches += 1

            step = float(
                np.linalg.norm(
                    track.position - previous_positions[track.track_id]
                )
            )
            max_filtered_step_m = max(max_filtered_step_m, step)
            previous_positions[track.track_id] = track.position.copy()

        for observation_index in result.unmatched_observation_indices:
            if observations[observation_index].truth_id == 999:
                unmatched_false_positives += 1

    print("Synthetic association validation")
    print(f"frames:                    {FRAME_COUNT}")
    print(f"tracks:                    {len(tracks)}")
    print(f"total matches:             {total_matches}")
    print(f"wrong identity matches:    {wrong_identity_matches}")
    print(f"false-positive matches:    {false_positive_matches}")
    print(f"unmatched false positives: {unmatched_false_positives}")
    print(f"gap prediction frames:     {gap_prediction_frames}")
    print(f"recovered after gap:       {recovered_after_gap}")
    print(f"max filtered step:         {max_filtered_step_m:.4f} m/frame")

    for track in tracks:
        snapshot = track.snapshot()
        print(
            f"track {track.track_id}: hits={snapshot.hits} "
            f"misses={snapshot.misses} "
            f"consecutive_misses={snapshot.consecutive_misses} "
            f"speed={snapshot.speed_mps:.3f} m/s"
        )

    if wrong_identity_matches != 0:
        raise SystemExit("FAIL: one or more tracks changed synthetic identity")

    if false_positive_matches != 0:
        raise SystemExit("FAIL: a far false detection stole an existing track")

    if unmatched_false_positives != 5:
        raise SystemExit(
            "FAIL: expected all five injected false detections to remain unmatched"
        )

    expected_gap_frames = GAP_END - GAP_START + 1
    if gap_prediction_frames != expected_gap_frames:
        raise SystemExit(
            f"FAIL: expected {expected_gap_frames} prediction-only gap frames, "
            f"got {gap_prediction_frames}"
        )

    if not recovered_after_gap:
        raise SystemExit("FAIL: missing track was not recovered after its gap")

    # At 30 FPS even a fast footballer should not move metres per frame.
    if max_filtered_step_m > 1.0:
        raise SystemExit(
            "FAIL: filtered trajectory contains an implausibly large frame jump"
        )

    print("PASS")


if __name__ == "__main__":
    main()
