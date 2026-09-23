#!/usr/bin/env python3
"""Synthetic validation for TrackManager lifecycle and visible-slot policy."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tracking.association import AssociationConfig
from tracking.kalman import KalmanConfig
from tracking.track_manager import (
    TrackLifecycle,
    TrackManager,
    TrackManagerConfig,
)


FPS = 30.0
DT = 1.0 / FPS
FRAME_COUNT = 180

SHORT_GAP_ID = 2
SHORT_GAP = range(70, 80)       # 10 frames ~= 0.33 s: should stay visible-eligible.

LONG_GAP_ID = 3
LONG_GAP = range(85, 110)       # 25 frames ~= 0.83 s: should become LOST,
                                 # then recover before 2 s memory expires.


@dataclass(frozen=True)
class SyntheticObservation:
    frame: int
    time_s: float
    truth_id: int
    world_x: float
    world_z: float
    confidence: float
    measurement_covariance: np.ndarray

    @property
    def world_position(self) -> np.ndarray:
        return np.asarray([self.world_x, self.world_z], dtype=np.float64)


def truth_position(truth_id: int, frame: int) -> np.ndarray:
    t = frame * DT
    angle = truth_id * 0.65
    x0 = -22.0 + truth_id * 6.2
    z0 = -15.0 + (truth_id % 4) * 9.0

    vx = 1.2 + (truth_id % 3) * 0.55
    vz = 0.45 * math.sin(angle)

    return np.asarray(
        [
            x0 + vx * t + 0.25 * math.sin(0.45 * t + angle),
            z0 + vz * t + 0.20 * math.cos(0.35 * t + angle),
        ],
        dtype=np.float64,
    )


def make_observations(frame: int, rng: np.random.Generator):
    time_s = frame * DT
    observations: list[SyntheticObservation] = []

    for truth_id in range(8):
        if truth_id == SHORT_GAP_ID and frame in SHORT_GAP:
            continue
        if truth_id == LONG_GAP_ID and frame in LONG_GAP:
            continue

        truth = truth_position(truth_id, frame)

        sigma_x = 0.07 + 0.01 * (truth_id % 2)
        sigma_z = 0.14 + 0.03 * (truth_id % 3)
        R = np.diag([sigma_x**2, sigma_z**2])
        measured = rng.multivariate_normal(truth, R)

        observations.append(
            SyntheticObservation(
                frame=frame,
                time_s=time_s,
                truth_id=truth_id,
                world_x=float(measured[0]),
                world_z=float(measured[1]),
                confidence=0.55 + 0.04 * (truth_id % 5),
                measurement_covariance=R,
            )
        )

    # One-frame clutter candidates. They must never become CONFIRMED.
    if frame in {25, 50, 75, 100, 125, 150}:
        observations.append(
            SyntheticObservation(
                frame=frame,
                time_s=time_s,
                truth_id=999,
                world_x=48.0 - 0.02 * frame,
                world_z=-26.0,
                confidence=0.92,
                measurement_covariance=np.diag([0.06**2, 0.10**2]),
            )
        )

    rng.shuffle(observations)
    return observations


def main() -> None:
    rng = np.random.default_rng(20260923)

    manager = TrackManager(
        config=TrackManagerConfig(
            min_confirmed_hits=3,
            tentative_max_gap_s=0.15,
            visible_prediction_horizon_s=0.50,
            lost_memory_horizon_s=2.00,
            max_visible_tracks=5,  # intentionally small to exercise the cap
            visible_retention_bonus=0.10,
        ),
        kalman_config=KalmanConfig(
            acceleration_std_mps2=4.0,
            initial_velocity_std_mps=6.0,
        ),
        association_config=AssociationConfig(
            mahalanobis_gate_squared=16.0,
            max_speed_mps=12.5,
            position_slack_m=1.50,
            euclidean_cost_weight=0.05,
        ),
    )

    track_origin_truth: dict[int, int] = {}
    truth_to_primary_track: dict[int, int] = {}

    wrong_identity_assignments = 0
    false_confirmations = 0
    max_visible_seen = 0

    short_gap_ever_lost = False
    long_gap_became_lost = False
    long_gap_reactivated_same_id = False

    confirmed_truth_ids: set[int] = set()

    for frame in range(FRAME_COUNT):
        time_s = frame * DT
        observations = make_observations(frame, rng)

        result = manager.process_frame(
            frame=frame,
            time_s=time_s,
            observations=observations,
        )

        # Capture origins of newly spawned internal candidates.
        active_by_id = {
            item.track_id: item
            for item in manager.active_tracks
        }
        for track_id in result.new_track_ids:
            managed = active_by_id.get(track_id)
            if managed is None:
                continue
            truth_id = int(managed.track.last_observation.truth_id)
            track_origin_truth[track_id] = truth_id
            if truth_id != 999:
                truth_to_primary_track.setdefault(truth_id, track_id)

        # Validate each association against synthetic truth.
        for assignment in result.assignments:
            observation = observations[assignment.observation_index]
            origin = track_origin_truth.get(assignment.track_id)
            if origin is not None and observation.truth_id != origin:
                wrong_identity_assignments += 1

        for track_id in result.newly_confirmed_track_ids:
            origin = track_origin_truth.get(track_id)
            if origin == 999:
                false_confirmations += 1
            elif origin is not None:
                confirmed_truth_ids.add(origin)

        max_visible_seen = max(max_visible_seen, len(result.visible_track_ids))

        short_id = truth_to_primary_track.get(SHORT_GAP_ID)
        long_id = truth_to_primary_track.get(LONG_GAP_ID)

        snapshots = {
            item.track_id: item
            for item in result.all_active_tracks
        }

        if short_id is not None and short_id in snapshots:
            if snapshots[short_id].lifecycle == TrackLifecycle.LOST.value:
                short_gap_ever_lost = True

        if long_id is not None:
            if long_id in result.newly_lost_track_ids:
                long_gap_became_lost = True
            if long_id in result.reactivated_track_ids:
                long_gap_reactivated_same_id = True

    active = manager.active_tracks
    retired = manager.retired_tracks

    confirmed_active_truths = {
        track_origin_truth.get(managed.track_id)
        for managed in active
        if managed.lifecycle == TrackLifecycle.CONFIRMED
    }
    confirmed_active_truths.discard(None)
    confirmed_active_truths.discard(999)

    retired_false_candidates = sum(
        1
        for managed in retired
        if track_origin_truth.get(managed.track_id) == 999
    )

    print("Synthetic track-manager validation")
    print(f"frames:                       {FRAME_COUNT}")
    print(f"true persons:                 8")
    print(f"confirmed true identities:    {len(confirmed_truth_ids)}")
    print(f"wrong identity assignments:   {wrong_identity_assignments}")
    print(f"false confirmations:          {false_confirmations}")
    print(f"retired one-frame clutter:    {retired_false_candidates}")
    print(f"max visible tracks:           {max_visible_seen}")
    print(f"short gap ever LOST:          {short_gap_ever_lost}")
    print(f"long gap became LOST:         {long_gap_became_lost}")
    print(f"long gap same-ID reactivated: {long_gap_reactivated_same_id}")
    print(f"active tracks at end:         {len(active)}")
    print(f"retired tracks at end:        {len(retired)}")

    if wrong_identity_assignments != 0:
        raise SystemExit("FAIL: synthetic identity switch detected")

    if false_confirmations != 0:
        raise SystemExit("FAIL: one-frame clutter became confirmed")

    if retired_false_candidates != 6:
        raise SystemExit(
            f"FAIL: expected 6 retired clutter candidates, got {retired_false_candidates}"
        )

    if max_visible_seen > 5:
        raise SystemExit("FAIL: visible-track cap was exceeded")

    if len(confirmed_truth_ids) != 8:
        raise SystemExit("FAIL: not all persistent true tracks were confirmed")

    if short_gap_ever_lost:
        raise SystemExit("FAIL: short gap should remain within visible prediction horizon")

    if not long_gap_became_lost:
        raise SystemExit("FAIL: long gap never entered LOST state")

    if not long_gap_reactivated_same_id:
        raise SystemExit("FAIL: long-gap track was not reactivated with the same ID")

    if len(confirmed_active_truths) != 8:
        raise SystemExit("FAIL: not all true tracks ended active and confirmed")

    print("PASS")


if __name__ == "__main__":
    main()
