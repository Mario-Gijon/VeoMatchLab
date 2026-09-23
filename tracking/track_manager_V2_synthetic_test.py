#!/usr/bin/env python3
"""V2 TrackManager tests focused on football contact and cascade priority.

Run after promoting:
    association_V2.py   -> association.py
    track_manager_V2.py -> track_manager.py

The test covers two separate failure modes:

1. A one-frame near-duplicate candidate appears beside an established player.
   On the next frame only one observation remains. CONFIRMED must get first
   claim; the TENTATIVE candidate must not steal it and become a replacement.

2. Two *real* players approach to close-contact distance, one is briefly
   occluded, and both later separate again. They must remain two independent
   tracks. No proximity merge/suppression is permitted.
"""

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


@dataclass(frozen=True)
class Obs:
    frame: int
    time_s: float
    truth_id: int
    world_x: float
    world_z: float
    confidence: float
    measurement_covariance: np.ndarray

    @property
    def world_position(self):
        return np.asarray([self.world_x, self.world_z], dtype=np.float64)


def obs(frame, truth_id, x, z, sigma=0.07, confidence=0.75):
    return Obs(
        frame=frame,
        time_s=frame * DT,
        truth_id=truth_id,
        world_x=float(x),
        world_z=float(z),
        confidence=float(confidence),
        measurement_covariance=np.diag([sigma**2, (sigma * 1.6)**2]),
    )


def manager():
    return TrackManager(
        config=TrackManagerConfig(
            min_confirmed_hits=3,
            tentative_max_gap_s=0.15,
            visible_prediction_horizon_s=0.50,
            lost_memory_horizon_s=2.00,
            max_visible_tracks=25,
        ),
        kalman_config=KalmanConfig(
            acceleration_std_mps2=4.0,
            initial_velocity_std_mps=6.0,
        ),
        association_config=AssociationConfig(
            mahalanobis_gate_squared=16.0,
            max_speed_mps=12.5,
            position_slack_m=0.10,
            uncertainty_sigma_multiplier=3.0,
            euclidean_cost_weight=0.05,
        ),
    )


def test_confirmed_priority():
    m = manager()

    # Establish one real player.
    for frame in range(3):
        m.process_frame(
            frame=frame,
            time_s=frame * DT,
            observations=[obs(frame, 10, 0.10 * frame, 0.0)],
        )

    confirmed = [
        managed for managed in m.active_tracks
        if managed.lifecycle == TrackLifecycle.CONFIRMED
    ]
    if len(confirmed) != 1:
        raise SystemExit("FAIL: base player did not confirm")
    established_id = confirmed[0].track_id

    # Frame 3: detector provides the established player plus a second nearby
    # observation. Since CONFIRMED consumes one, the independent second
    # observation legitimately becomes a tentative candidate.
    result3 = m.process_frame(
        frame=3,
        time_s=3 * DT,
        observations=[
            obs(3, 10, 0.30, 0.0),
            obs(3, 999, 0.34, 0.02),
        ],
    )
    if len(result3.new_track_ids) != 1:
        raise SystemExit("FAIL: expected one tentative nearby candidate")
    tentative_id = result3.new_track_ids[0]

    # Frame 4: only one observation exists and it is deliberately a little
    # closer to yesterday's tentative point. Cascade priority must still give
    # the established CONFIRMED track first claim.
    result4 = m.process_frame(
        frame=4,
        time_s=4 * DT,
        observations=[obs(4, 10, 0.39, 0.015)],
    )

    assigned_ids = {a.track_id for a in result4.assignments}
    if established_id not in assigned_ids:
        raise SystemExit("FAIL: tentative candidate stole established observation")
    if tentative_id in assigned_ids:
        raise SystemExit("FAIL: tentative candidate received already-claimed observation")

    # Continue the real player. The accidental candidate must never confirm.
    false_confirmed = False
    for frame in range(5, 12):
        result = m.process_frame(
            frame=frame,
            time_s=frame * DT,
            observations=[obs(frame, 10, 0.10 * frame, 0.0)],
        )
        if tentative_id in result.newly_confirmed_track_ids:
            false_confirmed = True

    if false_confirmed:
        raise SystemExit("FAIL: nearby one-frame candidate became confirmed")

    print("cascade priority: PASS")


def contact_positions(frame):
    """Two players pass at ~0.22 m separation while moving oppositely."""
    t = frame * DT
    # They cross in X around frame 45 but remain slightly separated in Z.
    ax = -4.5 + 3.0 * t
    bx = 4.5 - 3.0 * t

    # Minimum lateral separation around 22 cm.
    az = -0.11
    bz = +0.11
    return (ax, az), (bx, bz)


def test_contact_and_occlusion():
    rng = np.random.default_rng(20260923)
    m = manager()

    truth_to_primary_track = {}
    origin_truth = {}

    wrong_assignments = 0
    max_simultaneous_visible = 0
    contact_min_distance = float("inf")
    occluded_track_was_predicted = False
    same_id_recovered = False

    # Player B is hidden for 8 frames around the closest-contact region.
    occlusion = set(range(43, 51))

    for frame in range(90):
        (ax, az), (bx, bz) = contact_positions(frame)

        observations = []

        Ra = np.diag([0.05**2, 0.08**2])
        ma = rng.multivariate_normal([ax, az], Ra)
        observations.append(
            Obs(frame, frame * DT, 1, ma[0], ma[1], 0.80, Ra)
        )

        if frame not in occlusion:
            Rb = np.diag([0.05**2, 0.08**2])
            mb = rng.multivariate_normal([bx, bz], Rb)
            observations.append(
                Obs(frame, frame * DT, 2, mb[0], mb[1], 0.82, Rb)
            )

        rng.shuffle(observations)

        result = m.process_frame(
            frame=frame,
            time_s=frame * DT,
            observations=observations,
        )

        active_by_id = {managed.track_id: managed for managed in m.active_tracks}
        for track_id in result.new_track_ids:
            managed = active_by_id.get(track_id)
            if managed is None:
                continue
            tid = int(managed.track.last_observation.truth_id)
            origin_truth[track_id] = tid
            truth_to_primary_track.setdefault(tid, track_id)

        for assignment in result.assignments:
            o = observations[assignment.observation_index]
            origin = origin_truth.get(assignment.track_id)
            if origin is not None and origin != o.truth_id:
                wrong_assignments += 1

        a_id = truth_to_primary_track.get(1)
        b_id = truth_to_primary_track.get(2)

        snapshots = {s.track_id: s for s in result.all_active_tracks}
        if a_id in snapshots and b_id in snapshots:
            pa = np.asarray([snapshots[a_id].x, snapshots[a_id].z])
            pb = np.asarray([snapshots[b_id].x, snapshots[b_id].z])
            contact_min_distance = min(
                contact_min_distance,
                float(np.linalg.norm(pa - pb)),
            )

        if b_id is not None and frame in occlusion and b_id in snapshots:
            if snapshots[b_id].frame_state == "predicted":
                occluded_track_was_predicted = True

        if b_id is not None and frame == max(occlusion) + 1:
            if any(
                assignment.track_id == b_id
                and observations[assignment.observation_index].truth_id == 2
                for assignment in result.assignments
            ):
                same_id_recovered = True

        max_simultaneous_visible = max(
            max_simultaneous_visible,
            len(result.visible_track_ids),
        )

    a_id = truth_to_primary_track.get(1)
    b_id = truth_to_primary_track.get(2)

    if a_id is None or b_id is None or a_id == b_id:
        raise SystemExit("FAIL: two real close-contact players were not independent tracks")
    if wrong_assignments != 0:
        raise SystemExit("FAIL: identity switch occurred during contact/occlusion")
    if not occluded_track_was_predicted:
        raise SystemExit("FAIL: occluded player did not remain prediction-backed")
    if not same_id_recovered:
        raise SystemExit("FAIL: occluded player did not recover the same track ID")
    if contact_min_distance > 0.75:
        raise SystemExit(
            "FAIL: synthetic players never became close enough to test contact"
        )

    # Both should be confirmed and alive at the end.
    final = {managed.track_id: managed for managed in m.active_tracks}
    if final[a_id].lifecycle != TrackLifecycle.CONFIRMED:
        raise SystemExit("FAIL: player A not confirmed at end")
    if final[b_id].lifecycle != TrackLifecycle.CONFIRMED:
        raise SystemExit("FAIL: player B not confirmed at end")

    print(f"minimum tracked separation: {contact_min_distance:.4f} m")
    print(f"max simultaneous visible:   {max_simultaneous_visible}")
    print("football contact + occlusion: PASS")


def main():
    print("TrackManager V2 synthetic validation")
    test_confirmed_priority()
    test_contact_and_occlusion()
    print("PASS")


if __name__ == "__main__":
    main()
