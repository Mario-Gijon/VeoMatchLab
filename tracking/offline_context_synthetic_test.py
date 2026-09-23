#!/usr/bin/env python3
"""Synthetic validation for clip-global offline context tracking.

The test deliberately fragments one real person's online identity, inserts a
short isolated false positive, keeps two footballers in close contact, and
checks that the global linker uses future context without spatial merging.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from tracking.association import AssociationConfig
from tracking.kalman import KalmanConfig
from tracking.offline_context import (
    OfflineContextConfig,
    OfflineSample,
    ForwardPass,
    ForwardPassStats,
    build_tracklets,
    build_trajectories,
    calibrate_covariance_scale,
    link_tracklets,
    reconstruct_frames,
)


FPS = 30.0
DT = 1.0 / FPS


@dataclass(frozen=True)
class FakeFrame:
    frame: int
    time_s: float
    observations: tuple = ()


def sample(track_id, frame, x, z, *, sigma=0.08, confidence=0.8):
    return OfflineSample(
        source_track_id=track_id,
        frame=frame,
        time_s=frame * DT,
        observation_index=0,
        detection_index=0,
        x=float(x),
        z=float(z),
        confidence=float(confidence),
        covariance=np.diag([sigma**2, (sigma * 1.4)**2]),
    )


def make_fragment(track_id, start, end, x0, vx, z):
    return [
        sample(
            track_id,
            frame,
            x0 + vx * ((frame - start) * DT),
            z,
        )
        for frame in range(start, end + 1)
    ]


def main():
    cfg = OfflineContextConfig(
        calibration_min_triplets=10,
        tracklet_split_gap_s=0.75,
        linker_max_gap_s=3.0,
        final_min_observations=5,
        max_visible_tracks=25,
    )

    # Real player A is deliberately split into two source IDs with a 0.4 s gap.
    a1 = make_fragment(10, 0, 24, -8.0, 2.4, -0.12)
    a2 = make_fragment(51, 37, 89, -8.0 + 2.4 * (37 * DT), 2.4, -0.12)

    # Real player B moves oppositely only 24 cm away in Z around the crossing.
    b = [
        sample(
            20,
            frame,
            7.0 - 2.2 * (frame * DT),
            +0.12,
        )
        for frame in range(90)
    ]

    # Short isolated false positive: four frames, nowhere near a plausible
    # continuation. It must stay weak rather than being merged by proximity.
    false_fragment = [
        sample(99, frame, 30.0, 20.0, confidence=0.35)
        for frame in range(40, 44)
    ]

    preliminary = ForwardPass(
        samples_by_track={
            10: a1,
            51: a2,
            20: b,
            99: false_fragment,
        },
        ever_confirmed_track_ids={10, 51, 20},
        stats=ForwardPassStats(
            frames=90,
            observations=len(a1) + len(a2) + len(b) + len(false_fragment),
            assignments=0,
            unmatched_observations=0,
            tracks_created=4,
            confirmations=3,
            reactivations=0,
            lost_transitions=0,
            retirements=0,
            max_visible=3,
        ),
    )

    calibration = calibrate_covariance_scale(
        preliminary,
        config=cfg,
    )
    print("Offline-context synthetic validation")
    print(f"calibration status:          {calibration.status}")
    print(f"calibration triplets:        {calibration.triplet_count}")
    print(f"calibration scale:           {calibration.scale:.4f}")

    tracklets = build_tracklets(preliminary, config=cfg)
    links = link_tracklets(tracklets, config=cfg)
    trajectories = build_trajectories(
        tracklets,
        links,
        config=cfg,
        kalman_config=KalmanConfig(),
    )

    print(f"tracklets:                   {len(tracklets)}")
    print(f"accepted links:              {len(links.links)}")
    print(f"trajectories:                {len(trajectories)}")

    trajectory_by_sources = {
        tuple(sorted(set(item.source_track_ids))): item
        for item in trajectories
    }

    linked_a = None
    for trajectory in trajectories:
        sources = set(trajectory.source_track_ids)
        if {10, 51}.issubset(sources):
            linked_a = trajectory
            break

    if linked_a is None:
        raise SystemExit("FAIL: future context did not reconnect player A fragments")

    # Player B must remain a different trajectory despite close contact.
    if 20 in set(linked_a.source_track_ids):
        raise SystemExit("FAIL: close-contact player B was spatially merged into A")

    false_trajectory = None
    for trajectory in trajectories:
        if 99 in set(trajectory.source_track_ids):
            false_trajectory = trajectory
            break

    if false_trajectory is None:
        raise SystemExit("FAIL: false fragment disappeared from diagnostics")
    if false_trajectory.render_eligible:
        raise SystemExit("FAIL: isolated four-frame false fragment became render-eligible")

    frames = [
        FakeFrame(frame=i, time_s=i * DT)
        for i in range(90)
    ]
    reconstructed = reconstruct_frames(
        frames,
        trajectories,
        config=cfg,
    )

    # The 0.4 s gap between A fragments is below render_gap_s=0.75, so the
    # final offline result should fill it after seeing both past and future.
    gap_visible = 0
    for frame in range(25, 37):
        items = reconstructed[frame]
        if any(item.trajectory_id == linked_a.trajectory_id for item in items):
            gap_visible += 1

    print(f"A linked source IDs:         {linked_a.source_track_ids}")
    print(f"A fragment count:            {linked_a.fragment_count}")
    print(f"A filled gap frames:         {gap_visible}/12")
    print(f"false fragment eligible:     {false_trajectory.render_eligible}")

    if gap_visible != 12:
        raise SystemExit("FAIL: offline reconstruction did not fill A's short gap")

    # At the crossing both real people should exist simultaneously.
    simultaneous = 0
    b_trajectory_id = next(
        item.trajectory_id
        for item in trajectories
        if 20 in set(item.source_track_ids)
    )
    for frame in range(35, 55):
        ids = {item.trajectory_id for item in reconstructed[frame]}
        if linked_a.trajectory_id in ids and b_trajectory_id in ids:
            simultaneous += 1

    print(f"close-contact simultaneous:  {simultaneous} frames")
    if simultaneous == 0:
        raise SystemExit("FAIL: close-contact players were not independently preserved")

    print("PASS")


if __name__ == "__main__":
    main()
