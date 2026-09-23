#!/usr/bin/env python3
"""Synthetic tests for conservative offline context V2.

Run after copying offline_context_V2.py over tracking/offline_context.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tracking.kalman import KalmanConfig
from tracking.offline_context import (
    ForwardPass,
    ForwardPassStats,
    OfflineContextConfig,
    OfflineSample,
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


def fragment(track_id, start, end, x0, vx, z):
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
        anchor_min_observations=8,
        anchor_min_span_s=0.20,
        render_gap_s=0.50,
        endpoint_extrapolation_s=0.0,
        max_visible_tracks=25,
    )

    # Player A is fragmented into two independently confirmed proposal IDs.
    a1 = fragment(10, 0, 24, -8.0, 2.4, -0.12)
    a2_start_x = -8.0 + 2.4 * (37 * DT)
    a2 = fragment(51, 37, 89, a2_start_x, 2.4, -0.12)

    # Player B crosses close to A but must stay independent.
    b = [
        sample(
            20,
            frame,
            7.0 - 2.2 * (frame * DT),
            +0.12,
        )
        for frame in range(90)
    ]

    # A late real player C starts at frame 15. There is no evidence before
    # that time, so V2 must not back-extrapolate it to frame 0.
    c = fragment(30, 15, 60, -20.0, 1.0, 8.0)

    # Four-frame false positive: weak by design.
    false_fragment = [
        sample(99, frame, 30.0, 20.0, confidence=0.35)
        for frame in range(40, 44)
    ]

    # A source proposal with an impossible jump. Tracklet construction must
    # split it rather than fitting one absurd high-speed fragment.
    jumpy = fragment(70, 0, 12, 10.0, 0.5, -10.0)
    jumpy += [
        sample(70, 13, 18.0, -2.0, confidence=0.7),
        sample(70, 14, 18.1, -2.0, confidence=0.7),
    ]

    samples = {
        10: a1,
        51: a2,
        20: b,
        30: c,
        99: false_fragment,
        70: jumpy,
    }

    forward = ForwardPass(
        samples_by_track=samples,
        ever_confirmed_track_ids={10, 51, 20, 30, 70},
        stats=ForwardPassStats(
            frames=90,
            observations=sum(len(v) for v in samples.values()),
            assignments=0,
            unmatched_observations=0,
            tracks_created=6,
            confirmations=5,
            reactivations=0,
            lost_transitions=0,
            retirements=0,
            max_visible=5,
        ),
    )

    calibration = calibrate_covariance_scale(forward, config=cfg)
    tracklets = build_tracklets(forward, config=cfg)
    anchors = [item for item in tracklets if item.anchor]
    weak = [item for item in tracklets if not item.anchor]

    jumpy_tracklets = [
        item for item in tracklets
        if item.source_track_id == 70
    ]
    if len(jumpy_tracklets) < 2:
        raise SystemExit("FAIL: impossible jump did not split the tracklet")

    false_tracklets = [
        item for item in tracklets
        if item.source_track_id == 99
    ]
    if any(item.anchor for item in false_tracklets):
        raise SystemExit("FAIL: four-frame false fragment became anchor")

    links = link_tracklets(anchors, config=cfg)
    trajectories = build_trajectories(
        anchors,
        links,
        config=cfg,
        kalman_config=KalmanConfig(),
    )

    a_trajectory = next(
        (
            item for item in trajectories
            if {10, 51}.issubset(set(item.source_track_ids))
        ),
        None,
    )
    if a_trajectory is None:
        raise SystemExit("FAIL: A fragments were not globally linked")
    if 20 in set(a_trajectory.source_track_ids):
        raise SystemExit("FAIL: close player B was merged into A")

    frames = [
        FakeFrame(frame=i, time_s=i * DT)
        for i in range(90)
    ]
    reconstructed = reconstruct_frames(
        frames,
        trajectories,
        config=cfg,
    )

    filled = 0
    for frame in range(25, 37):
        ids = {item.trajectory_id for item in reconstructed[frame]}
        if a_trajectory.trajectory_id in ids:
            filled += 1
    if filled != 12:
        raise SystemExit("FAIL: A's bidirectionally supported short gap was not filled")

    c_trajectory = next(
        item for item in trajectories
        if 30 in set(item.source_track_ids)
    )
    if any(
        item.trajectory_id == c_trajectory.trajectory_id
        for item in reconstructed[0]
    ):
        raise SystemExit("FAIL: V2 extrapolated C backward without past evidence")

    b_trajectory = next(
        item for item in trajectories
        if 20 in set(item.source_track_ids)
    )
    simultaneous = sum(
        1
        for frame in range(35, 55)
        if {
            a_trajectory.trajectory_id,
            b_trajectory.trajectory_id,
        }.issubset({
            item.trajectory_id
            for item in reconstructed[frame]
        })
    )
    if simultaneous == 0:
        raise SystemExit("FAIL: close-contact players were not both preserved")

    print("Offline-context V2 synthetic validation")
    print(f"calibration scale:           {calibration.scale:.4f}")
    print(f"all tracklets:               {len(tracklets)}")
    print(f"anchor tracklets:            {len(anchors)}")
    print(f"weak tracklets:              {len(weak)}")
    print(f"accepted links:              {len(links.links)}")
    print(f"A filled gap frames:         {filled}/12")
    print(f"close-contact simultaneous:  {simultaneous}")
    print("endpoint back-extrapolation: False")
    print("four-frame false anchor:     False")
    print("impossible jump split:       True")
    print("PASS")


if __name__ == "__main__":
    main()
