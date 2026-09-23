#!/usr/bin/env python3
"""Synthetic validation for the VeoMatchLab V3 global identity solver.

The test covers the cases that matter for football:

- One real player duplicated into two overlapping V2 trajectories.
- Two different players only ~24 cm apart, with two independent raw detections:
  they MUST remain separate.
- The duplicated player's later fragment starts under a new source track ID:
  clip-global linking should recover the same identity.
- A person born at the sideline has a cheaper boundary birth than an otherwise
  equivalent interior birth.
- No identity is invented before its first supported state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from tracking.global_identity import (
    GlobalIdentityConfig,
    event_cost,
    solve_global_identities,
)


FPS = 30.0
DT = 1.0 / FPS
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


@dataclass(frozen=True)
class FakeObservation:
    world_position: np.ndarray


@dataclass(frozen=True)
class FakeFrame:
    frame: int
    time_s: float
    observations: tuple[FakeObservation, ...]


@dataclass(frozen=True)
class FakeSmoothedSample:
    frame: int
    time_s: float
    x: float
    z: float
    vx: float
    vz: float


@dataclass(frozen=True)
class FakeTrajectory:
    trajectory_id: int
    source_track_ids: tuple[int, ...]
    smoothed_samples: tuple[FakeSmoothedSample, ...]
    observed_count: int
    observed_span_s: float
    fragment_count: int
    mean_confidence: float


def trajectory(
    trajectory_id,
    source_track_ids,
    frame_start,
    frame_end,
    *,
    x0,
    vx,
    z,
    confidence=0.8,
    sample_step=1,
):
    sample_frames = list(range(frame_start, frame_end + 1, sample_step))
    if sample_frames[-1] != frame_end:
        sample_frames.append(frame_end)

    samples = []
    for frame in sample_frames:
        time_s = frame * DT
        x = x0 + vx * ((frame - frame_start) * DT)
        samples.append(
            FakeSmoothedSample(
                frame=frame,
                time_s=time_s,
                x=x,
                z=z,
                vx=vx,
                vz=0.0,
            )
        )

    return FakeTrajectory(
        trajectory_id=trajectory_id,
        source_track_ids=tuple(source_track_ids),
        smoothed_samples=tuple(samples),
        observed_count=len(samples),
        observed_span_s=(frame_end - frame_start) * DT,
        fragment_count=1,
        mean_confidence=confidence,
    )


def main():
    cfg = GlobalIdentityConfig(
        max_visible_tracks=25,
        duplicate_evidence_stride=1,
        bridge_evidence_stride=1,
        duplicate_min_informative_frames=6,
    )

    # Player A: true trajectory, later duplicated by V2 branch A'.
    a = trajectory(
        1,
        (10,),
        0,
        59,
        x0=-4.0,
        vx=1.5,
        z=0.0,
    )
    a_duplicate = trajectory(
        2,
        (10, 77),
        20,
        59,
        x0=-4.0 + 1.5 * (20 * DT) + 0.08,
        vx=1.48,
        z=0.05,
        confidence=0.65,
        # Sparse direct observations make this realistic: most of the duplicate
        # branch is V2 interpolation around the same underlying person.
        sample_step=5,
    )

    # Same real player A after a short fragmentation, but under a new source ID.
    # Raw bridge detections exist in frames 60..69.
    a_later = trajectory(
        4,
        (51,),
        70,
        89,
        x0=-4.0 + 1.5 * (70 * DT),
        vx=1.5,
        z=0.0,
    )

    # Player B is only 24 cm away in Z and follows similar X motion.  Crucially,
    # the detector supplies a second independent observation every frame.
    b = trajectory(
        3,
        (20,),
        0,
        89,
        x0=-4.0,
        vx=1.5,
        z=0.24,
        confidence=0.82,
    )

    # Staff/assistant-like boundary person.  Separate trajectory, born mid-clip.
    staff = trajectory(
        5,
        (99,),
        30,
        75,
        x0=10.0,
        vx=0.2,
        z=FIELD_WIDTH / 2.0 - 0.35,
        confidence=0.75,
    )

    trajectories = (a, a_duplicate, a_later, b, staff)

    frames = []
    for frame in range(90):
        time_s = frame * DT
        observations = []

        # One actual observation for A / A-duplicate.
        a_x = -4.0 + 1.5 * time_s
        observations.append(
            FakeObservation(
                np.asarray([a_x, 0.0], dtype=np.float64)
            )
        )

        # Independent evidence for close player B.
        observations.append(
            FakeObservation(
                np.asarray([a_x, 0.24], dtype=np.float64)
            )
        )

        # Bridge evidence for A during the online fragmentation.
        if 60 <= frame <= 69:
            observations.append(
                FakeObservation(
                    np.asarray([a_x, 0.01], dtype=np.float64)
                )
            )

        if 30 <= frame <= 75:
            staff_x = 10.0 + 0.2 * ((frame - 30) * DT)
            observations.append(
                FakeObservation(
                    np.asarray(
                        [staff_x, FIELD_WIDTH / 2.0 - 0.35],
                        dtype=np.float64,
                    )
                )
            )

        frames.append(
            FakeFrame(
                frame=frame,
                time_s=time_s,
                observations=tuple(observations),
            )
        )

    result = solve_global_identities(
        frames,
        trajectories,
        field_length_m=FIELD_LENGTH,
        field_width_m=FIELD_WIDTH,
        config=cfg,
    )

    duplicate_pairs = {
        frozenset((item.trajectory_a, item.trajectory_b))
        for item in result.duplicate_evidence
        if item.duplicate
    }

    if frozenset((1, 2)) not in duplicate_pairs:
        raise SystemExit("FAIL: duplicate A/A' was not identified")

    if frozenset((1, 3)) in duplicate_pairs:
        raise SystemExit(
            "FAIL: two close footballers with independent detections were merged"
        )

    # Find final identity containing original A.
    identity_a = next(
        identity
        for identity in result.identities
        if 1 in identity.member_trajectory_ids
    )

    if 2 not in identity_a.member_trajectory_ids:
        raise SystemExit("FAIL: duplicate A' was not collapsed into A")

    if 4 not in identity_a.member_trajectory_ids:
        raise SystemExit(
            "FAIL: future fragment A_later was not globally linked back to A"
        )

    identity_b = next(
        identity
        for identity in result.identities
        if 3 in identity.member_trajectory_ids
    )

    if identity_b.identity_id == identity_a.identity_id:
        raise SystemExit("FAIL: close player B collapsed into A")

    staff_identity = next(
        identity
        for identity in result.identities
        if 5 in identity.member_trajectory_ids
    )

    if staff_identity.identity_id == identity_a.identity_id:
        raise SystemExit("FAIL: sideline staff was linked onto player A")

    if min(staff_identity.states) < 30:
        raise SystemExit("FAIL: staff was invented before first evidence")

    midclip_time = 45 * DT
    boundary_cost = event_cost(
        position=np.asarray([0.0, FIELD_WIDTH / 2.0 - 0.2]),
        time_s=midclip_time,
        clip_edge_time_s=0.0,
        field_length_m=FIELD_LENGTH,
        field_width_m=FIELD_WIDTH,
        support_score=8.0,
        config=cfg,
    )
    interior_cost = event_cost(
        position=np.asarray([0.0, 0.0]),
        time_s=midclip_time,
        clip_edge_time_s=0.0,
        field_length_m=FIELD_LENGTH,
        field_width_m=FIELD_WIDTH,
        support_score=8.0,
        config=cfg,
    )

    if not boundary_cost < interior_cost:
        raise SystemExit("FAIL: boundary birth was not cheaper than interior birth")

    # Count simultaneous A+B presence around the close-contact section.
    simultaneous = 0
    for frame in range(20, 60):
        tracks = result.frame_tracks.get(frame, ())
        ids = {track.trajectory_id for track in tracks}
        if (
            identity_a.identity_id in ids
            and identity_b.identity_id in ids
        ):
            simultaneous += 1

    if simultaneous < 30:
        raise SystemExit(
            "FAIL: close-contact players were not preserved simultaneously"
        )

    print("Global Identity V3 synthetic validation")
    print(f"input V2 trajectories:       {len(trajectories)}")
    print(f"duplicate components:        {sum(len(c) > 1 for c in result.duplicate_components)}")
    print(f"accepted global links:       {len(result.accepted_links)}")
    print(f"final identities:            {len(result.identities)}")
    print(f"A member trajectories:       {identity_a.member_trajectory_ids}")
    print(f"A/B simultaneous frames:     {simultaneous}")
    print(f"boundary birth cost:         {boundary_cost:.3f}")
    print(f"interior birth cost:         {interior_cost:.3f}")
    print("close-player merge:          False")
    print("pre-evidence extrapolation:  False")
    print("PASS")


if __name__ == "__main__":
    main()
