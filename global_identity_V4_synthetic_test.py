#!/usr/bin/env python3
"""Synthetic validation for VeoMatchLab Global Identity V4."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from tracking.global_identity_V4 import (
    GlobalIdentityV4Config,
    make_config,
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
    trajectory_id: int,
    source_ids,
    start: int,
    end: int,
    *,
    x_at_frame0: float,
    vx: float,
    z: float,
    confidence: float = 0.8,
):
    samples = []
    for frame in range(start, end + 1):
        time_s = frame * DT
        samples.append(
            FakeSmoothedSample(
                frame=frame,
                time_s=time_s,
                x=x_at_frame0 + vx * time_s,
                z=z,
                vx=vx,
                vz=0.0,
            )
        )
    return FakeTrajectory(
        trajectory_id=trajectory_id,
        source_track_ids=tuple(source_ids),
        smoothed_samples=tuple(samples),
        observed_count=len(samples),
        observed_span_s=(end - start) * DT,
        fragment_count=1,
        mean_confidence=confidence,
    )


def main():
    cfg = make_config(max_visible_tracks=25)

    # Player A is fragmented with a five-frame observed overlap.  V3's old
    # sequential linker rejects this pattern; V4 may hand it off because raw
    # evidence in the overlap supports only one A observation.
    a1 = trajectory(1, (10,), 0, 30, x_at_frame0=-4.0, vx=1.5, z=0.0)
    a2 = trajectory(2, (10,), 26, 60, x_at_frame0=-3.95, vx=1.5, z=0.03)

    # Player B travels only 24 cm away with its own independent detection every
    # frame.  It must never merge with A.
    b = trajectory(3, (20,), 0, 60, x_at_frame0=-4.0, vx=1.5, z=0.24)

    # Player C has a short middle fragment.  Alone it is too weak to be a final
    # identity, but between two CORE fragments it should become a BRIDGE.
    c1 = trajectory(4, (30,), 0, 20, x_at_frame0=8.0, vx=-1.0, z=5.0)
    c_bridge = trajectory(5, (30,), 21, 27, x_at_frame0=8.0, vx=-1.0, z=5.0)
    c2 = trajectory(6, (30,), 28, 60, x_at_frame0=8.0, vx=-1.0, z=5.0)

    # Four-frame isolated false positive: diagnostic only.
    false = trajectory(7, (99,), 40, 43, x_at_frame0=25.0, vx=0.0, z=15.0, confidence=0.35)

    # A genuine boundary person (staff/assistant-like) persists long enough to
    # be CORE and should remain a separate identity.
    staff = trajectory(
        8,
        (120,),
        30,
        60,
        x_at_frame0=10.0,
        vx=0.2,
        z=FIELD_WIDTH / 2.0 - 0.35,
        confidence=0.75,
    )

    trajectories = (a1, a2, b, c1, c_bridge, c2, false, staff)

    frames = []
    for frame in range(61):
        t = frame * DT
        observations = []

        # A and B are independently observed.
        observations.append(FakeObservation(np.asarray([-4.0 + 1.5 * t, 0.0])))
        observations.append(FakeObservation(np.asarray([-4.0 + 1.5 * t, 0.24])))

        # One observation for the real C person, regardless of proposal split.
        observations.append(FakeObservation(np.asarray([8.0 - 1.0 * t, 5.0])))

        if 40 <= frame <= 43:
            observations.append(FakeObservation(np.asarray([25.0, 15.0])))

        if 30 <= frame <= 60:
            observations.append(
                FakeObservation(
                    np.asarray([
                        10.0 + 0.2 * t,
                        FIELD_WIDTH / 2.0 - 0.35,
                    ])
                )
            )

        frames.append(FakeFrame(frame=frame, time_s=t, observations=tuple(observations)))

    result = solve_global_identities(
        frames,
        trajectories,
        field_length_m=FIELD_LENGTH,
        field_width_m=FIELD_WIDTH,
        config=cfg,
    )

    identity_a = next(
        identity
        for identity in result.identities
        if 1 in identity.member_trajectory_ids
    )
    if 2 not in identity_a.member_trajectory_ids:
        raise SystemExit("FAIL: overlap-aware A handoff did not link A1 -> A2")

    identity_b = next(
        identity
        for identity in result.identities
        if 3 in identity.member_trajectory_ids
    )
    if identity_a.identity_id == identity_b.identity_id:
        raise SystemExit("FAIL: close player B was merged into A")

    identity_c = next(
        identity
        for identity in result.identities
        if 4 in identity.member_trajectory_ids
    )
    if not {4, 5, 6}.issubset(set(identity_c.member_trajectory_ids)):
        raise SystemExit("FAIL: short C bridge did not connect the two core fragments")

    if any(7 in identity.member_trajectory_ids for identity in result.identities):
        raise SystemExit("FAIL: isolated four-frame false positive became final identity")

    staff_identity = next(
        identity
        for identity in result.identities
        if 8 in identity.member_trajectory_ids
    )
    if staff_identity.identity_id in {identity_a.identity_id, identity_b.identity_id}:
        raise SystemExit("FAIL: boundary staff was fused into a player")

    overlap_edges = [
        candidate
        for candidate in result.link_candidates
        if candidate.handoff_mode == "observed_overlap"
    ]
    accepted = set(result.accepted_links)
    a_fragment = next(
        fragment.fragment_id
        for fragment in result.fragments
        if 1 in fragment.member_trajectory_ids
    )
    a2_fragment = next(
        fragment.fragment_id
        for fragment in result.fragments
        if 2 in fragment.member_trajectory_ids
    )
    if (a_fragment, a2_fragment) not in accepted:
        raise SystemExit("FAIL: expected observed-overlap edge was not accepted")

    bridge_fragment = next(
        fragment.fragment_id
        for fragment in result.fragments
        if 5 in fragment.member_trajectory_ids
    )
    if bridge_fragment not in set(result.bridge_fragment_ids):
        raise SystemExit("FAIL: contextual bridge fragment was not promoted to BRIDGE")

    false_fragment = next(
        fragment.fragment_id
        for fragment in result.fragments
        if 7 in fragment.member_trajectory_ids
    )
    if false_fragment not in set(result.ephemeral_fragment_ids):
        raise SystemExit("FAIL: isolated false fragment not classified EPHEMERAL")

    simultaneous_ab = 0
    for frame in range(0, 61):
        ids = {track.trajectory_id for track in result.frame_tracks.get(frame, ())}
        if identity_a.identity_id in ids and identity_b.identity_id in ids:
            simultaneous_ab += 1
    if simultaneous_ab < 50:
        raise SystemExit("FAIL: close-contact A/B were not preserved independently")

    print("Global Identity V4 synthetic validation")
    print(f"input V2 trajectories:       {len(trajectories)}")
    print(f"raw link candidates:         {result.raw_candidate_count}")
    print(f"retained link candidates:    {len(result.link_candidates)}")
    print(f"accepted global links:       {len(result.accepted_links)}")
    print(f"CORE fragments:              {len(result.core_fragment_ids)}")
    print(f"BRIDGE fragments:            {len(result.bridge_fragment_ids)}")
    print(f"EPHEMERAL fragments:         {len(result.ephemeral_fragment_ids)}")
    print(f"final identities:            {len(result.identities)}")
    print(f"observed-overlap candidates: {len(overlap_edges)}")
    print(f"A members:                   {identity_a.member_trajectory_ids}")
    print(f"C members:                   {identity_c.member_trajectory_ids}")
    print(f"A/B simultaneous frames:     {simultaneous_ab}")
    print("isolated false rendered:     False")
    print("boundary staff preserved:    True")
    print("PASS")


if __name__ == "__main__":
    main()
