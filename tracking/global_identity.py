"""Clip-global identity solver for VeoMatchLab Tracking V3.

This module sits *after* Offline Context V2.  V2 creates physically coherent
trajectory proposals from anchor tracklets.  V3 resolves identity globally over
the complete clip.

Design goals
------------
1. Never merge footballers merely because they are spatially close.
2. Use independent raw detections as explicit evidence that two close tracks
   can be two different people.
3. Collapse duplicate branches when they share lineage, move almost identically
   and lack independent evidence.
4. Link non-overlapping identity fragments with a clip-global path-cover
   optimization using past + future motion.
5. Make births/deaths cheaper near the pitch boundary and clip boundaries, so
   a coach, substitute or assistant entering from the sideline does not have to
   be forced onto an existing player identity.
6. Keep identity persistence separate from visualization.  A global identity
   may survive a long gap while remaining absent from rendered frames.
7. ``max_visible_tracks`` is a cap only, never a target.

The solver intentionally does not use team colours, appearance or jersey
numbers yet.  Those can later become additional edge evidence without changing
the graph structure.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


_FORBIDDEN = 1e12


@dataclass(frozen=True)
class GlobalIdentityConfig:
    # Build dense per-trajectory state maps from V2 smoothed observations.
    state_interpolation_gap_s: float = 0.50

    # Duplicate-branch evidence.
    duplicate_min_overlap_s: float = 0.40
    duplicate_min_overlap_frames: int = 12
    duplicate_median_distance_m: float = 0.65
    duplicate_p90_distance_m: float = 1.50
    duplicate_max_relative_speed_mps: float = 2.50
    duplicate_evidence_radius_m: float = 0.90
    duplicate_evidence_stride: int = 2
    duplicate_min_informative_frames: int = 6
    duplicate_max_independent_evidence_ratio: float = 0.25
    duplicate_max_both_observed_ratio: float = 0.25

    # Extremely conservative merge path when source lineage is not shared.
    no_lineage_duplicate_min_overlap_s: float = 1.50
    no_lineage_duplicate_median_distance_m: float = 0.25
    no_lineage_duplicate_p90_distance_m: float = 0.50
    no_lineage_duplicate_max_relative_speed_mps: float = 1.00
    no_lineage_duplicate_max_independent_ratio: float = 0.05

    # Transitive-cluster safeguard.  If two components have strong independent
    # evidence for two people, a third branch cannot accidentally fuse them.
    conflict_independent_evidence_ratio: float = 0.50
    conflict_min_informative_frames: int = 6
    conflict_max_distance_m: float = 1.25

    # Select one canonical branch inside a duplicate component.
    branch_switch_penalty: float = 0.12
    branch_interpolated_penalty: float = 0.30
    branch_motion_scale_m: float = 0.50

    # Sequential identity linking after duplicate resolution.
    linker_max_gap_s: float = 3.00
    linker_long_gap_requires_velocity_s: float = 0.50
    linker_max_speed_mps: float = 13.0
    linker_base_slack_m: float = 0.75
    linker_prediction_base_m: float = 0.75
    linker_prediction_growth_mps: float = 1.50
    linker_max_velocity_difference_mps: float = 8.0
    linker_gap_cost_weight: float = 0.25
    linker_velocity_cost_weight: float = 0.05
    linker_same_lineage_bonus: float = 0.80

    # Raw observations inside a gap can support a proposed continuation.
    bridge_evidence_radius_m: float = 1.00
    bridge_evidence_stride: int = 2
    bridge_bonus: float = 0.75
    bridge_required_gap_s: float = 0.75
    bridge_min_ratio_without_lineage: float = 0.15

    # Birth/death priors.  These are costs, not classifications.
    event_base_cost: float = 2.50
    event_min_cost: float = 0.40
    boundary_discount: float = 1.00
    boundary_scale_m: float = 2.00
    clip_edge_discount: float = 1.25
    clip_edge_scale_s: float = 0.75

    # Final display.
    render_gap_s: float = 0.50
    max_visible_tracks: int = 25

    def __post_init__(self) -> None:
        if self.duplicate_min_overlap_frames < 1:
            raise ValueError("duplicate_min_overlap_frames must be >= 1")
        if self.duplicate_evidence_stride < 1:
            raise ValueError("duplicate_evidence_stride must be >= 1")
        if self.bridge_evidence_stride < 1:
            raise ValueError("bridge_evidence_stride must be >= 1")
        if self.max_visible_tracks < 1:
            raise ValueError("max_visible_tracks must be >= 1")

        positive = (
            "state_interpolation_gap_s",
            "duplicate_min_overlap_s",
            "duplicate_median_distance_m",
            "duplicate_p90_distance_m",
            "duplicate_max_relative_speed_mps",
            "duplicate_evidence_radius_m",
            "no_lineage_duplicate_min_overlap_s",
            "no_lineage_duplicate_median_distance_m",
            "no_lineage_duplicate_p90_distance_m",
            "conflict_max_distance_m",
            "branch_motion_scale_m",
            "linker_max_gap_s",
            "linker_long_gap_requires_velocity_s",
            "linker_max_speed_mps",
            "linker_base_slack_m",
            "linker_prediction_base_m",
            "linker_prediction_growth_mps",
            "linker_max_velocity_difference_mps",
            "bridge_evidence_radius_m",
            "bridge_required_gap_s",
            "event_base_cost",
            "event_min_cost",
            "boundary_scale_m",
            "clip_edge_scale_s",
            "render_gap_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        bounded = (
            "duplicate_max_independent_evidence_ratio",
            "duplicate_max_both_observed_ratio",
            "no_lineage_duplicate_max_independent_ratio",
            "conflict_independent_evidence_ratio",
            "bridge_min_ratio_without_lineage",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class EvidenceState:
    frame: int
    time_s: float
    x: float
    z: float
    vx: float
    vz: float
    state: str  # observed | interpolated
    source_trajectory_id: int
    support_score: float

    @property
    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.z], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray([self.vx, self.vz], dtype=np.float64)


@dataclass(frozen=True)
class DuplicateEvidence:
    trajectory_a: int
    trajectory_b: int
    shared_source_track_ids: tuple[int, ...]
    overlap_frames: int
    overlap_duration_s: float
    median_distance_m: float
    p90_distance_m: float
    median_relative_speed_mps: float
    both_observed_ratio: float
    informative_evidence_frames: int
    independent_evidence_frames: int
    independent_evidence_ratio: float | None
    duplicate: bool
    duplicate_reason: str | None
    score: float


@dataclass(frozen=True)
class IdentityFragment:
    fragment_id: int
    member_trajectory_ids: tuple[int, ...]
    source_track_ids: tuple[int, ...]
    states: Mapping[int, EvidenceState]
    support_score: float

    @property
    def start_frame(self) -> int:
        return min(self.states)

    @property
    def end_frame(self) -> int:
        return max(self.states)

    @property
    def start_state(self) -> EvidenceState:
        return self.states[self.start_frame]

    @property
    def end_state(self) -> EvidenceState:
        return self.states[self.end_frame]

    @property
    def start_time_s(self) -> float:
        return self.start_state.time_s

    @property
    def end_time_s(self) -> float:
        return self.end_state.time_s


@dataclass(frozen=True)
class IdentityLinkCandidate:
    predecessor_fragment_id: int
    successor_fragment_id: int
    gap_s: float
    direct_distance_m: float
    physical_radius_m: float
    prediction_error_m: float
    prediction_radius_m: float
    velocity_difference_mps: float | None
    bridge_evidence_ratio: float
    shared_source_track_ids: tuple[int, ...]
    birth_cost_successor: float
    death_cost_predecessor: float
    link_cost: float
    savings_vs_birth_death: float


@dataclass(frozen=True)
class GlobalIdentityFrameTrack:
    # Kept as trajectory_id so the existing OfflinePitchRenderer can render V3
    # without any algorithm-specific UI changes.
    trajectory_id: int
    frame: int
    time_s: float
    state: str
    x: float
    z: float
    vx: float
    vz: float
    support_score: float
    member_trajectory_count: int
    source_track_count: int

    @property
    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.z], dtype=np.float64)

    @property
    def velocity(self) -> np.ndarray:
        return np.asarray([self.vx, self.vz], dtype=np.float64)

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx, self.vz)


@dataclass(frozen=True)
class GlobalIdentity:
    identity_id: int
    fragment_ids: tuple[int, ...]
    member_trajectory_ids: tuple[int, ...]
    source_track_ids: tuple[int, ...]
    states: Mapping[int, GlobalIdentityFrameTrack]
    support_score: float
    start_time_s: float
    end_time_s: float
    boundary_birth: bool
    boundary_death: bool


@dataclass(frozen=True)
class GlobalIdentityResult:
    duplicate_evidence: tuple[DuplicateEvidence, ...]
    duplicate_components: tuple[tuple[int, ...], ...]
    blocked_duplicate_merges: tuple[tuple[int, int], ...]
    fragments: tuple[IdentityFragment, ...]
    link_candidates: tuple[IdentityLinkCandidate, ...]
    accepted_links: tuple[tuple[int, int], ...]
    identities: tuple[GlobalIdentity, ...]
    frame_tracks: Mapping[int, tuple[GlobalIdentityFrameTrack, ...]]


def _trajectory_id(trajectory: Any) -> int:
    return int(trajectory.trajectory_id)


def _trajectory_sources(trajectory: Any) -> set[int]:
    return {int(value) for value in trajectory.source_track_ids}


def _trajectory_support(trajectory: Any) -> float:
    observed_count = max(0, int(getattr(trajectory, "observed_count", 0)))
    span = max(0.0, float(getattr(trajectory, "observed_span_s", 0.0)))
    confidence = max(
        0.0,
        min(1.0, float(getattr(trajectory, "mean_confidence", 0.0))),
    )
    fragments = max(1, int(getattr(trajectory, "fragment_count", 1)))

    return float(
        math.log1p(observed_count)
        + 0.45 * math.log1p(span)
        + 0.25 * math.log1p(fragments)
        + 0.50 * confidence
    )


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return (1.0 - alpha) * ordered[lower] + alpha * ordered[upper]


def _hermite(
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    *,
    duration: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(alpha)

    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    position = (
        h00 * p0
        + h10 * duration * v0
        + h01 * p1
        + h11 * duration * v1
    )

    dh00 = 6 * s**2 - 6 * s
    dh10 = 3 * s**2 - 4 * s + 1
    dh01 = -6 * s**2 + 6 * s
    dh11 = 3 * s**2 - 2 * s

    velocity = (
        dh00 * p0
        + dh10 * duration * v0
        + dh01 * p1
        + dh11 * duration * v1
    ) / duration

    return position, velocity


def build_trajectory_state_maps(
    frames: Sequence[Any],
    trajectories: Sequence[Any],
    *,
    config: GlobalIdentityConfig | None = None,
) -> dict[int, dict[int, EvidenceState]]:
    """Expand each V2 trajectory into an uncapped frame-state map.

    Only internal gaps already supported by V2 smoothed samples and shorter
    than ``state_interpolation_gap_s`` are filled.  Nothing is extrapolated
    before a trajectory starts or after it ends.
    """

    cfg = config or GlobalIdentityConfig()
    frame_time = {
        int(frame.frame): float(frame.time_s)
        for frame in frames
    }
    ordered_frame_ids = sorted(frame_time)
    frame_index = {
        frame_id: index
        for index, frame_id in enumerate(ordered_frame_ids)
    }

    result: dict[int, dict[int, EvidenceState]] = {}

    for trajectory in trajectories:
        trajectory_id = _trajectory_id(trajectory)
        support = _trajectory_support(trajectory)
        samples = sorted(
            trajectory.smoothed_samples,
            key=lambda sample: (float(sample.time_s), int(sample.frame)),
        )

        states: dict[int, EvidenceState] = {}

        for sample in samples:
            frame_id = int(sample.frame)
            states[frame_id] = EvidenceState(
                frame=frame_id,
                time_s=float(sample.time_s),
                x=float(sample.x),
                z=float(sample.z),
                vx=float(sample.vx),
                vz=float(sample.vz),
                state="observed",
                source_trajectory_id=trajectory_id,
                support_score=support,
            )

        for left, right in zip(samples, samples[1:]):
            left_frame = int(left.frame)
            right_frame = int(right.frame)

            if left_frame not in frame_index or right_frame not in frame_index:
                continue

            duration = float(right.time_s) - float(left.time_s)
            if (
                duration <= 0.0
                or duration > cfg.state_interpolation_gap_s
            ):
                continue

            left_index = frame_index[left_frame]
            right_index = frame_index[right_frame]
            if right_index <= left_index + 1:
                continue

            p0 = np.asarray([left.x, left.z], dtype=np.float64)
            v0 = np.asarray([left.vx, left.vz], dtype=np.float64)
            p1 = np.asarray([right.x, right.z], dtype=np.float64)
            v1 = np.asarray([right.vx, right.vz], dtype=np.float64)

            for index in range(left_index + 1, right_index):
                frame_id = ordered_frame_ids[index]
                if frame_id in states:
                    continue

                time_s = frame_time[frame_id]
                alpha = (time_s - float(left.time_s)) / duration
                if not 0.0 < alpha < 1.0:
                    continue

                position, velocity = _hermite(
                    p0,
                    v0,
                    p1,
                    v1,
                    duration=duration,
                    alpha=alpha,
                )
                states[frame_id] = EvidenceState(
                    frame=frame_id,
                    time_s=time_s,
                    x=float(position[0]),
                    z=float(position[1]),
                    vx=float(velocity[0]),
                    vz=float(velocity[1]),
                    state="interpolated",
                    source_trajectory_id=trajectory_id,
                    support_score=support,
                )

        result[trajectory_id] = states

    return result


def _raw_positions_by_frame(frames: Sequence[Any]) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for frame in frames:
        positions = [
            np.asarray(observation.world_position, dtype=np.float64)
            for observation in frame.observations
        ]
        if positions:
            result[int(frame.frame)] = np.asarray(positions, dtype=np.float64)
        else:
            result[int(frame.frame)] = np.zeros((0, 2), dtype=np.float64)
    return result


def _distinct_observation_support(
    position_a: np.ndarray,
    position_b: np.ndarray,
    observations: np.ndarray,
    *,
    radius_m: float,
) -> tuple[bool, bool]:
    """Return ``(informative, independent)`` for two nearby identity states.

    Independent evidence means that the *nearest supporting observation* for A
    is different from the nearest supporting observation for B.  This is more
    conservative than asking whether two arbitrary detections happen to fall
    inside a broad radius.

    That distinction matters in football: when two tracks are almost on top of
    each other and a third nearby player exists, the third player's detection
    must not be counted as proof that the two overlapping tracks are different
    people.
    """

    if observations.size == 0:
        return False, False

    distances_a = np.linalg.norm(observations - position_a[None, :], axis=1)
    distances_b = np.linalg.norm(observations - position_b[None, :], axis=1)

    index_a = int(np.argmin(distances_a))
    index_b = int(np.argmin(distances_b))

    supports_a = float(distances_a[index_a]) <= radius_m
    supports_b = float(distances_b[index_b]) <= radius_m

    informative = supports_a or supports_b
    independent = (
        supports_a
        and supports_b
        and index_a != index_b
    )

    return bool(informative), bool(independent)


def evaluate_duplicate_pair(
    trajectory_a: Any,
    trajectory_b: Any,
    states_a: Mapping[int, EvidenceState],
    states_b: Mapping[int, EvidenceState],
    raw_positions: Mapping[int, np.ndarray],
    *,
    config: GlobalIdentityConfig | None = None,
) -> DuplicateEvidence | None:
    cfg = config or GlobalIdentityConfig()

    overlap = sorted(set(states_a).intersection(states_b))
    if len(overlap) < cfg.duplicate_min_overlap_frames:
        return None

    duration = (
        states_a[overlap[-1]].time_s
        - states_a[overlap[0]].time_s
    )
    if duration < cfg.duplicate_min_overlap_s:
        return None

    distances: list[float] = []
    relative_speeds: list[float] = []
    both_observed = 0

    for frame_id in overlap:
        left = states_a[frame_id]
        right = states_b[frame_id]
        distances.append(
            float(np.linalg.norm(left.position - right.position))
        )
        relative_speeds.append(
            float(np.linalg.norm(left.velocity - right.velocity))
        )
        if left.state == "observed" and right.state == "observed":
            both_observed += 1

    median_distance = float(statistics.median(distances))
    p90_distance = float(_percentile(distances, 0.90))
    median_relative_speed = float(statistics.median(relative_speeds))
    both_observed_ratio = both_observed / len(overlap)

    sources_a = _trajectory_sources(trajectory_a)
    sources_b = _trajectory_sources(trajectory_b)
    shared_sources = tuple(sorted(sources_a.intersection(sources_b)))
    shared_lineage = bool(shared_sources)

    # Cheap rejection before inspecting raw observations.
    if shared_lineage:
        if (
            median_distance > cfg.duplicate_median_distance_m
            or p90_distance > cfg.duplicate_p90_distance_m
            or median_relative_speed > cfg.duplicate_max_relative_speed_mps
        ):
            return DuplicateEvidence(
                trajectory_a=_trajectory_id(trajectory_a),
                trajectory_b=_trajectory_id(trajectory_b),
                shared_source_track_ids=shared_sources,
                overlap_frames=len(overlap),
                overlap_duration_s=float(duration),
                median_distance_m=median_distance,
                p90_distance_m=p90_distance,
                median_relative_speed_mps=median_relative_speed,
                both_observed_ratio=both_observed_ratio,
                informative_evidence_frames=0,
                independent_evidence_frames=0,
                independent_evidence_ratio=None,
                duplicate=False,
                duplicate_reason=None,
                score=0.0,
            )
    else:
        if (
            duration < cfg.no_lineage_duplicate_min_overlap_s
            or median_distance > cfg.no_lineage_duplicate_median_distance_m
            or p90_distance > cfg.no_lineage_duplicate_p90_distance_m
            or median_relative_speed
            > cfg.no_lineage_duplicate_max_relative_speed_mps
        ):
            return None

    informative = 0
    independent = 0

    sampled_overlap = overlap[:: cfg.duplicate_evidence_stride]
    for frame_id in sampled_overlap:
        left = states_a[frame_id]
        right = states_b[frame_id]
        is_informative, is_independent = _distinct_observation_support(
            left.position,
            right.position,
            raw_positions.get(
                frame_id,
                np.zeros((0, 2), dtype=np.float64),
            ),
            radius_m=cfg.duplicate_evidence_radius_m,
        )
        if is_informative:
            informative += 1
            if is_independent:
                independent += 1

    independent_ratio = (
        independent / informative
        if informative > 0
        else None
    )

    duplicate = False
    reason = None

    if informative >= cfg.duplicate_min_informative_frames:
        if shared_lineage:
            duplicate = (
                independent_ratio is not None
                and independent_ratio
                <= cfg.duplicate_max_independent_evidence_ratio
                and both_observed_ratio
                <= cfg.duplicate_max_both_observed_ratio
            )
            if duplicate:
                reason = "shared_lineage_low_independent_evidence"
        else:
            duplicate = (
                independent_ratio is not None
                and independent_ratio
                <= cfg.no_lineage_duplicate_max_independent_ratio
                and both_observed_ratio <= 0.10
            )
            if duplicate:
                reason = "extreme_overlap_low_independent_evidence"

    # Purely diagnostic score; it is not a calibrated probability.
    threshold = (
        cfg.duplicate_median_distance_m
        if shared_lineage
        else cfg.no_lineage_duplicate_median_distance_m
    )
    p90_threshold = (
        cfg.duplicate_p90_distance_m
        if shared_lineage
        else cfg.no_lineage_duplicate_p90_distance_m
    )
    evidence_term = (
        0.0
        if independent_ratio is None
        else 1.0 - independent_ratio
    )
    score = (
        (2.0 if shared_lineage else 0.0)
        + max(0.0, 1.0 - median_distance / threshold)
        + max(0.0, 1.0 - p90_distance / p90_threshold)
        + evidence_term
        + min(1.0, duration / 2.0)
        - both_observed_ratio
    )

    return DuplicateEvidence(
        trajectory_a=_trajectory_id(trajectory_a),
        trajectory_b=_trajectory_id(trajectory_b),
        shared_source_track_ids=shared_sources,
        overlap_frames=len(overlap),
        overlap_duration_s=float(duration),
        median_distance_m=median_distance,
        p90_distance_m=p90_distance,
        median_relative_speed_mps=median_relative_speed,
        both_observed_ratio=both_observed_ratio,
        informative_evidence_frames=informative,
        independent_evidence_frames=independent,
        independent_evidence_ratio=independent_ratio,
        duplicate=duplicate,
        duplicate_reason=reason,
        score=float(score),
    )


class _DisjointSet:
    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {int(value): int(value) for value in values}

    def find(self, value: int) -> int:
        value = int(value)
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def members(self, value: int) -> set[int]:
        root = self.find(value)
        return {
            item
            for item in self.parent
            if self.find(item) == root
        }

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def _strong_independent_conflict(
    evidence: DuplicateEvidence | None,
    *,
    config: GlobalIdentityConfig,
) -> bool:
    if evidence is None:
        return False
    if (
        evidence.informative_evidence_frames
        < config.conflict_min_informative_frames
    ):
        return False
    if evidence.independent_evidence_ratio is None:
        return False
    return (
        evidence.independent_evidence_ratio
        >= config.conflict_independent_evidence_ratio
        and evidence.median_distance_m
        <= config.conflict_max_distance_m
        and evidence.overlap_duration_s
        >= config.duplicate_min_overlap_s
    )


def resolve_duplicate_components(
    trajectories: Sequence[Any],
    state_maps: Mapping[int, Mapping[int, EvidenceState]],
    frames: Sequence[Any],
    *,
    config: GlobalIdentityConfig | None = None,
) -> tuple[
    tuple[DuplicateEvidence, ...],
    tuple[tuple[int, ...], ...],
    tuple[tuple[int, int], ...],
]:
    cfg = config or GlobalIdentityConfig()
    trajectory_by_id = {
        _trajectory_id(trajectory): trajectory
        for trajectory in trajectories
    }
    ids = sorted(trajectory_by_id)
    raw_positions = _raw_positions_by_frame(frames)

    evidence_by_pair: dict[tuple[int, int], DuplicateEvidence] = {}
    duplicate_candidates: list[DuplicateEvidence] = []

    for index, left_id in enumerate(ids):
        left = trajectory_by_id[left_id]
        left_states = state_maps.get(left_id, {})

        for right_id in ids[index + 1:]:
            right = trajectory_by_id[right_id]
            right_states = state_maps.get(right_id, {})

            evidence = evaluate_duplicate_pair(
                left,
                right,
                left_states,
                right_states,
                raw_positions,
                config=cfg,
            )
            if evidence is None:
                continue

            key = (left_id, right_id)
            evidence_by_pair[key] = evidence
            if evidence.duplicate:
                duplicate_candidates.append(evidence)

    duplicate_candidates.sort(
        key=lambda item: (
            -item.score,
            item.trajectory_a,
            item.trajectory_b,
        )
    )

    dsu = _DisjointSet(ids)
    blocked: list[tuple[int, int]] = []

    for candidate in duplicate_candidates:
        left_root = dsu.find(candidate.trajectory_a)
        right_root = dsu.find(candidate.trajectory_b)

        if left_root == right_root:
            continue

        left_members = dsu.members(left_root)
        right_members = dsu.members(right_root)

        conflict = False
        for left_id in left_members:
            for right_id in right_members:
                key = (
                    min(left_id, right_id),
                    max(left_id, right_id),
                )
                evidence = evidence_by_pair.get(key)
                if _strong_independent_conflict(evidence, config=cfg):
                    conflict = True
                    break
            if conflict:
                break

        if conflict:
            blocked.append(
                (
                    candidate.trajectory_a,
                    candidate.trajectory_b,
                )
            )
            continue

        dsu.union(
            candidate.trajectory_a,
            candidate.trajectory_b,
        )

    components_by_root: dict[int, list[int]] = {}
    for trajectory_id in ids:
        root = dsu.find(trajectory_id)
        components_by_root.setdefault(root, []).append(trajectory_id)

    components = tuple(
        tuple(sorted(values))
        for _, values in sorted(
            components_by_root.items(),
            key=lambda item: min(item[1]),
        )
    )

    evidence_values = tuple(
        evidence_by_pair[key]
        for key in sorted(evidence_by_pair)
    )

    return evidence_values, components, tuple(blocked)


def _candidate_emission_cost(
    state: EvidenceState,
    *,
    config: GlobalIdentityConfig,
) -> float:
    state_penalty = (
        0.0
        if state.state == "observed"
        else config.branch_interpolated_penalty
    )
    return (
        state_penalty
        - 0.05 * state.support_score
    )


def _select_component_states(
    member_ids: Sequence[int],
    state_maps: Mapping[int, Mapping[int, EvidenceState]],
    *,
    config: GlobalIdentityConfig,
) -> dict[int, EvidenceState]:
    """Viterbi branch selection inside one duplicate component."""

    frame_candidates: dict[int, list[EvidenceState]] = {}

    for member_id in member_ids:
        for frame_id, state in state_maps.get(member_id, {}).items():
            frame_candidates.setdefault(frame_id, []).append(state)

    ordered_frames = sorted(frame_candidates)
    if not ordered_frames:
        return {}

    selected: dict[int, EvidenceState] = {}

    segment: list[int] = []

    def solve_segment(segment_frames: list[int]) -> None:
        if not segment_frames:
            return

        costs: list[dict[int, float]] = []
        back: list[dict[int, int | None]] = []

        first_frame = segment_frames[0]
        first_candidates = frame_candidates[first_frame]

        first_costs = {
            index: _candidate_emission_cost(
                state,
                config=config,
            )
            for index, state in enumerate(first_candidates)
        }
        costs.append(first_costs)
        back.append({index: None for index in first_costs})

        for local_index in range(1, len(segment_frames)):
            frame_id = segment_frames[local_index]
            previous_frame_id = segment_frames[local_index - 1]
            candidates = frame_candidates[frame_id]
            previous_candidates = frame_candidates[previous_frame_id]

            current_costs: dict[int, float] = {}
            current_back: dict[int, int | None] = {}

            for current_index, current in enumerate(candidates):
                best_cost = math.inf
                best_previous = None

                for previous_index, previous in enumerate(previous_candidates):
                    previous_total = costs[local_index - 1][previous_index]
                    dt = current.time_s - previous.time_s
                    if dt <= 0.0:
                        continue

                    expected = previous.position + previous.velocity * dt
                    motion_error = float(
                        np.linalg.norm(current.position - expected)
                    )
                    transition_scale = (
                        config.branch_motion_scale_m
                        + 2.0 * dt
                    )
                    transition_cost = motion_error / transition_scale

                    if (
                        previous.source_trajectory_id
                        != current.source_trajectory_id
                    ):
                        transition_cost += config.branch_switch_penalty

                    total = (
                        previous_total
                        + transition_cost
                        + _candidate_emission_cost(
                            current,
                            config=config,
                        )
                    )

                    if total < best_cost:
                        best_cost = total
                        best_previous = previous_index

                current_costs[current_index] = best_cost
                current_back[current_index] = best_previous

            costs.append(current_costs)
            back.append(current_back)

        last_index = min(
            costs[-1],
            key=lambda candidate_index: costs[-1][candidate_index],
        )

        chosen_indices = [last_index]
        for local_index in range(len(segment_frames) - 1, 0, -1):
            previous_index = back[local_index][chosen_indices[-1]]
            if previous_index is None:
                previous_index = min(
                    costs[local_index - 1],
                    key=lambda candidate_index: (
                        costs[local_index - 1][candidate_index]
                    ),
                )
            chosen_indices.append(previous_index)

        chosen_indices.reverse()

        for frame_id, candidate_index in zip(
            segment_frames,
            chosen_indices,
        ):
            selected[frame_id] = frame_candidates[frame_id][candidate_index]

    previous_time = None
    for frame_id in ordered_frames:
        current_time = frame_candidates[frame_id][0].time_s

        if (
            previous_time is not None
            and current_time - previous_time
            > config.state_interpolation_gap_s * 1.5
        ):
            solve_segment(segment)
            segment = []

        segment.append(frame_id)
        previous_time = current_time

    solve_segment(segment)
    return selected


def build_identity_fragments(
    trajectories: Sequence[Any],
    components: Sequence[Sequence[int]],
    state_maps: Mapping[int, Mapping[int, EvidenceState]],
    *,
    config: GlobalIdentityConfig | None = None,
) -> tuple[IdentityFragment, ...]:
    cfg = config or GlobalIdentityConfig()

    trajectory_by_id = {
        _trajectory_id(trajectory): trajectory
        for trajectory in trajectories
    }

    fragments: list[IdentityFragment] = []

    for fragment_id, members in enumerate(components, start=1):
        states = _select_component_states(
            members,
            state_maps,
            config=cfg,
        )
        if not states:
            continue

        source_track_ids = sorted(
            {
                source_id
                for member_id in members
                for source_id in _trajectory_sources(
                    trajectory_by_id[member_id]
                )
            }
        )
        support_score = max(
            _trajectory_support(trajectory_by_id[member_id])
            for member_id in members
        )

        fragments.append(
            IdentityFragment(
                fragment_id=fragment_id,
                member_trajectory_ids=tuple(sorted(members)),
                source_track_ids=tuple(source_track_ids),
                states=states,
                support_score=float(support_score),
            )
        )

    return tuple(fragments)


def _robust_endpoint_velocity(
    fragment: IdentityFragment,
    *,
    at_start: bool,
    window_s: float = 0.50,
) -> np.ndarray | None:
    states = [
        fragment.states[frame_id]
        for frame_id in sorted(fragment.states)
    ]
    anchor_time = (
        states[0].time_s
        if at_start
        else states[-1].time_s
    )

    selected = [
        state
        for state in states
        if (
            state.time_s - anchor_time <= window_s
            if at_start
            else anchor_time - state.time_s <= window_s
        )
    ]

    observed = [
        state for state in selected
        if state.state == "observed"
    ]
    if len(observed) >= 3:
        selected = observed

    if len(selected) < 2:
        return None

    velocities = np.asarray(
        [[state.vx, state.vz] for state in selected],
        dtype=np.float64,
    )
    velocity = np.median(velocities, axis=0)
    if not np.isfinite(velocity).all():
        return None
    return velocity


def _frame_lookup(frames: Sequence[Any]) -> tuple[list[int], dict[int, Any]]:
    ordered = sorted(int(frame.frame) for frame in frames)
    return ordered, {int(frame.frame): frame for frame in frames}


def _bridge_evidence_ratio(
    predecessor: IdentityFragment,
    successor: IdentityFragment,
    frames: Sequence[Any],
    *,
    start_velocity: np.ndarray | None,
    end_velocity: np.ndarray | None,
    config: GlobalIdentityConfig,
) -> float:
    ordered_frames, frame_by_id = _frame_lookup(frames)

    left_frame = predecessor.end_frame
    right_frame = successor.start_frame

    try:
        left_index = ordered_frames.index(left_frame)
        right_index = ordered_frames.index(right_frame)
    except ValueError:
        return 0.0

    if right_index <= left_index + 1:
        return 0.0

    p0 = predecessor.end_state.position
    p1 = successor.start_state.position
    duration = successor.start_time_s - predecessor.end_time_s

    if duration <= 0.0:
        return 0.0

    sampled = 0
    supported = 0

    middle_frames = ordered_frames[
        left_index + 1 : right_index : config.bridge_evidence_stride
    ]

    for frame_id in middle_frames:
        frame = frame_by_id[frame_id]
        alpha = (
            float(frame.time_s) - predecessor.end_time_s
        ) / duration

        if not 0.0 < alpha < 1.0:
            continue

        if start_velocity is not None and end_velocity is not None:
            expected, _ = _hermite(
                p0,
                start_velocity,
                p1,
                end_velocity,
                duration=duration,
                alpha=alpha,
            )
        else:
            expected = (1.0 - alpha) * p0 + alpha * p1

        sampled += 1

        observations = [
            np.asarray(observation.world_position, dtype=np.float64)
            for observation in frame.observations
        ]
        if not observations:
            continue

        distances = np.linalg.norm(
            np.asarray(observations) - expected[None, :],
            axis=1,
        )
        if bool(np.any(distances <= config.bridge_evidence_radius_m)):
            supported += 1

    if sampled == 0:
        return 0.0
    return supported / sampled


def boundary_distance_m(
    position: np.ndarray,
    *,
    field_length_m: float | None,
    field_width_m: float | None,
) -> float | None:
    if field_length_m is None or field_width_m is None:
        return None

    x = float(position[0])
    z = float(position[1])

    dx = field_length_m / 2.0 - abs(x)
    dz = field_width_m / 2.0 - abs(z)
    return max(0.0, min(dx, dz))


def event_cost(
    *,
    position: np.ndarray,
    time_s: float,
    clip_edge_time_s: float,
    field_length_m: float | None,
    field_width_m: float | None,
    support_score: float,
    config: GlobalIdentityConfig | None = None,
) -> float:
    """Birth/death cost used by the global path cover.

    ``clip_edge_time_s`` is the clip start for a birth and clip end for a death.
    The absolute temporal distance is used, so the same function serves both.
    """

    cfg = config or GlobalIdentityConfig()
    cost = cfg.event_base_cost

    boundary_distance = boundary_distance_m(
        position,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
    )
    if boundary_distance is not None:
        cost -= (
            cfg.boundary_discount
            * math.exp(
                -boundary_distance / cfg.boundary_scale_m
            )
        )

    temporal_distance = abs(float(time_s) - float(clip_edge_time_s))
    cost -= (
        cfg.clip_edge_discount
        * math.exp(
            -temporal_distance / cfg.clip_edge_scale_s
        )
    )

    # Short/weak fragments are easier to leave as independent births/deaths.
    support_factor = min(
        1.0,
        max(0.0, support_score / 8.0),
    )
    cost *= 0.72 + 0.28 * support_factor

    return max(cfg.event_min_cost, float(cost))


def evaluate_identity_link(
    predecessor: IdentityFragment,
    successor: IdentityFragment,
    frames: Sequence[Any],
    *,
    clip_start_s: float,
    clip_end_s: float,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityConfig | None = None,
) -> IdentityLinkCandidate | None:
    cfg = config or GlobalIdentityConfig()

    if successor.start_time_s <= predecessor.end_time_s:
        return None

    gap_s = successor.start_time_s - predecessor.end_time_s
    if gap_s > cfg.linker_max_gap_s:
        return None

    direct_distance = float(
        np.linalg.norm(
            successor.start_state.position
            - predecessor.end_state.position
        )
    )
    physical_radius = (
        cfg.linker_max_speed_mps * gap_s
        + cfg.linker_base_slack_m
    )
    if direct_distance > physical_radius:
        return None

    predecessor_velocity = _robust_endpoint_velocity(
        predecessor,
        at_start=False,
    )
    successor_velocity = _robust_endpoint_velocity(
        successor,
        at_start=True,
    )

    if (
        gap_s > cfg.linker_long_gap_requires_velocity_s
        and (
            predecessor_velocity is None
            or successor_velocity is None
        )
    ):
        return None

    errors: list[float] = []

    if predecessor_velocity is not None:
        forward = (
            predecessor.end_state.position
            + predecessor_velocity * gap_s
        )
        errors.append(
            float(
                np.linalg.norm(
                    successor.start_state.position - forward
                )
            )
        )

    if successor_velocity is not None:
        backward = (
            successor.start_state.position
            - successor_velocity * gap_s
        )
        errors.append(
            float(
                np.linalg.norm(
                    predecessor.end_state.position - backward
                )
            )
        )

    prediction_error = (
        float(statistics.fmean(errors))
        if errors
        else direct_distance
    )
    prediction_radius = (
        cfg.linker_prediction_base_m
        + cfg.linker_prediction_growth_mps * gap_s
    )
    if prediction_error > prediction_radius:
        return None

    velocity_difference = None
    if (
        predecessor_velocity is not None
        and successor_velocity is not None
    ):
        velocity_difference = float(
            np.linalg.norm(
                predecessor_velocity - successor_velocity
            )
        )
        if velocity_difference > cfg.linker_max_velocity_difference_mps:
            return None

    shared_sources = tuple(
        sorted(
            set(predecessor.source_track_ids)
            .intersection(successor.source_track_ids)
        )
    )

    bridge_ratio = _bridge_evidence_ratio(
        predecessor,
        successor,
        frames,
        start_velocity=predecessor_velocity,
        end_velocity=successor_velocity,
        config=cfg,
    )

    if (
        gap_s > cfg.bridge_required_gap_s
        and not shared_sources
        and bridge_ratio < cfg.bridge_min_ratio_without_lineage
    ):
        return None

    link_cost = (
        prediction_error / max(0.25, prediction_radius)
        + cfg.linker_gap_cost_weight * gap_s
        + (
            cfg.linker_velocity_cost_weight * velocity_difference
            if velocity_difference is not None
            else 0.0
        )
        - cfg.bridge_bonus * bridge_ratio
        - (
            cfg.linker_same_lineage_bonus
            if shared_sources
            else 0.0
        )
    )
    link_cost = max(0.0, float(link_cost))

    birth_cost = event_cost(
        position=successor.start_state.position,
        time_s=successor.start_time_s,
        clip_edge_time_s=clip_start_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        support_score=successor.support_score,
        config=cfg,
    )
    death_cost = event_cost(
        position=predecessor.end_state.position,
        time_s=predecessor.end_time_s,
        clip_edge_time_s=clip_end_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        support_score=predecessor.support_score,
        config=cfg,
    )

    savings = birth_cost + death_cost - link_cost
    if savings <= 0.0:
        return None

    return IdentityLinkCandidate(
        predecessor_fragment_id=predecessor.fragment_id,
        successor_fragment_id=successor.fragment_id,
        gap_s=float(gap_s),
        direct_distance_m=direct_distance,
        physical_radius_m=float(physical_radius),
        prediction_error_m=prediction_error,
        prediction_radius_m=float(prediction_radius),
        velocity_difference_mps=velocity_difference,
        bridge_evidence_ratio=float(bridge_ratio),
        shared_source_track_ids=shared_sources,
        birth_cost_successor=float(birth_cost),
        death_cost_predecessor=float(death_cost),
        link_cost=link_cost,
        savings_vs_birth_death=float(savings),
    )


def solve_fragment_path_cover(
    fragments: Sequence[IdentityFragment],
    frames: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityConfig | None = None,
) -> tuple[
    tuple[IdentityLinkCandidate, ...],
    tuple[tuple[int, int], ...],
]:
    cfg = config or GlobalIdentityConfig()
    fragments = tuple(fragments)
    count = len(fragments)

    if count == 0:
        return (), ()

    clip_start_s = float(frames[0].time_s)
    clip_end_s = float(frames[-1].time_s)

    matrix = np.full(
        (count, count + count),
        _FORBIDDEN,
        dtype=np.float64,
    )
    candidate_by_pair: dict[
        tuple[int, int],
        IdentityLinkCandidate,
    ] = {}

    for row, predecessor in enumerate(fragments):
        death_cost = event_cost(
            position=predecessor.end_state.position,
            time_s=predecessor.end_time_s,
            clip_edge_time_s=clip_end_s,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
            support_score=predecessor.support_score,
            config=cfg,
        )
        matrix[row, count + row] = death_cost

        for column, successor in enumerate(fragments):
            if row == column:
                continue

            candidate = evaluate_identity_link(
                predecessor,
                successor,
                frames,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                field_length_m=field_length_m,
                field_width_m=field_width_m,
                config=cfg,
            )
            if candidate is None:
                continue

            # Constant sum of birth costs is omitted from the assignment
            # objective.  Subtracting successor birth cost makes:
            #
            # link selected iff link_cost < predecessor_death + successor_birth
            adjusted_cost = (
                candidate.link_cost
                - candidate.birth_cost_successor
            )
            matrix[row, column] = adjusted_cost
            candidate_by_pair[(row, column)] = candidate

    rows, columns = linear_sum_assignment(matrix)

    accepted: list[tuple[int, int]] = []

    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= count:
            continue

        candidate = candidate_by_pair.get((row, column))
        if candidate is None:
            continue

        accepted.append(
            (
                candidate.predecessor_fragment_id,
                candidate.successor_fragment_id,
            )
        )

    accepted.sort()

    candidates = tuple(
        sorted(
            candidate_by_pair.values(),
            key=lambda item: (
                -item.savings_vs_birth_death,
                item.predecessor_fragment_id,
                item.successor_fragment_id,
            ),
        )
    )

    return candidates, tuple(accepted)


def _fragment_chains(
    fragments: Sequence[IdentityFragment],
    links: Sequence[tuple[int, int]],
) -> list[list[IdentityFragment]]:
    by_id = {
        fragment.fragment_id: fragment
        for fragment in fragments
    }
    successor = {left: right for left, right in links}
    predecessor = {right: left for left, right in links}

    starts = [
        fragment.fragment_id
        for fragment in fragments
        if fragment.fragment_id not in predecessor
    ]

    chains: list[list[IdentityFragment]] = []
    visited: set[int] = set()

    for start in sorted(
        starts,
        key=lambda fragment_id: (
            by_id[fragment_id].start_time_s,
            fragment_id,
        ),
    ):
        chain: list[IdentityFragment] = []
        current = start

        while current not in visited:
            visited.add(current)
            chain.append(by_id[current])

            if current not in successor:
                break
            current = successor[current]

        chains.append(chain)

    for fragment in fragments:
        if fragment.fragment_id not in visited:
            chains.append([fragment])

    return chains


def _interpolate_identity_gap(
    left: EvidenceState,
    right: EvidenceState,
    frames: Sequence[Any],
    *,
    config: GlobalIdentityConfig,
    identity_id: int,
    support_score: float,
    member_count: int,
    source_track_count: int,
) -> dict[int, GlobalIdentityFrameTrack]:
    duration = right.time_s - left.time_s
    if duration <= 0.0 or duration > config.render_gap_s:
        return {}

    frame_by_id = {
        int(frame.frame): frame
        for frame in frames
        if left.frame < int(frame.frame) < right.frame
    }

    result: dict[int, GlobalIdentityFrameTrack] = {}

    for frame_id in sorted(frame_by_id):
        frame = frame_by_id[frame_id]
        alpha = (
            float(frame.time_s) - left.time_s
        ) / duration

        if not 0.0 < alpha < 1.0:
            continue

        position, velocity = _hermite(
            left.position,
            left.velocity,
            right.position,
            right.velocity,
            duration=duration,
            alpha=alpha,
        )

        result[frame_id] = GlobalIdentityFrameTrack(
            trajectory_id=identity_id,
            frame=frame_id,
            time_s=float(frame.time_s),
            state="interpolated",
            x=float(position[0]),
            z=float(position[1]),
            vx=float(velocity[0]),
            vz=float(velocity[1]),
            support_score=float(support_score),
            member_trajectory_count=member_count,
            source_track_count=source_track_count,
        )

    return result


def build_global_identities(
    fragments: Sequence[IdentityFragment],
    links: Sequence[tuple[int, int]],
    frames: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityConfig | None = None,
) -> tuple[GlobalIdentity, ...]:
    cfg = config or GlobalIdentityConfig()
    chains = _fragment_chains(fragments, links)

    identities: list[GlobalIdentity] = []

    for identity_id, chain in enumerate(chains, start=1):
        member_trajectories = sorted(
            {
                trajectory_id
                for fragment in chain
                for trajectory_id in fragment.member_trajectory_ids
            }
        )
        source_track_ids = sorted(
            {
                source_id
                for fragment in chain
                for source_id in fragment.source_track_ids
            }
        )
        support = max(fragment.support_score for fragment in chain)

        states: dict[int, GlobalIdentityFrameTrack] = {}

        for fragment in chain:
            for frame_id, state in fragment.states.items():
                states[frame_id] = GlobalIdentityFrameTrack(
                    trajectory_id=identity_id,
                    frame=frame_id,
                    time_s=state.time_s,
                    state=state.state,
                    x=state.x,
                    z=state.z,
                    vx=state.vx,
                    vz=state.vz,
                    support_score=float(
                        support
                        + (0.50 if state.state == "observed" else 0.10)
                    ),
                    member_trajectory_count=len(member_trajectories),
                    source_track_count=len(source_track_ids),
                )

        for left_fragment, right_fragment in zip(chain, chain[1:]):
            if (
                right_fragment.start_time_s
                - left_fragment.end_time_s
                <= cfg.render_gap_s
            ):
                gap_states = _interpolate_identity_gap(
                    left_fragment.end_state,
                    right_fragment.start_state,
                    frames,
                    config=cfg,
                    identity_id=identity_id,
                    support_score=support + 0.10,
                    member_count=len(member_trajectories),
                    source_track_count=len(source_track_ids),
                )
                for frame_id, state in gap_states.items():
                    states.setdefault(frame_id, state)

        if not states:
            continue

        first = states[min(states)]
        last = states[max(states)]

        birth_distance = boundary_distance_m(
            first.position,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )
        death_distance = boundary_distance_m(
            last.position,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )

        identities.append(
            GlobalIdentity(
                identity_id=identity_id,
                fragment_ids=tuple(
                    fragment.fragment_id
                    for fragment in chain
                ),
                member_trajectory_ids=tuple(member_trajectories),
                source_track_ids=tuple(source_track_ids),
                states=states,
                support_score=float(support),
                start_time_s=float(first.time_s),
                end_time_s=float(last.time_s),
                boundary_birth=(
                    birth_distance is not None
                    and birth_distance <= 2.0
                ),
                boundary_death=(
                    death_distance is not None
                    and death_distance <= 2.0
                ),
            )
        )

    return tuple(identities)


def build_frame_tracks(
    frames: Sequence[Any],
    identities: Sequence[GlobalIdentity],
    *,
    config: GlobalIdentityConfig | None = None,
) -> dict[int, tuple[GlobalIdentityFrameTrack, ...]]:
    cfg = config or GlobalIdentityConfig()
    result: dict[int, tuple[GlobalIdentityFrameTrack, ...]] = {}

    for frame in frames:
        frame_id = int(frame.frame)
        candidates = [
            identity.states[frame_id]
            for identity in identities
            if frame_id in identity.states
        ]

        candidates.sort(
            key=lambda item: (
                -(
                    item.support_score
                    + (0.50 if item.state == "observed" else 0.0)
                ),
                item.trajectory_id,
            )
        )

        # Cap only.  Never pad to 25.
        result[frame_id] = tuple(
            candidates[: cfg.max_visible_tracks]
        )

    return result


def solve_global_identities(
    frames: Sequence[Any],
    trajectories: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityConfig | None = None,
) -> GlobalIdentityResult:
    cfg = config or GlobalIdentityConfig()

    state_maps = build_trajectory_state_maps(
        frames,
        trajectories,
        config=cfg,
    )

    duplicate_evidence, components, blocked = resolve_duplicate_components(
        trajectories,
        state_maps,
        frames,
        config=cfg,
    )

    fragments = build_identity_fragments(
        trajectories,
        components,
        state_maps,
        config=cfg,
    )

    link_candidates, accepted_links = solve_fragment_path_cover(
        fragments,
        frames,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=cfg,
    )

    identities = build_global_identities(
        fragments,
        accepted_links,
        frames,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=cfg,
    )

    frame_tracks = build_frame_tracks(
        frames,
        identities,
        config=cfg,
    )

    return GlobalIdentityResult(
        duplicate_evidence=duplicate_evidence,
        duplicate_components=components,
        blocked_duplicate_merges=blocked,
        fragments=fragments,
        link_candidates=link_candidates,
        accepted_links=accepted_links,
        identities=identities,
        frame_tracks=frame_tracks,
    )


__all__ = [
    "DuplicateEvidence",
    "EvidenceState",
    "GlobalIdentity",
    "GlobalIdentityConfig",
    "GlobalIdentityFrameTrack",
    "GlobalIdentityResult",
    "IdentityFragment",
    "IdentityLinkCandidate",
    "boundary_distance_m",
    "build_frame_tracks",
    "build_global_identities",
    "build_identity_fragments",
    "build_trajectory_state_maps",
    "event_cost",
    "evaluate_duplicate_pair",
    "evaluate_identity_link",
    "resolve_duplicate_components",
    "solve_fragment_path_cover",
    "solve_global_identities",
]
