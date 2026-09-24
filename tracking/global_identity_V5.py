"""VeoMatchLab clip-global identity solver V5.

V5 is an offline, full-clip identity resolver built on top of the validated
Offline Context V2 proposals and V3/V4 duplicate-resolution primitives.

Core ideas:
- local proposals are generated from STRONG detections only;
- WEAK detections can support continuity, but can never create an identity;
- identity continuation is solved over the full clip;
- short gaps (<= 4 s) are treated as normal football occlusion/dropout;
- medium gaps (4-10 s) require substantially stronger evidence;
- proximity alone never merges two footballers;
- final analytical trajectories preserve provenance:
  strong_observed / strong_bridge / weak_bridge / inferred.

This is intentionally versioned. It does not replace V3/V4.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field, replace
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import min_weight_full_bipartite_matching

from tracking import global_identity as v3
from tracking import global_identity_V4 as v4


_FORBIDDEN = 1e12


@dataclass(frozen=True)
class GlobalIdentityV5Config:
    base: v3.GlobalIdentityConfig = field(default_factory=v3.GlobalIdentityConfig)

    weak_threshold: float = 0.20
    strong_threshold: float = 0.40

    # Fragment role thresholds.
    core_min_observed_frames: int = 18
    core_min_observed_span_s: float = 0.55
    bridge_candidate_min_observed_frames: int = 6
    bridge_candidate_min_observed_span_s: float = 0.15

    # Time regimes.
    short_gap_max_s: float = 4.0
    medium_gap_max_s: float = 10.0
    overlap_max_s: float = 0.50

    # Time-indexed sparse candidate graph.
    candidate_top_k_outgoing: int = 8
    candidate_top_k_incoming: int = 8

    # Endpoint/motion compatibility.
    endpoint_velocity_window_s: float = 0.75
    short_max_speed_mps: float = 13.0
    medium_max_speed_mps: float = 11.5
    physical_slack_m: float = 1.00

    short_prediction_base_m: float = 1.50
    short_prediction_growth_mps: float = 3.00
    medium_prediction_base_m: float = 1.50
    medium_prediction_growth_mps: float = 2.50

    short_max_velocity_difference_mps: float = 14.0
    medium_max_velocity_difference_mps: float = 10.0

    # Medium gaps need real context, not just proximity.
    medium_min_context_span_s: float = 2.0
    medium_bridge_min_ratio: float = 0.08
    medium_bridge_min_supported_frames: int = 5

    # Bridge evidence from raw STRONG + WEAK observations.
    bridge_evidence_radius_m: float = 1.25
    bridge_evidence_stride: int = 3

    strong_bridge_bonus: float = 1.00
    weak_bridge_bonus: float = 0.45
    same_lineage_bonus: float = 0.80

    # Cost shaping.
    short_gap_cost_weight: float = 0.10
    medium_gap_cost_weight: float = 0.18
    velocity_cost_weight: float = 0.035
    medium_gap_fixed_penalty: float = 0.60
    context_bonus_scale: float = 0.12
    context_bonus_max: float = 0.85

    # Overlap handoff safety.
    overlap_evidence_radius_m: float = 0.90
    overlap_max_independent_ratio: float = 0.20
    overlap_max_median_distance_m: float = 0.90

    # Analytical reconstruction.
    analytical_fill_max_gap_s: float = 4.0
    bridge_support_radius_m: float = 1.25
    interpolation_velocity_cap_mps: float = 12.0
    interpolation_speed_limit_mps: float = 14.5
    pitch_clip_margin_m: float = 0.0

    expected_on_field: int = 23

    def __post_init__(self) -> None:
        if not 0.0 <= self.weak_threshold <= self.strong_threshold <= 1.0:
            raise ValueError(
                "thresholds must satisfy 0 <= weak_threshold <= strong_threshold <= 1"
            )

        integer_positive = (
            "core_min_observed_frames",
            "bridge_candidate_min_observed_frames",
            "candidate_top_k_outgoing",
            "candidate_top_k_incoming",
            "medium_bridge_min_supported_frames",
            "bridge_evidence_stride",
            "expected_on_field",
        )
        for name in integer_positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")

        positive = (
            "core_min_observed_span_s",
            "bridge_candidate_min_observed_span_s",
            "short_gap_max_s",
            "medium_gap_max_s",
            "overlap_max_s",
            "endpoint_velocity_window_s",
            "short_max_speed_mps",
            "medium_max_speed_mps",
            "physical_slack_m",
            "short_prediction_base_m",
            "short_prediction_growth_mps",
            "medium_prediction_base_m",
            "medium_prediction_growth_mps",
            "short_max_velocity_difference_mps",
            "medium_max_velocity_difference_mps",
            "medium_min_context_span_s",
            "bridge_evidence_radius_m",
            "overlap_evidence_radius_m",
            "overlap_max_median_distance_m",
            "analytical_fill_max_gap_s",
            "bridge_support_radius_m",
            "interpolation_velocity_cap_mps",
            "interpolation_speed_limit_mps",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        if self.medium_gap_max_s <= self.short_gap_max_s:
            raise ValueError("medium_gap_max_s must exceed short_gap_max_s")

        if not 0.0 <= self.medium_bridge_min_ratio <= 1.0:
            raise ValueError("medium_bridge_min_ratio must be in [0,1]")
        if not 0.0 <= self.overlap_max_independent_ratio <= 1.0:
            raise ValueError("overlap_max_independent_ratio must be in [0,1]")


@dataclass(frozen=True)
class BridgeEvidence:
    sampled_frames: int
    strong_supported_frames: int
    weak_supported_frames: int
    any_supported_frames: int

    @property
    def strong_ratio(self) -> float:
        return (
            self.strong_supported_frames / self.sampled_frames
            if self.sampled_frames
            else 0.0
        )

    @property
    def weak_ratio(self) -> float:
        return (
            self.weak_supported_frames / self.sampled_frames
            if self.sampled_frames
            else 0.0
        )

    @property
    def any_ratio(self) -> float:
        return (
            self.any_supported_frames / self.sampled_frames
            if self.sampled_frames
            else 0.0
        )


@dataclass(frozen=True)
class IdentityLinkCandidateV5:
    predecessor_fragment_id: int
    successor_fragment_id: int
    tier: str  # overlap | short | medium

    observed_gap_s: float
    observed_overlap_s: float

    direct_distance_m: float
    physical_radius_m: float
    prediction_error_m: float
    prediction_radius_m: float
    velocity_difference_mps: float | None

    bridge_sampled_frames: int
    bridge_strong_supported_frames: int
    bridge_weak_supported_frames: int
    bridge_any_supported_frames: int
    bridge_strong_ratio: float
    bridge_weak_ratio: float
    bridge_any_ratio: float

    overlap_independent_evidence_ratio: float | None
    overlap_informative_frames: int

    shared_source_track_ids: tuple[int, ...]
    predecessor_role: str
    successor_role: str

    predecessor_observed_span_s: float
    successor_observed_span_s: float
    context_bonus: float

    birth_cost_successor: float
    death_cost_predecessor: float
    link_cost: float
    savings_vs_birth_death: float


@dataclass(frozen=True)
class AnalyticalFrameTrack:
    identity_id: int
    frame: int
    time_s: float
    evidence: str  # strong_observed | strong_bridge | weak_bridge | inferred
    x: float
    z: float
    vx: float
    vz: float
    support_score: float
    member_trajectory_count: int
    source_track_count: int
    supporting_confidence: float | None = None
    supporting_detection_index: int | None = None

    @property
    def trajectory_id(self) -> int:
        # Compatibility convenience for downstream helpers/renderers.
        return self.identity_id

    @property
    def state(self) -> str:
        # Compatibility convenience. "observed" only means identity-anchored
        # STRONG evidence; all other states are reconstructed/supported.
        return "observed" if self.evidence == "strong_observed" else "interpolated"

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
class GlobalIdentityV5:
    identity_id: int
    fragment_ids: tuple[int, ...]
    member_trajectory_ids: tuple[int, ...]
    source_track_ids: tuple[int, ...]
    states: Mapping[int, AnalyticalFrameTrack]
    support_score: float
    observed_start_time_s: float
    observed_end_time_s: float
    boundary_birth: bool
    boundary_death: bool
    core_fragment_count: int
    bridge_fragment_count: int


@dataclass(frozen=True)
class GlobalIdentityV5Result:
    duplicate_evidence: tuple[v3.DuplicateEvidence, ...]
    duplicate_components: tuple[tuple[int, ...], ...]
    blocked_duplicate_merges: tuple[tuple[int, int], ...]
    fragments: tuple[v4.IdentityFragmentV4, ...]
    raw_candidate_count: int
    link_candidates: tuple[IdentityLinkCandidateV5, ...]
    accepted_links: tuple[tuple[int, int], ...]
    identities: tuple[GlobalIdentityV5, ...]
    frame_tracks: Mapping[int, tuple[AnalyticalFrameTrack, ...]]


@dataclass(frozen=True)
class _FrameEvidenceIndex:
    ordered_frames: tuple[int, ...]
    frame_by_id: Mapping[int, Any]
    frame_index: Mapping[int, int]
    strong_positions: Mapping[int, np.ndarray]
    weak_positions: Mapping[int, np.ndarray]
    strong_observations: Mapping[int, tuple[Any, ...]]
    weak_observations: Mapping[int, tuple[Any, ...]]


def make_config(
    *,
    weak_threshold: float = 0.20,
    strong_threshold: float = 0.40,
) -> GlobalIdentityV5Config:
    base = replace(
        v3.GlobalIdentityConfig(),
        linker_max_gap_s=10.0,
        render_gap_s=4.0,
    )
    return GlobalIdentityV5Config(
        base=base,
        weak_threshold=float(weak_threshold),
        strong_threshold=float(strong_threshold),
    )


def _evidence_index(
    frames: Sequence[Any],
    *,
    config: GlobalIdentityV5Config,
) -> _FrameEvidenceIndex:
    ordered = tuple(sorted(int(frame.frame) for frame in frames))
    frame_by_id = {int(frame.frame): frame for frame in frames}
    frame_index = {frame_id: index for index, frame_id in enumerate(ordered)}

    strong_positions: dict[int, np.ndarray] = {}
    weak_positions: dict[int, np.ndarray] = {}
    strong_observations: dict[int, tuple[Any, ...]] = {}
    weak_observations: dict[int, tuple[Any, ...]] = {}

    for frame_id in ordered:
        frame = frame_by_id[frame_id]
        strong = tuple(
            obs
            for obs in frame.observations
            if float(obs.confidence) >= config.strong_threshold
        )
        weak = tuple(
            obs
            for obs in frame.observations
            if config.weak_threshold
            <= float(obs.confidence)
            < config.strong_threshold
        )
        strong_observations[frame_id] = strong
        weak_observations[frame_id] = weak

        strong_positions[frame_id] = (
            np.asarray(
                [np.asarray(obs.world_position, dtype=np.float64) for obs in strong],
                dtype=np.float64,
            )
            if strong
            else np.zeros((0, 2), dtype=np.float64)
        )
        weak_positions[frame_id] = (
            np.asarray(
                [np.asarray(obs.world_position, dtype=np.float64) for obs in weak],
                dtype=np.float64,
            )
            if weak
            else np.zeros((0, 2), dtype=np.float64)
        )

    return _FrameEvidenceIndex(
        ordered_frames=ordered,
        frame_by_id=frame_by_id,
        frame_index=frame_index,
        strong_positions=strong_positions,
        weak_positions=weak_positions,
        strong_observations=strong_observations,
        weak_observations=weak_observations,
    )


def _fragment_role_config(
    config: GlobalIdentityV5Config,
) -> v4.GlobalIdentityV4Config:
    return v4.GlobalIdentityV4Config(
        base=config.base,
        core_min_observed_frames=config.core_min_observed_frames,
        core_min_observed_span_s=config.core_min_observed_span_s,
        bridge_candidate_min_observed_frames=(
            config.bridge_candidate_min_observed_frames
        ),
        bridge_candidate_min_observed_span_s=(
            config.bridge_candidate_min_observed_span_s
        ),
    )


def _robust_endpoint_velocity(
    fragment: v4.IdentityFragmentV4,
    *,
    at_start: bool,
    config: GlobalIdentityV5Config,
) -> np.ndarray | None:
    states = list(fragment.observed_states)
    if len(states) < 2:
        return None

    anchor_time = states[0].time_s if at_start else states[-1].time_s
    selected = [
        state
        for state in states
        if (
            state.time_s - anchor_time <= config.endpoint_velocity_window_s
            if at_start
            else anchor_time - state.time_s <= config.endpoint_velocity_window_s
        )
    ]
    if len(selected) < 2:
        return None

    velocities = np.asarray(
        [[state.vx, state.vz] for state in selected],
        dtype=np.float64,
    )
    velocity = np.median(velocities, axis=0)
    if not np.isfinite(velocity).all():
        return None
    return velocity.astype(np.float64)


def _cap_velocity(vector: np.ndarray | None, cap_mps: float) -> np.ndarray | None:
    if vector is None:
        return None
    value = np.asarray(vector, dtype=np.float64)
    speed = float(np.linalg.norm(value))
    if not math.isfinite(speed):
        return None
    if speed <= cap_mps or speed <= 1e-9:
        return value
    return value * (cap_mps / speed)


def _context_bonus(
    predecessor: v4.IdentityFragmentV4,
    successor: v4.IdentityFragmentV4,
    *,
    config: GlobalIdentityV5Config,
) -> float:
    raw = config.context_bonus_scale * (
        math.log1p(max(0.0, predecessor.observed_span_s))
        + math.log1p(max(0.0, successor.observed_span_s))
    )
    return float(min(config.context_bonus_max, max(0.0, raw)))


def _expected_position(
    left: v3.EvidenceState,
    right: v3.EvidenceState,
    *,
    time_s: float,
    start_velocity: np.ndarray | None,
    end_velocity: np.ndarray | None,
) -> np.ndarray:
    duration = float(right.time_s - left.time_s)
    if duration <= 0.0:
        return left.position.copy()

    alpha = (float(time_s) - float(left.time_s)) / duration
    alpha = min(1.0, max(0.0, alpha))

    if start_velocity is not None and end_velocity is not None:
        position, _ = v3._hermite(
            left.position,
            start_velocity,
            right.position,
            end_velocity,
            duration=duration,
            alpha=alpha,
        )
        if np.isfinite(position).all():
            return np.asarray(position, dtype=np.float64)

    return (
        (1.0 - alpha) * left.position
        + alpha * right.position
    ).astype(np.float64)


def _bridge_evidence(
    predecessor: v4.IdentityFragmentV4,
    successor: v4.IdentityFragmentV4,
    evidence: _FrameEvidenceIndex,
    *,
    start_velocity: np.ndarray | None,
    end_velocity: np.ndarray | None,
    config: GlobalIdentityV5Config,
) -> BridgeEvidence:
    left = predecessor.observed_end_state
    right = successor.observed_start_state

    if right.time_s <= left.time_s:
        return BridgeEvidence(0, 0, 0, 0)

    left_index = evidence.frame_index.get(left.frame)
    right_index = evidence.frame_index.get(right.frame)
    if left_index is None or right_index is None or right_index <= left_index + 1:
        return BridgeEvidence(0, 0, 0, 0)

    sampled = 0
    strong_supported = 0
    weak_supported = 0
    any_supported = 0

    for frame_id in evidence.ordered_frames[
        left_index + 1 : right_index : config.bridge_evidence_stride
    ]:
        frame = evidence.frame_by_id[frame_id]
        expected = _expected_position(
            left,
            right,
            time_s=float(frame.time_s),
            start_velocity=start_velocity,
            end_velocity=end_velocity,
        )

        sampled += 1
        strong = evidence.strong_positions[frame_id]
        weak = evidence.weak_positions[frame_id]

        strong_hit = (
            strong.size > 0
            and bool(
                np.any(
                    np.linalg.norm(strong - expected[None, :], axis=1)
                    <= config.bridge_evidence_radius_m
                )
            )
        )
        weak_hit = (
            weak.size > 0
            and bool(
                np.any(
                    np.linalg.norm(weak - expected[None, :], axis=1)
                    <= config.bridge_evidence_radius_m
                )
            )
        )

        strong_supported += int(strong_hit)
        weak_supported += int(weak_hit)
        any_supported += int(strong_hit or weak_hit)

    return BridgeEvidence(
        sampled_frames=sampled,
        strong_supported_frames=strong_supported,
        weak_supported_frames=weak_supported,
        any_supported_frames=any_supported,
    )


def _overlap_evidence(
    predecessor: v4.IdentityFragmentV4,
    successor: v4.IdentityFragmentV4,
    evidence: _FrameEvidenceIndex,
    *,
    config: GlobalIdentityV5Config,
) -> tuple[float, int, float]:
    common_frames = sorted(
        set(predecessor.states).intersection(successor.states)
    )
    if not common_frames:
        return 0.0, 0, math.inf

    informative = 0
    independent = 0
    distances: list[float] = []

    for frame_id in common_frames:
        left = predecessor.states[frame_id]
        right = successor.states[frame_id]

        if left.state != "observed" or right.state != "observed":
            continue

        distance = float(np.linalg.norm(left.position - right.position))
        distances.append(distance)

        strong_positions = evidence.strong_positions.get(
            frame_id,
            np.zeros((0, 2), dtype=np.float64),
        )
        is_informative, is_independent = v3._distinct_observation_support(
            left.position,
            right.position,
            strong_positions,
            radius_m=config.overlap_evidence_radius_m,
        )
        if is_informative:
            informative += 1
            independent += int(is_independent)

    ratio = independent / informative if informative else 0.0
    median_distance = (
        float(statistics.median(distances))
        if distances
        else math.inf
    )
    return float(ratio), int(informative), float(median_distance)


def _prediction_error(
    predecessor: v4.IdentityFragmentV4,
    successor: v4.IdentityFragmentV4,
    *,
    gap_s: float,
    predecessor_velocity: np.ndarray | None,
    successor_velocity: np.ndarray | None,
) -> float:
    left = predecessor.observed_end_state
    right = successor.observed_start_state
    errors: list[float] = []

    if predecessor_velocity is not None:
        forward = left.position + predecessor_velocity * gap_s
        errors.append(float(np.linalg.norm(right.position - forward)))

    if successor_velocity is not None:
        backward = right.position - successor_velocity * gap_s
        errors.append(float(np.linalg.norm(left.position - backward)))

    if errors:
        return float(statistics.fmean(errors))
    return float(np.linalg.norm(right.position - left.position))


def evaluate_identity_link(
    predecessor: v4.IdentityFragmentV4,
    successor: v4.IdentityFragmentV4,
    evidence: _FrameEvidenceIndex,
    *,
    clip_start_s: float,
    clip_end_s: float,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> IdentityLinkCandidateV5 | None:
    if predecessor.fragment_id == successor.fragment_id:
        return None
    if predecessor.role == "ephemeral" or successor.role == "ephemeral":
        return None

    # Directed acyclic orientation.
    if successor.observed_start_time_s <= predecessor.observed_start_time_s:
        return None

    gap_s = (
        successor.observed_start_time_s
        - predecessor.observed_end_time_s
    )
    overlap_s = max(0.0, -gap_s)

    shared_sources = tuple(
        sorted(
            set(predecessor.source_track_ids)
            .intersection(successor.source_track_ids)
        )
    )

    context_bonus = _context_bonus(
        predecessor,
        successor,
        config=config,
    )

    overlap_ratio = None
    overlap_informative = 0
    bridge = BridgeEvidence(0, 0, 0, 0)

    pred_velocity = _robust_endpoint_velocity(
        predecessor,
        at_start=False,
        config=config,
    )
    succ_velocity = _robust_endpoint_velocity(
        successor,
        at_start=True,
        config=config,
    )
    pred_velocity = _cap_velocity(
        pred_velocity,
        config.interpolation_velocity_cap_mps,
    )
    succ_velocity = _cap_velocity(
        succ_velocity,
        config.interpolation_velocity_cap_mps,
    )

    left = predecessor.observed_end_state
    right = successor.observed_start_state

    if gap_s < 0.0:
        if overlap_s > config.overlap_max_s:
            return None

        overlap_ratio, overlap_informative, median_distance = _overlap_evidence(
            predecessor,
            successor,
            evidence,
            config=config,
        )

        if median_distance > config.overlap_max_median_distance_m:
            return None

        if (
            overlap_informative > 0
            and overlap_ratio > config.overlap_max_independent_ratio
        ):
            return None

        # Without independent raw evidence or lineage, do not merge on proximity.
        if overlap_informative == 0 and not shared_sources:
            return None

        direct_distance = median_distance
        physical_radius = config.overlap_max_median_distance_m
        prediction_error = median_distance
        prediction_radius = config.overlap_max_median_distance_m
        velocity_difference = None

        if pred_velocity is not None and succ_velocity is not None:
            velocity_difference = float(
                np.linalg.norm(pred_velocity - succ_velocity)
            )

        tier = "overlap"
        link_cost = (
            prediction_error / max(0.25, prediction_radius)
            + 0.08 * overlap_s
            + (
                config.velocity_cost_weight * velocity_difference
                if velocity_difference is not None
                else 0.0
            )
            - (config.same_lineage_bonus if shared_sources else 0.0)
            - 0.35 * context_bonus
        )
    else:
        if gap_s > config.medium_gap_max_s:
            return None

        tier = "short" if gap_s <= config.short_gap_max_s else "medium"

        direct_distance = float(
            np.linalg.norm(right.position - left.position)
        )

        if tier == "short":
            max_speed = config.short_max_speed_mps
            prediction_radius = (
                config.short_prediction_base_m
                + config.short_prediction_growth_mps * gap_s
            )
            max_velocity_difference = config.short_max_velocity_difference_mps
        else:
            if predecessor.role != "core" or successor.role != "core":
                return None
            if (
                predecessor.observed_span_s < config.medium_min_context_span_s
                or successor.observed_span_s < config.medium_min_context_span_s
            ):
                return None
            if pred_velocity is None or succ_velocity is None:
                return None

            max_speed = config.medium_max_speed_mps
            prediction_radius = (
                config.medium_prediction_base_m
                + config.medium_prediction_growth_mps * gap_s
            )
            max_velocity_difference = config.medium_max_velocity_difference_mps

        physical_radius = (
            max_speed * gap_s
            + config.physical_slack_m
        )
        if direct_distance > physical_radius:
            return None

        prediction_error = _prediction_error(
            predecessor,
            successor,
            gap_s=gap_s,
            predecessor_velocity=pred_velocity,
            successor_velocity=succ_velocity,
        )

        # If both endpoint velocities exist, use them as a hard compatibility
        # safeguard. If a short gap lacks reliable velocity on one side, direct
        # motion still remains available as a conservative fallback.
        if (
            pred_velocity is not None
            and succ_velocity is not None
            and prediction_error > prediction_radius
        ):
            return None

        velocity_difference = None
        if pred_velocity is not None and succ_velocity is not None:
            velocity_difference = float(
                np.linalg.norm(pred_velocity - succ_velocity)
            )
            if velocity_difference > max_velocity_difference:
                return None

        bridge = _bridge_evidence(
            predecessor,
            successor,
            evidence,
            start_velocity=pred_velocity,
            end_velocity=succ_velocity,
            config=config,
        )

        if tier == "medium":
            has_bridge = (
                bridge.any_supported_frames
                >= config.medium_bridge_min_supported_frames
                and bridge.any_ratio >= config.medium_bridge_min_ratio
            )
            if not shared_sources and not has_bridge:
                return None

        normalized_prediction = (
            prediction_error / max(0.50, prediction_radius)
        )

        if tier == "short":
            temporal_cost = config.short_gap_cost_weight * gap_s
            tier_penalty = 0.0
        else:
            temporal_cost = config.medium_gap_cost_weight * gap_s
            tier_penalty = config.medium_gap_fixed_penalty

        bridge_bonus = (
            config.strong_bridge_bonus * bridge.strong_ratio
            + config.weak_bridge_bonus * bridge.weak_ratio
        )

        link_cost = (
            normalized_prediction
            + temporal_cost
            + tier_penalty
            + (
                config.velocity_cost_weight * velocity_difference
                if velocity_difference is not None
                else 0.0
            )
            - bridge_bonus
            - (config.same_lineage_bonus if shared_sources else 0.0)
            - context_bonus
        )

    link_cost = max(0.0, float(link_cost))

    birth_cost = v3.event_cost(
        position=right.position,
        time_s=right.time_s,
        clip_edge_time_s=clip_start_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        support_score=successor.support_score,
        config=config.base,
    )
    death_cost = v3.event_cost(
        position=left.position,
        time_s=left.time_s,
        clip_edge_time_s=clip_end_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        support_score=predecessor.support_score,
        config=config.base,
    )

    savings = birth_cost + death_cost - link_cost
    if savings <= 0.0:
        return None

    return IdentityLinkCandidateV5(
        predecessor_fragment_id=predecessor.fragment_id,
        successor_fragment_id=successor.fragment_id,
        tier=tier,
        observed_gap_s=float(gap_s),
        observed_overlap_s=float(overlap_s),
        direct_distance_m=float(direct_distance),
        physical_radius_m=float(physical_radius),
        prediction_error_m=float(prediction_error),
        prediction_radius_m=float(prediction_radius),
        velocity_difference_mps=velocity_difference,
        bridge_sampled_frames=bridge.sampled_frames,
        bridge_strong_supported_frames=bridge.strong_supported_frames,
        bridge_weak_supported_frames=bridge.weak_supported_frames,
        bridge_any_supported_frames=bridge.any_supported_frames,
        bridge_strong_ratio=float(bridge.strong_ratio),
        bridge_weak_ratio=float(bridge.weak_ratio),
        bridge_any_ratio=float(bridge.any_ratio),
        overlap_independent_evidence_ratio=(
            None if overlap_ratio is None else float(overlap_ratio)
        ),
        overlap_informative_frames=int(overlap_informative),
        shared_source_track_ids=shared_sources,
        predecessor_role=predecessor.role,
        successor_role=successor.role,
        predecessor_observed_span_s=float(predecessor.observed_span_s),
        successor_observed_span_s=float(successor.observed_span_s),
        context_bonus=float(context_bonus),
        birth_cost_successor=float(birth_cost),
        death_cost_predecessor=float(death_cost),
        link_cost=float(link_cost),
        savings_vs_birth_death=float(savings),
    )


def _top_k_pairs(
    candidates: Mapping[tuple[int, int], IdentityLinkCandidateV5],
    *,
    config: GlobalIdentityV5Config,
) -> set[tuple[int, int]]:
    by_pred: dict[int, list[tuple[tuple[int, int], IdentityLinkCandidateV5]]] = {}
    by_succ: dict[int, list[tuple[tuple[int, int], IdentityLinkCandidateV5]]] = {}

    for pair, candidate in candidates.items():
        by_pred.setdefault(candidate.predecessor_fragment_id, []).append(
            (pair, candidate)
        )
        by_succ.setdefault(candidate.successor_fragment_id, []).append(
            (pair, candidate)
        )

    def rank(item: tuple[tuple[int, int], IdentityLinkCandidateV5]):
        _, candidate = item
        tier_rank = {"overlap": 0, "short": 1, "medium": 2}[candidate.tier]
        return (
            -candidate.savings_vs_birth_death,
            tier_rank,
            candidate.link_cost,
            abs(candidate.observed_gap_s),
            candidate.successor_fragment_id,
        )

    keep: set[tuple[int, int]] = set()

    for values in by_pred.values():
        values.sort(key=rank)
        keep.update(
            pair
            for pair, _ in values[: config.candidate_top_k_outgoing]
        )

    for values in by_succ.values():
        values.sort(key=rank)
        keep.update(
            pair
            for pair, _ in values[: config.candidate_top_k_incoming]
        )

    return keep


def _candidate_pairs_time_indexed(
    fragments: Sequence[v4.IdentityFragmentV4],
    evidence: _FrameEvidenceIndex,
    *,
    clip_start_s: float,
    clip_end_s: float,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> dict[tuple[int, int], IdentityLinkCandidateV5]:
    fragments = tuple(fragments)

    eligible = [
        (index, fragment)
        for index, fragment in enumerate(fragments)
        if fragment.role != "ephemeral"
    ]
    by_start = sorted(
        eligible,
        key=lambda item: (
            item[1].observed_start_time_s,
            item[1].fragment_id,
        ),
    )
    start_times = [item[1].observed_start_time_s for item in by_start]

    result: dict[tuple[int, int], IdentityLinkCandidateV5] = {}

    for row, predecessor in eligible:
        low = (
            predecessor.observed_end_time_s
            - config.overlap_max_s
        )
        high = (
            predecessor.observed_end_time_s
            + config.medium_gap_max_s
        )

        lo = bisect_left(start_times, low)
        hi = bisect_right(start_times, high)

        for column, successor in by_start[lo:hi]:
            if row == column:
                continue
            if (
                successor.observed_start_time_s
                <= predecessor.observed_start_time_s
            ):
                continue

            candidate = evaluate_identity_link(
                predecessor,
                successor,
                evidence,
                clip_start_s=clip_start_s,
                clip_end_s=clip_end_s,
                field_length_m=field_length_m,
                field_width_m=field_width_m,
                config=config,
            )
            if candidate is not None:
                result[(row, column)] = candidate

    return result


def _sparse_path_cover(
    fragments: Sequence[v4.IdentityFragmentV4],
    candidates: Mapping[tuple[int, int], IdentityLinkCandidateV5],
    *,
    clip_end_s: float,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> tuple[tuple[IdentityLinkCandidateV5, ...], tuple[tuple[int, int], ...]]:
    fragments = tuple(fragments)
    count = len(fragments)
    if count == 0:
        return (), ()

    retained_pairs = _top_k_pairs(candidates, config=config)
    retained = {pair: candidates[pair] for pair in retained_pairs}

    row_indices: list[int] = []
    col_indices: list[int] = []
    costs: list[float] = []

    # Each row always has its own private death column, making a complete
    # matching possible without constructing a dense Nx2N matrix.
    raw_edge_costs: list[float] = []

    for row, predecessor in enumerate(fragments):
        death_cost = v3.event_cost(
            position=predecessor.observed_end_state.position,
            time_s=predecessor.observed_end_time_s,
            clip_edge_time_s=clip_end_s,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
            support_score=predecessor.support_score,
            config=config.base,
        )
        row_indices.append(row)
        col_indices.append(count + row)
        raw_edge_costs.append(float(death_cost))

    for (row, column), candidate in retained.items():
        row_indices.append(row)
        col_indices.append(column)
        raw_edge_costs.append(
            float(candidate.link_cost - candidate.birth_cost_successor)
        )

    min_cost = min(raw_edge_costs) if raw_edge_costs else 0.0
    shift = 1.0 - min(0.0, min_cost)
    costs = [value + shift for value in raw_edge_costs]

    matrix = coo_matrix(
        (
            np.asarray(costs, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int64),
                np.asarray(col_indices, dtype=np.int64),
            ),
        ),
        shape=(count, count * 2),
    ).tocsr()

    matched_rows, matched_columns = min_weight_full_bipartite_matching(matrix)

    accepted: list[tuple[int, int]] = []
    for row, column in zip(
        matched_rows.tolist(),
        matched_columns.tolist(),
    ):
        if column >= count:
            continue
        candidate = retained.get((row, column))
        if candidate is None:
            continue
        accepted.append(
            (
                candidate.predecessor_fragment_id,
                candidate.successor_fragment_id,
            )
        )

    ordered_candidates = tuple(
        sorted(
            retained.values(),
            key=lambda item: (
                -item.savings_vs_birth_death,
                item.predecessor_fragment_id,
                item.successor_fragment_id,
            ),
        )
    )
    return ordered_candidates, tuple(sorted(accepted))


def solve_fragment_path_cover(
    fragments: Sequence[v4.IdentityFragmentV4],
    frames: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> tuple[int, tuple[IdentityLinkCandidateV5, ...], tuple[tuple[int, int], ...]]:
    if not frames:
        return 0, (), ()

    evidence = _evidence_index(frames, config=config)
    clip_start_s = float(frames[0].time_s)
    clip_end_s = float(frames[-1].time_s)

    candidates = _candidate_pairs_time_indexed(
        fragments,
        evidence,
        clip_start_s=clip_start_s,
        clip_end_s=clip_end_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=config,
    )

    retained, accepted = _sparse_path_cover(
        fragments,
        candidates,
        clip_end_s=clip_end_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=config,
    )
    return len(candidates), retained, accepted


def _fragment_chains(
    fragments: Sequence[v4.IdentityFragmentV4],
    links: Sequence[tuple[int, int]],
) -> list[list[v4.IdentityFragmentV4]]:
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    successor = {left: right for left, right in links}
    predecessor = {right: left for left, right in links}

    starts = [
        fragment.fragment_id
        for fragment in fragments
        if fragment.fragment_id not in predecessor
    ]

    chains: list[list[v4.IdentityFragmentV4]] = []
    visited: set[int] = set()

    for start in sorted(
        starts,
        key=lambda fragment_id: (
            by_id[fragment_id].observed_start_time_s,
            fragment_id,
        ),
    ):
        chain: list[v4.IdentityFragmentV4] = []
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


def _nearest_supporting_observation(
    frame_id: int,
    position: np.ndarray,
    evidence: _FrameEvidenceIndex,
    *,
    config: GlobalIdentityV5Config,
) -> tuple[str | None, Any | None]:
    def nearest(observations: tuple[Any, ...]) -> tuple[float, Any] | None:
        best = None
        for observation in observations:
            distance = float(
                np.linalg.norm(
                    np.asarray(observation.world_position, dtype=np.float64)
                    - position
                )
            )
            if best is None or distance < best[0]:
                best = (distance, observation)
        return best

    strong = nearest(evidence.strong_observations.get(frame_id, ()))
    weak = nearest(evidence.weak_observations.get(frame_id, ()))

    if strong is not None and strong[0] <= config.bridge_support_radius_m:
        return "strong_bridge", strong[1]
    if weak is not None and weak[0] <= config.bridge_support_radius_m:
        return "weak_bridge", weak[1]
    return None, None


def _clip_pitch(
    position: np.ndarray,
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> np.ndarray:
    result = np.asarray(position, dtype=np.float64).copy()
    if field_length_m is not None:
        half = field_length_m / 2.0 + config.pitch_clip_margin_m
        result[0] = float(np.clip(result[0], -half, half))
    if field_width_m is not None:
        half = field_width_m / 2.0 + config.pitch_clip_margin_m
        result[1] = float(np.clip(result[1], -half, half))
    return result


def _gap_curve(
    left: v3.EvidenceState,
    right: v3.EvidenceState,
    frames: Sequence[Any],
    *,
    start_velocity: np.ndarray | None,
    end_velocity: np.ndarray | None,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    duration = float(right.time_s - left.time_s)
    if duration <= 0.0:
        return {}

    frame_time = {
        int(frame.frame): float(frame.time_s)
        for frame in frames
        if left.time_s < float(frame.time_s) < right.time_s
    }
    if not frame_time:
        return {}

    v0 = _cap_velocity(
        start_velocity,
        config.interpolation_velocity_cap_mps,
    )
    v1 = _cap_velocity(
        end_velocity,
        config.interpolation_velocity_cap_mps,
    )

    def build(use_hermite: bool) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        states: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        linear_velocity = (
            (right.position - left.position) / duration
        ).astype(np.float64)

        for frame_id, time_s in sorted(frame_time.items()):
            alpha = (time_s - left.time_s) / duration

            if use_hermite and v0 is not None and v1 is not None:
                position, velocity = v3._hermite(
                    left.position,
                    v0,
                    right.position,
                    v1,
                    duration=duration,
                    alpha=alpha,
                )
            else:
                position = (
                    (1.0 - alpha) * left.position
                    + alpha * right.position
                )
                velocity = linear_velocity

            position = _clip_pitch(
                np.asarray(position, dtype=np.float64),
                field_length_m=field_length_m,
                field_width_m=field_width_m,
                config=config,
            )
            velocity = np.asarray(velocity, dtype=np.float64)

            if not np.isfinite(position).all() or not np.isfinite(velocity).all():
                return {}
            states[frame_id] = (position, velocity)

        return states

    candidate = build(use_hermite=True)

    def curve_is_safe(
        states: Mapping[int, tuple[np.ndarray, np.ndarray]]
    ) -> bool:
        if not states:
            return False
        for _, velocity in states.values():
            if float(np.linalg.norm(velocity)) > config.interpolation_speed_limit_mps:
                return False
        return True

    if curve_is_safe(candidate):
        return candidate

    fallback = build(use_hermite=False)
    if curve_is_safe(fallback):
        return fallback

    return {}


def _analytical_from_fragment_state(
    *,
    identity_id: int,
    state: v3.EvidenceState,
    evidence: _FrameEvidenceIndex,
    support_score: float,
    member_count: int,
    source_track_count: int,
    config: GlobalIdentityV5Config,
) -> AnalyticalFrameTrack:
    supporting_confidence = None
    supporting_detection_index = None

    if state.state == "observed":
        label = "strong_observed"
    else:
        label, observation = _nearest_supporting_observation(
            state.frame,
            state.position,
            evidence,
            config=config,
        )
        if label is None:
            label = "inferred"
        elif observation is not None:
            supporting_confidence = float(observation.confidence)
            detection_index = getattr(observation, "detection_index", None)
            if detection_index is not None:
                supporting_detection_index = int(detection_index)

    return AnalyticalFrameTrack(
        identity_id=identity_id,
        frame=int(state.frame),
        time_s=float(state.time_s),
        evidence=label,
        x=float(state.x),
        z=float(state.z),
        vx=float(state.vx),
        vz=float(state.vz),
        support_score=float(support_score),
        member_trajectory_count=int(member_count),
        source_track_count=int(source_track_count),
        supporting_confidence=supporting_confidence,
        supporting_detection_index=supporting_detection_index,
    )


def _merge_state(
    states: dict[int, AnalyticalFrameTrack],
    candidate: AnalyticalFrameTrack,
) -> None:
    rank = {
        "strong_observed": 4,
        "strong_bridge": 3,
        "weak_bridge": 2,
        "inferred": 1,
    }
    existing = states.get(candidate.frame)
    if existing is None:
        states[candidate.frame] = candidate
        return

    existing_key = (
        rank.get(existing.evidence, 0),
        existing.support_score,
    )
    candidate_key = (
        rank.get(candidate.evidence, 0),
        candidate.support_score,
    )
    if candidate_key > existing_key:
        states[candidate.frame] = candidate


def build_global_identities(
    fragments: Sequence[v4.IdentityFragmentV4],
    links: Sequence[tuple[int, int]],
    frames: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config,
) -> tuple[GlobalIdentityV5, ...]:
    evidence = _evidence_index(frames, config=config)
    chains = _fragment_chains(fragments, links)
    identities: list[GlobalIdentityV5] = []

    for chain in chains:
        # Weak/ephemeral evidence alone must never become a final identity.
        if not any(fragment.role == "core" for fragment in chain):
            continue

        identity_id = len(identities) + 1
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
        member_count = len(member_trajectories)
        source_count = len(source_track_ids)
        support = max(fragment.support_score for fragment in chain)

        states: dict[int, AnalyticalFrameTrack] = {}

        for fragment in chain:
            for _, state in sorted(fragment.states.items()):
                analytical = _analytical_from_fragment_state(
                    identity_id=identity_id,
                    state=state,
                    evidence=evidence,
                    support_score=support,
                    member_count=member_count,
                    source_track_count=source_count,
                    config=config,
                )
                _merge_state(states, analytical)

        # Fill accepted SHORT gaps bidirectionally. Medium links preserve
        # identity continuity but intentionally do not fabricate positions.
        for left_fragment, right_fragment in zip(chain, chain[1:]):
            gap_s = (
                right_fragment.observed_start_time_s
                - left_fragment.observed_end_time_s
            )
            if not (0.0 < gap_s <= config.analytical_fill_max_gap_s):
                continue

            left = left_fragment.observed_end_state
            right = right_fragment.observed_start_state
            v0 = _robust_endpoint_velocity(
                left_fragment,
                at_start=False,
                config=config,
            )
            v1 = _robust_endpoint_velocity(
                right_fragment,
                at_start=True,
                config=config,
            )

            curve = _gap_curve(
                left,
                right,
                frames,
                start_velocity=v0,
                end_velocity=v1,
                field_length_m=field_length_m,
                field_width_m=field_width_m,
                config=config,
            )

            for frame_id, (position, velocity) in curve.items():
                frame = evidence.frame_by_id[frame_id]
                label, supporting_observation = _nearest_supporting_observation(
                    frame_id,
                    position,
                    evidence,
                    config=config,
                )
                if label is None:
                    label = "inferred"

                supporting_confidence = None
                supporting_detection_index = None
                if supporting_observation is not None:
                    supporting_confidence = float(
                        supporting_observation.confidence
                    )
                    detection_index = getattr(
                        supporting_observation,
                        "detection_index",
                        None,
                    )
                    if detection_index is not None:
                        supporting_detection_index = int(detection_index)

                candidate = AnalyticalFrameTrack(
                    identity_id=identity_id,
                    frame=int(frame_id),
                    time_s=float(frame.time_s),
                    evidence=label,
                    x=float(position[0]),
                    z=float(position[1]),
                    vx=float(velocity[0]),
                    vz=float(velocity[1]),
                    support_score=float(support + 0.10),
                    member_trajectory_count=member_count,
                    source_track_count=source_count,
                    supporting_confidence=supporting_confidence,
                    supporting_detection_index=supporting_detection_index,
                )
                _merge_state(states, candidate)

        if not states:
            continue

        first_observed = chain[0].observed_start_state
        last_observed = chain[-1].observed_end_state

        birth_distance = v3.boundary_distance_m(
            first_observed.position,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )
        death_distance = v3.boundary_distance_m(
            last_observed.position,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )

        identities.append(
            GlobalIdentityV5(
                identity_id=identity_id,
                fragment_ids=tuple(
                    fragment.fragment_id for fragment in chain
                ),
                member_trajectory_ids=tuple(member_trajectories),
                source_track_ids=tuple(source_track_ids),
                states=states,
                support_score=float(support),
                observed_start_time_s=float(first_observed.time_s),
                observed_end_time_s=float(last_observed.time_s),
                boundary_birth=(
                    birth_distance is not None and birth_distance <= 2.0
                ),
                boundary_death=(
                    death_distance is not None and death_distance <= 2.0
                ),
                core_fragment_count=sum(
                    fragment.role == "core" for fragment in chain
                ),
                bridge_fragment_count=sum(
                    fragment.role != "core" for fragment in chain
                ),
            )
        )

    return tuple(identities)


def build_frame_tracks(
    frames: Sequence[Any],
    identities: Sequence[GlobalIdentityV5],
) -> dict[int, tuple[AnalyticalFrameTrack, ...]]:
    result: dict[int, tuple[AnalyticalFrameTrack, ...]] = {}
    for frame in frames:
        frame_id = int(frame.frame)
        tracks = [
            identity.states[frame_id]
            for identity in identities
            if frame_id in identity.states
        ]
        tracks.sort(key=lambda item: item.identity_id)
        result[frame_id] = tuple(tracks)
    return result


def solve_global_identities(
    frames: Sequence[Any],
    strong_frames: Sequence[Any],
    trajectories: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5Config | None = None,
) -> GlobalIdentityV5Result:
    cfg = config or make_config()

    # Local proposals contain STRONG evidence only. Duplicate resolution also
    # uses STRONG raw detections so weak noise cannot prove two people exist.
    state_maps = v3.build_trajectory_state_maps(
        strong_frames,
        trajectories,
        config=cfg.base,
    )
    duplicate_evidence, components, blocked = v3.resolve_duplicate_components(
        trajectories,
        state_maps,
        strong_frames,
        config=cfg.base,
    )

    fragments = v4.build_identity_fragments(
        trajectories,
        components,
        state_maps,
        config=_fragment_role_config(cfg),
    )

    raw_candidate_count, candidates, accepted_links = solve_fragment_path_cover(
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

    frame_tracks = build_frame_tracks(frames, identities)

    return GlobalIdentityV5Result(
        duplicate_evidence=duplicate_evidence,
        duplicate_components=components,
        blocked_duplicate_merges=blocked,
        fragments=fragments,
        raw_candidate_count=raw_candidate_count,
        link_candidates=candidates,
        accepted_links=accepted_links,
        identities=identities,
        frame_tracks=frame_tracks,
    )


__all__ = [
    "AnalyticalFrameTrack",
    "BridgeEvidence",
    "GlobalIdentityV5",
    "GlobalIdentityV5Config",
    "GlobalIdentityV5Result",
    "IdentityLinkCandidateV5",
    "build_frame_tracks",
    "build_global_identities",
    "evaluate_identity_link",
    "make_config",
    "solve_fragment_path_cover",
    "solve_global_identities",
]
