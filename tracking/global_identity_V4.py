"""VeoMatchLab clip-global identity solver V4.

V4 is an experimental layer on top of the stable V3 identity solver.  It keeps
V3 duplicate-branch resolution, but replaces continuation generation and final
identity assembly.

Main changes from V3:
- observed support is separated from interpolated support;
- small temporal overlap can be a valid handoff when raw detections do not
  support two independent people;
- candidate generation evaluates a broad graph and retains Top-K incoming and
  outgoing alternatives before the global path-cover optimization;
- fragments are CORE, BRIDGE-CANDIDATE or EPHEMERAL;
- weak isolated chains are diagnostic only and do not become final visible IDs;
- proximity alone never merges footballers.

This module intentionally imports the current stable tracking.global_identity
(V3) for its validated duplicate-resolution machinery.  It does not overwrite
that stable module, which makes V3/V4 A/B runs straightforward.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking import global_identity as v3


_FORBIDDEN = 1e12

EvidenceState = v3.EvidenceState
DuplicateEvidence = v3.DuplicateEvidence
GlobalIdentityFrameTrack = v3.GlobalIdentityFrameTrack


@dataclass(frozen=True)
class GlobalIdentityV4Config:
    # All V3 duplicate/evidence/birth-death parameters remain available here.
    base: v3.GlobalIdentityConfig = field(default_factory=v3.GlobalIdentityConfig)

    # Contextual fragment roles.
    core_min_observed_frames: int = 18
    core_min_observed_span_s: float = 0.55
    bridge_candidate_min_observed_frames: int = 6
    bridge_candidate_min_observed_span_s: float = 0.15

    # Broader candidate graph.
    candidate_top_k_outgoing: int = 5
    candidate_top_k_incoming: int = 5

    # A short observed overlap can be a handoff if raw evidence supports only
    # one person.  This is deliberately checked independently of proximity.
    handoff_max_observed_overlap_s: float = 0.50
    handoff_overlap_evidence_radius_m: float = 0.90
    handoff_overlap_max_independent_ratio: float = 0.20
    handoff_overlap_max_median_distance_m: float = 0.90

    # Final output still uses V3's render-gap and max-visible settings through
    # ``base``.  Weak chains require at least one CORE fragment to render.

    def __post_init__(self) -> None:
        for name in (
            "core_min_observed_frames",
            "bridge_candidate_min_observed_frames",
            "candidate_top_k_outgoing",
            "candidate_top_k_incoming",
        ):
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")

        for name in (
            "core_min_observed_span_s",
            "bridge_candidate_min_observed_span_s",
            "handoff_max_observed_overlap_s",
            "handoff_overlap_evidence_radius_m",
            "handoff_overlap_max_median_distance_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        if not 0.0 <= self.handoff_overlap_max_independent_ratio <= 1.0:
            raise ValueError(
                "handoff_overlap_max_independent_ratio must be in [0,1]"
            )


@dataclass(frozen=True)
class IdentityFragmentV4:
    fragment_id: int
    member_trajectory_ids: tuple[int, ...]
    source_track_ids: tuple[int, ...]
    states: Mapping[int, EvidenceState]
    support_score: float
    role: str  # core | bridge_candidate | ephemeral

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

    @property
    def observed_states(self) -> tuple[EvidenceState, ...]:
        return tuple(
            self.states[frame_id]
            for frame_id in sorted(self.states)
            if self.states[frame_id].state == "observed"
        )

    @property
    def observed_count(self) -> int:
        return len(self.observed_states)

    @property
    def observed_start_state(self) -> EvidenceState:
        values = self.observed_states
        return values[0] if values else self.start_state

    @property
    def observed_end_state(self) -> EvidenceState:
        values = self.observed_states
        return values[-1] if values else self.end_state

    @property
    def observed_start_time_s(self) -> float:
        return self.observed_start_state.time_s

    @property
    def observed_end_time_s(self) -> float:
        return self.observed_end_state.time_s

    @property
    def observed_span_s(self) -> float:
        return max(
            0.0,
            self.observed_end_time_s - self.observed_start_time_s,
        )


@dataclass(frozen=True)
class IdentityLinkCandidateV4:
    predecessor_fragment_id: int
    successor_fragment_id: int

    # Positive = observed gap. Negative = small observed overlap.
    observed_gap_s: float
    rendered_gap_s: float
    observed_overlap_s: float
    handoff_mode: str  # gap | interpolated_overlap | observed_overlap

    direct_distance_m: float
    physical_radius_m: float
    prediction_error_m: float
    prediction_radius_m: float
    velocity_difference_mps: float | None

    bridge_evidence_ratio: float
    overlap_independent_evidence_ratio: float | None
    overlap_informative_frames: int

    shared_source_track_ids: tuple[int, ...]
    predecessor_role: str
    successor_role: str

    birth_cost_successor: float
    death_cost_predecessor: float
    link_cost: float
    savings_vs_birth_death: float


@dataclass(frozen=True)
class GlobalIdentityV4:
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
    core_fragment_count: int
    bridge_fragment_count: int


@dataclass(frozen=True)
class GlobalIdentityV4Result:
    duplicate_evidence: tuple[DuplicateEvidence, ...]
    duplicate_components: tuple[tuple[int, ...], ...]
    blocked_duplicate_merges: tuple[tuple[int, int], ...]
    fragments: tuple[IdentityFragmentV4, ...]
    raw_candidate_count: int
    link_candidates: tuple[IdentityLinkCandidateV4, ...]
    accepted_links: tuple[tuple[int, int], ...]
    core_fragment_ids: tuple[int, ...]
    bridge_fragment_ids: tuple[int, ...]
    ephemeral_fragment_ids: tuple[int, ...]
    identities: tuple[GlobalIdentityV4, ...]
    frame_tracks: Mapping[int, tuple[GlobalIdentityFrameTrack, ...]]


def make_config(
    *,
    state_interpolation_gap_s: float | None = None,
    render_gap_s: float | None = None,
    max_visible_tracks: int | None = None,
) -> GlobalIdentityV4Config:
    """Create V4 config while inheriting current V3 defaults."""

    base = v3.GlobalIdentityConfig()
    changes = {}
    if state_interpolation_gap_s is not None:
        changes["state_interpolation_gap_s"] = float(state_interpolation_gap_s)
    if render_gap_s is not None:
        changes["render_gap_s"] = float(render_gap_s)
    if max_visible_tracks is not None:
        changes["max_visible_tracks"] = int(max_visible_tracks)
    if changes:
        base = replace(base, **changes)
    return GlobalIdentityV4Config(base=base)


def _fragment_role(
    states: Mapping[int, EvidenceState],
    *,
    config: GlobalIdentityV4Config,
) -> str:
    observed = [
        state
        for _, state in sorted(states.items())
        if state.state == "observed"
    ]
    if not observed:
        return "ephemeral"

    count = len(observed)
    span = max(0.0, observed[-1].time_s - observed[0].time_s)

    if (
        count >= config.core_min_observed_frames
        and span >= config.core_min_observed_span_s
    ):
        return "core"

    if (
        count >= config.bridge_candidate_min_observed_frames
        and span >= config.bridge_candidate_min_observed_span_s
    ):
        return "bridge_candidate"

    return "ephemeral"


def build_identity_fragments(
    trajectories: Sequence[Any],
    components: Sequence[Sequence[int]],
    state_maps: Mapping[int, Mapping[int, EvidenceState]],
    *,
    config: GlobalIdentityV4Config | None = None,
) -> tuple[IdentityFragmentV4, ...]:
    cfg = config or GlobalIdentityV4Config()
    trajectory_by_id = {
        int(trajectory.trajectory_id): trajectory
        for trajectory in trajectories
    }

    fragments: list[IdentityFragmentV4] = []

    for fragment_id, members in enumerate(components, start=1):
        states = v3._select_component_states(
            members,
            state_maps,
            config=cfg.base,
        )
        if not states:
            continue

        source_track_ids = sorted({
            int(source_id)
            for member_id in members
            for source_id in trajectory_by_id[member_id].source_track_ids
        })
        support_score = max(
            v3._trajectory_support(trajectory_by_id[member_id])
            for member_id in members
        )

        fragments.append(
            IdentityFragmentV4(
                fragment_id=fragment_id,
                member_trajectory_ids=tuple(sorted(int(v) for v in members)),
                source_track_ids=tuple(source_track_ids),
                states=states,
                support_score=float(support_score),
                role=_fragment_role(states, config=cfg),
            )
        )

    return tuple(fragments)


def _robust_endpoint_velocity(
    fragment: IdentityFragmentV4,
    *,
    at_start: bool,
    window_s: float = 0.50,
) -> np.ndarray | None:
    # V4 deliberately uses OBSERVED support only.
    states = list(fragment.observed_states)
    if len(states) < 2:
        return None

    anchor_time = states[0].time_s if at_start else states[-1].time_s
    selected = [
        state
        for state in states
        if (
            state.time_s - anchor_time <= window_s
            if at_start
            else anchor_time - state.time_s <= window_s
        )
    ]
    if len(selected) < 2:
        return None

    velocity = np.median(
        np.asarray([[state.vx, state.vz] for state in selected], dtype=np.float64),
        axis=0,
    )
    return velocity if np.isfinite(velocity).all() else None


def _frame_lookup(frames: Sequence[Any]) -> tuple[list[int], dict[int, Any]]:
    ordered = sorted(int(frame.frame) for frame in frames)
    return ordered, {int(frame.frame): frame for frame in frames}


def _bridge_evidence_ratio(
    predecessor: IdentityFragmentV4,
    successor: IdentityFragmentV4,
    frames: Sequence[Any],
    *,
    start_velocity: np.ndarray | None,
    end_velocity: np.ndarray | None,
    config: GlobalIdentityV4Config,
) -> float:
    left = predecessor.observed_end_state
    right = successor.observed_start_state
    duration = right.time_s - left.time_s
    if duration <= 0.0:
        return 0.0

    ordered_frames, frame_by_id = _frame_lookup(frames)
    frame_index = {frame_id: i for i, frame_id in enumerate(ordered_frames)}
    if left.frame not in frame_index or right.frame not in frame_index:
        return 0.0

    left_index = frame_index[left.frame]
    right_index = frame_index[right.frame]
    if right_index <= left_index + 1:
        return 0.0

    sampled = 0
    supported = 0

    for frame_id in ordered_frames[
        left_index + 1 : right_index : config.base.bridge_evidence_stride
    ]:
        frame = frame_by_id[frame_id]
        alpha = (float(frame.time_s) - left.time_s) / duration
        if not 0.0 < alpha < 1.0:
            continue

        if start_velocity is not None and end_velocity is not None:
            expected, _ = v3._hermite(
                left.position,
                start_velocity,
                right.position,
                end_velocity,
                duration=duration,
                alpha=alpha,
            )
        else:
            expected = (1.0 - alpha) * left.position + alpha * right.position

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
        if bool(np.any(distances <= config.base.bridge_evidence_radius_m)):
            supported += 1

    return supported / sampled if sampled else 0.0


def _handoff_overlap_evidence(
    predecessor: IdentityFragmentV4,
    successor: IdentityFragmentV4,
    frames: Sequence[Any],
    *,
    config: GlobalIdentityV4Config,
) -> tuple[float, int, float]:
    start_time = max(
        predecessor.observed_start_time_s,
        successor.observed_start_time_s,
    )
    end_time = min(
        predecessor.observed_end_time_s,
        successor.observed_end_time_s,
    )
    if end_time <= start_time:
        return 0.0, 0, 0.0

    frame_by_id = {int(frame.frame): frame for frame in frames}
    common_frames = sorted(set(predecessor.states).intersection(successor.states))

    informative = 0
    independent = 0
    distances: list[float] = []

    for frame_id in common_frames:
        left = predecessor.states[frame_id]
        right = successor.states[frame_id]
        if not (
            left.state == "observed"
            and right.state == "observed"
            and start_time <= left.time_s <= end_time
            and start_time <= right.time_s <= end_time
        ):
            continue

        distances.append(float(np.linalg.norm(left.position - right.position)))
        frame = frame_by_id.get(frame_id)
        if frame is None:
            continue
        observations = np.asarray(
            [
                np.asarray(observation.world_position, dtype=np.float64)
                for observation in frame.observations
            ],
            dtype=np.float64,
        )
        if observations.size == 0:
            continue

        is_informative, is_independent = v3._distinct_observation_support(
            left.position,
            right.position,
            observations,
            radius_m=config.handoff_overlap_evidence_radius_m,
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
    return float(ratio), int(informative), median_distance


def _temporal_relation(
    predecessor: IdentityFragmentV4,
    successor: IdentityFragmentV4,
) -> tuple[float, float, float, str]:
    observed_gap = (
        successor.observed_start_time_s - predecessor.observed_end_time_s
    )
    rendered_gap = successor.start_time_s - predecessor.end_time_s
    overlap = max(0.0, -observed_gap)

    if observed_gap > 0.0:
        mode = "interpolated_overlap" if rendered_gap <= 0.0 else "gap"
    else:
        mode = "observed_overlap"
    return float(observed_gap), float(rendered_gap), float(overlap), mode


def evaluate_identity_link(
    predecessor: IdentityFragmentV4,
    successor: IdentityFragmentV4,
    frames: Sequence[Any],
    *,
    clip_start_s: float,
    clip_end_s: float,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV4Config | None = None,
) -> IdentityLinkCandidateV4 | None:
    cfg = config or GlobalIdentityV4Config()
    base = cfg.base

    if predecessor.fragment_id == successor.fragment_id:
        return None
    if predecessor.role == "ephemeral" or successor.role == "ephemeral":
        return None

    # Enforce DAG orientation by observed start time, independent of rendered
    # interpolation overlap.
    if successor.observed_start_time_s <= predecessor.observed_start_time_s:
        return None

    observed_gap, rendered_gap, overlap, mode = _temporal_relation(
        predecessor,
        successor,
    )
    if observed_gap > base.linker_max_gap_s:
        return None
    if overlap > cfg.handoff_max_observed_overlap_s:
        return None

    pred_velocity = _robust_endpoint_velocity(predecessor, at_start=False)
    succ_velocity = _robust_endpoint_velocity(successor, at_start=True)

    shared_sources = tuple(sorted(
        set(predecessor.source_track_ids).intersection(successor.source_track_ids)
    ))

    bridge_ratio = 0.0
    overlap_independent_ratio = None
    overlap_informative = 0
    left = predecessor.observed_end_state
    right = successor.observed_start_state

    if observed_gap >= 0.0:
        direct_distance = float(np.linalg.norm(right.position - left.position))
        physical_radius = base.linker_max_speed_mps * observed_gap + base.linker_base_slack_m
        if direct_distance > physical_radius:
            return None

        if (
            observed_gap > base.linker_long_gap_requires_velocity_s
            and (pred_velocity is None or succ_velocity is None)
        ):
            return None

        errors: list[float] = []
        if pred_velocity is not None:
            forward = left.position + pred_velocity * observed_gap
            errors.append(float(np.linalg.norm(right.position - forward)))
        if succ_velocity is not None:
            backward = right.position - succ_velocity * observed_gap
            errors.append(float(np.linalg.norm(left.position - backward)))

        prediction_error = float(statistics.fmean(errors)) if errors else direct_distance
        prediction_radius = (
            base.linker_prediction_base_m
            + base.linker_prediction_growth_mps * observed_gap
        )
        if prediction_error > prediction_radius:
            return None

        bridge_ratio = _bridge_evidence_ratio(
            predecessor,
            successor,
            frames,
            start_velocity=pred_velocity,
            end_velocity=succ_velocity,
            config=cfg,
        )
        if (
            observed_gap > base.bridge_required_gap_s
            and not shared_sources
            and bridge_ratio < base.bridge_min_ratio_without_lineage
        ):
            return None
    else:
        overlap_independent_ratio, overlap_informative, overlap_distance = (
            _handoff_overlap_evidence(
                predecessor,
                successor,
                frames,
                config=cfg,
            )
        )
        if overlap_distance > cfg.handoff_overlap_max_median_distance_m:
            return None
        if (
            overlap_informative > 0
            and overlap_independent_ratio > cfg.handoff_overlap_max_independent_ratio
        ):
            return None
        # No raw evidence at all: require lineage rather than linking purely
        # from spatial proximity.
        if overlap_informative == 0 and not shared_sources:
            return None

        direct_distance = float(overlap_distance)
        physical_radius = cfg.handoff_overlap_max_median_distance_m
        prediction_error = direct_distance
        prediction_radius = cfg.handoff_overlap_max_median_distance_m

    velocity_difference = None
    if pred_velocity is not None and succ_velocity is not None:
        velocity_difference = float(np.linalg.norm(pred_velocity - succ_velocity))
        if velocity_difference > base.linker_max_velocity_difference_mps:
            return None

    if observed_gap >= 0.0:
        temporal_cost = base.linker_gap_cost_weight * observed_gap
        evidence_bonus = base.bridge_bonus * bridge_ratio
    else:
        temporal_cost = base.linker_gap_cost_weight * overlap
        evidence_bonus = 0.0

    link_cost = (
        prediction_error / max(0.25, prediction_radius)
        + temporal_cost
        + (
            base.linker_velocity_cost_weight * velocity_difference
            if velocity_difference is not None
            else 0.0
        )
        - evidence_bonus
        - (base.linker_same_lineage_bonus if shared_sources else 0.0)
    )
    link_cost = max(0.0, float(link_cost))

    birth_cost = v3.event_cost(
        position=successor.observed_start_state.position,
        time_s=successor.observed_start_time_s,
        clip_edge_time_s=clip_start_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        support_score=successor.support_score,
        config=base,
    )
    death_cost = v3.event_cost(
        position=predecessor.observed_end_state.position,
        time_s=predecessor.observed_end_time_s,
        clip_edge_time_s=clip_end_s,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        support_score=predecessor.support_score,
        config=base,
    )

    savings = birth_cost + death_cost - link_cost
    if savings <= 0.0:
        return None

    return IdentityLinkCandidateV4(
        predecessor_fragment_id=predecessor.fragment_id,
        successor_fragment_id=successor.fragment_id,
        observed_gap_s=float(observed_gap),
        rendered_gap_s=float(rendered_gap),
        observed_overlap_s=float(overlap),
        handoff_mode=mode,
        direct_distance_m=float(direct_distance),
        physical_radius_m=float(physical_radius),
        prediction_error_m=float(prediction_error),
        prediction_radius_m=float(prediction_radius),
        velocity_difference_mps=velocity_difference,
        bridge_evidence_ratio=float(bridge_ratio),
        overlap_independent_evidence_ratio=(
            None
            if overlap_independent_ratio is None
            else float(overlap_independent_ratio)
        ),
        overlap_informative_frames=int(overlap_informative),
        shared_source_track_ids=shared_sources,
        predecessor_role=predecessor.role,
        successor_role=successor.role,
        birth_cost_successor=float(birth_cost),
        death_cost_predecessor=float(death_cost),
        link_cost=float(link_cost),
        savings_vs_birth_death=float(savings),
    )


def _top_k_pairs(
    candidates: Mapping[tuple[int, int], IdentityLinkCandidateV4],
    *,
    config: GlobalIdentityV4Config,
) -> set[tuple[int, int]]:
    by_pred: dict[int, list[tuple[tuple[int, int], IdentityLinkCandidateV4]]] = {}
    by_succ: dict[int, list[tuple[tuple[int, int], IdentityLinkCandidateV4]]] = {}

    for pair, candidate in candidates.items():
        by_pred.setdefault(candidate.predecessor_fragment_id, []).append((pair, candidate))
        by_succ.setdefault(candidate.successor_fragment_id, []).append((pair, candidate))

    def rank(item):
        _, candidate = item
        return (
            -candidate.savings_vs_birth_death,
            candidate.link_cost,
            abs(candidate.observed_gap_s),
            candidate.successor_fragment_id,
        )

    keep: set[tuple[int, int]] = set()
    for values in by_pred.values():
        values.sort(key=rank)
        keep.update(pair for pair, _ in values[: config.candidate_top_k_outgoing])
    for values in by_succ.values():
        values.sort(key=rank)
        keep.update(pair for pair, _ in values[: config.candidate_top_k_incoming])
    return keep


def solve_fragment_path_cover(
    fragments: Sequence[IdentityFragmentV4],
    frames: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV4Config | None = None,
) -> tuple[int, tuple[IdentityLinkCandidateV4, ...], tuple[tuple[int, int], ...]]:
    cfg = config or GlobalIdentityV4Config()
    base = cfg.base
    fragments = tuple(fragments)
    count = len(fragments)
    if count == 0:
        return 0, (), ()

    clip_start_s = float(frames[0].time_s)
    clip_end_s = float(frames[-1].time_s)

    all_candidates: dict[tuple[int, int], IdentityLinkCandidateV4] = {}
    for row, predecessor in enumerate(fragments):
        if predecessor.role == "ephemeral":
            continue
        for column, successor in enumerate(fragments):
            if row == column or successor.role == "ephemeral":
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
            if candidate is not None:
                all_candidates[(row, column)] = candidate

    retained = _top_k_pairs(all_candidates, config=cfg)
    candidate_by_pair = {pair: all_candidates[pair] for pair in retained}

    matrix = np.full((count, count + count), _FORBIDDEN, dtype=np.float64)
    for row, predecessor in enumerate(fragments):
        death_cost = v3.event_cost(
            position=predecessor.observed_end_state.position,
            time_s=predecessor.observed_end_time_s,
            clip_edge_time_s=clip_end_s,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
            support_score=predecessor.support_score,
            config=base,
        )
        matrix[row, count + row] = death_cost

    for (row, column), candidate in candidate_by_pair.items():
        matrix[row, column] = candidate.link_cost - candidate.birth_cost_successor

    rows, columns = linear_sum_assignment(matrix)
    accepted: list[tuple[int, int]] = []
    for row, column in zip(rows.tolist(), columns.tolist()):
        if column >= count:
            continue
        candidate = candidate_by_pair.get((row, column))
        if candidate is not None:
            accepted.append((
                candidate.predecessor_fragment_id,
                candidate.successor_fragment_id,
            ))

    candidates = tuple(sorted(
        candidate_by_pair.values(),
        key=lambda item: (
            -item.savings_vs_birth_death,
            item.predecessor_fragment_id,
            item.successor_fragment_id,
        ),
    ))
    return len(all_candidates), candidates, tuple(sorted(accepted))


def _fragment_chains(
    fragments: Sequence[IdentityFragmentV4],
    links: Sequence[tuple[int, int]],
) -> list[list[IdentityFragmentV4]]:
    by_id = {fragment.fragment_id: fragment for fragment in fragments}
    successor = {left: right for left, right in links}
    predecessor = {right: left for left, right in links}
    starts = [
        fragment.fragment_id
        for fragment in fragments
        if fragment.fragment_id not in predecessor
    ]

    chains: list[list[IdentityFragmentV4]] = []
    visited: set[int] = set()
    for start in sorted(
        starts,
        key=lambda fragment_id: (by_id[fragment_id].observed_start_time_s, fragment_id),
    ):
        chain = []
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


def _merge_state(
    states: dict[int, GlobalIdentityFrameTrack],
    candidate: GlobalIdentityFrameTrack,
) -> None:
    existing = states.get(candidate.frame)
    if existing is None:
        states[candidate.frame] = candidate
        return
    existing_rank = (
        1 if existing.state == "observed" else 0,
        existing.support_score,
    )
    candidate_rank = (
        1 if candidate.state == "observed" else 0,
        candidate.support_score,
    )
    if candidate_rank > existing_rank:
        states[candidate.frame] = candidate


def build_global_identities(
    fragments: Sequence[IdentityFragmentV4],
    links: Sequence[tuple[int, int]],
    frames: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV4Config | None = None,
) -> tuple[GlobalIdentityV4, ...]:
    cfg = config or GlobalIdentityV4Config()
    chains = _fragment_chains(fragments, links)
    identities: list[GlobalIdentityV4] = []

    for chain in chains:
        # A bridge candidate only becomes product-level evidence when attached
        # to at least one core.  Isolated weak/ephemeral chains are suppressed.
        if not any(fragment.role == "core" for fragment in chain):
            continue

        identity_id = len(identities) + 1
        member_trajectories = sorted({
            trajectory_id
            for fragment in chain
            for trajectory_id in fragment.member_trajectory_ids
        })
        source_track_ids = sorted({
            source_id
            for fragment in chain
            for source_id in fragment.source_track_ids
        })
        support = max(fragment.support_score for fragment in chain)
        states: dict[int, GlobalIdentityFrameTrack] = {}

        for fragment in chain:
            for frame_id, state in fragment.states.items():
                _merge_state(
                    states,
                    GlobalIdentityFrameTrack(
                        trajectory_id=identity_id,
                        frame=frame_id,
                        time_s=state.time_s,
                        state=state.state,
                        x=state.x,
                        z=state.z,
                        vx=state.vx,
                        vz=state.vz,
                        support_score=float(
                            support + (0.50 if state.state == "observed" else 0.10)
                        ),
                        member_trajectory_count=len(member_trajectories),
                        source_track_count=len(source_track_ids),
                    ),
                )

        # Fill only a short positive OBSERVED gap.  Overlap handoffs are resolved
        # by selecting the better observed state rather than double-rendering.
        for left_fragment, right_fragment in zip(chain, chain[1:]):
            observed_gap = (
                right_fragment.observed_start_time_s
                - left_fragment.observed_end_time_s
            )
            if 0.0 < observed_gap <= cfg.base.render_gap_s:
                gap_states = v3._interpolate_identity_gap(
                    left_fragment.observed_end_state,
                    right_fragment.observed_start_state,
                    frames,
                    config=cfg.base,
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
        birth_distance = v3.boundary_distance_m(
            first.position,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )
        death_distance = v3.boundary_distance_m(
            last.position,
            field_length_m=field_length_m,
            field_width_m=field_width_m,
        )

        identities.append(
            GlobalIdentityV4(
                identity_id=identity_id,
                fragment_ids=tuple(fragment.fragment_id for fragment in chain),
                member_trajectory_ids=tuple(member_trajectories),
                source_track_ids=tuple(source_track_ids),
                states=states,
                support_score=float(support),
                start_time_s=float(first.time_s),
                end_time_s=float(last.time_s),
                boundary_birth=(birth_distance is not None and birth_distance <= 2.0),
                boundary_death=(death_distance is not None and death_distance <= 2.0),
                core_fragment_count=sum(fragment.role == "core" for fragment in chain),
                bridge_fragment_count=sum(fragment.role != "core" for fragment in chain),
            )
        )

    return tuple(identities)


def build_frame_tracks(
    frames: Sequence[Any],
    identities: Sequence[GlobalIdentityV4],
    *,
    config: GlobalIdentityV4Config | None = None,
) -> dict[int, tuple[GlobalIdentityFrameTrack, ...]]:
    cfg = config or GlobalIdentityV4Config()
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
                -(item.support_score + (0.50 if item.state == "observed" else 0.0)),
                item.trajectory_id,
            )
        )
        result[frame_id] = tuple(
            candidates[: cfg.base.max_visible_tracks]
        )
    return result


def _post_roles(
    fragments: Sequence[IdentityFragmentV4],
    identities: Sequence[GlobalIdentityV4],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    core = {
        fragment.fragment_id
        for fragment in fragments
        if fragment.role == "core"
    }
    rendered = {
        fragment_id
        for identity in identities
        for fragment_id in identity.fragment_ids
    }
    bridge = {fragment_id for fragment_id in rendered if fragment_id not in core}
    ephemeral = {
        fragment.fragment_id
        for fragment in fragments
        if fragment.fragment_id not in rendered
    }
    return tuple(sorted(core)), tuple(sorted(bridge)), tuple(sorted(ephemeral))


def solve_global_identities(
    frames: Sequence[Any],
    trajectories: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV4Config | None = None,
) -> GlobalIdentityV4Result:
    cfg = config or GlobalIdentityV4Config()

    state_maps = v3.build_trajectory_state_maps(
        frames,
        trajectories,
        config=cfg.base,
    )
    duplicate_evidence, components, blocked = v3.resolve_duplicate_components(
        trajectories,
        state_maps,
        frames,
        config=cfg.base,
    )
    fragments = build_identity_fragments(
        trajectories,
        components,
        state_maps,
        config=cfg,
    )
    raw_candidate_count, link_candidates, accepted_links = solve_fragment_path_cover(
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
    core_ids, bridge_ids, ephemeral_ids = _post_roles(fragments, identities)
    frame_tracks = build_frame_tracks(frames, identities, config=cfg)

    return GlobalIdentityV4Result(
        duplicate_evidence=duplicate_evidence,
        duplicate_components=components,
        blocked_duplicate_merges=blocked,
        fragments=fragments,
        raw_candidate_count=raw_candidate_count,
        link_candidates=link_candidates,
        accepted_links=accepted_links,
        core_fragment_ids=core_ids,
        bridge_fragment_ids=bridge_ids,
        ephemeral_fragment_ids=ephemeral_ids,
        identities=identities,
        frame_tracks=frame_tracks,
    )


__all__ = [
    "DuplicateEvidence",
    "EvidenceState",
    "GlobalIdentityFrameTrack",
    "GlobalIdentityV4",
    "GlobalIdentityV4Config",
    "GlobalIdentityV4Result",
    "IdentityFragmentV4",
    "IdentityLinkCandidateV4",
    "build_frame_tracks",
    "build_global_identities",
    "build_identity_fragments",
    "evaluate_identity_link",
    "make_config",
    "solve_fragment_path_cover",
    "solve_global_identities",
]
