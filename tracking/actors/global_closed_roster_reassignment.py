"""Tracking V1 Hito 7I.6: global closed-roster reassignment.

This stage keeps only one immutable fact from the previous experiments:

    the 23 actor seeds present at the beginning of the clip.

Every later tracklet is mutable.  A tracklet can move from one actor to another,
move from NON_ROSTER back into the 23-person roster, or be evicted to
NON_ROSTER when a globally better assignment needs its slot.

The optimizer is deliberately scene/geometry-first:

- exact simultaneous occupancy is structural;
- short-range field geometry is structural;
- kit family is supporting evidence;
- ReID is only a low-weight confirmation / tie-break signal.

The implementation uses iterative counterfactual reassignment with conflict
repair.  For each mutable tracklet it scores all 23 actors after temporarily
removing that tracklet from its current owner.  If the desired actor is blocked
by one or two mutable tracklets, the optimizer can evaluate a swap or an
eviction move.  Accepted moves are applied globally and the process repeats
until stable.

This is not an online tracker and it does not create actor IDs.  The terminal
state of every tracklet is exactly one of:

- actor-0001 .. actor-0023
- NON_ROSTER

The stage also emits actor-level trajectory anchors so later rendering can
smooth/interpolate actor trajectories rather than individual tracklets.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


GLOBAL_REASSIGNMENT_VERSION = "global-closed-roster-reassignment-v1"
NON_ROSTER_ID = "NON_ROSTER"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a non-empty 1-D vector")
    if not np.isfinite(vector).all():
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding has zero norm")
    return vector / norm


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "min": None, "p50": None, "p90": None,
            "p95": None, "max": None, "mean": None,
        }
    ordered = sorted(float(value) for value in values)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * q
        lo = math.floor(pos)
        hi = math.ceil(pos)
        if lo == hi:
            return ordered[lo]
        alpha = pos - lo
        return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pct(.50),
        "p90": pct(.90),
        "p95": pct(.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


@dataclass(frozen=True, slots=True)
class GlobalClosedRosterConfig:
    expected_actor_slots: int = 23

    field_length_m: float = 105.0
    field_width_m: float = 68.0
    off_field_margin_m: float = 0.75

    max_transition_speed_mps: float = 12.0
    position_slack_m: float = 1.25
    hard_geometry_window_s: float = 2.5
    context_window_s: float = 7.5
    hard_conflict_relevance_s: float = 3.0

    geometry_weight: float = 0.58
    temporal_weight: float = 0.18
    family_weight: float = 0.18
    reid_weight: float = 0.06

    candidate_score_weight: float = 0.72
    roster_evidence_weight: float = 0.28

    direct_move_margin: float = 0.035
    swap_move_margin: float = 0.030
    eviction_move_margin: float = 0.055
    non_roster_move_margin: float = 0.060

    max_challenge_blockers: int = 2
    max_rounds: int = 8

    strong_non_roster_floor: float = 0.72
    short_extra_s: float = 2.0
    very_short_extra_s: float = 0.8
    off_field_strong_ratio: float = 0.80
    occupancy_strong_ratio: float = 0.80

    review_assignment_utility_floor: float = 0.48
    review_assignment_margin: float = 0.05

    visual_interpolation_max_gap_s: float = 0.50
    visual_interpolation_max_speed_mps: float = 12.5

    time_epsilon_s: float = 1e-6

    def __post_init__(self) -> None:
        if self.expected_actor_slots < 1:
            raise ValueError("expected_actor_slots must be >= 1")
        for name in (
            "field_length_m",
            "field_width_m",
            "max_transition_speed_mps",
            "position_slack_m",
            "hard_geometry_window_s",
            "context_window_s",
            "hard_conflict_relevance_s",
            "visual_interpolation_max_gap_s",
            "visual_interpolation_max_speed_mps",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.off_field_margin_m < 0:
            raise ValueError("off_field_margin_m must be >= 0")
        if self.max_challenge_blockers < 0:
            raise ValueError("max_challenge_blockers must be >= 0")
        if self.max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")

        evidence_weights = (
            self.geometry_weight,
            self.temporal_weight,
            self.family_weight,
            self.reid_weight,
        )
        if any(value < 0 for value in evidence_weights):
            raise ValueError("evidence weights must be non-negative")
        if sum(evidence_weights) <= 0:
            raise ValueError("evidence weights must sum to > 0")

        utility_weights = (
            self.candidate_score_weight,
            self.roster_evidence_weight,
        )
        if any(value < 0 for value in utility_weights):
            raise ValueError("utility weights must be non-negative")
        if abs(sum(utility_weights) - 1.0) > 1e-9:
            raise ValueError("utility weights must sum to 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Observation:
    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _Tracklet:
    tracklet_id: str
    observations: tuple[_Observation, ...]

    @property
    def frames(self) -> frozenset[int]:
        return frozenset(obs.frame_index for obs in self.observations)

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def start_time(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_time(self) -> float:
        return self.observations[-1].timestamp_s

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time - self.start_time)

    @property
    def mean_confidence(self) -> float:
        return sum(obs.confidence for obs in self.observations) / len(
            self.observations
        )


def _parse_tracklet(row: Mapping[str, Any]) -> _Tracklet:
    tid = str(row["tracklet_id"])
    raw = row.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{tid} has no observations")

    observations = []
    previous_frame = None
    previous_time = None
    for item in raw:
        obs = _Observation(
            detection_id=str(item["detection_id"]),
            frame_index=int(item["frame_index"]),
            timestamp_s=float(item["timestamp_s"]),
            x=float(item["x"]),
            z=float(item["z"]),
            confidence=float(item.get("confidence", 1.0)),
        )
        if obs.frame_index < 0:
            raise ValueError(f"{tid} has negative frame")
        if not all(
            math.isfinite(value)
            for value in (
                obs.timestamp_s,
                obs.x,
                obs.z,
                obs.confidence,
            )
        ):
            raise ValueError(f"{tid} contains non-finite values")
        if previous_frame is not None and obs.frame_index <= previous_frame:
            raise ValueError(f"{tid} frames are not strictly increasing")
        if previous_time is not None and obs.timestamp_s <= previous_time:
            raise ValueError(f"{tid} timestamps are not strictly increasing")
        previous_frame = obs.frame_index
        previous_time = obs.timestamp_s
        observations.append(obs)

    return _Tracklet(tid, tuple(observations))


@dataclass(frozen=True, slots=True)
class _Family:
    key: str | None
    confident: bool
    mass: float


def _family(row: Mapping[str, Any] | None) -> _Family:
    if row is None:
        return _Family(None, False, 0.0)

    state = str(row.get("state", "unknown"))
    if state in {"confident_primary", "confident_special"}:
        key = row.get("family_key")
        if isinstance(key, str) and key:
            return _Family(
                key,
                True,
                1.0 if state == "confident_primary" else .90,
            )

    if state in {"weak_primary", "weak_special"}:
        key = row.get("candidate_family_key")
        if isinstance(key, str) and key:
            return _Family(
                key,
                False,
                .35 if state == "weak_primary" else .15,
            )

    return _Family(None, False, 0.0)


@dataclass(frozen=True, slots=True)
class _Appearance:
    embedding: np.ndarray
    reliability: float
    effective_sample_count: float


def _appearance(row: Mapping[str, Any] | None) -> _Appearance | None:
    if row is None:
        return None
    values = row.get("aggregate_embedding")
    if not isinstance(values, list) or not values:
        return None
    return _Appearance(
        embedding=_unit(values),
        reliability=_clamp01(float(row.get("reliability", 0.0))),
        effective_sample_count=max(
            1.0,
            float(row.get("effective_sample_count", 1.0)),
        ),
    )


@dataclass(frozen=True, slots=True)
class _ActorContext:
    actor_id: str
    member_ids: tuple[str, ...]
    observations: tuple[tuple[_Observation, str], ...]
    observation_frames: tuple[int, ...]
    frame_owner: Mapping[int, str]
    family: _Family
    appearance: np.ndarray | None

    def previous(
        self,
        frame_index: int,
    ) -> tuple[_Observation, str] | None:
        index = bisect_left(self.observation_frames, frame_index) - 1
        return self.observations[index] if index >= 0 else None

    def after(
        self,
        frame_index: int,
    ) -> tuple[_Observation, str] | None:
        index = bisect_right(self.observation_frames, frame_index)
        return (
            self.observations[index]
            if index < len(self.observations)
            else None
        )


@dataclass(frozen=True, slots=True)
class ActorCandidate:
    actor_id: str
    tracklet_id: str
    viable: bool
    score: float
    utility: float
    roster_evidence: float

    blockers: tuple[str, ...]
    blocker_kinds: tuple[str, ...]
    seed_blocked: bool

    before_gap_s: float | None
    after_gap_s: float | None
    before_distance_m: float | None
    after_distance_m: float | None
    geometry_support: float
    temporal_support: float

    family_relation: str
    family_support: float

    reid_cosine: float | None
    reid_support: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReassignmentMove:
    round_index: int
    move_type: str
    tracklet_id: str
    from_actor_id: str | None
    to_actor_id: str | None
    swapped_tracklet_id: str | None
    evicted_tracklet_ids: tuple[str, ...]
    old_utility: float
    new_utility: float
    delta_utility: float
    candidate: ActorCandidate | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "reassignment_version": GLOBAL_REASSIGNMENT_VERSION,
            "round_index": self.round_index,
            "move_type": self.move_type,
            "tracklet_id": self.tracklet_id,
            "from_actor_id": self.from_actor_id,
            "to_actor_id": self.to_actor_id,
            "swapped_tracklet_id": self.swapped_tracklet_id,
            "evicted_tracklet_ids": list(self.evicted_tracklet_ids),
            "old_utility": self.old_utility,
            "new_utility": self.new_utility,
            "delta_utility": self.delta_utility,
            "candidate": (
                self.candidate.to_dict()
                if self.candidate is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class GlobalClosedRosterResult:
    actors: tuple[dict[str, Any], ...]
    tracklet_assignments: tuple[dict[str, Any], ...]
    non_roster_tracklets: tuple[dict[str, Any], ...]
    moves: tuple[ReassignmentMove, ...]
    actor_trajectories: tuple[dict[str, Any], ...]
    seed_tracklet_ids: tuple[str, ...]
    initial_owner: Mapping[str, str | None]
    final_owner: Mapping[str, str | None]
    rounds_executed: int
    final_review_rows: tuple[dict[str, Any], ...]
    config: GlobalClosedRosterConfig

    def summary(self) -> dict[str, Any]:
        changed = [
            tid
            for tid, initial in self.initial_owner.items()
            if self.final_owner.get(tid) != initial
        ]
        rescued = [
            tid
            for tid in changed
            if self.initial_owner.get(tid) is None
            and self.final_owner.get(tid) is not None
        ]
        demoted = [
            tid
            for tid in changed
            if self.initial_owner.get(tid) is not None
            and self.final_owner.get(tid) is None
        ]
        actor_to_actor = [
            tid
            for tid in changed
            if self.initial_owner.get(tid) is not None
            and self.final_owner.get(tid) is not None
        ]

        return {
            "schema_version": 1,
            "reassignment_version": GLOBAL_REASSIGNMENT_VERSION,
            "actor_slot_count": len(self.actors),
            "seed_count": len(self.seed_tracklet_ids),
            "seed_immutable": True,
            "new_actor_creation": False,
            "tracklet_count": (
                len(self.tracklet_assignments)
                + len(self.non_roster_tracklets)
            ),
            "assigned_tracklet_count": len(self.tracklet_assignments),
            "non_roster_tracklet_count": len(
                self.non_roster_tracklets
            ),
            "unresolved_tracklet_count": 0,
            "rounds_executed": self.rounds_executed,
            "accepted_move_count": len(self.moves),
            "changed_tracklet_count": len(changed),
            "rescued_from_non_roster_count": len(rescued),
            "demoted_to_non_roster_count": len(demoted),
            "actor_to_actor_reassignment_count": len(actor_to_actor),
            "move_type_counts": dict(sorted(Counter(
                row.move_type for row in self.moves
            ).items())),
            "needs_review_count": sum(
                bool(row["needs_review"])
                for row in self.final_review_rows
            ),
            "review_reason_counts": dict(sorted(Counter(
                reason
                for row in self.final_review_rows
                for reason in row["review_reasons"]
            ).items())),
            "actor_tracklet_count": _distribution([
                len(row["tracklet_ids"])
                for row in self.actors
            ]),
            "move_delta_utility": _distribution([
                row.delta_utility
                for row in self.moves
            ]),
            "config": self.config.to_dict(),
        }


def _family_relation(
    actor: _Family,
    target: _Family,
) -> tuple[str, float]:
    if actor.key is None or target.key is None:
        return "neutral", .50
    if actor.key == target.key:
        if actor.confident and target.confident:
            return "same_confident", 1.0
        return "same_weak", .82
    if actor.confident and target.confident:
        # Strong penalty, deliberately not a hard veto.  The global optimizer
        # is allowed to repair a wrong family aggregation instead of freezing
        # the whole actor forever.
        return "different_confident", .03
    return "different_weak", .22


def _hard_conflict_index(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        a = str(row["tracklet_a"])
        b = str(row["tracklet_b"])
        result[a].add(b)
        result[b].add(a)
    return result


def _interval_gap(a: _Tracklet, b: _Tracklet) -> float:
    if a.end_time < b.start_time:
        return b.start_time - a.end_time
    if b.end_time < a.start_time:
        return a.start_time - b.end_time
    return 0.0


def _aggregate_family(
    member_ids: Sequence[str],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
) -> _Family:
    mass: dict[str, float] = defaultdict(float)
    confident_mass: dict[str, float] = defaultdict(float)

    for tid in member_ids:
        fam = _family(family_by_tracklet.get(tid))
        if fam.key is None or fam.mass <= 0:
            continue
        mass[fam.key] += fam.mass
        if fam.confident:
            confident_mass[fam.key] += fam.mass

    if not mass:
        return _Family(None, False, 0.0)

    winner = max(mass, key=lambda key: (mass[key], key))
    total = sum(mass.values())
    dominance = mass[winner] / total if total else 0.0

    confident_total = sum(confident_mass.values())
    winner_confident = confident_mass.get(winner, 0.0)
    confident_dominance = (
        winner_confident / confident_total
        if confident_total > 0
        else 0.0
    )
    confident = (
        winner_confident >= .90
        and dominance >= .70
        and (
            confident_total <= 1e-12
            or confident_dominance >= .75
        )
    )
    return _Family(winner, confident, float(dominance))


def _aggregate_appearance(
    member_ids: Sequence[str],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
) -> np.ndarray | None:
    aggregate = None

    for tid in member_ids:
        item = _appearance(appearance_by_tracklet.get(tid))
        if item is None:
            continue
        weight = max(
            1e-6,
            item.reliability
            * math.sqrt(item.effective_sample_count),
        )
        weighted = item.embedding * weight
        aggregate = (
            weighted.copy()
            if aggregate is None
            else aggregate + weighted
        )

    return None if aggregate is None else _unit(aggregate)


def _build_context(
    *,
    actor_id: str,
    assignment: Mapping[str, str | None],
    tracklets: Mapping[str, _Tracklet],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    exclude_ids: frozenset[str] = frozenset(),
) -> _ActorContext:
    member_ids = tuple(sorted(
        tid
        for tid, owner in assignment.items()
        if owner == actor_id and tid not in exclude_ids
    ))

    observations = sorted(
        (
            (obs, tid)
            for tid in member_ids
            for obs in tracklets[tid].observations
        ),
        key=lambda item: (
            item[0].frame_index,
            item[0].timestamp_s,
            item[1],
        ),
    )

    frame_owner: dict[int, str] = {}
    for obs, tid in observations:
        previous = frame_owner.get(obs.frame_index)
        if previous is not None and previous != tid:
            raise ValueError(
                f"{actor_id} has exact-frame coexistence between "
                f"{previous} and {tid} at frame {obs.frame_index}"
            )
        frame_owner[obs.frame_index] = tid

    return _ActorContext(
        actor_id=actor_id,
        member_ids=member_ids,
        observations=tuple(observations),
        observation_frames=tuple(
            item[0].frame_index for item in observations
        ),
        frame_owner=frame_owner,
        family=_aggregate_family(
            member_ids,
            family_by_tracklet,
        ),
        appearance=_aggregate_appearance(
            member_ids,
            appearance_by_tracklet,
        ),
    )


def _build_all_contexts(
    *,
    actor_ids: Sequence[str],
    assignment: Mapping[str, str | None],
    tracklets: Mapping[str, _Tracklet],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
) -> dict[str, _ActorContext]:
    return {
        actor_id: _build_context(
            actor_id=actor_id,
            assignment=assignment,
            tracklets=tracklets,
            family_by_tracklet=family_by_tracklet,
            appearance_by_tracklet=appearance_by_tracklet,
        )
        for actor_id in actor_ids
    }


def _transition_support(
    actor_obs: _Observation,
    target_obs: _Observation,
    *,
    dt_s: float,
    config: GlobalClosedRosterConfig,
) -> tuple[float, float, bool]:
    distance = math.hypot(
        target_obs.x - actor_obs.x,
        target_obs.z - actor_obs.z,
    )
    limit = (
        config.position_slack_m
        + config.max_transition_speed_mps * max(dt_s, 0.0)
    )
    hard = (
        dt_s <= config.hard_geometry_window_s
        and distance > limit + 1e-9
    )

    if dt_s > config.context_window_s:
        return distance, .18, hard

    # A tighter soft scale than the hard 12 m/s envelope keeps proximity
    # dominant while still permitting normal football motion.
    soft_scale = max(
        config.position_slack_m,
        config.position_slack_m + 5.5 * dt_s,
    )
    support = math.exp(
        -0.5 * (distance / soft_scale) ** 2
    )
    return distance, float(_clamp01(support)), hard


def _roster_evidence(
    target: _Tracklet,
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    config: GlobalClosedRosterConfig,
) -> float:
    half_length = config.field_length_m / 2.0
    half_width = config.field_width_m / 2.0

    inside = sum(
        (
            abs(obs.x) <= half_length + config.off_field_margin_m
            and abs(obs.z) <= half_width + config.off_field_margin_m
        )
        for obs in target.observations
    ) / len(target.observations)

    duration_support = _clamp01(
        target.duration_s / max(.65, config.short_extra_s)
    )
    confidence_support = _clamp01(target.mean_confidence)

    fam = _family(family_by_tracklet.get(target.tracklet_id))
    family_presence = 1.0 if fam.key is not None else .35

    return float(_clamp01(
        .58 * inside
        + .17 * duration_support
        + .15 * confidence_support
        + .10 * family_presence
    ))


def _actor_presence_count(
    target: _Tracklet,
    contexts: Mapping[str, _ActorContext],
    *,
    exclude_actor: str | None,
) -> float:
    if not target.observations:
        return 0.0

    counts = []
    for obs in target.observations:
        count = sum(
            obs.frame_index in context.frame_owner
            for actor_id, context in contexts.items()
            if actor_id != exclude_actor
        )
        counts.append(count)
    return sum(counts) / len(counts)


def _non_roster_utility(
    target: _Tracklet,
    contexts: Mapping[str, _ActorContext],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    config: GlobalClosedRosterConfig,
    *,
    current_actor: str | None,
) -> float:
    half_length = config.field_length_m / 2.0
    half_width = config.field_width_m / 2.0

    off_field_ratio = sum(
        (
            abs(obs.x) > half_length + config.off_field_margin_m
            or abs(obs.z) > half_width + config.off_field_margin_m
        )
        for obs in target.observations
    ) / len(target.observations)

    shortness = _clamp01(
        1.0 - target.duration_s / max(
            config.short_extra_s,
            config.time_epsilon_s,
        )
    )

    presence = _actor_presence_count(
        target,
        contexts,
        exclude_actor=current_actor,
    )
    occupancy = _clamp01(
        presence / max(1, config.expected_actor_slots - 1)
    )

    fam = _family(family_by_tracklet.get(target.tracklet_id))
    unknown_family = 1.0 if fam.key is None else 0.0

    utility = (
        .55 * off_field_ratio
        + .20 * occupancy
        + .15 * shortness
        + .10 * unknown_family
    )

    # A long tracklet mostly inside the field needs positive evidence before
    # being considered an extra person.
    if off_field_ratio < .25 and target.duration_s > .8:
        utility = min(utility, .42)

    if (
        off_field_ratio >= config.off_field_strong_ratio
        and target.duration_s <= config.short_extra_s
    ):
        utility = max(utility, .78)

    if (
        occupancy >= config.occupancy_strong_ratio
        and target.duration_s <= config.very_short_extra_s
    ):
        utility = max(utility, .74)

    return float(_clamp01(utility))


def _candidate_for_actor(
    *,
    target: _Tracklet,
    actor_id: str,
    assignment: Mapping[str, str | None],
    contexts: Mapping[str, _ActorContext],
    tracklets: Mapping[str, _Tracklet],
    seed_ids: frozenset[str],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: Mapping[str, set[str]],
    config: GlobalClosedRosterConfig,
) -> ActorCandidate:
    current_actor = assignment.get(target.tracklet_id)
    base_exclude = (
        frozenset({target.tracklet_id})
        if current_actor == actor_id
        else frozenset()
    )

    context = (
        _build_context(
            actor_id=actor_id,
            assignment=assignment,
            tracklets=tracklets,
            family_by_tracklet=family_by_tracklet,
            appearance_by_tracklet=appearance_by_tracklet,
            exclude_ids=base_exclude,
        )
        if base_exclude
        else contexts[actor_id]
    )

    overlap_blockers = {
        owner
        for frame in target.frames
        for owner in [context.frame_owner.get(frame)]
        if owner is not None and owner != target.tracklet_id
    }

    conflict_blockers = set()
    target_conflicts = hard_conflicts.get(target.tracklet_id, set())
    for member in context.member_ids:
        if member not in target_conflicts:
            continue
        if (
            _interval_gap(target, tracklets[member])
            <= config.hard_conflict_relevance_s
        ):
            conflict_blockers.add(member)

    blockers = overlap_blockers | conflict_blockers
    seed_blocked = any(tid in seed_ids for tid in blockers)

    blocker_kinds = []
    if overlap_blockers:
        blocker_kinds.append("exact_frame_coexistence")
    if conflict_blockers:
        blocker_kinds.append("hard_conflict")

    if seed_blocked or len(blockers) > config.max_challenge_blockers:
        return ActorCandidate(
            actor_id=actor_id,
            tracklet_id=target.tracklet_id,
            viable=False,
            score=0.0,
            utility=0.0,
            roster_evidence=_roster_evidence(
                target,
                family_by_tracklet,
                config,
            ),
            blockers=tuple(sorted(blockers)),
            blocker_kinds=tuple(blocker_kinds),
            seed_blocked=seed_blocked,
            before_gap_s=None,
            after_gap_s=None,
            before_distance_m=None,
            after_distance_m=None,
            geometry_support=0.0,
            temporal_support=0.0,
            family_relation="blocked",
            family_support=0.0,
            reid_cosine=None,
            reid_support=.50,
        )

    if blockers:
        context = _build_context(
            actor_id=actor_id,
            assignment=assignment,
            tracklets=tracklets,
            family_by_tracklet=family_by_tracklet,
            appearance_by_tracklet=appearance_by_tracklet,
            exclude_ids=frozenset(blockers) | base_exclude,
        )

    before = context.previous(target.start_frame)
    after = context.after(target.end_frame)

    before_gap = None
    after_gap = None
    before_distance = None
    after_distance = None
    before_support = .18
    after_support = .18

    if before is not None:
        obs, _ = before
        before_gap = target.start_time - obs.timestamp_s
        if before_gap <= config.time_epsilon_s:
            return ActorCandidate(
                actor_id=actor_id,
                tracklet_id=target.tracklet_id,
                viable=False,
                score=0.0,
                utility=0.0,
                roster_evidence=_roster_evidence(
                    target,
                    family_by_tracklet,
                    config,
                ),
                blockers=tuple(sorted(blockers)),
                blocker_kinds=tuple(
                    [*blocker_kinds, "non_forward_before"]
                ),
                seed_blocked=False,
                before_gap_s=before_gap,
                after_gap_s=None,
                before_distance_m=None,
                after_distance_m=None,
                geometry_support=0.0,
                temporal_support=0.0,
                family_relation="unknown",
                family_support=.50,
                reid_cosine=None,
                reid_support=.50,
            )
        before_distance, before_support, hard = _transition_support(
            obs,
            target.observations[0],
            dt_s=before_gap,
            config=config,
        )
        if hard:
            blocker_kinds.append("before_geometry")

    if after is not None:
        obs, _ = after
        after_gap = obs.timestamp_s - target.end_time
        if after_gap <= config.time_epsilon_s:
            blocker_kinds.append("non_forward_after")
        else:
            after_distance, after_support, hard = _transition_support(
                target.observations[-1],
                obs,
                dt_s=after_gap,
                config=config,
            )
            if hard:
                blocker_kinds.append("after_geometry")

    hard_geometry = any(
        kind in {"before_geometry", "after_geometry", "non_forward_after"}
        for kind in blocker_kinds
    )
    if hard_geometry:
        return ActorCandidate(
            actor_id=actor_id,
            tracklet_id=target.tracklet_id,
            viable=False,
            score=0.0,
            utility=0.0,
            roster_evidence=_roster_evidence(
                target,
                family_by_tracklet,
                config,
            ),
            blockers=tuple(sorted(blockers)),
            blocker_kinds=tuple(blocker_kinds),
            seed_blocked=False,
            before_gap_s=before_gap,
            after_gap_s=after_gap,
            before_distance_m=before_distance,
            after_distance_m=after_distance,
            geometry_support=0.0,
            temporal_support=0.0,
            family_relation="geometry_blocked",
            family_support=0.0,
            reid_cosine=None,
            reid_support=.50,
        )

    geometry_terms = []
    temporal_terms = []

    if before is not None and before_gap is not None:
        geometry_terms.append(before_support)
        temporal_terms.append(
            math.exp(-before_gap / 1.6)
            if before_gap <= config.context_window_s
            else .08
        )
    if after is not None and after_gap is not None:
        geometry_terms.append(after_support)
        temporal_terms.append(
            math.exp(-after_gap / 1.6)
            if after_gap <= config.context_window_s
            else .08
        )

    geometry_support = (
        sum(geometry_terms) / len(geometry_terms)
        if geometry_terms
        else .15
    )
    temporal_support = (
        sum(temporal_terms) / len(temporal_terms)
        if temporal_terms
        else .10
    )

    relation, family_support = _family_relation(
        context.family,
        _family(family_by_tracklet.get(target.tracklet_id)),
    )

    reid_cosine = None
    reid_support = .50
    target_appearance = _appearance(
        appearance_by_tracklet.get(target.tracklet_id)
    )
    if context.appearance is not None and target_appearance is not None:
        reid_cosine = float(np.clip(
            np.dot(
                context.appearance,
                target_appearance.embedding,
            ),
            -1.0,
            1.0,
        ))
        reid_support = _clamp01((reid_cosine + 1.0) * .5)

    weights = (
        config.geometry_weight,
        config.temporal_weight,
        config.family_weight,
        config.reid_weight,
    )
    score = (
        weights[0] * geometry_support
        + weights[1] * temporal_support
        + weights[2] * family_support
        + weights[3] * reid_support
    ) / sum(weights)

    roster_evidence = _roster_evidence(
        target,
        family_by_tracklet,
        config,
    )
    utility = (
        config.candidate_score_weight * score
        + config.roster_evidence_weight * roster_evidence
    )

    return ActorCandidate(
        actor_id=actor_id,
        tracklet_id=target.tracklet_id,
        viable=True,
        score=float(_clamp01(score)),
        utility=float(_clamp01(utility)),
        roster_evidence=roster_evidence,
        blockers=tuple(sorted(blockers)),
        blocker_kinds=tuple(blocker_kinds),
        seed_blocked=False,
        before_gap_s=before_gap,
        after_gap_s=after_gap,
        before_distance_m=before_distance,
        after_distance_m=after_distance,
        geometry_support=float(geometry_support),
        temporal_support=float(temporal_support),
        family_relation=relation,
        family_support=float(family_support),
        reid_cosine=reid_cosine,
        reid_support=float(reid_support),
    )


def _current_assignment_utility(
    *,
    tid: str,
    assignment: Mapping[str, str | None],
    actor_ids: Sequence[str],
    contexts: Mapping[str, _ActorContext],
    tracklets: Mapping[str, _Tracklet],
    seed_ids: frozenset[str],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: Mapping[str, set[str]],
    config: GlobalClosedRosterConfig,
) -> float:
    actor_id = assignment.get(tid)
    target = tracklets[tid]

    if actor_id is None:
        return _non_roster_utility(
            target,
            contexts,
            family_by_tracklet,
            config,
            current_actor=None,
        )

    candidate = _candidate_for_actor(
        target=target,
        actor_id=actor_id,
        assignment=assignment,
        contexts=contexts,
        tracklets=tracklets,
        seed_ids=seed_ids,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
        hard_conflicts=hard_conflicts,
        config=config,
    )
    return candidate.utility if candidate.viable else 0.0


def _candidate_list(
    *,
    tid: str,
    assignment: Mapping[str, str | None],
    actor_ids: Sequence[str],
    contexts: Mapping[str, _ActorContext],
    tracklets: Mapping[str, _Tracklet],
    seed_ids: frozenset[str],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: Mapping[str, set[str]],
    config: GlobalClosedRosterConfig,
) -> list[ActorCandidate]:
    target = tracklets[tid]
    result = [
        _candidate_for_actor(
            target=target,
            actor_id=actor_id,
            assignment=assignment,
            contexts=contexts,
            tracklets=tracklets,
            seed_ids=seed_ids,
            family_by_tracklet=family_by_tracklet,
            appearance_by_tracklet=appearance_by_tracklet,
            hard_conflicts=hard_conflicts,
            config=config,
        )
        for actor_id in actor_ids
    ]
    return sorted(
        (row for row in result if row.viable),
        key=lambda row: (
            -row.utility,
            -row.score,
            row.actor_id,
        ),
    )


@dataclass(frozen=True, slots=True)
class _Proposal:
    move_type: str
    tid: str
    from_actor: str | None
    to_actor: str | None
    swap_tid: str | None
    evicted: tuple[str, ...]
    old_utility: float
    new_utility: float
    delta: float
    candidate: ActorCandidate | None


def _best_proposal(
    *,
    tid: str,
    assignment: Mapping[str, str | None],
    actor_ids: Sequence[str],
    contexts: Mapping[str, _ActorContext],
    tracklets: Mapping[str, _Tracklet],
    seed_ids: frozenset[str],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: Mapping[str, set[str]],
    config: GlobalClosedRosterConfig,
) -> _Proposal | None:
    if tid in seed_ids:
        return None

    target = tracklets[tid]
    current_actor = assignment.get(tid)
    old_utility = _current_assignment_utility(
        tid=tid,
        assignment=assignment,
        actor_ids=actor_ids,
        contexts=contexts,
        tracklets=tracklets,
        seed_ids=seed_ids,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
        hard_conflicts=hard_conflicts,
        config=config,
    )

    best_proposal = None

    non_roster = _non_roster_utility(
        target,
        contexts,
        family_by_tracklet,
        config,
        current_actor=current_actor,
    )
    if (
        current_actor is not None
        and non_roster >= old_utility + config.non_roster_move_margin
    ):
        best_proposal = _Proposal(
            move_type="to_non_roster",
            tid=tid,
            from_actor=current_actor,
            to_actor=None,
            swap_tid=None,
            evicted=(),
            old_utility=old_utility,
            new_utility=non_roster,
            delta=non_roster - old_utility,
            candidate=None,
        )

    for candidate in _candidate_list(
        tid=tid,
        assignment=assignment,
        actor_ids=actor_ids,
        contexts=contexts,
        tracklets=tracklets,
        seed_ids=seed_ids,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
        hard_conflicts=hard_conflicts,
        config=config,
    ):
        if candidate.actor_id == current_actor and not candidate.blockers:
            continue

        if not candidate.blockers:
            delta = candidate.utility - old_utility
            if delta >= config.direct_move_margin:
                proposal = _Proposal(
                    move_type="direct_reassign",
                    tid=tid,
                    from_actor=current_actor,
                    to_actor=candidate.actor_id,
                    swap_tid=None,
                    evicted=(),
                    old_utility=old_utility,
                    new_utility=candidate.utility,
                    delta=delta,
                    candidate=candidate,
                )
                if (
                    best_proposal is None
                    or proposal.delta > best_proposal.delta
                ):
                    best_proposal = proposal
            continue

        blockers = tuple(candidate.blockers)
        if any(blocker in seed_ids for blocker in blockers):
            continue

        # Exact one-for-one swap: target takes candidate actor while its one
        # blocker is tested against target's current actor.
        if (
            len(blockers) == 1
            and current_actor is not None
        ):
            blocker_tid = blockers[0]
            blocker_candidate = _candidate_for_actor(
                target=tracklets[blocker_tid],
                actor_id=current_actor,
                assignment=assignment,
                contexts=contexts,
                tracklets=tracklets,
                seed_ids=seed_ids,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=hard_conflicts,
                config=config,
            )

            # The target itself is allowed to be the only blocker in the swap.
            blocker_other_blockers = tuple(
                value
                for value in blocker_candidate.blockers
                if value != tid
            )

            if blocker_candidate.viable and not blocker_other_blockers:
                blocker_old = _current_assignment_utility(
                    tid=blocker_tid,
                    assignment=assignment,
                    actor_ids=actor_ids,
                    contexts=contexts,
                    tracklets=tracklets,
                    seed_ids=seed_ids,
                    family_by_tracklet=family_by_tracklet,
                    appearance_by_tracklet=appearance_by_tracklet,
                    hard_conflicts=hard_conflicts,
                    config=config,
                )
                new_total = (
                    candidate.utility + blocker_candidate.utility
                )
                old_total = old_utility + blocker_old
                delta = new_total - old_total

                if delta >= config.swap_move_margin:
                    proposal = _Proposal(
                        move_type="swap",
                        tid=tid,
                        from_actor=current_actor,
                        to_actor=candidate.actor_id,
                        swap_tid=blocker_tid,
                        evicted=(),
                        old_utility=old_total,
                        new_utility=new_total,
                        delta=delta,
                        candidate=candidate,
                    )
                    if (
                        best_proposal is None
                        or proposal.delta > best_proposal.delta
                    ):
                        best_proposal = proposal

        # Eviction challenge.  This is intentionally conservative: at most
        # `max_challenge_blockers` mutable blockers can be displaced.
        blocker_old_total = 0.0
        blocker_non_roster_total = 0.0
        valid = True
        for blocker_tid in blockers:
            blocker_old_total += _current_assignment_utility(
                tid=blocker_tid,
                assignment=assignment,
                actor_ids=actor_ids,
                contexts=contexts,
                tracklets=tracklets,
                seed_ids=seed_ids,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=hard_conflicts,
                config=config,
            )
            blocker_actor = assignment.get(blocker_tid)
            if blocker_actor is None:
                valid = False
                break
            blocker_non_roster_total += _non_roster_utility(
                tracklets[blocker_tid],
                contexts,
                family_by_tracklet,
                config,
                current_actor=blocker_actor,
            )

        if valid:
            new_total = candidate.utility + blocker_non_roster_total
            old_total = old_utility + blocker_old_total
            delta = new_total - old_total
            if delta >= config.eviction_move_margin:
                proposal = _Proposal(
                    move_type="evict_and_reassign",
                    tid=tid,
                    from_actor=current_actor,
                    to_actor=candidate.actor_id,
                    swap_tid=None,
                    evicted=blockers,
                    old_utility=old_total,
                    new_utility=new_total,
                    delta=delta,
                    candidate=candidate,
                )
                if (
                    best_proposal is None
                    or proposal.delta > best_proposal.delta
                ):
                    best_proposal = proposal

    return best_proposal


def _rebuild_affected_contexts(
    *,
    affected_actor_ids: set[str],
    contexts: dict[str, _ActorContext],
    assignment: Mapping[str, str | None],
    tracklets: Mapping[str, _Tracklet],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
) -> None:
    for actor_id in affected_actor_ids:
        contexts[actor_id] = _build_context(
            actor_id=actor_id,
            assignment=assignment,
            tracklets=tracklets,
            family_by_tracklet=family_by_tracklet,
            appearance_by_tracklet=appearance_by_tracklet,
        )


def _apply_proposal(
    *,
    proposal: _Proposal,
    round_index: int,
    assignment: dict[str, str | None],
    contexts: dict[str, _ActorContext],
    actor_ids: Sequence[str],
    tracklets: Mapping[str, _Tracklet],
    seed_ids: frozenset[str],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: Mapping[str, set[str]],
    config: GlobalClosedRosterConfig,
) -> ReassignmentMove | None:
    # Recompute on the fresh partition.  Proposals are generated synchronously,
    # but earlier accepted moves in the same round may have changed the scene.
    fresh = _best_proposal(
        tid=proposal.tid,
        assignment=assignment,
        actor_ids=actor_ids,
        contexts=contexts,
        tracklets=tracklets,
        seed_ids=seed_ids,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
        hard_conflicts=hard_conflicts,
        config=config,
    )
    if fresh is None:
        return None
    if fresh.move_type != proposal.move_type:
        # Accept the fresh best move instead of applying stale bookkeeping.
        proposal = fresh
    else:
        proposal = fresh

    affected = set()
    from_actor = assignment.get(proposal.tid)
    if from_actor is not None:
        affected.add(from_actor)

    if proposal.move_type == "to_non_roster":
        assignment[proposal.tid] = None

    elif proposal.move_type == "direct_reassign":
        if proposal.to_actor is None:
            return None
        assignment[proposal.tid] = proposal.to_actor
        affected.add(proposal.to_actor)

    elif proposal.move_type == "swap":
        if (
            proposal.to_actor is None
            or proposal.swap_tid is None
            or from_actor is None
        ):
            return None
        blocker_actor = assignment.get(proposal.swap_tid)
        if blocker_actor != proposal.to_actor:
            return None

        assignment[proposal.tid] = proposal.to_actor
        assignment[proposal.swap_tid] = from_actor
        affected.update({proposal.to_actor, from_actor})

    elif proposal.move_type == "evict_and_reassign":
        if proposal.to_actor is None:
            return None
        for blocker_tid in proposal.evicted:
            blocker_actor = assignment.get(blocker_tid)
            if blocker_actor is not None:
                affected.add(blocker_actor)
            assignment[blocker_tid] = None

        assignment[proposal.tid] = proposal.to_actor
        affected.add(proposal.to_actor)

    else:
        raise ValueError(f"unknown move type {proposal.move_type}")

    _rebuild_affected_contexts(
        affected_actor_ids=affected,
        contexts=contexts,
        assignment=assignment,
        tracklets=tracklets,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
    )

    return ReassignmentMove(
        round_index=round_index,
        move_type=proposal.move_type,
        tracklet_id=proposal.tid,
        from_actor_id=from_actor,
        to_actor_id=assignment.get(proposal.tid),
        swapped_tracklet_id=proposal.swap_tid,
        evicted_tracklet_ids=proposal.evicted,
        old_utility=proposal.old_utility,
        new_utility=proposal.new_utility,
        delta_utility=proposal.delta,
        candidate=proposal.candidate,
    )


def _validate_partition(
    *,
    actor_ids: Sequence[str],
    assignment: Mapping[str, str | None],
    tracklets: Mapping[str, _Tracklet],
    seed_ids: frozenset[str],
    seed_owner: Mapping[str, str],
) -> None:
    if set(assignment) != set(tracklets):
        raise ValueError("assignment must contain every tracklet")

    for tid in seed_ids:
        if assignment.get(tid) != seed_owner[tid]:
            raise ValueError(f"seed {tid} moved away from {seed_owner[tid]}")

    for actor_id in actor_ids:
        frame_owner: dict[int, str] = {}
        for tid, owner in assignment.items():
            if owner != actor_id:
                continue
            for frame in tracklets[tid].frames:
                previous = frame_owner.get(frame)
                if previous is not None and previous != tid:
                    raise ValueError(
                        f"{actor_id} exact-frame coexistence at frame "
                        f"{frame}: {previous}, {tid}"
                    )
                frame_owner[frame] = tid


def _trajectory_rows(
    *,
    actor_ids: Sequence[str],
    assignment: Mapping[str, str | None],
    tracklets: Mapping[str, _Tracklet],
    config: GlobalClosedRosterConfig,
) -> tuple[dict[str, Any], ...]:
    rows = []

    for actor_id in actor_ids:
        observations = sorted(
            (
                (obs, tid)
                for tid, owner in assignment.items()
                if owner == actor_id
                for obs in tracklets[tid].observations
            ),
            key=lambda item: (
                item[0].frame_index,
                item[0].timestamp_s,
                item[1],
            ),
        )

        points = []
        for index, (obs, tid) in enumerate(observations):
            next_gap_s = None
            next_distance_m = None
            next_speed_mps = None
            interpolate_to_next = False

            if index + 1 < len(observations):
                nxt, _ = observations[index + 1]
                next_gap_s = nxt.timestamp_s - obs.timestamp_s
                next_distance_m = math.hypot(
                    nxt.x - obs.x,
                    nxt.z - obs.z,
                )
                if next_gap_s > config.time_epsilon_s:
                    next_speed_mps = (
                        next_distance_m / next_gap_s
                    )
                    interpolate_to_next = (
                        next_gap_s
                        <= config.visual_interpolation_max_gap_s
                        + config.time_epsilon_s
                        and next_speed_mps
                        <= config.visual_interpolation_max_speed_mps
                    )

            points.append({
                "frame_index": obs.frame_index,
                "timestamp_s": obs.timestamp_s,
                "x": obs.x,
                "z": obs.z,
                "confidence": obs.confidence,
                "detection_id": obs.detection_id,
                "tracklet_id": tid,
                "next_gap_s": next_gap_s,
                "next_distance_m": next_distance_m,
                "next_speed_mps": next_speed_mps,
                "interpolate_to_next": interpolate_to_next,
            })

        rows.append({
            "schema_version": 1,
            "trajectory_version": "global-closed-roster-anchors-v1",
            "actor_id": actor_id,
            "tracklet_ids": sorted(
                tid
                for tid, owner in assignment.items()
                if owner == actor_id
            ),
            "point_count": len(points),
            "points": points,
            "visual_interpolation_max_gap_s": (
                config.visual_interpolation_max_gap_s
            ),
            "visual_interpolation_max_speed_mps": (
                config.visual_interpolation_max_speed_mps
            ),
        })

    return tuple(rows)


def optimize_global_closed_roster(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    fixed_actor_rows: Sequence[Mapping[str, Any]],
    fixed_assignment_rows: Sequence[Mapping[str, Any]],
    initial_assignment_rows: Sequence[Mapping[str, Any]],
    initial_non_roster_rows: Sequence[Mapping[str, Any]] = (),
    family_rows: Sequence[Mapping[str, Any]] = (),
    appearance_rows: Sequence[Mapping[str, Any]] = (),
    hard_conflicts: Sequence[Mapping[str, Any]] = (),
    config: GlobalClosedRosterConfig | None = None,
) -> GlobalClosedRosterResult:
    selected = config or GlobalClosedRosterConfig()

    tracklets = {
        parsed.tracklet_id: parsed
        for parsed in (_parse_tracklet(row) for row in tracklet_rows)
    }
    if len(tracklets) != len(tracklet_rows):
        raise ValueError("duplicate tracklet IDs")

    actor_ids = tuple(sorted(
        str(row["actor_id"])
        for row in fixed_actor_rows
    ))
    if len(actor_ids) != selected.expected_actor_slots:
        raise ValueError(
            f"expected {selected.expected_actor_slots} fixed actors, "
            f"got {len(actor_ids)}"
        )
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("duplicate fixed actor IDs")

    family_by_tracklet = {
        str(row["tracklet_id"]): row
        for row in family_rows
    }
    appearance_by_tracklet = {
        str(row["tracklet_id"]): row
        for row in appearance_rows
    }
    conflict_index = _hard_conflict_index(hard_conflicts)

    # Immutable seeds come only from fixed_actor_roster actor_sequence_index=0.
    seed_owner: dict[str, str] = {}
    for row in fixed_assignment_rows:
        if int(row.get("actor_sequence_index", -1)) != 0:
            continue
        tid = str(row["tracklet_id"])
        actor_id = str(row["actor_id"])
        if actor_id not in actor_ids:
            raise ValueError(f"unknown seed actor {actor_id}")
        if tid not in tracklets:
            raise ValueError(f"unknown seed tracklet {tid}")
        if tid in seed_owner:
            raise ValueError(f"duplicate seed tracklet {tid}")
        seed_owner[tid] = actor_id

    if len(seed_owner) != selected.expected_actor_slots:
        raise ValueError(
            f"expected exactly {selected.expected_actor_slots} seeds, "
            f"got {len(seed_owner)}"
        )
    if set(seed_owner.values()) != set(actor_ids):
        raise ValueError("each fixed actor must own exactly one seed")

    seed_ids = frozenset(seed_owner)

    # Start from 7I.5, but every non-seed assignment is mutable.
    assignment: dict[str, str | None] = {
        tid: None for tid in tracklets
    }

    for row in initial_assignment_rows:
        tid = str(row["tracklet_id"])
        actor_id = str(row["actor_id"])
        if tid not in tracklets:
            raise ValueError(f"initial assignment references unknown {tid}")
        if actor_id not in actor_ids:
            raise ValueError(
                f"initial assignment references unknown actor {actor_id}"
            )
        assignment[tid] = actor_id

    for row in initial_non_roster_rows:
        tid = str(row["tracklet_id"])
        if tid not in tracklets:
            raise ValueError(f"NON_ROSTER references unknown {tid}")
        if tid in seed_ids:
            raise ValueError("seed cannot be NON_ROSTER")
        assignment[tid] = None

    # Seeds override everything and are immutable.
    for tid, actor_id in seed_owner.items():
        assignment[tid] = actor_id

    initial_owner = dict(assignment)

    _validate_partition(
        actor_ids=actor_ids,
        assignment=assignment,
        tracklets=tracklets,
        seed_ids=seed_ids,
        seed_owner=seed_owner,
    )

    contexts = _build_all_contexts(
        actor_ids=actor_ids,
        assignment=assignment,
        tracklets=tracklets,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
    )

    moves: list[ReassignmentMove] = []
    rounds_executed = 0

    mutable_ids = tuple(
        tid for tid in sorted(tracklets)
        if tid not in seed_ids
    )

    for round_index in range(1, selected.max_rounds + 1):
        rounds_executed = round_index
        proposals = []

        for tid in mutable_ids:
            proposal = _best_proposal(
                tid=tid,
                assignment=assignment,
                actor_ids=actor_ids,
                contexts=contexts,
                tracklets=tracklets,
                seed_ids=seed_ids,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=conflict_index,
                config=selected,
            )
            if proposal is not None:
                proposals.append(proposal)

        if not proposals:
            break

        proposals.sort(
            key=lambda row: (
                -row.delta,
                row.move_type,
                row.tid,
            )
        )

        accepted_this_round = 0
        touched: set[str] = set()

        for proposal in proposals:
            related = {
                proposal.tid,
                *proposal.evicted,
            }
            if proposal.swap_tid is not None:
                related.add(proposal.swap_tid)

            # Prevent one tracklet from being modified repeatedly inside the
            # same synchronous round.  It can still be reconsidered next round.
            if related.intersection(touched):
                continue

            move = _apply_proposal(
                proposal=proposal,
                round_index=round_index,
                assignment=assignment,
                contexts=contexts,
                actor_ids=actor_ids,
                tracklets=tracklets,
                seed_ids=seed_ids,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=conflict_index,
                config=selected,
            )
            if move is None:
                continue

            moves.append(move)
            accepted_this_round += 1
            touched.update(related)

        _validate_partition(
            actor_ids=actor_ids,
            assignment=assignment,
            tracklets=tracklets,
            seed_ids=seed_ids,
            seed_owner=seed_owner,
        )

        if accepted_this_round == 0:
            break

    # Final review: no unresolved state exists, but low-confidence actor
    # assignments remain explicitly marked for visual audit.
    contexts = _build_all_contexts(
        actor_ids=actor_ids,
        assignment=assignment,
        tracklets=tracklets,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
    )

    final_review_rows = []
    for tid in mutable_ids:
        actor_id = assignment[tid]
        if actor_id is None:
            non_roster = _non_roster_utility(
                tracklets[tid],
                contexts,
                family_by_tracklet,
                selected,
                current_actor=None,
            )
            roster_evidence = _roster_evidence(
                tracklets[tid],
                family_by_tracklet,
                selected,
            )
            reasons = []
            if roster_evidence >= .65 and non_roster < .55:
                reasons.append("infield_roster_like_non_roster")
            final_review_rows.append({
                "tracklet_id": tid,
                "status": "non_roster",
                "actor_id": None,
                "needs_review": bool(reasons),
                "review_reasons": reasons,
                "non_roster_utility": non_roster,
                "roster_evidence": roster_evidence,
                "assignment_utility": None,
                "best_alternative_actor_id": None,
                "best_alternative_utility": None,
                "actor_margin": None,
            })
            continue

        current = _candidate_for_actor(
            target=tracklets[tid],
            actor_id=actor_id,
            assignment=assignment,
            contexts=contexts,
            tracklets=tracklets,
            seed_ids=seed_ids,
            family_by_tracklet=family_by_tracklet,
            appearance_by_tracklet=appearance_by_tracklet,
            hard_conflicts=conflict_index,
            config=selected,
        )

        alternatives = [
            row
            for row in _candidate_list(
                tid=tid,
                assignment=assignment,
                actor_ids=actor_ids,
                contexts=contexts,
                tracklets=tracklets,
                seed_ids=seed_ids,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=conflict_index,
                config=selected,
            )
            if row.actor_id != actor_id and not row.blockers
        ]
        best_alt = alternatives[0] if alternatives else None

        margin = (
            current.utility - best_alt.utility
            if current.viable and best_alt is not None
            else current.utility
            if current.viable
            else -1.0
        )

        reasons = []
        if not current.viable:
            reasons.append("current_assignment_not_viable")
        if (
            current.viable
            and current.utility
            < selected.review_assignment_utility_floor
        ):
            reasons.append("low_assignment_utility")
        if (
            current.viable
            and best_alt is not None
            and margin < selected.review_assignment_margin
        ):
            reasons.append("small_actor_margin")

        final_review_rows.append({
            "tracklet_id": tid,
            "status": "assigned",
            "actor_id": actor_id,
            "needs_review": bool(reasons),
            "review_reasons": reasons,
            "non_roster_utility": _non_roster_utility(
                tracklets[tid],
                contexts,
                family_by_tracklet,
                selected,
                current_actor=actor_id,
            ),
            "roster_evidence": _roster_evidence(
                tracklets[tid],
                family_by_tracklet,
                selected,
            ),
            "assignment_utility": (
                current.utility if current.viable else None
            ),
            "best_alternative_actor_id": (
                best_alt.actor_id if best_alt is not None else None
            ),
            "best_alternative_utility": (
                best_alt.utility if best_alt is not None else None
            ),
            "actor_margin": margin,
        })

    actor_rows = []
    for actor_id in actor_ids:
        member_ids = sorted(
            tid
            for tid, owner in assignment.items()
            if owner == actor_id
        )
        actor_rows.append({
            "schema_version": 1,
            "reassignment_version": GLOBAL_REASSIGNMENT_VERSION,
            "actor_id": actor_id,
            "seed_tracklet_id": next(
                tid
                for tid, owner in seed_owner.items()
                if owner == actor_id
            ),
            "tracklet_ids": member_ids,
            "tracklet_count": len(member_ids),
            "provisional": True,
        })

    assignment_rows = tuple(
        {
            "schema_version": 1,
            "reassignment_version": GLOBAL_REASSIGNMENT_VERSION,
            "tracklet_id": tid,
            "actor_id": owner,
            "assignment_status": "assigned",
            "seed": tid in seed_ids,
            "initial_actor_id": initial_owner.get(tid),
            "changed_from_initial": initial_owner.get(tid) != owner,
            "provisional": True,
        }
        for tid, owner in sorted(assignment.items())
        if owner is not None
    )

    non_roster_rows = tuple(
        {
            "schema_version": 1,
            "reassignment_version": GLOBAL_REASSIGNMENT_VERSION,
            "tracklet_id": tid,
            "identity_id": NON_ROSTER_ID,
            "assignment_status": "non_roster",
            "initial_actor_id": initial_owner.get(tid),
            "changed_from_initial": initial_owner.get(tid) is not None,
            "provisional": True,
        }
        for tid, owner in sorted(assignment.items())
        if owner is None
    )

    trajectories = _trajectory_rows(
        actor_ids=actor_ids,
        assignment=assignment,
        tracklets=tracklets,
        config=selected,
    )

    return GlobalClosedRosterResult(
        actors=tuple(actor_rows),
        tracklet_assignments=assignment_rows,
        non_roster_tracklets=non_roster_rows,
        moves=tuple(moves),
        actor_trajectories=trajectories,
        seed_tracklet_ids=tuple(sorted(seed_ids)),
        initial_owner=initial_owner,
        final_owner=dict(assignment),
        rounds_executed=rounds_executed,
        final_review_rows=tuple(final_review_rows),
        config=selected,
    )
