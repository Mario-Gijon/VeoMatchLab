"""Tracking V1 Hito 7I.5: closed-roster offline resolution.

The world is deliberately CLOSED for this experiment:

- exactly the actor slots produced by fixed-actor-roster-v1 exist;
- an unresolved tracklet must end as either:
    1) one of those actor slots, or
    2) NON_ROSTER (coach/bench/noise/duplicate/etc.);
- no new actor ID is ever created.

The resolver is deliberately geometry/scene first. ReID is only a small
confirmation/tie-break term.

Resolution order
----------------
1. Exact-frame coexistence and existing hard cannot-link constraints.
2. Before/after actor continuity in field X/Z.
3. Kit-family compatibility.
4. Closed-roster competition.
5. ReID as confirmation/tie-break only.
6. Strong NON_ROSTER evidence for short/off-field/fully-occupied extras.

The stage is offline: both the past and the future of every actor are available.
That makes "actor before -> unresolved fragment -> same actor after" a strong
piece of evidence and avoids solving every handoff only forwards in time.

No input artifact is modified.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CLOSED_ROSTER_VERSION = "closed-roster-resolution-v1"
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


def _stats(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "min": None, "median": None, "p90": None,
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
        "median": pct(.50),
        "p90": pct(.90),
        "p95": pct(.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


@dataclass(frozen=True, slots=True)
class ClosedRosterConfig:
    expected_actor_slots: int = 23

    field_length_m: float = 105.0
    field_width_m: float = 68.0
    off_field_margin_m: float = 0.75

    max_transition_speed_mps: float = 12.0
    position_slack_m: float = 1.25
    hard_geometry_window_s: float = 2.5
    context_window_s: float = 6.0

    geometry_weight: float = 0.52
    temporal_weight: float = 0.18
    family_weight: float = 0.22
    reid_weight: float = 0.08

    bridge_floor: float = 0.56
    one_sided_floor: float = 0.58
    unique_actor_floor: float = 0.45

    assignment_margin: float = 0.035
    reid_tie_break_margin: float = 0.015

    strong_non_roster_floor: float = 0.72
    non_roster_margin: float = 0.08
    short_extra_s: float = 2.0
    very_short_extra_s: float = 0.8
    off_field_strong_ratio: float = 0.80
    occupancy_strong_ratio: float = 0.80

    max_internal_speed_mps: float = 15.0

    visual_interpolation_max_gap_s: float = 0.50
    visual_interpolation_max_speed_mps: float = 12.5

    max_iterations: int = 12
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
            "max_internal_speed_mps",
            "visual_interpolation_max_gap_s",
            "visual_interpolation_max_speed_mps",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.off_field_margin_m < 0:
            raise ValueError("off_field_margin_m must be >= 0")
        weights = (
            self.geometry_weight,
            self.temporal_weight,
            self.family_weight,
            self.reid_weight,
        )
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("evidence weights must be non-negative")
        for name in (
            "bridge_floor",
            "one_sided_floor",
            "unique_actor_floor",
            "assignment_margin",
            "reid_tie_break_margin",
            "strong_non_roster_floor",
            "non_roster_margin",
            "off_field_strong_ratio",
            "occupancy_strong_ratio",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")

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
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def start_timestamp_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_timestamp_s(self) -> float:
        return self.observations[-1].timestamp_s

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_timestamp_s - self.start_timestamp_s)

    @property
    def frame_set(self) -> set[int]:
        return {obs.frame_index for obs in self.observations}


def _parse_tracklet(row: Mapping[str, Any]) -> _Tracklet:
    tid = str(row["tracklet_id"])
    raw_observations = row.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise ValueError(f"{tid} has no observations")

    observations = []
    previous_frame = None
    previous_time = None

    for raw in raw_observations:
        obs = _Observation(
            detection_id=str(raw["detection_id"]),
            frame_index=int(raw["frame_index"]),
            timestamp_s=float(raw["timestamp_s"]),
            x=float(raw["x"]),
            z=float(raw["z"]),
            confidence=float(raw.get("confidence", 1.0)),
        )
        if obs.frame_index < 0:
            raise ValueError(f"{tid} has negative frame")
        if not all(math.isfinite(v) for v in (
            obs.timestamp_s, obs.x, obs.z, obs.confidence
        )):
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


def _appearance(
    row: Mapping[str, Any] | None,
) -> _Appearance | None:
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


@dataclass(slots=True)
class _ActorState:
    actor_id: str
    tracklet_ids: list[str] = field(default_factory=list)
    occupied_frames: set[int] = field(default_factory=set)
    observations: list[_Observation] = field(default_factory=list)
    observation_frames: list[int] = field(default_factory=list)
    family_mass: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    confident_family_mass: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    appearance_sum: np.ndarray | None = None
    appearance_weight: float = 0.0

    def add(
        self,
        tracklet: _Tracklet,
        family: _Family,
        appearance: _Appearance | None,
    ) -> None:
        overlap = self.occupied_frames.intersection(tracklet.frame_set)
        if overlap:
            raise ValueError(
                f"{self.actor_id} exact-frame coexistence at {min(overlap)}"
            )

        self.tracklet_ids.append(tracklet.tracklet_id)
        self.occupied_frames.update(tracklet.frame_set)

        merged = sorted(
            [*self.observations, *tracklet.observations],
            key=lambda obs: (obs.frame_index, obs.timestamp_s, obs.detection_id),
        )
        if len({obs.frame_index for obs in merged}) != len(merged):
            raise ValueError(
                f"{self.actor_id} would contain duplicate observation frames"
            )
        self.observations = merged
        self.observation_frames = [obs.frame_index for obs in merged]

        if family.key is not None and family.mass > 0:
            self.family_mass[family.key] += family.mass
            if family.confident:
                self.confident_family_mass[family.key] += family.mass

        if appearance is not None:
            weight = max(
                1e-6,
                appearance.reliability
                * math.sqrt(appearance.effective_sample_count),
            )
            weighted = appearance.embedding * weight
            if self.appearance_sum is None:
                self.appearance_sum = weighted.copy()
            else:
                self.appearance_sum += weighted
            self.appearance_weight += weight

    def previous_observation(
        self,
        frame_index: int,
    ) -> _Observation | None:
        index = bisect_left(self.observation_frames, frame_index) - 1
        return self.observations[index] if index >= 0 else None

    def next_observation(
        self,
        frame_index: int,
    ) -> _Observation | None:
        index = bisect_right(self.observation_frames, frame_index)
        return (
            self.observations[index]
            if index < len(self.observations)
            else None
        )

    def family(self) -> _Family:
        if not self.family_mass:
            return _Family(None, False, 0.0)

        winner = max(
            self.family_mass,
            key=lambda key: (self.family_mass[key], key),
        )
        total = sum(self.family_mass.values())
        dominance = self.family_mass[winner] / total if total else 0.0

        confident_total = sum(self.confident_family_mass.values())
        confident_winner = self.confident_family_mass.get(winner, 0.0)
        confident_dominance = (
            confident_winner / confident_total
            if confident_total > 0
            else 0.0
        )
        confident = (
            confident_winner >= .90
            and dominance >= .70
            and (
                confident_total <= 1e-12
                or confident_dominance >= .75
            )
        )
        return _Family(winner, confident, float(dominance))

    def appearance(self) -> np.ndarray | None:
        if self.appearance_sum is None:
            return None
        return _unit(self.appearance_sum)


@dataclass(frozen=True, slots=True)
class ActorCandidate:
    actor_id: str
    tracklet_id: str
    score: float

    before_available: bool
    after_available: bool
    before_gap_s: float | None
    after_gap_s: float | None
    before_distance_m: float | None
    after_distance_m: float | None
    before_support: float
    after_support: float
    geometry_support: float
    temporal_support: float

    family_relation: str
    family_support: float

    reid_cosine: float | None
    reid_support: float

    physically_viable: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ResolutionDecision:
    tracklet_id: str
    status: str
    actor_id: str | None
    rule: str
    iteration: int
    needs_review: bool

    best_actor_score: float | None
    second_actor_score: float | None
    actor_margin: float | None

    non_roster_score: float
    off_field_ratio: float
    full_roster_occupancy_ratio: float
    shortness_support: float
    internal_motion_penalty: float

    candidate_count: int
    best_candidate: ActorCandidate | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "resolution_version": CLOSED_ROSTER_VERSION,
            "tracklet_id": self.tracklet_id,
            "status": self.status,
            "actor_id": self.actor_id,
            "rule": self.rule,
            "iteration": self.iteration,
            "needs_review": self.needs_review,
            "best_actor_score": self.best_actor_score,
            "second_actor_score": self.second_actor_score,
            "actor_margin": self.actor_margin,
            "non_roster_score": self.non_roster_score,
            "off_field_ratio": self.off_field_ratio,
            "full_roster_occupancy_ratio": self.full_roster_occupancy_ratio,
            "shortness_support": self.shortness_support,
            "internal_motion_penalty": self.internal_motion_penalty,
            "candidate_count": self.candidate_count,
            "best_candidate": (
                self.best_candidate.to_dict()
                if self.best_candidate is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ClosedRosterResult:
    actors: tuple[dict[str, Any], ...]
    tracklet_assignments: tuple[dict[str, Any], ...]
    non_roster_tracklets: tuple[dict[str, Any], ...]
    decisions: tuple[ResolutionDecision, ...]
    actor_trajectories: tuple[dict[str, Any], ...]
    initial_assignment_count: int
    recovered_assignment_count: int
    config: ClosedRosterConfig

    def summary(self) -> dict[str, Any]:
        review_count = sum(row.needs_review for row in self.decisions)
        input_tracklet_count = (
            len(self.tracklet_assignments)
            + len(self.non_roster_tracklets)
        )
        return {
            "schema_version": 1,
            "resolution_version": CLOSED_ROSTER_VERSION,
            "actor_slot_count": len(self.actors),
            "actor_count_is_fixed": True,
            "input_tracklet_count": input_tracklet_count,
            "initial_assignment_count": self.initial_assignment_count,
            "recovered_assignment_count": self.recovered_assignment_count,
            "assigned_tracklet_count": len(self.tracklet_assignments),
            "non_roster_tracklet_count": len(self.non_roster_tracklets),
            "unresolved_tracklet_count": 0,
            "assignment_or_non_roster_complete": True,
            "new_actor_creation": False,
            "reid_role": "confirmation_tie_break_only",
            "needs_review_count": review_count,
            "decision_rule_counts": dict(sorted(Counter(
                row.rule for row in self.decisions
            ).items())),
            "decision_status_counts": dict(sorted(Counter(
                row.status for row in self.decisions
            ).items())),
            "recovered_score": _stats([
                row.best_actor_score
                for row in self.decisions
                if row.status == "assigned"
                and row.best_actor_score is not None
            ]),
            "non_roster_score": _stats([
                row.non_roster_score
                for row in self.decisions
                if row.status == "non_roster"
            ]),
            "actor_tracklet_count": _stats([
                len(row["tracklet_ids"])
                for row in self.actors
            ]),
            "config": self.config.to_dict(),
        }


def _family_relation(
    actor_family: _Family,
    tracklet_family: _Family,
) -> tuple[str, float, bool]:
    if actor_family.key is None or tracklet_family.key is None:
        return "neutral", .50, False

    if actor_family.key == tracklet_family.key:
        if actor_family.confident and tracklet_family.confident:
            return "same_confident", 1.0, False
        return "same_weak", .82, False

    if actor_family.confident and tracklet_family.confident:
        return "different_confident", 0.0, True

    return "different_weak", .22, False


def _hard_pairs(
    rows: Iterable[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    return {
        tuple(sorted((
            str(row["tracklet_a"]),
            str(row["tracklet_b"]),
        )))
        for row in rows
    }


def _transition_support(
    *,
    actor_obs: _Observation,
    tracklet_obs: _Observation,
    dt_s: float,
    config: ClosedRosterConfig,
) -> tuple[float, float, bool]:
    distance = math.hypot(
        tracklet_obs.x - actor_obs.x,
        tracklet_obs.z - actor_obs.z,
    )
    reachability = (
        config.position_slack_m
        + config.max_transition_speed_mps * max(dt_s, 0.0)
    )

    hard_violation = (
        dt_s <= config.hard_geometry_window_s
        and distance > reachability + 1e-9
    )

    if dt_s > config.context_window_s:
        return distance, .20, hard_violation

    support = _clamp01(
        1.0 - distance / max(reachability, 1e-9)
    )
    return distance, support, hard_violation


def _candidate(
    *,
    actor: _ActorState,
    tracklet: _Tracklet,
    tracklet_family: _Family,
    tracklet_appearance: _Appearance | None,
    hard_conflicts: set[tuple[str, str]],
    config: ClosedRosterConfig,
) -> ActorCandidate | None:
    if actor.occupied_frames.intersection(tracklet.frame_set):
        return None

    for member in actor.tracklet_ids:
        if tuple(sorted((member, tracklet.tracklet_id))) in hard_conflicts:
            return None

    relation, family_support, hard_family = _family_relation(
        actor.family(),
        tracklet_family,
    )
    if hard_family:
        return None

    before = actor.previous_observation(tracklet.start_frame)
    after = actor.next_observation(tracklet.end_frame)

    before_gap = None
    after_gap = None
    before_distance = None
    after_distance = None
    before_support = .20
    after_support = .20

    if before is not None:
        before_gap = tracklet.start_timestamp_s - before.timestamp_s
        if before_gap <= config.time_epsilon_s:
            return None
        (
            before_distance,
            before_support,
            hard_before,
        ) = _transition_support(
            actor_obs=before,
            tracklet_obs=tracklet.observations[0],
            dt_s=before_gap,
            config=config,
        )
        if hard_before:
            return None

    if after is not None:
        after_gap = after.timestamp_s - tracklet.end_timestamp_s
        if after_gap <= config.time_epsilon_s:
            return None
        (
            after_distance,
            after_support,
            hard_after,
        ) = _transition_support(
            actor_obs=tracklet.observations[-1],
            tracklet_obs=after,
            dt_s=after_gap,
            config=config,
        )
        if hard_after:
            return None

    available_supports = []
    temporal_terms = []

    if before is not None and before_gap is not None:
        available_supports.append(before_support)
        temporal_terms.append(
            math.exp(-before_gap / 1.5)
            if before_gap <= config.context_window_s
            else .10
        )

    if after is not None and after_gap is not None:
        available_supports.append(after_support)
        temporal_terms.append(
            math.exp(-after_gap / 1.5)
            if after_gap <= config.context_window_s
            else .10
        )

    geometry_support = (
        sum(available_supports) / len(available_supports)
        if available_supports
        else .10
    )
    temporal_support = (
        sum(temporal_terms) / len(temporal_terms)
        if temporal_terms
        else .10
    )

    actor_embedding = actor.appearance()
    reid_cosine = None
    reid_support = .50
    if actor_embedding is not None and tracklet_appearance is not None:
        reid_cosine = float(np.clip(
            np.dot(actor_embedding, tracklet_appearance.embedding),
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

    return ActorCandidate(
        actor_id=actor.actor_id,
        tracklet_id=tracklet.tracklet_id,
        score=float(_clamp01(score)),
        before_available=before is not None,
        after_available=after is not None,
        before_gap_s=(
            float(before_gap)
            if before_gap is not None
            else None
        ),
        after_gap_s=(
            float(after_gap)
            if after_gap is not None
            else None
        ),
        before_distance_m=(
            float(before_distance)
            if before_distance is not None
            else None
        ),
        after_distance_m=(
            float(after_distance)
            if after_distance is not None
            else None
        ),
        before_support=float(before_support),
        after_support=float(after_support),
        geometry_support=float(geometry_support),
        temporal_support=float(temporal_support),
        family_relation=relation,
        family_support=float(family_support),
        reid_cosine=reid_cosine,
        reid_support=float(reid_support),
        physically_viable=True,
    )


def _off_field_ratio(
    tracklet: _Tracklet,
    config: ClosedRosterConfig,
) -> float:
    half_length = config.field_length_m / 2.0
    half_width = config.field_width_m / 2.0
    margin = config.off_field_margin_m

    outside = sum(
        (
            abs(obs.x) > half_length + margin
            or abs(obs.z) > half_width + margin
        )
        for obs in tracklet.observations
    )
    return outside / len(tracklet.observations)


def _internal_motion_penalty(
    tracklet: _Tracklet,
    config: ClosedRosterConfig,
) -> float:
    if len(tracklet.observations) < 2:
        return 0.0

    bad = 0
    total = 0

    for left, right in zip(
        tracklet.observations,
        tracklet.observations[1:],
    ):
        dt = right.timestamp_s - left.timestamp_s
        if dt <= config.time_epsilon_s:
            continue
        speed = math.hypot(
            right.x - left.x,
            right.z - left.z,
        ) / dt
        total += 1
        if speed > config.max_internal_speed_mps:
            bad += 1

    return bad / total if total else 0.0


def _occupied_actor_count_by_frame(
    actors: Mapping[str, _ActorState],
) -> dict[int, int]:
    result: dict[int, int] = defaultdict(int)
    for actor in actors.values():
        for frame in actor.occupied_frames:
            result[frame] += 1
    return result


def _non_roster_evidence(
    *,
    tracklet: _Tracklet,
    occupied_by_frame: Mapping[int, int],
    config: ClosedRosterConfig,
) -> tuple[float, float, float, float, float]:
    off_field = _off_field_ratio(tracklet, config)

    if config.short_extra_s <= config.time_epsilon_s:
        shortness = 0.0
    else:
        shortness = _clamp01(
            1.0 - tracklet.duration_s / config.short_extra_s
        )

    full_threshold = max(
        1,
        config.expected_actor_slots - 1,
    )
    occupied = sum(
        occupied_by_frame.get(obs.frame_index, 0) >= full_threshold
        for obs in tracklet.observations
    )
    occupancy_ratio = occupied / len(tracklet.observations)

    motion_penalty = _internal_motion_penalty(
        tracklet,
        config,
    )

    score = (
        .52 * off_field
        + .23 * occupancy_ratio
        + .15 * shortness
        + .10 * motion_penalty
    )

    if (
        off_field >= config.off_field_strong_ratio
        and tracklet.duration_s <= config.short_extra_s
    ):
        score = max(score, .78)

    if (
        occupancy_ratio >= config.occupancy_strong_ratio
        and tracklet.duration_s <= config.very_short_extra_s
    ):
        score = max(score, .74)

    if off_field < .25:
        score = min(score, .58)

    return (
        float(_clamp01(score)),
        float(off_field),
        float(occupancy_ratio),
        float(shortness),
        float(motion_penalty),
    )


def _score_candidates(
    *,
    tracklet: _Tracklet,
    actors: Mapping[str, _ActorState],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: set[tuple[str, str]],
    config: ClosedRosterConfig,
) -> list[ActorCandidate]:
    family = _family(family_by_tracklet.get(tracklet.tracklet_id))
    appearance = _appearance(
        appearance_by_tracklet.get(tracklet.tracklet_id)
    )

    candidates = []
    for actor in actors.values():
        candidate = _candidate(
            actor=actor,
            tracklet=tracklet,
            tracklet_family=family,
            tracklet_appearance=appearance,
            hard_conflicts=hard_conflicts,
            config=config,
        )
        if candidate is not None:
            candidates.append(candidate)

    return sorted(
        candidates,
        key=lambda row: (
            -row.score,
            -row.geometry_support,
            -(row.reid_cosine if row.reid_cosine is not None else -2.0),
            row.actor_id,
        ),
    )


def _decision_proposal(
    *,
    tracklet: _Tracklet,
    actors: Mapping[str, _ActorState],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
    hard_conflicts: set[tuple[str, str]],
    occupied_by_frame: Mapping[int, int],
    iteration: int,
    config: ClosedRosterConfig,
) -> tuple[ResolutionDecision, list[ActorCandidate]]:
    candidates = _score_candidates(
        tracklet=tracklet,
        actors=actors,
        family_by_tracklet=family_by_tracklet,
        appearance_by_tracklet=appearance_by_tracklet,
        hard_conflicts=hard_conflicts,
        config=config,
    )

    (
        non_roster_score,
        off_field_ratio,
        occupancy_ratio,
        shortness,
        motion_penalty,
    ) = _non_roster_evidence(
        tracklet=tracklet,
        occupied_by_frame=occupied_by_frame,
        config=config,
    )

    best = candidates[0] if candidates else None
    second = candidates[1] if len(candidates) > 1 else None

    best_score = best.score if best is not None else None
    second_score = second.score if second is not None else None
    margin = (
        best.score - second.score
        if best is not None and second is not None
        else best.score if best is not None
        else None
    )

    def make(
        *,
        status: str,
        actor_id: str | None,
        rule: str,
        review: bool,
    ) -> ResolutionDecision:
        return ResolutionDecision(
            tracklet_id=tracklet.tracklet_id,
            status=status,
            actor_id=actor_id,
            rule=rule,
            iteration=iteration,
            needs_review=review,
            best_actor_score=best_score,
            second_actor_score=second_score,
            actor_margin=margin,
            non_roster_score=non_roster_score,
            off_field_ratio=off_field_ratio,
            full_roster_occupancy_ratio=occupancy_ratio,
            shortness_support=shortness,
            internal_motion_penalty=motion_penalty,
            candidate_count=len(candidates),
            best_candidate=best,
        )

    if best is None:
        return (
            make(
                status="non_roster",
                actor_id=None,
                rule="no_physically_viable_actor",
                review=False,
            ),
            candidates,
        )

    # Strong explicit extra-person/noise evidence wins only if it is also
    # clearly stronger than the best roster explanation.
    if (
        non_roster_score >= config.strong_non_roster_floor
        and non_roster_score >= best.score + config.non_roster_margin
    ):
        return (
            make(
                status="non_roster",
                actor_id=None,
                rule="strong_non_roster_evidence",
                review=False,
            ),
            candidates,
        )

    has_before = best.before_available
    has_after = best.after_available

    if (
        has_before
        and has_after
        and best.score >= config.bridge_floor
        and (
            margin is None
            or margin >= config.assignment_margin
        )
    ):
        return (
            make(
                status="assigned",
                actor_id=best.actor_id,
                rule="bidirectional_bridge",
                review=False,
            ),
            candidates,
        )

    if (
        (has_before or has_after)
        and best.score >= config.one_sided_floor
        and (
            margin is None
            or margin >= config.assignment_margin
        )
    ):
        return (
            make(
                status="assigned",
                actor_id=best.actor_id,
                rule="one_sided_scene_continuity",
                review=False,
            ),
            candidates,
        )

    if (
        len(candidates) == 1
        and best.score >= config.unique_actor_floor
    ):
        return (
            make(
                status="assigned",
                actor_id=best.actor_id,
                rule="unique_physically_viable_actor",
                review=False,
            ),
            candidates,
        )

    # ReID is allowed to break a near score tie, never to rescue a physically
    # impossible actor because such actors were already removed above.
    if (
        second is not None
        and best.reid_cosine is not None
        and second.reid_cosine is not None
        and abs(best.score - second.score) < config.assignment_margin
        and (
            best.reid_cosine - second.reid_cosine
            >= config.reid_tie_break_margin
        )
        and best.score >= config.unique_actor_floor
    ):
        return (
            make(
                status="assigned",
                actor_id=best.actor_id,
                rule="reid_tie_break_after_geometry",
                review=True,
            ),
            candidates,
        )

    return (
        make(
            status="pending",
            actor_id=best.actor_id,
            rule="closed_world_pending",
            review=True,
        ),
        candidates,
    )


def _priority(decision: ResolutionDecision) -> tuple:
    order = {
        "bidirectional_bridge": 0,
        "one_sided_scene_continuity": 1,
        "unique_physically_viable_actor": 2,
        "reid_tie_break_after_geometry": 3,
        "strong_non_roster_evidence": 4,
        "no_physically_viable_actor": 5,
        "closed_world_pending": 6,
    }
    return (
        order.get(decision.rule, 99),
        -(decision.actor_margin or 0.0),
        -(decision.best_actor_score or 0.0),
        decision.tracklet_id,
    )


def _actor_rows(
    actors: Mapping[str, _ActorState],
) -> tuple[dict[str, Any], ...]:
    rows = []
    for actor_id in sorted(actors):
        actor = actors[actor_id]
        family = actor.family()
        rows.append({
            "schema_version": 1,
            "resolution_version": CLOSED_ROSTER_VERSION,
            "actor_id": actor_id,
            "tracklet_ids": list(actor.tracklet_ids),
            "tracklet_count": len(actor.tracklet_ids),
            "observation_count": len(actor.observations),
            "family_key": family.key,
            "family_confident": family.confident,
            "family_dominance": family.mass,
            "appearance_available": actor.appearance() is not None,
            "provisional": True,
        })
    return tuple(rows)


def _trajectory_rows(
    actors: Mapping[str, _ActorState],
    tracklet_owner: Mapping[str, str],
    tracklets: Mapping[str, _Tracklet],
    config: ClosedRosterConfig,
) -> tuple[dict[str, Any], ...]:
    rows = []

    for actor_id in sorted(actors):
        actor = actors[actor_id]

        # Tracklet owner guarantees at most one observation per actor/frame.
        source_by_detection: dict[str, str] = {}
        for tid in actor.tracklet_ids:
            for obs in tracklets[tid].observations:
                source_by_detection[obs.detection_id] = tid

        points = []
        observations = actor.observations

        for index, obs in enumerate(observations):
            next_gap_s = None
            next_distance_m = None
            next_speed_mps = None
            interpolate_to_next = False

            if index + 1 < len(observations):
                nxt = observations[index + 1]
                next_gap_s = nxt.timestamp_s - obs.timestamp_s
                next_distance_m = math.hypot(
                    nxt.x - obs.x,
                    nxt.z - obs.z,
                )
                if next_gap_s > config.time_epsilon_s:
                    next_speed_mps = next_distance_m / next_gap_s
                    interpolate_to_next = (
                        next_gap_s
                        <= config.visual_interpolation_max_gap_s
                        + config.time_epsilon_s
                        and next_speed_mps
                        <= config.visual_interpolation_max_speed_mps
                    )

            tid = source_by_detection[obs.detection_id]
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
            "trajectory_version": "closed-roster-anchors-v1",
            "actor_id": actor_id,
            "tracklet_ids": list(actor.tracklet_ids),
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


def resolve_closed_roster(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    initial_assignment_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]] = (),
    appearance_rows: Sequence[Mapping[str, Any]] = (),
    hard_conflicts: Sequence[Mapping[str, Any]] = (),
    config: ClosedRosterConfig | None = None,
) -> ClosedRosterResult:
    selected = config or ClosedRosterConfig()

    tracklets = {
        tracklet.tracklet_id: tracklet
        for tracklet in (
            _parse_tracklet(row)
            for row in tracklet_rows
        )
    }
    if len(tracklets) != len(tracklet_rows):
        raise ValueError("duplicate tracklet ids")

    declared_actor_ids = [
        str(row["actor_id"])
        for row in actor_rows
    ]
    if len(declared_actor_ids) != len(set(declared_actor_ids)):
        raise ValueError("duplicate actor ids")
    if len(declared_actor_ids) != selected.expected_actor_slots:
        raise ValueError(
            "closed roster requires exactly "
            f"{selected.expected_actor_slots} actor slots, got "
            f"{len(declared_actor_ids)}"
        )

    family_by_tracklet = {}
    for row in family_rows:
        tid = str(row["tracklet_id"])
        if tid in family_by_tracklet:
            raise ValueError(f"duplicate family row {tid}")
        family_by_tracklet[tid] = row

    appearance_by_tracklet = {}
    for row in appearance_rows:
        tid = str(row["tracklet_id"])
        if tid in appearance_by_tracklet:
            raise ValueError(f"duplicate appearance row {tid}")
        appearance_by_tracklet[tid] = row

    hard = _hard_pairs(hard_conflicts)

    initial_owner: dict[str, str] = {}
    for row in initial_assignment_rows:
        tid = str(row["tracklet_id"])
        actor_id = str(row["actor_id"])
        if tid not in tracklets:
            raise ValueError(f"initial assignment references unknown {tid}")
        if actor_id not in declared_actor_ids:
            raise ValueError(
                f"initial assignment references unknown actor {actor_id}"
            )
        if tid in initial_owner:
            raise ValueError(f"duplicate initial assignment for {tid}")
        initial_owner[tid] = actor_id

    actors = {
        actor_id: _ActorState(actor_id=actor_id)
        for actor_id in sorted(declared_actor_ids)
    }

    # Prefer the explicit initial assignments. actor_rows tracklet_ids are used
    # as a consistency check when present.
    declared_members = {}
    for row in actor_rows:
        actor_id = str(row["actor_id"])
        values = tuple(str(v) for v in row.get("tracklet_ids", ()))
        declared_members[actor_id] = set(values)

    for tid, actor_id in sorted(
        initial_owner.items(),
        key=lambda item: (
            tracklets[item[0]].start_frame,
            item[0],
        ),
    ):
        if (
            declared_members.get(actor_id)
            and tid not in declared_members[actor_id]
        ):
            raise ValueError(
                f"{tid} assignment disagrees with actor row {actor_id}"
            )
        actors[actor_id].add(
            tracklets[tid],
            _family(family_by_tracklet.get(tid)),
            _appearance(appearance_by_tracklet.get(tid)),
        )

    all_tracklet_ids = set(tracklets)
    unresolved = set(all_tracklet_ids - set(initial_owner))
    decisions: list[ResolutionDecision] = []
    final_owner = dict(initial_owner)
    non_roster: dict[str, ResolutionDecision] = {}

    for iteration in range(1, selected.max_iterations + 1):
        if not unresolved:
            break

        occupied_by_frame = _occupied_actor_count_by_frame(actors)
        proposals = []

        for tid in sorted(unresolved):
            decision, candidates = _decision_proposal(
                tracklet=tracklets[tid],
                actors=actors,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=hard,
                occupied_by_frame=occupied_by_frame,
                iteration=iteration,
                config=selected,
            )
            proposals.append((decision, candidates))

        progress = False

        for decision, _ in sorted(
            proposals,
            key=lambda item: _priority(item[0]),
        ):
            tid = decision.tracklet_id
            if tid not in unresolved:
                continue

            if decision.status == "non_roster":
                non_roster[tid] = decision
                decisions.append(decision)
                unresolved.remove(tid)
                progress = True
                continue

            if decision.status != "assigned" or decision.actor_id is None:
                continue

            # Re-evaluate immediately because earlier proposals in this same
            # iteration may have changed occupancy/neighbor context.
            fresh, _ = _decision_proposal(
                tracklet=tracklets[tid],
                actors=actors,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=hard,
                occupied_by_frame=_occupied_actor_count_by_frame(actors),
                iteration=iteration,
                config=selected,
            )
            if (
                fresh.status != "assigned"
                or fresh.actor_id is None
            ):
                continue

            actor = actors[fresh.actor_id]
            try:
                actor.add(
                    tracklets[tid],
                    _family(family_by_tracklet.get(tid)),
                    _appearance(appearance_by_tracklet.get(tid)),
                )
            except ValueError:
                continue

            final_owner[tid] = fresh.actor_id
            decisions.append(fresh)
            unresolved.remove(tid)
            progress = True

        if not progress:
            break

    # Closed-world completion. Any remaining tracklet is resolved to either
    # the best still-viable actor or NON_ROSTER. There is intentionally no
    # UNASSIGNED terminal state.
    if unresolved:
        occupied_by_frame = _occupied_actor_count_by_frame(actors)

        remaining = []
        for tid in sorted(unresolved):
            decision, candidates = _decision_proposal(
                tracklet=tracklets[tid],
                actors=actors,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=hard,
                occupied_by_frame=occupied_by_frame,
                iteration=selected.max_iterations + 1,
                config=selected,
            )
            remaining.append((decision, candidates))

        # Strongest actor explanations first. This makes overlapping extras
        # naturally lose roster slots to better-supported player fragments.
        remaining.sort(
            key=lambda item: (
                -(item[0].best_actor_score or -1.0),
                -(item[0].actor_margin or 0.0),
                item[0].tracklet_id,
            )
        )

        for prior, _ in remaining:
            tid = prior.tracklet_id
            if tid not in unresolved:
                continue

            fresh, candidates = _decision_proposal(
                tracklet=tracklets[tid],
                actors=actors,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
                hard_conflicts=hard,
                occupied_by_frame=_occupied_actor_count_by_frame(actors),
                iteration=selected.max_iterations + 1,
                config=selected,
            )

            # Respect clear NON_ROSTER conclusions.
            if fresh.status == "non_roster":
                non_roster[tid] = fresh
                decisions.append(fresh)
                unresolved.remove(tid)
                continue

            # If a viable actor remains, closed-world semantics assign the best
            # one even when ambiguity remains; such decisions are flagged.
            if candidates:
                best = candidates[0]

                # If NON_ROSTER is strongly more plausible than the best actor,
                # keep it outside the roster.
                if (
                    fresh.non_roster_score
                    >= selected.strong_non_roster_floor
                    and fresh.non_roster_score
                    >= best.score + selected.non_roster_margin
                ):
                    decision = ResolutionDecision(
                        tracklet_id=tid,
                        status="non_roster",
                        actor_id=None,
                        rule="closed_world_non_roster",
                        iteration=selected.max_iterations + 1,
                        needs_review=False,
                        best_actor_score=best.score,
                        second_actor_score=(
                            candidates[1].score
                            if len(candidates) > 1
                            else None
                        ),
                        actor_margin=(
                            best.score - candidates[1].score
                            if len(candidates) > 1
                            else best.score
                        ),
                        non_roster_score=fresh.non_roster_score,
                        off_field_ratio=fresh.off_field_ratio,
                        full_roster_occupancy_ratio=(
                            fresh.full_roster_occupancy_ratio
                        ),
                        shortness_support=fresh.shortness_support,
                        internal_motion_penalty=(
                            fresh.internal_motion_penalty
                        ),
                        candidate_count=len(candidates),
                        best_candidate=best,
                    )
                    non_roster[tid] = decision
                    decisions.append(decision)
                    unresolved.remove(tid)
                    continue

                # Closed-world fallback: try viable candidates in score order.
                # IMPORTANT: once one candidate is inserted successfully, commit
                # that assignment exactly once.  The previous implementation
                # inserted the best candidate in the try block and then inserted
                # it a second time below, which correctly tripped the exact-frame
                # coexistence guard on the same tracklet.
                chosen = None
                chosen_candidates = candidates

                for candidate in candidates:
                    actor = actors[candidate.actor_id]
                    try:
                        actor.add(
                            tracklets[tid],
                            _family(family_by_tracklet.get(tid)),
                            _appearance(appearance_by_tracklet.get(tid)),
                        )
                    except ValueError:
                        # This actor became unavailable after previous
                        # closed-world assignments. Try the next still-viable
                        # roster slot rather than duplicating or inventing an ID.
                        continue

                    chosen = candidate
                    break

                if chosen is not None:
                    # Re-score only for diagnostics after the assignment was
                    # committed; never add the same tracklet again.
                    second_score = next(
                        (
                            candidate.score
                            for candidate in chosen_candidates
                            if candidate.actor_id != chosen.actor_id
                        ),
                        None,
                    )
                    decision = ResolutionDecision(
                        tracklet_id=tid,
                        status="assigned",
                        actor_id=chosen.actor_id,
                        rule="closed_world_best_viable_actor",
                        iteration=selected.max_iterations + 1,
                        needs_review=True,
                        best_actor_score=chosen.score,
                        second_actor_score=second_score,
                        actor_margin=(
                            chosen.score - second_score
                            if second_score is not None
                            else chosen.score
                        ),
                        non_roster_score=fresh.non_roster_score,
                        off_field_ratio=fresh.off_field_ratio,
                        full_roster_occupancy_ratio=(
                            fresh.full_roster_occupancy_ratio
                        ),
                        shortness_support=fresh.shortness_support,
                        internal_motion_penalty=(
                            fresh.internal_motion_penalty
                        ),
                        candidate_count=len(chosen_candidates),
                        best_candidate=chosen,
                    )
                    final_owner[tid] = chosen.actor_id
                    decisions.append(decision)
                    unresolved.remove(tid)
                    continue

            # No actor can physically own it after the other high-confidence
            # assignments have been inserted: by definition it is outside this
            # closed 23-person roster.
            decision = ResolutionDecision(
                tracklet_id=tid,
                status="non_roster",
                actor_id=None,
                rule="closed_world_no_available_actor",
                iteration=selected.max_iterations + 1,
                needs_review=False,
                best_actor_score=None,
                second_actor_score=None,
                actor_margin=None,
                non_roster_score=fresh.non_roster_score,
                off_field_ratio=fresh.off_field_ratio,
                full_roster_occupancy_ratio=(
                    fresh.full_roster_occupancy_ratio
                ),
                shortness_support=fresh.shortness_support,
                internal_motion_penalty=fresh.internal_motion_penalty,
                candidate_count=0,
                best_candidate=None,
            )
            non_roster[tid] = decision
            decisions.append(decision)
            unresolved.remove(tid)

    if unresolved:
        raise RuntimeError(
            f"closed-roster resolution left unresolved tracklets: "
            f"{sorted(unresolved)[:10]}"
        )

    if set(final_owner).intersection(non_roster):
        raise RuntimeError("tracklet cannot be actor-assigned and NON_ROSTER")
    if set(final_owner).union(non_roster) != all_tracklet_ids:
        raise RuntimeError("not every input tracklet received a terminal status")

    assignment_rows = tuple(
        {
            "schema_version": 1,
            "resolution_version": CLOSED_ROSTER_VERSION,
            "tracklet_id": tid,
            "actor_id": final_owner[tid],
            "assignment_status": "assigned",
            "initial_assignment": tid in initial_owner,
            "provisional": True,
        }
        for tid in sorted(final_owner)
    )

    non_roster_rows = tuple(
        {
            **decision.to_dict(),
            "assignment_status": "non_roster",
            "identity_id": NON_ROSTER_ID,
        }
        for _, decision in sorted(non_roster.items())
    )

    actors_out = _actor_rows(actors)
    trajectories = _trajectory_rows(
        actors,
        final_owner,
        tracklets,
        selected,
    )

    return ClosedRosterResult(
        actors=actors_out,
        tracklet_assignments=assignment_rows,
        non_roster_tracklets=non_roster_rows,
        decisions=tuple(decisions),
        actor_trajectories=trajectories,
        initial_assignment_count=len(initial_owner),
        recovered_assignment_count=(
            len(final_owner) - len(initial_owner)
        ),
        config=selected,
    )
