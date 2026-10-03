"""Tracking V1 Hito 7I.2: bounded persistent actor-slot assignment.

The stage is deliberately ordered as:

1. TEMPORAL EXCLUSION
   Only actors that are actually LOST may receive a new tracklet. Actors that
   still have a tracklet at the successor start are excluded.

2. PHYSICAL PLAUSIBILITY
   Recent lost actors must be able to reach the new X/Z endpoint.

3. KIT COMPATIBILITY
   Confident anonymous-family contradictions are rejected.

4. APPEARANCE / ReID
   Existing tracklet embeddings are used to disambiguate the physically
   plausible LOST candidates, never to override a physical contradiction.

The first search tier is local handoff (short gap). Only when no local
candidate survives do we try a longer recovery tier, where ReID is required.

The football population limit is SOFT. It may slightly relax an already valid
candidate under pressure, but it never forces an impossible merge and it is
not a hard total-identity count (substitutions remain possible).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from tracking.tracklets.models import Tracklet


PERSISTENT_ACTOR_SLOT_VERSION = "persistent-actor-slots-v1"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("appearance embedding must be a finite 1-D vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("appearance embedding has zero norm")
    return vector / norm


def _stats(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "min": None, "median": None, "p90": None,
            "p95": None, "max": None, "mean": None,
        }
    ordered = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        a = pos - lo
        return ordered[lo] * (1.0 - a) + ordered[hi] * a

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
class PersistentActorSlotConfig:
    # Tier 1: "se perdió y reapareció cerca".
    local_gap_s: float = 1.25

    # Tier 2: longer recovery. ReID is mandatory here.
    recovery_gap_s: float = 5.0

    max_transition_speed_mps: float = 12.0
    position_slack_m: float = 1.25
    velocity_horizon_s: float = 0.85
    max_reliable_velocity_mps: float = 12.0

    # Proximity is the first measuring stick.
    local_shortlist_k: int = 4
    local_shortlist_extra_m: float = 2.5
    recovery_shortlist_k: int = 4

    # Scoring once hard/local filters have been applied.
    geometry_weight: float = 0.34
    recency_weight: float = 0.14
    family_weight: float = 0.12
    velocity_weight: float = 0.10
    reid_weight: float = 0.30

    local_min_score: float = 0.57
    local_unique_min_score: float = 0.50
    recovery_min_score: float = 0.64
    min_best_margin: float = 0.045

    # ReID is ranking/support, not a global identity threshold.
    recovery_reid_floor: float = 0.72
    recovery_reid_margin: float = 0.015

    soft_visible_actor_limit: int = 24
    pressure_score_relaxation: float = 0.04
    pressure_margin_relaxation: float = 0.015

    time_epsilon_s: float = 1e-6

    def __post_init__(self) -> None:
        if self.local_gap_s <= 0 or self.recovery_gap_s <= self.local_gap_s:
            raise ValueError("require 0 < local_gap_s < recovery_gap_s")
        for name in (
            "max_transition_speed_mps",
            "position_slack_m",
            "velocity_horizon_s",
            "max_reliable_velocity_mps",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.local_shortlist_k < 1 or self.recovery_shortlist_k < 1:
            raise ValueError("shortlist sizes must be >= 1")
        if self.local_shortlist_extra_m < 0:
            raise ValueError("local_shortlist_extra_m must be >= 0")
        weights = (
            self.geometry_weight, self.recency_weight, self.family_weight,
            self.velocity_weight, self.reid_weight,
        )
        if any(v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("evidence weights must be non-negative and non-zero")
        for name in (
            "local_min_score", "local_unique_min_score", "recovery_min_score",
            "min_best_margin", "recovery_reid_floor", "recovery_reid_margin",
            "pressure_score_relaxation", "pressure_margin_relaxation",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.soft_visible_actor_limit < 1:
            raise ValueError("soft_visible_actor_limit must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Family:
    key: str | None
    confident: bool
    weight: float


def _family(row: Mapping[str, Any] | None) -> _Family:
    if row is None:
        return _Family(None, False, 0.0)
    state = str(row.get("state", "unknown"))
    if state in {"confident_primary", "confident_special"}:
        key = row.get("family_key")
        if isinstance(key, str) and key:
            return _Family(key, True, 1.0 if state == "confident_primary" else .9)
    if state in {"weak_primary", "weak_special"}:
        key = row.get("candidate_family_key")
        if isinstance(key, str) and key:
            return _Family(key, False, .35 if state == "weak_primary" else .15)
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
        effective_sample_count=max(1.0, float(row.get("effective_sample_count", 1.0))),
    )


@dataclass(slots=True)
class _Actor:
    actor_id: str
    tracklet_ids: list[str] = field(default_factory=list)
    tracklets: list[Tracklet] = field(default_factory=list)
    family_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    confident_family_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    appearance_sum: np.ndarray | None = None
    appearance_weight: float = 0.0

    @property
    def start_time_s(self) -> float:
        return self.tracklets[0].start_timestamp_s

    @property
    def end_time_s(self) -> float:
        return self.tracklets[-1].end_timestamp_s

    @property
    def last_tracklet(self) -> Tracklet:
        return self.tracklets[-1]

    def add(
        self,
        tracklet: Tracklet,
        family: _Family,
        appearance: _Appearance | None,
    ) -> None:
        if self.tracklets and tracklet.start_timestamp_s <= self.end_time_s:
            raise ValueError(f"{self.actor_id} received overlapping tracklets")
        self.tracklets.append(tracklet)
        self.tracklet_ids.append(tracklet.tracklet_id)

        if family.key and family.weight > 0:
            self.family_mass[family.key] += family.weight
            if family.confident:
                self.confident_family_mass[family.key] += family.weight

        if appearance is not None:
            weight = max(
                1e-6,
                appearance.reliability
                * math.sqrt(max(1.0, appearance.effective_sample_count)),
            )
            vector = appearance.embedding * weight
            if self.appearance_sum is None:
                self.appearance_sum = vector.copy()
            else:
                self.appearance_sum += vector
            self.appearance_weight += weight

    def family(self) -> _Family:
        if not self.family_mass:
            return _Family(None, False, 0.0)
        key = max(self.family_mass, key=lambda k: (self.family_mass[k], k))
        total = sum(self.family_mass.values())
        dominance = self.family_mass[key] / total if total else 0.0
        confident_total = sum(self.confident_family_mass.values())
        confident_key = self.confident_family_mass.get(key, 0.0)
        confident_dominance = (
            confident_key / confident_total if confident_total else 0.0
        )
        confident = (
            confident_key >= .9
            and dominance >= .70
            and (confident_total <= 1e-12 or confident_dominance >= .75)
        )
        return _Family(key, confident, float(dominance))

    def appearance(self) -> np.ndarray | None:
        if self.appearance_sum is None or self.appearance_weight <= 1e-12:
            return None
        return _unit(self.appearance_sum)


@dataclass(frozen=True, slots=True)
class SlotCandidate:
    actor_id: str
    tracklet_id: str
    tier: str
    gap_s: float
    distance_m: float
    reachability_limit_m: float
    geometry_support: float
    recency_support: float
    family_relation: str
    family_support: float
    velocity_support: float
    velocity_prediction_error_m: float | None
    reid_cosine: float | None
    reid_support: float
    score: float
    population_pressure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "slot_version": PERSISTENT_ACTOR_SLOT_VERSION,
            "actor_id": self.actor_id,
            "tracklet_id": self.tracklet_id,
            "tier": self.tier,
            "gap_s": self.gap_s,
            "distance_m": self.distance_m,
            "reachability_limit_m": self.reachability_limit_m,
            "geometry_support": self.geometry_support,
            "recency_support": self.recency_support,
            "family_relation": self.family_relation,
            "family_support": self.family_support,
            "velocity_support": self.velocity_support,
            "velocity_prediction_error_m": self.velocity_prediction_error_m,
            "reid_cosine": self.reid_cosine,
            "reid_support": self.reid_support,
            "score": self.score,
            "population_pressure": self.population_pressure,
        }


@dataclass(frozen=True, slots=True)
class AcceptedSlotHandoff:
    actor_id: str
    predecessor_tracklet_id: str
    successor_tracklet_id: str
    candidate: SlotCandidate
    second_score: float | None
    score_margin: float | None
    second_reid_cosine: float | None
    reid_margin: float | None
    acceptance_rule: str

    def to_dict(self) -> dict[str, Any]:
        row = self.candidate.to_dict()
        row.update({
            "predecessor_tracklet_id": self.predecessor_tracklet_id,
            "successor_tracklet_id": self.successor_tracklet_id,
            "second_score": self.second_score,
            "score_margin": self.score_margin,
            "second_reid_cosine": self.second_reid_cosine,
            "reid_margin": self.reid_margin,
            "acceptance_rule": self.acceptance_rule,
            "identity_decision": "persistent_actor_slot_handoff",
        })
        return row


@dataclass(frozen=True, slots=True)
class NewSlotEvent:
    actor_id: str
    tracklet_id: str
    timestamp_s: float
    reason: str
    local_candidate_count: int
    recovery_candidate_count: int
    best_score: float | None
    best_reid_cosine: float | None
    visible_actor_count_before: int
    population_pressure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "slot_version": PERSISTENT_ACTOR_SLOT_VERSION,
            "actor_id": self.actor_id,
            "tracklet_id": self.tracklet_id,
            "timestamp_s": self.timestamp_s,
            "reason": self.reason,
            "local_candidate_count": self.local_candidate_count,
            "recovery_candidate_count": self.recovery_candidate_count,
            "best_score": self.best_score,
            "best_reid_cosine": self.best_reid_cosine,
            "visible_actor_count_before": self.visible_actor_count_before,
            "population_pressure": self.population_pressure,
        }


@dataclass(frozen=True, slots=True)
class PersistentActorSlotResult:
    actors: tuple[dict[str, Any], ...]
    tracklet_assignments: tuple[dict[str, Any], ...]
    accepted_handoffs: tuple[AcceptedSlotHandoff, ...]
    new_slot_events: tuple[NewSlotEvent, ...]
    candidate_count: int
    temporal_exclusion_count: int
    hard_geometry_rejection_count: int
    hard_family_rejection_count: int
    hard_conflict_rejection_count: int
    peak_visible_actor_count: int
    config: PersistentActorSlotConfig

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "slot_version": PERSISTENT_ACTOR_SLOT_VERSION,
            "actor_count": len(self.actors),
            "tracklet_count": len(self.tracklet_assignments),
            "accepted_handoff_count": len(self.accepted_handoffs),
            "new_actor_count": len(self.new_slot_events),
            "candidate_count": self.candidate_count,
            "temporal_exclusion_count": self.temporal_exclusion_count,
            "hard_geometry_rejection_count": self.hard_geometry_rejection_count,
            "hard_family_rejection_count": self.hard_family_rejection_count,
            "hard_conflict_rejection_count": self.hard_conflict_rejection_count,
            "peak_visible_actor_count": self.peak_visible_actor_count,
            "soft_visible_actor_limit": self.config.soft_visible_actor_limit,
            "population_hard_cap_applied": False,
            "population_forced_merge_count": 0,
            "previous_person_partition_used": False,
            "reid_used": True,
            "reid_is_hard_global_threshold": False,
            "missing_reid_is_negative_for_local": False,
            "missing_reid_allows_long_recovery": False,
            "actor_tracklet_count": _stats([
                len(row["tracklet_ids"]) for row in self.actors
            ]),
            "accepted_gap_s": _stats([
                row.candidate.gap_s for row in self.accepted_handoffs
            ]),
            "accepted_score": _stats([
                row.candidate.score for row in self.accepted_handoffs
            ]),
            "accepted_reid_cosine": _stats([
                row.candidate.reid_cosine
                for row in self.accepted_handoffs
                if row.candidate.reid_cosine is not None
            ]),
            "tier_counts": dict(sorted(Counter(
                row.candidate.tier for row in self.accepted_handoffs
            ).items())),
            "acceptance_rule_counts": dict(sorted(Counter(
                row.acceptance_rule for row in self.accepted_handoffs
            ).items())),
            "new_actor_reason_counts": dict(sorted(Counter(
                row.reason for row in self.new_slot_events
            ).items())),
            "config": self.config.to_dict(),
        }


def _hard_pairs(
    hard_conflicts: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    result = set()
    for row in hard_conflicts:
        a = str(row["tracklet_a"])
        b = str(row["tracklet_b"])
        result.add(tuple(sorted((a, b))))
    return result


def _family_relation(a: _Family, b: _Family) -> tuple[str, float, bool]:
    if a.key is None or b.key is None:
        return "neutral", .50, False
    if a.key == b.key:
        return (
            "same_confident" if a.confident and b.confident else "same_weak",
            1.0 if a.confident and b.confident else .80,
            False,
        )
    if a.confident and b.confident:
        return "different_confident", 0.0, True
    return "different_weak", .20, False


def _velocity(
    tracklet: Tracklet,
    config: PersistentActorSlotConfig,
) -> tuple[float, float] | None:
    if len(tracklet.observations) < 2:
        return None
    a = tracklet.observations[-2]
    b = tracklet.observations[-1]
    dt = b.timestamp_s - a.timestamp_s
    if dt <= config.time_epsilon_s:
        return None
    vx = (b.x - a.x) / dt
    vz = (b.z - a.z) / dt
    if math.hypot(vx, vz) > config.max_reliable_velocity_mps:
        return None
    return vx, vz


def _candidate(
    *,
    actor: _Actor,
    successor: Tracklet,
    successor_family: _Family,
    successor_appearance: _Appearance | None,
    tier: str,
    population_pressure: bool,
    hard_pairs: set[tuple[str, str]],
    config: PersistentActorSlotConfig,
) -> tuple[SlotCandidate | None, str | None]:
    # First rule: the actor must have been LOST before this tracklet starts.
    gap = successor.start_timestamp_s - actor.end_time_s
    if gap <= config.time_epsilon_s:
        return None, "temporal_visible"

    max_gap = config.local_gap_s if tier == "local" else config.recovery_gap_s
    if gap > max_gap + config.time_epsilon_s:
        return None, "gap_too_large"

    # Explicit historical cannot-link/coexistence evidence, if present.
    for member in actor.tracklet_ids:
        if tuple(sorted((member, successor.tracklet_id))) in hard_pairs:
            return None, "hard_conflict"

    last = actor.last_tracklet.observations[-1]
    first = successor.observations[0]
    distance = math.hypot(first.x - last.x, first.z - last.z)
    limit = config.position_slack_m + config.max_transition_speed_mps * gap
    if distance > limit + 1e-9:
        return None, "hard_geometry"

    relation, family_support, hard_family = _family_relation(
        actor.family(), successor_family
    )
    if hard_family:
        return None, "hard_family"

    geometry_support = _clamp01(1.0 - distance / max(limit, 1e-9))
    recency_support = math.exp(
        -gap / (.55 if tier == "local" else 2.0)
    )

    velocity_error = None
    velocity_support = .50
    velocity = _velocity(actor.last_tracklet, config)
    if velocity is not None and gap <= config.velocity_horizon_s:
        vx, vz = velocity
        px = last.x + vx * gap
        pz = last.z + vz * gap
        velocity_error = math.hypot(first.x - px, first.z - pz)
        velocity_support = math.exp(
            -velocity_error / max(1e-6, config.position_slack_m * 1.5)
        )

    actor_embedding = actor.appearance()
    reid_cosine = None
    reid_support = .50
    if actor_embedding is not None and successor_appearance is not None:
        reid_cosine = float(np.clip(
            np.dot(actor_embedding, successor_appearance.embedding),
            -1.0,
            1.0,
        ))
        # Continuous support only. The candidate pool has already been cut by
        # temporal/physical evidence; ReID now helps choose among plausible LOST actors.
        reid_support = _clamp01((reid_cosine + 1.0) * .5)

    if tier == "recovery":
        if reid_cosine is None:
            return None, "recovery_missing_reid"
        if reid_cosine < config.recovery_reid_floor:
            return None, "recovery_reid_below_floor"

    weights = (
        config.geometry_weight,
        config.recency_weight,
        config.family_weight,
        config.velocity_weight,
        config.reid_weight,
    )
    score = (
        weights[0] * geometry_support
        + weights[1] * recency_support
        + weights[2] * family_support
        + weights[3] * velocity_support
        + weights[4] * reid_support
    ) / sum(weights)

    return SlotCandidate(
        actor_id=actor.actor_id,
        tracklet_id=successor.tracklet_id,
        tier=tier,
        gap_s=float(gap),
        distance_m=float(distance),
        reachability_limit_m=float(limit),
        geometry_support=float(geometry_support),
        recency_support=float(recency_support),
        family_relation=relation,
        family_support=float(family_support),
        velocity_support=float(velocity_support),
        velocity_prediction_error_m=(
            float(velocity_error) if velocity_error is not None else None
        ),
        reid_cosine=reid_cosine,
        reid_support=float(reid_support),
        score=float(_clamp01(score)),
        population_pressure=population_pressure,
    ), None


def _shortlist(
    rows: Sequence[SlotCandidate],
    *,
    tier: str,
    config: PersistentActorSlotConfig,
) -> list[SlotCandidate]:
    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda r: (
            r.distance_m,
            r.gap_s,
            -r.score,
            r.actor_id,
        ),
    )
    k = (
        config.local_shortlist_k
        if tier == "local"
        else config.recovery_shortlist_k
    )
    selected = ordered[:k]
    if tier == "local":
        nearest = ordered[0].distance_m
        selected = [
            row for row in selected
            if row.distance_m <= nearest + config.local_shortlist_extra_m
        ]
    return sorted(selected, key=lambda r: (-r.score, r.actor_id))


def _visible_actor_count(
    actors: Mapping[str, _Actor],
    timestamp_s: float,
    *,
    epsilon: float,
) -> int:
    total = 0
    for actor in actors.values():
        last = actor.last_tracklet
        if (
            last.start_timestamp_s - epsilon
            <= timestamp_s
            <= last.end_timestamp_s + epsilon
        ):
            total += 1
    return total


def _peak_visible_actor_count(
    actors: Mapping[str, _Actor],
) -> int:
    events: list[tuple[int, int]] = []
    # Frame-level inclusive intervals. Accepted handoffs never overlap within actor.
    for actor in actors.values():
        for tracklet in actor.tracklets:
            events.append((tracklet.start_frame, +1))
            events.append((tracklet.end_frame + 1, -1))
    events.sort()
    current = peak = 0
    i = 0
    while i < len(events):
        frame = events[i][0]
        delta = 0
        while i < len(events) and events[i][0] == frame:
            delta += events[i][1]
            i += 1
        current += delta
        peak = max(peak, current)
    return peak


def build_persistent_actor_slots(
    *,
    tracklet_rows: Sequence[Mapping[str, Any] | Tracklet],
    family_rows: Sequence[Mapping[str, Any]] = (),
    appearance_rows: Sequence[Mapping[str, Any]] = (),
    hard_conflicts: Sequence[Mapping[str, Any]] = (),
    config: PersistentActorSlotConfig | None = None,
) -> PersistentActorSlotResult:
    selected = config or PersistentActorSlotConfig()

    tracklets = [
        row if isinstance(row, Tracklet) else Tracklet.from_dict(row)
        for row in tracklet_rows
    ]
    tracklets.sort(
        key=lambda t: (t.start_timestamp_s, t.start_frame, t.tracklet_id)
    )
    if len({t.tracklet_id for t in tracklets}) != len(tracklets):
        raise ValueError("duplicate tracklet ids")

    family_by_tracklet: dict[str, Mapping[str, Any]] = {}
    for row in family_rows:
        tid = str(row["tracklet_id"])
        if tid in family_by_tracklet:
            raise ValueError(f"duplicate family row {tid}")
        family_by_tracklet[tid] = row

    appearance_by_tracklet: dict[str, Mapping[str, Any]] = {}
    for row in appearance_rows:
        tid = str(row["tracklet_id"])
        if tid in appearance_by_tracklet:
            raise ValueError(f"duplicate appearance profile {tid}")
        appearance_by_tracklet[tid] = row

    cannot_link = _hard_pairs(hard_conflicts)

    actors: dict[str, _Actor] = {}
    assignments: dict[str, str] = {}
    accepted: list[AcceptedSlotHandoff] = []
    new_events: list[NewSlotEvent] = []

    temporal_exclusions = 0
    geometry_rejections = 0
    family_rejections = 0
    conflict_rejections = 0
    candidate_count = 0
    next_actor_id = 1

    batches: list[list[Tracklet]] = []
    for tracklet in tracklets:
        if (
            not batches
            or abs(
                batches[-1][0].start_timestamp_s
                - tracklet.start_timestamp_s
            ) > selected.time_epsilon_s
        ):
            batches.append([tracklet])
        else:
            batches[-1].append(tracklet)

    def create_actor(
        successor: Tracklet,
        *,
        reason: str,
        local_count: int,
        recovery_count: int,
        best: SlotCandidate | None,
        visible_before: int,
        pressure: bool,
    ) -> None:
        nonlocal next_actor_id
        actor_id = f"actor-{next_actor_id:04d}"
        next_actor_id += 1
        actor = _Actor(actor_id=actor_id)
        actor.add(
            successor,
            _family(family_by_tracklet.get(successor.tracklet_id)),
            _appearance(appearance_by_tracklet.get(successor.tracklet_id)),
        )
        actors[actor_id] = actor
        assignments[successor.tracklet_id] = actor_id
        new_events.append(
            NewSlotEvent(
                actor_id=actor_id,
                tracklet_id=successor.tracklet_id,
                timestamp_s=successor.start_timestamp_s,
                reason=reason,
                local_candidate_count=local_count,
                recovery_candidate_count=recovery_count,
                best_score=best.score if best else None,
                best_reid_cosine=best.reid_cosine if best else None,
                visible_actor_count_before=visible_before,
                population_pressure=pressure,
            )
        )

    for batch in batches:
        now = batch[0].start_timestamp_s
        visible_before = _visible_actor_count(
            actors, now, epsilon=selected.time_epsilon_s
        )
        pressure = (
            visible_before + len(batch)
            > selected.soft_visible_actor_limit
        )

        local_by_tracklet: dict[str, list[SlotCandidate]] = {}
        recovery_by_tracklet: dict[str, list[SlotCandidate]] = {}
        candidate_by_actor: dict[str, list[SlotCandidate]] = defaultdict(list)

        for successor in batch:
            sf = _family(family_by_tracklet.get(successor.tracklet_id))
            sa = _appearance(appearance_by_tracklet.get(successor.tracklet_id))

            local_raw: list[SlotCandidate] = []
            recovery_raw: list[SlotCandidate] = []

            for actor in actors.values():
                gap = successor.start_timestamp_s - actor.end_time_s
                if gap <= selected.time_epsilon_s:
                    temporal_exclusions += 1
                    continue
                if gap > selected.recovery_gap_s + selected.time_epsilon_s:
                    continue

                tier = "local" if gap <= selected.local_gap_s else "recovery"
                row, rejection = _candidate(
                    actor=actor,
                    successor=successor,
                    successor_family=sf,
                    successor_appearance=sa,
                    tier=tier,
                    population_pressure=pressure,
                    hard_pairs=cannot_link,
                    config=selected,
                )
                if row is not None:
                    candidate_count += 1
                    if tier == "local":
                        local_raw.append(row)
                    else:
                        recovery_raw.append(row)
                elif rejection == "hard_geometry":
                    geometry_rejections += 1
                elif rejection == "hard_family":
                    family_rejections += 1
                elif rejection == "hard_conflict":
                    conflict_rejections += 1

            # Crucial priority: if plausible recent LOST actors exist, longer
            # historical candidates do not compete with them.
            local = _shortlist(local_raw, tier="local", config=selected)
            recovery = (
                []
                if local
                else _shortlist(
                    recovery_raw, tier="recovery", config=selected
                )
            )
            local_by_tracklet[successor.tracklet_id] = local
            recovery_by_tracklet[successor.tracklet_id] = recovery

            for row in (local if local else recovery):
                candidate_by_actor[row.actor_id].append(row)

        actor_best: dict[str, str] = {}
        for actor_id, rows in candidate_by_actor.items():
            best = min(
                rows,
                key=lambda r: (-r.score, r.tracklet_id),
            )
            actor_best[actor_id] = best.tracklet_id

        proposals = []
        diagnostics = {}

        for successor in batch:
            local = local_by_tracklet[successor.tracklet_id]
            recovery = recovery_by_tracklet[successor.tracklet_id]
            rows = local if local else recovery
            tier = "local" if local else "recovery"

            if not rows:
                diagnostics[successor.tracklet_id] = (
                    False, "no_plausible_lost_actor", None, None, None
                )
                continue

            ordered = sorted(rows, key=lambda r: (-r.score, r.actor_id))
            best = ordered[0]
            second = ordered[1] if len(ordered) > 1 else None
            score_margin = best.score - second.score if second else None

            score_floor = (
                selected.local_unique_min_score
                if tier == "local" and len(ordered) == 1
                else selected.local_min_score
                if tier == "local"
                else selected.recovery_min_score
            )
            margin_floor = selected.min_best_margin

            if pressure:
                score_floor = max(
                    0.0, score_floor - selected.pressure_score_relaxation
                )
                margin_floor = max(
                    0.0, margin_floor - selected.pressure_margin_relaxation
                )

            if best.score < score_floor:
                diagnostics[successor.tracklet_id] = (
                    False, "best_below_floor", best, second, score_margin
                )
                continue

            if second is not None and (
                score_margin is None or score_margin < margin_floor
            ):
                diagnostics[successor.tracklet_id] = (
                    False, "ambiguous_local_competition", best, second, score_margin
                )
                continue

            if tier == "recovery":
                reid_second_values = sorted(
                    (
                        row.reid_cosine
                        for row in ordered[1:]
                        if row.reid_cosine is not None
                    ),
                    reverse=True,
                )
                second_reid = (
                    reid_second_values[0] if reid_second_values else None
                )
                reid_margin = (
                    best.reid_cosine - second_reid
                    if best.reid_cosine is not None
                    and second_reid is not None
                    else None
                )
                if (
                    second_reid is not None
                    and reid_margin is not None
                    and reid_margin < selected.recovery_reid_margin
                ):
                    diagnostics[successor.tracklet_id] = (
                        False, "ambiguous_reid_recovery", best, second, score_margin
                    )
                    continue

            if actor_best.get(best.actor_id) != successor.tracklet_id:
                diagnostics[successor.tracklet_id] = (
                    False, "actor_prefers_other_successor", best, second, score_margin
                )
                continue

            rule = (
                "local_unique"
                if tier == "local" and len(ordered) == 1
                else "local_competition"
                if tier == "local"
                else "reid_recovery"
            )
            proposals.append(
                (
                    -best.score,
                    successor.tracklet_id,
                    best,
                    second,
                    score_margin,
                    rule,
                )
            )
            diagnostics[successor.tracklet_id] = (
                True, rule, best, second, score_margin
            )

        proposals.sort()
        by_tracklet = {t.tracklet_id: t for t in batch}
        used_actors: set[str] = set()
        used_tracklets: set[str] = set()

        for (
            _,
            tracklet_id,
            best,
            second,
            score_margin,
            rule,
        ) in proposals:
            if best.actor_id in used_actors or tracklet_id in used_tracklets:
                continue

            actor = actors[best.actor_id]
            successor = by_tracklet[tracklet_id]
            predecessor_id = actor.tracklet_ids[-1]

            second_reid = (
                second.reid_cosine
                if second is not None
                else None
            )
            reid_margin = (
                best.reid_cosine - second_reid
                if best.reid_cosine is not None
                and second_reid is not None
                else None
            )

            actor.add(
                successor,
                _family(family_by_tracklet.get(tracklet_id)),
                _appearance(appearance_by_tracklet.get(tracklet_id)),
            )
            assignments[tracklet_id] = actor.actor_id
            used_actors.add(actor.actor_id)
            used_tracklets.add(tracklet_id)

            accepted.append(
                AcceptedSlotHandoff(
                    actor_id=actor.actor_id,
                    predecessor_tracklet_id=predecessor_id,
                    successor_tracklet_id=tracklet_id,
                    candidate=best,
                    second_score=second.score if second else None,
                    score_margin=score_margin,
                    second_reid_cosine=second_reid,
                    reid_margin=reid_margin,
                    acceptance_rule=rule,
                )
            )

        for successor in batch:
            if successor.tracklet_id in used_tracklets:
                continue

            accepted_flag, reason, best, _, _ = diagnostics.get(
                successor.tracklet_id,
                (False, "no_plausible_lost_actor", None, None, None),
            )
            if accepted_flag:
                reason = "assignment_collision"

            create_actor(
                successor,
                reason=reason,
                local_count=len(local_by_tracklet[successor.tracklet_id]),
                recovery_count=len(recovery_by_tracklet[successor.tracklet_id]),
                best=best,
                visible_before=visible_before,
                pressure=pressure,
            )

    actor_rows = []
    assignment_rows = []
    for actor_id in sorted(actors):
        actor = actors[actor_id]
        fam = actor.family()
        actor_rows.append({
            "schema_version": 1,
            "slot_version": PERSISTENT_ACTOR_SLOT_VERSION,
            "actor_id": actor_id,
            "tracklet_ids": list(actor.tracklet_ids),
            "start_time_s": actor.start_time_s,
            "end_time_s": actor.end_time_s,
            "family_key": fam.key,
            "family_confident": fam.confident,
            "family_dominance": fam.weight,
            "appearance_available": actor.appearance() is not None,
            "provisional": True,
        })
        for sequence_index, tracklet_id in enumerate(actor.tracklet_ids):
            assignment_rows.append({
                "schema_version": 1,
                "slot_version": PERSISTENT_ACTOR_SLOT_VERSION,
                "tracklet_id": tracklet_id,
                "actor_id": actor_id,
                "actor_sequence_index": sequence_index,
                "provisional": True,
            })

    assignment_rows.sort(key=lambda row: row["tracklet_id"])

    return PersistentActorSlotResult(
        actors=tuple(actor_rows),
        tracklet_assignments=tuple(assignment_rows),
        accepted_handoffs=tuple(accepted),
        new_slot_events=tuple(new_events),
        candidate_count=candidate_count,
        temporal_exclusion_count=temporal_exclusions,
        hard_geometry_rejection_count=geometry_rejections,
        hard_family_rejection_count=family_rejections,
        hard_conflict_rejection_count=conflict_rejections,
        peak_visible_actor_count=_peak_visible_actor_count(actors),
        config=selected,
    )
