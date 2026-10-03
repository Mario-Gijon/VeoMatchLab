"""Tracking V1 Hito 7I.1: local persistent-actor handoff.

This stage repairs the common local fragmentation case:

    actor visible -> short tracklet ends -> nearby compatible tracklet appears

The solver works directly from conservative tracklets. It does NOT trust the
previous ``person-*`` partition as an input prior.

Evidence:
* short temporal gap;
* hard X/Z reachability;
* anonymous kit-family compatibility;
* short-horizon velocity as soft evidence;
* one-to-one competition between simultaneous new tracklets.

The 24-live-actor limit is a SOFT football prior. It may lower the acceptance
floor slightly when the scene is saturated, but it never overrides physics,
never overrides a confident kit contradiction, and never forces a merge.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

from tracking.tracklets.models import Tracklet


PERSISTENT_ACTOR_HANDOFF_VERSION = "persistent-actor-handoff-v1"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _stats(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)

    def pct(value: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * value
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        alpha = position - lower
        return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


@dataclass(frozen=True, slots=True)
class PersistentActorHandoffConfig:
    max_handoff_gap_s: float = 1.50
    max_transition_speed_mps: float = 12.0
    position_slack_m: float = 1.25
    velocity_horizon_s: float = 0.75
    velocity_max_reliable_speed_mps: float = 12.0
    recency_tau_s: float = 0.60

    geometry_weight: float = 0.45
    recency_weight: float = 0.25
    family_weight: float = 0.20
    velocity_weight: float = 0.10

    min_score: float = 0.58
    unique_candidate_min_score: float = 0.50
    population_pressure_min_score: float = 0.52
    min_best_margin: float = 0.07

    soft_live_actor_limit: int = 24
    live_actor_memory_s: float = 2.0
    time_epsilon_s: float = 1e-6

    def __post_init__(self) -> None:
        for name in (
            "max_handoff_gap_s",
            "max_transition_speed_mps",
            "position_slack_m",
            "velocity_horizon_s",
            "velocity_max_reliable_speed_mps",
            "recency_tau_s",
            "live_actor_memory_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        weights = (
            self.geometry_weight,
            self.recency_weight,
            self.family_weight,
            self.velocity_weight,
        )
        if any((not math.isfinite(v) or v < 0.0) for v in weights):
            raise ValueError("evidence weights must be finite and >= 0")
        if sum(weights) <= 0.0:
            raise ValueError("at least one evidence weight must be positive")

        for name in (
            "min_score",
            "unique_candidate_min_score",
            "population_pressure_min_score",
            "min_best_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")

        if self.soft_live_actor_limit < 1:
            raise ValueError("soft_live_actor_limit must be >= 1")
        if self.time_epsilon_s < 0.0:
            raise ValueError("time_epsilon_s must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FamilyEvidence:
    key: str | None
    confident: bool
    state: str
    weight: float


def _tracklet_family(
    row: Mapping[str, Any] | None,
) -> FamilyEvidence:
    if row is None:
        return FamilyEvidence(None, False, "missing", 0.0)

    state = str(row.get("state", "unknown"))
    family_key = row.get("family_key")
    candidate_key = row.get("candidate_family_key")

    if (
        state in {"confident_primary", "confident_special"}
        and isinstance(family_key, str)
        and family_key
    ):
        return FamilyEvidence(
            family_key,
            True,
            state,
            1.0 if state == "confident_primary" else 0.90,
        )

    if (
        state in {"weak_primary", "weak_special"}
        and isinstance(candidate_key, str)
        and candidate_key
    ):
        return FamilyEvidence(
            candidate_key,
            False,
            state,
            0.35 if state == "weak_primary" else 0.15,
        )

    return FamilyEvidence(None, False, state, 0.0)


@dataclass(slots=True)
class _ActorState:
    actor_id: str
    tracklet_ids: list[str] = field(default_factory=list)
    start_time_s: float = 0.0
    end_time_s: float = 0.0
    last_tracklet: Tracklet | None = None
    family_mass: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    confident_family_mass: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )

    def add_tracklet(
        self,
        tracklet: Tracklet,
        family: FamilyEvidence,
    ) -> None:
        if (
            self.last_tracklet is not None
            and tracklet.start_timestamp_s
            <= self.last_tracklet.end_timestamp_s
        ):
            raise ValueError(
                f"actor {self.actor_id} received overlapping tracklets"
            )

        if not self.tracklet_ids:
            self.start_time_s = tracklet.start_timestamp_s

        self.end_time_s = tracklet.end_timestamp_s
        self.last_tracklet = tracklet
        self.tracklet_ids.append(tracklet.tracklet_id)

        if family.key is not None and family.weight > 0.0:
            self.family_mass[family.key] += family.weight
            if family.confident:
                self.confident_family_mass[family.key] += family.weight

    def family_evidence(self) -> FamilyEvidence:
        if not self.family_mass:
            return FamilyEvidence(None, False, "unknown", 0.0)

        key = max(
            self.family_mass,
            key=lambda value: (
                self.family_mass[value],
                value,
            ),
        )
        total = sum(self.family_mass.values())
        dominance = self.family_mass[key] / total if total > 0.0 else 0.0

        confident_total = sum(self.confident_family_mass.values())
        confident_winner = self.confident_family_mass.get(key, 0.0)
        confident_dominance = (
            confident_winner / confident_total
            if confident_total > 0.0
            else 0.0
        )
        confident = (
            confident_winner >= 0.90
            and dominance >= 0.70
            and (
                confident_total <= 1e-12
                or confident_dominance >= 0.75
            )
        )
        return FamilyEvidence(
            key,
            confident,
            "actor_confident" if confident else "actor_weak",
            float(dominance),
        )


@dataclass(frozen=True, slots=True)
class HandoffCandidate:
    actor_id: str
    tracklet_id: str
    gap_s: float
    endpoint_distance_m: float
    reachability_limit_m: float
    geometry_support: float
    recency_support: float
    family_support: float
    family_relation: str
    velocity_support: float
    velocity_prediction_error_m: float | None
    score: float
    actor_family_key: str | None
    tracklet_family_key: str | None
    population_pressure: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "handoff_version": PERSISTENT_ACTOR_HANDOFF_VERSION,
            "actor_id": self.actor_id,
            "tracklet_id": self.tracklet_id,
            "gap_s": self.gap_s,
            "endpoint_distance_m": self.endpoint_distance_m,
            "reachability_limit_m": self.reachability_limit_m,
            "geometry_support": self.geometry_support,
            "recency_support": self.recency_support,
            "family_support": self.family_support,
            "family_relation": self.family_relation,
            "velocity_support": self.velocity_support,
            "velocity_prediction_error_m": (
                self.velocity_prediction_error_m
            ),
            "score": self.score,
            "actor_family_key": self.actor_family_key,
            "tracklet_family_key": self.tracklet_family_key,
            "population_pressure": self.population_pressure,
        }


@dataclass(frozen=True, slots=True)
class AcceptedHandoff:
    actor_id: str
    predecessor_tracklet_id: str
    successor_tracklet_id: str
    score: float
    second_best_score: float | None
    score_margin: float | None
    candidate_count: int
    acceptance_rule: str
    candidate: HandoffCandidate

    def to_dict(self) -> dict[str, Any]:
        row = self.candidate.to_dict()
        row.update({
            "predecessor_tracklet_id": self.predecessor_tracklet_id,
            "successor_tracklet_id": self.successor_tracklet_id,
            "second_best_score": self.second_best_score,
            "score_margin": self.score_margin,
            "candidate_count": self.candidate_count,
            "acceptance_rule": self.acceptance_rule,
            "identity_decision": "local_actor_handoff",
        })
        return row


@dataclass(frozen=True, slots=True)
class NewActorEvent:
    actor_id: str
    tracklet_id: str
    timestamp_s: float
    plausible_candidate_count: int
    best_candidate_score: float | None
    reason: str
    live_actor_count_before: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "handoff_version": PERSISTENT_ACTOR_HANDOFF_VERSION,
            "actor_id": self.actor_id,
            "tracklet_id": self.tracklet_id,
            "timestamp_s": self.timestamp_s,
            "plausible_candidate_count": self.plausible_candidate_count,
            "best_candidate_score": self.best_candidate_score,
            "reason": self.reason,
            "live_actor_count_before": self.live_actor_count_before,
        }


@dataclass(frozen=True, slots=True)
class PersistentActorHandoffResult:
    actors: tuple[dict[str, Any], ...]
    tracklet_assignments: tuple[dict[str, Any], ...]
    accepted_handoffs: tuple[AcceptedHandoff, ...]
    new_actor_events: tuple[NewActorEvent, ...]
    candidate_count: int
    hard_geometry_rejection_count: int
    hard_family_rejection_count: int
    peak_live_actor_count: int
    population_pressure_handoff_count: int
    config: PersistentActorHandoffConfig

    def summary(self) -> dict[str, Any]:
        family_relations = Counter(
            row.candidate.family_relation
            for row in self.accepted_handoffs
        )
        acceptance_rules = Counter(
            row.acceptance_rule
            for row in self.accepted_handoffs
        )
        return {
            "schema_version": 1,
            "handoff_version": PERSISTENT_ACTOR_HANDOFF_VERSION,
            "actor_count": len(self.actors),
            "tracklet_count": len(self.tracklet_assignments),
            "accepted_handoff_count": len(self.accepted_handoffs),
            "new_actor_count": len(self.new_actor_events),
            "candidate_count": self.candidate_count,
            "hard_geometry_rejection_count": (
                self.hard_geometry_rejection_count
            ),
            "hard_family_rejection_count": (
                self.hard_family_rejection_count
            ),
            "peak_live_actor_count": self.peak_live_actor_count,
            "soft_live_actor_limit": self.config.soft_live_actor_limit,
            "population_pressure_handoff_count": (
                self.population_pressure_handoff_count
            ),
            "population_hard_cap_applied": False,
            "population_forced_merge_count": 0,
            "previous_person_partition_used": False,
            "reid_used": False,
            "missing_family_is_negative": False,
            "actor_tracklet_count": _stats([
                len(row["tracklet_ids"]) for row in self.actors
            ]),
            "accepted_gap_s": _stats([
                row.candidate.gap_s
                for row in self.accepted_handoffs
            ]),
            "accepted_score": _stats([
                row.score for row in self.accepted_handoffs
            ]),
            "accepted_family_relation_counts": dict(
                sorted(family_relations.items())
            ),
            "acceptance_rule_counts": dict(
                sorted(acceptance_rules.items())
            ),
            "config": self.config.to_dict(),
        }


def _velocity(
    tracklet: Tracklet,
    config: PersistentActorHandoffConfig,
) -> tuple[float, float] | None:
    if len(tracklet.observations) < 2:
        return None
    first = tracklet.observations[-2]
    second = tracklet.observations[-1]
    dt = second.timestamp_s - first.timestamp_s
    if dt <= config.time_epsilon_s:
        return None
    vx = (second.x - first.x) / dt
    vz = (second.z - first.z) / dt
    if math.hypot(vx, vz) > config.velocity_max_reliable_speed_mps:
        return None
    return vx, vz


def _family_relation(
    actor: FamilyEvidence,
    tracklet: FamilyEvidence,
) -> tuple[str, float, bool]:
    if actor.key is None or tracklet.key is None:
        return "neutral", 0.50, False

    if actor.key == tracklet.key:
        if actor.confident and tracklet.confident:
            return "same_confident", 1.00, False
        return "same_weak", 0.80, False

    if actor.confident and tracklet.confident:
        return "different_confident", 0.0, True

    return "different_weak", 0.20, False


def _candidate(
    *,
    actor: _ActorState,
    successor: Tracklet,
    successor_family: FamilyEvidence,
    population_pressure: bool,
    config: PersistentActorHandoffConfig,
) -> tuple[HandoffCandidate | None, str | None]:
    predecessor = actor.last_tracklet
    if predecessor is None:
        return None, "missing_predecessor"

    gap_s = successor.start_timestamp_s - predecessor.end_timestamp_s
    if gap_s <= config.time_epsilon_s:
        return None, "overlap_or_same_time"
    if gap_s > config.max_handoff_gap_s + config.time_epsilon_s:
        return None, "gap_too_large"

    left = predecessor.observations[-1]
    right = successor.observations[0]
    distance = math.hypot(right.x - left.x, right.z - left.z)
    reachability_limit = (
        config.position_slack_m
        + config.max_transition_speed_mps * gap_s
    )
    if distance > reachability_limit + 1e-9:
        return None, "hard_geometry"

    actor_family = actor.family_evidence()
    relation, family_support, hard_family = _family_relation(
        actor_family,
        successor_family,
    )
    if hard_family:
        return None, "hard_family"

    geometry_support = _clamp01(
        1.0 - distance / max(reachability_limit, 1e-9)
    )
    recency_support = math.exp(-gap_s / config.recency_tau_s)

    velocity_prediction_error = None
    velocity_support = 0.50
    velocity = _velocity(predecessor, config)
    if velocity is not None and gap_s <= config.velocity_horizon_s:
        vx, vz = velocity
        predicted_x = left.x + vx * gap_s
        predicted_z = left.z + vz * gap_s
        velocity_prediction_error = math.hypot(
            right.x - predicted_x,
            right.z - predicted_z,
        )
        velocity_support = math.exp(
            -velocity_prediction_error
            / max(config.position_slack_m * 1.5, 1e-6)
        )

    weight_sum = (
        config.geometry_weight
        + config.recency_weight
        + config.family_weight
        + config.velocity_weight
    )
    score = (
        config.geometry_weight * geometry_support
        + config.recency_weight * recency_support
        + config.family_weight * family_support
        + config.velocity_weight * velocity_support
    ) / weight_sum

    return HandoffCandidate(
        actor_id=actor.actor_id,
        tracklet_id=successor.tracklet_id,
        gap_s=float(gap_s),
        endpoint_distance_m=float(distance),
        reachability_limit_m=float(reachability_limit),
        geometry_support=float(geometry_support),
        recency_support=float(recency_support),
        family_support=float(family_support),
        family_relation=relation,
        velocity_support=float(velocity_support),
        velocity_prediction_error_m=(
            float(velocity_prediction_error)
            if velocity_prediction_error is not None
            else None
        ),
        score=float(_clamp01(score)),
        actor_family_key=actor_family.key,
        tracklet_family_key=successor_family.key,
        population_pressure=population_pressure,
    ), None


def _acceptance(
    candidates: Sequence[HandoffCandidate],
    *,
    population_pressure: bool,
    config: PersistentActorHandoffConfig,
) -> tuple[bool, str, float | None, float | None]:
    if not candidates:
        return False, "no_plausible_candidate", None, None

    ordered = sorted(
        candidates,
        key=lambda row: (-row.score, row.actor_id),
    )
    best = ordered[0]
    second = ordered[1].score if len(ordered) > 1 else None
    margin = (
        best.score - second
        if second is not None
        else None
    )

    if len(ordered) == 1:
        floor = (
            min(
                config.unique_candidate_min_score,
                config.population_pressure_min_score,
            )
            if population_pressure
            else config.unique_candidate_min_score
        )
        if best.score >= floor:
            return True, (
                "unique_candidate_population_pressure"
                if population_pressure
                else "unique_candidate"
            ), second, margin
        return False, "unique_candidate_below_floor", second, margin

    floor = (
        min(config.min_score, config.population_pressure_min_score)
        if population_pressure
        else config.min_score
    )
    if best.score < floor:
        return False, "best_below_floor", second, margin
    if margin is None or margin < config.min_best_margin:
        return False, "ambiguous_competition", second, margin
    return True, (
        "best_margin_population_pressure"
        if population_pressure
        else "best_margin"
    ), second, margin


def build_persistent_actor_handoffs(
    *,
    tracklet_rows: Sequence[Mapping[str, Any] | Tracklet],
    family_rows: Sequence[Mapping[str, Any]] = (),
    config: PersistentActorHandoffConfig | None = None,
) -> PersistentActorHandoffResult:
    selected = config or PersistentActorHandoffConfig()

    tracklets = [
        row if isinstance(row, Tracklet) else Tracklet.from_dict(row)
        for row in tracklet_rows
    ]
    tracklets.sort(
        key=lambda row: (
            row.start_timestamp_s,
            row.start_frame,
            row.tracklet_id,
        )
    )
    if len({row.tracklet_id for row in tracklets}) != len(tracklets):
        raise ValueError("duplicate tracklet ids")

    family_by_tracklet: dict[str, Mapping[str, Any]] = {}
    for row in family_rows:
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id in family_by_tracklet:
            raise ValueError(f"duplicate family row {tracklet_id}")
        family_by_tracklet[tracklet_id] = row

    actors: dict[str, _ActorState] = {}
    assignments: dict[str, str] = {}
    handoffs: list[AcceptedHandoff] = []
    new_actor_events: list[NewActorEvent] = []

    candidate_count = 0
    hard_geometry_rejections = 0
    hard_family_rejections = 0
    pressure_handoffs = 0
    peak_live = 0
    next_actor = 1

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
        tracklet: Tracklet,
        *,
        plausible_count: int,
        best_score: float | None,
        reason: str,
        live_before: int,
    ) -> None:
        nonlocal next_actor
        actor_id = f"actor-{next_actor:04d}"
        next_actor += 1
        actor = _ActorState(actor_id=actor_id)
        actor.add_tracklet(
            tracklet,
            _tracklet_family(
                family_by_tracklet.get(tracklet.tracklet_id)
            ),
        )
        actors[actor_id] = actor
        assignments[tracklet.tracklet_id] = actor_id
        new_actor_events.append(
            NewActorEvent(
                actor_id=actor_id,
                tracklet_id=tracklet.tracklet_id,
                timestamp_s=tracklet.start_timestamp_s,
                plausible_candidate_count=plausible_count,
                best_candidate_score=best_score,
                reason=reason,
                live_actor_count_before=live_before,
            )
        )

    for batch in batches:
        now = batch[0].start_timestamp_s
        live_actor_ids = [
            actor_id
            for actor_id, actor in actors.items()
            if (
                now - actor.end_time_s
                <= selected.live_actor_memory_s
                + selected.time_epsilon_s
            )
        ]
        population_pressure = (
            len(live_actor_ids) + len(batch)
            > selected.soft_live_actor_limit
        )

        candidates_by_tracklet: dict[str, list[HandoffCandidate]] = {}
        candidates_by_actor: dict[str, list[HandoffCandidate]] = defaultdict(list)

        for successor in batch:
            rows: list[HandoffCandidate] = []
            successor_family = _tracklet_family(
                family_by_tracklet.get(successor.tracklet_id)
            )
            for actor_id in live_actor_ids:
                candidate, rejection = _candidate(
                    actor=actors[actor_id],
                    successor=successor,
                    successor_family=successor_family,
                    population_pressure=population_pressure,
                    config=selected,
                )
                if candidate is not None:
                    rows.append(candidate)
                    candidates_by_actor[actor_id].append(candidate)
                    candidate_count += 1
                elif rejection == "hard_geometry":
                    hard_geometry_rejections += 1
                elif rejection == "hard_family":
                    hard_family_rejections += 1

            rows.sort(key=lambda row: (-row.score, row.actor_id))
            candidates_by_tracklet[successor.tracklet_id] = rows

        actor_best: dict[str, str] = {}
        for actor_id, rows in candidates_by_actor.items():
            best = min(
                rows,
                key=lambda row: (-row.score, row.tracklet_id),
            )
            actor_best[actor_id] = best.tracklet_id

        proposals = []
        for successor in batch:
            rows = candidates_by_tracklet[successor.tracklet_id]
            accepted, rule, second, margin = _acceptance(
                rows,
                population_pressure=population_pressure,
                config=selected,
            )
            if not rows:
                continue
            best = rows[0]
            if (
                accepted
                and actor_best.get(best.actor_id)
                == successor.tracklet_id
            ):
                proposals.append(
                    (
                        -best.score,
                        successor.tracklet_id,
                        best,
                        rule,
                        second,
                        margin,
                        len(rows),
                    )
                )

        proposals.sort()
        by_tracklet = {row.tracklet_id: row for row in batch}
        accepted_tracklets: set[str] = set()
        accepted_actors: set[str] = set()

        for (
            _,
            tracklet_id,
            candidate,
            rule,
            second,
            margin,
            row_count,
        ) in proposals:
            if (
                tracklet_id in accepted_tracklets
                or candidate.actor_id in accepted_actors
            ):
                continue

            actor = actors[candidate.actor_id]
            successor = by_tracklet[tracklet_id]
            predecessor_id = actor.tracklet_ids[-1]
            actor.add_tracklet(
                successor,
                _tracklet_family(
                    family_by_tracklet.get(successor.tracklet_id)
                ),
            )
            assignments[tracklet_id] = actor.actor_id
            accepted_tracklets.add(tracklet_id)
            accepted_actors.add(actor.actor_id)

            if candidate.population_pressure:
                pressure_handoffs += 1

            handoffs.append(
                AcceptedHandoff(
                    actor_id=actor.actor_id,
                    predecessor_tracklet_id=predecessor_id,
                    successor_tracklet_id=tracklet_id,
                    score=candidate.score,
                    second_best_score=second,
                    score_margin=margin,
                    candidate_count=row_count,
                    acceptance_rule=rule,
                    candidate=candidate,
                )
            )

        for successor in batch:
            if successor.tracklet_id in accepted_tracklets:
                continue
            rows = candidates_by_tracklet[successor.tracklet_id]
            _, reason, _, _ = _acceptance(
                rows,
                population_pressure=population_pressure,
                config=selected,
            )
            create_actor(
                successor,
                plausible_count=len(rows),
                best_score=rows[0].score if rows else None,
                reason=reason,
                live_before=len(live_actor_ids),
            )

        current_live = sum(
            (
                now - actor.end_time_s
                <= selected.live_actor_memory_s
                + selected.time_epsilon_s
            )
            for actor in actors.values()
        )
        peak_live = max(peak_live, current_live)

    actor_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for actor_id in sorted(actors):
        actor = actors[actor_id]
        family = actor.family_evidence()
        actor_rows.append({
            "schema_version": 1,
            "handoff_version": PERSISTENT_ACTOR_HANDOFF_VERSION,
            "actor_id": actor_id,
            "tracklet_ids": list(actor.tracklet_ids),
            "start_time_s": actor.start_time_s,
            "end_time_s": actor.end_time_s,
            "family_key": family.key,
            "family_state": family.state,
            "family_confident": family.confident,
            "family_dominance": family.weight,
            "provisional": True,
        })
        for sequence_index, tracklet_id in enumerate(actor.tracklet_ids):
            assignment_rows.append({
                "schema_version": 1,
                "handoff_version": PERSISTENT_ACTOR_HANDOFF_VERSION,
                "tracklet_id": tracklet_id,
                "actor_id": actor_id,
                "actor_sequence_index": sequence_index,
                "provisional": True,
            })

    assignment_rows.sort(key=lambda row: row["tracklet_id"])

    return PersistentActorHandoffResult(
        actors=tuple(actor_rows),
        tracklet_assignments=tuple(assignment_rows),
        accepted_handoffs=tuple(handoffs),
        new_actor_events=tuple(new_actor_events),
        candidate_count=candidate_count,
        hard_geometry_rejection_count=hard_geometry_rejections,
        hard_family_rejection_count=hard_family_rejections,
        peak_live_actor_count=peak_live,
        population_pressure_handoff_count=pressure_handoffs,
        config=selected,
    )
