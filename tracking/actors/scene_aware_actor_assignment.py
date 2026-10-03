"""Tracking V1 Hito 7I.3: scene-aware persistent actor assignment.

Order of evidence:
1) exact current-frame visibility, 2) JUST_LOST pool, 3) hard physics /
cannot-link, 4) proximity shortlist, 5) kit compatibility, 6) ReID,
7) global one-to-one LOST<->NEW assignment.

Internal holes of a still-running tracklet are not treated as a disappearance.
The 24-player population value is a soft diagnostic prior only; no merge is
forced to hit a target count.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from tracking.tracklets.models import Tracklet


SCENE_AWARE_ACTOR_VERSION = "scene-aware-actor-assignment-v1"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("embedding must be a finite non-empty 1-D vector")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding has zero norm")
    return vector / norm


def _stats(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "median": None, "p90": None,
                "p95": None, "max": None, "mean": None}
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
        "count": len(ordered), "min": ordered[0], "median": pct(.50),
        "p90": pct(.90), "p95": pct(.95), "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


@dataclass(frozen=True, slots=True)
class SceneAwareActorConfig:
    local_lost_gap_s: float = 1.25
    recovery_gap_s: float = 5.0

    max_transition_speed_mps: float = 12.0
    position_slack_m: float = 1.25
    velocity_horizon_s: float = 0.85
    max_reliable_velocity_mps: float = 12.0

    local_shortlist_k: int = 4
    local_shortlist_extra_m: float = 2.5
    recovery_shortlist_k: int = 4

    geometry_weight: float = 0.38
    recency_weight: float = 0.17
    family_weight: float = 0.12
    velocity_weight: float = 0.08
    reid_weight: float = 0.25

    local_min_score: float = 0.54
    local_unique_min_score: float = 0.48
    recovery_min_score: float = 0.64
    recovery_reid_floor: float = 0.72
    recovery_reid_margin: float = 0.015

    soft_visible_actor_limit: int = 24
    population_score_relaxation: float = 0.03
    time_epsilon_s: float = 1e-6

    def __post_init__(self) -> None:
        if self.local_lost_gap_s <= 0:
            raise ValueError("local_lost_gap_s must be > 0")
        if self.recovery_gap_s <= self.local_lost_gap_s:
            raise ValueError("recovery_gap_s must be > local_lost_gap_s")
        for name in ("max_transition_speed_mps", "position_slack_m",
                     "velocity_horizon_s", "max_reliable_velocity_mps"):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.local_shortlist_k < 1 or self.recovery_shortlist_k < 1:
            raise ValueError("shortlist sizes must be >= 1")
        if self.local_shortlist_extra_m < 0:
            raise ValueError("local_shortlist_extra_m must be >= 0")
        weights = (self.geometry_weight, self.recency_weight,
                   self.family_weight, self.velocity_weight, self.reid_weight)
        if any(v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("evidence weights must be non-negative")
        for name in ("local_min_score", "local_unique_min_score",
                     "recovery_min_score", "recovery_reid_floor",
                     "recovery_reid_margin", "population_score_relaxation"):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.soft_visible_actor_limit < 1:
            raise ValueError("soft_visible_actor_limit must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
            return _Family(key, True, 1.0 if state == "confident_primary" else .90)
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
    tracklets: list[Tracklet] = field(default_factory=list)
    tracklet_ids: list[str] = field(default_factory=list)
    family_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    confident_family_mass: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    appearance_sum: np.ndarray | None = None

    @property
    def last_tracklet(self) -> Tracklet:
        return self.tracklets[-1]

    @property
    def end_time_s(self) -> float:
        return self.last_tracklet.end_timestamp_s

    @property
    def end_frame(self) -> int:
        return self.last_tracklet.end_frame

    @property
    def start_time_s(self) -> float:
        return self.tracklets[0].start_timestamp_s

    def add(self, tracklet: Tracklet, family: _Family,
            appearance: _Appearance | None) -> None:
        if self.tracklets and tracklet.start_frame <= self.end_frame:
            raise ValueError(
                f"{self.actor_id} received successor before prior tracklet ended"
            )
        self.tracklets.append(tracklet)
        self.tracklet_ids.append(tracklet.tracklet_id)

        if family.key is not None and family.mass > 0:
            self.family_mass[family.key] += family.mass
            if family.confident:
                self.confident_family_mass[family.key] += family.mass

        if appearance is not None:
            weight = max(
                1e-6,
                appearance.reliability * math.sqrt(appearance.effective_sample_count),
            )
            weighted = appearance.embedding * weight
            self.appearance_sum = (
                weighted.copy()
                if self.appearance_sum is None
                else self.appearance_sum + weighted
            )

    def family(self) -> _Family:
        if not self.family_mass:
            return _Family(None, False, 0.0)
        key = max(self.family_mass, key=lambda k: (self.family_mass[k], k))
        total = sum(self.family_mass.values())
        dominance = self.family_mass[key] / total if total else 0.0
        confident_total = sum(self.confident_family_mass.values())
        winner = self.confident_family_mass.get(key, 0.0)
        confident_dominance = winner / confident_total if confident_total else 0.0
        confident = (
            winner >= .90 and dominance >= .70
            and (confident_total <= 1e-12 or confident_dominance >= .75)
        )
        return _Family(key, confident, float(dominance))

    def appearance(self) -> np.ndarray | None:
        return None if self.appearance_sum is None else _unit(self.appearance_sum)


@dataclass(frozen=True, slots=True)
class SceneCandidate:
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
    visible_actor_count: int
    lost_actor_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "assignment_version": SCENE_AWARE_ACTOR_VERSION,
            "actor_id": self.actor_id, "tracklet_id": self.tracklet_id,
            "tier": self.tier, "gap_s": self.gap_s,
            "distance_m": self.distance_m,
            "reachability_limit_m": self.reachability_limit_m,
            "geometry_support": self.geometry_support,
            "recency_support": self.recency_support,
            "family_relation": self.family_relation,
            "family_support": self.family_support,
            "velocity_support": self.velocity_support,
            "velocity_prediction_error_m": self.velocity_prediction_error_m,
            "reid_cosine": self.reid_cosine,
            "reid_support": self.reid_support, "score": self.score,
            "visible_actor_count": self.visible_actor_count,
            "lost_actor_count": self.lost_actor_count,
        }


@dataclass(frozen=True, slots=True)
class AcceptedSceneHandoff:
    actor_id: str
    predecessor_tracklet_id: str
    successor_tracklet_id: str
    candidate: SceneCandidate
    successor_alternative_score: float | None
    actor_alternative_score: float | None
    rule: str

    def to_dict(self) -> dict[str, Any]:
        row = self.candidate.to_dict()
        row.update({
            "predecessor_tracklet_id": self.predecessor_tracklet_id,
            "successor_tracklet_id": self.successor_tracklet_id,
            "successor_alternative_score": self.successor_alternative_score,
            "actor_alternative_score": self.actor_alternative_score,
            "rule": self.rule,
            "identity_decision": "scene_aware_global_handoff",
        })
        return row


@dataclass(frozen=True, slots=True)
class NewActorEvent:
    actor_id: str
    tracklet_id: str
    frame_index: int
    timestamp_s: float
    reason: str
    visible_actor_count: int
    local_candidate_count: int
    recovery_candidate_count: int
    best_candidate_score: float | None
    best_reid_cosine: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1, "assignment_version": SCENE_AWARE_ACTOR_VERSION,
            "actor_id": self.actor_id, "tracklet_id": self.tracklet_id,
            "frame_index": self.frame_index, "timestamp_s": self.timestamp_s,
            "reason": self.reason, "visible_actor_count": self.visible_actor_count,
            "local_candidate_count": self.local_candidate_count,
            "recovery_candidate_count": self.recovery_candidate_count,
            "best_candidate_score": self.best_candidate_score,
            "best_reid_cosine": self.best_reid_cosine,
        }


@dataclass(frozen=True, slots=True)
class SceneAwareActorResult:
    actors: tuple[dict[str, Any], ...]
    tracklet_assignments: tuple[dict[str, Any], ...]
    accepted_handoffs: tuple[AcceptedSceneHandoff, ...]
    new_actor_events: tuple[NewActorEvent, ...]
    candidate_count: int
    visible_exclusion_count: int
    not_ended_exclusion_count: int
    hard_geometry_rejection_count: int
    hard_family_rejection_count: int
    hard_conflict_rejection_count: int
    scene_batch_count: int
    peak_exact_observation_count: int
    peak_visible_actor_count: int
    config: SceneAwareActorConfig

    def summary(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "assignment_version": SCENE_AWARE_ACTOR_VERSION,
            "actor_count": len(self.actors),
            "tracklet_count": len(self.tracklet_assignments),
            "accepted_handoff_count": len(self.accepted_handoffs),
            "new_actor_count": len(self.new_actor_events),
            "candidate_count": self.candidate_count,
            "scene_batch_count": self.scene_batch_count,
            "visible_exclusion_count": self.visible_exclusion_count,
            "not_ended_exclusion_count": self.not_ended_exclusion_count,
            "hard_geometry_rejection_count": self.hard_geometry_rejection_count,
            "hard_family_rejection_count": self.hard_family_rejection_count,
            "hard_conflict_rejection_count": self.hard_conflict_rejection_count,
            "peak_exact_observation_count": self.peak_exact_observation_count,
            "peak_visible_actor_count": self.peak_visible_actor_count,
            "soft_visible_actor_limit": self.config.soft_visible_actor_limit,
            "population_hard_cap_applied": False,
            "population_forced_merge_count": 0,
            "previous_person_partition_used": False,
            "exact_frame_scene_used": True,
            "global_assignment_used": True,
            "reid_used_after_physical_shortlist": True,
            "actor_tracklet_count": _stats([len(row["tracklet_ids"]) for row in self.actors]),
            "accepted_gap_s": _stats([row.candidate.gap_s for row in self.accepted_handoffs]),
            "accepted_score": _stats([row.candidate.score for row in self.accepted_handoffs]),
            "accepted_reid_cosine": _stats([
                row.candidate.reid_cosine for row in self.accepted_handoffs
                if row.candidate.reid_cosine is not None
            ]),
            "tier_counts": dict(sorted(Counter(
                row.candidate.tier for row in self.accepted_handoffs
            ).items())),
            "new_actor_reason_counts": dict(sorted(Counter(
                row.reason for row in self.new_actor_events
            ).items())),
            "config": self.config.to_dict(),
        }


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


def _velocity(tracklet: Tracklet, config: SceneAwareActorConfig) -> tuple[float, float] | None:
    if len(tracklet.observations) < 2:
        return None
    a, b = tracklet.observations[-2], tracklet.observations[-1]
    dt = b.timestamp_s - a.timestamp_s
    if dt <= config.time_epsilon_s:
        return None
    vx, vz = (b.x - a.x) / dt, (b.z - a.z) / dt
    return None if math.hypot(vx, vz) > config.max_reliable_velocity_mps else (vx, vz)


def _hard_pairs(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str]]:
    return {
        tuple(sorted((str(row["tracklet_a"]), str(row["tracklet_b"]))))
        for row in rows
    }


def _candidate(*, actor: _Actor, successor: Tracklet,
               successor_family: _Family,
               successor_appearance: _Appearance | None,
               tier: str, visible_count: int, lost_count: int,
               hard_pairs: set[tuple[str, str]],
               config: SceneAwareActorConfig) -> tuple[SceneCandidate | None, str | None]:
    gap = successor.start_timestamp_s - actor.end_time_s
    max_gap = config.local_lost_gap_s if tier == "local" else config.recovery_gap_s
    if gap <= config.time_epsilon_s:
        return None, "not_lost"
    if gap > max_gap + config.time_epsilon_s:
        return None, "gap_too_large"

    for member in actor.tracklet_ids:
        if tuple(sorted((member, successor.tracklet_id))) in hard_pairs:
            return None, "hard_conflict"

    end = actor.last_tracklet.observations[-1]
    start = successor.observations[0]
    distance = math.hypot(start.x - end.x, start.z - end.z)
    reachability = config.position_slack_m + config.max_transition_speed_mps * gap
    if distance > reachability + 1e-9:
        return None, "hard_geometry"

    relation, family_support, hard_family = _family_relation(
        actor.family(), successor_family
    )
    if hard_family:
        return None, "hard_family"

    geometry_support = _clamp01(1.0 - distance / max(reachability, 1e-9))
    recency_support = math.exp(-gap / (.55 if tier == "local" else 2.0))

    velocity_error = None
    velocity_support = .50
    velocity = _velocity(actor.last_tracklet, config)
    if velocity is not None and gap <= config.velocity_horizon_s:
        vx, vz = velocity
        velocity_error = math.hypot(
            start.x - (end.x + vx * gap),
            start.z - (end.z + vz * gap),
        )
        velocity_support = math.exp(
            -velocity_error / max(1e-6, config.position_slack_m * 1.5)
        )

    actor_embedding = actor.appearance()
    reid_cosine = None
    reid_support = .50
    if actor_embedding is not None and successor_appearance is not None:
        reid_cosine = float(np.clip(
            np.dot(actor_embedding, successor_appearance.embedding), -1.0, 1.0
        ))
        reid_support = _clamp01((reid_cosine + 1.0) * .5)

    if tier == "recovery":
        if reid_cosine is None:
            return None, "recovery_missing_reid"
        if reid_cosine < config.recovery_reid_floor:
            return None, "recovery_reid_below_floor"

    weights = (config.geometry_weight, config.recency_weight,
               config.family_weight, config.velocity_weight, config.reid_weight)
    score = (
        weights[0] * geometry_support
        + weights[1] * recency_support
        + weights[2] * family_support
        + weights[3] * velocity_support
        + weights[4] * reid_support
    ) / sum(weights)

    return SceneCandidate(
        actor_id=actor.actor_id, tracklet_id=successor.tracklet_id, tier=tier,
        gap_s=float(gap), distance_m=float(distance),
        reachability_limit_m=float(reachability),
        geometry_support=float(geometry_support),
        recency_support=float(recency_support),
        family_relation=relation, family_support=float(family_support),
        velocity_support=float(velocity_support),
        velocity_prediction_error_m=(
            float(velocity_error) if velocity_error is not None else None
        ),
        reid_cosine=reid_cosine, reid_support=float(reid_support),
        score=float(_clamp01(score)), visible_actor_count=visible_count,
        lost_actor_count=lost_count,
    ), None


def _shortlist(rows: Sequence[SceneCandidate], *, tier: str,
               config: SceneAwareActorConfig) -> list[SceneCandidate]:
    if not rows:
        return []
    ordered = sorted(
        rows,
        key=lambda row: (row.distance_m, row.gap_s, -row.score, row.actor_id),
    )
    k = config.local_shortlist_k if tier == "local" else config.recovery_shortlist_k
    result = ordered[:k]
    if tier == "local":
        nearest = result[0].distance_m
        result = [
            row for row in result
            if row.distance_m <= nearest + config.local_shortlist_extra_m
        ]
    return sorted(result, key=lambda row: (-row.score, row.actor_id))


def _hungarian_min(cost: list[list[float]]) -> list[int]:
    """Rectangular Hungarian algorithm, rows <= columns."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m:
        raise ValueError("Hungarian requires rows <= columns")

    u, v = [0.0] * (n + 1), [0.0] * (m + 1)
    p, way = [0] * (m + 1), [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], math.inf, 0
            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j], way[j] = cur, j0
                if minv[j] < delta:
                    delta, j1 = minv[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break

        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break

    result = [-1] * n
    for j in range(1, m + 1):
        if p[j]:
            result[p[j] - 1] = j - 1
    return result


def _global_assignment(*, successors: Sequence[Tracklet],
                       edges_by_tracklet: Mapping[str, Sequence[SceneCandidate]],
                       visible_count: int,
                       config: SceneAwareActorConfig) -> dict[str, SceneCandidate]:
    actor_ids = sorted({
        edge.actor_id
        for rows in edges_by_tracklet.values()
        for edge in rows
    })
    if not successors or not actor_ids:
        return {}

    actor_index = {actor_id: i for i, actor_id in enumerate(actor_ids)}
    n, m_real = len(successors), len(actor_ids)
    impossible = 1_000_000.0
    cost = [[impossible] * (m_real + n) for _ in successors]
    lookup: dict[tuple[int, int], SceneCandidate] = {}

    pressure = visible_count + len(successors) > config.soft_visible_actor_limit

    for i, successor in enumerate(successors):
        rows = list(edges_by_tracklet.get(successor.tracklet_id, ()))
        unique = len(rows) == 1
        for edge in rows:
            floor = (
                config.local_unique_min_score
                if edge.tier == "local" and unique
                else config.local_min_score
                if edge.tier == "local"
                else config.recovery_min_score
            )
            if pressure:
                floor = max(0.0, floor - config.population_score_relaxation)
            if edge.score < floor:
                continue
            j = actor_index[edge.actor_id]
            cost[i][j] = -edge.score
            lookup[(i, j)] = edge
        cost[i][m_real + i] = 0.0  # private unmatched dummy

    selected = _hungarian_min(cost)
    return {
        successors[i].tracklet_id: lookup[(i, j)]
        for i, j in enumerate(selected)
        if (i, j) in lookup
    }


def _peak_exact_observations(tracklets: Sequence[Tracklet]) -> int:
    counts: dict[int, int] = defaultdict(int)
    for tracklet in tracklets:
        for observation in tracklet.observations:
            counts[observation.frame_index] += 1
    return max(counts.values(), default=0)


def build_scene_aware_actor_assignment(
    *, tracklet_rows: Sequence[Mapping[str, Any] | Tracklet],
    family_rows: Sequence[Mapping[str, Any]] = (),
    appearance_rows: Sequence[Mapping[str, Any]] = (),
    hard_conflicts: Sequence[Mapping[str, Any]] = (),
    config: SceneAwareActorConfig | None = None,
) -> SceneAwareActorResult:
    selected = config or SceneAwareActorConfig()

    tracklets = [
        row if isinstance(row, Tracklet) else Tracklet.from_dict(row)
        for row in tracklet_rows
    ]
    tracklets.sort(key=lambda row: (
        row.start_frame, row.start_timestamp_s, row.tracklet_id
    ))
    if len({row.tracklet_id for row in tracklets}) != len(tracklets):
        raise ValueError("duplicate tracklet ids")

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

    observations_by_frame: dict[int, list[str]] = defaultdict(list)
    for tracklet in tracklets:
        for observation in tracklet.observations:
            observations_by_frame[observation.frame_index].append(tracklet.tracklet_id)

    batches: list[list[Tracklet]] = []
    for tracklet in tracklets:
        if not batches or batches[-1][0].start_frame != tracklet.start_frame:
            batches.append([tracklet])
        else:
            batches[-1].append(tracklet)

    actors: dict[str, _Actor] = {}
    assignment_by_tracklet: dict[str, str] = {}
    accepted: list[AcceptedSceneHandoff] = []
    new_events: list[NewActorEvent] = []

    candidate_count = 0
    visible_exclusions = 0
    not_ended_exclusions = 0
    geometry_rejections = 0
    family_rejections = 0
    conflict_rejections = 0
    next_actor = 1
    peak_visible = 0

    def create_actor(successor: Tracklet, *, reason: str, visible_count: int,
                     local_count: int, recovery_count: int,
                     best: SceneCandidate | None) -> None:
        nonlocal next_actor
        actor_id = f"actor-{next_actor:04d}"
        next_actor += 1
        actor = _Actor(actor_id=actor_id)
        actor.add(
            successor,
            _family(family_by_tracklet.get(successor.tracklet_id)),
            _appearance(appearance_by_tracklet.get(successor.tracklet_id)),
        )
        actors[actor_id] = actor
        assignment_by_tracklet[successor.tracklet_id] = actor_id
        new_events.append(NewActorEvent(
            actor_id=actor_id, tracklet_id=successor.tracklet_id,
            frame_index=successor.start_frame,
            timestamp_s=successor.start_timestamp_s, reason=reason,
            visible_actor_count=visible_count,
            local_candidate_count=local_count,
            recovery_candidate_count=recovery_count,
            best_candidate_score=best.score if best else None,
            best_reid_cosine=best.reid_cosine if best else None,
        ))

    for batch in batches:
        frame = batch[0].start_frame
        exact_visible_actor_ids = {
            assignment_by_tracklet[tid]
            for tid in observations_by_frame.get(frame, ())
            if tid in assignment_by_tracklet
        }
        visible_count = len(exact_visible_actor_ids)
        peak_visible = max(peak_visible, visible_count + len(batch))

        local_by_tracklet: dict[str, list[SceneCandidate]] = {}
        recovery_by_tracklet: dict[str, list[SceneCandidate]] = {}

        for successor in batch:
            sf = _family(family_by_tracklet.get(successor.tracklet_id))
            sa = _appearance(appearance_by_tracklet.get(successor.tracklet_id))

            lost_ids = [
                actor_id for actor_id, actor in actors.items()
                if actor_id not in exact_visible_actor_ids
                and actor.end_frame < frame
                and 0 < successor.start_timestamp_s - actor.end_time_s
                <= selected.recovery_gap_s + selected.time_epsilon_s
            ]
            lost_count = len(lost_ids)
            local_raw, recovery_raw = [], []

            for actor_id, actor in actors.items():
                if actor_id in exact_visible_actor_ids:
                    visible_exclusions += 1
                    continue
                # A hole inside a tracklet that later resumes is not a loss.
                if actor.end_frame >= frame:
                    not_ended_exclusions += 1
                    continue

                gap = successor.start_timestamp_s - actor.end_time_s
                if gap <= selected.time_epsilon_s or gap > selected.recovery_gap_s:
                    continue
                tier = "local" if gap <= selected.local_lost_gap_s else "recovery"

                edge, rejection = _candidate(
                    actor=actor, successor=successor,
                    successor_family=sf, successor_appearance=sa,
                    tier=tier, visible_count=visible_count,
                    lost_count=lost_count, hard_pairs=hard, config=selected,
                )
                if edge is not None:
                    candidate_count += 1
                    (local_raw if tier == "local" else recovery_raw).append(edge)
                elif rejection == "hard_geometry":
                    geometry_rejections += 1
                elif rejection == "hard_family":
                    family_rejections += 1
                elif rejection == "hard_conflict":
                    conflict_rejections += 1

            local = _shortlist(local_raw, tier="local", config=selected)
            recovery = [] if local else _shortlist(
                recovery_raw, tier="recovery", config=selected
            )

            # Long recovery must also be distinguishable by ReID.
            if recovery:
                reids = sorted(
                    [edge.reid_cosine for edge in recovery if edge.reid_cosine is not None],
                    reverse=True,
                )
                if len(reids) >= 2 and reids[0] - reids[1] < selected.recovery_reid_margin:
                    recovery = []

            local_by_tracklet[successor.tracklet_id] = local
            recovery_by_tracklet[successor.tracklet_id] = recovery

        edges_by_tracklet = {
            successor.tracklet_id: (
                local_by_tracklet[successor.tracklet_id]
                or recovery_by_tracklet[successor.tracklet_id]
            )
            for successor in batch
        }
        chosen = _global_assignment(
            successors=batch, edges_by_tracklet=edges_by_tracklet,
            visible_count=visible_count, config=selected,
        )

        for successor in batch:
            edge = chosen.get(successor.tracklet_id)
            if edge is None:
                continue
            actor = actors[edge.actor_id]
            predecessor_id = actor.tracklet_ids[-1]

            succ_alt = sorted(
                [row.score for row in edges_by_tracklet[successor.tracklet_id]
                 if row.actor_id != edge.actor_id],
                reverse=True,
            )
            actor_alt = sorted(
                [row.score
                 for other in batch if other.tracklet_id != successor.tracklet_id
                 for row in edges_by_tracklet[other.tracklet_id]
                 if row.actor_id == edge.actor_id],
                reverse=True,
            )

            actor.add(
                successor,
                _family(family_by_tracklet.get(successor.tracklet_id)),
                _appearance(appearance_by_tracklet.get(successor.tracklet_id)),
            )
            assignment_by_tracklet[successor.tracklet_id] = actor.actor_id
            accepted.append(AcceptedSceneHandoff(
                actor_id=actor.actor_id,
                predecessor_tracklet_id=predecessor_id,
                successor_tracklet_id=successor.tracklet_id,
                candidate=edge,
                successor_alternative_score=succ_alt[0] if succ_alt else None,
                actor_alternative_score=actor_alt[0] if actor_alt else None,
                rule=("global_local_assignment" if edge.tier == "local"
                      else "global_reid_recovery"),
            ))

        for successor in batch:
            if successor.tracklet_id in chosen:
                continue
            local = local_by_tracklet[successor.tracklet_id]
            recovery = recovery_by_tracklet[successor.tracklet_id]
            edges = local or recovery
            best = max(edges, key=lambda row: row.score, default=None)

            if not edges:
                reason = "no_scene_candidate"
            else:
                floor = (
                    selected.local_unique_min_score
                    if best.tier == "local" and len(edges) == 1
                    else selected.local_min_score
                    if best.tier == "local"
                    else selected.recovery_min_score
                )
                if visible_count + len(batch) > selected.soft_visible_actor_limit:
                    floor = max(0.0, floor - selected.population_score_relaxation)
                reason = (
                    "candidate_below_floor"
                    if best.score < floor
                    else "global_assignment_left_unmatched"
                )
            create_actor(
                successor, reason=reason, visible_count=visible_count,
                local_count=len(local), recovery_count=len(recovery), best=best,
            )

    actor_rows, assignment_rows = [], []
    for actor_id in sorted(actors):
        actor = actors[actor_id]
        fam = actor.family()
        actor_rows.append({
            "schema_version": 1, "assignment_version": SCENE_AWARE_ACTOR_VERSION,
            "actor_id": actor_id, "tracklet_ids": list(actor.tracklet_ids),
            "start_time_s": actor.start_time_s, "end_time_s": actor.end_time_s,
            "family_key": fam.key, "family_confident": fam.confident,
            "family_dominance": fam.mass,
            "appearance_available": actor.appearance() is not None,
            "provisional": True,
        })
        for sequence_index, tracklet_id in enumerate(actor.tracklet_ids):
            assignment_rows.append({
                "schema_version": 1,
                "assignment_version": SCENE_AWARE_ACTOR_VERSION,
                "tracklet_id": tracklet_id, "actor_id": actor_id,
                "actor_sequence_index": sequence_index, "provisional": True,
            })

    assignment_rows.sort(key=lambda row: row["tracklet_id"])
    return SceneAwareActorResult(
        actors=tuple(actor_rows), tracklet_assignments=tuple(assignment_rows),
        accepted_handoffs=tuple(accepted), new_actor_events=tuple(new_events),
        candidate_count=candidate_count,
        visible_exclusion_count=visible_exclusions,
        not_ended_exclusion_count=not_ended_exclusions,
        hard_geometry_rejection_count=geometry_rejections,
        hard_family_rejection_count=family_rejections,
        hard_conflict_rejection_count=conflict_rejections,
        scene_batch_count=len(batches),
        peak_exact_observation_count=_peak_exact_observations(tracklets),
        peak_visible_actor_count=peak_visible,
        config=selected,
    )
