"""Tracking V1 Hito 7I.4: fixed persistent actor roster.

Football does not create a new physical actor every time a detector/tracker
fragment starts. This stage creates a fixed set of actor slots (23 by default),
seeds them from an early exact-frame scene, and only ever re-attaches future
tracklets to those same slots.

An unmatched tracklet does NOT create a new actor. It remains unassigned for
later recovery/audit.

Decision order:
1. Exact-frame occupancy / coexistence.
2. Existing hard cannot-link conflicts.
3. Recent physical continuity and endpoint distance.
4. Anonymous kit-family compatibility.
5. ReID after the physical shortlist.
6. One-to-one global assignment for births in the same frame.

Tracklet spans may overlap because conservative tracklets can contain internal
holes. Availability is therefore based on actual observation frames, not only
[start_frame, end_frame].
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import math
from typing import Any, Mapping, Sequence

import numpy as np

from tracking.tracklets.models import Tracklet


FIXED_ACTOR_ROSTER_VERSION = "fixed-actor-roster-v1"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("embedding has zero norm")
    return vector / norm


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
    ordered = sorted(float(v) for v in values)

    def pct(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        pos = (len(ordered) - 1) * p
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
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
class FixedActorRosterConfig:
    actor_slots: int = 23
    bootstrap_window_s: float = 3.0
    bootstrap_fill_window_s: float = 2.0

    local_gap_s: float = 1.25
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

    local_min_score: float = 0.53
    local_unique_min_score: float = 0.47
    recovery_min_score: float = 0.63
    recovery_reid_floor: float = 0.72

    time_epsilon_s: float = 1e-6

    def __post_init__(self) -> None:
        if self.actor_slots < 1:
            raise ValueError("actor_slots must be >= 1")
        if self.bootstrap_window_s <= 0:
            raise ValueError("bootstrap_window_s must be > 0")
        if self.bootstrap_fill_window_s < 0:
            raise ValueError("bootstrap_fill_window_s must be >= 0")
        if not 0 < self.local_gap_s < self.recovery_gap_s:
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
            self.geometry_weight,
            self.recency_weight,
            self.family_weight,
            self.velocity_weight,
            self.reid_weight,
        )
        if any(v < 0 for v in weights) or sum(weights) <= 0:
            raise ValueError("evidence weights must be non-negative")
        for name in (
            "local_min_score",
            "local_unique_min_score",
            "recovery_min_score",
            "recovery_reid_floor",
        ):
            value = float(getattr(self, name))
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0,1]")

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
            return _Family(
                key,
                True,
                1.0 if state == "confident_primary" else 0.90,
            )

    if state in {"weak_primary", "weak_special"}:
        key = row.get("candidate_family_key")
        if isinstance(key, str) and key:
            return _Family(
                key,
                False,
                0.35 if state == "weak_primary" else 0.15,
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


@dataclass(slots=True)
class _Slot:
    actor_id: str
    tracklets: list[Tracklet] = field(default_factory=list)
    tracklet_ids: list[str] = field(default_factory=list)
    occupied_frames: set[int] = field(default_factory=set)
    family_mass: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    confident_family_mass: dict[str, float] = field(
        default_factory=lambda: defaultdict(float)
    )
    appearance_sum: np.ndarray | None = None
    appearance_weight: float = 0.0

    @property
    def initialized(self) -> bool:
        return bool(self.tracklets)

    def add(
        self,
        tracklet: Tracklet,
        family: _Family,
        appearance: _Appearance | None,
    ) -> None:
        frames = {obs.frame_index for obs in tracklet.observations}
        overlap = self.occupied_frames.intersection(frames)
        if overlap:
            raise ValueError(
                f"{self.actor_id} has exact-frame coexistence at "
                f"{min(overlap)}"
            )

        self.tracklets.append(tracklet)
        self.tracklet_ids.append(tracklet.tracklet_id)
        self.occupied_frames.update(frames)

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

    def family(self) -> _Family:
        if not self.family_mass:
            return _Family(None, False, 0.0)

        key = max(
            self.family_mass,
            key=lambda k: (self.family_mass[k], k),
        )
        total = sum(self.family_mass.values())
        dominance = self.family_mass[key] / total if total else 0.0

        confident_total = sum(self.confident_family_mass.values())
        confident_winner = self.confident_family_mass.get(key, 0.0)
        confident_dominance = (
            confident_winner / confident_total
            if confident_total > 0
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
        return _Family(key, confident, float(dominance))

    def appearance(self) -> np.ndarray | None:
        if self.appearance_sum is None:
            return None
        return _unit(self.appearance_sum)

    def observations(self):
        result = []
        for tracklet in self.tracklets:
            result.extend(tracklet.observations)
        return sorted(
            result,
            key=lambda obs: (
                obs.frame_index,
                obs.timestamp_s,
                obs.detection_id,
            ),
        )

    def latest_before(self, frame_index: int):
        candidates = [
            obs for obs in self.observations()
            if obs.frame_index < frame_index
        ]
        return candidates[-1] if candidates else None

    def previous_before(self, frame_index: int):
        candidates = [
            obs for obs in self.observations()
            if obs.frame_index < frame_index
        ]
        return candidates[-2] if len(candidates) >= 2 else None


@dataclass(frozen=True, slots=True)
class RosterCandidate:
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
    reid_cosine: float | None
    reid_support: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "roster_version": FIXED_ACTOR_ROSTER_VERSION,
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
            "reid_cosine": self.reid_cosine,
            "reid_support": self.reid_support,
            "score": self.score,
        }


@dataclass(frozen=True, slots=True)
class AcceptedRosterHandoff:
    actor_id: str
    tracklet_id: str
    candidate: RosterCandidate
    phase: str
    rule: str

    def to_dict(self) -> dict[str, Any]:
        row = self.candidate.to_dict()
        row.update({
            "phase": self.phase,
            "rule": self.rule,
            "identity_decision": "fixed_actor_slot_assignment",
        })
        return row


@dataclass(frozen=True, slots=True)
class FixedActorRosterResult:
    actors: tuple[dict[str, Any], ...]
    tracklet_assignments: tuple[dict[str, Any], ...]
    accepted_handoffs: tuple[AcceptedRosterHandoff, ...]
    unassigned_tracklets: tuple[dict[str, Any], ...]
    bootstrap_frame: int
    bootstrap_timestamp_s: float
    bootstrap_observation_count: int
    bootstrap_seed_count: int
    candidate_count: int
    exact_visible_exclusion_count: int
    exact_coexistence_rejection_count: int
    hard_conflict_rejection_count: int
    hard_geometry_rejection_count: int
    hard_family_rejection_count: int
    config: FixedActorRosterConfig

    def summary(self) -> dict[str, Any]:
        assigned = len(self.tracklet_assignments)
        unassigned = len(self.unassigned_tracklets)
        return {
            "schema_version": 1,
            "roster_version": FIXED_ACTOR_ROSTER_VERSION,
            "actor_slot_count": len(self.actors),
            "actor_count_is_hard_capped": True,
            "tracklet_count": assigned + unassigned,
            "assigned_tracklet_count": assigned,
            "unassigned_tracklet_count": unassigned,
            "assignment_coverage": (
                assigned / (assigned + unassigned)
                if assigned + unassigned
                else 0.0
            ),
            "accepted_handoff_count": len(self.accepted_handoffs),
            "bootstrap_frame": self.bootstrap_frame,
            "bootstrap_timestamp_s": self.bootstrap_timestamp_s,
            "bootstrap_observation_count": self.bootstrap_observation_count,
            "bootstrap_seed_count": self.bootstrap_seed_count,
            "candidate_count": self.candidate_count,
            "exact_visible_exclusion_count": self.exact_visible_exclusion_count,
            "exact_coexistence_rejection_count": (
                self.exact_coexistence_rejection_count
            ),
            "hard_conflict_rejection_count": self.hard_conflict_rejection_count,
            "hard_geometry_rejection_count": self.hard_geometry_rejection_count,
            "hard_family_rejection_count": self.hard_family_rejection_count,
            "new_actor_creation_after_bootstrap": False,
            "unmatched_tracklet_creates_actor": False,
            "reid_used_after_physical_shortlist": True,
            "actor_tracklet_count": _stats([
                len(row["tracklet_ids"]) for row in self.actors
            ]),
            "accepted_score": _stats([
                row.candidate.score for row in self.accepted_handoffs
            ]),
            "accepted_reid_cosine": _stats([
                row.candidate.reid_cosine
                for row in self.accepted_handoffs
                if row.candidate.reid_cosine is not None
            ]),
            "phase_counts": dict(sorted(Counter(
                row.phase for row in self.accepted_handoffs
            ).items())),
            "tier_counts": dict(sorted(Counter(
                row.candidate.tier for row in self.accepted_handoffs
            ).items())),
            "unassigned_reason_counts": dict(sorted(Counter(
                str(row["reason"]) for row in self.unassigned_tracklets
            ).items())),
            "config": self.config.to_dict(),
        }


def _family_relation(
    actor: _Family,
    successor: _Family,
) -> tuple[str, float, bool]:
    if actor.key is None or successor.key is None:
        return "neutral", 0.50, False
    if actor.key == successor.key:
        if actor.confident and successor.confident:
            return "same_confident", 1.0, False
        return "same_weak", 0.80, False
    if actor.confident and successor.confident:
        return "different_confident", 0.0, True
    return "different_weak", 0.20, False


def _hard_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    return {
        tuple(sorted((str(row["tracklet_a"]), str(row["tracklet_b"]))))
        for row in rows
    }


def _candidate(
    *,
    slot: _Slot,
    successor: Tracklet,
    successor_family: _Family,
    successor_appearance: _Appearance | None,
    hard_pairs: set[tuple[str, str]],
    config: FixedActorRosterConfig,
) -> tuple[RosterCandidate | None, str | None]:
    if successor.start_frame in slot.occupied_frames:
        return None, "exact_visible"

    successor_frames = {obs.frame_index for obs in successor.observations}
    if slot.occupied_frames.intersection(successor_frames):
        return None, "exact_coexistence"

    for member in slot.tracklet_ids:
        if tuple(sorted((member, successor.tracklet_id))) in hard_pairs:
            return None, "hard_conflict"

    previous = slot.latest_before(successor.start_frame)
    if previous is None:
        return None, "no_previous_observation"

    first = successor.observations[0]
    gap_s = first.timestamp_s - previous.timestamp_s
    if gap_s <= config.time_epsilon_s:
        return None, "non_forward"
    if gap_s > config.recovery_gap_s + config.time_epsilon_s:
        return None, "gap_too_large"

    tier = "local" if gap_s <= config.local_gap_s else "recovery"

    distance = math.hypot(first.x - previous.x, first.z - previous.z)
    limit = config.position_slack_m + config.max_transition_speed_mps * gap_s
    if distance > limit + 1e-9:
        return None, "hard_geometry"

    relation, family_support, hard_family = _family_relation(
        slot.family(), successor_family
    )
    if hard_family:
        return None, "hard_family"

    geometry_support = _clamp01(1.0 - distance / max(limit, 1e-9))
    recency_support = math.exp(-gap_s / (0.55 if tier == "local" else 2.0))

    velocity_support = 0.50
    earlier = slot.previous_before(successor.start_frame)
    if earlier is not None and gap_s <= config.velocity_horizon_s:
        dt = previous.timestamp_s - earlier.timestamp_s
        if dt > config.time_epsilon_s:
            vx = (previous.x - earlier.x) / dt
            vz = (previous.z - earlier.z) / dt
            speed = math.hypot(vx, vz)
            if speed <= config.max_reliable_velocity_mps:
                px = previous.x + vx * gap_s
                pz = previous.z + vz * gap_s
                error = math.hypot(first.x - px, first.z - pz)
                velocity_support = math.exp(
                    -error / max(1e-6, config.position_slack_m * 1.5)
                )

    reid_cosine = None
    reid_support = 0.50
    slot_embedding = slot.appearance()
    if slot_embedding is not None and successor_appearance is not None:
        reid_cosine = float(np.clip(
            np.dot(slot_embedding, successor_appearance.embedding),
            -1.0,
            1.0,
        ))
        reid_support = _clamp01((reid_cosine + 1.0) * 0.5)

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

    return RosterCandidate(
        actor_id=slot.actor_id,
        tracklet_id=successor.tracklet_id,
        tier=tier,
        gap_s=float(gap_s),
        distance_m=float(distance),
        reachability_limit_m=float(limit),
        geometry_support=float(geometry_support),
        recency_support=float(recency_support),
        family_relation=relation,
        family_support=float(family_support),
        velocity_support=float(velocity_support),
        reid_cosine=reid_cosine,
        reid_support=float(reid_support),
        score=float(_clamp01(score)),
    ), None


def _shortlist(
    rows: Sequence[RosterCandidate],
    config: FixedActorRosterConfig,
) -> list[RosterCandidate]:
    if not rows:
        return []

    local = [row for row in rows if row.tier == "local"]
    recovery = [row for row in rows if row.tier == "recovery"]

    if local:
        ordered = sorted(
            local,
            key=lambda row: (
                row.distance_m,
                row.gap_s,
                -row.score,
                row.actor_id,
            ),
        )[:config.local_shortlist_k]
        nearest = ordered[0].distance_m
        ordered = [
            row for row in ordered
            if row.distance_m <= nearest + config.local_shortlist_extra_m
        ]
        return sorted(ordered, key=lambda row: (-row.score, row.actor_id))

    ordered = sorted(
        recovery,
        key=lambda row: (
            row.distance_m,
            row.gap_s,
            -row.score,
            row.actor_id,
        ),
    )[:config.recovery_shortlist_k]
    return sorted(ordered, key=lambda row: (-row.score, row.actor_id))


def _hungarian_min(cost: list[list[float]]) -> list[int]:
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if n > m:
        raise ValueError("Hungarian requires rows <= columns")

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [math.inf] * (m + 1)
        used = [False] * (m + 1)

        while True:
            used[j0] = True
            i0 = p[j0]
            delta = math.inf
            j1 = 0

            for j in range(1, m + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j

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

    assignment = [-1] * n
    for j in range(1, m + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    return assignment


def _solve_batch(
    *,
    tracklets: Sequence[Tracklet],
    candidates_by_tracklet: Mapping[str, Sequence[RosterCandidate]],
    config: FixedActorRosterConfig,
) -> dict[str, RosterCandidate]:
    actor_ids = sorted({
        row.actor_id
        for rows in candidates_by_tracklet.values()
        for row in rows
    })
    if not tracklets or not actor_ids:
        return {}

    actor_index = {actor_id: index for index, actor_id in enumerate(actor_ids)}
    columns = len(actor_ids) + len(tracklets)
    impossible = 1_000_000.0
    cost = [[impossible] * columns for _ in tracklets]
    lookup: dict[tuple[int, int], RosterCandidate] = {}

    for row_index, tracklet in enumerate(tracklets):
        rows = list(candidates_by_tracklet.get(tracklet.tracklet_id, ()))
        unique = len(rows) == 1

        for candidate in rows:
            floor = (
                config.local_unique_min_score
                if candidate.tier == "local" and unique
                else config.local_min_score
                if candidate.tier == "local"
                else config.recovery_min_score
            )
            if candidate.score < floor:
                continue

            col = actor_index[candidate.actor_id]
            cost[row_index][col] = -candidate.score
            lookup[(row_index, col)] = candidate

        cost[row_index][len(actor_ids) + row_index] = 0.0

    assignment = _hungarian_min(cost)
    result: dict[str, RosterCandidate] = {}

    for row_index, col in enumerate(assignment):
        candidate = lookup.get((row_index, col))
        if candidate is not None:
            result[tracklets[row_index].tracklet_id] = candidate

    return result


def _bootstrap_frame(
    tracklets: Sequence[Tracklet],
    config: FixedActorRosterConfig,
) -> tuple[int, float, list[Tracklet]]:
    observations: dict[int, list[Tracklet]] = defaultdict(list)
    timestamp_by_frame: dict[int, float] = {}

    first_timestamp = min(
        obs.timestamp_s
        for tracklet in tracklets
        for obs in tracklet.observations
    )

    for tracklet in tracklets:
        for obs in tracklet.observations:
            if (
                obs.timestamp_s
                <= first_timestamp + config.bootstrap_window_s + config.time_epsilon_s
            ):
                observations[obs.frame_index].append(tracklet)
                timestamp_by_frame.setdefault(obs.frame_index, obs.timestamp_s)

    if not observations:
        raise ValueError("no observations in bootstrap window")

    def key(frame: int):
        count = len(observations[frame])
        return (
            abs(count - config.actor_slots),
            0 if count >= config.actor_slots else 1,
            frame,
        )

    frame = min(observations, key=key)
    timestamp = timestamp_by_frame[frame]
    active = list(observations[frame])

    active.sort(
        key=lambda tracklet: (
            -tracklet.observation_count,
            -tracklet.mean_confidence,
            -tracklet.duration_s,
            tracklet.tracklet_id,
        )
    )
    active = active[:config.actor_slots]

    def position(tracklet: Tracklet):
        obs = next(
            obs for obs in tracklet.observations
            if obs.frame_index == frame
        )
        return (obs.z, obs.x, tracklet.tracklet_id)

    active.sort(key=position)
    return frame, timestamp, active


def build_fixed_actor_roster(
    *,
    tracklet_rows: Sequence[Mapping[str, Any] | Tracklet],
    family_rows: Sequence[Mapping[str, Any]] = (),
    appearance_rows: Sequence[Mapping[str, Any]] = (),
    hard_conflicts: Sequence[Mapping[str, Any]] = (),
    config: FixedActorRosterConfig | None = None,
) -> FixedActorRosterResult:
    selected = config or FixedActorRosterConfig()

    tracklets = [
        row if isinstance(row, Tracklet) else Tracklet.from_dict(row)
        for row in tracklet_rows
    ]
    tracklets.sort(
        key=lambda row: (
            row.start_frame,
            row.start_timestamp_s,
            row.tracklet_id,
        )
    )
    if not tracklets:
        raise ValueError("tracklets are empty")
    if len({row.tracklet_id for row in tracklets}) != len(tracklets):
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
            raise ValueError(f"duplicate appearance row {tid}")
        appearance_by_tracklet[tid] = row

    hard = _hard_pairs(hard_conflicts)
    bootstrap_frame, bootstrap_time, seeds = _bootstrap_frame(tracklets, selected)

    slots = {
        f"actor-{index:04d}": _Slot(actor_id=f"actor-{index:04d}")
        for index in range(1, selected.actor_slots + 1)
    }

    assignment_by_tracklet: dict[str, str] = {}
    accepted: list[AcceptedRosterHandoff] = []
    unassigned: dict[str, dict[str, Any]] = {}

    for actor_id, tracklet in zip(sorted(slots), seeds):
        slot = slots[actor_id]
        slot.add(
            tracklet,
            _family(family_by_tracklet.get(tracklet.tracklet_id)),
            _appearance(appearance_by_tracklet.get(tracklet.tracklet_id)),
        )
        assignment_by_tracklet[tracklet.tracklet_id] = actor_id

    counters = Counter()
    candidate_count = 0

    forward = [
        tracklet
        for tracklet in tracklets
        if (
            tracklet.tracklet_id not in assignment_by_tracklet
            and tracklet.start_frame > bootstrap_frame
        )
    ]

    batches: list[list[Tracklet]] = []
    for tracklet in forward:
        if not batches or batches[-1][0].start_frame != tracklet.start_frame:
            batches.append([tracklet])
        else:
            batches[-1].append(tracklet)

    for batch in batches:
        candidate_rows: dict[str, list[RosterCandidate]] = {}

        for successor in batch:
            rows: list[RosterCandidate] = []

            for slot in slots.values():
                if not slot.initialized:
                    continue

                candidate, rejection = _candidate(
                    slot=slot,
                    successor=successor,
                    successor_family=_family(
                        family_by_tracklet.get(successor.tracklet_id)
                    ),
                    successor_appearance=_appearance(
                        appearance_by_tracklet.get(successor.tracklet_id)
                    ),
                    hard_pairs=hard,
                    config=selected,
                )

                if candidate is not None:
                    rows.append(candidate)
                    candidate_count += 1
                elif rejection is not None:
                    counters[rejection] += 1

            candidate_rows[successor.tracklet_id] = _shortlist(rows, selected)

        chosen = _solve_batch(
            tracklets=batch,
            candidates_by_tracklet=candidate_rows,
            config=selected,
        )

        for successor in batch:
            candidate = chosen.get(successor.tracklet_id)
            if candidate is None:
                continue

            slot = slots[candidate.actor_id]
            slot.add(
                successor,
                _family(family_by_tracklet.get(successor.tracklet_id)),
                _appearance(appearance_by_tracklet.get(successor.tracklet_id)),
            )
            assignment_by_tracklet[successor.tracklet_id] = candidate.actor_id
            accepted.append(
                AcceptedRosterHandoff(
                    actor_id=candidate.actor_id,
                    tracklet_id=successor.tracklet_id,
                    candidate=candidate,
                    phase="forward",
                    rule="global_fixed_roster_assignment",
                )
            )

        empty_slots = [slot for slot in slots.values() if not slot.initialized]
        within_fill_window = (
            batch[0].start_timestamp_s
            <= bootstrap_time
            + selected.bootstrap_fill_window_s
            + selected.time_epsilon_s
        )

        for successor in batch:
            if successor.tracklet_id in assignment_by_tracklet:
                continue

            if empty_slots and within_fill_window:
                slot = empty_slots.pop(0)
                slot.add(
                    successor,
                    _family(family_by_tracklet.get(successor.tracklet_id)),
                    _appearance(appearance_by_tracklet.get(successor.tracklet_id)),
                )
                assignment_by_tracklet[successor.tracklet_id] = slot.actor_id
                continue

            rows = candidate_rows[successor.tracklet_id]
            best = rows[0] if rows else None
            reason = (
                "no_candidate"
                if not rows
                else "global_assignment_unmatched_or_below_floor"
            )
            unassigned[successor.tracklet_id] = {
                "schema_version": 1,
                "roster_version": FIXED_ACTOR_ROSTER_VERSION,
                "tracklet_id": successor.tracklet_id,
                "start_frame": successor.start_frame,
                "start_timestamp_s": successor.start_timestamp_s,
                "reason": reason,
                "best_candidate_actor_id": best.actor_id if best else None,
                "best_candidate_score": best.score if best else None,
                "best_candidate_reid_cosine": (
                    best.reid_cosine if best else None
                ),
            }

    for tracklet in tracklets:
        if tracklet.tracklet_id in assignment_by_tracklet:
            continue
        if tracklet.tracklet_id in unassigned:
            continue
        unassigned[tracklet.tracklet_id] = {
            "schema_version": 1,
            "roster_version": FIXED_ACTOR_ROSTER_VERSION,
            "tracklet_id": tracklet.tracklet_id,
            "start_frame": tracklet.start_frame,
            "start_timestamp_s": tracklet.start_timestamp_s,
            "reason": "prebootstrap_or_bootstrap_overflow",
            "best_candidate_actor_id": None,
            "best_candidate_score": None,
            "best_candidate_reid_cosine": None,
        }

    actor_rows = []
    assignment_rows = []

    for actor_id in sorted(slots):
        slot = slots[actor_id]
        family = slot.family()
        actor_rows.append({
            "schema_version": 1,
            "roster_version": FIXED_ACTOR_ROSTER_VERSION,
            "actor_id": actor_id,
            "initialized": slot.initialized,
            "tracklet_ids": list(slot.tracklet_ids),
            "family_key": family.key,
            "family_confident": family.confident,
            "family_dominance": family.mass,
            "appearance_available": slot.appearance() is not None,
            "provisional": True,
        })

        for index, tracklet_id in enumerate(slot.tracklet_ids):
            assignment_rows.append({
                "schema_version": 1,
                "roster_version": FIXED_ACTOR_ROSTER_VERSION,
                "tracklet_id": tracklet_id,
                "actor_id": actor_id,
                "actor_sequence_index": index,
                "provisional": True,
            })

    assignment_rows.sort(key=lambda row: row["tracklet_id"])

    return FixedActorRosterResult(
        actors=tuple(actor_rows),
        tracklet_assignments=tuple(assignment_rows),
        accepted_handoffs=tuple(accepted),
        unassigned_tracklets=tuple(unassigned[key] for key in sorted(unassigned)),
        bootstrap_frame=bootstrap_frame,
        bootstrap_timestamp_s=bootstrap_time,
        bootstrap_observation_count=sum(
            any(obs.frame_index == bootstrap_frame for obs in tracklet.observations)
            for tracklet in tracklets
        ),
        bootstrap_seed_count=len(seeds),
        candidate_count=candidate_count,
        exact_visible_exclusion_count=counters["exact_visible"],
        exact_coexistence_rejection_count=counters["exact_coexistence"],
        hard_conflict_rejection_count=counters["hard_conflict"],
        hard_geometry_rejection_count=counters["hard_geometry"],
        hard_family_rejection_count=counters["hard_family"],
        config=selected,
    )
