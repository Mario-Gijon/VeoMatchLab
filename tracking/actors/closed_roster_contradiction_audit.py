"""Tracking V1 Hito 7I.5A: closed-roster contradiction audit.

READ-ONLY audit over `closed-roster-resolution-v1`.

Purpose
-------
The 7I.5 solver produced two deliberately suspicious groups:

- `no_physically_viable_actor`
- `closed_world_best_viable_actor`

This audit does not change a single assignment.  It explains, against the
FINAL 23-slot partition, why every actor was rejected or accepted for each
target tracklet and whether the blocking evidence came from:

- an initial 7I.4 assignment;
- a recovered 7I.5 assignment;
- geometry;
- confident kit-family contradiction;
- sparse hard cannot-link evidence.

For forced closed-world assignments the target tracklet is temporarily removed
from its own actor while auditing it, otherwise it would trivially conflict
with itself.

Outputs are diagnostic only.  Heuristic labels such as
`initial_assignment_pressure_signal` and `strong_non_roster_signal` are audit
signals, NOT automatic reassignment decisions.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


AUDIT_VERSION = "closed-roster-contradiction-audit-v1"

TARGET_RULES = frozenset({
    "no_physically_viable_actor",
    "closed_world_best_viable_actor",
})


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


@dataclass(frozen=True, slots=True)
class ContradictionAuditConfig:
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

    strong_non_roster_floor: float = 0.72
    short_extra_s: float = 2.0
    very_short_extra_s: float = 0.8
    off_field_strong_ratio: float = 0.80
    occupancy_strong_ratio: float = 0.80

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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Obs:
    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _Tracklet:
    tracklet_id: str
    observations: tuple[_Obs, ...]

    @property
    def frames(self) -> set[int]:
        return {obs.frame_index for obs in self.observations}

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


def _tracklet(row: Mapping[str, Any]) -> _Tracklet:
    tid = str(row["tracklet_id"])
    raw = row.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{tid} has no observations")

    result = []
    previous_frame = None
    previous_time = None
    for item in raw:
        obs = _Obs(
            detection_id=str(item["detection_id"]),
            frame_index=int(item["frame_index"]),
            timestamp_s=float(item["timestamp_s"]),
            x=float(item["x"]),
            z=float(item["z"]),
            confidence=float(item.get("confidence", 1.0)),
        )
        if previous_frame is not None and obs.frame_index <= previous_frame:
            raise ValueError(f"{tid} frames are not strictly increasing")
        if previous_time is not None and obs.timestamp_s <= previous_time:
            raise ValueError(f"{tid} timestamps are not strictly increasing")
        previous_frame = obs.frame_index
        previous_time = obs.timestamp_s
        result.append(obs)

    return _Tracklet(tid, tuple(result))


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


@dataclass(slots=True)
class _ActorAuditState:
    actor_id: str
    member_ids: tuple[str, ...]
    member_tracklets: tuple[_Tracklet, ...]
    member_initial: Mapping[str, bool]
    observations: tuple[tuple[_Obs, str], ...]
    observation_frames: tuple[int, ...]
    occupied_frames: frozenset[int]
    family_value: _Family
    appearance_value: np.ndarray | None

    def previous(self, frame_index: int) -> tuple[_Obs, str] | None:
        index = bisect_left(self.observation_frames, frame_index) - 1
        return self.observations[index] if index >= 0 else None

    def after(self, frame_index: int) -> tuple[_Obs, str] | None:
        index = bisect_right(self.observation_frames, frame_index)
        return (
            self.observations[index]
            if index < len(self.observations)
            else None
        )


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
            item.reliability * math.sqrt(item.effective_sample_count),
        )
        weighted = item.embedding * weight
        aggregate = (
            weighted.copy()
            if aggregate is None
            else aggregate + weighted
        )

    return None if aggregate is None else _unit(aggregate)


def _actor_state(
    *,
    actor_id: str,
    all_member_ids: Sequence[str],
    exclude_tracklet_id: str | None,
    tracklets: Mapping[str, _Tracklet],
    initial_by_tracklet: Mapping[str, bool],
    family_by_tracklet: Mapping[str, Mapping[str, Any]],
    appearance_by_tracklet: Mapping[str, Mapping[str, Any]],
) -> _ActorAuditState:
    member_ids = tuple(
        tid
        for tid in all_member_ids
        if tid != exclude_tracklet_id
    )
    members = tuple(tracklets[tid] for tid in member_ids)

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

    # Final partition is expected to be exact-frame clean.
    frames = [item[0].frame_index for item in observations]
    if len(frames) != len(set(frames)):
        raise ValueError(
            f"{actor_id} final partition contains duplicate observation frames"
        )

    return _ActorAuditState(
        actor_id=actor_id,
        member_ids=member_ids,
        member_tracklets=members,
        member_initial={
            tid: bool(initial_by_tracklet.get(tid, False))
            for tid in member_ids
        },
        observations=tuple(observations),
        observation_frames=tuple(frames),
        occupied_frames=frozenset(frames),
        family_value=_aggregate_family(
            member_ids,
            family_by_tracklet,
        ),
        appearance_value=_aggregate_appearance(
            member_ids,
            appearance_by_tracklet,
        ),
    )


def _family_relation(
    actor: _Family,
    target: _Family,
) -> tuple[str, float, bool]:
    if actor.key is None or target.key is None:
        return "neutral", .50, False
    if actor.key == target.key:
        if actor.confident and target.confident:
            return "same_confident", 1.0, False
        return "same_weak", .82, False
    if actor.confident and target.confident:
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


def _transition(
    actor_obs: _Obs,
    target_obs: _Obs,
    *,
    dt_s: float,
    config: ContradictionAuditConfig,
) -> tuple[float, float, float, bool]:
    distance = math.hypot(
        target_obs.x - actor_obs.x,
        target_obs.z - actor_obs.z,
    )
    reachability = (
        config.position_slack_m
        + config.max_transition_speed_mps * max(dt_s, 0.0)
    )
    support = (
        .20
        if dt_s > config.context_window_s
        else _clamp01(
            1.0 - distance / max(reachability, 1e-9)
        )
    )
    hard = (
        dt_s <= config.hard_geometry_window_s
        and distance > reachability + 1e-9
    )
    return distance, reachability, support, hard


def _off_field_ratio(
    target: _Tracklet,
    config: ContradictionAuditConfig,
) -> float:
    half_length = config.field_length_m / 2.0
    half_width = config.field_width_m / 2.0

    outside = sum(
        (
            abs(obs.x) > half_length + config.off_field_margin_m
            or abs(obs.z) > half_width + config.off_field_margin_m
        )
        for obs in target.observations
    )
    return outside / len(target.observations)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0, "min": None, "p50": None, "p90": None,
            "p95": None, "max": None, "mean": None,
        }
    ordered = sorted(float(value) for value in values)

    def pct(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        lo = math.floor(position)
        hi = math.ceil(position)
        if lo == hi:
            return ordered[lo]
        alpha = position - lo
        return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": pct(.50),
        "p90": pct(.90),
        "p95": pct(.95),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _blocker_sources(
    tracklet_ids: Sequence[str],
    initial_by_tracklet: Mapping[str, bool],
) -> dict[str, int]:
    result = {"initial": 0, "recovered": 0}
    for tid in set(tracklet_ids):
        if initial_by_tracklet.get(tid, False):
            result["initial"] += 1
        else:
            result["recovered"] += 1
    return result


def audit_actor_for_tracklet(
    *,
    target: _Tracklet,
    state: _ActorAuditState,
    target_family: _Family,
    target_appearance: _Appearance | None,
    hard_conflicts: set[tuple[str, str]],
    initial_by_tracklet: Mapping[str, bool],
    config: ContradictionAuditConfig,
) -> dict[str, Any]:
    blockers: list[str] = []
    blocking_tracklets: set[str] = set()

    overlap_members = [
        member.tracklet_id
        for member in state.member_tracklets
        if member.frames.intersection(target.frames)
    ]
    if overlap_members:
        blockers.append("exact_frame_coexistence")
        blocking_tracklets.update(overlap_members)

    hard_members = [
        tid
        for tid in state.member_ids
        if tuple(sorted((tid, target.tracklet_id))) in hard_conflicts
    ]
    if hard_members:
        blockers.append("hard_conflict")
        blocking_tracklets.update(hard_members)

    relation, family_support, hard_family = _family_relation(
        state.family_value,
        target_family,
    )
    if hard_family:
        blockers.append("confident_family_mismatch")

    before = state.previous(target.start_frame)
    after = state.after(target.end_frame)

    before_payload = None
    before_support = .20
    if before is not None:
        obs, source_tid = before
        gap = target.start_time - obs.timestamp_s
        distance, limit, before_support, hard = _transition(
            obs,
            target.observations[0],
            dt_s=gap,
            config=config,
        )
        before_payload = {
            "tracklet_id": source_tid,
            "initial_assignment": bool(
                initial_by_tracklet.get(source_tid, False)
            ),
            "frame_index": obs.frame_index,
            "timestamp_s": obs.timestamp_s,
            "gap_s": gap,
            "distance_m": distance,
            "reachability_limit_m": limit,
            "support": before_support,
            "hard_violation": hard,
        }
        if hard:
            blockers.append("before_geometry")
            blocking_tracklets.add(source_tid)

    after_payload = None
    after_support = .20
    if after is not None:
        obs, source_tid = after
        gap = obs.timestamp_s - target.end_time
        distance, limit, after_support, hard = _transition(
            target.observations[-1],
            obs,
            dt_s=gap,
            config=config,
        )
        after_payload = {
            "tracklet_id": source_tid,
            "initial_assignment": bool(
                initial_by_tracklet.get(source_tid, False)
            ),
            "frame_index": obs.frame_index,
            "timestamp_s": obs.timestamp_s,
            "gap_s": gap,
            "distance_m": distance,
            "reachability_limit_m": limit,
            "support": after_support,
            "hard_violation": hard,
        }
        if hard:
            blockers.append("after_geometry")
            blocking_tracklets.add(source_tid)

    available_supports = []
    temporal_terms = []

    if before_payload is not None:
        available_supports.append(before_support)
        gap = before_payload["gap_s"]
        temporal_terms.append(
            math.exp(-gap / 1.5)
            if gap <= config.context_window_s
            else .10
        )

    if after_payload is not None:
        available_supports.append(after_support)
        gap = after_payload["gap_s"]
        temporal_terms.append(
            math.exp(-gap / 1.5)
            if gap <= config.context_window_s
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

    reid_cosine = None
    reid_support = .50
    if (
        state.appearance_value is not None
        and target_appearance is not None
    ):
        reid_cosine = float(np.clip(
            np.dot(
                state.appearance_value,
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

    blocker_sources = _blocker_sources(
        tuple(blocking_tracklets),
        initial_by_tracklet,
    )

    # Strong signal that an otherwise plausible actor is blocked only by
    # occupancy/cannot-link inherited from 7I.4.  This is intentionally a
    # diagnostic signal, not a reassignment.
    only_partition_blockers = bool(blockers) and set(blockers).issubset({
        "exact_frame_coexistence",
        "hard_conflict",
    })
    initial_only_partition_pressure = (
        only_partition_blockers
        and blocker_sources["initial"] > 0
        and blocker_sources["recovered"] == 0
    )

    return {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "tracklet_id": target.tracklet_id,
        "actor_id": state.actor_id,
        "viable": not blockers,
        "blockers": blockers,
        "blocking_tracklet_ids": sorted(blocking_tracklets),
        "blocker_sources": blocker_sources,
        "initial_only_partition_pressure": (
            initial_only_partition_pressure
        ),
        "overlap_tracklet_ids": sorted(overlap_members),
        "hard_conflict_tracklet_ids": sorted(hard_members),
        "family_relation": relation,
        "family_support": family_support,
        "actor_family_key": state.family_value.key,
        "actor_family_confident": state.family_value.confident,
        "target_family_key": target_family.key,
        "target_family_confident": target_family.confident,
        "before": before_payload,
        "after": after_payload,
        "geometry_support": geometry_support,
        "temporal_support": temporal_support,
        "reid_cosine": reid_cosine,
        "reid_support": reid_support,
        "audit_score_if_viable": float(_clamp01(score)),
    }


def build_contradiction_audit(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
    decision_rows: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]] = (),
    appearance_rows: Sequence[Mapping[str, Any]] = (),
    hard_conflicts: Sequence[Mapping[str, Any]] = (),
    config: ContradictionAuditConfig | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    selected = config or ContradictionAuditConfig()

    tracklets = {
        parsed.tracklet_id: parsed
        for parsed in (_tracklet(row) for row in tracklet_rows)
    }
    if len(tracklets) != len(tracklet_rows):
        raise ValueError("duplicate tracklet IDs")

    actor_ids = tuple(sorted(str(row["actor_id"]) for row in actor_rows))
    if len(actor_ids) != selected.expected_actor_slots:
        raise ValueError(
            f"expected {selected.expected_actor_slots} actors, got "
            f"{len(actor_ids)}"
        )
    if len(actor_ids) != len(set(actor_ids)):
        raise ValueError("duplicate actor IDs")

    family_by_tracklet = {
        str(row["tracklet_id"]): row
        for row in family_rows
    }
    appearance_by_tracklet = {
        str(row["tracklet_id"]): row
        for row in appearance_rows
    }
    hard = _hard_pairs(hard_conflicts)

    owner: dict[str, str] = {}
    initial_by_tracklet: dict[str, bool] = {}
    members: dict[str, list[str]] = {
        actor_id: [] for actor_id in actor_ids
    }

    for row in assignment_rows:
        tid = str(row["tracklet_id"])
        actor_id = str(row["actor_id"])
        if tid not in tracklets:
            raise ValueError(f"assignment references unknown {tid}")
        if actor_id not in members:
            raise ValueError(f"assignment references unknown {actor_id}")
        if tid in owner:
            raise ValueError(f"duplicate assignment for {tid}")
        owner[tid] = actor_id
        initial_by_tracklet[tid] = bool(
            row.get("initial_assignment", False)
        )
        members[actor_id].append(tid)

    targets = [
        row
        for row in decision_rows
        if str(row.get("rule")) in TARGET_RULES
    ]

    actor_audits = []
    target_audits = []

    # Final roster occupancy is used only as a scene diagnostic.
    occupied_count: dict[int, int] = defaultdict(int)
    for tid in owner:
        for frame in tracklets[tid].frames:
            occupied_count[frame] += 1

    for decision in sorted(
        targets,
        key=lambda row: str(row["tracklet_id"]),
    ):
        tid = str(decision["tracklet_id"])
        if tid not in tracklets:
            raise ValueError(f"decision references unknown {tid}")

        target = tracklets[tid]
        assigned_actor = owner.get(tid)

        target_family = _family(family_by_tracklet.get(tid))
        target_appearance = _appearance(
            appearance_by_tracklet.get(tid)
        )

        rows = []
        for actor_id in actor_ids:
            # For a forced assignment, audit its owner counterfactually without
            # the target itself, otherwise exact coexistence is guaranteed.
            state = _actor_state(
                actor_id=actor_id,
                all_member_ids=members[actor_id],
                exclude_tracklet_id=(
                    tid if assigned_actor == actor_id else None
                ),
                tracklets=tracklets,
                initial_by_tracklet=initial_by_tracklet,
                family_by_tracklet=family_by_tracklet,
                appearance_by_tracklet=appearance_by_tracklet,
            )
            row = audit_actor_for_tracklet(
                target=target,
                state=state,
                target_family=target_family,
                target_appearance=target_appearance,
                hard_conflicts=hard,
                initial_by_tracklet=initial_by_tracklet,
                config=selected,
            )
            rows.append(row)
            actor_audits.append(row)

        viable = sorted(
            (row for row in rows if row["viable"]),
            key=lambda row: (
                -row["audit_score_if_viable"],
                row["actor_id"],
            ),
        )

        blocker_counter = Counter(
            blocker
            for row in rows
            for blocker in row["blockers"]
        )

        off_field_ratio = _off_field_ratio(target, selected)

        occupancy_values = [
            occupied_count.get(obs.frame_index, 0)
            - (
                1
                if assigned_actor is not None
                and obs.frame_index in tracklets[tid].frames
                else 0
            )
            for obs in target.observations
        ]
        full_23_ratio = sum(
            value >= selected.expected_actor_slots
            for value in occupancy_values
        ) / len(occupancy_values)
        near_full_ratio = sum(
            value >= selected.expected_actor_slots - 1
            for value in occupancy_values
        ) / len(occupancy_values)

        strong_non_roster_signal = (
            (
                off_field_ratio >= selected.off_field_strong_ratio
                and target.duration_s <= selected.short_extra_s
            )
            or (
                near_full_ratio >= selected.occupancy_strong_ratio
                and target.duration_s <= selected.very_short_extra_s
            )
        )

        initial_pressure_actors = [
            row["actor_id"]
            for row in rows
            if row["initial_only_partition_pressure"]
        ]

        chosen_audit = next(
            (
                row
                for row in rows
                if row["actor_id"] == assigned_actor
            ),
            None,
        )

        target_audits.append({
            "schema_version": 1,
            "audit_version": AUDIT_VERSION,
            "tracklet_id": tid,
            "source_rule": str(decision["rule"]),
            "source_status": str(decision["status"]),
            "assigned_actor_id": assigned_actor,
            "duration_s": target.duration_s,
            "start_frame": target.start_frame,
            "end_frame": target.end_frame,
            "start_timestamp_s": target.start_time,
            "end_timestamp_s": target.end_time,
            "off_field_ratio": off_field_ratio,
            "full_roster_occupancy_ratio": full_23_ratio,
            "near_full_roster_occupancy_ratio": near_full_ratio,
            "strong_non_roster_signal": strong_non_roster_signal,
            "viable_actor_count_final_partition": len(viable),
            "viable_actors": [
                {
                    "actor_id": row["actor_id"],
                    "score": row["audit_score_if_viable"],
                    "family_relation": row["family_relation"],
                    "reid_cosine": row["reid_cosine"],
                }
                for row in viable
            ],
            "chosen_actor_audit": chosen_audit,
            "blocker_counts_across_23": dict(
                sorted(blocker_counter.items())
            ),
            "initial_assignment_pressure_signal": bool(
                initial_pressure_actors
            ),
            "initial_pressure_actor_ids": initial_pressure_actors,
            "source_best_actor_score": decision.get(
                "best_actor_score"
            ),
            "source_second_actor_score": decision.get(
                "second_actor_score"
            ),
            "source_actor_margin": decision.get(
                "actor_margin"
            ),
            "source_non_roster_score": decision.get(
                "non_roster_score"
            ),
            "source_needs_review": bool(
                decision.get("needs_review", False)
            ),
        })

    pair_blockers = Counter(
        blocker
        for row in actor_audits
        for blocker in row["blockers"]
    )

    no_viable_targets = [
        row for row in target_audits
        if row["source_rule"] == "no_physically_viable_actor"
    ]
    forced_targets = [
        row for row in target_audits
        if row["source_rule"] == "closed_world_best_viable_actor"
    ]

    summary = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "target_count": len(target_audits),
        "actor_pair_audit_count": len(actor_audits),
        "no_physically_viable_actor_count": len(no_viable_targets),
        "closed_world_best_viable_actor_count": len(forced_targets),
        "pair_blocker_counts": dict(sorted(pair_blockers.items())),
        "no_viable_strong_non_roster_signal_count": sum(
            row["strong_non_roster_signal"]
            for row in no_viable_targets
        ),
        "no_viable_initial_assignment_pressure_signal_count": sum(
            row["initial_assignment_pressure_signal"]
            for row in no_viable_targets
        ),
        "no_viable_has_viable_actor_in_final_partition_count": sum(
            row["viable_actor_count_final_partition"] > 0
            for row in no_viable_targets
        ),
        "forced_chosen_actor_still_viable_count": sum(
            row["chosen_actor_audit"] is not None
            and row["chosen_actor_audit"]["viable"]
            for row in forced_targets
        ),
        "forced_chosen_actor_now_blocked_count": sum(
            row["chosen_actor_audit"] is not None
            and not row["chosen_actor_audit"]["viable"]
            for row in forced_targets
        ),
        "forced_source_score": _distribution([
            float(row["source_best_actor_score"])
            for row in forced_targets
            if row["source_best_actor_score"] is not None
        ]),
        "forced_source_margin": _distribution([
            float(row["source_actor_margin"])
            for row in forced_targets
            if row["source_actor_margin"] is not None
        ]),
        "forced_low_score_below_0_45_count": sum(
            row["source_best_actor_score"] is not None
            and float(row["source_best_actor_score"]) < .45
            for row in forced_targets
        ),
        "forced_low_margin_below_0_035_count": sum(
            row["source_actor_margin"] is not None
            and float(row["source_actor_margin"]) < .035
            for row in forced_targets
        ),
        "config": selected.to_dict(),
    }

    return (
        tuple(target_audits),
        tuple(actor_audits),
        summary,
    )
