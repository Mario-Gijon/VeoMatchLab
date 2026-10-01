"""Tracking V1 Hito 7G.2: geometry-first global trajectory candidates.

7G.1 represents each 7F.B identity as one or more contiguous observed segments.
This stage proposes two kinds of *physically plausible* continuation:

APPEND
    identity A ends -> identity B begins

INSERT
    an entire identity B fits inside one internal observation gap of identity A

This is candidate generation only.  It does not merge identities.

Evidence policy
---------------
* Exact X/Z endpoint geometry provides the hard reachability gate.
* The hard gate intentionally matches the already validated V1 semantics:
  distance <= position_slack + max_speed * dt.
* Endpoint velocity is soft diagnostic evidence only and is never a hard gate.
* Velocity prediction is used only over a short horizon; after that it becomes
  unknown instead of pretending constant velocity remains meaningful.
* Family evidence is attached for later scoring, but is not a hard constraint.
* Missing evidence is unknown, never negative.
* No target player count is assumed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import bisect
import math
from typing import Any, Mapping, Sequence


GLOBAL_TRAJECTORY_CANDIDATE_VERSION = "global-trajectory-candidates-v1"


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryCandidateConfig:
    max_append_gap_s: float = 15.0
    max_transition_speed_mps: float = 12.0
    transition_position_slack_m: float = 1.5
    velocity_use_horizon_s: float = 2.5
    time_epsilon_s: float = 1e-9

    probable_primary_min_dominance: float = 0.60
    probable_primary_min_confidence: float = 0.20
    probable_primary_min_evidence_coverage: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "max_append_gap_s",
            "max_transition_speed_mps",
            "transition_position_slack_m",
            "velocity_use_horizon_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")
        if self.time_epsilon_s < 0.0:
            raise ValueError("time_epsilon_s must be >= 0")
        for name in (
            "probable_primary_min_dominance",
            "probable_primary_min_confidence",
            "probable_primary_min_evidence_coverage",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TransitionGeometry:
    gap_s: float
    endpoint_distance_m: float
    reachability_limit_m: float
    reachability_margin_m: float
    required_speed_after_slack_mps: float | None
    forward_prediction_error_m: float | None
    backward_prediction_error_m: float | None
    velocity_direction_cosine: float | None
    predecessor_velocity_reliability: float | None
    successor_velocity_reliability: float | None
    velocity_prediction_used: bool
    physically_reachable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "gap_s": self.gap_s,
            "endpoint_distance_m": self.endpoint_distance_m,
            "reachability_limit_m": self.reachability_limit_m,
            "reachability_margin_m": self.reachability_margin_m,
            "required_speed_after_slack_mps": (
                self.required_speed_after_slack_mps
            ),
            "forward_prediction_error_m": self.forward_prediction_error_m,
            "backward_prediction_error_m": self.backward_prediction_error_m,
            "velocity_direction_cosine": self.velocity_direction_cosine,
            "predecessor_velocity_reliability": (
                self.predecessor_velocity_reliability
            ),
            "successor_velocity_reliability": (
                self.successor_velocity_reliability
            ),
            "velocity_prediction_used": self.velocity_prediction_used,
            "physically_reachable": self.physically_reachable,
        }


@dataclass(frozen=True, slots=True)
class AppendTrajectoryCandidate:
    predecessor_identity_id: str
    successor_identity_id: str
    geometry: TransitionGeometry
    family_relation: str
    predecessor_family_state: str | None
    successor_family_state: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate_version": GLOBAL_TRAJECTORY_CANDIDATE_VERSION,
            "candidate_type": "append",
            "predecessor_identity_id": self.predecessor_identity_id,
            "successor_identity_id": self.successor_identity_id,
            "geometry": self.geometry.to_dict(),
            "gap_s": self.geometry.gap_s,
            "family_relation": self.family_relation,
            "predecessor_family_state": self.predecessor_family_state,
            "successor_family_state": self.successor_family_state,
            "hard_physical_gate_passed": self.geometry.physically_reachable,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class InsertTrajectoryCandidate:
    host_identity_id: str
    inserted_identity_id: str
    host_gap_index: int
    host_preceding_segment_index: int
    host_following_segment_index: int
    left_geometry: TransitionGeometry
    right_geometry: TransitionGeometry
    family_relation: str
    host_family_state: str | None
    inserted_family_state: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate_version": GLOBAL_TRAJECTORY_CANDIDATE_VERSION,
            "candidate_type": "insert",
            "host_identity_id": self.host_identity_id,
            "inserted_identity_id": self.inserted_identity_id,
            "host_gap_index": self.host_gap_index,
            "host_preceding_segment_index": (
                self.host_preceding_segment_index
            ),
            "host_following_segment_index": (
                self.host_following_segment_index
            ),
            "left_geometry": self.left_geometry.to_dict(),
            "right_geometry": self.right_geometry.to_dict(),
            "left_gap_s": self.left_geometry.gap_s,
            "right_gap_s": self.right_geometry.gap_s,
            "family_relation": self.family_relation,
            "host_family_state": self.host_family_state,
            "inserted_family_state": self.inserted_family_state,
            "hard_physical_gate_passed": (
                self.left_geometry.physically_reachable
                and self.right_geometry.physically_reachable
            ),
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryCandidateResult:
    append_candidates: tuple[AppendTrajectoryCandidate, ...]
    insert_candidates: tuple[InsertTrajectoryCandidate, ...]
    append_pair_evaluated_count: int
    append_physically_rejected_count: int
    insert_pair_evaluated_count: int
    insert_physically_rejected_count: int
    host_gap_with_candidate_count: int
    config: GlobalTrajectoryCandidateConfig

    def summary(self) -> dict[str, Any]:
        append_relations = Counter(
            row.family_relation for row in self.append_candidates
        )
        insert_relations = Counter(
            row.family_relation for row in self.insert_candidates
        )
        return {
            "schema_version": 1,
            "candidate_version": GLOBAL_TRAJECTORY_CANDIDATE_VERSION,
            "append_candidate_count": len(self.append_candidates),
            "insert_candidate_count": len(self.insert_candidates),
            "candidate_count": (
                len(self.append_candidates) + len(self.insert_candidates)
            ),
            "append_pair_evaluated_count": self.append_pair_evaluated_count,
            "append_physically_rejected_count": (
                self.append_physically_rejected_count
            ),
            "insert_pair_evaluated_count": self.insert_pair_evaluated_count,
            "insert_physically_rejected_count": (
                self.insert_physically_rejected_count
            ),
            "host_gap_with_candidate_count": (
                self.host_gap_with_candidate_count
            ),
            "append_family_relation_counts": dict(
                sorted(append_relations.items())
            ),
            "insert_family_relation_counts": dict(
                sorted(insert_relations.items())
            ),
            "config": self.config.to_dict(),
            "velocity_is_hard_constraint": False,
            "family_is_hard_constraint": False,
            "population_forced": False,
            "identity_decisions_changed": False,
        }


def _velocity(endpoint: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = endpoint.get("velocity")
    return raw if isinstance(raw, Mapping) else None


def _transition_geometry(
    predecessor_endpoint: Mapping[str, Any],
    successor_endpoint: Mapping[str, Any],
    *,
    config: GlobalTrajectoryCandidateConfig,
) -> TransitionGeometry:
    t0 = float(predecessor_endpoint["timestamp_s"])
    t1 = float(successor_endpoint["timestamp_s"])
    gap_s = t1 - t0
    if gap_s < -config.time_epsilon_s:
        raise ValueError("transition is temporally reversed")
    gap_s = max(0.0, gap_s)

    x0 = float(predecessor_endpoint["x"])
    z0 = float(predecessor_endpoint["z"])
    x1 = float(successor_endpoint["x"])
    z1 = float(successor_endpoint["z"])

    distance = math.hypot(x1 - x0, z1 - z0)
    limit = (
        config.transition_position_slack_m
        + config.max_transition_speed_mps * gap_s
    )
    margin = limit - distance
    reachable = margin >= -config.time_epsilon_s

    required_speed = None
    if gap_s > config.time_epsilon_s:
        required_speed = max(
            0.0,
            distance - config.transition_position_slack_m,
        ) / gap_s
    elif distance <= config.transition_position_slack_m:
        required_speed = 0.0

    pred_v = _velocity(predecessor_endpoint)
    succ_v = _velocity(successor_endpoint)

    pred_rel = (
        float(pred_v.get("reliability"))
        if pred_v is not None and pred_v.get("reliability") is not None
        else None
    )
    succ_rel = (
        float(succ_v.get("reliability"))
        if succ_v is not None and succ_v.get("reliability") is not None
        else None
    )

    use_velocity = gap_s <= (
        config.velocity_use_horizon_s + config.time_epsilon_s
    )
    forward_error = None
    backward_error = None
    direction_cosine = None

    if use_velocity and pred_v is not None:
        px = x0 + float(pred_v["vx_mps"]) * gap_s
        pz = z0 + float(pred_v["vz_mps"]) * gap_s
        forward_error = math.hypot(x1 - px, z1 - pz)

    if use_velocity and succ_v is not None:
        px = x1 - float(succ_v["vx_mps"]) * gap_s
        pz = z1 - float(succ_v["vz_mps"]) * gap_s
        backward_error = math.hypot(x0 - px, z0 - pz)

    if use_velocity and pred_v is not None and succ_v is not None:
        ax = float(pred_v["vx_mps"])
        az = float(pred_v["vz_mps"])
        bx = float(succ_v["vx_mps"])
        bz = float(succ_v["vz_mps"])
        denom = math.hypot(ax, az) * math.hypot(bx, bz)
        if denom > 1e-12:
            direction_cosine = max(
                -1.0,
                min(1.0, (ax * bx + az * bz) / denom),
            )

    return TransitionGeometry(
        gap_s=float(gap_s),
        endpoint_distance_m=float(distance),
        reachability_limit_m=float(limit),
        reachability_margin_m=float(margin),
        required_speed_after_slack_mps=(
            float(required_speed)
            if required_speed is not None
            else None
        ),
        forward_prediction_error_m=(
            float(forward_error)
            if forward_error is not None
            else None
        ),
        backward_prediction_error_m=(
            float(backward_error)
            if backward_error is not None
            else None
        ),
        velocity_direction_cosine=(
            float(direction_cosine)
            if direction_cosine is not None
            else None
        ),
        predecessor_velocity_reliability=pred_rel,
        successor_velocity_reliability=succ_rel,
        velocity_prediction_used=bool(use_velocity),
        physically_reachable=bool(reachable),
    )


def _primary_evidence(
    row: Mapping[str, Any] | None,
    *,
    config: GlobalTrajectoryCandidateConfig,
) -> tuple[str | None, bool]:
    """Return (primary_key, confident).

    Weak primary evidence may only create a probable contradiction later.  It
    never creates a positive same-primary relation.
    """
    if row is None:
        return None, False

    family_type = row.get("family_type")
    family_key = row.get("family_key")
    if (
        family_type == "primary"
        and isinstance(family_key, str)
        and bool(row.get("usable_for_tracking", False))
    ):
        return family_key, True

    candidate_type = row.get("candidate_family_type")
    candidate_key = row.get("candidate_family_key")
    if candidate_type != "primary" or not isinstance(candidate_key, str):
        return None, False

    try:
        dominance = float(row.get("dominance", 0.0))
        confidence = float(row.get("confidence", 0.0))
        coverage = float(row.get("evidence_coverage", 0.0))
    except (TypeError, ValueError):
        return None, False

    if (
        dominance >= config.probable_primary_min_dominance
        and confidence >= config.probable_primary_min_confidence
        and coverage >= config.probable_primary_min_evidence_coverage
    ):
        return candidate_key, False
    return None, False


def _family_relation(
    left: Mapping[str, Any] | None,
    right: Mapping[str, Any] | None,
    *,
    config: GlobalTrajectoryCandidateConfig,
) -> str:
    left_key, left_confident = _primary_evidence(left, config=config)
    right_key, right_confident = _primary_evidence(right, config=config)

    if left_key is None or right_key is None:
        return "neutral"

    if left_key == right_key:
        if left_confident and right_confident:
            return "same_primary"
        return "neutral"

    if left_confident and right_confident:
        return "different_primary"
    return "probable_different_primary"


def _family_state(row: Mapping[str, Any] | None) -> str | None:
    if row is None:
        return None
    value = row.get("state")
    return str(value) if value is not None else None


def build_global_trajectory_candidates(
    *,
    trajectory_profiles: Sequence[Mapping[str, Any]],
    family_rows: Sequence[Mapping[str, Any]] = (),
    config: GlobalTrajectoryCandidateConfig | None = None,
) -> GlobalTrajectoryCandidateResult:
    selected = config or GlobalTrajectoryCandidateConfig()

    profiles = [dict(row) for row in trajectory_profiles]
    by_id: dict[str, dict[str, Any]] = {}
    for row in profiles:
        identity_id = str(row["identity_id"])
        if identity_id in by_id:
            raise ValueError(f"duplicate identity profile {identity_id}")
        by_id[identity_id] = row

    family_by_id: dict[str, dict[str, Any]] = {}
    for row in family_rows:
        identity_id = str(row.get("identity_id", row.get("entity_id")))
        if identity_id in family_by_id:
            raise ValueError(f"duplicate family row {identity_id}")
        family_by_id[identity_id] = dict(row)

    starts = sorted(
        (
            float(row["start"]["timestamp_s"]),
            str(row["identity_id"]),
        )
        for row in profiles
    )
    start_times = [value[0] for value in starts]

    append_rows: list[AppendTrajectoryCandidate] = []
    append_evaluated = 0
    append_rejected = 0

    for predecessor in profiles:
        predecessor_id = str(predecessor["identity_id"])
        end_t = float(predecessor["end"]["timestamp_s"])

        low = bisect.bisect_left(
            start_times,
            end_t - selected.time_epsilon_s,
        )
        high = bisect.bisect_right(
            start_times,
            end_t
            + selected.max_append_gap_s
            + selected.time_epsilon_s,
        )

        for _, successor_id in starts[low:high]:
            if successor_id == predecessor_id:
                continue
            successor = by_id[successor_id]
            start_t = float(successor["start"]["timestamp_s"])
            if start_t < end_t - selected.time_epsilon_s:
                continue

            append_evaluated += 1
            geometry = _transition_geometry(
                predecessor["end"],
                successor["start"],
                config=selected,
            )
            if not geometry.physically_reachable:
                append_rejected += 1
                continue

            relation = _family_relation(
                family_by_id.get(predecessor_id),
                family_by_id.get(successor_id),
                config=selected,
            )
            append_rows.append(
                AppendTrajectoryCandidate(
                    predecessor_identity_id=predecessor_id,
                    successor_identity_id=successor_id,
                    geometry=geometry,
                    family_relation=relation,
                    predecessor_family_state=_family_state(
                        family_by_id.get(predecessor_id)
                    ),
                    successor_family_state=_family_state(
                        family_by_id.get(successor_id)
                    ),
                )
            )

    # INSERT candidates: candidate identity must fit completely inside one
    # explicit host gap.  We use the same start-time index, then verify its end.
    insert_rows: list[InsertTrajectoryCandidate] = []
    insert_evaluated = 0
    insert_rejected = 0
    host_gaps_with_candidate: set[tuple[str, int]] = set()

    for host in profiles:
        host_id = str(host["identity_id"])
        segments = {
            int(row["segment_index"]): row
            for row in host.get("segments", ())
        }

        for gap in host.get("internal_gaps", ()):
            gap_index = int(gap["gap_index"])
            gap_start = float(gap["start_time_s"])
            gap_end = float(gap["end_time_s"])
            preceding_index = int(gap["preceding_segment_index"])
            following_index = int(gap["following_segment_index"])

            try:
                left_segment = segments[preceding_index]
                right_segment = segments[following_index]
            except KeyError as exc:
                raise ValueError(
                    f"{host_id} gap {gap_index} references missing segment"
                ) from exc

            low = bisect.bisect_left(
                start_times,
                gap_start - selected.time_epsilon_s,
            )
            high = bisect.bisect_right(
                start_times,
                gap_end + selected.time_epsilon_s,
            )

            for _, inserted_id in starts[low:high]:
                if inserted_id == host_id:
                    continue
                inserted = by_id[inserted_id]
                inserted_start = float(inserted["start"]["timestamp_s"])
                inserted_end = float(inserted["end"]["timestamp_s"])

                if inserted_start < gap_start - selected.time_epsilon_s:
                    continue
                if inserted_end > gap_end + selected.time_epsilon_s:
                    continue

                insert_evaluated += 1
                left_geometry = _transition_geometry(
                    left_segment["end"],
                    inserted["start"],
                    config=selected,
                )
                right_geometry = _transition_geometry(
                    inserted["end"],
                    right_segment["start"],
                    config=selected,
                )

                if (
                    not left_geometry.physically_reachable
                    or not right_geometry.physically_reachable
                ):
                    insert_rejected += 1
                    continue

                relation = _family_relation(
                    family_by_id.get(host_id),
                    family_by_id.get(inserted_id),
                    config=selected,
                )
                insert_rows.append(
                    InsertTrajectoryCandidate(
                        host_identity_id=host_id,
                        inserted_identity_id=inserted_id,
                        host_gap_index=gap_index,
                        host_preceding_segment_index=preceding_index,
                        host_following_segment_index=following_index,
                        left_geometry=left_geometry,
                        right_geometry=right_geometry,
                        family_relation=relation,
                        host_family_state=_family_state(
                            family_by_id.get(host_id)
                        ),
                        inserted_family_state=_family_state(
                            family_by_id.get(inserted_id)
                        ),
                    )
                )
                host_gaps_with_candidate.add((host_id, gap_index))

    append_rows.sort(
        key=lambda row: (
            row.predecessor_identity_id,
            row.geometry.gap_s,
            row.successor_identity_id,
        )
    )
    insert_rows.sort(
        key=lambda row: (
            row.host_identity_id,
            row.host_gap_index,
            row.left_geometry.gap_s + row.right_geometry.gap_s,
            row.inserted_identity_id,
        )
    )

    return GlobalTrajectoryCandidateResult(
        append_candidates=tuple(append_rows),
        insert_candidates=tuple(insert_rows),
        append_pair_evaluated_count=append_evaluated,
        append_physically_rejected_count=append_rejected,
        insert_pair_evaluated_count=insert_evaluated,
        insert_physically_rejected_count=insert_rejected,
        host_gap_with_candidate_count=len(host_gaps_with_candidate),
        config=selected,
    )
