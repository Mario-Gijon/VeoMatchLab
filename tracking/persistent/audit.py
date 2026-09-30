"""Hito 6D: audit persistent-actor population and trajectory invariants.

This module is intentionally read-only.  It consumes the provisional Hito 6B
actor assignment plus Hito 6C dense trajectories and checks whether the result
is safe to send to visual validation (Hito 6E).  It never mutates Tracking V1,
actor assignments, or trajectory points.

The audit separates *blocking* invariants from *review* diagnostics:

Blocking examples
-----------------
* population/actor-count mismatch;
* original tracklet-level ``cannot_link`` violation;
* duplicate actor/frame output;
* malformed/unknown trajectory origin;
* prediction beyond the reconstruction policy;
* internal disappearance after Hito 6C;
* occupancy rows inconsistent with the trajectory payload.

Review examples
---------------
* a small number of edge frames without an actor position;
* high-but-bounded trajectory uncertainty;
* unusually high smoothed speed/acceleration;
* very close actors that deserve visual inspection.

A ``PASS_WITH_REVIEW`` result therefore means that all hard invariants hold,
but Hito 6E should inspect the listed review cases before publication.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


AUDIT_VERSION = "persistent-population-audit-v1"
TRAJECTORY_VERSION = "persistent-trajectory-v1"
ASSIGNMENT_SOLVER_VERSION = "anchored-actor-assignment-v1"
VALID_ORIGINS = {"observed", "fused", "interpolated", "predicted"}
DIRECT_ORIGINS = {"observed", "fused"}


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return ordered[low]
        weight = position - low
        return ordered[low] * (1.0 - weight) + ordered[high] * weight

    return {
        "count": len(ordered),
        "min": ordered[0],
        "median": percentile(0.50),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


@dataclass(frozen=True, slots=True)
class PersistentPopulationAuditConfig:
    """Policy for 6D.

    Coverage thresholds are intentionally gates for the *current short-window
    persistent representation*, not universal football constants.  The
    expected actor count itself comes from Hito 6B's population estimate.
    """

    min_represented_coverage_fraction: float = 0.98
    min_full_population_frame_fraction: float = 0.95
    prediction_age_tolerance_s: float = 0.01
    position_std_review_m: float = 1.00
    speed_review_mps: float = 12.0
    acceleration_review_mps2: float = 20.0
    close_actor_review_m: float = 0.35
    field_abs_x_review_m: float = 60.0
    field_abs_z_review_m: float = 40.0
    max_issue_examples: int = 50

    def __post_init__(self) -> None:
        for name in (
            "min_represented_coverage_fraction",
            "min_full_population_frame_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        for name in (
            "prediction_age_tolerance_s",
            "position_std_review_m",
            "speed_review_mps",
            "acceleration_review_mps2",
            "close_actor_review_m",
            "field_abs_x_review_m",
            "field_abs_z_review_m",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be >= 0")
        if self.max_issue_examples <= 0:
            raise ValueError("max_issue_examples must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_represented_coverage_fraction": self.min_represented_coverage_fraction,
            "min_full_population_frame_fraction": self.min_full_population_frame_fraction,
            "prediction_age_tolerance_s": self.prediction_age_tolerance_s,
            "position_std_review_m": self.position_std_review_m,
            "speed_review_mps": self.speed_review_mps,
            "acceleration_review_mps2": self.acceleration_review_mps2,
            "close_actor_review_m": self.close_actor_review_m,
            "field_abs_x_review_m": self.field_abs_x_review_m,
            "field_abs_z_review_m": self.field_abs_z_review_m,
            "max_issue_examples": self.max_issue_examples,
        }


def _issue(
    *,
    severity: str,
    code: str,
    message: str,
    **details: Any,
) -> dict[str, Any]:
    if severity not in {"blocker", "review"}:
        raise ValueError("invalid issue severity")
    return {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "severity": severity,
        "code": code,
        "message": message,
        "details": details,
    }


def _assignment_map(
    assignment_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, str | None], list[dict[str, Any]]]:
    result: dict[str, str | None] = {}
    issues: list[dict[str, Any]] = []
    for row in assignment_rows:
        tracklet_id = str(row["tracklet_id"])
        actor = row.get("actor_id")
        actor_id = None if actor is None else str(actor)
        if tracklet_id in result:
            issues.append(
                _issue(
                    severity="blocker",
                    code="duplicate_tracklet_assignment",
                    message=f"tracklet {tracklet_id} appears more than once in 6B assignments",
                    tracklet_id=tracklet_id,
                )
            )
        result[tracklet_id] = actor_id
    return result, issues


def _hard_physics_violations(
    hard_pair_rows: Sequence[Mapping[str, Any]],
    assignment: Mapping[str, str | None],
) -> tuple[int, list[dict[str, Any]], int]:
    violations: list[dict[str, Any]] = []
    hard_count = 0
    for row in hard_pair_rows:
        if str(row.get("status")) != "cannot_link":
            continue
        hard_count += 1
        a = str(row["tracklet_a"])
        b = str(row["tracklet_b"])
        actor_a = assignment.get(a)
        actor_b = assignment.get(b)
        if actor_a is not None and actor_a == actor_b:
            violations.append(
                {
                    "tracklet_a": a,
                    "tracklet_b": b,
                    "actor_id": actor_a,
                }
            )
    return hard_count, violations, len(violations)


def _trajectory_index(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, Mapping[str, Any]],
    dict[int, dict[str, Mapping[str, Any]]],
    list[dict[str, Any]],
]:
    by_actor: dict[str, Mapping[str, Any]] = {}
    by_frame: dict[int, dict[str, Mapping[str, Any]]] = collections.defaultdict(dict)
    issues: list[dict[str, Any]] = []

    for trajectory in trajectory_rows:
        actor_id = str(trajectory["actor_id"])
        if actor_id in by_actor:
            issues.append(
                _issue(
                    severity="blocker",
                    code="duplicate_actor_trajectory",
                    message=f"actor {actor_id} has multiple trajectory rows",
                    actor_id=actor_id,
                )
            )
            continue
        by_actor[actor_id] = trajectory

        previous_frame: int | None = None
        seen_frames: set[int] = set()
        for point in trajectory.get("points", []):
            frame = int(point["frame_index"])
            origin = str(point.get("origin"))
            if origin not in VALID_ORIGINS:
                issues.append(
                    _issue(
                        severity="blocker",
                        code="invalid_point_origin",
                        message=f"{actor_id} frame {frame} has invalid origin {origin!r}",
                        actor_id=actor_id,
                        frame_index=frame,
                        origin=origin,
                    )
                )
            if frame in seen_frames:
                issues.append(
                    _issue(
                        severity="blocker",
                        code="duplicate_actor_frame",
                        message=f"{actor_id} contains duplicate frame {frame}",
                        actor_id=actor_id,
                        frame_index=frame,
                    )
                )
            seen_frames.add(frame)
            if previous_frame is not None and frame <= previous_frame:
                issues.append(
                    _issue(
                        severity="blocker",
                        code="non_monotonic_actor_frames",
                        message=f"{actor_id} trajectory frames are not strictly increasing",
                        actor_id=actor_id,
                        previous_frame_index=previous_frame,
                        frame_index=frame,
                    )
                )
            previous_frame = frame
            if actor_id in by_frame[frame]:
                issues.append(
                    _issue(
                        severity="blocker",
                        code="duplicate_actor_frame_global",
                        message=f"duplicate actor/frame output for {actor_id} frame {frame}",
                        actor_id=actor_id,
                        frame_index=frame,
                    )
                )
            by_frame[frame][actor_id] = point
    return by_actor, by_frame, issues


def _infer_cadence(
    occupancy_rows: Sequence[Mapping[str, Any]],
    trajectory_summary: Mapping[str, Any],
) -> tuple[tuple[int, ...], int, float]:
    cadence = trajectory_summary.get("cadence") or {}
    frames = tuple(sorted(int(row["frame_index"]) for row in occupancy_rows))
    if not frames:
        raise ValueError("occupancy rows are empty")
    frame_step = int(cadence.get("frame_step", 0) or 0)
    time_step_s = float(cadence.get("time_step_s", 0.0) or 0.0)
    if frame_step <= 0 and len(frames) > 1:
        frame_step = min(b - a for a, b in zip(frames, frames[1:]) if b > a)
    if frame_step <= 0:
        frame_step = 1
    if time_step_s <= 0.0 and len(occupancy_rows) > 1:
        ordered = sorted(occupancy_rows, key=lambda row: int(row["frame_index"]))
        dt = float(ordered[1]["timestamp_s"]) - float(ordered[0]["timestamp_s"])
        time_step_s = max(dt, 1e-9)
    if time_step_s <= 0.0:
        time_step_s = 1.0
    return frames, frame_step, time_step_s


def audit_persistent_population(
    *,
    trajectory_rows: Sequence[Mapping[str, Any]],
    occupancy_rows: Sequence[Mapping[str, Any]],
    gap_rows: Sequence[Mapping[str, Any]],
    trajectory_summary: Mapping[str, Any],
    assignment_rows: Sequence[Mapping[str, Any]],
    assignment_summary: Mapping[str, Any],
    hard_pair_rows: Sequence[Mapping[str, Any]],
    config: PersistentPopulationAuditConfig | None = None,
) -> dict[str, Any]:
    """Audit a 6B+6C result without mutating any source artifact."""

    config = config or PersistentPopulationAuditConfig()
    issues: list[dict[str, Any]] = []

    if trajectory_summary.get("trajectory_version") != TRAJECTORY_VERSION:
        issues.append(
            _issue(
                severity="blocker",
                code="trajectory_version_mismatch",
                message="6D requires persistent-trajectory-v1",
                value=trajectory_summary.get("trajectory_version"),
            )
        )
    if assignment_summary.get("solver_version") != ASSIGNMENT_SOLVER_VERSION:
        issues.append(
            _issue(
                severity="blocker",
                code="assignment_version_mismatch",
                message="6D requires anchored-actor-assignment-v1",
                value=assignment_summary.get("solver_version"),
            )
        )

    expected_actor_count = int(assignment_summary.get("population_estimate", 0) or 0)
    trajectory_actor_count = len({str(row["actor_id"]) for row in trajectory_rows})
    assignment_actor_count = int(assignment_summary.get("actor_count", 0) or 0)
    if expected_actor_count <= 0:
        issues.append(
            _issue(
                severity="blocker",
                code="invalid_population_estimate",
                message="6B population estimate is missing or non-positive",
                value=assignment_summary.get("population_estimate"),
            )
        )
    if not (
        expected_actor_count == trajectory_actor_count == assignment_actor_count
    ):
        issues.append(
            _issue(
                severity="blocker",
                code="actor_count_mismatch",
                message="population estimate, 6B actor count and 6C trajectory count disagree",
                population_estimate=expected_actor_count,
                assignment_actor_count=assignment_actor_count,
                trajectory_actor_count=trajectory_actor_count,
            )
        )

    assignment, assignment_issues = _assignment_map(assignment_rows)
    issues.extend(assignment_issues)
    assigned_actor_ids = {value for value in assignment.values() if value is not None}

    by_actor, by_frame, trajectory_issues = _trajectory_index(trajectory_rows)
    issues.extend(trajectory_issues)
    trajectory_actor_ids = set(by_actor)
    if assigned_actor_ids != trajectory_actor_ids:
        issues.append(
            _issue(
                severity="blocker",
                code="assignment_trajectory_actor_set_mismatch",
                message="6B assigned actor IDs and 6C trajectory actor IDs differ",
                only_in_assignment=sorted(assigned_actor_ids - trajectory_actor_ids),
                only_in_trajectories=sorted(trajectory_actor_ids - assigned_actor_ids),
            )
        )

    hard_count, hard_violations, hard_violation_count = _hard_physics_violations(
        hard_pair_rows, assignment
    )
    if hard_violation_count:
        issues.append(
            _issue(
                severity="blocker",
                code="hard_physics_violation",
                message=f"{hard_violation_count} original tracklet cannot_link constraints are violated",
                count=hard_violation_count,
                examples=hard_violations[: config.max_issue_examples],
            )
        )
    reported_hard_count = int(assignment_summary.get("hard_cannot_link_count", -1))
    if reported_hard_count >= 0 and hard_count != reported_hard_count:
        issues.append(
            _issue(
                severity="blocker",
                code="hard_pair_count_mismatch",
                message="6D hard-pair input does not match the 6B provenance count",
                audited=hard_count,
                assignment_summary=reported_hard_count,
            )
        )

    frames, frame_step, time_step_s = _infer_cadence(occupancy_rows, trajectory_summary)
    frame_set = set(frames)
    occupancy_by_frame = {int(row["frame_index"]): row for row in occupancy_rows}
    if len(occupancy_by_frame) != len(occupancy_rows):
        issues.append(
            _issue(
                severity="blocker",
                code="duplicate_occupancy_frame",
                message="occupancy artifact contains duplicate frame rows",
            )
        )

    missing_rows: list[dict[str, Any]] = []
    full_population_frames = 0
    represented_actor_frames = 0
    direct_actor_frames = 0
    minimum_population = trajectory_actor_count

    for frame in frames:
        row = occupancy_by_frame[frame]
        actual = by_frame.get(frame, {})
        actual_actor_ids = set(actual)
        reported_actor_ids = {str(value) for value in row.get("actor_ids", [])}
        reported_missing = {str(value) for value in row.get("missing_actor_ids", [])}
        expected_missing = trajectory_actor_ids - actual_actor_ids
        if reported_actor_ids != actual_actor_ids or reported_missing != expected_missing:
            issues.append(
                _issue(
                    severity="blocker",
                    code="occupancy_trajectory_mismatch",
                    message=f"occupancy row at frame {frame} disagrees with trajectory points",
                    frame_index=frame,
                    reported_actor_ids=sorted(reported_actor_ids),
                    actual_actor_ids=sorted(actual_actor_ids),
                    reported_missing_actor_ids=sorted(reported_missing),
                    actual_missing_actor_ids=sorted(expected_missing),
                )
            )
        represented = len(actual_actor_ids)
        direct = sum(
            str(point.get("origin")) in DIRECT_ORIGINS for point in actual.values()
        )
        represented_actor_frames += represented
        direct_actor_frames += direct
        minimum_population = min(minimum_population, represented)
        if represented == trajectory_actor_count:
            full_population_frames += 1
        for actor_id in sorted(expected_missing):
            missing_rows.append(
                {
                    "schema_version": 1,
                    "audit_version": AUDIT_VERSION,
                    "actor_id": actor_id,
                    "frame_index": frame,
                    "timestamp_s": float(row["timestamp_s"]),
                }
            )

    total_possible = trajectory_actor_count * len(frames)
    represented_coverage = _safe_fraction(represented_actor_frames, total_possible)
    direct_coverage = _safe_fraction(direct_actor_frames, total_possible)
    full_population_fraction = _safe_fraction(full_population_frames, len(frames))
    if represented_coverage < config.min_represented_coverage_fraction:
        issues.append(
            _issue(
                severity="blocker",
                code="represented_coverage_below_gate",
                message="represented actor-frame coverage is below the 6D gate",
                value=represented_coverage,
                minimum=config.min_represented_coverage_fraction,
            )
        )
    if full_population_fraction < config.min_full_population_frame_fraction:
        issues.append(
            _issue(
                severity="blocker",
                code="full_population_coverage_below_gate",
                message="too few frames represent the complete persistent population",
                value=full_population_fraction,
                minimum=config.min_full_population_frame_fraction,
            )
        )

    missing_by_actor: dict[str, list[int]] = collections.defaultdict(list)
    for row in missing_rows:
        missing_by_actor[str(row["actor_id"])].append(int(row["frame_index"]))

    internal_missing: list[dict[str, Any]] = []
    actor_reviews: list[dict[str, Any]] = []
    for actor_id in sorted(trajectory_actor_ids):
        actor_frames = sorted(
            int(point["frame_index"])
            for point in by_actor[actor_id].get("points", [])
        )
        if not actor_frames:
            issues.append(
                _issue(
                    severity="blocker",
                    code="actor_without_points",
                    message=f"{actor_id} has no trajectory points",
                    actor_id=actor_id,
                )
            )
            continue
        missing = sorted(missing_by_actor.get(actor_id, []))
        actor_internal = [
            frame for frame in missing if actor_frames[0] < frame < actor_frames[-1]
        ]
        if actor_internal:
            internal_missing.extend(
                {"actor_id": actor_id, "frame_index": frame}
                for frame in actor_internal
            )
            issues.append(
                _issue(
                    severity="blocker",
                    code="internal_disappearance",
                    message=f"{actor_id} still has missing frames inside its represented interval",
                    actor_id=actor_id,
                    frame_indices=actor_internal[: config.max_issue_examples],
                    count=len(actor_internal),
                )
            )
        if missing:
            actor_reviews.append(
                {
                    "actor_id": actor_id,
                    "missing_frame_count": len(missing),
                    "missing_frame_indices": missing,
                    "internal_missing_frame_count": len(actor_internal),
                    "represented_frame_fraction": _safe_fraction(
                        len(actor_frames), len(frames)
                    ),
                }
            )
            if not actor_internal:
                issues.append(
                    _issue(
                        severity="review",
                        code="edge_frames_missing",
                        message=f"{actor_id} has unresolved edge frames but no internal disappearance",
                        actor_id=actor_id,
                        frame_indices=missing[: config.max_issue_examples],
                        count=len(missing),
                    )
                )

    configured_prediction_limit = float(
        (trajectory_summary.get("config") or {}).get("max_edge_prediction_s", 0.0)
        or 0.0
    )
    allowed_prediction_age = configured_prediction_limit + config.prediction_age_tolerance_s
    prediction_ages: list[float] = []
    position_stds: list[float] = []
    speeds: list[float] = []
    prediction_age_violations: list[dict[str, Any]] = []
    uncertainty_reviews: list[dict[str, Any]] = []
    speed_reviews: list[dict[str, Any]] = []
    field_reviews: list[dict[str, Any]] = []
    acceleration_reviews: list[dict[str, Any]] = []

    for actor_id, trajectory in by_actor.items():
        points = list(trajectory.get("points", []))
        previous: Mapping[str, Any] | None = None
        for point in points:
            frame = int(point["frame_index"])
            origin = str(point.get("origin"))
            x = _finite_float("x", point["x"])
            z = _finite_float("z", point["z"])
            std = _finite_float("position_std_m", point.get("position_std_m", 0.0))
            speed = _finite_float("speed_mps", point.get("speed_mps", 0.0))
            position_stds.append(std)
            speeds.append(speed)
            if std > config.position_std_review_m:
                uncertainty_reviews.append(
                    {
                        "actor_id": actor_id,
                        "frame_index": frame,
                        "origin": origin,
                        "position_std_m": std,
                    }
                )
            if speed > config.speed_review_mps:
                speed_reviews.append(
                    {
                        "actor_id": actor_id,
                        "frame_index": frame,
                        "origin": origin,
                        "speed_mps": speed,
                    }
                )
            if abs(x) > config.field_abs_x_review_m or abs(z) > config.field_abs_z_review_m:
                field_reviews.append(
                    {
                        "actor_id": actor_id,
                        "frame_index": frame,
                        "x": x,
                        "z": z,
                        "origin": origin,
                    }
                )
            if origin == "predicted":
                age = _finite_float("prediction_age_s", point.get("prediction_age_s", 0.0))
                prediction_ages.append(age)
                if age > allowed_prediction_age + 1e-9:
                    prediction_age_violations.append(
                        {
                            "actor_id": actor_id,
                            "frame_index": frame,
                            "prediction_age_s": age,
                            "allowed_s": allowed_prediction_age,
                        }
                    )
            if previous is not None:
                dt = float(point["timestamp_s"]) - float(previous["timestamp_s"])
                if dt > 0.0:
                    dvx = float(point.get("vx_mps", 0.0)) - float(previous.get("vx_mps", 0.0))
                    dvz = float(point.get("vz_mps", 0.0)) - float(previous.get("vz_mps", 0.0))
                    acceleration = math.hypot(dvx, dvz) / dt
                    if acceleration > config.acceleration_review_mps2:
                        acceleration_reviews.append(
                            {
                                "actor_id": actor_id,
                                "frame_index": frame,
                                "previous_frame_index": int(previous["frame_index"]),
                                "acceleration_mps2": acceleration,
                            }
                        )
            previous = point

    if prediction_age_violations:
        issues.append(
            _issue(
                severity="blocker",
                code="prediction_age_exceeded",
                message="6C emitted predictions beyond its configured edge horizon",
                count=len(prediction_age_violations),
                examples=prediction_age_violations[: config.max_issue_examples],
            )
        )
    if uncertainty_reviews:
        issues.append(
            _issue(
                severity="review",
                code="high_position_uncertainty",
                message="some persistent points exceed the visual-review uncertainty threshold",
                count=len(uncertainty_reviews),
                threshold_m=config.position_std_review_m,
                examples=uncertainty_reviews[: config.max_issue_examples],
            )
        )
    if speed_reviews:
        issues.append(
            _issue(
                severity="review",
                code="high_smoothed_speed",
                message="some smoothed persistent states exceed the speed review threshold",
                count=len(speed_reviews),
                threshold_mps=config.speed_review_mps,
                examples=speed_reviews[: config.max_issue_examples],
            )
        )
    if acceleration_reviews:
        issues.append(
            _issue(
                severity="review",
                code="high_smoothed_acceleration",
                message="some smoothed persistent states exceed the acceleration review threshold",
                count=len(acceleration_reviews),
                threshold_mps2=config.acceleration_review_mps2,
                examples=acceleration_reviews[: config.max_issue_examples],
            )
        )
    if field_reviews:
        issues.append(
            _issue(
                severity="review",
                code="coordinate_sanity_review",
                message="some actor points fall outside the broad pitch-coordinate sanity envelope",
                count=len(field_reviews),
                abs_x_review_m=config.field_abs_x_review_m,
                abs_z_review_m=config.field_abs_z_review_m,
                examples=field_reviews[: config.max_issue_examples],
            )
        )

    collision_rows: list[dict[str, Any]] = []
    for frame in frames:
        points = by_frame.get(frame, {})
        actor_ids = sorted(points)
        for i, actor_a in enumerate(actor_ids):
            pa = points[actor_a]
            ax = float(pa["x"])
            az = float(pa["z"])
            for actor_b in actor_ids[i + 1 :]:
                pb = points[actor_b]
                distance = math.hypot(ax - float(pb["x"]), az - float(pb["z"]))
                if distance < config.close_actor_review_m:
                    collision_rows.append(
                        {
                            "schema_version": 1,
                            "audit_version": AUDIT_VERSION,
                            "frame_index": frame,
                            "timestamp_s": float(pa["timestamp_s"]),
                            "actor_a": actor_a,
                            "actor_b": actor_b,
                            "distance_m": distance,
                            "origin_a": str(pa["origin"]),
                            "origin_b": str(pb["origin"]),
                        }
                    )
    if collision_rows:
        issues.append(
            _issue(
                severity="review",
                code="close_actor_pairs",
                message="distinct persistent actors occupy unusually close field positions",
                count=len(collision_rows),
                threshold_m=config.close_actor_review_m,
                examples=collision_rows[: config.max_issue_examples],
            )
        )

    # Cross-check 6C's own high-level summary; recomputation remains the source
    # of truth for this audit.
    source_summary_checks = {
        "actor_count": trajectory_actor_count,
        "frame_count": len(frames),
        "represented_actor_frame_count": represented_actor_frames,
        "full_population_frame_count_after": full_population_frames,
    }
    for key, audited_value in source_summary_checks.items():
        if key in trajectory_summary and int(trajectory_summary[key]) != int(audited_value):
            issues.append(
                _issue(
                    severity="blocker",
                    code="trajectory_summary_mismatch",
                    message=f"6C summary field {key} does not match recomputed 6D value",
                    field=key,
                    source_value=trajectory_summary[key],
                    audited_value=audited_value,
                )
            )

    preserved_gaps = [
        dict(row)
        for row in gap_rows
        if str(row.get("action")) in {"preserved", "partially_predicted"}
    ]

    blocker_count = sum(row["severity"] == "blocker" for row in issues)
    review_count = sum(row["severity"] == "review" for row in issues)
    status = "FAIL" if blocker_count else "PASS_WITH_REVIEW" if review_count else "PASS"

    summary = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "trajectory_version": TRAJECTORY_VERSION,
        "assignment_solver_version": ASSIGNMENT_SOLVER_VERSION,
        "canonical": False,
        "provisional": True,
        "tracking_v1_unchanged": True,
        "status": status,
        "blocking_issue_count": blocker_count,
        "review_issue_count": review_count,
        "eligible_for_visual_validation": blocker_count == 0,
        "actor_count": trajectory_actor_count,
        "expected_actor_count": expected_actor_count,
        "frame_count": len(frames),
        "cadence": {
            "frame_step": frame_step,
            "time_step_s": time_step_s,
            "first_frame_index": frames[0],
            "last_frame_index": frames[-1],
        },
        "hard_cannot_link_count": hard_count,
        "hard_constraint_violation_count": hard_violation_count,
        "total_possible_actor_frames": total_possible,
        "direct_actor_frame_count": direct_actor_frames,
        "represented_actor_frame_count": represented_actor_frames,
        "direct_coverage_fraction": direct_coverage,
        "represented_coverage_fraction": represented_coverage,
        "full_population_frame_count": full_population_frames,
        "full_population_frame_fraction": full_population_fraction,
        "minimum_represented_actor_count": minimum_population,
        "missing_actor_frame_count": len(missing_rows),
        "actors_with_missing_frames": sorted(missing_by_actor),
        "internal_missing_actor_frame_count": len(internal_missing),
        "disappearance_event_count": sum(
            issue["code"] == "internal_disappearance" for issue in issues
        ),
        "predicted_point_count": len(prediction_ages),
        "prediction_age_s": _distribution(prediction_ages),
        "position_std_m": _distribution(position_stds),
        "speed_mps": _distribution(speeds),
        "prediction_age_violation_count": len(prediction_age_violations),
        "high_uncertainty_point_count": len(uncertainty_reviews),
        "high_speed_point_count": len(speed_reviews),
        "high_acceleration_transition_count": len(acceleration_reviews),
        "coordinate_sanity_review_count": len(field_reviews),
        "close_actor_pair_count": len(collision_rows),
        "preserved_gap_count": len(preserved_gaps),
        "config": config.to_dict(),
    }

    return {
        "summary": summary,
        "issues": issues,
        "missing_actor_frames": missing_rows,
        "actor_reviews": actor_reviews,
        "collision_reviews": collision_rows,
        "preserved_gaps": preserved_gaps,
    }
