"""Final trajectory diagnostics for Tracking V1.

Hito 4E is audit-only. It NEVER changes the smoothed trajectories.

Findings are separated into:
- blockers: structural inconsistencies that make publication unsafe;
- review flags: unusual/weakly-supported physical cases that deserve explicit
  inspection but do not silently alter the trajectory.

The 4F publication gate is `blocker_count == 0`. Review flags remain in the
provenance so unusual cases are never hidden.
"""

from __future__ import annotations

import collections
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


FINAL_DIAGNOSTICS_VERSION = "final-trajectory-audit-v1"
SMOOTHING_VERSION = "rts-cv-v1"


@dataclass(frozen=True, slots=True)
class FinalDiagnosticsConfig:
    review_speed_mps: float = 12.0
    review_acceleration_mps2: float = 12.0
    review_position_adjustment_m: float = 1.0
    review_position_std_m: float = 0.50
    review_velocity_std_mps: float = 2.50
    cadence_tolerance_s: float = 0.02
    numeric_tolerance: float = 1e-6
    top_n_review_items: int = 25

    def __post_init__(self) -> None:
        for name in (
            "review_speed_mps",
            "review_acceleration_mps2",
            "review_position_adjustment_m",
            "review_position_std_m",
            "review_velocity_std_mps",
            "numeric_tolerance",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.cadence_tolerance_s < 0:
            raise ValueError("cadence_tolerance_s must be >= 0")
        if self.top_n_review_items < 1:
            raise ValueError("top_n_review_items must be >= 1")


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source}:{line_number} is not a JSON object"
                )
            yield payload


def load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} is not a JSON object")
    return payload


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)

    def percentile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * q
        low = math.floor(position)
        high = math.ceil(position)
        if low == high:
            return ordered[low]
        alpha = position - low
        return ordered[low] * (1.0 - alpha) + ordered[high] * alpha

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


def _infer_cadence(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> tuple[int, float]:
    frame_deltas: list[int] = []
    time_deltas_by_frame: dict[int, list[float]] = collections.defaultdict(list)

    for trajectory in trajectory_rows:
        points = trajectory.get("points") or []
        for left, right in zip(points, points[1:]):
            if left.get("segment_id") != right.get("segment_id"):
                continue
            frame_delta = int(right["frame_index"]) - int(left["frame_index"])
            time_delta = float(right["timestamp_s"]) - float(left["timestamp_s"])
            if frame_delta <= 0 or time_delta <= 0:
                continue
            frame_deltas.append(frame_delta)
            time_deltas_by_frame[frame_delta].append(time_delta)

    if not frame_deltas:
        raise ValueError("cannot infer sampling cadence")

    counts = collections.Counter(frame_deltas)
    highest = max(counts.values())
    frame_step = min(
        delta for delta, count in counts.items() if count == highest
    )
    time_step_s = statistics.median(time_deltas_by_frame[frame_step])
    return frame_step, float(time_step_s)


def _expected_breaks(
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    result = []
    for trajectory in trajectory_rows:
        identity_id = str(trajectory["identity_id"])
        points = trajectory.get("points") or []
        for left, right in zip(points, points[1:]):
            if left.get("segment_id") == right.get("segment_id"):
                continue
            result.append(
                {
                    "identity_id": identity_id,
                    "left_frame_index": int(left["frame_index"]),
                    "right_frame_index": int(right["frame_index"]),
                    "left_timestamp_s": float(left["timestamp_s"]),
                    "right_timestamp_s": float(right["timestamp_s"]),
                    "duration_s": (
                        float(right["timestamp_s"])
                        - float(left["timestamp_s"])
                    ),
                }
            )
    return result


def _normalized_break(
    row: Mapping[str, Any],
) -> tuple[str, int, int]:
    return (
        str(row["identity_id"]),
        int(row["left_frame_index"]),
        int(row["right_frame_index"]),
    )


def _point_review_flags(
    point: Mapping[str, Any],
    config: FinalDiagnosticsConfig,
) -> list[str]:
    flags = []
    speed = point.get("speed_mps")
    if (
        speed is not None
        and _is_finite(speed)
        and float(speed) > config.review_speed_mps
    ):
        flags.append("speed_gt_12mps")

    acceleration = point.get("acceleration_mps2")
    if (
        acceleration is not None
        and _is_finite(acceleration)
        and float(acceleration) > config.review_acceleration_mps2
    ):
        flags.append("acceleration_gt_12mps2")

    adjustment = point.get("position_adjustment_m")
    if (
        adjustment is not None
        and _is_finite(adjustment)
        and float(adjustment) > config.review_position_adjustment_m
    ):
        flags.append("smoothing_adjustment_gt_1m")

    position_std = point.get("position_std_m")
    if (
        position_std is not None
        and _is_finite(position_std)
        and float(position_std) > config.review_position_std_m
    ):
        flags.append("position_uncertainty_gt_0_5m")

    velocity_std = point.get("velocity_std_mps")
    if (
        velocity_std is not None
        and _is_finite(velocity_std)
        and float(velocity_std) > config.review_velocity_std_mps
    ):
        flags.append("velocity_uncertainty_gt_2_5mps")
    return flags


def _point_blockers(
    point: Mapping[str, Any],
    config: FinalDiagnosticsConfig,
) -> list[str]:
    blockers = []
    for name in (
        "frame_index",
        "timestamp_s",
        "measurement_x",
        "measurement_z",
        "smoothed_x",
        "smoothed_z",
        "measurement_confidence",
        "measurement_sigma_m",
        "position_adjustment_m",
        "position_std_m",
    ):
        if not _is_finite(point.get(name)):
            blockers.append(f"nonfinite_{name}")

    if blockers:
        return blockers

    adjustment = math.hypot(
        float(point["smoothed_x"]) - float(point["measurement_x"]),
        float(point["smoothed_z"]) - float(point["measurement_z"]),
    )
    if (
        abs(adjustment - float(point["position_adjustment_m"]))
        > config.numeric_tolerance
    ):
        blockers.append("position_adjustment_inconsistent")

    vx = point.get("vx_mps")
    vz = point.get("vz_mps")
    speed = point.get("speed_mps")
    ax = point.get("ax_mps2")
    az = point.get("az_mps2")
    acceleration = point.get("acceleration_mps2")
    velocity_std = point.get("velocity_std_mps")
    quality = str(point.get("kinematics_quality"))
    smoothing_applied = bool(
        (point.get("metadata") or {}).get("smoothing_applied", False)
    )

    if quality == "rts_cv":
        if not smoothing_applied:
            blockers.append("rts_quality_without_smoothing")
        for name, value in (
            ("vx_mps", vx),
            ("vz_mps", vz),
            ("speed_mps", speed),
            ("ax_mps2", ax),
            ("az_mps2", az),
            ("acceleration_mps2", acceleration),
            ("velocity_std_mps", velocity_std),
        ):
            if not _is_finite(value):
                blockers.append(f"rts_missing_{name}")

    elif quality == "two_point_velocity_only":
        if smoothing_applied:
            blockers.append("two_point_segment_was_smoothed")
        for name, value in (
            ("vx_mps", vx),
            ("vz_mps", vz),
            ("speed_mps", speed),
        ):
            if not _is_finite(value):
                blockers.append(f"two_point_missing_{name}")
        if any(
            value is not None
            for value in (ax, az, acceleration, velocity_std)
        ):
            blockers.append("two_point_has_unsupported_kinematics")

    elif quality == "singleton_no_kinematics":
        if smoothing_applied:
            blockers.append("singleton_was_smoothed")
        if any(
            value is not None
            for value in (vx, vz, speed, ax, az, acceleration, velocity_std)
        ):
            blockers.append("singleton_has_kinematics")
    else:
        blockers.append("unknown_kinematics_quality")

    if _is_finite(vx) and _is_finite(vz) and _is_finite(speed):
        expected_speed = math.hypot(float(vx), float(vz))
        if (
            abs(expected_speed - float(speed))
            > config.numeric_tolerance
        ):
            blockers.append("speed_vector_inconsistent")

    if (
        _is_finite(ax)
        and _is_finite(az)
        and _is_finite(acceleration)
    ):
        expected_acceleration = math.hypot(float(ax), float(az))
        if (
            abs(expected_acceleration - float(acceleration))
            > config.numeric_tolerance
        ):
            blockers.append("acceleration_vector_inconsistent")

    return blockers


def audit_final_trajectories(
    *,
    trajectory_rows: Sequence[Mapping[str, Any]],
    segment_break_rows: Sequence[Mapping[str, Any]] = (),
    source_summary: Mapping[str, Any] | None = None,
    config: FinalDiagnosticsConfig | None = None,
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
    dict[str, Any],
]:
    if config is None:
        config = FinalDiagnosticsConfig()

    trajectory_rows = tuple(dict(row) for row in trajectory_rows)
    if not trajectory_rows:
        raise ValueError("no trajectory rows supplied")

    frame_step, time_step_s = _infer_cadence(trajectory_rows)

    blockers: list[dict[str, Any]] = []
    point_reviews: list[dict[str, Any]] = []
    segment_audits: list[dict[str, Any]] = []
    identity_audits: list[dict[str, Any]] = []

    seen_identities: set[str] = set()
    total_points = 0
    origin_counts: collections.Counter[str] = collections.Counter()
    review_flag_counts: collections.Counter[str] = collections.Counter()
    blocker_flag_counts: collections.Counter[str] = collections.Counter()

    all_speeds: list[float] = []
    all_accelerations: list[float] = []
    all_adjustments: list[float] = []
    all_position_stds: list[float] = []
    all_velocity_stds: list[float] = []

    for trajectory in sorted(
        trajectory_rows,
        key=lambda row: str(row["identity_id"]),
    ):
        identity_id = str(trajectory["identity_id"])

        if identity_id in seen_identities:
            blocker = {
                "scope": "identity",
                "identity_id": identity_id,
                "flag": "duplicate_identity_id",
            }
            blockers.append(blocker)
            blocker_flag_counts[blocker["flag"]] += 1
            continue
        seen_identities.add(identity_id)

        if trajectory.get("trajectory_version") != SMOOTHING_VERSION:
            blocker = {
                "scope": "identity",
                "identity_id": identity_id,
                "flag": "trajectory_version_mismatch",
            }
            blockers.append(blocker)
            blocker_flag_counts[blocker["flag"]] += 1

        points = list(trajectory.get("points") or [])
        total_points += len(points)

        if int(trajectory.get("point_count", len(points))) != len(points):
            blocker = {
                "scope": "identity",
                "identity_id": identity_id,
                "flag": "point_count_mismatch",
            }
            blockers.append(blocker)
            blocker_flag_counts[blocker["flag"]] += 1

        if not points:
            blocker = {
                "scope": "identity",
                "identity_id": identity_id,
                "flag": "empty_trajectory",
            }
            blockers.append(blocker)
            blocker_flag_counts[blocker["flag"]] += 1
            continue

        segments: collections.OrderedDict[
            str, list[dict[str, Any]]
        ] = collections.OrderedDict()
        previous_frame = None
        previous_time = None
        previous_segment = None
        identity_review_flags: set[str] = set()

        for point in points:
            segment_id = str(point["segment_id"])
            segments.setdefault(segment_id, []).append(point)
            origin_counts[str(point.get("origin"))] += 1

            frame = int(point["frame_index"])
            timestamp = float(point["timestamp_s"])
            if previous_frame is not None:
                if frame <= previous_frame:
                    blocker = {
                        "scope": "point",
                        "identity_id": identity_id,
                        "segment_id": segment_id,
                        "frame_index": frame,
                        "flag": "frame_not_strictly_increasing",
                    }
                    blockers.append(blocker)
                    blocker_flag_counts[blocker["flag"]] += 1
                if timestamp <= previous_time:
                    blocker = {
                        "scope": "point",
                        "identity_id": identity_id,
                        "segment_id": segment_id,
                        "frame_index": frame,
                        "flag": "timestamp_not_strictly_increasing",
                    }
                    blockers.append(blocker)
                    blocker_flag_counts[blocker["flag"]] += 1

                frame_delta = frame - previous_frame
                time_delta = timestamp - previous_time

                if segment_id == previous_segment:
                    if frame_delta != frame_step:
                        blocker = {
                            "scope": "point",
                            "identity_id": identity_id,
                            "segment_id": segment_id,
                            "frame_index": frame,
                            "flag": "intra_segment_frame_gap",
                        }
                        blockers.append(blocker)
                        blocker_flag_counts[blocker["flag"]] += 1
                    if (
                        abs(time_delta - time_step_s)
                        > config.cadence_tolerance_s
                    ):
                        blocker = {
                            "scope": "point",
                            "identity_id": identity_id,
                            "segment_id": segment_id,
                            "frame_index": frame,
                            "flag": "intra_segment_time_gap",
                        }
                        blockers.append(blocker)
                        blocker_flag_counts[blocker["flag"]] += 1
                else:
                    if (
                        frame_delta == frame_step
                        and abs(time_delta - time_step_s)
                        <= config.cadence_tolerance_s
                    ):
                        blocker = {
                            "scope": "segment",
                            "identity_id": identity_id,
                            "segment_id": segment_id,
                            "frame_index": frame,
                            "flag": "segment_break_without_gap",
                        }
                        blockers.append(blocker)
                        blocker_flag_counts[blocker["flag"]] += 1

            for flag in _point_blockers(point, config):
                blocker = {
                    "scope": "point",
                    "identity_id": identity_id,
                    "segment_id": segment_id,
                    "frame_index": frame,
                    "timestamp_s": timestamp,
                    "flag": flag,
                }
                blockers.append(blocker)
                blocker_flag_counts[flag] += 1

            review_flags = _point_review_flags(point, config)
            if review_flags:
                review = {
                    "schema_version": 1,
                    "diagnostics_version": FINAL_DIAGNOSTICS_VERSION,
                    "identity_id": identity_id,
                    "segment_id": segment_id,
                    "frame_index": frame,
                    "timestamp_s": timestamp,
                    "origin": str(point.get("origin")),
                    "kinematics_quality": str(
                        point.get("kinematics_quality")
                    ),
                    "speed_mps": point.get("speed_mps"),
                    "acceleration_mps2": point.get("acceleration_mps2"),
                    "position_adjustment_m": point.get(
                        "position_adjustment_m"
                    ),
                    "position_std_m": point.get("position_std_m"),
                    "velocity_std_mps": point.get("velocity_std_mps"),
                    "review_flags": review_flags,
                }
                point_reviews.append(review)
                for flag in review_flags:
                    review_flag_counts[flag] += 1
                    identity_review_flags.add(flag)

            for key, destination in (
                ("speed_mps", all_speeds),
                ("acceleration_mps2", all_accelerations),
                ("position_adjustment_m", all_adjustments),
                ("position_std_m", all_position_stds),
                ("velocity_std_mps", all_velocity_stds),
            ):
                value = point.get(key)
                if value is not None and _is_finite(value):
                    destination.append(float(value))

            previous_frame = frame
            previous_time = timestamp
            previous_segment = segment_id

        if int(
            trajectory.get("segment_count", len(segments))
        ) != len(segments):
            blocker = {
                "scope": "identity",
                "identity_id": identity_id,
                "flag": "segment_count_mismatch",
            }
            blockers.append(blocker)
            blocker_flag_counts[blocker["flag"]] += 1

        reviews_by_segment: dict[str, set[str]] = collections.defaultdict(set)
        for row in point_reviews:
            if row["identity_id"] == identity_id:
                reviews_by_segment[row["segment_id"]].update(
                    row["review_flags"]
                )

        for segment_index, (segment_id, segment_points) in enumerate(
            segments.items(),
            start=1,
        ):
            review_flags = set(reviews_by_segment.get(segment_id, set()))
            if len(segment_points) < 3:
                review_flags.add("short_segment")
                identity_review_flags.add("short_segment")

            segment_audits.append(
                {
                    "schema_version": 1,
                    "diagnostics_version": FINAL_DIAGNOSTICS_VERSION,
                    "identity_id": identity_id,
                    "segment_id": segment_id,
                    "segment_index": segment_index,
                    "point_count": len(segment_points),
                    "start_frame_index": int(
                        segment_points[0]["frame_index"]
                    ),
                    "end_frame_index": int(
                        segment_points[-1]["frame_index"]
                    ),
                    "start_timestamp_s": float(
                        segment_points[0]["timestamp_s"]
                    ),
                    "end_timestamp_s": float(
                        segment_points[-1]["timestamp_s"]
                    ),
                    "kinematics_quality_counts": dict(
                        collections.Counter(
                            str(point.get("kinematics_quality"))
                            for point in segment_points
                        )
                    ),
                    "max_speed_mps": max(
                        (
                            float(point["speed_mps"])
                            for point in segment_points
                            if (
                                point.get("speed_mps") is not None
                                and _is_finite(point.get("speed_mps"))
                            )
                        ),
                        default=None,
                    ),
                    "max_acceleration_mps2": max(
                        (
                            float(point["acceleration_mps2"])
                            for point in segment_points
                            if (
                                point.get("acceleration_mps2") is not None
                                and _is_finite(
                                    point.get("acceleration_mps2")
                                )
                            )
                        ),
                        default=None,
                    ),
                    "max_position_adjustment_m": max(
                        float(point["position_adjustment_m"])
                        for point in segment_points
                    ),
                    "max_position_std_m": max(
                        float(point["position_std_m"])
                        for point in segment_points
                    ),
                    "review_flags": sorted(review_flags),
                    "review_required": bool(review_flags),
                }
            )

        identity_audits.append(
            {
                "schema_version": 1,
                "diagnostics_version": FINAL_DIAGNOSTICS_VERSION,
                "identity_id": identity_id,
                "point_count": len(points),
                "segment_count": len(segments),
                "review_flags": sorted(identity_review_flags),
                "review_required": bool(identity_review_flags),
            }
        )

    expected_breaks = _expected_breaks(trajectory_rows)
    expected_break_keys = {
        _normalized_break(row) for row in expected_breaks
    }
    supplied_break_keys = {
        _normalized_break(row) for row in segment_break_rows
    }

    for missing in sorted(expected_break_keys - supplied_break_keys):
        blocker = {
            "scope": "segment_break",
            "identity_id": missing[0],
            "left_frame_index": missing[1],
            "right_frame_index": missing[2],
            "flag": "missing_segment_break_artifact",
        }
        blockers.append(blocker)
        blocker_flag_counts[blocker["flag"]] += 1

    for extra in sorted(supplied_break_keys - expected_break_keys):
        blocker = {
            "scope": "segment_break",
            "identity_id": extra[0],
            "left_frame_index": extra[1],
            "right_frame_index": extra[2],
            "flag": "unexpected_segment_break_artifact",
        }
        blockers.append(blocker)
        blocker_flag_counts[blocker["flag"]] += 1

    segment_count = sum(
        int(row.get("segment_count", 0))
        for row in trajectory_rows
    )

    if source_summary is not None:
        if source_summary.get("smoothing_version") != SMOOTHING_VERSION:
            blocker = {
                "scope": "summary",
                "flag": "source_smoothing_version_mismatch",
            }
            blockers.append(blocker)
            blocker_flag_counts[blocker["flag"]] += 1

        expected_counts = {
            "trajectory_count": len(trajectory_rows),
            "output_point_count": total_points,
            "segment_count": segment_count,
            "preserved_gap_break_count": len(expected_breaks),
        }
        for key, expected in expected_counts.items():
            actual = source_summary.get(key)
            if actual != expected:
                blocker = {
                    "scope": "summary",
                    "flag": f"source_{key}_mismatch",
                    "expected": expected,
                    "actual": actual,
                }
                blockers.append(blocker)
                blocker_flag_counts[blocker["flag"]] += 1

    speed_reviews = sorted(
        (
            row
            for row in point_reviews
            if "speed_gt_12mps" in row["review_flags"]
        ),
        key=lambda row: float(row["speed_mps"]),
        reverse=True,
    )
    acceleration_reviews = sorted(
        (
            row
            for row in point_reviews
            if "acceleration_gt_12mps2" in row["review_flags"]
        ),
        key=lambda row: float(row["acceleration_mps2"]),
        reverse=True,
    )
    adjustment_reviews = sorted(
        (
            row
            for row in point_reviews
            if "smoothing_adjustment_gt_1m" in row["review_flags"]
        ),
        key=lambda row: float(row["position_adjustment_m"]),
        reverse=True,
    )
    uncertainty_reviews = sorted(
        (
            row
            for row in point_reviews
            if any(
                flag.startswith("position_uncertainty")
                or flag.startswith("velocity_uncertainty")
                for flag in row["review_flags"]
            )
        ),
        key=lambda row: max(
            float(row.get("position_std_m") or 0.0),
            float(row.get("velocity_std_mps") or 0.0),
        ),
        reverse=True,
    )
    short_segments = [
        row
        for row in segment_audits
        if "short_segment" in row["review_flags"]
    ]

    segment_review_count = sum(
        bool(row["review_flags"]) for row in segment_audits
    )
    identity_review_count = sum(
        bool(row["review_flags"]) for row in identity_audits
    )

    review_queue = {
        "schema_version": 1,
        "diagnostics_version": FINAL_DIAGNOSTICS_VERSION,
        "provisional": True,
        "publishable": len(blockers) == 0,
        "speed_reviews": speed_reviews[: config.top_n_review_items],
        "acceleration_reviews": acceleration_reviews[
            : config.top_n_review_items
        ],
        "adjustment_reviews": adjustment_reviews[
            : config.top_n_review_items
        ],
        "uncertainty_reviews": uncertainty_reviews[
            : config.top_n_review_items
        ],
        "short_segments": short_segments[: config.top_n_review_items],
    }

    summary = {
        "schema_version": 1,
        "diagnostics_version": FINAL_DIAGNOSTICS_VERSION,
        "source_smoothing_version": SMOOTHING_VERSION,
        "provisional": True,
        "final_stage_completed": False,
        "publishable": len(blockers) == 0,
        "trajectory_count": len(trajectory_rows),
        "point_count": total_points,
        "segment_count": segment_count,
        "segment_break_count": len(expected_breaks),
        "blocker_count": len(blockers),
        "blocker_flag_counts": dict(sorted(blocker_flag_counts.items())),
        "review_point_count": len(point_reviews),
        "review_segment_count": segment_review_count,
        "review_identity_count": identity_review_count,
        "review_flag_counts": dict(sorted(review_flag_counts.items())),
        "short_segment_count": len(short_segments),
        "point_origin_counts": dict(sorted(origin_counts.items())),
        "speed_mps": _distribution(all_speeds),
        "acceleration_mps2": _distribution(all_accelerations),
        "position_adjustment_m": _distribution(all_adjustments),
        "position_std_m": _distribution(all_position_stds),
        "velocity_std_mps": _distribution(all_velocity_stds),
        "sampling_cadence": {
            "frame_step": frame_step,
            "time_step_s": time_step_s,
        },
        "review_thresholds": {
            "speed_mps": config.review_speed_mps,
            "acceleration_mps2": config.review_acceleration_mps2,
            "position_adjustment_m": config.review_position_adjustment_m,
            "position_std_m": config.review_position_std_m,
            "velocity_std_mps": config.review_velocity_std_mps,
        },
    }

    return (
        tuple(identity_audits),
        tuple(segment_audits),
        tuple(point_reviews),
        tuple(blockers),
        review_queue,
        summary,
    )
