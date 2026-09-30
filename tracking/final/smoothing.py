"""Offline temporal smoothing + kinematics for Tracking V1.

Hito 4D consumes `safe-gap-v1` trajectories.  Remaining gaps are HARD segment
boundaries: the smoother never sees observations from the opposite side of a
preserved gap.

Within each continuous segment we use a constant-velocity Kalman filter plus a
Rauch-Tung-Striebel backward pass.  Measurement trust depends on provenance:
real observations are trusted most, fused observations remain strong, and
4C-interpolated points are deliberately weaker measurements.

No point/frame is created or removed here.  Input positions are preserved as
`measurement_x/z`; the smoother writes separate `smoothed_x/z` and kinematics.
"""

from __future__ import annotations

import collections
import json
import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

from .models import (
    FusionMode,
    IdentityTrajectory,
    TrajectoryPoint,
    TrajectoryPointOrigin,
)
from .reconstruction import (
    SamplingCadence,
    infer_sampling_cadence,
)


SMOOTHING_VERSION = "rts-cv-v1"


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("kinematic value must be finite")
    return result


@dataclass(frozen=True, slots=True)
class TemporalSmoothingConfig:
    process_acceleration_sigma_mps2: float = 12.0
    observed_position_sigma_m: float = 0.30
    fused_position_sigma_m: float = 0.35
    interpolated_position_sigma_m: float = 0.90
    fused_spread_weight: float = 0.50
    minimum_confidence: float = 0.15
    initial_velocity_sigma_mps: float = 8.0
    initial_velocity_fit_points: int = 5
    minimum_rts_segment_points: int = 3
    cadence_tolerance_s: float = 0.02
    adjustment_review_threshold_m: float = 1.50

    def __post_init__(self) -> None:
        positive = (
            "process_acceleration_sigma_mps2",
            "observed_position_sigma_m",
            "fused_position_sigma_m",
            "interpolated_position_sigma_m",
            "initial_velocity_sigma_mps",
            "cadence_tolerance_s",
            "adjustment_review_threshold_m",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.fused_spread_weight < 0:
            raise ValueError(
                "fused_spread_weight must be >= 0"
            )
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError(
                "minimum_confidence must be in (0,1]"
            )
        if self.initial_velocity_fit_points < 2:
            raise ValueError(
                "initial_velocity_fit_points must be >= 2"
            )
        if self.minimum_rts_segment_points < 3:
            raise ValueError(
                "minimum_rts_segment_points must be >= 3"
            )


@dataclass(frozen=True, slots=True)
class SmoothedTrajectoryPoint:
    identity_id: str
    segment_id: str
    segment_index: int
    frame_index: int
    timestamp_s: float
    measurement_x: float
    measurement_z: float
    smoothed_x: float
    smoothed_z: float
    measurement_confidence: float
    measurement_sigma_m: float
    origin: str
    fusion_mode: str
    source_tracklet_ids: tuple[str, ...]
    source_detection_ids: tuple[str, ...]
    source_count: int
    position_spread_m: float
    position_adjustment_m: float
    position_std_m: float
    vx_mps: float | None
    vz_mps: float | None
    speed_mps: float | None
    ax_mps2: float | None
    az_mps2: float | None
    acceleration_mps2: float | None
    velocity_std_mps: float | None
    kinematics_quality: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "smoothing_version": SMOOTHING_VERSION,
            "identity_id": self.identity_id,
            "segment_id": self.segment_id,
            "segment_index": self.segment_index,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "measurement_x": self.measurement_x,
            "measurement_z": self.measurement_z,
            "smoothed_x": self.smoothed_x,
            "smoothed_z": self.smoothed_z,
            "measurement_confidence": (
                self.measurement_confidence
            ),
            "measurement_sigma_m": (
                self.measurement_sigma_m
            ),
            "origin": self.origin,
            "fusion_mode": self.fusion_mode,
            "source_tracklet_ids": list(
                self.source_tracklet_ids
            ),
            "source_detection_ids": list(
                self.source_detection_ids
            ),
            "source_count": self.source_count,
            "position_spread_m": (
                self.position_spread_m
            ),
            "position_adjustment_m": (
                self.position_adjustment_m
            ),
            "position_std_m": self.position_std_m,
            "vx_mps": self.vx_mps,
            "vz_mps": self.vz_mps,
            "speed_mps": self.speed_mps,
            "ax_mps2": self.ax_mps2,
            "az_mps2": self.az_mps2,
            "acceleration_mps2": (
                self.acceleration_mps2
            ),
            "velocity_std_mps": (
                self.velocity_std_mps
            ),
            "kinematics_quality": (
                self.kinematics_quality
            ),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SmoothedIdentityTrajectory:
    identity_id: str
    tracklet_ids: tuple[str, ...]
    points: tuple[SmoothedTrajectoryPoint, ...]
    segment_count: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trajectory_version": SMOOTHING_VERSION,
            "identity_id": self.identity_id,
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": len(self.tracklet_ids),
            "point_count": len(self.points),
            "segment_count": self.segment_count,
            "points": [
                point.to_dict()
                for point in self.points
            ],
            "metadata": dict(self.metadata),
        }


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source}:{line_number} is not an object"
                )
            yield payload


def load_trajectories(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[IdentityTrajectory, ...]:
    result = tuple(
        IdentityTrajectory.from_dict(row)
        for row in rows
    )
    if not result:
        raise ValueError("no trajectories supplied")
    if len({row.identity_id for row in result}) != len(result):
        raise ValueError("duplicate identity trajectory")
    return result


def _distribution(
    values: Sequence[float],
) -> dict[str, Any]:
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
        return (
            ordered[low] * (1.0 - alpha)
            + ordered[high] * alpha
        )

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


def _measurement_sigma(
    point: TrajectoryPoint,
    config: TemporalSmoothingConfig,
) -> float:
    confidence = max(
        point.confidence,
        config.minimum_confidence,
    )
    if point.origin is TrajectoryPointOrigin.OBSERVED:
        base = config.observed_position_sigma_m
    elif point.origin is TrajectoryPointOrigin.FUSED:
        base = math.sqrt(
            config.fused_position_sigma_m**2
            + (
                config.fused_spread_weight
                * point.position_spread_m
            ) ** 2
        )
    elif point.origin is TrajectoryPointOrigin.INTERPOLATED:
        base = config.interpolated_position_sigma_m
    else:  # defensive for future enum extension
        raise ValueError(
            f"unsupported point origin: {point.origin}"
        )
    return base / math.sqrt(confidence)


def split_continuous_segments(
    trajectory: IdentityTrajectory,
    cadence: SamplingCadence,
    *,
    tolerance_s: float,
) -> tuple[
    tuple[tuple[TrajectoryPoint, ...], ...],
    tuple[dict[str, Any], ...],
]:
    segments: list[list[TrajectoryPoint]] = [
        [trajectory.points[0]]
    ]
    breaks: list[dict[str, Any]] = []

    for left, right in zip(
        trajectory.points,
        trajectory.points[1:],
    ):
        frame_delta = (
            right.frame_index - left.frame_index
        )
        time_delta = (
            right.timestamp_s - left.timestamp_s
        )
        continuous = (
            frame_delta == cadence.frame_step
            and abs(
                time_delta - cadence.time_step_s
            )
            <= tolerance_s
        )
        if continuous:
            segments[-1].append(right)
            continue

        breaks.append(
            {
                "schema_version": 1,
                "smoothing_version": SMOOTHING_VERSION,
                "identity_id": trajectory.identity_id,
                "left_frame_index": left.frame_index,
                "right_frame_index": right.frame_index,
                "left_timestamp_s": left.timestamp_s,
                "right_timestamp_s": right.timestamp_s,
                "frame_delta": frame_delta,
                "duration_s": time_delta,
                "reason": "preserved_gap_boundary",
            }
        )
        segments.append([right])

    return (
        tuple(tuple(segment) for segment in segments),
        tuple(breaks),
    )


def _initial_velocity(
    points: Sequence[TrajectoryPoint],
    config: TemporalSmoothingConfig,
) -> tuple[float, float]:
    count = min(
        len(points),
        config.initial_velocity_fit_points,
    )
    if count < 2:
        return 0.0, 0.0

    timestamps = np.array(
        [
            points[index].timestamp_s
            for index in range(count)
        ],
        dtype=float,
    )
    tau = timestamps - timestamps[0]
    design = np.column_stack(
        [np.ones(count), tau]
    )
    x_values = np.array(
        [points[index].x for index in range(count)],
        dtype=float,
    )
    z_values = np.array(
        [points[index].z for index in range(count)],
        dtype=float,
    )
    vx = float(
        np.linalg.lstsq(
            design, x_values, rcond=None
        )[0][1]
    )
    vz = float(
        np.linalg.lstsq(
            design, z_values, rcond=None
        )[0][1]
    )
    return vx, vz


def _filter_and_rts(
    points: Sequence[TrajectoryPoint],
    config: TemporalSmoothingConfig,
) -> tuple[np.ndarray, list[np.ndarray]]:
    count = len(points)
    if count < config.minimum_rts_segment_points:
        raise ValueError(
            "RTS requires a sufficiently long segment"
        )

    vx0, vz0 = _initial_velocity(
        points, config
    )
    first_sigma = _measurement_sigma(
        points[0], config
    )
    state = np.array(
        [
            points[0].x,
            points[0].z,
            vx0,
            vz0,
        ],
        dtype=float,
    )
    covariance = np.diag(
        [
            first_sigma**2,
            first_sigma**2,
            config.initial_velocity_sigma_mps**2,
            config.initial_velocity_sigma_mps**2,
        ]
    )

    observation_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    identity = np.eye(4, dtype=float)

    filtered_states: list[np.ndarray] = []
    filtered_covariances: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    predicted_covariances: list[np.ndarray] = []
    transitions: list[np.ndarray] = []

    for index, point in enumerate(points):
        if index == 0:
            transition = np.eye(4, dtype=float)
            predicted_state = state.copy()
            predicted_covariance = (
                covariance.copy()
            )
        else:
            dt = (
                point.timestamp_s
                - points[index - 1].timestamp_s
            )
            transition = np.array(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            noise_map = np.array(
                [
                    [0.5 * dt * dt, 0.0],
                    [0.0, 0.5 * dt * dt],
                    [dt, 0.0],
                    [0.0, dt],
                ],
                dtype=float,
            )
            process_covariance = (
                config.process_acceleration_sigma_mps2
                ** 2
            ) * (
                noise_map @ noise_map.T
            )
            predicted_state = (
                transition @ state
            )
            predicted_covariance = (
                transition
                @ covariance
                @ transition.T
                + process_covariance
            )

        sigma = _measurement_sigma(
            point, config
        )
        measurement_covariance = np.eye(
            2, dtype=float
        ) * (sigma**2)
        measurement = np.array(
            [point.x, point.z],
            dtype=float,
        )
        innovation = (
            measurement
            - observation_matrix
            @ predicted_state
        )
        innovation_covariance = (
            observation_matrix
            @ predicted_covariance
            @ observation_matrix.T
            + measurement_covariance
        )
        kalman_gain = (
            predicted_covariance
            @ observation_matrix.T
            @ np.linalg.inv(
                innovation_covariance
            )
        )
        state = (
            predicted_state
            + kalman_gain @ innovation
        )
        covariance = (
            identity
            - kalman_gain
            @ observation_matrix
        ) @ predicted_covariance

        transitions.append(transition)
        predicted_states.append(
            predicted_state.copy()
        )
        predicted_covariances.append(
            predicted_covariance.copy()
        )
        filtered_states.append(
            state.copy()
        )
        filtered_covariances.append(
            covariance.copy()
        )

    smoothed_states = [
        value.copy()
        for value in filtered_states
    ]
    smoothed_covariances = [
        value.copy()
        for value in filtered_covariances
    ]

    for index in range(
        count - 2,
        -1,
        -1,
    ):
        transition = transitions[index + 1]
        smoother_gain = (
            filtered_covariances[index]
            @ transition.T
            @ np.linalg.inv(
                predicted_covariances[
                    index + 1
                ]
            )
        )
        smoothed_states[index] = (
            filtered_states[index]
            + smoother_gain
            @ (
                smoothed_states[index + 1]
                - predicted_states[index + 1]
            )
        )
        smoothed_covariances[index] = (
            filtered_covariances[index]
            + smoother_gain
            @ (
                smoothed_covariances[
                    index + 1
                ]
                - predicted_covariances[
                    index + 1
                ]
            )
            @ smoother_gain.T
        )

    return (
        np.stack(smoothed_states),
        smoothed_covariances,
    )


def _short_segment_states(
    points: Sequence[TrajectoryPoint],
) -> tuple[np.ndarray, list[np.ndarray]]:
    count = len(points)
    states = np.zeros(
        (count, 4),
        dtype=float,
    )
    covariances = [
        np.zeros((4, 4), dtype=float)
        for _ in points
    ]
    for index, point in enumerate(points):
        states[index, 0] = point.x
        states[index, 1] = point.z

    if count == 2:
        dt = (
            points[1].timestamp_s
            - points[0].timestamp_s
        )
        vx = (
            points[1].x - points[0].x
        ) / dt
        vz = (
            points[1].z - points[0].z
        ) / dt
        states[:, 2] = vx
        states[:, 3] = vz

    return states, covariances


def _accelerations(
    points: Sequence[TrajectoryPoint],
    states: np.ndarray,
) -> list[tuple[float, float] | None]:
    count = len(points)
    if count < 3:
        return [None] * count

    result: list[
        tuple[float, float] | None
    ] = []
    for index in range(count):
        if index == 0:
            left_index = 0
            right_index = 1
        elif index == count - 1:
            left_index = count - 2
            right_index = count - 1
        else:
            left_index = index - 1
            right_index = index + 1

        dt = (
            points[right_index].timestamp_s
            - points[left_index].timestamp_s
        )
        if dt <= 0:
            result.append(None)
            continue
        ax = (
            states[right_index, 2]
            - states[left_index, 2]
        ) / dt
        az = (
            states[right_index, 3]
            - states[left_index, 3]
        ) / dt
        result.append(
            (float(ax), float(az))
        )
    return result


def _finite_difference_metrics(
    points: Sequence[TrajectoryPoint],
    positions: Sequence[tuple[float, float]],
) -> tuple[list[float], list[float]]:
    velocities: list[
        tuple[float, float, float]
    ] = []
    speeds: list[float] = []

    for index in range(len(points) - 1):
        dt = (
            points[index + 1].timestamp_s
            - points[index].timestamp_s
        )
        x0, z0 = positions[index]
        x1, z1 = positions[index + 1]
        vx = (x1 - x0) / dt
        vz = (z1 - z0) / dt
        velocities.append((dt, vx, vz))
        speeds.append(math.hypot(vx, vz))

    accelerations: list[float] = []
    for index in range(
        len(velocities) - 1
    ):
        dt = 0.5 * (
            velocities[index][0]
            + velocities[index + 1][0]
        )
        if dt <= 0:
            continue
        ax = (
            velocities[index + 1][1]
            - velocities[index][1]
        ) / dt
        az = (
            velocities[index + 1][2]
            - velocities[index][2]
        ) / dt
        accelerations.append(
            math.hypot(ax, az)
        )

    return speeds, accelerations


def smooth_trajectories(
    *,
    trajectories: Sequence[
        IdentityTrajectory
    ],
    config: TemporalSmoothingConfig | None = None,
    cadence: SamplingCadence | None = None,
) -> tuple[
    tuple[SmoothedIdentityTrajectory, ...],
    tuple[dict[str, Any], ...],
    dict[str, Any],
]:
    if config is None:
        config = TemporalSmoothingConfig()
    if cadence is None:
        cadence = infer_sampling_cadence(
            trajectories
        )

    output: list[
        SmoothedIdentityTrajectory
    ] = []
    segment_breaks: list[dict[str, Any]] = []

    all_adjustments: list[float] = []
    observed_adjustments: list[float] = []
    fused_adjustments: list[float] = []
    interpolated_adjustments: list[float] = []
    position_stds: list[float] = []
    velocity_stds: list[float] = []
    state_speeds: list[float] = []
    state_accelerations: list[float] = []
    raw_fd_speeds: list[float] = []
    raw_fd_accelerations: list[float] = []
    smooth_fd_speeds: list[float] = []
    smooth_fd_accelerations: list[float] = []

    rts_segment_count = 0
    short_segment_count = 0
    singleton_segment_count = 0
    two_point_segment_count = 0
    adjustment_review_count = 0

    for trajectory in sorted(
        trajectories,
        key=lambda item: item.identity_id,
    ):
        segments, breaks = (
            split_continuous_segments(
                trajectory,
                cadence,
                tolerance_s=(
                    config.cadence_tolerance_s
                ),
            )
        )
        segment_breaks.extend(breaks)
        identity_points: list[
            SmoothedTrajectoryPoint
        ] = []

        for segment_index, segment in enumerate(
            segments,
            start=1,
        ):
            segment_id = (
                f"{trajectory.identity_id}"
                f":segment-{segment_index:03d}"
            )
            if (
                len(segment)
                >= config.minimum_rts_segment_points
            ):
                states, covariances = (
                    _filter_and_rts(
                        segment,
                        config,
                    )
                )
                quality = "rts_cv"
                rts_segment_count += 1
            else:
                states, covariances = (
                    _short_segment_states(
                        segment
                    )
                )
                quality = (
                    "singleton_no_kinematics"
                    if len(segment) == 1
                    else "two_point_velocity_only"
                )
                short_segment_count += 1
                singleton_segment_count += (
                    len(segment) == 1
                )
                two_point_segment_count += (
                    len(segment) == 2
                )

            accelerations = _accelerations(
                segment,
                states,
            )

            raw_positions = [
                (point.x, point.z)
                for point in segment
            ]
            smooth_positions = [
                (
                    float(states[index, 0]),
                    float(states[index, 1]),
                )
                for index in range(
                    len(segment)
                )
            ]
            raw_speed, raw_accel = (
                _finite_difference_metrics(
                    segment,
                    raw_positions,
                )
            )
            smooth_speed, smooth_accel = (
                _finite_difference_metrics(
                    segment,
                    smooth_positions,
                )
            )
            raw_fd_speeds.extend(raw_speed)
            raw_fd_accelerations.extend(
                raw_accel
            )
            smooth_fd_speeds.extend(
                smooth_speed
            )
            smooth_fd_accelerations.extend(
                smooth_accel
            )

            for index, point in enumerate(
                segment
            ):
                sx = float(states[index, 0])
                sz = float(states[index, 1])
                adjustment = math.hypot(
                    sx - point.x,
                    sz - point.z,
                )
                all_adjustments.append(
                    adjustment
                )
                if (
                    point.origin
                    is TrajectoryPointOrigin.OBSERVED
                ):
                    observed_adjustments.append(
                        adjustment
                    )
                elif (
                    point.origin
                    is TrajectoryPointOrigin.FUSED
                ):
                    fused_adjustments.append(
                        adjustment
                    )
                else:
                    interpolated_adjustments.append(
                        adjustment
                    )

                sigma = _measurement_sigma(
                    point,
                    config,
                )

                if quality == "rts_cv":
                    covariance = (
                        covariances[index]
                    )
                    position_std = math.sqrt(
                        max(
                            0.0,
                            0.5
                            * (
                                covariance[0, 0]
                                + covariance[1, 1]
                            ),
                        )
                    )
                    velocity_std = math.sqrt(
                        max(
                            0.0,
                            0.5
                            * (
                                covariance[2, 2]
                                + covariance[3, 3]
                            ),
                        )
                    )
                    vx = float(
                        states[index, 2]
                    )
                    vz = float(
                        states[index, 3]
                    )
                    speed = math.hypot(
                        vx, vz
                    )
                elif len(segment) == 2:
                    position_std = sigma
                    velocity_std = None
                    vx = float(
                        states[index, 2]
                    )
                    vz = float(
                        states[index, 3]
                    )
                    speed = math.hypot(
                        vx, vz
                    )
                else:
                    position_std = sigma
                    velocity_std = None
                    vx = None
                    vz = None
                    speed = None

                accel = accelerations[index]
                if accel is None:
                    ax = None
                    az = None
                    acceleration = None
                else:
                    ax, az = accel
                    acceleration = math.hypot(
                        ax, az
                    )

                position_stds.append(
                    position_std
                )
                if velocity_std is not None:
                    velocity_stds.append(
                        velocity_std
                    )
                if speed is not None:
                    state_speeds.append(
                        speed
                    )
                if acceleration is not None:
                    state_accelerations.append(
                        acceleration
                    )
                if (
                    adjustment
                    > config.adjustment_review_threshold_m
                ):
                    adjustment_review_count += 1

                identity_points.append(
                    SmoothedTrajectoryPoint(
                        identity_id=(
                            trajectory.identity_id
                        ),
                        segment_id=segment_id,
                        segment_index=(
                            segment_index
                        ),
                        frame_index=(
                            point.frame_index
                        ),
                        timestamp_s=(
                            point.timestamp_s
                        ),
                        measurement_x=point.x,
                        measurement_z=point.z,
                        smoothed_x=sx,
                        smoothed_z=sz,
                        measurement_confidence=(
                            point.confidence
                        ),
                        measurement_sigma_m=sigma,
                        origin=point.origin.value,
                        fusion_mode=(
                            point.fusion_mode.value
                        ),
                        source_tracklet_ids=(
                            point.source_tracklet_ids
                        ),
                        source_detection_ids=(
                            point.source_detection_ids
                        ),
                        source_count=(
                            point.source_count
                        ),
                        position_spread_m=(
                            point.position_spread_m
                        ),
                        position_adjustment_m=(
                            adjustment
                        ),
                        position_std_m=(
                            position_std
                        ),
                        vx_mps=vx,
                        vz_mps=vz,
                        speed_mps=speed,
                        ax_mps2=ax,
                        az_mps2=az,
                        acceleration_mps2=(
                            acceleration
                        ),
                        velocity_std_mps=(
                            velocity_std
                        ),
                        kinematics_quality=(
                            quality
                        ),
                        metadata={
                            "source_trajectory_version": (
                                trajectory.trajectory_version
                            ),
                            "measurement_metadata": (
                                dict(point.metadata)
                            ),
                            "smoothing_applied": (
                                quality == "rts_cv"
                            ),
                        },
                    )
                )

        metadata = dict(
            trajectory.metadata
        )
        metadata.update(
            {
                "builder": SMOOTHING_VERSION,
                "source_trajectory_version": (
                    trajectory.trajectory_version
                ),
                "smoothed": True,
                "gap_boundaries_respected": True,
                "segment_count": len(
                    segments
                ),
            }
        )
        output.append(
            SmoothedIdentityTrajectory(
                identity_id=(
                    trajectory.identity_id
                ),
                tracklet_ids=(
                    trajectory.tracklet_ids
                ),
                points=tuple(
                    identity_points
                ),
                segment_count=len(
                    segments
                ),
                metadata=metadata,
            )
        )

    input_point_count = sum(
        len(item.points)
        for item in trajectories
    )
    output_point_count = sum(
        len(item.points)
        for item in output
    )
    if input_point_count != output_point_count:
        raise AssertionError(
            "smoothing changed point count"
        )

    origin_counts = collections.Counter(
        point.origin.value
        for trajectory in trajectories
        for point in trajectory.points
    )

    summary = {
        "schema_version": 1,
        "smoothing_version": SMOOTHING_VERSION,
        "provisional": True,
        "final_stage_completed": False,
        "trajectory_count": len(output),
        "input_point_count": input_point_count,
        "output_point_count": output_point_count,
        "segment_count": (
            rts_segment_count
            + short_segment_count
        ),
        "preserved_gap_break_count": len(
            segment_breaks
        ),
        "rts_segment_count": (
            rts_segment_count
        ),
        "short_segment_count": (
            short_segment_count
        ),
        "singleton_segment_count": (
            singleton_segment_count
        ),
        "two_point_segment_count": (
            two_point_segment_count
        ),
        "point_origin_counts": dict(
            sorted(origin_counts.items())
        ),
        "position_adjustment_m": (
            _distribution(
                all_adjustments
            )
        ),
        "observed_position_adjustment_m": (
            _distribution(
                observed_adjustments
            )
        ),
        "fused_position_adjustment_m": (
            _distribution(
                fused_adjustments
            )
        ),
        "interpolated_position_adjustment_m": (
            _distribution(
                interpolated_adjustments
            )
        ),
        "position_std_m": _distribution(
            position_stds
        ),
        "velocity_std_mps": _distribution(
            velocity_stds
        ),
        "state_speed_mps": _distribution(
            state_speeds
        ),
        "state_acceleration_mps2": (
            _distribution(
                state_accelerations
            )
        ),
        "raw_finite_difference_speed_mps": (
            _distribution(
                raw_fd_speeds
            )
        ),
        "smoothed_finite_difference_speed_mps": (
            _distribution(
                smooth_fd_speeds
            )
        ),
        "raw_finite_difference_acceleration_mps2": (
            _distribution(
                raw_fd_accelerations
            )
        ),
        "smoothed_finite_difference_acceleration_mps2": (
            _distribution(
                smooth_fd_accelerations
            )
        ),
        "adjustment_review_threshold_m": (
            config.adjustment_review_threshold_m
        ),
        "adjustment_review_count": (
            adjustment_review_count
        ),
        "sampling_cadence": {
            "frame_step": cadence.frame_step,
            "time_step_s": cadence.time_step_s,
        },
        "config": {
            "process_acceleration_sigma_mps2": (
                config.process_acceleration_sigma_mps2
            ),
            "observed_position_sigma_m": (
                config.observed_position_sigma_m
            ),
            "fused_position_sigma_m": (
                config.fused_position_sigma_m
            ),
            "interpolated_position_sigma_m": (
                config.interpolated_position_sigma_m
            ),
            "fused_spread_weight": (
                config.fused_spread_weight
            ),
            "minimum_confidence": (
                config.minimum_confidence
            ),
            "initial_velocity_sigma_mps": (
                config.initial_velocity_sigma_mps
            ),
            "initial_velocity_fit_points": (
                config.initial_velocity_fit_points
            ),
            "minimum_rts_segment_points": (
                config.minimum_rts_segment_points
            ),
            "cadence_tolerance_s": (
                config.cadence_tolerance_s
            ),
        },
    }

    return (
        tuple(output),
        tuple(segment_breaks),
        summary,
    )
