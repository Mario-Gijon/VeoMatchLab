"""Hito 6C: dense persistent-actor trajectories with explicit uncertainty.

This stage consumes the Hito 6B V1 anchored actor assignment and the original
Tracking V1 tracklets.  It does **not** mutate Tracking V1 or re-run identity
assignment.

Design goals
------------
* Preserve every real assigned observation as evidence.
* Fuse duplicate same-actor/same-frame observations conservatively.
* Run an offline constant-velocity Kalman filter + RTS smoother over each
  actor's directly-supported interval.
* Fill only bounded internal gaps.  Long or physically implausible gaps stay
  visible as gaps instead of being silently invented.
* Allow only short leading/trailing predictions, with explicit prediction age
  and position uncertainty.
* Keep provenance on every output point so the debug UI can distinguish
  ``observed``, ``fused``, ``interpolated`` and ``predicted`` states.

The output is diagnostic/provisional.  Hito 6E visual validation must happen
before any persistent trajectory is promoted to a canonical downstream stage.
"""

from __future__ import annotations

import collections
import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


TRAJECTORY_VERSION = "persistent-trajectory-v1"
ASSIGNMENT_SOLVER_VERSION = "anchored-actor-assignment-v1"

DIRECT_ORIGINS = {"observed", "fused"}
SYNTHETIC_ORIGINS = {"interpolated", "predicted"}
ALL_ORIGINS = DIRECT_ORIGINS | SYNTHETIC_ORIGINS


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


@dataclass(frozen=True, slots=True)
class PersistentTrajectoryConfig:
    """Conservative 6C reconstruction policy.

    The default gap limits are intentionally much shorter than a tactical
    event.  An internal gap must be supported on both sides; edge prediction
    is shorter still.  Position uncertainty can veto synthetic points even
    when the duration gate passes.
    """

    max_internal_gap_s: float = 1.50
    max_edge_prediction_s: float = 0.50
    max_internal_endpoint_speed_mps: float = 12.0
    max_position_std_m: float = 2.50
    process_acceleration_sigma_mps2: float = 10.0
    observed_position_sigma_m: float = 0.30
    fused_position_sigma_m: float = 0.40
    fused_spread_weight: float = 0.50
    minimum_confidence: float = 0.15
    initial_velocity_sigma_mps: float = 8.0
    initial_velocity_fit_points: int = 5
    minimum_edge_support_points: int = 3
    weighted_fusion_max_spread_m: float = 1.50
    cadence_tolerance_s: float = 0.025

    def __post_init__(self) -> None:
        positive = (
            "max_internal_gap_s",
            "max_edge_prediction_s",
            "max_internal_endpoint_speed_mps",
            "max_position_std_m",
            "process_acceleration_sigma_mps2",
            "observed_position_sigma_m",
            "fused_position_sigma_m",
            "initial_velocity_sigma_mps",
            "weighted_fusion_max_spread_m",
            "cadence_tolerance_s",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.fused_spread_weight < 0:
            raise ValueError("fused_spread_weight must be >= 0")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in (0,1]")
        if self.initial_velocity_fit_points < 2:
            raise ValueError("initial_velocity_fit_points must be >= 2")
        if self.minimum_edge_support_points < 2:
            raise ValueError("minimum_edge_support_points must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_internal_gap_s": self.max_internal_gap_s,
            "max_edge_prediction_s": self.max_edge_prediction_s,
            "max_internal_endpoint_speed_mps": self.max_internal_endpoint_speed_mps,
            "max_position_std_m": self.max_position_std_m,
            "process_acceleration_sigma_mps2": self.process_acceleration_sigma_mps2,
            "observed_position_sigma_m": self.observed_position_sigma_m,
            "fused_position_sigma_m": self.fused_position_sigma_m,
            "fused_spread_weight": self.fused_spread_weight,
            "minimum_confidence": self.minimum_confidence,
            "initial_velocity_sigma_mps": self.initial_velocity_sigma_mps,
            "initial_velocity_fit_points": self.initial_velocity_fit_points,
            "minimum_edge_support_points": self.minimum_edge_support_points,
            "weighted_fusion_max_spread_m": self.weighted_fusion_max_spread_m,
            "cadence_tolerance_s": self.cadence_tolerance_s,
        }


@dataclass(frozen=True, slots=True)
class SamplingCadence:
    frame_step: int
    time_step_s: float
    first_frame_index: int
    last_frame_index: int
    first_timestamp_s: float

    def __post_init__(self) -> None:
        if self.frame_step <= 0:
            raise ValueError("frame_step must be > 0")
        if self.time_step_s <= 0:
            raise ValueError("time_step_s must be > 0")
        if self.last_frame_index < self.first_frame_index:
            raise ValueError("last_frame_index must be >= first_frame_index")

    @property
    def frames(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.first_frame_index,
                self.last_frame_index + 1,
                self.frame_step,
            )
        )

    def timestamp_for_frame(self, frame_index: int) -> float:
        offset = (frame_index - self.first_frame_index) / self.frame_step
        return self.first_timestamp_s + offset * self.time_step_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_step": self.frame_step,
            "time_step_s": self.time_step_s,
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "first_timestamp_s": self.first_timestamp_s,
            "frame_count": len(self.frames),
        }


@dataclass(frozen=True, slots=True)
class _Observation:
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _Measurement:
    actor_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float
    origin: str
    fusion_mode: str
    source_tracklet_ids: tuple[str, ...]
    source_detection_ids: tuple[str, ...]
    position_spread_m: float
    fusion_conflict: bool

    @property
    def source_count(self) -> int:
        return len(self.source_detection_ids)


@dataclass(frozen=True, slots=True)
class _StateSeries:
    frames: tuple[int, ...]
    timestamps: tuple[float, ...]
    states: np.ndarray
    covariances: tuple[np.ndarray, ...]
    measurements: Mapping[int, _Measurement]


@dataclass(slots=True)
class _GapEvent:
    actor_id: str
    gap_kind: str
    start_frame_index: int
    end_frame_index: int
    missing_frame_count: int
    duration_s: float
    action: str
    reason: str
    inserted_frame_indices: list[int] = field(default_factory=list)
    left_support_frame_index: int | None = None
    right_support_frame_index: int | None = None
    endpoint_distance_m: float | None = None
    endpoint_speed_mps: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trajectory_version": TRAJECTORY_VERSION,
            "actor_id": self.actor_id,
            "gap_kind": self.gap_kind,
            "start_frame_index": self.start_frame_index,
            "end_frame_index": self.end_frame_index,
            "missing_frame_count": self.missing_frame_count,
            "duration_s": self.duration_s,
            "action": self.action,
            "reason": self.reason,
            "inserted_frame_indices": list(self.inserted_frame_indices),
            "inserted_point_count": len(self.inserted_frame_indices),
            "left_support_frame_index": self.left_support_frame_index,
            "right_support_frame_index": self.right_support_frame_index,
            "endpoint_distance_m": self.endpoint_distance_m,
            "endpoint_speed_mps": self.endpoint_speed_mps,
        }


def _load_tracklets(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[_Observation, ...]]:
    result: dict[str, tuple[_Observation, ...]] = {}
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        if not tracklet_id or tracklet_id in result:
            raise ValueError(f"invalid/duplicate tracklet_id: {tracklet_id!r}")
        raw_observations = row.get("observations")
        if not isinstance(raw_observations, list) or not raw_observations:
            raise ValueError(
                f"{tracklet_id} must contain canonical tracklet observations"
            )
        observations: list[_Observation] = []
        previous_frame: int | None = None
        previous_time: float | None = None
        for raw in raw_observations:
            frame_index = int(raw["frame_index"])
            timestamp_s = _finite_float("timestamp_s", raw["timestamp_s"])
            if previous_frame is not None and frame_index <= previous_frame:
                raise ValueError(f"{tracklet_id} frames must increase")
            if previous_time is not None and timestamp_s <= previous_time:
                raise ValueError(f"{tracklet_id} timestamps must increase")
            confidence = _finite_float("confidence", raw["confidence"])
            if not 0 <= confidence <= 1:
                raise ValueError("observation confidence must be in [0,1]")
            observations.append(
                _Observation(
                    tracklet_id=tracklet_id,
                    detection_id=str(raw["detection_id"]),
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                    x=_finite_float("x", raw["x"]),
                    z=_finite_float("z", raw["z"]),
                    confidence=confidence,
                )
            )
            previous_frame = frame_index
            previous_time = timestamp_s
        result[tracklet_id] = tuple(observations)
    if not result:
        raise ValueError("no tracklets supplied")
    return result


def _load_assignment(
    rows: Sequence[Mapping[str, Any]],
    *,
    known_tracklets: set[str],
    known_actors: set[str],
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for row in rows:
        if row.get("solver_version") not in {None, ASSIGNMENT_SOLVER_VERSION}:
            raise ValueError("persistent assignment solver version mismatch")
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id not in known_tracklets:
            raise ValueError(f"assignment references unknown tracklet {tracklet_id}")
        if tracklet_id in result:
            raise ValueError(f"duplicate assignment for {tracklet_id}")
        actor_raw = row.get("actor_id")
        actor_id = None if actor_raw is None else str(actor_raw)
        status = str(row.get("status", "assigned" if actor_id else "noise"))
        if actor_id is None:
            if status != "noise":
                raise ValueError(f"{tracklet_id} has null actor but status={status}")
        else:
            if actor_id not in known_actors:
                raise ValueError(f"assignment references unknown actor {actor_id}")
            if status != "assigned":
                raise ValueError(f"{tracklet_id} actor assignment has status={status}")
        result[tracklet_id] = actor_id
    if set(result) != known_tracklets:
        missing = sorted(known_tracklets - set(result))
        extra = sorted(set(result) - known_tracklets)
        raise ValueError(
            "persistent assignments must cover tracklets exactly; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return result


def _load_actors(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        if row.get("solver_version") not in {None, ASSIGNMENT_SOLVER_VERSION}:
            raise ValueError("persistent actor solver version mismatch")
        actor_id = str(row["actor_id"])
        if not actor_id or actor_id in result:
            raise ValueError(f"invalid/duplicate actor_id: {actor_id!r}")
        result[actor_id] = row
    if not result:
        raise ValueError("no persistent actors supplied")
    return result


def infer_sampling_cadence(
    tracklets: Mapping[str, Sequence[_Observation]],
) -> SamplingCadence:
    by_frame: dict[int, list[float]] = collections.defaultdict(list)
    for observations in tracklets.values():
        for obs in observations:
            by_frame[obs.frame_index].append(obs.timestamp_s)
    if len(by_frame) < 2:
        raise ValueError("cannot infer cadence from fewer than two observed frames")

    frame_times = {
        frame: float(statistics.median(values))
        for frame, values in by_frame.items()
    }
    frames = sorted(frame_times)
    deltas = [right - left for left, right in zip(frames, frames[1:]) if right > left]
    if not deltas:
        raise ValueError("cannot infer positive frame cadence")
    counts = collections.Counter(deltas)
    highest = max(counts.values())
    frame_step = min(delta for delta, count in counts.items() if count == highest)

    time_steps = [
        frame_times[right] - frame_times[left]
        for left, right in zip(frames, frames[1:])
        if right - left == frame_step and frame_times[right] > frame_times[left]
    ]
    if not time_steps:
        raise ValueError("cannot infer timestamp cadence")
    time_step_s = float(statistics.median(time_steps))

    first_frame = frames[0]
    last_frame = frames[-1]
    # Anchor the reconstructed clock to the median real timestamp at the first
    # observed frame.  Real measurement timestamps are retained on direct
    # points; this clock is used only for synthetic grid points.
    first_timestamp_s = frame_times[first_frame]
    return SamplingCadence(
        frame_step=frame_step,
        time_step_s=time_step_s,
        first_frame_index=first_frame,
        last_frame_index=last_frame,
        first_timestamp_s=first_timestamp_s,
    )


def _max_pair_distance(observations: Sequence[_Observation]) -> float:
    maximum = 0.0
    for i, left in enumerate(observations):
        for right in observations[i + 1 :]:
            maximum = max(maximum, math.hypot(right.x - left.x, right.z - left.z))
    return maximum


def _fuse_measurement(
    actor_id: str,
    observations: Sequence[_Observation],
    config: PersistentTrajectoryConfig,
) -> _Measurement:
    if not observations:
        raise ValueError("cannot fuse zero observations")
    ordered = sorted(observations, key=lambda item: (item.tracklet_id, item.detection_id))
    frame = ordered[0].frame_index
    if any(item.frame_index != frame for item in ordered):
        raise ValueError("fusion requires one frame")
    timestamp = float(statistics.median(item.timestamp_s for item in ordered))

    if len(ordered) == 1:
        item = ordered[0]
        return _Measurement(
            actor_id=actor_id,
            frame_index=frame,
            timestamp_s=item.timestamp_s,
            x=item.x,
            z=item.z,
            confidence=item.confidence,
            origin="observed",
            fusion_mode="single",
            source_tracklet_ids=(item.tracklet_id,),
            source_detection_ids=(item.detection_id,),
            position_spread_m=0.0,
            fusion_conflict=False,
        )

    max_pair = _max_pair_distance(ordered)
    weights = np.asarray(
        [max(config.minimum_confidence, item.confidence) for item in ordered],
        dtype=np.float64,
    )
    positions = np.asarray([(item.x, item.z) for item in ordered], dtype=np.float64)

    conflict = max_pair > config.weighted_fusion_max_spread_m
    if not conflict:
        center = np.average(positions, axis=0, weights=weights)
        x, z = float(center[0]), float(center[1])
        mode = "weighted_mean"
    else:
        # Robust fallback: select the confidence-weighted medoid instead of
        # averaging two spatially inconsistent detections into empty space.
        costs = []
        for i in range(len(ordered)):
            distance = np.linalg.norm(positions - positions[i], axis=1)
            costs.append(float(np.sum(distance * weights)))
        chosen = min(
            range(len(ordered)),
            key=lambda index: (costs[index], -ordered[index].confidence, ordered[index].detection_id),
        )
        x, z = ordered[chosen].x, ordered[chosen].z
        mode = "medoid_selected"

    spread = max(math.hypot(item.x - x, item.z - z) for item in ordered)
    confidence = _clamp01(1.0 - math.prod(1.0 - item.confidence for item in ordered))
    return _Measurement(
        actor_id=actor_id,
        frame_index=frame,
        timestamp_s=timestamp,
        x=x,
        z=z,
        confidence=confidence,
        origin="fused",
        fusion_mode=mode,
        source_tracklet_ids=tuple(sorted({item.tracklet_id for item in ordered})),
        source_detection_ids=tuple(sorted({item.detection_id for item in ordered})),
        position_spread_m=spread,
        fusion_conflict=conflict,
    )


def _measurement_sigma(
    measurement: _Measurement,
    config: PersistentTrajectoryConfig,
) -> float:
    confidence = max(config.minimum_confidence, measurement.confidence)
    if measurement.origin == "observed":
        base = config.observed_position_sigma_m
    else:
        base = math.sqrt(
            config.fused_position_sigma_m**2
            + (config.fused_spread_weight * measurement.position_spread_m) ** 2
        )
        if measurement.fusion_conflict:
            base *= 1.5
    return base / math.sqrt(confidence)


def _actor_measurements(
    *,
    actors: Mapping[str, Mapping[str, Any]],
    assignment: Mapping[str, str | None],
    tracklets: Mapping[str, Sequence[_Observation]],
    config: PersistentTrajectoryConfig,
) -> dict[str, dict[int, _Measurement]]:
    grouped: dict[str, dict[int, list[_Observation]]] = {
        actor_id: collections.defaultdict(list) for actor_id in actors
    }
    for tracklet_id, actor_id in assignment.items():
        if actor_id is None:
            continue
        for observation in tracklets[tracklet_id]:
            grouped[actor_id][observation.frame_index].append(observation)

    result: dict[str, dict[int, _Measurement]] = {}
    for actor_id in sorted(actors):
        frame_map = grouped[actor_id]
        if not frame_map:
            raise ValueError(f"{actor_id} has no assigned direct observations")
        result[actor_id] = {
            frame: _fuse_measurement(actor_id, values, config)
            for frame, values in sorted(frame_map.items())
        }
    return result


def _initial_velocity(
    measurements: Sequence[_Measurement],
    config: PersistentTrajectoryConfig,
) -> tuple[float, float]:
    count = min(len(measurements), config.initial_velocity_fit_points)
    if count < 2:
        return 0.0, 0.0
    selected = measurements[:count]
    timestamps = np.asarray([item.timestamp_s for item in selected], dtype=np.float64)
    tau = timestamps - timestamps[0]
    design = np.column_stack([np.ones(count), tau])
    x = np.asarray([item.x for item in selected], dtype=np.float64)
    z = np.asarray([item.z for item in selected], dtype=np.float64)
    vx = float(np.linalg.lstsq(design, x, rcond=None)[0][1])
    vz = float(np.linalg.lstsq(design, z, rcond=None)[0][1])
    return vx, vz


def _transition(dt: float, acceleration_sigma: float) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    noise_map = np.array(
        [
            [0.5 * dt * dt, 0.0],
            [0.0, 0.5 * dt * dt],
            [dt, 0.0],
            [0.0, dt],
        ],
        dtype=np.float64,
    )
    process = acceleration_sigma**2 * (noise_map @ noise_map.T)
    return matrix, process


def _filter_and_rts(
    *,
    measurements: Mapping[int, _Measurement],
    cadence: SamplingCadence,
    config: PersistentTrajectoryConfig,
) -> _StateSeries:
    support_frames = sorted(measurements)
    first = support_frames[0]
    last = support_frames[-1]
    if (first - cadence.first_frame_index) % cadence.frame_step != 0:
        raise ValueError("actor first frame is off global cadence")
    if (last - cadence.first_frame_index) % cadence.frame_step != 0:
        raise ValueError("actor last frame is off global cadence")
    frames = tuple(range(first, last + 1, cadence.frame_step))
    timestamps = tuple(cadence.timestamp_for_frame(frame) for frame in frames)
    first_measurement = measurements[first]
    ordered_measurements = [measurements[frame] for frame in support_frames]
    vx0, vz0 = _initial_velocity(ordered_measurements, config)
    first_sigma = _measurement_sigma(first_measurement, config)

    state = np.asarray([first_measurement.x, first_measurement.z, vx0, vz0], dtype=np.float64)
    covariance = np.diag(
        [
            first_sigma**2,
            first_sigma**2,
            config.initial_velocity_sigma_mps**2,
            config.initial_velocity_sigma_mps**2,
        ]
    )
    H = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
    I = np.eye(4, dtype=np.float64)

    filtered_states: list[np.ndarray] = []
    filtered_covs: list[np.ndarray] = []
    predicted_states: list[np.ndarray] = []
    predicted_covs: list[np.ndarray] = []
    transitions: list[np.ndarray] = []

    for index, frame in enumerate(frames):
        if index == 0:
            F = np.eye(4, dtype=np.float64)
            predicted_state = state.copy()
            predicted_cov = covariance.copy()
        else:
            dt = timestamps[index] - timestamps[index - 1]
            F, Q = _transition(dt, config.process_acceleration_sigma_mps2)
            predicted_state = F @ state
            predicted_cov = F @ covariance @ F.T + Q

        measurement = measurements.get(frame)
        if measurement is not None:
            sigma = _measurement_sigma(measurement, config)
            R = np.eye(2, dtype=np.float64) * sigma**2
            z = np.asarray([measurement.x, measurement.z], dtype=np.float64)
            innovation = z - H @ predicted_state
            S = H @ predicted_cov @ H.T + R
            K = predicted_cov @ H.T @ np.linalg.inv(S)
            state = predicted_state + K @ innovation
            covariance = (I - K @ H) @ predicted_cov
        else:
            state = predicted_state
            covariance = predicted_cov

        transitions.append(F)
        predicted_states.append(predicted_state.copy())
        predicted_covs.append(predicted_cov.copy())
        filtered_states.append(state.copy())
        filtered_covs.append(covariance.copy())

    smoothed_states = [value.copy() for value in filtered_states]
    smoothed_covs = [value.copy() for value in filtered_covs]
    for index in range(len(frames) - 2, -1, -1):
        F = transitions[index + 1]
        predicted_cov = predicted_covs[index + 1]
        gain = filtered_covs[index] @ F.T @ np.linalg.pinv(predicted_cov)
        smoothed_states[index] = filtered_states[index] + gain @ (
            smoothed_states[index + 1] - predicted_states[index + 1]
        )
        smoothed_covs[index] = filtered_covs[index] + gain @ (
            smoothed_covs[index + 1] - predicted_cov
        ) @ gain.T

    return _StateSeries(
        frames=frames,
        timestamps=timestamps,
        states=np.stack(smoothed_states),
        covariances=tuple(smoothed_covs),
        measurements=measurements,
    )


def _position_std(covariance: np.ndarray) -> float:
    # Largest axis std is deliberately conservative and easier to reason about
    # than RMS when deciding whether a synthetic point is publishable.
    return math.sqrt(max(0.0, float(covariance[0, 0]), float(covariance[1, 1])))


def _support_neighbors(
    frame: int,
    support_frames: Sequence[int],
) -> tuple[int | None, int | None]:
    left = None
    right = None
    for candidate in support_frames:
        if candidate < frame:
            left = candidate
            continue
        if candidate > frame:
            right = candidate
            break
    return left, right


def _point_payload(
    *,
    actor_id: str,
    frame_index: int,
    timestamp_s: float,
    state: np.ndarray,
    covariance: np.ndarray,
    measurement: _Measurement | None,
    origin: str,
    segment_id: str,
    support_before: int | None,
    support_after: int | None,
    prediction_age_s: float,
    synthetic_support: Sequence[_Measurement],
    cadence: SamplingCadence,
    config: PersistentTrajectoryConfig,
) -> dict[str, Any]:
    if origin not in ALL_ORIGINS:
        raise ValueError(f"invalid persistent point origin: {origin}")
    position_std = _position_std(covariance)
    speed = math.hypot(float(state[2]), float(state[3]))

    if measurement is not None:
        confidence = measurement.confidence
        measurement_x = measurement.x
        measurement_z = measurement.z
        fusion_mode = measurement.fusion_mode
        source_tracklets = list(measurement.source_tracklet_ids)
        source_detections = list(measurement.source_detection_ids)
        source_count = measurement.source_count
        spread = measurement.position_spread_m
        fusion_conflict = measurement.fusion_conflict
    else:
        nearest_frames = [value for value in (support_before, support_after) if value is not None]
        nearest_distance_s = (
            min(abs(frame_index - value) for value in nearest_frames)
            / cadence.frame_step
            * cadence.time_step_s
            if nearest_frames
            else 0.0
        )
        support_confidence = (
            min(item.confidence for item in synthetic_support)
            if synthetic_support
            else config.minimum_confidence
        )
        confidence = _clamp01(
            support_confidence / (1.0 + position_std + 0.5 * nearest_distance_s)
        )
        measurement_x = None
        measurement_z = None
        fusion_mode = "not_applicable"
        source_tracklets = sorted(
            {value for item in synthetic_support for value in item.source_tracklet_ids}
        )
        source_detections = sorted(
            {value for item in synthetic_support for value in item.source_detection_ids}
        )
        source_count = len(source_detections)
        spread = max((item.position_spread_m for item in synthetic_support), default=0.0)
        fusion_conflict = any(item.fusion_conflict for item in synthetic_support)

    support_before_age_s = (
        None
        if support_before is None
        else (frame_index - support_before) / cadence.frame_step * cadence.time_step_s
    )
    support_after_age_s = (
        None
        if support_after is None
        else (support_after - frame_index) / cadence.frame_step * cadence.time_step_s
    )

    return {
        "schema_version": 1,
        "trajectory_version": TRAJECTORY_VERSION,
        "actor_id": actor_id,
        "segment_id": segment_id,
        "frame_index": frame_index,
        "timestamp_s": timestamp_s,
        "x": float(state[0]),
        "z": float(state[1]),
        "measurement_x": measurement_x,
        "measurement_z": measurement_z,
        "confidence": confidence,
        "origin": origin,
        "fusion_mode": fusion_mode,
        "source_tracklet_ids": source_tracklets,
        "source_detection_ids": source_detections,
        "source_count": source_count,
        "position_spread_m": spread,
        "position_std_m": position_std,
        "vx_mps": float(state[2]),
        "vz_mps": float(state[3]),
        "speed_mps": speed,
        "prediction_age_s": prediction_age_s,
        "support_before_frame_index": support_before,
        "support_after_frame_index": support_after,
        "support_before_age_s": support_before_age_s,
        "support_after_age_s": support_after_age_s,
        "support_mode": (
            "direct"
            if measurement is not None
            else "two_sided"
            if support_before is not None and support_after is not None
            else "one_sided"
        ),
        "fusion_conflict": fusion_conflict,
        "synthetic": origin in SYNTHETIC_ORIGINS,
        "quality": (
            "direct_rts"
            if origin in DIRECT_ORIGINS
            else "bounded_rts_interpolation"
            if origin == "interpolated"
            else "bounded_cv_edge_prediction"
        ),
        "config_max_position_std_m": config.max_position_std_m,
    }


def _internal_gap_events(
    *,
    actor_id: str,
    measurements: Mapping[int, _Measurement],
    series: _StateSeries,
    cadence: SamplingCadence,
    config: PersistentTrajectoryConfig,
) -> tuple[dict[int, str], list[_GapEvent]]:
    support = sorted(measurements)
    synthetic_origins: dict[int, str] = {}
    events: list[_GapEvent] = []
    state_index = {frame: index for index, frame in enumerate(series.frames)}

    for left_frame, right_frame in zip(support, support[1:]):
        missing_frames = list(range(left_frame + cadence.frame_step, right_frame, cadence.frame_step))
        if not missing_frames:
            continue
        left = measurements[left_frame]
        right = measurements[right_frame]
        duration = right.timestamp_s - left.timestamp_s
        distance = math.hypot(right.x - left.x, right.z - left.z)
        speed = distance / duration if duration > 0 else math.inf
        event = _GapEvent(
            actor_id=actor_id,
            gap_kind="internal",
            start_frame_index=missing_frames[0],
            end_frame_index=missing_frames[-1],
            missing_frame_count=len(missing_frames),
            duration_s=duration,
            action="preserved",
            reason="",
            left_support_frame_index=left_frame,
            right_support_frame_index=right_frame,
            endpoint_distance_m=distance,
            endpoint_speed_mps=speed,
        )
        if duration > config.max_internal_gap_s + config.cadence_tolerance_s:
            event.reason = "internal_gap_too_long"
        elif speed > config.max_internal_endpoint_speed_mps:
            event.reason = "endpoint_speed_too_high"
        else:
            candidates = []
            for frame in missing_frames:
                index = state_index[frame]
                if _position_std(series.covariances[index]) <= config.max_position_std_m:
                    candidates.append(frame)
            if len(candidates) != len(missing_frames):
                event.reason = "position_uncertainty_too_high"
            else:
                event.action = "interpolated"
                event.reason = "bounded_two_sided_support"
                event.inserted_frame_indices.extend(missing_frames)
                for frame in missing_frames:
                    synthetic_origins[frame] = "interpolated"
        events.append(event)
    return synthetic_origins, events


def _edge_predictions(
    *,
    actor_id: str,
    measurements: Mapping[int, _Measurement],
    series: _StateSeries,
    cadence: SamplingCadence,
    config: PersistentTrajectoryConfig,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray, float]], list[_GapEvent]]:
    support = sorted(measurements)
    result: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}
    events: list[_GapEvent] = []
    if len(support) < config.minimum_edge_support_points:
        # Record unfilled edges for diagnostics but avoid extrapolating a
        # velocity from one/two points.
        for kind, start, end in (
            ("leading", cadence.first_frame_index, support[0] - cadence.frame_step),
            ("trailing", support[-1] + cadence.frame_step, cadence.last_frame_index),
        ):
            if start <= end:
                missing = list(range(start, end + cadence.frame_step, cadence.frame_step))
                events.append(
                    _GapEvent(
                        actor_id=actor_id,
                        gap_kind=kind,
                        start_frame_index=missing[0],
                        end_frame_index=missing[-1],
                        missing_frame_count=len(missing),
                        duration_s=len(missing) * cadence.time_step_s,
                        action="preserved",
                        reason="insufficient_edge_support",
                        left_support_frame_index=(None if kind == "leading" else support[-1]),
                        right_support_frame_index=(support[0] if kind == "leading" else None),
                    )
                )
        return result, events

    first_frame = series.frames[0]
    last_frame = series.frames[-1]
    first_state = series.states[0]
    first_cov = series.covariances[0]
    last_state = series.states[-1]
    last_cov = series.covariances[-1]

    def extrapolate(kind: str) -> None:
        if kind == "leading":
            all_missing = list(
                range(cadence.first_frame_index, first_frame, cadence.frame_step)
            )
            all_missing.reverse()  # nearest support first
            base_frame, base_state, base_cov = first_frame, first_state, first_cov
            direction = -1.0
            left_support, right_support = None, first_frame
        else:
            all_missing = list(
                range(last_frame + cadence.frame_step, cadence.last_frame_index + 1, cadence.frame_step)
            )
            base_frame, base_state, base_cov = last_frame, last_state, last_cov
            direction = 1.0
            left_support, right_support = last_frame, None

        if not all_missing:
            return
        inserted: list[int] = []
        stop_reason = "edge_prediction_limit"
        state = base_state.copy()
        cov = base_cov.copy()
        previous_frame = base_frame
        for frame in all_missing:
            age = abs(frame - base_frame) / cadence.frame_step * cadence.time_step_s
            if age > config.max_edge_prediction_s + config.cadence_tolerance_s:
                stop_reason = "edge_prediction_limit"
                break
            dt = direction * (
                abs(frame - previous_frame) / cadence.frame_step * cadence.time_step_s
            )
            F, Q = _transition(dt, config.process_acceleration_sigma_mps2)
            state = F @ state
            # Q is symmetric in dt except cross terms generated by noise_map;
            # for backward extrapolation we still want positive process growth.
            _, Q_abs = _transition(abs(dt), config.process_acceleration_sigma_mps2)
            cov = F @ cov @ F.T + Q_abs
            if _position_std(cov) > config.max_position_std_m:
                stop_reason = "position_uncertainty_too_high"
                break
            result[frame] = (state.copy(), cov.copy(), age)
            inserted.append(frame)
            previous_frame = frame

        missing_sorted = sorted(all_missing)
        events.append(
            _GapEvent(
                actor_id=actor_id,
                gap_kind=kind,
                start_frame_index=missing_sorted[0],
                end_frame_index=missing_sorted[-1],
                missing_frame_count=len(missing_sorted),
                duration_s=len(missing_sorted) * cadence.time_step_s,
                action=(
                    "predicted"
                    if len(inserted) == len(all_missing)
                    else "partially_predicted"
                    if inserted
                    else "preserved"
                ),
                reason=("bounded_edge_prediction" if len(inserted) == len(all_missing) else stop_reason),
                inserted_frame_indices=sorted(inserted),
                left_support_frame_index=left_support,
                right_support_frame_index=right_support,
            )
        )

    extrapolate("leading")
    extrapolate("trailing")
    return result, events


def _build_actor_trajectory(
    *,
    actor_row: Mapping[str, Any],
    measurements: Mapping[int, _Measurement],
    cadence: SamplingCadence,
    config: PersistentTrajectoryConfig,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    actor_id = str(actor_row["actor_id"])
    series = _filter_and_rts(
        measurements=measurements,
        cadence=cadence,
        config=config,
    )
    state_index = {frame: index for index, frame in enumerate(series.frames)}
    synthetic_internal, internal_events = _internal_gap_events(
        actor_id=actor_id,
        measurements=measurements,
        series=series,
        cadence=cadence,
        config=config,
    )
    edge_states, edge_events = _edge_predictions(
        actor_id=actor_id,
        measurements=measurements,
        series=series,
        cadence=cadence,
        config=config,
    )

    emitted_frames = set(measurements) | set(synthetic_internal) | set(edge_states)
    ordered_frames = sorted(emitted_frames)
    support_frames = sorted(measurements)
    points: list[dict[str, Any]] = []
    segment_index = 0
    previous_frame: int | None = None

    for frame in ordered_frames:
        if previous_frame is None or frame - previous_frame != cadence.frame_step:
            segment_index += 1
        segment_id = f"{actor_id}:segment-{segment_index:03d}"
        measurement = measurements.get(frame)
        if measurement is not None:
            index = state_index[frame]
            state = series.states[index]
            covariance = series.covariances[index]
            origin = measurement.origin
            support_before, support_after = _support_neighbors(frame, support_frames)
            prediction_age = 0.0
        elif frame in synthetic_internal:
            index = state_index[frame]
            state = series.states[index]
            covariance = series.covariances[index]
            origin = "interpolated"
            support_before, support_after = _support_neighbors(frame, support_frames)
            prediction_age = 0.0
        else:
            state, covariance, prediction_age = edge_states[frame]
            origin = "predicted"
            support_before, support_after = _support_neighbors(frame, support_frames)

        points.append(
            _point_payload(
                actor_id=actor_id,
                frame_index=frame,
                timestamp_s=(measurement.timestamp_s if measurement is not None else cadence.timestamp_for_frame(frame)),
                state=state,
                covariance=covariance,
                measurement=measurement,
                origin=origin,
                segment_id=segment_id,
                support_before=support_before,
                support_after=support_after,
                prediction_age_s=prediction_age,
                synthetic_support=(
                    ()
                    if measurement is not None
                    else tuple(
                        measurements[value]
                        for value in (support_before, support_after)
                        if value is not None
                    )
                ),
                cadence=cadence,
                config=config,
            )
        )
        previous_frame = frame

    counts = collections.Counter(point["origin"] for point in points)
    direct_count = counts["observed"] + counts["fused"]
    synthetic_count = counts["interpolated"] + counts["predicted"]
    max_std = max(float(point["position_std_m"]) for point in points)
    max_prediction_age = max(float(point["prediction_age_s"]) for point in points)
    max_speed = max(float(point["speed_mps"]) for point in points)

    trajectory = {
        "schema_version": 1,
        "trajectory_version": TRAJECTORY_VERSION,
        "actor_id": actor_id,
        "anchor_frame_index": actor_row.get("anchor_frame_index"),
        "anchor_identity_id": actor_row.get("anchor_identity_id"),
        "tracklet_ids": list(actor_row.get("tracklet_ids", [])),
        "source_identity_ids": list(actor_row.get("source_identity_ids", [])),
        "point_count": len(points),
        "direct_point_count": direct_count,
        "observed_point_count": counts["observed"],
        "fused_point_count": counts["fused"],
        "interpolated_point_count": counts["interpolated"],
        "predicted_point_count": counts["predicted"],
        "synthetic_point_count": synthetic_count,
        "segment_count": segment_index,
        "represented_frame_fraction": len(points) / len(cadence.frames),
        "max_position_std_m": max_std,
        "max_prediction_age_s": max_prediction_age,
        "max_speed_mps": max_speed,
        "provisional": True,
        "canonical": False,
        "points": points,
    }
    events = tuple(event.to_dict() for event in (*internal_events, *edge_events))
    return trajectory, events


def _occupancy_rows(
    trajectories: Sequence[Mapping[str, Any]],
    cadence: SamplingCadence,
) -> tuple[dict[str, Any], ...]:
    by_frame: dict[int, dict[str, Mapping[str, Any]]] = collections.defaultdict(dict)
    actor_ids = sorted(str(row["actor_id"]) for row in trajectories)
    for trajectory in trajectories:
        actor_id = str(trajectory["actor_id"])
        for point in trajectory["points"]:
            frame = int(point["frame_index"])
            if actor_id in by_frame[frame]:
                raise ValueError(f"duplicate actor/frame output: {actor_id} frame {frame}")
            by_frame[frame][actor_id] = point

    rows = []
    for frame in cadence.frames:
        points = by_frame.get(frame, {})
        origins = collections.Counter(str(point["origin"]) for point in points.values())
        represented = sorted(points)
        missing = sorted(set(actor_ids) - set(represented))
        rows.append(
            {
                "schema_version": 1,
                "trajectory_version": TRAJECTORY_VERSION,
                "frame_index": frame,
                "timestamp_s": cadence.timestamp_for_frame(frame),
                "represented_actor_count": len(represented),
                "direct_actor_count": origins["observed"] + origins["fused"],
                "observed_actor_count": origins["observed"],
                "fused_actor_count": origins["fused"],
                "interpolated_actor_count": origins["interpolated"],
                "predicted_actor_count": origins["predicted"],
                "synthetic_actor_count": origins["interpolated"] + origins["predicted"],
                "missing_actor_count": len(missing),
                "actor_ids": represented,
                "missing_actor_ids": missing,
            }
        )
    return tuple(rows)


def _disappearance_event_count(
    occupancy_by_actor: Mapping[str, set[int]],
    cadence: SamplingCadence,
) -> int:
    count = 0
    for frames in occupancy_by_actor.values():
        ordered = sorted(frames)
        for left, right in zip(ordered, ordered[1:]):
            if right - left > cadence.frame_step:
                count += 1
    return count


def build_persistent_actor_trajectories(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
    assignment_summary: Mapping[str, Any] | None = None,
    config: PersistentTrajectoryConfig | None = None,
) -> dict[str, Any]:
    """Build serialization-ready 6C trajectories and diagnostics."""

    config = config or PersistentTrajectoryConfig()
    actors = _load_actors(actor_rows)
    tracklets = _load_tracklets(tracklet_rows)
    assignment = _load_assignment(
        assignment_rows,
        known_tracklets=set(tracklets),
        known_actors=set(actors),
    )
    if assignment_summary is not None:
        if assignment_summary.get("solver_version") not in {None, ASSIGNMENT_SOLVER_VERSION}:
            raise ValueError("6B summary solver version mismatch")
        expected_actor_count = assignment_summary.get("actor_count")
        if expected_actor_count is not None and int(expected_actor_count) != len(actors):
            raise ValueError("6B actor count does not match actors.jsonl")
        violations = assignment_summary.get("hard_constraint_violation_count")
        if violations not in {None, 0}:
            raise ValueError("6C refuses 6B assignments with hard constraint violations")

    # Make the actor row contract and assignment contract agree exactly.
    assigned_by_actor: dict[str, set[str]] = {actor_id: set() for actor_id in actors}
    for tracklet_id, actor_id in assignment.items():
        if actor_id is not None:
            assigned_by_actor[actor_id].add(tracklet_id)
    for actor_id, row in actors.items():
        declared = set(str(value) for value in row.get("tracklet_ids", []))
        if declared and declared != assigned_by_actor[actor_id]:
            raise ValueError(
                f"{actor_id} actors.jsonl membership differs from assignments.jsonl"
            )

    cadence = infer_sampling_cadence(tracklets)
    measurements = _actor_measurements(
        actors=actors,
        assignment=assignment,
        tracklets=tracklets,
        config=config,
    )

    trajectories: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    for actor_id in sorted(actors):
        trajectory, actor_gaps = _build_actor_trajectory(
            actor_row=actors[actor_id],
            measurements=measurements[actor_id],
            cadence=cadence,
            config=config,
        )
        trajectories.append(trajectory)
        gaps.extend(actor_gaps)

    occupancy = _occupancy_rows(trajectories, cadence)
    actor_count = len(actors)
    frame_count = len(cadence.frames)
    total_possible = actor_count * frame_count
    direct_actor_frames = sum(int(row["direct_actor_count"]) for row in occupancy)
    represented_actor_frames = sum(int(row["represented_actor_count"]) for row in occupancy)
    synthetic_actor_frames = represented_actor_frames - direct_actor_frames

    before_frames: dict[str, set[int]] = {
        actor_id: set(measurements[actor_id]) for actor_id in actors
    }
    after_frames: dict[str, set[int]] = {
        str(row["actor_id"]): {int(point["frame_index"]) for point in row["points"]}
        for row in trajectories
    }
    origin_counts = collections.Counter(
        str(point["origin"])
        for trajectory in trajectories
        for point in trajectory["points"]
    )
    represented_counts = [float(row["represented_actor_count"]) for row in occupancy]
    direct_counts = [float(row["direct_actor_count"]) for row in occupancy]
    position_stds = [
        float(point["position_std_m"])
        for trajectory in trajectories
        for point in trajectory["points"]
    ]
    prediction_ages = [
        float(point["prediction_age_s"])
        for trajectory in trajectories
        for point in trajectory["points"]
        if point["origin"] == "predicted"
    ]
    fusion_conflict_points = sum(
        bool(point["fusion_conflict"])
        for trajectory in trajectories
        for point in trajectory["points"]
        if point["origin"] == "fused"
    )

    full_population_frames = sum(
        int(row["represented_actor_count"]) == actor_count for row in occupancy
    )
    full_direct_frames = sum(int(row["direct_actor_count"]) == actor_count for row in occupancy)
    at_least_minus_one = sum(
        int(row["represented_actor_count"]) >= max(0, actor_count - 1) for row in occupancy
    )
    at_least_minus_two = sum(
        int(row["represented_actor_count"]) >= max(0, actor_count - 2) for row in occupancy
    )
    gap_actions = collections.Counter(str(row["action"]) for row in gaps)
    gap_kinds = collections.Counter(str(row["gap_kind"]) for row in gaps)

    summary = {
        "schema_version": 1,
        "trajectory_version": TRAJECTORY_VERSION,
        "assignment_solver_version": ASSIGNMENT_SOLVER_VERSION,
        "canonical": False,
        "provisional": True,
        "tracking_v1_unchanged": True,
        "actor_count": actor_count,
        "frame_count": frame_count,
        "cadence": cadence.to_dict(),
        "total_possible_actor_frames": total_possible,
        "direct_actor_frame_count": direct_actor_frames,
        "represented_actor_frame_count": represented_actor_frames,
        "synthetic_actor_frame_count": synthetic_actor_frames,
        "direct_coverage_fraction": (
            direct_actor_frames / total_possible if total_possible else 0.0
        ),
        "represented_coverage_fraction": (
            represented_actor_frames / total_possible if total_possible else 0.0
        ),
        "observed_point_count": origin_counts["observed"],
        "fused_point_count": origin_counts["fused"],
        "interpolated_point_count": origin_counts["interpolated"],
        "predicted_point_count": origin_counts["predicted"],
        "fusion_conflict_point_count": fusion_conflict_points,
        "full_population_frame_count_before": full_direct_frames,
        "full_population_frame_fraction_before": full_direct_frames / frame_count,
        "full_population_frame_count_after": full_population_frames,
        "full_population_frame_fraction_after": full_population_frames / frame_count,
        "at_least_population_minus_one_frame_fraction_after": at_least_minus_one / frame_count,
        "at_least_population_minus_two_frame_fraction_after": at_least_minus_two / frame_count,
        "represented_actor_count": _distribution(represented_counts),
        "direct_actor_count": _distribution(direct_counts),
        "position_std_m": _distribution(position_stds),
        "prediction_age_s": _distribution(prediction_ages),
        "disappearance_event_count_before": _disappearance_event_count(before_frames, cadence),
        "disappearance_event_count_after": _disappearance_event_count(after_frames, cadence),
        "gap_count": len(gaps),
        "gap_action_counts": dict(sorted(gap_actions.items())),
        "gap_kind_counts": dict(sorted(gap_kinds.items())),
        "preserved_gap_count": sum(
            row["action"] in {"preserved", "partially_predicted"} for row in gaps
        ),
        "actor_segment_count": _distribution(
            [float(row["segment_count"]) for row in trajectories]
        ),
        "actor_coverage_fraction": _distribution(
            [float(row["represented_frame_fraction"]) for row in trajectories]
        ),
        "config": config.to_dict(),
    }

    return {
        "trajectories": trajectories,
        "occupancy": list(occupancy),
        "gaps": gaps,
        "summary": summary,
    }
