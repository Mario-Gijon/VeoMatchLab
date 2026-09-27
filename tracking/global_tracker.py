from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from .kalman import ConstantVelocityKalman
from .models import (
    PositionKind,
    TrackLifecycle,
    TrackSample,
    TrackSnapshot,
    TrackerConfig,
    TrackingFrame,
    TrackingMeasurement,
)


_LARGE_COST = 1e9


@dataclass(slots=True)
class _Track:
    track_id: int
    kalman: ConstantVelocityKalman
    lifecycle: TrackLifecycle
    birth_timestamp: float
    last_timestamp: float
    last_observed_timestamp: float
    last_observed_position: np.ndarray
    hits: int = 1
    history: list[TrackSample] = field(default_factory=list)

    @property
    def missed_seconds(self) -> float:
        return max(0.0, self.last_timestamp - self.last_observed_timestamp)

    @property
    def age_seconds(self) -> float:
        return max(0.0, self.last_timestamp - self.birth_timestamp)


class GlobalTracker:
    """
    Tracking V1 for globally-associated football-player measurements.

    Important design choices:
    - Input is already same-frame deduplicated.
    - Tracking happens in field coordinates, not per virtual camera.
    - Track IDs are monotonically increasing and never reused.
    - Confirmed tracks may coast without observations for several seconds.
    - Every returned position explicitly says whether it was OBSERVED or
      PREDICTED. INTERPOLATED is reserved for the later offline refinement pass.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        self.config = config or TrackerConfig()
        self._tracks: dict[int, _Track] = {}
        self._next_track_id = 1
        self._last_timestamp: float | None = None
        self._last_frame_index: int | None = None

    @property
    def next_track_id(self) -> int:
        return self._next_track_id

    def update(
        self,
        measurements: Iterable[TrackingMeasurement],
        *,
        timestamp: float,
        frame_index: int,
    ) -> TrackingFrame:
        measurements = tuple(measurements)
        self._validate_frame(measurements, timestamp, frame_index)

        live_tracks = [
            track
            for track in self._tracks.values()
            if track.lifecycle is not TrackLifecycle.LOST
        ]

        # Predict every living track to the current timestamp.
        for track in live_tracks:
            dt = timestamp - track.last_timestamp
            track.kalman.predict(dt)
            track.last_timestamp = timestamp

        # Expire tracks before association. A track may be reacquired at exactly
        # the configured coast limit, but not after silently exceeding it.
        for track in live_tracks:
            if (
                track.lifecycle is TrackLifecycle.TENTATIVE
                and track.missed_seconds
                > self.config.tentative_max_missed_seconds
            ):
                track.lifecycle = TrackLifecycle.LOST
            elif (
                track.lifecycle
                in (TrackLifecycle.ACTIVE, TrackLifecycle.COASTING)
                and track.missed_seconds > self.config.max_coast_seconds
            ):
                track.lifecycle = TrackLifecycle.LOST

        candidate_tracks = [
            track
            for track in live_tracks
            if track.lifecycle is not TrackLifecycle.LOST
        ]

        matched_indices, unmatched_track_indices, unmatched_measurement_indices = (
            self._associate(candidate_tracks, measurements)
        )

        matched_pairs: list[tuple[int, Hashable]] = []
        unmatched_track_ids: list[int] = []

        # 1) Apply real observations to matched tracks.
        for track_idx, measurement_idx in matched_indices:
            track = candidate_tracks[track_idx]
            measurement = measurements[measurement_idx]
            covariance = self._measurement_covariance(measurement)

            track.kalman.update(
                np.array([measurement.x, measurement.z], dtype=np.float64),
                covariance,
            )
            track.kalman.clamp_speed(self.config.max_speed_m_s)
            track.last_observed_timestamp = timestamp
            track.last_observed_position = np.array(
                [measurement.x, measurement.z],
                dtype=np.float64,
            )
            track.hits += 1

            if (
                track.lifecycle is TrackLifecycle.TENTATIVE
                and track.hits >= self.config.min_confirm_hits
            ):
                track.lifecycle = TrackLifecycle.ACTIVE
            elif track.lifecycle in (
                TrackLifecycle.ACTIVE,
                TrackLifecycle.COASTING,
            ):
                track.lifecycle = TrackLifecycle.ACTIVE

            sample = self._make_sample(
                track,
                frame_index=frame_index,
                position_kind=PositionKind.OBSERVED,
                measurement_id=measurement.measurement_id,
                confidence=measurement.confidence,
            )
            track.history.append(sample)
            matched_pairs.append((track.track_id, measurement.measurement_id))

        # 2) Age unmatched tracks without inventing observations.
        for track_idx in unmatched_track_indices:
            track = candidate_tracks[track_idx]
            missed = track.missed_seconds

            if track.lifecycle is TrackLifecycle.TENTATIVE:
                if missed > self.config.tentative_max_missed_seconds:
                    track.lifecycle = TrackLifecycle.LOST
                else:
                    track.history.append(
                        self._make_sample(
                            track,
                            frame_index=frame_index,
                            position_kind=PositionKind.PREDICTED,
                            measurement_id=None,
                            confidence=None,
                        )
                    )
            else:
                if missed > self.config.max_coast_seconds:
                    track.lifecycle = TrackLifecycle.LOST
                else:
                    track.lifecycle = TrackLifecycle.COASTING
                    track.history.append(
                        self._make_sample(
                            track,
                            frame_index=frame_index,
                            position_kind=PositionKind.PREDICTED,
                            measurement_id=None,
                            confidence=None,
                        )
                    )

            unmatched_track_ids.append(track.track_id)

        # 3) Start tentative tracks for genuinely unmatched measurements.
        created_track_ids: list[int] = []
        for measurement_idx in unmatched_measurement_indices:
            measurement = measurements[measurement_idx]
            track = self._new_track(measurement)
            self._tracks[track.track_id] = track
            created_track_ids.append(track.track_id)

        self._last_timestamp = timestamp
        self._last_frame_index = frame_index

        # Return all tracks that still exist at this frame. LOST tracks are kept
        # internally for audit/history but are not rendered as current objects.
        snapshots = [
            self._make_snapshot(track, frame_index)
            for track in self._tracks.values()
            if track.lifecycle is not TrackLifecycle.LOST
            and np.isclose(track.last_timestamp, timestamp)
        ]
        snapshots.sort(key=lambda snapshot: snapshot.track_id)

        unmatched_measurement_ids = tuple(
            measurements[idx].measurement_id
            for idx in unmatched_measurement_indices
        )

        # New tentative tracks are also unmatched measurements by definition;
        # their IDs remain discoverable through snapshots.
        _ = created_track_ids

        return TrackingFrame(
            frame_index=frame_index,
            timestamp=timestamp,
            tracks=tuple(snapshots),
            matched_pairs=tuple(matched_pairs),
            unmatched_track_ids=tuple(unmatched_track_ids),
            unmatched_measurement_ids=unmatched_measurement_ids,
        )

    def trajectories(
        self,
        *,
        include_lost: bool = True,
    ) -> dict[int, tuple[TrackSample, ...]]:
        """Return a copy of the accumulated forward-pass trajectories."""
        output: dict[int, tuple[TrackSample, ...]] = {}
        for track_id, track in self._tracks.items():
            if not include_lost and track.lifecycle is TrackLifecycle.LOST:
                continue
            output[track_id] = tuple(track.history)
        return output

    def track_lifecycle(self, track_id: int) -> TrackLifecycle:
        return self._tracks[track_id].lifecycle

    def reset(self) -> None:
        self._tracks.clear()
        self._next_track_id = 1
        self._last_timestamp = None
        self._last_frame_index = None

    def _associate(
        self,
        tracks: list[_Track],
        measurements: tuple[TrackingMeasurement, ...],
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        if not tracks or not measurements:
            return (
                [],
                list(range(len(tracks))),
                list(range(len(measurements))),
            )

        costs = np.full(
            (len(tracks), len(measurements)),
            np.inf,
            dtype=np.float64,
        )

        for track_idx, track in enumerate(tracks):
            predicted_position = track.kalman.position
            predicted_velocity = track.kalman.velocity
            hard_gate = min(
                self.config.max_reacquire_distance_m,
                self.config.base_match_distance_m
                + self.config.match_distance_growth_m_per_s
                * track.missed_seconds,
            )

            for measurement_idx, measurement in enumerate(measurements):
                measured_position = np.array(
                    [measurement.x, measurement.z],
                    dtype=np.float64,
                )
                delta = measured_position - predicted_position
                euclidean_distance = float(np.linalg.norm(delta))

                if euclidean_distance > hard_gate:
                    continue

                measurement_covariance = self._measurement_covariance(measurement)
                mahalanobis_squared = track.kalman.mahalanobis_squared(
                    measured_position,
                    measurement_covariance,
                )

                if mahalanobis_squared > self.config.mahalanobis_gate_squared:
                    continue

                mahalanobis_distance = float(np.sqrt(mahalanobis_squared))

                # A very small velocity-consistency term helps break ties without
                # making motion dominate the identity decision. It is computed
                # from the last REAL observation, never from the predicted point.
                dt_from_observation = (
                    track.last_timestamp - track.last_observed_timestamp
                )
                if dt_from_observation > 1e-3:
                    implied_velocity = (
                        measured_position - track.last_observed_position
                    ) / dt_from_observation
                    velocity_change = float(
                        np.linalg.norm(implied_velocity - predicted_velocity)
                    )
                else:
                    velocity_change = 0.0

                costs[track_idx, measurement_idx] = (
                    mahalanobis_distance
                    + self.config.euclidean_cost_weight * euclidean_distance
                    + self.config.velocity_change_cost_weight * velocity_change
                )

        if not np.isfinite(costs).any():
            return (
                [],
                list(range(len(tracks))),
                list(range(len(measurements))),
            )

        safe_costs = np.where(np.isfinite(costs), costs, _LARGE_COST)
        row_indices, column_indices = linear_sum_assignment(safe_costs)

        matched: list[tuple[int, int]] = []
        matched_track_indices: set[int] = set()
        matched_measurement_indices: set[int] = set()

        for track_idx, measurement_idx in zip(
            row_indices.tolist(),
            column_indices.tolist(),
        ):
            if not np.isfinite(costs[track_idx, measurement_idx]):
                continue
            matched.append((track_idx, measurement_idx))
            matched_track_indices.add(track_idx)
            matched_measurement_indices.add(measurement_idx)

        unmatched_tracks = [
            idx for idx in range(len(tracks)) if idx not in matched_track_indices
        ]
        unmatched_measurements = [
            idx
            for idx in range(len(measurements))
            if idx not in matched_measurement_indices
        ]

        return matched, unmatched_tracks, unmatched_measurements

    def _measurement_covariance(
        self,
        measurement: TrackingMeasurement,
    ) -> np.ndarray:
        if measurement.position_covariance is not None:
            return measurement.position_covariance

        confidence = max(
            measurement.confidence,
            self.config.min_confidence_for_noise_scaling,
        )
        scale = min(
            self.config.max_measurement_noise_scale,
            1.0 / np.sqrt(confidence),
        )
        std = self.config.default_measurement_std * scale
        return np.diag([std * std, std * std]).astype(np.float64)

    def _new_track(self, measurement: TrackingMeasurement) -> _Track:
        track_id = self._next_track_id
        self._next_track_id += 1

        kalman = ConstantVelocityKalman(
            x=measurement.x,
            z=measurement.z,
            process_acceleration_std=self.config.process_acceleration_std,
            initial_position_std=self.config.initial_position_std,
            initial_velocity_std=self.config.initial_velocity_std,
        )

        track = _Track(
            track_id=track_id,
            kalman=kalman,
            lifecycle=TrackLifecycle.TENTATIVE,
            birth_timestamp=measurement.timestamp,
            last_timestamp=measurement.timestamp,
            last_observed_timestamp=measurement.timestamp,
            last_observed_position=np.array(
                [measurement.x, measurement.z],
                dtype=np.float64,
            ),
        )
        track.history.append(
            self._make_sample(
                track,
                frame_index=measurement.frame_index,
                position_kind=PositionKind.OBSERVED,
                measurement_id=measurement.measurement_id,
                confidence=measurement.confidence,
            )
        )
        return track

    def _make_sample(
        self,
        track: _Track,
        *,
        frame_index: int,
        position_kind: PositionKind,
        measurement_id: Hashable | None,
        confidence: float | None,
    ) -> TrackSample:
        state = track.kalman.state
        return TrackSample(
            track_id=track.track_id,
            frame_index=frame_index,
            timestamp=track.last_timestamp,
            x=float(state[0]),
            z=float(state[1]),
            vx=float(state[2]),
            vz=float(state[3]),
            lifecycle=track.lifecycle,
            position_kind=position_kind,
            measurement_id=measurement_id,
            confidence=confidence,
        )

    def _make_snapshot(
        self,
        track: _Track,
        frame_index: int,
    ) -> TrackSnapshot:
        state = track.kalman.state
        last_sample = track.history[-1] if track.history else None

        position_kind = (
            last_sample.position_kind
            if last_sample is not None
            and last_sample.frame_index == frame_index
            else PositionKind.PREDICTED
        )
        measurement_id = (
            last_sample.measurement_id
            if last_sample is not None
            and last_sample.frame_index == frame_index
            else None
        )
        confidence = (
            last_sample.confidence
            if last_sample is not None
            and last_sample.frame_index == frame_index
            else None
        )

        return TrackSnapshot(
            track_id=track.track_id,
            frame_index=frame_index,
            timestamp=track.last_timestamp,
            x=float(state[0]),
            z=float(state[1]),
            vx=float(state[2]),
            vz=float(state[3]),
            lifecycle=track.lifecycle,
            position_kind=position_kind,
            measurement_id=measurement_id,
            confidence=confidence,
            age_seconds=track.age_seconds,
            missed_seconds=track.missed_seconds,
            hits=track.hits,
        )

    def _validate_frame(
        self,
        measurements: tuple[TrackingMeasurement, ...],
        timestamp: float,
        frame_index: int,
    ) -> None:
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if frame_index < 0:
            raise ValueError("frame_index must be >= 0")

        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError(
                "GlobalTracker.update requires non-decreasing timestamps"
            )
        if (
            self._last_frame_index is not None
            and frame_index < self._last_frame_index
        ):
            raise ValueError(
                "GlobalTracker.update requires non-decreasing frame indices"
            )

        measurement_ids: set[Hashable] = set()
        for measurement in measurements:
            if not np.isclose(measurement.timestamp, timestamp):
                raise ValueError(
                    "All measurements must have the same timestamp passed "
                    "to GlobalTracker.update"
                )
            if measurement.frame_index != frame_index:
                raise ValueError(
                    "All measurements must have the same frame_index passed "
                    "to GlobalTracker.update"
                )
            if measurement.measurement_id in measurement_ids:
                raise ValueError(
                    f"Duplicate measurement_id: {measurement.measurement_id!r}"
                )
            measurement_ids.add(measurement.measurement_id)
