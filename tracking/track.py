"""Single-object track state built on top of the metric Kalman filter.

This module owns the temporal state of one anonymous person track.  It does not
decide which observation belongs to which track; that is the job of
tracking.association.

V1 deliberately keeps lifecycle semantics small:
- OBSERVED: the track was corrected by a detector observation this frame.
- PREDICTED: no observation was assigned this frame; the Kalman prediction is
  kept so the 2D field does not blink.

Track creation/deletion/reliability/max-visible-slots come in a later module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any

import numpy as np

from tracking.kalman import (
    FieldKalmanFilter,
    KalmanConfig,
    KalmanPrediction,
    KalmanUpdate,
)


class TrackFrameState(str, Enum):
    OBSERVED = "observed"
    PREDICTED = "predicted"


@dataclass(frozen=True)
class TrackSnapshot:
    """Immutable diagnostic snapshot of one track at one frame."""

    track_id: int
    frame: int
    time_s: float
    frame_state: str

    x: float
    z: float
    vx: float
    vz: float
    speed_mps: float

    age_frames: int
    hits: int
    misses: int
    consecutive_misses: int

    last_observed_frame: int
    last_observed_time_s: float


class Track:
    """Anonymous metric field track."""

    def __init__(
        self,
        *,
        track_id: int,
        initial_observation: Any,
        kalman_config: KalmanConfig | None = None,
    ) -> None:
        self.track_id = int(track_id)
        if self.track_id < 0:
            raise ValueError("track_id must be >= 0")

        frame = int(initial_observation.frame)
        time_s = float(initial_observation.time_s)
        if not math.isfinite(time_s):
            raise ValueError("initial observation time must be finite")

        position = _observation_position(initial_observation)
        measurement_covariance = _observation_covariance(initial_observation)

        self.kalman = FieldKalmanFilter.from_measurement(
            position,
            measurement_covariance,
            config=kalman_config,
        )

        self.birth_frame = frame
        self.birth_time_s = time_s

        self.current_frame = frame
        self.current_time_s = time_s

        self.last_observed_frame = frame
        self.last_observed_time_s = time_s
        self.last_observed_position = position.copy()
        self.last_observation = initial_observation

        self.age_frames = 1
        self.hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        self.frame_state = TrackFrameState.OBSERVED

        self._prediction_ready = False
        self._last_prediction: KalmanPrediction | None = None
        self._last_update: KalmanUpdate | None = None

    @property
    def position(self) -> np.ndarray:
        return self.kalman.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kalman.velocity

    @property
    def covariance(self) -> np.ndarray:
        return self.kalman.covariance

    @property
    def speed_mps(self) -> float:
        return self.kalman.speed_mps

    @property
    def predicted_position(self) -> np.ndarray:
        """Current predicted position after predict_to().

        Before association this is the position all candidates are compared to.
        """
        return self.position

    @property
    def seconds_since_observation(self) -> float:
        return max(0.0, self.current_time_s - self.last_observed_time_s)

    @property
    def frames_since_observation(self) -> int:
        return max(0, self.current_frame - self.last_observed_frame)

    @property
    def last_prediction(self) -> KalmanPrediction | None:
        return self._last_prediction

    @property
    def last_update(self) -> KalmanUpdate | None:
        return self._last_update

    def predict_to(self, *, frame: int, time_s: float) -> KalmanPrediction:
        """Advance exactly once to a later frame/time before association."""

        frame = int(frame)
        time_s = float(time_s)

        if frame <= self.current_frame:
            raise ValueError(
                f"track {self.track_id}: frame must advance "
                f"({frame} <= {self.current_frame})"
            )
        if not math.isfinite(time_s) or time_s <= self.current_time_s:
            raise ValueError(
                f"track {self.track_id}: time_s must advance "
                f"({time_s} <= {self.current_time_s})"
            )
        if self._prediction_ready:
            raise RuntimeError(
                f"track {self.track_id}: previous prediction has not been "
                "resolved by update() or mark_missed()"
            )

        dt = time_s - self.current_time_s
        prediction = self.kalman.predict(dt)

        self.age_frames += frame - self.current_frame
        self.current_frame = frame
        self.current_time_s = time_s
        self.frame_state = TrackFrameState.PREDICTED

        self._prediction_ready = True
        self._last_prediction = prediction
        self._last_update = None
        return prediction

    def update(self, observation: Any) -> KalmanUpdate:
        """Correct the current prediction using its assigned observation."""

        self._require_pending_prediction()

        if int(observation.frame) != self.current_frame:
            raise ValueError(
                f"track {self.track_id}: observation frame "
                f"{observation.frame} != predicted frame {self.current_frame}"
            )
        if not math.isclose(
            float(observation.time_s),
            self.current_time_s,
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise ValueError(
                f"track {self.track_id}: observation time "
                f"{observation.time_s} != predicted time {self.current_time_s}"
            )

        position = _observation_position(observation)
        measurement_covariance = _observation_covariance(observation)

        result = self.kalman.update(
            position,
            measurement_covariance,
        )

        self.hits += 1
        self.consecutive_misses = 0
        self.last_observed_frame = self.current_frame
        self.last_observed_time_s = self.current_time_s
        self.last_observed_position = position.copy()
        self.last_observation = observation
        self.frame_state = TrackFrameState.OBSERVED

        self._prediction_ready = False
        self._last_update = result
        return result

    def mark_missed(self) -> None:
        """Keep the prediction for this frame when no observation was assigned."""

        self._require_pending_prediction()
        self.misses += 1
        self.consecutive_misses += 1
        self.frame_state = TrackFrameState.PREDICTED
        self._prediction_ready = False
        self._last_update = None

    def mahalanobis_squared(self, observation: Any) -> float:
        """Statistical candidate distance from current predicted state."""

        self._require_pending_prediction()
        return self.kalman.mahalanobis_squared(
            _observation_position(observation),
            _observation_covariance(observation),
        )

    def snapshot(self) -> TrackSnapshot:
        position = self.position
        velocity = self.velocity
        return TrackSnapshot(
            track_id=self.track_id,
            frame=self.current_frame,
            time_s=self.current_time_s,
            frame_state=self.frame_state.value,
            x=float(position[0]),
            z=float(position[1]),
            vx=float(velocity[0]),
            vz=float(velocity[1]),
            speed_mps=self.speed_mps,
            age_frames=self.age_frames,
            hits=self.hits,
            misses=self.misses,
            consecutive_misses=self.consecutive_misses,
            last_observed_frame=self.last_observed_frame,
            last_observed_time_s=self.last_observed_time_s,
        )

    def _require_pending_prediction(self) -> None:
        if not self._prediction_ready:
            raise RuntimeError(
                f"track {self.track_id}: call predict_to() before association/"
                "update for a new frame"
            )


def _observation_position(observation: Any) -> np.ndarray:
    if hasattr(observation, "world_position"):
        value = observation.world_position
    else:
        value = (observation.world_x, observation.world_z)

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all():
        raise ValueError("observation world position must be a finite 2-vector")
    return result


def _observation_covariance(observation: Any) -> np.ndarray:
    if hasattr(observation, "measurement_covariance"):
        value = observation.measurement_covariance
    elif hasattr(observation, "quality"):
        value = observation.quality.measurement_covariance
    else:
        raise TypeError("observation does not expose measurement covariance")

    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 2) or not np.isfinite(result).all():
        raise ValueError("observation covariance must be finite 2x2")
    return result


__all__ = [
    "Track",
    "TrackFrameState",
    "TrackSnapshot",
]
