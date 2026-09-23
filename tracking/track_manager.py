"""Lifecycle, reliability and visible-slot management for field tracks.

This module sits above:
- tracking.track
- tracking.association

It intentionally does NOT know about teams, shirt numbers, player identity,
appearance embeddings or detector internals.

V1 goals:
- one-frame detections do not immediately appear on the 2D field,
- confirmed tracks survive short detector gaps as Kalman predictions,
- longer gaps hide a track but keep it in memory for conservative recovery,
- internal candidates are unlimited,
- the final 2D view exposes at most max_visible_tracks (25 by default),
- existing visible tracks get mild hysteresis so the display does not flicker
  when two candidates have almost identical reliability.

The reliability score is an operational stability score, not a calibrated
probability. Its weights are explicit and configurable so they can later be
replaced by evidence from real match data.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Iterable, Sequence

from tracking.association import (
    AssociationConfig,
    AssociationResult,
    associate,
)
from tracking.kalman import KalmanConfig
from tracking.track import Track, TrackFrameState


class TrackLifecycle(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    RETIRED = "retired"


@dataclass(frozen=True)
class TrackManagerConfig:
    # Birth confirmation.
    min_confirmed_hits: int = 3
    tentative_max_gap_s: float = 0.15

    # A confirmed track remains visible as prediction for this long.
    visible_prediction_horizon_s: float = 0.50

    # After becoming hidden/LOST, keep predicting and allow reassociation until
    # this total time since the last real observation.
    lost_memory_horizon_s: float = 2.00

    # Final 2D display capacity only. Internal candidates are not capped.
    max_visible_tracks: int = 25

    # EMA for detector confidence, used only as one component of operational
    # reliability.
    confidence_ema_alpha: float = 0.15

    # Reliability weights; normalized internally.
    reliability_hit_ratio_weight: float = 0.45
    reliability_age_weight: float = 0.20
    reliability_confidence_weight: float = 0.20
    reliability_recency_weight: float = 0.15

    # Display hysteresis. Existing visible tracks receive this small ranking
    # bonus before max_visible_tracks is applied.
    visible_retention_bonus: float = 0.10

    # Number of hits at which the age component saturates.
    age_saturation_hits: int = 30

    def __post_init__(self) -> None:
        if self.min_confirmed_hits < 1:
            raise ValueError("min_confirmed_hits must be >= 1")
        if self.tentative_max_gap_s <= 0.0:
            raise ValueError("tentative_max_gap_s must be > 0")
        if self.visible_prediction_horizon_s <= 0.0:
            raise ValueError("visible_prediction_horizon_s must be > 0")
        if self.lost_memory_horizon_s <= self.visible_prediction_horizon_s:
            raise ValueError(
                "lost_memory_horizon_s must be greater than "
                "visible_prediction_horizon_s"
            )
        if self.max_visible_tracks < 1:
            raise ValueError("max_visible_tracks must be >= 1")
        if not 0.0 < self.confidence_ema_alpha <= 1.0:
            raise ValueError("confidence_ema_alpha must be in (0,1]")
        if self.visible_retention_bonus < 0.0:
            raise ValueError("visible_retention_bonus must be >= 0")
        if self.age_saturation_hits < 1:
            raise ValueError("age_saturation_hits must be >= 1")

        weights = (
            self.reliability_hit_ratio_weight,
            self.reliability_age_weight,
            self.reliability_confidence_weight,
            self.reliability_recency_weight,
        )
        if any((not math.isfinite(w) or w < 0.0) for w in weights):
            raise ValueError("reliability weights must be finite and >= 0")
        if sum(weights) <= 0.0:
            raise ValueError("at least one reliability weight must be > 0")


@dataclass
class ManagedTrack:
    track: Track
    lifecycle: TrackLifecycle
    confidence_ema: float
    created_from_confidence: float
    confirmed_frame: int | None = None
    confirmed_time_s: float | None = None
    visible: bool = False

    @property
    def track_id(self) -> int:
        return self.track.track_id


@dataclass(frozen=True)
class ManagerAssignment:
    track_id: int
    observation_index: int
    mahalanobis_squared: float
    predicted_distance_m: float


@dataclass(frozen=True)
class ManagedTrackSnapshot:
    track_id: int
    lifecycle: str
    visible: bool
    reliability: float

    frame: int
    time_s: float
    frame_state: str

    x: float
    z: float
    vx: float
    vz: float
    speed_mps: float

    hits: int
    misses: int
    consecutive_misses: int
    seconds_since_observation: float
    confidence_ema: float


@dataclass(frozen=True)
class TrackManagerFrameResult:
    frame: int
    time_s: float

    assignments: tuple[ManagerAssignment, ...]
    unmatched_observation_indices: tuple[int, ...]

    new_track_ids: tuple[int, ...]
    newly_confirmed_track_ids: tuple[int, ...]
    reactivated_track_ids: tuple[int, ...]
    newly_lost_track_ids: tuple[int, ...]
    retired_track_ids: tuple[int, ...]

    visible_track_ids: tuple[int, ...]
    visible_tracks: tuple[ManagedTrackSnapshot, ...]
    all_active_tracks: tuple[ManagedTrackSnapshot, ...]


def _observation_confidence(observation: Any) -> float:
    value = float(getattr(observation, "confidence", 1.0))
    if not math.isfinite(value):
        raise ValueError("observation confidence must be finite")
    return max(0.0, min(1.0, value))


class TrackManager:
    """Own all anonymous tracks for one continuous detector run."""

    def __init__(
        self,
        *,
        config: TrackManagerConfig | None = None,
        kalman_config: KalmanConfig | None = None,
        association_config: AssociationConfig | None = None,
        first_track_id: int = 1,
    ) -> None:
        self.config = config or TrackManagerConfig()
        self.kalman_config = kalman_config or KalmanConfig()
        self.association_config = association_config or AssociationConfig()

        self._next_track_id = int(first_track_id)
        if self._next_track_id < 0:
            raise ValueError("first_track_id must be >= 0")

        self._managed: dict[int, ManagedTrack] = {}
        self._retired: dict[int, ManagedTrack] = {}
        self._visible_ids: set[int] = set()

        self._last_frame: int | None = None
        self._last_time_s: float | None = None

    @property
    def active_tracks(self) -> tuple[ManagedTrack, ...]:
        return tuple(
            self._managed[track_id]
            for track_id in sorted(self._managed)
            if self._managed[track_id].lifecycle != TrackLifecycle.RETIRED
        )

    @property
    def retired_tracks(self) -> tuple[ManagedTrack, ...]:
        return tuple(
            self._retired[track_id]
            for track_id in sorted(self._retired)
        )

    def reliability(self, managed: ManagedTrack) -> float:
        """Operational stability score in [0,1], not a probability."""

        track = managed.track
        cfg = self.config

        total_resolutions = track.hits + track.misses
        hit_ratio = (
            track.hits / total_resolutions
            if total_resolutions > 0
            else 0.0
        )
        age_score = min(
            1.0,
            track.hits / float(cfg.age_saturation_hits),
        )
        confidence_score = max(0.0, min(1.0, managed.confidence_ema))

        if cfg.visible_prediction_horizon_s > 0.0:
            recency_score = max(
                0.0,
                1.0
                - track.seconds_since_observation
                / cfg.visible_prediction_horizon_s,
            )
        else:
            recency_score = 0.0

        weights = (
            cfg.reliability_hit_ratio_weight,
            cfg.reliability_age_weight,
            cfg.reliability_confidence_weight,
            cfg.reliability_recency_weight,
        )
        components = (
            hit_ratio,
            age_score,
            confidence_score,
            recency_score,
        )
        denominator = sum(weights)
        score = sum(w * c for w, c in zip(weights, components)) / denominator
        return max(0.0, min(1.0, float(score)))

    def process_frame(
        self,
        *,
        frame: int,
        time_s: float,
        observations: Sequence[Any],
    ) -> TrackManagerFrameResult:
        frame = int(frame)
        time_s = float(time_s)
        observations = tuple(observations)

        self._validate_frame_clock(frame, time_s)
        self._validate_observations(frame, time_s, observations)

        new_track_ids: list[int] = []
        newly_confirmed: list[int] = []
        reactivated: list[int] = []
        newly_lost: list[int] = []
        retired: list[int] = []

        # First frame: every observation starts as a hidden tentative track.
        if self._last_frame is None:
            for observation in observations:
                managed = self._spawn(observation)
                new_track_ids.append(managed.track_id)

            self._last_frame = frame
            self._last_time_s = time_s
            self._select_visible_tracks()

            return self._frame_result(
                frame=frame,
                time_s=time_s,
                assignments=(),
                unmatched_observation_indices=tuple(range(len(observations))),
                new_track_ids=tuple(new_track_ids),
                newly_confirmed_track_ids=(),
                reactivated_track_ids=(),
                newly_lost_track_ids=(),
                retired_track_ids=(),
            )

        # Predict every active track to the new frame.
        for managed in list(self._managed.values()):
            if managed.lifecycle == TrackLifecycle.RETIRED:
                continue
            managed.track.predict_to(frame=frame, time_s=time_s)

        # Age lifecycle before association. Lost tracks still participate while
        # within memory; expired tracks do not.
        for managed in list(self._managed.values()):
            transition = self._age_lifecycle(managed)
            if transition == "lost":
                newly_lost.append(managed.track_id)
            elif transition == "retired":
                retired.append(managed.track_id)

        associable = [
            managed
            for managed in self._managed.values()
            if managed.lifecycle != TrackLifecycle.RETIRED
        ]
        association = associate(
            [managed.track for managed in associable],
            observations,
            config=self.association_config,
        )

        assignments: list[ManagerAssignment] = []
        matched_track_indices = {
            match.track_index for match in association.matches
        }

        for match in association.matches:
            managed = associable[match.track_index]
            observation = observations[match.observation_index]
            previous_lifecycle = managed.lifecycle

            managed.track.update(observation)
            self._update_confidence(managed, observation)

            if previous_lifecycle == TrackLifecycle.LOST:
                managed.lifecycle = TrackLifecycle.CONFIRMED
                reactivated.append(managed.track_id)
            elif (
                previous_lifecycle == TrackLifecycle.TENTATIVE
                and managed.track.hits >= self.config.min_confirmed_hits
            ):
                managed.lifecycle = TrackLifecycle.CONFIRMED
                managed.confirmed_frame = frame
                managed.confirmed_time_s = time_s
                newly_confirmed.append(managed.track_id)

            assignments.append(
                ManagerAssignment(
                    track_id=managed.track_id,
                    observation_index=match.observation_index,
                    mahalanobis_squared=match.mahalanobis_squared,
                    predicted_distance_m=match.predicted_distance_m,
                )
            )

        # Resolve unmatched predictions.
        for track_index, managed in enumerate(associable):
            if track_index in matched_track_indices:
                continue
            managed.track.mark_missed()

            transition = self._age_lifecycle(managed)
            if transition == "lost" and managed.track_id not in newly_lost:
                newly_lost.append(managed.track_id)
            elif transition == "retired" and managed.track_id not in retired:
                retired.append(managed.track_id)

        # Every unmatched observation becomes an internal tentative candidate,
        # never an immediately visible player.
        for observation_index in association.unmatched_observation_indices:
            managed = self._spawn(observations[observation_index])
            new_track_ids.append(managed.track_id)

        # Move retired tracks to archive after all association bookkeeping.
        for track_id in retired:
            self._retire(track_id)

        self._last_frame = frame
        self._last_time_s = time_s
        self._select_visible_tracks()

        return self._frame_result(
            frame=frame,
            time_s=time_s,
            assignments=tuple(assignments),
            unmatched_observation_indices=association.unmatched_observation_indices,
            new_track_ids=tuple(new_track_ids),
            newly_confirmed_track_ids=tuple(sorted(set(newly_confirmed))),
            reactivated_track_ids=tuple(sorted(set(reactivated))),
            newly_lost_track_ids=tuple(sorted(set(newly_lost))),
            retired_track_ids=tuple(sorted(set(retired))),
        )

    def _spawn(self, observation: Any) -> ManagedTrack:
        track_id = self._next_track_id
        self._next_track_id += 1

        confidence = _observation_confidence(observation)
        track = Track(
            track_id=track_id,
            initial_observation=observation,
            kalman_config=self.kalman_config,
        )
        managed = ManagedTrack(
            track=track,
            lifecycle=TrackLifecycle.TENTATIVE,
            confidence_ema=confidence,
            created_from_confidence=confidence,
        )

        if self.config.min_confirmed_hits <= 1:
            managed.lifecycle = TrackLifecycle.CONFIRMED
            managed.confirmed_frame = track.current_frame
            managed.confirmed_time_s = track.current_time_s

        self._managed[track_id] = managed
        return managed

    def _update_confidence(self, managed: ManagedTrack, observation: Any) -> None:
        alpha = self.config.confidence_ema_alpha
        confidence = _observation_confidence(observation)
        managed.confidence_ema = (
            alpha * confidence
            + (1.0 - alpha) * managed.confidence_ema
        )

    def _age_lifecycle(self, managed: ManagedTrack) -> str | None:
        if managed.lifecycle == TrackLifecycle.RETIRED:
            return None

        elapsed = managed.track.seconds_since_observation

        if managed.lifecycle == TrackLifecycle.TENTATIVE:
            if elapsed > self.config.tentative_max_gap_s:
                managed.lifecycle = TrackLifecycle.RETIRED
                return "retired"
            return None

        if elapsed > self.config.lost_memory_horizon_s:
            managed.lifecycle = TrackLifecycle.RETIRED
            return "retired"

        if (
            managed.lifecycle == TrackLifecycle.CONFIRMED
            and elapsed > self.config.visible_prediction_horizon_s
        ):
            managed.lifecycle = TrackLifecycle.LOST
            return "lost"

        return None

    def _retire(self, track_id: int) -> None:
        managed = self._managed.pop(track_id, None)
        if managed is None:
            return
        managed.lifecycle = TrackLifecycle.RETIRED
        managed.visible = False
        self._visible_ids.discard(track_id)
        self._retired[track_id] = managed

    def _select_visible_tracks(self) -> None:
        cfg = self.config

        eligible = [
            managed
            for managed in self._managed.values()
            if (
                managed.lifecycle == TrackLifecycle.CONFIRMED
                and managed.track.seconds_since_observation
                <= cfg.visible_prediction_horizon_s
            )
        ]

        ranked = sorted(
            eligible,
            key=lambda managed: (
                -(
                    self.reliability(managed)
                    + (
                        cfg.visible_retention_bonus
                        if managed.track_id in self._visible_ids
                        else 0.0
                    )
                ),
                -managed.track.hits,
                managed.track_id,
            ),
        )

        selected_ids = {
            managed.track_id
            for managed in ranked[: cfg.max_visible_tracks]
        }

        for managed in self._managed.values():
            managed.visible = managed.track_id in selected_ids

        self._visible_ids = selected_ids

    def _snapshot(self, managed: ManagedTrack) -> ManagedTrackSnapshot:
        track_snapshot = managed.track.snapshot()
        return ManagedTrackSnapshot(
            track_id=managed.track_id,
            lifecycle=managed.lifecycle.value,
            visible=managed.visible,
            reliability=self.reliability(managed),
            frame=track_snapshot.frame,
            time_s=track_snapshot.time_s,
            frame_state=track_snapshot.frame_state,
            x=track_snapshot.x,
            z=track_snapshot.z,
            vx=track_snapshot.vx,
            vz=track_snapshot.vz,
            speed_mps=track_snapshot.speed_mps,
            hits=track_snapshot.hits,
            misses=track_snapshot.misses,
            consecutive_misses=track_snapshot.consecutive_misses,
            seconds_since_observation=managed.track.seconds_since_observation,
            confidence_ema=managed.confidence_ema,
        )

    def _frame_result(
        self,
        *,
        frame: int,
        time_s: float,
        assignments: tuple[ManagerAssignment, ...],
        unmatched_observation_indices: tuple[int, ...],
        new_track_ids: tuple[int, ...],
        newly_confirmed_track_ids: tuple[int, ...],
        reactivated_track_ids: tuple[int, ...],
        newly_lost_track_ids: tuple[int, ...],
        retired_track_ids: tuple[int, ...],
    ) -> TrackManagerFrameResult:
        active = [
            self._snapshot(managed)
            for managed in self._managed.values()
            if managed.lifecycle != TrackLifecycle.RETIRED
        ]
        active.sort(key=lambda item: item.track_id)

        visible = [item for item in active if item.visible]
        visible.sort(key=lambda item: item.track_id)

        return TrackManagerFrameResult(
            frame=frame,
            time_s=time_s,
            assignments=assignments,
            unmatched_observation_indices=unmatched_observation_indices,
            new_track_ids=new_track_ids,
            newly_confirmed_track_ids=newly_confirmed_track_ids,
            reactivated_track_ids=reactivated_track_ids,
            newly_lost_track_ids=newly_lost_track_ids,
            retired_track_ids=retired_track_ids,
            visible_track_ids=tuple(item.track_id for item in visible),
            visible_tracks=tuple(visible),
            all_active_tracks=tuple(active),
        )

    def _validate_frame_clock(self, frame: int, time_s: float) -> None:
        if not math.isfinite(time_s):
            raise ValueError("time_s must be finite")

        if self._last_frame is not None:
            if frame <= self._last_frame:
                raise ValueError("frames must be strictly increasing")
            assert self._last_time_s is not None
            if time_s <= self._last_time_s:
                raise ValueError("time_s must be strictly increasing")

    @staticmethod
    def _validate_observations(
        frame: int,
        time_s: float,
        observations: Sequence[Any],
    ) -> None:
        for observation in observations:
            if int(observation.frame) != frame:
                raise ValueError("all observations must belong to frame")
            if not math.isclose(
                float(observation.time_s),
                time_s,
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                raise ValueError("all observations must share frame time")


__all__ = [
    "ManagedTrack",
    "ManagedTrackSnapshot",
    "ManagerAssignment",
    "TrackLifecycle",
    "TrackManager",
    "TrackManagerConfig",
    "TrackManagerFrameResult",
]
