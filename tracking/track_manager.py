"""TrackManager V2.1: V2 behaviour plus association diagnostics.

The tracking decisions are intentionally the same as V2:
    CONFIRMED -> LOST -> TENTATIVE -> births

V2.1 only instruments those decisions so real match runs can answer:
- which lifecycle stage rejected candidate pairs,
- whether physical or Mahalanobis gating is responsible,
- why observations still unmatched after the full cascade became births.

No proximity deduplication or new tracking heuristic is introduced here.

Workflow note:
This revision is delivered as track_manager_V2_1.py.  When promoted in the
project it should replace the stable tracking/track_manager.py so existing
runtime imports remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Any, Sequence

from tracking.association import (
    AssociationConfig,
    PairDiagnostics,
    associate,
    evaluate_pair,
)
from tracking.kalman import KalmanConfig
from tracking.track import Track


class TrackLifecycle(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"
    RETIRED = "retired"


@dataclass(frozen=True)
class TrackManagerConfig:
    min_confirmed_hits: int = 3
    tentative_max_gap_s: float = 0.15

    visible_prediction_horizon_s: float = 0.50
    lost_memory_horizon_s: float = 2.00

    max_visible_tracks: int = 25

    confidence_ema_alpha: float = 0.15

    reliability_hit_ratio_weight: float = 0.45
    reliability_age_weight: float = 0.20
    reliability_confidence_weight: float = 0.20
    reliability_recency_weight: float = 0.15

    visible_retention_bonus: float = 0.10
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
class StageAssociationDiagnostics:
    lifecycle: str
    track_count: int
    observation_count: int

    pairs_evaluated: int
    allowed_pairs: int
    physical_gate_rejections: int
    mahalanobis_gate_rejections: int
    non_finite_mahalanobis_rejections: int
    other_rejections: int

    matches: int

    @property
    def rejected_pairs(self) -> int:
        return self.pairs_evaluated - self.allowed_pairs


@dataclass(frozen=True)
class NearestTrackDiagnostic:
    lifecycle: str
    track_id: int
    predicted_distance_m: float
    displacement_from_last_observed_m: float
    physical_radius_m: float
    uncertainty_slack_m: float
    mahalanobis_squared: float | None
    allowed: bool
    reason: str


@dataclass(frozen=True)
class UnmatchedObservationDiagnostic:
    observation_index: int
    detection_index: int | None
    confidence: float

    nearest_confirmed: NearestTrackDiagnostic | None
    nearest_lost: NearestTrackDiagnostic | None
    nearest_tentative: NearestTrackDiagnostic | None


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

    association_stage_diagnostics: tuple[StageAssociationDiagnostics, ...] = ()
    unmatched_observation_diagnostics: tuple[
        UnmatchedObservationDiagnostic, ...
    ] = ()


def _observation_confidence(observation: Any) -> float:
    value = float(getattr(observation, "confidence", 1.0))
    if not math.isfinite(value):
        raise ValueError("observation confidence must be finite")
    return max(0.0, min(1.0, value))


def _summarize_stage(
    *,
    lifecycle: TrackLifecycle,
    track_count: int,
    observation_count: int,
    pairs: Sequence[PairDiagnostics],
    matches: int,
) -> StageAssociationDiagnostics:
    counts = {
        "allowed": 0,
        "physical_gate": 0,
        "mahalanobis_gate": 0,
        "non_finite_mahalanobis": 0,
        "other": 0,
    }

    for pair in pairs:
        if pair.allowed:
            counts["allowed"] += 1
        elif pair.reason in counts:
            counts[pair.reason] += 1
        else:
            counts["other"] += 1

    return StageAssociationDiagnostics(
        lifecycle=lifecycle.value,
        track_count=track_count,
        observation_count=observation_count,
        pairs_evaluated=len(pairs),
        allowed_pairs=counts["allowed"],
        physical_gate_rejections=counts["physical_gate"],
        mahalanobis_gate_rejections=counts["mahalanobis_gate"],
        non_finite_mahalanobis_rejections=counts["non_finite_mahalanobis"],
        other_rejections=counts["other"],
        matches=matches,
    )


class TrackManager:
    """Anonymous football-person track manager with diagnostic cascade."""

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

        recency_score = max(
            0.0,
            1.0
            - track.seconds_since_observation
            / cfg.visible_prediction_horizon_s,
        )

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
        score = (
            sum(w * c for w, c in zip(weights, components))
            / sum(weights)
        )
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
        assignments: list[ManagerAssignment] = []
        stage_diagnostics: list[StageAssociationDiagnostics] = []

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
                association_stage_diagnostics=(),
                unmatched_observation_diagnostics=(),
            )

        for managed in list(self._managed.values()):
            if managed.lifecycle != TrackLifecycle.RETIRED:
                managed.track.predict_to(frame=frame, time_s=time_s)

        for managed in list(self._managed.values()):
            transition = self._age_lifecycle(managed)
            if transition == "lost":
                newly_lost.append(managed.track_id)
            elif transition == "retired":
                retired.append(managed.track_id)

        for track_id in retired:
            self._retire(track_id)

        remaining_observation_indices = list(range(len(observations)))
        matched_track_ids: set[int] = set()

        lifecycle_stages = (
            TrackLifecycle.CONFIRMED,
            TrackLifecycle.LOST,
            TrackLifecycle.TENTATIVE,
        )

        for lifecycle in lifecycle_stages:
            stage_tracks = [
                managed
                for managed in self._managed.values()
                if managed.lifecycle == lifecycle
                and managed.track_id not in matched_track_ids
            ]

            if not stage_tracks or not remaining_observation_indices:
                stage_diagnostics.append(
                    StageAssociationDiagnostics(
                        lifecycle=lifecycle.value,
                        track_count=len(stage_tracks),
                        observation_count=len(remaining_observation_indices),
                        pairs_evaluated=0,
                        allowed_pairs=0,
                        physical_gate_rejections=0,
                        mahalanobis_gate_rejections=0,
                        non_finite_mahalanobis_rejections=0,
                        other_rejections=0,
                        matches=0,
                    )
                )
                continue

            stage_observations = [
                observations[index]
                for index in remaining_observation_indices
            ]

            result = associate(
                [managed.track for managed in stage_tracks],
                stage_observations,
                config=self.association_config,
                include_pair_diagnostics=True,
            )

            stage_diagnostics.append(
                _summarize_stage(
                    lifecycle=lifecycle,
                    track_count=len(stage_tracks),
                    observation_count=len(stage_observations),
                    pairs=result.pair_diagnostics,
                    matches=len(result.matches),
                )
            )

            matched_stage_observation_positions: set[int] = set()

            for match in result.matches:
                managed = stage_tracks[match.track_index]
                global_observation_index = remaining_observation_indices[
                    match.observation_index
                ]
                observation = observations[global_observation_index]

                previous_lifecycle = managed.lifecycle
                managed.track.update(observation)
                self._update_confidence(managed, observation)
                matched_track_ids.add(managed.track_id)
                matched_stage_observation_positions.add(
                    match.observation_index
                )

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
                        observation_index=global_observation_index,
                        mahalanobis_squared=match.mahalanobis_squared,
                        predicted_distance_m=match.predicted_distance_m,
                    )
                )

            remaining_observation_indices = [
                global_index
                for local_index, global_index
                in enumerate(remaining_observation_indices)
                if local_index not in matched_stage_observation_positions
            ]

        # Capture diagnostics while every unmatched track is still in the
        # prediction-ready state.  mark_missed() resolves that state.
        unmatched_observation_diagnostics = self._diagnose_unmatched(
            observations=observations,
            observation_indices=remaining_observation_indices,
            matched_track_ids=matched_track_ids,
        )

        for managed in list(self._managed.values()):
            if managed.track_id in matched_track_ids:
                continue

            managed.track.mark_missed()

            transition = self._age_lifecycle(managed)
            if transition == "lost" and managed.track_id not in newly_lost:
                newly_lost.append(managed.track_id)
            elif transition == "retired" and managed.track_id not in retired:
                retired.append(managed.track_id)

        for track_id in retired:
            self._retire(track_id)

        for observation_index in remaining_observation_indices:
            managed = self._spawn(observations[observation_index])
            new_track_ids.append(managed.track_id)

        self._last_frame = frame
        self._last_time_s = time_s
        self._select_visible_tracks()

        return self._frame_result(
            frame=frame,
            time_s=time_s,
            assignments=tuple(assignments),
            unmatched_observation_indices=tuple(
                remaining_observation_indices
            ),
            new_track_ids=tuple(new_track_ids),
            newly_confirmed_track_ids=tuple(sorted(set(newly_confirmed))),
            reactivated_track_ids=tuple(sorted(set(reactivated))),
            newly_lost_track_ids=tuple(sorted(set(newly_lost))),
            retired_track_ids=tuple(sorted(set(retired))),
            association_stage_diagnostics=tuple(stage_diagnostics),
            unmatched_observation_diagnostics=unmatched_observation_diagnostics,
        )

    def _diagnose_unmatched(
        self,
        *,
        observations: Sequence[Any],
        observation_indices: Sequence[int],
        matched_track_ids: set[int],
    ) -> tuple[UnmatchedObservationDiagnostic, ...]:
        candidates_by_lifecycle: dict[TrackLifecycle, list[ManagedTrack]] = {
            TrackLifecycle.CONFIRMED: [],
            TrackLifecycle.LOST: [],
            TrackLifecycle.TENTATIVE: [],
        }

        for managed in self._managed.values():
            if managed.track_id in matched_track_ids:
                continue
            if managed.lifecycle in candidates_by_lifecycle:
                candidates_by_lifecycle[managed.lifecycle].append(managed)

        result: list[UnmatchedObservationDiagnostic] = []

        for observation_index in observation_indices:
            observation = observations[observation_index]

            nearest = {}
            for lifecycle in (
                TrackLifecycle.CONFIRMED,
                TrackLifecycle.LOST,
                TrackLifecycle.TENTATIVE,
            ):
                best = None
                for managed in candidates_by_lifecycle[lifecycle]:
                    pair = evaluate_pair(
                        managed.track,
                        observation,
                        track_index=0,
                        observation_index=observation_index,
                        config=self.association_config,
                    )
                    diagnostic = NearestTrackDiagnostic(
                        lifecycle=lifecycle.value,
                        track_id=managed.track_id,
                        predicted_distance_m=pair.predicted_distance_m,
                        displacement_from_last_observed_m=(
                            pair.displacement_from_last_observed_m
                        ),
                        physical_radius_m=pair.physical_radius_m,
                        uncertainty_slack_m=pair.uncertainty_slack_m,
                        mahalanobis_squared=pair.mahalanobis_squared,
                        allowed=pair.allowed,
                        reason=pair.reason,
                    )

                    if (
                        best is None
                        or diagnostic.predicted_distance_m
                        < best.predicted_distance_m
                    ):
                        best = diagnostic

                nearest[lifecycle] = best

            detection_index_value = getattr(
                observation,
                "detection_index",
                None,
            )
            detection_index = (
                int(detection_index_value)
                if detection_index_value is not None
                else None
            )

            result.append(
                UnmatchedObservationDiagnostic(
                    observation_index=observation_index,
                    detection_index=detection_index,
                    confidence=_observation_confidence(observation),
                    nearest_confirmed=nearest[TrackLifecycle.CONFIRMED],
                    nearest_lost=nearest[TrackLifecycle.LOST],
                    nearest_tentative=nearest[TrackLifecycle.TENTATIVE],
                )
            )

        return tuple(result)

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
        association_stage_diagnostics: tuple[
            StageAssociationDiagnostics, ...
        ],
        unmatched_observation_diagnostics: tuple[
            UnmatchedObservationDiagnostic, ...
        ],
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
            association_stage_diagnostics=association_stage_diagnostics,
            unmatched_observation_diagnostics=(
                unmatched_observation_diagnostics
            ),
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
    "NearestTrackDiagnostic",
    "StageAssociationDiagnostics",
    "TrackLifecycle",
    "TrackManager",
    "TrackManagerConfig",
    "TrackManagerFrameResult",
    "UnmatchedObservationDiagnostic",
]
