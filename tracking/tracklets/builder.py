"""Conservative local tracklet construction for Tracking V1.

This module deliberately solves only the *short-term continuity* problem.
It does not try to recover a player's identity across long disappearances,
dense ambiguous interactions, or major occlusions. Those decisions belong to
later offline/global stages.

Design principle
----------------
When an association is ambiguous, fragment the trajectory instead of risking
identity contamination.

The builder therefore uses:
* field-space X/Z geometry;
* constant-velocity prediction when enough history exists;
* hard physical gating;
* short miss tolerance;
* mutual-best matching;
* explicit ambiguity margins on both sides of an association.

The implementation has no scipy/numpy dependency and is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

from tracking.detections import CanonicalDetection, CanonicalDetectionFrame

from .models import Tracklet, TrackletObservation, TrackletSet


@dataclass(frozen=True, slots=True)
class TrackletBuilderConfig:
    """Configuration for conservative short-term association.

    Defaults intentionally prefer fragmentation to a doubtful association.

    ``max_gap_frames`` and ``max_gap_s`` limit how long a local tracklet may
    survive without an observation. Both constraints must be satisfied.

    ``max_speed_mps`` is a hard field-space feasibility bound from the most
    recent observation. ``prediction_gate_m`` is the uncertainty radius around
    the constant-velocity prediction, expanded by
    ``uncertainty_growth_mps * dt``.

    ``ambiguity_margin`` is expressed in normalized association-cost units.
    The best candidate must beat the second-best candidate by at least this
    margin for both the tracklet and the detection.
    """

    max_gap_frames: int = 8
    max_gap_s: float = 0.35
    max_speed_mps: float = 12.0
    prediction_gate_m: float = 0.85
    uncertainty_growth_mps: float = 1.75
    ambiguity_margin: float = 0.18
    confidence_penalty: float = 0.08
    velocity_smoothing: float = 0.70

    def __post_init__(self) -> None:
        if isinstance(self.max_gap_frames, bool) or not isinstance(
            self.max_gap_frames, int
        ):
            raise ValueError("max_gap_frames must be an integer")
        if self.max_gap_frames < 0:
            raise ValueError("max_gap_frames must be >= 0")

        for name in (
            "max_gap_s",
            "max_speed_mps",
            "prediction_gate_m",
            "uncertainty_growth_mps",
            "ambiguity_margin",
            "confidence_penalty",
            "velocity_smoothing",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")

        if self.max_gap_s < 0:
            raise ValueError("max_gap_s must be >= 0")
        if self.max_speed_mps <= 0:
            raise ValueError("max_speed_mps must be > 0")
        if self.prediction_gate_m <= 0:
            raise ValueError("prediction_gate_m must be > 0")
        if self.uncertainty_growth_mps < 0:
            raise ValueError("uncertainty_growth_mps must be >= 0")
        if self.ambiguity_margin < 0:
            raise ValueError("ambiguity_margin must be >= 0")
        if self.confidence_penalty < 0:
            raise ValueError("confidence_penalty must be >= 0")
        if not 0.0 <= self.velocity_smoothing <= 1.0:
            raise ValueError("velocity_smoothing must be in [0, 1]")


@dataclass(slots=True)
class _DraftTracklet:
    serial: int
    observations: list[TrackletObservation]
    vx: float = 0.0
    vz: float = 0.0
    has_velocity: bool = False

    @property
    def last(self) -> TrackletObservation:
        return self.observations[-1]

    def predicted_position(
        self,
        timestamp_s: float,
    ) -> tuple[float, float]:
        dt = timestamp_s - self.last.timestamp_s
        if dt <= 0.0 or not self.has_velocity:
            return self.last.x, self.last.z
        return (
            self.last.x + self.vx * dt,
            self.last.z + self.vz * dt,
        )

    def append(
        self,
        observation: TrackletObservation,
        *,
        velocity_smoothing: float,
    ) -> None:
        previous = self.last
        dt = observation.timestamp_s - previous.timestamp_s

        if dt > 0.0:
            measured_vx = (observation.x - previous.x) / dt
            measured_vz = (observation.z - previous.z) / dt

            if self.has_velocity:
                keep = velocity_smoothing
                inject = 1.0 - keep
                self.vx = keep * self.vx + inject * measured_vx
                self.vz = keep * self.vz + inject * measured_vz
            else:
                self.vx = measured_vx
                self.vz = measured_vz
                self.has_velocity = True

        self.observations.append(observation)


@dataclass(frozen=True, slots=True)
class _Candidate:
    track_index: int
    detection_index: int
    cost: float


class ConservativeTrackletBuilder:
    """Build short, conservative field-space tracklets.

    The builder is intentionally *not* a full player tracker. A player can and
    often should become several tracklets if local evidence is weak.
    """

    VERSION = "conservative-local-v1"

    def __init__(self, config: TrackletBuilderConfig | None = None) -> None:
        self.config = config or TrackletBuilderConfig()

    def build(
        self,
        frames: Iterable[CanonicalDetectionFrame],
    ) -> TrackletSet:
        active: list[_DraftTracklet] = []
        finished: list[_DraftTracklet] = []
        next_serial = 1

        previous_frame_index: int | None = None
        previous_timestamp_s: float | None = None

        for raw_frame in frames:
            frame = (
                raw_frame
                if isinstance(raw_frame, CanonicalDetectionFrame)
                else CanonicalDetectionFrame.from_dict(raw_frame)
            )

            if (
                previous_frame_index is not None
                and frame.frame_index <= previous_frame_index
            ):
                raise ValueError("frames must be strictly increasing by frame_index")
            if (
                previous_timestamp_s is not None
                and frame.timestamp_s <= previous_timestamp_s
            ):
                raise ValueError("frames must be strictly increasing by timestamp_s")

            previous_frame_index = frame.frame_index
            previous_timestamp_s = frame.timestamp_s

            still_active: list[_DraftTracklet] = []
            for draft in active:
                if self._within_survival_window(draft, frame):
                    still_active.append(draft)
                else:
                    finished.append(draft)
            active = still_active

            detections = tuple(frame.detections)
            accepted = self._associate(active, detections, frame)

            matched_tracks: set[int] = set()
            matched_detections: set[int] = set()

            for track_index, detection_index in accepted:
                observation = self._observation(detections[detection_index])
                active[track_index].append(
                    observation,
                    velocity_smoothing=self.config.velocity_smoothing,
                )
                matched_tracks.add(track_index)
                matched_detections.add(detection_index)

            # Unmatched active drafts remain alive until their local survival
            # window expires. This permits a few missed detector frames.
            _ = matched_tracks

            for detection_index, detection in enumerate(detections):
                if detection_index in matched_detections:
                    continue
                active.append(
                    _DraftTracklet(
                        serial=next_serial,
                        observations=[self._observation(detection)],
                    )
                )
                next_serial += 1

        finished.extend(active)

        tracklets = tuple(
            self._finalize(draft)
            for draft in sorted(finished, key=lambda item: item.serial)
        )

        return TrackletSet(
            tracklets=tracklets,
            metadata={
                "builder": self.VERSION,
                "config": {
                    "max_gap_frames": self.config.max_gap_frames,
                    "max_gap_s": self.config.max_gap_s,
                    "max_speed_mps": self.config.max_speed_mps,
                    "prediction_gate_m": self.config.prediction_gate_m,
                    "uncertainty_growth_mps": self.config.uncertainty_growth_mps,
                    "ambiguity_margin": self.config.ambiguity_margin,
                    "confidence_penalty": self.config.confidence_penalty,
                    "velocity_smoothing": self.config.velocity_smoothing,
                },
            },
        )

    def _within_survival_window(
        self,
        draft: _DraftTracklet,
        frame: CanonicalDetectionFrame,
    ) -> bool:
        frame_delta = frame.frame_index - draft.last.frame_index
        time_delta = frame.timestamp_s - draft.last.timestamp_s

        if frame_delta <= 0 or time_delta <= 0:
            return False

        missed_frames = frame_delta - 1
        return (
            missed_frames <= self.config.max_gap_frames
            and time_delta <= self.config.max_gap_s
        )

    def _associate(
        self,
        active: Sequence[_DraftTracklet],
        detections: Sequence[CanonicalDetection],
        frame: CanonicalDetectionFrame,
    ) -> tuple[tuple[int, int], ...]:
        if not active or not detections:
            return ()

        candidates: list[_Candidate] = []

        for track_index, draft in enumerate(active):
            for detection_index, detection in enumerate(detections):
                cost = self._association_cost(draft, detection, frame)
                if cost is not None:
                    candidates.append(
                        _Candidate(
                            track_index=track_index,
                            detection_index=detection_index,
                            cost=cost,
                        )
                    )

        if not candidates:
            return ()

        by_track: dict[int, list[_Candidate]] = {}
        by_detection: dict[int, list[_Candidate]] = {}

        for candidate in candidates:
            by_track.setdefault(candidate.track_index, []).append(candidate)
            by_detection.setdefault(candidate.detection_index, []).append(candidate)

        for values in by_track.values():
            values.sort(key=lambda item: (item.cost, item.detection_index))
        for values in by_detection.values():
            values.sort(key=lambda item: (item.cost, item.track_index))

        accepted: list[tuple[int, int]] = []

        for track_index in sorted(by_track):
            options = by_track[track_index]
            best = options[0]

            detection_options = by_detection[best.detection_index]
            if detection_options[0].track_index != track_index:
                continue

            if not self._has_required_margin(options):
                continue
            if not self._has_required_margin(detection_options):
                continue

            accepted.append((track_index, best.detection_index))

        # Mutual-best construction already guarantees one-to-one matching.
        accepted.sort()
        return tuple(accepted)

    def _has_required_margin(self, options: Sequence[_Candidate]) -> bool:
        if len(options) < 2:
            return True
        return (
            options[1].cost - options[0].cost
            >= self.config.ambiguity_margin
        )

    def _association_cost(
        self,
        draft: _DraftTracklet,
        detection: CanonicalDetection,
        frame: CanonicalDetectionFrame,
    ) -> float | None:
        dt = frame.timestamp_s - draft.last.timestamp_s
        if dt <= 0.0:
            return None

        # First hard gate: physically feasible travel since last observation.
        distance_from_last = math.hypot(
            detection.x - draft.last.x,
            detection.z - draft.last.z,
        )
        physical_gate = (
            self.config.prediction_gate_m
            + self.config.max_speed_mps * dt
        )
        if distance_from_last > physical_gate:
            return None

        predicted_x, predicted_z = draft.predicted_position(frame.timestamp_s)
        prediction_error = math.hypot(
            detection.x - predicted_x,
            detection.z - predicted_z,
        )

        uncertainty_gate = (
            self.config.prediction_gate_m
            + self.config.uncertainty_growth_mps * dt
        )
        if prediction_error > uncertainty_gate:
            return None

        normalized_position_error = prediction_error / uncertainty_gate
        confidence_cost = (
            self.config.confidence_penalty * (1.0 - detection.confidence)
        )

        return normalized_position_error + confidence_cost

    @staticmethod
    def _observation(detection: CanonicalDetection) -> TrackletObservation:
        return TrackletObservation(
            detection_id=detection.detection_id,
            frame_index=detection.frame_index,
            timestamp_s=detection.timestamp_s,
            x=detection.x,
            z=detection.z,
            confidence=detection.confidence,
        )

    def _finalize(self, draft: _DraftTracklet) -> Tracklet:
        return Tracklet(
            tracklet_id=f"trk-{draft.serial:06d}",
            observations=tuple(draft.observations),
            metadata={"builder": self.VERSION},
        )


def build_tracklets(
    frames: Iterable[CanonicalDetectionFrame],
    config: TrackletBuilderConfig | None = None,
) -> TrackletSet:
    """Convenience wrapper around :class:`ConservativeTrackletBuilder`."""
    return ConservativeTrackletBuilder(config).build(frames)
