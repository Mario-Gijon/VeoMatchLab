"""VeoMatchLab offline context tracker V2.

V2 is deliberately conservative.  The online TrackManager remains only a
proposal generator; final identities are built from *anchor tracklets* that
have enough temporal support to be useful.

Key rules:
- Never merge people merely because they are close in X/Z.
- Weak/sparse fragments remain diagnostics and cannot create a final visible
  identity by themselves.
- Tracklets are split on temporal gaps *and* physically implausible jumps.
- Long-gap linking requires reliable motion at both sides and is evaluated
  bidirectionally (past -> future and future -> past).
- The final renderer never tries to "fill" max_visible_tracks.  The value is a
  hard upper bound only.
- Final positions are reconstructed after the whole clip is known.  Internal
  gaps may be interpolated when both past and future support them; clip
  endpoints are not extrapolated by default.
"""

from __future__ import annotations

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
import math
import statistics
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.association import AssociationConfig
from tracking.kalman import FieldKalmanFilter, KalmanConfig
from tracking.track_manager import TrackManager, TrackManagerConfig


CHI2_DF2_P95 = 5.991464547107979
_FORBIDDEN = 1e12


@dataclass(frozen=True)
class OfflineContextConfig:
    # Conservative residual calibration.  Never shrink R from the geometric
    # baseline; only widen it when the upper residual tail asks for it.
    calibration_max_triplet_gap_s: float = 0.15
    calibration_min_triplets: int = 50
    covariance_scale_min: float = 1.0
    covariance_scale_max: float = 9.0

    # Split online proposal tracks into locally coherent fragments.
    tracklet_split_gap_s: float = 0.25
    tracklet_break_max_speed_mps: float = 13.5
    tracklet_break_base_slack_m: float = 0.10
    tracklet_break_uncertainty_sigma_multiplier: float = 3.0

    # Robust endpoint motion estimation.
    velocity_window_s: float = 0.50
    velocity_min_observations: int = 5
    velocity_min_span_s: float = 0.12

    # Only these fragments may define final identities.
    anchor_min_observations: int = 8
    anchor_min_span_s: float = 0.20
    anchor_min_coverage: float = 0.50
    anchor_max_endpoint_speed_mps: float = 14.0

    # Global tracklet linking.
    linker_max_gap_s: float = 2.50
    linker_long_gap_requires_velocity_s: float = 0.50
    linker_max_speed_mps: float = 13.0
    linker_base_slack_m: float = 0.50
    linker_uncertainty_sigma_multiplier: float = 3.0
    linker_prediction_growth_mps: float = 1.50
    linker_velocity_cost_weight: float = 0.10
    linker_gap_cost_weight: float = 0.35
    linker_same_source_bonus: float = 0.40
    linker_unmatched_cost: float = 3.75

    # Final reconstruction.
    final_min_observations: int = 8
    final_min_span_s: float = 0.20
    render_gap_s: float = 0.50
    endpoint_extrapolation_s: float = 0.0
    max_visible_tracks: int = 25

    def __post_init__(self) -> None:
        if self.calibration_min_triplets < 1:
            raise ValueError("calibration_min_triplets must be >= 1")
        if self.velocity_min_observations < 2:
            raise ValueError("velocity_min_observations must be >= 2")
        if self.anchor_min_observations < 2:
            raise ValueError("anchor_min_observations must be >= 2")
        if not 0.0 < self.anchor_min_coverage <= 1.0:
            raise ValueError("anchor_min_coverage must be in (0,1]")
        if self.final_min_observations < 2:
            raise ValueError("final_min_observations must be >= 2")
        if self.max_visible_tracks < 1:
            raise ValueError("max_visible_tracks must be >= 1")
        if self.endpoint_extrapolation_s < 0.0:
            raise ValueError("endpoint_extrapolation_s must be >= 0")

        positive = (
            "calibration_max_triplet_gap_s",
            "covariance_scale_min",
            "covariance_scale_max",
            "tracklet_split_gap_s",
            "tracklet_break_max_speed_mps",
            "tracklet_break_base_slack_m",
            "tracklet_break_uncertainty_sigma_multiplier",
            "velocity_window_s",
            "velocity_min_span_s",
            "anchor_min_span_s",
            "anchor_max_endpoint_speed_mps",
            "linker_max_gap_s",
            "linker_long_gap_requires_velocity_s",
            "linker_max_speed_mps",
            "linker_base_slack_m",
            "linker_uncertainty_sigma_multiplier",
            "linker_prediction_growth_mps",
            "linker_unmatched_cost",
            "final_min_span_s",
            "render_gap_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        if self.covariance_scale_max < self.covariance_scale_min:
            raise ValueError(
                "covariance_scale_max must be >= covariance_scale_min"
            )


class ScaledObservation:
    __slots__ = ("_observation", "_scale")

    def __init__(self, observation: Any, scale: float) -> None:
        self._observation = observation
        self._scale = float(scale)

    @property
    def measurement_covariance(self) -> np.ndarray:
        return (
            np.asarray(
                self._observation.measurement_covariance,
                dtype=np.float64,
            )
            * self._scale
        )

    @property
    def world_position(self) -> np.ndarray:
        return np.asarray(
            self._observation.world_position,
            dtype=np.float64,
        )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._observation, name)


@dataclass(frozen=True)
class OfflineSample:
    source_track_id: int
    frame: int
    time_s: float
    observation_index: int
    detection_index: int | None
    x: float
    z: float
    confidence: float
    covariance: np.ndarray

    @property
    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.z], dtype=np.float64)


@dataclass(frozen=True)
class ForwardPassStats:
    frames: int
    observations: int
    assignments: int
    unmatched_observations: int
    tracks_created: int
    confirmations: int
    reactivations: int
    lost_transitions: int
    retirements: int
    max_visible: int


@dataclass
class ForwardPass:
    samples_by_track: dict[int, list[OfflineSample]]
    ever_confirmed_track_ids: set[int]
    stats: ForwardPassStats


@dataclass(frozen=True)
class CovarianceCalibration:
    scale: float
    triplet_count: int
    nis_median: float | None
    nis_mean: float | None
    nis_p95: float | None
    raw_scale: float | None
    status: str


@dataclass
class Tracklet:
    tracklet_id: int
    source_track_id: int
    source_confirmed: bool
    samples: tuple[OfflineSample, ...]

    start_position: np.ndarray
    end_position: np.ndarray
    start_velocity: np.ndarray | None
    end_velocity: np.ndarray | None

    mean_confidence: float
    frame_span: int
    coverage: float
    anchor: bool
    max_endpoint_speed_mps: float | None

    @property
    def start_frame(self) -> int:
        return self.samples[0].frame

    @property
    def end_frame(self) -> int:
        return self.samples[-1].frame

    @property
    def start_time_s(self) -> float:
        return self.samples[0].time_s

    @property
    def end_time_s(self) -> float:
        return self.samples[-1].time_s

    @property
    def observed_count(self) -> int:
        return len(self.samples)

    @property
    def observed_span_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


@dataclass(frozen=True)
class LinkCandidate:
    predecessor_tracklet_id: int
    successor_tracklet_id: int
    gap_s: float
    direct_distance_m: float
    physical_radius_m: float
    symmetric_prediction_error_m: float
    prediction_radius_m: float
    velocity_difference_mps: float | None
    cost: float


@dataclass(frozen=True)
class TrackletLinkResult:
    links: tuple[tuple[int, int], ...]
    accepted_candidates: tuple[LinkCandidate, ...]
    candidate_count: int


@dataclass
class SmoothedSample:
    frame: int
    time_s: float
    x: float
    z: float
    vx: float
    vz: float
    covariance: np.ndarray


@dataclass
class OfflineTrajectory:
    trajectory_id: int
    tracklet_ids: tuple[int, ...]
    source_track_ids: tuple[int, ...]
    samples: tuple[OfflineSample, ...]
    smoothed_samples: tuple[SmoothedSample, ...]
    mean_confidence: float
    observed_count: int
    observed_span_s: float
    fragment_count: int
    max_observed_gap_s: float
    render_eligible: bool

    @property
    def start_time_s(self) -> float:
        return self.samples[0].time_s

    @property
    def end_time_s(self) -> float:
        return self.samples[-1].time_s


@dataclass(frozen=True)
class OfflineFrameTrack:
    trajectory_id: int
    frame: int
    time_s: float
    state: str
    x: float
    z: float
    vx: float
    vz: float
    support_score: float
    observed_count: int
    fragment_count: int
    mean_confidence: float


@dataclass
class OfflineContextResult:
    preliminary: ForwardPass
    calibrated: ForwardPass
    calibration: CovarianceCalibration
    tracklets: tuple[Tracklet, ...]
    anchor_tracklets: tuple[Tracklet, ...]
    weak_tracklets: tuple[Tracklet, ...]
    link_result: TrackletLinkResult
    trajectories: tuple[OfflineTrajectory, ...]
    frame_tracks: dict[int, tuple[OfflineFrameTrack, ...]]


def _observation_covariance(observation: Any) -> np.ndarray:
    value = np.asarray(
        observation.measurement_covariance,
        dtype=np.float64,
    )
    if value.shape != (2, 2) or not np.isfinite(value).all():
        raise ValueError(
            "observation covariance must be finite 2x2"
        )
    return 0.5 * (value + value.T)


def _sample(*, track_id: int, observation: Any, observation_index: int):
    detection_index_value = getattr(
        observation,
        "detection_index",
        None,
    )
    return OfflineSample(
        source_track_id=int(track_id),
        frame=int(observation.frame),
        time_s=float(observation.time_s),
        observation_index=int(observation_index),
        detection_index=(
            int(detection_index_value)
            if detection_index_value is not None
            else None
        ),
        x=float(observation.world_position[0]),
        z=float(observation.world_position[1]),
        confidence=float(
            getattr(observation, "confidence", 1.0)
        ),
        covariance=_observation_covariance(observation),
    )


def run_forward_pass(
    frames: Sequence[Any],
    *,
    covariance_scale: float,
    manager_config: TrackManagerConfig | None = None,
    association_config: AssociationConfig | None = None,
    kalman_config: KalmanConfig | None = None,
) -> ForwardPass:
    manager = TrackManager(
        config=manager_config or TrackManagerConfig(),
        association_config=(
            association_config or AssociationConfig()
        ),
        kalman_config=kalman_config or KalmanConfig(),
    )

    samples_by_track: dict[int, list[OfflineSample]] = defaultdict(list)
    ever_confirmed: set[int] = set()
    totals = {
        "observations": 0,
        "assignments": 0,
        "unmatched": 0,
        "tracks_created": 0,
        "confirmations": 0,
        "reactivations": 0,
        "lost": 0,
        "retirements": 0,
        "max_visible": 0,
    }

    for frame in frames:
        observations = tuple(
            ScaledObservation(item, covariance_scale)
            for item in frame.observations
        )
        result = manager.process_frame(
            frame=frame.frame,
            time_s=frame.time_s,
            observations=observations,
        )

        totals["observations"] += len(observations)
        totals["assignments"] += len(result.assignments)
        totals["unmatched"] += len(
            result.unmatched_observation_indices
        )
        totals["tracks_created"] += len(result.new_track_ids)
        totals["confirmations"] += len(
            result.newly_confirmed_track_ids
        )
        totals["reactivations"] += len(
            result.reactivated_track_ids
        )
        totals["lost"] += len(
            result.newly_lost_track_ids
        )
        totals["retirements"] += len(
            result.retired_track_ids
        )
        totals["max_visible"] = max(
            totals["max_visible"],
            len(result.visible_track_ids),
        )

        ever_confirmed.update(
            result.newly_confirmed_track_ids
        )
        ever_confirmed.update(
            result.reactivated_track_ids
        )
        for snapshot in result.all_active_tracks:
            if snapshot.lifecycle == "confirmed":
                ever_confirmed.add(snapshot.track_id)

        for assignment in result.assignments:
            observation = observations[
                assignment.observation_index
            ]
            samples_by_track[
                assignment.track_id
            ].append(
                _sample(
                    track_id=assignment.track_id,
                    observation=observation,
                    observation_index=(
                        assignment.observation_index
                    ),
                )
            )

        if len(result.new_track_ids) != len(
            result.unmatched_observation_indices
        ):
            raise RuntimeError(
                "TrackManager birth/unmatched contract changed"
            )

        for track_id, observation_index in zip(
            result.new_track_ids,
            result.unmatched_observation_indices,
        ):
            observation = observations[observation_index]
            samples_by_track[track_id].append(
                _sample(
                    track_id=track_id,
                    observation=observation,
                    observation_index=observation_index,
                )
            )

    for values in samples_by_track.values():
        values.sort(
            key=lambda item: (
                item.time_s,
                item.frame,
            )
        )

    return ForwardPass(
        samples_by_track=dict(samples_by_track),
        ever_confirmed_track_ids=ever_confirmed,
        stats=ForwardPassStats(
            frames=len(frames),
            observations=totals["observations"],
            assignments=totals["assignments"],
            unmatched_observations=totals["unmatched"],
            tracks_created=totals["tracks_created"],
            confirmations=totals["confirmations"],
            reactivations=totals["reactivations"],
            lost_transitions=totals["lost"],
            retirements=totals["retirements"],
            max_visible=totals["max_visible"],
        ),
    )


def _quadratic_form(
    vector: np.ndarray,
    covariance: np.ndarray,
) -> float:
    try:
        solved = np.linalg.solve(
            covariance,
            vector,
        )
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(covariance) @ vector
    return float(vector.T @ solved)


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return (
        (1.0 - alpha) * ordered[lower]
        + alpha * ordered[upper]
    )


def calibrate_covariance_scale(
    preliminary: ForwardPass,
    *,
    config: OfflineContextConfig | None = None,
) -> CovarianceCalibration:
    """Conservatively widen R from the 95th-percentile residual tail.

    The old V1 median estimator was strongly biased toward the very dense
    centre of already-successful matches.  V2 never shrinks the geometric
    baseline and uses the upper residual tail, which is more relevant to
    association gating.
    """

    cfg = config or OfflineContextConfig()
    values: list[float] = []

    for track_id in preliminary.ever_confirmed_track_ids:
        samples = preliminary.samples_by_track.get(
            track_id,
            (),
        )
        if len(samples) < 3:
            continue

        for previous, current, following in zip(
            samples,
            samples[1:],
            samples[2:],
        ):
            dt_before = (
                current.time_s - previous.time_s
            )
            dt_after = (
                following.time_s - current.time_s
            )
            total = (
                following.time_s - previous.time_s
            )
            if (
                dt_before <= 0.0
                or dt_after <= 0.0
                or total <= 0.0
                or dt_before
                > cfg.calibration_max_triplet_gap_s
                or dt_after
                > cfg.calibration_max_triplet_gap_s
            ):
                continue

            alpha = dt_before / total
            interpolated = (
                (1.0 - alpha) * previous.position
                + alpha * following.position
            )
            residual = current.position - interpolated
            covariance = (
                current.covariance
                + ((1.0 - alpha) ** 2)
                * previous.covariance
                + (alpha ** 2)
                * following.covariance
            )
            covariance = 0.5 * (
                covariance + covariance.T
            )

            nis = _quadratic_form(
                residual,
                covariance,
            )
            if math.isfinite(nis) and nis >= 0.0:
                values.append(nis)

    if len(values) < cfg.calibration_min_triplets:
        return CovarianceCalibration(
            scale=1.0,
            triplet_count=len(values),
            nis_median=(
                statistics.median(values)
                if values
                else None
            ),
            nis_mean=(
                statistics.fmean(values)
                if values
                else None
            ),
            nis_p95=(
                _percentile(values, 0.95)
                if values
                else None
            ),
            raw_scale=None,
            status="insufficient_triplets",
        )

    p95 = float(_percentile(values, 0.95))
    raw_scale = p95 / CHI2_DF2_P95
    scale = min(
        cfg.covariance_scale_max,
        max(cfg.covariance_scale_min, raw_scale),
    )

    return CovarianceCalibration(
        scale=float(scale),
        triplet_count=len(values),
        nis_median=float(
            statistics.median(values)
        ),
        nis_mean=float(
            statistics.fmean(values)
        ),
        nis_p95=p95,
        raw_scale=float(raw_scale),
        status="calibrated",
    )


def _major_sigma(covariance: np.ndarray) -> float:
    covariance = 0.5 * (
        covariance + covariance.T
    )
    largest = max(
        0.0,
        float(
            np.max(
                np.linalg.eigvalsh(covariance)
            )
        ),
    )
    return math.sqrt(largest)


def _samples_are_locally_continuous(
    previous: OfflineSample,
    current: OfflineSample,
    *,
    config: OfflineContextConfig,
) -> bool:
    dt = current.time_s - previous.time_s
    if dt <= 0.0:
        return False
    if dt > config.tracklet_split_gap_s:
        return False

    distance = float(
        np.linalg.norm(
            current.position - previous.position
        )
    )
    uncertainty = (
        config.tracklet_break_uncertainty_sigma_multiplier
        * _major_sigma(
            previous.covariance
            + current.covariance
        )
    )
    radius = (
        config.tracklet_break_max_speed_mps * dt
        + config.tracklet_break_base_slack_m
        + uncertainty
    )
    return distance <= radius


def _robust_endpoint_state(
    samples: Sequence[OfflineSample],
    *,
    at_start: bool,
    config: OfflineContextConfig,
) -> tuple[np.ndarray, np.ndarray | None]:
    if not samples:
        raise ValueError("cannot fit empty samples")

    ordered = list(samples)
    anchor_time = (
        ordered[0].time_s
        if at_start
        else ordered[-1].time_s
    )

    if at_start:
        selected = [
            item
            for item in ordered
            if (
                item.time_s - anchor_time
                <= config.velocity_window_s + 1e-12
            )
        ]
    else:
        selected = [
            item
            for item in ordered
            if (
                anchor_time - item.time_s
                <= config.velocity_window_s + 1e-12
            )
        ]

    if len(selected) < config.velocity_min_observations:
        anchor = (
            ordered[0]
            if at_start
            else ordered[-1]
        )
        return anchor.position.copy(), None

    span = (
        selected[-1].time_s
        - selected[0].time_s
    )
    if span < config.velocity_min_span_s:
        anchor = (
            ordered[0]
            if at_start
            else ordered[-1]
        )
        return anchor.position.copy(), None

    slopes_x: list[float] = []
    slopes_z: list[float] = []
    for index, left in enumerate(selected):
        for right in selected[index + 1:]:
            dt = right.time_s - left.time_s
            if dt <= 1e-6:
                continue
            slopes_x.append(
                (right.x - left.x) / dt
            )
            slopes_z.append(
                (right.z - left.z) / dt
            )

    if not slopes_x:
        anchor = (
            ordered[0]
            if at_start
            else ordered[-1]
        )
        return anchor.position.copy(), None

    velocity = np.asarray(
        [
            statistics.median(slopes_x),
            statistics.median(slopes_z),
        ],
        dtype=np.float64,
    )

    intercepts = np.asarray(
        [
            item.position
            - velocity
            * (item.time_s - anchor_time)
            for item in selected
        ],
        dtype=np.float64,
    )
    position = np.median(
        intercepts,
        axis=0,
    )

    return (
        position.astype(np.float64),
        velocity.astype(np.float64),
    )


def _tracklet_is_anchor(
    *,
    observed_count: int,
    observed_span_s: float,
    coverage: float,
    start_velocity: np.ndarray | None,
    end_velocity: np.ndarray | None,
    config: OfflineContextConfig,
) -> tuple[bool, float | None]:
    speeds = [
        float(np.linalg.norm(value))
        for value in (
            start_velocity,
            end_velocity,
        )
        if value is not None
    ]
    max_speed = max(speeds) if speeds else None

    anchor = (
        observed_count
        >= config.anchor_min_observations
        and observed_span_s
        >= config.anchor_min_span_s
        and coverage
        >= config.anchor_min_coverage
        and start_velocity is not None
        and end_velocity is not None
        and (
            max_speed is None
            or max_speed
            <= config.anchor_max_endpoint_speed_mps
        )
    )
    return bool(anchor), max_speed


def build_tracklets(
    calibrated: ForwardPass,
    *,
    config: OfflineContextConfig | None = None,
) -> tuple[Tracklet, ...]:
    cfg = config or OfflineContextConfig()
    tracklets: list[Tracklet] = []
    next_id = 1

    for source_track_id in sorted(
        calibrated.samples_by_track
    ):
        source_samples = calibrated.samples_by_track[
            source_track_id
        ]
        if not source_samples:
            continue

        segments: list[list[OfflineSample]] = [[]]

        for sample in source_samples:
            if (
                segments[-1]
                and not _samples_are_locally_continuous(
                    segments[-1][-1],
                    sample,
                    config=cfg,
                )
            ):
                segments.append([])
            segments[-1].append(sample)

        for segment in segments:
            if not segment:
                continue

            start_position, start_velocity = (
                _robust_endpoint_state(
                    segment,
                    at_start=True,
                    config=cfg,
                )
            )
            end_position, end_velocity = (
                _robust_endpoint_state(
                    segment,
                    at_start=False,
                    config=cfg,
                )
            )

            frame_span = (
                segment[-1].frame
                - segment[0].frame
                + 1
            )
            coverage = (
                len(segment) / frame_span
                if frame_span > 0
                else 0.0
            )
            span = (
                segment[-1].time_s
                - segment[0].time_s
            )

            anchor, max_endpoint_speed = (
                _tracklet_is_anchor(
                    observed_count=len(segment),
                    observed_span_s=span,
                    coverage=coverage,
                    start_velocity=start_velocity,
                    end_velocity=end_velocity,
                    config=cfg,
                )
            )

            tracklets.append(
                Tracklet(
                    tracklet_id=next_id,
                    source_track_id=source_track_id,
                    source_confirmed=(
                        source_track_id
                        in calibrated.ever_confirmed_track_ids
                    ),
                    samples=tuple(segment),
                    start_position=start_position,
                    end_position=end_position,
                    start_velocity=start_velocity,
                    end_velocity=end_velocity,
                    mean_confidence=float(
                        statistics.fmean(
                            item.confidence
                            for item in segment
                        )
                    ),
                    frame_span=int(frame_span),
                    coverage=float(coverage),
                    anchor=anchor,
                    max_endpoint_speed_mps=(
                        float(max_endpoint_speed)
                        if max_endpoint_speed
                        is not None
                        else None
                    ),
                )
            )
            next_id += 1

    return tuple(tracklets)


def evaluate_tracklet_link(
    predecessor: Tracklet,
    successor: Tracklet,
    *,
    config: OfflineContextConfig | None = None,
) -> LinkCandidate | None:
    cfg = config or OfflineContextConfig()

    if not predecessor.anchor or not successor.anchor:
        return None
    if successor.start_time_s <= predecessor.end_time_s:
        return None

    gap_s = (
        successor.start_time_s
        - predecessor.end_time_s
    )
    if gap_s > cfg.linker_max_gap_s:
        return None

    if (
        gap_s
        > cfg.linker_long_gap_requires_velocity_s
        and (
            predecessor.end_velocity is None
            or successor.start_velocity is None
        )
    ):
        return None

    direct_distance = float(
        np.linalg.norm(
            successor.start_position
            - predecessor.end_position
        )
    )
    uncertainty = (
        cfg.linker_uncertainty_sigma_multiplier
        * _major_sigma(
            predecessor.samples[-1].covariance
            + successor.samples[0].covariance
        )
    )
    physical_radius = (
        cfg.linker_max_speed_mps * gap_s
        + cfg.linker_base_slack_m
        + uncertainty
    )
    if direct_distance > physical_radius:
        return None

    errors: list[float] = []

    if predecessor.end_velocity is not None:
        forward = (
            predecessor.end_position
            + predecessor.end_velocity * gap_s
        )
        errors.append(
            float(
                np.linalg.norm(
                    successor.start_position
                    - forward
                )
            )
        )

    if successor.start_velocity is not None:
        backward = (
            successor.start_position
            - successor.start_velocity * gap_s
        )
        errors.append(
            float(
                np.linalg.norm(
                    predecessor.end_position
                    - backward
                )
            )
        )

    if not errors:
        if (
            gap_s
            > cfg.linker_long_gap_requires_velocity_s
        ):
            return None
        symmetric_error = direct_distance
    else:
        symmetric_error = float(
            statistics.fmean(errors)
        )

    prediction_radius = (
        cfg.linker_base_slack_m
        + uncertainty
        + cfg.linker_prediction_growth_mps
        * gap_s
    )
    if symmetric_error > prediction_radius:
        return None

    velocity_difference = None
    if (
        predecessor.end_velocity is not None
        and successor.start_velocity is not None
    ):
        velocity_difference = float(
            np.linalg.norm(
                predecessor.end_velocity
                - successor.start_velocity
            )
        )

    normalization = max(
        0.25,
        cfg.linker_base_slack_m
        + uncertainty,
    )

    cost = symmetric_error / normalization
    cost += cfg.linker_gap_cost_weight * gap_s

    if velocity_difference is not None:
        cost += (
            cfg.linker_velocity_cost_weight
            * velocity_difference
        )

    if (
        predecessor.source_track_id
        == successor.source_track_id
    ):
        cost -= cfg.linker_same_source_bonus

    cost = max(0.0, float(cost))
    if cost >= cfg.linker_unmatched_cost:
        return None

    return LinkCandidate(
        predecessor_tracklet_id=(
            predecessor.tracklet_id
        ),
        successor_tracklet_id=(
            successor.tracklet_id
        ),
        gap_s=float(gap_s),
        direct_distance_m=direct_distance,
        physical_radius_m=float(physical_radius),
        symmetric_prediction_error_m=(
            symmetric_error
        ),
        prediction_radius_m=float(
            prediction_radius
        ),
        velocity_difference_mps=(
            velocity_difference
        ),
        cost=cost,
    )


def link_tracklets(
    tracklets: Sequence[Tracklet],
    *,
    config: OfflineContextConfig | None = None,
) -> TrackletLinkResult:
    cfg = config or OfflineContextConfig()
    anchors = tuple(
        item for item in tracklets
        if item.anchor
    )
    count = len(anchors)

    if count == 0:
        return TrackletLinkResult(
            links=(),
            accepted_candidates=(),
            candidate_count=0,
        )

    matrix = np.full(
        (count, count + count),
        _FORBIDDEN,
        dtype=np.float64,
    )
    candidates: dict[
        tuple[int, int],
        LinkCandidate,
    ] = {}

    for row, predecessor in enumerate(anchors):
        matrix[
            row,
            count + row,
        ] = cfg.linker_unmatched_cost

        for column, successor in enumerate(anchors):
            if row == column:
                continue

            candidate = evaluate_tracklet_link(
                predecessor,
                successor,
                config=cfg,
            )
            if candidate is None:
                continue

            matrix[row, column] = candidate.cost
            candidates[(row, column)] = candidate

    rows, columns = linear_sum_assignment(matrix)
    links: list[tuple[int, int]] = []
    accepted: list[LinkCandidate] = []

    for row, column in zip(
        rows.tolist(),
        columns.tolist(),
    ):
        if column >= count:
            continue

        candidate = candidates.get(
            (row, column)
        )
        if candidate is None:
            continue

        if candidate.cost >= cfg.linker_unmatched_cost:
            continue

        links.append(
            (
                candidate.predecessor_tracklet_id,
                candidate.successor_tracklet_id,
            )
        )
        accepted.append(candidate)

    links.sort()
    accepted.sort(
        key=lambda item: (
            item.predecessor_tracklet_id,
            item.successor_tracklet_id,
        )
    )

    return TrackletLinkResult(
        links=tuple(links),
        accepted_candidates=tuple(accepted),
        candidate_count=len(candidates),
    )


def _trajectory_chains(
    anchors: Sequence[Tracklet],
    links: Sequence[tuple[int, int]],
) -> list[list[Tracklet]]:
    by_id = {
        item.tracklet_id: item
        for item in anchors
    }
    successor = {
        left: right
        for left, right in links
    }
    predecessor = {
        right: left
        for left, right in links
    }

    starts = [
        item.tracklet_id
        for item in anchors
        if item.tracklet_id not in predecessor
    ]

    chains: list[list[Tracklet]] = []
    visited: set[int] = set()

    for start in sorted(starts):
        current = start
        chain: list[Tracklet] = []

        while current not in visited:
            visited.add(current)
            chain.append(by_id[current])

            if current not in successor:
                break
            current = successor[current]

        chains.append(chain)

    for item in anchors:
        if item.tracklet_id not in visited:
            chains.append([item])

    return chains


def _initial_velocity(
    samples: Sequence[OfflineSample],
    *,
    config: OfflineContextConfig,
) -> np.ndarray:
    _, velocity = _robust_endpoint_state(
        samples,
        at_start=True,
        config=config,
    )
    if velocity is None:
        return np.zeros(
            2,
            dtype=np.float64,
        )
    return velocity


def _rts_smooth(
    samples: Sequence[OfflineSample],
    *,
    kalman_config: KalmanConfig,
    context_config: OfflineContextConfig,
) -> tuple[SmoothedSample, ...]:
    if not samples:
        return ()

    ordered = sorted(
        samples,
        key=lambda item: (
            item.time_s,
            item.frame,
        ),
    )

    initial_velocity = _initial_velocity(
        ordered,
        config=context_config,
    )

    kalman = FieldKalmanFilter.from_measurement(
        ordered[0].position,
        ordered[0].covariance,
        initial_velocity=initial_velocity,
        config=kalman_config,
    )

    filtered_states = [
        kalman.state
    ]
    filtered_covariances = [
        kalman.covariance
    ]
    predicted_states: list[
        np.ndarray | None
    ] = [None]
    predicted_covariances: list[
        np.ndarray | None
    ] = [None]
    transitions: list[
        np.ndarray | None
    ] = [None]

    previous = ordered[0]

    for current in ordered[1:]:
        dt = (
            current.time_s
            - previous.time_s
        )
        if dt <= 0.0:
            raise ValueError(
                "trajectory samples must increase in time"
            )

        transition = (
            FieldKalmanFilter.transition_matrix(dt)
        )
        prediction = kalman.predict(dt)
        update = kalman.update(
            current.position,
            current.covariance,
        )

        transitions.append(transition)
        predicted_states.append(
            prediction.state.copy()
        )
        predicted_covariances.append(
            prediction.covariance.copy()
        )
        filtered_states.append(
            update.state.copy()
        )
        filtered_covariances.append(
            update.covariance.copy()
        )
        previous = current

    smoothed_states = [
        value.copy()
        for value in filtered_states
    ]
    smoothed_covariances = [
        value.copy()
        for value in filtered_covariances
    ]

    for index in range(
        len(ordered) - 2,
        -1,
        -1,
    ):
        transition = transitions[index + 1]
        predicted_state = (
            predicted_states[index + 1]
        )
        predicted_covariance = (
            predicted_covariances[index + 1]
        )

        assert transition is not None
        assert predicted_state is not None
        assert predicted_covariance is not None

        numerator = (
            filtered_covariances[index]
            @ transition.T
        )
        try:
            gain = np.linalg.solve(
                predicted_covariance.T,
                numerator.T,
            ).T
        except np.linalg.LinAlgError:
            gain = (
                numerator
                @ np.linalg.pinv(
                    predicted_covariance
                )
            )

        smoothed_states[index] = (
            filtered_states[index]
            + gain
            @ (
                smoothed_states[index + 1]
                - predicted_state
            )
        )
        smoothed_covariances[index] = (
            filtered_covariances[index]
            + gain
            @ (
                smoothed_covariances[index + 1]
                - predicted_covariance
            )
            @ gain.T
        )
        smoothed_covariances[index] = (
            0.5
            * (
                smoothed_covariances[index]
                + smoothed_covariances[index].T
            )
        )

    result: list[SmoothedSample] = []
    for sample, state, covariance in zip(
        ordered,
        smoothed_states,
        smoothed_covariances,
    ):
        result.append(
            SmoothedSample(
                frame=sample.frame,
                time_s=sample.time_s,
                x=float(state[0]),
                z=float(state[1]),
                vx=float(state[2]),
                vz=float(state[3]),
                covariance=covariance.copy(),
            )
        )

    return tuple(result)


def build_trajectories(
    tracklets: Sequence[Tracklet],
    link_result: TrackletLinkResult,
    *,
    config: OfflineContextConfig | None = None,
    kalman_config: KalmanConfig | None = None,
) -> tuple[OfflineTrajectory, ...]:
    cfg = config or OfflineContextConfig()
    kcfg = kalman_config or KalmanConfig()
    anchors = tuple(
        item for item in tracklets
        if item.anchor
    )

    chains = _trajectory_chains(
        anchors,
        link_result.links,
    )
    trajectories: list[OfflineTrajectory] = []

    for trajectory_id, chain in enumerate(
        chains,
        start=1,
    ):
        samples = sorted(
            (
                sample
                for tracklet in chain
                for sample in tracklet.samples
            ),
            key=lambda item: (
                item.time_s,
                item.frame,
            ),
        )

        deduped: list[OfflineSample] = []
        seen_frames: set[int] = set()
        for sample in samples:
            if sample.frame in seen_frames:
                continue
            seen_frames.add(sample.frame)
            deduped.append(sample)
        samples = deduped

        if not samples:
            continue

        gaps = [
            right.time_s - left.time_s
            for left, right in zip(
                samples,
                samples[1:],
            )
        ]
        max_gap = max(gaps) if gaps else 0.0
        span = (
            samples[-1].time_s
            - samples[0].time_s
        )
        count = len(samples)

        render_eligible = (
            count >= cfg.final_min_observations
            and span >= cfg.final_min_span_s
        )

        smoothed = _rts_smooth(
            samples,
            kalman_config=kcfg,
            context_config=cfg,
        )

        trajectories.append(
            OfflineTrajectory(
                trajectory_id=trajectory_id,
                tracklet_ids=tuple(
                    item.tracklet_id
                    for item in chain
                ),
                source_track_ids=tuple(
                    item.source_track_id
                    for item in chain
                ),
                samples=tuple(samples),
                smoothed_samples=smoothed,
                mean_confidence=float(
                    statistics.fmean(
                        item.confidence
                        for item in samples
                    )
                ),
                observed_count=count,
                observed_span_s=float(span),
                fragment_count=len(chain),
                max_observed_gap_s=float(max_gap),
                render_eligible=bool(
                    render_eligible
                ),
            )
        )

    return tuple(trajectories)


def _hermite(
    p0: np.ndarray,
    v0: np.ndarray,
    p1: np.ndarray,
    v1: np.ndarray,
    *,
    duration: float,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    s = float(alpha)

    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2

    position = (
        h00 * p0
        + h10 * duration * v0
        + h01 * p1
        + h11 * duration * v1
    )

    dh00 = 6 * s**2 - 6 * s
    dh10 = 3 * s**2 - 4 * s + 1
    dh01 = -6 * s**2 + 6 * s
    dh11 = 3 * s**2 - 2 * s

    velocity = (
        dh00 * p0
        + dh10 * duration * v0
        + dh01 * p1
        + dh11 * duration * v1
    ) / duration

    return position, velocity


def _trajectory_state_at(
    trajectory: OfflineTrajectory,
    *,
    frame: int,
    time_s: float,
    config: OfflineContextConfig,
) -> OfflineFrameTrack | None:
    if not trajectory.render_eligible:
        return None

    values = trajectory.smoothed_samples
    times = [
        item.time_s
        for item in values
    ]
    index = bisect_left(
        times,
        time_s,
    )

    state = None
    position = None
    velocity = None

    candidates = []
    if index < len(values):
        candidates.append(values[index])
    if index > 0:
        candidates.append(values[index - 1])

    for candidate in candidates:
        if abs(
            candidate.time_s - time_s
        ) <= 1e-6:
            state = "observed"
            position = np.asarray(
                [candidate.x, candidate.z],
                dtype=np.float64,
            )
            velocity = np.asarray(
                [candidate.vx, candidate.vz],
                dtype=np.float64,
            )
            break

    if (
        state is None
        and 0 < index < len(values)
    ):
        before = values[index - 1]
        after = values[index]
        gap = (
            after.time_s
            - before.time_s
        )

        if (
            0.0 < gap
            <= config.render_gap_s
        ):
            alpha = (
                time_s - before.time_s
            ) / gap
            if 0.0 <= alpha <= 1.0:
                position, velocity = _hermite(
                    np.asarray(
                        [before.x, before.z],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [before.vx, before.vz],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [after.x, after.z],
                        dtype=np.float64,
                    ),
                    np.asarray(
                        [after.vx, after.vz],
                        dtype=np.float64,
                    ),
                    duration=gap,
                    alpha=alpha,
                )
                state = "interpolated"

    if (
        state is None
        and config.endpoint_extrapolation_s > 0.0
        and index == 0
    ):
        first = values[0]
        delta = (
            first.time_s - time_s
        )
        if (
            0.0 < delta
            <= config.endpoint_extrapolation_s
        ):
            velocity = np.asarray(
                [first.vx, first.vz],
                dtype=np.float64,
            )
            position = (
                np.asarray(
                    [first.x, first.z],
                    dtype=np.float64,
                )
                - velocity * delta
            )
            state = "extrapolated"

    if (
        state is None
        and config.endpoint_extrapolation_s > 0.0
        and index >= len(values)
    ):
        last = values[-1]
        delta = (
            time_s - last.time_s
        )
        if (
            0.0 < delta
            <= config.endpoint_extrapolation_s
        ):
            velocity = np.asarray(
                [last.vx, last.vz],
                dtype=np.float64,
            )
            position = (
                np.asarray(
                    [last.x, last.z],
                    dtype=np.float64,
                )
                + velocity * delta
            )
            state = "extrapolated"

    if (
        state is None
        or position is None
        or velocity is None
    ):
        return None

    state_bonus = {
        "observed": 0.60,
        "interpolated": 0.20,
        "extrapolated": 0.0,
    }[state]

    support = (
        math.log1p(
            trajectory.observed_count
        )
        + 0.50
        * math.log1p(
            max(
                0.0,
                trajectory.observed_span_s,
            )
        )
        + 0.35
        * trajectory.fragment_count
        + 0.50
        * trajectory.mean_confidence
        + state_bonus
    )

    return OfflineFrameTrack(
        trajectory_id=trajectory.trajectory_id,
        frame=int(frame),
        time_s=float(time_s),
        state=state,
        x=float(position[0]),
        z=float(position[1]),
        vx=float(velocity[0]),
        vz=float(velocity[1]),
        support_score=float(support),
        observed_count=(
            trajectory.observed_count
        ),
        fragment_count=(
            trajectory.fragment_count
        ),
        mean_confidence=(
            trajectory.mean_confidence
        ),
    )


def reconstruct_frames(
    frames: Sequence[Any],
    trajectories: Sequence[OfflineTrajectory],
    *,
    config: OfflineContextConfig | None = None,
) -> dict[int, tuple[OfflineFrameTrack, ...]]:
    cfg = config or OfflineContextConfig()
    result: dict[
        int,
        tuple[OfflineFrameTrack, ...],
    ] = {}

    for frame in frames:
        candidates: list[
            OfflineFrameTrack
        ] = []

        for trajectory in trajectories:
            value = _trajectory_state_at(
                trajectory,
                frame=frame.frame,
                time_s=frame.time_s,
                config=cfg,
            )
            if value is not None:
                candidates.append(value)

        candidates.sort(
            key=lambda item: (
                -item.support_score,
                item.trajectory_id,
            )
        )

        # This is a CAP, not a target.  If only 19 trajectories have
        # bidirectionally supported state at this frame, exactly 19 are shown.
        result[int(frame.frame)] = tuple(
            candidates[
                : cfg.max_visible_tracks
            ]
        )

    return result


def process_offline_context(
    frames: Sequence[Any],
    *,
    context_config: OfflineContextConfig | None = None,
    manager_config: TrackManagerConfig | None = None,
    association_config: AssociationConfig | None = None,
    kalman_config: KalmanConfig | None = None,
) -> OfflineContextResult:
    cfg = context_config or OfflineContextConfig()

    manager_cfg = (
        manager_config
        or TrackManagerConfig(
            max_visible_tracks=(
                cfg.max_visible_tracks
            )
        )
    )
    association_cfg = (
        association_config
        or AssociationConfig()
    )
    kalman_cfg = (
        kalman_config
        or KalmanConfig()
    )

    preliminary = run_forward_pass(
        frames,
        covariance_scale=1.0,
        manager_config=manager_cfg,
        association_config=association_cfg,
        kalman_config=kalman_cfg,
    )

    calibration = calibrate_covariance_scale(
        preliminary,
        config=cfg,
    )

    calibrated = run_forward_pass(
        frames,
        covariance_scale=calibration.scale,
        manager_config=manager_cfg,
        association_config=association_cfg,
        kalman_config=kalman_cfg,
    )

    tracklets = build_tracklets(
        calibrated,
        config=cfg,
    )
    anchors = tuple(
        item for item in tracklets
        if item.anchor
    )
    weak = tuple(
        item for item in tracklets
        if not item.anchor
    )

    links = link_tracklets(
        anchors,
        config=cfg,
    )
    trajectories = build_trajectories(
        anchors,
        links,
        config=cfg,
        kalman_config=kalman_cfg,
    )
    frame_tracks = reconstruct_frames(
        frames,
        trajectories,
        config=cfg,
    )

    return OfflineContextResult(
        preliminary=preliminary,
        calibrated=calibrated,
        calibration=calibration,
        tracklets=tracklets,
        anchor_tracklets=anchors,
        weak_tracklets=weak,
        link_result=links,
        trajectories=trajectories,
        frame_tracks=frame_tracks,
    )


__all__ = [
    "CovarianceCalibration",
    "ForwardPass",
    "ForwardPassStats",
    "LinkCandidate",
    "OfflineContextConfig",
    "OfflineContextResult",
    "OfflineFrameTrack",
    "OfflineSample",
    "OfflineTrajectory",
    "ScaledObservation",
    "SmoothedSample",
    "Tracklet",
    "TrackletLinkResult",
    "build_tracklets",
    "build_trajectories",
    "calibrate_covariance_scale",
    "evaluate_tracklet_link",
    "link_tracklets",
    "process_offline_context",
    "reconstruct_frames",
    "run_forward_pass",
]
