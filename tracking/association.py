"""Tracking association V2 for VeoMatchLab.

V2 keeps the field-centric, one-to-one Hungarian assignment from V1 but
replaces the fixed 1.5 m physical slack with a Veo-aware uncertainty term.

Important:
- proximity never means "same person";
- two observations can remain two independent tracks even when their X/Z
  positions are extremely close;
- the association only answers whether one concrete observation is compatible
  with one concrete predicted track;
- lifecycle priority (CONFIRMED -> LOST -> TENTATIVE) is implemented by
  tracking.track_manager, not by merging nearby tracks here.

The user-facing workflow keeps stable imports.  This revision is delivered as
association_V2.py, but when promoted it should replace tracking/association.py.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from tracking.track import Track


_FORBIDDEN_COST = 1e12


@dataclass(frozen=True)
class AssociationConfig:
    """Association gates and costs.

    mahalanobis_gate_squared:
        Statistical compatibility gate using Kalman uncertainty + each
        observation's own measurement covariance.

    max_speed_mps:
        Hard human-motion allowance measured from the last actual observation.

    position_slack_m:
        Small fixed numerical/localisation floor retained for backward API
        compatibility with V1.  V2's default is intentionally small.  It is no
        longer the main source of tolerance.

    uncertainty_sigma_multiplier:
        Adds k-sigma uncertainty from the *difference* between the previous and
        current Veo measurements.  This makes the physical gate tighter in
        well-conditioned areas and wider where the camera geometry is noisy.

    euclidean_cost_weight:
        Small tie-breaker on metric distance after the hard gates.
    """

    mahalanobis_gate_squared: float = 16.0
    max_speed_mps: float = 12.5
    position_slack_m: float = 0.10
    uncertainty_sigma_multiplier: float = 3.0
    euclidean_cost_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "mahalanobis_gate_squared",
            "max_speed_mps",
            "position_slack_m",
            "uncertainty_sigma_multiplier",
            "euclidean_cost_weight",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if value < 0.0:
                raise ValueError(f"{name} must be >= 0")

        if self.mahalanobis_gate_squared <= 0.0:
            raise ValueError("mahalanobis_gate_squared must be > 0")
        if self.max_speed_mps <= 0.0:
            raise ValueError("max_speed_mps must be > 0")


@dataclass(frozen=True)
class PairDiagnostics:
    track_index: int
    observation_index: int
    allowed: bool
    reason: str

    predicted_distance_m: float
    displacement_from_last_observed_m: float

    motion_radius_m: float
    uncertainty_slack_m: float
    physical_radius_m: float

    mahalanobis_squared: float | None
    cost: float | None


@dataclass(frozen=True)
class AssociationMatch:
    track_index: int
    observation_index: int
    cost: float
    mahalanobis_squared: float
    predicted_distance_m: float


@dataclass(frozen=True)
class AssociationResult:
    matches: tuple[AssociationMatch, ...]
    unmatched_track_indices: tuple[int, ...]
    unmatched_observation_indices: tuple[int, ...]

    pair_diagnostics: tuple[PairDiagnostics, ...]
    cost_matrix: np.ndarray

    @property
    def match_count(self) -> int:
        return len(self.matches)


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

    result = 0.5 * (result + result.T)
    eigenvalues = np.linalg.eigvalsh(result)
    if float(np.min(eigenvalues)) < -1e-10:
        raise ValueError("observation covariance must be positive semidefinite")
    return result


def _major_sigma(covariance: np.ndarray) -> float:
    """1-sigma length along the major axis of a 2D covariance."""
    covariance = 0.5 * (covariance + covariance.T)
    largest_variance = max(0.0, float(np.max(np.linalg.eigvalsh(covariance))))
    return math.sqrt(largest_variance)


def measurement_difference_sigma(
    previous_observation: Any,
    current_observation: Any,
) -> float:
    """Major-axis sigma for the difference of two noisy X/Z measurements.

    The two measurement errors are treated as independent for this *gate
    tolerance* calculation, so covariance of the difference is R_prev + R_now.

    This is not sensor fusion and does not claim Panorama/Ultra independence.
    It only converts the uncertainty already attached to two observations into
    a physically interpretable slack.
    """

    previous_covariance = _observation_covariance(previous_observation)
    current_covariance = _observation_covariance(current_observation)
    return _major_sigma(previous_covariance + current_covariance)


def physical_gate_components(
    track: Track,
    observation: Any,
    *,
    config: AssociationConfig,
) -> tuple[float, float, float]:
    """Return motion radius, uncertainty slack and total physical radius."""

    elapsed = float(observation.time_s) - float(track.last_observed_time_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(
            f"observation time must be after track {track.track_id}'s "
            "last observed time"
        )

    motion_radius = config.max_speed_mps * elapsed

    difference_sigma = measurement_difference_sigma(
        track.last_observation,
        observation,
    )
    uncertainty_slack = (
        config.position_slack_m
        + config.uncertainty_sigma_multiplier * difference_sigma
    )

    return (
        float(motion_radius),
        float(uncertainty_slack),
        float(motion_radius + uncertainty_slack),
    )


def physical_gate_radius(
    track: Track,
    observation: Any,
    *,
    config: AssociationConfig,
) -> float:
    """Backward-compatible convenience wrapper returning total radius."""
    return physical_gate_components(track, observation, config=config)[2]


def evaluate_pair(
    track: Track,
    observation: Any,
    *,
    track_index: int,
    observation_index: int,
    config: AssociationConfig,
) -> PairDiagnostics:
    """Evaluate one candidate pair after the track is predicted to this frame."""

    obs_position = _observation_position(observation)

    predicted_distance = float(
        np.linalg.norm(obs_position - track.predicted_position)
    )
    displacement_from_last_observed = float(
        np.linalg.norm(obs_position - track.last_observed_position)
    )

    motion_radius, uncertainty_slack, physical_radius = physical_gate_components(
        track,
        observation,
        config=config,
    )

    if displacement_from_last_observed > physical_radius:
        return PairDiagnostics(
            track_index=track_index,
            observation_index=observation_index,
            allowed=False,
            reason="physical_gate",
            predicted_distance_m=predicted_distance,
            displacement_from_last_observed_m=displacement_from_last_observed,
            motion_radius_m=motion_radius,
            uncertainty_slack_m=uncertainty_slack,
            physical_radius_m=physical_radius,
            mahalanobis_squared=None,
            cost=None,
        )

    d2 = float(track.mahalanobis_squared(observation))

    if not math.isfinite(d2):
        return PairDiagnostics(
            track_index=track_index,
            observation_index=observation_index,
            allowed=False,
            reason="non_finite_mahalanobis",
            predicted_distance_m=predicted_distance,
            displacement_from_last_observed_m=displacement_from_last_observed,
            motion_radius_m=motion_radius,
            uncertainty_slack_m=uncertainty_slack,
            physical_radius_m=physical_radius,
            mahalanobis_squared=d2,
            cost=None,
        )

    if d2 > config.mahalanobis_gate_squared:
        return PairDiagnostics(
            track_index=track_index,
            observation_index=observation_index,
            allowed=False,
            reason="mahalanobis_gate",
            predicted_distance_m=predicted_distance,
            displacement_from_last_observed_m=displacement_from_last_observed,
            motion_radius_m=motion_radius,
            uncertainty_slack_m=uncertainty_slack,
            physical_radius_m=physical_radius,
            mahalanobis_squared=d2,
            cost=None,
        )

    cost = float(
        d2
        + config.euclidean_cost_weight * predicted_distance
    )

    return PairDiagnostics(
        track_index=track_index,
        observation_index=observation_index,
        allowed=True,
        reason="allowed",
        predicted_distance_m=predicted_distance,
        displacement_from_last_observed_m=displacement_from_last_observed,
        motion_radius_m=motion_radius,
        uncertainty_slack_m=uncertainty_slack,
        physical_radius_m=physical_radius,
        mahalanobis_squared=d2,
        cost=cost,
    )


def associate(
    tracks: Sequence[Track],
    observations: Sequence[Any],
    *,
    config: AssociationConfig | None = None,
    include_pair_diagnostics: bool = False,
) -> AssociationResult:
    """Hungarian one-to-one association for one lifecycle stage."""

    cfg = config or AssociationConfig()

    track_count = len(tracks)
    observation_count = len(observations)

    if track_count == 0:
        return AssociationResult(
            matches=(),
            unmatched_track_indices=(),
            unmatched_observation_indices=tuple(range(observation_count)),
            pair_diagnostics=(),
            cost_matrix=np.empty((0, observation_count), dtype=np.float64),
        )

    if observation_count == 0:
        return AssociationResult(
            matches=(),
            unmatched_track_indices=tuple(range(track_count)),
            unmatched_observation_indices=(),
            pair_diagnostics=(),
            cost_matrix=np.empty((track_count, 0), dtype=np.float64),
        )

    cost_matrix = np.full(
        (track_count, observation_count),
        _FORBIDDEN_COST,
        dtype=np.float64,
    )
    diagnostics: list[PairDiagnostics] = []

    for track_index, track in enumerate(tracks):
        for observation_index, observation in enumerate(observations):
            pair = evaluate_pair(
                track,
                observation,
                track_index=track_index,
                observation_index=observation_index,
                config=cfg,
            )

            if include_pair_diagnostics or pair.allowed:
                diagnostics.append(pair)

            if pair.allowed:
                assert pair.cost is not None
                cost_matrix[track_index, observation_index] = pair.cost

    row_indices, column_indices = linear_sum_assignment(cost_matrix)

    matches: list[AssociationMatch] = []
    matched_tracks: set[int] = set()
    matched_observations: set[int] = set()

    for track_index, observation_index in zip(
        row_indices.tolist(),
        column_indices.tolist(),
    ):
        cost = float(cost_matrix[track_index, observation_index])

        if cost >= _FORBIDDEN_COST * 0.5:
            continue

        pair = evaluate_pair(
            tracks[track_index],
            observations[observation_index],
            track_index=track_index,
            observation_index=observation_index,
            config=cfg,
        )
        if not pair.allowed or pair.mahalanobis_squared is None:
            continue

        matches.append(
            AssociationMatch(
                track_index=track_index,
                observation_index=observation_index,
                cost=cost,
                mahalanobis_squared=float(pair.mahalanobis_squared),
                predicted_distance_m=float(pair.predicted_distance_m),
            )
        )
        matched_tracks.add(track_index)
        matched_observations.add(observation_index)

    matches.sort(key=lambda item: item.track_index)

    return AssociationResult(
        matches=tuple(matches),
        unmatched_track_indices=tuple(
            index for index in range(track_count)
            if index not in matched_tracks
        ),
        unmatched_observation_indices=tuple(
            index for index in range(observation_count)
            if index not in matched_observations
        ),
        pair_diagnostics=tuple(diagnostics),
        cost_matrix=cost_matrix.copy(),
    )


__all__ = [
    "AssociationConfig",
    "AssociationMatch",
    "AssociationResult",
    "PairDiagnostics",
    "associate",
    "evaluate_pair",
    "measurement_difference_sigma",
    "physical_gate_components",
    "physical_gate_radius",
]
