"""Global association of predicted field tracks to metric observations.

The association is deliberately field-centric rather than bbox-centric.

For each track/observation pair V1 applies:
1. a hard physical displacement gate from the last observed field position,
2. a Mahalanobis gate using the predicted Kalman state and the observation's
   own Veo-specific X/Z covariance,
3. Hungarian global assignment over the remaining finite costs.

No appearance, team colour, jersey number or pixel IoU is used in V1.
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
    """Association gates and cost weights.

    mahalanobis_gate_squared:
        Hard statistical gate.  16 is deliberately permissive for V1 while
        still rejecting wildly incompatible candidates.

    max_speed_mps:
        Human motion allowance used by the hard displacement gate.

    position_slack_m:
        Additional allowance for detector/projection noise.  This is important
        at ~30 FPS, where dividing a sub-metre observation error by one frame
        would otherwise look like an impossible instantaneous speed.

    euclidean_cost_weight:
        Small tie-breaker in metres on top of the Mahalanobis cost.
    """

    mahalanobis_gate_squared: float = 16.0
    max_speed_mps: float = 12.5
    position_slack_m: float = 1.50
    euclidean_cost_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "mahalanobis_gate_squared",
            "max_speed_mps",
            "position_slack_m",
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


def physical_gate_radius(
    track: Track,
    observation: Any,
    *,
    config: AssociationConfig,
) -> float:
    """Maximum displacement from the track's last actual observation."""

    elapsed = float(observation.time_s) - float(track.last_observed_time_s)
    if not math.isfinite(elapsed) or elapsed <= 0.0:
        raise ValueError(
            f"observation time must be after track {track.track_id}'s "
            "last observed time"
        )

    return float(
        config.position_slack_m
        + config.max_speed_mps * elapsed
    )


def evaluate_pair(
    track: Track,
    observation: Any,
    *,
    track_index: int,
    observation_index: int,
    config: AssociationConfig,
) -> PairDiagnostics:
    """Evaluate one pair after the track has been predicted to this frame."""

    obs_position = _observation_position(observation)

    predicted_distance = float(
        np.linalg.norm(obs_position - track.predicted_position)
    )
    displacement_from_last_observed = float(
        np.linalg.norm(obs_position - track.last_observed_position)
    )
    physical_radius = physical_gate_radius(
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
    """Globally associate already-predicted tracks to this frame's observations.

    Preconditions:
    - every track has already received predict_to(frame=..., time_s=...)
    - every observation belongs to that same target frame/time
    """

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

        # Hungarian always returns an assignment when a matrix entry exists,
        # including our forbidden sentinel.  Do not accept those pairs.
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

    unmatched_tracks = tuple(
        index
        for index in range(track_count)
        if index not in matched_tracks
    )
    unmatched_observations = tuple(
        index
        for index in range(observation_count)
        if index not in matched_observations
    )

    return AssociationResult(
        matches=tuple(matches),
        unmatched_track_indices=unmatched_tracks,
        unmatched_observation_indices=unmatched_observations,
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
    "physical_gate_radius",
]
