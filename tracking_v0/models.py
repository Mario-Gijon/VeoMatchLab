from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Hashable, Mapping

import numpy as np


class TrackLifecycle(str, Enum):
    """Lifecycle of a temporal track."""

    TENTATIVE = "tentative"
    ACTIVE = "active"
    COASTING = "coasting"
    LOST = "lost"


class PositionKind(str, Enum):
    """Origin of a position in a trajectory."""

    OBSERVED = "observed"
    PREDICTED = "predicted"
    INTERPOLATED = "interpolated"


@dataclass(slots=True)
class TrackingMeasurement:
    """
    Minimal input consumed by the global tracker.

    This intentionally does not depend on the current detector/association
    implementation. Later, the pipeline only needs a small adapter from the
    existing PersonMeasurement object to this structure.
    """

    measurement_id: Hashable
    frame_index: int
    timestamp: float
    x: float
    z: float
    confidence: float = 1.0
    position_covariance: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")
        if not np.isfinite(self.x) or not np.isfinite(self.z):
            raise ValueError("x and z must be finite")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")

        if self.position_covariance is not None:
            cov = np.asarray(self.position_covariance, dtype=np.float64)
            if cov.shape != (2, 2):
                raise ValueError("position_covariance must have shape (2, 2)")
            if not np.all(np.isfinite(cov)):
                raise ValueError("position_covariance must contain finite values")
            if not np.allclose(cov, cov.T, atol=1e-9):
                raise ValueError("position_covariance must be symmetric")
            if np.any(np.linalg.eigvalsh(cov) < -1e-9):
                raise ValueError("position_covariance must be positive semidefinite")
            self.position_covariance = cov


@dataclass(slots=True, frozen=True)
class TrackSample:
    """One temporal sample belonging to a track."""

    track_id: int
    frame_index: int
    timestamp: float
    x: float
    z: float
    vx: float
    vz: float
    lifecycle: TrackLifecycle
    position_kind: PositionKind
    measurement_id: Hashable | None
    confidence: float | None


@dataclass(slots=True, frozen=True)
class TrackSnapshot:
    """Current state of a track returned for a processed frame."""

    track_id: int
    frame_index: int
    timestamp: float
    x: float
    z: float
    vx: float
    vz: float
    lifecycle: TrackLifecycle
    position_kind: PositionKind
    measurement_id: Hashable | None
    confidence: float | None
    age_seconds: float
    missed_seconds: float
    hits: int

    @property
    def is_confirmed(self) -> bool:
        return self.lifecycle in (TrackLifecycle.ACTIVE, TrackLifecycle.COASTING)


@dataclass(slots=True, frozen=True)
class TrackingFrame:
    """Output produced after processing one timestamp/frame."""

    frame_index: int
    timestamp: float
    tracks: tuple[TrackSnapshot, ...]
    matched_pairs: tuple[tuple[int, Hashable], ...]
    unmatched_track_ids: tuple[int, ...]
    unmatched_measurement_ids: tuple[Hashable, ...]


@dataclass(slots=True)
class TrackerConfig:
    """
    Parameters for Tracking V1.

    Defaults are intentionally conservative and are expected to be tuned on a
    20-30 second diagnostic clip before processing complete matches.
    """

    # Track lifecycle.
    min_confirm_hits: int = 3
    tentative_max_missed_seconds: float = 0.50
    max_coast_seconds: float = 3.0

    # Constant-velocity Kalman filter.
    process_acceleration_std: float = 3.5
    default_measurement_std: float = 0.85
    initial_position_std: float = 1.25
    initial_velocity_std: float = 5.0

    # Statistical gating. 9.21 ~= chi-square 99% gate for 2 dimensions.
    mahalanobis_gate_squared: float = 9.21

    # Additional physical gate. It grows while a track is unobserved.
    base_match_distance_m: float = 4.0
    match_distance_growth_m_per_s: float = 4.0
    max_reacquire_distance_m: float = 18.0

    # Soft association terms.
    euclidean_cost_weight: float = 0.08
    velocity_change_cost_weight: float = 0.025

    # Safety limit applied after measurement updates.
    max_speed_m_s: float = 12.5

    # Confidence can slightly alter observation covariance. This is deliberately
    # mild because detector confidence is not a calibrated geometry error.
    min_confidence_for_noise_scaling: float = 0.25
    max_measurement_noise_scale: float = 2.0

    def __post_init__(self) -> None:
        if self.min_confirm_hits < 1:
            raise ValueError("min_confirm_hits must be >= 1")
        if self.tentative_max_missed_seconds < 0:
            raise ValueError("tentative_max_missed_seconds must be >= 0")
        if self.max_coast_seconds <= 0:
            raise ValueError("max_coast_seconds must be > 0")
        if self.process_acceleration_std <= 0:
            raise ValueError("process_acceleration_std must be > 0")
        if self.default_measurement_std <= 0:
            raise ValueError("default_measurement_std must be > 0")
        if self.initial_position_std <= 0:
            raise ValueError("initial_position_std must be > 0")
        if self.initial_velocity_std <= 0:
            raise ValueError("initial_velocity_std must be > 0")
        if self.mahalanobis_gate_squared <= 0:
            raise ValueError("mahalanobis_gate_squared must be > 0")
        if self.base_match_distance_m <= 0:
            raise ValueError("base_match_distance_m must be > 0")
        if self.match_distance_growth_m_per_s < 0:
            raise ValueError("match_distance_growth_m_per_s must be >= 0")
        if self.max_reacquire_distance_m < self.base_match_distance_m:
            raise ValueError(
                "max_reacquire_distance_m must be >= base_match_distance_m"
            )
        if self.max_speed_m_s <= 0:
            raise ValueError("max_speed_m_s must be > 0")
