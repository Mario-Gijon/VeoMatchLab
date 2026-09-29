from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import hypot, isfinite, sqrt
from statistics import median
from typing import Mapping, Sequence

import numpy as np

from .models import PositionKind, TrackSample


@dataclass(slots=True, frozen=True)
class SmoothingConfig:
    """
    Offline constant-velocity Kalman + Rauch-Tung-Striebel smoother.

    The forward tracker only knows the past. This stage reruns each already
    resolved participant trajectory offline and then performs an RTS backward
    pass so every state can use both past and future observations.

    Only OBSERVED samples are treated as measurements. PREDICTED and
    INTERPOLATED positions are never fed back as measurements.

    This stage does NOT create 30 FPS samples. Interpolation is a later stage.
    """

    process_acceleration_std_m_s2: float = 3.5
    measurement_std_m: float = 0.85

    initial_position_std_m: float = 1.25
    initial_velocity_std_m_s: float = 5.0

    min_dt_seconds: float = 1e-4

    # Confidence can modestly alter measurement noise without letting detector
    # confidence dominate geometry. At confidence=1.0 the base std is used.
    use_confidence_weighting: bool = True
    confidence_floor: float = 0.35
    confidence_ceiling: float = 1.0
    max_measurement_std_scale: float = 1.60

    def __post_init__(self) -> None:
        positive = (
            self.process_acceleration_std_m_s2,
            self.measurement_std_m,
            self.initial_position_std_m,
            self.initial_velocity_std_m_s,
            self.min_dt_seconds,
            self.max_measurement_std_scale,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("SmoothingConfig positive parameters must be > 0")

        if not 0.0 < self.confidence_floor <= 1.0:
            raise ValueError("confidence_floor must be in (0, 1]")

        if not self.confidence_floor <= self.confidence_ceiling <= 1.0:
            raise ValueError(
                "confidence_ceiling must be in [confidence_floor, 1]"
            )

        if self.max_measurement_std_scale < 1.0:
            raise ValueError("max_measurement_std_scale must be >= 1")


@dataclass(slots=True, frozen=True)
class TrackSmoothingMetrics:
    track_id: int
    sample_count: int
    observed_count: int
    predicted_count: int
    interpolated_count: int

    first_timestamp_s: float
    last_timestamp_s: float
    duration_s: float
    median_sample_period_s: float
    longest_sample_gap_s: float

    observed_rms_correction_m: float
    observed_median_correction_m: float
    observed_max_correction_m: float

    predicted_rms_correction_m: float
    predicted_max_correction_m: float

    raw_max_speed_m_s: float
    smoothed_max_speed_m_s: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class SmoothingReport:
    tracks: Mapping[int, TrackSmoothingMetrics]

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tracks))

    def to_dict(self) -> dict[str, object]:
        return {
            "track_ids": list(self.track_ids),
            "tracks": {
                str(track_id): metrics.to_dict()
                for track_id, metrics in sorted(self.tracks.items())
            },
        }


def smooth_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    config: SmoothingConfig | None = None,
) -> tuple[dict[int, tuple[TrackSample, ...]], SmoothingReport]:
    """
    Smooth every trajectory independently with a variable-dt RTS smoother.

    Input and output contain the same track IDs and the same sample timestamps.
    PositionKind, lifecycle, measurement_id and confidence are preserved.

    Only x/z/vx/vz are replaced by their smoothed estimates.
    """
    cfg = config or SmoothingConfig()

    smoothed: dict[int, tuple[TrackSample, ...]] = {}
    reports: dict[int, TrackSmoothingMetrics] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(track_id, raw_samples)
        if not samples:
            continue

        output = smooth_trajectory(
            samples,
            config=cfg,
        )
        smoothed[track_id] = output
        reports[track_id] = _metrics(
            original=samples,
            smoothed=output,
        )

    return smoothed, SmoothingReport(tracks=reports)


def smooth_trajectory(
    samples: Sequence[TrackSample],
    *,
    config: SmoothingConfig | None = None,
) -> tuple[TrackSample, ...]:
    """
    Run variable-dt Kalman filtering followed by an RTS backward pass.

    State:
        [x, z, vx, vz]

    Observation:
        [x, z]

    The sequence may contain irregular time gaps. A later interpolation stage
    can fill those gaps after smoothing has made the endpoint states mutually
    consistent.
    """
    cfg = config or SmoothingConfig()

    if not samples:
        return ()

    track_id = samples[0].track_id
    ordered = _validated_samples(track_id, samples)

    n = len(ordered)
    if n == 1:
        return ordered

    filtered_state: list[np.ndarray] = [
        np.zeros(4, dtype=np.float64)
        for _ in range(n)
    ]
    filtered_cov: list[np.ndarray] = [
        np.zeros((4, 4), dtype=np.float64)
        for _ in range(n)
    ]

    predicted_state: list[np.ndarray] = [
        np.zeros(4, dtype=np.float64)
        for _ in range(n)
    ]
    predicted_cov: list[np.ndarray] = [
        np.zeros((4, 4), dtype=np.float64)
        for _ in range(n)
    ]

    # transition_to[k] maps state k-1 -> k. Index 0 is unused identity.
    transition_to: list[np.ndarray] = [
        np.eye(4, dtype=np.float64)
        for _ in range(n)
    ]

    first = ordered[0]
    x0 = np.array(
        [first.x, first.z, first.vx, first.vz],
        dtype=np.float64,
    )
    p0 = np.diag(
        [
            cfg.initial_position_std_m**2,
            cfg.initial_position_std_m**2,
            cfg.initial_velocity_std_m_s**2,
            cfg.initial_velocity_std_m_s**2,
        ]
    ).astype(np.float64)

    predicted_state[0] = x0.copy()
    predicted_cov[0] = p0.copy()

    if first.position_kind is PositionKind.OBSERVED:
        filtered_state[0], filtered_cov[0] = _measurement_update(
            x0,
            p0,
            first,
            cfg,
        )
    else:
        filtered_state[0] = x0.copy()
        filtered_cov[0] = p0.copy()

    for index in range(1, n):
        previous = ordered[index - 1]
        current = ordered[index]

        dt = max(
            cfg.min_dt_seconds,
            float(current.timestamp - previous.timestamp),
        )

        f = _transition_matrix(dt)
        q = _process_covariance(
            dt,
            cfg.process_acceleration_std_m_s2,
        )

        transition_to[index] = f

        x_pred = f @ filtered_state[index - 1]
        p_pred = (
            f
            @ filtered_cov[index - 1]
            @ f.T
            + q
        )
        p_pred = _symmetrize(p_pred)

        predicted_state[index] = x_pred
        predicted_cov[index] = p_pred

        if current.position_kind is PositionKind.OBSERVED:
            x_filt, p_filt = _measurement_update(
                x_pred,
                p_pred,
                current,
                cfg,
            )
        else:
            x_filt = x_pred
            p_filt = p_pred

        filtered_state[index] = x_filt
        filtered_cov[index] = p_filt

    smoothed_state: list[np.ndarray] = [
        state.copy()
        for state in filtered_state
    ]
    smoothed_cov: list[np.ndarray] = [
        covariance.copy()
        for covariance in filtered_cov
    ]

    # Rauch-Tung-Striebel backward pass.
    for index in range(n - 2, -1, -1):
        f_next = transition_to[index + 1]
        p_filt = filtered_cov[index]
        p_pred_next = predicted_cov[index + 1]

        # C_k = P_k|k F^T (P_{k+1|k})^-1
        # Solve instead of forming an explicit inverse.
        cross = p_filt @ f_next.T
        gain = np.linalg.solve(
            p_pred_next.T,
            cross.T,
        ).T

        smoothed_state[index] = (
            filtered_state[index]
            + gain
            @ (
                smoothed_state[index + 1]
                - predicted_state[index + 1]
            )
        )

        smoothed_cov[index] = _symmetrize(
            p_filt
            + gain
            @ (
                smoothed_cov[index + 1]
                - p_pred_next
            )
            @ gain.T
        )

    result: list[TrackSample] = []

    for raw, state in zip(ordered, smoothed_state):
        x, z, vx, vz = (
            float(state[0]),
            float(state[1]),
            float(state[2]),
            float(state[3]),
        )

        if not all(isfinite(value) for value in (x, z, vx, vz)):
            raise RuntimeError(
                f"RTS smoothing produced non-finite state for track {track_id}"
            )

        result.append(
            replace(
                raw,
                x=x,
                z=z,
                vx=vx,
                vz=vz,
            )
        )

    return tuple(result)


def _measurement_update(
    state: np.ndarray,
    covariance: np.ndarray,
    sample: TrackSample,
    config: SmoothingConfig,
) -> tuple[np.ndarray, np.ndarray]:
    h = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )

    measurement = np.array(
        [sample.x, sample.z],
        dtype=np.float64,
    )

    std = _measurement_std(sample, config)
    r = np.eye(2, dtype=np.float64) * (std**2)

    innovation = measurement - h @ state
    s = h @ covariance @ h.T + r

    ph_t = covariance @ h.T
    gain = np.linalg.solve(
        s.T,
        ph_t.T,
    ).T

    updated_state = state + gain @ innovation

    # Joseph stabilized covariance update.
    identity = np.eye(4, dtype=np.float64)
    kh = gain @ h
    updated_cov = (
        (identity - kh)
        @ covariance
        @ (identity - kh).T
        + gain @ r @ gain.T
    )

    return (
        updated_state,
        _symmetrize(updated_cov),
    )


def _measurement_std(
    sample: TrackSample,
    config: SmoothingConfig,
) -> float:
    if (
        not config.use_confidence_weighting
        or sample.confidence is None
        or not isfinite(float(sample.confidence))
    ):
        return config.measurement_std_m

    confidence = min(
        config.confidence_ceiling,
        max(
            config.confidence_floor,
            float(sample.confidence),
        ),
    )

    # Lower confidence => slightly larger measurement noise.
    raw_scale = 1.0 / sqrt(confidence)
    scale = min(
        config.max_measurement_std_scale,
        max(1.0, raw_scale),
    )
    return config.measurement_std_m * scale


def _transition_matrix(dt: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _process_covariance(
    dt: float,
    acceleration_std: float,
) -> np.ndarray:
    variance = acceleration_std**2

    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    return variance * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=np.float64,
    )


def _metrics(
    *,
    original: Sequence[TrackSample],
    smoothed: Sequence[TrackSample],
) -> TrackSmoothingMetrics:
    observed_corrections: list[float] = []
    predicted_corrections: list[float] = []

    for raw, smooth in zip(original, smoothed):
        correction = hypot(
            smooth.x - raw.x,
            smooth.z - raw.z,
        )

        if raw.position_kind is PositionKind.OBSERVED:
            observed_corrections.append(correction)
        elif raw.position_kind is PositionKind.PREDICTED:
            predicted_corrections.append(correction)

    deltas = [
        float(b.timestamp - a.timestamp)
        for a, b in zip(original, original[1:])
        if b.timestamp > a.timestamp
    ]

    observed_count = sum(
        sample.position_kind is PositionKind.OBSERVED
        for sample in original
    )
    predicted_count = sum(
        sample.position_kind is PositionKind.PREDICTED
        for sample in original
    )
    interpolated_count = sum(
        sample.position_kind is PositionKind.INTERPOLATED
        for sample in original
    )

    return TrackSmoothingMetrics(
        track_id=original[0].track_id,
        sample_count=len(original),
        observed_count=observed_count,
        predicted_count=predicted_count,
        interpolated_count=interpolated_count,
        first_timestamp_s=float(original[0].timestamp),
        last_timestamp_s=float(original[-1].timestamp),
        duration_s=max(
            0.0,
            float(original[-1].timestamp - original[0].timestamp),
        ),
        median_sample_period_s=(
            float(median(deltas))
            if deltas
            else 0.0
        ),
        longest_sample_gap_s=(
            max(deltas)
            if deltas
            else 0.0
        ),
        observed_rms_correction_m=_rms(observed_corrections),
        observed_median_correction_m=(
            float(median(observed_corrections))
            if observed_corrections
            else 0.0
        ),
        observed_max_correction_m=(
            max(observed_corrections)
            if observed_corrections
            else 0.0
        ),
        predicted_rms_correction_m=_rms(predicted_corrections),
        predicted_max_correction_m=(
            max(predicted_corrections)
            if predicted_corrections
            else 0.0
        ),
        raw_max_speed_m_s=max(
            hypot(sample.vx, sample.vz)
            for sample in original
        ),
        smoothed_max_speed_m_s=max(
            hypot(sample.vx, sample.vz)
            for sample in smoothed
        ),
    )


def _rms(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sqrt(
        sum(value * value for value in values)
        / len(values)
    )


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.timestamp,
                sample.frame_index,
            ),
        )
    )

    previous_timestamp: float | None = None

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match "
                f"sample track_id {sample.track_id}"
            )

        if not isfinite(sample.timestamp):
            raise ValueError("Track sample timestamp must be finite")

        if not all(
            isfinite(value)
            for value in (
                sample.x,
                sample.z,
                sample.vx,
                sample.vz,
            )
        ):
            raise ValueError(
                f"Track {track_id} contains non-finite state"
            )

        if (
            previous_timestamp is not None
            and sample.timestamp <= previous_timestamp
        ):
            raise ValueError(
                f"Track {track_id} timestamps must be strictly increasing"
            )

        previous_timestamp = sample.timestamp

    return ordered
