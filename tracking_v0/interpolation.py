from __future__ import annotations

from dataclasses import asdict, dataclass
from math import hypot, isfinite
from typing import Mapping, Sequence

from .models import PositionKind, TrackSample


@dataclass(slots=True, frozen=True)
class InterpolationConfig:
    """
    Offline temporal interpolation after identity resolution + RTS smoothing.

    Existing samples are preserved exactly. Missing source frames are filled
    with cubic Hermite interpolation using the smoothed endpoint position and
    velocity.

    Inserted samples are marked internally as INTERPOLATED, but the UI does not
    need to render them differently.

    No roster-size assumption is used.
    """

    target_fps: float = 30.0

    # Do not invent arbitrarily long trajectories from two distant endpoints.
    # The current forward tracker coasts for up to ~3 s, so 3 s is a sensible
    # default upper bound for interpolation after identity resolution.
    max_gap_seconds: float = 3.0

    # Physical guard on Hermite tangents.
    max_speed_m_s: float = 12.5

    # Prevent large endpoint velocities from creating cubic loops/overshoot
    # when the endpoint displacement is small.
    max_tangent_to_chord_speed_ratio: float = 3.0

    min_dt_seconds: float = 1e-6

    def __post_init__(self) -> None:
        if self.target_fps <= 0.0:
            raise ValueError("target_fps must be > 0")
        if self.max_gap_seconds <= 0.0:
            raise ValueError("max_gap_seconds must be > 0")
        if self.max_speed_m_s <= 0.0:
            raise ValueError("max_speed_m_s must be > 0")
        if self.max_tangent_to_chord_speed_ratio <= 0.0:
            raise ValueError(
                "max_tangent_to_chord_speed_ratio must be > 0"
            )
        if self.min_dt_seconds <= 0.0:
            raise ValueError("min_dt_seconds must be > 0")


@dataclass(slots=True, frozen=True)
class TrackInterpolationMetrics:
    track_id: int

    input_samples: int
    output_samples: int
    inserted_samples: int

    first_frame_index: int
    last_frame_index: int

    filled_gaps: int
    skipped_large_gaps: int

    max_filled_gap_seconds: float
    max_skipped_gap_seconds: float

    max_interpolated_speed_m_s: float
    max_hermite_vs_linear_deviation_m: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class InterpolationReport:
    tracks: Mapping[int, TrackInterpolationMetrics]

    @property
    def track_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self.tracks))

    @property
    def total_inserted_samples(self) -> int:
        return sum(
            metrics.inserted_samples
            for metrics in self.tracks.values()
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "track_ids": list(self.track_ids),
            "total_inserted_samples": self.total_inserted_samples,
            "tracks": {
                str(track_id): metrics.to_dict()
                for track_id, metrics in sorted(self.tracks.items())
            },
        }


def interpolate_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    config: InterpolationConfig | None = None,
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    InterpolationReport,
]:
    cfg = config or InterpolationConfig()

    interpolated: dict[int, tuple[TrackSample, ...]] = {}
    reports: dict[int, TrackInterpolationMetrics] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(
            track_id,
            raw_samples,
        )
        if not samples:
            continue

        output, metrics = interpolate_trajectory(
            samples,
            config=cfg,
        )
        interpolated[track_id] = output
        reports[track_id] = metrics

    return (
        interpolated,
        InterpolationReport(tracks=reports),
    )


def interpolate_trajectory(
    samples: Sequence[TrackSample],
    *,
    config: InterpolationConfig | None = None,
) -> tuple[
    tuple[TrackSample, ...],
    TrackInterpolationMetrics,
]:
    """
    Fill missing integer frame indices between existing trajectory samples.

    Example at source 30 FPS with a 15 Hz tracker:

        frame 100  OBSERVED/PREDICTED
        frame 102  OBSERVED/PREDICTED

    becomes:

        frame 100  original
        frame 101  INTERPOLATED
        frame 102  original

    Existing samples are never modified.
    """
    cfg = config or InterpolationConfig()

    if not samples:
        raise ValueError("interpolate_trajectory requires at least one sample")

    track_id = samples[0].track_id
    ordered = _validated_samples(
        track_id,
        samples,
    )

    if len(ordered) == 1:
        metrics = TrackInterpolationMetrics(
            track_id=track_id,
            input_samples=1,
            output_samples=1,
            inserted_samples=0,
            first_frame_index=ordered[0].frame_index,
            last_frame_index=ordered[0].frame_index,
            filled_gaps=0,
            skipped_large_gaps=0,
            max_filled_gap_seconds=0.0,
            max_skipped_gap_seconds=0.0,
            max_interpolated_speed_m_s=0.0,
            max_hermite_vs_linear_deviation_m=0.0,
        )
        return ordered, metrics

    output: list[TrackSample] = []

    inserted_samples = 0
    filled_gaps = 0
    skipped_large_gaps = 0

    max_filled_gap_seconds = 0.0
    max_skipped_gap_seconds = 0.0
    max_interpolated_speed = 0.0
    max_curve_deviation = 0.0

    for left, right in zip(
        ordered,
        ordered[1:],
    ):
        output.append(left)

        frame_delta = right.frame_index - left.frame_index
        if frame_delta <= 1:
            continue

        dt = float(right.timestamp - left.timestamp)
        if dt <= cfg.min_dt_seconds:
            raise ValueError(
                f"Track {track_id}: non-positive timestamp gap "
                f"between frames {left.frame_index} and {right.frame_index}"
            )

        if dt > cfg.max_gap_seconds:
            skipped_large_gaps += 1
            max_skipped_gap_seconds = max(
                max_skipped_gap_seconds,
                dt,
            )
            continue

        filled_gaps += 1
        max_filled_gap_seconds = max(
            max_filled_gap_seconds,
            dt,
        )

        left_velocity, right_velocity = _guarded_tangents(
            left,
            right,
            dt=dt,
            config=cfg,
        )

        for frame_index in range(
            left.frame_index + 1,
            right.frame_index,
        ):
            alpha = (
                (frame_index - left.frame_index)
                / frame_delta
            )

            timestamp = (
                left.timestamp
                + alpha * dt
            )

            (
                x,
                z,
                vx,
                vz,
            ) = _hermite_state(
                left_x=left.x,
                left_z=left.z,
                left_vx=left_velocity[0],
                left_vz=left_velocity[1],
                right_x=right.x,
                right_z=right.z,
                right_vx=right_velocity[0],
                right_vz=right_velocity[1],
                dt=dt,
                alpha=alpha,
            )

            if not all(
                isfinite(value)
                for value in (
                    x,
                    z,
                    vx,
                    vz,
                    timestamp,
                )
            ):
                raise RuntimeError(
                    f"Track {track_id}: interpolation produced "
                    f"non-finite state at frame {frame_index}"
                )

            speed = hypot(vx, vz)
            if speed > cfg.max_speed_m_s:
                scale = cfg.max_speed_m_s / speed
                vx *= scale
                vz *= scale
                speed = cfg.max_speed_m_s

            max_interpolated_speed = max(
                max_interpolated_speed,
                speed,
            )

            linear_x = (
                left.x
                + alpha * (right.x - left.x)
            )
            linear_z = (
                left.z
                + alpha * (right.z - left.z)
            )
            curve_deviation = hypot(
                x - linear_x,
                z - linear_z,
            )
            max_curve_deviation = max(
                max_curve_deviation,
                curve_deviation,
            )

            output.append(
                TrackSample(
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    x=x,
                    z=z,
                    vx=vx,
                    vz=vz,
                    lifecycle=left.lifecycle,
                    position_kind=PositionKind.INTERPOLATED,
                    measurement_id=None,
                    confidence=None,
                )
            )
            inserted_samples += 1

    output.append(ordered[-1])

    result = tuple(
        sorted(
            output,
            key=lambda sample: (
                sample.frame_index,
                sample.timestamp,
            ),
        )
    )

    metrics = TrackInterpolationMetrics(
        track_id=track_id,
        input_samples=len(ordered),
        output_samples=len(result),
        inserted_samples=inserted_samples,
        first_frame_index=result[0].frame_index,
        last_frame_index=result[-1].frame_index,
        filled_gaps=filled_gaps,
        skipped_large_gaps=skipped_large_gaps,
        max_filled_gap_seconds=max_filled_gap_seconds,
        max_skipped_gap_seconds=max_skipped_gap_seconds,
        max_interpolated_speed_m_s=max_interpolated_speed,
        max_hermite_vs_linear_deviation_m=max_curve_deviation,
    )

    return result, metrics


def _guarded_tangents(
    left: TrackSample,
    right: TrackSample,
    *,
    dt: float,
    config: InterpolationConfig,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
]:
    chord_x = right.x - left.x
    chord_z = right.z - left.z
    chord_distance = hypot(
        chord_x,
        chord_z,
    )

    chord_speed = chord_distance / dt

    # Permit some curvature, but do not let a noisy velocity estimate produce
    # a giant cubic loop when the endpoints themselves are close.
    chord_based_cap = (
        chord_speed
        * config.max_tangent_to_chord_speed_ratio
    )

    tangent_speed_cap = min(
        config.max_speed_m_s,
        max(
            chord_based_cap,
            min(
                config.max_speed_m_s,
                chord_speed,
            ),
        ),
    )

    # If the endpoints are effectively identical, zero tangents are the safest
    # interpolation. Any large velocity there would create a loop.
    if chord_distance <= 1e-6:
        return (
            (0.0, 0.0),
            (0.0, 0.0),
        )

    left_velocity = _clamp_velocity(
        left.vx,
        left.vz,
        tangent_speed_cap,
    )
    right_velocity = _clamp_velocity(
        right.vx,
        right.vz,
        tangent_speed_cap,
    )

    return (
        left_velocity,
        right_velocity,
    )


def _clamp_velocity(
    vx: float,
    vz: float,
    max_speed: float,
) -> tuple[float, float]:
    speed = hypot(vx, vz)

    if speed <= max_speed or speed <= 1e-12:
        return (
            float(vx),
            float(vz),
        )

    scale = max_speed / speed
    return (
        float(vx * scale),
        float(vz * scale),
    )


def _hermite_state(
    *,
    left_x: float,
    left_z: float,
    left_vx: float,
    left_vz: float,
    right_x: float,
    right_z: float,
    right_vx: float,
    right_vz: float,
    dt: float,
    alpha: float,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")

    u = alpha
    u2 = u * u
    u3 = u2 * u

    h00 = 2.0 * u3 - 3.0 * u2 + 1.0
    h10 = u3 - 2.0 * u2 + u
    h01 = -2.0 * u3 + 3.0 * u2
    h11 = u3 - u2

    x = (
        h00 * left_x
        + h10 * dt * left_vx
        + h01 * right_x
        + h11 * dt * right_vx
    )
    z = (
        h00 * left_z
        + h10 * dt * left_vz
        + h01 * right_z
        + h11 * dt * right_vz
    )

    dh00 = 6.0 * u2 - 6.0 * u
    dh10 = 3.0 * u2 - 4.0 * u + 1.0
    dh01 = -6.0 * u2 + 6.0 * u
    dh11 = 3.0 * u2 - 2.0 * u

    vx = (
        dh00 * left_x
        + dh10 * dt * left_vx
        + dh01 * right_x
        + dh11 * dt * right_vx
    ) / dt

    vz = (
        dh00 * left_z
        + dh10 * dt * left_vz
        + dh01 * right_z
        + dh11 * dt * right_vz
    ) / dt

    return (
        float(x),
        float(z),
        float(vx),
        float(vz),
    )


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.frame_index,
                sample.timestamp,
            ),
        )
    )

    previous_frame: int | None = None
    previous_timestamp: float | None = None

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match "
                f"sample track_id {sample.track_id}"
            )

        if not isfinite(sample.timestamp):
            raise ValueError(
                "Track sample timestamp must be finite"
            )

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
            previous_frame is not None
            and sample.frame_index <= previous_frame
        ):
            raise ValueError(
                f"Track {track_id} frame indices must be strictly increasing"
            )

        if (
            previous_timestamp is not None
            and sample.timestamp <= previous_timestamp
        ):
            raise ValueError(
                f"Track {track_id} timestamps must be strictly increasing"
            )

        previous_frame = sample.frame_index
        previous_timestamp = sample.timestamp

    return ordered
