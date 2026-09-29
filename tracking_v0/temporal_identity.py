"""Temporal identity evidence for football tracklet stitching.

This module adds the missing "before / after" dimension to identity resolution.

It operates on OBSERVED field points and asks whether one identity unit can
plausibly continue as another after a short disappearance or a short duplicate
overlap.  It intentionally does not decide canonical identities by itself.

Two transition regimes are handled:

1. GAP / handover:
       A ends ---- missing ---- B starts

   We fit recent velocity for A and early velocity for B, extrapolate both ways,
   and score the symmetric prediction error, velocity agreement and gap length.

2. SHORT OVERLAP / duplicate handover:
       A ---------
              -------- B

   We compare simultaneous field positions and local motion.  Long/sustained
   overlaps should already be represented as STRONG conflicts by identity
   evidence; this layer is for short duplicate-ID handovers.

The same motion primitives also expose a conservative gate for future
TEMP_OUTSIDE tracking: an outside observation is admissible only when an
already-tracked inside player was near a boundary, moving outward, and the
outside candidate remains close to the predicted continuation corridor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Sequence

import numpy as np

from .identity_evidence import ObservedTrackPoint


EPS = 1e-9


@dataclass(frozen=True)
class TemporalTransitionConfig:
    fit_window_s: float = 0.90
    min_fit_points: int = 3

    max_gap_s: float = 5.0
    max_overlap_s: float = 3.0

    base_position_sigma_m: float = 0.85
    position_sigma_per_gap_s: float = 1.20
    overlap_position_sigma_m: float = 1.35

    max_transition_error_m: float = 9.0
    max_overlap_q90_distance_m: float = 4.5

    velocity_difference_sigma_mps: float = 3.5
    stationary_speed_mps: float = 0.45
    max_implied_speed_mps: float = 13.5

    gap_decay_s: float = 3.0

    # Temporal evidence is tri-state. A weak/noisy motion estimate is NEUTRAL,
    # not automatically negative evidence.
    support_score_threshold: float = 0.58
    neutral_score: float = 0.50
    overlap_contradiction_min_fraction: float = 0.30
    overlap_contradiction_median_m: float = 5.0

    def __post_init__(self) -> None:
        if self.fit_window_s <= 0.0:
            raise ValueError("fit_window_s must be > 0")
        if self.min_fit_points < 2:
            raise ValueError("min_fit_points must be >= 2")
        if self.max_gap_s <= 0.0:
            raise ValueError("max_gap_s must be > 0")
        if self.max_overlap_s <= 0.0:
            raise ValueError("max_overlap_s must be > 0")
        for name in (
            "base_position_sigma_m",
            "position_sigma_per_gap_s",
            "overlap_position_sigma_m",
            "max_transition_error_m",
            "max_overlap_q90_distance_m",
            "velocity_difference_sigma_mps",
            "stationary_speed_mps",
            "max_implied_speed_mps",
            "gap_decay_s",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")

        for name in (
            "support_score_threshold",
            "neutral_score",
            "overlap_contradiction_min_fraction",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.overlap_contradiction_median_m <= 0.0:
            raise ValueError(
                "overlap_contradiction_median_m must be > 0"
            )


@dataclass(frozen=True)
class MotionFit:
    reference_time: float
    x: float
    z: float
    vx: float
    vz: float
    rmse_m: float
    point_count: int

    @property
    def speed_mps(self) -> float:
        return float(
            math.hypot(
                self.vx,
                self.vz,
            )
        )

    def predict(self, timestamp: float) -> tuple[float, float]:
        dt = float(timestamp - self.reference_time)
        return (
            self.x + self.vx * dt,
            self.z + self.vz * dt,
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TemporalTransitionEvidence:
    source_key: str
    target_key: str

    source_track_id: int
    target_track_id: int

    gap_s: float
    overlap_s: float

    forward_error_m: float | None
    backward_error_m: float | None
    symmetric_error_m: float | None

    overlap_median_distance_m: float | None
    overlap_q90_distance_m: float | None

    implied_speed_mps: float | None
    velocity_cosine: float | None
    velocity_difference_mps: float | None

    position_score: float
    velocity_score: float
    gap_score: float
    temporal_score: float

    valid: bool
    regime: str
    reason: str

    # "support": motion actively supports same identity.
    # "neutral": temporal model cannot reliably decide; identity evidence may
    #            still connect the units.
    # "contradiction": physically implausible clean continuation.
    verdict: str = "neutral"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class FieldBounds:
    min_x: float
    max_x: float
    min_z: float
    max_z: float

    def __post_init__(self) -> None:
        if self.max_x <= self.min_x:
            raise ValueError("invalid X bounds")
        if self.max_z <= self.min_z:
            raise ValueError("invalid Z bounds")

    def contains(
        self,
        x: float,
        z: float,
        *,
        margin_m: float = 0.0,
    ) -> bool:
        return (
            self.min_x - margin_m
            <= x
            <= self.max_x + margin_m
            and self.min_z - margin_m
            <= z
            <= self.max_z + margin_m
        )

    def outside_distance_m(
        self,
        x: float,
        z: float,
    ) -> float:
        dx = max(
            self.min_x - x,
            0.0,
            x - self.max_x,
        )
        dz = max(
            self.min_z - z,
            0.0,
            z - self.max_z,
        )
        return float(
            math.hypot(
                dx,
                dz,
            )
        )


@dataclass(frozen=True)
class OutsideTrackingConfig:
    boundary_exit_band_m: float = 2.0
    outside_corridor_m: float = 4.0
    max_outside_gap_s: float = 3.0
    prediction_sigma_m: float = 1.5
    min_outward_speed_mps: float = 0.25
    min_score: float = 0.50

    def __post_init__(self) -> None:
        for name in (
            "boundary_exit_band_m",
            "outside_corridor_m",
            "max_outside_gap_s",
            "prediction_sigma_m",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be > 0")
        if self.min_outward_speed_mps < 0.0:
            raise ValueError("min_outward_speed_mps must be >= 0")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError("min_score must be in [0, 1]")


@dataclass(frozen=True)
class OutsideContinuationEvidence:
    allowed: bool
    gap_s: float
    outside_distance_m: float
    prediction_error_m: float | None
    score: float
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_temporal_transition(
    key_a: str,
    key_b: str,
    *,
    rows_a: Sequence[ObservedTrackPoint],
    rows_b: Sequence[ObservedTrackPoint],
    config: TemporalTransitionConfig | None = None,
) -> TemporalTransitionEvidence:
    """Score the most plausible directed continuation between two units."""

    cfg = config or TemporalTransitionConfig()

    a = _sorted_points(rows_a)
    b = _sorted_points(rows_b)

    if not a or not b:
        raise ValueError(
            "temporal transition requires observations for both units"
        )

    # Stable orientation by observed ending time.  In duplicate-overlap cases
    # this chooses the unit that disappears first as the source.
    if (
        a[-1].timestamp,
        a[-1].frame_index,
        a[0].timestamp,
        key_a,
    ) <= (
        b[-1].timestamp,
        b[-1].frame_index,
        b[0].timestamp,
        key_b,
    ):
        source_key = key_a
        target_key = key_b
        source = a
        target = b
    else:
        source_key = key_b
        target_key = key_a
        source = b
        target = a

    gap_s = float(
        target[0].timestamp
        - source[-1].timestamp
    )
    overlap_s = max(
        0.0,
        min(
            source[-1].timestamp,
            target[-1].timestamp,
        )
        - max(
            source[0].timestamp,
            target[0].timestamp,
        ),
    )

    source_fit = fit_motion(
        source,
        side="tail",
        config=cfg,
    )
    target_fit = fit_motion(
        target,
        side="head",
        config=cfg,
    )

    velocity_cosine = _velocity_cosine(
        source_fit,
        target_fit,
        stationary_speed_mps=cfg.stationary_speed_mps,
    )
    velocity_difference = abs(
        source_fit.speed_mps
        - target_fit.speed_mps
    )
    velocity_magnitude_score = math.exp(
        -0.5
        * (
            velocity_difference
            / cfg.velocity_difference_sigma_mps
        )
        ** 2
    )
    direction_score = (
        0.75
        if velocity_cosine is None
        else 0.5
        + 0.5
        * max(
            -1.0,
            min(
                1.0,
                velocity_cosine,
            ),
        )
    )
    velocity_score = (
        0.60 * direction_score
        + 0.40 * velocity_magnitude_score
    )

    if gap_s >= 0.0:
        return _score_gap_transition(
            source_key=source_key,
            target_key=target_key,
            source=source,
            target=target,
            source_fit=source_fit,
            target_fit=target_fit,
            velocity_cosine=velocity_cosine,
            velocity_difference=velocity_difference,
            velocity_score=velocity_score,
            gap_s=gap_s,
            config=cfg,
        )

    return _score_overlap_transition(
        source_key=source_key,
        target_key=target_key,
        source=source,
        target=target,
        source_fit=source_fit,
        target_fit=target_fit,
        velocity_cosine=velocity_cosine,
        velocity_difference=velocity_difference,
        velocity_score=velocity_score,
        gap_s=gap_s,
        overlap_s=overlap_s,
        config=cfg,
    )


def fit_motion(
    rows: Sequence[ObservedTrackPoint],
    *,
    side: str,
    config: TemporalTransitionConfig | None = None,
) -> MotionFit:
    cfg = config or TemporalTransitionConfig()
    points = _sorted_points(rows)

    if not points:
        raise ValueError("motion fit requires observations")
    if side not in {"head", "tail"}:
        raise ValueError("side must be 'head' or 'tail'")

    if side == "tail":
        reference = points[-1]
        selected = [
            point
            for point in points
            if (
                reference.timestamp
                - cfg.fit_window_s
                <= point.timestamp
                <= reference.timestamp
            )
        ]
    else:
        reference = points[0]
        selected = [
            point
            for point in points
            if (
                reference.timestamp
                <= point.timestamp
                <= reference.timestamp
                + cfg.fit_window_s
            )
        ]

    if len(selected) < cfg.min_fit_points:
        selected = (
            points[
                -max(
                    cfg.min_fit_points,
                    min(
                        len(points),
                        6,
                    ),
                ):
            ]
            if side == "tail"
            else points[
                : max(
                    cfg.min_fit_points,
                    min(
                        len(points),
                        6,
                    ),
                )
            ]
        )

    if len(selected) == 1:
        point = selected[0]
        return MotionFit(
            reference_time=point.timestamp,
            x=point.x,
            z=point.z,
            vx=0.0,
            vz=0.0,
            rmse_m=0.0,
            point_count=1,
        )

    ref_time = (
        selected[-1].timestamp
        if side == "tail"
        else selected[0].timestamp
    )

    times = np.asarray(
        [
            point.timestamp
            - ref_time
            for point in selected
        ],
        dtype=np.float64,
    )
    xs = np.asarray(
        [point.x for point in selected],
        dtype=np.float64,
    )
    zs = np.asarray(
        [point.z for point in selected],
        dtype=np.float64,
    )

    design = np.column_stack(
        [
            np.ones_like(times),
            times,
        ]
    )

    coef_x, *_ = np.linalg.lstsq(
        design,
        xs,
        rcond=None,
    )
    coef_z, *_ = np.linalg.lstsq(
        design,
        zs,
        rcond=None,
    )

    predicted_x = design @ coef_x
    predicted_z = design @ coef_z

    residual = np.sqrt(
        (predicted_x - xs) ** 2
        + (predicted_z - zs) ** 2
    )

    return MotionFit(
        reference_time=float(ref_time),
        x=float(coef_x[0]),
        z=float(coef_z[0]),
        vx=float(coef_x[1]),
        vz=float(coef_z[1]),
        rmse_m=float(
            np.sqrt(
                np.mean(
                    residual ** 2
                )
            )
        ),
        point_count=len(selected),
    )


def evaluate_temporary_outside_candidate(
    history: Sequence[ObservedTrackPoint],
    candidate: ObservedTrackPoint,
    *,
    field: FieldBounds,
    config: OutsideTrackingConfig | None = None,
    transition_config: TemporalTransitionConfig | None = None,
) -> OutsideContinuationEvidence:
    """Gate a future outside-field detection for an already tracked player.

    This does NOT admit arbitrary people outside the pitch.  The candidate is
    eligible only if:
      * the identity has an inside history,
      * the last inside observation is near a field boundary,
      * recent velocity points outward through that boundary,
      * the outside candidate is close in time and to the motion prediction,
      * it remains in a small outside corridor.

    The detector can therefore continue a known player during a throw-in or
    brief excursion without opening the normal player pool to bench/sideline
    people.
    """

    cfg = config or OutsideTrackingConfig()
    transition_cfg = (
        transition_config
        or TemporalTransitionConfig()
    )

    rows = _sorted_points(history)

    if not rows:
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=math.inf,
            outside_distance_m=field.outside_distance_m(
                candidate.x,
                candidate.z,
            ),
            prediction_error_m=None,
            score=0.0,
            reason="no_inside_history",
        )

    if field.contains(
        candidate.x,
        candidate.z,
    ):
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=max(
                0.0,
                candidate.timestamp
                - rows[-1].timestamp,
            ),
            outside_distance_m=0.0,
            prediction_error_m=None,
            score=0.0,
            reason="candidate_is_inside",
        )

    last_inside = None
    for point in reversed(rows):
        if field.contains(
            point.x,
            point.z,
        ):
            last_inside = point
            break

    if last_inside is None:
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=math.inf,
            outside_distance_m=field.outside_distance_m(
                candidate.x,
                candidate.z,
            ),
            prediction_error_m=None,
            score=0.0,
            reason="history_never_inside",
        )

    gap_s = float(
        candidate.timestamp
        - last_inside.timestamp
    )
    outside_distance = field.outside_distance_m(
        candidate.x,
        candidate.z,
    )

    if gap_s < 0.0 or gap_s > cfg.max_outside_gap_s:
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=gap_s,
            outside_distance_m=outside_distance,
            prediction_error_m=None,
            score=0.0,
            reason="outside_gap_too_large",
        )

    if outside_distance > cfg.outside_corridor_m:
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=gap_s,
            outside_distance_m=outside_distance,
            prediction_error_m=None,
            score=0.0,
            reason="outside_corridor_exceeded",
        )

    boundary = _nearest_boundary(
        last_inside.x,
        last_inside.z,
        field=field,
    )

    if boundary[0] > cfg.boundary_exit_band_m:
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=gap_s,
            outside_distance_m=outside_distance,
            prediction_error_m=None,
            score=0.0,
            reason="last_inside_not_near_boundary",
        )

    motion = fit_motion(
        rows,
        side="tail",
        config=transition_cfg,
    )

    outward_speed = _outward_speed(
        motion,
        boundary_name=boundary[1],
    )

    if outward_speed < cfg.min_outward_speed_mps:
        return OutsideContinuationEvidence(
            allowed=False,
            gap_s=gap_s,
            outside_distance_m=outside_distance,
            prediction_error_m=None,
            score=0.0,
            reason="motion_not_outward",
        )

    predicted_x, predicted_z = motion.predict(
        candidate.timestamp
    )
    prediction_error = math.hypot(
        predicted_x - candidate.x,
        predicted_z - candidate.z,
    )

    prediction_score = math.exp(
        -0.5
        * (
            prediction_error
            / cfg.prediction_sigma_m
        )
        ** 2
    )
    time_score = math.exp(
        -gap_s
        / max(
            EPS,
            cfg.max_outside_gap_s,
        )
    )
    corridor_score = max(
        0.0,
        1.0
        - outside_distance
        / cfg.outside_corridor_m,
    )

    score = (
        0.65 * prediction_score
        + 0.20 * time_score
        + 0.15 * corridor_score
    )

    return OutsideContinuationEvidence(
        allowed=bool(
            score >= cfg.min_score
        ),
        gap_s=gap_s,
        outside_distance_m=outside_distance,
        prediction_error_m=float(
            prediction_error
        ),
        score=float(score),
        reason=(
            "admissible_known_exit_continuation"
            if score >= cfg.min_score
            else "outside_continuation_score_too_low"
        ),
    )


def _score_gap_transition(
    *,
    source_key: str,
    target_key: str,
    source: Sequence[ObservedTrackPoint],
    target: Sequence[ObservedTrackPoint],
    source_fit: MotionFit,
    target_fit: MotionFit,
    velocity_cosine: float | None,
    velocity_difference: float,
    velocity_score: float,
    gap_s: float,
    config: TemporalTransitionConfig,
) -> TemporalTransitionEvidence:
    source_end = source[-1]
    target_start = target[0]

    forward_x, forward_z = source_fit.predict(
        target_start.timestamp
    )
    backward_x, backward_z = target_fit.predict(
        source_end.timestamp
    )

    forward_error = math.hypot(
        forward_x - target_start.x,
        forward_z - target_start.z,
    )
    backward_error = math.hypot(
        backward_x - source_end.x,
        backward_z - source_end.z,
    )

    symmetric_error = math.sqrt(
        0.5
        * (
            forward_error ** 2
            + backward_error ** 2
        )
    )

    endpoint_distance = math.hypot(
        target_start.x - source_end.x,
        target_start.z - source_end.z,
    )
    implied_speed = (
        endpoint_distance / gap_s
        if gap_s > EPS
        else 0.0
    )

    sigma = (
        config.base_position_sigma_m
        + config.position_sigma_per_gap_s
        * gap_s
        + 0.25
        * (
            source_fit.rmse_m
            + target_fit.rmse_m
        )
    )

    position_score = math.exp(
        -0.5
        * (
            symmetric_error
            / max(
                EPS,
                sigma,
            )
        )
        ** 2
    )

    gap_score = math.exp(
        -gap_s
        / config.gap_decay_s
    )

    raw_score = (
        0.68 * position_score
        + 0.20 * velocity_score
        + 0.12 * gap_score
    )

    # A gap beyond our motion horizon is absence of temporal information, not
    # proof that the identities differ.
    if gap_s > config.max_gap_s:
        verdict = "neutral"
        valid = True
        reason = "gap_beyond_motion_horizon"
        temporal_score = config.neutral_score

    # These are genuine physical contradictions for a clean, non-overlapping
    # handover and may veto a temporal edge.
    elif symmetric_error > config.max_transition_error_m:
        verdict = "contradiction"
        valid = False
        reason = "prediction_error_too_large"
        temporal_score = 0.0

    elif implied_speed > config.max_implied_speed_mps:
        verdict = "contradiction"
        valid = False
        reason = "implied_speed_too_large"
        temporal_score = 0.0

    elif raw_score >= config.support_score_threshold:
        verdict = "support"
        valid = True
        reason = "plausible_gap_handover"
        temporal_score = raw_score

    else:
        # Motion fits in football can be noisy because field projection error is
        # heteroscedastic.  Weak-but-not-impossible motion is therefore neutral.
        verdict = "neutral"
        valid = True
        reason = "weak_gap_motion_neutral"
        temporal_score = config.neutral_score

    return TemporalTransitionEvidence(
        source_key=source_key,
        target_key=target_key,
        source_track_id=source[0].track_id,
        target_track_id=target[0].track_id,
        gap_s=float(gap_s),
        overlap_s=0.0,
        forward_error_m=float(
            forward_error
        ),
        backward_error_m=float(
            backward_error
        ),
        symmetric_error_m=float(
            symmetric_error
        ),
        overlap_median_distance_m=None,
        overlap_q90_distance_m=None,
        implied_speed_mps=float(
            implied_speed
        ),
        velocity_cosine=(
            float(velocity_cosine)
            if velocity_cosine is not None
            else None
        ),
        velocity_difference_mps=float(
            velocity_difference
        ),
        position_score=float(
            position_score
        ),
        velocity_score=float(
            velocity_score
        ),
        gap_score=float(
            gap_score
        ),
        temporal_score=float(
            temporal_score
        ),
        valid=bool(valid),
        regime="gap",
        reason=reason,
        verdict=verdict,
    )

def _score_overlap_transition(
    *,
    source_key: str,
    target_key: str,
    source: Sequence[ObservedTrackPoint],
    target: Sequence[ObservedTrackPoint],
    source_fit: MotionFit,
    target_fit: MotionFit,
    velocity_cosine: float | None,
    velocity_difference: float,
    velocity_score: float,
    gap_s: float,
    overlap_s: float,
    config: TemporalTransitionConfig,
) -> TemporalTransitionEvidence:
    by_frame_source = {
        point.frame_index: point
        for point in source
    }
    by_frame_target = {
        point.frame_index: point
        for point in target
    }

    common_frames = sorted(
        set(by_frame_source)
        & set(by_frame_target)
    )

    simultaneous_fraction = (
        len(common_frames)
        / max(
            1,
            min(
                len(source),
                len(target),
            ),
        )
    )

    distances = [
        math.hypot(
            by_frame_source[frame].x
            - by_frame_target[frame].x,
            by_frame_source[frame].z
            - by_frame_target[frame].z,
        )
        for frame in common_frames
    ]

    # IMPORTANT: support-interval overlap is not simultaneous observation.
    # Raw IDs can interleave for many seconds while almost never being observed
    # in the same frame.  In that case temporal evidence is simply neutral and
    # must not zero a strong ReID/team/geometry handover such as 1<->29.
    if not distances:
        return TemporalTransitionEvidence(
            source_key=source_key,
            target_key=target_key,
            source_track_id=source[0].track_id,
            target_track_id=target[0].track_id,
            gap_s=float(gap_s),
            overlap_s=float(overlap_s),
            forward_error_m=None,
            backward_error_m=None,
            symmetric_error_m=None,
            overlap_median_distance_m=None,
            overlap_q90_distance_m=None,
            implied_speed_mps=None,
            velocity_cosine=(
                float(velocity_cosine)
                if velocity_cosine is not None
                else None
            ),
            velocity_difference_mps=float(
                velocity_difference
            ),
            position_score=0.5,
            velocity_score=float(
                velocity_score
            ),
            gap_score=0.5,
            temporal_score=float(
                config.neutral_score
            ),
            valid=True,
            regime="interleaved_overlap",
            reason="no_simultaneous_observations_neutral",
            verdict="neutral",
        )

    values = np.asarray(
        distances,
        dtype=np.float64,
    )
    overlap_median = float(
        np.median(values)
    )
    overlap_q90 = float(
        np.quantile(
            values,
            0.90,
        )
    )

    position_score = math.exp(
        -0.5
        * (
            overlap_median
            / config.overlap_position_sigma_m
        )
        ** 2
    )

    if len(common_frames) >= 2:
        observed_overlap_s = (
            by_frame_source[
                common_frames[-1]
            ].timestamp
            - by_frame_source[
                common_frames[0]
            ].timestamp
        )
    else:
        observed_overlap_s = 0.0

    overlap_duration_score = math.exp(
        -max(
            0.0,
            observed_overlap_s,
        )
        / config.max_overlap_s
    )

    raw_score = (
        0.72 * position_score
        + 0.20 * velocity_score
        + 0.08 * overlap_duration_score
    )

    # A real temporal contradiction requires substantial simultaneous support
    # and clearly different positions. Sparse simultaneous samples are not
    # enough to overrule identity evidence.
    if (
        simultaneous_fraction
        >= config.overlap_contradiction_min_fraction
        and overlap_median
        >= config.overlap_contradiction_median_m
    ):
        verdict = "contradiction"
        valid = False
        reason = "sustained_overlap_positions_inconsistent"
        temporal_score = 0.0

    elif (
        overlap_q90
        <= config.max_overlap_q90_distance_m
        and raw_score
        >= config.support_score_threshold
    ):
        verdict = "support"
        valid = True
        reason = "plausible_duplicate_overlap_handover"
        temporal_score = raw_score

    else:
        verdict = "neutral"
        valid = True
        reason = "sparse_or_noisy_overlap_neutral"
        temporal_score = config.neutral_score

    return TemporalTransitionEvidence(
        source_key=source_key,
        target_key=target_key,
        source_track_id=source[0].track_id,
        target_track_id=target[0].track_id,
        gap_s=float(gap_s),
        overlap_s=float(
            overlap_s
        ),
        forward_error_m=None,
        backward_error_m=None,
        symmetric_error_m=None,
        overlap_median_distance_m=float(
            overlap_median
        ),
        overlap_q90_distance_m=float(
            overlap_q90
        ),
        implied_speed_mps=None,
        velocity_cosine=(
            float(velocity_cosine)
            if velocity_cosine is not None
            else None
        ),
        velocity_difference_mps=float(
            velocity_difference
        ),
        position_score=float(
            position_score
        ),
        velocity_score=float(
            velocity_score
        ),
        gap_score=float(
            overlap_duration_score
        ),
        temporal_score=float(
            temporal_score
        ),
        valid=bool(valid),
        regime="observed_overlap",
        reason=reason,
        verdict=verdict,
    )

def _velocity_cosine(
    left: MotionFit,
    right: MotionFit,
    *,
    stationary_speed_mps: float,
) -> float | None:
    speed_left = left.speed_mps
    speed_right = right.speed_mps

    if (
        speed_left < stationary_speed_mps
        or speed_right < stationary_speed_mps
    ):
        return None

    dot = (
        left.vx * right.vx
        + left.vz * right.vz
    )

    return float(
        dot
        / max(
            EPS,
            speed_left * speed_right,
        )
    )


def _nearest_boundary(
    x: float,
    z: float,
    *,
    field: FieldBounds,
) -> tuple[float, str]:
    distances = (
        (abs(x - field.min_x), "min_x"),
        (abs(x - field.max_x), "max_x"),
        (abs(z - field.min_z), "min_z"),
        (abs(z - field.max_z), "max_z"),
    )
    return min(
        distances,
        key=lambda item: item[0],
    )


def _outward_speed(
    motion: MotionFit,
    *,
    boundary_name: str,
) -> float:
    if boundary_name == "min_x":
        return float(-motion.vx)
    if boundary_name == "max_x":
        return float(motion.vx)
    if boundary_name == "min_z":
        return float(-motion.vz)
    if boundary_name == "max_z":
        return float(motion.vz)

    raise ValueError(
        f"unknown boundary {boundary_name}"
    )


def _sorted_points(
    rows: Sequence[ObservedTrackPoint],
) -> tuple[ObservedTrackPoint, ...]:
    return tuple(
        sorted(
            rows,
            key=lambda point: (
                point.timestamp,
                point.frame_index,
            ),
        )
    )
