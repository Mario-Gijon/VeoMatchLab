"""Sprint B.2: conservative football-role inference from 2D motion + kit appearance.

This module intentionally does NOT force "exactly two goalkeepers + one referee".
It scores every raw track independently and only assigns a role when:

  * the track has enough real observed support,
  * one role score is high enough,
  * and that score beats the runner-up by a configurable margin.

Otherwise the result is UNKNOWN.

Inputs:
  * raw/fused 2D track samples in centred pitch coordinates (X length, Z width),
  * Sprint B.1 appearance labels (TEAM_A / TEAM_B / SPECIAL / MIXED / UNKNOWN).

Outputs:
  * GOALKEEPER_LEFT / GOALKEEPER_RIGHT / REFEREE / OUTFIELD / UNKNOWN,
  * interpretable geometry metrics,
  * per-role scores and score margin,
  * global candidate rankings for audit/debug.

No V2.7 identity fusion or frontend publishing happens here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from .models import PositionKind, TrackSample
from .team_appearance import AppearanceLabel


EPS = 1e-9


class FootballRole(str, Enum):
    GOALKEEPER_LEFT = "goalkeeper_left"
    GOALKEEPER_RIGHT = "goalkeeper_right"
    REFEREE = "referee"
    OUTFIELD = "outfield"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RoleInferenceConfig:
    field_length_m: float = 105.0
    field_width_m: float = 68.0

    # Goalkeeper geometry.  These are deliberately a little wider than the
    # official penalty area because a keeper can step outside it.
    goalkeeper_depth_m: float = 20.0
    goalkeeper_half_width_m: float = 23.0
    goalkeeper_distance_sigma_m: float = 14.0
    goalkeeper_opposite_side_penalty: float = 0.35

    # Central / referee geometry.
    referee_central_x_fraction: float = 0.35
    referee_central_z_fraction: float = 0.42
    referee_goal_exclusion_m: float = 22.0
    referee_span_x_reference_m: float = 24.0
    referee_span_z_reference_m: float = 15.0

    # Evidence support.
    support_observed_count_reference: int = 45
    support_span_reference_s: float = 3.0
    min_observed_count: int = 12
    min_observed_span_s: float = 1.0

    # Conservative decision policy.
    min_goalkeeper_score: float = 0.72
    min_referee_score: float = 0.72
    min_outfield_score: float = 0.68
    min_role_margin: float = 0.10

    # Appearance confidence.
    min_appearance_purity: float = 0.65

    def __post_init__(self) -> None:
        positive = (
            self.field_length_m,
            self.field_width_m,
            self.goalkeeper_depth_m,
            self.goalkeeper_half_width_m,
            self.goalkeeper_distance_sigma_m,
            self.referee_goal_exclusion_m,
            self.referee_span_x_reference_m,
            self.referee_span_z_reference_m,
            self.support_span_reference_s,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive role-inference geometry values must be > 0")
        if self.support_observed_count_reference < 1:
            raise ValueError("support_observed_count_reference must be >= 1")
        if self.min_observed_count < 1:
            raise ValueError("min_observed_count must be >= 1")
        if self.min_observed_span_s < 0.0:
            raise ValueError("min_observed_span_s must be >= 0")

        fractions = (
            self.referee_central_x_fraction,
            self.referee_central_z_fraction,
            self.goalkeeper_opposite_side_penalty,
            self.min_goalkeeper_score,
            self.min_referee_score,
            self.min_outfield_score,
            self.min_role_margin,
            self.min_appearance_purity,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("role-inference fraction/score values must be in [0, 1]")


@dataclass(frozen=True)
class TrackAppearanceEvidence:
    track_id: int
    label: AppearanceLabel
    top_share: float
    second_share: float
    ui_color_hex: str | None = None

    @property
    def purity(self) -> float:
        return float(self.top_share)

    @classmethod
    def from_report_row(
        cls,
        track_id: int,
        row: Mapping[str, Any],
    ) -> "TrackAppearanceEvidence":
        raw_label = str(row.get("label", "unknown"))
        try:
            label = AppearanceLabel(raw_label)
        except ValueError:
            label = AppearanceLabel.UNKNOWN

        return cls(
            track_id=int(track_id),
            label=label,
            top_share=float(row.get("top_share", 0.0) or 0.0),
            second_share=float(row.get("second_share", 0.0) or 0.0),
            ui_color_hex=(
                str(row["ui_color_hex"])
                if row.get("ui_color_hex") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class TrackRoleGeometry:
    track_id: int

    observed_count: int
    observed_span_s: float
    support_score: float

    median_x: float
    median_z: float
    x_span_m: float
    z_span_m: float
    path_length_m: float
    median_speed_m_s: float

    median_left_goal_distance_m: float
    median_right_goal_distance_m: float

    left_goal_zone_fraction: float
    right_goal_zone_fraction: float
    left_side_fraction: float
    right_side_fraction: float
    central_fraction: float
    goal_exclusion_fraction: float

    left_goal_geometry_score: float
    right_goal_geometry_score: float
    referee_geometry_score: float
    outfield_geometry_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class RoleScores:
    goalkeeper_left: float
    goalkeeper_right: float
    referee: float
    outfield: float

    def as_mapping(self) -> Mapping[FootballRole, float]:
        return {
            FootballRole.GOALKEEPER_LEFT: self.goalkeeper_left,
            FootballRole.GOALKEEPER_RIGHT: self.goalkeeper_right,
            FootballRole.REFEREE: self.referee,
            FootballRole.OUTFIELD: self.outfield,
        }

    def to_dict(self) -> dict[str, float]:
        return {
            role.value: float(score)
            for role, score in self.as_mapping().items()
        }


@dataclass(frozen=True)
class TrackRoleDecision:
    track_id: int
    role: FootballRole
    role_score: float
    runner_up_role: FootballRole | None
    runner_up_score: float
    margin: float

    appearance_label: AppearanceLabel
    appearance_purity: float
    ui_color_hex: str | None

    geometry: TrackRoleGeometry
    scores: RoleScores

    reason: str

    @property
    def confident(self) -> bool:
        return self.role is not FootballRole.UNKNOWN

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "role": self.role.value,
            "role_score": self.role_score,
            "runner_up_role": (
                self.runner_up_role.value
                if self.runner_up_role is not None
                else None
            ),
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
            "appearance_label": self.appearance_label.value,
            "appearance_purity": self.appearance_purity,
            "ui_color_hex": self.ui_color_hex,
            "geometry": self.geometry.to_dict(),
            "scores": self.scores.to_dict(),
            "reason": self.reason,
            "confident": self.confident,
        }


@dataclass(frozen=True)
class RoleInferenceResult:
    decisions: Mapping[int, TrackRoleDecision]

    left_goalkeeper_ranking: tuple[int, ...]
    right_goalkeeper_ranking: tuple[int, ...]
    referee_ranking: tuple[int, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "decisions": {
                str(track_id): decision.to_dict()
                for track_id, decision in sorted(self.decisions.items())
            },
            "rankings": {
                "goalkeeper_left": list(self.left_goalkeeper_ranking),
                "goalkeeper_right": list(self.right_goalkeeper_ranking),
                "referee": list(self.referee_ranking),
            },
        }


def infer_roles(
    trajectories: Mapping[int, Sequence[TrackSample]],
    appearance: Mapping[int, TrackAppearanceEvidence] | None = None,
    *,
    config: RoleInferenceConfig | None = None,
) -> RoleInferenceResult:
    """Infer conservative football-role candidates for each raw/fused track."""

    cfg = config or RoleInferenceConfig()
    appearance = appearance or {}

    decisions: dict[int, TrackRoleDecision] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(track_id, raw_samples)
        observed = tuple(
            sample
            for sample in samples
            if sample.position_kind is PositionKind.OBSERVED
        )
        if not observed:
            continue

        appearance_row = appearance.get(
            track_id,
            TrackAppearanceEvidence(
                track_id=track_id,
                label=AppearanceLabel.UNKNOWN,
                top_share=0.0,
                second_share=0.0,
                ui_color_hex=None,
            ),
        )

        geometry = compute_track_role_geometry(
            track_id,
            observed,
            config=cfg,
        )
        scores = compute_role_scores(
            geometry,
            appearance_row,
            config=cfg,
        )
        decisions[track_id] = decide_role(
            track_id,
            geometry,
            appearance_row,
            scores,
            config=cfg,
        )

    left_ranking = _ranking(
        decisions,
        FootballRole.GOALKEEPER_LEFT,
    )
    right_ranking = _ranking(
        decisions,
        FootballRole.GOALKEEPER_RIGHT,
    )
    referee_ranking = _ranking(
        decisions,
        FootballRole.REFEREE,
    )

    return RoleInferenceResult(
        decisions=decisions,
        left_goalkeeper_ranking=left_ranking,
        right_goalkeeper_ranking=right_ranking,
        referee_ranking=referee_ranking,
    )


def compute_track_role_geometry(
    track_id: int,
    observed: Sequence[TrackSample],
    *,
    config: RoleInferenceConfig | None = None,
) -> TrackRoleGeometry:
    cfg = config or RoleInferenceConfig()
    if not observed:
        raise ValueError("compute_track_role_geometry requires observed samples")

    rows = tuple(
        sorted(
            observed,
            key=lambda sample: (
                sample.frame_index,
                sample.timestamp,
            ),
        )
    )

    xs = np.asarray([sample.x for sample in rows], dtype=np.float64)
    zs = np.asarray([sample.z for sample in rows], dtype=np.float64)
    times = np.asarray([sample.timestamp for sample in rows], dtype=np.float64)

    span_s = float(max(0.0, times[-1] - times[0]))
    support_score = _support_score(
        observed_count=len(rows),
        observed_span_s=span_s,
        config=cfg,
    )

    half_length = cfg.field_length_m / 2.0
    half_width = cfg.field_width_m / 2.0

    left_goal_x = -half_length
    right_goal_x = half_length

    left_dist = np.hypot(xs - left_goal_x, zs)
    right_dist = np.hypot(xs - right_goal_x, zs)

    left_zone = (
        (xs <= left_goal_x + cfg.goalkeeper_depth_m)
        & (np.abs(zs) <= cfg.goalkeeper_half_width_m)
    )
    right_zone = (
        (xs >= right_goal_x - cfg.goalkeeper_depth_m)
        & (np.abs(zs) <= cfg.goalkeeper_half_width_m)
    )

    # "Side fraction" is deliberately broader than the goal box. It rewards a
    # keeper that briefly leaves the area but still clearly lives on one end.
    left_side = xs < -cfg.field_length_m / 6.0
    right_side = xs > cfg.field_length_m / 6.0

    central = (
        (np.abs(xs) <= half_length * cfg.referee_central_x_fraction)
        & (np.abs(zs) <= half_width * cfg.referee_central_z_fraction)
    )

    goal_exclusion = (
        (left_dist >= cfg.referee_goal_exclusion_m)
        & (right_dist >= cfg.referee_goal_exclusion_m)
    )

    x_span = float(np.max(xs) - np.min(xs))
    z_span = float(np.max(zs) - np.min(zs))

    path_length = 0.0
    if len(rows) >= 2:
        for left, right in zip(rows, rows[1:]):
            # Only accumulate plausible chronological transitions. Large gaps
            # are still allowed, but no extrapolation is invented.
            if right.timestamp <= left.timestamp:
                continue
            path_length += math.hypot(
                right.x - left.x,
                right.z - left.z,
            )

    speeds = [
        math.hypot(sample.vx, sample.vz)
        for sample in rows
        if math.isfinite(sample.vx) and math.isfinite(sample.vz)
    ]
    median_speed = float(median(speeds)) if speeds else 0.0

    left_goal_geometry = _goalkeeper_geometry_score(
        median_goal_distance=float(np.median(left_dist)),
        goal_zone_fraction=float(np.mean(left_zone)),
        same_side_fraction=float(np.mean(left_side)),
        opposite_side_fraction=float(np.mean(right_side)),
        config=cfg,
    )
    right_goal_geometry = _goalkeeper_geometry_score(
        median_goal_distance=float(np.median(right_dist)),
        goal_zone_fraction=float(np.mean(right_zone)),
        same_side_fraction=float(np.mean(right_side)),
        opposite_side_fraction=float(np.mean(left_side)),
        config=cfg,
    )

    mobility_x = min(1.0, x_span / cfg.referee_span_x_reference_m)
    mobility_z = min(1.0, z_span / cfg.referee_span_z_reference_m)
    mobility = math.sqrt(max(EPS, mobility_x) * max(EPS, mobility_z))

    central_fraction = float(np.mean(central))
    goal_exclusion_fraction = float(np.mean(goal_exclusion))

    referee_geometry = _weighted_geometric_mean(
        (
            (max(EPS, central_fraction), 0.40),
            (max(EPS, goal_exclusion_fraction), 0.35),
            (max(EPS, mobility), 0.25),
        )
    )

    max_goal_geometry = max(
        left_goal_geometry,
        right_goal_geometry,
    )
    broad_field_presence = min(
        1.0,
        0.55 * min(1.0, x_span / 18.0)
        + 0.45 * min(1.0, z_span / 12.0),
    )
    outfield_geometry = _weighted_geometric_mean(
        (
            (max(EPS, 1.0 - max_goal_geometry), 0.65),
            (max(EPS, 0.35 + 0.65 * broad_field_presence), 0.35),
        )
    )

    return TrackRoleGeometry(
        track_id=int(track_id),
        observed_count=len(rows),
        observed_span_s=span_s,
        support_score=float(support_score),
        median_x=float(np.median(xs)),
        median_z=float(np.median(zs)),
        x_span_m=x_span,
        z_span_m=z_span,
        path_length_m=float(path_length),
        median_speed_m_s=median_speed,
        median_left_goal_distance_m=float(np.median(left_dist)),
        median_right_goal_distance_m=float(np.median(right_dist)),
        left_goal_zone_fraction=float(np.mean(left_zone)),
        right_goal_zone_fraction=float(np.mean(right_zone)),
        left_side_fraction=float(np.mean(left_side)),
        right_side_fraction=float(np.mean(right_side)),
        central_fraction=central_fraction,
        goal_exclusion_fraction=goal_exclusion_fraction,
        left_goal_geometry_score=float(left_goal_geometry),
        right_goal_geometry_score=float(right_goal_geometry),
        referee_geometry_score=float(referee_geometry),
        outfield_geometry_score=float(outfield_geometry),
    )


def compute_role_scores(
    geometry: TrackRoleGeometry,
    appearance: TrackAppearanceEvidence,
    *,
    config: RoleInferenceConfig | None = None,
) -> RoleScores:
    cfg = config or RoleInferenceConfig()

    appearance_purity = (
        appearance.purity
        if appearance.purity >= cfg.min_appearance_purity
        else 0.0
    )

    is_team = appearance.label in {
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
    }
    is_special = appearance.label is AppearanceLabel.SPECIAL
    is_mixed = appearance.label is AppearanceLabel.MIXED

    # A goalkeeper may wear either a special kit or, occasionally, be visually
    # absorbed into one team cluster. So appearance only nudges GK scores.
    goalkeeper_appearance = (
        1.0
        if is_special
        else 0.78
        if is_team
        else 0.48
        if is_mixed
        else 0.55
    )
    goalkeeper_appearance = (
        0.55 + 0.45 * appearance_purity
        if is_special
        else goalkeeper_appearance
    )

    # Referee appearance is more diagnostic, but still not decisive on its own.
    referee_appearance = (
        1.0
        if is_special
        else 0.18
        if is_team
        else 0.25
        if is_mixed
        else 0.35
    )
    if is_special:
        referee_appearance = 0.55 + 0.45 * appearance_purity

    outfield_appearance = (
        1.0
        if is_team
        else 0.20
        if is_special
        else 0.35
        if is_mixed
        else 0.45
    )
    if is_team:
        outfield_appearance = 0.55 + 0.45 * appearance_purity

    left = geometry.support_score * _weighted_geometric_mean(
        (
            (max(EPS, geometry.left_goal_geometry_score), 0.78),
            (max(EPS, goalkeeper_appearance), 0.22),
        )
    )
    right = geometry.support_score * _weighted_geometric_mean(
        (
            (max(EPS, geometry.right_goal_geometry_score), 0.78),
            (max(EPS, goalkeeper_appearance), 0.22),
        )
    )
    referee = geometry.support_score * _weighted_geometric_mean(
        (
            (max(EPS, geometry.referee_geometry_score), 0.62),
            (max(EPS, referee_appearance), 0.38),
        )
    )
    outfield = geometry.support_score * _weighted_geometric_mean(
        (
            (max(EPS, geometry.outfield_geometry_score), 0.58),
            (max(EPS, outfield_appearance), 0.42),
        )
    )

    return RoleScores(
        goalkeeper_left=float(_clamp01(left)),
        goalkeeper_right=float(_clamp01(right)),
        referee=float(_clamp01(referee)),
        outfield=float(_clamp01(outfield)),
    )


def decide_role(
    track_id: int,
    geometry: TrackRoleGeometry,
    appearance: TrackAppearanceEvidence,
    scores: RoleScores,
    *,
    config: RoleInferenceConfig | None = None,
) -> TrackRoleDecision:
    cfg = config or RoleInferenceConfig()

    ranking = sorted(
        scores.as_mapping().items(),
        key=lambda item: (
            -item[1],
            item[0].value,
        ),
    )
    best_role, best_score = ranking[0]
    runner_role, runner_score = ranking[1]
    margin = float(best_score - runner_score)

    enough_support = (
        geometry.observed_count >= cfg.min_observed_count
        and geometry.observed_span_s >= cfg.min_observed_span_s
    )

    threshold = {
        FootballRole.GOALKEEPER_LEFT: cfg.min_goalkeeper_score,
        FootballRole.GOALKEEPER_RIGHT: cfg.min_goalkeeper_score,
        FootballRole.REFEREE: cfg.min_referee_score,
        FootballRole.OUTFIELD: cfg.min_outfield_score,
    }[best_role]

    if not enough_support:
        role = FootballRole.UNKNOWN
        reason = (
            "insufficient_observed_support:"
            f"count={geometry.observed_count},"
            f"span_s={geometry.observed_span_s:.3f}"
        )
    elif best_score < threshold:
        role = FootballRole.UNKNOWN
        reason = (
            "best_score_below_threshold:"
            f"{best_role.value}={best_score:.3f}<"
            f"{threshold:.3f}"
        )
    elif margin < cfg.min_role_margin:
        role = FootballRole.UNKNOWN
        reason = (
            "role_margin_too_small:"
            f"best={best_role.value}:{best_score:.3f},"
            f"runner_up={runner_role.value}:{runner_score:.3f},"
            f"margin={margin:.3f}"
        )
    else:
        role = best_role
        reason = "confident"

    return TrackRoleDecision(
        track_id=int(track_id),
        role=role,
        role_score=float(best_score if role is not FootballRole.UNKNOWN else 0.0),
        runner_up_role=runner_role,
        runner_up_score=float(runner_score),
        margin=margin,
        appearance_label=appearance.label,
        appearance_purity=float(appearance.purity),
        ui_color_hex=appearance.ui_color_hex,
        geometry=geometry,
        scores=scores,
        reason=reason,
    )


def _goalkeeper_geometry_score(
    *,
    median_goal_distance: float,
    goal_zone_fraction: float,
    same_side_fraction: float,
    opposite_side_fraction: float,
    config: RoleInferenceConfig,
) -> float:
    distance_term = math.exp(
        -0.5
        * (
            median_goal_distance
            / config.goalkeeper_distance_sigma_m
        ) ** 2
    )
    side_term = max(
        EPS,
        same_side_fraction
        * (
            1.0
            - config.goalkeeper_opposite_side_penalty
            * opposite_side_fraction
        ),
    )
    zone_term = max(EPS, goal_zone_fraction)

    return _weighted_geometric_mean(
        (
            (max(EPS, distance_term), 0.42),
            (zone_term, 0.38),
            (side_term, 0.20),
        )
    )


def _support_score(
    *,
    observed_count: int,
    observed_span_s: float,
    config: RoleInferenceConfig,
) -> float:
    count_term = min(
        1.0,
        observed_count
        / config.support_observed_count_reference,
    )
    span_term = min(
        1.0,
        observed_span_s
        / config.support_span_reference_s,
    )
    return math.sqrt(
        max(EPS, count_term)
        * max(EPS, span_term)
    )


def _ranking(
    decisions: Mapping[int, TrackRoleDecision],
    role: FootballRole,
) -> tuple[int, ...]:
    return tuple(
        track_id
        for track_id, _ in sorted(
            decisions.items(),
            key=lambda item: (
                -item[1].scores.as_mapping()[role],
                item[0],
            ),
        )
    )


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    rows = tuple(
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

    for sample in rows:
        if sample.track_id != track_id:
            raise ValueError(
                f"trajectory key {track_id} does not match sample "
                f"track_id {sample.track_id}"
            )
        if not all(
            math.isfinite(value)
            for value in (
                sample.timestamp,
                sample.x,
                sample.z,
                sample.vx,
                sample.vz,
            )
        ):
            raise ValueError(f"track {track_id} contains non-finite values")
        if (
            previous_frame is not None
            and sample.frame_index == previous_frame
        ):
            raise ValueError(
                f"track {track_id} contains duplicate frame "
                f"{sample.frame_index}"
            )
        if (
            previous_timestamp is not None
            and sample.timestamp <= previous_timestamp
        ):
            raise ValueError(
                f"track {track_id} timestamps must be strictly increasing"
            )

        previous_frame = sample.frame_index
        previous_timestamp = sample.timestamp

    return rows


def _weighted_geometric_mean(
    terms: Sequence[tuple[float, float]],
) -> float:
    total_weight = sum(weight for _, weight in terms)
    if total_weight <= 0.0:
        raise ValueError("at least one positive weight is required")

    acc = 0.0
    for value, weight in terms:
        if weight <= 0.0:
            continue
        acc += weight * math.log(
            max(EPS, min(1.0, float(value)))
        )

    return math.exp(acc / total_weight)


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))
