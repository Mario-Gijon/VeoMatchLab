"""Sprint B.2.1: conservative global selection for unique football roles.

Per-track role scores are useful diagnostics, but independent classification can
produce nonsense such as four simultaneous "left goalkeepers".  This module
adds only global *selection constraints*:

  * at most one GOALKEEPER_LEFT,
  * at most one GOALKEEPER_RIGHT,
  * at most one REFEREE.

A winner is selected only when:
  * its absolute score clears the existing role threshold, and
  * it beats the next candidate by a configurable global margin.

No role is forced.  Missing / weak evidence remains UNKNOWN.
OUTFIELD is intentionally not globally constrained.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .role_inference import (
    FootballRole,
    RoleInferenceResult,
    TrackRoleDecision,
)


@dataclass(frozen=True)
class GlobalRoleSelectionConfig:
    min_goalkeeper_score: float = 0.72
    min_referee_score: float = 0.72

    min_goalkeeper_global_margin: float = 0.08
    min_referee_global_margin: float = 0.10

    def __post_init__(self) -> None:
        for name in (
            "min_goalkeeper_score",
            "min_referee_score",
            "min_goalkeeper_global_margin",
            "min_referee_global_margin",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class GlobalRoleWinner:
    role: FootballRole
    track_id: int | None
    score: float
    runner_up_track_id: int | None
    runner_up_score: float
    margin: float
    reason: str

    @property
    def selected(self) -> bool:
        return self.track_id is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role.value,
            "track_id": self.track_id,
            "score": self.score,
            "runner_up_track_id": self.runner_up_track_id,
            "runner_up_score": self.runner_up_score,
            "margin": self.margin,
            "reason": self.reason,
            "selected": self.selected,
        }


@dataclass(frozen=True)
class GlobalRoleSelection:
    goalkeeper_left: GlobalRoleWinner
    goalkeeper_right: GlobalRoleWinner
    referee: GlobalRoleWinner

    def winners_by_role(self) -> Mapping[FootballRole, GlobalRoleWinner]:
        return {
            FootballRole.GOALKEEPER_LEFT: self.goalkeeper_left,
            FootballRole.GOALKEEPER_RIGHT: self.goalkeeper_right,
            FootballRole.REFEREE: self.referee,
        }

    def selected_track_ids(self) -> Mapping[FootballRole, int]:
        return {
            role: winner.track_id
            for role, winner in self.winners_by_role().items()
            if winner.track_id is not None
        }

    def to_dict(self) -> dict[str, object]:
        return {
            role.value: winner.to_dict()
            for role, winner in self.winners_by_role().items()
        }


def select_global_unique_roles(
    result: RoleInferenceResult,
    *,
    config: GlobalRoleSelectionConfig | None = None,
) -> GlobalRoleSelection:
    cfg = config or GlobalRoleSelectionConfig()

    left = _select_one(
        result.decisions,
        FootballRole.GOALKEEPER_LEFT,
        min_score=cfg.min_goalkeeper_score,
        min_margin=cfg.min_goalkeeper_global_margin,
        blocked=frozenset(),
    )

    blocked = (
        frozenset({left.track_id})
        if left.track_id is not None
        else frozenset()
    )

    right = _select_one(
        result.decisions,
        FootballRole.GOALKEEPER_RIGHT,
        min_score=cfg.min_goalkeeper_score,
        min_margin=cfg.min_goalkeeper_global_margin,
        blocked=blocked,
    )

    blocked = frozenset(
        track_id
        for track_id in (
            left.track_id,
            right.track_id,
        )
        if track_id is not None
    )

    referee = _select_one(
        result.decisions,
        FootballRole.REFEREE,
        min_score=cfg.min_referee_score,
        min_margin=cfg.min_referee_global_margin,
        blocked=blocked,
    )

    return GlobalRoleSelection(
        goalkeeper_left=left,
        goalkeeper_right=right,
        referee=referee,
    )


def _select_one(
    decisions: Mapping[int, TrackRoleDecision],
    role: FootballRole,
    *,
    min_score: float,
    min_margin: float,
    blocked: frozenset[int],
) -> GlobalRoleWinner:
    ranked = sorted(
        (
            (
                track_id,
                float(
                    decision.scores.as_mapping()[role]
                ),
            )
            for track_id, decision in decisions.items()
            if track_id not in blocked
        ),
        key=lambda item: (
            -item[1],
            item[0],
        ),
    )

    if not ranked:
        return GlobalRoleWinner(
            role=role,
            track_id=None,
            score=0.0,
            runner_up_track_id=None,
            runner_up_score=0.0,
            margin=0.0,
            reason="no_candidates",
        )

    best_id, best_score = ranked[0]
    if len(ranked) >= 2:
        runner_id, runner_score = ranked[1]
    else:
        runner_id, runner_score = None, 0.0

    margin = best_score - runner_score

    if best_score < min_score:
        return GlobalRoleWinner(
            role=role,
            track_id=None,
            score=best_score,
            runner_up_track_id=runner_id,
            runner_up_score=runner_score,
            margin=margin,
            reason=(
                f"best_score_below_threshold:"
                f"{best_score:.3f}<{min_score:.3f}"
            ),
        )

    if margin < min_margin:
        return GlobalRoleWinner(
            role=role,
            track_id=None,
            score=best_score,
            runner_up_track_id=runner_id,
            runner_up_score=runner_score,
            margin=margin,
            reason=(
                f"global_margin_too_small:"
                f"{margin:.3f}<{min_margin:.3f}"
            ),
        )

    return GlobalRoleWinner(
        role=role,
        track_id=best_id,
        score=best_score,
        runner_up_track_id=runner_id,
        runner_up_score=runner_score,
        margin=margin,
        reason="selected",
    )
