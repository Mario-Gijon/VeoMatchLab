from __future__ import annotations

import math

import pytest

from tracking.models import (
    PositionKind,
    TrackLifecycle,
    TrackSample,
)
from tracking.role_inference import (
    FootballRole,
    RoleInferenceConfig,
    TrackAppearanceEvidence,
    compute_track_role_geometry,
    infer_roles,
)
from tracking.team_appearance import AppearanceLabel


DT = 1.0 / 15.0


def sample(
    track_id: int,
    frame: int,
    *,
    x: float,
    z: float,
    vx: float = 0.0,
    vz: float = 0.0,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame,
        timestamp=frame * DT,
        x=x,
        z=z,
        vx=vx,
        vz=vz,
        lifecycle=TrackLifecycle.ACTIVE,
        position_kind=PositionKind.OBSERVED,
        measurement_id=f"{frame}:{track_id}",
        confidence=0.9,
    )


def trajectory(
    track_id: int,
    *,
    frames: int,
    x_fn,
    z_fn,
) -> tuple[TrackSample, ...]:
    rows = []
    previous_x = None
    previous_z = None

    for frame in range(frames):
        x = float(x_fn(frame))
        z = float(z_fn(frame))
        if previous_x is None:
            vx = 0.0
            vz = 0.0
        else:
            vx = (x - previous_x) / DT
            vz = (z - previous_z) / DT

        rows.append(
            sample(
                track_id,
                frame,
                x=x,
                z=z,
                vx=vx,
                vz=vz,
            )
        )
        previous_x = x
        previous_z = z

    return tuple(rows)


def appearance(
    track_id: int,
    label: AppearanceLabel,
    *,
    purity: float = 0.95,
) -> TrackAppearanceEvidence:
    return TrackAppearanceEvidence(
        track_id=track_id,
        label=label,
        top_share=purity,
        second_share=max(0.0, 1.0 - purity),
        ui_color_hex="#123456",
    )


def test_left_goalkeeper_geometry_is_strong_near_left_goal():
    rows = trajectory(
        1,
        frames=150,
        x_fn=lambda i: -48.0 + 1.5 * math.sin(i / 20.0),
        z_fn=lambda i: 5.0 * math.sin(i / 12.0),
    )

    geometry = compute_track_role_geometry(
        1,
        rows,
    )

    assert geometry.left_goal_zone_fraction > 0.90
    assert geometry.left_goal_geometry_score > 0.75
    assert geometry.right_goal_geometry_score < 0.05


def test_right_goalkeeper_is_classified_conservatively():
    rows = trajectory(
        2,
        frames=180,
        x_fn=lambda i: 48.0 + 1.2 * math.sin(i / 15.0),
        z_fn=lambda i: 4.0 * math.sin(i / 10.0),
    )

    result = infer_roles(
        {2: rows},
        {
            2: appearance(
                2,
                AppearanceLabel.SPECIAL,
            )
        },
    )

    assert result.decisions[2].role is FootballRole.GOALKEEPER_RIGHT
    assert result.decisions[2].scores.goalkeeper_right > 0.72


def test_special_mobile_central_track_can_be_referee():
    rows = trajectory(
        3,
        frames=240,
        x_fn=lambda i: -18.0 + 36.0 * (i / 239.0),
        z_fn=lambda i: 12.0 * math.sin(i / 22.0),
    )

    result = infer_roles(
        {3: rows},
        {
            3: appearance(
                3,
                AppearanceLabel.SPECIAL,
            )
        },
    )

    decision = result.decisions[3]
    assert decision.role is FootballRole.REFEREE
    assert decision.scores.referee > decision.scores.outfield


def test_team_coloured_mobile_track_prefers_outfield_over_referee():
    rows = trajectory(
        4,
        frames=240,
        x_fn=lambda i: -25.0 + 50.0 * (i / 239.0),
        z_fn=lambda i: 14.0 * math.sin(i / 20.0),
    )

    result = infer_roles(
        {4: rows},
        {
            4: appearance(
                4,
                AppearanceLabel.TEAM_A,
            )
        },
    )

    decision = result.decisions[4]
    assert decision.role is FootballRole.OUTFIELD
    assert decision.scores.outfield > decision.scores.referee


def test_short_ghost_track_stays_unknown_even_if_near_goal():
    rows = trajectory(
        5,
        frames=8,
        x_fn=lambda i: -50.0,
        z_fn=lambda i: 0.5 * i,
    )

    result = infer_roles(
        {5: rows},
        {
            5: appearance(
                5,
                AppearanceLabel.SPECIAL,
            )
        },
    )

    decision = result.decisions[5]
    assert decision.role is FootballRole.UNKNOWN
    assert "insufficient_observed_support" in decision.reason


def test_ambiguous_special_track_can_remain_unknown():
    rows = trajectory(
        6,
        frames=120,
        x_fn=lambda i: -24.0 + 8.0 * math.sin(i / 30.0),
        z_fn=lambda i: 6.0 * math.sin(i / 15.0),
    )

    cfg = RoleInferenceConfig(
        min_role_margin=0.20,
        min_goalkeeper_score=0.75,
        min_referee_score=0.75,
        min_outfield_score=0.75,
    )

    result = infer_roles(
        {6: rows},
        {
            6: appearance(
                6,
                AppearanceLabel.SPECIAL,
                purity=0.72,
            )
        },
        config=cfg,
    )

    assert result.decisions[6].role is FootballRole.UNKNOWN


def test_rankings_do_not_force_role_assignment():
    left_keeper = trajectory(
        1,
        frames=150,
        x_fn=lambda i: -49.0,
        z_fn=lambda i: 4.0 * math.sin(i / 10.0),
    )
    ghost = trajectory(
        2,
        frames=7,
        x_fn=lambda i: 49.0,
        z_fn=lambda i: 0.0,
    )
    referee = trajectory(
        3,
        frames=180,
        x_fn=lambda i: -15.0 + 30.0 * (i / 179.0),
        z_fn=lambda i: 10.0 * math.sin(i / 18.0),
    )

    result = infer_roles(
        {
            1: left_keeper,
            2: ghost,
            3: referee,
        },
        {
            1: appearance(1, AppearanceLabel.SPECIAL),
            2: appearance(2, AppearanceLabel.SPECIAL),
            3: appearance(3, AppearanceLabel.SPECIAL),
        },
    )

    assert result.left_goalkeeper_ranking[0] == 1
    assert result.referee_ranking[0] == 3
    assert result.decisions[2].role is FootballRole.UNKNOWN


def test_report_row_parser_handles_unknown_label():
    row = TrackAppearanceEvidence.from_report_row(
        9,
        {
            "label": "something_new",
            "top_share": 0.8,
            "second_share": 0.2,
            "ui_color_hex": "#ABCDEF",
        },
    )

    assert row.label is AppearanceLabel.UNKNOWN
    assert row.top_share == pytest.approx(0.8)
    assert row.ui_color_hex == "#ABCDEF"
