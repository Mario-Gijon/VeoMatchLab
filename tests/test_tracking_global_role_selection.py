from __future__ import annotations

from tracking_v0.global_role_selection import (
    GlobalRoleSelectionConfig,
    select_global_unique_roles,
)
from tracking_v0.role_inference import (
    FootballRole,
    RoleInferenceResult,
    RoleScores,
    TrackAppearanceEvidence,
    TrackRoleDecision,
    TrackRoleGeometry,
)
from tracking_v0.team_appearance import AppearanceLabel


def geometry(track_id: int):
    return TrackRoleGeometry(
        track_id=track_id,
        observed_count=100,
        observed_span_s=10.0,
        support_score=1.0,
        median_x=0.0,
        median_z=0.0,
        x_span_m=20.0,
        z_span_m=10.0,
        path_length_m=30.0,
        median_speed_m_s=2.0,
        median_left_goal_distance_m=20.0,
        median_right_goal_distance_m=80.0,
        left_goal_zone_fraction=0.5,
        right_goal_zone_fraction=0.0,
        left_side_fraction=0.8,
        right_side_fraction=0.1,
        central_fraction=0.2,
        goal_exclusion_fraction=0.5,
        left_goal_geometry_score=0.8,
        right_goal_geometry_score=0.1,
        referee_geometry_score=0.2,
        outfield_geometry_score=0.5,
    )


def decision(track_id, *, gl, gr, ref, out):
    scores = RoleScores(
        goalkeeper_left=gl,
        goalkeeper_right=gr,
        referee=ref,
        outfield=out,
    )
    ranking = sorted(scores.as_mapping().items(), key=lambda x: -x[1])
    return TrackRoleDecision(
        track_id=track_id,
        role=ranking[0][0],
        role_score=ranking[0][1],
        runner_up_role=ranking[1][0],
        runner_up_score=ranking[1][1],
        margin=ranking[0][1] - ranking[1][1],
        appearance_label=AppearanceLabel.SPECIAL,
        appearance_purity=1.0,
        ui_color_hex="#000000",
        geometry=geometry(track_id),
        scores=scores,
        reason="test",
    )


def result(decisions):
    ids = tuple(sorted(decisions))
    return RoleInferenceResult(
        decisions=decisions,
        left_goalkeeper_ranking=ids,
        right_goalkeeper_ranking=ids,
        referee_ranking=ids,
    )


def test_selects_only_one_left_goalkeeper():
    r = result({
        5: decision(5, gl=0.94, gr=0.01, ref=0.1, out=0.2),
        38: decision(38, gl=0.82, gr=0.01, ref=0.1, out=0.4),
        3: decision(3, gl=0.76, gr=0.01, ref=0.1, out=0.5),
    })

    selected = select_global_unique_roles(r)

    assert selected.goalkeeper_left.track_id == 5
    assert selected.goalkeeper_left.margin == 0.12


def test_weak_right_goalkeeper_remains_unselected():
    r = result({
        5: decision(5, gl=0.94, gr=0.01, ref=0.1, out=0.2),
        23: decision(23, gl=0.05, gr=0.33, ref=0.1, out=0.4),
    })

    selected = select_global_unique_roles(r)

    assert selected.goalkeeper_right.track_id is None
    assert "below_threshold" in selected.goalkeeper_right.reason


def test_weak_referee_remains_unselected():
    r = result({
        5: decision(5, gl=0.94, gr=0.01, ref=0.10, out=0.2),
        20: decision(20, gl=0.01, gr=0.01, ref=0.46, out=0.9),
    })

    selected = select_global_unique_roles(r)

    assert selected.referee.track_id is None


def test_global_margin_can_reject_close_goalkeeper_candidates():
    r = result({
        1: decision(1, gl=0.86, gr=0.01, ref=0.1, out=0.2),
        2: decision(2, gl=0.83, gr=0.01, ref=0.1, out=0.3),
    })

    selected = select_global_unique_roles(
        r,
        config=GlobalRoleSelectionConfig(
            min_goalkeeper_score=0.72,
            min_referee_score=0.72,
            min_goalkeeper_global_margin=0.08,
            min_referee_global_margin=0.10,
        ),
    )

    assert selected.goalkeeper_left.track_id is None
    assert "global_margin_too_small" in selected.goalkeeper_left.reason
