from __future__ import annotations

import numpy as np

from tracking_v0.team_appearance import (
    AppearanceCluster,
    AppearanceLabel,
    CropAppearance,
    KitAppearanceResult,
    TrackAppearanceDecision,
)
from tracking_v0.team_similarity import (
    RelativeTeamLabel,
    TeamSimilarityConfig,
    compute_team_similarity,
)


def unit(vec):
    arr = np.asarray(vec, dtype=np.float64)
    return arr / np.linalg.norm(arr)


def crop(track_id, frame, vec, quality=0.9):
    return CropAppearance(
        track_id=track_id,
        frame_index=frame,
        quality=quality,
        descriptor=unit(vec),
        representative_rgb=(128, 128, 128),
        torso_valid_fraction=1.0,
        lower_valid_fraction=1.0,
    )


def decision(track_id, label, top=1.0):
    return TrackAppearanceDecision(
        track_id=track_id,
        label=label,
        cluster_id=0,
        crop_count=4,
        support_weight=4.0,
        top_share=top,
        second_share=1.0 - top,
        representative_rgb=(128, 128, 128),
        ui_color_hex="#808080",
        cluster_shares={0: top},
    )


def make_result(crops, cluster_ids, decisions):
    clusters = (
        AppearanceCluster(
            cluster_id=0,
            crop_count=sum(x == 0 for x in cluster_ids),
            distinct_track_count=2,
            support_weight=10.0,
            representative_rgb=(180, 30, 30),
            ui_color_hex="#B41E1E",
            semantic_label=AppearanceLabel.TEAM_A,
        ),
        AppearanceCluster(
            cluster_id=1,
            crop_count=sum(x == 1 for x in cluster_ids),
            distinct_track_count=2,
            support_weight=10.0,
            representative_rgb=(220, 220, 220),
            ui_color_hex="#DCDCDC",
            semantic_label=AppearanceLabel.TEAM_B,
        ),
    )
    return KitAppearanceResult(
        chosen_k=2,
        silhouette_score=0.5,
        crop_cluster_ids=tuple(cluster_ids),
        clusters=clusters,
        track_decisions=decisions,
    )


def test_relative_team_similarity_prefers_nearest_prototype():
    crops = [
        crop(1, 0, [1.0, 0.0, 0.0]),
        crop(1, 1, [0.95, 0.05, 0.0]),
        crop(2, 0, [0.0, 1.0, 0.0]),
        crop(2, 1, [0.05, 0.95, 0.0]),
    ]
    result = make_result(
        crops,
        [0, 0, 1, 1],
        {
            1: decision(1, AppearanceLabel.TEAM_A),
            2: decision(2, AppearanceLabel.TEAM_B),
        },
    )

    scored = compute_team_similarity(
        crops,
        result,
        config=TeamSimilarityConfig(
            min_track_crops=2,
            min_track_similarity=0.4,
            min_track_margin=0.1,
            min_track_vote_share=0.5,
        ),
    )

    assert scored.tracks[1].label is RelativeTeamLabel.TEAM_A
    assert scored.tracks[2].label is RelativeTeamLabel.TEAM_B
    assert scored.tracks[1].margin > 0.5
    assert scored.tracks[2].margin > 0.5


def test_ambiguous_greyish_track_remains_unknown():
    crops = [
        crop(1, 0, [1.0, 0.0]),
        crop(1, 1, [1.0, 0.0]),
        crop(2, 0, [0.0, 1.0]),
        crop(2, 1, [0.0, 1.0]),
        crop(3, 0, [0.72, 0.69]),
        crop(3, 1, [0.70, 0.71]),
        crop(3, 2, [0.71, 0.70]),
    ]
    result = make_result(
        crops,
        [0, 0, 1, 1, 0, 1, 0],
        {
            1: decision(1, AppearanceLabel.TEAM_A),
            2: decision(2, AppearanceLabel.TEAM_B),
            3: decision(3, AppearanceLabel.MIXED, top=0.55),
        },
    )

    scored = compute_team_similarity(
        crops,
        result,
        config=TeamSimilarityConfig(
            min_track_crops=3,
            min_track_similarity=0.4,
            min_track_margin=0.12,
            min_track_vote_share=0.6,
        ),
    )

    assert scored.tracks[3].label is RelativeTeamLabel.UNKNOWN
    assert scored.tracks[3].margin < 0.12


def test_special_track_is_not_force_assigned_to_team():
    crops = [
        crop(1, 0, [1.0, 0.0]),
        crop(1, 1, [1.0, 0.0]),
        crop(2, 0, [0.0, 1.0]),
        crop(2, 1, [0.0, 1.0]),
        crop(9, 0, [0.95, 0.05]),
        crop(9, 1, [0.96, 0.04]),
        crop(9, 2, [0.94, 0.06]),
    ]
    result = make_result(
        crops,
        [0, 0, 1, 1, 0, 0, 0],
        {
            1: decision(1, AppearanceLabel.TEAM_A),
            2: decision(2, AppearanceLabel.TEAM_B),
            9: decision(9, AppearanceLabel.SPECIAL),
        },
    )

    scored = compute_team_similarity(
        crops,
        result,
        config=TeamSimilarityConfig(
            min_track_crops=3,
            min_track_similarity=0.4,
            min_track_margin=0.1,
            min_track_vote_share=0.5,
            preserve_special=True,
        ),
    )

    assert scored.tracks[9].label is RelativeTeamLabel.UNKNOWN
    assert scored.tracks[9].reason == "special_appearance_preserved"


def test_prototype_is_robust_to_one_switched_crop():
    crops = [
        crop(1, 0, [1.0, 0.0]),
        crop(1, 1, [1.0, 0.0]),
        crop(1, 2, [1.0, 0.0]),
        crop(1, 3, [0.0, 1.0]),  # contaminated team-A cluster crop
        crop(2, 0, [0.0, 1.0]),
        crop(2, 1, [0.0, 1.0]),
        crop(2, 2, [0.0, 1.0]),
        crop(2, 3, [0.0, 1.0]),
    ]
    result = make_result(
        crops,
        [0, 0, 0, 0, 1, 1, 1, 1],
        {
            1: decision(1, AppearanceLabel.TEAM_A),
            2: decision(2, AppearanceLabel.TEAM_B),
        },
    )

    scored = compute_team_similarity(
        crops,
        result,
        config=TeamSimilarityConfig(
            min_track_crops=3,
            min_track_similarity=0.4,
            min_track_margin=0.1,
            min_track_vote_share=0.5,
        ),
    )

    assert scored.tracks[1].team_a_similarity > scored.tracks[1].team_b_similarity
