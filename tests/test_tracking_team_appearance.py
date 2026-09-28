from __future__ import annotations

import numpy as np
import pytest

from tracking.team_appearance import (
    AppearanceLabel,
    KitAppearanceConfig,
    appearance_similarity,
    extract_crop_appearance,
    fit_kit_appearance,
    rgb_to_hex,
)


def synthetic_player_crop(
    *,
    torso_bgr: tuple[int, int, int],
    lower_bgr: tuple[int, int, int],
    grass_bgr: tuple[int, int, int] = (45, 100, 45),
    jitter: int = 0,
) -> np.ndarray:
    image = np.zeros((256, 128, 3), dtype=np.uint8)
    image[:] = grass_bgr

    # Person silhouette roughly matches the rectified Sprint A crop geometry.
    image[25:150, 38:90] = torso_bgr
    image[145:235, 42:86] = lower_bgr

    if jitter:
        noise = np.random.default_rng(jitter).integers(
            -jitter,
            jitter + 1,
            size=image.shape,
        )
        image = np.clip(
            image.astype(np.int16) + noise,
            0,
            255,
        ).astype(np.uint8)

    return image


def crop(
    track_id: int,
    frame: int,
    *,
    torso_bgr: tuple[int, int, int],
    lower_bgr: tuple[int, int, int],
    quality: float = 0.9,
):
    return extract_crop_appearance(
        synthetic_player_crop(
            torso_bgr=torso_bgr,
            lower_bgr=lower_bgr,
            jitter=(frame % 3),
        ),
        track_id=track_id,
        frame_index=frame,
        quality=quality,
    )


def test_same_kit_similarity_is_higher_than_red_vs_white():
    red_a = crop(
        1,
        0,
        torso_bgr=(30, 35, 210),
        lower_bgr=(20, 20, 35),
    )
    red_b = crop(
        2,
        1,
        torso_bgr=(35, 40, 220),
        lower_bgr=(22, 22, 40),
    )
    white = crop(
        3,
        2,
        torso_bgr=(225, 225, 225),
        lower_bgr=(230, 230, 230),
    )

    same = appearance_similarity(red_a, red_b)
    different = appearance_similarity(red_a, white)

    assert same > different
    assert same > 0.65
    assert same > different * 2.0


def test_green_pitch_is_suppressed_from_representative_colour():
    item = crop(
        1,
        0,
        torso_bgr=(20, 30, 220),
        lower_bgr=(15, 15, 30),
    )

    r, g, b = item.representative_rgb

    assert r > 150
    assert r > g * 2
    assert r > b * 2


def test_two_dominant_teams_and_small_special_cluster_are_discovered():
    crops = []

    # Red team: 3 tracks x 4 crops.
    for track_id in (1, 2, 3):
        for j in range(4):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=(25, 35, 215 + j),
                    lower_bgr=(20, 20, 35),
                )
            )

    # White team: 3 tracks x 4 crops.
    for track_id in (10, 11, 12):
        for j in range(4):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=(220 + j, 220 + j, 220 + j),
                    lower_bgr=(215, 215, 215),
                )
            )

    # One dark special track, e.g. referee/goalkeeper candidate.
    for j in range(4):
        crops.append(
            crop(
                20,
                j,
                torso_bgr=(25, 25, 25),
                lower_bgr=(30, 30, 30),
            )
        )

    result = fit_kit_appearance(
        crops,
        config=KitAppearanceConfig(
            min_clusters=3,
            max_clusters=3,
            min_team_distinct_tracks=2,
        ),
    )

    labels = {
        decision.label
        for decision in result.track_decisions.values()
    }

    assert AppearanceLabel.TEAM_A in labels
    assert AppearanceLabel.TEAM_B in labels
    assert result.track_decisions[20].label is AppearanceLabel.SPECIAL


def test_track_with_red_and_white_crops_is_marked_mixed():
    crops = []

    # Stable red and white populations define the two team clusters.
    for track_id in (1, 2):
        for j in range(5):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=(25, 30, 220),
                    lower_bgr=(20, 20, 30),
                )
            )
    for track_id in (10, 11):
        for j in range(5):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=(225, 225, 225),
                    lower_bgr=(225, 225, 225),
                )
            )

    # Raw track 99 switches physical person / kit halfway through.
    for j in range(3):
        crops.append(
            crop(
                99,
                j,
                torso_bgr=(25, 30, 220),
                lower_bgr=(20, 20, 30),
            )
        )
    for j in range(3, 6):
        crops.append(
            crop(
                99,
                j,
                torso_bgr=(225, 225, 225),
                lower_bgr=(225, 225, 225),
            )
        )

    result = fit_kit_appearance(
        crops,
        config=KitAppearanceConfig(
            min_clusters=2,
            max_clusters=2,
            min_track_purity=0.70,
            mixed_second_share=0.22,
        ),
    )

    decision = result.track_decisions[99]

    assert decision.label is AppearanceLabel.MIXED
    assert decision.top_share < 0.70
    assert decision.second_share > 0.22


def test_track_with_too_few_crops_is_unknown():
    crops = []

    for track_id in (1, 2):
        for j in range(4):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=(25, 30, 220),
                    lower_bgr=(20, 20, 30),
                )
            )
    for track_id in (10, 11):
        for j in range(4):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=(225, 225, 225),
                    lower_bgr=(225, 225, 225),
                )
            )

    crops.append(
        crop(
            77,
            0,
            torso_bgr=(25, 30, 220),
            lower_bgr=(20, 20, 30),
        )
    )

    result = fit_kit_appearance(
        crops,
        config=KitAppearanceConfig(
            min_clusters=2,
            max_clusters=2,
            min_track_crops=3,
        ),
    )

    assert result.track_decisions[77].label is AppearanceLabel.UNKNOWN


def test_ui_colour_is_hex():
    assert rgb_to_hex((255, 0, 16)) == "#FF0010"


def test_clustering_is_deterministic():
    crops = []
    for track_id, colour in (
        (1, (20, 30, 220)),
        (2, (20, 30, 220)),
        (10, (225, 225, 225)),
        (11, (225, 225, 225)),
    ):
        for j in range(4):
            crops.append(
                crop(
                    track_id,
                    j,
                    torso_bgr=colour,
                    lower_bgr=colour,
                )
            )

    cfg = KitAppearanceConfig(
        min_clusters=2,
        max_clusters=3,
    )
    first = fit_kit_appearance(crops, config=cfg)
    second = fit_kit_appearance(crops, config=cfg)

    assert first.chosen_k == second.chosen_k
    assert first.crop_cluster_ids == second.crop_cluster_ids
    assert {
        track_id: decision.label
        for track_id, decision in first.track_decisions.items()
    } == {
        track_id: decision.label
        for track_id, decision in second.track_decisions.items()
    }
