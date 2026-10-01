from __future__ import annotations

import cv2
import numpy as np

from tracking.appearance.kit_families import (
    KitFamilyConfig,
    KitFamilyProfile,
    build_kit_family_profiles,
    cluster_kit_families,
    extract_kit_family_sample,
)


def _crop(
    *,
    torso=(0, 0, 255),
    shorts=(0, 0, 180),
    grass=(0, 150, 0),
    scale=1.0,
):
    image = np.zeros((120, 52, 3), dtype=np.uint8)
    image[:] = np.asarray(grass, dtype=np.float64) * scale
    image = np.clip(image, 0, 255).astype(np.uint8)

    t = np.clip(
        np.asarray(torso, dtype=np.float64) * scale,
        0, 255,
    ).astype(np.uint8)
    s = np.clip(
        np.asarray(shorts, dtype=np.float64) * scale,
        0, 255,
    ).astype(np.uint8)
    cv2.rectangle(image, (10, 12), (42, 68), tuple(map(int, t)), -1)
    cv2.rectangle(image, (13, 62), (39, 104), tuple(map(int, s)), -1)
    return image


def _sample(tracklet_id, detection_id, image, t=0.0):
    return extract_kit_family_sample(
        image,
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        frame_index=int(t * 10),
        timestamp_s=t,
        crop_ref=f"{detection_id}.jpg",
        quality=0.95,
    )


def _profile(tracklet_id, vector, reliability=0.85, samples=3):
    vector = np.asarray(vector, dtype=np.float64)
    vector /= np.linalg.norm(vector)
    return KitFamilyProfile(
        tracklet_id=tracklet_id,
        signature=tuple(float(v) for v in vector),
        sample_count=samples,
        reliability=reliability,
        first_timestamp_s=0.0,
        last_timestamp_s=1.0,
    )


def test_same_kit_bright_and_shadow_have_high_similarity():
    bright = _sample(
        "a", "a1",
        _crop(
            torso=(0, 0, 230),
            shorts=(10, 10, 180),
            grass=(0, 160, 0),
            scale=1.0,
        ),
    )
    shadow = _sample(
        "b", "b1",
        _crop(
            torso=(0, 0, 230),
            shorts=(10, 10, 180),
            grass=(0, 160, 0),
            scale=0.48,
        ),
    )
    assert bright is not None
    assert shadow is not None
    cosine = float(np.dot(bright.signature, shadow.signature))
    assert cosine > 0.95


def test_different_colour_families_are_less_similar():
    red = _sample(
        "a", "a1",
        _crop(torso=(0, 0, 230), shorts=(0, 0, 180)),
    )
    white = _sample(
        "b", "b1",
        _crop(torso=(235, 235, 235), shorts=(220, 220, 220)),
    )
    assert red is not None and white is not None
    cosine = float(np.dot(red.signature, white.signature))
    assert cosine < 0.95


def test_exactly_two_primary_families_are_always_modelled():
    dim = 8
    a = np.eye(dim)[0]
    b = np.eye(dim)[1]
    profiles = [
        _profile(f"a{i:02d}", a + 0.01 * np.eye(dim)[2], samples=3)
        for i in range(30)
    ] + [
        _profile(f"b{i:02d}", b + 0.01 * np.eye(dim)[3], samples=3)
        for i in range(30)
    ]

    result = cluster_kit_families(
        profiles,
        KitFamilyConfig(
            min_primary_core_count=10,
            max_special_families=0,
        ),
    )
    primary = [
        row for row in result.clusters
        if row.family_type == "primary"
    ]
    assert len(primary) == 2
    assert result.special_family_count == 0


def test_minority_special_does_not_become_third_primary_team():
    dim = 8
    a = np.eye(dim)[0]
    b = np.eye(dim)[1]
    special = np.eye(dim)[2]

    profiles = []
    for i in range(35):
        profiles.append(_profile(f"a{i:02d}", a, samples=3))
        profiles.append(_profile(f"b{i:02d}", b, samples=3))
    for i in range(8):
        profiles.append(_profile(f"s{i:02d}", special, samples=3))

    result = cluster_kit_families(
        profiles,
        KitFamilyConfig(
            min_primary_core_count=10,
            min_special_profiles=4,
            max_special_families=3,
            max_special_primary_cosine=0.90,
        ),
    )

    primary_ids = {
        row.family_id
        for row in result.assignments
        if row.family_type == "primary"
    }
    assert primary_ids == {0, 1}
    assert result.special_family_count >= 1

    special_rows = [
        row for row in result.assignments
        if row.tracklet_id.startswith("s")
    ]
    assert any(row.family_type == "special" for row in special_rows)


def test_low_reliability_can_remain_unknown():
    dim = 8
    a = np.eye(dim)[0]
    b = np.eye(dim)[1]
    profiles = [
        _profile(f"a{i:02d}", a, samples=3)
        for i in range(25)
    ] + [
        _profile(f"b{i:02d}", b, samples=3)
        for i in range(25)
    ] + [
        _profile("weak", a, reliability=0.05, samples=1)
    ]

    result = cluster_kit_families(
        profiles,
        KitFamilyConfig(
            min_primary_core_count=10,
            max_special_families=0,
        ),
    )
    weak = next(
        row for row in result.assignments
        if row.tracklet_id == "weak"
    )
    assert weak.family_type == "unknown"
    assert weak.state == "low_reliability"


def test_family_difference_never_creates_identity_hard_constraint():
    dim = 8
    a = np.eye(dim)[0]
    b = np.eye(dim)[1]
    profiles = [
        _profile(f"a{i:02d}", a, samples=3)
        for i in range(25)
    ] + [
        _profile(f"b{i:02d}", b, samples=3)
        for i in range(25)
    ]

    result = cluster_kit_families(
        profiles,
        KitFamilyConfig(
            min_primary_core_count=10,
            max_special_families=0,
        ),
    )
    summary = result.summary()
    assert summary["identity_hard_constraints_created"] is False
    assert summary["football_structure"]["dominant_primary_families"] == 2
