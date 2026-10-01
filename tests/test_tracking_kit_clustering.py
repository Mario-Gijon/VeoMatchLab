from __future__ import annotations

import numpy as np

from tracking.appearance.kit import (
    KitTrackletProfile,
)
from tracking.appearance.kit_clustering import (
    KitClusteringConfig,
    cluster_kit_profiles,
)


def _profile(
    tracklet_id: str,
    vector,
    *,
    reliability=0.8,
    sample_count=3,
):
    vector = np.asarray(vector, dtype=np.float64)
    vector = vector / np.linalg.norm(vector)
    return KitTrackletProfile(
        tracklet_id=tracklet_id,
        aggregate_signature=tuple(float(v) for v in vector),
        sample_count=sample_count,
        effective_sample_count=float(sample_count),
        reliability=reliability,
        mean_quality=0.9,
        internal_similarity={
            "count": sample_count,
            "min": 0.95,
            "p10": 0.96,
            "median": 0.98,
            "p90": 0.99,
            "p95": 0.995,
            "max": 1.0,
            "mean": 0.98,
            "std": 0.01,
        },
        first_timestamp_s=0.0,
        last_timestamp_s=1.0,
    )


def test_two_clean_groups_are_separated():
    profiles = []
    for index in range(20):
        profiles.append(
            _profile(
                f"a-{index:02d}",
                [1.0, 0.01 * (index % 3), 0.0],
            )
        )
        profiles.append(
            _profile(
                f"b-{index:02d}",
                [0.0, 0.01 * (index % 3), 1.0],
            )
        )

    result = cluster_kit_profiles(
        profiles,
        KitClusteringConfig(
            min_k=2,
            max_k=2,
            min_anchor_count=10,
        ),
    )

    labels_a = {
        item.cluster_id
        for item in result.assignments
        if item.tracklet_id.startswith("a-")
    }
    labels_b = {
        item.cluster_id
        for item in result.assignments
        if item.tracklet_id.startswith("b-")
    }

    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert labels_a != labels_b


def test_singletons_are_assigned_but_do_not_need_to_be_anchors():
    profiles = [
        _profile(f"a-{i}", [1.0, 0.02 * i, 0.0])
        for i in range(10)
    ] + [
        _profile(f"b-{i}", [0.0, 0.02 * i, 1.0])
        for i in range(10)
    ] + [
        _profile(
            "weak-singleton",
            [0.98, 0.05, 0.0],
            reliability=0.2,
            sample_count=1,
        )
    ]

    result = cluster_kit_profiles(
        profiles,
        KitClusteringConfig(
            min_k=2,
            max_k=2,
            min_anchor_count=10,
        ),
    )
    weak = next(
        item
        for item in result.assignments
        if item.tracklet_id == "weak-singleton"
    )

    assert weak.anchor_used_for_fit is False
    assert 0.0 <= weak.best_cosine <= 1.0
    assert weak.margin >= 0.0


def test_cluster_ids_are_deterministic_under_input_order():
    profiles = [
        _profile(f"a-{i}", [1.0, 0.01 * i, 0.0])
        for i in range(12)
    ] + [
        _profile(f"b-{i}", [0.0, 0.01 * i, 1.0])
        for i in range(12)
    ]

    config = KitClusteringConfig(
        min_k=2,
        max_k=2,
        min_anchor_count=10,
    )
    first = cluster_kit_profiles(profiles, config)
    second = cluster_kit_profiles(list(reversed(profiles)), config)

    first_map = {
        item.tracklet_id: item.cluster_id
        for item in first.assignments
    }
    second_map = {
        item.tracklet_id: item.cluster_id
        for item in second.assignments
    }

    assert first_map == second_map


def test_model_scores_cover_requested_k_range():
    profiles = []
    directions = (
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    )
    for group, direction in enumerate(directions):
        for index in range(15):
            vector = np.asarray(direction, dtype=np.float64)
            vector = vector + 0.01 * np.asarray(
                [index % 2, (index + group) % 2, index % 3],
                dtype=np.float64,
            )
            profiles.append(
                _profile(
                    f"g{group}-{index:02d}",
                    vector,
                )
            )

    result = cluster_kit_profiles(
        profiles,
        KitClusteringConfig(
            min_k=2,
            max_k=4,
            min_anchor_count=12,
        ),
    )

    assert [item.k for item in result.model_scores] == [2, 3, 4]
    assert 2 <= result.suggested_k <= 4


def test_assignments_preserve_no_semantic_labels():
    profiles = [
        _profile(f"a-{i}", [1.0, 0.01 * i, 0.0])
        for i in range(10)
    ] + [
        _profile(f"b-{i}", [0.0, 0.01 * i, 1.0])
        for i in range(10)
    ]

    result = cluster_kit_profiles(
        profiles,
        KitClusteringConfig(
            min_k=2,
            max_k=2,
            min_anchor_count=10,
        ),
    )

    payload = result.assignments[0].to_dict()
    assert payload["team_label"] is None
    assert payload["role_label"] is None
    assert payload["semantic_label"] is None
