from __future__ import annotations

import numpy as np

from tracking.appearance.kit import KitTrackletProfile
from tracking.appearance.kit_clustering_adaptive import (
    AdaptiveKitClusteringConfig,
    cluster_kit_profiles_adaptive,
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


def _group(prefix, centre, count, noise=0.012):
    centre = np.asarray(centre, dtype=np.float64)
    result = []
    for index in range(count):
        delta = np.asarray(
            [
                ((index * 3 + 1) % 5) - 2,
                ((index * 5 + 2) % 7) - 3,
                ((index * 7 + 3) % 5) - 2,
                ((index * 11 + 1) % 7) - 3,
            ],
            dtype=np.float64,
        )
        vector = centre + noise * delta
        result.append(
            _profile(f"{prefix}-{index:02d}", vector)
        )
    return result


def test_two_clean_groups_stay_two_groups():
    profiles = (
        _group("a", [1.0, 0.05, 0.0, 0.0], 30)
        + _group("b", [0.0, 0.05, 1.0, 0.0], 30)
    )
    result = cluster_kit_profiles_adaptive(
        profiles,
        AdaptiveKitClusteringConfig(
            min_k=2,
            max_k=5,
            min_anchor_count=20,
        ),
    )

    assert result.selected_k == 2
    assert result.silhouette_best_k == 2


def test_coherent_minority_group_can_survive_model_selection():
    profiles = (
        _group("a", [1.0, 0.0, 0.0, 0.0], 40)
        + _group("b", [0.0, 0.0, 1.0, 0.0], 40)
        + _group("minor", [0.0, 1.0, 0.0, 0.0], 8)
    )
    result = cluster_kit_profiles_adaptive(
        profiles,
        AdaptiveKitClusteringConfig(
            min_k=2,
            max_k=5,
            min_anchor_count=20,
        ),
    )

    assert result.selected_k >= 3
    minor_labels = {
        item.cluster_id
        for item in result.assignments
        if item.tracklet_id.startswith("minor-")
        and item.usable_for_identity
    }
    assert len(minor_labels) == 1


def test_no_semantic_role_or_team_is_invented():
    profiles = (
        _group("a", [1.0, 0.0, 0.0, 0.0], 20)
        + _group("b", [0.0, 0.0, 1.0, 0.0], 20)
        + _group("c", [0.0, 1.0, 0.0, 0.0], 8)
    )
    result = cluster_kit_profiles_adaptive(
        profiles,
        AdaptiveKitClusteringConfig(
            min_k=2,
            max_k=5,
            min_anchor_count=20,
        ),
    )

    payload = result.assignments[0].to_dict()
    assert payload["semantic_label"] is None
    assert payload["team_label"] is None
    assert payload["role_label"] is None


def test_low_reliability_assignment_is_not_used_for_identity():
    profiles = (
        _group("a", [1.0, 0.0, 0.0, 0.0], 20)
        + _group("b", [0.0, 0.0, 1.0, 0.0], 20)
    )
    profiles.append(
        _profile(
            "weak",
            [0.99, 0.01, 0.0, 0.0],
            reliability=0.05,
            sample_count=1,
        )
    )

    result = cluster_kit_profiles_adaptive(
        profiles,
        AdaptiveKitClusteringConfig(
            min_k=2,
            max_k=2,
            min_anchor_count=20,
        ),
    )
    weak = next(
        item for item in result.assignments
        if item.tracklet_id == "weak"
    )
    assert weak.assignment_state == "low_reliability"
    assert weak.usable_for_identity is False


def test_input_order_does_not_change_cluster_assignment():
    profiles = (
        _group("a", [1.0, 0.0, 0.0, 0.0], 20)
        + _group("b", [0.0, 0.0, 1.0, 0.0], 20)
        + _group("c", [0.0, 1.0, 0.0, 0.0], 8)
    )
    config = AdaptiveKitClusteringConfig(
        min_k=2,
        max_k=5,
        min_anchor_count=20,
    )
    first = cluster_kit_profiles_adaptive(profiles, config)
    second = cluster_kit_profiles_adaptive(
        list(reversed(profiles)),
        config,
    )

    assert first.selected_k == second.selected_k
    first_map = {
        item.tracklet_id: (
            item.cluster_id,
            item.assignment_state,
        )
        for item in first.assignments
    }
    second_map = {
        item.tracklet_id: (
            item.cluster_id,
            item.assignment_state,
        )
        for item in second.assignments
    }
    assert first_map == second_map


def test_assignments_create_no_hard_identity_decision():
    profiles = (
        _group("a", [1.0, 0.0, 0.0, 0.0], 20)
        + _group("b", [0.0, 0.0, 1.0, 0.0], 20)
    )
    result = cluster_kit_profiles_adaptive(
        profiles,
        AdaptiveKitClusteringConfig(
            min_k=2,
            max_k=2,
            min_anchor_count=20,
        ),
    )

    summary = result.summary()
    assert summary["identity_hard_constraints_created"] is False
    assert summary["clip_adaptive"] is True
