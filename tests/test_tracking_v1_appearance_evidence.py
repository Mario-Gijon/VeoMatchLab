from __future__ import annotations

import math

import numpy as np
import pytest

from tracking.appearance.evidence import (
    AppearanceEvidenceConfig,
    AppearanceEvidenceSample,
    build_pairwise_evidence,
    build_tracklet_profile,
    build_tracklet_profiles,
    compare_tracklet_profiles,
    sample_from_embedding_row,
)


CFG = AppearanceEvidenceConfig(model_id="test-reid")


def _sample(
    tracklet_id: str,
    sample_id: str,
    vector,
    *,
    quality: float = 0.9,
    t: float = 0.0,
):
    return AppearanceEvidenceSample(
        sample_id=sample_id,
        tracklet_id=tracklet_id,
        detection_id=f"det-{sample_id}",
        frame_index=int(round(t * 10)),
        timestamp_s=t,
        quality=quality,
        embedding=tuple(vector),
    )


def test_profile_preserves_samples_and_builds_unit_aggregate():
    profile = build_tracklet_profile(
        [
            _sample("trk-a", "a1", [1.0, 0.0, 0.0], t=0.0),
            _sample("trk-a", "a2", [0.98, 0.05, 0.0], t=1.0),
            _sample("trk-a", "a3", [0.95, 0.08, 0.0], t=2.0),
        ],
        CFG,
    )

    assert profile.tracklet_id == "trk-a"
    assert profile.sample_count == 3
    assert len(profile.samples) == 3
    assert np.linalg.norm(profile.aggregate_embedding) == pytest.approx(1.0)
    assert sum(sample.aggregation_weight for sample in profile.samples) == pytest.approx(1.0)
    assert 1.0 <= profile.effective_sample_count <= 3.0
    assert 0.0 <= profile.reliability <= 1.0


def test_robust_aggregation_smoothly_downweights_visual_outlier():
    profile = build_tracklet_profile(
        [
            _sample("trk-a", "a1", [1.0, 0.0], quality=0.9, t=0),
            _sample("trk-a", "a2", [0.99, 0.05], quality=0.9, t=1),
            _sample("trk-a", "a3", [0.98, -0.05], quality=0.9, t=2),
            _sample("trk-a", "outlier", [-1.0, 0.0], quality=0.9, t=3),
        ],
        CFG,
    )
    weights = {sample.sample_id: sample.aggregation_weight for sample in profile.samples}

    assert weights["outlier"] < weights["a1"]
    assert weights["outlier"] > 0.0  # no hard rejection
    assert profile.aggregate_embedding[0] > 0.9


def test_pair_evidence_is_symmetric_and_contains_no_identity_decision():
    a = build_tracklet_profile(
        [_sample("trk-a", "a1", [1, 0]), _sample("trk-a", "a2", [0.98, 0.1], t=1)],
        CFG,
    )
    b = build_tracklet_profile(
        [_sample("trk-b", "b1", [0.95, 0.2]), _sample("trk-b", "b2", [0.9, 0.3], t=1)],
        CFG,
    )

    ab = compare_tracklet_profiles(a, b)
    ba = compare_tracklet_profiles(b, a)

    assert ab.to_dict() == ba.to_dict()
    assert ab.cross_similarity.count == 4
    assert ab.bidirectional_best_similarity.count == 4
    assert "same_identity" not in ab.to_dict()
    assert "merge" not in ab.to_dict()


def test_build_pairwise_evidence_emits_n_choose_2_edges():
    profiles = build_tracklet_profiles(
        [
            _sample("trk-a", "a", [1, 0]),
            _sample("trk-b", "b", [0, 1]),
            _sample("trk-c", "c", [0.7, 0.7]),
        ],
        CFG,
    )
    pairs = build_pairwise_evidence(profiles)
    assert len(pairs) == 3
    assert [(p.tracklet_a, p.tracklet_b) for p in pairs] == [
        ("trk-a", "trk-b"),
        ("trk-a", "trk-c"),
        ("trk-b", "trk-c"),
    ]


def test_embedding_row_selects_exact_descriptor_and_normalizes():
    row = {
        "sample_id": "sample-1",
        "tracklet_id": "trk-a",
        "detection_id": "det-1",
        "frame_index": 12,
        "timestamp_s": 1.2,
        "quality": 0.88,
        "crop_ref": "crop.jpg",
        "view_name": "tight_far_c1",
        "descriptors": [
            {
                "name": "reid",
                "model_id": "test-reid",
                "values": [3.0, 4.0],
                "metric": "cosine",
            }
        ],
    }
    sample = sample_from_embedding_row(row, CFG)
    assert sample.embedding == pytest.approx((0.6, 0.8))


def test_profile_rejects_mixed_dimensions():
    with pytest.raises(ValueError, match="dimension"):
        build_tracklet_profile(
            [
                _sample("trk-a", "a1", [1, 0]),
                _sample("trk-a", "a2", [1, 0, 0]),
            ],
            CFG,
        )
