from __future__ import annotations

import pytest

from tracking.actors.reid_retrieval import (
    SparseReIDRetrievalConfig,
    build_sparse_reid_retrieval,
)
from tracking.appearance.evidence import (
    DistributionStats,
    TrackletAppearanceProfile,
)


def _profile(
    tracklet_id: str,
    embedding: tuple[float, ...],
) -> TrackletAppearanceProfile:
    stats = DistributionStats(
        count=1,
        min=1.0,
        p10=1.0,
        median=1.0,
        p90=1.0,
        p95=1.0,
        max=1.0,
        mean=1.0,
        std=0.0,
    )
    return TrackletAppearanceProfile(
        tracklet_id=tracklet_id,
        model_id="prtreid-soccernet-baseline-global-v1",
        descriptor_name="reid",
        embedding_dim=len(embedding),
        aggregate_embedding=embedding,
        medoid_sample_id=f"{tracklet_id}-sample",
        samples=(),
        quality_stats=stats,
        centroid_similarity=stats,
        internal_pair_similarity=DistributionStats(count=0),
        effective_sample_count=1.0,
        dispersion=0.0,
        reliability=1.0,
        first_timestamp_s=0.0,
        last_timestamp_s=0.0,
    )


def _pairs(result):
    return {
        (item.tracklet_a, item.tracklet_b)
        for item in result.candidates
    }


def test_top_k_excludes_self():
    profiles = [
        _profile("a", (1.0, 0.0)),
        _profile("b", (0.9, 0.1)),
        _profile("c", (0.0, 1.0)),
    ]

    result = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=1,
            cosine_block_size=2,
        ),
    )

    assert all(
        item.query_tracklet_id
        != item.candidate_tracklet_id
        for item in result.neighbors
    )


def test_hard_conflict_overrides_high_reid_similarity():
    profiles = [
        _profile("a", (1.0, 0.0)),
        _profile("b", (1.0, 0.0)),
        _profile("c", (0.0, 1.0)),
    ]

    result = build_sparse_reid_retrieval(
        profiles,
        hard_conflicts=[
            {
                "tracklet_a": "a",
                "tracklet_b": "b",
            }
        ],
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=1,
        ),
    )

    assert ("a", "b") not in _pairs(result)


def test_union_is_canonical_and_has_no_duplicates():
    profiles = [
        _profile("z", (1.0, 0.0)),
        _profile("a", (0.99, 0.01)),
        _profile("m", (0.0, 1.0)),
    ]

    result = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=2,
        ),
    )

    pairs = [
        (item.tracklet_a, item.tracklet_b)
        for item in result.candidates
    ]
    assert len(pairs) == len(set(pairs))
    assert all(a < b for a, b in pairs)


def test_retrieval_uses_no_cosine_threshold():
    profiles = [
        _profile("a", (1.0, 0.0)),
        _profile("b", (-1.0, 0.0)),
    ]

    result = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=1,
        ),
    )

    assert _pairs(result) == {("a", "b")}
    assert result.candidates[0].aggregate_cosine == pytest.approx(-1.0)


def test_mutual_topk_is_recorded():
    profiles = [
        _profile("a", (1.0, 0.0)),
        _profile("b", (0.99, 0.01)),
        _profile("c", (0.0, 1.0)),
    ]

    result = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=1,
        ),
    )

    pair = next(
        item
        for item in result.candidates
        if (item.tracklet_a, item.tracklet_b) == ("a", "b")
    )
    assert pair.mutual is True
    assert pair.rank_a_to_b == 1
    assert pair.rank_b_to_a == 1


def test_block_size_does_not_change_result():
    profiles = [
        _profile("a", (1.0, 0.0, 0.0)),
        _profile("b", (0.9, 0.1, 0.0)),
        _profile("c", (0.8, 0.2, 0.0)),
        _profile("d", (0.0, 0.0, 1.0)),
        _profile("e", (0.0, 0.1, 0.9)),
    ]

    small = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=2,
            cosine_block_size=1,
        ),
    )
    large = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=2,
            cosine_block_size=100,
        ),
    )

    assert [
        item.to_dict() for item in small.neighbors
    ] == [
        item.to_dict() for item in large.neighbors
    ]
    assert [
        item.to_dict() for item in small.candidates
    ] == [
        item.to_dict() for item in large.candidates
    ]


def test_top_k_output_is_sparse_not_all_pairs():
    profiles = [
        _profile(
            f"trk-{index:03d}",
            (1.0, float(index + 1)),
        )
        for index in range(100)
    ]

    result = build_sparse_reid_retrieval(
        profiles,
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=3,
            cosine_block_size=17,
        ),
    )

    all_pairs = len(profiles) * (len(profiles) - 1) // 2
    assert len(result.neighbors) <= len(profiles) * 3
    assert len(result.candidates) < all_pairs // 5


def test_missing_profiles_are_not_required():
    # The retrieval stage is appearance-only: it must be valid to receive only
    # the subset of tracklets that have usable ReID.
    result = build_sparse_reid_retrieval(
        [
            _profile("a", (1.0, 0.0)),
            _profile("c", (0.9, 0.1)),
        ],
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=1,
        ),
    )

    assert _pairs(result) == {("a", "c")}
