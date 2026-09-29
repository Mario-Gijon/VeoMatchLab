from __future__ import annotations

import numpy as np

from tracking_v0.appearance_bank import (
    AppearanceBankConfig,
    ReIDEmbedding,
    build_appearance_banks,
    compare_track_banks,
    within_track_coherence,
)


def e(track_id, frame, vector, quality=0.9):
    return ReIDEmbedding(
        track_id=track_id,
        frame_index=frame,
        quality=quality,
        vector=np.asarray(
            vector,
            dtype=np.float32,
        ),
    )


def test_build_bank_filters_low_quality_and_limits_count():
    rows = [
        e(1, 1, [1, 0], 0.95),
        e(1, 2, [1, 0], 0.80),
        e(1, 3, [1, 0], 0.20),
        e(1, 4, [1, 0], 0.05),
    ]

    banks = build_appearance_banks(
        rows,
        config=AppearanceBankConfig(
            min_quality=0.10,
            max_embeddings_per_track=2,
        ),
    )

    assert banks[1].count == 2
    assert [
        item.frame_index
        for item in banks[1].embeddings
    ] == [1, 2]


def test_same_person_like_banks_score_higher_than_different():
    rows = [
        e(1, 0, [1.0, 0.0, 0.0]),
        e(1, 1, [0.98, 0.05, 0.0]),
        e(2, 0, [0.99, 0.03, 0.0]),
        e(2, 1, [0.97, 0.06, 0.0]),
        e(3, 0, [0.0, 1.0, 0.0]),
        e(3, 1, [0.05, 0.98, 0.0]),
    ]

    banks = build_appearance_banks(rows)

    same = compare_track_banks(
        banks[1],
        banks[2],
    )
    different = compare_track_banks(
        banks[1],
        banks[3],
    )

    assert (
        same.bidirectional_best_median
        > different.bidirectional_best_median
    )
    assert (
        same.unique_topk_similarity
        > different.unique_topk_similarity
    )


def test_within_track_coherence_detects_mixed_bank():
    rows = [
        e(1, 0, [1.0, 0.0]),
        e(1, 1, [0.99, 0.05]),
        e(1, 2, [0.98, 0.04]),
        e(2, 0, [1.0, 0.0]),
        e(2, 1, [0.0, 1.0]),
        e(2, 2, [0.7, 0.7]),
    ]

    banks = build_appearance_banks(rows)

    clean = within_track_coherence(
        banks[1]
    )
    mixed = within_track_coherence(
        banks[2]
    )

    assert (
        clean.median_similarity
        > mixed.median_similarity
    )
    assert (
        clean.q10_similarity
        > mixed.q10_similarity
    )


def test_unique_topk_does_not_reuse_one_crop_forever():
    rows = [
        e(1, 0, [1.0, 0.0]),
        e(1, 1, [0.0, 1.0]),
        e(2, 0, [1.0, 0.0]),
        e(2, 1, [1.0, 0.0]),
    ]

    banks = build_appearance_banks(rows)

    pair = compare_track_banks(
        banks[1],
        banks[2],
        config=AppearanceBankConfig(
            unique_top_k=2,
        ),
    )

    # One perfect pair + one orthogonal pair => unique matching cannot simply
    # reuse the first crop twice.
    assert pair.best_similarity > 0.99
    assert pair.unique_topk_similarity < 0.75
