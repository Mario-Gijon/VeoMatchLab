from __future__ import annotations

import numpy as np

from tracking_v0.appearance_bank import (
    ReIDEmbedding,
    TrackAppearanceBank,
)
from tracking_v0.appearance_segments import (
    AppearanceSegmentationConfig,
    segment_appearance_bank,
)
from tracking_v0.team_appearance import AppearanceLabel


def e(frame, vector, quality=0.9):
    return ReIDEmbedding(
        track_id=1,
        frame_index=frame,
        quality=quality,
        vector=np.asarray(vector, dtype=np.float32),
    )


def bank(rows):
    return TrackAppearanceBank(
        track_id=1,
        embeddings=tuple(rows),
    )


def test_clean_track_is_not_split():
    rows = [
        e(0, [1.0, 0.02, 0.0]),
        e(10, [0.99, 0.04, 0.0]),
        e(20, [0.98, 0.06, 0.0]),
        e(30, [1.0, 0.01, 0.0]),
        e(40, [0.99, 0.03, 0.0]),
        e(50, [0.98, 0.05, 0.0]),
    ]

    result = segment_appearance_bank(
        bank(rows)
    )

    assert len(result.segments) == 1
    assert not result.segmented


def test_clear_temporal_identity_switch_is_split():
    rows = [
        e(0, [1.0, 0.0, 0.0]),
        e(10, [0.99, 0.03, 0.0]),
        e(20, [0.98, 0.04, 0.0]),
        e(30, [0.0, 1.0, 0.0]),
        e(40, [0.03, 0.99, 0.0]),
        e(50, [0.04, 0.98, 0.0]),
    ]

    result = segment_appearance_bank(
        bank(rows)
    )

    assert len(result.segments) == 2
    assert result.segments[0].end_frame == 20
    assert result.segments[1].start_frame == 30
    assert result.accepted_splits[0].split_gain > 0.5


def test_mixed_label_can_accept_moderately_weaker_split():
    rows = [
        e(0, [1.0, 0.0]),
        e(10, [0.98, 0.20]),
        e(20, [0.96, 0.28]),
        e(30, [0.55, 0.84]),
        e(40, [0.48, 0.88]),
        e(50, [0.42, 0.91]),
    ]

    strict = AppearanceSegmentationConfig(
        min_split_gain=0.24,
        mixed_split_gain_scale=0.70,
        max_cross_similarity=0.82,
        min_child_coherence=0.90,
    )

    normal = segment_appearance_bank(
        bank(rows),
        source_label=AppearanceLabel.TEAM_A,
        config=strict,
    )
    mixed = segment_appearance_bank(
        bank(rows),
        source_label=AppearanceLabel.MIXED,
        config=strict,
    )

    assert len(mixed.segments) >= len(normal.segments)


def test_segment_prototypes_are_normalized():
    rows = [
        e(0, [1.0, 0.0]),
        e(10, [0.9, 0.1]),
        e(20, [0.0, 1.0]),
        e(30, [0.1, 0.9]),
    ]

    result = segment_appearance_bank(
        bank(rows),
        config=AppearanceSegmentationConfig(
            min_split_gain=0.05,
            max_cross_similarity=0.9,
            min_child_coherence=0.8,
        ),
    )

    for segment in result.segments:
        assert np.isclose(
            np.linalg.norm(segment.prototype),
            1.0,
        )
