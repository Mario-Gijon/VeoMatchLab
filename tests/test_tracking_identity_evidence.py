from __future__ import annotations

import numpy as np

from tracking_v0.appearance_bank import (
    AppearanceBankConfig,
    ReIDEmbedding,
    TrackAppearanceBank,
)
from tracking_v0.appearance_segments import (
    segment_appearance_bank,
)
from tracking_v0.identity_evidence import (
    IdentityEvidenceLabel,
    ObservedTrackPoint,
    build_identity_pair_evidence,
    compute_geometry_pair_evidence_from_points,
)
from tracking_v0.team_appearance import AppearanceLabel
from tracking_v0.team_similarity import RelativeTeamLabel


def emb(track, frame, vector):
    return ReIDEmbedding(
        track_id=track,
        frame_index=frame,
        quality=0.9,
        vector=np.asarray(vector, dtype=np.float32),
    )


def segmentation(track, rows):
    return segment_appearance_bank(
        TrackAppearanceBank(
            track_id=track,
            embeddings=tuple(rows),
        ),
        source_label=AppearanceLabel.UNKNOWN,
    )


def points(track, start, count, x0, z0, dx=0.1):
    return tuple(
        ObservedTrackPoint(
            track_id=track,
            frame_index=start + index,
            timestamp=(start + index) / 15.0,
            x=x0 + dx * index,
            z=z0,
        )
        for index in range(count)
    )


def test_team_conflict_is_hard_conflict_even_with_visual_similarity():
    segs = {
        1: segmentation(
            1,
            [
                emb(1, 0, [1.0, 0.0]),
                emb(1, 10, [0.99, 0.02]),
            ],
        ),
        2: segmentation(
            2,
            [
                emb(2, 20, [1.0, 0.0]),
                emb(2, 30, [0.99, 0.02]),
            ],
        ),
    }

    observed = {
        1: points(1, 0, 20, 0.0, 0.0),
        2: points(2, 20, 20, 2.0, 0.0),
    }

    result = build_identity_pair_evidence(
        1,
        2,
        segmentations=segs,
        observed_points=observed,
        team_labels={
            1: RelativeTeamLabel.TEAM_A,
            2: RelativeTeamLabel.TEAM_B,
        },
    )

    assert (
        result.label
        is IdentityEvidenceLabel.HARD_CONFLICT
    )


def test_sustained_simultaneous_tracks_are_strong_conflict():
    segs = {
        1: segmentation(
            1,
            [
                emb(1, 0, [1.0, 0.0]),
                emb(1, 10, [0.98, 0.03]),
            ],
        ),
        2: segmentation(
            2,
            [
                emb(2, 0, [0.98, 0.03]),
                emb(2, 10, [1.0, 0.0]),
            ],
        ),
    }

    observed = {
        1: points(1, 0, 30, 0.0, 0.0),
        2: points(2, 0, 30, 3.0, 0.0),
    }

    result = build_identity_pair_evidence(
        1,
        2,
        segmentations=segs,
        observed_points=observed,
        team_labels={
            1: RelativeTeamLabel.TEAM_A,
            2: RelativeTeamLabel.TEAM_A,
        },
    )

    assert result.geometry.simultaneous_conflict
    assert (
        result.label
        is IdentityEvidenceLabel.STRONG_CONFLICT
    )


def test_clean_handover_with_visual_match_has_support():
    segs = {
        1: segmentation(
            1,
            [
                emb(1, 0, [1.0, 0.0, 0.0]),
                emb(1, 10, [0.99, 0.03, 0.0]),
                emb(1, 20, [0.98, 0.04, 0.0]),
            ],
        ),
        2: segmentation(
            2,
            [
                emb(2, 30, [0.99, 0.02, 0.0]),
                emb(2, 40, [1.0, 0.01, 0.0]),
                emb(2, 50, [0.98, 0.04, 0.0]),
            ],
        ),
    }

    observed = {
        1: points(1, 0, 25, 0.0, 0.0),
        2: points(2, 25, 25, 2.5, 0.0),
    }

    result = build_identity_pair_evidence(
        1,
        2,
        segmentations=segs,
        observed_points=observed,
        team_labels={
            1: RelativeTeamLabel.TEAM_A,
            2: RelativeTeamLabel.TEAM_A,
        },
    )

    assert result.reid_score > 0.95
    assert result.geometry_score > 0.8
    assert result.fused_support_score > 0.8
    assert result.label in {
        IdentityEvidenceLabel.STRONG_SUPPORT,
        IdentityEvidenceLabel.MODERATE_SUPPORT,
    }


def test_segment_aware_matching_can_recover_one_matching_mode():
    segs = {
        1: segmentation(
            1,
            [
                emb(1, 0, [1.0, 0.0]),
                emb(1, 10, [1.0, 0.0]),
                emb(1, 20, [0.0, 1.0]),
                emb(1, 30, [0.0, 1.0]),
            ],
        ),
        2: segmentation(
            2,
            [
                emb(2, 40, [0.0, 1.0]),
                emb(2, 50, [0.02, 0.99]),
            ],
        ),
    }

    observed = {
        1: points(1, 0, 35, 0.0, 0.0),
        2: points(2, 40, 20, 3.5, 0.0),
    }

    result = build_identity_pair_evidence(
        1,
        2,
        segmentations=segs,
        observed_points=observed,
        team_labels={
            1: RelativeTeamLabel.UNKNOWN,
            2: RelativeTeamLabel.UNKNOWN,
        },
    )

    assert result.reid_score > 0.95
    assert (
        result.best_segment_pair.segment_a
        == "1:1"
    )

def test_explicit_geometry_sequences_do_not_alias_same_raw_track_id():
    left = tuple(
        ObservedTrackPoint(
            track_id=13,
            frame_index=frame,
            timestamp=frame / 15.0,
            x=0.1 * frame,
            z=0.0,
        )
        for frame in range(0, 20)
    )
    right = tuple(
        ObservedTrackPoint(
            track_id=13,
            frame_index=frame,
            timestamp=frame / 15.0,
            x=0.1 * frame,
            z=0.0,
        )
        for frame in range(25, 45)
    )

    result = compute_geometry_pair_evidence_from_points(
        13,
        13,
        rows_a=left,
        rows_b=right,
    )

    assert result.simultaneous_count == 0
    assert result.simultaneous_fraction == 0.0
    assert result.support_gap_s > 0.0
    assert not result.simultaneous_conflict
    assert result.transition_geometry_score < 1.0

