from __future__ import annotations

import numpy as np

from tracking_v0.identity_segmentation import (
    IdentityAppearanceSample,
    IdentitySegmentationConfig,
    SegmentIdentityQuality,
    segment_track_identity,
)
from tracking_v0.team_appearance import AppearanceLabel


def unit(x, y):
    value = np.asarray([x, y], dtype=np.float32)
    return value / np.linalg.norm(value)


def sample(frame, vector, appearance, *, raw_id=27, kit=None):
    return IdentityAppearanceSample(
        raw_track_id=raw_id,
        frame_index=frame,
        quality=0.9,
        reid_vector=np.asarray(vector, dtype=np.float32),
        appearance_label=appearance,
        kit_vector=(
            np.asarray(kit, dtype=np.float32)
            if kit is not None
            else None
        ),
    )


def test_stable_raw_track_is_not_split():
    samples = [
        sample(10, unit(1.0, 0.05), AppearanceLabel.TEAM_A),
        sample(20, unit(1.0, 0.02), AppearanceLabel.TEAM_A),
        sample(30, unit(0.99, 0.06), AppearanceLabel.TEAM_A),
        sample(40, unit(1.0, 0.01), AppearanceLabel.TEAM_A),
        sample(50, unit(0.98, 0.08), AppearanceLabel.TEAM_A),
        sample(60, unit(1.0, 0.04), AppearanceLabel.TEAM_A),
    ]

    result = segment_track_identity(samples)

    assert not result.segmented
    assert result.boundaries_after_frame == ()
    assert len(result.segments) == 1


def test_single_wrong_kit_crop_does_not_split_track():
    samples = [
        sample(10, unit(1.0, 0.03), AppearanceLabel.TEAM_A),
        sample(20, unit(1.0, 0.04), AppearanceLabel.TEAM_A),
        sample(30, unit(0.99, 0.05), AppearanceLabel.TEAM_B),  # contamination
        sample(40, unit(1.0, 0.02), AppearanceLabel.TEAM_A),
        sample(50, unit(0.99, 0.06), AppearanceLabel.TEAM_A),
        sample(60, unit(1.0, 0.01), AppearanceLabel.TEAM_A),
    ]

    result = segment_track_identity(samples)

    assert not result.segmented


def test_persistent_opposite_kit_and_reid_change_is_split():
    samples = [
        sample(10, unit(1.0, 0.05), AppearanceLabel.TEAM_A),
        sample(20, unit(0.99, 0.08), AppearanceLabel.TEAM_A),
        sample(30, unit(1.0, 0.02), AppearanceLabel.TEAM_A),
        sample(40, unit(0.05, 1.0), AppearanceLabel.TEAM_B),
        sample(50, unit(0.08, 0.99), AppearanceLabel.TEAM_B),
        sample(60, unit(0.02, 1.0), AppearanceLabel.TEAM_B),
    ]

    result = segment_track_identity(samples)

    assert result.segmented
    assert result.boundaries_after_frame == (30,)
    assert len(result.segments) == 2
    assert result.segments[0].appearance_label is AppearanceLabel.TEAM_A
    assert result.segments[1].appearance_label is AppearanceLabel.TEAM_B
    assert result.accepted_splits[0].persistent_appearance_conflict


def test_strong_same_team_reid_switch_can_split_without_team_change():
    samples = [
        sample(10, unit(1.0, 0.02), AppearanceLabel.TEAM_A),
        sample(20, unit(0.99, 0.04), AppearanceLabel.TEAM_A),
        sample(30, unit(1.0, 0.06), AppearanceLabel.TEAM_A),
        sample(40, unit(0.02, 1.0), AppearanceLabel.TEAM_A),
        sample(50, unit(0.04, 0.99), AppearanceLabel.TEAM_A),
        sample(60, unit(0.06, 1.0), AppearanceLabel.TEAM_A),
    ]

    result = segment_track_identity(
        samples,
        config=IdentitySegmentationConfig(
            min_strong_reid_gain=0.20,
            strong_max_cross_similarity=0.40,
            strong_min_child_coherence=0.90,
        ),
    )

    assert result.segmented
    assert result.boundaries_after_frame == (30,)
    assert result.accepted_splits[0].reason == "strong_reid_discontinuity"


def test_persistent_special_to_team_switch_splits_raw_track():
    samples = [
        sample(10, unit(1.0, 0.05), AppearanceLabel.SPECIAL),
        sample(20, unit(0.98, 0.08), AppearanceLabel.SPECIAL),
        sample(30, unit(0.95, 0.12), AppearanceLabel.SPECIAL),
        sample(40, unit(0.20, 0.98), AppearanceLabel.TEAM_B),
        sample(50, unit(0.15, 0.99), AppearanceLabel.TEAM_B),
        sample(60, unit(0.10, 1.0), AppearanceLabel.TEAM_B),
    ]

    result = segment_track_identity(samples)

    assert result.segmented
    assert result.boundaries_after_frame == (30,)
    assert result.segments[0].appearance_label is AppearanceLabel.SPECIAL
    assert result.segments[1].appearance_label is AppearanceLabel.TEAM_B


def test_persistent_kit_descriptor_shift_splits_even_when_semantic_labels_do_not():
    """Regression for the raw35-type failure.

    Global kit clustering can label both sides as the same team even though the
    actual crop appearance changes persistently. The raw kit descriptor must
    still be able to trigger a split, with ReID acting only as support.
    """
    reid_left = [
        unit(1.00, 0.04),
        unit(0.99, 0.08),
        unit(1.00, 0.02),
        unit(0.98, 0.10),
    ]
    reid_right = [
        unit(0.74, 0.67),
        unit(0.72, 0.69),
        unit(0.76, 0.65),
        unit(0.70, 0.71),
    ]
    kit_left = [
        unit(1.00, 0.03),
        unit(0.99, 0.05),
        unit(1.00, 0.01),
        unit(0.98, 0.08),
    ]
    kit_right = [
        unit(0.05, 1.00),
        unit(0.02, 1.00),
        unit(0.08, 0.99),
        unit(0.04, 1.00),
    ]

    samples = []
    for index, frame in enumerate((100, 200, 300, 400)):
        samples.append(
            sample(
                frame,
                reid_left[index],
                AppearanceLabel.TEAM_A,
                raw_id=35,
                kit=kit_left[index],
            )
        )
    for index, frame in enumerate((500, 600, 700, 800)):
        samples.append(
            sample(
                frame,
                reid_right[index],
                AppearanceLabel.TEAM_A,
                raw_id=35,
                kit=kit_right[index],
            )
        )

    result = segment_track_identity(samples)

    assert result.segmented
    assert result.boundaries_after_frame == (400,)
    split = result.accepted_splits[0]
    assert split.persistent_kit_shift
    assert split.reason == "persistent_kit_shift+reid"
    assert split.cross_kit_similarity is not None
    assert split.cross_kit_similarity < 0.20


def test_single_kit_descriptor_outlier_does_not_split():
    stable_kit = unit(1.0, 0.05)
    outlier_kit = unit(0.05, 1.0)

    samples = [
        sample(10, unit(1.0, 0.03), AppearanceLabel.TEAM_A, kit=stable_kit),
        sample(20, unit(1.0, 0.04), AppearanceLabel.TEAM_A, kit=stable_kit),
        sample(30, unit(0.99, 0.05), AppearanceLabel.TEAM_A, kit=outlier_kit),
        sample(40, unit(1.0, 0.02), AppearanceLabel.TEAM_A, kit=stable_kit),
        sample(50, unit(0.99, 0.06), AppearanceLabel.TEAM_A, kit=stable_kit),
        sample(60, unit(1.0, 0.01), AppearanceLabel.TEAM_A, kit=stable_kit),
        sample(70, unit(0.99, 0.04), AppearanceLabel.TEAM_A, kit=stable_kit),
        sample(80, unit(1.0, 0.03), AppearanceLabel.TEAM_A, kit=stable_kit),
    ]

    result = segment_track_identity(samples)

    assert not result.segmented


def test_stable_track_with_pose_variation_in_kit_descriptor_is_not_split():
    kits = [
        unit(1.0, 0.12),
        unit(0.99, 0.16),
        unit(1.0, 0.08),
        unit(0.98, 0.18),
        unit(0.99, 0.14),
        unit(1.0, 0.10),
        unit(0.98, 0.17),
        unit(1.0, 0.11),
    ]
    samples = [
        sample(
            10 * (index + 1),
            unit(1.0, 0.03 + 0.005 * index),
            AppearanceLabel.TEAM_A,
            raw_id=10,
            kit=kit,
        )
        for index, kit in enumerate(kits)
    ]

    result = segment_track_identity(samples)

    assert not result.segmented


def angle_vector(degrees: float) -> np.ndarray:
    radians = np.deg2rad(degrees)
    return np.asarray(
        [np.cos(radians), np.sin(radians)],
        dtype=np.float32,
    )


def test_v3_raw35_like_noisy_prefix_then_persistent_state_is_split():
    """Regression for the real raw35 pattern.

    Semantic labels before the boundary are contaminated/mixed:
        B A A B | A A A A A A
    so neither side is a clean opposite-team block.  The new state after the
    boundary is persistent, while kit + ReID both support a real visual change.
    """

    frames = (500, 559, 587, 701, 751, 775, 799, 825, 861, 885)
    labels = (
        AppearanceLabel.TEAM_B,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
    )

    # Persistent but not enormous kit shift: cross-state cosine is around the
    # high 0.8s, deliberately above V2's ordinary 0.80 hard cutoff.
    kit_angles = (
        0.0, 6.0, -4.0, 8.0,
        30.0, 34.0, 26.0, 31.0, 28.0, 33.0,
    )

    # ReID supports the change, but does not get to be the primary decision.
    reid_angles = (
        -8.0, 8.0, -12.0, 12.0,
        42.0, 52.0, 37.0, 48.0, 40.0, 46.0,
    )

    samples = [
        sample(
            frame,
            angle_vector(reid_angle),
            label,
            raw_id=35,
            kit=angle_vector(kit_angle),
        )
        for frame, label, kit_angle, reid_angle in zip(
            frames,
            labels,
            kit_angles,
            reid_angles,
        )
    ]

    result = segment_track_identity(
        samples,
        config=IdentitySegmentationConfig(
            # Make sure this regression is testing the V3 replacement rule,
            # not the pre-existing strong-ReID fallback.
            min_strong_reid_gain=1.0,
        ),
    )

    assert result.segmented
    assert result.boundaries_after_frame == (701,)
    assert result.accepted_splits[0].reason == (
        "persistent_state_replacement+kit+reid"
    )


def test_v3_mixed_semantic_prefix_alone_does_not_split_same_person():
    """A dirty TEAM label history is insufficient without visual change."""

    frames = (500, 559, 587, 701, 751, 775, 799, 825, 861, 885)
    labels = (
        AppearanceLabel.TEAM_B,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
    )

    samples = [
        sample(
            frame,
            angle_vector(2.0 + (index % 3) * 1.5),
            label,
            raw_id=99,
            kit=angle_vector(5.0 + (index % 2) * 2.0),
        )
        for index, (frame, label) in enumerate(
            zip(frames, labels)
        )
    ]

    result = segment_track_identity(samples)

    assert not result.segmented


def test_v3_pose_drift_with_kit_gain_but_same_reid_does_not_split():
    """Kit pose/lighting drift cannot trigger state replacement by itself."""

    frames = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    labels = (
        AppearanceLabel.TEAM_B,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_A,
    )

    kit_angles = (
        0.0, 4.0, -3.0, 6.0,
        30.0, 34.0, 27.0, 31.0, 29.0, 33.0,
    )

    # Same-person ReID remains almost identical across the apparent kit drift.
    reid_angles = (
        1.0, -1.0, 2.0, 0.0,
        2.0, 1.0, -1.0, 0.0, 1.5, -0.5,
    )

    samples = [
        sample(
            frame,
            angle_vector(reid_angle),
            label,
            raw_id=100,
            kit=angle_vector(kit_angle),
        )
        for frame, label, kit_angle, reid_angle in zip(
            frames,
            labels,
            kit_angles,
            reid_angles,
        )
    ]

    result = segment_track_identity(
        samples,
        config=IdentitySegmentationConfig(
            min_strong_reid_gain=1.0,
        ),
    )

    assert not result.segmented


def test_v4_clean_stable_segment_is_marked_clean():
    samples = [
        sample(10, unit(1.0, 0.02), AppearanceLabel.TEAM_A, kit=unit(1.0, 0.03)),
        sample(20, unit(0.99, 0.04), AppearanceLabel.TEAM_A, kit=unit(1.0, 0.02)),
        sample(30, unit(1.0, 0.05), AppearanceLabel.TEAM_A, kit=unit(0.99, 0.04)),
        sample(40, unit(0.98, 0.06), AppearanceLabel.TEAM_A, kit=unit(1.0, 0.01)),
    ]

    result = segment_track_identity(samples)

    assert not result.segmented
    assert result.segments[0].identity_quality is SegmentIdentityQuality.CLEAN
    assert result.segments[0].identity_quality_reason == "stable_single_state"


def test_v4_sparse_two_crop_segment_is_ambiguous_not_clean():
    samples = [
        sample(10, unit(1.0, 0.02), AppearanceLabel.TEAM_A, kit=unit(1.0, 0.03)),
        sample(20, unit(0.99, 0.04), AppearanceLabel.TEAM_A, kit=unit(1.0, 0.02)),
    ]

    result = segment_track_identity(samples)

    assert not result.segmented
    assert result.segments[0].identity_quality is SegmentIdentityQuality.AMBIGUOUS


def test_v4_persistent_mixed_segment_is_contaminated():
    # One unsplit span contains two repeated semantic states and disagreement
    # in both raw kit and ReID space.  It must not be trusted as one person.
    samples = [
        sample(10, angle_vector(0), AppearanceLabel.TEAM_A, kit=angle_vector(0)),
        sample(20, angle_vector(4), AppearanceLabel.TEAM_A, kit=angle_vector(5)),
        sample(30, angle_vector(50), AppearanceLabel.TEAM_B, kit=angle_vector(42)),
        sample(40, angle_vector(55), AppearanceLabel.TEAM_B, kit=angle_vector(47)),
        sample(50, angle_vector(7), AppearanceLabel.TEAM_A, kit=angle_vector(3)),
        sample(60, angle_vector(58), AppearanceLabel.TEAM_B, kit=angle_vector(45)),
    ]

    # Disable further splitting to inspect the quality classifier itself.
    result = segment_track_identity(
        samples,
        config=IdentitySegmentationConfig(
            max_segments_per_track=1,
        ),
    )

    assert not result.segmented
    segment = result.segments[0]
    assert segment.identity_quality is SegmentIdentityQuality.CONTAMINATED
    assert segment.secondary_appearance_count >= 2
    assert segment.identity_quality_reason == "persistent_multi_state_contamination"


def test_v4_special_goalkeeper_like_track_can_still_be_clean():
    samples = [
        sample(10, angle_vector(0), AppearanceLabel.SPECIAL, kit=angle_vector(0)),
        sample(20, angle_vector(2), AppearanceLabel.SPECIAL, kit=angle_vector(1)),
        sample(30, angle_vector(-2), AppearanceLabel.SPECIAL, kit=angle_vector(-1)),
        sample(40, angle_vector(1), AppearanceLabel.SPECIAL, kit=angle_vector(2)),
    ]

    result = segment_track_identity(samples)

    segment = result.segments[0]
    assert segment.identity_quality is SegmentIdentityQuality.CLEAN
    # SPECIAL is intentionally not forced into TEAM_A/TEAM_B.
    assert segment.team_confidence == 0.0
