from __future__ import annotations

import numpy as np

from tracking.appearance.kit_family_consensus import (
    KitFamilyConsensusConfig,
    aggregate_tracklet_consensus,
    build_kit_family_consensus,
    SampleFamilyEvidence,
)


def _cluster(family_type, family_id, vector):
    vector = np.asarray(vector, dtype=float)
    vector /= np.linalg.norm(vector)
    return {
        "family_type": family_type,
        "family_id": family_id,
        "centroid": vector.tolist(),
    }


def _profile(tracklet_id, vector):
    vector = np.asarray(vector, dtype=float)
    vector /= np.linalg.norm(vector)
    return {
        "tracklet_id": tracklet_id,
        "signature": vector.tolist(),
    }


def _profile_assignment(tracklet_id, family_type, family_id):
    return {
        "tracklet_id": tracklet_id,
        "family_type": family_type,
        "family_id": family_id,
        "usable_for_tracking": True,
    }


def _sample(tracklet_id, detection_id, vector, t, quality=0.95):
    vector = np.asarray(vector, dtype=float)
    vector /= np.linalg.norm(vector)
    return {
        "tracklet_id": tracklet_id,
        "detection_id": detection_id,
        "frame_index": int(round(t * 10)),
        "timestamp_s": t,
        "crop_ref": f"{detection_id}.jpg",
        "quality": quality,
        "signature": vector.tolist(),
    }


def _base_sources():
    primary0 = np.asarray([1.0, 0.0, 0.0, 0.0])
    primary1 = np.asarray([0.0, 1.0, 0.0, 0.0])
    special0 = np.asarray([0.0, 0.0, 1.0, 0.0])

    clusters = [
        _cluster("primary", 0, primary0),
        _cluster("primary", 1, primary1),
        _cluster("special", 0, special0),
    ]

    profiles = []
    assignments = []
    for family_type, family_id, vector in [
        ("primary", 0, primary0),
        ("primary", 1, primary1),
        ("special", 0, special0),
    ]:
        for index in range(6):
            tracklet_id = f"cal-{family_type}-{family_id}-{index}"
            profiles.append(_profile(tracklet_id, vector))
            assignments.append(
                _profile_assignment(
                    tracklet_id, family_type, family_id
                )
            )

    return clusters, profiles, assignments, primary0, primary1, special0


def test_one_contaminated_crop_does_not_flip_primary_tracklet():
    (
        clusters, profiles, assignments,
        p0, p1, _special,
    ) = _base_sources()

    samples = [
        _sample("trk", f"a{i}", p0, i * 0.2)
        for i in range(5)
    ]
    samples.append(_sample("trk", "bad", p1, 1.1))

    result = build_kit_family_consensus(
        sample_rows=samples,
        profile_rows=profiles,
        profile_assignment_rows=assignments,
        cluster_rows=clusters,
    )
    row = result.tracklets[0]

    assert row.family_type == "primary"
    assert row.family_id == 0
    assert row.usable_for_tracking is True
    assert row.support_samples == 5
    assert row.conflicting_samples == 1
    assert row.dominance > 0.80


def test_even_strong_single_wrong_crop_cannot_overturn_many_good_votes():
    (
        clusters, profiles, assignments,
        p0, p1, _special,
    ) = _base_sources()

    samples = [
        _sample("trk", f"good{i}", p0, i * 0.2, quality=0.80)
        for i in range(7)
    ]
    samples.append(
        _sample("trk", "wrong", p1, 1.5, quality=1.0)
    )

    result = build_kit_family_consensus(
        sample_rows=samples,
        profile_rows=profiles,
        profile_assignment_rows=assignments,
        cluster_rows=clusters,
    )
    row = result.tracklets[0]
    assert row.family_key == "primary-0"
    assert row.conflicting_samples == 1


def test_mixed_tracklet_stays_unknown():
    config = KitFamilyConsensusConfig()
    rows = []
    for index in range(3):
        rows.append(
            SampleFamilyEvidence(
                tracklet_id="mixed",
                detection_id=f"a{index}",
                frame_index=index,
                timestamp_s=index * 0.2,
                crop_ref=f"a{index}.jpg",
                quality=1.0,
                winner_family_type="primary",
                winner_family_id=0,
                best_cosine=0.95,
                second_best_cosine=0.2,
                margin=0.75,
                cosine_threshold=0.8,
                margin_threshold=0.05,
                vote_weight=1.0,
                state="confident_vote",
            )
        )
        rows.append(
            SampleFamilyEvidence(
                tracklet_id="mixed",
                detection_id=f"b{index}",
                frame_index=10 + index,
                timestamp_s=index * 0.2 + 0.1,
                crop_ref=f"b{index}.jpg",
                quality=1.0,
                winner_family_type="primary",
                winner_family_id=1,
                best_cosine=0.95,
                second_best_cosine=0.2,
                margin=0.75,
                cosine_threshold=0.8,
                margin_threshold=0.05,
                vote_weight=1.0,
                state="confident_vote",
            )
        )

    result = aggregate_tracklet_consensus(
        sample_evidence=rows,
        config=config,
    )[0]
    assert result.family_type == "unknown"
    assert result.state == "weak_primary"


def test_special_requires_repeated_temporal_support():
    config = KitFamilyConsensusConfig(
        special_temporal_bin_s=0.5,
        special_min_temporal_bins=2,
    )

    rows = [
        SampleFamilyEvidence(
            tracklet_id="special",
            detection_id="s1",
            frame_index=0,
            timestamp_s=0.00,
            crop_ref="s1.jpg",
            quality=1.0,
            winner_family_type="special",
            winner_family_id=0,
            best_cosine=0.95,
            second_best_cosine=0.2,
            margin=0.75,
            cosine_threshold=0.8,
            margin_threshold=0.05,
            vote_weight=1.0,
            state="confident_vote",
        ),
        SampleFamilyEvidence(
            tracklet_id="special",
            detection_id="s2",
            frame_index=10,
            timestamp_s=0.80,
            crop_ref="s2.jpg",
            quality=1.0,
            winner_family_type="special",
            winner_family_id=0,
            best_cosine=0.96,
            second_best_cosine=0.2,
            margin=0.76,
            cosine_threshold=0.8,
            margin_threshold=0.05,
            vote_weight=1.0,
            state="confident_vote",
        ),
    ]

    result = aggregate_tracklet_consensus(
        sample_evidence=rows,
        config=config,
    )[0]
    assert result.family_key == "special-0"
    assert result.support_temporal_bins == 2
    assert result.usable_for_tracking is True


def test_special_burst_in_one_temporal_bin_is_not_trusted():
    config = KitFamilyConsensusConfig(
        special_temporal_bin_s=0.75,
        special_min_temporal_bins=2,
    )
    rows = []
    for index, timestamp in enumerate((0.0, 0.1, 0.2)):
        rows.append(
            SampleFamilyEvidence(
                tracklet_id="burst",
                detection_id=f"s{index}",
                frame_index=index,
                timestamp_s=timestamp,
                crop_ref=f"s{index}.jpg",
                quality=1.0,
                winner_family_type="special",
                winner_family_id=0,
                best_cosine=0.95,
                second_best_cosine=0.2,
                margin=0.75,
                cosine_threshold=0.8,
                margin_threshold=0.05,
                vote_weight=1.0,
                state="confident_vote",
            )
        )

    result = aggregate_tracklet_consensus(
        sample_evidence=rows,
        config=config,
    )[0]
    assert result.family_type == "unknown"
    assert result.state == "weak_special"
    assert result.support_temporal_bins == 1


def test_consensus_never_creates_hard_identity_constraint():
    (
        clusters, profiles, assignments,
        p0, _p1, _special,
    ) = _base_sources()

    samples = [
        _sample("trk", "a", p0, 0.0),
        _sample("trk", "b", p0, 1.0),
    ]

    result = build_kit_family_consensus(
        sample_rows=samples,
        profile_rows=profiles,
        profile_assignment_rows=assignments,
        cluster_rows=clusters,
    )
    summary = result.summary()

    assert summary["identity_hard_constraints_created"] is False
    assert summary["decision_unit"] == "whole_conservative_tracklet"
    assert summary["contaminated_crop_tolerance"] is True
