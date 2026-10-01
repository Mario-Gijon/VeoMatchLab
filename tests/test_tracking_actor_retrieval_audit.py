from __future__ import annotations

import pytest

from tracking.actors.retrieval_audit import (
    RetrievalAuditConfig,
    audit_local_physics,
    audit_reid_split_retrieval,
)
from tracking.appearance.evidence import (
    AppearanceEvidenceConfig,
    AppearanceEvidenceSample,
    build_tracklet_profiles,
)
from tracking.tracklets.models import (
    Tracklet,
    TrackletObservation,
)


def _tracklet(
    tracklet_id: str,
    rows: list[tuple[int, float, float, float]],
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=tuple(
            TrackletObservation(
                detection_id=f"{tracklet_id}-det-{index}",
                frame_index=frame,
                timestamp_s=timestamp,
                x=x,
                z=z,
                confidence=0.95,
            )
            for index, (frame, timestamp, x, z) in enumerate(rows)
        ),
    )


def _appearance_samples(
    tracklet_id: str,
    embeddings: list[tuple[float, ...]],
    *,
    frame_base: int = 0,
):
    return [
        AppearanceEvidenceSample(
            sample_id=f"{tracklet_id}-sample-{index}",
            tracklet_id=tracklet_id,
            detection_id=f"{tracklet_id}-det-{index}",
            frame_index=frame_base + index,
            timestamp_s=float(frame_base + index) * 0.1,
            quality=0.9,
            embedding=embedding,
        )
        for index, embedding in enumerate(embeddings)
    ]


def test_local_proxy_recovers_normal_same_tracklet_motion():
    tracklet = _tracklet(
        "a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 0.8, 0.0),
            (6, 0.2, 1.6, 0.0),
        ],
    )
    result = audit_local_physics(
        [tracklet],
        RetrievalAuditConfig(),
    )

    assert result.eligible_adjacent_transition_count == 2
    assert result.recovered_transition_count == 2
    assert result.recall == pytest.approx(1.0)


def test_local_proxy_surfaces_impossible_internal_jump():
    tracklet = _tracklet(
        "a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 10.0, 0.0),
        ],
    )
    result = audit_local_physics(
        [tracklet],
        RetrievalAuditConfig(),
    )

    assert result.recovered_transition_count == 0
    assert result.missed_transition_count == 1
    assert result.worst_misses[0]["tracklet_id"] == "a"


def test_reid_split_mate_ranks_first_for_clean_identity():
    evidence = AppearanceEvidenceConfig()
    samples = (
        _appearance_samples(
            "a",
            [
                (1.0, 0.0),
                (0.99, 0.01),
                (0.98, 0.02),
                (0.97, 0.03),
            ],
        )
        + _appearance_samples(
            "b",
            [
                (0.0, 1.0),
                (0.01, 0.99),
                (0.02, 0.98),
                (0.03, 0.97),
            ],
            frame_base=100,
        )
    )
    profiles = build_tracklet_profiles(samples, evidence)

    result = audit_reid_split_retrieval(
        samples,
        profiles,
        evidence,
        RetrievalAuditConfig(rank_cutoffs=(1, 2)),
    )

    assert result.query_count == 4
    assert result.recall_at["1"] == pytest.approx(1.0)
    assert all(
        item.mate_rank == 1
        for item in result.worst_queries
    )


def test_reid_split_excludes_original_full_profile_from_gallery():
    evidence = AppearanceEvidenceConfig()
    samples = _appearance_samples(
        "a",
        [
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
        ],
    )
    profiles = build_tracklet_profiles(samples, evidence)

    result = audit_reid_split_retrieval(
        samples,
        profiles,
        evidence,
        RetrievalAuditConfig(rank_cutoffs=(1,)),
    )

    assert result.query_count == 2
    assert result.recall_at["1"] == pytest.approx(1.0)
    assert all(
        item.gallery_size == 1
        for item in result.worst_queries
    )


def test_tracklets_with_too_few_samples_are_not_split():
    evidence = AppearanceEvidenceConfig()
    samples = _appearance_samples(
        "a",
        [
            (1.0, 0.0),
            (1.0, 0.0),
            (1.0, 0.0),
        ],
    )
    profiles = build_tracklet_profiles(samples, evidence)

    result = audit_reid_split_retrieval(
        samples,
        profiles,
        evidence,
        RetrievalAuditConfig(rank_cutoffs=(1,)),
    )

    assert result.eligible_tracklet_count == 0
    assert result.query_count == 0
    assert result.recall_at["1"] is None
