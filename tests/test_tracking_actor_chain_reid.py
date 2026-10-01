from __future__ import annotations

from tracking.actors.chain_reid import (
    ChainReIDConfig,
    build_chain_candidates,
    build_chain_profiles,
    fit_chain_calibration,
)
from tracking.appearance.evidence import (
    DistributionStats,
    TrackletAppearanceProfile,
)


def _stats():
    return DistributionStats(
        count=1,
        min=0.9,
        p10=0.9,
        median=0.9,
        p90=0.9,
        p95=0.9,
        max=0.9,
        mean=0.9,
        std=0.0,
    )


def _profile(tracklet_id, embedding, time):
    return TrackletAppearanceProfile(
        tracklet_id=tracklet_id,
        model_id="m",
        descriptor_name="reid",
        embedding_dim=len(embedding),
        aggregate_embedding=embedding,
        medoid_sample_id=f"{tracklet_id}-s",
        samples=(),
        quality_stats=_stats(),
        centroid_similarity=_stats(),
        internal_pair_similarity=DistributionStats(count=0),
        effective_sample_count=1.0,
        dispersion=0.0,
        reliability=0.9,
        first_timestamp_s=time,
        last_timestamp_s=time,
    )


def _actor(actor_id, tracklets, start, end):
    return {
        "actor_id": actor_id,
        "tracklet_ids": tracklets,
        "start_time_s": start,
        "end_time_s": end,
    }


def _assign(tracklet, actor):
    return {
        "tracklet_id": tracklet,
        "actor_id": actor,
    }


def _kit(tracklet, cluster):
    return {
        "tracklet_id": tracklet,
        "cluster_id": cluster,
        "best_cosine": 0.95,
        "margin": 0.25,
        "profile_reliability": 0.9,
    }


def test_chain_profile_aggregates_multiple_tracklets():
    profiles = [
        _profile("a1", (1.0, 0.0), 0.0),
        _profile("a2", (0.99, 0.01), 0.5),
    ]
    result = build_chain_profiles(
        actor_rows=[_actor("A", ["a1", "a2"], 0.0, 1.0)],
        assignments=[
            _assign("a1", "A"),
            _assign("a2", "A"),
        ],
        tracklet_profiles=profiles,
        kit_assignments=[
            _kit("a1", 0),
            _kit("a2", 0),
        ],
    )
    assert len(result) == 1
    assert result[0].profiled_tracklet_count == 2
    assert result[0].kit_cluster_id == 0


def test_chain_calibration_uses_split_positive_and_hard_negative():
    actors = [
        _actor("A", ["a1", "a2"], 0.0, 1.0),
        _actor("B", ["b1", "b2"], 0.0, 1.0),
    ]
    assignments = [
        _assign("a1", "A"),
        _assign("a2", "A"),
        _assign("b1", "B"),
        _assign("b2", "B"),
    ]
    profiles = [
        _profile("a1", (1.0, 0.0), 0.0),
        _profile("a2", (0.99, 0.01), 0.5),
        _profile("b1", (0.0, 1.0), 0.0),
        _profile("b2", (0.01, 0.99), 0.5),
    ]
    kits = [
        _kit("a1", 0), _kit("a2", 0),
        _kit("b1", 1), _kit("b2", 1),
    ]
    chains = build_chain_profiles(
        actor_rows=actors,
        assignments=assignments,
        tracklet_profiles=profiles,
        kit_assignments=kits,
    )
    calibration, positives, negatives = fit_chain_calibration(
        chain_profiles=chains,
        tracklet_profiles=profiles,
        assignments=assignments,
        hard_conflicts=[
            {"tracklet_a": "a1", "tracklet_b": "b1"},
        ],
    )
    assert len(positives) == 2
    assert len(negatives) == 1
    assert calibration.positive[0] > 0.99
    assert calibration.different_kit_negative[0] < 0.1


def test_retrieval_only_allows_sequential_chains():
    profiles = [
        _profile("a1", (1.0, 0.0), 0.0),
        _profile("a2", (0.99, 0.01), 0.5),
        _profile("b1", (1.0, 0.01), 2.0),
        _profile("b2", (0.99, 0.02), 2.5),
        _profile("c1", (0.0, 1.0), 0.2),
        _profile("c2", (0.01, 0.99), 0.7),
    ]
    actors = [
        _actor("A", ["a1", "a2"], 0.0, 1.0),
        _actor("B", ["b1", "b2"], 2.0, 3.0),
        _actor("C", ["c1", "c2"], 0.2, 1.2),
    ]
    assignments = [
        _assign("a1", "A"), _assign("a2", "A"),
        _assign("b1", "B"), _assign("b2", "B"),
        _assign("c1", "C"), _assign("c2", "C"),
    ]
    kits = [
        _kit("a1", 0), _kit("a2", 0),
        _kit("b1", 0), _kit("b2", 0),
        _kit("c1", 1), _kit("c2", 1),
    ]
    chains = build_chain_profiles(
        actor_rows=actors,
        assignments=assignments,
        tracklet_profiles=profiles,
        kit_assignments=kits,
    )
    calibration, _, _ = fit_chain_calibration(
        chain_profiles=chains,
        tracklet_profiles=profiles,
        assignments=assignments,
        hard_conflicts=[
            {"tracklet_a": "a1", "tracklet_b": "c1"},
        ],
    )
    candidates = build_chain_candidates(
        chain_profiles=chains,
        calibration=calibration,
        hard_actor_pairs={("A", "C")},
        config=ChainReIDConfig(
            top_k_successors=4,
            top_k_predecessors=4,
        ),
    )
    pairs = {
        (row.predecessor_actor_id, row.successor_actor_id)
        for row in candidates
    }
    assert ("A", "B") in pairs
    assert ("A", "C") not in pairs
    assert ("C", "A") not in pairs


def test_chain_competition_marks_best_successor_and_predecessor():
    # Manual calibration avoids requiring extra negative actor pairs here.
    from tracking.actors.chain_reid import (
        ChainCalibration,
        ChainProfile,
    )

    chains = (
        ChainProfile(
            "A", ("a",), ("a",), (1.0, 0.0),
            0.9, 0.0, 1.0, 0, 1.0,
        ),
        ChainProfile(
            "B", ("b",), ("b",), (0.999, 0.001),
            0.9, 2.0, 3.0, 0, 1.0,
        ),
        ChainProfile(
            "C", ("c",), ("c",), (0.8, 0.2),
            0.9, 2.0, 3.0, 0, 1.0,
        ),
    )
    calibration = ChainCalibration(
        positive=(0.9, 0.95, 0.99),
        same_kit_negative=(0.4, 0.6, 0.8),
        different_kit_negative=(0.2, 0.4),
        unknown_kit_negative=(),
    )
    candidates = build_chain_candidates(
        chain_profiles=chains,
        calibration=calibration,
        hard_actor_pairs=set(),
        config=ChainReIDConfig(
            top_k_successors=4,
            top_k_predecessors=4,
        ),
    )
    by_pair = {
        (row.predecessor_actor_id, row.successor_actor_id): row
        for row in candidates
    }
    assert by_pair[("A", "B")].outgoing_score_rank == 1


def test_trusted_positive_override_is_used_for_calibration():
    actors = [
        _actor("A", ["a1", "a2"], 0.0, 1.0),
        _actor("B", ["b1", "b2"], 0.0, 1.0),
    ]
    assignments = [
        _assign("a1", "A"), _assign("a2", "A"),
        _assign("b1", "B"), _assign("b2", "B"),
    ]
    profiles = [
        _profile("a1", (1.0, 0.0), 0.0),
        _profile("a2", (0.2, 0.98), 0.5),
        _profile("b1", (0.0, 1.0), 0.0),
        _profile("b2", (0.01, 0.99), 0.5),
    ]
    kits = [
        _kit("a1", 0), _kit("a2", 0),
        _kit("b1", 1), _kit("b2", 1),
    ]
    chains = build_chain_profiles(
        actor_rows=actors,
        assignments=assignments,
        tracklet_profiles=profiles,
        kit_assignments=kits,
    )
    trusted = (0.90, 0.94, 0.98)
    calibration, chain_split_rows, _ = fit_chain_calibration(
        chain_profiles=chains,
        tracklet_profiles=profiles,
        assignments=assignments,
        hard_conflicts=[
            {"tracklet_a": "a1", "tracklet_b": "b1"},
        ],
        trusted_positive_cosines=trusted,
    )

    assert calibration.positive == trusted
    assert len(chain_split_rows) == 2
