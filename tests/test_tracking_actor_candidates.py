from __future__ import annotations

from tracking.actors.candidates import (
    UnifiedCandidateConfig,
    build_unified_candidate_graph,
)
from tracking.actors.index import (
    SparseTrackletIndexConfig,
    build_sparse_tracklet_index,
)
from tracking.actors.reid_retrieval import (
    SparseReIDRetrievalConfig,
    build_sparse_reid_retrieval,
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
    samples: list[tuple[int, float, float, float]],
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
            for index, (frame, timestamp, x, z) in enumerate(samples)
        ),
    )


def _profiles(
    rows: dict[str, tuple[float, ...]],
):
    config = AppearanceEvidenceConfig()
    samples = []
    for index, (tracklet_id, embedding) in enumerate(sorted(rows.items())):
        samples.append(
            AppearanceEvidenceSample(
                sample_id=f"sample-{index}",
                tracklet_id=tracklet_id,
                detection_id=f"{tracklet_id}-appearance",
                frame_index=index,
                timestamp_s=float(index),
                quality=0.9,
                embedding=embedding,
            )
        )
    return tuple(build_tracklet_profiles(samples, config))


def _build(
    tracklets,
    profiles,
    *,
    horizon=1.0,
    top_k=2,
    compare_fn=None,
):
    physical = build_sparse_tracklet_index(
        tracklets,
        SparseTrackletIndexConfig(
            local_gap_horizon_s=horizon,
        ),
    )
    reid = build_sparse_reid_retrieval(
        profiles,
        hard_conflicts=physical.hard_coexistence_conflicts(),
        config=SparseReIDRetrievalConfig(
            top_k_per_tracklet=top_k,
        ),
    )
    kwargs = {}
    if compare_fn is not None:
        kwargs["appearance_compare_fn"] = compare_fn
    return build_unified_candidate_graph(
        physical_index=physical,
        reid_retrieval=reid,
        profiles=profiles,
        config=UnifiedCandidateConfig(
            local_gap_horizon_s=horizon,
        ),
        **kwargs,
    )


def _candidate_map(graph):
    return {
        (item.tracklet_a, item.tracklet_b): item
        for item in graph.candidates
    }


def _conflict_map(graph):
    return {
        (item.tracklet_a, item.tracklet_b): item
        for item in graph.hard_conflicts
    }


def test_reachable_short_transition_is_candidate():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(3, 0.1, 1.0, 0.0)])

    graph = _build([a, b], ())
    candidate = _candidate_map(graph)[("a", "b")]

    assert "local_physics" in candidate.reasons


def test_unreachable_short_transition_becomes_hard_conflict():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(3, 0.1, 10.0, 0.0)])

    graph = _build([a, b], ())
    conflict = _conflict_map(graph)[("a", "b")]

    assert "short_gap_unreachable" in conflict.reasons
    assert ("a", "b") not in _candidate_map(graph)


def test_simultaneous_far_pair_is_hard_even_with_high_reid():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(0, 0.0, 5.0, 0.0)])
    profiles = _profiles(
        {
            "a": (1.0, 0.0),
            "b": (1.0, 0.0),
        }
    )

    graph = _build([a, b], profiles, top_k=1)

    assert ("a", "b") in _conflict_map(graph)
    assert ("a", "b") not in _candidate_map(graph)


def test_close_overlap_survives_as_candidate():
    a = _tracklet(
        "a",
        [(0, 0.0, 0.0, 0.0), (3, 0.1, 0.2, 0.0)],
    )
    b = _tracklet(
        "b",
        [(0, 0.0, 0.4, 0.0), (3, 0.1, 0.5, 0.0)],
    )

    graph = _build([a, b], ())
    candidate = _candidate_map(graph)[("a", "b")]

    assert "close_overlap" in candidate.reasons


def test_global_reid_recovers_long_gap_candidate():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(900, 30.0, 20.0, 0.0)])
    c = _tracklet("c", [(900, 30.0, -20.0, 0.0)])
    profiles = _profiles(
        {
            "a": (1.0, 0.0),
            "b": (0.99, 0.01),
            "c": (0.0, 1.0),
        }
    )

    graph = _build([a, b, c], profiles, top_k=1)
    candidate = _candidate_map(graph)[("a", "b")]

    assert candidate.reasons == ("appearance_topk",)


def test_missing_reid_does_not_remove_local_candidate():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(3, 0.1, 1.0, 0.0)])
    profiles = _profiles({"a": (1.0, 0.0)})

    graph = _build([a, b], profiles, top_k=1)
    candidate = _candidate_map(graph)[("a", "b")]

    assert "local_physics" in candidate.reasons
    assert candidate.appearance_evidence is None


def test_reasons_are_unioned_not_duplicated():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(3, 0.1, 1.0, 0.0)])
    profiles = _profiles(
        {
            "a": (1.0, 0.0),
            "b": (0.99, 0.01),
        }
    )

    graph = _build([a, b], profiles, top_k=1)
    candidate = _candidate_map(graph)[("a", "b")]

    assert candidate.reasons == (
        "appearance_topk",
        "local_physics",
    )


def test_rich_appearance_is_computed_only_for_sparse_union():
    tracklets = [
        _tracklet("a", [(0, 0.0, 0.0, 0.0)]),
        _tracklet("b", [(300, 10.0, 0.0, 0.0)]),
        _tracklet("c", [(600, 20.0, 0.0, 0.0)]),
        _tracklet("d", [(900, 30.0, 0.0, 0.0)]),
    ]
    profiles = _profiles(
        {
            "a": (1.0, 0.0),
            "b": (0.99, 0.01),
            "c": (0.0, 1.0),
            "d": (0.01, 0.99),
        }
    )

    calls = []

    from tracking.appearance.evidence import compare_tracklet_profiles

    def recording_compare(first, second):
        calls.append((first.tracklet_id, second.tracklet_id))
        return compare_tracklet_profiles(first, second)

    graph = _build(
        tracklets,
        profiles,
        top_k=1,
        compare_fn=recording_compare,
    )

    assert len(calls) == len(graph.candidates)
    assert len(calls) < 6  # 4 choose 2


def test_summary_does_not_treat_absent_pairs_as_cannot_link():
    a = _tracklet("a", [(0, 0.0, 0.0, 0.0)])
    b = _tracklet("b", [(300, 10.0, 0.0, 0.0)])
    c = _tracklet("c", [(600, 20.0, 0.0, 0.0)])
    profiles = _profiles(
        {
            "a": (1.0, 0.0),
            "b": (0.9, 0.1),
            "c": (0.0, 1.0),
        }
    )

    graph = _build([a, b, c], profiles, top_k=1)
    summary = graph.summary()

    assert summary["absence_of_candidate_is_cannot_link"] is False
    assert summary["identity_decisions"] is False
