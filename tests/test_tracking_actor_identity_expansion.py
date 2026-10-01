from __future__ import annotations

from tracking.actors.identity_expansion import (
    ExpansionCandidate,
    IdentityCalibration,
    IdentityExpansionConfig,
    build_expansion_candidates,
    build_hard_identity_pairs,
    build_identity_family_evidence,
    build_identity_profiles,
    family_pair_relation,
    family_score_adjustment,
    solve_seeded_expansion,
)


def _identity(identity_id, actors, tracklets, start, end):
    return {
        "identity_id": identity_id,
        "actor_ids": list(actors),
        "tracklet_ids": list(tracklets),
        "start_time_s": start,
        "end_time_s": end,
    }


def _chain_profile(
    actor_id,
    tracklets,
    embedding,
    start,
    end,
    *,
    reliability=0.9,
    kit=0,
):
    return {
        "actor_id": actor_id,
        "profiled_tracklet_ids": list(tracklets),
        "profiled_tracklet_count": len(tracklets),
        "aggregate_embedding": list(embedding),
        "reliability": reliability,
        "first_timestamp_s": start,
        "last_timestamp_s": end,
        "kit_cluster_id": kit,
        "kit_dominance": 1.0,
    }


def _actor_assignment(actor, identity):
    return {
        "actor_id": actor,
        "identity_id": identity,
    }


def _tracklet_assignment(tracklet, identity):
    return {
        "tracklet_id": tracklet,
        "identity_id": identity,
    }


def _candidate(
    predecessor,
    successor,
    *,
    score=0.8,
    gap=1.0,
    kit="same_kit",
    mutual_best=True,
    mutual_top3=True,
):
    return ExpansionCandidate(
        predecessor_identity_id=predecessor,
        successor_identity_id=successor,
        gap_s=gap,
        aggregate_cosine=0.98,
        kit_family=kit,
        empirical_support=0.9,
        profile_confidence=0.8,
        forward_retrieval_rank=1,
        backward_retrieval_rank=1,
        retrieval_support=1.0,
        rank_score=score,
        outgoing_score_rank=1,
        incoming_score_rank=1,
        mutual_best=mutual_best,
        mutual_top3=mutual_top3,
    )


def test_identity_profile_seed_requires_multiple_profiled_tracklets():
    identities = [
        _identity("I1", ["A"], ["a1", "a2"], 0.0, 1.0),
        _identity("I2", ["B"], ["b1"], 2.0, 3.0),
    ]
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile(
                "A",
                ["a1", "a2"],
                (1.0, 0.0),
                0.0,
                1.0,
            ),
            _chain_profile(
                "B",
                ["b1"],
                (0.99, 0.01),
                2.0,
                3.0,
            ),
        ],
        config=IdentityExpansionConfig(
            min_seed_profiled_tracklets=2
        ),
    )
    by_id = {row.identity_id: row for row in profiles}
    assert by_id["I1"].initial_seed is True
    assert by_id["I2"].initial_seed is False


def test_unseeded_pair_is_not_merged():
    identities = [
        _identity("I1", ["A"], ["a"], 0.0, 1.0),
        _identity("I2", ["B"], ["b"], 2.0, 3.0),
    ]
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile(
                "A", ["a"], (1.0, 0.0), 0.0, 1.0
            ),
            _chain_profile(
                "B", ["b"], (0.99, 0.01), 2.0, 3.0
            ),
        ],
        config=IdentityExpansionConfig(
            min_seed_profiled_tracklets=2
        ),
    )
    selected, expanded, *_rest = solve_seeded_expansion(
        identity_rows=identities,
        persistent_actor_assignments=[
            _actor_assignment("A", "I1"),
            _actor_assignment("B", "I2"),
        ],
        persistent_tracklet_assignments=[
            _tracklet_assignment("a", "I1"),
            _tracklet_assignment("b", "I2"),
        ],
        profiles=profiles,
        candidates=[_candidate("I1", "I2")],
        hard_identity_pairs=set(),
        config=IdentityExpansionConfig(
            min_seed_profiled_tracklets=2
        ),
    )
    assert selected == ()
    assert len(expanded) == 2


def test_seed_can_absorb_and_then_grow_outward():
    identities = [
        _identity(
            "I1", ["A"], ["a1", "a2"], 0.0, 1.0
        ),
        _identity("I2", ["B"], ["b"], 2.0, 3.0),
        _identity("I3", ["C"], ["c"], 4.0, 5.0),
    ]
    chain_profiles = [
        _chain_profile(
            "A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0
        ),
        _chain_profile(
            "B", ["b"], (0.99, 0.01), 2.0, 3.0
        ),
        _chain_profile(
            "C", ["c"], (0.98, 0.02), 4.0, 5.0
        ),
    ]
    config = IdentityExpansionConfig(
        min_seed_profiled_tracklets=2
    )
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=chain_profiles,
        config=config,
    )
    selected, expanded, *_rest = solve_seeded_expansion(
        identity_rows=identities,
        persistent_actor_assignments=[
            _actor_assignment("A", "I1"),
            _actor_assignment("B", "I2"),
            _actor_assignment("C", "I3"),
        ],
        persistent_tracklet_assignments=[
            _tracklet_assignment("a1", "I1"),
            _tracklet_assignment("a2", "I1"),
            _tracklet_assignment("b", "I2"),
            _tracklet_assignment("c", "I3"),
        ],
        profiles=profiles,
        candidates=[
            _candidate("I2", "I3", score=0.90),
            _candidate("I1", "I2", score=0.80),
        ],
        hard_identity_pairs=set(),
        config=config,
    )
    # I2->I3 is initially deferred because neither is seeded.
    # I1->I2 seeds the component, then I2->I3 is accepted next pass.
    assert len(selected) == 2
    assert len(expanded) == 1
    assert expanded[0].source_identity_ids == (
        "I1", "I2", "I3"
    )


def test_transitive_hard_conflict_blocks_seed_growth():
    identities = [
        _identity(
            "I1", ["A"], ["a1", "a2"], 0.0, 1.0
        ),
        _identity("I2", ["B"], ["b"], 2.0, 3.0),
        _identity("I3", ["C"], ["c"], 4.0, 5.0),
    ]
    config = IdentityExpansionConfig(
        min_seed_profiled_tracklets=2
    )
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile(
                "A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0
            ),
            _chain_profile(
                "B", ["b"], (0.99, 0.01), 2.0, 3.0
            ),
            _chain_profile(
                "C", ["c"], (0.98, 0.02), 4.0, 5.0
            ),
        ],
        config=config,
    )
    (
        selected,
        expanded,
        _identity_assignments,
        _actor_assignments,
        _tracklet_assignments,
        rejections,
    ) = solve_seeded_expansion(
        identity_rows=identities,
        persistent_actor_assignments=[
            _actor_assignment("A", "I1"),
            _actor_assignment("B", "I2"),
            _actor_assignment("C", "I3"),
        ],
        persistent_tracklet_assignments=[
            _tracklet_assignment("a1", "I1"),
            _tracklet_assignment("a2", "I1"),
            _tracklet_assignment("b", "I2"),
            _tracklet_assignment("c", "I3"),
        ],
        profiles=profiles,
        candidates=[
            _candidate("I1", "I2", score=0.90),
            _candidate("I2", "I3", score=0.85),
        ],
        hard_identity_pairs={("I1", "I3")},
        config=config,
    )
    assert len(selected) == 1
    assert len(expanded) == 2
    assert rejections["component_hard_conflict"] == 1


def test_different_kit_is_soft_policy_rejection_not_hard_pair():
    identities = [
        _identity(
            "I1", ["A"], ["a1", "a2"], 0.0, 1.0
        ),
        _identity("I2", ["B"], ["b"], 2.0, 3.0),
    ]
    config = IdentityExpansionConfig(
        min_seed_profiled_tracklets=2
    )
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile(
                "A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0,
                kit=0,
            ),
            _chain_profile(
                "B", ["b"], (0.99, 0.01), 2.0, 3.0,
                kit=1,
            ),
        ],
        config=config,
    )
    (
        selected,
        expanded,
        *_middle,
        rejections,
    ) = solve_seeded_expansion(
        identity_rows=identities,
        persistent_actor_assignments=[
            _actor_assignment("A", "I1"),
            _actor_assignment("B", "I2"),
        ],
        persistent_tracklet_assignments=[
            _tracklet_assignment("a1", "I1"),
            _tracklet_assignment("a2", "I1"),
            _tracklet_assignment("b", "I2"),
        ],
        profiles=profiles,
        candidates=[
            _candidate(
                "I1",
                "I2",
                score=0.9,
                kit="different_kit",
            )
        ],
        hard_identity_pairs=set(),
        config=config,
    )
    assert selected == ()
    assert len(expanded) == 2
    assert rejections["different_kit"] == 1


def test_hard_identity_pairs_are_deduplicated():
    rows = [
        _tracklet_assignment("a1", "I1"),
        _tracklet_assignment("a2", "I1"),
        _tracklet_assignment("b1", "I2"),
    ]
    hard = build_hard_identity_pairs(
        tracklet_assignment_rows=rows,
        hard_conflicts=[
            {"tracklet_a": "a1", "tracklet_b": "b1"},
            {"tracklet_a": "a2", "tracklet_b": "b1"},
        ],
    )
    assert hard == {("I1", "I2")}


def test_single_chain_identity_preserves_chain_reliability_scale():
    identities = [
        _identity("I1", ["A"], ["a1"], 0.0, 1.0),
    ]
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile(
                "A",
                ["a1"],
                (1.0, 0.0),
                0.0,
                1.0,
                reliability=0.73,
            ),
        ],
        config=IdentityExpansionConfig(
            min_seed_profiled_tracklets=2
        ),
    )

    assert len(profiles) == 1
    assert abs(profiles[0].reliability - 0.73) < 1e-12


def test_multi_chain_identity_uses_weighted_chain_reliability_without_second_penalty():
    identities = [
        _identity(
            "I1",
            ["A", "B"],
            ["a1", "a2", "b1"],
            0.0,
            3.0,
        ),
    ]
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile(
                "A",
                ["a1", "a2"],
                (1.0, 0.0),
                0.0,
                1.0,
                reliability=0.80,
            ),
            _chain_profile(
                "B",
                ["b1"],
                (0.99, 0.01),
                2.0,
                3.0,
                reliability=0.60,
            ),
        ],
        config=IdentityExpansionConfig(
            min_seed_profiled_tracklets=2
        ),
    )

    assert len(profiles) == 1
    assert 0.60 <= profiles[0].reliability <= 0.80
    assert profiles[0].initial_seed is True



def _family_row(identity_id, family_id, *, confidence=0.9, dominance=0.95):
    return {
        "identity_id": identity_id,
        "state": "confident_primary",
        "family_type": "primary",
        "family_id": family_id,
        "confidence": confidence,
        "dominance": dominance,
        "evidence_coverage": 0.8,
        "usable_for_tracking": True,
    }


def test_family_score_adjustment_is_asymmetric_by_design():
    config = IdentityExpansionConfig(
        same_primary_bonus=0.035,
        different_primary_penalty=0.150,
    )
    same = family_score_adjustment(
        relation="same_primary",
        pair_strength=0.8,
        config=config,
    )
    different = family_score_adjustment(
        relation="different_primary",
        pair_strength=0.8,
        config=config,
    )
    neutral = family_score_adjustment(
        relation="neutral",
        pair_strength=1.0,
        config=config,
    )

    assert 0.0 < same < abs(different)
    assert different < 0.0
    assert neutral == 0.0


def test_family_aware_candidate_same_primary_gets_small_bonus():
    identities = [
        _identity("I1", ["A"], ["a1", "a2"], 0.0, 1.0),
        _identity("I2", ["B"], ["b1"], 2.0, 3.0),
    ]
    config = IdentityExpansionConfig(min_seed_profiled_tracklets=2)
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile("A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0),
            _chain_profile("B", ["b1"], (0.995, 0.005), 2.0, 3.0),
        ],
        config=config,
    )
    family = build_identity_family_evidence([
        _family_row("I1", 0),
        _family_row("I2", 0),
    ])
    relation, strength = family_pair_relation("I1", "I2", family)
    assert relation == "same_primary"
    assert strength > 0.0

    calibration = IdentityCalibration(
        positive=(0.80, 0.90, 0.96),
        same_kit_negative=(0.30, 0.50, 0.75),
        different_kit_negative=(0.20, 0.40, 0.60),
        unknown_kit_negative=(0.25, 0.45, 0.65),
    )
    candidates = build_expansion_candidates(
        profiles=profiles,
        calibration=calibration,
        hard_identity_pairs=set(),
        config=config,
        identity_family_evidence=family,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.family_relation == "same_primary"
    assert candidate.family_adjustment > 0.0
    assert candidate.rank_score > candidate.base_rank_score
    assert candidate.family_evidence_source == "entity_consensus"


def test_family_aware_candidate_different_primary_is_softly_penalized():
    identities = [
        _identity("I1", ["A"], ["a1", "a2"], 0.0, 1.0),
        _identity("I2", ["B"], ["b1"], 2.0, 3.0),
    ]
    config = IdentityExpansionConfig(min_seed_profiled_tracklets=2)
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile("A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0),
            _chain_profile("B", ["b1"], (0.995, 0.005), 2.0, 3.0),
        ],
        config=config,
    )
    family = build_identity_family_evidence([
        _family_row("I1", 0),
        _family_row("I2", 1),
    ])
    calibration = IdentityCalibration(
        positive=(0.80, 0.90, 0.96),
        same_kit_negative=(0.30, 0.50, 0.75),
        different_kit_negative=(0.20, 0.40, 0.60),
        unknown_kit_negative=(0.25, 0.45, 0.65),
    )
    candidates = build_expansion_candidates(
        profiles=profiles,
        calibration=calibration,
        hard_identity_pairs=set(),
        config=config,
        identity_family_evidence=family,
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.family_relation == "different_primary"
    assert candidate.family_adjustment < 0.0
    assert candidate.rank_score < candidate.base_rank_score
    # It remains a candidate: family mismatch is not a cannot-link.
    assert candidate.family_evidence_source == "entity_consensus"


def test_family_aware_different_primary_is_not_legacy_policy_rejection():
    identities = [
        _identity("I1", ["A"], ["a1", "a2"], 0.0, 1.0),
        _identity("I2", ["B"], ["b1"], 2.0, 3.0),
    ]
    config = IdentityExpansionConfig(
        min_seed_profiled_tracklets=2,
        min_rank_score=0.60,
        allow_different_kit=False,
    )
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile("A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0),
            _chain_profile("B", ["b1"], (0.99, 0.01), 2.0, 3.0),
        ],
        config=config,
    )
    candidate = _candidate(
        "I1",
        "I2",
        score=0.80,
        kit="different_kit",
    )
    candidate = ExpansionCandidate(
        predecessor_identity_id=candidate.predecessor_identity_id,
        successor_identity_id=candidate.successor_identity_id,
        gap_s=candidate.gap_s,
        aggregate_cosine=candidate.aggregate_cosine,
        kit_family="different_kit",
        empirical_support=candidate.empirical_support,
        profile_confidence=candidate.profile_confidence,
        forward_retrieval_rank=1,
        backward_retrieval_rank=1,
        retrieval_support=1.0,
        base_rank_score=0.90,
        family_relation="different_primary",
        family_pair_strength=0.8,
        family_adjustment=-0.10,
        family_evidence_source="entity_consensus",
        rank_score=0.80,
        outgoing_score_rank=1,
        incoming_score_rank=1,
        mutual_best=True,
        mutual_top3=True,
    )

    selected, expanded, *_middle, rejections = solve_seeded_expansion(
        identity_rows=identities,
        persistent_actor_assignments=[
            _actor_assignment("A", "I1"),
            _actor_assignment("B", "I2"),
        ],
        persistent_tracklet_assignments=[
            _tracklet_assignment("a1", "I1"),
            _tracklet_assignment("a2", "I1"),
            _tracklet_assignment("b1", "I2"),
        ],
        profiles=profiles,
        candidates=[candidate],
        hard_identity_pairs=set(),
        config=config,
    )

    assert len(selected) == 1
    assert len(expanded) == 1
    assert rejections.get("different_kit", 0) == 0


def _weak_family_row(
    identity_id,
    family_id,
    *,
    confidence=0.35,
    dominance=0.72,
    coverage=0.35,
):
    return {
        "identity_id": identity_id,
        "state": "weak_primary",
        "family_type": "unknown",
        "family_id": None,
        "candidate_family_type": "primary",
        "candidate_family_id": family_id,
        "confidence": confidence,
        "dominance": dominance,
        "evidence_coverage": coverage,
        "usable_for_tracking": False,
        "family_distribution": {
            f"primary-{family_id}": dominance,
            f"primary-{1 - family_id}": 1.0 - dominance,
        },
    }


def test_weak_opposite_primary_is_negative_only_guard():
    config = IdentityExpansionConfig(
        probable_primary_min_dominance=0.60,
        probable_primary_min_confidence=0.20,
        probable_primary_min_evidence_coverage=0.15,
    )
    family = build_identity_family_evidence([
        _weak_family_row("I1", 0),
        _weak_family_row("I2", 1),
    ])

    relation, strength = family_pair_relation(
        "I1",
        "I2",
        family,
        config=config,
    )

    assert relation == "probable_different_primary"
    assert strength > 0.0
    assert family_score_adjustment(
        relation=relation,
        pair_strength=strength,
        config=config,
    ) < 0.0


def test_weak_same_primary_never_gets_positive_bonus():
    config = IdentityExpansionConfig()
    family = build_identity_family_evidence([
        _weak_family_row("I1", 0),
        _weak_family_row("I2", 0),
    ])

    relation, strength = family_pair_relation(
        "I1",
        "I2",
        family,
        config=config,
    )

    assert relation == "neutral"
    assert strength == 0.0


def test_low_coverage_opposite_primary_stays_neutral():
    config = IdentityExpansionConfig(
        probable_primary_min_evidence_coverage=0.15,
    )
    family = build_identity_family_evidence([
        _weak_family_row("I1", 0, coverage=0.05),
        _weak_family_row("I2", 1, coverage=0.05),
    ])

    relation, strength = family_pair_relation(
        "I1",
        "I2",
        family,
        config=config,
    )

    assert relation == "neutral"
    assert strength == 0.0


def test_probable_different_primary_candidate_is_penalized_below_base():
    identities = [
        _identity("I1", ["A"], ["a1", "a2"], 0.0, 1.0),
        _identity("I2", ["B"], ["b1"], 2.0, 3.0),
    ]
    config = IdentityExpansionConfig(
        min_seed_profiled_tracklets=2,
        min_rank_score=0.60,
        probable_different_primary_penalty=0.20,
    )
    profiles = build_identity_profiles(
        identity_rows=identities,
        chain_profile_rows=[
            _chain_profile("A", ["a1", "a2"], (1.0, 0.0), 0.0, 1.0),
            _chain_profile("B", ["b1"], (0.995, 0.005), 2.0, 3.0),
        ],
        config=config,
    )
    family = build_identity_family_evidence([
        _weak_family_row("I1", 0),
        _weak_family_row("I2", 1),
    ])
    calibration = IdentityCalibration(
        positive=(0.80, 0.90, 0.96),
        same_kit_negative=(0.30, 0.50, 0.75),
        different_kit_negative=(0.20, 0.40, 0.60),
        unknown_kit_negative=(0.25, 0.45, 0.65),
    )

    candidates = build_expansion_candidates(
        profiles=profiles,
        calibration=calibration,
        hard_identity_pairs=set(),
        config=config,
        identity_family_evidence=family,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.family_relation == "probable_different_primary"
    assert candidate.kit_family == "different_kit"
    assert candidate.family_adjustment < 0.0
    assert candidate.rank_score < candidate.base_rank_score
