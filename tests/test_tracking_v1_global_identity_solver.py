from __future__ import annotations

import pytest

from tracking.identities.global_solver import (
    BackboneAugmentationConfig,
    build_solver_input,
    load_tracklet_windows,
    solve_global_backbone_partition,
    solve_global_partition,
    utility_from_scored_link,
    validate_backbone_partition,
    validate_partition,
)


def _candidate(a, b, status):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": status,
    }


def _score(a, b, score, confidence=1.0, kind="continuation"):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "rank_score": score,
        "evidence_confidence": confidence,
        "link_kind": kind,
        "identity_decision": None,
    }


def test_confidence_shrinks_score_toward_neutral():
    high = utility_from_scored_link(
        _score("a", "b", 1.0, confidence=1.0)
    )
    low = utility_from_scored_link(
        _score("a", "b", 1.0, confidence=0.2)
    )

    assert high.effective_affinity == pytest.approx(1.0)
    assert high.utility == pytest.approx(1.0)
    assert low.effective_affinity == pytest.approx(0.6)
    assert low.utility == pytest.approx(0.2)


def test_solver_respects_hard_cannot_link_despite_two_good_edges():
    candidate_edges = [
        _candidate("a", "b", "candidate"),
        _candidate("a", "c", "cannot_link"),
        _candidate("b", "c", "candidate"),
    ]
    scored = [
        _score("a", "b", 0.95, 1.0),
        _score("b", "c", 0.94, 1.0),
    ]

    solver_input = build_solver_input(
        candidate_edges,
        scored,
    )
    result, _ = solve_global_partition(solver_input)
    assignment = result.assignment()

    assert assignment["a"] != assignment["c"]
    assert result.identity_count == 2
    validation = validate_partition(
        result,
        solver_input,
    )
    assert validation["hard_constraint_violation_count"] == 0


def test_global_objective_avoids_bad_transitive_cluster():
    # a-b and b-c look attractive individually, but a-c is allowed and has
    # strong negative evidence.  A simple connected-components linker would
    # merge all three.  The partition objective should keep one pair apart.
    candidate_edges = [
        _candidate("a", "b", "candidate"),
        _candidate("a", "c", "candidate"),
        _candidate("b", "c", "candidate"),
    ]
    scored = [
        _score("a", "b", 0.95, 1.0),
        _score("b", "c", 0.90, 1.0),
        _score("a", "c", 0.00, 1.0),
    ]

    solver_input = build_solver_input(
        candidate_edges,
        scored,
    )
    result, _ = solve_global_partition(solver_input)

    assert result.identity_count == 2
    assert len(max(result.clusters, key=len)) == 2


def test_low_confidence_extreme_does_not_dominate():
    candidate_edges = [
        _candidate("a", "b", "candidate"),
        _candidate("a", "c", "candidate"),
        _candidate("b", "c", "candidate"),
    ]
    scored = [
        _score("a", "b", 0.95, 0.9),
        _score("b", "c", 0.95, 0.9),
        # extreme negative but almost no evidence confidence
        _score("a", "c", 0.0, 0.01),
    ]

    solver_input = build_solver_input(
        candidate_edges,
        scored,
    )
    result, _ = solve_global_partition(solver_input)

    assert result.identity_count == 1


def test_solver_does_not_require_predefined_identity_count():
    candidate_edges = []
    scored = []
    nodes = ["a", "b", "c", "d"]
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            candidate_edges.append(
                _candidate(a, b, "candidate")
            )
            score = 0.95 if {a, b} in (
                {"a", "b"},
                {"c", "d"},
            ) else 0.05
            scored.append(_score(a, b, score, 1.0))

    solver_input = build_solver_input(
        candidate_edges,
        scored,
    )
    result, _ = solve_global_partition(solver_input)

    assert result.identity_count == 2
    assert {frozenset(cluster) for cluster in result.clusters} == {
        frozenset({"a", "b"}),
        frozenset({"c", "d"}),
    }


def test_equal_objective_tie_is_conservative():
    # Neutral edge has zero utility; singleton partition should survive rather
    # than merge for no evidence gain.
    candidate_edges = [
        _candidate("a", "b", "candidate"),
    ]
    scored = [
        _score("a", "b", 0.5, 1.0),
    ]

    solver_input = build_solver_input(
        candidate_edges,
        scored,
    )
    result, _ = solve_global_partition(solver_input)

    assert result.identity_count == 2



def _candidate_transition(
    a,
    b,
    *,
    pred,
    succ,
    gap=0.1,
):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": "candidate",
        "reason": "transition_reachable",
        "physics": {
            "predecessor": pred,
            "successor": succ,
            "gap_s": gap,
            "endpoint_distance_m": 1.0,
        },
    }


def _node(tracklet_id, start, end):
    return {
        "tracklet_id": tracklet_id,
        "start_time_s": start,
        "end_time_s": end,
    }


def test_backbone_repairs_strong_trajectory_vs_clique_conflict():
    # Baseline clique prefers {b,c} + {a} because the distant a-c pair is
    # negative.  The V1 backbone stage should preserve {b,c} and attach a via
    # the strong, boundary-aligned a->b handoff.
    candidates = [
        _candidate_transition(
            "a", "b", pred="a", succ="b",
        ),
        _candidate_transition(
            "a", "c", pred="a", succ="c", gap=1.1,
        ),
        _candidate_transition(
            "b", "c", pred="b", succ="c",
        ),
    ]
    scored = [
        _score("a", "b", 0.85, 1.0),
        _score("a", "c", 0.00, 1.0),
        _score("b", "c", 0.95, 1.0),
    ]
    solver_input = build_solver_input(
        candidates,
        scored,
    )
    windows = load_tracklet_windows(
        [
            _node("a", 0.0, 1.0),
            _node("b", 1.1, 2.0),
            _node("c", 2.1, 3.0),
        ]
    )

    final, baseline, selected, diagnostics, _ = (
        solve_global_backbone_partition(
            solver_input,
            candidate_edges=candidates,
            tracklet_windows=windows,
            config=BackboneAugmentationConfig(
                strong_positive_quantile=0.95,
                strong_utility_floor=0.35,
            ),
        )
    )

    assert baseline.identity_count == 2
    assert final.identity_count == 1
    assert len(selected) == 1
    assert (
        selected[0].predecessor_tracklet_id,
        selected[0].successor_tracklet_id,
    ) == ("a", "b")
    assert diagnostics["selected_backbone_link_count"] == 1

    validation = validate_backbone_partition(
        final_result=final,
        baseline_result=baseline,
        selected_links=selected,
        solver_input=solver_input,
    )
    assert validation["hard_constraint_violation_count"] == 0
    assert validation["baseline_cluster_preserved"] is True


def test_backbone_does_not_merge_weak_fragment():
    # b-c is a strong baseline identity. a has one weakly positive boundary
    # link to b, but a-c is strongly negative, so clique keeps a separate and
    # the backbone floor must NOT force the weak repair.
    candidates = [
        _candidate_transition(
            "a", "b", pred="a", succ="b",
        ),
        _candidate_transition(
            "a", "c", pred="a", succ="c", gap=1.1,
        ),
        _candidate_transition(
            "b", "c", pred="b", succ="c",
        ),
    ]
    scored = [
        _score("a", "b", 0.55, 0.5),
        _score("a", "c", 0.00, 1.0),
        _score("b", "c", 0.95, 1.0),
    ]
    solver_input = build_solver_input(
        candidates,
        scored,
    )
    windows = load_tracklet_windows(
        [
            _node("a", 0.0, 1.0),
            _node("b", 1.1, 2.0),
            _node("c", 2.1, 3.0),
        ]
    )

    final, baseline, selected, _, _ = (
        solve_global_backbone_partition(
            solver_input,
            candidate_edges=candidates,
            tracklet_windows=windows,
        )
    )
    assert baseline.identity_count == 2
    assert final.identity_count == 2
    assert selected == ()


def test_backbone_never_overrides_hard_conflict_between_clusters():
    candidates = [
        _candidate_transition(
            "a", "b", pred="a", succ="b",
        ),
        _candidate("a", "c", "cannot_link"),
        _candidate_transition(
            "b", "c", pred="b", succ="c",
        ),
    ]
    scored = [
        _score("a", "b", 0.99, 1.0),
        _score("b", "c", 0.95, 1.0),
    ]
    solver_input = build_solver_input(
        candidates,
        scored,
    )
    windows = load_tracklet_windows(
        [
            _node("a", 0.0, 1.0),
            _node("b", 1.1, 2.0),
            _node("c", 2.1, 3.0),
        ]
    )

    final, baseline, selected, _, _ = (
        solve_global_backbone_partition(
            solver_input,
            candidate_edges=candidates,
            tracklet_windows=windows,
        )
    )
    validation = validate_backbone_partition(
        final_result=final,
        baseline_result=baseline,
        selected_links=selected,
        solver_input=solver_input,
    )
    assert validation["hard_constraint_violation_count"] == 0


def test_strong_non_boundary_link_is_not_used_for_augmentation():
    # a->c is very strong, but a is not the tail of its baseline identity
    # because d continues later.
    candidates = [
        _candidate_transition(
            "a", "b", pred="a", succ="b",
        ),
        _candidate_transition(
            "a", "c", pred="a", succ="c",
        ),
        _candidate_transition(
            "a", "d", pred="a", succ="d",
        ),
        _candidate_transition(
            "b", "c", pred="b", succ="c",
        ),
        _candidate_transition(
            "b", "d", pred="b", succ="d",
        ),
        _candidate_transition(
            "c", "d", pred="c", succ="d",
        ),
    ]
    scored = [
        _score("a", "d", 0.95, 1.0),
        _score("a", "c", 0.99, 1.0),
        _score("a", "b", 0.05, 1.0),
        _score("b", "c", 0.05, 1.0),
        _score("b", "d", 0.05, 1.0),
        _score("c", "d", 0.05, 1.0),
    ]
    solver_input = build_solver_input(
        candidates,
        scored,
    )
    windows = load_tracklet_windows(
        [
            _node("a", 0.0, 1.0),
            _node("d", 1.1, 4.0),
            _node("c", 1.1, 2.0),
            _node("b", 5.0, 6.0),
        ]
    )

    final, _, selected, _, _ = (
        solve_global_backbone_partition(
            solver_input,
            candidate_edges=candidates,
            tracklet_windows=windows,
            config=BackboneAugmentationConfig(
                strong_positive_quantile=0.90,
                strong_utility_floor=0.35,
                boundary_tolerance_s=0.25,
            ),
        )
    )
    assert not any(
        link.predecessor_tracklet_id == "a"
        and link.successor_tracklet_id == "c"
        for link in selected
    )
