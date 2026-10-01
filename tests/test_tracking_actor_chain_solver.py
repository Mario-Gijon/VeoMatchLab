from __future__ import annotations

from tracking.actors.chain_solver import (
    PersistentChainSolverConfig,
    solve_persistent_chain_paths,
)


def _actor(actor_id, tracklets, start, end):
    return {
        "actor_id": actor_id,
        "tracklet_ids": list(tracklets),
        "start_time_s": start,
        "end_time_s": end,
    }


def _assignment(tracklet_id, actor_id):
    return {
        "tracklet_id": tracklet_id,
        "actor_id": actor_id,
    }


def _candidate(
    predecessor,
    successor,
    *,
    score=0.8,
    confidence=0.9,
    gap=1.0,
    kit="same_kit",
    mutual_best=True,
    mutual_top3=True,
):
    return {
        "predecessor_actor_id": predecessor,
        "successor_actor_id": successor,
        "rank_score": score,
        "aggregate_cosine": 0.97,
        "profile_confidence": confidence,
        "gap_s": gap,
        "kit_family": kit,
        "mutual_best": mutual_best,
        "mutual_top3": mutual_top3,
    }


def _base_three():
    actors = [
        _actor("A", ["a"], 0.0, 1.0),
        _actor("B", ["b"], 2.0, 3.0),
        _actor("C", ["c"], 4.0, 5.0),
    ]
    assignments = [
        _assignment("a", "A"),
        _assignment("b", "B"),
        _assignment("c", "C"),
    ]
    return actors, assignments


def test_builds_persistent_path_longer_than_two():
    actors, assignments = _base_three()
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate("A", "B", score=0.82),
            _candidate("B", "C", score=0.80),
        ],
        hard_conflicts=[],
    )

    assert len(result.identities) == 1
    assert result.identities[0].actor_ids == ("A", "B", "C")
    assert result.identities[0].tracklet_ids == ("a", "b", "c")


def test_transitive_hard_conflict_blocks_chain_growth():
    actors, assignments = _base_three()
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate("A", "B", score=0.82),
            _candidate("B", "C", score=0.80),
        ],
        hard_conflicts=[
            {"tracklet_a": "a", "tracklet_b": "c"},
        ],
    )

    assert len(result.selected_links) == 1
    assert (
        result.rejection_counts["component_hard_conflict"]
        == 1
    )


def test_one_successor_and_one_predecessor_are_enforced():
    actors = [
        _actor("A", ["a"], 0.0, 1.0),
        _actor("B", ["b"], 2.0, 3.0),
        _actor("C", ["c"], 2.1, 3.1),
        _actor("D", ["d"], 4.0, 5.0),
    ]
    assignments = [
        _assignment("a", "A"),
        _assignment("b", "B"),
        _assignment("c", "C"),
        _assignment("d", "D"),
    ]
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate("A", "B", score=0.90),
            _candidate("A", "C", score=0.80),
            _candidate("B", "D", score=0.85),
            _candidate("C", "D", score=0.75),
        ],
        hard_conflicts=[],
    )

    selected = {
        (
            row.predecessor_actor_id,
            row.successor_actor_id,
        )
        for row in result.selected_links
    }
    assert ("A", "B") in selected
    assert ("A", "C") not in selected
    assert ("B", "D") in selected


def test_different_kit_is_rejected_by_default():
    actors = [
        _actor("A", ["a"], 0.0, 1.0),
        _actor("B", ["b"], 2.0, 3.0),
    ]
    assignments = [
        _assignment("a", "A"),
        _assignment("b", "B"),
    ]
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate(
                "A",
                "B",
                score=0.95,
                kit="different_kit",
            )
        ],
        hard_conflicts=[],
    )

    assert len(result.identities) == 2
    assert result.selected_links == ()
    assert result.rejection_counts["different_kit"] == 1


def test_unknown_kit_is_allowed():
    actors = [
        _actor("A", ["a"], 0.0, 1.0),
        _actor("B", ["b"], 2.0, 3.0),
    ]
    assignments = [
        _assignment("a", "A"),
        _assignment("b", "B"),
    ]
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate(
                "A",
                "B",
                score=0.80,
                kit="unknown_kit",
            )
        ],
        hard_conflicts=[],
    )

    assert len(result.identities) == 1


def test_mutual_best_required_by_default():
    actors = [
        _actor("A", ["a"], 0.0, 1.0),
        _actor("B", ["b"], 2.0, 3.0),
    ]
    assignments = [
        _assignment("a", "A"),
        _assignment("b", "B"),
    ]
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate(
                "A",
                "B",
                score=0.90,
                mutual_best=False,
                mutual_top3=True,
            )
        ],
        hard_conflicts=[],
    )

    assert len(result.identities) == 2
    assert result.rejection_counts["competition_filter"] == 1


def test_mutual_top3_can_be_enabled_explicitly():
    actors = [
        _actor("A", ["a"], 0.0, 1.0),
        _actor("B", ["b"], 2.0, 3.0),
    ]
    assignments = [
        _assignment("a", "A"),
        _assignment("b", "B"),
    ]
    result = solve_persistent_chain_paths(
        actor_rows=actors,
        backbone_assignments=assignments,
        chain_candidates=[
            _candidate(
                "A",
                "B",
                score=0.90,
                mutual_best=False,
                mutual_top3=True,
            )
        ],
        hard_conflicts=[],
        config=PersistentChainSolverConfig(
            min_rank_score=0.70,
            competition_mode="mutual_top3",
        ),
    )

    assert len(result.identities) == 1
