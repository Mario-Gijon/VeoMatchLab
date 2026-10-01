from tracking.actors.global_trajectory_solver import (
    GlobalTrajectoryProposalSolverConfig,
    solve_global_trajectory_proposals,
)


def _append(a, b, score, *, mutual=True, family="neutral"):
    return {
        "candidate_type": "append",
        "predecessor_identity_id": a,
        "successor_identity_id": b,
        "trajectory_score": score,
        "profile_confidence": 0.7,
        "geometry_support": 0.8,
        "family_relation": family,
        "mutual_best": mutual,
        "mutual_top3": mutual,
    }


def _insert(h, i, score, *, gap=1, mutual=True, family="neutral"):
    return {
        "candidate_type": "insert",
        "host_identity_id": h,
        "inserted_identity_id": i,
        "host_gap_index": gap,
        "host_preceding_segment_index": gap,
        "host_following_segment_index": gap + 1,
        "trajectory_score": score,
        "profile_confidence": 0.8,
        "geometry_support": 0.9,
        "family_relation": family,
        "mutual_best": mutual,
        "mutual_top3": mutual,
    }


def test_floor_and_competition_filter_candidates():
    result = solve_global_trajectory_proposals(
        append_rows=[
            _append("A", "B", 0.69),
            _append("C", "D", 0.80, mutual=False),
            _append("E", "F", 0.75),
        ],
        insert_rows=[],
        hard_identity_pairs=set(),
    )
    assert len(result.selected) == 1
    assert result.selected[0].first_identity_id == "E"
    assert result.rejection_counts["score_below_floor"] == 1
    assert result.rejection_counts["competition_filter"] == 1


def test_hard_conflict_remains_inviolable():
    result = solve_global_trajectory_proposals(
        append_rows=[_append("A", "B", 0.9)],
        insert_rows=[],
        hard_identity_pairs={("A", "B")},
    )
    assert not result.selected
    assert result.rejection_counts["hard_identity_conflict"] == 1


def test_selected_proposals_are_identity_disjoint():
    result = solve_global_trajectory_proposals(
        append_rows=[
            _append("A", "B", 0.90),
            _append("A", "C", 0.85),
            _append("D", "E", 0.80),
        ],
        insert_rows=[],
        hard_identity_pairs=set(),
    )
    pairs = {
        frozenset((
            row.first_identity_id,
            row.second_identity_id,
        ))
        for row in result.selected
    }
    assert pairs == {
        frozenset(("A", "B")),
        frozenset(("D", "E")),
    }


def test_append_and_insert_compete_globally():
    result = solve_global_trajectory_proposals(
        append_rows=[_append("A", "B", 0.80)],
        insert_rows=[_insert("A", "C", 0.90)],
        hard_identity_pairs=set(),
    )
    assert len(result.selected) == 1
    assert result.selected[0].candidate_type == "insert"
    assert result.selected[0].second_identity_id == "C"


def test_different_primary_is_soft_not_hard():
    result = solve_global_trajectory_proposals(
        append_rows=[
            _append(
                "A",
                "B",
                0.80,
                family="different_primary",
            )
        ],
        insert_rows=[],
        hard_identity_pairs=set(),
    )
    assert len(result.selected) == 1


def test_missing_score_is_unknown_and_not_selected():
    row = _append("A", "B", 0.8)
    row["trajectory_score"] = None
    result = solve_global_trajectory_proposals(
        append_rows=[row],
        insert_rows=[],
        hard_identity_pairs=set(),
    )
    assert not result.selected
    assert result.rejection_counts["missing_score"] == 1


def test_mutual_top3_mode_can_be_selected():
    row = _append("A", "B", 0.75, mutual=False)
    row["mutual_top3"] = True
    result = solve_global_trajectory_proposals(
        append_rows=[row],
        insert_rows=[],
        hard_identity_pairs=set(),
        config=GlobalTrajectoryProposalSolverConfig(
            min_trajectory_score=0.70,
            competition_mode="mutual_top3",
        ),
    )
    assert len(result.selected) == 1


def test_summary_is_proposal_only():
    result = solve_global_trajectory_proposals(
        append_rows=[_append("A", "B", 0.8)],
        insert_rows=[],
        hard_identity_pairs=set(),
    )
    summary = result.summary(source_identity_count=10)
    assert summary["selected_proposal_count"] == 1
    assert summary["projected_identity_count_if_all_accepted"] == 9
    assert summary["identity_partition_changed"] is False
    assert summary["selection_is_proposal_only"] is True
