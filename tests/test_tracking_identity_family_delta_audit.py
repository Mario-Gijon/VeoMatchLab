from __future__ import annotations

from tracking.actors.identity_family_delta_audit import (
    build_delta_records,
    summarize_delta,
)


def _row(
    left,
    right,
    score,
    *,
    relation=None,
    adjustment=None,
    strength=None,
):
    row = {
        "predecessor_identity_id": left,
        "successor_identity_id": right,
        "rank_score": score,
        "gap_s": 3.0,
        "aggregate_cosine": 0.96,
        "profile_confidence": 0.5,
        "outgoing_score_rank": 1,
        "incoming_score_rank": 1,
        "mutual_best": True,
    }
    if relation is not None:
        row.update({
            "base_rank_score": score - (adjustment or 0.0),
            "family_relation": relation,
            "family_pair_strength": strength or 0.0,
            "family_adjustment": adjustment or 0.0,
        })
    return row


def test_only_new_and_lost_are_returned():
    baseline_selected = [
        _row("A", "B", 0.65),
        _row("C", "D", 0.66),
    ]
    family_selected = [
        _row("A", "B", 0.68, relation="same_primary", adjustment=0.03),
        _row("E", "F", 0.67, relation="same_primary", adjustment=0.02),
    ]

    rows = build_delta_records(
        baseline_selected_rows=baseline_selected,
        family_selected_rows=family_selected,
        baseline_candidate_rows=baseline_selected,
        family_candidate_rows=family_selected,
    )

    keys = {
        (
            row["status"],
            row["predecessor_identity_id"],
            row["successor_identity_id"],
        )
        for row in rows
    }
    assert keys == {
        ("lost", "C", "D"),
        ("new", "E", "F"),
    }


def test_lost_edge_is_enriched_from_family_candidate_table():
    baseline_selected = [_row("A", "B", 0.63)]
    family_candidate = _row(
        "A",
        "B",
        0.49,
        relation="different_primary",
        adjustment=-0.14,
        strength=0.93,
    )

    rows = build_delta_records(
        baseline_selected_rows=baseline_selected,
        family_selected_rows=[],
        baseline_candidate_rows=baseline_selected,
        family_candidate_rows=[family_candidate],
    )
    row = rows[0]

    assert row["status"] == "lost"
    assert row["family_relation"] == "different_primary"
    assert row["family_adjustment"] == -0.14
    assert row["family_aware_selected"] is False


def test_new_edge_shows_same_primary_bonus():
    family_selected = [
        _row(
            "A",
            "B",
            0.64,
            relation="same_primary",
            adjustment=0.03,
            strength=0.90,
        )
    ]
    baseline_candidate = _row("A", "B", 0.61)

    rows = build_delta_records(
        baseline_selected_rows=[],
        family_selected_rows=family_selected,
        baseline_candidate_rows=[baseline_candidate],
        family_candidate_rows=family_selected,
    )
    row = rows[0]

    assert row["status"] == "new"
    assert row["family_relation"] == "same_primary"
    assert abs(row["score_delta"] - 0.03) < 1e-9


def test_source_identity_family_evidence_is_attached():
    baseline_selected = [_row("A", "B", 0.65)]
    family_candidate = _row(
        "A",
        "B",
        0.50,
        relation="different_primary",
        adjustment=-0.15,
        strength=0.95,
    )
    family_rows = [
        {
            "identity_id": "A",
            "family_key": "primary-0",
            "candidate_family_key": "primary-0",
            "state": "confident_primary",
            "confidence": 0.92,
            "dominance": 0.98,
            "evidence_coverage": 0.70,
        },
        {
            "identity_id": "B",
            "family_key": "primary-1",
            "candidate_family_key": "primary-1",
            "state": "confident_primary",
            "confidence": 0.90,
            "dominance": 0.96,
            "evidence_coverage": 0.68,
        },
    ]

    row = build_delta_records(
        baseline_selected_rows=baseline_selected,
        family_selected_rows=[],
        baseline_candidate_rows=baseline_selected,
        family_candidate_rows=[family_candidate],
        source_family_rows=family_rows,
    )[0]

    assert row["predecessor_family"]["family_key"] == "primary-0"
    assert row["successor_family"]["family_key"] == "primary-1"


def test_common_link_is_not_audited():
    common = _row(
        "A",
        "B",
        0.70,
        relation="same_primary",
        adjustment=0.02,
    )
    rows = build_delta_records(
        baseline_selected_rows=[_row("A", "B", 0.68)],
        family_selected_rows=[common],
        baseline_candidate_rows=[_row("A", "B", 0.68)],
        family_candidate_rows=[common],
    )
    assert rows == ()


def test_summary_separates_new_and_lost_relations():
    baseline_selected = [
        _row("A", "B", 0.65),
    ]
    family_selected = [
        _row(
            "C",
            "D",
            0.65,
            relation="same_primary",
            adjustment=0.03,
        )
    ]
    family_candidates = [
        _row(
            "A",
            "B",
            0.50,
            relation="different_primary",
            adjustment=-0.15,
        ),
        family_selected[0],
    ]

    rows = build_delta_records(
        baseline_selected_rows=baseline_selected,
        family_selected_rows=family_selected,
        baseline_candidate_rows=baseline_selected,
        family_candidate_rows=family_candidates,
    )
    summary = summarize_delta(rows)

    assert summary["status_counts"] == {"lost": 1, "new": 1}
    assert summary["lost_family_relation_counts"] == {
        "different_primary": 1
    }
    assert summary["new_family_relation_counts"] == {
        "same_primary": 1
    }
    assert summary["identity_decisions_changed"] is False
