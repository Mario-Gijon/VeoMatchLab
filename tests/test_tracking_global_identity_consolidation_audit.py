from tracking.actors.global_identity_consolidation_audit import (
    enrich_risk_rows,
    select_risk_audit_rows,
    summarize_risk_audit,
)


def _link(
    round_index,
    left,
    right,
    score,
    *,
    relation="same_primary",
    gap=1.0,
    confidence=0.5,
    left_initial=(),
    right_initial=(),
):
    return {
        "round_index": round_index,
        "predecessor_identity_id": left,
        "successor_identity_id": right,
        "rank_score": score,
        "gap_s": gap,
        "aggregate_cosine": 0.96,
        "profile_confidence": confidence,
        "family_relation": relation,
        "predecessor_initial_identity_ids": list(left_initial),
        "successor_initial_identity_ids": list(right_initial),
    }


def test_round_one_is_excluded_by_default():
    rows = enrich_risk_rows(
        selected_links=[
            _link(1, "A", "B", 0.61),
            _link(2, "C", "D", 0.62),
        ],
        final_lineage_rows=[],
    )
    assert len(rows) == 1
    assert rows[0]["round_index"] == 2


def test_final_component_metadata_is_attached():
    rows = enrich_risk_rows(
        selected_links=[
            _link(
                3,
                "A",
                "B",
                0.61,
                left_initial=("I1", "I2"),
                right_initial=("I3",),
            )
        ],
        final_lineage_rows=[
            {
                "identity_id": "F1",
                "initial_identity_ids": ["I1", "I2", "I3", "I4", "I5"],
                "initial_identity_count": 5,
            }
        ],
    )
    row = rows[0]
    assert row["combined_initial_identity_count"] == 3
    assert row["final_identity_id"] == "F1"
    assert row["final_initial_identity_count"] == 5
    assert "feeds_large_final_component" in row["risk_reasons"]


def test_neutral_near_floor_long_gap_is_prioritized():
    rows = enrich_risk_rows(
        selected_links=[
            _link(
                2, "A", "B", 0.64,
                relation="same_primary", gap=1.0
            ),
            _link(
                2, "C", "D", 0.605,
                relation="neutral", gap=20.0
            ),
        ],
        final_lineage_rows=[],
    )
    assert rows[0]["predecessor_identity_id"] == "C"
    assert "neutral_family" in rows[0]["risk_reasons"]
    assert "near_score_floor" in rows[0]["risk_reasons"]
    assert "long_gap" in rows[0]["risk_reasons"]


def test_stratified_selection_keeps_each_round():
    rows = [
        _link(2, f"A{i}", f"B{i}", 0.60 + i / 1000)
        for i in range(10)
    ] + [
        _link(3, f"C{i}", f"D{i}", 0.60 + i / 1000)
        for i in range(10)
    ]
    enriched = enrich_risk_rows(
        selected_links=rows,
        final_lineage_rows=[],
    )
    selected = select_risk_audit_rows(
        enriched,
        max_links=6,
        min_per_round=2,
    )
    rounds = [row["round_index"] for row in selected]
    assert rounds.count(2) >= 2
    assert rounds.count(3) >= 2
    assert len(selected) == 6


def test_selection_has_unique_links():
    rows = enrich_risk_rows(
        selected_links=[
            _link(2, "A", "B", 0.61),
            _link(2, "C", "D", 0.62),
            _link(3, "E", "F", 0.63),
        ],
        final_lineage_rows=[],
    )
    selected = select_risk_audit_rows(
        rows,
        max_links=3,
        min_per_round=2,
    )
    keys = {
        (
            row["round_index"],
            row["predecessor_identity_id"],
            row["successor_identity_id"],
        )
        for row in selected
    }
    assert len(keys) == len(selected)


def test_summary_is_diagnostic_only():
    rows = enrich_risk_rows(
        selected_links=[
            _link(2, "A", "B", 0.61, relation="neutral"),
            _link(3, "C", "D", 0.64),
        ],
        final_lineage_rows=[],
    )
    selected = select_risk_audit_rows(
        rows,
        max_links=2,
        min_per_round=1,
    )
    summary = summarize_risk_audit(
        all_later_rows=rows,
        audited_rows=selected,
    )
    assert summary["later_round_selected_link_count"] == 2
    assert summary["audited_link_count"] == 2
    assert summary["selection_changed_identity_decisions"] is False
