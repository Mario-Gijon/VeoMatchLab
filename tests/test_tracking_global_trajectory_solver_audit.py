from tracking.actors.global_trajectory_solver_audit import (
    enrich_proposal_audit_rows,
    proposal_gap_s,
    proposal_identity_ids,
    summarize_proposal_audit,
)


def _append(
    a="A",
    b="B",
    score=0.705,
    family="neutral",
    confidence=0.15,
    gap=12.0,
):
    return {
        "candidate_type": "append",
        "predecessor_identity_id": a,
        "successor_identity_id": b,
        "proposal_rank": 1,
        "trajectory_score": score,
        "appearance_score": 0.52,
        "aggregate_cosine": 0.94,
        "profile_confidence": confidence,
        "geometry_support": 0.88,
        "family_relation": family,
        "gap_s": gap,
    }


def _insert(
    host="H",
    inserted="I",
    score=0.72,
    family="same_primary",
    left_gap=10.0,
    right_gap=25.0,
):
    return {
        "candidate_type": "insert",
        "host_identity_id": host,
        "inserted_identity_id": inserted,
        "host_gap_index": 1,
        "proposal_rank": 2,
        "trajectory_score": score,
        "appearance_score": 0.60,
        "aggregate_cosine": 0.95,
        "profile_confidence": 0.3,
        "geometry_support": 0.95,
        "family_relation": family,
        "left_gap_s": left_gap,
        "right_gap_s": right_gap,
    }


def test_identity_ids_for_both_candidate_types():
    assert proposal_identity_ids(_append()) == ("A", "B")
    assert proposal_identity_ids(_insert()) == ("H", "I")


def test_gap_semantics_differ_between_append_and_insert():
    assert proposal_gap_s(_append(gap=4.0)) == 4.0
    assert proposal_gap_s(
        _insert(left_gap=4.0, right_gap=6.0)
    ) == 10.0


def test_risk_reasons_capture_neutral_low_conf_long_append():
    row = enrich_proposal_audit_rows([_append()])[0]
    reasons = set(row["risk_reasons"])
    assert "neutral_family" in reasons
    assert "near_score_floor" in reasons
    assert "low_profile_confidence" in reasons
    assert "long_append_gap" in reasons
    assert "very_long_append_gap" in reasons


def test_long_insert_bridge_is_flagged():
    row = enrich_proposal_audit_rows([_insert()])[0]
    reasons = set(row["risk_reasons"])
    assert "long_insert_bridge" in reasons
    assert "very_long_insert_bridge" in reasons


def test_rows_are_sorted_riskiest_first():
    risky = _append(a="R1", b="R2")
    safe = _append(
        a="S1",
        b="S2",
        score=0.74,
        family="same_primary",
        confidence=0.4,
        gap=0.2,
    )
    rows = enrich_proposal_audit_rows([safe, risky])
    assert rows[0]["first_identity_id"] == "R1"
    assert rows[0]["audit_rank"] == 1


def test_summary_is_diagnostic_only():
    rows = enrich_proposal_audit_rows([_append(), _insert()])
    summary = summarize_proposal_audit(rows)
    assert summary["audited_proposal_count"] == 2
    assert summary["identity_decisions_changed"] is False
    assert summary["audit_is_diagnostic_only"] is True
