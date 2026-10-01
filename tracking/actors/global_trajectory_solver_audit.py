"""Tracking V1 Hito 7G.4A: visual-risk audit for trajectory proposals.

Diagnostic only.  The solver proposals remain uncommitted identity hypotheses.

The audit normally inspects every selected proposal because 7G.4 intentionally
produces a small conservative set.  Rows are ordered by audit risk so the most
uncertain cases appear first on the rendered pages.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence


GLOBAL_TRAJECTORY_SOLVER_AUDIT_VERSION = (
    "global-trajectory-proposal-audit-v1"
)


def _as_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def proposal_identity_ids(
    row: Mapping[str, Any],
) -> tuple[str, str]:
    candidate_type = str(row["candidate_type"])
    if candidate_type == "append":
        return (
            str(row["predecessor_identity_id"]),
            str(row["successor_identity_id"]),
        )
    if candidate_type == "insert":
        return (
            str(row["host_identity_id"]),
            str(row["inserted_identity_id"]),
        )
    raise ValueError(
        f"unsupported candidate_type {candidate_type!r}"
    )


def proposal_gap_s(row: Mapping[str, Any]) -> float:
    if str(row["candidate_type"]) == "append":
        return float(row.get("gap_s", 0.0))
    return (
        float(row.get("left_gap_s", 0.0))
        + float(row.get("right_gap_s", 0.0))
    )


def enrich_proposal_audit_rows(
    proposals: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    enriched: list[dict[str, Any]] = []

    for source in proposals:
        row = dict(source)
        score = _as_float(row, "trajectory_score")
        confidence = _as_float(row, "profile_confidence")
        geometry = _as_float(row, "geometry_support")
        appearance = _as_float(row, "appearance_score")
        gap = proposal_gap_s(row)
        candidate_type = str(row["candidate_type"])
        relation = str(row.get("family_relation") or "neutral")

        reasons: list[str] = []
        if relation == "neutral":
            reasons.append("neutral_family")
        if score is not None and score < 0.715:
            reasons.append("near_score_floor")
        if confidence is not None and confidence < 0.20:
            reasons.append("low_profile_confidence")
        if appearance is not None and appearance < 0.55:
            reasons.append("weak_appearance_score")
        if geometry is not None and geometry < 0.90:
            reasons.append("weaker_geometry")
        if candidate_type == "append":
            if gap >= 5.0:
                reasons.append("long_append_gap")
            if gap >= 10.0:
                reasons.append("very_long_append_gap")
        else:
            if gap >= 15.0:
                reasons.append("long_insert_bridge")
            if gap >= 30.0:
                reasons.append("very_long_insert_bridge")

        first, second = proposal_identity_ids(row)
        row.update({
            "audit_version": GLOBAL_TRAJECTORY_SOLVER_AUDIT_VERSION,
            "first_identity_id": first,
            "second_identity_id": second,
            "audit_gap_s": gap,
            "risk_reasons": reasons,
        })
        enriched.append(row)

    enriched.sort(
        key=lambda row: (
            0 if str(row.get("family_relation") or "neutral") == "neutral"
            else 1,
            float(row.get("trajectory_score", 1.0)),
            float(row.get("profile_confidence") or 1.0),
            -proposal_gap_s(row),
            float(row.get("geometry_support") or 1.0),
            int(row.get("proposal_rank", 10**9)),
            str(row["first_identity_id"]),
            str(row["second_identity_id"]),
        )
    )

    for rank, row in enumerate(enriched, start=1):
        row["audit_rank"] = rank

    return tuple(enriched)


def summarize_proposal_audit(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "audit_version": GLOBAL_TRAJECTORY_SOLVER_AUDIT_VERSION,
        "audited_proposal_count": len(rows),
        "candidate_type_counts": dict(sorted(Counter(
            str(row["candidate_type"]) for row in rows
        ).items())),
        "family_relation_counts": dict(sorted(Counter(
            str(row.get("family_relation") or "neutral")
            for row in rows
        ).items())),
        "risk_reason_counts": dict(sorted(Counter(
            reason
            for row in rows
            for reason in row.get("risk_reasons", ())
        ).items())),
        "identity_decisions_changed": False,
        "audit_is_diagnostic_only": True,
    }
