"""Hito 7E.5A: compare baseline vs family-aware identity links.

Pure diagnostic helpers for the family-aware identity-expansion audit.

The important detail is that NEW/LOST status is determined only from selected
links, but metrics are enriched from the complete candidate tables.  This lets
a LOST link still show the family relation/penalty that caused it to disappear.

No identity decisions are changed here.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


DELTA_AUDIT_VERSION = "identity-family-delta-audit-v1"


def link_key(row: Mapping[str, Any]) -> tuple[str, str]:
    return (
        str(row["predecessor_identity_id"]),
        str(row["successor_identity_id"]),
    )


def _optional_float(
    row: Mapping[str, Any] | None,
    key: str,
) -> float | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    return float(value)


def _optional_int(
    row: Mapping[str, Any] | None,
    key: str,
) -> int | None:
    if row is None:
        return None
    value = row.get(key)
    if value is None:
        return None
    return int(value)


def _family_public(
    row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if row is None:
        return {
            "family_key": None,
            "candidate_family_key": None,
            "state": None,
            "confidence": None,
            "dominance": None,
            "evidence_coverage": None,
        }

    return {
        "family_key": row.get("family_key"),
        "candidate_family_key": row.get("candidate_family_key"),
        "state": row.get("state"),
        "confidence": _optional_float(row, "confidence"),
        "dominance": _optional_float(row, "dominance"),
        "evidence_coverage": _optional_float(
            row,
            "evidence_coverage",
        ),
    }


def build_delta_records(
    *,
    baseline_selected_rows: Sequence[Mapping[str, Any]],
    family_selected_rows: Sequence[Mapping[str, Any]],
    baseline_candidate_rows: Sequence[Mapping[str, Any]],
    family_candidate_rows: Sequence[Mapping[str, Any]],
    source_family_rows: Sequence[Mapping[str, Any]] = (),
) -> tuple[dict[str, Any], ...]:
    """Return only NEW and LOST selected links, enriched with full candidates."""

    baseline_selected = {
        link_key(row): dict(row)
        for row in baseline_selected_rows
    }
    family_selected = {
        link_key(row): dict(row)
        for row in family_selected_rows
    }
    baseline_candidates = {
        link_key(row): dict(row)
        for row in baseline_candidate_rows
    }
    family_candidates = {
        link_key(row): dict(row)
        for row in family_candidate_rows
    }
    family_by_identity = {
        str(row.get("identity_id", row.get("entity_id"))): dict(row)
        for row in source_family_rows
        if row.get("identity_id", row.get("entity_id")) is not None
    }

    baseline_keys = set(baseline_selected)
    family_keys = set(family_selected)

    lost = baseline_keys - family_keys
    new = family_keys - baseline_keys

    records: list[dict[str, Any]] = []

    for status, keys in (("lost", lost), ("new", new)):
        for key in keys:
            baseline_selected_row = baseline_selected.get(key)
            family_selected_row = family_selected.get(key)
            baseline_candidate = baseline_candidates.get(
                key,
                baseline_selected_row,
            )
            family_candidate = family_candidates.get(
                key,
                family_selected_row,
            )

            # Prefer family-aware candidate values for common physical/ReID
            # fields because that is the scoring universe being audited.
            metrics_row = (
                family_candidate
                or baseline_candidate
                or {}
            )

            baseline_score = _optional_float(
                baseline_candidate,
                "rank_score",
            )
            family_score = _optional_float(
                family_candidate,
                "rank_score",
            )
            family_base_score = _optional_float(
                family_candidate,
                "base_rank_score",
            )
            if family_base_score is None:
                family_base_score = baseline_score

            predecessor = key[0]
            successor = key[1]

            records.append({
                "schema_version": 1,
                "audit_version": DELTA_AUDIT_VERSION,
                "status": status,
                "predecessor_identity_id": predecessor,
                "successor_identity_id": successor,
                "baseline_selected": baseline_selected_row is not None,
                "family_aware_selected": family_selected_row is not None,
                "baseline_rank_score": baseline_score,
                "family_base_rank_score": family_base_score,
                "family_aware_rank_score": family_score,
                "score_delta": (
                    None
                    if baseline_score is None or family_score is None
                    else family_score - baseline_score
                ),
                "gap_s": _optional_float(metrics_row, "gap_s"),
                "aggregate_cosine": _optional_float(
                    metrics_row,
                    "aggregate_cosine",
                ),
                "profile_confidence": _optional_float(
                    metrics_row,
                    "profile_confidence",
                ),
                "kit_family": metrics_row.get("kit_family"),
                "family_relation": (
                    None
                    if family_candidate is None
                    else family_candidate.get(
                        "family_relation",
                        "neutral",
                    )
                ),
                "family_pair_strength": _optional_float(
                    family_candidate,
                    "family_pair_strength",
                ),
                "family_adjustment": _optional_float(
                    family_candidate,
                    "family_adjustment",
                ),
                "family_evidence_source": (
                    None
                    if family_candidate is None
                    else family_candidate.get(
                        "family_evidence_source"
                    )
                ),
                "baseline_outgoing_score_rank": _optional_int(
                    baseline_candidate,
                    "outgoing_score_rank",
                ),
                "baseline_incoming_score_rank": _optional_int(
                    baseline_candidate,
                    "incoming_score_rank",
                ),
                "family_outgoing_score_rank": _optional_int(
                    family_candidate,
                    "outgoing_score_rank",
                ),
                "family_incoming_score_rank": _optional_int(
                    family_candidate,
                    "incoming_score_rank",
                ),
                "baseline_mutual_best": (
                    None
                    if baseline_candidate is None
                    else bool(
                        baseline_candidate.get(
                            "mutual_best",
                            False,
                        )
                    )
                ),
                "family_mutual_best": (
                    None
                    if family_candidate is None
                    else bool(
                        family_candidate.get(
                            "mutual_best",
                            False,
                        )
                    )
                ),
                "predecessor_family": _family_public(
                    family_by_identity.get(predecessor)
                ),
                "successor_family": _family_public(
                    family_by_identity.get(successor)
                ),
            })

    relation_priority = {
        "different_primary": 0,
        "same_primary": 1,
        "neutral": 2,
        None: 3,
    }

    records.sort(
        key=lambda row: (
            0 if row["status"] == "lost" else 1,
            relation_priority.get(row["family_relation"], 4),
            -abs(float(row["family_adjustment"] or 0.0)),
            -float(row["gap_s"] or 0.0),
            str(row["predecessor_identity_id"]),
            str(row["successor_identity_id"]),
        )
    )

    for index, row in enumerate(records, start=1):
        row["audit_rank"] = index

    return tuple(records)


def summarize_delta(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = Counter(str(row["status"]) for row in rows)
    relations = Counter(
        str(row.get("family_relation") or "missing")
        for row in rows
    )
    lost_relations = Counter(
        str(row.get("family_relation") or "missing")
        for row in rows
        if row["status"] == "lost"
    )
    new_relations = Counter(
        str(row.get("family_relation") or "missing")
        for row in rows
        if row["status"] == "new"
    )

    adjustments = [
        float(row["family_adjustment"])
        for row in rows
        if row.get("family_adjustment") is not None
    ]

    return {
        "schema_version": 1,
        "audit_version": DELTA_AUDIT_VERSION,
        "delta_link_count": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "family_relation_counts": dict(sorted(relations.items())),
        "lost_family_relation_counts": dict(
            sorted(lost_relations.items())
        ),
        "new_family_relation_counts": dict(
            sorted(new_relations.items())
        ),
        "family_adjustment_min": (
            min(adjustments) if adjustments else None
        ),
        "family_adjustment_max": (
            max(adjustments) if adjustments else None
        ),
        "identity_decisions_changed": False,
    }
