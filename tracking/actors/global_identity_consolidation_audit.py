"""Tracking V1 Hito 7F.A: risk-oriented audit selection for later consolidation rounds.

Diagnostic only: no identity decisions are changed.

Selection policy:
- audit only rounds >= min_round (default 2);
- guarantee a small quota from every non-empty round;
- fill remaining budget globally with highest-risk links;
- transparent deterministic risk ordering:
  neutral family first, then lower score, longer gap, later round,
  larger accumulated source component, lower profile confidence.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Sequence


CONSOLIDATION_RISK_AUDIT_VERSION = (
    "global-identity-consolidation-risk-audit-v1"
)


def link_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["round_index"]),
        str(row["predecessor_identity_id"]),
        str(row["successor_identity_id"]),
    )


def _as_float(row: Mapping[str, Any], key: str) -> float | None:
    value = row.get(key)
    return None if value is None else float(value)


def _initial_ids(
    row: Mapping[str, Any],
    prefix: str,
) -> tuple[str, ...]:
    raw = row.get(f"{prefix}_initial_identity_ids", ())
    return tuple(str(value) for value in raw)


def _risk_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    relation = str(row.get("family_relation") or "neutral")
    relation_rank = 0 if relation == "neutral" else 1

    score = _as_float(row, "rank_score")
    gap = _as_float(row, "gap_s")
    confidence = _as_float(row, "profile_confidence")

    accumulated_source_count = (
        len(_initial_ids(row, "predecessor"))
        + len(_initial_ids(row, "successor"))
    )

    return (
        relation_rank,
        float("inf") if score is None else score,
        -0.0 if gap is None else -gap,
        -int(row["round_index"]),
        -accumulated_source_count,
        float("inf") if confidence is None else confidence,
        str(row["predecessor_identity_id"]),
        str(row["successor_identity_id"]),
    )


def enrich_risk_rows(
    *,
    selected_links: Sequence[Mapping[str, Any]],
    final_lineage_rows: Sequence[Mapping[str, Any]],
    min_round: int = 2,
) -> tuple[dict[str, Any], ...]:
    if min_round < 1:
        raise ValueError("min_round must be >= 1")

    final_by_initial: dict[str, tuple[str, int]] = {}
    for row in final_lineage_rows:
        final_id = str(row["identity_id"])
        initial_ids = tuple(
            str(value)
            for value in row.get("initial_identity_ids", ())
        )
        final_count = int(
            row.get("initial_identity_count", len(initial_ids))
        )
        for initial_id in initial_ids:
            previous = final_by_initial.get(initial_id)
            value = (final_id, final_count)
            if previous is not None and previous != value:
                raise ValueError(
                    f"initial identity {initial_id} belongs to multiple "
                    "final identities"
                )
            final_by_initial[initial_id] = value

    result: list[dict[str, Any]] = []

    for source in selected_links:
        round_index = int(source["round_index"])
        if round_index < min_round:
            continue

        row = dict(source)
        predecessor_initial = _initial_ids(row, "predecessor")
        successor_initial = _initial_ids(row, "successor")
        combined_initial = tuple(
            dict.fromkeys(predecessor_initial + successor_initial)
        )

        final_matches = {
            final_by_initial[initial_id]
            for initial_id in combined_initial
            if initial_id in final_by_initial
        }
        final_identity_id = None
        final_initial_count = None
        if len(final_matches) == 1:
            final_identity_id, final_initial_count = next(
                iter(final_matches)
            )

        score = _as_float(row, "rank_score")
        gap = _as_float(row, "gap_s")
        confidence = _as_float(row, "profile_confidence")

        reasons: list[str] = []
        if str(row.get("family_relation") or "neutral") == "neutral":
            reasons.append("neutral_family")
        if score is not None and score < 0.625:
            reasons.append("near_score_floor")
        if gap is not None and gap >= 10.0:
            reasons.append("long_gap")
        if gap is not None and gap >= 30.0:
            reasons.append("very_long_gap")
        if confidence is not None and confidence < 0.40:
            reasons.append("low_profile_confidence")
        if len(predecessor_initial) > 1:
            reasons.append("predecessor_already_merged")
        if len(successor_initial) > 1:
            reasons.append("successor_already_merged")
        if final_initial_count is not None and final_initial_count >= 5:
            reasons.append("feeds_large_final_component")

        row.update({
            "audit_version": CONSOLIDATION_RISK_AUDIT_VERSION,
            "predecessor_initial_identity_count": len(predecessor_initial),
            "successor_initial_identity_count": len(successor_initial),
            "combined_initial_identity_count": len(combined_initial),
            "final_identity_id": final_identity_id,
            "final_initial_identity_count": final_initial_count,
            "risk_reasons": reasons,
        })
        result.append(row)

    result.sort(key=_risk_key)
    return tuple(result)


def select_risk_audit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_links: int = 30,
    min_per_round: int = 5,
) -> tuple[dict[str, Any], ...]:
    if max_links < 1:
        raise ValueError("max_links must be >= 1")
    if min_per_round < 0:
        raise ValueError("min_per_round must be >= 0")

    prepared = [dict(row) for row in rows]
    prepared.sort(key=_risk_key)

    by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in prepared:
        by_round[int(row["round_index"])].append(row)

    selected: list[dict[str, Any]] = []
    selected_keys: set[tuple[int, str, str]] = set()

    for round_index in sorted(by_round):
        for row in by_round[round_index][:min_per_round]:
            if len(selected) >= max_links:
                break
            key = link_key(row)
            selected.append(row)
            selected_keys.add(key)
        if len(selected) >= max_links:
            break

    if len(selected) < max_links:
        for row in prepared:
            if len(selected) >= max_links:
                break
            key = link_key(row)
            if key in selected_keys:
                continue
            selected.append(row)
            selected_keys.add(key)

    selected.sort(
        key=lambda row: (
            -int(row["round_index"]),
            _risk_key(row),
        )
    )
    for rank, row in enumerate(selected, start=1):
        row["audit_rank"] = rank

    return tuple(selected)


def summarize_risk_audit(
    *,
    all_later_rows: Sequence[Mapping[str, Any]],
    audited_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    all_rounds = Counter(
        int(row["round_index"]) for row in all_later_rows
    )
    audited_rounds = Counter(
        int(row["round_index"]) for row in audited_rows
    )
    audited_relations = Counter(
        str(row.get("family_relation") or "neutral")
        for row in audited_rows
    )
    audited_reasons = Counter(
        reason
        for row in audited_rows
        for reason in row.get("risk_reasons", ())
    )

    return {
        "schema_version": 1,
        "audit_version": CONSOLIDATION_RISK_AUDIT_VERSION,
        "later_round_selected_link_count": len(all_later_rows),
        "audited_link_count": len(audited_rows),
        "later_round_counts": {
            str(key): value for key, value in sorted(all_rounds.items())
        },
        "audited_round_counts": {
            str(key): value for key, value in sorted(audited_rounds.items())
        },
        "audited_family_relation_counts": dict(
            sorted(audited_relations.items())
        ),
        "audited_risk_reason_counts": dict(
            sorted(audited_reasons.items())
        ),
        "selection_changed_identity_decisions": False,
    }
