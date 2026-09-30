"""Publish validated Tracking V1 physical identities as the canonical stage.

Hito 3F is intentionally boring: it does not solve, merge, split, re-score, or
repair identities. It only promotes an already validated `global-backbone-v1`
partition into `tracking/v1/identities/` after strict guardrail checks.

The publisher is conservative:
- every canonical tracklet must be assigned exactly once;
- solver assignments must match solver identities exactly;
- no hard cannot-link may be internal to a canonical identity;
- 3C V1 must report a clean backbone-aware audit;
- 3E V1 must report no remaining reciprocal strong boundary handoff;
- 3D V2 must refer to the same identity/tracklet population;
- selected backbone links must remain internal to the final partition.

Only the runner mutates TrackingArtifactStore / TrackingStage.IDENTITIES.
This module is pure validation + canonical-record preparation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


PUBLISH_VERSION = "identity-publish-v1"
SOLVER_VERSION = "global-backbone-v1"
CANDIDATE_GRAPH_VERSION = "physical-compatibility-v1"
SCORING_VERSION = "soft-link-v1"
DIAGNOSTICS_VERSION = "identity-backbone-audit-v1"
VISUAL_AUDIT_VERSION = "visual-backbone-audit-v1"
FRAGMENTATION_AUDIT_VERSION = "fragmentation-backbone-audit-v1"


@dataclass(frozen=True, slots=True)
class IdentityPublishInputs:
    identities: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, Any], ...]
    intra_identity_links: tuple[dict[str, Any], ...]
    selected_backbone_links: tuple[dict[str, Any], ...]
    tracklets: tuple[dict[str, Any], ...]
    candidate_edges: tuple[dict[str, Any], ...]
    solver_summary: dict[str, Any]
    diagnostics_summary: dict[str, Any]
    visual_summary: dict[str, Any]
    fragmentation_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class IdentityPublishResult:
    identities: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, Any], ...]
    identity_links: tuple[dict[str, Any], ...]
    summary: dict[str, Any]
    provenance: dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source}:{line_number} is not a JSON object"
                )
            yield payload


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} is not a JSON object")
    return payload


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("link pair must contain two distinct tracklets")
    return (a, b) if a < b else (b, a)


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _identity_map(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    identities: dict[str, tuple[str, ...]] = {}
    assignment: dict[str, str] = {}

    for row in rows:
        identity_id = str(row["identity_id"])
        if identity_id in identities:
            raise ValueError(f"duplicate identity_id: {identity_id}")

        members = tuple(str(value) for value in row["tracklet_ids"])
        if not members:
            raise ValueError(f"{identity_id} has no tracklets")
        if len(set(members)) != len(members):
            raise ValueError(
                f"{identity_id} contains a duplicate tracklet"
            )

        members = tuple(sorted(members))
        identities[identity_id] = members
        for tracklet_id in members:
            previous = assignment.get(tracklet_id)
            if previous is not None:
                raise ValueError(
                    f"{tracklet_id} belongs to both "
                    f"{previous} and {identity_id}"
                )
            assignment[tracklet_id] = identity_id

    return identities, assignment


def _solver_assignment_map(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        identity_id = str(row["identity_id"])
        if tracklet_id in result:
            raise ValueError(
                f"duplicate solver assignment for {tracklet_id}"
            )
        result[tracklet_id] = identity_id
    return result


def _tracklet_ids(
    rows: Sequence[Mapping[str, Any]],
) -> set[str]:
    result: set[str] = set()
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id in result:
            raise ValueError(
                f"duplicate canonical tracklet: {tracklet_id}"
            )
        result.add(tracklet_id)
    return result


def _hard_cannot_links(
    candidate_edges: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    seen: set[tuple[str, str]] = set()

    for row in candidate_edges:
        pair = _canonical_pair(
            str(row["tracklet_a"]),
            str(row["tracklet_b"]),
        )
        if pair in seen:
            raise ValueError(
                f"duplicate candidate-graph pair: {pair}"
            )
        seen.add(pair)
        status = row.get("status")
        if status == "cannot_link":
            result.add(pair)
        elif status != "candidate":
            raise ValueError(
                f"candidate graph pair {pair} has invalid status {status!r}"
            )
    return result


def _validate_versions(inputs: IdentityPublishInputs) -> None:
    _require_equal(
        "solver version",
        inputs.solver_summary.get("solver_version"),
        SOLVER_VERSION,
    )
    _require_equal(
        "3C diagnostics version",
        inputs.diagnostics_summary.get("diagnostics_version"),
        DIAGNOSTICS_VERSION,
    )
    _require_equal(
        "3D visual audit version",
        inputs.visual_summary.get("visual_audit_version"),
        VISUAL_AUDIT_VERSION,
    )
    _require_equal(
        "3E fragmentation audit version",
        inputs.fragmentation_summary.get("fragmentation_audit_version"),
        FRAGMENTATION_AUDIT_VERSION,
    )


def _validate_diagnostics(
    inputs: IdentityPublishInputs,
    *,
    identity_count: int,
    tracklet_count: int,
) -> None:
    diag = inputs.diagnostics_summary

    required_zero = {
        "baseline_cluster_preservation_violation_count": 0,
        "baseline_hard_constraint_violation_count": 0,
        "final_hard_constraint_violation_count": 0,
        "bridge_validation_failure_count": 0,
        "unselected_reciprocal_eligible_boundary_handoff_count": 0,
        "unexpected_legacy_clique_split_pressure_count": 0,
    }
    for key, expected in required_zero.items():
        _require_equal(f"3C {key}", diag.get(key), expected)

    _require_equal(
        "3C bridge graph acyclic",
        diag.get("bridge_graph_acyclic"),
        True,
    )
    _require_equal(
        "3C partition matches bridge components",
        diag.get("final_partition_matches_bridge_components"),
        True,
    )
    _require_equal(
        "3C final identity count",
        diag.get("final_identity_count"),
        identity_count,
    )
    _require_equal(
        "3C tracklet count",
        diag.get("tracklet_count"),
        tracklet_count,
    )

    baseline = diag.get("baseline_clique_diagnostics")
    if not isinstance(baseline, Mapping):
        raise ValueError("3C baseline_clique_diagnostics is missing")
    for key in (
        "profitable_exact_split_identity_count",
        "profitable_single_tracklet_move_count",
        "profitable_whole_identity_merge_count",
    ):
        _require_equal(f"3C baseline {key}", baseline.get(key), 0)


def _validate_visual(
    inputs: IdentityPublishInputs,
    *,
    identity_count: int,
    tracklet_count: int,
) -> None:
    visual = inputs.visual_summary
    _require_equal(
        "3D identity count",
        visual.get("identity_count"),
        identity_count,
    )
    _require_equal(
        "3D tracklet count",
        visual.get("tracklet_count"),
        tracklet_count,
    )

    # 3D review flags are intentionally diagnostic, not publication blockers.
    # Hard physics has already been enforced by 3A and re-checked here.


def _validate_fragmentation(
    inputs: IdentityPublishInputs,
    *,
    identity_count: int,
) -> None:
    frag = inputs.fragmentation_summary
    _require_equal(
        "3E identity count",
        frag.get("identity_count"),
        identity_count,
    )
    _require_equal(
        "3E remaining reciprocal strong boundary handoffs",
        frag.get(
            "remaining_reciprocal_strong_boundary_handoff_count"
        ),
        0,
    )
    _require_equal(
        "3E trajectory-vs-clique conflicts",
        frag.get("trajectory_vs_clique_conflict_count"),
        0,
    )
    _require_equal(
        "3E strong boundary handoffs",
        frag.get("strong_boundary_handoff_count"),
        0,
    )


def _selected_backbone_pairs(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        pair = _canonical_pair(
            str(row["predecessor_tracklet_id"]),
            str(row["successor_tracklet_id"]),
        )
        if pair in result:
            raise ValueError(
                f"duplicate selected backbone pair: {pair}"
            )
        result[pair] = dict(row)
    return result


def prepare_identity_publication(
    inputs: IdentityPublishInputs,
) -> IdentityPublishResult:
    """Validate all publication guardrails and prepare canonical records."""

    _validate_versions(inputs)

    identities, assignment = _identity_map(inputs.identities)
    solver_assignment = _solver_assignment_map(inputs.assignments)
    tracklet_ids = _tracklet_ids(inputs.tracklets)

    if set(assignment) != tracklet_ids:
        missing = sorted(tracklet_ids - set(assignment))
        extra = sorted(set(assignment) - tracklet_ids)
        raise ValueError(
            "identity partition does not cover canonical tracklets exactly; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )

    if solver_assignment != assignment:
        raise ValueError(
            "solver assignments do not exactly match solver identities"
        )

    identity_count = len(identities)
    tracklet_count = len(tracklet_ids)

    _require_equal(
        "solver final identity count",
        inputs.solver_summary.get("final_identity_count"),
        identity_count,
    )
    _require_equal(
        "solver tracklet count",
        inputs.solver_summary.get("tracklet_count"),
        tracklet_count,
    )
    _require_equal(
        "solver hard violations",
        (
            inputs.solver_summary.get("validation") or {}
        ).get("hard_constraint_violation_count"),
        0,
    )

    hard = _hard_cannot_links(inputs.candidate_edges)
    hard_violations = [
        pair
        for pair in sorted(hard)
        if assignment.get(pair[0]) == assignment.get(pair[1])
    ]
    if hard_violations:
        raise ValueError(
            "canonical partition contains hard cannot-links: "
            f"{hard_violations[:10]}"
        )

    _validate_diagnostics(
        inputs,
        identity_count=identity_count,
        tracklet_count=tracklet_count,
    )
    _validate_visual(
        inputs,
        identity_count=identity_count,
        tracklet_count=tracklet_count,
    )
    _validate_fragmentation(
        inputs,
        identity_count=identity_count,
    )

    selected = _selected_backbone_pairs(
        inputs.selected_backbone_links
    )
    _require_equal(
        "solver selected backbone link count",
        inputs.solver_summary.get(
            "selected_backbone_link_count"
        ),
        len(selected),
    )

    for pair in selected:
        if assignment[pair[0]] != assignment[pair[1]]:
            raise ValueError(
                f"selected backbone pair is cut in final partition: {pair}"
            )

    provenance = {
        "publisher_version": PUBLISH_VERSION,
        "solver_version": SOLVER_VERSION,
        "candidate_graph_version": CANDIDATE_GRAPH_VERSION,
        "scoring_version": SCORING_VERSION,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "visual_audit_version": VISUAL_AUDIT_VERSION,
        "fragmentation_audit_version": FRAGMENTATION_AUDIT_VERSION,
    }

    canonical_identities = []
    for row in sorted(
        inputs.identities,
        key=lambda item: str(item["identity_id"]),
    ):
        identity_id = str(row["identity_id"])
        members = identities[identity_id]
        canonical_identities.append(
            {
                **dict(row),
                "schema_version": 1,
                "identity_id": identity_id,
                "tracklet_ids": list(members),
                "tracklet_count": len(members),
                "provisional": False,
                "canonical": True,
                "published_by": PUBLISH_VERSION,
                "source_solver_version": SOLVER_VERSION,
            }
        )

    canonical_assignments = [
        {
            "schema_version": 1,
            "tracklet_id": tracklet_id,
            "identity_id": assignment[tracklet_id],
            "canonical": True,
            "published_by": PUBLISH_VERSION,
            "source_solver_version": SOLVER_VERSION,
        }
        for tracklet_id in sorted(tracklet_ids)
    ]

    canonical_links = []
    seen_links: set[tuple[str, str]] = set()
    selected_seen: set[tuple[str, str]] = set()

    for row in inputs.intra_identity_links:
        pair = _canonical_pair(
            str(row["tracklet_a"]),
            str(row["tracklet_b"]),
        )
        if pair in seen_links:
            raise ValueError(
                f"duplicate intra-identity soft link: {pair}"
            )
        seen_links.add(pair)

        identity_a = assignment.get(pair[0])
        identity_b = assignment.get(pair[1])
        if identity_a is None or identity_a != identity_b:
            raise ValueError(
                f"solver intra-identity link is not internal: {pair}"
            )

        selected_payload = selected.get(pair)
        if selected_payload is not None:
            selected_seen.add(pair)

        canonical_links.append(
            {
                **dict(row),
                "schema_version": 1,
                "identity_id": identity_a,
                "tracklet_a": pair[0],
                "tracklet_b": pair[1],
                "canonical": True,
                "published_by": PUBLISH_VERSION,
                "selected_backbone_bridge": (
                    selected_payload is not None
                ),
                "selected_backbone_metadata": (
                    selected_payload
                    if selected_payload is not None
                    else None
                ),
            }
        )

    missing_selected = sorted(set(selected) - selected_seen)
    if missing_selected:
        raise ValueError(
            "selected backbone links are missing from intra-identity links: "
            f"{missing_selected}"
        )

    canonical_links.sort(
        key=lambda row: (
            row["identity_id"],
            -float(row.get("utility", 0.0)),
            row["tracklet_a"],
            row["tracklet_b"],
        )
    )

    singleton_count = sum(
        len(members) == 1
        for members in identities.values()
    )

    summary = {
        "schema_version": 1,
        "publisher_version": PUBLISH_VERSION,
        "canonical": True,
        "identity_stage_completed": True,
        "identity_count": identity_count,
        "tracklet_count": tracklet_count,
        "singleton_identity_count": singleton_count,
        "multi_tracklet_identity_count": (
            identity_count - singleton_count
        ),
        "identity_link_count": len(canonical_links),
        "selected_backbone_bridge_count": len(selected),
        "hard_cannot_link_count": len(hard),
        "hard_constraint_violation_count": 0,
        "guardrails": {
            "all_tracklets_assigned_exactly_once": True,
            "solver_assignments_match_identities": True,
            "hard_constraints_clean": True,
            "backbone_diagnostics_clean": True,
            "fragmentation_audit_clean": True,
            "visual_population_matches": True,
            "selected_backbone_links_internal": True,
        },
        "provenance": provenance,
    }

    return IdentityPublishResult(
        identities=tuple(canonical_identities),
        assignments=tuple(canonical_assignments),
        identity_links=tuple(canonical_links),
        summary=summary,
        provenance=provenance,
    )
