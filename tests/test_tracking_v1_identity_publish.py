from __future__ import annotations

import pytest

from tracking.identities.publish import (
    IdentityPublishInputs,
    prepare_identity_publication,
)


def _inputs():
    identities = (
        {
            "schema_version": 1,
            "solver_version": "global-backbone-v1",
            "identity_id": "identity-001",
            "tracklet_ids": ["a", "b"],
            "tracklet_count": 2,
            "provisional": True,
        },
        {
            "schema_version": 1,
            "solver_version": "global-backbone-v1",
            "identity_id": "identity-002",
            "tracklet_ids": ["c"],
            "tracklet_count": 1,
            "provisional": True,
        },
    )
    assignments = (
        {
            "tracklet_id": "a",
            "identity_id": "identity-001",
        },
        {
            "tracklet_id": "b",
            "identity_id": "identity-001",
        },
        {
            "tracklet_id": "c",
            "identity_id": "identity-002",
        },
    )
    intra = (
        {
            "tracklet_a": "a",
            "tracklet_b": "b",
            "utility": 0.7,
            "rank_score": 0.9,
            "evidence_confidence": 0.8,
            "link_kind": "continuation",
        },
    )
    selected = (
        {
            "predecessor_tracklet_id": "a",
            "successor_tracklet_id": "b",
            "utility": 0.7,
        },
    )
    tracklets = (
        {"tracklet_id": "a", "observations": [{}]},
        {"tracklet_id": "b", "observations": [{}]},
        {"tracklet_id": "c", "observations": [{}]},
    )
    candidates = (
        {
            "tracklet_a": "a",
            "tracklet_b": "b",
            "status": "candidate",
        },
        {
            "tracklet_a": "a",
            "tracklet_b": "c",
            "status": "cannot_link",
        },
        {
            "tracklet_a": "b",
            "tracklet_b": "c",
            "status": "cannot_link",
        },
    )
    solver_summary = {
        "solver_version": "global-backbone-v1",
        "tracklet_count": 3,
        "final_identity_count": 2,
        "selected_backbone_link_count": 1,
        "validation": {
            "hard_constraint_violation_count": 0,
        },
    }
    diagnostics_summary = {
        "diagnostics_version": "identity-backbone-audit-v1",
        "final_identity_count": 2,
        "tracklet_count": 3,
        "baseline_cluster_preservation_violation_count": 0,
        "baseline_hard_constraint_violation_count": 0,
        "final_hard_constraint_violation_count": 0,
        "bridge_validation_failure_count": 0,
        "unselected_reciprocal_eligible_boundary_handoff_count": 0,
        "unexpected_legacy_clique_split_pressure_count": 0,
        "bridge_graph_acyclic": True,
        "final_partition_matches_bridge_components": True,
        "baseline_clique_diagnostics": {
            "profitable_exact_split_identity_count": 0,
            "profitable_single_tracklet_move_count": 0,
            "profitable_whole_identity_merge_count": 0,
        },
    }
    visual_summary = {
        "visual_audit_version": "visual-backbone-audit-v1",
        "identity_count": 2,
        "tracklet_count": 3,
    }
    fragmentation_summary = {
        "fragmentation_audit_version": (
            "fragmentation-backbone-audit-v1"
        ),
        "identity_count": 2,
        "remaining_reciprocal_strong_boundary_handoff_count": 0,
        "trajectory_vs_clique_conflict_count": 0,
        "strong_boundary_handoff_count": 0,
    }
    return IdentityPublishInputs(
        identities=identities,
        assignments=assignments,
        intra_identity_links=intra,
        selected_backbone_links=selected,
        tracklets=tracklets,
        candidate_edges=candidates,
        solver_summary=solver_summary,
        diagnostics_summary=diagnostics_summary,
        visual_summary=visual_summary,
        fragmentation_summary=fragmentation_summary,
    )


def test_prepare_publication_marks_records_canonical():
    result = prepare_identity_publication(_inputs())

    assert result.summary["identity_count"] == 2
    assert result.summary["tracklet_count"] == 3
    assert result.summary["singleton_identity_count"] == 1
    assert result.summary["hard_constraint_violation_count"] == 0
    assert all(row["canonical"] for row in result.identities)
    assert all(
        row["canonical"] for row in result.assignments
    )
    assert result.identity_links[0][
        "selected_backbone_bridge"
    ] is True
    assert result.identities[0]["provisional"] is False


def test_rejects_tracklet_missing_from_partition():
    inputs = _inputs()
    broken = IdentityPublishInputs(
        identities=(inputs.identities[0],),
        assignments=inputs.assignments,
        intra_identity_links=inputs.intra_identity_links,
        selected_backbone_links=inputs.selected_backbone_links,
        tracklets=inputs.tracklets,
        candidate_edges=inputs.candidate_edges,
        solver_summary=inputs.solver_summary,
        diagnostics_summary=inputs.diagnostics_summary,
        visual_summary=inputs.visual_summary,
        fragmentation_summary=inputs.fragmentation_summary,
    )
    with pytest.raises(
        ValueError,
        match="does not cover canonical tracklets exactly",
    ):
        prepare_identity_publication(broken)


def test_rejects_hard_cannot_link_inside_identity():
    inputs = _inputs()
    candidates = tuple(
        {
            **row,
            "status": (
                "cannot_link"
                if {row["tracklet_a"], row["tracklet_b"]}
                == {"a", "b"}
                else row["status"]
            ),
        }
        for row in inputs.candidate_edges
    )
    broken = IdentityPublishInputs(
        identities=inputs.identities,
        assignments=inputs.assignments,
        intra_identity_links=inputs.intra_identity_links,
        selected_backbone_links=inputs.selected_backbone_links,
        tracklets=inputs.tracklets,
        candidate_edges=candidates,
        solver_summary=inputs.solver_summary,
        diagnostics_summary=inputs.diagnostics_summary,
        visual_summary=inputs.visual_summary,
        fragmentation_summary=inputs.fragmentation_summary,
    )
    with pytest.raises(
        ValueError,
        match="hard cannot-links",
    ):
        prepare_identity_publication(broken)


def test_rejects_dirty_fragmentation_audit():
    inputs = _inputs()
    frag = dict(inputs.fragmentation_summary)
    frag[
        "remaining_reciprocal_strong_boundary_handoff_count"
    ] = 1
    broken = IdentityPublishInputs(
        identities=inputs.identities,
        assignments=inputs.assignments,
        intra_identity_links=inputs.intra_identity_links,
        selected_backbone_links=inputs.selected_backbone_links,
        tracklets=inputs.tracklets,
        candidate_edges=inputs.candidate_edges,
        solver_summary=inputs.solver_summary,
        diagnostics_summary=inputs.diagnostics_summary,
        visual_summary=inputs.visual_summary,
        fragmentation_summary=frag,
    )
    with pytest.raises(
        ValueError,
        match="remaining reciprocal",
    ):
        prepare_identity_publication(broken)


def test_rejects_dirty_backbone_diagnostics():
    inputs = _inputs()
    diag = dict(inputs.diagnostics_summary)
    diag["bridge_validation_failure_count"] = 1
    broken = IdentityPublishInputs(
        identities=inputs.identities,
        assignments=inputs.assignments,
        intra_identity_links=inputs.intra_identity_links,
        selected_backbone_links=inputs.selected_backbone_links,
        tracklets=inputs.tracklets,
        candidate_edges=inputs.candidate_edges,
        solver_summary=inputs.solver_summary,
        diagnostics_summary=diag,
        visual_summary=inputs.visual_summary,
        fragmentation_summary=inputs.fragmentation_summary,
    )
    with pytest.raises(
        ValueError,
        match="bridge_validation_failure_count",
    ):
        prepare_identity_publication(broken)
