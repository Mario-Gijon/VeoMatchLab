from __future__ import annotations

import json

from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import (
    StageStatus,
    TrackingStage,
)
from experiments.tracking.run_v1_publish_identities import (
    run_publish_identities,
)


def _complete_prerequisites(store):
    for stage in (
        TrackingStage.DETECTIONS,
        TrackingStage.TRACKLETS,
        TrackingStage.APPEARANCE,
    ):
        store.begin_stage(
            stage,
            version="test-v1",
        )
        artifact = store.write_json(
            stage,
            "index.json",
            {"stage": stage.value},
        )
        store.complete_stage(
            stage,
            artifact=artifact,
        )


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _source_fixture(tmp_path):
    source = tmp_path / "source"
    identities = source / "identities.jsonl"
    assignments = source / "assignments.jsonl"
    intra = source / "intra.jsonl"
    selected = source / "selected.jsonl"
    tracklets = source / "tracklets.jsonl"
    candidates = source / "candidates.jsonl"
    solver_summary = source / "solver_summary.json"
    diagnostics_summary = source / "diagnostics_summary.json"
    visual_summary = source / "visual_summary.json"
    fragmentation_summary = source / "fragmentation_summary.json"

    _write_jsonl(
        identities,
        [
            {
                "solver_version": "global-backbone-v1",
                "identity_id": "identity-001",
                "tracklet_ids": ["a", "b"],
                "tracklet_count": 2,
                "provisional": True,
            },
            {
                "solver_version": "global-backbone-v1",
                "identity_id": "identity-002",
                "tracklet_ids": ["c"],
                "tracklet_count": 1,
                "provisional": True,
            },
        ],
    )
    _write_jsonl(
        assignments,
        [
            {"tracklet_id": "a", "identity_id": "identity-001"},
            {"tracklet_id": "b", "identity_id": "identity-001"},
            {"tracklet_id": "c", "identity_id": "identity-002"},
        ],
    )
    _write_jsonl(
        intra,
        [
            {
                "tracklet_a": "a",
                "tracklet_b": "b",
                "utility": 0.7,
                "rank_score": 0.9,
                "evidence_confidence": 0.8,
                "link_kind": "continuation",
            }
        ],
    )
    _write_jsonl(
        selected,
        [
            {
                "predecessor_tracklet_id": "a",
                "successor_tracklet_id": "b",
                "utility": 0.7,
            }
        ],
    )
    _write_jsonl(
        tracklets,
        [
            {"tracklet_id": "a"},
            {"tracklet_id": "b"},
            {"tracklet_id": "c"},
        ],
    )
    _write_jsonl(
        candidates,
        [
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
        ],
    )
    _write_json(
        solver_summary,
        {
            "solver_version": "global-backbone-v1",
            "tracklet_count": 3,
            "final_identity_count": 2,
            "selected_backbone_link_count": 1,
            "validation": {
                "hard_constraint_violation_count": 0,
            },
        },
    )
    _write_json(
        diagnostics_summary,
        {
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
        },
    )
    _write_json(
        visual_summary,
        {
            "visual_audit_version": "visual-backbone-audit-v1",
            "identity_count": 2,
            "tracklet_count": 3,
        },
    )
    _write_json(
        fragmentation_summary,
        {
            "fragmentation_audit_version": (
                "fragmentation-backbone-audit-v1"
            ),
            "identity_count": 2,
            "remaining_reciprocal_strong_boundary_handoff_count": 0,
            "trajectory_vs_clique_conflict_count": 0,
            "strong_boundary_handoff_count": 0,
        },
    )
    return {
        "identities": identities,
        "assignments": assignments,
        "intra_identity_links": intra,
        "selected_backbone_links": selected,
        "tracklets": tracklets,
        "candidate_edges": candidates,
        "solver_summary": solver_summary,
        "diagnostics_summary": diagnostics_summary,
        "visual_summary": visual_summary,
        "fragmentation_summary": fragmentation_summary,
    }


def test_runner_publishes_and_completes_identity_stage(tmp_path):
    data_root = tmp_path / "data"
    clip_dir = (
        data_root
        / "matches"
        / "match_test"
        / "clips"
        / "clip_test"
    )
    clip_dir.mkdir(parents=True)

    store = TrackingArtifactStore.for_clip(
        data_root,
        "match_test",
        "clip_test",
    )
    store.ensure_manifest()
    _complete_prerequisites(store)

    summary = run_publish_identities(
        store=store,
        paths=_source_fixture(tmp_path),
    )

    assert summary["manifest_status"] == "completed"
    assert summary["identity_count"] == 2
    assert summary["tracklet_count"] == 3

    manifest = store.load_manifest()
    identity_stage = manifest.stage(
        TrackingStage.IDENTITIES
    )
    assert identity_stage.status is StageStatus.COMPLETED
    assert identity_stage.version == "identity-publish-v1"

    assert (
        store.root / "identities" / "identities.jsonl"
    ).is_file()
    assert (
        store.root / "identities" / "assignments.jsonl"
    ).is_file()
    assert (
        store.root / "identities" / "identity_links.jsonl"
    ).is_file()
    assert (
        store.root / "identities" / "summary.json"
    ).is_file()
    assert (
        store.root / "identities" / "index.json"
    ).is_file()

    final_stage = manifest.stage(TrackingStage.FINAL)
    assert final_stage.status is StageStatus.NOT_RUN


def test_validation_failure_does_not_begin_identity_stage(tmp_path):
    data_root = tmp_path / "data"
    clip_dir = (
        data_root
        / "matches"
        / "match_test"
        / "clips"
        / "clip_test"
    )
    clip_dir.mkdir(parents=True)

    store = TrackingArtifactStore.for_clip(
        data_root,
        "match_test",
        "clip_test",
    )
    store.ensure_manifest()
    _complete_prerequisites(store)

    paths = _source_fixture(tmp_path)
    frag = json.loads(
        paths["fragmentation_summary"].read_text(
            encoding="utf-8"
        )
    )
    frag[
        "remaining_reciprocal_strong_boundary_handoff_count"
    ] = 1
    _write_json(paths["fragmentation_summary"], frag)

    try:
        run_publish_identities(
            store=store,
            paths=paths,
        )
        assert False, "expected publication validation to fail"
    except ValueError:
        pass

    manifest = store.load_manifest()
    assert (
        manifest.stage(TrackingStage.IDENTITIES).status
        is StageStatus.NOT_RUN
    )
