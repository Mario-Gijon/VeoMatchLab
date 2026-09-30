from __future__ import annotations

import json

from experiments.tracking.run_v1_global_identity_solver import (
    run_global_identity_solver,
)
from tracking.identities.global_solver import (
    BackboneAugmentationConfig,
)


def test_runner_writes_backbone_solver_artifacts(tmp_path):
    candidates = tmp_path / "edges.jsonl"
    scores = tmp_path / "scores.jsonl"
    nodes = tmp_path / "nodes.jsonl"
    output = tmp_path / "out"

    edge_rows = [
        {
            "tracklet_a": "a",
            "tracklet_b": "b",
            "status": "candidate",
            "reason": "transition_reachable",
            "physics": {
                "predecessor": "a",
                "successor": "b",
                "gap_s": 0.1,
                "endpoint_distance_m": 1.0,
            },
        },
        {
            "tracklet_a": "a",
            "tracklet_b": "c",
            "status": "candidate",
            "reason": "transition_reachable",
            "physics": {
                "predecessor": "a",
                "successor": "c",
                "gap_s": 1.1,
                "endpoint_distance_m": 2.0,
            },
        },
        {
            "tracklet_a": "b",
            "tracklet_b": "c",
            "status": "candidate",
            "reason": "transition_reachable",
            "physics": {
                "predecessor": "b",
                "successor": "c",
                "gap_s": 0.1,
                "endpoint_distance_m": 1.0,
            },
        },
    ]
    score_rows = [
        {
            "tracklet_a": "a",
            "tracklet_b": "b",
            "rank_score": 0.95,
            "evidence_confidence": 1.0,
            "link_kind": "continuation",
            "identity_decision": None,
        },
        {
            "tracklet_a": "a",
            "tracklet_b": "c",
            "rank_score": 0.05,
            "evidence_confidence": 1.0,
            "link_kind": "continuation",
            "identity_decision": None,
        },
        {
            "tracklet_a": "b",
            "tracklet_b": "c",
            "rank_score": 0.90,
            "evidence_confidence": 1.0,
            "link_kind": "continuation",
            "identity_decision": None,
        },
    ]
    node_rows = [
        {
            "tracklet_id": "a",
            "start_time_s": 0.0,
            "end_time_s": 1.0,
        },
        {
            "tracklet_id": "b",
            "start_time_s": 1.1,
            "end_time_s": 2.0,
        },
        {
            "tracklet_id": "c",
            "start_time_s": 2.1,
            "end_time_s": 3.0,
        },
    ]

    candidates.write_text(
        "\n".join(json.dumps(row) for row in edge_rows) + "\n",
        encoding="utf-8",
    )
    scores.write_text(
        "\n".join(json.dumps(row) for row in score_rows) + "\n",
        encoding="utf-8",
    )
    nodes.write_text(
        "\n".join(json.dumps(row) for row in node_rows) + "\n",
        encoding="utf-8",
    )

    summary = run_global_identity_solver(
        candidate_edges_path=candidates,
        scored_links_path=scores,
        nodes_path=nodes,
        output_dir=output,
        config=BackboneAugmentationConfig(),
    )

    assert summary["solver_version"] == "global-backbone-v1"
    assert summary["baseline_identity_count"] == 2
    assert summary["final_identity_count"] == 1
    assert summary["selected_backbone_link_count"] == 1
    assert summary["validation"][
        "hard_constraint_violation_count"
    ] == 0

    for name in (
        "index.json",
        "identities.jsonl",
        "assignments.jsonl",
        "intra_identity_links.jsonl",
        "cut_positive_links.jsonl",
        "selected_backbone_links.jsonl",
        "baseline_comparison.json",
        "solver_trace.json",
        "summary.json",
    ):
        assert (output / name).is_file()

    index = json.loads(
        (output / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["provisional"] is True
    assert index["selected_backbone_link_count"] == 1
