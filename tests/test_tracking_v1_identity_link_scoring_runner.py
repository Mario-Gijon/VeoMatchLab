from __future__ import annotations

import json

from experiments.tracking.run_v1_identity_link_scoring import (
    run_soft_link_scoring,
)


def test_runner_scores_candidates_and_preserves_cannot_links(tmp_path):
    edges = tmp_path / "edges.jsonl"
    nodes = tmp_path / "nodes.jsonl"
    benchmark = tmp_path / "summary.json"
    output = tmp_path / "out"

    edge_rows = [
        {
            "tracklet_a": "trk-a",
            "tracklet_b": "trk-b",
            "status": "candidate",
            "reason": "transition_reachable",
            "temporal_relation": "non_overlapping",
            "physics": {
                "gap_s": 0.1,
                "required_speed_after_slack_mps": 1.0,
                "prediction_error_mean_m": 1.0,
                "velocity_direction_cosine": 1.0,
            },
            "appearance": {
                "aggregate_cosine": 0.97,
                "bidirectional_best_similarity": {
                    "median": 0.96,
                },
                "reliability": 0.8,
                "support_quality": 0.9,
            },
        },
        {
            "tracklet_a": "trk-a",
            "tracklet_b": "trk-c",
            "status": "cannot_link",
            "reason": "simultaneous_spatial_separation",
            "temporal_relation": "simultaneous_samples",
            "physics": {},
            "appearance": {
                "aggregate_cosine": 0.99,
                "reliability": 0.9,
                "support_quality": 0.9,
            },
        },
        {
            "tracklet_a": "trk-b",
            "tracklet_b": "trk-c",
            "status": "candidate",
            "reason": "transition_reachable",
            "temporal_relation": "non_overlapping",
            "physics": {
                "gap_s": 2.0,
                "required_speed_after_slack_mps": 6.0,
                "prediction_error_mean_m": 8.0,
                "velocity_direction_cosine": 0.0,
            },
            "appearance": None,
        },
    ]
    edges.write_text(
        "".join(json.dumps(row) + "\n" for row in edge_rows),
        encoding="utf-8",
    )

    node_rows = [
        {"tracklet_id": "trk-a", "mean_confidence": 0.9},
        {"tracklet_id": "trk-b", "mean_confidence": 0.8},
        {"tracklet_id": "trk-c", "mean_confidence": 0.7},
    ]
    nodes.write_text(
        "".join(json.dumps(row) + "\n" for row in node_rows),
        encoding="utf-8",
    )

    benchmark.write_text(
        json.dumps(
            {
                "tracklet_level": {
                    "split_positive": {"p10": 0.9344},
                    "concurrent_negative": {"p95": 0.9095},
                    "roc_auc": 0.984,
                    "p10_positive_minus_p95_negative": 0.0249,
                }
            }
        ),
        encoding="utf-8",
    )

    summary = run_soft_link_scoring(
        candidate_edges_path=edges,
        nodes_path=nodes,
        benchmark_summary_path=benchmark,
        output_dir=output,
    )

    assert summary["source_pair_count"] == 3
    assert summary["source_candidate_count"] == 2
    assert summary["source_cannot_link_count"] == 1
    assert summary["scored_link_count"] == 2
    assert summary["hard_constraints_modified"] is False
    assert summary["identity_decisions"] is False

    scored = [
        json.loads(line)
        for line in (output / "scored_links.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(scored) == 2
    assert all(row["identity_decision"] is None for row in scored)
    assert not any(
        row["tracklet_a"] == "trk-a"
        and row["tracklet_b"] == "trk-c"
        for row in scored
    )

    index = json.loads(
        (output / "index.json").read_text(encoding="utf-8")
    )
    assert index["rank_score_is_probability"] is False
    assert index["hard_constraints_modified"] is False
