from __future__ import annotations

import json

from experiments.tracking.run_v1_identity_diagnostics import (
    run_identity_diagnostics,
)


def test_runner_writes_backbone_aware_identity_audit(tmp_path):
    final_identities = tmp_path / "final.jsonl"
    baseline_identities = tmp_path / "baseline.jsonl"
    selected_links = tmp_path / "selected.jsonl"
    solver_summary = tmp_path / "solver_summary.json"
    tracklets = tmp_path / "tracklets.jsonl"
    scores = tmp_path / "scores.jsonl"
    candidates = tmp_path / "candidates.jsonl"
    output = tmp_path / "out"

    final_identities.write_text(
        json.dumps(
            {
                "identity_id": "identity-final",
                "tracklet_ids": ["a", "b", "c"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline_identities.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "identity_id": "identity-a",
                        "tracklet_ids": ["a"],
                    }
                ),
                json.dumps(
                    {
                        "identity_id": "identity-b",
                        "tracklet_ids": ["b", "c"],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    selected_links.write_text(
        json.dumps(
            {
                "predecessor_tracklet_id": "a",
                "successor_tracklet_id": "b",
                "utility": 0.7,
                "rank_score": 0.85,
                "evidence_confidence": 1.0,
                "gap_s": 0.1,
                "endpoint_distance_m": 1.0,
                "source_tail_offset_s": 0.0,
                "target_head_offset_s": 0.0,
                "whole_clique_merge_delta": -0.3,
                "strong_utility_threshold": 0.5,
                "reciprocal_best": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    solver_summary.write_text(
        json.dumps(
            {
                "strong_utility_threshold": 0.5,
            }
        ),
        encoding="utf-8",
    )
    tracklets.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tracklet_id": "a",
                        "observations": [
                            {
                                "frame_index": 0,
                                "timestamp_s": 0.0,
                            },
                            {
                                "frame_index": 10,
                                "timestamp_s": 1.0,
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "tracklet_id": "b",
                        "observations": [
                            {
                                "frame_index": 11,
                                "timestamp_s": 1.1,
                            },
                            {
                                "frame_index": 20,
                                "timestamp_s": 2.0,
                            },
                        ],
                    }
                ),
                json.dumps(
                    {
                        "tracklet_id": "c",
                        "observations": [
                            {
                                "frame_index": 21,
                                "timestamp_s": 2.1,
                            },
                            {
                                "frame_index": 30,
                                "timestamp_s": 3.0,
                            },
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tracklet_a": "a",
                        "tracklet_b": "b",
                        "rank_score": 0.85,
                        "evidence_confidence": 1.0,
                        "link_kind": "continuation",
                    }
                ),
                json.dumps(
                    {
                        "tracklet_a": "a",
                        "tracklet_b": "c",
                        "rank_score": 0.0,
                        "evidence_confidence": 1.0,
                        "link_kind": "continuation",
                    }
                ),
                json.dumps(
                    {
                        "tracklet_a": "b",
                        "tracklet_b": "c",
                        "rank_score": 0.95,
                        "evidence_confidence": 1.0,
                        "link_kind": "continuation",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
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
                            "required_speed_after_slack_mps": 0.0,
                            "prediction_error_mean_m": 0.5,
                            "velocity_direction_cosine": 0.9,
                        },
                    }
                ),
                json.dumps(
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
                    }
                ),
                json.dumps(
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
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_identity_diagnostics(
        identities_path=final_identities,
        baseline_identities_path=baseline_identities,
        selected_backbone_links_path=selected_links,
        solver_summary_path=solver_summary,
        tracklets_path=tracklets,
        scored_links_path=scores,
        candidate_edges_path=candidates,
        output_dir=output,
    )

    assert summary["baseline_identity_count"] == 2
    assert summary["final_identity_count"] == 1
    assert summary["selected_backbone_bridge_count"] == 1
    assert summary["bridge_validation_failure_count"] == 0
    assert summary["identity_stage_completed"] is False

    for name in (
        "index.json",
        "identity_audits.jsonl",
        "bridge_audits.jsonl",
        "baseline_identity_audits.jsonl",
        "review_queue.json",
        "summary.json",
    ):
        assert (output / name).is_file()

    index = json.loads(
        (output / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["provisional"] is True
    assert index["bridge_audit_count"] == 1
