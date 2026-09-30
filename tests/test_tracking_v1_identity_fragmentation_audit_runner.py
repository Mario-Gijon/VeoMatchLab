from __future__ import annotations

import json

from experiments.tracking.run_v1_identity_fragmentation_audit import (
    run_fragmentation_audit,
)
from tracking.identities.fragmentation_audit import (
    FragmentationAuditConfig,
)


def test_runner_writes_fragmentation_audit_artifacts(tmp_path):
    identities = tmp_path / "identities.jsonl"
    tracklets = tmp_path / "tracklets.jsonl"
    candidates = tmp_path / "edges.jsonl"
    scores = tmp_path / "scores.jsonl"
    output = tmp_path / "out"

    identities.write_text(
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
                        "tracklet_ids": ["b"],
                    }
                ),
            ]
        )
        + "\n",
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
                                "frame_index": 1,
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
                                "frame_index": 2,
                                "timestamp_s": 1.1,
                            },
                            {
                                "frame_index": 3,
                                "timestamp_s": 2.0,
                            },
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    candidates.write_text(
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
                    "prediction_error_mean_m": 1.0,
                    "velocity_direction_cosine": 1.0,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "tracklet_a": "a",
                "tracklet_b": "b",
                "rank_score": 0.9,
                "evidence_confidence": 0.8,
                "link_kind": "continuation",
                "identity_decision": None,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_fragmentation_audit(
        identities_path=identities,
        tracklets_path=tracklets,
        candidate_edges_path=candidates,
        scored_links_path=scores,
        output_dir=output,
        config=FragmentationAuditConfig(),
    )

    assert summary["identity_count"] == 2
    assert summary["singleton_identity_count"] == 2
    assert summary["identity_stage_completed"] is False
    assert summary["identity_decisions"] is False

    for name in (
        "index.json",
        "boundary_links.jsonl",
        "identity_fragmentation.jsonl",
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
    assert index["identity_stage_completed"] is False



def test_backbone_fragmentation_runner_version():
    from experiments.tracking.run_v1_identity_fragmentation_audit import (
        SOLVER_VERSION,
    )
    assert SOLVER_VERSION == "global-backbone-v1"
