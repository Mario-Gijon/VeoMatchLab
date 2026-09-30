from __future__ import annotations

import json

from experiments.tracking.run_v1_identity_visual_audit import (
    run_visual_identity_audit,
)
from tracking.identities.visual_audit import (
    VisualAuditConfig,
)


def test_runner_writes_diagnostics_without_video(tmp_path):
    identities = tmp_path / "identities.jsonl"
    tracklets = tmp_path / "tracklets.jsonl"
    audits = tmp_path / "audits.jsonl"
    scores = tmp_path / "scores.jsonl"
    output = tmp_path / "out"

    identities.write_text(
        json.dumps(
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a", "trk-b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    tracklets.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "tracklet_id": "trk-a",
                        "observations": [
                            {
                                "frame_index": 0,
                                "timestamp_s": 0.0,
                                "x": 0.0,
                                "z": 0.0,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ),
                json.dumps(
                    {
                        "tracklet_id": "trk-b",
                        "observations": [
                            {
                                "frame_index": 3,
                                "timestamp_s": 0.3,
                                "x": 1.0,
                                "z": 0.0,
                                "confidence": 0.9,
                            }
                        ],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audits.write_text(
        json.dumps(
            {
                "identity_id": "identity-001",
                "flags": ["contains_temporal_gaps"],
                "backbone": {
                    "edges": [
                        {
                            "tracklet_a": "trk-a",
                            "tracklet_b": "trk-b",
                        }
                    ]
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    scores.write_text(
        json.dumps(
            {
                "tracklet_a": "trk-a",
                "tracklet_b": "trk-b",
                "rank_score": 0.8,
                "evidence_confidence": 0.7,
                "link_kind": "continuation",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_visual_identity_audit(
        identities_path=identities,
        tracklets_path=tracklets,
        identity_audits_path=audits,
        scored_links_path=scores,
        output_dir=output,
        config=VisualAuditConfig(),
        render_video=False,
    )

    assert summary["identity_count"] == 1
    assert summary["tracklet_count"] == 2
    assert summary["transition_count"] == 1
    assert summary["identity_stage_completed"] is False
    assert summary["video"] is None
    assert (output / "index.json").is_file()
    assert (output / "transitions.jsonl").is_file()
    assert (output / "review_queue.json").is_file()
    assert (output / "summary.json").is_file()

    index = json.loads(
        (output / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["provisional"] is True
    assert index["video"] is None



def test_backbone_runner_versions_are_selected():
    from experiments.tracking.run_v1_identity_visual_audit import (
        IDENTITY_AUDIT_VERSION,
        SOLVER_VERSION,
    )
    assert SOLVER_VERSION == "global-backbone-v1"
    assert IDENTITY_AUDIT_VERSION == "identity-backbone-audit-v1"
