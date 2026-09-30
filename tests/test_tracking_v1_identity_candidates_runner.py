from __future__ import annotations

import json

from experiments.tracking.run_v1_identity_candidates import (
    run_candidate_diagnostics,
)
from tracking.identities.candidates import PhysicalCompatibilityConfig


def test_runner_writes_complete_diagnostics(tmp_path):
    tracklets = tmp_path / "tracklets.jsonl"
    appearance = tmp_path / "appearance.jsonl"
    output = tmp_path / "out"

    rows = [
        {
            "tracklet_id": "trk-a",
            "observations": [
                {
                    "detection_id": "det-a",
                    "frame_index": 0,
                    "timestamp_s": 0.0,
                    "x": 0.0,
                    "z": 0.0,
                    "confidence": 0.9,
                }
            ],
            "metadata": {},
        },
        {
            "tracklet_id": "trk-b",
            "observations": [
                {
                    "detection_id": "det-b",
                    "frame_index": 3,
                    "timestamp_s": 0.1,
                    "x": 0.5,
                    "z": 0.0,
                    "confidence": 0.9,
                }
            ],
            "metadata": {},
        },
        {
            "tracklet_id": "trk-c",
            "observations": [
                {
                    "detection_id": "det-c",
                    "frame_index": 0,
                    "timestamp_s": 0.0,
                    "x": 20.0,
                    "z": 0.0,
                    "confidence": 0.9,
                }
            ],
            "metadata": {},
        },
    ]
    tracklets.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    appearance.write_text(
        json.dumps(
            {
                "tracklet_a": "trk-a",
                "tracklet_b": "trk-b",
                "aggregate_cosine": 0.97,
                "reliability": 0.8,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = run_candidate_diagnostics(
        tracklets_path=tracklets,
        appearance_edges_path=appearance,
        output_dir=output,
        config=PhysicalCompatibilityConfig(),
    )

    assert summary["node_count"] == 3
    assert summary["pair_count"] == 3
    assert summary["appearance_attached_count"] == 1
    assert summary["identity_decisions"] is False
    assert (output / "index.json").is_file()
    assert (output / "nodes.jsonl").is_file()
    assert (output / "edges.jsonl").is_file()
    assert (output / "summary.json").is_file()

    index = json.loads((output / "index.json").read_text())
    assert index["hard_constraints_only_from_physics"] is True
    assert index["identity_decisions"] is False
