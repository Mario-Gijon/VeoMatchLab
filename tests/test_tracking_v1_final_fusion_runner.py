from __future__ import annotations

import json

from experiments.tracking.run_v1_final_observation_fusion import (
    run_observation_fusion,
)
from tracking.final import ObservationFusionConfig


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_runner_writes_intermediate_fusion_artifacts(tmp_path):
    identities = tmp_path / "identities.jsonl"
    tracklets = tmp_path / "tracklets.jsonl"
    output = tmp_path / "out"

    _write_jsonl(
        identities,
        [
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a", "trk-b"],
                "canonical": True,
                "provisional": False,
            }
        ],
    )
    _write_jsonl(
        tracklets,
        [
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
            },
            {
                "tracklet_id": "trk-b",
                "observations": [
                    {
                        "detection_id": "det-b",
                        "frame_index": 0,
                        "timestamp_s": 0.0,
                        "x": 0.5,
                        "z": 0.0,
                        "confidence": 0.8,
                    }
                ],
            },
        ],
    )

    summary = run_observation_fusion(
        identities_path=identities,
        tracklets_path=tracklets,
        output_dir=output,
        config=ObservationFusionConfig(),
    )

    assert summary["trajectory_count"] == 1
    assert summary["input_observation_count"] == 2
    assert summary["output_point_count"] == 1
    assert summary["same_frame_fusion_event_count"] == 1
    assert summary["final_stage_completed"] is False

    for name in (
        "index.json",
        "trajectories.jsonl",
        "fusion_events.jsonl",
        "summary.json",
    ):
        assert (output / name).is_file()

    index = json.loads(
        (output / "index.json").read_text(
            encoding="utf-8"
        )
    )
    assert index["provisional"] is True
    assert index["final_stage_completed"] is False
