from __future__ import annotations

import json

from experiments.tracking.run_v1_final_smoothing import (
    run_temporal_smoothing,
)
from tracking.final import (
    TemporalSmoothingConfig,
)


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + "\n",
        encoding="utf-8",
    )


def test_runner_writes_provisional_smoothing_artifacts(
    tmp_path,
):
    trajectories = (
        tmp_path / "trajectories.jsonl"
    )
    output = tmp_path / "out"

    points = []
    for index in range(4):
        frame = index * 3
        points.append(
            {
                "identity_id": "identity-001",
                "frame_index": frame,
                "timestamp_s": (
                    index * 0.1001
                ),
                "x": index * 0.1,
                "z": 0.0,
                "confidence": 0.9,
                "origin": "observed",
                "fusion_mode": "single",
                "source_tracklet_ids": [
                    "trk-a"
                ],
                "source_detection_ids": [
                    f"det-{frame}"
                ],
                "source_count": 1,
                "position_spread_m": 0.0,
                "metadata": {
                    "fusion_conflict": False
                },
            }
        )

    _write_jsonl(
        trajectories,
        [
            {
                "trajectory_version": (
                    "safe-gap-v1"
                ),
                "identity_id": (
                    "identity-001"
                ),
                "tracklet_ids": ["trk-a"],
                "points": points,
                "metadata": {},
            }
        ],
    )

    summary = run_temporal_smoothing(
        trajectories_path=trajectories,
        output_dir=output,
        config=TemporalSmoothingConfig(),
    )

    assert summary["trajectory_count"] == 1
    assert summary["input_point_count"] == 4
    assert summary["output_point_count"] == 4
    assert summary["segment_count"] == 1
    assert summary["final_stage_completed"] is False

    for name in (
        "index.json",
        "trajectories.jsonl",
        "segment_breaks.jsonl",
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
