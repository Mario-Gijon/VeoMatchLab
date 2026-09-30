from __future__ import annotations

import json

from experiments.tracking.run_v1_final_diagnostics import (
    run_final_diagnostics,
)
from tracking.final.diagnostics import FinalDiagnosticsConfig


def _write_jsonl(path, rows):
    path.write_text(
        "\n".join(json.dumps(row) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def test_runner_writes_final_diagnostic_artifacts(tmp_path):
    trajectories = tmp_path / "trajectories.jsonl"
    breaks = tmp_path / "segment_breaks.jsonl"
    smoothing_summary = tmp_path / "summary.json"
    output = tmp_path / "out"

    points = []
    for index in range(3):
        frame = index * 3
        points.append(
            {
                "identity_id": "identity-001",
                "segment_id": "identity-001:segment-001",
                "segment_index": 1,
                "frame_index": frame,
                "timestamp_s": index * 0.1001,
                "measurement_x": float(frame),
                "measurement_z": 0.0,
                "smoothed_x": float(frame) + 0.1,
                "smoothed_z": 0.0,
                "measurement_confidence": 0.9,
                "measurement_sigma_m": 0.3,
                "origin": "observed",
                "fusion_mode": "single",
                "source_tracklet_ids": ["trk-a"],
                "source_detection_ids": [f"det-{frame}"],
                "source_count": 1,
                "position_spread_m": 0.0,
                "position_adjustment_m": 0.1,
                "position_std_m": 0.15,
                "vx_mps": 1.0,
                "vz_mps": 0.0,
                "speed_mps": 1.0,
                "ax_mps2": 1.0,
                "az_mps2": 0.0,
                "acceleration_mps2": 1.0,
                "velocity_std_mps": 0.9,
                "kinematics_quality": "rts_cv",
                "metadata": {"smoothing_applied": True},
            }
        )

    _write_jsonl(
        trajectories,
        [
            {
                "trajectory_version": "rts-cv-v1",
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a"],
                "tracklet_count": 1,
                "point_count": 3,
                "segment_count": 1,
                "points": points,
                "metadata": {},
            }
        ],
    )
    _write_jsonl(breaks, [])
    smoothing_summary.write_text(
        json.dumps(
            {
                "smoothing_version": "rts-cv-v1",
                "trajectory_count": 1,
                "output_point_count": 3,
                "segment_count": 1,
                "preserved_gap_break_count": 0,
            }
        ),
        encoding="utf-8",
    )

    summary = run_final_diagnostics(
        trajectories_path=trajectories,
        segment_breaks_path=breaks,
        smoothing_summary_path=smoothing_summary,
        output_dir=output,
        config=FinalDiagnosticsConfig(),
    )

    assert summary["publishable"] is True
    assert summary["blocker_count"] == 0
    assert summary["trajectory_count"] == 1
    assert summary["point_count"] == 3

    for name in (
        "identity_audits.jsonl",
        "segment_audits.jsonl",
        "point_reviews.jsonl",
        "blockers.jsonl",
        "review_queue.json",
        "summary.json",
        "index.json",
    ):
        assert (output / name).is_file()
