import json

from tracking.models import PositionKind, TrackLifecycle, TrackSample
from web_app.backend.app.tracking_postprocess import (
    final_trajectories_to_rows,
    json_safe,
    postprocess_tracking_rows,
    raw_rows_to_trajectories,
    write_postprocess_outputs,
)


def _raw_rows():
    return [
        {"frame_index": frame, "timestamp": index / 15, "timestamp_s": index / 15,
         "tracks": [{"track_id": 7, "x": 10.0 + index * .1, "z": 20.0,
                     "vx": 1.5, "vz": 0.0, "lifecycle": "active",
                     "position_kind": "observed", "measurement_id": f"m-{frame}"}]}
        for index, frame in enumerate((100, 102, 104, 106))
    ]


def test_raw_frame_rows_reconstruct_source_trajectories():
    trajectories = raw_rows_to_trajectories(_raw_rows())
    assert list(trajectories) == [7]
    assert [sample.frame_index for sample in trajectories[7]] == [100, 102, 104, 106]
    assert [sample.timestamp for sample in trajectories[7]] == [0.0, 1 / 15, 2 / 15, 3 / 15]
    assert all(sample.position_kind is PositionKind.OBSERVED for sample in trajectories[7])


def test_final_samples_become_sorted_frame_rows_and_keep_interpolation():
    trajectories = {7: (
        TrackSample(7, 100, 0.0, 1.0, 2.0, 1.0, 0.0, TrackLifecycle.ACTIVE, PositionKind.OBSERVED, "m-100", None),
        TrackSample(7, 101, 1 / 30, 1.03, 2.0, 1.0, 0.0, TrackLifecycle.ACTIVE, PositionKind.INTERPOLATED, None, None),
        TrackSample(7, 102, 2 / 30, 1.06, 2.0, 1.0, 0.0, TrackLifecycle.ACTIVE, PositionKind.OBSERVED, "m-102", None),
    )}
    rows = final_trajectories_to_rows(trajectories)
    assert [row["frame_index"] for row in rows] == [100, 101, 102]
    assert [row["timestamp_s"] for row in rows] == [0.0, 1 / 30, 2 / 30]
    assert rows[1]["tracks"] == [{
        "track_id": 7, "x": 1.03, "z": 2.0, "vx": 1.0, "vz": 0.0,
        "lifecycle": "active", "position_kind": "interpolated", "measurement_id": None,
    }]


def test_postprocess_writes_final_separately_from_raw_diagnostic(tmp_path):
    raw_rows = _raw_rows()
    final_rows, report = postprocess_tracking_rows(raw_rows)
    assert report["summary"]["raw_track_count"] == 1
    assert report["summary"]["final_track_count"] != 23
    assert any(track["position_kind"] == "interpolated" for row in final_rows for track in row["tracks"])

    raw_path = tmp_path / "tracking_frames.jsonl"
    raw_path.write_text("".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8")
    write_postprocess_outputs(tmp_path, final_rows, report)

    assert [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines()] == raw_rows
    assert (tmp_path / "tracking_frames_final.jsonl").is_file()
    written_report = json.loads((tmp_path / "tracking_postprocess_report.json").read_text(encoding="utf-8"))
    assert set(written_report) >= {"identity_resolution", "participant_validation", "smoothing", "interpolation"}
    assert json_safe({"bad": float("nan")}) == {"bad": None}
