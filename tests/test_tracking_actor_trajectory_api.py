from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from web_app.backend.app import main
from web_app.backend.app.storage import WorkspaceStore
from web_app.backend.app.tracking_actor_trajectory_api import tracking_actor_trajectory_router


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _endpoint_artifact(store, match_id, clip_id):
    root = store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "diagnostics" / "actor_trajectory_reconstruction" / "actor-trajectory-reconstruction-v1"
    anchor = {"detection_id": "det-left", "frame_index": 0, "timestamp_s": 0.0, "representative_view": "tight_near_c1", "virtual_bbox_xyxy": [1, 2, 3, 4], "source_lens": "left", "source_bbox_xyxy": [10, 20, 30, 40]}
    right = {**anchor, "detection_id": "det-right", "frame_index": 3, "timestamp_s": 0.1, "representative_view": "tight_near_c2", "source_lens": "right", "source_bbox_xyxy": [50, 60, 70, 80]}
    rows = [
        {"actor_id": "actor-0001", "tracklet_ids": ["trk-1"], "segment_count": 1, "points": [
            {"frame_index": 0, "timestamp_s": 0.0, "x": 1.0, "z": -2.0, "origin": "observed", "segment_id": "actor-0001:segment-001", "source_tracklet_ids": ["trk-1"], "visual_anchor": anchor, "visual_interpolation": None},
            {"frame_index": 1, "timestamp_s": 1 / 30, "x": 1.1, "z": -2.1, "origin": "interpolated", "segment_id": "actor-0001:segment-001", "source_tracklet_ids": ["trk-1"], "visual_anchor": None, "visual_interpolation": {"left": anchor, "right": right, "alpha": 1 / 3, "visual_interpolation_allowed": True, "source_bbox_interpolation_allowed": False}},
        ]},
        {"actor_id": "actor-0002", "tracklet_ids": ["trk-2"], "segment_count": 1, "points": [{"frame_index": 0, "timestamp_s": 0.0, "x": -1.0, "z": 2.0, "origin": "observed", "segment_id": "actor-0002:segment-001", "source_tracklet_ids": ["trk-2"], "visual_anchor": anchor, "visual_interpolation": None}]},
    ]
    _write_jsonl(root / "trajectories.jsonl", rows)
    _write_jsonl(root / "segments.jsonl", [])
    _write_jsonl(root / "gap_events.jsonl", [])
    root.joinpath("index.json").write_text("{}", encoding="utf-8")
    root.joinpath("summary.json").write_text(json.dumps({"trajectory_version": "actor-trajectory-reconstruction-v1", "actor_count": 2, "display_point_count": 3, "observed_display_point_count": 2, "interpolated_display_point_count": 1, "segment_count": 2, "sampling_cadence": {"frame_step": 1, "time_step_s": 1 / 30, "estimated_fps": 30.0}}), encoding="utf-8")
    return root


@pytest.fixture
def trajectory_context(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", WorkspaceStore(tmp_path))
    match = main.store.create_match("Trajectories", "", None)
    clip = main.store.create_clip(match["id"], "source", 0.0, 1.0, "Trajectory clip")
    return tracking_actor_trajectory_router(lambda: main.store).routes[0].endpoint, match["id"], clip["id"]


def test_actor_trajectory_endpoint_reads_only_artifact_rows_and_visual_evidence(trajectory_context):
    endpoint, match_id, clip_id = trajectory_context
    _endpoint_artifact(main.store, match_id, clip_id)
    payload = endpoint(match_id, clip_id)
    assert payload["trajectory_version"] == "actor-trajectory-reconstruction-v1"
    assert payload["provisional"] is True
    assert payload["actor_count"] == len(payload["actors"]) == 2
    assert payload["display_point_count"] == 3
    assert payload["observed_point_count"] == 2
    assert payload["interpolated_point_count"] == 1
    assert all(point["origin"] in {"observed", "interpolated"} and isinstance(point["x"], float) and isinstance(point["z"], float) for actor in payload["actors"] for point in actor["points"])
    interpolated = payload["actors"][0]["points"][1]
    assert interpolated["visual_interpolation"]["left"]["representative_view"] == "tight_near_c1"
    assert interpolated["visual_interpolation"]["right"]["representative_view"] == "tight_near_c2"


def test_actor_trajectory_endpoint_rejects_missing_and_invalid_rows(trajectory_context):
    endpoint, match_id, clip_id = trajectory_context
    with pytest.raises(HTTPException) as missing:
        endpoint(match_id, clip_id)
    assert missing.value.detail["code"] == "actor_trajectory_missing"
    root = _endpoint_artifact(main.store, match_id, clip_id)
    rows = [json.loads(line) for line in root.joinpath("trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[0]["points"][1]["frame_index"] = 0
    _write_jsonl(root / "trajectories.jsonl", rows)
    with pytest.raises(HTTPException) as invalid:
        endpoint(match_id, clip_id)
    assert invalid.value.detail["code"] == "actor_trajectory_invalid"


def test_actor_trajectory_endpoint_never_promotes_non_roster_rows_to_actors(trajectory_context):
    endpoint, match_id, clip_id = trajectory_context
    root = _endpoint_artifact(main.store, match_id, clip_id)
    rows = [json.loads(line) for line in root.joinpath("trajectories.jsonl").read_text(encoding="utf-8").splitlines()]
    rows[1]["actor_id"] = "NON_ROSTER"
    _write_jsonl(root / "trajectories.jsonl", rows)
    with pytest.raises(HTTPException) as invalid:
        endpoint(match_id, clip_id)
    assert invalid.value.detail["code"] == "actor_trajectory_invalid"
