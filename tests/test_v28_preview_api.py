import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from web_app.backend.app import main
from web_app.backend.app.artifacts import LocalArtifactStore
from web_app.backend.app.match_data import Provenance
from web_app.backend.app.storage import WorkspaceStore


def test_debug_frames_serves_v28_preview_without_changing_actual_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", WorkspaceStore(tmp_path))
    match = main.store.create_match("Preview", "", None)
    clip = main.store.create_clip(match["id"], "source", 0.0, 10.0, "Preview clip")
    artifacts = LocalArtifactStore(main.store)
    provenance = Provenance(source="our_analysis", run_id="test", parser_version="1")
    actual = artifacts.write_records(match["id"], "tracking.frames", [
        {"timestamp_s": 0.0, "frame_index": 0, "tracks": [{"track_id": 99, "x": 1.0, "z": 2.0}]},
    ], provenance)
    association = artifacts.write_records(match["id"], "association.frames", [
        {"timestamp_s": 0.0, "frame_index": 0, "observations": [], "measurements": []},
    ], provenance)
    main.store.update_clip(match["id"], clip["id"], lambda value: value.update(analysis={
        "tracking_artifact_id": actual["id"], "association_artifact_id": association["id"],
    }))
    monkeypatch.setattr(main, "cached_clip_view_geometry", lambda *_args: (None, [
        SimpleNamespace(name="center", longitude=0.0, latitude=0.0, vfov=1.0),
    ], {}))
    monkeypatch.setattr(main, "ROOT", tmp_path)
    preview_path = tmp_path / "experiments" / "tracking" / "reid_similarity_debug" / "previews" / clip["id"] / "tracking_frames_v28_preview.jsonl"
    preview_path.parent.mkdir(parents=True)
    preview_path.write_text(json.dumps({"timestamp_s": 0.0, "frame_index": 0, "tracks": [{"track_id": 1, "x": 3.0, "z": 4.0}]}) + "\n", encoding="utf-8")
    v8_path = main.store.asset_dir(match["id"], "clips") / clip["id"] / "tracking" / "v8" / "preview_frames.jsonl"
    v8_path.parent.mkdir(parents=True)
    v8_person = {"canonical_id": 8, "track_id": 8, "role": "referee", "x": 5.0, "z": 6.0, "position_kind": "predicted", "lifecycle": "active"}
    v8_path.write_text(json.dumps({"timestamp_s": 0.0, "frame_index": 0, "tracks": {"person": [v8_person]}}) + "\n", encoding="utf-8")

    actual_response = main.clip_debug_frames(match["id"], clip["id"])
    preview_response = main.clip_debug_frames(match["id"], clip["id"], "v28_preview")
    v8_response = main.clip_debug_frames(match["id"], clip["id"], "v8_canonical")

    assert actual_response["tracks"][0]["tracks"][0]["track_id"] == 99
    assert preview_response["tracks"][0]["tracks"][0]["track_id"] == 1
    assert v8_response["tracks"][0]["tracks"] == [v8_person]
    assert preview_response["association"] == actual_response["association"]
    assert preview_response["camera_specs"] == actual_response["camera_specs"]
    with pytest.raises(HTTPException, match="tracking_mode") as invalid:
        main.clip_debug_frames(match["id"], clip["id"], "unknown")
    assert invalid.value.status_code == 422
    assert invalid.value.detail["code"] == "invalid_tracking_mode"

    preview_path.unlink()
    with pytest.raises(HTTPException) as missing:
        main.clip_debug_frames(match["id"], clip["id"], "v28_preview")
    assert missing.value.status_code == 404
    assert missing.value.detail["code"] == "clip_v28_preview_missing"
