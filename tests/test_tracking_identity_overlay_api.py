from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from web_app.backend.app import main
from web_app.backend.app.storage import WorkspaceStore
from web_app.backend.app.tracking_identity_overlay_api import tracking_identity_overlay_router


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _overlay_artifacts(store, match_id, clip_id, round_name="round-10-v1"):
    root = store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1"
    state = root / "diagnostics" / "global_trajectory_round_state" / round_name
    _write_jsonl(root / "detections" / "frames.jsonl", [{"detections": [
        {"detection_id": "det-1", "timestamp_s": 1.0, "source_lens": "right", "source_bbox_xyxy": [10, 1100, 30, 1140], "virtual_bbox_xyxy": [900, 1, 950, 50]},
    ]}])
    _write_jsonl(root / "tracklets" / "tracklets.jsonl", [{"tracklet_id": "trk-1", "observations": [{"detection_id": "det-1"}]}])
    _write_jsonl(state / "final_tracklet_assignments.jsonl", [{"tracklet_id": "trk-1", "identity_id": "person-0001"}])
    _write_jsonl(state / "final_identities.jsonl", [{"identity_id": "person-0001", "tracklet_ids": ["trk-1"]}])


@pytest.fixture
def overlay_context(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", WorkspaceStore(tmp_path))
    match = main.store.create_match("Overlay", "", None)
    clip = main.store.create_clip(match["id"], "source", 0.0, 1.0, "Overlay clip")
    media = {"id": "media_source", "name": "source.mp4", "path": str(tmp_path / "source.mp4"), "source": "generated", "role": "clip", "kind": "clip", "width": 2048, "height": 2048}
    main.store.attach_media(match["id"], media)
    main.store.update_clip(match["id"], clip["id"], lambda item: item.update(clip_media_id=media["id"]))
    endpoint = tracking_identity_overlay_router(lambda: main.store).routes[0].endpoint
    return endpoint, match["id"], clip["id"]


def test_overlay_resolves_numeric_latest_and_uses_source_bbox_without_lens_offset(overlay_context):
    endpoint, match_id, clip_id = overlay_context
    _overlay_artifacts(main.store, match_id, clip_id, "round-09-v1")
    _overlay_artifacts(main.store, match_id, clip_id, "round-10-v1")
    payload = endpoint(match_id, clip_id)
    assert payload["state_id"] == "round-10-v1"
    assert payload["identity_count"] == 1
    assert payload["tracks"] == [{"tracklet_id": "trk-1", "identity_id": "person-0001", "observations": [{"timestamp_s": 1.0, "bbox": [10.0, 1100.0, 30.0, 1140.0], "source_lens": "right"}]}]


def test_overlay_rejects_missing_or_incomplete_state_and_outside_paths(overlay_context, tmp_path):
    endpoint, match_id, clip_id = overlay_context
    with pytest.raises(HTTPException) as missing:
        endpoint(match_id, clip_id)
    assert missing.value.detail["code"] == "tracking_identity_overlay_missing"
    _overlay_artifacts(main.store, match_id, clip_id)
    state = main.store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "diagnostics" / "global_trajectory_round_state" / "round-10-v1"
    (state / "final_identities.jsonl").unlink()
    with pytest.raises(HTTPException) as incomplete:
        endpoint(match_id, clip_id)
    assert incomplete.value.detail["code"] == "tracking_identity_overlay_missing"
    _overlay_artifacts(main.store, match_id, clip_id)
    frames_path = main.store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "detections" / "frames.jsonl"
    outside = tmp_path / "outside-frames.jsonl"
    outside.write_text("", encoding="utf-8")
    frames_path.unlink()
    frames_path.symlink_to(outside)
    with pytest.raises(HTTPException) as outside_path:
        endpoint(match_id, clip_id)
    assert outside_path.value.detail["code"] == "tracking_identity_overlay_missing"
    with pytest.raises(HTTPException) as traversal:
        endpoint(match_id, "../outside")
    assert traversal.value.detail["code"] == "clip_not_found"
