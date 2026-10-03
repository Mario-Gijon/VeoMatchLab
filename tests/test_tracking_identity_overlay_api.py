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


def _fixed_roster_artifacts(store, match_id, clip_id, *, actor_id="actor-0007", assignment_actor_id=None, assigned_tracklet="trk-1", unassigned_tracklet="trk-2"):
    root = store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1"
    roster = root / "diagnostics" / "fixed_actor_roster" / "fixed-actor-roster-v1"
    _write_jsonl(root / "detections" / "frames.jsonl", [{"detections": [
        {"detection_id": "det-1", "timestamp_s": 1.0, "source_lens": "right", "source_bbox_xyxy": [10, 1100, 30, 1140], "virtual_bbox_xyxy": [900, 1, 950, 50]},
        {"detection_id": "det-2", "timestamp_s": 1.1, "source_lens": "left", "source_bbox_xyxy": [40, 50, 70, 100], "virtual_bbox_xyxy": [800, 2, 850, 60]},
    ]}])
    _write_jsonl(root / "tracklets" / "tracklets.jsonl", [
        {"tracklet_id": "trk-1", "observations": [{"detection_id": "det-1"}]},
        {"tracklet_id": "trk-2", "observations": [{"detection_id": "det-2"}]},
    ])
    _write_jsonl(roster / "actors.jsonl", [{"actor_id": actor_id}])
    _write_jsonl(roster / "tracklet_assignments.jsonl", [{"tracklet_id": assigned_tracklet, "actor_id": assignment_actor_id or actor_id}])
    _write_jsonl(roster / "unassigned_tracklets.jsonl", [{"tracklet_id": unassigned_tracklet}])


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
    payload = endpoint(match_id, clip_id, source="round_state")
    assert payload["state_id"] == "round-10-v1"
    assert payload["source_kind"] == "round_state"
    assert payload["identity_count"] == 1
    assert payload["tracks"] == [{"tracklet_id": "trk-1", "identity_id": "person-0001", "assignment_status": "assigned", "observations": [{"timestamp_s": 1.0, "bbox": [10.0, 1100.0, 30.0, 1140.0], "source_lens": "right"}]}]


def test_overlay_rejects_missing_or_incomplete_state_and_outside_paths(overlay_context, tmp_path):
    endpoint, match_id, clip_id = overlay_context
    with pytest.raises(HTTPException) as missing:
        endpoint(match_id, clip_id, source="round_state")
    assert missing.value.detail["code"] == "tracking_identity_overlay_missing"
    _overlay_artifacts(main.store, match_id, clip_id)
    state = main.store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "diagnostics" / "global_trajectory_round_state" / "round-10-v1"
    (state / "final_identities.jsonl").unlink()
    with pytest.raises(HTTPException) as incomplete:
        endpoint(match_id, clip_id, source="round_state")
    assert incomplete.value.detail["code"] == "tracking_identity_overlay_missing"
    _overlay_artifacts(main.store, match_id, clip_id)
    frames_path = main.store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "detections" / "frames.jsonl"
    outside = tmp_path / "outside-frames.jsonl"
    outside.write_text("", encoding="utf-8")
    frames_path.unlink()
    frames_path.symlink_to(outside)
    with pytest.raises(HTTPException) as outside_path:
        endpoint(match_id, clip_id, source="round_state")
    assert outside_path.value.detail["code"] == "tracking_identity_overlay_missing"
    with pytest.raises(HTTPException) as traversal:
        endpoint(match_id, "../outside", source="round_state")
    assert traversal.value.detail["code"] == "clip_not_found"


def test_fixed_roster_is_default_and_keeps_unassigned_outside_the_actor_roster(overlay_context):
    endpoint, match_id, clip_id = overlay_context
    _fixed_roster_artifacts(main.store, match_id, clip_id)
    payload = endpoint(match_id, clip_id)
    assert payload["source_id"] == "fixed-actor-roster-v1"
    assert payload["source_kind"] == "fixed_roster"
    assert payload["identity_count"] == 1
    assert payload["assigned_tracklet_count"] == 1
    assert payload["unassigned_tracklet_count"] == 1
    assert payload["tracks"] == [
        {"tracklet_id": "trk-1", "identity_id": "actor-0007", "assignment_status": "assigned", "observations": [{"timestamp_s": 1.0, "bbox": [10.0, 1100.0, 30.0, 1140.0], "source_lens": "right"}]},
        {"tracklet_id": "trk-2", "identity_id": "UNASSIGNED", "assignment_status": "unassigned", "observations": [{"timestamp_s": 1.1, "bbox": [40.0, 50.0, 70.0, 100.0], "source_lens": "left"}]},
    ]


@pytest.mark.parametrize(("assignment_actor_id", "assigned_tracklet", "unassigned_tracklet"), [
    ("actor-missing", "trk-1", "trk-2"),
    (None, "trk-1", "trk-1"),
    (None, "trk-missing", "trk-2"),
])
def test_fixed_roster_rejects_invalid_assignments(overlay_context, assignment_actor_id, assigned_tracklet, unassigned_tracklet):
    endpoint, match_id, clip_id = overlay_context
    _fixed_roster_artifacts(main.store, match_id, clip_id, assignment_actor_id=assignment_actor_id, assigned_tracklet=assigned_tracklet, unassigned_tracklet=unassigned_tracklet)
    with pytest.raises(HTTPException) as invalid:
        endpoint(match_id, clip_id)
    assert invalid.value.detail["code"] == "tracking_identity_overlay_invalid"
