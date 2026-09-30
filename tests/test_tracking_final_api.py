from __future__ import annotations

import json

import pytest
from fastapi import HTTPException

from web_app.backend.app import main
from web_app.backend.app.storage import WorkspaceStore
from web_app.backend.app.tracking_final_api import read_final_jsonl, tracking_final_router


def _point(frame: int, timestamp: float, segment: str) -> dict:
    return {
        "identity_id": "identity-001", "segment_id": segment,
        "frame_index": frame, "timestamp_s": timestamp,
        "measurement_x": 1.0, "measurement_z": 2.0,
        "smoothed_x": 1.1, "smoothed_z": 2.1,
        "origin": "observed", "speed_mps": 1.0,
        "acceleration_mps2": 0.0, "position_adjustment_m": 0.1,
        "position_std_m": 0.2, "kinematics_quality": "rts_cv",
    }


def _publish_final(store: WorkspaceStore, match_id: str, clip_id: str) -> None:
    root = store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "final"
    root.mkdir(parents=True)
    (root / "index.json").write_text(json.dumps({
        "publisher_version": "final-publish-v1", "canonical": True,
        "final_stage_completed": True,
    }), encoding="utf-8")
    (root / "summary.json").write_text(json.dumps({
        "publisher_version": "final-publish-v1", "canonical": True,
        "final_stage_completed": True, "point_count": 2,
    }), encoding="utf-8")
    (root / "review_queue.json").write_text(json.dumps({
        "speed_reviews": [{"identity_id": "identity-001", "frame_index": 3, "timestamp_s": 0.1}],
        "acceleration_reviews": [], "adjustment_reviews": [],
        "uncertainty_reviews": [], "short_segments": [],
    }), encoding="utf-8")
    trajectory = {"identity_id": "identity-001", "points": [_point(0, 0.0, "one"), _point(3, 0.1, "one")]}
    (root / "trajectories.jsonl").write_text(json.dumps(trajectory) + "\n", encoding="utf-8")
    (root / "segment_breaks.jsonl").write_text("", encoding="utf-8")


@pytest.fixture
def final_context(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", WorkspaceStore(tmp_path))
    match = main.store.create_match("Final", "", None)
    clip = main.store.create_clip(match["id"], "source", 0.0, 1.0, "Final clip")
    router = tracking_final_router(lambda: main.store)
    endpoints = {route.path.rsplit("/final", 1)[-1] or "/": route.endpoint for route in router.routes}
    return endpoints, match["id"], clip["id"]


def test_final_tracking_endpoints_serve_only_published_final(final_context):
    endpoints, match_id, clip_id = final_context
    _publish_final(main.store, match_id, clip_id)

    final = endpoints["/"](match_id, clip_id)
    assert final["trajectories"][0]["points"][1]["smoothed_x"] == 1.1
    assert endpoints["/summary"](match_id, clip_id)["point_count"] == 2
    assert endpoints["/reviews"](match_id, clip_id)["speed_reviews"][0]["frame_index"] == 3


def test_final_tracking_routes_are_registered_on_the_application():
    paths = main.app.openapi()["paths"]
    assert "/api/tracking/matches/{match_id}/clips/{clip_id}/final" in paths
    assert "/api/tracking/matches/{match_id}/clips/{clip_id}/final/summary" in paths
    assert "/api/tracking/matches/{match_id}/clips/{clip_id}/final/reviews" in paths


def test_final_tracking_is_clear_when_final_is_not_published(final_context):
    endpoints, match_id, clip_id = final_context
    with pytest.raises(HTTPException) as missing:
        endpoints["/"](match_id, clip_id)
    assert missing.value.status_code == 404
    assert missing.value.detail["code"] == "tracking_final_missing"


def test_final_tracking_rejects_noncanonical_or_invalid_jsonl(final_context):
    endpoints, match_id, clip_id = final_context
    _publish_final(main.store, match_id, clip_id)
    root = main.store.asset_dir(match_id, "clips") / clip_id / "tracking" / "v1" / "final"
    (root / "index.json").write_text(json.dumps({"publisher_version": "draft", "canonical": False}), encoding="utf-8")
    with pytest.raises(HTTPException) as not_canonical:
        endpoints["/"](match_id, clip_id)
    assert not_canonical.value.status_code == 409
    (root / "trajectories.jsonl").write_text("[]\n", encoding="utf-8")
    with pytest.raises(HTTPException) as invalid:
        read_final_jsonl(root / "trajectories.jsonl")
    assert invalid.value.status_code == 422
