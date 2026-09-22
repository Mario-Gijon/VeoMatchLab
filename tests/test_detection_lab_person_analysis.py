import json
import time
from pathlib import Path

import pytest

from detection_lab.app import create_app
from detection_lab.person_analysis import (
    AnalysisRequest,
    PersonAnalysisService,
    cache_key,
    cache_path,
    normalize_result,
    stacked_overlay_bbox,
    visible_detections,
)
from web_app.backend.app.storage import WorkspaceStore


def request(**overrides):
    return AnalysisRequest.from_mapping("match_example", {
        "start_s": 12.5, "duration_s": 10, "sample_fps": 5,
        "inference_threshold": 0.10, **overrides,
    })


def test_person_analysis_parameter_validation():
    assert request().model_id == "rfdetr_large"
    with pytest.raises(ValueError, match="duración"):
        request(duration_s=0)
    with pytest.raises(ValueError, match="sample_fps"):
        request(sample_fps=31)
    with pytest.raises(ValueError, match="umbral"):
        request(inference_threshold=1.1)
    with pytest.raises(ValueError, match="solo"):
        request(model_id="yolo26x")


def test_cache_key_tracks_expensive_inference_inputs_but_is_stable(tmp_path):
    first = request()
    assert cache_key(first) == cache_key(request())
    assert cache_key(first) != cache_key(request(sample_fps=2))
    assert cache_key(first) != cache_key(request(inference_threshold=0.25))
    assert cache_path(first, tmp_path).parent == tmp_path / "person"


def test_normalized_result_serialization_and_display_filtering():
    result = normalize_result({
        "video": {"fps": 30, "width": 2048, "height": 2048},
        "samples": [{"timestamp_s": 12.5, "frame_index": 375, "detections": [
            {"lens": "left", "bbox": [1, 2, 3, 4], "confidence": 0.10},
            {"lens": "right", "bbox": [5, 6, 7, 8], "confidence": 0.30},
        ]}],
    }, request())
    assert json.loads(json.dumps(result))["samples"][0]["frame_index"] == 375
    assert [item["confidence"] for item in visible_detections(result["samples"][0], 0.25)] == [0.30]
    assert len(result["samples"][0]["detections"]) == 2


def test_right_lens_overlay_uses_stacked_vertical_offset():
    assert stacked_overlay_bbox({"lens": "left", "bbox": [1, 2, 3, 4]}) == [1.0, 2.0, 3.0, 4.0]
    assert stacked_overlay_bbox({"lens": "right", "bbox": [1, 2, 3, 4]}) == [1.0, 1026.0, 3.0, 1028.0]


def test_service_writes_and_reuses_inspectable_cache_without_ml(tmp_path):
    calls = []
    def fake_runner(analysis_request, video_path):
        calls.append((analysis_request, video_path))
        return {"video": {"fps": 30}, "samples": []}
    service = PersonAnalysisService(root=tmp_path, runner=fake_runner)
    started = service.start(request(), Path("/video.mp4"))
    for _ in range(50):
        status = service.status(started["job_id"])
        if status["state"] != "processing":
            break
        time.sleep(0.01)
    assert status["state"] == "complete"
    assert cache_path(request(), tmp_path).is_file()
    reused = service.start(request(), Path("/video.mp4"))
    assert reused["state"] == "cached"
    assert len(calls) == 1


def test_web_app_creation_and_person_endpoint_do_not_require_gpu(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = store.create_match("Partido", "", None)
    media_dir = store.match_dir(match["id"]) / "media"
    media_dir.mkdir()
    (media_dir / "dual_lens.mp4").write_bytes(b"placeholder")
    store.update_match(match["id"], lambda item: item.update(media=[{
        "id": "dual", "role": "dual_lens", "path": str(media_dir / "dual_lens.mp4"), "ownership": "managed",
    }]))
    service = PersonAnalysisService(root=tmp_path / "cache", runner=lambda *_: {"samples": []})
    app = create_app(store, service)
    route = next(route for route in app.routes if route.path == "/api/matches/{match_id}/person/analyses")
    response = route.endpoint(match["id"], {"start_s": 0, "duration_s": 5})
    assert response["state"] == "processing"
