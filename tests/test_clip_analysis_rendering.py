import inspect

from web_app.backend.app.clip_jobs import ClipAnalysisService, ClipJob
from web_app.backend.app.match_data import Provenance
from web_app.backend.app.storage import WorkspaceStore


def test_run_all_does_not_create_tracking_debug_video_and_keeps_ultra_base():
    source = inspect.getsource(ClipAnalysisService._run_all)
    assert "TrackingDebugVideo" not in source
    assert "ultra_base.mp4" in source
    assert "render_ultra_presentation" in source
    assert "frame_callback=" not in source


def test_cached_tracking_does_not_render_ultra_base_again():
    source = inspect.getsource(ClipAnalysisService._run_cached_tracking)
    assert "render_ultra_panorama" not in source
    assert "TrackingDebugVideo" not in source
    assert "read_frame" not in source
    assert "render_ultra_presentation" not in source


def test_full_and_cached_tracking_share_the_v2_postprocess_helper():
    assert "self._postprocess_tracking(" in inspect.getsource(ClipAnalysisService._run_all)
    assert "self._postprocess_tracking(" in inspect.getsource(ClipAnalysisService._run_cached_tracking)


def test_cached_tracking_uses_only_cached_measurements_and_preserves_existing_artifacts(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = store.create_match("Cached", "", None)
    clip = store.create_clip(match["id"], "source", 0.0, 0.2, "Cached clip")
    service = ClipAnalysisService(store)
    rows = [
        {"timestamp_s": 0.0, "frame_index": 0, "measurements": [{"measurement_id": "0:1", "frame_index": 0, "timestamp": 0.0, "x": 1.0, "z": 2.0, "confidence": 0.9, "metadata": {}}]},
        {"timestamp_s": 1 / 15, "frame_index": 1, "measurements": [{"measurement_id": "1:1", "frame_index": 1, "timestamp": 1 / 15, "x": 1.1, "z": 2.0, "confidence": 0.9, "metadata": {}}]},
    ]
    detection = service.artifacts.write_records(match["id"], "detections.person_measurements", rows, Provenance(source="our_analysis", run_id="detection", parser_version="1"), source_media_ids=["source"])
    store.update_clip(match["id"], clip["id"], lambda value: value.update(analysis={"detection_artifact_id": detection["id"], "association_artifact_id": "association_old", "ultra_base": "clips/old/ultra_base.mp4"}))
    job = ClipJob("job_cached", match["id"], clip["id"], "tracking", total_frames=2)
    service._run_cached_tracking(job, store.match(match["id"]), store.clip(match["id"], clip["id"]), {"id": "source", "path": str(tmp_path / "does-not-exist.mp4")})
    updated = store.clip(match["id"], clip["id"])
    assert updated["analysis"]["detection_artifact_id"] == detection["id"]
    assert updated["analysis"]["association_artifact_id"] == "association_old"
    assert updated["analysis"]["ultra_base"] == "clips/old/ultra_base.mp4"
    assert "debug_video" not in updated["analysis"]
    assert not (tmp_path / "assets" / match["id"] / "clips" / clip["id"] / job.job_id / "tracking_debug.mp4").exists()
    output_dir = store.asset_dir(match["id"], "clips") / clip["id"] / job.job_id
    assert (output_dir / "tracking_frames.jsonl").is_file()
    assert (output_dir / "tracking_frames_final.jsonl").is_file()
    assert (output_dir / "tracking_postprocess_report.json").is_file()
    tracks = list(service.artifacts.read_window(match["id"], updated["analysis"]["tracking_artifact_id"]))
    assert len(tracks) == len(rows)
