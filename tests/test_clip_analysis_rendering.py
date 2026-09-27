import inspect

from web_app.backend.app.clip_jobs import ClipAnalysisService


def test_run_all_does_not_create_tracking_debug_video_and_keeps_ultra_base():
    source = inspect.getsource(ClipAnalysisService._run_all)
    assert "TrackingDebugVideo" not in source
    assert "ultra_base.mp4" in source
    assert "build_ultra_panorama_maps" in source


def test_cached_tracking_does_not_render_ultra_base_again():
    source = inspect.getsource(ClipAnalysisService._run_cached_tracking)
    assert "render_ultra_panorama" not in source
