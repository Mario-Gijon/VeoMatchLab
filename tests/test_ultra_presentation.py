import inspect

from experiments.tracking.run_global_tracker import sample_timestamps
from web_app.backend.app.clip_jobs import ClipAnalysisService
from web_app.backend.app.ultra_presentation import source_frame_indices


def test_ultra_source_frame_cadence_is_independent_from_15hz_detection_sampling():
    ultra_frames = list(source_frame_indices(0.0, 10.0, 30.0))
    detector_samples = sample_timestamps(0.0, 10.0, 1 / 15)
    assert ultra_frames == list(range(300))
    assert len(detector_samples) == 150


def test_ultra_presentation_respects_available_physical_source_frames():
    assert list(source_frame_indices(0.0, 10.0, 29.97, frame_count=299)) == list(range(299))


def test_full_analysis_uses_a_separate_ultra_presentation_pass():
    source = inspect.getsource(ClipAnalysisService._run_all)
    assert "render_ultra_presentation(" in source
    assert "run_tracker(args, measurement_callback=" in source
