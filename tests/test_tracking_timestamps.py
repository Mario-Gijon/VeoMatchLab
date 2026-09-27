import pytest

from experiments.tracking.run_global_tracker import sample_timestamps
from web_app.backend.app.clip_jobs import ClipAnalysisService, SAMPLE_PERIOD_S
from web_app.backend.app.storage import WorkspaceStore


@pytest.mark.parametrize(("duration", "count"), [(10.0, 150), (5.0, 75), (3.0, 45)])
def test_exact_duration_uses_half_open_samples(duration, count):
    values = sample_timestamps(0.0, duration, 1 / 15)
    assert len(values) == count
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(duration - 1 / 15)
    assert all(value < duration for value in values)


def test_non_multiple_duration_includes_only_times_before_end():
    values = sample_timestamps(2.0, 1.01, .1)
    assert values == pytest.approx([2.0 + .1 * index for index in range(11)])
    assert all(value < 3.01 for value in values)


def test_clip_job_uses_runner_sample_count(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = store.create_match("Samples", "", None)
    clip = {"id": "clip_samples", "duration_s": 10.0}
    source = {"id": "media", "path": "/unused"}
    job = ClipAnalysisService(store).start(match, clip, source, action="all")
    # The background worker fails immediately with the deliberately incomplete
    # fixture, but construction must share the runner's count exactly.
    assert job.total_frames == len(sample_timestamps(0.0, 10.0, SAMPLE_PERIOD_S)) == 150
