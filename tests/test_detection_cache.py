from pathlib import Path
from tracking_v0.detection_cache import BaseDetectionCache

class Detection:
    def __init__(self, confidence): self.xyxy=(1., 2., 3., 4.); self.confidence=confidence

def config(**extra):
    return {"video": "a", "calibration": "b", "weights": "c", "width": 1280, "height": 720, "start_s": 0., "duration_s": 10., **extra}

def test_miss_then_hit_filters_without_mutating_cache(tmp_path):
    cache = BaseDetectionCache(tmp_path, config(), .40)
    assert not cache.hit
    assert [row["confidence"] for row in cache.candidates(1, .0, "cam", lambda: [Detection(.35), Detection(.46)])] == [.46]
    cache.commit()
    saved = (tmp_path / "base_candidates.jsonl").read_text()
    cached = BaseDetectionCache(tmp_path, config(), .30)
    assert cached.hit
    assert [row["confidence"] for row in cached.candidates(1, .0, "cam", lambda: (_ for _ in ()).throw(AssertionError()))] == [.35, .46]
    assert (tmp_path / "base_candidates.jsonl").read_text() == saved

def test_threshold_and_temporal_fingerprint_rules(tmp_path):
    first = BaseDetectionCache(tmp_path, config(), .40); first.candidates(1, 0., "cam", lambda: []); first.commit()
    assert BaseDetectionCache(tmp_path, config(), .50).hit
    assert not BaseDetectionCache(tmp_path, config(), .10).hit
    assert not BaseDetectionCache(tmp_path, config(start_s=1.), .40).hit
    assert not BaseDetectionCache(tmp_path, config(duration_s=11.), .40).hit

def test_missing_metadata_or_corrupt_rows_are_misses(tmp_path):
    cache = BaseDetectionCache(tmp_path, config(), .40); cache.candidates(1, 0., "cam", lambda: []); cache.commit()
    (tmp_path / "metadata.json").unlink()
    assert not BaseDetectionCache(tmp_path, config(), .40).hit
