from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from experiments.tracking_v0.run_global_tracker import PipelineProfile
from experiments.virtual_camera_coverage import ultra_panorama_overlay
from geometry.reproduce_veo_panorama import build_panorama_maps, render, render_panorama_with_maps


def _metadata():
    identity = np.eye(4, dtype=np.float64)
    lens = SimpleNamespace(lens2camera=identity, intrinsic=np.array([[1800.0, 0.0, 1920.0], [0.0, 1800.0, 1080.0], [0.0, 0.0, 1.0]]), distortion=np.zeros(8))
    return SimpleNamespace(
        va_presentation={"left": -0.5, "right": 0.5, "top": 0.3, "bottom": -0.3},
        camera2world=identity, camera2presentation=identity, left=lens, right=lens,
    )


def test_precomputed_panorama_maps_are_bit_exact_for_multiple_frames():
    metadata = _metadata()
    frames = [np.full((64, 64, 3), value, dtype=np.uint8) for value in (17, 203)]
    maps = build_panorama_maps(metadata, 64, 64, 32, 16)
    for frame in frames:
        legacy = render(frame, metadata, 32, 16)
        cached = render_panorama_with_maps(frame, maps)
        assert np.array_equal(legacy["panorama"], cached["panorama"])


def test_ultra_maps_are_built_once_and_reused_for_multiple_frames():
    metadata = _metadata()
    frames = [np.zeros((64, 64, 3), dtype=np.uint8), np.ones((64, 64, 3), dtype=np.uint8)]
    with patch.object(ultra_panorama_overlay, "ultra_panorama_size", return_value=(32, 16)), patch.object(ultra_panorama_overlay, "build_panorama_maps", wraps=ultra_panorama_overlay.build_panorama_maps) as build:
        maps = ultra_panorama_overlay.build_ultra_panorama_maps(frames[0], metadata)
        for frame in frames:
            ultra_panorama_overlay.render_ultra_panorama(frame, metadata, maps)
    assert build.call_count == 1


def test_profile_accumulators_keep_base_and_rescue_detector_separate():
    profile = PipelineProfile(samples=3, base_detector_calls=18, rescue_detector_calls=2, measurements=66, tracks=70)
    profile.detector_base_seconds = 1.25
    profile.rescue_detector_seconds = 0.25
    values = profile.as_dict()
    assert values["samples"] == 3
    assert values["base_detector_calls"] == 18
    assert values["rescue_detector_calls"] == 2
    assert values["detector_base_seconds"] == 1.25
    assert values["rescue_detector_seconds"] == 0.25
