from pathlib import Path

import numpy as np
import pytest

from geometry.panorama_ground import ground_intersection_for_pixel
from geometry.panorama_landmarks import field_landmarks
from geometry.raw_ground import inverse_source_compression, raw_pixel_to_ground
from geometry.reproduce_veo_panorama import apply_source_compression, project_world_target_to_lens
from geometry.veo_metadata import load_veo_metadata

CALIBRATIONS = sorted((Path(__file__).resolve().parent / "fixtures/calibrations").glob("*.veo"))


def test_source_mapping_inverse_including_piecewise_boundary():
    for value in [0., .1, 19 / 45 - 1e-9, 19 / 45, 19 / 45 + 1e-9, .9, 1.]:
        compressed = apply_source_compression(np.array([value]))[0]
        assert inverse_source_compression(compressed) == pytest.approx(value, abs=1e-14)


@pytest.mark.parametrize("path", CALIBRATIONS, ids=lambda path: path.stem[-8:])
@pytest.mark.parametrize("lens_name", ["left", "right"])
def test_known_field_landmark_forward_raw_inverse_round_trip(path, lens_name):
    metadata = load_veo_metadata(path)
    lens = metadata.lens(lens_name)
    tested = 0
    for landmark in field_landmarks(metadata.field_length, metadata.field_width).values():
        target = np.array([[landmark.x, 0, landmark.z, 1.]])
        direction = np.linalg.inv(metadata.camera2world @ lens.lens2camera) @ target[0]
        x, y, valid = project_world_target_to_lens(target, lens, metadata.camera2world, 2048, 2048, lens_name == "left")
        if not valid[0] or direction[2] <= 0:
            continue
        # Forward renderer returns float32 maps: tolerance includes that rounding.
        point = raw_pixel_to_ground(metadata, float(x[0]), float(y[0]))
        np.testing.assert_allclose([point.world_x, point.world_z], [landmark.x, landmark.z], atol=6e-5, rtol=0)
        tested += 1
    assert tested >= 7


@pytest.mark.parametrize("path", CALIBRATIONS, ids=lambda path: path.stem[-8:])
def test_panorama_ground_and_raw_lens_rays_agree(path):
    metadata = load_veo_metadata(path)
    counts = {"left": 0, "right": 0}
    for u in range(80, 1920, 180):
        for v in range(350, 1050, 140):
            pano = ground_intersection_for_pixel(metadata, u, v, 1920, 1080)
            if not pano.inside_field:
                continue
            target = np.array([[pano.world_x, 0, pano.world_z, 1.]])
            for lens in (metadata.left, metadata.right):
                if (np.linalg.inv(metadata.camera2world @ lens.lens2camera) @ target[0])[2] <= 0:
                    continue
                x, y, valid = project_world_target_to_lens(target, lens, metadata.camera2world, 2048, 2048, lens.name == "left")
                if valid[0]:
                    raw = raw_pixel_to_ground(metadata, float(x[0]), float(y[0]))
                    np.testing.assert_allclose([raw.world_x, raw.world_z], [pano.world_x, pano.world_z], atol=1e-4, rtol=0)
                    counts[lens.name] += 1
    assert min(counts.values()) > 5


@pytest.mark.parametrize("x,y", [(-1, 0), (2048, 0), (0, 2048), (float('nan'), 0)])
def test_raw_invalid_coordinates_rejected(x, y):
    with pytest.raises(ValueError):
        raw_pixel_to_ground(load_veo_metadata(CALIBRATIONS[0]), x, y)
