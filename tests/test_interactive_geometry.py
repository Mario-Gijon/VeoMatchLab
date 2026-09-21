"""Cross-language checks against the untouched authoritative Panorama geometry."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from geometry.panorama_ground import ground_intersection_from_uv, panorama_ray
from geometry.veo_metadata import load_veo_metadata
from web_app.backend.app.view_geometry import view_context

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', sorted((ROOT / 'tests/fixtures/calibrations').glob('*.veo')))
def test_interactive_rays_match_panorama_conventions_for_real_calibrations(path):
    context = view_context(str(path))
    metadata = load_veo_metadata(path)
    script = """
      import {readFileSync} from 'node:fs';
      import {initialCamera, interactiveRay, interactiveGroundPoint} from './web_app/frontend/src/components/interactiveCamera.ts';
      const context = JSON.parse(readFileSync(0, 'utf8'));
      const samples = [];
      for (const longitude of [-.4, 0, .6]) for (const latitude of [-.3, 0, .4]) for (const fov of [.5, 1, 1.5]) {
        const camera = {...initialCamera(context), longitude, latitude, fov};
        for (const [x,y,width,height] of [[50,50,101,101],[30,70,151,101],[200,100,601,301]])
          samples.push({camera,x,y,width,height,ray:interactiveRay(context,camera,x,y,width,height),
            point:interactiveGroundPoint(context,camera,x,y,width,height)});
      }
      console.log(JSON.stringify(samples));
    """
    completed = subprocess.run(['node', '--experimental-strip-types', '--input-type=module', '-e', script],
        cwd=ROOT, input=json.dumps(context), text=True, capture_output=True, check=True, timeout=30)
    valid = 0
    for sample in json.loads(completed.stdout):
        # Convert the perspective direction into equivalent Panorama lon/lat,
        # independently in NumPy, then use the validated ray/intersection helpers.
        camera = sample['camera']
        lon, lat, fov = camera['longitude'], camera['latitude'], camera['fov']
        x, y, w, h = (sample[k] for k in ('x', 'y', 'width', 'height'))
        direction = np.array([((x + .5) / w * 2 - 1) * np.tan(fov / 2) * w / h,
                              ((y + .5) / h * 2 - 1) * np.tan(fov / 2), 1])
        ry = np.array([[np.cos(lon), 0, np.sin(lon)], [0, 1, 0], [-np.sin(lon), 0, np.cos(lon)]])
        rx = np.array([[1, 0, 0], [0, np.cos(lat), -np.sin(lat)], [0, np.sin(lat), np.cos(lat)]])
        presentation = ry @ rx @ direction
        magnitude = np.linalg.norm(presentation)
        equivalent_lon = np.arctan2(presentation[0], presentation[2])
        equivalent_lat = -np.arcsin(presentation[1] / magnitude)
        va = metadata.va_presentation
        u = (equivalent_lon - va['left']) / (va['right'] - va['left'])
        v = (equivalent_lat - va['bottom']) / (va['top'] - va['bottom'])
        origin, ray, _, _ = panorama_ray(metadata, u, v)
        np.testing.assert_allclose(sample['ray']['origin'], origin, atol=1e-12)
        np.testing.assert_allclose(sample['ray']['direction'], ray * magnitude, atol=1e-12)
        ground = ground_intersection_from_uv(metadata, u, v)
        meaningful = ground and abs(ground[0]) <= metadata.field_length / 2 and abs(ground[1]) <= metadata.field_width / 2
        if meaningful:
            valid += 1
            np.testing.assert_allclose([sample['point']['world_x'], sample['point']['world_z']], ground[:2], atol=1e-10)
        else:
            assert sample['point'] is None
    assert valid > 5
