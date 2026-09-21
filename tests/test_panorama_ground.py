from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from geometry.panorama_ground import (
    ground_intersection_for_pixel,
    ground_intersection_from_uv,
    is_inside_field,
    panorama_pixel_center_uv,
)


def metadata_for(latitude_top: float = 0.8, latitude_bottom: float = 0.2):
    camera2world = np.eye(4, dtype=np.float64)
    camera2world[1, 3] = 10.0
    return SimpleNamespace(
        va_presentation={"left": -0.8, "right": 0.8, "top": latitude_top, "bottom": latitude_bottom},
        camera2world=camera2world,
        camera2presentation=np.eye(4, dtype=np.float64),
        field_length=105.0,
        field_width=68.0,
    )


class PanoramaGroundTests(unittest.TestCase):
    def test_pixel_center_uv_handles_center_and_edges(self):
        self.assertAlmostEqual(panorama_pixel_center_uv(0, 0, 100, 50)[0], 0.005)
        self.assertAlmostEqual(panorama_pixel_center_uv(0, 0, 100, 50)[1], 0.99)
        self.assertAlmostEqual(panorama_pixel_center_uv(99, 49, 100, 50)[0], 0.995)
        self.assertAlmostEqual(panorama_pixel_center_uv(99, 49, 100, 50)[1], 0.01)

    def test_downward_ray_has_finite_ground_intersection(self):
        result = ground_intersection_for_pixel(metadata_for(), 50, 50, 101, 101)
        self.assertNotEqual(result.status, "no_ground_intersection")
        self.assertTrue(np.isfinite(result.world_x))
        self.assertTrue(np.isfinite(result.world_z))
        self.assertGreater(result.distance, 0.0)

    def test_upward_ray_has_no_ground_intersection_in_front_of_camera(self):
        result = ground_intersection_for_pixel(metadata_for(-0.2, -0.8), 50, 50, 101, 101)
        self.assertEqual(result.status, "no_ground_intersection")
        self.assertIsNone(result.world_x)
        self.assertIsNone(result.world_z)

    def test_inside_and_outside_field_classification(self):
        self.assertTrue(is_inside_field(0.0, 0.0, 105.0, 68.0))
        self.assertFalse(is_inside_field(52.6, 0.0, 105.0, 68.0))
        self.assertFalse(is_inside_field(0.0, -34.1, 105.0, 68.0))

    def test_equivalent_pixel_center_uv_is_resolution_independent(self):
        metadata = metadata_for()
        u_small, v_small = panorama_pixel_center_uv(25, 25, 101, 101)
        u_large, v_large = panorama_pixel_center_uv(76, 76, 303, 303)
        self.assertAlmostEqual(u_small, u_large)
        self.assertAlmostEqual(v_small, v_large)
        small = ground_intersection_from_uv(metadata, u_small, v_small)
        large = ground_intersection_from_uv(metadata, u_large, v_large)
        self.assertIsNotNone(small)
        self.assertIsNotNone(large)
        np.testing.assert_allclose(small[:3], large[:3], rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
