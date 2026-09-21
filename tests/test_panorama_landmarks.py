from __future__ import annotations

import unittest

from geometry.panorama_ground import GroundIntersection
from geometry.panorama_landmarks import field_landmarks, landmark_validation


def measured_point(x: float | None, z: float | None) -> GroundIntersection:
    return GroundIntersection(
        pixel_x=100,
        pixel_y=200,
        u=0.1,
        v=0.2,
        longitude=0.0,
        latitude=0.0,
        world_x=x,
        world_z=z,
        distance=10.0 if x is not None else None,
        inside_field=True if x is not None else None,
        status="inside_field" if x is not None else "no_ground_intersection",
    )


class PanoramaLandmarkTests(unittest.TestCase):
    def test_center_mark_and_penalty_spots(self):
        landmarks = field_landmarks(105.0, 68.0)
        self.assertEqual((landmarks["center_mark"].x, landmarks["center_mark"].z), (0.0, 0.0))
        self.assertEqual((landmarks["left_penalty_spot"].x, landmarks["left_penalty_spot"].z), (-41.5, 0.0))
        self.assertEqual((landmarks["right_penalty_spot"].x, landmarks["right_penalty_spot"].z), (41.5, 0.0))

    def test_pitch_corners_follow_each_field_width(self):
        narrow = field_landmarks(100.0, 60.0)
        wide = field_landmarks(100.0, 80.0)
        self.assertEqual((narrow["corner_left_far"].x, narrow["corner_left_far"].z), (-50.0, 30.0))
        self.assertEqual((wide["corner_right_near"].x, wide["corner_right_near"].z), (50.0, -40.0))

    def test_penalty_area_outer_corners(self):
        landmarks = field_landmarks(105.0, 68.0)
        self.assertEqual((landmarks["left_penalty_area_near"].x, landmarks["left_penalty_area_near"].z), (-36.0, -20.16))
        self.assertEqual((landmarks["right_penalty_area_far"].x, landmarks["right_penalty_area_far"].z), (36.0, 20.16))

    def test_error_metric_uses_euclidean_field_distance(self):
        landmark = field_landmarks(105.0, 68.0)["center_mark"]
        validation = landmark_validation(landmark, measured_point(3.0, -4.0))
        self.assertEqual(validation["delta_x"], 3.0)
        self.assertEqual(validation["delta_z"], -4.0)
        self.assertEqual(validation["field_error_m"], 5.0)

    def test_no_ground_intersection_has_no_error_metric(self):
        landmark = field_landmarks(105.0, 68.0)["center_mark"]
        validation = landmark_validation(landmark, measured_point(None, None))
        self.assertIsNone(validation["delta_x"])
        self.assertIsNone(validation["field_error_m"])


if __name__ == "__main__":
    unittest.main()
