"""Known pitch landmarks used only to validate panorama ground intersections."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot

from geometry.panorama_ground import GroundIntersection


@dataclass(frozen=True)
class PitchLandmark:
    identifier: str
    name: str
    x: float
    z: float

    def to_dict(self) -> dict:
        return {"id": self.identifier, "name": self.name, "x": self.x, "z": self.z}


def field_landmarks(field_length: float, field_width: float) -> dict[str, PitchLandmark]:
    """Build validation references from the current calibration's field size."""
    half_length = field_length / 2.0
    half_width = field_width / 2.0
    values = (
        PitchLandmark("center_mark", "Center mark", 0.0, 0.0),
        PitchLandmark("left_penalty_spot", "Left penalty spot", -half_length + 11.0, 0.0),
        PitchLandmark("right_penalty_spot", "Right penalty spot", half_length - 11.0, 0.0),
        PitchLandmark("corner_left_near", "Left near corner", -half_length, -half_width),
        PitchLandmark("corner_left_far", "Left far corner", -half_length, half_width),
        PitchLandmark("corner_right_near", "Right near corner", half_length, -half_width),
        PitchLandmark("corner_right_far", "Right far corner", half_length, half_width),
        PitchLandmark("left_penalty_area_near", "Left penalty-area near corner", -half_length + 16.5, -20.16),
        PitchLandmark("left_penalty_area_far", "Left penalty-area far corner", -half_length + 16.5, 20.16),
        PitchLandmark("right_penalty_area_near", "Right penalty-area near corner", half_length - 16.5, -20.16),
        PitchLandmark("right_penalty_area_far", "Right penalty-area far corner", half_length - 16.5, 20.16),
        PitchLandmark("left_goal_area_near", "Left goal-area near corner", -half_length + 5.5, -9.16),
        PitchLandmark("left_goal_area_far", "Left goal-area far corner", -half_length + 5.5, 9.16),
        PitchLandmark("right_goal_area_near", "Right goal-area near corner", half_length - 5.5, -9.16),
        PitchLandmark("right_goal_area_far", "Right goal-area far corner", half_length - 5.5, 9.16),
    )
    return {landmark.identifier: landmark for landmark in values}


def landmark_validation(landmark: PitchLandmark, measured: GroundIntersection) -> dict:
    """Return validation-only error fields without changing camera geometry."""
    result = {
        "landmark": landmark.identifier,
        "landmark_name": landmark.name,
        "expected_x": landmark.x,
        "expected_z": landmark.z,
        "measured_x": measured.world_x,
        "measured_z": measured.world_z,
        "delta_x": None,
        "delta_z": None,
        "field_error_m": None,
    }
    if measured.world_x is not None and measured.world_z is not None:
        delta_x = measured.world_x - landmark.x
        delta_z = measured.world_z - landmark.z
        result.update(
            delta_x=delta_x,
            delta_z=delta_z,
            field_error_m=hypot(delta_x, delta_z),
        )
    return result
