"""Ground projection for detector boxes in the calibrated panorama view."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.views.panorama_view.renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_HEIGHT,
    MAX_WIDTH,
    default_panorama_size,
)


DEFAULT_FAR_TOUCHLINE_MARGIN_M = 0.50
EPSILON = 1e-12


@dataclass(frozen=True)
class ProjectionResult:
    source_view: str
    pixel_x: float
    pixel_y: float
    world_x: float | None
    world_z: float | None
    ray_distance: float | None
    inside_strict_field: bool | None
    inside_allowed_field: bool | None
    field_status: str
    owner_view: str | None
    is_owner: bool | None
    far_touchline_z: float
    far_touchline_name: str
    far_touchline_margin_m: float
    status: str

    def to_dict(self) -> dict:
        return asdict(self)


class PanoramaProjector:
    """Inverse of the calibrated panorama pixel angular parameterization."""

    view_names = ("panorama",)

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
        *,
        width: int | None = None,
        height: int | None = None,
        far_touchline_margin_m: float = DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    ) -> None:
        self.calibration = Path(calibration)
        self.metadata = load_veo_metadata(self.calibration)
        if width is None and height is None:
            self.width, self.height = default_panorama_size(self.metadata)
        else:
            self.height = int(DEFAULT_HEIGHT if height is None else height)
            self.width = int(
                default_panorama_size(self.metadata, height=self.height)[0]
                if width is None
                else width
            )
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Panorama width and height must be positive")

        self.far_touchline_margin_m = float(far_touchline_margin_m)
        if self.far_touchline_margin_m < 0.0:
            raise ValueError("far_touchline_margin_m must be >= 0")

        self.presentation_to_world = (
            self.metadata.camera2world
            @ np.linalg.inv(self.metadata.camera2presentation)
        )
        self.camera = (
            self.metadata.camera2world
            @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        self.half_length = float(self.metadata.field_length) / 2.0
        self.half_width = float(self.metadata.field_width) / 2.0

        plus_z_distance = abs(float(self.camera[2]) - self.half_width)
        minus_z_distance = abs(float(self.camera[2]) + self.half_width)
        if plus_z_distance >= minus_z_distance:
            self.far_touchline_z = self.half_width
            self.near_touchline_z = -self.half_width
            self.far_touchline_name = "+Z"
        else:
            self.far_touchline_z = -self.half_width
            self.near_touchline_z = self.half_width
            self.far_touchline_name = "-Z"

    def _ray_for_pixel(
        self,
        pixel_x: float,
        pixel_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not np.isfinite([pixel_x, pixel_y]).all():
            raise ValueError("pixel coordinates must be finite")
        if not 0.0 <= pixel_x < self.width:
            raise ValueError(f"pixel_x must be in [0, {self.width}), got {pixel_x}")
        if not 0.0 <= pixel_y < self.height:
            raise ValueError(f"pixel_y must be in [0, {self.height}), got {pixel_y}")

        va = self.metadata.va_presentation
        u = (float(pixel_x) + 0.5) / self.width
        v = 1.0 - (float(pixel_y) + 0.5) / self.height
        lon = u * (va["right"] - va["left"]) + va["left"]
        lat = v * (va["top"] - va["bottom"]) + va["bottom"]
        direction_presentation = np.array(
            [
                math.sin(lon) * math.cos(lat),
                -math.sin(lat),
                math.cos(lon) * math.cos(lat),
                0.0,
            ],
            dtype=np.float64,
        )
        direction_world = (self.presentation_to_world @ direction_presentation)[:3]
        if not np.isfinite(direction_world).all():
            raise ValueError("Non-finite world ray")
        return self.camera.copy(), direction_world

    def pixel_for_world(self, world_x: float, world_z: float) -> tuple[float, float] | None:
        """Return panorama pixel coordinates for a world ground point."""
        world = np.array([float(world_x), 0.0, float(world_z)], dtype=np.float64)
        direction = world - self.camera
        direction_h = np.array([*direction, 0.0], dtype=np.float64)
        presentation = np.linalg.inv(self.presentation_to_world) @ direction_h
        if not np.isfinite(presentation).all():
            return None
        lon = math.atan2(float(presentation[0]), float(presentation[2]))
        lat = math.atan2(
            -float(presentation[1]),
            math.hypot(float(presentation[0]), float(presentation[2])),
        )
        va = self.metadata.va_presentation
        span_lon = float(va["right"] - va["left"])
        span_lat = float(va["top"] - va["bottom"])
        u = (lon - va["left"]) / span_lon
        v = (lat - va["bottom"]) / span_lat
        return u * self.width - 0.5, (1.0 - v) * self.height - 0.5

    def classify_field(self, world_x: float, world_z: float) -> tuple[bool, bool, str]:
        x = float(world_x)
        z = float(world_z)
        inside_x = -self.half_length <= x <= self.half_length
        inside_z = -self.half_width <= z <= self.half_width
        inside_strict = inside_x and inside_z
        if self.far_touchline_z > 0.0:
            allowed_z_min = -self.half_width
            allowed_z_max = self.half_width + self.far_touchline_margin_m
        else:
            allowed_z_min = -self.half_width - self.far_touchline_margin_m
            allowed_z_max = self.half_width
        inside_allowed = inside_x and allowed_z_min <= z <= allowed_z_max
        if inside_strict:
            status = "inside_field"
        elif inside_allowed:
            status = "far_touchline_margin"
        else:
            status = "outside_field"
        return inside_strict, inside_allowed, status

    def project_pixel(self, pixel_x: float, pixel_y: float) -> ProjectionResult:
        origin, direction = self._ray_for_pixel(pixel_x, pixel_y)
        dy = float(direction[1])
        if abs(dy) <= EPSILON:
            return self._no_ground_result(pixel_x, pixel_y, "no_ground_intersection")

        distance = -float(origin[1]) / dy
        if not np.isfinite(distance) or distance <= 0.0:
            return self._no_ground_result(pixel_x, pixel_y, "no_ground_intersection")

        world = origin + distance * direction
        if not np.isfinite(world).all():
            return self._no_ground_result(pixel_x, pixel_y, "no_ground_intersection")

        world_x, world_z = float(world[0]), float(world[2])
        inside_strict, inside_allowed, field_status = self.classify_field(world_x, world_z)
        status = "accepted" if inside_allowed else "rejected_outside_field"
        return ProjectionResult(
            source_view="panorama",
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            world_x=world_x,
            world_z=world_z,
            ray_distance=float(np.linalg.norm(world - origin)),
            inside_strict_field=inside_strict,
            inside_allowed_field=inside_allowed,
            field_status=field_status,
            owner_view="panorama",
            is_owner=True,
            far_touchline_z=self.far_touchline_z,
            far_touchline_name=self.far_touchline_name,
            far_touchline_margin_m=self.far_touchline_margin_m,
            status=status,
        )

    def project_bbox(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> ProjectionResult:
        values = np.asarray([x1, y1, x2, y2], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")
        foot_x = min(max(0.5 * (float(x1) + float(x2)), 0.0), self.width - 1.0)
        foot_y = min(max(float(y2) - 0.5, 0.0), self.height - 1.0)
        return self.project_pixel(foot_x, foot_y)

    def _no_ground_result(
        self,
        pixel_x: float,
        pixel_y: float,
        status: str,
    ) -> ProjectionResult:
        return ProjectionResult(
            source_view="panorama",
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            world_x=None,
            world_z=None,
            ray_distance=None,
            inside_strict_field=None,
            inside_allowed_field=None,
            field_status="no_ground_intersection",
            owner_view=None,
            is_owner=None,
            far_touchline_z=self.far_touchline_z,
            far_touchline_name=self.far_touchline_name,
            far_touchline_margin_m=self.far_touchline_margin_m,
            status=status,
        )

    def summary(self) -> dict:
        va = self.metadata.va_presentation
        return {
            "calibration": str(self.calibration),
            "panorama_size": [self.width, self.height],
            "field_length_m": float(self.metadata.field_length),
            "field_width_m": float(self.metadata.field_width),
            "camera_world_xyz": [float(value) for value in self.camera],
            "far_touchline": {
                "name": self.far_touchline_name,
                "z": self.far_touchline_z,
                "margin_outward_m": self.far_touchline_margin_m,
            },
            "presentation_range": {key: float(value) for key, value in va.items()},
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Project panorama pixels onto the pitch.")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--point", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--bbox", nargs=4, type=float, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--far-margin", type=float, default=DEFAULT_FAR_TOUCHLINE_MARGIN_M)
    args = parser.parse_args()
    if (args.point is None) == (args.bbox is None):
        parser.error("provide exactly one of --point or --bbox")
    return args


def main() -> None:
    args = parse_args()
    projector = PanoramaProjector(
        args.calibration,
        width=args.width,
        height=args.height,
        far_touchline_margin_m=args.far_margin,
    )
    if args.point is not None:
        result = projector.project_pixel(*args.point)
    else:
        result = projector.project_bbox(*args.bbox)
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
