#!/usr/bin/env python3
"""Project detections from LEFT/CENTER/RIGHT views onto the football pitch.

This is the second clean stage of the new player-detection pipeline.

Input:
    virtual view name + image point (normally bbox bottom-center)

Output:
    world X/Z ground point
    strict field membership
    allowed field membership with +0.50 m ONLY on the far touchline
    logical owner view (LEFT / CENTER / RIGHT)
    whether the source view is the owner

Important design rule:
    rendered views MAY overlap visually;
    logical ownership NEVER overlaps.

No detector-specific code lives here.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect.interactive_renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    OUTPUT_H,
    OUTPUT_W,
    OVERLAP,
    ViewSpec,
    camera_world,
    derive_views,
    presentation_to_world,
    rot_x,
    rot_y,
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
    presentation_longitude_rad: float | None
    presentation_longitude_deg: float | None

    far_touchline_z: float
    far_touchline_name: str
    far_touchline_margin_m: float

    status: str

    def to_dict(self) -> dict:
        return asdict(self)


def _unwrap_angle(value: float, reference: float) -> float:
    return reference + math.atan2(
        math.sin(value - reference),
        math.cos(value - reference),
    )


class FieldProjector:
    """Geometry shared by every detector backend."""

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
        *,
        width: int = OUTPUT_W,
        height: int = OUTPUT_H,
        overlap: float = OVERLAP,
        far_touchline_margin_m: float = DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    ) -> None:
        # Reuse the same metadata + view derivation as the renderer so there
        # cannot be a renderer/projector disagreement about camera orientation.
        from geometry.veo_metadata import load_veo_metadata

        self.calibration = Path(calibration)
        self.metadata = load_veo_metadata(self.calibration)
        self.width = int(width)
        self.height = int(height)
        self.far_touchline_margin_m = float(far_touchline_margin_m)

        if self.width <= 0 or self.height <= 0:
            raise ValueError("width and height must be positive")
        if self.far_touchline_margin_m < 0.0:
            raise ValueError("far_touchline_margin_m must be >= 0")

        self.specs, self.envelope = derive_views(
            self.metadata,
            self.width,
            self.height,
            overlap,
        )
        self.spec_by_name = {spec.name: spec for spec in self.specs}

        if tuple(self.spec_by_name) != ("left", "center", "right"):
            raise RuntimeError(
                "Expected exactly LEFT/CENTER/RIGHT view specifications"
            )

        self.camera = camera_world(self.metadata)
        self.presentation_to_world_matrix = presentation_to_world(self.metadata)
        self.world_to_presentation_matrix = np.linalg.inv(
            self.presentation_to_world_matrix
        )

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

        self.owner_reference = 0.5 * (
            self.specs[0].owner_min + self.specs[-1].owner_max
        )

    def _spec(self, view_name: str) -> ViewSpec:
        key = view_name.lower()
        try:
            return self.spec_by_name[key]
        except KeyError as exc:
            raise ValueError(
                f"Unknown view {view_name!r}; expected left, center or right"
            ) from exc

    def _ray_for_pixel(
        self,
        spec: ViewSpec,
        pixel_x: float,
        pixel_y: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return world-space ray using renderer-identical perspective math.

        pixel_x/pixel_y are OpenCV-style pixel coordinates. Integer values are
        interpreted as pixel indices, therefore the ray passes through that
        pixel's center via the +0.5 convention used by the renderer.
        """
        if not np.isfinite([pixel_x, pixel_y]).all():
            raise ValueError("pixel coordinates must be finite")
        if not (0.0 <= pixel_x < self.width):
            raise ValueError(
                f"pixel_x must be in [0, {self.width}), got {pixel_x}"
            )
        if not (0.0 <= pixel_y < self.height):
            raise ValueError(
                f"pixel_y must be in [0, {self.height}), got {pixel_y}"
            )

        u = (float(pixel_x) + 0.5) / self.width
        v = (float(pixel_y) + 0.5) / self.height

        tan_v = math.tan(spec.vfov / 2.0)
        aspect = self.width / self.height

        view_ray = np.array(
            [
                (2.0 * u - 1.0) * tan_v * aspect,
                (2.0 * v - 1.0) * tan_v,
                1.0,
                0.0,
            ],
            dtype=np.float64,
        )

        transform = (
            self.presentation_to_world_matrix
            @ rot_y(spec.longitude)
            @ rot_x(spec.latitude)
        )
        direction = (transform @ view_ray)[:3]

        if not np.isfinite(direction).all():
            raise ValueError("Non-finite world ray")

        return self.camera.copy(), direction

    def _world_longitude(self, world_x: float, world_z: float) -> float:
        point = np.array(
            [world_x, 0.0, world_z],
            dtype=np.float64,
        )
        direction_world = point - self.camera
        direction_h = np.array(
            [
                direction_world[0],
                direction_world[1],
                direction_world[2],
                0.0,
            ],
            dtype=np.float64,
        )
        p = self.world_to_presentation_matrix @ direction_h
        longitude = math.atan2(float(p[0]), float(p[2]))
        return _unwrap_angle(longitude, self.owner_reference)

    def owner_for_world_point(
        self,
        world_x: float,
        world_z: float,
    ) -> tuple[str | None, float]:
        """Return the ONE logical owner of a ground point, if any.

        Sectors are half-open:
            LEFT   [min, max)
            CENTER [min, max)
            RIGHT  [min, max]

        Therefore no longitude can belong to two views.
        """
        longitude = self._world_longitude(world_x, world_z)

        for spec in self.specs:
            if spec.name == "right":
                owned = (
                    longitude >= spec.owner_min - EPSILON
                    and longitude <= spec.owner_max + EPSILON
                )
            else:
                owned = (
                    longitude >= spec.owner_min - EPSILON
                    and longitude < spec.owner_max - EPSILON
                )

            if owned:
                return spec.name, longitude

        return None, longitude

    def classify_field(
        self,
        world_x: float,
        world_z: float,
    ) -> tuple[bool, bool, str]:
        """Apply strict field bounds + margin only on the FAR touchline."""
        x = float(world_x)
        z = float(world_z)

        inside_x = -self.half_length <= x <= self.half_length
        inside_z_strict = -self.half_width <= z <= self.half_width
        inside_strict = inside_x and inside_z_strict

        if self.far_touchline_z > 0.0:
            allowed_z_min = -self.half_width
            allowed_z_max = (
                self.half_width + self.far_touchline_margin_m
            )
        else:
            allowed_z_min = (
                -self.half_width - self.far_touchline_margin_m
            )
            allowed_z_max = self.half_width

        inside_allowed = (
            inside_x
            and allowed_z_min <= z <= allowed_z_max
        )

        if inside_strict:
            status = "inside_field"
        elif inside_allowed:
            status = "far_touchline_margin"
        else:
            status = "outside_field"

        return inside_strict, inside_allowed, status

    def project_pixel(
        self,
        view_name: str,
        pixel_x: float,
        pixel_y: float,
    ) -> ProjectionResult:
        spec = self._spec(view_name)
        origin, direction = self._ray_for_pixel(
            spec,
            pixel_x,
            pixel_y,
        )

        dy = float(direction[1])
        if abs(dy) <= EPSILON:
            return self._no_ground_result(
                spec.name,
                pixel_x,
                pixel_y,
                "ray_parallel_to_ground",
            )

        t = -float(origin[1]) / dy
        if not np.isfinite(t) or t <= 0.0:
            return self._no_ground_result(
                spec.name,
                pixel_x,
                pixel_y,
                "ground_behind_camera",
            )

        world = origin + t * direction
        if not np.isfinite(world).all():
            return self._no_ground_result(
                spec.name,
                pixel_x,
                pixel_y,
                "non_finite_ground_intersection",
            )

        world_x = float(world[0])
        world_z = float(world[2])
        ray_distance = float(t * np.linalg.norm(direction))

        inside_strict, inside_allowed, field_status = (
            self.classify_field(world_x, world_z)
        )
        owner, longitude = self.owner_for_world_point(
            world_x,
            world_z,
        )

        is_owner = owner == spec.name

        if not inside_allowed:
            status = "rejected_outside_field"
        elif owner is None:
            status = "rejected_no_owner"
        elif not is_owner:
            status = "rejected_not_owner"
        else:
            status = "accepted"

        return ProjectionResult(
            source_view=spec.name,
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            world_x=world_x,
            world_z=world_z,
            ray_distance=ray_distance,
            inside_strict_field=inside_strict,
            inside_allowed_field=inside_allowed,
            field_status=field_status,
            owner_view=owner,
            is_owner=is_owner,
            presentation_longitude_rad=longitude,
            presentation_longitude_deg=math.degrees(longitude),
            far_touchline_z=self.far_touchline_z,
            far_touchline_name=self.far_touchline_name,
            far_touchline_margin_m=self.far_touchline_margin_m,
            status=status,
        )

    def project_bbox(
        self,
        view_name: str,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
    ) -> ProjectionResult:
        """Project bbox bottom-center, our player foot-point convention."""
        values = np.asarray([x1, y1, x2, y2], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")

        foot_x = 0.5 * (float(x1) + float(x2))

        # Detection bboxes use an outer edge for y2. Move half a pixel inward
        # to obtain a valid image-sample position while staying at the feet.
        foot_y = min(
            max(float(y2) - 0.5, 0.0),
            self.height - 1.0,
        )
        foot_x = min(
            max(foot_x, 0.0),
            self.width - 1.0,
        )

        return self.project_pixel(
            view_name,
            foot_x,
            foot_y,
        )

    def _no_ground_result(
        self,
        view_name: str,
        pixel_x: float,
        pixel_y: float,
        status: str,
    ) -> ProjectionResult:
        return ProjectionResult(
            source_view=view_name,
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
            presentation_longitude_rad=None,
            presentation_longitude_deg=None,
            far_touchline_z=self.far_touchline_z,
            far_touchline_name=self.far_touchline_name,
            far_touchline_margin_m=self.far_touchline_margin_m,
            status=status,
        )

    def summary(self) -> dict:
        return {
            "calibration": str(self.calibration),
            "image_size": [self.width, self.height],
            "field_length_m": float(self.metadata.field_length),
            "field_width_m": float(self.metadata.field_width),
            "camera_world_xyz": [float(v) for v in self.camera],
            "far_touchline": {
                "axis": "z",
                "name": self.far_touchline_name,
                "z": self.far_touchline_z,
                "margin_outward_m": self.far_touchline_margin_m,
            },
            "ownership": [
                {
                    "view": spec.name,
                    "min_rad": spec.owner_min,
                    "max_rad": spec.owner_max,
                    "min_deg": math.degrees(spec.owner_min),
                    "max_deg": math.degrees(spec.owner_max),
                    "right_edge_inclusive": spec.name == "right",
                }
                for spec in self.specs
            ],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a LEFT/CENTER/RIGHT image point or bbox onto the "
            "football pitch."
        )
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--view",
        choices=("left", "center", "right"),
        required=True,
    )
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
        help="Image pixel point to project.",
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Detection bbox; bottom-center is projected.",
    )
    parser.add_argument("--width", type=int, default=OUTPUT_W)
    parser.add_argument("--height", type=int, default=OUTPUT_H)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument(
        "--far-margin",
        type=float,
        default=DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print projector geometry summary before the result.",
    )
    args = parser.parse_args()

    if (args.point is None) == (args.bbox is None):
        parser.error("provide exactly one of --point or --bbox")

    return args


def main() -> None:
    args = parse_args()

    projector = FieldProjector(
        args.calibration,
        width=args.width,
        height=args.height,
        overlap=args.overlap,
        far_touchline_margin_m=args.far_margin,
    )

    if args.summary:
        print(json.dumps(projector.summary(), indent=2))
        print()

    if args.point is not None:
        result = projector.project_pixel(
            args.view,
            args.point[0],
            args.point[1],
        )
    else:
        result = projector.project_bbox(
            args.view,
            *args.bbox,
        )

    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
