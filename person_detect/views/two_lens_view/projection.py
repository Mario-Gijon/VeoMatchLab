#!/usr/bin/env python3
"""Project detections from the two physical Veo lenses onto the football pitch.

Input:
    physical lens name + local 2048x1024 image point
    (normally the detector bbox bottom-center)

Output:
    world X/Z ground point
    strict field membership
    allowed field membership with +0.50 m ONLY on the far touchline
    unique logical owner lens
    whether the source lens is the owner

Ownership design
----------------
The same physical player may be visible in BOTH lenses. We do not merge nearby
players by world-space distance.

Instead, every world point has one deterministic owner:
  1. if visible in only one physical lens -> that lens owns it
  2. if visible in both -> the lens whose optical axis is closer to that point
     owns it (smaller off-axis angle)
  3. exact ties go to LEFT for deterministic non-overlap

Thus overlap is allowed visually, but logical ownership is non-overlapping.
A narrow boundary reconciler may still be added later if real detector
foot-point noise makes the same player cross the ownership seam in opposite
directions. That must be tested rather than assumed.
"""

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

from geometry.raw_ground import raw_pixel_to_ground  # noqa: E402
from geometry.reproduce_veo_panorama import (  # noqa: E402
    project_world_target_to_lens,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.views.two_lens_view.renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    LENS_H,
    LENS_W,
    STACKED_H,
    STACKED_W,
)


DEFAULT_FAR_TOUCHLINE_MARGIN_M = 0.50
EPSILON = 1e-12


@dataclass(frozen=True)
class LensVisibility:
    lens: str
    visible: bool
    pixel_x: float | None
    pixel_y: float | None
    off_axis_angle_rad: float | None
    off_axis_angle_deg: float | None

    def to_dict(self) -> dict:
        return asdict(self)


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

    left_visible: bool | None
    right_visible: bool | None
    left_off_axis_deg: float | None
    right_off_axis_deg: float | None

    status: str

    def to_dict(self) -> dict:
        return asdict(self)


class TwoLensProjector:
    """Detector-independent physical-lens ground projection + ownership."""

    view_names = ("left", "right")

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
        *,
        far_touchline_margin_m: float = DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    ) -> None:
        self.calibration = Path(calibration)
        self.metadata = load_veo_metadata(self.calibration)
        self.far_touchline_margin_m = float(far_touchline_margin_m)

        if self.far_touchline_margin_m < 0.0:
            raise ValueError("far_touchline_margin_m must be >= 0")

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

    @staticmethod
    def _validate_view(view_name: str) -> str:
        key = view_name.lower()
        if key not in ("left", "right"):
            raise ValueError(
                f"Unknown lens {view_name!r}; expected left or right"
            )
        return key

    def classify_field(
        self,
        world_x: float,
        world_z: float,
    ) -> tuple[bool, bool, str]:
        x = float(world_x)
        z = float(world_z)

        inside_x = -self.half_length <= x <= self.half_length
        inside_z_strict = -self.half_width <= z <= self.half_width
        inside_strict = inside_x and inside_z_strict

        if self.far_touchline_z > 0.0:
            allowed_z_min = -self.half_width
            allowed_z_max = self.half_width + self.far_touchline_margin_m
        else:
            allowed_z_min = -self.half_width - self.far_touchline_margin_m
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

    def visibility_for_world_point(
        self,
        lens_name: str,
        world_x: float,
        world_z: float,
    ) -> LensVisibility:
        """Return source-pixel visibility + optical-axis quality for one lens."""
        lens_name = self._validate_view(lens_name)
        lens = self.metadata.lens(lens_name)

        target_world = np.array(
            [[float(world_x), 0.0, float(world_z), 1.0]],
            dtype=np.float64,
        )

        map_x, map_y, valid = project_world_target_to_lens(
            target_world,
            lens,
            self.metadata.camera2world,
            STACKED_W,
            STACKED_H,
            lens_name == "left",
        )

        lens_transform_world = (
            self.metadata.camera2world
            @ lens.lens2camera
        )
        point_lens = (
            np.linalg.inv(lens_transform_world)
            @ target_world[0]
        )

        z_lens = float(point_lens[2])
        physically_forward = (
            np.isfinite(point_lens[:3]).all()
            and z_lens > EPSILON
        )

        is_valid = bool(valid[0]) and physically_forward

        if lens_name == "left":
            local_y = float(map_y[0])
        else:
            local_y = float(map_y[0]) - LENS_H

        local_x = float(map_x[0])

        is_valid = (
            is_valid
            and 0.0 <= local_x < LENS_W
            and 0.0 <= local_y < LENS_H
        )

        if not is_valid:
            return LensVisibility(
                lens=lens_name,
                visible=False,
                pixel_x=None,
                pixel_y=None,
                off_axis_angle_rad=None,
                off_axis_angle_deg=None,
            )

        ideal_x = float(point_lens[0]) / z_lens
        ideal_y = float(point_lens[1]) / z_lens
        angle = math.atan(math.hypot(ideal_x, ideal_y))

        return LensVisibility(
            lens=lens_name,
            visible=True,
            pixel_x=local_x,
            pixel_y=local_y,
            off_axis_angle_rad=angle,
            off_axis_angle_deg=math.degrees(angle),
        )

    def owner_for_world_point(
        self,
        world_x: float,
        world_z: float,
    ) -> tuple[str | None, LensVisibility, LensVisibility]:
        left = self.visibility_for_world_point(
            "left",
            world_x,
            world_z,
        )
        right = self.visibility_for_world_point(
            "right",
            world_x,
            world_z,
        )

        if left.visible and not right.visible:
            return "left", left, right

        if right.visible and not left.visible:
            return "right", left, right

        if not left.visible and not right.visible:
            return None, left, right

        assert left.off_axis_angle_rad is not None
        assert right.off_axis_angle_rad is not None

        # Exact ties deliberately belong to LEFT so ownership cannot overlap.
        if (
            left.off_axis_angle_rad
            <= right.off_axis_angle_rad + EPSILON
        ):
            return "left", left, right

        return "right", left, right

    def _stacked_pixel(
        self,
        view_name: str,
        pixel_x: float,
        pixel_y: float,
    ) -> tuple[float, float]:
        view_name = self._validate_view(view_name)

        if not np.isfinite([pixel_x, pixel_y]).all():
            raise ValueError("pixel coordinates must be finite")

        if not 0.0 <= pixel_x < LENS_W:
            raise ValueError(
                f"pixel_x must be in [0, {LENS_W}), got {pixel_x}"
            )

        if not 0.0 <= pixel_y < LENS_H:
            raise ValueError(
                f"pixel_y must be in [0, {LENS_H}), got {pixel_y}"
            )

        stacked_y = (
            float(pixel_y)
            if view_name == "left"
            else float(pixel_y) + LENS_H
        )

        return float(pixel_x), stacked_y

    def project_pixel(
        self,
        view_name: str,
        pixel_x: float,
        pixel_y: float,
    ) -> ProjectionResult:
        view_name = self._validate_view(view_name)
        stacked_x, stacked_y = self._stacked_pixel(
            view_name,
            pixel_x,
            pixel_y,
        )

        try:
            ground = raw_pixel_to_ground(
                self.metadata,
                stacked_x,
                stacked_y,
            )
        except ValueError:
            return self._no_ground_result(
                view_name,
                pixel_x,
                pixel_y,
                "no_ground_intersection",
            )

        if (
            ground.world_x is None
            or ground.world_z is None
            or ground.status == "no_ground_intersection"
        ):
            return self._no_ground_result(
                view_name,
                pixel_x,
                pixel_y,
                "no_ground_intersection",
            )

        world_x = float(ground.world_x)
        world_z = float(ground.world_z)

        inside_strict, inside_allowed, field_status = (
            self.classify_field(world_x, world_z)
        )

        owner, left_visibility, right_visibility = (
            self.owner_for_world_point(world_x, world_z)
        )

        is_owner = owner == view_name

        lens = self.metadata.lens(view_name)
        lens_origin = (
            self.metadata.camera2world
            @ lens.lens2camera
            @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        )[:3]
        world_point = np.array(
            [world_x, 0.0, world_z],
            dtype=np.float64,
        )
        ray_distance = float(
            np.linalg.norm(world_point - lens_origin)
        )

        if not inside_allowed:
            status = "rejected_outside_field"
        elif owner is None:
            status = "rejected_no_owner"
        elif not is_owner:
            status = "rejected_not_owner"
        else:
            status = "accepted"

        return ProjectionResult(
            source_view=view_name,
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
            far_touchline_z=self.far_touchline_z,
            far_touchline_name=self.far_touchline_name,
            far_touchline_margin_m=self.far_touchline_margin_m,
            left_visible=left_visibility.visible,
            right_visible=right_visibility.visible,
            left_off_axis_deg=left_visibility.off_axis_angle_deg,
            right_off_axis_deg=right_visibility.off_axis_angle_deg,
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
        """Project detector bbox bottom-center (player foot-point convention)."""
        values = np.asarray([x1, y1, x2, y2], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("bbox coordinates must be finite")
        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")

        foot_x = min(
            max(0.5 * (float(x1) + float(x2)), 0.0),
            LENS_W - 1.0,
        )
        foot_y = min(
            max(float(y2) - 0.5, 0.0),
            LENS_H - 1.0,
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
            far_touchline_z=self.far_touchline_z,
            far_touchline_name=self.far_touchline_name,
            far_touchline_margin_m=self.far_touchline_margin_m,
            left_visible=None,
            right_visible=None,
            left_off_axis_deg=None,
            right_off_axis_deg=None,
            status=status,
        )

    def coverage_summary(
        self,
        *,
        nx: int = 105,
        nz: int = 65,
    ) -> dict:
        """Sample the pitch and report physical-lens visibility/ownership."""
        xs = np.linspace(-self.half_length, self.half_length, nx)
        zs = np.linspace(-self.half_width, self.half_width, nz)

        counts = {
            "sampled": 0,
            "left_only_visible": 0,
            "right_only_visible": 0,
            "both_visible": 0,
            "neither_visible": 0,
            "left_owner": 0,
            "right_owner": 0,
            "no_owner": 0,
        }

        for z in zs:
            for x in xs:
                owner, left, right = self.owner_for_world_point(
                    float(x),
                    float(z),
                )
                counts["sampled"] += 1

                if left.visible and right.visible:
                    counts["both_visible"] += 1
                elif left.visible:
                    counts["left_only_visible"] += 1
                elif right.visible:
                    counts["right_only_visible"] += 1
                else:
                    counts["neither_visible"] += 1

                if owner == "left":
                    counts["left_owner"] += 1
                elif owner == "right":
                    counts["right_owner"] += 1
                else:
                    counts["no_owner"] += 1

        sampled = counts["sampled"]
        return {
            **counts,
            "both_visible_fraction": counts["both_visible"] / sampled,
            "neither_visible_fraction": counts["neither_visible"] / sampled,
            "left_owner_fraction": counts["left_owner"] / sampled,
            "right_owner_fraction": counts["right_owner"] / sampled,
        }

    def summary(self) -> dict:
        return {
            "calibration": str(self.calibration),
            "lens_image_size": [LENS_W, LENS_H],
            "field_length_m": float(self.metadata.field_length),
            "field_width_m": float(self.metadata.field_width),
            "camera_world_xyz": [float(v) for v in self.camera],
            "far_touchline": {
                "axis": "z",
                "name": self.far_touchline_name,
                "z": self.far_touchline_z,
                "margin_outward_m": self.far_touchline_margin_m,
            },
            "ownership_rule": (
                "single visible lens wins; if both visible, smaller optical "
                "off-axis angle wins; exact ties go to left"
            ),
            "coverage": self.coverage_summary(),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Project a physical LEFT/RIGHT lens point or bbox onto the pitch."
        )
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--view",
        choices=("left", "right"),
        required=True,
    )
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        metavar=("X", "Y"),
    )
    parser.add_argument(
        "--bbox",
        nargs=4,
        type=float,
        metavar=("X1", "Y1", "X2", "Y2"),
    )
    parser.add_argument(
        "--far-margin",
        type=float,
        default=DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    )
    parser.add_argument(
        "--summary",
        action="store_true",
    )
    args = parser.parse_args()

    if (args.point is None) == (args.bbox is None):
        parser.error("provide exactly one of --point or --bbox")

    return args


def main() -> None:
    args = parse_args()

    projector = TwoLensProjector(
        args.calibration,
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
