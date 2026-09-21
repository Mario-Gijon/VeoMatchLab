"""Panorama-pixel to ground-plane validation helpers.

The ray construction deliberately mirrors ``geometry.reproduce_veo_panorama``.
It is kept separate so the validated renderer remains unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EPSILON = 1e-12


@dataclass(frozen=True)
class GroundIntersection:
    """Authoritative result for one rendered-panorama pixel."""

    pixel_x: int
    pixel_y: int
    u: float
    v: float
    longitude: float
    latitude: float
    world_x: float | None
    world_z: float | None
    distance: float | None
    inside_field: bool | None
    status: str

    def to_dict(self) -> dict:
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "u": self.u,
            "v": self.v,
            "longitude": self.longitude,
            "latitude": self.latitude,
            "world_x": self.world_x,
            "world_z": self.world_z,
            "distance": self.distance,
            "inside_field": self.inside_field,
            "status": self.status,
        }


def panorama_pixel_center_uv(pixel_x: int, pixel_y: int, width: int, height: int) -> tuple[float, float]:
    """Return the pixel-center UV used by the WebGL panorama reproduction."""
    if not 0 <= pixel_x < width or not 0 <= pixel_y < height:
        raise ValueError("Panorama pixel is outside the rendered image")
    return (pixel_x + 0.5) / width, 1.0 - (pixel_y + 0.5) / height


def panorama_ray(metadata, u: float, v: float) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return the renderer-equivalent world origin/direction for panorama UV."""
    va = metadata.va_presentation
    longitude = u * (va["right"] - va["left"]) + va["left"]
    latitude = v * (va["top"] - va["bottom"]) + va["bottom"]

    cos_latitude = np.cos(latitude)
    direction_presentation = np.array(
        [
            np.sin(longitude) * cos_latitude,
            -np.sin(latitude),
            np.cos(longitude) * cos_latitude,
            0.0,
        ],
        dtype=np.float64,
    )

    # This is exactly the effective convention in build_output_world_geometry:
    # camera_to_world_presentation = camera2world @ inverse(camera2presentation).
    camera_to_world_presentation = metadata.camera2world @ np.linalg.inv(metadata.camera2presentation)
    direction_world = (camera_to_world_presentation @ direction_presentation)[:3]
    camera_world = (metadata.camera2world @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64))[:3]
    return camera_world, direction_world, float(longitude), float(latitude)


def is_inside_field(world_x: float, world_z: float, field_length: float, field_width: float) -> bool:
    """Classify an X/Z ground point against the field dimensions in metadata."""
    return abs(world_x) <= field_length / 2.0 and abs(world_z) <= field_width / 2.0


def ground_intersection_from_uv(metadata, u: float, v: float) -> tuple[float, float, float, float, float] | None:
    """Intersect the renderer-equivalent panorama ray with Y=0 in front of camera."""
    camera_world, direction_world, longitude, latitude = panorama_ray(metadata, u, v)
    direction_y = direction_world[1]
    if not np.isfinite(direction_world).all() or abs(direction_y) <= EPSILON:
        return None

    distance = -camera_world[1] / direction_y
    if not np.isfinite(distance) or distance <= 0.0:
        return None

    world = camera_world + distance * direction_world
    if not np.isfinite(world).all():
        return None
    return float(world[0]), float(world[2]), float(distance), longitude, latitude


def ground_intersection_for_pixel(metadata, pixel_x: int, pixel_y: int, width: int, height: int) -> GroundIntersection:
    """Convert one rendered panorama pixel into a world X/Z validation result."""
    u, v = panorama_pixel_center_uv(pixel_x, pixel_y, width, height)
    result = ground_intersection_from_uv(metadata, u, v)
    if result is None:
        _, _, longitude, latitude = panorama_ray(metadata, u, v)
        return GroundIntersection(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            u=u,
            v=v,
            longitude=longitude,
            latitude=latitude,
            world_x=None,
            world_z=None,
            distance=None,
            inside_field=None,
            status="no_ground_intersection",
        )

    world_x, world_z, distance, longitude, latitude = result
    inside = is_inside_field(world_x, world_z, metadata.field_length, metadata.field_width)
    return GroundIntersection(
        pixel_x=pixel_x,
        pixel_y=pixel_y,
        u=u,
        v=v,
        longitude=longitude,
        latitude=latitude,
        world_x=world_x,
        world_z=world_z,
        distance=distance,
        inside_field=inside,
        status="inside_field" if inside else "outside_field",
    )
