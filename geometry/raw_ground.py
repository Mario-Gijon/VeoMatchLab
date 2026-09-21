"""Inverse of the validated Veo lens sampling path; no fitted field warp.

RAW pixels refer to texel centers in the entire stacked 2048×2048 source.
The rational lens inversion is solved against the existing forward function,
and rejected if its forward residual is not below a calibration-pixel tolerance.
"""
from __future__ import annotations

import numpy as np

from geometry.panorama_ground import GroundIntersection, is_inside_field
from geometry.reproduce_veo_panorama import (
    CALIB_WIDTH, CALIB_HEIGHT, TOP_DEST_RATIO, TOP_SOURCE_RATIO, distort_points,
)


def inverse_source_compression(value: float) -> float:
    if value < TOP_SOURCE_RATIO:
        return value / TOP_SOURCE_RATIO * TOP_DEST_RATIO
    return TOP_DEST_RATIO + (value - TOP_SOURCE_RATIO) / (1 - TOP_SOURCE_RATIO) * (1 - TOP_DEST_RATIO)


def invert_distortion(pixel: np.ndarray, lens) -> np.ndarray:
    initial = np.linalg.solve(lens.intrinsic[:2, :2], pixel - lens.intrinsic[:2, 2])
    point = initial.copy()

    def forward(p):
        return np.asarray(distort_points(p[0], p[1], lens.intrinsic, lens.distortion))

    # Damped Newton inversion of the canonical forward model, not a calibration
    # optimization. Bounds/line search avoid divergent rational-model branches.
    for _ in range(80):
        residual = forward(point) - pixel
        if np.linalg.norm(residual) < 1e-7:
            return point
        step = 1e-5
        jacobian = np.column_stack([(forward(point + np.eye(2)[i] * step) - forward(point - np.eye(2)[i] * step)) / (2 * step) for i in range(2)])
        try:
            delta = np.linalg.solve(jacobian, residual)
        except np.linalg.LinAlgError:
            break
        improved = False
        for power in range(16):
            candidate = point - delta * 0.5 ** power
            if np.isfinite(candidate).all() and np.linalg.norm(candidate) < 100 and np.linalg.norm(forward(candidate) - pixel) < np.linalg.norm(residual):
                point = candidate
                improved = True
                break
        if not improved:
            break
    raise ValueError("Lens distortion inversion did not converge for this pixel; no ground point reported.")


def raw_pixel_to_ground(metadata, x: float, y: float) -> GroundIntersection:
    if not np.isfinite([x, y]).all() or not (0 <= x < 2048 and 0 <= y < 2048):
        raise ValueError("RAW pixel must be inside the 2048×2048 stacked source")
    left = y < 1024
    lens = metadata.left if left else metadata.right
    v = (y + .5 - (0 if left else 1024)) / 1024
    pixel = np.array([(x + .5) / 2048 * CALIB_WIDTH, inverse_source_compression(v) * CALIB_HEIGHT])
    ideal = invert_distortion(pixel, lens)
    transform = metadata.camera2world @ lens.lens2camera
    origin = (transform @ np.array([0, 0, 0, 1.]))[:3]
    direction = (transform @ np.array([*ideal, 1., 0.]))[:3]
    t = -origin[1] / direction[1] if abs(direction[1]) > 1e-12 else -1
    world = origin + t * direction
    valid = t > 0 and np.isfinite(world).all()
    inside = bool(is_inside_field(world[0], world[2], metadata.field_length, metadata.field_width)) if valid else None
    return GroundIntersection(
        pixel_x=x, pixel_y=y, u=(x + .5) / 2048, v=1 - (y + .5) / 2048,
        longitude=0., latitude=0., world_x=float(world[0]) if valid else None,
        world_z=float(world[2]) if valid else None, distance=float(t) if valid else None,
        inside_field=inside, status=("inside_field" if inside else "outside_field") if valid else "no_ground_intersection",
    )
