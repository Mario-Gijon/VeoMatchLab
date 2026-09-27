#!/usr/bin/env python3
# CHATGPT DELIVERY MARKER: VIRTUAL_CAMERA_COVERAGE_GEOMETRY_V13
# GEOMETRY/RENDER ONLY: do not import association.py or detect.py.
"""Asymmetric six-camera Veo coverage experiment: 4 FAR + 2 NEAR.

Purpose
-------
Test a deliberately non-uniform observation layout based on the real failure
modes of football detection from a sideline Veo camera:

- FAR players are smaller and harder to detect -> allocate FOUR stronger views.
- NEAR players are larger and usually easier -> allocate TWO wider views.

The pitch is split by physical depth from the calibrated near touchline:
    NEAR: 0 .. split_depth
    FAR:  split_depth .. far touchline

Within each depth band, ownership is divided in camera longitude:
    NEAR -> 2 non-overlapping angular sectors
    FAR  -> 4 non-overlapping angular sectors

Logical ownership never overlaps. Rendered views receive a small physical
guard around their ownership mask so players at boundaries are not cut.

All cameras in the same depth band share one optimized latitude/pitch.
Each camera keeps its own longitude/pan. This preserves vertical alignment
across FAR views and across NEAR views without sacrificing much zoom.

TIGHT:
    minimum rectilinear FOV that contains ownership + guard + player height.

CONTEXT:
    same logical layout, but the sampled physical area includes:
      - near_context_m beyond the near touchline,
      - outer_context_m beyond the other pitch edges.
    This is only a diagnostic wider mode; logical field ownership is unchanged.

Every output pixel is sampled DIRECTLY from the original stacked 2048x2048
dual-lens frame through the repository's already-validated Veo geometry.
No panorama is used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.reproduce_veo_panorama import project_world_target_to_lens  # noqa: E402
from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.views.interactive_view.renderer import (  # noqa: E402
    CONTEXT_HEIGHT_M,
    SOURCE_H,
    SOURCE_W,
    camera_world,
    presentation_to_world,
    rot_x,
    rot_y,
    virtual_geometry as validated_virtual_geometry,
)


DEFAULT_NEAR_CAMERAS = 2
DEFAULT_FAR_CAMERAS = 4
DEFAULT_SPLIT_FRACTION = 0.50
DEFAULT_GUARD_M = 0.50
DEFAULT_NEAR_CONTEXT_M = 1.00
DEFAULT_OUTER_CONTEXT_M = 0.50
DEFAULT_GRID_STEP_M = 0.50
DEFAULT_OUT = Path("experiments/virtual_camera_coverage/output")

FOV_SAFETY_SCALE = 1.005
MAX_VFOV_DEG = 125.0
SEARCH_GRID = 15
SEARCH_REFINEMENTS = 4
BAND_LAT_SEARCH_GRID = 15
BAND_LAT_REFINEMENTS = 4
FIXED_LAT_LON_SEARCH_GRID = 15
FIXED_LAT_LON_REFINEMENTS = 4
EPS = 1e-12


@dataclass(frozen=True)
class PhysicalBounds:
    x_min: float
    x_max: float
    z_min: float
    z_max: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CellOwnership:
    name: str
    band: str
    index_in_band: int
    owner_id: int
    lon_min: float
    lon_max: float
    depth_min_m: float
    depth_max_m: float

    def to_dict(self) -> dict:
        result = asdict(self)
        result["lon_min_deg"] = math.degrees(self.lon_min)
        result["lon_max_deg"] = math.degrees(self.lon_max)
        return result


@dataclass(frozen=True)
class AsymmetricOwnershipLayout:
    reference_lon: float
    near_z: float
    far_z: float
    depth_sign: float
    total_depth_m: float
    split_depth_m: float
    near_count: int
    far_count: int
    near_lon_edges: tuple[float, ...]
    far_lon_edges: tuple[float, ...]

    def cells(self) -> list[CellOwnership]:
        result: list[CellOwnership] = []

        for i in range(self.near_count):
            result.append(
                CellOwnership(
                    name=f"near_c{i + 1}",
                    band="near",
                    index_in_band=i,
                    owner_id=i,
                    lon_min=self.near_lon_edges[i],
                    lon_max=self.near_lon_edges[i + 1],
                    depth_min_m=0.0,
                    depth_max_m=self.split_depth_m,
                )
            )

        for i in range(self.far_count):
            result.append(
                CellOwnership(
                    name=f"far_c{i + 1}",
                    band="far",
                    index_in_band=i,
                    owner_id=self.near_count + i,
                    lon_min=self.far_lon_edges[i],
                    lon_max=self.far_lon_edges[i + 1],
                    depth_min_m=self.split_depth_m,
                    depth_max_m=self.total_depth_m,
                )
            )

        return result

    def cell_by_id(self, owner_id: int) -> CellOwnership:
        cells = self.cells()
        if not 0 <= owner_id < len(cells):
            raise ValueError(f"invalid owner_id: {owner_id}")
        return cells[owner_id]

    def to_dict(self) -> dict:
        return {
            "reference_lon_rad": self.reference_lon,
            "reference_lon_deg": math.degrees(self.reference_lon),
            "near_touchline_z": self.near_z,
            "far_touchline_z": self.far_z,
            "depth_sign": self.depth_sign,
            "total_depth_m": self.total_depth_m,
            "split_depth_m": self.split_depth_m,
            "split_fraction": self.split_depth_m / self.total_depth_m,
            "near_count": self.near_count,
            "far_count": self.far_count,
            "near_lon_edges_rad": list(self.near_lon_edges),
            "near_lon_edges_deg": [math.degrees(v) for v in self.near_lon_edges],
            "far_lon_edges_rad": list(self.far_lon_edges),
            "far_lon_edges_deg": [math.degrees(v) for v in self.far_lon_edges],
            "cells": [cell.to_dict() for cell in self.cells()],
        }


@dataclass(frozen=True)
class GroundGrid:
    bounds: PhysicalBounds
    xs: np.ndarray
    zs: np.ndarray
    points: np.ndarray
    longitude: np.ndarray
    depth_m: np.ndarray
    owner_id: np.ndarray


@dataclass(frozen=True)
class ViewFit:
    longitude: float
    latitude: float
    hfov: float
    vfov: float
    half_tan_vertical: float


@dataclass(frozen=True)
class CameraSpec:
    name: str
    mode: str
    cell: CellOwnership
    longitude: float
    latitude: float
    hfov: float
    vfov: float
    required_ground_points: int
    required_world_points: int
    required_point_valid_fraction: float
    required_invalid_world_points: int
    physical_bounds: PhysicalBounds

    def json(self) -> dict:
        result = asdict(self)
        for key in ("longitude", "latitude", "hfov", "vfov"):
            result[f"{key}_deg"] = math.degrees(getattr(self, key))
        result["cell"] = self.cell.to_dict()
        result["physical_bounds"] = self.physical_bounds.to_dict()
        return result


@dataclass
class CameraMap:
    spec: CameraSpec
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    blend: np.ndarray


def presentation_reference(metadata) -> float:
    va = metadata.va_presentation
    return 0.5 * (float(va["left"]) + float(va["right"]))


def player_top_y(metadata) -> float:
    camera_y = float(camera_world(metadata)[1])
    if abs(camera_y) <= EPS:
        raise RuntimeError("calibrated camera height is approximately zero")
    return math.copysign(float(CONTEXT_HEIGHT_M), camera_y)


def near_and_far_touchline_z(metadata) -> tuple[float, float]:
    camera_z = float(camera_world(metadata)[2])
    half_width = float(metadata.field_width) / 2.0

    plus_distance = abs(camera_z - half_width)
    minus_distance = abs(camera_z + half_width)

    if plus_distance <= minus_distance:
        return half_width, -half_width
    return -half_width, half_width


def depth_geometry(metadata) -> tuple[float, float, float, float]:
    near_z, far_z = near_and_far_touchline_z(metadata)
    delta = far_z - near_z
    if abs(delta) <= EPS:
        raise RuntimeError("near and far touchlines collapse to the same Z")
    sign = math.copysign(1.0, delta)
    total = abs(delta)
    return near_z, far_z, sign, total


def depth_from_near(z: np.ndarray | float, near_z: float, depth_sign: float):
    return (np.asarray(z) - near_z) * depth_sign


def tight_physical_bounds(metadata) -> PhysicalBounds:
    return PhysicalBounds(
        x_min=-float(metadata.field_length) / 2.0,
        x_max=float(metadata.field_length) / 2.0,
        z_min=-float(metadata.field_width) / 2.0,
        z_max=float(metadata.field_width) / 2.0,
    )


def context_physical_bounds(
    metadata,
    *,
    near_context_m: float,
    outer_context_m: float,
) -> PhysicalBounds:
    tight = tight_physical_bounds(metadata)
    near_z, _ = near_and_far_touchline_z(metadata)

    z_min = tight.z_min - outer_context_m
    z_max = tight.z_max + outer_context_m

    if near_z < 0.0:
        z_min = tight.z_min - near_context_m
    else:
        z_max = tight.z_max + near_context_m

    return PhysicalBounds(
        x_min=tight.x_min - outer_context_m,
        x_max=tight.x_max + outer_context_m,
        z_min=z_min,
        z_max=z_max,
    )


def axis_samples(minimum: float, maximum: float, max_step_m: float) -> np.ndarray:
    if max_step_m <= 0.0:
        raise ValueError("grid step must be positive")
    span = maximum - minimum
    intervals = max(1, int(math.ceil(span / max_step_m)))
    return np.linspace(minimum, maximum, intervals + 1, dtype=np.float64)


def presentation_directions(metadata, world_points: np.ndarray) -> np.ndarray:
    points = np.asarray(world_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("world_points must have shape (N, 3)")

    camera = camera_world(metadata)
    world_dirs = points - camera[None, :]
    world_dirs_h = np.column_stack((world_dirs, np.zeros(len(world_dirs))))
    world_to_presentation = np.linalg.inv(presentation_to_world(metadata))
    directions = (world_dirs_h @ world_to_presentation.T)[:, :3]

    if not np.isfinite(directions).all():
        raise RuntimeError("non-finite presentation direction")

    return directions


def spherical_longitudes(
    directions: np.ndarray,
    reference_lon: float,
) -> np.ndarray:
    raw = np.arctan2(directions[:, 0], directions[:, 2])
    return reference_lon + np.arctan2(
        np.sin(raw - reference_lon),
        np.cos(raw - reference_lon),
    )


def raw_ground_grid(
    metadata,
    bounds: PhysicalBounds,
    *,
    max_step_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs = axis_samples(bounds.x_min, bounds.x_max, max_step_m)
    zs = axis_samples(bounds.z_min, bounds.z_max, max_step_m)
    xx, zz = np.meshgrid(xs, zs)

    points = np.stack((xx, np.zeros_like(xx), zz), axis=-1)
    flat = points.reshape(-1, 3)

    directions = presentation_directions(metadata, flat)
    longitude = spherical_longitudes(
        directions,
        presentation_reference(metadata),
    ).reshape(xx.shape)

    near_z, _, depth_sign, _ = depth_geometry(metadata)
    depth = depth_from_near(zz, near_z, depth_sign)

    return xs, zs, points, longitude, depth


def derive_asymmetric_layout(
    metadata,
    *,
    split_fraction: float,
    near_count: int,
    far_count: int,
    max_step_m: float,
) -> AsymmetricOwnershipLayout:
    if near_count < 1 or far_count < 1:
        raise ValueError("near_count and far_count must both be >= 1")
    if not 0.05 < split_fraction < 0.95:
        raise ValueError("split_fraction must be between 0.05 and 0.95")

    bounds = tight_physical_bounds(metadata)
    _, _, points, longitude, depth = raw_ground_grid(
        metadata,
        bounds,
        max_step_m=max_step_m,
    )

    near_z, far_z, depth_sign, total_depth = depth_geometry(metadata)
    split_depth = total_depth * split_fraction

    # Boundary ownership is FAR. This makes the split exact and non-overlapping.
    near_mask = depth < split_depth
    far_mask = ~near_mask

    if not np.any(near_mask) or not np.any(far_mask):
        raise RuntimeError("split produced an empty depth band")

    near_lon_min = float(longitude[near_mask].min())
    near_lon_max = float(longitude[near_mask].max())
    far_lon_min = float(longitude[far_mask].min())
    far_lon_max = float(longitude[far_mask].max())

    near_edges = tuple(
        float(v)
        for v in np.linspace(near_lon_min, near_lon_max, near_count + 1)
    )
    far_edges = tuple(
        float(v)
        for v in np.linspace(far_lon_min, far_lon_max, far_count + 1)
    )

    return AsymmetricOwnershipLayout(
        reference_lon=presentation_reference(metadata),
        near_z=near_z,
        far_z=far_z,
        depth_sign=depth_sign,
        total_depth_m=total_depth,
        split_depth_m=split_depth,
        near_count=near_count,
        far_count=far_count,
        near_lon_edges=near_edges,
        far_lon_edges=far_edges,
    )


def _bin_index(values: np.ndarray, edges: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(edges, dtype=np.float64)
    idx = np.searchsorted(arr, values, side="right") - 1
    return np.clip(idx, 0, len(arr) - 2)


def owner_ids(
    longitude: np.ndarray,
    depth_m: np.ndarray,
    layout: AsymmetricOwnershipLayout,
) -> np.ndarray:
    result = np.empty(longitude.shape, dtype=np.int16)

    near_mask = depth_m < layout.split_depth_m
    far_mask = ~near_mask

    result[near_mask] = _bin_index(
        longitude[near_mask],
        layout.near_lon_edges,
    )

    result[far_mask] = (
        layout.near_count
        + _bin_index(
            longitude[far_mask],
            layout.far_lon_edges,
        )
    )

    return result


def build_ground_grid(
    metadata,
    bounds: PhysicalBounds,
    layout: AsymmetricOwnershipLayout,
    *,
    max_step_m: float,
) -> GroundGrid:
    xs, zs, points, longitude, depth = raw_ground_grid(
        metadata,
        bounds,
        max_step_m=max_step_m,
    )

    return GroundGrid(
        bounds=bounds,
        xs=xs,
        zs=zs,
        points=points,
        longitude=longitude,
        depth_m=depth,
        owner_id=owner_ids(longitude, depth, layout),
    )


def guard_kernel(grid: GroundGrid, guard_margin_m: float) -> np.ndarray:
    if guard_margin_m <= 0.0:
        return np.ones((1, 1), dtype=np.uint8)

    dx = float(np.median(np.diff(grid.xs))) if len(grid.xs) > 1 else guard_margin_m
    dz = float(np.median(np.diff(grid.zs))) if len(grid.zs) > 1 else guard_margin_m

    radius_x = max(1, int(math.ceil(guard_margin_m / max(dx, EPS))))
    radius_z = max(1, int(math.ceil(guard_margin_m / max(dz, EPS))))

    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * radius_x + 1, 2 * radius_z + 1),
    )


def guarded_cell_mask(
    grid: GroundGrid,
    owner_id: int,
    *,
    guard_margin_m: float,
) -> np.ndarray:
    logical = (grid.owner_id == owner_id).astype(np.uint8)

    if not np.any(logical):
        raise RuntimeError(f"owner {owner_id} has no sampled ground points")

    if guard_margin_m <= 0.0:
        return logical.astype(bool)

    return cv2.dilate(
        logical,
        guard_kernel(grid, guard_margin_m),
        iterations=1,
    ).astype(bool)


def render_world_points_for_cell(
    metadata,
    grid: GroundGrid,
    owner_id: int,
    *,
    guard_margin_m: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    render_mask = guarded_cell_mask(
        grid,
        owner_id,
        guard_margin_m=guard_margin_m,
    )

    logical_mask = grid.owner_id == owner_id
    ground = grid.points[render_mask]

    top = ground.copy()
    top[:, 1] = player_top_y(metadata)

    return np.concatenate((ground, top), axis=0), logical_mask, len(ground)


def world_point_lens_validity(metadata, world_points: np.ndarray) -> np.ndarray:
    points = np.asarray(world_points, dtype=np.float64)
    target = np.column_stack((points, np.ones(len(points), dtype=np.float64)))

    _, _, left_valid = project_world_target_to_lens(
        target,
        metadata.left,
        metadata.camera2world,
        SOURCE_W,
        SOURCE_H,
        True,
    )
    _, _, right_valid = project_world_target_to_lens(
        target,
        metadata.right,
        metadata.camera2world,
        SOURCE_W,
        SOURCE_H,
        False,
    )

    return left_valid | right_valid


def required_fov_for_orientation(
    presentation_dirs: np.ndarray,
    longitude: float,
    latitude: float,
    aspect: float,
) -> tuple[float, float, float] | None:
    rotation = (rot_y(longitude) @ rot_x(latitude))[:3, :3]

    local = presentation_dirs @ rotation
    z = local[:, 2]

    if np.any(~np.isfinite(local)) or np.any(z <= EPS):
        return None

    nx = local[:, 0] / z
    ny = local[:, 1] / z

    half_tan_h = float(np.max(np.abs(nx)))
    half_tan_v = float(np.max(np.abs(ny)))

    half_tan_vertical = max(
        half_tan_v,
        half_tan_h / aspect,
    ) * FOV_SAFETY_SCALE

    if not math.isfinite(half_tan_vertical) or half_tan_vertical <= 0.0:
        return None

    vfov = 2.0 * math.atan(half_tan_vertical)
    hfov = 2.0 * math.atan(half_tan_vertical * aspect)

    if vfov >= math.radians(MAX_VFOV_DEG):
        return None

    return hfov, vfov, half_tan_vertical


def fit_rectilinear_camera(
    metadata,
    world_points: np.ndarray,
    width: int,
    height: int,
) -> ViewFit:
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")

    aspect = width / height
    directions = presentation_directions(metadata, world_points)

    raw_lon = spherical_longitudes(
        directions,
        presentation_reference(metadata),
    )
    raw_lat = np.arctan2(
        -directions[:, 1],
        np.hypot(directions[:, 0], directions[:, 2]),
    )

    lon_min = float(raw_lon.min())
    lon_max = float(raw_lon.max())
    lat_min = float(raw_lat.min())
    lat_max = float(raw_lat.max())

    lon_mid = 0.5 * (lon_min + lon_max)
    lat_mid = 0.5 * (lat_min + lat_max)

    lon_span = max(lon_max - lon_min, math.radians(0.20))
    lat_span = max(lat_max - lat_min, math.radians(0.20))

    box = [
        lon_min - max(0.20 * lon_span, math.radians(0.25)),
        lon_max + max(0.20 * lon_span, math.radians(0.25)),
        lat_min - max(0.20 * lat_span, math.radians(0.25)),
        lat_max + max(0.20 * lat_span, math.radians(0.25)),
    ]

    def search(
        search_box: list[float],
        grid_size: int,
    ) -> tuple[ViewFit | None, float, float]:
        local_best: ViewFit | None = None

        lon_values = np.linspace(
            search_box[0],
            search_box[1],
            grid_size,
        )
        lat_values = np.linspace(
            search_box[2],
            search_box[3],
            grid_size,
        )

        for lon in lon_values:
            for lat in lat_values:
                fitted = required_fov_for_orientation(
                    directions,
                    float(lon),
                    float(lat),
                    aspect,
                )
                if fitted is None:
                    continue

                hfov, vfov, score = fitted
                candidate = ViewFit(
                    float(lon),
                    float(lat),
                    hfov,
                    vfov,
                    score,
                )

                if local_best is None:
                    local_best = candidate
                    continue

                tie = (
                    abs(candidate.longitude - lon_mid)
                    + abs(candidate.latitude - lat_mid)
                )
                best_tie = (
                    abs(local_best.longitude - lon_mid)
                    + abs(local_best.latitude - lat_mid)
                )

                if (
                    candidate.half_tan_vertical
                    < local_best.half_tan_vertical - 1e-12
                    or (
                        abs(
                            candidate.half_tan_vertical
                            - local_best.half_tan_vertical
                        )
                        <= 1e-12
                        and tie < best_tie
                    )
                ):
                    local_best = candidate

        lon_step = (
            search_box[1] - search_box[0]
        ) / max(grid_size - 1, 1)
        lat_step = (
            search_box[3] - search_box[2]
        ) / max(grid_size - 1, 1)

        return local_best, lon_step, lat_step

    best, lon_step, lat_step = search(box, SEARCH_GRID)

    if best is None:
        raise RuntimeError(
            "could not fit a rectilinear camera to required world points"
        )

    for _ in range(SEARCH_REFINEMENTS):
        box = [
            best.longitude - 1.5 * lon_step,
            best.longitude + 1.5 * lon_step,
            best.latitude - 1.5 * lat_step,
            best.latitude + 1.5 * lat_step,
        ]

        refined, lon_step, lat_step = search(box, 11)
        if refined is not None:
            best = refined

    return best


def fit_longitude_at_fixed_latitude(
    metadata,
    world_points: np.ndarray,
    width: int,
    height: int,
    *,
    latitude: float,
) -> ViewFit:
    """Optimize horizontal pan while keeping a prescribed latitude/pitch.

    This is used after the common pitch for a depth band has been selected.
    Every camera retains an independent longitude so it can stay centered on
    its own logical sector.
    """
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")

    aspect = width / height
    directions = presentation_directions(
        metadata,
        world_points,
    )

    raw_lon = spherical_longitudes(
        directions,
        presentation_reference(metadata),
    )

    lon_min = float(raw_lon.min())
    lon_max = float(raw_lon.max())
    lon_mid = 0.5 * (lon_min + lon_max)
    lon_span = max(
        lon_max - lon_min,
        math.radians(0.20),
    )

    margin = max(
        0.20 * lon_span,
        math.radians(0.25),
    )

    def search(
        minimum: float,
        maximum: float,
        grid_size: int,
    ) -> tuple[ViewFit | None, float]:
        best: ViewFit | None = None

        for longitude in np.linspace(
            minimum,
            maximum,
            grid_size,
        ):
            fitted = required_fov_for_orientation(
                directions,
                float(longitude),
                float(latitude),
                aspect,
            )

            if fitted is None:
                continue

            hfov, vfov, score = fitted

            candidate = ViewFit(
                longitude=float(longitude),
                latitude=float(latitude),
                hfov=hfov,
                vfov=vfov,
                half_tan_vertical=score,
            )

            if best is None:
                best = candidate
                continue

            candidate_tie = abs(
                candidate.longitude - lon_mid
            )
            best_tie = abs(
                best.longitude - lon_mid
            )

            if (
                candidate.half_tan_vertical
                < best.half_tan_vertical - 1e-12
                or (
                    abs(
                        candidate.half_tan_vertical
                        - best.half_tan_vertical
                    )
                    <= 1e-12
                    and candidate_tie < best_tie
                )
            ):
                best = candidate

        step = (
            maximum - minimum
        ) / max(grid_size - 1, 1)

        return best, step

    minimum = lon_min - margin
    maximum = lon_max + margin

    best, step = search(
        minimum,
        maximum,
        FIXED_LAT_LON_SEARCH_GRID,
    )

    if best is None:
        raise RuntimeError(
            "could not fit camera longitude at fixed latitude"
        )

    for _ in range(FIXED_LAT_LON_REFINEMENTS):
        minimum = best.longitude - 1.5 * step
        maximum = best.longitude + 1.5 * step

        refined, new_step = search(
            minimum,
            maximum,
            11,
        )

        if refined is not None:
            best = refined

        step = new_step

    return best


def fit_band_with_shared_latitude(
    metadata,
    cell_world_points: list[
        tuple[CellOwnership, np.ndarray]
    ],
    width: int,
    height: int,
) -> dict[int, ViewFit]:
    """Jointly optimize one latitude/pitch for a complete depth band.

    FAR cameras share one pitch and NEAR cameras share another.  For each
    candidate shared latitude, every camera independently optimizes its own
    longitude.  The selected band latitude minimizes:

      1. worst required FOV in the band,
      2. mean required FOV,
      3. distance from the median independent optimum.

    The FOV score is ``half_tan_vertical``.  At a fixed output aspect ratio it
    is monotonic with both vertical and horizontal FOV, so minimizing it is
    equivalent to maximizing useful zoom.
    """
    if not cell_world_points:
        raise ValueError(
            "cell_world_points cannot be empty"
        )

    independent = [
        fit_rectilinear_camera(
            metadata,
            world_points,
            width,
            height,
        )
        for _, world_points in cell_world_points
    ]

    independent_latitudes = np.asarray(
        [
            fit.latitude
            for fit in independent
        ],
        dtype=np.float64,
    )

    target_latitude = float(
        np.median(independent_latitudes)
    )

    latitude_min = float(
        independent_latitudes.min()
    )
    latitude_max = float(
        independent_latitudes.max()
    )

    latitude_span = max(
        latitude_max - latitude_min,
        math.radians(0.50),
    )

    margin = max(
        0.75 * latitude_span,
        math.radians(1.00),
    )

    def evaluate(
        latitude: float,
    ) -> tuple[
        tuple[float, float, float],
        dict[int, ViewFit],
    ] | None:
        fits: dict[int, ViewFit] = {}

        for cell, world_points in cell_world_points:
            try:
                fit = (
                    fit_longitude_at_fixed_latitude(
                        metadata,
                        world_points,
                        width,
                        height,
                        latitude=float(latitude),
                    )
                )
            except RuntimeError:
                return None

            fits[cell.owner_id] = fit

        scores = np.asarray(
            [
                fit.half_tan_vertical
                for fit in fits.values()
            ],
            dtype=np.float64,
        )

        objective = (
            float(scores.max()),
            float(scores.mean()),
            abs(
                float(latitude)
                - target_latitude
            ),
        )

        return objective, fits

    def search(
        minimum: float,
        maximum: float,
        grid_size: int,
    ) -> tuple[
        tuple[float, float, float] | None,
        dict[int, ViewFit] | None,
        float | None,
        float,
    ]:
        local_objective = None
        local_fits = None
        local_latitude = None

        for latitude in np.linspace(
            minimum,
            maximum,
            grid_size,
        ):
            evaluated = evaluate(
                float(latitude)
            )

            if evaluated is None:
                continue

            objective, fits = evaluated

            if (
                local_objective is None
                or objective < local_objective
            ):
                local_objective = objective
                local_fits = fits
                local_latitude = float(latitude)

        step = (
            maximum - minimum
        ) / max(grid_size - 1, 1)

        return (
            local_objective,
            local_fits,
            local_latitude,
            step,
        )

    minimum = latitude_min - margin
    maximum = latitude_max + margin

    (
        best_objective,
        best_fits,
        best_latitude,
        step,
    ) = search(
        minimum,
        maximum,
        BAND_LAT_SEARCH_GRID,
    )

    if (
        best_objective is None
        or best_fits is None
        or best_latitude is None
    ):
        raise RuntimeError(
            "could not fit a shared latitude for band"
        )

    for _ in range(BAND_LAT_REFINEMENTS):
        minimum = best_latitude - 1.5 * step
        maximum = best_latitude + 1.5 * step

        (
            local_objective,
            local_fits,
            local_latitude,
            new_step,
        ) = search(
            minimum,
            maximum,
            11,
        )

        if (
            local_objective is not None
            and local_fits is not None
            and local_latitude is not None
            and local_objective < best_objective
        ):
            best_objective = local_objective
            best_fits = local_fits
            best_latitude = local_latitude

        step = new_step

    # Force bit-identical latitude in every camera of the band.  The fits were
    # already evaluated at this exact value; this simply makes the invariant
    # explicit in the returned objects.
    shared_latitude = float(
        best_latitude
    )

    return {
        owner_id: ViewFit(
            longitude=fit.longitude,
            latitude=shared_latitude,
            hfov=fit.hfov,
            vfov=fit.vfov,
            half_tan_vertical=fit.half_tan_vertical,
        )
        for owner_id, fit in best_fits.items()
    }


def derive_mode_specs(
    metadata,
    *,
    mode: str,
    grid: GroundGrid,
    layout: AsymmetricOwnershipLayout,
    width: int,
    height: int,
    guard_margin_m: float,
) -> list[CameraSpec]:
    """Derive one mode with a shared pitch per NEAR/FAR depth band."""
    prepared: dict[
        int,
        tuple[
            CellOwnership,
            np.ndarray,
            int,
            np.ndarray,
        ],
    ] = {}

    for cell in layout.cells():
        (
            world_points,
            _,
            ground_count,
        ) = render_world_points_for_cell(
            metadata,
            grid,
            cell.owner_id,
            guard_margin_m=guard_margin_m,
        )

        valid = world_point_lens_validity(
            metadata,
            world_points,
        )

        prepared[cell.owner_id] = (
            cell,
            world_points,
            ground_count,
            valid,
        )

    fits_by_owner: dict[int, ViewFit] = {}

    for band in ("near", "far"):
        band_items = [
            (
                cell,
                world_points,
            )
            for (
                cell,
                world_points,
                _,
                _,
            ) in prepared.values()
            if cell.band == band
        ]

        band_fits = (
            fit_band_with_shared_latitude(
                metadata,
                band_items,
                width,
                height,
            )
        )

        fits_by_owner.update(
            band_fits
        )

    specs: list[CameraSpec] = []

    # Preserve layout.cells() order so filenames and contact sheets remain
    # stable across versions.
    for cell in layout.cells():
        (
            _,
            world_points,
            ground_count,
            valid,
        ) = prepared[cell.owner_id]

        fit = fits_by_owner[
            cell.owner_id
        ]

        specs.append(
            CameraSpec(
                name=f"{mode}_{cell.name}",
                mode=mode,
                cell=cell,
                longitude=fit.longitude,
                latitude=fit.latitude,
                hfov=fit.hfov,
                vfov=fit.vfov,
                required_ground_points=ground_count,
                required_world_points=len(
                    world_points
                ),
                required_point_valid_fraction=float(
                    valid.mean()
                ),
                required_invalid_world_points=int(
                    (~valid).sum()
                ),
                physical_bounds=grid.bounds,
            )
        )

    return specs


def derive_specs(
    metadata,
    width: int,
    height: int,
    *,
    split_fraction: float = DEFAULT_SPLIT_FRACTION,
    near_count: int = DEFAULT_NEAR_CAMERAS,
    far_count: int = DEFAULT_FAR_CAMERAS,
    guard_margin_m: float = DEFAULT_GUARD_M,
    near_context_m: float = DEFAULT_NEAR_CONTEXT_M,
    outer_context_m: float = DEFAULT_OUTER_CONTEXT_M,
    grid_step_m: float = DEFAULT_GRID_STEP_M,
) -> tuple[
    list[CameraSpec],
    AsymmetricOwnershipLayout,
    GroundGrid,
    GroundGrid,
]:
    if near_count + far_count < 2:
        raise ValueError("at least two cameras are required")
    if guard_margin_m < 0.0:
        raise ValueError("guard_margin_m must be >= 0")
    if near_context_m < 0.0 or outer_context_m < 0.0:
        raise ValueError("context values must be >= 0")
    if grid_step_m <= 0.0:
        raise ValueError("grid_step_m must be > 0")

    layout = derive_asymmetric_layout(
        metadata,
        split_fraction=split_fraction,
        near_count=near_count,
        far_count=far_count,
        max_step_m=grid_step_m,
    )

    tight_grid = build_ground_grid(
        metadata,
        tight_physical_bounds(metadata),
        layout,
        max_step_m=grid_step_m,
    )

    context_grid = build_ground_grid(
        metadata,
        context_physical_bounds(
            metadata,
            near_context_m=near_context_m,
            outer_context_m=outer_context_m,
        ),
        layout,
        max_step_m=grid_step_m,
    )

    specs = derive_mode_specs(
        metadata,
        mode="tight",
        grid=tight_grid,
        layout=layout,
        width=width,
        height=height,
        guard_margin_m=guard_margin_m,
    )

    specs.extend(
        derive_mode_specs(
            metadata,
            mode="context",
            grid=context_grid,
            layout=layout,
            width=width,
            height=height,
            guard_margin_m=guard_margin_m,
        )
    )

    return specs, layout, tight_grid, context_grid


def build_map(
    metadata,
    spec: CameraSpec,
    width: int,
    height: int,
) -> CameraMap:
    target_world, blend = validated_virtual_geometry(
        metadata,
        spec,
        width,
        height,
    )

    left_x, left_y, left_valid = project_world_target_to_lens(
        target_world,
        metadata.left,
        metadata.camera2world,
        SOURCE_W,
        SOURCE_H,
        True,
    )

    right_x, right_y, right_valid = project_world_target_to_lens(
        target_world,
        metadata.right,
        metadata.camera2world,
        SOURCE_W,
        SOURCE_H,
        False,
    )

    shape = (height, width)

    return CameraMap(
        spec=spec,
        left_x=left_x.reshape(shape),
        left_y=left_y.reshape(shape),
        right_x=right_x.reshape(shape),
        right_y=right_y.reshape(shape),
        left_valid=left_valid.reshape(shape),
        right_valid=right_valid.reshape(shape),
        blend=blend,
    )


def render_map(
    frame: np.ndarray,
    camera_map: CameraMap,
) -> np.ndarray:
    left = cv2.remap(
        frame,
        camera_map.left_x,
        camera_map.left_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    right = cv2.remap(
        frame,
        camera_map.right_x,
        camera_map.right_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    blend3 = camera_map.blend[..., None].astype(np.float32)

    output = (
        left.astype(np.float32) * (1.0 - blend3)
        + right.astype(np.float32) * blend3
    )

    only_left = camera_map.left_valid & ~camera_map.right_valid
    only_right = camera_map.right_valid & ~camera_map.left_valid
    neither = ~(camera_map.left_valid | camera_map.right_valid)

    output[only_left] = left[only_left]
    output[only_right] = right[only_right]
    output[neither] = 0

    return np.clip(output, 0, 255).astype(np.uint8)


def world_points_to_virtual_pixels(
    metadata,
    spec: CameraSpec,
    world_points: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    dirs = presentation_directions(
        metadata,
        np.asarray(world_points, dtype=np.float64),
    )

    rotation = (
        rot_y(spec.longitude)
        @ rot_x(spec.latitude)
    )[:3, :3]

    local = dirs @ rotation
    z = local[:, 2]

    valid = (
        np.isfinite(local).all(axis=1)
        & (z > EPS)
    )

    points = np.full((len(local), 2), np.nan, dtype=np.float64)

    tan_v = math.tan(spec.vfov / 2.0)
    aspect = width / height

    nx = np.zeros(len(local), dtype=np.float64)
    ny = np.zeros(len(local), dtype=np.float64)

    nx[valid] = local[valid, 0] / z[valid]
    ny[valid] = local[valid, 1] / z[valid]

    u = 0.5 * (
        nx / (tan_v * aspect)
        + 1.0
    )

    v = 0.5 * (
        ny / tan_v
        + 1.0
    )

    points[:, 0] = u * width
    points[:, 1] = v * height

    return points


def draw_ownership_hull(
    image: np.ndarray,
    metadata,
    spec: CameraSpec,
    grid: GroundGrid,
) -> np.ndarray:
    output = image.copy()

    logical_ground = grid.points[
        grid.owner_id == spec.cell.owner_id
    ]

    if len(logical_ground) < 3:
        return output

    pixels = world_points_to_virtual_pixels(
        metadata,
        spec,
        logical_ground,
        image.shape[1],
        image.shape[0],
    )

    valid = np.isfinite(pixels).all(axis=1)
    pixels = pixels[valid]

    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < image.shape[1])
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < image.shape[0])
    )

    pixels = pixels[inside]

    if len(pixels) < 3:
        return output

    hull = cv2.convexHull(
        np.round(pixels).astype(np.int32)
    )

    cv2.polylines(
        output,
        [hull],
        True,
        (255, 0, 255),
        3,
        cv2.LINE_AA,
    )

    return output


def read_frame(
    video: Path,
    time_s: float,
) -> tuple[np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")

    fps = float(
        cap.get(cv2.CAP_PROP_FPS)
    )

    if fps <= 0.0:
        cap.release()
        raise RuntimeError("video reports invalid FPS")

    requested = max(
        0,
        int(round(time_s * fps)),
    )

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        requested,
    )

    ok, frame = cap.read()

    actual = int(
        cap.get(cv2.CAP_PROP_POS_FRAMES)
    ) - 1

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame near {time_s:.3f}s"
        )

    h, w = frame.shape[:2]

    if (w, h) != (SOURCE_W, SOURCE_H):
        raise RuntimeError(
            f"Expected {SOURCE_W}x{SOURCE_H} source frame, got {w}x{h}"
        )

    return frame, {
        "fps": fps,
        "requested_time_s": float(time_s),
        "requested_frame": requested,
        "actual_frame": actual,
        "actual_time_s": actual / fps,
        "source_width": w,
        "source_height": h,
    }


def camera_diagnostics(camera_map: CameraMap) -> dict:
    either = camera_map.left_valid | camera_map.right_valid
    both = camera_map.left_valid & camera_map.right_valid

    result = camera_map.spec.json()

    result.update(
        {
            "left_valid_fraction": float(
                camera_map.left_valid.mean()
            ),
            "right_valid_fraction": float(
                camera_map.right_valid.mean()
            ),
            "either_lens_valid_fraction": float(
                either.mean()
            ),
            "both_lenses_valid_fraction": float(
                both.mean()
            ),
            "rendered_invalid_fraction": float(
                (~either).mean()
            ),
        }
    )

    return result


def label_panel(
    panel: np.ndarray,
    text: str,
) -> np.ndarray:
    output = panel.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 42),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        output,
        text,
        (8, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output


def _mode_row(
    *,
    mode: str,
    band: str,
    count: int,
    images: dict[str, np.ndarray],
    maps: dict[str, CameraMap],
    panel_width: int,
    total_columns: int,
) -> np.ndarray:
    row_width = total_columns * panel_width
    panels: list[np.ndarray] = []

    for i in range(count):
        name = f"{mode}_{band}_c{i + 1}"
        camera_map = maps[name]
        image = images[name]

        panel_height = int(
            round(
                image.shape[0]
                * panel_width
                / image.shape[1]
            )
        )

        panel = cv2.resize(
            image,
            (panel_width, panel_height),
            interpolation=cv2.INTER_AREA,
        )

        either = (
            camera_map.left_valid
            | camera_map.right_valid
        )

        spec = camera_map.spec

        text = (
            f"{name}  "
            f"H {math.degrees(spec.hfov):.1f}  "
            f"V {math.degrees(spec.vfov):.1f}  "
            f"img {100.0 * float(either.mean()):.1f}%  "
            f"req {100.0 * spec.required_point_valid_fraction:.1f}%"
        )

        panels.append(
            label_panel(panel, text)
        )

    content = np.concatenate(
        panels,
        axis=1,
    )

    canvas = np.zeros(
        (
            content.shape[0],
            row_width,
            3,
        ),
        dtype=np.uint8,
    )

    x0 = (
        row_width
        - content.shape[1]
    ) // 2

    canvas[
        :,
        x0:x0 + content.shape[1],
    ] = content

    return canvas


def contact_sheet(
    images: dict[str, np.ndarray],
    maps: dict[str, CameraMap],
    *,
    near_count: int,
    far_count: int,
    panel_width: int,
) -> np.ndarray:
    total_columns = max(
        near_count,
        far_count,
    )

    sections: list[np.ndarray] = []

    for mode in ("tight", "context"):
        title = np.zeros(
            (
                46,
                total_columns * panel_width,
                3,
            ),
            dtype=np.uint8,
        )

        cv2.putText(
            title,
            mode.upper(),
            (12, 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        sections.append(title)

        sections.append(
            _mode_row(
                mode=mode,
                band="far",
                count=far_count,
                images=images,
                maps=maps,
                panel_width=panel_width,
                total_columns=total_columns,
            )
        )

        sections.append(
            _mode_row(
                mode=mode,
                band="near",
                count=near_count,
                images=images,
                maps=maps,
                panel_width=panel_width,
                total_columns=total_columns,
            )
        )

    return np.concatenate(
        sections,
        axis=0,
    )


def ownership_summary(
    grid: GroundGrid,
    layout: AsymmetricOwnershipLayout,
) -> dict:
    total = int(grid.owner_id.size)

    counts = {
        cell.name: int(
            np.sum(
                grid.owner_id == cell.owner_id
            )
        )
        for cell in layout.cells()
    }

    return {
        "sampled_ground_points": total,
        "owner_counts": counts,
        "all_points_have_exactly_one_owner": (
            sum(counts.values()) == total
        ),
    }


def run(args: argparse.Namespace) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    if not args.calibration.is_file():
        raise FileNotFoundError(args.calibration)

    metadata = load_veo_metadata(
        args.calibration
    )

    print(
        "Asymmetric Veo coverage experiment: "
        f"{args.far_cameras} FAR + "
        f"{args.near_cameras} NEAR"
    )

    specs, layout, tight_grid, context_grid = derive_specs(
        metadata,
        args.width,
        args.height,
        split_fraction=args.split_fraction,
        near_count=args.near_cameras,
        far_count=args.far_cameras,
        guard_margin_m=args.guard_margin_m,
        near_context_m=args.near_context_m,
        outer_context_m=args.outer_context_m,
        grid_step_m=args.grid_step_m,
    )

    print(
        f"  field: "
        f"{metadata.field_length:.3f} x "
        f"{metadata.field_width:.3f} m"
    )

    print(
        f"  camera world: "
        f"{camera_world(metadata).tolist()}"
    )

    print(
        f"  split: "
        f"{layout.split_depth_m:.3f} m "
        f"from near touchline "
        f"({100.0 * args.split_fraction:.1f}% depth)"
    )

    print(
        "  NEAR longitude edges: "
        + ", ".join(
            f"{math.degrees(v):.2f}deg"
            for v in layout.near_lon_edges
        )
    )

    print(
        "  FAR longitude edges:  "
        + ", ".join(
            f"{math.degrees(v):.2f}deg"
            for v in layout.far_lon_edges
        )
    )

    for spec in specs:
        print(
            f"  {spec.name:18s} "
            f"HFOV={math.degrees(spec.hfov):6.2f}deg "
            f"VFOV={math.degrees(spec.vfov):6.2f}deg "
            f"required-visible="
            f"{100.0 * spec.required_point_valid_fraction:6.2f}%"
        )

    print(
        "Building static direct-lens remap grids..."
    )

    maps = {
        spec.name: build_map(
            metadata,
            spec,
            args.width,
            args.height,
        )
        for spec in specs
    }

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "experiment": "asymmetric_virtual_camera_coverage_4far_2near",
        "video": str(args.video),
        "calibration": str(args.calibration),
        "field_length_m": float(
            metadata.field_length
        ),
        "field_width_m": float(
            metadata.field_width
        ),
        "camera_world": camera_world(
            metadata
        ).tolist(),
        "player_context_height_m": float(
            CONTEXT_HEIGHT_M
        ),
        "player_top_world_y": player_top_y(
            metadata
        ),
        "guard_margin_m": args.guard_margin_m,
        "near_context_m": args.near_context_m,
        "outer_context_m": args.outer_context_m,
        "grid_step_m": args.grid_step_m,
        "render_size_per_view": [
            args.width,
            args.height,
        ],
        "ownership_layout": layout.to_dict(),
        "tight_ownership_diagnostics": ownership_summary(
            tight_grid,
            layout,
        ),
        "context_ownership_diagnostics": ownership_summary(
            context_grid,
            layout,
        ),
        "shared_band_latitudes_deg": {
            mode: {
                band: math.degrees(next(
                    spec.latitude
                    for spec in specs
                    if spec.mode == mode and spec.cell.band == band
                ))
                for band in ("near", "far")
            }
            for mode in ("tight", "context")
        },
        "cameras": [
            camera_diagnostics(
                maps[spec.name]
            )
            for spec in specs
        ],
        "frames": [],
    }

    grid_for_mode = {
        "tight": tight_grid,
        "context": context_grid,
    }

    for time_s in args.times:
        print(
            f"Rendering t={time_s:.3f}s ..."
        )

        frame, frame_info = read_frame(
            args.video,
            time_s,
        )

        raw_images = {
            name: render_map(
                frame,
                camera_map,
            )
            for name, camera_map
            in maps.items()
        }

        annotated_images = {
            name: draw_ownership_hull(
                image,
                metadata,
                maps[name].spec,
                grid_for_mode[
                    maps[name].spec.mode
                ],
            )
            for name, image
            in raw_images.items()
        }

        frame_dir = (
            args.out_dir
            / f"frame_{frame_info['actual_frame']:06d}"
        )

        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                frame_dir
                / "source_stacked.png"
            ),
            frame,
        )

        for name, image in raw_images.items():
            cv2.imwrite(
                str(
                    frame_dir
                    / f"{name}.png"
                ),
                image,
            )

        for name, image in annotated_images.items():
            cv2.imwrite(
                str(
                    frame_dir
                    / f"{name}_ownership.png"
                ),
                image,
            )

        comparison = contact_sheet(
            annotated_images,
            maps,
            near_count=args.near_cameras,
            far_count=args.far_cameras,
            panel_width=args.panel_width,
        )

        comparison_path = (
            frame_dir
            / "comparison.jpg"
        )

        cv2.imwrite(
            str(comparison_path),
            comparison,
        )

        summary["frames"].append(
            {
                "source": frame_info,
                "comparison": str(
                    comparison_path
                ),
            }
        )

        print(
            f"  -> {comparison_path}"
        )

    summary_path = (
        args.out_dir
        / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Summary: {summary_path}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--times",
        type=float,
        nargs="+",
        required=True,
        help="Exactly five source timestamps in seconds.",
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
    )

    parser.add_argument(
        "--near-cameras",
        type=int,
        default=DEFAULT_NEAR_CAMERAS,
    )

    parser.add_argument(
        "--far-cameras",
        type=int,
        default=DEFAULT_FAR_CAMERAS,
    )

    parser.add_argument(
        "--split-fraction",
        type=float,
        default=DEFAULT_SPLIT_FRACTION,
        help=(
            "Physical pitch depth assigned to NEAR, measured from the "
            "calibrated near touchline. Default: 0.50."
        ),
    )

    parser.add_argument(
        "--guard-margin-m",
        type=float,
        default=DEFAULT_GUARD_M,
    )

    parser.add_argument(
        "--near-context-m",
        type=float,
        default=DEFAULT_NEAR_CONTEXT_M,
    )

    parser.add_argument(
        "--outer-context-m",
        type=float,
        default=DEFAULT_OUTER_CONTEXT_M,
    )

    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=DEFAULT_GRID_STEP_M,
    )

    parser.add_argument(
        "--panel-width",
        type=int,
        default=420,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )

    args = parser.parse_args()

    if len(args.times) != 5:
        parser.error(
            "--times must contain exactly five timestamps"
        )

    if (
        args.width <= 0
        or args.height <= 0
        or args.panel_width <= 0
    ):
        parser.error(
            "image dimensions must be positive"
        )

    if (
        args.near_cameras < 1
        or args.far_cameras < 1
    ):
        parser.error(
            "--near-cameras and --far-cameras must be >= 1"
        )

    if (
        args.near_cameras
        + args.far_cameras
        != 6
    ):
        parser.error(
            "this experiment is intentionally fixed to six cameras total"
        )

    if not 0.05 < args.split_fraction < 0.95:
        parser.error(
            "--split-fraction must be between 0.05 and 0.95"
        )

    if args.guard_margin_m < 0.0:
        parser.error(
            "--guard-margin-m must be >= 0"
        )

    if (
        args.near_context_m < 0.0
        or args.outer_context_m < 0.0
    ):
        parser.error(
            "context distances must be >= 0"
        )

    if args.grid_step_m <= 0.0:
        parser.error(
            "--grid-step-m must be > 0"
        )

    return args


if __name__ == "__main__":
    run(
        parse_args()
    )
