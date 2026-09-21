"""Exact Veo Tracking/Follow geometry reconstructed from the web player.

This module mirrors the code path used by Veo's own "draw on pitch" /
player-tracking overlay for the pre-rendered Standard/Follow video.

Observed Veo implementation:
- video frame rate constant: 30000 / 1001
- camera row: round(video_time * fps) - 1, clamped at 0
- calibration >= 6.2:
    camera2logic = alignment.camera2logic
    logic2camera = inverse(camera2logic)
- dynamic view transform:
    view_to_world =
        camera2world
        @ logic2camera
        @ Ry(yaw)
        @ Rx(pitch)
- vertical FOV from detections_constrained.det
- normalized screen coordinates have origin at TOP-LEFT:
    u = 0.5 + (x/z) / viewport_width
    v = 0.5 + (y/z) / viewport_height

This deliberately does not modify the already validated panorama/RAW geometry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np


TRACKING_FPS = 30000.0 / 1001.0
TRACKING_ASPECT = 16.0 / 9.0
EPSILON = 1e-12


class TrackingGeometryMetadata(Protocol):
    raw_json: dict[str, Any]
    camera2world: np.ndarray
    field_length: float
    field_width: float


@dataclass(frozen=True)
class CameraDirectionSample:
    time_s: float
    row_index: int
    frame_position: float
    yaw: float
    pitch: float
    fov: float
    extra: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "time_s": self.time_s,
            "row_index": self.row_index,
            "frame_position": self.frame_position,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "fov": self.fov,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class TrackingCameraState:
    sample: CameraDirectionSample
    logic2camera: np.ndarray
    view_to_world: np.ndarray
    world_to_view: np.ndarray
    camera_world: np.ndarray
    viewport_width: float
    viewport_height: float

    @property
    def fov(self) -> float:
        return self.sample.fov

    def to_dict(self, *, include_matrices: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            **self.sample.to_dict(),
            "viewport_width": self.viewport_width,
            "viewport_height": self.viewport_height,
            "camera_world": self.camera_world.tolist(),
        }
        if include_matrices:
            result["logic2camera"] = self.logic2camera.tolist()
            result["view_to_world"] = self.view_to_world.tolist()
            result["world_to_view"] = self.world_to_view.tolist()
        return result


@dataclass(frozen=True)
class TrackingProjection:
    world_x: float
    world_y: float
    world_z: float
    pixel_x: float | None
    pixel_y: float | None
    u: float | None
    v: float | None
    visible: bool
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "world_x": self.world_x,
            "world_y": self.world_y,
            "world_z": self.world_z,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "u": self.u,
            "v": self.v,
            "visible": self.visible,
            "status": self.status,
        }


@dataclass(frozen=True)
class TrackingGroundIntersection:
    pixel_x: float
    pixel_y: float
    width: int
    height: int
    u: float
    v: float
    time_s: float
    row_index: int
    yaw: float
    pitch: float
    fov: float
    world_x: float | None
    world_z: float | None
    ground_distance_m: float | None
    inside_field: bool | None
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "width": self.width,
            "height": self.height,
            "u": self.u,
            "v": self.v,
            "time_s": self.time_s,
            "row_index": self.row_index,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "fov": self.fov,
            "world_x": self.world_x,
            "world_z": self.world_z,
            "ground_distance_m": self.ground_distance_m,
            "inside_field": self.inside_field,
            "status": self.status,
        }


class LegacyCameraDirections:
    """Veo detections_constrained.det: one CSV row per Standard video frame."""

    def __init__(self, values: np.ndarray, source_path: Path | None = None):
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 4:
            raise ValueError("Camera directions must have shape (N, 4)")
        if len(values) < 1:
            raise ValueError("Camera directions file is empty")
        self.values = values
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path) -> "LegacyCameraDirections":
        source_path = Path(path)
        rows: list[list[float]] = []

        for line_number, line in enumerate(
            source_path.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1,
        ):
            if not line.strip():
                continue
            parts = [item.strip() for item in line.split(",")]
            if len(parts) != 4:
                raise ValueError(
                    f"Invalid camera-direction row {line_number}: expected 4 values"
                )
            rows.append([float(item) for item in parts])

        return cls(np.asarray(rows, dtype=np.float64), source_path=source_path)

    @property
    def sample_count(self) -> int:
        return len(self.values)

    def sample_at_time(
        self,
        time_s: float,
        *,
        fps: float = TRACKING_FPS,
    ) -> CameraDirectionSample | None:
        if not math.isfinite(time_s) or time_s < 0.0:
            return None

        frame_position = float(time_s) * float(fps)

        # Exact observed Veo player-tracking provider:
        #   n = Math.round(videoTime * FPS)
        #   row = cameraDirections[Math.max(0, n - 1)]
        row_index = max(0, int(math.floor(frame_position + 0.5)) - 1)

        if row_index >= len(self.values):
            return None

        yaw, pitch, fov, extra = self.values[row_index]
        return CameraDirectionSample(
            time_s=float(time_s),
            row_index=row_index,
            frame_position=frame_position,
            yaw=float(yaw),
            pitch=float(pitch),
            fov=float(fov),
            extra=float(extra),
        )


def _parse_matrix(value: str) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"Expected 16 matrix values, got {values.size}")
    return values.reshape(4, 4)


def logic_matrices(metadata: TrackingGeometryMetadata) -> tuple[np.ndarray, np.ndarray]:
    """Mirror Veo's calibration parser exactly."""
    alignment = metadata.raw_json.get("alignment", {})
    raw_value = alignment.get("camera2logic")
    if not raw_value:
        raise ValueError("Calibration does not contain alignment.camera2logic")

    raw = _parse_matrix(raw_value)
    version = float(metadata.raw_json.get("calibration_version", 0.0))

    if version >= 6.2:
        camera2logic = raw
        logic2camera = np.linalg.inv(raw)
    else:
        logic2camera = raw
        camera2logic = np.linalg.inv(raw)

    return camera2logic, logic2camera


def rotation_x(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def rotation_y(angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.asarray(
        [
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def camera_state_at_time(
    metadata: TrackingGeometryMetadata,
    directions: LegacyCameraDirections,
    time_s: float,
    *,
    fps: float = TRACKING_FPS,
    aspect: float = TRACKING_ASPECT,
) -> TrackingCameraState | None:
    sample = directions.sample_at_time(time_s, fps=fps)
    if sample is None:
        return None

    if not (EPSILON < sample.fov < math.pi - EPSILON):
        raise ValueError(f"Invalid Tracking FOV: {sample.fov}")

    _, logic2camera = logic_matrices(metadata)
    camera2world = np.asarray(metadata.camera2world, dtype=np.float64)

    # Literal Veo "use-draw-on-pitch-matrix-buffers" path.
    view_to_world = (
        camera2world
        @ logic2camera
        @ rotation_y(sample.yaw)
        @ rotation_x(sample.pitch)
    )

    world_to_view = np.linalg.inv(view_to_world)

    camera_world = np.asarray(
        [
            camera2world[0, 3],
            camera2world[1, 3],
            camera2world[2, 3],
        ],
        dtype=np.float64,
    )

    horizontal_fov = 2.0 * math.atan(math.tan(sample.fov / 2.0) * aspect)
    viewport_width = 2.0 * math.tan(horizontal_fov / 2.0)
    viewport_height = 2.0 * math.tan(sample.fov / 2.0)

    return TrackingCameraState(
        sample=sample,
        logic2camera=logic2camera,
        view_to_world=view_to_world,
        world_to_view=world_to_view,
        camera_world=camera_world,
        viewport_width=viewport_width,
        viewport_height=viewport_height,
    )


def world_point_to_tracking_pixel(
    metadata: TrackingGeometryMetadata,
    directions: LegacyCameraDirections,
    time_s: float,
    world_x: float,
    world_y: float,
    world_z: float,
    width: int,
    height: int,
    *,
    fps: float = TRACKING_FPS,
    aspect: float = TRACKING_ASPECT,
) -> TrackingProjection:
    state = camera_state_at_time(
        metadata,
        directions,
        time_s,
        fps=fps,
        aspect=aspect,
    )

    if state is None:
        return TrackingProjection(
            float(world_x), float(world_y), float(world_z),
            None, None, None, None, False, "camera_state_unavailable",
        )

    world_point = np.asarray([world_x, world_y, world_z], dtype=np.float64)
    direction_world = world_point - state.camera_world
    direction_world_h = np.asarray(
        [direction_world[0], direction_world[1], direction_world[2], 0.0],
        dtype=np.float64,
    )

    direction_view = state.world_to_view @ direction_world_h
    z = float(direction_view[2])

    if not np.isfinite(direction_view).all() or z < 0.01:
        return TrackingProjection(
            float(world_x), float(world_y), float(world_z),
            None, None, None, None, False, "behind_tracking_camera",
        )

    u = float(direction_view[0] / z) / state.viewport_width + 0.5
    v = float(direction_view[1] / z) / state.viewport_height + 0.5

    pixel_x = u * float(width)
    pixel_y = v * float(height)
    visible = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0

    return TrackingProjection(
        world_x=float(world_x),
        world_y=float(world_y),
        world_z=float(world_z),
        pixel_x=float(pixel_x),
        pixel_y=float(pixel_y),
        u=u,
        v=v,
        visible=visible,
        status="visible" if visible else "outside_tracking_view",
    )


def field_point_to_tracking_pixel(
    metadata: TrackingGeometryMetadata,
    directions: LegacyCameraDirections,
    time_s: float,
    world_x: float,
    world_z: float,
    width: int,
    height: int,
    *,
    fps: float = TRACKING_FPS,
    aspect: float = TRACKING_ASPECT,
) -> TrackingProjection:
    return world_point_to_tracking_pixel(
        metadata,
        directions,
        time_s,
        world_x,
        0.0,
        world_z,
        width,
        height,
        fps=fps,
        aspect=aspect,
    )


def tracking_pixel_to_world_ray(
    metadata: TrackingGeometryMetadata,
    directions: LegacyCameraDirections,
    time_s: float,
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    *,
    fps: float = TRACKING_FPS,
    aspect: float = TRACKING_ASPECT,
) -> tuple[np.ndarray, np.ndarray, TrackingCameraState, float, float] | None:
    if width <= 0 or height <= 0:
        raise ValueError("Tracking image dimensions must be positive")

    state = camera_state_at_time(
        metadata,
        directions,
        time_s,
        fps=fps,
        aspect=aspect,
    )
    if state is None:
        return None

    # Veo screen coordinates are normalized directly from top-left.
    u = float(pixel_x) / float(width)
    v = float(pixel_y) / float(height)

    direction_view = np.asarray(
        [
            (u - 0.5) * state.viewport_width,
            (v - 0.5) * state.viewport_height,
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )

    direction_world_h = state.view_to_world @ direction_view
    direction_world = direction_world_h[:3]

    return (
        state.camera_world.copy(),
        direction_world,
        state,
        u,
        v,
    )


def is_inside_field(
    world_x: float,
    world_z: float,
    field_length: float,
    field_width: float,
) -> bool:
    return (
        abs(float(world_x)) <= float(field_length) / 2.0
        and abs(float(world_z)) <= float(field_width) / 2.0
    )


def tracking_ground_for_pixel(
    metadata: TrackingGeometryMetadata,
    directions: LegacyCameraDirections,
    time_s: float,
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
    *,
    fps: float = TRACKING_FPS,
    aspect: float = TRACKING_ASPECT,
) -> TrackingGroundIntersection:
    ray = tracking_pixel_to_world_ray(
        metadata,
        directions,
        time_s,
        pixel_x,
        pixel_y,
        width,
        height,
        fps=fps,
        aspect=aspect,
    )

    if ray is None:
        return TrackingGroundIntersection(
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            width=int(width),
            height=int(height),
            u=float(pixel_x) / float(width),
            v=float(pixel_y) / float(height),
            time_s=float(time_s),
            row_index=-1,
            yaw=float("nan"),
            pitch=float("nan"),
            fov=float("nan"),
            world_x=None,
            world_z=None,
            ground_distance_m=None,
            inside_field=None,
            status="camera_state_unavailable",
        )

    origin, direction, state, u, v = ray
    dy = float(direction[1])

    if not np.isfinite(direction).all() or abs(dy) <= EPSILON:
        world_x = world_z = distance = None
        inside = None
        status = "ray_parallel_to_ground"
    else:
        ray_parameter = -float(origin[1]) / dy

        if not math.isfinite(ray_parameter) or ray_parameter <= 0.0:
            world_x = world_z = distance = None
            inside = None
            status = "no_ground_intersection"
        else:
            world = origin + ray_parameter * direction

            if not np.isfinite(world).all():
                world_x = world_z = distance = None
                inside = None
                status = "invalid_ground_intersection"
            else:
                world_x = float(world[0])
                world_z = float(world[2])
                distance = float(np.linalg.norm(world - origin))
                inside = is_inside_field(
                    world_x,
                    world_z,
                    metadata.field_length,
                    metadata.field_width,
                )
                status = "inside_field" if inside else "outside_field"

    return TrackingGroundIntersection(
        pixel_x=float(pixel_x),
        pixel_y=float(pixel_y),
        width=int(width),
        height=int(height),
        u=u,
        v=v,
        time_s=float(time_s),
        row_index=state.sample.row_index,
        yaw=state.sample.yaw,
        pitch=state.sample.pitch,
        fov=state.sample.fov,
        world_x=world_x,
        world_z=world_z,
        ground_distance_m=distance,
        inside_field=inside,
        status=status,
    )
