"""Tracking-view geometry for Veo Standard/Follow video.

This module reproduces the camera-direction path used by Veo's Standard player:

    tracking pixel + time
      -> detections_constrained.det (yaw, pitch, vertical fov)
      -> legacy virtual camera
      -> legacy2logic
      -> camera2world * camera2logic * virtualcam
      -> world ray
      -> Y=0 field intersection

It intentionally lives beside, rather than modifying, the already validated
panorama/RAW geometry modules.  The public functions are suitable for reuse by
the FastAPI backend later: they are pure geometry apart from loading the .det
file and have no browser/UI dependency.

The implementation mirrors the currently observed Veo WebGL player behaviour:
- camera-direction lookup uses frame = 29.97 * video.currentTime;
- .det samples are linearly interpolated;
- legacy virtualcam = Ry(yaw) @ Rx(pitch);
- calibration 6.2 inverts alignment.camera2logic before use;
- legacy2logic = inverse(camera2world @ camera2logic), with translation removed;
- Standard shader uses a 16:9 view model and vertical FOV;
- browser pixels use a top-left origin while the WebGL UV has a bottom-left origin.

The inverse projection (world/field -> Tracking pixel) is included because it is
needed for future cross-view entity highlighting.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence

import numpy as np


# Veo's player code uses this literal value, rather than 30000/1001.
TRACKING_FPS = 29.97
TRACKING_ASPECT = 16.0 / 9.0
EPSILON = 1e-12


class TrackingGeometryMetadata(Protocol):
    """Minimal metadata surface required by this module."""

    raw_json: dict[str, Any]
    camera2world: np.ndarray
    field_length: float
    field_width: float


@dataclass(frozen=True)
class CameraDirectionSample:
    """Interpolated legacy camera directions for one video time/frame."""

    time_s: float
    frame: float
    yaw: float
    pitch: float
    fov: float
    extra: float

    def to_dict(self) -> dict[str, float]:
        return {
            "time_s": self.time_s,
            "frame": self.frame,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "fov": self.fov,
            "extra": self.extra,
        }


@dataclass(frozen=True)
class TrackingCameraState:
    """Full camera state used by the Standard/Follow shader at one instant."""

    sample: CameraDirectionSample
    virtualcam_legacy: np.ndarray
    virtualcam_logic: np.ndarray
    yaw_logic: float
    pitch_logic: float

    @property
    def time_s(self) -> float:
        return self.sample.time_s

    @property
    def frame(self) -> float:
        return self.sample.frame

    @property
    def yaw(self) -> float:
        return self.sample.yaw

    @property
    def pitch(self) -> float:
        return self.sample.pitch

    @property
    def fov(self) -> float:
        return self.sample.fov

    def to_dict(self, *, include_matrices: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            **self.sample.to_dict(),
            "yaw_logic": self.yaw_logic,
            "pitch_logic": self.pitch_logic,
        }
        if include_matrices:
            result["virtualcam_legacy"] = self.virtualcam_legacy.tolist()
            result["virtualcam_logic"] = self.virtualcam_logic.tolist()
        return result


@dataclass(frozen=True)
class TrackingGroundIntersection:
    """Result of mapping one Tracking-view pixel onto the Y=0 pitch plane."""

    pixel_x: float
    pixel_y: float
    width: int
    height: int
    u: float
    v: float
    time_s: float
    frame: float
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
            "frame": self.frame,
            "yaw": self.yaw,
            "pitch": self.pitch,
            "fov": self.fov,
            "world_x": self.world_x,
            "world_z": self.world_z,
            "ground_distance_m": self.ground_distance_m,
            "inside_field": self.inside_field,
            "status": self.status,
        }


@dataclass(frozen=True)
class TrackingProjection:
    """Projection of a world point into the Tracking view."""

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
class _DirectionSequence:
    times: np.ndarray
    values: np.ndarray
    origin: str = "machine_detection"

    def __post_init__(self) -> None:
        if self.times.ndim != 1:
            raise ValueError("Direction sequence times must be one-dimensional")
        if self.values.ndim != 2 or self.values.shape[1] != 4:
            raise ValueError("Direction sequence values must have shape (N, 4)")
        if len(self.times) != len(self.values):
            raise ValueError("Direction sequence times/values length mismatch")
        if len(self.times) < 2:
            raise ValueError("Direction sequence needs at least two samples")
        if not np.all(np.diff(self.times) > 0):
            raise ValueError("Direction sequence times must be strictly increasing")

    def evaluate(self, frame: float) -> np.ndarray | None:
        """Mirror Veo Sequence.evaluateBS() linear interpolation."""
        first = float(self.times[0])
        last = float(self.times[-1])
        if not math.isfinite(frame) or frame < first or frame > last:
            return None

        if frame == last:
            return self.values[-1].astype(np.float64, copy=True)

        index = int(np.searchsorted(self.times, frame, side="right") - 1)
        index = max(0, min(index, len(self.times) - 2))

        t0 = float(self.times[index])
        t1 = float(self.times[index + 1])
        v0 = self.values[index]
        v1 = self.values[index + 1]

        alpha = (frame - t0) / (t1 - t0)
        return (1.0 - alpha) * v0 + alpha * v1


class LegacyCameraDirections:
    """Parsed Veo camera-direction sequence(s) from detections_constrained.det."""

    def __init__(self, sequences: Sequence[_DirectionSequence], source_path: Path | None = None):
        if not sequences:
            raise ValueError("At least one camera-direction sequence is required")
        self._sequences = tuple(sequences)
        self.source_path = source_path

    @classmethod
    def load(cls, path: str | Path) -> "LegacyCameraDirections":
        source_path = Path(path)
        text = source_path.read_text(encoding="utf-8", errors="replace")
        return cls.from_text(text, source_path=source_path)

    @classmethod
    def from_text(
        cls,
        text: str,
        *,
        source_path: Path | None = None,
    ) -> "LegacyCameraDirections":
        stripped = text.lstrip()
        if not stripped:
            raise ValueError("Camera-direction file is empty")

        if stripped.startswith("{"):
            raw = json.loads(text)
            sequences: list[_DirectionSequence] = []
            for sequence in raw.get("sequences", []):
                times = np.asarray(sequence["times"], dtype=np.float64)
                values = np.asarray(sequence["values"], dtype=np.float64)
                sequences.append(
                    _DirectionSequence(
                        times=times,
                        values=values,
                        origin=str(sequence.get("origin", "unknown")),
                    )
                )
            return cls(sequences, source_path=source_path)

        lines = text.splitlines()
        if not lines:
            raise ValueError("Camera-direction file contains no lines")

        first_parts = [part.strip() for part in lines[0].split(",")]
        first_has_offset = len(first_parts) == 1
        offset = float(first_parts[0]) if first_has_offset else 0.0

        rows: list[list[float]] = []
        times: list[float] = []

        # Veo's legacy parser uses the original line index as the sample time.
        # Therefore, when a one-value offset line exists, the first four-value
        # row has time offset + 1.  Preserve that convention exactly.
        for line_index, line in enumerate(lines):
            if not line.strip():
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 4:
                if line_index == 0 and first_has_offset:
                    continue
                raise ValueError(
                    f"Invalid camera-direction row {line_index + 1}: expected 4 values"
                )
            values = [float(part) for part in parts]
            rows.append(values)
            times.append(float(line_index) + offset)

        if len(rows) < 2:
            raise ValueError("Camera-direction file needs at least two valid rows")

        sequence = _DirectionSequence(
            times=np.asarray(times, dtype=np.float64),
            values=np.asarray(rows, dtype=np.float64),
            origin="machine_detection",
        )
        return cls([sequence], source_path=source_path)

    @property
    def sample_count(self) -> int:
        return sum(len(sequence.times) for sequence in self._sequences)

    @property
    def earliest_frame(self) -> float:
        return min(float(sequence.times[0]) for sequence in self._sequences)

    @property
    def latest_frame(self) -> float:
        return max(float(sequence.times[-1]) for sequence in self._sequences)

    def evaluate_frame(self, frame: float) -> np.ndarray | None:
        # Veo SequenceStack searches from the last inserted sequence backwards.
        for sequence in reversed(self._sequences):
            value = sequence.evaluate(frame)
            if value is not None:
                return value
        return None

    def evaluate_time(self, time_s: float, *, fps: float = TRACKING_FPS) -> CameraDirectionSample | None:
        if not math.isfinite(time_s) or time_s < 0.0:
            return None
        frame = float(fps) * float(time_s)
        values = self.evaluate_frame(frame)
        if values is None:
            return None
        return CameraDirectionSample(
            time_s=float(time_s),
            frame=frame,
            yaw=float(values[0]),
            pitch=float(values[1]),
            fov=float(values[2]),
            extra=float(values[3]),
        )


def _parse_matrix(value: str, rows: int = 4, cols: int = 4) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.size != rows * cols:
        raise ValueError(f"Expected {rows * cols} matrix values, got {values.size}")
    return values.reshape(rows, cols)


def camera2logic_matrix(metadata: TrackingGeometryMetadata) -> np.ndarray:
    """Return the matrix that Veo actually uploads as camera2logic.

    Observed player calibration parser:
      - calibration 6.2: inverse(alignment.camera2logic)
      - calibration 6.1 / 6.0: alignment.camera2logic directly

    Older formats are deliberately rejected here because the repository's
    current metadata reader is 6.x-oriented and silently guessing would make
    metric coordinates unsafe.
    """
    raw = metadata.raw_json
    version = str(raw.get("calibration_version", ""))
    alignment = raw.get("alignment", {})
    value = alignment.get("camera2logic")
    if not value:
        raise ValueError("Calibration does not contain alignment.camera2logic")

    parsed = _parse_matrix(value)
    if version == "6.2":
        return np.linalg.inv(parsed)
    if version in {"6.1", "6.0"}:
        return parsed
    raise NotImplementedError(
        f"Tracking geometry is verified only for Veo calibration 6.0/6.1/6.2; got {version!r}"
    )


def legacy2logic_matrix(metadata: TrackingGeometryMetadata) -> np.ndarray:
    """Mirror Veo's legacy2logic = inverse(camera2world * camera2logic)."""
    camera2logic = camera2logic_matrix(metadata)
    matrix = np.linalg.inv(np.asarray(metadata.camera2world, dtype=np.float64) @ camera2logic)
    matrix = matrix.copy()
    matrix[0, 3] = 0.0
    matrix[1, 3] = 0.0
    matrix[2, 3] = 0.0
    return matrix


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


def legacy_virtual_camera(pitch: float, yaw: float) -> np.ndarray:
    """Mirror Veo virtualcam(pitch, yaw) = Ry(yaw) @ Rx(pitch)."""
    return rotation_y(float(yaw)) @ rotation_x(float(pitch))


def camera_state_at_time(
    metadata: TrackingGeometryMetadata,
    directions: LegacyCameraDirections,
    time_s: float,
    *,
    fps: float = TRACKING_FPS,
) -> TrackingCameraState | None:
    sample = directions.evaluate_time(time_s, fps=fps)
    if sample is None:
        return None
    if not (EPSILON < sample.fov < math.pi - EPSILON):
        raise ValueError(f"Invalid vertical FOV in camera directions: {sample.fov}")

    virtualcam_legacy = legacy_virtual_camera(sample.pitch, sample.yaw)
    virtualcam_logic = legacy2logic_matrix(metadata) @ virtualcam_legacy

    # Veo derives these values after converting the legacy camera to logic.
    forward = virtualcam_logic @ np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float64)
    yaw_logic = math.atan2(float(forward[0]), float(forward[2]))
    pitch_logic = math.atan2(float(-forward[1]), float(forward[2]))

    return TrackingCameraState(
        sample=sample,
        virtualcam_legacy=virtualcam_legacy,
        virtualcam_logic=virtualcam_logic,
        yaw_logic=yaw_logic,
        pitch_logic=pitch_logic,
    )


def tracking_pixel_center_uv(
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Convert browser/top-left pixel coordinates to Veo WebGL UV."""
    if width <= 0 or height <= 0:
        raise ValueError("Tracking dimensions must be positive")
    if not (-0.5 <= pixel_x <= width - 0.5 and -0.5 <= pixel_y <= height - 0.5):
        raise ValueError("Tracking pixel is outside the image")
    u = (float(pixel_x) + 0.5) / float(width)
    v = 1.0 - (float(pixel_y) + 0.5) / float(height)
    return u, v


def tracking_view_direction(
    u: float,
    v: float,
    fov: float,
    *,
    aspect: float = TRACKING_ASPECT,
) -> np.ndarray:
    """Literal CPU equivalent of Standard shader cal_viewDir()."""
    if aspect <= 0.0:
        raise ValueError("Aspect ratio must be positive")
    if not (EPSILON < fov < math.pi - EPSILON):
        raise ValueError("Vertical FOV must be between 0 and pi radians")

    return np.asarray(
        [
            (float(u) - 0.5) * float(aspect),
            -(float(v) - 0.5),
            0.5 / math.tan(float(fov) / 2.0),
            0.0,
        ],
        dtype=np.float64,
    )


def tracking_world_ray_from_uv(
    metadata: TrackingGeometryMetadata,
    state: TrackingCameraState,
    u: float,
    v: float,
    *,
    aspect: float = TRACKING_ASPECT,
) -> tuple[np.ndarray, np.ndarray]:
    """Return camera world origin and non-normalized world ray direction."""
    camera2world = np.asarray(metadata.camera2world, dtype=np.float64)
    camera2logic = camera2logic_matrix(metadata)
    view_direction = tracking_view_direction(u, v, state.fov, aspect=aspect)

    direction_world_h = (
        camera2world
        @ camera2logic
        @ state.virtualcam_logic
        @ view_direction
    )
    origin_world_h = camera2world @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

    direction_world = direction_world_h[:3]
    origin_world = origin_world_h[:3]
    return origin_world, direction_world


def tracking_world_ray_for_pixel(
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
    state = camera_state_at_time(metadata, directions, time_s, fps=fps)
    if state is None:
        return None
    u, v = tracking_pixel_center_uv(pixel_x, pixel_y, width, height)
    origin, direction = tracking_world_ray_from_uv(metadata, state, u, v, aspect=aspect)
    return origin, direction, state, u, v


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
    """Map a Tracking pixel to canonical field X/Z metres at a video time."""
    ray = tracking_world_ray_for_pixel(
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
        frame = float(time_s) * float(fps)
        u, v = tracking_pixel_center_uv(pixel_x, pixel_y, width, height)
        return TrackingGroundIntersection(
            pixel_x=float(pixel_x),
            pixel_y=float(pixel_y),
            width=int(width),
            height=int(height),
            u=u,
            v=v,
            time_s=float(time_s),
            frame=frame,
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
        status = "ray_parallel_to_ground"
        world_x = world_z = distance = None
        inside = None
    else:
        ray_parameter = -float(origin[1]) / dy
        if not math.isfinite(ray_parameter) or ray_parameter <= 0.0:
            status = "no_ground_intersection"
            world_x = world_z = distance = None
            inside = None
        else:
            world = origin + ray_parameter * direction
            if not np.isfinite(world).all():
                status = "invalid_ground_intersection"
                world_x = world_z = distance = None
                inside = None
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
        frame=state.frame,
        yaw=state.yaw,
        pitch=state.pitch,
        fov=state.fov,
        world_x=world_x,
        world_z=world_z,
        ground_distance_m=distance,
        inside_field=inside,
        status=status,
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
    """Project any world point into Tracking using the same dynamic camera.

    This inverse operation is what will later let the app highlight a canonical
    player/ball entity in Tracking after it was selected in another view.
    """
    state = camera_state_at_time(metadata, directions, time_s, fps=fps)
    if state is None:
        return TrackingProjection(
            world_x=float(world_x),
            world_y=float(world_y),
            world_z=float(world_z),
            pixel_x=None,
            pixel_y=None,
            u=None,
            v=None,
            visible=False,
            status="camera_state_unavailable",
        )

    camera2world = np.asarray(metadata.camera2world, dtype=np.float64)
    camera2logic = camera2logic_matrix(metadata)
    camera_origin = (camera2world @ np.asarray([0.0, 0.0, 0.0, 1.0]))[:3]
    target = np.asarray([world_x, world_y, world_z], dtype=np.float64)
    direction_world_h = np.asarray(
        [
            target[0] - camera_origin[0],
            target[1] - camera_origin[1],
            target[2] - camera_origin[2],
            0.0,
        ],
        dtype=np.float64,
    )

    view_to_world = camera2world @ camera2logic @ state.virtualcam_logic
    world_to_view = np.linalg.inv(view_to_world)
    direction_view = world_to_view @ direction_world_h

    if not np.isfinite(direction_view).all() or float(direction_view[2]) <= EPSILON:
        return TrackingProjection(
            world_x=float(world_x),
            world_y=float(world_y),
            world_z=float(world_z),
            pixel_x=None,
            pixel_y=None,
            u=None,
            v=None,
            visible=False,
            status="behind_tracking_camera",
        )

    z0 = 0.5 / math.tan(state.fov / 2.0)
    ratio_x = float(direction_view[0] / direction_view[2])
    ratio_y = float(direction_view[1] / direction_view[2])

    u = 0.5 + ratio_x * z0 / float(aspect)
    v = 0.5 - ratio_y * z0

    pixel_x = u * float(width) - 0.5
    pixel_y = (1.0 - v) * float(height) - 0.5
    visible = 0.0 <= u <= 1.0 and 0.0 <= v <= 1.0

    return TrackingProjection(
        world_x=float(world_x),
        world_y=float(world_y),
        world_z=float(world_z),
        pixel_x=float(pixel_x),
        pixel_y=float(pixel_y),
        u=float(u),
        v=float(v),
        visible=bool(visible),
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
    """Convenience wrapper for a ground-plane (Y=0) field point."""
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
