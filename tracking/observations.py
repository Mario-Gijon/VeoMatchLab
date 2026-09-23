"""Tracking observations built from validated detector outputs.

This module is deliberately downstream from person detection and geometry.
It does not run a detector and it does not modify the validated projection
math. It turns accepted detections.jsonl records into a stable contract for
the field tracker.

Initial scope:
- panorama_view
- ultra_panorama_view

The schema is intentionally generic enough to support other views later.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from tracking.veo_quality import (
    ObservationQuality,
    VeoObservationQualityModel,
    VeoQualityConfig,
)


@dataclass(frozen=True)
class Observation:
    """One accepted person observation for one video frame."""

    frame: int
    time_s: float
    detection_index: int

    view: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    foot_u: float
    foot_v: float

    world_x: float
    world_z: float

    bbox_width: float
    bbox_height: float

    inside_strict_field: bool | None
    field_status: str
    owner_view: str | None

    reprojection_error_m: float
    quality: ObservationQuality

    @property
    def world_position(self) -> np.ndarray:
        return np.asarray([self.world_x, self.world_z], dtype=np.float64)

    @property
    def measurement_covariance(self) -> np.ndarray:
        return self.quality.measurement_covariance.copy()

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["xyxy"] = list(self.xyxy)
        result["quality"] = self.quality.to_dict()
        return result


@dataclass(frozen=True)
class ObservationFrame:
    """All accepted observations for one processed video frame."""

    frame: int
    time_s: float
    observations: tuple[Observation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame": self.frame,
            "time_s": self.time_s,
            "observations": [item.to_dict() for item in self.observations],
        }


@dataclass
class ObservationRun:
    """Loaded detector run plus the projector used to interpret it."""

    run_dir: Path
    config: dict[str, Any]
    frames: list[ObservationFrame]
    projector: Any

    def iter_observations(self) -> Iterator[Observation]:
        for frame in self.frames:
            yield from frame.observations

    @property
    def observation_count(self) -> int:
        return sum(len(frame.observations) for frame in self.frames)


def bbox_footpoint(
    xyxy: Iterable[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float]:
    """Match the validated panorama projector's bbox footpoint exactly."""

    x1, y1, x2, y2 = [float(value) for value in xyxy]
    values = np.asarray([x1, y1, x2, y2], dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("bbox coordinates must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2>x1 and y2>y1")
    if width <= 0 or height <= 0:
        raise ValueError("projection dimensions must be positive")

    foot_u = min(max(0.5 * (x1 + x2), 0.0), float(width) - 1.0)
    foot_v = min(max(y2 - 0.5, 0.0), float(height) - 1.0)
    return foot_u, foot_v


def _build_observation(
    *,
    frame: int,
    time_s: float,
    detection_index: int,
    detection: dict[str, Any],
    projector: Any,
    quality_model: VeoObservationQualityModel,
) -> Observation:
    required = ("confidence", "xyxy", "world_x", "world_z")
    missing = [key for key in required if key not in detection]
    if missing:
        raise ValueError(
            f"detection at frame {frame} is missing required fields: {missing}"
        )

    confidence = float(detection["confidence"])
    if not math.isfinite(confidence):
        raise ValueError(f"non-finite confidence at frame {frame}")

    xyxy_values = tuple(float(value) for value in detection["xyxy"])
    if len(xyxy_values) != 4:
        raise ValueError(f"xyxy must contain 4 values at frame {frame}")

    x1, y1, x2, y2 = xyxy_values
    bbox_width = x2 - x1
    bbox_height = y2 - y1

    foot_u, foot_v = bbox_footpoint(
        xyxy_values,
        width=int(projector.width),
        height=int(projector.height),
    )

    stored_x = float(detection["world_x"])
    stored_z = float(detection["world_z"])
    if not np.isfinite([stored_x, stored_z]).all():
        raise ValueError(f"non-finite stored world position at frame {frame}")

    reprojection = projector.project_pixel(foot_u, foot_v)
    if reprojection.world_x is None or reprojection.world_z is None:
        raise ValueError(
            f"stored accepted detection has no reprojection ground intersection "
            f"at frame {frame}, detection {detection_index}"
        )

    reprojection_error_m = math.hypot(
        stored_x - float(reprojection.world_x),
        stored_z - float(reprojection.world_z),
    )

    quality = quality_model.evaluate(
        pixel_x=foot_u,
        pixel_y=foot_v,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        confidence=confidence,
    )

    return Observation(
        frame=int(frame),
        time_s=float(time_s),
        detection_index=int(detection_index),
        view=str(detection.get("view", "panorama")),
        confidence=confidence,
        xyxy=xyxy_values,
        foot_u=foot_u,
        foot_v=foot_v,
        world_x=stored_x,
        world_z=stored_z,
        bbox_width=float(bbox_width),
        bbox_height=float(bbox_height),
        inside_strict_field=detection.get("inside_strict_field"),
        field_status=str(detection.get("field_status", "unknown")),
        owner_view=detection.get("owner_view"),
        reprojection_error_m=float(reprojection_error_m),
        quality=quality,
    )


def load_detection_frames(
    detections_path: str | Path,
    *,
    projector: Any,
    quality_model: VeoObservationQualityModel | None = None,
) -> list[ObservationFrame]:
    """Load the current detections.jsonl schema into tracker observations.

    Every line in the detector output represents one processed frame, including
    frames with zero accepted detections. Empty frames are therefore preserved.
    """

    source = Path(detections_path)
    if not source.exists():
        raise FileNotFoundError(source)

    if quality_model is None:
        quality_model = VeoObservationQualityModel(projector)

    frames: list[ObservationFrame] = []

    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            record = json.loads(line)
            if "frame" not in record or "time_s" not in record:
                raise ValueError(
                    f"{source}:{line_number}: expected frame and time_s"
                )

            raw_detections = record.get("detections", [])
            if not isinstance(raw_detections, list):
                raise ValueError(
                    f"{source}:{line_number}: detections must be a list"
                )

            frame_number = int(record["frame"])
            time_s = float(record["time_s"])

            observations = tuple(
                _build_observation(
                    frame=frame_number,
                    time_s=time_s,
                    detection_index=index,
                    detection=detection,
                    projector=projector,
                    quality_model=quality_model,
                )
                for index, detection in enumerate(raw_detections)
            )

            frames.append(
                ObservationFrame(
                    frame=frame_number,
                    time_s=time_s,
                    observations=observations,
                )
            )

    return frames


def projector_from_run_config(config: dict[str, Any]) -> Any:
    """Reconstruct the panorama projector used by an existing detector run."""

    view = str(config.get("view", ""))
    calibration_raw = config.get("calibration")
    if not calibration_raw:
        raise ValueError("config.json does not contain calibration")

    calibration = Path(calibration_raw)
    panorama_size = config.get("panorama_size")
    if not isinstance(panorama_size, list) or len(panorama_size) != 2:
        raise ValueError(
            "tracking observation V1 requires config.panorama_size=[width,height]"
        )

    width = int(panorama_size[0])
    height = int(panorama_size[1])
    far_margin = float(config.get("far_touchline_margin_m", 0.50))

    if view == "panorama_view":
        from person_detect.views.panorama_view.projection import PanoramaProjector

        return PanoramaProjector(
            calibration,
            width=width,
            height=height,
            far_touchline_margin_m=far_margin,
        )

    if view == "ultra_panorama_view":
        from person_detect.views.ultra_panorama_view.projection import PanoramaProjector

        return PanoramaProjector(
            calibration,
            width=width,
            height=height,
            far_touchline_margin_m=far_margin,
        )

    raise ValueError(
        "tracking observation V1 currently supports panorama_view and "
        f"ultra_panorama_view, got {view!r}"
    )


def load_run_observations(
    run_dir: str | Path,
    *,
    quality_config: VeoQualityConfig | None = None,
) -> ObservationRun:
    """Load an existing detector run without running inference again."""

    run_path = Path(run_dir)
    config_path = run_path / "config.json"
    detections_path = run_path / "detections.jsonl"

    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not detections_path.exists():
        raise FileNotFoundError(detections_path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    projector = projector_from_run_config(config)
    quality_model = VeoObservationQualityModel(
        projector,
        config=quality_config,
    )
    frames = load_detection_frames(
        detections_path,
        projector=projector,
        quality_model=quality_model,
    )

    return ObservationRun(
        run_dir=run_path,
        config=config,
        frames=frames,
        projector=projector,
    )


__all__ = [
    "Observation",
    "ObservationFrame",
    "ObservationRun",
    "bbox_footpoint",
    "load_detection_frames",
    "load_run_observations",
    "projector_from_run_config",
]
