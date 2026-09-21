"""Minimal reader for Veo calibration metadata used by this repository.

The values exposed here are initialization and reference data for future
calibration work.  This module deliberately does not project pixels, intersect
rays with the field, or otherwise treat Veo's optical model as ground truth.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def parse_matrix(value: str, rows: int, cols: int) -> np.ndarray:
    values = np.asarray([float(item) for item in value.split(",")], dtype=np.float64)
    if values.size != rows * cols:
        raise ValueError(f"Expected {rows * cols} matrix values, got {values.size}")
    return values.reshape(rows, cols)


def parse_vector(value: str) -> np.ndarray:
    return np.asarray([float(item) for item in value.split(",")], dtype=np.float64)


@dataclass(frozen=True)
class LensMetadata:
    name: str
    intrinsic: np.ndarray
    distortion: np.ndarray
    lens2camera: np.ndarray


@dataclass(frozen=True)
class SelectedCorner:
    name: str
    lens: str
    raw_pixel: np.ndarray


@dataclass
class VeoMetadata:
    source_path: Path
    raw_json: dict
    left: LensMetadata
    right: LensMetadata
    field_length: float
    field_width: float
    camera2world: np.ndarray
    camera2presentation: np.ndarray
    va_presentation: dict[str, float]

    def lens(self, lens_name: str) -> LensMetadata:
        if lens_name == "left":
            return self.left
        if lens_name == "right":
            return self.right
        raise ValueError(f"Unknown lens: {lens_name!r}")

    def selected_corners(
        self,
        *,
        stacked_width: int = 2048,
        stacked_height: int = 2048,
    ) -> list[SelectedCorner]:
        """Return valid Veo corner anchors in local RAW lens coordinates."""
        if stacked_height % 2:
            raise ValueError("Stacked frame height must be divisible by two")
        lens_height = stacked_height // 2
        corners = self.raw_json.get("selected_corners", {}).get("corners_stacked", {})
        observations: list[SelectedCorner] = []
        for name, corner in corners.items():
            for lens_name in ("left", "right"):
                entry = corner.get(lens_name)
                if not entry or not entry.get("valid"):
                    continue
                normalized_x, normalized_y = entry["stacked"]
                raw_x = float(normalized_x) * stacked_width
                stacked_y = float(normalized_y) * stacked_height
                raw_y = stacked_y if lens_name == "left" else stacked_y - lens_height
                observations.append(
                    SelectedCorner(
                        name=name,
                        lens=lens_name,
                        raw_pixel=np.asarray([raw_x, raw_y], dtype=np.float64),
                    )
                )
        return observations


def load_veo_metadata(path: str | Path) -> VeoMetadata:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as source_file:
        raw_json = json.load(source_file)
    alignment = raw_json["alignment"]

    def lens(name: str) -> LensMetadata:
        return LensMetadata(
            name=name,
            intrinsic=parse_matrix(alignment[f"intrinsic_{name}"], 3, 3),
            distortion=parse_vector(alignment[f"distortion_{name}"]),
            lens2camera=parse_matrix(alignment[f"lens_{name}2camera"], 4, 4),
        )

    return VeoMetadata(
        source_path=source_path,
        raw_json=raw_json,
        left=lens("left"),
        right=lens("right"),
        field_length=float(alignment["field_length"]),
        field_width=float(alignment["field_width"]),
        camera2world=parse_matrix(alignment["camera2world"], 4, 4),
        camera2presentation=parse_matrix(alignment["camera2presentation"], 4, 4),
        va_presentation={key: float(value) for key, value in alignment["va_presentation"].items()},
    )
