"""Pure crop helpers for Tracking V1 appearance diagnostics.

This module has no detector, video-decoder, Veo-geometry or ReID dependency.
It only turns a selected virtual-view bbox into a safe padded image crop and
provides deterministic metadata for downstream appearance extraction.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


def _finite(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _non_negative(name: str, value: float) -> float:
    parsed = _finite(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


@dataclass(frozen=True, slots=True)
class AppearanceCropConfig:
    """Model-independent crop policy.

    Crops stay at native rendered-view resolution.  Resizing/normalization is a
    model-specific concern and intentionally belongs to the later descriptor
    extractor, not to this diagnostic stage.
    """

    padding_x_fraction: float = 0.08
    padding_y_fraction: float = 0.06
    min_crop_width_px: int = 6
    min_crop_height_px: int = 12

    def __post_init__(self) -> None:
        _non_negative("padding_x_fraction", self.padding_x_fraction)
        _non_negative("padding_y_fraction", self.padding_y_fraction)
        _positive_int("min_crop_width_px", self.min_crop_width_px)
        _positive_int("min_crop_height_px", self.min_crop_height_px)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CropResult:
    """Result of cropping one rendered virtual-view image."""

    crop: np.ndarray
    bbox_xyxy_int: tuple[int, int, int, int]
    original_bbox_xyxy: tuple[float, float, float, float]
    clipped: bool

    @property
    def width(self) -> int:
        return int(self.crop.shape[1])

    @property
    def height(self) -> int:
        return int(self.crop.shape[0])


def normalize_bbox_xyxy(
    bbox: Sequence[float],
) -> tuple[float, float, float, float]:
    if isinstance(bbox, (str, bytes)) or len(bbox) != 4:
        raise ValueError("bbox must contain exactly four numeric values")

    values = tuple(
        _finite(f"bbox[{index}]", value)
        for index, value in enumerate(bbox)
    )
    x1, y1, x2, y2 = values
    if not x2 > x1 or not y2 > y1:
        raise ValueError("bbox must have positive width and height")
    return values


def padded_bbox_xyxy(
    bbox: Sequence[float],
    *,
    image_width: int,
    image_height: int,
    config: AppearanceCropConfig | None = None,
) -> tuple[tuple[int, int, int, int], bool]:
    """Pad, round outward and clip an xyxy bbox to image bounds."""
    selected = config or AppearanceCropConfig()
    _positive_int("image_width", image_width)
    _positive_int("image_height", image_height)

    x1, y1, x2, y2 = normalize_bbox_xyxy(bbox)
    width = x2 - x1
    height = y2 - y1

    px = width * selected.padding_x_fraction
    py = height * selected.padding_y_fraction

    raw_x1 = math.floor(x1 - px)
    raw_y1 = math.floor(y1 - py)
    raw_x2 = math.ceil(x2 + px)
    raw_y2 = math.ceil(y2 + py)

    x1i = max(0, min(image_width, raw_x1))
    y1i = max(0, min(image_height, raw_y1))
    x2i = max(0, min(image_width, raw_x2))
    y2i = max(0, min(image_height, raw_y2))

    clipped = (
        x1i != raw_x1
        or y1i != raw_y1
        or x2i != raw_x2
        or y2i != raw_y2
    )

    if x2i <= x1i or y2i <= y1i:
        raise ValueError("bbox lies outside the image after clipping")

    return (x1i, y1i, x2i, y2i), clipped


def crop_virtual_person(
    image: np.ndarray,
    bbox: Sequence[float],
    *,
    config: AppearanceCropConfig | None = None,
) -> CropResult:
    """Extract one padded native-resolution player crop."""
    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must have shape HxW or HxWxC")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image must be non-empty")

    selected = config or AppearanceCropConfig()
    original = normalize_bbox_xyxy(bbox)
    bbox_int, clipped = padded_bbox_xyxy(
        original,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
        config=selected,
    )
    x1, y1, x2, y2 = bbox_int
    crop = image[y1:y2, x1:x2].copy()

    if crop.shape[1] < selected.min_crop_width_px:
        raise ValueError(
            f"crop width {crop.shape[1]} is below "
            f"{selected.min_crop_width_px}px"
        )
    if crop.shape[0] < selected.min_crop_height_px:
        raise ValueError(
            f"crop height {crop.shape[0]} is below "
            f"{selected.min_crop_height_px}px"
        )

    return CropResult(
        crop=crop,
        bbox_xyxy_int=bbox_int,
        original_bbox_xyxy=original,
        clipped=clipped,
    )


def safe_filename_token(value: str) -> str:
    """Return a deterministic conservative token for diagnostic filenames."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("filename token must be a non-empty string")

    token = "".join(
        character
        if character.isalnum() or character in ("-", "_")
        else "_"
        for character in value.strip()
    )
    token = token.strip("_")
    if not token:
        raise ValueError("filename token has no usable characters")
    return token
