"""Ultra-panorama debug overlays for virtual-camera detections.

This module is presentation/debug only. Detection and association still happen
in the six TIGHT virtual cameras.

Important:
The overlay is projected from the REPRESENTATIVE DETECTION PIXELS themselves,
not from estimated ground X/Z. That is deliberate: tiny/far bbox footpoints can
produce metre-scale X/Z error, while the visual ray of the detection is still
known precisely. For visual detector auditing, pixel-ray -> panorama is the
correct mapping.

This V3 rendering uses only a compact bottom-edge marker plus the foot point
(no full rectangle).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from geometry.reproduce_veo_panorama import (
    PanoramaRenderMaps,
    build_panorama_maps,
    render as render_panorama,
    render_panorama_with_maps,
)
from person_detect.views.interactive_view.renderer import rot_x, rot_y


ULTRA_PANORAMA_HEIGHT = 1024
ULTRA_PANORAMA_MAX_WIDTH = 4096
PANORAMA_FALLBACK_SIZE = (1920, 1080)


@dataclass(frozen=True)
class PanoramaDetection:
    """One final unique person represented by one virtual-camera bbox."""

    person_id: int
    view_name: str
    polygon: tuple[
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
        tuple[float, float],
    ]
    foot: tuple[float, float]


def ultra_panorama_size(metadata) -> tuple[int, int]:
    """Match the frontend ultra-panorama sizing rule."""
    try:
        va = metadata.va_presentation
        left = float(va["left"])
        right = float(va["right"])
        top = float(va["top"])
        bottom = float(va["bottom"])
    except Exception:
        return PANORAMA_FALLBACK_SIZE

    horizontal_span = right - left
    vertical_span = top - bottom

    if (
        not np.isfinite(horizontal_span)
        or not np.isfinite(vertical_span)
        or horizontal_span <= 0.0
        or vertical_span <= 0.0
    ):
        return PANORAMA_FALLBACK_SIZE

    width = int(
        round(
            ULTRA_PANORAMA_HEIGHT
            * horizontal_span
            / vertical_span
        )
    )
    width = max(
        1,
        min(
            ULTRA_PANORAMA_MAX_WIDTH,
            width,
        ),
    )

    return width, ULTRA_PANORAMA_HEIGHT


def render_ultra_panorama(
    frame: np.ndarray,
    metadata,
    maps: PanoramaRenderMaps | None = None,
) -> np.ndarray:
    """Render the panorama with the repository's validated CPU renderer."""
    width, height = ultra_panorama_size(
        metadata
    )
    result = (render_panorama_with_maps(frame, maps) if maps is not None else render_panorama(frame, metadata, width, height))

    return result["panorama"]


def build_ultra_panorama_maps(frame: np.ndarray, metadata) -> PanoramaRenderMaps:
    """Build the static Ultra panorama remap tables once for a clip."""
    width, height = ultra_panorama_size(metadata)
    raw_height, raw_width = frame.shape[:2]
    return build_panorama_maps(metadata, raw_width, raw_height, width, height)


def _presentation_direction_from_virtual_pixel(
    spec,
    pixel_x: float,
    pixel_y: float,
    virtual_width: int,
    virtual_height: int,
) -> np.ndarray | None:
    """Invert coverage.world_points_to_virtual_pixels for a single image ray.

    coverage.py uses:
        local = presentation_direction @ rotation

    so:
        presentation_direction = local @ rotation.T
    """
    if (
        virtual_width <= 0
        or virtual_height <= 0
        or not np.isfinite(
            [
                pixel_x,
                pixel_y,
                spec.longitude,
                spec.latitude,
                spec.vfov,
            ]
        ).all()
    ):
        return None

    u = float(pixel_x) / float(
        virtual_width
    )
    v = float(pixel_y) / float(
        virtual_height
    )

    tan_v = np.tan(
        float(spec.vfov) / 2.0
    )
    aspect = (
        float(virtual_width)
        / float(virtual_height)
    )

    local = np.array(
        [
            (2.0 * u - 1.0)
            * tan_v
            * aspect,
            (2.0 * v - 1.0)
            * tan_v,
            1.0,
        ],
        dtype=np.float64,
    )

    rotation = (
        rot_y(float(spec.longitude))
        @ rot_x(float(spec.latitude))
    )[:3, :3]

    direction = (
        local
        @ rotation.T
    )

    norm = float(
        np.linalg.norm(direction)
    )

    if (
        not np.isfinite(direction).all()
        or not np.isfinite(norm)
        or norm <= 1e-12
    ):
        return None

    return direction / norm


def virtual_pixel_to_panorama_pixel(
    metadata,
    spec,
    pixel_x: float,
    pixel_y: float,
    *,
    virtual_width: int,
    virtual_height: int,
    panorama_width: int,
    panorama_height: int,
) -> tuple[float, float] | None:
    """Map one virtual-camera pixel ray directly into Veo panorama pixels.

    No field X/Z is used.
    """
    direction = (
        _presentation_direction_from_virtual_pixel(
            spec,
            pixel_x,
            pixel_y,
            virtual_width,
            virtual_height,
        )
    )

    if direction is None:
        return None

    dx, dy, dz = [
        float(value)
        for value in direction
    ]

    longitude = float(
        np.arctan2(
            dx,
            dz,
        )
    )

    latitude = float(
        np.arctan2(
            -dy,
            np.hypot(
                dx,
                dz,
            ),
        )
    )

    va = metadata.va_presentation

    left = float(va["left"])
    right = float(va["right"])
    bottom = float(va["bottom"])
    top = float(va["top"])

    horizontal_span = right - left
    vertical_span = top - bottom

    if (
        horizontal_span <= 0.0
        or vertical_span <= 0.0
    ):
        return None

    u = (
        longitude - left
    ) / horizontal_span

    v = (
        latitude - bottom
    ) / vertical_span

    # Allow only a tiny numerical tolerance around the image boundary.
    eps = 1e-6

    if (
        u < -eps
        or u > 1.0 + eps
        or v < -eps
        or v > 1.0 + eps
    ):
        return None

    u = min(
        1.0,
        max(
            0.0,
            u,
        ),
    )
    v = min(
        1.0,
        max(
            0.0,
            v,
        ),
    )

    panorama_x = (
        u * panorama_width
    )

    # Panorama renderer uses image y=0 at top while panorama v grows upward.
    panorama_y = (
        (1.0 - v)
        * panorama_height
    )

    return (
        float(panorama_x),
        float(panorama_y),
    )


def virtual_bbox_to_panorama_detection(
    metadata,
    spec,
    *,
    person_id: int,
    view_name: str,
    xyxy: tuple[
        float,
        float,
        float,
        float,
    ],
    virtual_width: int,
    virtual_height: int,
    panorama_width: int,
    panorama_height: int,
) -> PanoramaDetection | None:
    """Project the representative detection bbox itself into the panorama."""
    x1, y1, x2, y2 = [
        float(value)
        for value in xyxy
    ]

    if (
        not np.isfinite(
            [
                x1,
                y1,
                x2,
                y2,
            ]
        ).all()
        or x2 <= x1
        or y2 <= y1
    ):
        return None

    virtual_points = [
        (x1, y1),
        (x2, y1),
        (x2, y2),
        (x1, y2),
    ]

    panorama_points: list[
        tuple[float, float]
    ] = []

    for px, py in virtual_points:
        mapped = (
            virtual_pixel_to_panorama_pixel(
                metadata,
                spec,
                px,
                py,
                virtual_width=virtual_width,
                virtual_height=virtual_height,
                panorama_width=panorama_width,
                panorama_height=panorama_height,
            )
        )

        if mapped is None:
            return None

        panorama_points.append(
            mapped
        )

    foot = (
        virtual_pixel_to_panorama_pixel(
            metadata,
            spec,
            0.5 * (x1 + x2),
            y2,
            virtual_width=virtual_width,
            virtual_height=virtual_height,
            panorama_width=panorama_width,
            panorama_height=panorama_height,
        )
    )

    if foot is None:
        return None

    return PanoramaDetection(
        person_id=int(person_id),
        view_name=str(view_name),
        polygon=(
            panorama_points[0],
            panorama_points[1],
            panorama_points[2],
            panorama_points[3],
        ),
        foot=foot,
    )


def draw_panorama_detections(
    panorama_bgr: np.ndarray,
    detections: list[
        PanoramaDetection
    ],
    *,
    numbered: bool,
) -> np.ndarray:
    """Draw a minimal overlay: bottom edge + foot point.

    This is intentionally lighter than a full rectangle so nearby players remain
    readable in crowded zones.
    """
    output = panorama_bgr.copy()

    for detection in detections:
        p1, p2, p3, p4 = detection.polygon

        # Bottom edge of the projected bbox in panorama space.
        base_a = (
            int(round(p4[0])),
            int(round(p4[1])),
        )
        base_b = (
            int(round(p3[0])),
            int(round(p3[1])),
        )

        # Keep the projected base visually compact when the warped bottom edge
        # becomes too long in the panorama.
        foot_x = int(round(detection.foot[0]))
        foot_y = int(round(detection.foot[1]))
        half_len = min(
            max(6, abs(base_b[0] - base_a[0]) // 2),
            18,
        )
        compact_a = (foot_x - half_len, foot_y)
        compact_b = (foot_x + half_len, foot_y)

        cv2.line(
            output,
            compact_a,
            compact_b,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.line(
            output,
            compact_a,
            compact_b,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.circle(
            output,
            (foot_x, foot_y),
            5,
            (0, 0, 0),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            output,
            (foot_x, foot_y),
            3,
            (255, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        if not numbered:
            continue

        label = str(detection.person_id)
        label_x = foot_x + 7
        label_y = foot_y + 18

        cv2.putText(
            output,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return output
