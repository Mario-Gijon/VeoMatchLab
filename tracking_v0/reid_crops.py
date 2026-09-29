"""Rectified, player-centred visual crops for future ReID.

Sprint A deliberately contains no identity model, team classifier, goalkeeper
classifier or jersey-number recognition.  It only turns one *observed* tracker
measurement into a stable visual crop and a transparent quality score.

The source detection is already expressed in one of VeoMatchLab's rectilinear
virtual cameras.  We use that detection only to recover its presentation-space
ray and angular extent; then we render a new, narrow perspective camera
directly from the calibrated 2048x2048 dual-lens source frame.

No raw fisheye crop is enlarged.  The new crop is generated through the same
validated Veo reprojection used by the six virtual detection cameras.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ReIDCropConfig:
    """Geometry and quality policy for rectified ReID crops."""

    # Small dynamic view: enough pixels for appearance while keeping remap
    # generation affordable for an offline audit.
    render_width: int = 160
    render_height: int = 320

    # Final tensor-friendly aspect.  No ReID network is used in Sprint A.
    output_width: int = 128
    output_height: int = 256

    # Try to make the detector bbox occupy this much of the narrow view.
    target_player_fill: float = 0.72
    min_vfov_deg: float = 3.5
    max_vfov_deg: float = 32.0

    # Extra body context after the original bbox rays are mapped into the
    # player-centred camera.
    horizontal_padding_fraction: float = 0.10
    top_padding_fraction: float = 0.08
    bottom_padding_fraction: float = 0.06

    # Candidate proposal quality (cheap; before rendering).
    proposal_reference_height_fraction: float = 0.18
    proposal_size_weight: float = 0.45
    proposal_confidence_weight: float = 0.35
    proposal_edge_weight: float = 0.20

    # Final crop quality.
    final_size_reference_px: float = 180.0
    blur_scale: float = 100.0
    size_weight: float = 0.30
    clip_weight: float = 0.25
    blur_weight: float = 0.20
    confidence_weight: float = 0.20
    valid_weight: float = 0.05

    # Temporal diversity for a track appearance bank.
    max_crops_per_track: int = 10
    min_time_gap_seconds: float = 0.75

    def __post_init__(self) -> None:
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("render dimensions must be positive")
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("output dimensions must be positive")
        if not 0.0 < self.target_player_fill < 1.0:
            raise ValueError("target_player_fill must be in (0, 1)")
        if not 0.0 < self.min_vfov_deg < self.max_vfov_deg < 170.0:
            raise ValueError("invalid VFOV limits")
        if self.proposal_reference_height_fraction <= 0.0:
            raise ValueError("proposal_reference_height_fraction must be > 0")
        if self.final_size_reference_px <= 0.0 or self.blur_scale <= 0.0:
            raise ValueError("quality scales must be > 0")
        if self.max_crops_per_track < 1:
            raise ValueError("max_crops_per_track must be >= 1")
        if self.min_time_gap_seconds < 0.0:
            raise ValueError("min_time_gap_seconds must be >= 0")

        proposal_weights = (
            self.proposal_size_weight,
            self.proposal_confidence_weight,
            self.proposal_edge_weight,
        )
        final_weights = (
            self.size_weight,
            self.clip_weight,
            self.blur_weight,
            self.confidence_weight,
            self.valid_weight,
        )
        if any(value < 0.0 for value in (*proposal_weights, *final_weights)):
            raise ValueError("quality weights must be >= 0")
        if abs(sum(proposal_weights) - 1.0) > 1e-9:
            raise ValueError("proposal quality weights must sum to 1")
        if abs(sum(final_weights) - 1.0) > 1e-9:
            raise ValueError("final quality weights must sum to 1")


@dataclass(frozen=True)
class ReIDCropCandidate:
    track_id: int
    frame_index: int
    timestamp: float
    source_camera: str
    source_bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float

    angular_height_deg: float
    source_height_fraction: float
    source_edge_score: float
    proposal_quality: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReIDCropMetrics:
    track_id: int
    frame_index: int
    timestamp: float
    source_camera: str
    source_bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float

    centred_longitude_deg: float
    centred_latitude_deg: float
    centred_vfov_deg: float
    centred_hfov_deg: float

    mapped_bbox_xyxy: tuple[float, float, float, float]
    crop_bbox_xyxy: tuple[int, int, int, int]

    player_height_px: float
    clipped_fraction: float
    valid_pixel_fraction: float

    laplacian_variance: float
    blur_score: float
    size_score: float
    clipping_score: float
    quality: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ReIDCrop:
    image: np.ndarray
    metrics: ReIDCropMetrics


def _rot_y3(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def _rot_x3(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ],
        dtype=np.float64,
    )


def _camera_rotation(spec: Any) -> np.ndarray:
    return _rot_y3(float(spec.longitude)) @ _rot_x3(float(spec.latitude))


def _validate_bbox(
    bbox: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    if len(bbox) != 4:
        raise ValueError("bbox must contain exactly four values")

    x1, y1, x2, y2 = (float(value) for value in bbox)

    if not np.isfinite([x1, y1, x2, y2]).all():
        raise ValueError("bbox values must be finite")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must satisfy x2>x1 and y2>y1")
    if x2 <= 0.0 or y2 <= 0.0 or x1 >= width or y1 >= height:
        raise ValueError("bbox does not overlap the source virtual image")

    return x1, y1, x2, y2


def pixel_to_presentation_direction(
    spec: Any,
    pixel_x: float,
    pixel_y: float,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Map one rectilinear virtual-camera pixel to a presentation-space ray."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not np.isfinite([pixel_x, pixel_y]).all():
        raise ValueError("pixel coordinates must be finite")

    # Keep the exact +0.5 convention used by VeoMatchLab's validated renderer.
    u = (float(pixel_x) + 0.5) / width
    v = (float(pixel_y) + 0.5) / height

    tan_v = math.tan(float(spec.vfov) / 2.0)
    aspect = width / height

    local = np.asarray(
        [
            (2.0 * u - 1.0) * tan_v * aspect,
            (2.0 * v - 1.0) * tan_v,
            1.0,
        ],
        dtype=np.float64,
    )

    presentation = local @ _camera_rotation(spec).T
    norm = float(np.linalg.norm(presentation))
    if not np.isfinite(presentation).all() or norm <= EPS:
        raise ValueError("invalid presentation ray")

    return presentation / norm


def presentation_direction_to_pixel(
    spec: Any,
    direction: Sequence[float],
    *,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Project a presentation-space ray into a rectilinear virtual camera."""

    ray = np.asarray(direction, dtype=np.float64)
    if ray.shape != (3,) or not np.isfinite(ray).all():
        raise ValueError("direction must be a finite 3-vector")

    # Inverse of pixel_to_presentation_direction(): row-pres @ R = row-local.
    local = ray @ _camera_rotation(spec)

    if float(local[2]) <= EPS:
        return None

    tan_v = math.tan(float(spec.vfov) / 2.0)
    aspect = width / height

    nx = float(local[0] / local[2])
    ny = float(local[1] / local[2])

    u = 0.5 * (nx / (tan_v * aspect) + 1.0)
    v = 0.5 * (ny / tan_v + 1.0)

    # Undo the renderer's pixel-centre convention.
    return u * width - 0.5, v * height - 0.5


def direction_angles(direction: Sequence[float]) -> tuple[float, float]:
    ray = np.asarray(direction, dtype=np.float64)
    if ray.shape != (3,) or not np.isfinite(ray).all():
        raise ValueError("direction must be a finite 3-vector")

    norm = float(np.linalg.norm(ray))
    if norm <= EPS:
        raise ValueError("direction norm must be > 0")

    ray = ray / norm
    longitude = math.atan2(float(ray[0]), float(ray[2]))
    latitude = math.atan2(
        -float(ray[1]),
        math.hypot(float(ray[0]), float(ray[2])),
    )
    return longitude, latitude


def angular_distance(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != (3,) or bb.shape != (3,):
        raise ValueError("angular_distance expects two 3-vectors")

    aa_norm = float(np.linalg.norm(aa))
    bb_norm = float(np.linalg.norm(bb))
    if aa_norm <= EPS or bb_norm <= EPS:
        raise ValueError("ray norms must be > 0")

    cosine = float(np.dot(aa / aa_norm, bb / bb_norm))
    if cosine >= 1.0 - 1e-12:
        return 0.0
    if cosine <= -1.0 + 1e-12:
        return math.pi
    return math.acos(min(1.0, max(-1.0, cosine)))


def candidate_from_detection(
    *,
    track_id: int,
    frame_index: int,
    timestamp: float,
    source_camera: str,
    source_spec: Any,
    source_bbox_xyxy: Sequence[float],
    detection_confidence: float,
    source_width: int,
    source_height: int,
    config: ReIDCropConfig | None = None,
) -> ReIDCropCandidate:
    """Create a cheap crop proposal without touching the 2048x2048 frame."""

    cfg = config or ReIDCropConfig()
    x1, y1, x2, y2 = _validate_bbox(
        source_bbox_xyxy,
        width=source_width,
        height=source_height,
    )

    cx = 0.5 * (x1 + x2)
    top = pixel_to_presentation_direction(
        source_spec,
        cx,
        max(0.0, min(source_height - 1.0, y1)),
        width=source_width,
        height=source_height,
    )
    bottom = pixel_to_presentation_direction(
        source_spec,
        cx,
        max(0.0, min(source_height - 1.0, y2 - 0.5)),
        width=source_width,
        height=source_height,
    )

    angular_height = angular_distance(top, bottom)
    angular_height_deg = math.degrees(angular_height)

    visible_y1 = max(0.0, y1)
    visible_y2 = min(float(source_height), y2)
    visible_height = max(0.0, visible_y2 - visible_y1)
    source_height_fraction = visible_height / source_height

    left_margin = max(0.0, min(1.0, x1 / source_width))
    right_margin = max(0.0, min(1.0, (source_width - x2) / source_width))
    top_margin = max(0.0, min(1.0, y1 / source_height))
    bottom_margin = max(0.0, min(1.0, (source_height - y2) / source_height))
    edge_margin = min(left_margin, right_margin, top_margin, bottom_margin)

    # Full credit once the bbox is ~3% away from every image edge.
    source_edge_score = min(1.0, edge_margin / 0.03)

    size_score = min(
        1.0,
        max(
            1e-6,
            source_height_fraction / cfg.proposal_reference_height_fraction,
        ),
    )
    confidence_score = min(
        1.0,
        max(1e-6, float(detection_confidence)),
    )
    edge_score = max(1e-6, source_edge_score)

    proposal_quality = _weighted_geometric_mean(
        (
            (size_score, cfg.proposal_size_weight),
            (confidence_score, cfg.proposal_confidence_weight),
            (edge_score, cfg.proposal_edge_weight),
        )
    )

    return ReIDCropCandidate(
        track_id=int(track_id),
        frame_index=int(frame_index),
        timestamp=float(timestamp),
        source_camera=str(source_camera),
        source_bbox_xyxy=(x1, y1, x2, y2),
        detection_confidence=float(detection_confidence),
        angular_height_deg=float(angular_height_deg),
        source_height_fraction=float(source_height_fraction),
        source_edge_score=float(source_edge_score),
        proposal_quality=float(proposal_quality),
    )


def centred_spec_for_candidate(
    source_spec: Any,
    candidate: ReIDCropCandidate,
    *,
    source_width: int,
    source_height: int,
    config: ReIDCropConfig | None = None,
) -> Any:
    """Return a narrow CameraSpec of the same dataclass type as source_spec."""

    cfg = config or ReIDCropConfig()
    x1, y1, x2, y2 = candidate.source_bbox_xyxy
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)

    centre_ray = pixel_to_presentation_direction(
        source_spec,
        cx,
        cy,
        width=source_width,
        height=source_height,
    )
    longitude, latitude = direction_angles(centre_ray)

    requested_vfov = math.radians(
        max(candidate.angular_height_deg, 1e-6)
    ) / cfg.target_player_fill
    vfov = min(
        math.radians(cfg.max_vfov_deg),
        max(math.radians(cfg.min_vfov_deg), requested_vfov),
    )

    aspect = cfg.render_width / cfg.render_height
    hfov = 2.0 * math.atan(
        math.tan(vfov / 2.0) * aspect
    )

    # CameraSpec is a frozen dataclass in coverage.py.  Reusing the original
    # object's non-geometric metadata means Sprint A stays compatible with the
    # current six-camera layout without duplicating that dataclass here.
    return replace(
        source_spec,
        longitude=float(longitude),
        latitude=float(latitude),
        vfov=float(vfov),
        hfov=float(hfov),
    )


def map_source_bbox_to_spec(
    source_spec: Any,
    target_spec: Any,
    source_bbox_xyxy: Sequence[float],
    *,
    source_width: int,
    source_height: int,
    target_width: int,
    target_height: int,
) -> tuple[float, float, float, float]:
    """Reproject the four source-bbox corner rays into another virtual view."""

    x1, y1, x2, y2 = _validate_bbox(
        source_bbox_xyxy,
        width=source_width,
        height=source_height,
    )

    corners = (
        (x1, y1),
        (x2, y1),
        (x1, y2 - 0.5),
        (x2, y2 - 0.5),
    )

    projected: list[tuple[float, float]] = []
    for x, y in corners:
        ray = pixel_to_presentation_direction(
            source_spec,
            max(0.0, min(source_width - 1.0, x)),
            max(0.0, min(source_height - 1.0, y)),
            width=source_width,
            height=source_height,
        )
        point = presentation_direction_to_pixel(
            target_spec,
            ray,
            width=target_width,
            height=target_height,
        )
        if point is not None:
            projected.append(point)

    if len(projected) < 2:
        raise ValueError("source bbox is not visible in centred ReID view")

    xs = [point[0] for point in projected]
    ys = [point[1] for point in projected]

    return (
        float(min(xs)),
        float(min(ys)),
        float(max(xs)),
        float(max(ys)),
    )


def padded_crop_box(
    mapped_bbox_xyxy: Sequence[float],
    *,
    width: int,
    height: int,
    config: ReIDCropConfig | None = None,
) -> tuple[tuple[int, int, int, int], float]:
    """Return clipped integer ROI + fraction of requested ROI outside image."""

    cfg = config or ReIDCropConfig()
    x1, y1, x2, y2 = (float(value) for value in mapped_bbox_xyxy)

    box_width = max(EPS, x2 - x1)
    box_height = max(EPS, y2 - y1)

    desired_x1 = x1 - cfg.horizontal_padding_fraction * box_width
    desired_x2 = x2 + cfg.horizontal_padding_fraction * box_width
    desired_y1 = y1 - cfg.top_padding_fraction * box_height
    desired_y2 = y2 + cfg.bottom_padding_fraction * box_height

    desired_area = max(
        EPS,
        (desired_x2 - desired_x1) * (desired_y2 - desired_y1),
    )

    clipped_x1 = max(0.0, desired_x1)
    clipped_y1 = max(0.0, desired_y1)
    clipped_x2 = min(float(width), desired_x2)
    clipped_y2 = min(float(height), desired_y2)

    inside_area = max(0.0, clipped_x2 - clipped_x1) * max(
        0.0,
        clipped_y2 - clipped_y1,
    )
    clipped_fraction = 1.0 - min(1.0, inside_area / desired_area)

    ix1 = max(0, min(width - 1, int(math.floor(clipped_x1))))
    iy1 = max(0, min(height - 1, int(math.floor(clipped_y1))))
    ix2 = max(ix1 + 1, min(width, int(math.ceil(clipped_x2))))
    iy2 = max(iy1 + 1, min(height, int(math.ceil(clipped_y2))))

    return (ix1, iy1, ix2, iy2), float(clipped_fraction)


def normalize_person_crop(
    image: np.ndarray,
    *,
    output_width: int,
    output_height: int,
) -> np.ndarray:
    """Aspect-preserving resize + centred zero padding."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("person crop must be an HxWx3 image")
    if image.size == 0:
        raise ValueError("person crop is empty")
    if output_width <= 0 or output_height <= 0:
        raise ValueError("output dimensions must be positive")

    h, w = image.shape[:2]
    scale = min(output_width / w, output_height / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    interpolation = (
        cv2.INTER_AREA
        if scale < 1.0
        else cv2.INTER_CUBIC
    )
    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=interpolation,
    )

    output = np.zeros(
        (output_height, output_width, 3),
        dtype=np.uint8,
    )
    x0 = (output_width - new_w) // 2
    y0 = (output_height - new_h) // 2
    output[y0:y0 + new_h, x0:x0 + new_w] = resized
    return output


def crop_quality(
    crop: np.ndarray,
    *,
    player_height_px: float,
    clipped_fraction: float,
    valid_pixel_fraction: float,
    detection_confidence: float,
    config: ReIDCropConfig | None = None,
) -> tuple[float, float, float, float]:
    """Return (quality, laplacian_variance, blur_score, size_score)."""

    cfg = config or ReIDCropConfig()
    if crop.size == 0:
        raise ValueError("crop must not be empty")

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(
        cv2.Laplacian(gray, cv2.CV_64F).var()
    )

    blur_score = 1.0 - math.exp(
        -max(0.0, laplacian_variance) / cfg.blur_scale
    )
    size_score = min(
        1.0,
        max(1e-6, player_height_px / cfg.final_size_reference_px),
    )
    clipping_score = max(
        1e-6,
        1.0 - min(1.0, max(0.0, clipped_fraction)),
    )
    confidence_score = max(
        1e-6,
        min(1.0, float(detection_confidence)),
    )
    valid_score = max(
        1e-6,
        min(1.0, float(valid_pixel_fraction)),
    )

    quality = _weighted_geometric_mean(
        (
            (size_score, cfg.size_weight),
            (clipping_score, cfg.clip_weight),
            (max(1e-6, blur_score), cfg.blur_weight),
            (confidence_score, cfg.confidence_weight),
            (valid_score, cfg.valid_weight),
        )
    )

    return (
        float(quality),
        float(laplacian_variance),
        float(blur_score),
        float(size_score),
    )


def render_reid_crop(
    *,
    frame: np.ndarray,
    metadata: Any,
    source_spec: Any,
    candidate: ReIDCropCandidate,
    source_width: int,
    source_height: int,
    config: ReIDCropConfig | None = None,
) -> ReIDCrop:
    """Render one calibrated player-centred crop from the raw Veo frame."""

    cfg = config or ReIDCropConfig()

    # Lazy import keeps the geometry-free unit tests light.
    from experiments.virtual_camera_coverage.coverage import (
        build_map,
        render_map,
    )

    centred_spec = centred_spec_for_candidate(
        source_spec,
        candidate,
        source_width=source_width,
        source_height=source_height,
        config=cfg,
    )

    camera_map = build_map(
        metadata,
        centred_spec,
        cfg.render_width,
        cfg.render_height,
    )
    centred_image = render_map(
        frame,
        camera_map,
    )

    mapped_bbox = map_source_bbox_to_spec(
        source_spec,
        centred_spec,
        candidate.source_bbox_xyxy,
        source_width=source_width,
        source_height=source_height,
        target_width=cfg.render_width,
        target_height=cfg.render_height,
    )

    crop_box, clipped_fraction = padded_crop_box(
        mapped_bbox,
        width=cfg.render_width,
        height=cfg.render_height,
        config=cfg,
    )
    x1, y1, x2, y2 = crop_box

    raw_crop = centred_image[y1:y2, x1:x2]
    if raw_crop.size == 0:
        raise ValueError("centred crop is empty")

    valid = camera_map.left_valid | camera_map.right_valid
    valid_roi = valid[y1:y2, x1:x2]
    valid_fraction = float(valid_roi.mean()) if valid_roi.size else 0.0

    player_height_px = max(
        0.0,
        float(mapped_bbox[3] - mapped_bbox[1]),
    )

    quality, lap_var, blur_score, size_score = crop_quality(
        raw_crop,
        player_height_px=player_height_px,
        clipped_fraction=clipped_fraction,
        valid_pixel_fraction=valid_fraction,
        detection_confidence=candidate.detection_confidence,
        config=cfg,
    )

    normalized = normalize_person_crop(
        raw_crop,
        output_width=cfg.output_width,
        output_height=cfg.output_height,
    )

    metrics = ReIDCropMetrics(
        track_id=candidate.track_id,
        frame_index=candidate.frame_index,
        timestamp=candidate.timestamp,
        source_camera=candidate.source_camera,
        source_bbox_xyxy=candidate.source_bbox_xyxy,
        detection_confidence=candidate.detection_confidence,
        centred_longitude_deg=math.degrees(
            float(centred_spec.longitude)
        ),
        centred_latitude_deg=math.degrees(
            float(centred_spec.latitude)
        ),
        centred_vfov_deg=math.degrees(
            float(centred_spec.vfov)
        ),
        centred_hfov_deg=math.degrees(
            float(centred_spec.hfov)
        ),
        mapped_bbox_xyxy=tuple(
            float(value) for value in mapped_bbox
        ),
        crop_bbox_xyxy=crop_box,
        player_height_px=float(player_height_px),
        clipped_fraction=float(clipped_fraction),
        valid_pixel_fraction=float(valid_fraction),
        laplacian_variance=float(lap_var),
        blur_score=float(blur_score),
        size_score=float(size_score),
        clipping_score=float(
            max(0.0, 1.0 - clipped_fraction)
        ),
        quality=float(quality),
    )

    return ReIDCrop(
        image=normalized,
        metrics=metrics,
    )


def select_diverse_candidates(
    candidates: Sequence[ReIDCropCandidate],
    *,
    max_crops: int | None = None,
    min_time_gap_seconds: float | None = None,
    config: ReIDCropConfig | None = None,
) -> tuple[ReIDCropCandidate, ...]:
    """Quality-first greedy selection with temporal diversity."""

    cfg = config or ReIDCropConfig()
    limit = (
        cfg.max_crops_per_track
        if max_crops is None
        else int(max_crops)
    )
    gap = (
        cfg.min_time_gap_seconds
        if min_time_gap_seconds is None
        else float(min_time_gap_seconds)
    )

    if limit < 1:
        raise ValueError("max_crops must be >= 1")
    if gap < 0.0:
        raise ValueError("min_time_gap_seconds must be >= 0")

    ordered = sorted(
        candidates,
        key=lambda item: (
            -item.proposal_quality,
            item.timestamp,
            item.frame_index,
        ),
    )

    selected: list[ReIDCropCandidate] = []

    for candidate in ordered:
        if any(
            abs(candidate.timestamp - existing.timestamp) < gap
            for existing in selected
        ):
            continue

        selected.append(candidate)
        if len(selected) >= limit:
            break

    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.timestamp,
                item.frame_index,
            ),
        )
    )


def select_diverse_crops(
    crops: Sequence[ReIDCrop],
    *,
    max_crops: int,
    min_time_gap_seconds: float,
) -> tuple[ReIDCrop, ...]:
    """Same policy after rendering, now using the expensive final quality."""

    if max_crops < 1:
        raise ValueError("max_crops must be >= 1")
    if min_time_gap_seconds < 0.0:
        raise ValueError("min_time_gap_seconds must be >= 0")

    ordered = sorted(
        crops,
        key=lambda item: (
            -item.metrics.quality,
            item.metrics.timestamp,
            item.metrics.frame_index,
        ),
    )

    selected: list[ReIDCrop] = []
    for crop in ordered:
        if any(
            abs(
                crop.metrics.timestamp
                - existing.metrics.timestamp
            ) < min_time_gap_seconds
            for existing in selected
        ):
            continue

        selected.append(crop)
        if len(selected) >= max_crops:
            break

    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.metrics.timestamp,
                item.metrics.frame_index,
            ),
        )
    )


def candidates_by_track(
    candidates: Sequence[ReIDCropCandidate],
) -> dict[int, tuple[ReIDCropCandidate, ...]]:
    grouped: dict[int, list[ReIDCropCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(
            candidate.track_id,
            [],
        ).append(candidate)

    return {
        track_id: tuple(
            sorted(
                rows,
                key=lambda item: (
                    item.timestamp,
                    item.frame_index,
                ),
            )
        )
        for track_id, rows in sorted(grouped.items())
    }


def _weighted_geometric_mean(
    terms: Sequence[tuple[float, float]],
) -> float:
    total_weight = sum(weight for _, weight in terms)
    if total_weight <= 0.0:
        raise ValueError("at least one positive weight is required")

    log_sum = 0.0
    for value, weight in terms:
        if weight <= 0.0:
            continue
        log_sum += weight * math.log(
            max(1e-9, min(1.0, float(value)))
        )

    return float(math.exp(log_sum / total_weight))
