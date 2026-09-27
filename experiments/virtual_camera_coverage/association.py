#!/usr/bin/env python3
# CHATGPT DELIVERY MARKER: ASSOCIATION_V5
"""Same-frame PRIMARY/RESCUE association plus image-confirmed rescue promotion.

This module does NOT track players over time.

Its only responsibility is to take raw person detections from the six
overlapping virtual cameras of the coverage experiment and consolidate
multiple observations of the SAME physical player within ONE source frame.

Design principles
-----------------
1. Proximity in field X/Z is NEVER sufficient to declare two detections equal.
2. Render overlap is useful and is intentionally preserved.
3. Logical ownership is exclusive, but a non-owner observation is not thrown
   away if it is the only valid observation of a player.
4. The strongest same-frame identity signal is that two virtual detections map
   back to the same physical pixels in the original Veo lens image.
5. Matching is one-to-one per camera pair.
6. Ambiguous evidence is kept separate rather than aggressively merged.
7. Every final PersonMeasurement retains all contributing observations and
   provenance for later tracking / offline solving.

The association score therefore uses:
- overlap of the detections' footprints in the ORIGINAL Veo lens coordinates,
- distance between their original-lens foot anchors,
- field X/Z compatibility as a permissive gate only,
- topology: only camera pairs that can geometrically overlap are compared.

No appearance/ReID and no temporal information are used yet.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable

import cv2
import numpy as np

from experiments.virtual_camera_coverage.coverage import (
    AsymmetricOwnershipLayout,
    CameraMap,
    CameraSpec,
    depth_from_near,
    owner_ids,
    presentation_directions,
    spherical_longitudes,
)
from person_detect.views.interactive_view.renderer import (
    camera_world,
    presentation_to_world,
    rot_x,
    rot_y,
)


EPS = 1e-12

DEFAULT_FIELD_MARGIN_M = 0.60

# Conservative by design: it is better to leave one duplicated person
# unresolved than to merge two real nearby players.
DEFAULT_MIN_SOURCE_IOU = 0.25
DEFAULT_MAX_SOURCE_ANCHOR_NORM = 0.35
DEFAULT_MAX_WORLD_DISTANCE_M = 1.75

# Partial / scale-mismatched duplicate handling. These values are normalized
# against the smaller original-source footprint, so they transfer across Veo
# matches and output resolutions better than raw pixel thresholds.
DEFAULT_MIN_SOURCE_IOM = 0.58
DEFAULT_MIN_HORIZONTAL_IOM = 0.72
DEFAULT_MIN_VERTICAL_IOM = 0.45
DEFAULT_MAX_CENTER_X_NORM = 0.45
DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M = 2.75
DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M = 6.0

# A detector box this close to a virtual-image edge is considered clipped.
DEFAULT_TRUNCATION_EDGE_FRACTION = 0.015
DEFAULT_TRUNCATION_EDGE_MIN_PX = 6.0

# PRIMARY/RESCUE policy.
# A non-owner observation may help an owner-backed person, but cannot create a
# new person by itself. Two or more mutually consistent RESCUE observations
# may be promoted if they are not source-ambiguous with any PRIMARY.
DEFAULT_RESCUE_CONFLICT_MIN_IOM = 0.40
DEFAULT_RESCUE_CONFLICT_MIN_HORIZONTAL_IOM = 0.55
DEFAULT_RESCUE_CONFLICT_MAX_CENTER_X_NORM = 0.80

# Source-footprint sampling density inside each virtual bbox.
DEFAULT_FOOTPRINT_GRID = 7


@dataclass(frozen=True)
class SourceLensEvidence:
    """Where one virtual bbox came from in one original physical Veo lens."""

    lens: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    anchor_x: float
    anchor_y: float
    sample_count: int

    @property
    def width(self) -> float:
        return max(0.0, self.x_max - self.x_min)

    @property
    def height(self) -> float:
        return max(0.0, self.y_max - self.y_min)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center_x(self) -> float:
        return 0.5 * (self.x_min + self.x_max)

    @property
    def center_y(self) -> float:
        return 0.5 * (self.y_min + self.y_max)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "width": self.width,
                "height": self.height,
                "area": self.area,
                "center_x": self.center_x,
                "center_y": self.center_y,
            }
        )
        return result


@dataclass(frozen=True)
class Observation:
    """One detector bbox in one virtual camera."""

    observation_id: int
    frame_number: int
    time_s: float

    view_name: str
    band: str
    index_in_band: int

    xyxy: tuple[float, float, float, float]
    confidence: float

    foot_x: float
    foot_y: float

    world_x: float | None
    world_z: float | None
    inside_strict_field: bool | None
    inside_allowed_field: bool | None

    owner_cell: str | None
    is_owner: bool | None

    source_lenses: tuple[SourceLensEvidence, ...]

    truncated_top: bool
    truncated_bottom: bool
    truncated_left: bool
    truncated_right: bool

    status: str

    @property
    def is_truncated(self) -> bool:
        return (
            self.truncated_top
            or self.truncated_bottom
            or self.truncated_left
            or self.truncated_right
        )

    @property
    def ownership_role(self) -> str:
        if self.status != "accepted":
            return "invalid"
        if self.is_owner:
            return "primary"
        return "rescue"

    def lens_evidence(self, lens: str) -> SourceLensEvidence | None:
        for item in self.source_lenses:
            if item.lens == lens:
                return item
        return None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["source_lenses"] = [
            item.to_dict()
            for item in self.source_lenses
        ]
        result["is_truncated"] = self.is_truncated
        result["ownership_role"] = self.ownership_role
        return result


@dataclass(frozen=True)
class MatchEdge:
    """Accepted duplicate relation between observations from two views."""

    observation_a: int
    observation_b: int
    view_a: str
    view_b: str

    source_lens: str
    match_mode: str

    source_iou: float
    source_iom: float
    source_horizontal_iom: float
    source_vertical_iom: float
    source_center_x_distance_norm: float

    source_anchor_distance_px: float
    source_anchor_distance_norm: float

    world_distance_m: float
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourcePairMetrics:
    lens: str
    iou: float
    iom: float
    horizontal_iom: float
    vertical_iom: float
    center_x_distance_norm: float
    anchor_distance_px: float
    anchor_distance_norm: float


@dataclass(frozen=True)
class PersonMeasurement:
    """One unique per-frame person hypothesis, with full provenance."""

    measurement_id: int
    observation_ids: tuple[int, ...]
    primary_observation_id: int

    representative_world_x: float | None
    representative_world_z: float | None

    fused_world_x: float | None
    fused_world_z: float | None
    position_spread_m: float | None

    confidence: float
    owner_supported: bool
    multi_view: bool
    creation_source: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssociationResult:
    observations: tuple[Observation, ...]
    matches: tuple[MatchEdge, ...]
    measurements: tuple[PersonMeasurement, ...]
    deferred_rescue_observation_ids: tuple[int, ...]

    @property
    def primary_observation_count(self) -> int:
        return sum(
            item.status == "accepted"
            and bool(item.is_owner)
            for item in self.observations
        )

    @property
    def rescue_observation_count(self) -> int:
        return sum(
            item.status == "accepted"
            and not bool(item.is_owner)
            for item in self.observations
        )

    @property
    def promoted_rescue_measurement_count(self) -> int:
        return sum(
            item.creation_source == "promoted_rescue"
            for item in self.measurements
        )

    @property
    def image_confirmed_rescue_measurement_count(self) -> int:
        return sum(
            item.creation_source == "image_confirmed_rescue"
            for item in self.measurements
        )

    @property
    def owner_backed_measurement_count(self) -> int:
        return sum(
            item.creation_source == "primary"
            for item in self.measurements
        )

    def measurement_for_observation(
        self,
        observation_id: int,
    ) -> PersonMeasurement | None:
        for measurement in self.measurements:
            if observation_id in measurement.observation_ids:
                return measurement
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "observations": [
                item.to_dict()
                for item in self.observations
            ],
            "matches": [
                item.to_dict()
                for item in self.matches
            ],
            "measurements": [
                item.to_dict()
                for item in self.measurements
            ],
            "deferred_rescue_observation_ids": list(
                self.deferred_rescue_observation_ids
            ),
            "ownership_summary": {
                "primary_observations": self.primary_observation_count,
                "rescue_observations": self.rescue_observation_count,
                "owner_backed_measurements": (
                    self.owner_backed_measurement_count
                ),
                "promoted_rescue_measurements": (
                    self.promoted_rescue_measurement_count
                ),
                "image_confirmed_rescue_measurements": (
                    self.image_confirmed_rescue_measurement_count
                ),
                "deferred_rescue_observations": len(
                    self.deferred_rescue_observation_ids
                ),
            },
        }


def _clip_point(
    x: float,
    y: float,
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        min(max(float(x), 0.0), width - 1.0),
        min(max(float(y), 0.0), height - 1.0),
    )


def bbox_footpoint(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float]:
    x1, y1, x2, y2 = xyxy

    foot_x = 0.5 * (float(x1) + float(x2))
    foot_y = float(y2) - 0.5

    return _clip_point(
        foot_x,
        foot_y,
        width,
        height,
    )


def virtual_pixel_to_ground(
    metadata,
    spec: CameraSpec,
    pixel_x: float,
    pixel_y: float,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    """Inverse of the rectilinear virtual-camera projection for Y=0 ground."""
    if width <= 0 or height <= 0:
        raise ValueError("virtual image dimensions must be positive")

    x, y = _clip_point(
        pixel_x,
        pixel_y,
        width,
        height,
    )

    u = (x + 0.5) / float(width)
    v = (y + 0.5) / float(height)

    tan_v = math.tan(spec.vfov / 2.0)
    aspect = width / height

    view_ray = np.asarray(
        [
            (2.0 * u - 1.0) * tan_v * aspect,
            (2.0 * v - 1.0) * tan_v,
            1.0,
            0.0,
        ],
        dtype=np.float64,
    )

    transform = (
        presentation_to_world(metadata)
        @ rot_y(spec.longitude)
        @ rot_x(spec.latitude)
    )

    direction = (transform @ view_ray)[:3]
    origin = camera_world(metadata)

    if not np.isfinite(direction).all():
        return None

    dy = float(direction[1])

    if abs(dy) <= EPS:
        return None

    ray_parameter = -float(origin[1]) / dy

    if not math.isfinite(ray_parameter) or ray_parameter <= 0.0:
        return None

    world = origin + ray_parameter * direction

    if not np.isfinite(world).all():
        return None

    return float(world[0]), float(world[2])


def classify_field_point(
    metadata,
    world_x: float,
    world_z: float,
    *,
    margin_m: float,
) -> tuple[bool, bool]:
    if margin_m < 0.0:
        raise ValueError("field margin must be non-negative")

    half_length = float(metadata.field_length) / 2.0
    half_width = float(metadata.field_width) / 2.0

    strict = (
        -half_length <= world_x <= half_length
        and -half_width <= world_z <= half_width
    )

    allowed = (
        -half_length - margin_m
        <= world_x
        <= half_length + margin_m
        and -half_width - margin_m
        <= world_z
        <= half_width + margin_m
    )

    return strict, allowed


def logical_owner_for_ground(
    metadata,
    layout: AsymmetricOwnershipLayout,
    world_x: float,
    world_z: float,
) -> str:
    point = np.asarray(
        [[world_x, 0.0, world_z]],
        dtype=np.float64,
    )

    direction = presentation_directions(
        metadata,
        point,
    )

    longitude = spherical_longitudes(
        direction,
        layout.reference_lon,
    )

    depth = depth_from_near(
        np.asarray([world_z], dtype=np.float64),
        layout.near_z,
        layout.depth_sign,
    )

    owner_id = int(
        owner_ids(
            longitude,
            depth,
            layout,
        )[0]
    )

    return layout.cell_by_id(owner_id).name


def _bbox_sample_indices(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if grid_size < 2:
        raise ValueError("grid_size must be >= 2")

    x1, y1, x2, y2 = (
        float(value)
        for value in xyxy
    )

    # Stay inside the outer bbox edges so every sample corresponds to a real
    # virtual output pixel.
    left = min(max(x1 + 0.5, 0.0), width - 1.0)
    right = min(max(x2 - 0.5, 0.0), width - 1.0)
    top = min(max(y1 + 0.5, 0.0), height - 1.0)
    bottom = min(max(y2 - 0.5, 0.0), height - 1.0)

    if right < left:
        left = right = min(
            max(0.5 * (x1 + x2), 0.0),
            width - 1.0,
        )

    if bottom < top:
        top = bottom = min(
            max(0.5 * (y1 + y2), 0.0),
            height - 1.0,
        )

    xs = np.linspace(
        left,
        right,
        grid_size,
    )
    ys = np.linspace(
        top,
        bottom,
        grid_size,
    )

    xx, yy = np.meshgrid(
        xs,
        ys,
    )

    xi = np.clip(
        np.rint(xx).astype(np.int32),
        0,
        width - 1,
    )

    yi = np.clip(
        np.rint(yy).astype(np.int32),
        0,
        height - 1,
    )

    return xi, yi


def _source_lens_evidence(
    camera_map: CameraMap,
    xyxy: tuple[float, float, float, float],
    *,
    lens: str,
    grid_size: int,
) -> SourceLensEvidence | None:
    height, width = camera_map.left_x.shape

    if lens == "left":
        map_x = camera_map.left_x
        map_y = camera_map.left_y
        valid = camera_map.left_valid
    elif lens == "right":
        map_x = camera_map.right_x
        map_y = camera_map.right_y
        valid = camera_map.right_valid
    else:
        raise ValueError(f"unknown lens: {lens!r}")

    xi, yi = _bbox_sample_indices(
        xyxy,
        width,
        height,
        grid_size,
    )

    sample_valid = valid[yi, xi]

    sx = map_x[yi, xi][sample_valid]
    sy = map_y[yi, xi][sample_valid]

    finite = (
        np.isfinite(sx)
        & np.isfinite(sy)
        & (sx >= 0.0)
        & (sy >= 0.0)
    )

    sx = sx[finite]
    sy = sy[finite]

    if len(sx) < 3:
        return None

    x_min = float(sx.min())
    x_max = float(sx.max())
    y_min = float(sy.min())
    y_max = float(sy.max())

    # Try the exact virtual footpoint first.
    foot_x, foot_y = bbox_footpoint(
        xyxy,
        width,
        height,
    )

    fx = int(
        np.clip(
            round(foot_x),
            0,
            width - 1,
        )
    )
    fy = int(
        np.clip(
            round(foot_y),
            0,
            height - 1,
        )
    )

    if bool(valid[fy, fx]):
        anchor_x = float(map_x[fy, fx])
        anchor_y = float(map_y[fy, fx])
    else:
        # Fallback: bottom-centre of the mapped source footprint.  This is less
        # precise, but keeps the evidence usable near a lens-validity boundary.
        anchor_x = 0.5 * (x_min + x_max)
        anchor_y = y_max

    if not np.isfinite(
        [
            x_min,
            x_max,
            y_min,
            y_max,
            anchor_x,
            anchor_y,
        ]
    ).all():
        return None

    return SourceLensEvidence(
        lens=lens,
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        sample_count=int(len(sx)),
    )


def source_lens_evidence_for_bbox(
    camera_map: CameraMap,
    xyxy: tuple[float, float, float, float],
    *,
    grid_size: int = DEFAULT_FOOTPRINT_GRID,
) -> tuple[SourceLensEvidence, ...]:
    result: list[SourceLensEvidence] = []

    for lens in ("left", "right"):
        item = _source_lens_evidence(
            camera_map,
            xyxy,
            lens=lens,
            grid_size=grid_size,
        )

        if item is not None:
            result.append(item)

    return tuple(result)


def _bbox_truncation_flags(
    xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[bool, bool, bool, bool]:
    x1, y1, x2, y2 = (
        float(value)
        for value in xyxy
    )

    margin_x = max(
        DEFAULT_TRUNCATION_EDGE_MIN_PX,
        DEFAULT_TRUNCATION_EDGE_FRACTION * width,
    )
    margin_y = max(
        DEFAULT_TRUNCATION_EDGE_MIN_PX,
        DEFAULT_TRUNCATION_EDGE_FRACTION * height,
    )

    return (
        y1 <= margin_y,
        y2 >= height - margin_y,
        x1 <= margin_x,
        x2 >= width - margin_x,
    )


def build_observation(
    *,
    observation_id: int,
    frame_number: int,
    time_s: float,
    metadata,
    layout: AsymmetricOwnershipLayout,
    spec: CameraSpec,
    camera_map: CameraMap,
    xyxy: tuple[float, float, float, float],
    confidence: float,
    field_margin_m: float = DEFAULT_FIELD_MARGIN_M,
    footprint_grid: int = DEFAULT_FOOTPRINT_GRID,
) -> Observation:
    height, width = camera_map.left_x.shape

    foot_x, foot_y = bbox_footpoint(
        xyxy,
        width,
        height,
    )

    (
        truncated_top,
        truncated_bottom,
        truncated_left,
        truncated_right,
    ) = _bbox_truncation_flags(
        xyxy,
        width,
        height,
    )

    ground = virtual_pixel_to_ground(
        metadata,
        spec,
        foot_x,
        foot_y,
        width,
        height,
    )

    source_lenses = source_lens_evidence_for_bbox(
        camera_map,
        xyxy,
        grid_size=footprint_grid,
    )

    common_kwargs = {
        "observation_id": observation_id,
        "frame_number": frame_number,
        "time_s": time_s,
        "view_name": spec.name,
        "band": spec.cell.band,
        "index_in_band": spec.cell.index_in_band,
        "xyxy": xyxy,
        "confidence": float(confidence),
        "foot_x": foot_x,
        "foot_y": foot_y,
        "source_lenses": source_lenses,
        "truncated_top": truncated_top,
        "truncated_bottom": truncated_bottom,
        "truncated_left": truncated_left,
        "truncated_right": truncated_right,
    }

    if ground is None:
        return Observation(
            **common_kwargs,
            world_x=None,
            world_z=None,
            inside_strict_field=None,
            inside_allowed_field=None,
            owner_cell=None,
            is_owner=None,
            status="no_ground_intersection",
        )

    world_x, world_z = ground

    strict, allowed = classify_field_point(
        metadata,
        world_x,
        world_z,
        margin_m=field_margin_m,
    )

    owner_cell = logical_owner_for_ground(
        metadata,
        layout,
        world_x,
        world_z,
    )

    is_owner = owner_cell == spec.cell.name

    if not allowed:
        status = "outside_field"
    elif not source_lenses:
        status = "no_source_reference"
    else:
        status = "accepted"

    return Observation(
        **common_kwargs,
        world_x=world_x,
        world_z=world_z,
        inside_strict_field=strict,
        inside_allowed_field=allowed,
        owner_cell=owner_cell,
        is_owner=is_owner,
        status=status,
    )


def rectangle_overlap_metrics(
    a: SourceLensEvidence,
    b: SourceLensEvidence,
) -> tuple[float, float, float, float, float]:
    """Return IoU, IoM, horizontal-IoM, vertical-IoM and normalized dx.

    IoM = intersection over the smaller footprint. It is crucial for the
    common case where one virtual camera sees only a torso while another sees
    the complete player: IoU can be small although the partial box is almost
    entirely contained in the full one.
    """
    ix1 = max(a.x_min, b.x_min)
    iy1 = max(a.y_min, b.y_min)
    ix2 = min(a.x_max, b.x_max)
    iy2 = min(a.y_max, b.y_max)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    intersection = iw * ih

    union = a.area + b.area - intersection
    iou = (
        intersection / union
        if union > EPS
        else 0.0
    )

    smaller_area = min(a.area, b.area)
    iom = (
        intersection / smaller_area
        if smaller_area > EPS
        else 0.0
    )

    smaller_width = max(
        min(a.width, b.width),
        1.0,
    )
    smaller_height = max(
        min(a.height, b.height),
        1.0,
    )

    horizontal_iom = iw / smaller_width
    vertical_iom = ih / smaller_height

    center_x_distance_norm = (
        abs(a.center_x - b.center_x)
        / smaller_width
    )

    return (
        float(iou),
        float(iom),
        float(horizontal_iom),
        float(vertical_iom),
        float(center_x_distance_norm),
    )


def rectangle_iou(
    a: SourceLensEvidence,
    b: SourceLensEvidence,
) -> float:
    return rectangle_overlap_metrics(
        a,
        b,
    )[0]


def source_pair_metrics(
    a: Observation,
    b: Observation,
) -> SourcePairMetrics | None:
    """Return the strongest common-original-lens relation."""
    best: tuple[
        tuple[float, float, float, float],
        SourcePairMetrics,
    ] | None = None

    for lens in ("left", "right"):
        evidence_a = a.lens_evidence(lens)
        evidence_b = b.lens_evidence(lens)

        if evidence_a is None or evidence_b is None:
            continue

        (
            iou,
            iom,
            horizontal_iom,
            vertical_iom,
            center_x_norm,
        ) = rectangle_overlap_metrics(
            evidence_a,
            evidence_b,
        )

        anchor_distance = math.hypot(
            evidence_a.anchor_x - evidence_b.anchor_x,
            evidence_a.anchor_y - evidence_b.anchor_y,
        )

        reference_height = max(
            min(
                evidence_a.height,
                evidence_b.height,
            ),
            1.0,
        )

        anchor_norm = (
            anchor_distance
            / reference_height
        )

        metrics = SourcePairMetrics(
            lens=lens,
            iou=iou,
            iom=iom,
            horizontal_iom=horizontal_iom,
            vertical_iom=vertical_iom,
            center_x_distance_norm=center_x_norm,
            anchor_distance_px=float(anchor_distance),
            anchor_distance_norm=float(anchor_norm),
        )

        # IoM and horizontal alignment are deliberately ranked before IoU:
        # this lets a torso-only crop match the full-body observation from a
        # neighbouring zoom while still separating side-by-side players.
        rank = (
            iom,
            horizontal_iom,
            -center_x_norm,
            iou,
        )

        if best is None or rank > best[0]:
            best = (
                rank,
                metrics,
            )

    if best is None:
        return None

    return best[1]


def world_distance(
    a: Observation,
    b: Observation,
) -> float | None:
    if (
        a.world_x is None
        or a.world_z is None
        or b.world_x is None
        or b.world_z is None
    ):
        return None

    return float(
        math.hypot(
            a.world_x - b.world_x,
            a.world_z - b.world_z,
        )
    )


def candidate_view_pairs(
    layout: AsymmetricOwnershipLayout,
) -> tuple[tuple[str, str], ...]:
    """Only return view pairs whose logical cells touch / overlap topologically."""
    cells = layout.cells()
    result: set[tuple[str, str]] = set()

    # Same-band neighbours.
    for band in ("near", "far"):
        same_band = sorted(
            [
                cell
                for cell in cells
                if cell.band == band
            ],
            key=lambda item: item.index_in_band,
        )

        for left, right in zip(
            same_band,
            same_band[1:],
        ):
            result.add(
                (
                    f"tight_{left.name}",
                    f"tight_{right.name}",
                )
            )

    # NEAR/FAR touch at the split depth. Compare only angularly overlapping
    # cells; these are the only cross-band pairs that can share a player.
    near_cells = [
        cell
        for cell in cells
        if cell.band == "near"
    ]
    far_cells = [
        cell
        for cell in cells
        if cell.band == "far"
    ]

    for near in near_cells:
        for far in far_cells:
            overlap = min(
                near.lon_max,
                far.lon_max,
            ) - max(
                near.lon_min,
                far.lon_min,
            )

            if overlap >= -1e-9:
                result.add(
                    (
                        f"tight_{near.name}",
                        f"tight_{far.name}",
                    )
                )

    return tuple(
        sorted(
            tuple(sorted(pair))
            for pair in result
        )
    )


def candidate_edge(
    a: Observation,
    b: Observation,
    *,
    min_source_iou: float = DEFAULT_MIN_SOURCE_IOU,
    max_source_anchor_norm: float = DEFAULT_MAX_SOURCE_ANCHOR_NORM,
    max_world_distance_m: float = DEFAULT_MAX_WORLD_DISTANCE_M,
    min_source_iom: float = DEFAULT_MIN_SOURCE_IOM,
    min_horizontal_iom: float = DEFAULT_MIN_HORIZONTAL_IOM,
    min_vertical_iom: float = DEFAULT_MIN_VERTICAL_IOM,
    max_center_x_norm: float = DEFAULT_MAX_CENTER_X_NORM,
    max_containment_world_distance_m: float = DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
    max_truncated_world_distance_m: float = DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
) -> MatchEdge | None:
    """Conservative duplicate test driven by the ORIGINAL Veo pixels.

    There are three accepted evidence modes:

    strict_iou
        Full/full detections with good source IoU and foot-anchor agreement.

    source_containment
        Scale-mismatched detections where the smaller source footprint is
        substantially contained inside the larger one.

    truncated_source_containment
        A clipped torso/partial player in one virtual camera versus a fuller
        observation in another. In this mode bbox bottom-center is not trusted
        as a footpoint, so X/Z is only a broad safety gate.

    Field proximity alone can never create an edge.
    """
    if a.view_name == b.view_name:
        return None

    if (
        a.status != "accepted"
        or b.status != "accepted"
    ):
        return None

    source = source_pair_metrics(
        a,
        b,
    )

    if source is None:
        return None

    distance = world_distance(
        a,
        b,
    )

    if distance is None:
        return None

    strict = (
        source.iou >= min_source_iou
        and source.anchor_distance_norm
        <= max_source_anchor_norm
        and distance
        <= max_world_distance_m
    )

    source_containment = (
        source.iom >= min_source_iom
        and source.horizontal_iom
        >= min_horizontal_iom
        and source.vertical_iom
        >= min_vertical_iom
        and source.center_x_distance_norm
        <= max_center_x_norm
        and distance
        <= max_containment_world_distance_m
    )

    truncated_containment = (
        (a.is_truncated or b.is_truncated)
        and source.iom >= min_source_iom
        and source.horizontal_iom
        >= min_horizontal_iom
        and source.vertical_iom
        >= min_vertical_iom
        and source.center_x_distance_norm
        <= max_center_x_norm
        and distance
        <= max_truncated_world_distance_m
    )

    if strict:
        mode = "strict_iou"
        anchor_similarity = max(
            0.0,
            1.0
            - source.anchor_distance_norm
            / max(
                max_source_anchor_norm,
                EPS,
            ),
        )
        world_similarity = max(
            0.0,
            1.0
            - distance
            / max(
                max_world_distance_m,
                EPS,
            ),
        )
        score = (
            0.50 * source.iou
            + 0.20 * source.iom
            + 0.20 * anchor_similarity
            + 0.10 * world_similarity
        )

    elif truncated_containment:
        mode = "truncated_source_containment"
        center_similarity = max(
            0.0,
            1.0
            - source.center_x_distance_norm
            / max(
                max_center_x_norm,
                EPS,
            ),
        )
        world_similarity = max(
            0.0,
            1.0
            - distance
            / max(
                max_truncated_world_distance_m,
                EPS,
            ),
        )
        score = (
            0.45 * source.iom
            + 0.30 * source.horizontal_iom
            + 0.15 * center_similarity
            + 0.10 * world_similarity
        )

    elif source_containment:
        mode = "source_containment"
        center_similarity = max(
            0.0,
            1.0
            - source.center_x_distance_norm
            / max(
                max_center_x_norm,
                EPS,
            ),
        )
        world_similarity = max(
            0.0,
            1.0
            - distance
            / max(
                max_containment_world_distance_m,
                EPS,
            ),
        )
        score = (
            0.45 * source.iom
            + 0.30 * source.horizontal_iom
            + 0.15 * center_similarity
            + 0.10 * world_similarity
        )

    else:
        return None

    return MatchEdge(
        observation_a=a.observation_id,
        observation_b=b.observation_id,
        view_a=a.view_name,
        view_b=b.view_name,
        source_lens=source.lens,
        match_mode=mode,
        source_iou=source.iou,
        source_iom=source.iom,
        source_horizontal_iom=(
            source.horizontal_iom
        ),
        source_vertical_iom=(
            source.vertical_iom
        ),
        source_center_x_distance_norm=(
            source.center_x_distance_norm
        ),
        source_anchor_distance_px=(
            source.anchor_distance_px
        ),
        source_anchor_distance_norm=(
            source.anchor_distance_norm
        ),
        world_distance_m=distance,
        score=float(score),
    )


def one_to_one_matches_for_view_pair(
    observations_a: Iterable[Observation],
    observations_b: Iterable[Observation],
    **edge_kwargs,
) -> list[MatchEdge]:
    """Conservative maximum-score greedy bipartite matching.

    The candidate gate is intentionally strong, making this sparse.  Sorting by
    score and consuming each observation at most once prevents two close real
    players in one view from both collapsing onto one detection in the other.

    A future global solver may replace this local matcher, but the one-to-one
    invariant must remain.
    """
    edges: list[MatchEdge] = []

    for a in observations_a:
        for b in observations_b:
            edge = candidate_edge(
                a,
                b,
                **edge_kwargs,
            )

            if edge is not None:
                edges.append(edge)

    edges.sort(
        key=lambda item: (
            item.score,
            item.source_iom,
            item.source_horizontal_iom,
            item.source_iou,
            -item.source_center_x_distance_norm,
            -item.world_distance_m,
        ),
        reverse=True,
    )

    used_a: set[int] = set()
    used_b: set[int] = set()
    selected: list[MatchEdge] = []

    for edge in edges:
        if (
            edge.observation_a in used_a
            or edge.observation_b in used_b
        ):
            continue

        used_a.add(
            edge.observation_a
        )
        used_b.add(
            edge.observation_b
        )
        selected.append(edge)

    return selected


class _DisjointSet:
    def __init__(self, ids: Iterable[int]):
        self.parent = {
            value: value
            for value in ids
        }

    def find(self, value: int) -> int:
        root = value

        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[value] != value:
            next_value = self.parent[value]
            self.parent[value] = root
            value = next_value

        return root

    def union(
        self,
        a: int,
        b: int,
    ) -> None:
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a != root_b:
            self.parent[root_b] = root_a


def _component_has_duplicate_view(
    observation_ids: set[int],
    by_id: dict[int, Observation],
) -> bool:
    views = [
        by_id[value].view_name
        for value in observation_ids
    ]

    return len(views) != len(set(views))


def merge_match_components(
    observations: list[Observation],
    matches: list[MatchEdge],
) -> list[list[Observation]]:
    """Merge accepted pair matches while preserving at most one obs per view."""
    by_id = {
        item.observation_id: item
        for item in observations
    }

    accepted_ids = {
        item.observation_id
        for item in observations
        if item.status == "accepted"
    }

    dsu = _DisjointSet(
        accepted_ids
    )

    # Strongest edges first.  Before joining two components, enforce the
    # no-two-observations-from-the-same-view invariant.
    for edge in sorted(
        matches,
        key=lambda item: item.score,
        reverse=True,
    ):
        root_a = dsu.find(
            edge.observation_a
        )
        root_b = dsu.find(
            edge.observation_b
        )

        if root_a == root_b:
            continue

        group_a = {
            value
            for value in accepted_ids
            if dsu.find(value) == root_a
        }
        group_b = {
            value
            for value in accepted_ids
            if dsu.find(value) == root_b
        }

        merged = group_a | group_b

        if _component_has_duplicate_view(
            merged,
            by_id,
        ):
            continue

        dsu.union(
            edge.observation_a,
            edge.observation_b,
        )

    groups: dict[int, list[Observation]] = {}

    for observation_id in sorted(
        accepted_ids
    ):
        root = dsu.find(
            observation_id
        )

        groups.setdefault(
            root,
            [],
        ).append(
            by_id[observation_id]
        )

    return list(
        groups.values()
    )


def _primary_observation(
    observations: list[Observation],
) -> Observation:
    """Choose the observation with the most trustworthy ground position.

    A clipped torso can have very high confidence but its bbox bottom is not a
    footpoint. Therefore non-truncated evidence is preferred before ownership.
    """
    def key(
        item: Observation,
    ) -> tuple[int, int, float, float]:
        x1, y1, x2, y2 = item.xyxy

        area = max(
            0.0,
            (x2 - x1) * (y2 - y1),
        )

        return (
            1 if not item.is_truncated else 0,
            1 if item.is_owner else 0,
            item.confidence,
            area,
        )

    return max(
        observations,
        key=key,
    )


def _fused_position(
    observations: list[Observation],
) -> tuple[
    float | None,
    float | None,
    float | None,
]:
    usable = [
        item
        for item in observations
        if (
            item.world_x is not None
            and item.world_z is not None
        )
    ]

    if not usable:
        return None, None, None

    # If at least one complete player observation exists, do not let a clipped
    # torso corrupt the fused ground position.
    complete = [
        item
        for item in usable
        if not item.is_truncated
    ]

    if complete:
        usable = complete

    weights = np.asarray(
        [
            max(
                item.confidence,
                1e-3,
            )
            * (
                1.0
                if item.is_owner
                else 0.75
            )
            for item in usable
        ],
        dtype=np.float64,
    )

    xs = np.asarray(
        [
            item.world_x
            for item in usable
        ],
        dtype=np.float64,
    )
    zs = np.asarray(
        [
            item.world_z
            for item in usable
        ],
        dtype=np.float64,
    )

    fused_x = float(
        np.average(
            xs,
            weights=weights,
        )
    )
    fused_z = float(
        np.average(
            zs,
            weights=weights,
        )
    )

    spread = float(
        max(
            math.hypot(
                float(x) - fused_x,
                float(z) - fused_z,
            )
            for x, z in zip(
                xs,
                zs,
                strict=True,
            )
        )
    )

    return (
        fused_x,
        fused_z,
        spread,
    )


def measurements_from_components(
    components: list[list[Observation]],
    *,
    start_index: int = 1,
    creation_source: str = "primary",
) -> list[PersonMeasurement]:
    measurements: list[PersonMeasurement] = []

    ordered = sorted(
        components,
        key=lambda group: min(
            item.observation_id
            for item in group
        ),
    )

    for offset, group in enumerate(
        ordered,
    ):
        primary = _primary_observation(
            group
        )

        fused_x, fused_z, spread = _fused_position(
            group
        )

        measurements.append(
            PersonMeasurement(
                measurement_id=start_index + offset,
                observation_ids=tuple(
                    sorted(
                        item.observation_id
                        for item in group
                    )
                ),
                primary_observation_id=primary.observation_id,
                representative_world_x=primary.world_x,
                representative_world_z=primary.world_z,
                fused_world_x=fused_x,
                fused_world_z=fused_z,
                position_spread_m=spread,
                confidence=max(
                    item.confidence
                    for item in group
                ),
                owner_supported=any(
                    bool(item.is_owner)
                    for item in group
                ),
                multi_view=len(group) > 1,
                creation_source=creation_source,
            )
        )

    return measurements


def _edge_sort_key(
    edge: MatchEdge,
) -> tuple[float, float, float, float, float]:
    return (
        edge.score,
        edge.source_iom,
        edge.source_horizontal_iom,
        edge.source_iou,
        -edge.source_center_x_distance_norm,
    )


def _attach_rescues_to_primary_components(
    primary_components: list[list[Observation]],
    rescues: list[Observation],
    matches: list[MatchEdge],
) -> tuple[list[list[Observation]], set[int]]:
    """Grow owner-backed components without ever merging two PRIMARY seeds.

    A RESCUE may attach to exactly one PRIMARY component. Once attached, it can
    help pull another RESCUE into that same component through a strong selected
    match edge. Edges between two already owner-backed components are ignored,
    so a RESCUE can never bridge two distinct players.
    """
    components = [
        list(group)
        for group in primary_components
    ]

    observation_to_component: dict[int, int] = {}

    for component_index, group in enumerate(
        components
    ):
        for observation in group:
            observation_to_component[
                observation.observation_id
            ] = component_index

    rescue_by_id = {
        item.observation_id: item
        for item in rescues
    }

    unassigned = set(
        rescue_by_id
    )

    ordered_edges = sorted(
        matches,
        key=_edge_sort_key,
        reverse=True,
    )

    changed = True

    while changed:
        changed = False

        for edge in ordered_edges:
            a_component = observation_to_component.get(
                edge.observation_a
            )
            b_component = observation_to_component.get(
                edge.observation_b
            )

            # Exactly one side must already belong to an owner-backed
            # component. The other side must be an unassigned RESCUE.
            if (
                a_component is not None
                and b_component is None
                and edge.observation_b in unassigned
            ):
                component_index = a_component
                rescue_id = edge.observation_b

            elif (
                b_component is not None
                and a_component is None
                and edge.observation_a in unassigned
            ):
                component_index = b_component
                rescue_id = edge.observation_a

            else:
                continue

            rescue = rescue_by_id[
                rescue_id
            ]

            existing_views = {
                item.view_name
                for item in components[
                    component_index
                ]
            }

            if rescue.view_name in existing_views:
                continue

            components[
                component_index
            ].append(
                rescue
            )
            observation_to_component[
                rescue_id
            ] = component_index
            unassigned.remove(
                rescue_id
            )
            changed = True

    return components, unassigned


def _rescue_component_has_primary_conflict(
    rescue_group: list[Observation],
    primaries: list[Observation],
) -> bool:
    """Return True when a rescue-only group is source-ambiguous with PRIMARY.

    Ambiguous evidence is deferred instead of creating a second person. This is
    deliberately looser than the normal duplicate matcher because the cost of
    a false promotion here is exactly the duplicate problem we are avoiding.
    """
    for rescue in rescue_group:
        for primary in primaries:
            metrics = source_pair_metrics(
                rescue,
                primary,
            )

            if metrics is None:
                continue

            if (
                metrics.iom
                >= DEFAULT_RESCUE_CONFLICT_MIN_IOM
                and metrics.horizontal_iom
                >= DEFAULT_RESCUE_CONFLICT_MIN_HORIZONTAL_IOM
                and metrics.center_x_distance_norm
                <= DEFAULT_RESCUE_CONFLICT_MAX_CENTER_X_NORM
            ):
                return True

    return False


def _promotable_rescue_components(
    rescues: list[Observation],
    rescue_matches: list[MatchEdge],
    primaries: list[Observation],
) -> tuple[
    list[list[Observation]],
    set[int],
]:
    """Promote only corroborated rescue-only people.

    A single unmatched RESCUE is never allowed to create a new same-frame
    person. It is preserved as deferred evidence for future temporal tracking.

    Two or more RESCUE observations from distinct views may be promoted when
    their own selected source evidence joins them and they are not source-
    ambiguous with any PRIMARY.
    """
    if not rescues:
        return [], set()

    components = merge_match_components(
        rescues,
        rescue_matches,
    )

    promoted: list[list[Observation]] = []
    deferred: set[int] = set()

    for group in components:
        observation_ids = {
            item.observation_id
            for item in group
        }

        if len(group) < 2:
            deferred.update(
                observation_ids
            )
            continue

        if _rescue_component_has_primary_conflict(
            group,
            primaries,
        ):
            deferred.update(
                observation_ids
            )
            continue

        promoted.append(
            group
        )

    return promoted, deferred




def observations_share_identity_evidence(
    a: Observation,
    b: Observation,
    *,
    min_source_iou: float = DEFAULT_MIN_SOURCE_IOU,
    max_source_anchor_norm: float = DEFAULT_MAX_SOURCE_ANCHOR_NORM,
    max_world_distance_m: float = DEFAULT_MAX_WORLD_DISTANCE_M,
    min_source_iom: float = DEFAULT_MIN_SOURCE_IOM,
    min_horizontal_iom: float = DEFAULT_MIN_HORIZONTAL_IOM,
    min_vertical_iom: float = DEFAULT_MIN_VERTICAL_IOM,
    max_center_x_norm: float = DEFAULT_MAX_CENTER_X_NORM,
    max_containment_world_distance_m: float = DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
    max_truncated_world_distance_m: float = DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
) -> bool:
    """Return True when two observations have normal identity-grade evidence.

    Unlike candidate_edge(), this intentionally permits observations from the
    SAME virtual view. That matters for the targeted owner-crop second pass:
    a crop detection must be checked against the already represented detection
    from the full owner image, and both share the same view_name.

    X/Z proximity alone is never sufficient.
    """
    if (
        a.status != "accepted"
        or b.status != "accepted"
    ):
        return False

    source = source_pair_metrics(
        a,
        b,
    )

    if source is None:
        return False

    distance = world_distance(
        a,
        b,
    )

    if distance is None:
        return False

    strict = (
        source.iou >= min_source_iou
        and source.anchor_distance_norm <= max_source_anchor_norm
        and distance <= max_world_distance_m
    )

    containment = (
        source.iom >= min_source_iom
        and source.horizontal_iom >= min_horizontal_iom
        and source.vertical_iom >= min_vertical_iom
        and source.center_x_distance_norm <= max_center_x_norm
        and distance <= max_containment_world_distance_m
    )

    truncated_containment = (
        (a.is_truncated or b.is_truncated)
        and source.iom >= min_source_iom
        and source.horizontal_iom >= min_horizontal_iom
        and source.vertical_iom >= min_vertical_iom
        and source.center_x_distance_norm <= max_center_x_norm
        and distance <= max_truncated_world_distance_m
    )

    return bool(
        strict
        or containment
        or truncated_containment
    )


def observation_matching_measurements(
    observation: Observation,
    association: AssociationResult,
) -> tuple[int, ...]:
    """Return existing persons that match with identity-grade evidence.

    This is intentionally stricter than observation_conflicting_measurements(),
    whose loose gate is useful for ambiguity diagnostics but is too conservative
    to veto the owner-image confirmation pass before that pass even runs.
    """
    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    matches: list[int] = []

    for measurement in association.measurements:
        for represented_id in measurement.observation_ids:
            represented = observation_by_id.get(
                represented_id
            )

            if represented is None:
                continue

            if observations_share_identity_evidence(
                observation,
                represented,
            ):
                matches.append(
                    measurement.measurement_id
                )
                break

    return tuple(
        sorted(
            set(
                matches
            )
        )
    )


def observation_conflicting_measurements(
    observation: Observation,
    association: AssociationResult,
    *,
    min_iom: float = DEFAULT_RESCUE_CONFLICT_MIN_IOM,
    min_horizontal_iom: float = DEFAULT_RESCUE_CONFLICT_MIN_HORIZONTAL_IOM,
    max_center_x_norm: float = DEFAULT_RESCUE_CONFLICT_MAX_CENTER_X_NORM,
) -> tuple[int, ...]:
    """Return existing person measurements source-ambiguous with observation.

    This deliberately uses a looser ORIGINAL-lens overlap gate than normal
    identity matching. It is a safety check for second-pass image rescue:
    before a deferred observation is allowed to create a new person, make sure
    it does not look like evidence belonging to an already represented person.
    """
    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    conflicts: list[int] = []

    for measurement in association.measurements:
        has_conflict = False

        for represented_id in measurement.observation_ids:
            represented = observation_by_id.get(
                represented_id
            )

            if represented is None:
                continue

            metrics = source_pair_metrics(
                observation,
                represented,
            )

            if metrics is None:
                continue

            if (
                metrics.iom >= min_iom
                and metrics.horizontal_iom >= min_horizontal_iom
                and metrics.center_x_distance_norm <= max_center_x_norm
            ):
                has_conflict = True
                break

        if has_conflict:
            conflicts.append(
                measurement.measurement_id
            )

    return tuple(
        sorted(
            conflicts
        )
    )


def promote_image_confirmed_rescues(
    association: AssociationResult,
    observation_ids: list[int] | tuple[int, ...] | set[int],
) -> AssociationResult:
    """Promote explicitly image-confirmed deferred RESCUE observations.

    This function does not decide whether a rescue is valid. The caller must
    have independently confirmed it from the OWNER virtual image. It only
    performs the immutable bookkeeping step.
    """
    requested = {
        int(value)
        for value in observation_ids
    }

    if not requested:
        return association

    deferred = set(
        association.deferred_rescue_observation_ids
    )

    invalid = requested - deferred

    if invalid:
        raise ValueError(
            "Only deferred RESCUE observations may be image-promoted: "
            f"{sorted(invalid)}"
        )

    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    groups: list[list[Observation]] = []

    for observation_id in sorted(
        requested
    ):
        observation = observation_by_id.get(
            observation_id
        )

        if observation is None:
            raise ValueError(
                f"Unknown observation id {observation_id}"
            )

        if (
            observation.status != "accepted"
            or bool(observation.is_owner)
        ):
            raise ValueError(
                "Image-confirmed promotion requires an accepted non-owner "
                f"RESCUE observation, got {observation_id}"
            )

        groups.append(
            [observation]
        )

    promoted = measurements_from_components(
        groups,
        start_index=len(association.measurements) + 1,
        creation_source="image_confirmed_rescue",
    )

    return AssociationResult(
        observations=association.observations,
        matches=association.matches,
        measurements=tuple(
            [
                *association.measurements,
                *promoted,
            ]
        ),
        deferred_rescue_observation_ids=tuple(
            sorted(
                deferred - requested
            )
        ),
    )


def associate_observations(
    observations: list[Observation],
    layout: AsymmetricOwnershipLayout,
    *,
    min_source_iou: float = DEFAULT_MIN_SOURCE_IOU,
    max_source_anchor_norm: float = DEFAULT_MAX_SOURCE_ANCHOR_NORM,
    max_world_distance_m: float = DEFAULT_MAX_WORLD_DISTANCE_M,
    min_source_iom: float = DEFAULT_MIN_SOURCE_IOM,
    min_horizontal_iom: float = DEFAULT_MIN_HORIZONTAL_IOM,
    min_vertical_iom: float = DEFAULT_MIN_VERTICAL_IOM,
    max_center_x_norm: float = DEFAULT_MAX_CENTER_X_NORM,
    max_containment_world_distance_m: float = DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
    max_truncated_world_distance_m: float = DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
) -> AssociationResult:
    """Consolidate one source frame using exclusive PRIMARY/RESCUE ownership.

    Policy:
    1. Accepted observations whose projected footpoint belongs to their own
       logical camera cell are PRIMARY and may seed a person.
    2. Accepted observations seen through overlap but owned by another camera
       are RESCUE and cannot seed a person by themselves.
    3. Strong source-pixel matches attach RESCUE observations to PRIMARY seeds.
       A RESCUE can never bridge two PRIMARY components.
    4. Rescue-only evidence is promoted only when at least two distinct virtual
       views corroborate it and it is not source-ambiguous with any PRIMARY.
    5. Singleton / ambiguous RESCUE observations are preserved as deferred raw
       evidence for future temporal tracking, not emitted as new people.

    This removes overlap duplicates structurally instead of forcing the frame to
    a target population such as 23.
    """
    accepted = [
        item
        for item in observations
        if item.status == "accepted"
    ]

    primaries = [
        item
        for item in accepted
        if bool(item.is_owner)
    ]

    rescues = [
        item
        for item in accepted
        if not bool(item.is_owner)
    ]

    by_view: dict[str, list[Observation]] = {}

    for item in accepted:
        by_view.setdefault(
            item.view_name,
            [],
        ).append(item)

    matches: list[MatchEdge] = []

    for view_a, view_b in candidate_view_pairs(
        layout
    ):
        matches.extend(
            one_to_one_matches_for_view_pair(
                by_view.get(view_a, []),
                by_view.get(view_b, []),
                min_source_iou=min_source_iou,
                max_source_anchor_norm=max_source_anchor_norm,
                max_world_distance_m=max_world_distance_m,
                min_source_iom=min_source_iom,
                min_horizontal_iom=min_horizontal_iom,
                min_vertical_iom=min_vertical_iom,
                max_center_x_norm=max_center_x_norm,
                max_containment_world_distance_m=max_containment_world_distance_m,
                max_truncated_world_distance_m=max_truncated_world_distance_m,
            )
        )

    primary_ids = {
        item.observation_id
        for item in primaries
    }
    rescue_ids = {
        item.observation_id
        for item in rescues
    }

    primary_primary_matches = [
        edge
        for edge in matches
        if (
            edge.observation_a in primary_ids
            and edge.observation_b in primary_ids
        )
    ]

    primary_components = merge_match_components(
        primaries,
        primary_primary_matches,
    )

    owner_backed_components, unassigned_rescue_ids = (
        _attach_rescues_to_primary_components(
            primary_components,
            rescues,
            matches,
        )
    )

    unassigned_rescues = [
        item
        for item in rescues
        if item.observation_id
        in unassigned_rescue_ids
    ]

    unassigned_id_set = {
        item.observation_id
        for item in unassigned_rescues
    }

    rescue_rescue_matches = [
        edge
        for edge in matches
        if (
            edge.observation_a
            in unassigned_id_set
            and edge.observation_b
            in unassigned_id_set
        )
    ]

    (
        promoted_rescue_components,
        deferred_rescue_ids,
    ) = _promotable_rescue_components(
        unassigned_rescues,
        rescue_rescue_matches,
        primaries,
    )

    owner_measurements = measurements_from_components(
        owner_backed_components,
        start_index=1,
        creation_source="primary",
    )

    promoted_measurements = measurements_from_components(
        promoted_rescue_components,
        start_index=len(owner_measurements) + 1,
        creation_source="promoted_rescue",
    )

    # Any unassigned RESCUE not present in a promoted group must remain
    # deferred. This also catches rescue groups rejected by ambiguity.
    promoted_ids = {
        item.observation_id
        for group in promoted_rescue_components
        for item in group
    }

    deferred_rescue_ids.update(
        unassigned_id_set
        - promoted_ids
    )

    measurements = [
        *owner_measurements,
        *promoted_measurements,
    ]

    return AssociationResult(
        observations=tuple(observations),
        matches=tuple(matches),
        measurements=tuple(measurements),
        deferred_rescue_observation_ids=tuple(
            sorted(
                deferred_rescue_ids
            )
        ),
    )

