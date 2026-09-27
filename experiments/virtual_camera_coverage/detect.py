#!/usr/bin/env python3
"""Run RF-DETR 2XL C2 on six TIGHT cameras with PRIMARY/RESCUE and owner-image confirmation.

CHATGPT DELIVERY MARKER: DETECT_COVERAGE_V14

Pipeline for ONE source frame:

    original stacked Veo frame
        -> 4 FAR + 2 NEAR TIGHT virtual cameras
        -> RF-DETR 2XL C2 independently per camera
        -> raw Observation per bbox
        -> bbox footpoint -> field X/Z
        -> bbox footprint -> ORIGINAL Veo lens coordinates
        -> conservative one-to-one cross-view association
        -> unique per-frame PersonMeasurement objects

This is still NOT tracking.

The goal is to validate the detection/position layer first:
- preserve useful render overlap,
- collapse the same physical player seen in adjacent zooms,
- do not merge two nearby real players merely because their X/Z positions are close,
- retain all raw observations and provenance for later tracking.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.virtual_camera_coverage.association import (  # noqa: E402
    DEFAULT_FIELD_MARGIN_M,
    DEFAULT_MAX_CENTER_X_NORM,
    DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
    DEFAULT_MAX_SOURCE_ANCHOR_NORM,
    DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
    DEFAULT_MAX_WORLD_DISTANCE_M,
    DEFAULT_MIN_HORIZONTAL_IOM,
    DEFAULT_MIN_SOURCE_IOM,
    DEFAULT_MIN_SOURCE_IOU,
    DEFAULT_MIN_VERTICAL_IOM,
    AssociationResult,
    Observation,
    associate_observations,
    build_observation,
    observation_conflicting_measurements,
    observation_matching_measurements,
    observations_share_identity_evidence,
    promote_image_confirmed_rescues,
)
from experiments.virtual_camera_coverage.coverage import (  # noqa: E402
    DEFAULT_FAR_CAMERAS,
    DEFAULT_GRID_STEP_M,
    DEFAULT_GUARD_M,
    DEFAULT_NEAR_CAMERAS,
    DEFAULT_SPLIT_FRACTION,
    build_map,
    derive_specs,
    player_top_y,
    read_frame,
    render_map,
    world_points_to_virtual_pixels,
)
from experiments.virtual_camera_coverage.ultra_panorama_overlay import (  # noqa: E402
    draw_panorama_detections,
    render_ultra_panorama,
    virtual_bbox_to_panorama_detection,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.models.rfdetr_2xl import RFDETR2XLBackend  # noqa: E402


DEFAULT_C2_WEIGHTS = (
    ROOT
    / "person_detect"
    / "model_weights"
    / "rfdetr_finetuned"
    / "veomatch_person_v1"
    / "c2_football_images_plus_veo"
    / "best.pth"
)

DEFAULT_THRESHOLD = 0.30

# Targeted second-pass confirmation for deferred RESCUE observations.
DEFAULT_RESCUE_CONFIRM_THRESHOLD = 0.30
DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M = 2.50
DEFAULT_RESCUE_CONFIRM_MIN_EXPECTED_HEIGHT_PX = 5.0
DEFAULT_RESCUE_CONFIRM_MIN_CROP_WIDTH_PX = 240
DEFAULT_RESCUE_CONFIRM_MIN_CROP_HEIGHT_PX = 220
DEFAULT_RESCUE_CONFIRM_X_HEIGHT_SCALE = 2.75
DEFAULT_RESCUE_CONFIRM_Y_HEIGHT_SCALE = 1.00

DEFAULT_OUT = Path(
    "experiments/virtual_camera_coverage/association_output"
)


@dataclass(frozen=True)
class PersonDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float

    @property
    def width_px(self) -> float:
        return self.xyxy[2] - self.xyxy[0]

    @property
    def height_px(self) -> float:
        return self.xyxy[3] - self.xyxy[1]

    @property
    def area_px2(self) -> float:
        return self.width_px * self.height_px

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            {
                "width_px": self.width_px,
                "height_px": self.height_px,
                "area_px2": self.area_px2,
            }
        )
        return result


@dataclass(frozen=True)
class RescueImageConfirmation:
    rescue_observation_id: int
    owner_view_name: str | None
    status: str
    confirmed: bool

    source_conflicting_measurement_ids: tuple[int, ...]

    expected_owner_foot_px: tuple[float, float] | None
    expected_owner_top_px: tuple[float, float] | None
    expected_height_px: float | None

    crop_xyxy: tuple[int, int, int, int] | None
    crop_detection_count: int

    candidate_full_xyxy: tuple[float, float, float, float] | None
    candidate_confidence: float | None
    candidate_world_x: float | None
    candidate_world_z: float | None
    candidate_world_distance_m: float | None
    candidate_is_owner: bool | None
    candidate_conflicting_measurement_ids: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def predict_persons(
    backend: RFDETR2XLBackend,
    image_bgr: np.ndarray,
    *,
    threshold: float,
) -> list[PersonDetection]:
    """Run one image through the repository's existing RF-DETR 2XL backend."""
    if image_bgr is None or image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
        raise ValueError(
            "image_bgr must be a valid HxWx3 BGR image"
        )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1"
        )

    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB,
    )

    raw = backend._predict_one(  # noqa: SLF001 - isolated experiment
        image_rgb,
        threshold=threshold,
    )

    names = backend._class_names(  # noqa: SLF001 - isolated experiment
        raw
    )

    xyxy = np.asarray(
        raw.xyxy,
        dtype=np.float64,
    )

    confidence = np.asarray(
        raw.confidence,
        dtype=np.float64,
    )

    if (
        len(names) != len(xyxy)
        or len(confidence) != len(xyxy)
    ):
        raise RuntimeError(
            "Inconsistent RF-DETR output lengths"
        )

    persons: list[PersonDetection] = []

    for box, score, class_name in zip(
        xyxy,
        confidence,
        names,
        strict=True,
    ):
        if str(class_name).lower() != "person":
            continue

        values = [
            float(value)
            for value in box
        ]
        score_value = float(score)

        if not np.isfinite(
            [*values, score_value]
        ).all():
            continue

        x1, y1, x2, y2 = values

        if x2 <= x1 or y2 <= y1:
            continue

        persons.append(
            PersonDetection(
                xyxy=(x1, y1, x2, y2),
                confidence=score_value,
            )
        )

    return persons


def detection_metrics(
    detections: list[PersonDetection],
) -> dict[str, float | int | None]:
    if not detections:
        return {
            "count": 0,
            "mean_confidence": None,
            "median_confidence": None,
            "mean_bbox_width_px": None,
            "median_bbox_width_px": None,
            "mean_bbox_height_px": None,
            "median_bbox_height_px": None,
            "mean_bbox_area_px2": None,
            "median_bbox_area_px2": None,
        }

    confidences = np.asarray(
        [item.confidence for item in detections],
        dtype=np.float64,
    )
    widths = np.asarray(
        [item.width_px for item in detections],
        dtype=np.float64,
    )
    heights = np.asarray(
        [item.height_px for item in detections],
        dtype=np.float64,
    )
    areas = np.asarray(
        [item.area_px2 for item in detections],
        dtype=np.float64,
    )

    return {
        "count": len(detections),
        "mean_confidence": float(
            confidences.mean()
        ),
        "median_confidence": float(
            np.median(confidences)
        ),
        "mean_bbox_width_px": float(
            widths.mean()
        ),
        "median_bbox_width_px": float(
            np.median(widths)
        ),
        "mean_bbox_height_px": float(
            heights.mean()
        ),
        "median_bbox_height_px": float(
            np.median(heights)
        ),
        "mean_bbox_area_px2": float(
            areas.mean()
        ),
        "median_bbox_area_px2": float(
            np.median(areas)
        ),
    }


def ordered_tight_specs(
    specs,
) -> list:
    """Return FAR left-to-right first, then NEAR left-to-right."""
    tight = [
        spec
        for spec in specs
        if spec.mode == "tight"
    ]

    far = sorted(
        [
            spec
            for spec in tight
            if spec.cell.band == "far"
        ],
        key=lambda spec: spec.cell.index_in_band,
    )

    near = sorted(
        [
            spec
            for spec in tight
            if spec.cell.band == "near"
        ],
        key=lambda spec: spec.cell.index_in_band,
    )

    if (
        len(far) != DEFAULT_FAR_CAMERAS
        or len(near) != DEFAULT_NEAR_CAMERAS
    ):
        raise RuntimeError(
            "Expected exactly 4 FAR and 2 NEAR TIGHT cameras, got "
            f"{len(far)} FAR + {len(near)} NEAR"
        )

    return [
        *far,
        *near,
    ]


def _measurement_lookup(
    association: AssociationResult,
) -> tuple[
    dict[int, int],
    set[int],
]:
    observation_to_measurement: dict[int, int] = {}
    primary_ids: set[int] = set()

    for measurement in association.measurements:
        primary_ids.add(
            measurement.primary_observation_id
        )

        for observation_id in measurement.observation_ids:
            observation_to_measurement[
                observation_id
            ] = measurement.measurement_id

    return observation_to_measurement, primary_ids



def draw_unique_representatives(
    image: np.ndarray,
    *,
    view_name: str,
    association: AssociationResult,
    numbered: bool = False,
) -> np.ndarray:
    """Draw exactly one bbox per final unique person, in one neutral style.

    This is diagnostic-only. No PRIMARY/RESCUE/deferred semantics are shown.
    Each final PersonMeasurement is represented by its single
    primary_observation_id, so the complete six-view sheet contains exactly
    len(association.measurements) boxes in total.

    WHITE means only: this final unique person exists.
    """
    output = image.copy()

    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    for measurement in association.measurements:
        observation = observation_by_id.get(
            measurement.primary_observation_id
        )

        if (
            observation is None
            or observation.view_name != view_name
        ):
            continue

        x1, y1, x2, y2 = observation.xyxy

        p1 = (
            int(round(x1)),
            int(round(y1)),
        )
        p2 = (
            int(round(x2)),
            int(round(y2)),
        )

        # One neutral color only. No semantic color coding.
        color = (255, 255, 255)

        cv2.rectangle(
            output,
            p1,
            p2,
            color,
            3,
            cv2.LINE_AA,
        )

        if numbered:
            label = str(
                measurement.measurement_id
            )

            label_x = max(
                0,
                int(round(x1)),
            )
            label_y = max(
                20,
                int(round(y1)) - 6,
            )

            # Small dark backing only for readability; the player marking
            # itself remains a single neutral white style.
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                2,
            )

            cv2.rectangle(
                output,
                (
                    label_x - 2,
                    label_y - text_h - 3,
                ),
                (
                    label_x + text_w + 3,
                    label_y + baseline + 2,
                ),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                output,
                label,
                (label_x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

    return output


def _clean_unique_panel(
    image: np.ndarray,
    *,
    name: str,
    panel_width: int,
) -> np.ndarray:
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

    # Camera name only: no raw/PRIMARY/RESCUE counters.
    cv2.rectangle(
        panel,
        (0, 0),
        (panel.shape[1], 34),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        panel,
        name,
        (8, 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return panel


def clean_unique_contact_sheet(
    unique_rendered: dict[str, np.ndarray],
    association: AssociationResult,
    *,
    panel_width: int,
    frame_number: int,
    time_s: float,
    numbered: bool,
) -> np.ndarray:
    """Six-camera contact sheet showing only final unique-person boxes."""
    far_names = [
        f"tight_far_c{i}"
        for i in range(1, 5)
    ]
    near_names = [
        f"tight_near_c{i}"
        for i in range(1, 3)
    ]

    full_width = 4 * panel_width

    far_panels = [
        _clean_unique_panel(
            unique_rendered[name],
            name=name,
            panel_width=panel_width,
        )
        for name in far_names
    ]

    near_panels = [
        _clean_unique_panel(
            unique_rendered[name],
            name=name,
            panel_width=panel_width,
        )
        for name in near_names
    ]

    far_row = np.concatenate(
        far_panels,
        axis=1,
    )

    near_content = np.concatenate(
        near_panels,
        axis=1,
    )

    near_row = np.zeros(
        (
            near_content.shape[0],
            full_width,
            3,
        ),
        dtype=np.uint8,
    )

    near_x0 = (
        full_width
        - near_content.shape[1]
    ) // 2

    near_row[
        :,
        near_x0:near_x0 + near_content.shape[1],
    ] = near_content

    title = np.zeros(
        (52, full_width, 3),
        dtype=np.uint8,
    )

    mode_text = (
        "numbered"
        if numbered
        else "boxes only"
    )

    cv2.putText(
        title,
        (
            f"FINAL UNIQUE PERSONS ONLY | unique={len(association.measurements)} "
            f"| {mode_text} | frame={frame_number} | t={time_s:.2f}s"
        ),
        (12, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    far_title = np.zeros(
        (32, full_width, 3),
        dtype=np.uint8,
    )
    cv2.putText(
        far_title,
        "FAR",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    near_title = np.zeros(
        (32, full_width, 3),
        dtype=np.uint8,
    )
    cv2.putText(
        near_title,
        "NEAR",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return np.concatenate(
        [
            title,
            far_title,
            far_row,
            near_title,
            near_row,
        ],
        axis=0,
    )


def draw_associated_detections(
    image: np.ndarray,
    view_observations: list[Observation],
    association: AssociationResult,
) -> np.ndarray:
    """Draw PRIMARY, attached RESCUE, promoted RESCUE and deferred evidence."""
    output = image.copy()

    observation_to_measurement, representative_ids = _measurement_lookup(
        association
    )

    deferred_ids = set(
        association.deferred_rescue_observation_ids
    )

    measurement_by_id = {
        item.measurement_id: item
        for item in association.measurements
    }

    for observation in view_observations:
        x1, y1, x2, y2 = observation.xyxy

        p1 = (
            int(round(x1)),
            int(round(y1)),
        )
        p2 = (
            int(round(x2)),
            int(round(y2)),
        )

        measurement_id = observation_to_measurement.get(
            observation.observation_id
        )

        if observation.status != "accepted":
            color = (0, 0, 255)
            label = (
                f"OBS{observation.observation_id} "
                f"{observation.status}"
            )

        elif observation.observation_id in deferred_ids:
            # Deferred RESCUE remains available for future temporal tracking,
            # but does not create a same-frame person.
            color = (255, 0, 255)
            owner = (
                observation.owner_cell
                if observation.owner_cell is not None
                else "?"
            )
            label = (
                f"R-DEFER O{observation.observation_id} "
                f"->{owner} c={observation.confidence:.2f}"
            )

        elif measurement_id is None:
            color = (255, 0, 255)
            label = (
                f"OBS{observation.observation_id} unassigned"
            )

        else:
            measurement = measurement_by_id[
                measurement_id
            ]

            is_representative = (
                observation.observation_id
                in representative_ids
            )

            if measurement.creation_source == "image_confirmed_rescue":
                color = (255, 255, 0)
                role = "R-IMG"

            elif measurement.creation_source == "promoted_rescue":
                color = (255, 255, 0)
                role = (
                    "R-PROM"
                    if is_representative
                    else "R-CORR"
                )

            elif observation.is_owner:
                color = (0, 255, 0)
                role = (
                    "PRIMARY"
                    if is_representative
                    else "P-EVID"
                )

            else:
                color = (0, 190, 255)
                role = "RESCUE"

            trunc_flag = (
                " clip"
                if observation.is_truncated
                else ""
            )

            label = (
                f"P{measurement_id:02d} {role} "
                f"{observation.confidence:.2f}"
                f"{trunc_flag}"
            )

        box_thickness = (
            4
            if observation.observation_id in deferred_ids
            else 2
        )

        cv2.rectangle(
            output,
            p1,
            p2,
            color,
            box_thickness,
            cv2.LINE_AA,
        )

        foot = (
            int(round(observation.foot_x)),
            int(round(observation.foot_y)),
        )

        cv2.circle(
            output,
            foot,
            4,
            color,
            -1,
            cv2.LINE_AA,
        )

        label_y = max(
            18,
            int(round(y1)) - 5,
        )

        cv2.putText(
            output,
            label,
            (int(round(x1)), label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            color,
            1,
            cv2.LINE_AA,
        )

    return output


def pitch_to_pixel(
    world_x: float,
    world_z: float,
    *,
    field_length: float,
    field_width: float,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    u = (
        world_x + field_length / 2.0
    ) / field_length

    v = (
        field_width / 2.0 - world_z
    ) / field_width

    return (
        x0 + int(round(u * width)),
        y0 + int(round(v * height)),
    )


def draw_unique_radar(
    metadata,
    association: AssociationResult,
    *,
    width: int = 1280,
    height: int = 520,
) -> np.ndarray:
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    field_length = float(
        metadata.field_length
    )
    field_width = float(
        metadata.field_width
    )

    margin = 48
    available_w = width - 2 * margin
    available_h = height - 2 * margin - 36

    pitch_w = available_w
    pitch_h = int(
        round(
            pitch_w
            * field_width
            / field_length
        )
    )

    if pitch_h > available_h:
        pitch_h = available_h
        pitch_w = int(
            round(
                pitch_h
                * field_length
                / field_width
            )
        )

    x0 = (width - pitch_w) // 2
    y0 = 50 + (
        available_h - pitch_h
    ) // 2

    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + pitch_w, y0 + pitch_h),
        (35, 90, 35),
        -1,
    )

    line = (235, 235, 235)

    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + pitch_w, y0 + pitch_h),
        line,
        2,
        cv2.LINE_AA,
    )

    mid_x = x0 + pitch_w // 2

    cv2.line(
        canvas,
        (mid_x, y0),
        (mid_x, y0 + pitch_h),
        line,
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        (
            f"UNIQUE={len(association.measurements)}  "
            f"OWNER-BACKED={association.owner_backed_measurement_count}  "
            f"PROMOTED-RESCUE={association.promoted_rescue_measurement_count}  "
            f"IMG-CONF={association.image_confirmed_rescue_measurement_count}  "
            f"DEFERRED-RESCUE={len(association.deferred_rescue_observation_ids)}"
        ),
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for measurement in association.measurements:
        world_x = (
            measurement.fused_world_x
            if measurement.fused_world_x is not None
            else measurement.representative_world_x
        )
        world_z = (
            measurement.fused_world_z
            if measurement.fused_world_z is not None
            else measurement.representative_world_z
        )

        if world_x is None or world_z is None:
            continue

        px, py = pitch_to_pixel(
            world_x,
            world_z,
            field_length=field_length,
            field_width=field_width,
            x0=x0,
            y0=y0,
            width=pitch_w,
            height=pitch_h,
        )

        cv2.circle(
            canvas,
            (px, py),
            8,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )

        cv2.circle(
            canvas,
            (px, py),
            9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"P{measurement.measurement_id:02d}",
            (px + 10, py - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    for observation_id in association.deferred_rescue_observation_ids:
        observation = observation_by_id.get(
            observation_id
        )

        if (
            observation is None
            or observation.world_x is None
            or observation.world_z is None
        ):
            continue

        px, py = pitch_to_pixel(
            observation.world_x,
            observation.world_z,
            field_length=field_length,
            field_width=field_width,
            x0=x0,
            y0=y0,
            width=pitch_w,
            height=pitch_h,
        )

        # Magenta X: evidence exists, but it has not been promoted to a person.
        cv2.line(
            canvas,
            (px - 8, py - 8),
            (px + 8, py + 8),
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )
        cv2.line(
            canvas,
            (px - 8, py + 8),
            (px + 8, py - 8),
            (255, 0, 255),
            3,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            f"R{observation_id}",
            (px + 10, py + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _label_panel(
    panel: np.ndarray,
    *,
    name: str,
    raw_count: int,
    accepted_count: int,
) -> np.ndarray:
    output = panel.copy()

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 43),
        (0, 0, 0),
        -1,
    )

    text = (
        f"{name}  raw {raw_count}  "
        f"field {accepted_count}"
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


def _contact_row(
    names: list[str],
    annotated: dict[str, np.ndarray],
    observations_by_view: dict[str, list[Observation]],
    raw_counts: dict[str, int],
    *,
    panel_width: int,
    total_columns: int,
) -> np.ndarray:
    panels: list[np.ndarray] = []

    for name in names:
        image = annotated[name]

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

        accepted_count = sum(
            item.status == "accepted"
            for item in observations_by_view.get(
                name,
                [],
            )
        )

        panel = _label_panel(
            panel,
            name=name,
            raw_count=raw_counts.get(
                name,
                0,
            ),
            accepted_count=accepted_count,
        )

        panels.append(panel)

    content = np.concatenate(
        panels,
        axis=1,
    )

    canvas = np.zeros(
        (
            content.shape[0],
            total_columns * panel_width,
            3,
        ),
        dtype=np.uint8,
    )

    x0 = (
        canvas.shape[1]
        - content.shape[1]
    ) // 2

    canvas[
        :,
        x0:x0 + content.shape[1],
    ] = content

    return canvas


def contact_sheet(
    annotated: dict[str, np.ndarray],
    observations_by_view: dict[str, list[Observation]],
    raw_counts: dict[str, int],
    association: AssociationResult,
    *,
    panel_width: int,
    threshold: float,
    frame_number: int,
    time_s: float,
) -> np.ndarray:
    far_names = [
        f"tight_far_c{i}"
        for i in range(1, 5)
    ]
    near_names = [
        f"tight_near_c{i}"
        for i in range(1, 3)
    ]

    full_width = 4 * panel_width

    accepted_count = sum(
        item.status == "accepted"
        for item in association.observations
    )

    collapsed = (
        accepted_count
        - len(association.measurements)
        - len(
            association.deferred_rescue_observation_ids
        )
    )

    multi_view = sum(
        item.multi_view
        for item in association.measurements
    )

    title = np.zeros(
        (68, full_width, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        title,
        (
            "RF-DETR 2XL C2 | PRIMARY/RESCUE same-frame association | "
            f"thr={threshold:.2f} | frame={frame_number} | t={time_s:.2f}s"
        ),
        (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.64,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        title,
        (
            f"primary_obs={association.primary_observation_count}  "
            f"rescue_obs={association.rescue_observation_count}  "
            f"unique={len(association.measurements)}  "
            f"deferred_rescue={len(association.deferred_rescue_observation_ids)}  "
            f"multi-view={multi_view}"
        ),
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    far_title = np.zeros(
        (36, full_width, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        far_title,
        "FAR - priority zone",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    near_title = np.zeros(
        (36, full_width, 3),
        dtype=np.uint8,
    )

    cv2.putText(
        near_title,
        "NEAR - easier zone",
        (12, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    far_row = _contact_row(
        far_names,
        annotated,
        observations_by_view,
        raw_counts,
        panel_width=panel_width,
        total_columns=4,
    )

    near_row = _contact_row(
        near_names,
        annotated,
        observations_by_view,
        raw_counts,
        panel_width=panel_width,
        total_columns=4,
    )

    return np.concatenate(
        [
            title,
            far_title,
            far_row,
            near_title,
            near_row,
        ],
        axis=0,
    )


def _clamped_crop_bounds(
    image: np.ndarray,
    *,
    center_x: float,
    top_y: float,
    foot_y: float,
    expected_height_px: float,
) -> tuple[int, int, int, int] | None:
    height, width = image.shape[:2]

    half_width = max(
        DEFAULT_RESCUE_CONFIRM_MIN_CROP_WIDTH_PX / 2.0,
        expected_height_px * DEFAULT_RESCUE_CONFIRM_X_HEIGHT_SCALE,
    )

    vertical_extra = max(
        60.0,
        expected_height_px * DEFAULT_RESCUE_CONFIRM_Y_HEIGHT_SCALE,
    )

    x1 = int(
        math.floor(
            center_x - half_width
        )
    )
    x2 = int(
        math.ceil(
            center_x + half_width
        )
    )

    y1 = int(
        math.floor(
            min(top_y, foot_y) - vertical_extra
        )
    )
    y2 = int(
        math.ceil(
            max(top_y, foot_y) + vertical_extra
        )
    )

    # Guarantee a useful minimum vertical crop size.
    current_h = y2 - y1
    if current_h < DEFAULT_RESCUE_CONFIRM_MIN_CROP_HEIGHT_PX:
        extra = (
            DEFAULT_RESCUE_CONFIRM_MIN_CROP_HEIGHT_PX
            - current_h
        ) / 2.0
        y1 = int(
            math.floor(
                y1 - extra
            )
        )
        y2 = int(
            math.ceil(
                y2 + extra
            )
        )

    x1 = max(
        0,
        x1,
    )
    y1 = max(
        0,
        y1,
    )
    x2 = min(
        width,
        x2,
    )
    y2 = min(
        height,
        y2,
    )

    if x2 - x1 < 32 or y2 - y1 < 32:
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


def _owner_expected_pixels(
    metadata,
    spec,
    observation: Observation,
    *,
    width: int,
    height: int,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    float,
] | None:
    if (
        observation.world_x is None
        or observation.world_z is None
    ):
        return None

    world_points = np.asarray(
        [
            [
                observation.world_x,
                0.0,
                observation.world_z,
            ],
            [
                observation.world_x,
                player_top_y(metadata),
                observation.world_z,
            ],
        ],
        dtype=np.float64,
    )

    pixels = world_points_to_virtual_pixels(
        metadata,
        spec,
        world_points,
        width,
        height,
    )

    if (
        pixels.shape != (2, 2)
        or not np.isfinite(pixels).all()
    ):
        return None

    foot = (
        float(pixels[0, 0]),
        float(pixels[0, 1]),
    )
    top = (
        float(pixels[1, 0]),
        float(pixels[1, 1]),
    )

    expected_height = abs(
        foot[1] - top[1]
    )

    if (
        not math.isfinite(expected_height)
        or expected_height < DEFAULT_RESCUE_CONFIRM_MIN_EXPECTED_HEIGHT_PX
    ):
        return None

    return (
        foot,
        top,
        expected_height,
    )


def _candidate_full_box(
    detection: PersonDetection,
    crop_xyxy: tuple[int, int, int, int],
) -> tuple[float, float, float, float]:
    crop_x1, crop_y1, _, _ = crop_xyxy
    x1, y1, x2, y2 = detection.xyxy

    return (
        x1 + crop_x1,
        y1 + crop_y1,
        x2 + crop_x1,
        y2 + crop_y1,
    )


def confirm_deferred_rescues_from_owner_images(
    *,
    backend: RFDETR2XLBackend,
    metadata,
    layout,
    rendered: dict[str, np.ndarray],
    maps,
    spec_by_name,
    association: AssociationResult,
    frame_number: int,
    time_s: float,
    field_margin_m: float,
    threshold: float,
    max_world_distance_m: float,
) -> tuple[
    AssociationResult,
    list[RescueImageConfirmation],
    dict[int, dict[str, Any]],
]:
    """Try to recover deferred RESCUE observations using the OWNER image only.

    This is still a same-frame, image-only procedure:
    - use the deferred rescue X/Z only to locate the expected player in the
      owner virtual camera;
    - crop that owner image around the expected position;
    - rerun RF-DETR on the crop, which effectively gives the tiny/far player
      more pixels at the detector input;
    - accept only a candidate that projects back into the same owner cell,
      is geometrically close to the rescue, and is not source-ambiguous with an
      already represented person.

    No player-count prior and no temporal tracking are used.
    """
    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    confirmations: list[RescueImageConfirmation] = []
    debug_payload: dict[int, dict[str, Any]] = {}
    confirmed_ids: list[int] = []

    synthetic_observation_id = (
        max(
            observation_by_id,
            default=0,
        )
        + 1_000_000
    )

    for rescue_id in association.deferred_rescue_observation_ids:
        rescue = observation_by_id.get(
            rescue_id
        )

        if rescue is None:
            continue

        existing_conflicts = observation_conflicting_measurements(
            rescue,
            association,
        )

        owner_view_name = (
            f"tight_{rescue.owner_cell}"
            if rescue.owner_cell is not None
            else None
        )

        base_kwargs = {
            "rescue_observation_id": rescue_id,
            "owner_view_name": owner_view_name,
            "source_conflicting_measurement_ids": existing_conflicts,
        }

        # A visibly clipped rescue is useful evidence but is a poor seed for
        # creating a brand-new person. Keep it deferred.
        if rescue.is_truncated:
            confirmations.append(
                RescueImageConfirmation(
                    **base_kwargs,
                    status="rescue_truncated",
                    confirmed=False,
                    expected_owner_foot_px=None,
                    expected_owner_top_px=None,
                    expected_height_px=None,
                    crop_xyxy=None,
                    crop_detection_count=0,
                    candidate_full_xyxy=None,
                    candidate_confidence=None,
                    candidate_world_x=None,
                    candidate_world_z=None,
                    candidate_world_distance_m=None,
                    candidate_is_owner=None,
                    candidate_conflicting_measurement_ids=(),
                )
            )
            continue

        if (
            owner_view_name is None
            or owner_view_name not in rendered
            or owner_view_name not in spec_by_name
            or owner_view_name not in maps
        ):
            confirmations.append(
                RescueImageConfirmation(
                    **base_kwargs,
                    status="owner_view_unavailable",
                    confirmed=False,
                    expected_owner_foot_px=None,
                    expected_owner_top_px=None,
                    expected_height_px=None,
                    crop_xyxy=None,
                    crop_detection_count=0,
                    candidate_full_xyxy=None,
                    candidate_confidence=None,
                    candidate_world_x=None,
                    candidate_world_z=None,
                    candidate_world_distance_m=None,
                    candidate_is_owner=None,
                    candidate_conflicting_measurement_ids=(),
                )
            )
            continue

        owner_image = rendered[
            owner_view_name
        ]
        owner_spec = spec_by_name[
            owner_view_name
        ]
        owner_map = maps[
            owner_view_name
        ]

        expected = _owner_expected_pixels(
            metadata,
            owner_spec,
            rescue,
            width=owner_image.shape[1],
            height=owner_image.shape[0],
        )

        if expected is None:
            confirmations.append(
                RescueImageConfirmation(
                    **base_kwargs,
                    status="owner_projection_invalid",
                    confirmed=False,
                    expected_owner_foot_px=None,
                    expected_owner_top_px=None,
                    expected_height_px=None,
                    crop_xyxy=None,
                    crop_detection_count=0,
                    candidate_full_xyxy=None,
                    candidate_confidence=None,
                    candidate_world_x=None,
                    candidate_world_z=None,
                    candidate_world_distance_m=None,
                    candidate_is_owner=None,
                    candidate_conflicting_measurement_ids=(),
                )
            )
            continue

        expected_foot, expected_top, expected_height = expected

        crop_xyxy = _clamped_crop_bounds(
            owner_image,
            center_x=expected_foot[0],
            top_y=expected_top[1],
            foot_y=expected_foot[1],
            expected_height_px=expected_height,
        )

        if crop_xyxy is None:
            confirmations.append(
                RescueImageConfirmation(
                    **base_kwargs,
                    status="owner_crop_invalid",
                    confirmed=False,
                    expected_owner_foot_px=expected_foot,
                    expected_owner_top_px=expected_top,
                    expected_height_px=expected_height,
                    crop_xyxy=None,
                    crop_detection_count=0,
                    candidate_full_xyxy=None,
                    candidate_confidence=None,
                    candidate_world_x=None,
                    candidate_world_z=None,
                    candidate_world_distance_m=None,
                    candidate_is_owner=None,
                    candidate_conflicting_measurement_ids=(),
                )
            )
            continue

        crop_x1, crop_y1, crop_x2, crop_y2 = crop_xyxy
        crop = owner_image[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        detections = predict_persons(
            backend,
            crop,
            threshold=threshold,
        )

        candidates: list[
            tuple[
                float,
                float,
                Observation,
                tuple[float, float, float, float],
                tuple[int, ...],
            ]
        ] = []

        candidate_records: list[dict[str, Any]] = []

        for candidate_index, detection in enumerate(
            detections
        ):
            full_box = _candidate_full_box(
                detection,
                crop_xyxy,
            )

            candidate = build_observation(
                observation_id=(
                    synthetic_observation_id
                    + candidate_index
                ),
                frame_number=frame_number,
                time_s=time_s,
                metadata=metadata,
                layout=layout,
                spec=owner_spec,
                camera_map=owner_map,
                xyxy=full_box,
                confidence=detection.confidence,
                field_margin_m=field_margin_m,
            )

            if (
                candidate.world_x is None
                or candidate.world_z is None
                or rescue.world_x is None
                or rescue.world_z is None
            ):
                distance = math.inf
            else:
                distance = math.hypot(
                    candidate.world_x - rescue.world_x,
                    candidate.world_z - rescue.world_z,
                )

            # The owner-crop detection must itself match the RESCUE with
            # normal identity-grade ORIGINAL-lens evidence. X/Z is only an
            # additional safety gate; it never creates identity by itself.
            matches_rescue_identity = observations_share_identity_evidence(
                rescue,
                candidate,
            )

            # This is the important duplicate veto. It is stricter than the
            # loose ambiguity diagnostic and intentionally works even when the
            # crop candidate and an existing PRIMARY come from the SAME owner
            # virtual view.
            candidate_conflicts = observation_matching_measurements(
                candidate,
                association,
            )

            candidate_records.append(
                {
                    "xyxy": list(
                        full_box
                    ),
                    "confidence": detection.confidence,
                    "status": candidate.status,
                    "is_owner": candidate.is_owner,
                    "owner_cell": candidate.owner_cell,
                    "world_x": candidate.world_x,
                    "world_z": candidate.world_z,
                    "world_distance_to_rescue_m": (
                        None
                        if not math.isfinite(distance)
                        else distance
                    ),
                    "matches_rescue_identity": matches_rescue_identity,
                    "strict_conflicting_measurement_ids": list(
                        candidate_conflicts
                    ),
                    # Backwards-compatible debug key.
                    "conflicting_measurement_ids": list(
                        candidate_conflicts
                    ),
                }
            )

            if candidate.status != "accepted":
                continue

            if not bool(candidate.is_owner):
                continue

            if (
                not math.isfinite(distance)
                or distance > max_world_distance_m
            ):
                continue

            if not matches_rescue_identity:
                continue

            if candidate_conflicts:
                continue

            candidates.append(
                (
                    distance,
                    -detection.confidence,
                    candidate,
                    full_box,
                    candidate_conflicts,
                )
            )

        debug_payload[
            rescue_id
        ] = {
            "owner_view_name": owner_view_name,
            "crop_xyxy": list(
                crop_xyxy
            ),
            "expected_owner_foot_px": list(
                expected_foot
            ),
            "expected_owner_top_px": list(
                expected_top
            ),
            "expected_height_px": expected_height,
            "detections": candidate_records,
        }

        if not candidates:
            confirmations.append(
                RescueImageConfirmation(
                    **base_kwargs,
                    status="owner_crop_no_new_identity_candidate",
                    confirmed=False,
                    expected_owner_foot_px=expected_foot,
                    expected_owner_top_px=expected_top,
                    expected_height_px=expected_height,
                    crop_xyxy=crop_xyxy,
                    crop_detection_count=len(
                        detections
                    ),
                    candidate_full_xyxy=None,
                    candidate_confidence=None,
                    candidate_world_x=None,
                    candidate_world_z=None,
                    candidate_world_distance_m=None,
                    candidate_is_owner=None,
                    candidate_conflicting_measurement_ids=(),
                )
            )
            continue

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        (
            best_distance,
            _,
            best_candidate,
            best_full_box,
            best_conflicts,
        ) = candidates[0]

        confirmed_ids.append(
            rescue_id
        )

        confirmations.append(
            RescueImageConfirmation(
                **base_kwargs,
                status="confirmed_in_owner_crop",
                confirmed=True,
                expected_owner_foot_px=expected_foot,
                expected_owner_top_px=expected_top,
                expected_height_px=expected_height,
                crop_xyxy=crop_xyxy,
                crop_detection_count=len(
                    detections
                ),
                candidate_full_xyxy=best_full_box,
                candidate_confidence=best_candidate.confidence,
                candidate_world_x=best_candidate.world_x,
                candidate_world_z=best_candidate.world_z,
                candidate_world_distance_m=best_distance,
                candidate_is_owner=best_candidate.is_owner,
                candidate_conflicting_measurement_ids=best_conflicts,
            )
        )

        synthetic_observation_id += max(
            len(detections),
            1,
        ) + 100

    updated = promote_image_confirmed_rescues(
        association,
        confirmed_ids,
    )

    return (
        updated,
        confirmations,
        debug_payload,
    )


def draw_owner_confirmation_debug(
    rendered: dict[str, np.ndarray],
    confirmations: list[RescueImageConfirmation],
    debug_payload: dict[int, dict[str, Any]],
    *,
    card_width: int = 1320,
    card_height: int = 650,
) -> np.ndarray | None:
    """Visual diagnostic for the targeted OWNER-image second pass.

    Nothing in this function changes association decisions. It only makes the
    reason for each accepted/rejected owner-crop candidate visible:

    - magenta X: expected RESCUE footpoint projected into the OWNER view
    - magenta line/circle: expected player extent (top -> foot)
    - C1, C2, ...: second-pass RF-DETR detections
    - green bbox: candidate passes every current confirmation condition
    - red bbox: candidate is rejected by at least one condition
    - right panel: confidence, X/Z distance, rescue identity match, owner flag,
      and strict conflicts with already represented persons
    """
    cards: list[np.ndarray] = []

    header_h = 112
    image_panel_w = 780
    diagnostics_x = image_panel_w + 24
    diagnostics_w = card_width - diagnostics_x - 18
    body_h = card_height - header_h

    for confirmation in confirmations:
        payload = debug_payload.get(
            confirmation.rescue_observation_id
        )

        if payload is None:
            continue

        owner_name = confirmation.owner_view_name

        if (
            owner_name is None
            or owner_name not in rendered
            or confirmation.crop_xyxy is None
        ):
            continue

        image = rendered[
            owner_name
        ]

        x1, y1, x2, y2 = confirmation.crop_xyxy
        crop = image[
            y1:y2,
            x1:x2,
        ].copy()

        expected_foot = payload.get(
            "expected_owner_foot_px"
        )
        expected_top = payload.get(
            "expected_owner_top_px"
        )

        # Draw the expected RESCUE location before resizing so it scales with
        # the crop exactly like the detector boxes.
        if (
            expected_foot is not None
            and len(expected_foot) == 2
        ):
            foot_local = (
                int(round(expected_foot[0] - x1)),
                int(round(expected_foot[1] - y1)),
            )

            marker_size = 12

            cv2.line(
                crop,
                (
                    foot_local[0] - marker_size,
                    foot_local[1] - marker_size,
                ),
                (
                    foot_local[0] + marker_size,
                    foot_local[1] + marker_size,
                ),
                (255, 0, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.line(
                crop,
                (
                    foot_local[0] - marker_size,
                    foot_local[1] + marker_size,
                ),
                (
                    foot_local[0] + marker_size,
                    foot_local[1] - marker_size,
                ),
                (255, 0, 255),
                4,
                cv2.LINE_AA,
            )

            cv2.putText(
                crop,
                "EXPECTED",
                (
                    foot_local[0] + 15,
                    max(
                        18,
                        foot_local[1] - 10,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )

            if (
                expected_top is not None
                and len(expected_top) == 2
            ):
                top_local = (
                    int(round(expected_top[0] - x1)),
                    int(round(expected_top[1] - y1)),
                )

                cv2.line(
                    crop,
                    top_local,
                    foot_local,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

                cv2.circle(
                    crop,
                    top_local,
                    6,
                    (255, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        for candidate_index, candidate in enumerate(
            payload["detections"],
            start=1,
        ):
            bx1, by1, bx2, by2 = candidate["xyxy"]

            local_p1 = (
                int(round(bx1 - x1)),
                int(round(by1 - y1)),
            )
            local_p2 = (
                int(round(bx2 - x1)),
                int(round(by2 - y1)),
            )

            distance = candidate.get(
                "world_distance_to_rescue_m"
            )

            valid = (
                candidate["status"] == "accepted"
                and bool(candidate["is_owner"])
                and bool(candidate.get("matches_rescue_identity"))
                and not candidate["conflicting_measurement_ids"]
                and distance is not None
            )

            color = (
                (0, 255, 0)
                if valid
                else (0, 0, 255)
            )

            cv2.rectangle(
                crop,
                local_p1,
                local_p2,
                color,
                3,
                cv2.LINE_AA,
            )

            cv2.putText(
                crop,
                f"C{candidate_index}",
                (
                    local_p1[0],
                    max(
                        18,
                        local_p1[1] - 7,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )

        scale = min(
            image_panel_w / max(crop.shape[1], 1),
            body_h / max(crop.shape[0], 1),
        )

        resized_w = max(
            1,
            int(round(crop.shape[1] * scale)),
        )
        resized_h = max(
            1,
            int(round(crop.shape[0] * scale)),
        )

        resized = cv2.resize(
            crop,
            (resized_w, resized_h),
            interpolation=(
                cv2.INTER_CUBIC
                if scale > 1.0
                else cv2.INTER_AREA
            ),
        )

        card = np.zeros(
            (card_height, card_width, 3),
            dtype=np.uint8,
        )

        image_x0 = (
            image_panel_w - resized_w
        ) // 2
        image_y0 = header_h + (
            body_h - resized_h
        ) // 2

        card[
            image_y0:image_y0 + resized_h,
            image_x0:image_x0 + resized_w,
        ] = resized

        status_color = (
            (0, 255, 0)
            if confirmation.confirmed
            else (255, 0, 255)
        )

        cv2.putText(
            card,
            (
                f"RESCUE O{confirmation.rescue_observation_id} -> "
                f"{owner_name}"
            ),
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            card,
            f"status={confirmation.status}",
            (14, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            status_color,
            1,
            cv2.LINE_AA,
        )

        loose_conflicts = (
            confirmation.source_conflicting_measurement_ids
        )

        expected_h_text = (
            f"{confirmation.expected_height_px:.1f}px"
            if confirmation.expected_height_px is not None
            else "?"
        )

        cv2.putText(
            card,
            (
                f"crop detections={confirmation.crop_detection_count}  "
                f"expected_h={expected_h_text}  "
                f"loose_rescue_conflicts="
                f"{list(loose_conflicts) if loose_conflicts else '-'}"
            ),
            (14, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(
            card,
            "MAGENTA=X expected RESCUE | GREEN=passes all | RED=rejected",
            (14, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        # Diagnostic panel: one compact two-line block per crop candidate.
        cv2.putText(
            card,
            "OWNER-CROP CANDIDATES",
            (diagnostics_x, header_h + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y = header_h + 54

        for candidate_index, candidate in enumerate(
            payload["detections"],
            start=1,
        ):
            if y + 48 >= card_height:
                cv2.putText(
                    card,
                    "...",
                    (diagnostics_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                break

            distance = candidate.get(
                "world_distance_to_rescue_m"
            )

            distance_text = (
                "?"
                if distance is None
                else f"{distance:.2f}m"
            )

            rescue_match = bool(
                candidate.get(
                    "matches_rescue_identity"
                )
            )
            owner_ok = bool(
                candidate.get(
                    "is_owner"
                )
            )
            conflicts = candidate.get(
                "conflicting_measurement_ids",
                [],
            )

            accepted = (
                candidate.get("status") == "accepted"
            )

            passes = (
                accepted
                and owner_ok
                and rescue_match
                and not conflicts
                and distance is not None
            )

            color = (
                (0, 255, 0)
                if passes
                else (0, 0, 255)
            )

            confidence = float(
                candidate.get(
                    "confidence",
                    0.0,
                )
            )

            cv2.putText(
                card,
                (
                    f"C{candidate_index} conf={confidence:.3f} "
                    f"d={distance_text} rescue={'Y' if rescue_match else 'N'} "
                    f"owner={'Y' if owner_ok else 'N'}"
                ),
                (diagnostics_x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                color,
                1,
                cv2.LINE_AA,
            )

            status = str(
                candidate.get(
                    "status",
                    "?",
                )
            )

            conflict_text = (
                "-"
                if not conflicts
                else ",".join(
                    f"P{value}"
                    for value in conflicts
                )
            )

            cv2.putText(
                card,
                (
                    f"   status={status} conflicts={conflict_text}"
                ),
                (diagnostics_x, y + 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

            y += 50

        cards.append(
            card
        )

    if not cards:
        return None

    return np.concatenate(
        cards,
        axis=0,
    )


def deferred_rescue_diagnostics(
    association: AssociationResult,
) -> list[dict[str, Any]]:
    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    diagnostics: list[dict[str, Any]] = []

    for observation_id in association.deferred_rescue_observation_ids:
        observation = observation_by_id.get(
            observation_id
        )

        if observation is None:
            continue

        truncation = [
            name
            for name, active in (
                ("top", observation.truncated_top),
                ("bottom", observation.truncated_bottom),
                ("left", observation.truncated_left),
                ("right", observation.truncated_right),
            )
            if active
        ]

        diagnostics.append(
            {
                "observation_id": observation.observation_id,
                "view_name": observation.view_name,
                "confidence": observation.confidence,
                "owner_cell": observation.owner_cell,
                "world_x": observation.world_x,
                "world_z": observation.world_z,
                "xyxy": list(observation.xyxy),
                "foot_pixel": [
                    observation.foot_x,
                    observation.foot_y,
                ],
                "inside_strict_field": observation.inside_strict_field,
                "inside_allowed_field": observation.inside_allowed_field,
                "truncated": truncation,
                "source_lenses": [
                    item.to_dict()
                    for item in observation.source_lenses
                ],
            }
        )

    return diagnostics


def draw_deferred_rescue_debug(
    rendered: dict[str, np.ndarray],
    association: AssociationResult,
    *,
    crop_width: int = 720,
    crop_height: int = 420,
    padding_px: int = 120,
) -> np.ndarray | None:
    """Build enlarged crops for every deferred RESCUE in the frame."""
    observation_by_id = {
        item.observation_id: item
        for item in association.observations
    }

    cards: list[np.ndarray] = []

    for observation_id in association.deferred_rescue_observation_ids:
        observation = observation_by_id.get(
            observation_id
        )

        if observation is None:
            continue

        image = rendered.get(
            observation.view_name
        )

        if image is None:
            continue

        x1, y1, x2, y2 = observation.xyxy

        ix1 = max(
            0,
            int(round(x1)) - padding_px,
        )
        iy1 = max(
            0,
            int(round(y1)) - padding_px,
        )
        ix2 = min(
            image.shape[1],
            int(round(x2)) + padding_px,
        )
        iy2 = min(
            image.shape[0],
            int(round(y2)) + padding_px,
        )

        if ix2 <= ix1 or iy2 <= iy1:
            continue

        crop = image[
            iy1:iy2,
            ix1:ix2,
        ].copy()

        local_p1 = (
            int(round(x1)) - ix1,
            int(round(y1)) - iy1,
        )
        local_p2 = (
            int(round(x2)) - ix1,
            int(round(y2)) - iy1,
        )

        cv2.rectangle(
            crop,
            local_p1,
            local_p2,
            (255, 0, 255),
            4,
            cv2.LINE_AA,
        )

        foot = (
            int(round(observation.foot_x)) - ix1,
            int(round(observation.foot_y)) - iy1,
        )

        cv2.circle(
            crop,
            foot,
            7,
            (255, 0, 255),
            -1,
            cv2.LINE_AA,
        )

        scale = min(
            crop_width / max(crop.shape[1], 1),
            (crop_height - 86) / max(crop.shape[0], 1),
        )

        resized_w = max(
            1,
            int(round(crop.shape[1] * scale)),
        )
        resized_h = max(
            1,
            int(round(crop.shape[0] * scale)),
        )

        resized = cv2.resize(
            crop,
            (resized_w, resized_h),
            interpolation=(
                cv2.INTER_CUBIC
                if scale > 1.0
                else cv2.INTER_AREA
            ),
        )

        card = np.zeros(
            (crop_height, crop_width, 3),
            dtype=np.uint8,
        )

        x0 = (
            crop_width - resized_w
        ) // 2
        y0 = 86 + (
            crop_height - 86 - resized_h
        ) // 2

        card[
            y0:y0 + resized_h,
            x0:x0 + resized_w,
        ] = resized

        owner = (
            observation.owner_cell
            if observation.owner_cell is not None
            else "?"
        )

        world_text = (
            "XZ=?"
            if (
                observation.world_x is None
                or observation.world_z is None
            )
            else (
                f"XZ=({observation.world_x:.2f}, "
                f"{observation.world_z:.2f})m"
            )
        )

        cv2.putText(
            card,
            (
                f"R-DEFER O{observation.observation_id} | "
                f"view={observation.view_name} | owner={owner}"
            ),
            (14, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            card,
            (
                f"conf={observation.confidence:.3f} | {world_text} | "
                f"strict={observation.inside_strict_field} | "
                f"allowed={observation.inside_allowed_field} | "
                f"clip={observation.is_truncated}"
            ),
            (14, 61),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.47,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cards.append(
            card
        )

    if not cards:
        return None

    return np.concatenate(
        cards,
        axis=0,
    )


def association_summary(
    association: AssociationResult,
) -> dict[str, Any]:
    accepted = [
        item
        for item in association.observations
        if item.status == "accepted"
    ]

    rejected = [
        item
        for item in association.observations
        if item.status != "accepted"
    ]

    multi_view = [
        item
        for item in association.measurements
        if item.multi_view
    ]

    represented_observation_ids = {
        observation_id
        for measurement in association.measurements
        for observation_id in measurement.observation_ids
    }

    collapsed_duplicates = (
        len(represented_observation_ids)
        - len(association.measurements)
    )

    return {
        "raw_observations": len(
            association.observations
        ),
        "accepted_field_observations": len(
            accepted
        ),
        "rejected_observations": len(
            rejected
        ),
        "primary_observations": (
            association.primary_observation_count
        ),
        "rescue_observations": (
            association.rescue_observation_count
        ),
        "unique_person_measurements": len(
            association.measurements
        ),
        "owner_backed_measurements": (
            association.owner_backed_measurement_count
        ),
        "promoted_rescue_measurements": (
            association.promoted_rescue_measurement_count
        ),
        "image_confirmed_rescue_measurements": (
            association.image_confirmed_rescue_measurement_count
        ),
        "deferred_rescue_observations": len(
            association.deferred_rescue_observation_ids
        ),
        "collapsed_duplicate_observations": (
            collapsed_duplicates
        ),
        "multi_view_measurements": len(
            multi_view
        ),
        "selected_match_edges": len(
            association.matches
        ),
        "population_delta_from_23_prior_only": (
            len(association.measurements)
            - 23
        ),
    }


def run(
    args: argparse.Namespace,
) -> None:
    if not args.video.is_file():
        raise FileNotFoundError(
            args.video
        )

    if not args.calibration.is_file():
        raise FileNotFoundError(
            args.calibration
        )

    if not args.weights.is_file():
        raise FileNotFoundError(
            args.weights
        )

    metadata = load_veo_metadata(
        args.calibration
    )

    print(
        "Deriving validated 4 FAR + 2 NEAR TIGHT geometry..."
    )

    specs, layout, _, _ = derive_specs(
        metadata,
        args.width,
        args.height,
        split_fraction=args.split_fraction,
        near_count=DEFAULT_NEAR_CAMERAS,
        far_count=DEFAULT_FAR_CAMERAS,
        guard_margin_m=args.guard_margin_m,
        grid_step_m=args.grid_step_m,
    )

    tight_specs = ordered_tight_specs(
        specs
    )

    spec_by_name = {
        spec.name: spec
        for spec in tight_specs
    }

    print(
        "Building six static direct-lens TIGHT remap grids..."
    )

    maps = {
        spec.name: build_map(
            metadata,
            spec,
            args.width,
            args.height,
        )
        for spec in tight_specs
    }

    print(
        "Loading RF-DETR 2XL C2..."
    )

    backend = RFDETR2XLBackend(
        args.weights,
        device=args.device,
    )

    print(
        json.dumps(
            backend.info(),
            indent=2,
        )
    )

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    result: dict[str, Any] = {
        "experiment": (
            "six_tight_rfdetr_2xl_c2_same_frame_cross_view_association"
        ),
        "scope": (
            "per-frame detection + position + conservative cross-view "
            "association; NO temporal tracking"
        ),
        "video": str(
            args.video
        ),
        "calibration": str(
            args.calibration
        ),
        "weights": str(
            args.weights
        ),
        "threshold": args.threshold,
        "model": backend.info(),
        "render_size_per_view": [
            args.width,
            args.height,
        ],
        "ultra_panorama_debug": {
            "enabled": True,
            "purpose": (
                "presentation-only projection of representative detection rays; "
                "not used for detection, association, or X/Z estimation"
            ),
        },
        "split_fraction": args.split_fraction,
        "split_depth_m": layout.split_depth_m,
        "guard_margin_m": args.guard_margin_m,
        "grid_step_m": args.grid_step_m,
        "association_config": {
            "field_margin_m": args.field_margin_m,
            "image_rescue_confirmation": {
                "enabled": True,
                "threshold": args.rescue_confirm_threshold,
                "max_world_distance_m": (
                    args.rescue_confirm_max_world_distance_m
                ),
                "method": (
                    "project deferred rescue X/Z into OWNER virtual camera, "
                    "crop owner image, rerun RF-DETR, require identity-grade match to RESCUE, reject strict duplicate matches"
                ),
                "diagnostic_overlay": (
                    "expected RESCUE marker + per-candidate distance, rescue match, owner flag, and strict conflicts"
                ),
            },
            "min_source_iou": args.assoc_min_source_iou,
            "max_source_anchor_norm": (
                args.assoc_max_anchor_norm
            ),
            "max_world_distance_m": (
                args.assoc_max_world_distance_m
            ),
            "min_source_iom": (
                args.assoc_min_source_iom
            ),
            "min_horizontal_iom": (
                args.assoc_min_horizontal_iom
            ),
            "min_vertical_iom": (
                args.assoc_min_vertical_iom
            ),
            "max_center_x_norm": (
                args.assoc_max_center_x_norm
            ),
            "max_containment_world_distance_m": (
                args.assoc_max_containment_world_distance_m
            ),
            "max_truncated_world_distance_m": (
                args.assoc_max_truncated_world_distance_m
            ),
            "identity_rule": (
                "exclusive logical owner cells create PRIMARY observations. "
                "Overlap observations are RESCUE evidence and cannot create a "
                "same-frame person alone. Strong source-pixel matches attach "
                "RESCUE to PRIMARY. Corroborated rescue-only groups may be "
                "promoted; singleton/ambiguous rescues are deferred for future "
                "temporal tracking. X/Z never creates identity by itself."
            ),
        },
        "camera_order": [
            spec.name
            for spec in tight_specs
        ],
        "frames": [],
    }

    total_inference_s = 0.0
    total_images = 0

    for requested_time in args.times:
        frame, frame_info = read_frame(
            args.video,
            requested_time,
        )

        frame_number = int(
            frame_info["actual_frame"]
        )
        time_s = float(
            frame_info["actual_time_s"]
        )

        print(
            f"Frame {frame_number} "
            f"t={time_s:.3f}s"
        )

        frame_dir = (
            args.out_dir
            / f"frame_{frame_number:06d}"
        )

        frame_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        rendered: dict[
            str,
            np.ndarray,
        ] = {}

        detections_by_name: dict[
            str,
            list[PersonDetection],
        ] = {}

        raw_counts: dict[
            str,
            int,
        ] = {}

        view_records: list[
            dict[str, Any]
        ] = []

        observation_id = 1
        observations: list[
            Observation
        ] = []
        observations_by_view: dict[
            str,
            list[Observation],
        ] = {}

        for spec in tight_specs:
            name = spec.name

            image = render_map(
                frame,
                maps[name],
            )

            rendered[name] = image

            start = time.perf_counter()

            detections = predict_persons(
                backend,
                image,
                threshold=args.threshold,
            )

            inference_s = (
                time.perf_counter()
                - start
            )

            total_inference_s += inference_s
            total_images += 1

            detections_by_name[
                name
            ] = detections

            raw_counts[
                name
            ] = len(detections)

            current_observations: list[
                Observation
            ] = []

            for detection in detections:
                observation = build_observation(
                    observation_id=observation_id,
                    frame_number=frame_number,
                    time_s=time_s,
                    metadata=metadata,
                    layout=layout,
                    spec=spec,
                    camera_map=maps[name],
                    xyxy=detection.xyxy,
                    confidence=detection.confidence,
                    field_margin_m=args.field_margin_m,
                )

                observations.append(
                    observation
                )

                current_observations.append(
                    observation
                )

                observation_id += 1

            observations_by_view[
                name
            ] = current_observations

            metrics = detection_metrics(
                detections
            )

            view_records.append(
                {
                    "name": name,
                    "band": spec.cell.band,
                    "index_in_band": (
                        spec.cell.index_in_band
                    ),
                    "hfov_deg": math.degrees(
                        spec.hfov
                    ),
                    "vfov_deg": math.degrees(
                        spec.vfov
                    ),
                    "inference_s": inference_s,
                    "metrics": metrics,
                    "raw_detections": [
                        item.to_dict()
                        for item in detections
                    ],
                    "observations": [
                        item.to_dict()
                        for item in current_observations
                    ],
                }
            )

            cv2.imwrite(
                str(
                    frame_dir
                    / f"{name}.png"
                ),
                image,
            )

        association = associate_observations(
            observations,
            layout,
            min_source_iou=(
                args.assoc_min_source_iou
            ),
            max_source_anchor_norm=(
                args.assoc_max_anchor_norm
            ),
            max_world_distance_m=(
                args.assoc_max_world_distance_m
            ),
            min_source_iom=(
                args.assoc_min_source_iom
            ),
            min_horizontal_iom=(
                args.assoc_min_horizontal_iom
            ),
            min_vertical_iom=(
                args.assoc_min_vertical_iom
            ),
            max_center_x_norm=(
                args.assoc_max_center_x_norm
            ),
            max_containment_world_distance_m=(
                args.assoc_max_containment_world_distance_m
            ),
            max_truncated_world_distance_m=(
                args.assoc_max_truncated_world_distance_m
            ),
        )

        (
            association,
            image_confirmations,
            image_confirmation_debug_payload,
        ) = confirm_deferred_rescues_from_owner_images(
            backend=backend,
            metadata=metadata,
            layout=layout,
            rendered=rendered,
            maps=maps,
            spec_by_name=spec_by_name,
            association=association,
            frame_number=frame_number,
            time_s=time_s,
            field_margin_m=args.field_margin_m,
            threshold=args.rescue_confirm_threshold,
            max_world_distance_m=(
                args.rescue_confirm_max_world_distance_m
            ),
        )

        # Clean visual audit: exactly one neutral box per final unique
        # PersonMeasurement across all six views. No PRIMARY/RESCUE colors,
        # no deferred evidence, no match/debug semantics.
        unique_only_rendered = {
            name: draw_unique_representatives(
                rendered[name],
                view_name=name,
                association=association,
                numbered=False,
            )
            for name in rendered
        }

        unique_numbered_rendered = {
            name: draw_unique_representatives(
                rendered[name],
                view_name=name,
                association=association,
                numbered=True,
            )
            for name in rendered
        }

        unique_only_sheet = clean_unique_contact_sheet(
            unique_only_rendered,
            association,
            panel_width=args.panel_width,
            frame_number=frame_number,
            time_s=time_s,
            numbered=False,
        )

        unique_numbered_sheet = clean_unique_contact_sheet(
            unique_numbered_rendered,
            association,
            panel_width=args.panel_width,
            frame_number=frame_number,
            time_s=time_s,
            numbered=True,
        )

        unique_only_path = (
            frame_dir
            / "comparison_unique_only.jpg"
        )
        unique_numbered_path = (
            frame_dir
            / "comparison_unique_numbered.jpg"
        )

        cv2.imwrite(
            str(unique_only_path),
            unique_only_sheet,
        )
        cv2.imwrite(
            str(unique_numbered_path),
            unique_numbered_sheet,
        )

        # Single ultra-panorama visual audit.
        #
        # IMPORTANT: use the representative DETECTION BBOX ray itself, not
        # fused X/Z. Tiny/far bbox footpoints can move several metres on the
        # ground plane even when the visual detection is correct. This debug
        # image answers only: "which visible players did the final unique
        # detections correspond to?"
        ultra_panorama = render_ultra_panorama(
            frame,
            metadata,
        )

        panorama_h, panorama_w = (
            ultra_panorama.shape[:2]
        )

        observation_by_id = {
            item.observation_id: item
            for item in association.observations
        }

        panorama_detections = []

        for measurement in association.measurements:
            representative = observation_by_id.get(
                measurement.primary_observation_id
            )

            if representative is None:
                continue

            spec = spec_by_name.get(
                representative.view_name
            )

            if spec is None:
                continue

            projected = (
                virtual_bbox_to_panorama_detection(
                    metadata,
                    spec,
                    person_id=measurement.measurement_id,
                    view_name=representative.view_name,
                    xyxy=representative.xyxy,
                    virtual_width=args.width,
                    virtual_height=args.height,
                    panorama_width=panorama_w,
                    panorama_height=panorama_h,
                )
            )

            if projected is not None:
                panorama_detections.append(
                    projected
                )

        ultra_unique_only = draw_panorama_detections(
            ultra_panorama,
            panorama_detections,
            numbered=False,
        )

        ultra_unique_numbered = draw_panorama_detections(
            ultra_panorama,
            panorama_detections,
            numbered=True,
        )

        ultra_unique_only_path = (
            frame_dir
            / "ultra_panorama_unique_only.jpg"
        )
        ultra_unique_numbered_path = (
            frame_dir
            / "ultra_panorama_unique_numbered.jpg"
        )

        cv2.imwrite(
            str(
                ultra_unique_only_path
            ),
            ultra_unique_only,
        )
        cv2.imwrite(
            str(
                ultra_unique_numbered_path
            ),
            ultra_unique_numbered,
        )

        annotated = {
            name: draw_associated_detections(
                rendered[name],
                observations_by_view.get(
                    name,
                    [],
                ),
                association,
            )
            for name in rendered
        }

        for name, image in annotated.items():
            cv2.imwrite(
                str(
                    frame_dir
                    / f"{name}_associated.png"
                ),
                image,
            )

        owner_confirmation_debug = draw_owner_confirmation_debug(
            rendered,
            image_confirmations,
            image_confirmation_debug_payload,
        )

        owner_confirmation_debug_path: Path | None = None

        if owner_confirmation_debug is not None:
            owner_confirmation_debug_path = (
                frame_dir
                / "owner_image_confirmation_debug.jpg"
            )

            cv2.imwrite(
                str(
                    owner_confirmation_debug_path
                ),
                owner_confirmation_debug,
            )

        radar = draw_unique_radar(
            metadata,
            association,
        )

        radar_path = (
            frame_dir
            / "unique_radar.jpg"
        )

        cv2.imwrite(
            str(radar_path),
            radar,
        )

        sheet = contact_sheet(
            annotated,
            observations_by_view,
            raw_counts,
            association,
            panel_width=args.panel_width,
            threshold=args.threshold,
            frame_number=frame_number,
            time_s=time_s,
        )

        sheet_path = (
            frame_dir
            / "comparison_associated.jpg"
        )

        cv2.imwrite(
            str(sheet_path),
            sheet,
        )

        summary = association_summary(
            association
        )

        deferred_diagnostics = deferred_rescue_diagnostics(
            association
        )

        deferred_debug = draw_deferred_rescue_debug(
            rendered,
            association,
        )

        deferred_debug_path: Path | None = None

        if deferred_debug is not None:
            deferred_debug_path = (
                frame_dir
                / "deferred_rescue_debug.jpg"
            )

            cv2.imwrite(
                str(
                    deferred_debug_path
                ),
                deferred_debug,
            )

        print(
            "  "
            f"raw={summary['raw_observations']} "
            f"accepted={summary['accepted_field_observations']} "
            f"primary={summary['primary_observations']} "
            f"rescue={summary['rescue_observations']} "
            f"unique={summary['unique_person_measurements']} "
            f"owner_backed={summary['owner_backed_measurements']} "
            f"promoted_rescue={summary['promoted_rescue_measurements']} "
            f"image_confirmed={summary['image_confirmed_rescue_measurements']} "
            f"deferred_rescue={summary['deferred_rescue_observations']} "
            f"collapsed={summary['collapsed_duplicate_observations']} "
            f"matches={summary['selected_match_edges']}"
        )

        for confirmation in image_confirmations:
            if confirmation.confirmed:
                print(
                    "    IMG-CONFIRM "
                    f"rescue={confirmation.rescue_observation_id} "
                    f"owner_view={confirmation.owner_view_name} "
                    f"conf={confirmation.candidate_confidence:.3f} "
                    f"XZ=({confirmation.candidate_world_x:.2f},"
                    f"{confirmation.candidate_world_z:.2f})m "
                    f"distance={confirmation.candidate_world_distance_m:.2f}m"
                )
            else:
                print(
                    "    IMG-CHECK "
                    f"rescue={confirmation.rescue_observation_id} "
                    f"owner_view={confirmation.owner_view_name} "
                    f"status={confirmation.status} "
                    f"crop_detections={confirmation.crop_detection_count}"
                )

        for confirmation in image_confirmations:
            payload = image_confirmation_debug_payload.get(
                confirmation.rescue_observation_id
            )

            if payload is None:
                continue

            for candidate_index, candidate in enumerate(
                payload["detections"],
                start=1,
            ):
                distance = candidate.get(
                    "world_distance_to_rescue_m"
                )

                distance_text = (
                    "?"
                    if distance is None
                    else f"{distance:.2f}m"
                )

                conflicts = candidate.get(
                    "conflicting_measurement_ids",
                    [],
                )

                print(
                    "      CAND "
                    f"C{candidate_index} "
                    f"conf={candidate['confidence']:.3f} "
                    f"d={distance_text} "
                    f"rescue_match={bool(candidate.get('matches_rescue_identity'))} "
                    f"owner={bool(candidate.get('is_owner'))} "
                    f"status={candidate.get('status')} "
                    f"conflicts={conflicts or '-'}"
                )

        for item in deferred_diagnostics:
            world_text = (
                "XZ=?"
                if (
                    item["world_x"] is None
                    or item["world_z"] is None
                )
                else (
                    f"XZ=({item['world_x']:.2f},"
                    f"{item['world_z']:.2f})m"
                )
            )

            trunc_text = (
                ",".join(
                    item["truncated"]
                )
                if item["truncated"]
                else "none"
            )

            print(
                "    DEFER "
                f"obs={item['observation_id']} "
                f"view={item['view_name']} "
                f"owner={item['owner_cell']} "
                f"conf={item['confidence']:.3f} "
                f"{world_text} "
                f"strict={item['inside_strict_field']} "
                f"allowed={item['inside_allowed_field']} "
                f"truncated={trunc_text} "
                f"bbox={tuple(round(v, 1) for v in item['xyxy'])}"
            )

        for edge in association.matches:
            print(
                "    MATCH "
                f"{edge.view_a}#{edge.observation_a}"
                " <-> "
                f"{edge.view_b}#{edge.observation_b} "
                f"mode={edge.match_mode} "
                f"lens={edge.source_lens} "
                f"IoU={edge.source_iou:.3f} "
                f"IoM={edge.source_iom:.3f} "
                f"hIoM={edge.source_horizontal_iom:.3f} "
                f"dx={edge.source_center_x_distance_norm:.3f}w "
                f"anchor={edge.source_anchor_distance_norm:.3f}h "
                f"XZ={edge.world_distance_m:.2f}m "
                f"score={edge.score:.3f}"
            )

        frame_record = {
            "source": frame_info,
            "summary": summary,
            "association": (
                association.to_dict()
            ),
            "image_rescue_confirmations": [
                item.to_dict()
                for item in image_confirmations
            ],
            "owner_image_confirmation_debug": (
                str(owner_confirmation_debug_path)
                if owner_confirmation_debug_path is not None
                else None
            ),
            "deferred_rescue_diagnostics": (
                deferred_diagnostics
            ),
            "deferred_rescue_debug": (
                str(deferred_debug_path)
                if deferred_debug_path is not None
                else None
            ),
            "views": view_records,
            "comparison": str(
                sheet_path
            ),
            "comparison_unique_only": str(
                unique_only_path
            ),
            "comparison_unique_numbered": str(
                unique_numbered_path
            ),
            "ultra_panorama_unique_only": str(
                ultra_unique_only_path
            ),
            "ultra_panorama_unique_numbered": str(
                ultra_unique_numbered_path
            ),
            "radar": str(
                radar_path
            ),
        }

        result[
            "frames"
        ].append(
            frame_record
        )

        frame_json = (
            frame_dir
            / "association.json"
        )

        frame_json.write_text(
            json.dumps(
                frame_record,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print(
            f"  -> {sheet_path}"
        )
        print(
            f"  -> {unique_only_path}"
        )
        print(
            f"  -> {unique_numbered_path}"
        )
        print(
            f"  -> {ultra_unique_only_path}"
        )
        print(
            f"  -> {ultra_unique_numbered_path}"
        )
        print(
            f"  -> {radar_path}"
        )

        if deferred_debug_path is not None:
            print(
                f"  -> {deferred_debug_path}"
            )

        if owner_confirmation_debug_path is not None:
            print(
                f"  -> {owner_confirmation_debug_path}"
            )

    result[
        "runtime"
    ] = {
        "total_detector_images": (
            total_images
        ),
        "total_inference_s": (
            total_inference_s
        ),
        "mean_inference_s_per_view": (
            total_inference_s / total_images
            if total_images
            else None
        ),
        "detector_views_per_source_frame": 6,
    }

    output_json = (
        args.out_dir
        / "association_results.json"
    )

    output_json.write_text(
        json.dumps(
            result,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"Results: {output_json}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
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
        "--weights",
        type=Path,
        default=DEFAULT_C2_WEIGHTS,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
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
        "--split-fraction",
        type=float,
        default=DEFAULT_SPLIT_FRACTION,
    )

    parser.add_argument(
        "--guard-margin-m",
        type=float,
        default=DEFAULT_GUARD_M,
    )

    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=DEFAULT_GRID_STEP_M,
    )

    parser.add_argument(
        "--field-margin-m",
        type=float,
        default=DEFAULT_FIELD_MARGIN_M,
    )

    parser.add_argument(
        "--assoc-min-source-iou",
        type=float,
        default=DEFAULT_MIN_SOURCE_IOU,
    )

    parser.add_argument(
        "--assoc-max-anchor-norm",
        type=float,
        default=DEFAULT_MAX_SOURCE_ANCHOR_NORM,
    )

    parser.add_argument(
        "--assoc-max-world-distance-m",
        type=float,
        default=DEFAULT_MAX_WORLD_DISTANCE_M,
    )

    parser.add_argument(
        "--assoc-min-source-iom",
        type=float,
        default=DEFAULT_MIN_SOURCE_IOM,
        help=(
            "Minimum original-source intersection-over-smaller-footprint "
            "for scale-mismatched / partial-player association."
        ),
    )

    parser.add_argument(
        "--assoc-min-horizontal-iom",
        type=float,
        default=DEFAULT_MIN_HORIZONTAL_IOM,
    )

    parser.add_argument(
        "--assoc-min-vertical-iom",
        type=float,
        default=DEFAULT_MIN_VERTICAL_IOM,
    )

    parser.add_argument(
        "--assoc-max-center-x-norm",
        type=float,
        default=DEFAULT_MAX_CENTER_X_NORM,
    )

    parser.add_argument(
        "--assoc-max-containment-world-distance-m",
        type=float,
        default=DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
    )

    parser.add_argument(
        "--assoc-max-truncated-world-distance-m",
        type=float,
        default=DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
    )

    parser.add_argument(
        "--rescue-confirm-threshold",
        type=float,
        default=DEFAULT_RESCUE_CONFIRM_THRESHOLD,
        help=(
            "RF-DETR threshold for targeted second-pass owner-view crops. "
            "Default is the same 0.30 used by the main pass."
        ),
    )

    parser.add_argument(
        "--rescue-confirm-max-world-distance-m",
        type=float,
        default=DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M,
        help=(
            "Maximum X/Z distance between a deferred rescue and the candidate "
            "re-detected in its owner crop."
        ),
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

    if not 0.0 <= args.threshold <= 1.0:
        parser.error(
            "--threshold must be between 0 and 1"
        )

    if not 0.0 <= args.rescue_confirm_threshold <= 1.0:
        parser.error(
            "--rescue-confirm-threshold must be between 0 and 1"
        )

    if args.rescue_confirm_max_world_distance_m <= 0.0:
        parser.error(
            "--rescue-confirm-max-world-distance-m must be > 0"
        )

    if (
        args.width <= 0
        or args.height <= 0
        or args.panel_width <= 0
    ):
        parser.error(
            "image dimensions must be positive"
        )

    if not 0.05 < args.split_fraction < 0.95:
        parser.error(
            "--split-fraction must be between 0.05 and 0.95"
        )

    if (
        args.guard_margin_m < 0.0
        or args.grid_step_m <= 0.0
        or args.field_margin_m < 0.0
    ):
        parser.error(
            "geometry margins must be non-negative and grid step positive"
        )

    if not 0.0 <= args.assoc_min_source_iou <= 1.0:
        parser.error(
            "--assoc-min-source-iou must be between 0 and 1"
        )

    if args.assoc_max_anchor_norm < 0.0:
        parser.error(
            "--assoc-max-anchor-norm must be >= 0"
        )

    if args.assoc_max_world_distance_m <= 0.0:
        parser.error(
            "--assoc-max-world-distance-m must be > 0"
        )

    for name, value in (
        ("--assoc-min-source-iom", args.assoc_min_source_iom),
        ("--assoc-min-horizontal-iom", args.assoc_min_horizontal_iom),
        ("--assoc-min-vertical-iom", args.assoc_min_vertical_iom),
    ):
        if not 0.0 <= value <= 1.0:
            parser.error(
                f"{name} must be between 0 and 1"
            )

    if args.assoc_max_center_x_norm < 0.0:
        parser.error(
            "--assoc-max-center-x-norm must be >= 0"
        )

    if args.assoc_max_containment_world_distance_m <= 0.0:
        parser.error(
            "--assoc-max-containment-world-distance-m must be > 0"
        )

    if args.assoc_max_truncated_world_distance_m <= 0.0:
        parser.error(
            "--assoc-max-truncated-world-distance-m must be > 0"
        )

    return args


if __name__ == "__main__":
    run(
        parse_args()
    )
