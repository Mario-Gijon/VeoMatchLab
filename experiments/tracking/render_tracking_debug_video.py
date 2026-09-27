#!/usr/bin/env python3
"""Render a two-panel Tracking V1 diagnostic video in one tracking pass."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tracking.run_global_tracker import parse_args, run  # noqa: E402
from experiments.virtual_camera_coverage.ultra_panorama_overlay import (  # noqa: E402
    draw_panorama_detections,
    render_ultra_panorama,
    virtual_bbox_to_panorama_detection,
)


VIDEO_NAME = "tracking_debug.mp4"
FIELD_PANEL_WIDTH = 760


def _field_pixel(
    x: float,
    z: float,
    *,
    field_length: float,
    field_width: float,
    left: int,
    top: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    u = (x + field_length / 2.0) / field_length
    v = (field_width / 2.0 - z) / field_width
    return (
        left + int(round(u * width)),
        top + int(round(v * height)),
    )


def _draw_field(
    snapshots: tuple[Any, ...],
    metadata: Any,
    *,
    timestamp: float,
    frame_index: int,
    measurement_count: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, FIELD_PANEL_WIDTH, 3), dtype=np.uint8)
    canvas[:] = (18, 24, 28)

    field_length = float(metadata.field_length)
    field_width = float(metadata.field_width)
    margin_x, margin_y = 48, 112
    field_w = FIELD_PANEL_WIDTH - 2 * margin_x
    field_h = int(round(field_w * field_width / field_length))
    if field_h > canvas.shape[0] - margin_y - 42:
        field_h = canvas.shape[0] - margin_y - 42
        field_w = int(round(field_h * field_length / field_width))
    left = (FIELD_PANEL_WIDTH - field_w) // 2
    top = margin_y + (canvas.shape[0] - margin_y - field_h - 42) // 2

    cv2.rectangle(canvas, (left, top), (left + field_w, top + field_h), (35, 105, 48), -1)
    cv2.rectangle(canvas, (left, top), (left + field_w, top + field_h), (225, 235, 225), 2)
    mid_x = left + field_w // 2
    cv2.line(canvas, (mid_x, top), (mid_x, top + field_h), (220, 230, 220), 1, cv2.LINE_AA)
    cv2.circle(canvas, (mid_x, top + field_h // 2), max(1, field_w // 12), (220, 230, 220), 1, cv2.LINE_AA)

    active = sum(item.lifecycle.value == "active" for item in snapshots)
    coasting = sum(item.lifecycle.value == "coasting" for item in snapshots)
    tentative = sum(item.lifecycle.value == "tentative" for item in snapshots)
    cv2.putText(
        canvas,
        f"FIELD  t={timestamp:.2f}s  frame={frame_index}  measurements={measurement_count}",
        (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (235, 235, 235), 1, cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"ACTIVE {active}  COASTING {coasting}  TENTATIVE {tentative}",
        (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (235, 235, 235), 1, cv2.LINE_AA,
    )

    for snapshot in snapshots:
        px, py = _field_pixel(
            snapshot.x, snapshot.z,
            field_length=field_length, field_width=field_width,
            left=left, top=top, width=field_w, height=field_h,
        )
        lifecycle = snapshot.lifecycle.value
        kind = snapshot.position_kind.value
        if lifecycle == "tentative":
            color = (0, 190, 255)
        elif lifecycle == "coasting":
            color = (255, 190, 40)
        else:
            color = (70, 225, 90)

        if kind == "observed":
            cv2.circle(canvas, (px, py), 7, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, (px, py), 9, (20, 20, 20), 1, cv2.LINE_AA)
        else:
            cv2.circle(canvas, (px, py), 9, color, 2, cv2.LINE_AA)
            cv2.line(canvas, (px - 6, py - 6), (px + 6, py + 6), color, 2, cv2.LINE_AA)
            cv2.line(canvas, (px - 6, py + 6), (px + 6, py - 6), color, 2, cv2.LINE_AA)

        label = f"{snapshot.track_id} {lifecycle[:1].upper()}/{kind[:1].upper()}"
        cv2.putText(canvas, label, (px + 11, py - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.39, color, 1, cv2.LINE_AA)

    legend_y = canvas.shape[0] - 30
    cv2.circle(canvas, (24, legend_y - 4), 6, (70, 225, 90), -1, cv2.LINE_AA)
    cv2.putText(canvas, "OBSERVED", (36, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.circle(canvas, (145, legend_y - 4), 7, (70, 225, 90), 2, cv2.LINE_AA)
    cv2.putText(canvas, "PREDICTED", (158, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1, cv2.LINE_AA)
    return canvas


class TrackingDebugVideo:
    def __init__(self, output_path: Path, *, width: int, height: int) -> None:
        self.output_path = output_path
        self.width = width
        self.height = height
        self.writer: cv2.VideoWriter | None = None

    def __call__(self, *, frame: np.ndarray, metadata: Any, spec_by_name: dict[str, Any], tracking_frame: Any, measurements: tuple[Any, ...], record: dict[str, Any]) -> None:
        panorama = render_ultra_panorama(frame, metadata)
        panorama_h, panorama_w = panorama.shape[:2]
        measurement_by_id = {str(item.measurement_id): item for item in measurements}
        panorama_detections = []
        observed_track_ids: set[int] = set()

        for snapshot in tracking_frame.tracks:
            if snapshot.position_kind.value != "observed" or snapshot.measurement_id is None:
                continue
            measurement = measurement_by_id.get(str(snapshot.measurement_id))
            if measurement is None:
                continue
            camera = measurement.metadata.get("representative_camera")
            bbox = measurement.metadata.get("representative_bbox_xyxy")
            spec = spec_by_name.get(camera)
            if spec is None or bbox is None:
                continue
            projected = virtual_bbox_to_panorama_detection(
                metadata, spec, person_id=snapshot.track_id, view_name=str(camera),
                xyxy=tuple(float(value) for value in bbox),
                virtual_width=self.width, virtual_height=self.height,
                panorama_width=panorama_w, panorama_height=panorama_h,
            )
            if projected is not None:
                panorama_detections.append(projected)
                observed_track_ids.add(snapshot.track_id)

        panorama = draw_panorama_detections(panorama, panorama_detections, numbered=True)
        cv2.putText(panorama, "ULTRA PANORAMA  (OBSERVED ONLY)", (18, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(panorama, f"observed IDs={len(observed_track_ids)}  predicted IDs shown in field only", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (235, 235, 235), 1, cv2.LINE_AA)
        field = _draw_field(
            tracking_frame.tracks, metadata,
            timestamp=tracking_frame.timestamp, frame_index=tracking_frame.frame_index,
            measurement_count=record["measurement_count"],
            height=panorama_h,
        )
        output = np.concatenate((panorama, field), axis=1)
        if self.writer is None:
            self.writer = cv2.VideoWriter(
                str(self.output_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                (output.shape[1], output.shape[0]),
            )
            if not self.writer.isOpened():
                raise RuntimeError(f"Could not open video writer: {self.output_path}")
        self.writer.write(output)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


def main() -> None:
    args = parse_args()
    renderer = TrackingDebugVideo(args.out_dir / VIDEO_NAME, width=args.width, height=args.height)
    try:
        run(args, frame_callback=renderer)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
