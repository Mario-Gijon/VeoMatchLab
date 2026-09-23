"""Small metric-only pitch renderer for tracking V1 diagnostics."""

from __future__ import annotations

import cv2
import numpy as np


class PitchRenderer:
    """Render metric X/Z positions without depending on camera imagery."""

    width = 1920
    height = 1080

    def __init__(self, projector, *, width: int = 1920, height: int = 1080) -> None:
        self.projector = projector
        self.width = int(width)
        self.height = int(height)
        self.field_length = float(projector.metadata.field_length)
        self.field_width = float(projector.metadata.field_width)

    def _geometry(self, *, top: int = 0, bottom: int | None = None):
        bottom = self.height if bottom is None else int(bottom)
        available_h = max(1, bottom - top)
        margin = 70
        pitch_h = available_h - 2 * margin
        pitch_w = int(round(pitch_h * self.field_length / self.field_width))
        max_w = self.width - 2 * margin
        if pitch_w > max_w:
            pitch_w = max_w
            pitch_h = int(round(pitch_w * self.field_width / self.field_length))
        x0 = (self.width - pitch_w) // 2
        y0 = top + (available_h - pitch_h) // 2
        return x0, y0, pitch_w, pitch_h

    def _pixel(self, x: float, z: float, geometry) -> tuple[int, int]:
        x0, y0, pitch_w, pitch_h = geometry
        px = x0 + int(round((float(x) + self.field_length / 2) / self.field_length * pitch_w))
        py = y0 + int(round((self.field_width / 2 - float(z)) / self.field_width * pitch_h))
        return px, py

    def _draw_pitch(self, canvas: np.ndarray, geometry) -> None:
        x0, y0, pitch_w, pitch_h = geometry
        line = (235, 235, 235)
        cv2.rectangle(canvas, (x0, y0), (x0 + pitch_w, y0 + pitch_h), (35, 90, 35), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + pitch_w, y0 + pitch_h), line, 3, cv2.LINE_AA)
        mid_x = x0 + pitch_w // 2
        cv2.line(canvas, (mid_x, y0), (mid_x, y0 + pitch_h), line, 2, cv2.LINE_AA)
        center = (mid_x, y0 + pitch_h // 2)
        cv2.circle(canvas, center, int(round(pitch_h * 9.15 / self.field_width)), line, 2, cv2.LINE_AA)
        cv2.circle(canvas, center, 4, line, -1, cv2.LINE_AA)
        penalty_depth = int(round(pitch_w * 16.5 / self.field_length))
        penalty_width = int(round(pitch_h * 40.32 / self.field_width))
        penalty_y = y0 + (pitch_h - penalty_width) // 2
        cv2.rectangle(canvas, (x0, penalty_y), (x0 + penalty_depth, penalty_y + penalty_width), line, 2, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0 + pitch_w - penalty_depth, penalty_y), (x0 + pitch_w, penalty_y + penalty_width), line, 2, cv2.LINE_AA)

    def _draw_marker(self, canvas, geometry, x, z, label, *, observed: bool) -> None:
        point = self._pixel(x, z, geometry)
        color = (0, 220, 255) if observed else (255, 180, 0)
        cv2.circle(canvas, point, 9, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 11, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, label, (point[0] + 13, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    def render_result(self, tracks, *, frame: int, time_s: float) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        geometry = self._geometry()
        self._draw_pitch(canvas, geometry)
        for track in tracks:
            self._draw_marker(canvas, geometry, track.x, track.z, str(track.track_id), observed=track.frame_state == "observed")
        cv2.putText(canvas, f"frame={frame}  t={time_s:.2f}s  visible={len(tracks)}", (30, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, "OBSERVED" , (30, self.height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "PREDICTED", (170, self.height - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 180, 0), 2, cv2.LINE_AA)
        return canvas

    def render_debug(self, observations, tracks, *, frame: int, time_s: float) -> np.ndarray:
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        half = self.width // 2
        left = self._geometry(bottom=self.height)
        # Scale the same metric mapping into each half, preserving field aspect ratio.
        def panel_geometry(offset):
            available_w = half - 40
            available_h = self.height - 110
            pitch_h = min(available_h, int(round(available_w * self.field_width / self.field_length)))
            pitch_w = int(round(pitch_h * self.field_length / self.field_width))
            return offset + (half - pitch_w) // 2, (self.height - pitch_h) // 2, pitch_w, pitch_h
        for offset in (0, half):
            geom = panel_geometry(offset)
            self._draw_pitch(canvas, geom)
        for observation in observations:
            self._draw_marker(canvas, panel_geometry(0), observation.world_x, observation.world_z, str(observation.detection_index), observed=True)
        for track in tracks:
            self._draw_marker(canvas, panel_geometry(half), track.x, track.z, str(track.track_id), observed=track.frame_state == "observed")
        cv2.putText(canvas, "RAW OBSERVATIONS", (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, "TRACKED V1", (half + 30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (240, 240, 240), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"frame={frame}  t={time_s:.2f}s", (30, self.height - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
        return canvas
