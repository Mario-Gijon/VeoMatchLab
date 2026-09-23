"""2D renderer for clip-global offline tracking."""

from __future__ import annotations

import cv2
import numpy as np


class OfflinePitchRenderer:
    width = 1920
    height = 1080

    def __init__(self, projector, *, width: int = 1920, height: int = 1080) -> None:
        self.projector = projector
        self.width = int(width)
        self.height = int(height)
        self.field_length = float(projector.metadata.field_length)
        self.field_width = float(projector.metadata.field_width)

    def _panel_geometry(self, *, offset_x: int = 0, panel_width: int | None = None):
        panel_width = self.width if panel_width is None else int(panel_width)
        margin_x = 50
        margin_y = 80
        available_w = panel_width - 2 * margin_x
        available_h = self.height - 2 * margin_y

        pitch_w = available_w
        pitch_h = int(round(pitch_w * self.field_width / self.field_length))
        if pitch_h > available_h:
            pitch_h = available_h
            pitch_w = int(round(pitch_h * self.field_length / self.field_width))

        x0 = offset_x + (panel_width - pitch_w) // 2
        y0 = (self.height - pitch_h) // 2
        return x0, y0, pitch_w, pitch_h

    def _pixel(self, x: float, z: float, geometry):
        x0, y0, pitch_w, pitch_h = geometry
        px = x0 + int(round((x + self.field_length / 2) / self.field_length * pitch_w))
        py = y0 + int(round((self.field_width / 2 - z) / self.field_width * pitch_h))
        return px, py

    def _draw_pitch(self, canvas, geometry):
        x0, y0, pitch_w, pitch_h = geometry
        line = (235, 235, 235)
        cv2.rectangle(canvas, (x0, y0), (x0 + pitch_w, y0 + pitch_h), (35, 90, 35), -1)
        cv2.rectangle(canvas, (x0, y0), (x0 + pitch_w, y0 + pitch_h), line, 3, cv2.LINE_AA)

        mid_x = x0 + pitch_w // 2
        cv2.line(canvas, (mid_x, y0), (mid_x, y0 + pitch_h), line, 2, cv2.LINE_AA)
        center = (mid_x, y0 + pitch_h // 2)
        radius = int(round(pitch_h * 9.15 / self.field_width))
        cv2.circle(canvas, center, radius, line, 2, cv2.LINE_AA)
        cv2.circle(canvas, center, 4, line, -1, cv2.LINE_AA)

        penalty_depth = int(round(pitch_w * 16.5 / self.field_length))
        penalty_width = int(round(pitch_h * 40.32 / self.field_width))
        penalty_y = y0 + (pitch_h - penalty_width) // 2
        cv2.rectangle(
            canvas,
            (x0, penalty_y),
            (x0 + penalty_depth, penalty_y + penalty_width),
            line,
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(
            canvas,
            (x0 + pitch_w - penalty_depth, penalty_y),
            (x0 + pitch_w, penalty_y + penalty_width),
            line,
            2,
            cv2.LINE_AA,
        )

    def _draw_track(self, canvas, geometry, track):
        point = self._pixel(track.x, track.z, geometry)
        if track.state == "observed":
            color = (0, 220, 255)
        elif track.state == "interpolated":
            color = (255, 180, 0)
        else:
            color = (180, 180, 180)

        cv2.circle(canvas, point, 9, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 11, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(track.trajectory_id),
            (point[0] + 13, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_raw(self, canvas, geometry, observation):
        point = self._pixel(observation.world_x, observation.world_z, geometry)
        cv2.circle(canvas, point, 7, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, 9, (0, 0, 0), 2, cv2.LINE_AA)

    def render_result(self, tracks, *, frame: int, time_s: float):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        geometry = self._panel_geometry()
        self._draw_pitch(canvas, geometry)

        for track in tracks:
            self._draw_track(canvas, geometry, track)

        cv2.putText(
            canvas,
            f"OFFLINE CONTEXT | frame={frame}  t={time_s:.2f}s  visible={len(tracks)}",
            (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "observed",
            (30, self.height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "interpolated",
            (150, self.height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 180, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "short extrapolation",
            (335, self.height - 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (180, 180, 180),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def render_debug(self, observations, tracks, *, frame: int, time_s: float):
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        half = self.width // 2
        left_geometry = self._panel_geometry(offset_x=0, panel_width=half)
        right_geometry = self._panel_geometry(offset_x=half, panel_width=half)
        self._draw_pitch(canvas, left_geometry)
        self._draw_pitch(canvas, right_geometry)

        for observation in observations:
            self._draw_raw(canvas, left_geometry, observation)
        for track in tracks:
            self._draw_track(canvas, right_geometry, track)

        cv2.putText(
            canvas,
            "RAW OBSERVATIONS",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "OFFLINE GLOBAL CONTEXT",
            (half + 30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.85,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"frame={frame}  t={time_s:.2f}s",
            (30, self.height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        return canvas
