"""2D analytical renderer for VeoMatchLab Tracking V5."""

from __future__ import annotations

import cv2
import numpy as np


class OfflinePitchRendererV5:
    width = 1920
    height = 1080

    def __init__(
        self,
        projector,
        *,
        weak_threshold: float = 0.20,
        strong_threshold: float = 0.40,
        width: int = 1920,
        height: int = 1080,
    ) -> None:
        self.projector = projector
        self.width = int(width)
        self.height = int(height)
        self.field_length = float(projector.metadata.field_length)
        self.field_width = float(projector.metadata.field_width)
        self.weak_threshold = float(weak_threshold)
        self.strong_threshold = float(strong_threshold)

    def _panel_geometry(
        self,
        *,
        offset_x: int = 0,
        panel_width: int | None = None,
    ):
        panel_width = self.width if panel_width is None else int(panel_width)
        margin_x = 50
        margin_y = 80
        available_w = panel_width - 2 * margin_x
        available_h = self.height - 2 * margin_y

        pitch_w = available_w
        pitch_h = int(
            round(pitch_w * self.field_width / self.field_length)
        )
        if pitch_h > available_h:
            pitch_h = available_h
            pitch_w = int(
                round(pitch_h * self.field_length / self.field_width)
            )

        x0 = offset_x + (panel_width - pitch_w) // 2
        y0 = (self.height - pitch_h) // 2
        return x0, y0, pitch_w, pitch_h

    def _pixel(self, x: float, z: float, geometry):
        x0, y0, pitch_w, pitch_h = geometry
        px = x0 + int(
            round(
                (x + self.field_length / 2.0)
                / self.field_length
                * pitch_w
            )
        )
        py = y0 + int(
            round(
                (self.field_width / 2.0 - z)
                / self.field_width
                * pitch_h
            )
        )
        return px, py

    def _draw_pitch(self, canvas, geometry):
        x0, y0, pitch_w, pitch_h = geometry
        line = (235, 235, 235)

        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + pitch_w, y0 + pitch_h),
            (35, 90, 35),
            -1,
        )
        cv2.rectangle(
            canvas,
            (x0, y0),
            (x0 + pitch_w, y0 + pitch_h),
            line,
            3,
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
        center = (mid_x, y0 + pitch_h // 2)
        radius = int(
            round(pitch_h * 9.15 / self.field_width)
        )
        cv2.circle(
            canvas,
            center,
            radius,
            line,
            2,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            center,
            4,
            line,
            -1,
            cv2.LINE_AA,
        )

        penalty_depth = int(
            round(pitch_w * 16.5 / self.field_length)
        )
        penalty_width = int(
            round(pitch_h * 40.32 / self.field_width)
        )
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

    @staticmethod
    def _evidence_color(evidence: str):
        # BGR. Technical/debug only: provenance is intentionally visible.
        if evidence == "strong_observed":
            return (0, 220, 255)
        if evidence == "strong_bridge":
            return (80, 220, 80)
        if evidence == "weak_bridge":
            return (220, 120, 255)
        return (255, 180, 0)

    def _draw_analytical_track(self, canvas, geometry, track):
        point = self._pixel(track.x, track.z, geometry)
        color = self._evidence_color(track.evidence)

        cv2.circle(
            canvas,
            point,
            9,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            point,
            11,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(track.identity_id),
            (point[0] + 13, point[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def _draw_raw_observation(self, canvas, geometry, observation):
        point = self._pixel(
            float(observation.world_x),
            float(observation.world_z),
            geometry,
        )
        confidence = float(observation.confidence)

        if confidence >= self.strong_threshold:
            color = (0, 220, 255)
            radius = 7
        elif confidence >= self.weak_threshold:
            color = (220, 120, 255)
            radius = 6
        else:
            color = (120, 120, 120)
            radius = 5

        cv2.circle(
            canvas,
            point,
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            point,
            radius + 2,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def _draw_legend(self, canvas):
        items = [
            ("strong observed", "strong_observed"),
            ("strong bridge", "strong_bridge"),
            ("weak bridge", "weak_bridge"),
            ("inferred", "inferred"),
        ]

        x = 30
        y = self.height - 42
        for label, evidence in items:
            color = self._evidence_color(evidence)
            cv2.circle(
                canvas,
                (x + 7, y - 6),
                6,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                label,
                (x + 20, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                color,
                2,
                cv2.LINE_AA,
            )
            x += 175

    def render_result(self, tracks, *, frame: int, time_s: float):
        canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )
        geometry = self._panel_geometry()
        self._draw_pitch(canvas, geometry)

        for track in tracks:
            self._draw_analytical_track(
                canvas,
                geometry,
                track,
            )

        counts = {
            "strong_observed": 0,
            "strong_bridge": 0,
            "weak_bridge": 0,
            "inferred": 0,
        }
        for track in tracks:
            counts[track.evidence] = counts.get(track.evidence, 0) + 1

        cv2.putText(
            canvas,
            (
                f"V5 ANALYTICAL | frame={frame}  t={time_s:.2f}s  "
                f"visible={len(tracks)}"
            ),
            (30, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.82,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"strong={counts['strong_observed']}  "
                f"strong-bridge={counts['strong_bridge']}  "
                f"weak-bridge={counts['weak_bridge']}  "
                f"inferred={counts['inferred']}"
            ),
            (30, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (215, 215, 215),
            2,
            cv2.LINE_AA,
        )

        self._draw_legend(canvas)
        return canvas

    def render_debug(
        self,
        observations,
        tracks,
        *,
        frame: int,
        time_s: float,
    ):
        canvas = np.zeros(
            (self.height, self.width, 3),
            dtype=np.uint8,
        )
        half = self.width // 2
        left_geometry = self._panel_geometry(
            offset_x=0,
            panel_width=half,
        )
        right_geometry = self._panel_geometry(
            offset_x=half,
            panel_width=half,
        )

        self._draw_pitch(canvas, left_geometry)
        self._draw_pitch(canvas, right_geometry)

        for observation in observations:
            self._draw_raw_observation(
                canvas,
                left_geometry,
                observation,
            )

        for track in tracks:
            self._draw_analytical_track(
                canvas,
                right_geometry,
                track,
            )

        cv2.putText(
            canvas,
            "RAW STRONG + WEAK EVIDENCE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "V5 FINAL ANALYTICAL IDENTITIES",
            (half + 30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            canvas,
            (
                f"frame={frame}  t={time_s:.2f}s  "
                f"raw={len(observations)}  final={len(tracks)}"
            ),
            (30, self.height - 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.64,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )

        return canvas
