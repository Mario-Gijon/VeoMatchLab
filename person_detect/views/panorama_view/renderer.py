"""Fixed 16:9 calibrated panorama renderer.

The implementation delegates to the natural panorama renderer so both
representations use exactly the same Veo geometry and differ only in output
sampling dimensions.
"""

from __future__ import annotations

from pathlib import Path

from person_detect.views.ultra_panorama_view.renderer import (
    DEFAULT_CALIBRATION,
    DEFAULT_VIDEO,
    PanoramaRenderer as _CalibratedPanoramaRenderer,
    read_frame,
)

DEFAULT_WIDTH = 3840
DEFAULT_HEIGHT = 2160
DEFAULT_OUT = Path(__file__).resolve().parents[3] / "person_detect" / "outputs" / "panorama_view" / "diagnostics"


class PanoramaRenderer(_CalibratedPanoramaRenderer):
    """Render the full calibrated panorama into a fixed 3840×2160 canvas."""

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
        *,
        width: int | None = DEFAULT_WIDTH,
        height: int | None = DEFAULT_HEIGHT,
    ) -> None:
        super().__init__(
            calibration,
            width=DEFAULT_WIDTH if width is None else width,
            height=DEFAULT_HEIGHT if height is None else height,
        )


__all__ = [
    "DEFAULT_CALIBRATION",
    "DEFAULT_HEIGHT",
    "DEFAULT_OUT",
    "DEFAULT_VIDEO",
    "DEFAULT_WIDTH",
    "PanoramaRenderer",
    "read_frame",
]
