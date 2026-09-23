"""Panorama projection at the fixed benchmark resolution."""

from __future__ import annotations

from pathlib import Path

from person_detect.views.panorama_view.renderer import (
    DEFAULT_CALIBRATION,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
)
from person_detect.views.ultra_panorama_view.projection import (
    DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    PanoramaProjector as _CalibratedPanoramaProjector,
)


class PanoramaProjector(_CalibratedPanoramaProjector):
    """Project fixed 3840×2160 panorama pixels using calibrated Veo geometry."""

    def __init__(
        self,
        calibration: str | Path = DEFAULT_CALIBRATION,
        *,
        width: int | None = DEFAULT_WIDTH,
        height: int | None = DEFAULT_HEIGHT,
        far_touchline_margin_m: float = DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    ) -> None:
        super().__init__(
            calibration,
            width=DEFAULT_WIDTH if width is None else width,
            height=DEFAULT_HEIGHT if height is None else height,
            far_touchline_margin_m=far_touchline_margin_m,
        )


__all__ = ["DEFAULT_FAR_TOUCHLINE_MARGIN_M", "PanoramaProjector"]
