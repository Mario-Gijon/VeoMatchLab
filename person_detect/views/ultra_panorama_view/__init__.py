"""Natural-aspect calibrated Veo panorama benchmark view."""

from .projection import PanoramaProjector
from .renderer import PanoramaRenderer

UltraPanoramaProjector = PanoramaProjector
UltraPanoramaRenderer = PanoramaRenderer

__all__ = [
    "PanoramaProjector",
    "PanoramaRenderer",
    "UltraPanoramaProjector",
    "UltraPanoramaRenderer",
]
