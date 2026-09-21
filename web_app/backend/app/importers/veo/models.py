"""Data models for sanitized inspection output and in-memory imports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit
from ...match_data import DatasetObservation


def importable(candidate: VeoCandidate) -> bool:
    """Only positively identified complete media files and parsed calibration."""
    path = urlsplit(candidate.url).path.lower()
    if candidate.role == "calibration":
        return path.endswith(".veo") and candidate.field_length is not None and candidate.field_width is not None
    if candidate.role == "ball_tracking" and not (candidate.filename.lower() == "video.mp4" and "initial" in candidate.discovery_phases) and not any(reason in candidate.reasons for reason in ("explicit Standard/Follow video metadata", "authoritative initial player video")):
        return False
    if candidate.role == "panorama" and not (candidate.filename.lower().startswith("transcode-") and "after_interactive" in candidate.discovery_phases):
        return False
    if candidate.role == "camera_directions":
        return path.endswith("detections_constrained.det")
    return candidate.role in {"ball_tracking", "panorama", "veo_highlights"} and path.endswith(".mp4")


@dataclass
class VeoCandidate:
    identifier: str
    url: str
    safe_url: str
    filename: str
    content_type: str | None
    role: str
    confidence: float
    reasons: list[str]
    discovery_phases: list[str] = field(default_factory=list)
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    duration_s: float | None = None
    field_length: float | None = None
    field_width: float | None = None
    calibration_version: str | None = None
    size_bytes: int | None = None
    cookies: list[dict] = field(default_factory=list, repr=False)

    def public_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("url")
        value.pop("cookies")
        return value


@dataclass
class VeoInspection:
    identifier: str
    match_url: str
    safe_match_url: str
    metadata: dict[str, Any] = field(default_factory=dict)
    candidates: list[VeoCandidate] = field(default_factory=list)
    interaction: dict[str, Any] = field(default_factory=dict)
    diagnostic_path: str | None = None
    access_error: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    dataset_observations: list[DatasetObservation] = field(default_factory=list, repr=False)

    def public_dict(self) -> dict[str, Any]:
        resolved = {}
        for role in ("ball_tracking", "panorama", "calibration", "camera_directions", "veo_highlights"):
            choices = sorted((item for item in self.candidates if item.role == role and importable(item)), key=lambda item: item.confidence, reverse=True)
            if choices:
                resolved[role] = choices[0].identifier
        return {
            "id": self.identifier,
            "match_url": self.safe_match_url,
            "metadata": self.metadata,
            "candidates": [candidate.public_dict() for candidate in self.candidates],
            "interaction": self.interaction,
            "diagnostic_path": self.diagnostic_path,
            "access_error": self.access_error,
            "resolved_assets": resolved,
            "ready": all(role in resolved for role in ("ball_tracking", "panorama", "calibration")),
            "manifest": self.manifest,
        }
