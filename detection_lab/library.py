"""Read-only Detection Lab view of the existing workspace match library."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from web_app.backend.app.storage import WorkspaceStore


def _file_available(resource: dict[str, Any] | None) -> bool:
    """Treat absent or stale manifest references as unavailable, never as errors."""
    if not resource or not resource.get("path"):
        return False
    return Path(str(resource["path"])).is_file()


def _media_for_role(match: dict[str, Any], role: str) -> dict[str, Any] | None:
    return next((item for item in match.get("media", []) if item.get("role") == role), None)


class MatchLibrary:
    """Expose only the small read-only subset needed by Detection Lab."""

    def __init__(self, store: WorkspaceStore) -> None:
        self.store = store

    @staticmethod
    def title(match: dict[str, Any]) -> str:
        metadata = match.get("metadata") if isinstance(match.get("metadata"), dict) else {}
        return str(match.get("name") or metadata.get("title") or match["id"])

    def summary(self, match: dict[str, Any]) -> dict[str, str]:
        return {"id": str(match["id"]), "title": self.title(match)}

    def matches(self) -> list[dict[str, str]]:
        return [self.summary(match) for match in self.store.matches()]

    def resource_path(self, match_id: str, resource: dict[str, Any] | None, role: str) -> Path | None:
        """Resolve a manifest resource, including legacy managed-path records.

        Veo imports have a canonical managed name inside the existing match
        directory. This fallback only applies to managed resources whose old
        absolute manifest path is no longer valid; it never copies or alters a
        manifest and it does not apply to external references.
        """
        if _file_available(resource):
            return Path(str(resource["path"]))
        if not resource or resource.get("ownership") != "managed":
            return None
        relative_paths = {
            "dual_lens": Path("media/dual_lens.mp4"),
            "camera_directions": Path("media/detections_constrained.det"),
            "calibration": Path("calibration/camera_alignment.veo"),
        }
        relative_path = relative_paths.get(role)
        if relative_path is None:
            return None
        candidate = self.store.match_dir(match_id) / relative_path
        return candidate if candidate.is_file() else None

    def detail(self, match_id: str) -> dict[str, Any] | None:
        match = self.store.match(match_id)
        if match is None:
            return None
        dual_lens = _media_for_role(match, "dual_lens")
        camera_directions = _media_for_role(match, "camera_directions")
        calibration = match.get("calibration") if isinstance(match.get("calibration"), dict) else None
        has_structured_metadata = bool(match.get("structured_data"))
        video_available = self.resource_path(match_id, dual_lens, "dual_lens") is not None
        return {
            **self.summary(match),
            "video_url": f"/api/matches/{match_id}/dual-lens" if video_available else None,
            "assets": [
                {"label": "Vídeo de doble lente", "available": video_available},
                {"label": "Calibración (.veo)", "available": self.resource_path(match_id, calibration, "calibration") is not None},
                {"label": "Direcciones de cámara (.det)", "available": self.resource_path(match_id, camera_directions, "camera_directions") is not None},
                {"label": "Metadatos estructurados", "available": has_structured_metadata},
            ],
        }

    def dual_lens_path(self, match_id: str) -> Path | None:
        """Return the manifest-referenced video only when it is still readable."""
        match = self.store.match(match_id)
        if match is None:
            return None
        return self.resource_path(match_id, _media_for_role(match, "dual_lens"), "dual_lens")
