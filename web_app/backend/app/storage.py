"""Small JSON-backed workspace store for the local-only developer app."""

from __future__ import annotations

import json
import threading
import uuid
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


class WorkspaceStore:
    """Persist workspace metadata, while media itself stays in referenced files."""

    def __init__(self, runtime_root: Path) -> None:
        self.runtime_root = runtime_root
        self.state_path = runtime_root / "workspaces.json"
        self._lock = threading.RLock()
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        if not self.state_path.exists():
            self._write({"matches": []})
        # One-time metadata migration. Source assets and the legacy manifest remain
        # untouched so older workspaces are recoverable without copying videos.
        marker = runtime_root / "matches" / ".migrated"
        if not marker.exists():
            for match in self._read()["matches"]:
                path = self.match_dir(match["id"]) / "match.json"
                if not path.exists():
                    self._save_match(match)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    def match_dir(self, match_id: str) -> Path:
        if not re.fullmatch(r"match_[a-zA-Z0-9_-]+", match_id):
            raise ValueError("Invalid match identifier")
        directory = self.runtime_root / "matches" / match_id
        parent = self.runtime_root / "matches"
        if parent.is_symlink() or directory.is_symlink() or directory.resolve().parent != parent.resolve():
            raise ValueError("Match directory is outside managed storage")
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def asset_dir(self, match_id: str, category: str) -> Path:
        if category not in {"media", "frames", "calibration", "cache", "diagnostics", "clips", "analysis", "datasets", "artifacts"}:
            raise ValueError("Invalid asset category")
        path = self.match_dir(match_id) / category
        if path.is_symlink() or path.resolve().parent != self.match_dir(match_id).resolve():
            raise ValueError("Asset directory is outside managed storage")
        path.mkdir(exist_ok=True)
        return path

    def _save_match(self, match: dict) -> None:
        path = self.match_dir(match["id"]) / "match.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(match, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def now() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid.uuid4().hex[:12]}"

    def _read(self) -> dict[str, Any]:
        with self.state_path.open("r", encoding="utf-8") as source:
            return json.load(source)

    def _write(self, state: dict[str, Any]) -> None:
        temporary = self.state_path.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as destination:
            json.dump(state, destination, indent=2, sort_keys=True)
        temporary.replace(self.state_path)

    def matches(self) -> list[dict[str, Any]]:
        with self._lock:
            return [json.loads(path.read_text()) for path in sorted((self.runtime_root / "matches").glob("match_*/match.json"))]

    def match(self, match_id: str) -> dict[str, Any] | None:
        return next((item for item in self.matches() if item["id"] == match_id), None)

    def create_match(self, name: str, description: str, calibration: dict[str, Any] | None) -> dict[str, Any]:
        with self._lock:
            item = {
                "id": self.new_id("match"),
                "name": name,
                "description": description,
                "created_at": self.now(),
                "updated_at": self.now(),
                "calibration": calibration,
                "media": [],
                "metadata": {},
                "artifacts": [],
                "clips": [],
            }
            self._save_match(item)
            return item

    def update_match(self, match_id: str, callback) -> dict[str, Any] | None:
        with self._lock:
            item = self.match(match_id)
            if item is None:
                return None
            callback(item)
            item["updated_at"] = self.now()
            self._save_match(item)
            return item

    def clips(self, match_id: str) -> list[dict]:
        match = self.match(match_id)
        return list(match.get("clips", [])) if match else []

    def clip(self, match_id: str, clip_id: str) -> dict | None:
        return next((item for item in self.clips(match_id) if item["id"] == clip_id), None)

    def create_clip(self, match_id: str, source_media_id: str, start_s: float, end_s: float, name: str) -> dict | None:
        clip = {"id": self.new_id("clip"), "source_media_id": source_media_id,
                "start_s": start_s, "end_s": end_s, "name": name,
                "created_at": self.now(), "analysis": None}
        result = self.update_match(match_id, lambda item: item.setdefault("clips", []).append(clip))
        return clip if result else None

    def update_clip(self, match_id: str, clip_id: str, callback) -> dict | None:
        updated = None
        def apply(match):
            nonlocal updated
            for clip in match.setdefault("clips", []):
                if clip["id"] == clip_id:
                    callback(clip)
                    updated = clip
                    break
        self.update_match(match_id, apply)
        return updated

    def delete_clip(self, match_id: str, clip_id: str) -> bool:
        match = self.match(match_id)
        if match is None or not any(item["id"] == clip_id for item in match.get("clips", [])):
            return False
        self.update_match(match_id, lambda item: item.update(clips=[value for value in item.get("clips", []) if value["id"] != clip_id]))
        return True

    def media(self, media_id: str) -> tuple[dict[str, Any], dict[str, Any]] | None:
        for match in self.matches():
            item = next((value for value in match["media"] if value["id"] == media_id), None)
            if item is not None:
                return match, item
        return None

    def _remove_managed_file(self, match_id: str, resource: dict | None) -> None:
        if not resource or resource.get("ownership", "managed" if resource.get("source") in {"upload", "veo_import", "extracted"} else "external") != "managed":
            return
        path = Path(resource["path"])
        root = self.match_dir(match_id).resolve()
        # Even legacy metadata may only authorize deletion inside this match.
        if path.resolve().is_relative_to(root) and path.is_file():
            path.unlink()

    def attach_media(self, match_id: str, record: dict) -> dict | None:
        with self._lock:
            old = []
            def update(item):
                if record.get("role") in {"tracking", "dual_lens", "camera_directions"}:
                    old.extend(value for value in item["media"] if value.get("role") == record["role"] and value["id"] != record["id"])
                item["media"] = [value for value in item["media"] if value not in old and value["id"] != record["id"]] + [record]
            result = self.update_match(match_id, update)
            for resource in old:
                if resource["path"] != record["path"]:
                    self._remove_managed_file(match_id, resource)
            return result

    def set_calibration(self, match_id: str, calibration: dict | None) -> dict | None:
        with self._lock:
            match = self.match(match_id)
            if match is None:
                return None
            result = self.update_match(match_id, lambda item: item.update(calibration=calibration))
            if match.get("calibration") and (not calibration or match["calibration"]["path"] != calibration["path"]):
                self._remove_managed_file(match_id, match["calibration"])
            return result

    def remove_media(self, match_id: str, media_id: str) -> None:
        with self._lock:
            match = self.match(match_id)
            if match is None:
                return
            resource = next((item for item in match["media"] if item["id"] == media_id), None)
            self.update_match(match_id, lambda item: item.update(media=[value for value in item["media"] if value["id"] != media_id]))
            self._remove_managed_file(match_id, resource)

    def delete_match(self, match_id: str) -> bool:
        with self._lock:
            if self.match(match_id) is None:
                return False
            directory = self.match_dir(match_id)
            if directory.is_symlink() or directory.resolve().parent != (self.runtime_root / "matches").resolve():
                raise ValueError("Match directory is outside managed storage")
            # No paths from resource metadata are traversed. rmtree does not follow
            # child symlinks, so referenced input files and output/ are untouched.
            shutil.rmtree(directory)
            return True
