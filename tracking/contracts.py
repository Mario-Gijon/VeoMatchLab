"""Shared contracts for Tracking V1 artifacts and pipeline stages.

This module deliberately contains no filesystem, detector, tracker, frontend or
legacy V0 dependencies.  It defines only the stable data contract between
Tracking V1 stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Mapping


class TrackingStage(str, Enum):
    DETECTIONS = "detections"
    TRACKLETS = "tracklets"
    APPEARANCE = "appearance"
    IDENTITIES = "identities"
    FINAL = "final"


class StageStatus(str, Enum):
    NOT_RUN = "not_run"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


def _json_safe_dict(name: str, value: Mapping[str, Any] | None) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain JSON-serializable values") from exc
    return result


def _optional_relative_path(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty relative path or None")

    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{name} must stay relative to the Tracking V1 artifact root")
    return path.as_posix()


def _stage_name(value: TrackingStage | str) -> str:
    if isinstance(value, TrackingStage):
        return value.value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("stage name must be a non-empty string")
    return value.strip()


def _default_stages() -> dict[str, "StageManifest"]:
    return {stage.value: StageManifest() for stage in TrackingStage}


@dataclass(slots=True)
class StageManifest:
    """Manifest entry describing the publication state of one V1 stage."""

    status: StageStatus = StageStatus.NOT_RUN
    artifact: str | None = None
    summary: str | None = None
    version: str | int | None = None
    updated_at: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, StageStatus):
            try:
                self.status = StageStatus(self.status)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid stage status: {self.status!r}") from exc

        self.artifact = _optional_relative_path("artifact", self.artifact)
        self.summary = _optional_relative_path("summary", self.summary)

        if isinstance(self.version, bool) or (
            self.version is not None and not isinstance(self.version, (str, int))
        ):
            raise ValueError("version must be str, int, or None")
        if isinstance(self.version, str):
            if not self.version.strip():
                raise ValueError("version string must not be empty")
            self.version = self.version.strip()

        if self.updated_at is not None:
            if not isinstance(self.updated_at, str) or not self.updated_at.strip():
                raise ValueError("updated_at must be a non-empty string or None")
            self.updated_at = self.updated_at.strip()

        self.config = _json_safe_dict("config", self.config)
        self.metadata = _json_safe_dict("metadata", self.metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "artifact": self.artifact,
            "summary": self.summary,
            "version": self.version,
            "updated_at": self.updated_at,
            "config": dict(self.config),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "StageManifest":
        return cls(
            status=payload.get("status", StageStatus.NOT_RUN.value),
            artifact=payload.get("artifact"),
            summary=payload.get("summary"),
            version=payload.get("version"),
            updated_at=payload.get("updated_at"),
            config=payload.get("config", {}),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class TrackingManifest:
    """Top-level manifest for one clip's Tracking V1 artifacts."""

    match_id: str
    clip_id: str
    schema_version: int = 1
    stages: dict[str, StageManifest] = field(default_factory=_default_stages)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.match_id, str) or not self.match_id.strip():
            raise ValueError("match_id must be a non-empty string")
        if not isinstance(self.clip_id, str) or not self.clip_id.strip():
            raise ValueError("clip_id must be a non-empty string")
        self.match_id = self.match_id.strip()
        self.clip_id = self.clip_id.strip()

        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version < 1
        ):
            raise ValueError("schema_version must be an integer >= 1")

        incoming = self.stages
        if not isinstance(incoming, Mapping):
            raise ValueError("stages must be a mapping")

        normalized: dict[str, StageManifest] = {}
        for raw_name, raw_manifest in incoming.items():
            name = _stage_name(raw_name)
            manifest = (
                raw_manifest
                if isinstance(raw_manifest, StageManifest)
                else StageManifest.from_dict(raw_manifest)
            )
            normalized[name] = manifest

        # Every known V1 stage always exists in the manifest, even before it is run.
        for known_stage in TrackingStage:
            normalized.setdefault(known_stage.value, StageManifest())

        self.stages = normalized
        self.metadata = _json_safe_dict("metadata", self.metadata)

    def stage(self, stage: TrackingStage | str) -> StageManifest:
        name = _stage_name(stage)
        try:
            return self.stages[name]
        except KeyError as exc:
            raise KeyError(f"stage {name!r} is not present in this manifest") from exc

    def set_stage(self, stage: TrackingStage | str, manifest: StageManifest) -> None:
        if not isinstance(manifest, StageManifest):
            raise TypeError("manifest must be a StageManifest")
        self.stages[_stage_name(stage)] = manifest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "match_id": self.match_id,
            "clip_id": self.clip_id,
            "stages": {name: manifest.to_dict() for name, manifest in self.stages.items()},
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackingManifest":
        raw_stages = payload.get("stages", {})
        if not isinstance(raw_stages, Mapping):
            raise ValueError("stages must be a mapping")
        return cls(
            schema_version=payload.get("schema_version", 1),
            match_id=payload["match_id"],
            clip_id=payload["clip_id"],
            stages={
                str(name): (
                    manifest
                    if isinstance(manifest, StageManifest)
                    else StageManifest.from_dict(manifest)
                )
                for name, manifest in raw_stages.items()
            },
            metadata=payload.get("metadata", {}),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )

    @classmethod
    def from_json(cls, raw: str) -> "TrackingManifest":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("TrackingManifest JSON must encode an object")
        return cls.from_dict(payload)
