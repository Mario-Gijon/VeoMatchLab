"""Filesystem-backed artifact store for the clean Tracking V1 pipeline.

The store is deliberately independent from ``web_app`` and ``tracking_v0``.
It owns the small, stable publication contract for one clip:

    data/matches/<match_id>/clips/<clip_id>/tracking/v1/
        manifest.json
        detections/
        tracklets/
        appearance/
        identities/
        final/

Stage directories are created lazily.  A stage is visible to consumers only
when its manifest status is ``completed`` and its referenced files exist.

Correctness rules
-----------------
* Writes of JSON/JSONL files are atomic (temporary file + replace).
* A stage must be ``running`` before it can receive artifacts.
* Completing a stage validates that the referenced artifact/summary exist.
* Re-running an upstream stage invalidates all downstream stages.
* Re-running ``tracklets`` never invalidates ``detections``.
* Partial files from a failed run are never referenced by a completed manifest.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Iterator, Mapping

from .contracts import (
    StageManifest,
    StageStatus,
    TrackingManifest,
    TrackingStage,
)


_MATCH_ID_RE = re.compile(r"match_[A-Za-z0-9_-]+")
_CLIP_ID_RE = re.compile(r"clip_[A-Za-z0-9_-]+")

_STAGE_ORDER: tuple[TrackingStage, ...] = (
    TrackingStage.DETECTIONS,
    TrackingStage.TRACKLETS,
    TrackingStage.APPEARANCE,
    TrackingStage.IDENTITIES,
    TrackingStage.FINAL,
)

_STAGE_PREREQUISITE: dict[TrackingStage, TrackingStage | None] = {
    TrackingStage.DETECTIONS: None,
    TrackingStage.TRACKLETS: TrackingStage.DETECTIONS,
    TrackingStage.APPEARANCE: TrackingStage.TRACKLETS,
    TrackingStage.IDENTITIES: TrackingStage.APPEARANCE,
    TrackingStage.FINAL: TrackingStage.IDENTITIES,
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _coerce_stage(stage: TrackingStage | str) -> TrackingStage:
    if isinstance(stage, TrackingStage):
        return stage
    try:
        return TrackingStage(stage)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unknown Tracking V1 stage: {stage!r}") from exc


def _downstream_stages(stage: TrackingStage) -> tuple[TrackingStage, ...]:
    index = _STAGE_ORDER.index(stage)
    return _STAGE_ORDER[index + 1 :]


def _normalize_relative_path(value: str | Path, *, label: str) -> PurePosixPath:
    raw = str(value).strip().replace("\\", "/")
    if not raw:
        raise ValueError(f"{label} must be a non-empty relative path")

    path = PurePosixPath(raw)
    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(f"{label} must stay inside the Tracking V1 artifact root")
    if not path.parts:
        raise ValueError(f"{label} must be a non-empty relative path")
    return path


def _json_line(record: Any) -> str:
    if isinstance(record, Mapping):
        payload = dict(record)
    else:
        to_dict = getattr(record, "to_dict", None)
        if not callable(to_dict):
            raise TypeError("JSONL records must be mappings or expose to_dict()")
        payload = to_dict()
        if not isinstance(payload, Mapping):
            raise TypeError("record.to_dict() must return a mapping")

    return json.dumps(
        dict(payload),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )


@dataclass(frozen=True, slots=True)
class ArtifactFile:
    """Information about one file atomically published inside a V1 stage."""

    ref: str
    path: Path
    size_bytes: int
    sha256: str
    record_count: int | None = None


class TrackingArtifactStore:
    """Manage the current Tracking V1 artifacts for one clip.

    ``data_root`` is the workspace data directory itself (normally ``data``),
    not the repository root.

    The store never creates missing matches or clips.  The clip directory must
    already exist unless ``require_clip_exists=False`` is explicitly requested
    for isolated tests.
    """

    def __init__(
        self,
        data_root: str | Path,
        match_id: str,
        clip_id: str,
        *,
        require_clip_exists: bool = True,
    ) -> None:
        if not isinstance(match_id, str) or _MATCH_ID_RE.fullmatch(match_id) is None:
            raise ValueError(f"invalid match_id: {match_id!r}")
        if not isinstance(clip_id, str) or _CLIP_ID_RE.fullmatch(clip_id) is None:
            raise ValueError(f"invalid clip_id: {clip_id!r}")

        self.data_root = Path(data_root).expanduser()
        self.match_id = match_id
        self.clip_id = clip_id
        self.matches_root = self.data_root / "matches"
        self.match_dir = self.matches_root / match_id
        self.clip_dir = self.match_dir / "clips" / clip_id
        self.root = self.clip_dir / "tracking" / "v1"
        self.manifest_path = self.root / "manifest.json"
        self._lock = threading.RLock()

        if require_clip_exists and not self.clip_dir.is_dir():
            raise FileNotFoundError(f"clip directory does not exist: {self.clip_dir}")

        self._validate_managed_paths()

    @classmethod
    def for_clip(
        cls,
        data_root: str | Path,
        match_id: str,
        clip_id: str,
        *,
        require_clip_exists: bool = True,
    ) -> "TrackingArtifactStore":
        return cls(
            data_root=data_root,
            match_id=match_id,
            clip_id=clip_id,
            require_clip_exists=require_clip_exists,
        )

    def _validate_managed_paths(self) -> None:
        for path, label in (
            (self.match_dir, "match directory"),
            (self.clip_dir, "clip directory"),
            (self.clip_dir / "tracking", "tracking directory"),
            (self.root, "Tracking V1 root"),
        ):
            if path.exists() and path.is_symlink():
                raise ValueError(f"{label} must not be a symlink: {path}")

        matches_root = self.matches_root.resolve(strict=False)
        match_dir = self.match_dir.resolve(strict=False)
        if match_dir.parent != matches_root:
            raise ValueError("match directory is outside the managed data root")

        clips_root = (self.match_dir / "clips").resolve(strict=False)
        clip_dir = self.clip_dir.resolve(strict=False)
        if clip_dir.parent != clips_root:
            raise ValueError("clip directory is outside the managed match")

    def _ensure_root(self) -> None:
        self._validate_managed_paths()
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink():
            raise ValueError("Tracking V1 root must not be a symlink")

    def _atomic_write_bytes(self, path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as destination:
                destination.write(content)
                destination.flush()
                os.fsync(destination.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _write_manifest(self, manifest: TrackingManifest) -> None:
        if manifest.match_id != self.match_id or manifest.clip_id != self.clip_id:
            raise ValueError("manifest identifiers do not match this artifact store")
        self._ensure_root()
        content = (manifest.to_json(indent=2) + "\n").encode("utf-8")
        self._atomic_write_bytes(self.manifest_path, content)

    def manifest_exists(self) -> bool:
        return self.manifest_path.is_file()

    def load_manifest(self) -> TrackingManifest:
        """Load the current manifest without creating filesystem state."""
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Tracking V1 manifest does not exist: {self.manifest_path}")

        manifest = TrackingManifest.from_json(self.manifest_path.read_text(encoding="utf-8"))
        if manifest.match_id != self.match_id or manifest.clip_id != self.clip_id:
            raise ValueError("manifest identifiers do not match its clip directory")
        return manifest

    def ensure_manifest(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrackingManifest:
        """Create the initial manifest if necessary; never creates stage dirs."""
        with self._lock:
            if self.manifest_path.is_file():
                return self.load_manifest()

            manifest_metadata = dict(metadata or {})
            manifest_metadata.setdefault("created_at", _utc_now())
            manifest = TrackingManifest(
                match_id=self.match_id,
                clip_id=self.clip_id,
                metadata=manifest_metadata,
            )
            self._write_manifest(manifest)
            return manifest

    def _stage_dir(self, stage: TrackingStage) -> Path:
        path = self.root / stage.value
        if path.exists() and path.is_symlink():
            raise ValueError(f"stage directory must not be a symlink: {path}")
        resolved = path.resolve(strict=False)
        root = self.root.resolve(strict=False)
        if resolved.parent != root:
            raise ValueError("stage directory is outside the Tracking V1 root")
        return path

    def _remove_stage_dir(self, stage: TrackingStage) -> None:
        path = self._stage_dir(stage)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            raise ValueError(f"stage path is not a directory: {path}")

    def _stage_relative_ref(
        self,
        stage: TrackingStage,
        relative_path: str | Path,
    ) -> str:
        path = _normalize_relative_path(relative_path, label="stage artifact path")
        if path.parts[0] == stage.value:
            # Accept already-canonical refs such as detections/frames.jsonl.
            canonical = path
        else:
            canonical = PurePosixPath(stage.value) / path
        if canonical.parts[0] != stage.value:
            raise ValueError(f"artifact must live inside stage {stage.value!r}")
        return canonical.as_posix()

    def _resolve_ref(self, ref: str | Path) -> Path:
        relative = _normalize_relative_path(ref, label="artifact reference")
        candidate = self.root.joinpath(*relative.parts)
        root = self.root.resolve(strict=False)
        resolved = candidate.resolve(strict=False)
        if resolved != root and root not in resolved.parents:
            raise ValueError("artifact reference escapes the Tracking V1 root")
        return candidate

    def _require_stage_running(
        self,
        stage: TrackingStage,
        manifest: TrackingManifest | None = None,
    ) -> TrackingManifest:
        current = manifest or self.load_manifest()
        status = current.stage(stage).status
        if status is not StageStatus.RUNNING:
            raise RuntimeError(
                f"stage {stage.value!r} must be running before writing artifacts "
                f"(current status: {status.value})"
            )
        return current

    def _check_prerequisite(self, stage: TrackingStage, manifest: TrackingManifest) -> None:
        prerequisite = _STAGE_PREREQUISITE[stage]
        if prerequisite is None:
            return
        status = manifest.stage(prerequisite).status
        if status is not StageStatus.COMPLETED:
            raise RuntimeError(
                f"stage {stage.value!r} requires completed stage "
                f"{prerequisite.value!r}; current status is {status.value!r}"
            )

    def begin_stage(
        self,
        stage: TrackingStage | str,
        *,
        version: str | int | None = None,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        invalidate_downstream: bool = True,
        clean_outputs: bool = True,
        restart: bool = False,
    ) -> TrackingManifest:
        """Transition one stage to RUNNING.

        Starting an upstream stage invalidates all later stages by default.
        Existing files for the stage/downstream stages are removed only after the
        manifest stops referencing them.  If cleanup fails, stale files may
        remain on disk but are never advertised as completed artifacts.
        """
        selected = _coerce_stage(stage)

        with self._lock:
            manifest = self.ensure_manifest()
            current = manifest.stage(selected)
            if current.status is StageStatus.RUNNING and not restart:
                raise RuntimeError(
                    f"stage {selected.value!r} is already running; pass restart=True "
                    "only when intentionally recovering a stale run"
                )

            self._check_prerequisite(selected, manifest)
            now = _utc_now()

            manifest.set_stage(
                selected,
                StageManifest(
                    status=StageStatus.RUNNING,
                    version=version,
                    updated_at=now,
                    config=dict(config or {}),
                    metadata=dict(metadata or {}),
                ),
            )

            invalidated = _downstream_stages(selected) if invalidate_downstream else ()
            for downstream in invalidated:
                manifest.set_stage(
                    downstream,
                    StageManifest(
                        status=StageStatus.NOT_RUN,
                        updated_at=now,
                        metadata={"invalidated_by": selected.value},
                    ),
                )

            # Publish state first so consumers stop using any previous outputs.
            self._write_manifest(manifest)

            if clean_outputs:
                self._remove_stage_dir(selected)
            for downstream in invalidated:
                self._remove_stage_dir(downstream)

            return manifest

    def _write_stage_bytes(
        self,
        stage: TrackingStage,
        relative_path: str | Path,
        content: bytes,
        *,
        record_count: int | None = None,
    ) -> ArtifactFile:
        self._require_stage_running(stage)
        ref = self._stage_relative_ref(stage, relative_path)
        path = self._resolve_ref(ref)

        stage_dir = self._stage_dir(stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write_bytes(path, content)

        return ArtifactFile(
            ref=ref,
            path=path,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
            record_count=record_count,
        )

    def write_json(
        self,
        stage: TrackingStage | str,
        relative_path: str | Path,
        payload: Any,
        *,
        indent: int | None = 2,
    ) -> ArtifactFile:
        """Atomically write a JSON artifact inside a RUNNING stage."""
        selected = _coerce_stage(stage)
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
        )
        if indent is not None:
            encoded += "\n"
        return self._write_stage_bytes(
            selected,
            relative_path,
            encoded.encode("utf-8"),
        )

    def write_jsonl(
        self,
        stage: TrackingStage | str,
        relative_path: str | Path,
        records: Iterable[Any],
    ) -> ArtifactFile:
        """Stream JSONL to a temporary file, then publish it atomically."""
        selected = _coerce_stage(stage)
        self._require_stage_running(selected)

        ref = self._stage_relative_ref(selected, relative_path)
        path = self._resolve_ref(ref)
        stage_dir = self._stage_dir(selected)
        stage_dir.mkdir(parents=True, exist_ok=True)
        path.parent.mkdir(parents=True, exist_ok=True)

        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        digest = hashlib.sha256()
        record_count = 0

        try:
            with temporary.open("wb") as destination:
                for record in records:
                    line = (_json_line(record) + "\n").encode("utf-8")
                    destination.write(line)
                    digest.update(line)
                    record_count += 1
                destination.flush()
                os.fsync(destination.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

        return ArtifactFile(
            ref=ref,
            path=path,
            size_bytes=path.stat().st_size,
            sha256=digest.hexdigest(),
            record_count=record_count,
        )

    @staticmethod
    def _coerce_artifact_ref(value: str | Path | ArtifactFile) -> str:
        if isinstance(value, ArtifactFile):
            return value.ref
        return str(value)

    def _validate_completed_ref(
        self,
        stage: TrackingStage,
        value: str | Path | ArtifactFile,
        *,
        label: str,
    ) -> str:
        ref = self._stage_relative_ref(stage, self._coerce_artifact_ref(value))
        path = self._resolve_ref(ref)
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
        return ref

    def complete_stage(
        self,
        stage: TrackingStage | str,
        *,
        artifact: str | Path | ArtifactFile,
        summary: str | Path | ArtifactFile | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrackingManifest:
        """Publish one stage as COMPLETED only after its files exist."""
        selected = _coerce_stage(stage)

        with self._lock:
            manifest = self.load_manifest()
            current = manifest.stage(selected)
            if current.status is not StageStatus.RUNNING:
                raise RuntimeError(
                    f"stage {selected.value!r} can only complete from running "
                    f"(current status: {current.status.value})"
                )

            artifact_ref = self._validate_completed_ref(
                selected,
                artifact,
                label="stage artifact",
            )
            summary_ref = (
                self._validate_completed_ref(selected, summary, label="stage summary")
                if summary is not None
                else None
            )

            merged_metadata = dict(current.metadata)
            merged_metadata.update(dict(metadata or {}))
            merged_metadata.pop("error", None)

            manifest.set_stage(
                selected,
                StageManifest(
                    status=StageStatus.COMPLETED,
                    artifact=artifact_ref,
                    summary=summary_ref,
                    version=current.version,
                    updated_at=_utc_now(),
                    config=current.config,
                    metadata=merged_metadata,
                ),
            )
            self._write_manifest(manifest)
            return manifest

    def fail_stage(
        self,
        stage: TrackingStage | str,
        error: str | BaseException,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> TrackingManifest:
        """Mark a RUNNING stage as FAILED without publishing partial files."""
        selected = _coerce_stage(stage)
        message = str(error).strip() or error.__class__.__name__

        with self._lock:
            manifest = self.load_manifest()
            current = manifest.stage(selected)
            if current.status is not StageStatus.RUNNING:
                raise RuntimeError(
                    f"stage {selected.value!r} can only fail from running "
                    f"(current status: {current.status.value})"
                )

            merged_metadata = dict(current.metadata)
            merged_metadata.update(dict(metadata or {}))
            merged_metadata["error"] = message

            manifest.set_stage(
                selected,
                StageManifest(
                    status=StageStatus.FAILED,
                    version=current.version,
                    updated_at=_utc_now(),
                    config=current.config,
                    metadata=merged_metadata,
                ),
            )
            self._write_manifest(manifest)
            return manifest

    def invalidate_stage(
        self,
        stage: TrackingStage | str,
        *,
        include_downstream: bool = True,
        delete_files: bool = True,
        reason: str | None = None,
    ) -> TrackingManifest:
        """Reset a stage to NOT_RUN and optionally invalidate everything after it."""
        selected = _coerce_stage(stage)

        with self._lock:
            manifest = self.ensure_manifest()
            targets = (selected,) + (
                _downstream_stages(selected) if include_downstream else ()
            )
            now = _utc_now()

            for target in targets:
                metadata: dict[str, Any] = {}
                if target is not selected:
                    metadata["invalidated_by"] = selected.value
                if reason:
                    metadata["reason"] = reason
                manifest.set_stage(
                    target,
                    StageManifest(
                        status=StageStatus.NOT_RUN,
                        updated_at=now,
                        metadata=metadata,
                    ),
                )

            self._write_manifest(manifest)

            if delete_files:
                for target in targets:
                    self._remove_stage_dir(target)

            return manifest

    def resolve_artifact(self, ref: str | Path, *, require_file: bool = True) -> Path:
        """Resolve a manifest-relative artifact reference safely."""
        path = self._resolve_ref(ref)
        if require_file and not path.is_file():
            raise FileNotFoundError(path)
        return path

    def read_json(self, ref: str | Path) -> Any:
        path = self.resolve_artifact(ref)
        return json.loads(path.read_text(encoding="utf-8"))

    def iter_jsonl(self, ref: str | Path) -> Iterator[dict[str, Any]]:
        path = self.resolve_artifact(ref)
        with path.open("r", encoding="utf-8") as source:
            for line_number, raw in enumerate(source, start=1):
                if not raw.strip():
                    continue
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"JSONL row {line_number} in {ref!s} is not an object"
                    )
                yield payload

    def stage_completed(self, stage: TrackingStage | str) -> bool:
        if not self.manifest_exists():
            return False
        return self.load_manifest().stage(_coerce_stage(stage)).status is StageStatus.COMPLETED
