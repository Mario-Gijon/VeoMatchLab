"""Replaceable local artifact store: immutable payloads, small atomic indexes.

JSON for small datasets; sorted JSONL chunks for large observations. Chunk
selection happens before reading rows, so a time window never loads a whole run.
Single-process writes share WorkspaceStore's lock. No execution is implied.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Iterator, Protocol

from .analysis_layers import AnalysisArtifact, AnalysisRun
from .match_data import Provenance
from .storage import WorkspaceStore


def read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}-{WorkspaceStore.new_id('write')}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ArtifactStore(Protocol):
    def list_artifacts(self, match_id: str) -> list[dict]: ...
    def write_json(self, match_id: str, kind: str, payload: Any, provenance: Provenance, **kwargs) -> dict: ...
    def write_records(self, match_id: str, kind: str, records: Iterable[dict], provenance: Provenance, **kwargs) -> dict: ...
    def read_window(self, match_id: str, artifact_id: str, start_s: float = 0, end_s: float = math.inf) -> Iterator[dict]: ...


class LocalArtifactStore:
    def __init__(self, store: WorkspaceStore):
        self.store = store

    def _root(self, match_id: str) -> Path:
        if self.store.match(match_id) is None:
            raise ValueError("Match does not exist")
        return self.store.asset_dir(match_id, "artifacts")

    def _directory(self, match_id: str, artifact_id: str) -> Path:
        if not re.fullmatch(r"artifact_[a-zA-Z0-9_-]+", artifact_id):
            raise ValueError("Invalid artifact identifier")
        root = self._root(match_id)
        directory = root / artifact_id
        if directory.is_symlink() or directory.resolve().parent != root.resolve():
            raise ValueError("Invalid artifact directory")
        return directory

    def list_artifacts(self, match_id: str) -> list[dict]:
        return read_json(self._root(match_id) / "index.json", [])

    def _publish(self, metadata: dict) -> dict:
        match_id = metadata["match_id"]
        index = self.list_artifacts(match_id)
        if not any(item["id"] == metadata["id"] for item in index):
            atomic_json(self._root(match_id) / "index.json", [*index, metadata])
        self._link_match(match_id)
        return metadata

    def _link_match(self, match_id: str) -> None:
        self.store.update_match(match_id, lambda item: item.update(structured_data={
            "schema_version": "1", "dataset_index": "datasets/index.json",
            "artifact_index": "artifacts/index.json", "run_index": "analysis/index.json",
        }))

    def _metadata(self, match_id, artifact_id, kind, format, count, provenance, source_media_ids, start=None, end=None):
        return AnalysisArtifact(id=artifact_id, match_id=match_id, kind=kind,
            source=provenance.source, source_dataset=provenance.source_dataset,
            run_id=provenance.run_id, source_media_ids=source_media_ids or [],
            format=format, record_count=count, start_s=start, end_s=end,
            created_at=self.store.now(), provenance=provenance).model_dump(mode="json")

    def write_json(self, match_id: str, kind: str, payload: Any, provenance: Provenance,
                   *, source_media_ids: list[str] | None = None) -> dict:
        encoded = json.dumps([kind, payload, provenance.model_dump(), source_media_ids], sort_keys=True, allow_nan=False)
        artifact_id = "artifact_" + hashlib.sha256(encoded.encode()).hexdigest()[:24]
        with self.store._lock:
            directory = self._directory(match_id, artifact_id)
            previous = read_json(directory / "manifest.json")
            if previous:
                return self._publish(previous["metadata"])
            metadata = self._metadata(match_id, artifact_id, kind, "json", len(payload) if isinstance(payload, list) else 1,
                                      provenance, source_media_ids)
            atomic_json(directory / "payload.json", payload)
            atomic_json(directory / "manifest.json", {"metadata": metadata})
            return self._publish(metadata)

    def read_payload(self, match_id: str, artifact_id: str) -> Any:
        return read_json(self._directory(match_id, artifact_id) / "payload.json")

    def write_records(self, match_id: str, kind: str, records: Iterable[dict], provenance: Provenance,
                      *, source_media_ids: list[str] | None = None, chunk_seconds: float = 60,
                      max_rows: int = 10000) -> dict:
        """Consume timestamp-sorted rows with bounded memory; publish only on success.

        A failed writer may leave an unindexed directory, never a visible artifact.
        Future workers must validate their stage-specific contracts before calling.
        """
        if chunk_seconds <= 0 or not math.isfinite(chunk_seconds) or max_rows < 1:
            raise ValueError("Invalid chunk bounds")
        artifact_id = self.store.new_id("artifact")
        directory = self._directory(match_id, artifact_id)
        directory.mkdir()
        chunks, rows = [], []
        previous = -1.
        def flush():
            if not rows:
                return
            name = f"{len(chunks):06d}.jsonl"
            content = "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows)
            (directory / name).write_text(content, encoding="utf-8")
            chunks.append({"file": name, "start_s": rows[0]["timestamp_s"],
                "end_s": rows[-1]["timestamp_s"], "record_count": len(rows),
                "sha256": hashlib.sha256(content.encode()).hexdigest()})
            rows.clear()
        # Stream outside the metadata lock: a long detector run must not block
        # normal match/media reads. This unique directory is not yet published.
        for record in records:
            timestamp = record.get("timestamp_s")
            if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp) or timestamp < previous or timestamp < 0:
                raise ValueError("Rows require sorted nonnegative match timestamps")
            if rows and (len(rows) >= max_rows or timestamp - rows[0]["timestamp_s"] >= chunk_seconds):
                flush()
            rows.append(record)
            previous = timestamp
        flush()
        metadata = self._metadata(match_id, artifact_id, kind, "jsonl_chunks", sum(c["record_count"] for c in chunks),
            provenance, source_media_ids, chunks[0]["start_s"] if chunks else None, chunks[-1]["end_s"] if chunks else None)
        with self.store._lock:
            self._root(match_id)  # The match may have been deleted during writing.
            atomic_json(directory / "manifest.json", {"metadata": metadata, "chunks": chunks})
            return self._publish(metadata)

    def read_window(self, match_id: str, artifact_id: str, start_s: float = 0, end_s: float = math.inf) -> Iterator[dict]:
        directory = self._directory(match_id, artifact_id)
        manifest = read_json(directory / "manifest.json")
        if not manifest:
            raise ValueError("Artifact not found")
        if manifest["metadata"]["format"] == "json":
            payload = self.read_payload(match_id, artifact_id)
            if not isinstance(payload, list):
                raise ValueError("Time-window reads require a row artifact")
            for row in payload:
                start = row.get("timestamp_s", row.get("start_s", 0))
                if start <= end_s and (row.get("end_s") or start) >= start_s:
                    yield row
            return
        for chunk in manifest["chunks"]:
            if chunk["end_s"] < start_s or chunk["start_s"] > end_s:
                continue
            if not re.fullmatch(r"\d+\.jsonl", chunk["file"]):
                raise ValueError("Invalid chunk reference")
            with (directory / chunk["file"]).open(encoding="utf-8") as source:
                for line in source:
                    row = json.loads(line)
                    if start_s <= row["timestamp_s"] <= end_s:
                        yield row

    def save_run(self, run: AnalysisRun) -> None:
        """Store truthful run metadata. Callers own real lifecycle transitions."""
        if not re.fullmatch(r"run_[a-zA-Z0-9_-]+", run.id):
            raise ValueError("Invalid run identifier")
        with self.store._lock:
            self._root(run.match_id)
            root = self.store.asset_dir(run.match_id, "analysis")
            atomic_json(root / f"{run.id}.json", run.model_dump(mode="json"))
            ids = read_json(root / "index.json", [])
            atomic_json(root / "index.json", sorted(set([*ids, run.id])))
            self._link_match(run.match_id)

    def list_runs(self, match_id: str) -> list[dict]:
        self._root(match_id)
        root = self.store.asset_dir(match_id, "analysis")
        return [AnalysisRun.model_validate(read_json(path)).model_dump(mode="json") for path in sorted(root.glob("run_*.json"))]
