"""Immutable, clip-scoped RF-DETR candidate cache primitives.

Candidates are deliberately stored before geometry/association.  Thresholding
always returns derived in-memory rows and never rewrites the cache.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Iterable

DETECTOR_CACHE_FLOOR = .20

def fingerprint(configuration: dict) -> str:
    """Inference fingerprint; caller must omit requested detection threshold."""
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def compatible(metadata: dict | None, configuration: dict, requested_threshold: float) -> bool:
    if not metadata or requested_threshold < float(metadata.get("floor", 1.1)) or metadata.get("fingerprint") != fingerprint(configuration):
        return False
    path = Path(metadata.get("candidates_path", ""))
    if not path.is_file(): return False
    content = path.read_bytes()
    return metadata.get("candidate_count") == len([line for line in content.splitlines() if line.strip()]) and metadata.get("sha256") == hashlib.sha256(content).hexdigest()

def filter_candidates(rows: Iterable[dict], threshold: float) -> list[dict]:
    return [row for row in rows if float(row["confidence"]) >= threshold]

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


class BaseDetectionCache:
    """Atomic pre-association candidate cache owned by a single clip."""
    def __init__(self, directory: Path, configuration: dict, requested_threshold: float):
        self.directory, self.configuration = directory, configuration
        self.requested_threshold = requested_threshold
        self.candidates_path = directory / "base_candidates.jsonl"
        self.metadata_path = directory / "metadata.json"
        self._rows: list[dict] | None = None
        self._index: dict[tuple[int, str], list[dict]] = {}
        self.hit = False
        self._pending: list[dict] = []
        self._metadata = {"version": 1, "floor": DETECTOR_CACHE_FLOOR,
                          "fingerprint": fingerprint(configuration), "configuration": configuration}
        if self.metadata_path.is_file() and self.candidates_path.is_file():
            try:
                current = json.loads(self.metadata_path.read_text(encoding="utf-8"))
                current["candidates_path"] = str(self.candidates_path)
                if compatible(current, configuration, requested_threshold):
                    self._rows, self.hit = read_jsonl(self.candidates_path), True
                    for row in self._rows: self._index.setdefault((row["frame_index"], row["virtual_camera"]), []).append(row)
            except (OSError, ValueError, json.JSONDecodeError):
                pass

    def candidates(self, frame_index: int, timestamp_s: float, camera: str, infer) -> list[dict]:
        if self.hit:
            rows = self._index.get((frame_index, camera), [])
        else:
            rows = [{"frame_index": frame_index, "timestamp_s": timestamp_s, "virtual_camera": camera,
                     "candidate_index": index, "bbox_xyxy": list(item.xyxy), "confidence": item.confidence}
                    for index, item in enumerate(infer())]
            self._pending.extend(rows)
        return filter_candidates(rows, self.requested_threshold)

    def commit(self) -> None:
        if self.hit:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        # Metadata is the completion marker. Remove it before replacing rows so
        # an interrupted publish is always a cache MISS.
        self.metadata_path.unlink(missing_ok=True)
        temporary = self.directory / ".base_candidates.tmp.jsonl"
        content = "".join(json.dumps(row, allow_nan=False) + "\n" for row in self._pending)
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(self.candidates_path)
        self._metadata.update(candidates_path=str(self.candidates_path), candidate_count=len(self._pending), sha256=hashlib.sha256(content.encode()).hexdigest())
        meta_tmp = self.directory / ".metadata.tmp.json"
        meta_tmp.write_text(json.dumps(self._metadata, indent=2) + "\n", encoding="utf-8")
        meta_tmp.replace(self.metadata_path)
