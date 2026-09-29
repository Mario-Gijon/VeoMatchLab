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
    return bool(metadata and requested_threshold >= float(metadata.get("floor", 1.1)) and metadata.get("fingerprint") == fingerprint(configuration))

def filter_candidates(rows: Iterable[dict], threshold: float) -> list[dict]:
    return [row for row in rows if float(row["confidence"]) >= threshold]

def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")
