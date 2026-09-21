"""Conservative Veo dataset normalization and match-scoped persistence.

Only the observed bare-array Highlights schema has a domain parser today.
Other datasets retain evidence and sanitized bodies without invented semantics.
"""
from __future__ import annotations

import hashlib
import math
import re
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from ...artifacts import LocalArtifactStore, atomic_json, read_json
from ...match_data import DatasetObservation, MatchEvent, Provenance
from ...storage import WorkspaceStore
from .discovery import DATASETS, StructuredDiscovery, classify_endpoint

PARSER_VERSION = "veo-highlights/1"


def match_key(url: str) -> str | None:
    parsed = urlsplit(url)
    if parsed.hostname not in {"app.veo.co", "veo.co"}:
        return None
    match = re.search(r"/matches/([^/]+)(?:/|$)", parsed.path)
    return match.group(1) if match else None


def display_url(value) -> str | None:
    if not isinstance(value, str) or "redacted" in value.lower():
        return None
    parsed = urlsplit(value)
    # References only: never fetch remote resources in the backend/parser.
    if parsed.scheme == "https" and parsed.hostname and not parsed.username and not parsed.password:
        return value
    return None


def nonnegative(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        return None
    return float(value)


def parse_highlights(payload, match_id: str, artifact_id: str) -> tuple[list[dict], dict]:
    """Skip invalid rows, never coerce clip offsets, confidence, teams or players."""
    if not isinstance(payload, list):
        return [], {"parser_version": PARSER_VERSION, "supported_schema": False, "skipped": None}
    events, skipped = {}, 0
    for row in payload:
        if not isinstance(row, dict):
            skipped += 1
            continue
        start = nonnegative(row.get("start"))
        kind, original_id = row.get("type"), row.get("id")
        if start is None or not isinstance(kind, str) or not kind.strip() or not isinstance(original_id, str) or not original_id:
            skipped += 1
            continue
        duration = nonnegative(row.get("duration"))
        identity = hashlib.sha256(f"{match_id}:veo:highlights:{original_id}".encode()).hexdigest()[:24]
        clips = []
        for video in row.get("videos", []) if isinstance(row.get("videos"), list) else []:
            if isinstance(video, dict) and (url := display_url(video.get("url"))):
                clips.append({"url": url, **{key: video[key] for key in ("mime_type", "width", "height") if key in video}})
        event = MatchEvent(id=f"event_{identity}", match_id=match_id, source="veo", source_dataset="highlights",
            type=kind, label={"goal": "Gol", "shot_on_goal": "Tiro a puerta"}.get(kind, kind),
            start_s=start, duration_s=duration, end_s=start + duration if duration is not None else None,
            thumbnail_url=display_url(row.get("thumbnail")), clip_references=clips,
            metadata={"source_id": original_id, **{key: row[key] for key in ("is_ai_generated", "created") if key in row}},
            provenance=Provenance(source="veo", source_dataset="highlights", parser_version=PARSER_VERSION,
                                  upstream_artifact_ids=[artifact_id]))
        events[event.id] = event.model_dump(mode="json", exclude_none=True)
    return sorted(events.values(), key=lambda e: (e["start_s"], e["id"])), {
        "parser_version": PARSER_VERSION, "supported_schema": True, "skipped": skipped,
        "input_records": len(payload), "event_count": len(events)}


def persist_datasets(store: WorkspaceStore, match_id: str, source_url: str,
                     observations: list[DatasetObservation]) -> dict:
    """Publish dataset refs separately; retrying the same observations is idempotent.

    Keep history but expose the latest successful snapshot per dataset. A failed
    reinspection does not erase previously fetched data. Source namespaces and
    analysis runs never overwrite one another.
    """
    with store._lock:
        match = store.match(match_id)
        expected = match_key(source_url)
        if not match or not expected or match_key(match.get("metadata", {}).get("veo_url", "")) != expected:
            raise ValueError("Dataset source does not match the workspace Veo match")
        root = store.asset_dir(match_id, "datasets")
        index = read_json(root / "index.json", {"schema_version": "1", "datasets": {}})
        artifacts = LocalArtifactStore(store)
        # Capture provenance can contain other API calls; do not import other matches.
        for observation in sorted(observations, key=lambda o: (o.observed_at, o.id)):
            if observation.source_match != expected or observation.dataset not in DATASETS:
                continue
            key = f"veo:{observation.dataset}"
            dataset = index["datasets"].setdefault(key, {"id": key, "source": "veo", "name": observation.dataset,
                "status": "not_observed", "observations": [], "artifact_ids": [], "event_artifact_id": None})
            if any(o["id"] == observation.id for o in dataset["observations"]):
                continue
            evidence = observation.model_dump(mode="json", exclude={"payload_path"})
            raw = None
            if observation.payload_observed and observation.payload_path:
                path = Path(observation.payload_path)
                if path.is_file():
                    payload = read_json(path)
                    raw = artifacts.write_json(match_id, f"veo.raw.{observation.dataset}", payload,
                        Provenance(source="veo", source_dataset=observation.dataset, parser_version=observation.parser_version))
                    evidence["raw_artifact_id"] = raw["id"]
                    dataset["artifact_ids"] = list(dict.fromkeys([*dataset["artifact_ids"], raw["id"]]))
            if observation.payload_observed and raw is None:
                evidence.update(status="not_observed", payload_observed=False, capture_error="payload_missing")
            dataset["observations"].append(evidence)
            successful = evidence["status"] in {"observed", "empty"} and raw is not None
            newer = observation.observed_at >= dataset.get("snapshot_at", "")
            if successful and newer:
                dataset.update(status=evidence["status"], snapshot_at=observation.observed_at,
                    raw_artifact_id=raw["id"], event_artifact_id=None, event_count=0)
                if observation.dataset == "highlights":
                    events, report = parse_highlights(payload, match_id, raw["id"])
                    normalized = artifacts.write_json(match_id, "match_events", events,
                        Provenance(source="veo", source_dataset="highlights", parser_version=PARSER_VERSION,
                                   upstream_artifact_ids=[raw["id"]]))
                    dataset.update(event_artifact_id=normalized["id"], event_count=len(events), parser=report)
            elif not dataset.get("raw_artifact_id"):
                dataset["status"] = evidence["status"]
        atomic_json(root / "index.json", index)
        store.update_match(match_id, lambda item: item.update(structured_data={"schema_version": "1",
            "dataset_index": "datasets/index.json", "artifact_index": "artifacts/index.json", "run_index": "analysis/index.json"}))
        return index


def replay_capture(root: Path, responses: list[dict], saved_bodies: dict,
                   capture_dir: Path) -> StructuredDiscovery:
    """Same interpreter for offline captures and live inspector, without networking."""
    discovery = StructuredDiscovery(capture_dir)
    by_url = {item.get("url"): item for item in responses}
    seen = set()
    for url, relative in saved_bodies.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or path.suffix != ".json" or not path.is_file():
            continue
        response = by_url.get(url, {})
        timestamp = response.get("time", path.stat().st_mtime)
        observed_at = datetime.fromtimestamp(timestamp, UTC).isoformat() if isinstance(timestamp, (float, int)) else None
        try:
            payload = read_json(path)
        except (ValueError, OSError):
            continue  # Network evidence below still records the missing body.
        discovery.observe_json(url, payload, response.get("phase", "captured"),
            "standalone_json_capture", response.get("status", 200), observed_at)
        seen.add(url)
    for response in responses:
        url = response.get("url", "")
        if url not in seen and classify_endpoint(url):
            timestamp = response.get("time")
            discovery.observe_json(url, None, response.get("phase", "captured"), "network_response", response.get("status", 0),
                datetime.fromtimestamp(timestamp, UTC).isoformat() if isinstance(timestamp, (int, float)) else None)
    return discovery
