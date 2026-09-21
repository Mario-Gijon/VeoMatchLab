"""Shared, bounded interpretation of Veo match discovery evidence."""

from __future__ import annotations

import json
import re
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .classifier import sanitize_url
from ...match_data import DatasetObservation
from ...artifacts import atomic_json

MAX_UNKNOWN_ITEMS = 20
MAX_UNKNOWN_BYTES = 48_000
MAX_REFERENCES = 300
MAX_DATASET_BYTES = 48_000
DATASETS = ("highlights", "events", "lineup", "roster", "player_tracking", "player_moments", "heatmaps", "physical_metrics", "match_report")
CAPABILITIES = ("has_tracking_data", "analytics_available", "events_available", "player_moments_available", "heatmaps_available", "interactive_available")
MATCH_PATH = re.compile(r"/(?:api/)?(?:app/)?matches/[^/]+(?:/|$)", re.I)
KNOWN_ENDPOINTS = {
    "highlights": re.compile(r"/matches/[^/]+/highlights/?(?:$|\?)", re.I),
    "events": re.compile(r"/matches/[^/]+/(?:events|annotations|timeline)/?(?:$|\?)", re.I),
    "lineup": re.compile(r"/matches/[^/]+/lineups?/?(?:$|\?)", re.I),
    "roster": re.compile(r"/matches/[^/]+/(?:roster|players)/?(?:$|\?)", re.I),
    "player_tracking": re.compile(r"/matches/[^/]+/(?:player[-_]?tracking|tracks)/?(?:$|\?)", re.I),
    "player_moments": re.compile(r"/matches/[^/]+/player[-_]?moments/?(?:$|\?)", re.I),
    "heatmaps": re.compile(r"/matches/[^/]+/(?:heatmaps?|heatmap)/?(?:$|\?)", re.I),
    "physical_metrics": re.compile(r"/matches/[^/]+/(?:physical[-_]?metrics|physical[-_]?stats)/?(?:$|\?)", re.I),
    "match_report": re.compile(r"/matches/[^/]+/(?:match[-_]?report|report)/?(?:$|\?)", re.I),
}
SECRET_KEYS = ("token", "signature", "authorization", "cookie", "password", "secret", "credential")
JWT_PATTERN = re.compile(r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b")
BEARER_PATTERN = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{10,}")
URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+")


def _safe_value(value: Any, key: str | None = None, *, full: bool = False) -> Any:
    if key and any(secret in key.lower() for secret in SECRET_KEYS):
        return "<REDACTED>"
    if isinstance(value, dict):
        return {str(name): _safe_value(item, str(name), full=full) for name, item in value.items()}
    if isinstance(value, list):
        return [_safe_value(item, full=full) for item in (value if full else value[:200])]
    if isinstance(value, str) and value.startswith(("https://", "http://")):
        return sanitize_url(value)
    if isinstance(value, str):
        value = JWT_PATTERN.sub("<JWT_REDACTED>", value)
        value = BEARER_PATTERN.sub("Bearer <REDACTED>", value)
        value = URL_PATTERN.sub(lambda match: sanitize_url(match.group(0)), value)
    return value


def _bounded(value: Any, limit: int) -> Any:
    safe = _safe_value(value)
    serialized = json.dumps(safe, ensure_ascii=False, default=str)
    if len(serialized.encode("utf-8")) <= limit:
        return safe
    return {"truncated": True, "preview": serialized[:limit]}


def referenced_urls(value: Any, limit: int = MAX_REFERENCES) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    def visit(item: Any, path: str, depth: int) -> None:
        if depth > 14 or len(found) >= limit:
            return
        if isinstance(item, dict):
            for key, child in item.items():
                child_path = f"{path}.{key}" if path else str(key)
                if isinstance(child, str) and child.startswith(("https://", "http://")):
                    found.append({"url": sanitize_url(child), "field": child_path})
                else:
                    visit(child, child_path, depth + 1)
        elif isinstance(item, list):
            for index, child in enumerate(item[:200]):
                visit(child, f"{path}[{index}]", depth + 1)
    visit(value, "", 0)
    return found


def classify_endpoint(url: str) -> str | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if not (host == "veo.co" or host.endswith(".veo.co") or host == "veocdn.com" or host.endswith(".veocdn.com")):
        return None
    for dataset, pattern in KNOWN_ENDPOINTS.items():
        if pattern.search(path):
            return dataset
    return None


def dataset_status(dataset: str, status_code: int, payload: Any) -> str:
    if status_code in {401, 403}:
        return "forbidden"
    if status_code == 404:
        return "unavailable"
    if not 200 <= status_code < 300:
        return "not_observed"
    if payload is None:
        return "not_observed"
    if payload == [] or payload == {}:
        return "empty"
    if isinstance(payload, dict):
        for key in ("results", "items", "data", dataset):
            if key in payload and payload[key] == []:
                return "empty"
        signals = {"metadata", "permissions", "capabilities", "match", "tracking_available", *CAPABILITIES}
        values = [value for key, value in payload.items() if key.lower() not in signals]
        if not values:
            return "not_observed"
        if all(value is None or value == [] or value == {} for value in values):
            return "empty"
    return "observed"


class StructuredDiscovery:
    def __init__(self, capture_dir: Path | None = None) -> None:
        self.capture_dir = capture_dir
        self.observations: list[DatasetObservation] = []
        self.responses: list[dict[str, Any]] = []
        self.references: list[dict[str, Any]] = []
        self.unknown: list[dict[str, Any]] = []
        self.dataset_evidence: dict[str, list[dict[str, Any]]] = {name: [] for name in DATASETS}
        self.capability_evidence: dict[str, list[dict[str, Any]]] = {name: [] for name in CAPABILITIES}

    def observe_json(self, url: str, payload: Any, phase: str, source: str = "json_response", status_code: int = 200,
                     observed_at: str | None = None) -> None:
        category = classify_endpoint(url)
        parsed = urlsplit(url)
        path = parsed.path
        safe_endpoint = sanitize_url(urlunsplit((parsed.scheme, parsed.netloc, path, "", "")))
        if category:
            status = dataset_status(category, status_code, payload)
            timestamp = observed_at or datetime.now(UTC).isoformat()
            # Diagnostics are bounded; persisted datasets must not silently lose
            # rows. HTTP capture still applies its existing response-byte limit.
            safe_payload = _safe_value(payload, full=True)
            identity = json.dumps([safe_endpoint, phase, timestamp, status_code, safe_payload], sort_keys=True)
            identifier = "observation_" + hashlib.sha256(identity.encode()).hexdigest()[:24]
            observation = DatasetObservation(id=identifier,
                source_match=re.search(r"/matches/([^/]+)", path, re.I).group(1),
                dataset=category, endpoint=safe_endpoint, phase=phase,
                observed_at=timestamp, context=source, status=status, http_status=status_code,
                payload_observed=payload is not None)
            if self.capture_dir:
                if payload is not None:
                    destination = self.capture_dir / f"{identifier}-payload.json"
                    atomic_json(destination, safe_payload)
                    observation.payload_path = str(destination.resolve())
                atomic_json(self.capture_dir / f"{identifier}.json", observation.model_dump(mode="json"))
            self.observations.append(observation)
            evidence = {"status": status, "endpoint": safe_endpoint, "phase": phase, "source": source,
                        "http_status": status_code, "payload_observed": payload is not None,
                        "observed_at": timestamp, "parser_version": observation.parser_version,
                        "payload": _bounded(payload, MAX_DATASET_BYTES)}
            self.dataset_evidence[category].append(evidence)
            self.responses.append({"url": safe_endpoint, "phase": phase, "source": source, "http_status": status_code, "category": category, "dataset_status": status})
        elif MATCH_PATH.search(path):
            self.responses.append({"url": safe_endpoint, "phase": phase, "source": source, "http_status": status_code, "category": "match_metadata"})
        # Capabilities require explicit match metadata permission/feature evidence,
        # or the caller recording a confirmed interaction. Generic key presence
        # in SDK/configuration responses is never sufficient.
        if MATCH_PATH.search(path):
            self._observe_match_capabilities(payload, safe_endpoint, phase, source)
        if (parsed.hostname or "").lower().endswith(("veo.co", "veocdn.com")):
            self.references.extend({**item, "phase": phase, "source": source} for item in referenced_urls(payload))
        if category is None and MATCH_PATH.search(path) and len(self.unknown) < MAX_UNKNOWN_ITEMS:
            self.unknown.append({"endpoint": safe_endpoint, "phase": phase, "http_status": status_code,
                                 "payload": _bounded(payload, MAX_UNKNOWN_BYTES)})

    def _observe_match_capabilities(self, payload: Any, endpoint: str, phase: str, source: str) -> None:
        def visit(item: Any, depth: int = 0) -> None:
            if depth > 10:
                return
            if isinstance(item, dict):
                for key, value in item.items():
                    normalized = key.lower()
                    permission_heatmap = "read_match_heatmap" in normalized
                    allowed = value is True or (isinstance(value, str) and value.strip().lower() in {"true", "allowed", "granted"})
                    if permission_heatmap and allowed:
                        name = "heatmaps_available"
                        self.capability_evidence[name].append({"status": "observed", "source": source, "endpoint": endpoint, "phase": phase, "signal": key})
                    if normalized in {"has_tracking_data", "tracking_available", "analytics_available", "events_available", "player_moments_available"} and value is True:
                        name = "has_tracking_data" if "tracking" in normalized else "analytics_available" if "analytics" in normalized else "events_available" if "events" in normalized else "player_moments_available"
                        self.capability_evidence[name].append({"status": "observed", "source": source, "endpoint": endpoint, "phase": phase, "signal": key})
                    visit(value, depth + 1)
            elif isinstance(item, list):
                for child in item[:100]:
                    visit(child, depth + 1)
        visit(payload)

    def record_interactive(self, confirmed: bool, evidence: dict[str, Any] | None = None) -> None:
        if confirmed:
            self.capability_evidence["interactive_available"].append({"status": "observed", "source": "player_interaction",
                                                                          "evidence": _safe_value(evidence or {})})

    def manifest(self, match: dict[str, Any], assets: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
        datasets = {}
        for name, evidence in self.dataset_evidence.items():
            status = "not_observed"
            if evidence:
                # Prefer a successful observed result; otherwise report the
                # strongest definitive failure across attempts.
                priority = {"observed": 5, "empty": 4, "forbidden": 3, "unavailable": 2, "not_observed": 1}
                status = max((entry["status"] for entry in evidence), key=lambda value: priority[value])
            datasets[name] = {"status": status, "evidence": evidence[:10]}
        capabilities = {name: {"status": "observed" if evidence else "not_observed", "evidence": evidence[:10]}
                        for name, evidence in self.capability_evidence.items()}
        references = list({(item["url"], item["field"], item["phase"]): item for item in self.references}.values())[:MAX_REFERENCES]
        return {"match": _safe_value(match), "assets": _safe_value(assets), "datasets": datasets,
                "capabilities": capabilities, "endpoints": self.responses[:100], "referenced_resources": references,
                "unknown_structured_data": self.unknown, "provenance": _safe_value(provenance)}


def write_manifest(path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
