"""Read-only HTTP boundary for the published Tracking V1 final artifact.

This module intentionally knows only the final publication layout.  It never
falls back to tracklets, solver output, or diagnostics when serving a clip.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter, HTTPException


FINAL_PUBLISHER = "final-publish-v1"


def _error(status: int, message: str, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"message": message, "code": code})


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _error(422, "The published Tracking V1 artifact is not valid JSON.", "tracking_final_invalid") from exc
    if not isinstance(payload, dict):
        raise _error(422, "The published Tracking V1 artifact must contain a JSON object.", "tracking_final_invalid")
    return payload


def read_final_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a final JSONL asset without interpreting or deriving its records."""
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError(f"line {line_number} is not an object")
                rows.append(payload)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise _error(422, "The published Tracking V1 JSONL artifact is invalid.", "tracking_final_invalid") from exc
    return rows


def tracking_final_router(store_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix="/api/tracking/matches/{match_id}/clips/{clip_id}/final")

    def final_root(match_id: str, clip_id: str) -> Path:
        store = store_provider()
        if store.match(match_id) is None:
            raise _error(404, "Workspace not found.", "match_not_found")
        if store.clip(match_id, clip_id) is None:
            raise _error(404, "Clip not found.", "clip_not_found")
        clips_root = store.asset_dir(match_id, "clips").resolve()
        root = (clips_root / clip_id / "tracking" / "v1" / "final").resolve()
        if not root.is_relative_to(clips_root):
            raise _error(404, "Tracking V1 FINAL is not available for this clip.", "tracking_final_missing")
        index_path = root / "index.json"
        if not index_path.is_file():
            raise _error(404, "Tracking V1 FINAL has not been published for this clip.", "tracking_final_missing")
        index = _json_object(index_path)
        if not (index.get("publisher_version") == FINAL_PUBLISHER and index.get("canonical") is True and index.get("final_stage_completed") is True):
            raise _error(409, "Tracking V1 FINAL is present but is not the canonical final-publish-v1 artifact.", "tracking_final_not_canonical")
        return root

    def published_json(root: Path, filename: str) -> dict[str, Any]:
        path = root / filename
        if not path.is_file():
            raise _error(404, "Tracking V1 FINAL is incomplete.", "tracking_final_missing")
        return _json_object(path)

    @router.get("")
    def final_tracking(match_id: str, clip_id: str) -> dict[str, Any]:
        root = final_root(match_id, clip_id)
        trajectories_path = root / "trajectories.jsonl"
        breaks_path = root / "segment_breaks.jsonl"
        if not trajectories_path.is_file() or not breaks_path.is_file():
            raise _error(404, "Tracking V1 FINAL is incomplete.", "tracking_final_missing")
        return {
            "index": published_json(root, "index.json"),
            "summary": published_json(root, "summary.json"),
            "trajectories": read_final_jsonl(trajectories_path),
            "segment_breaks": read_final_jsonl(breaks_path),
        }

    @router.get("/summary")
    def final_tracking_summary(match_id: str, clip_id: str) -> dict[str, Any]:
        return published_json(final_root(match_id, clip_id), "summary.json")

    @router.get("/reviews")
    def final_tracking_reviews(match_id: str, clip_id: str) -> dict[str, Any]:
        return published_json(final_root(match_id, clip_id), "review_queue.json")

    return router


def persistent_tracking_router(store_provider: Callable[[], Any]) -> APIRouter:
    """Expose published Persistent V1 diagnostics without executing tracking."""
    router = APIRouter(prefix="/api/tracking/matches/{match_id}/clips/{clip_id}/persistent")

    def persistent_root(match_id: str, clip_id: str) -> Path:
        store = store_provider()
        if store.match(match_id) is None:
            raise _error(404, "Workspace not found.", "match_not_found")
        if store.clip(match_id, clip_id) is None:
            raise _error(404, "Clip not found.", "clip_not_found")
        clips_root = store.asset_dir(match_id, "clips").resolve()
        root = (clips_root / clip_id / "tracking" / "v1" / "diagnostics" / "persistent_actor_trajectories" / "persistent-trajectory-v1").resolve()
        if not root.is_relative_to(clips_root) or not (root / "index.json").is_file():
            raise _error(404, "Persistent V1 has not been published for this clip.", "tracking_persistent_missing")
        index = _json_object(root / "index.json")
        if index.get("trajectory_version") != "persistent-trajectory-v1":
            raise _error(409, "Persistent V1 has an unexpected trajectory version.", "tracking_persistent_invalid")
        return root

    @router.get("")
    def persistent_tracking(match_id: str, clip_id: str) -> dict[str, Any]:
        root = persistent_root(match_id, clip_id)
        required = {name: root / name for name in ("trajectories.jsonl", "gaps.jsonl", "summary.json")}
        if any(not path.is_file() for path in required.values()):
            raise _error(404, "Persistent V1 is incomplete.", "tracking_persistent_missing")
        audit_root = (root.parent.parent / "persistent_population_audit" / "persistent-population-audit-v1").resolve()
        clips_root = store_provider().asset_dir(match_id, "clips").resolve()
        audit = _json_object(audit_root / "summary.json") if audit_root.is_relative_to(clips_root) and (audit_root / "summary.json").is_file() else None
        return {
            "index": _json_object(root / "index.json"),
            "summary": _json_object(required["summary.json"]),
            "audit": audit,
            "trajectories": read_final_jsonl(required["trajectories.jsonl"]),
            "gaps": read_final_jsonl(required["gaps.jsonl"]),
        }

    return router
