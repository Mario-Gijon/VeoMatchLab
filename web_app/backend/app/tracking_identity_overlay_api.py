"""Read-only HTTP boundary for provisional Tracking V1 identity overlays."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter

from .tracking_final_api import _error


ROUND_NAME = re.compile(r"round-(\d+)-v1$")
REQUIRED_STATE_FILES = ("final_identities.jsonl", "final_tracklet_assignments.jsonl")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {number} is not an object")
                rows.append(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise _error(422, "The Tracking V1 overlay artifact is invalid.", "tracking_identity_overlay_invalid") from exc
    return rows


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise _error(404, "Tracking V1 overlay is not available for this clip.", "tracking_identity_overlay_missing")
    return resolved


def _state_root(clip_root: Path, state: str) -> tuple[str, Path]:
    states_root = _inside(clip_root / "tracking" / "v1" / "diagnostics" / "global_trajectory_round_state", clip_root)
    if not states_root.is_dir():
        raise _error(404, "Tracking V1 overlay is not available for this clip.", "tracking_identity_overlay_missing")

    def valid(candidate: Path) -> bool:
        return candidate.is_dir() and all(_inside(candidate / filename, clip_root).is_file() for filename in REQUIRED_STATE_FILES)

    if state == "latest":
        choices: list[tuple[int, str, Path]] = []
        for candidate in states_root.iterdir():
            match = ROUND_NAME.fullmatch(candidate.name)
            if match and valid(candidate):
                choices.append((int(match.group(1)), candidate.name, candidate))
        if not choices:
            raise _error(404, "Tracking V1 overlay is not available for this clip.", "tracking_identity_overlay_missing")
        _, state_id, root = max(choices, key=lambda item: item[0])
        return state_id, root

    if not ROUND_NAME.fullmatch(state):
        raise _error(404, "Tracking V1 overlay state is not available for this clip.", "tracking_identity_overlay_missing")
    root = _inside(states_root / state, clip_root)
    if not valid(root):
        raise _error(404, "Tracking V1 overlay state is incomplete.", "tracking_identity_overlay_missing")
    return state, root


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise _error(422, "Tracking V1 overlay contains an invalid source bounding box.", "tracking_identity_overlay_invalid")
    parsed = [_finite_number(item) for item in value]
    if any(item is None for item in parsed) or parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
        raise _error(422, "Tracking V1 overlay contains an invalid source bounding box.", "tracking_identity_overlay_invalid")
    return [float(item) for item in parsed]


def build_identity_overlay(clip_root: Path, state_id: str, state_root: Path, source_width: int, source_height: int) -> dict[str, Any]:
    """Resolve detection -> tracklet -> provisional identity without deriving tracking."""
    state_root = _inside(state_root, clip_root)
    paths = {
        "frames": _inside(clip_root / "tracking" / "v1" / "detections" / "frames.jsonl", clip_root),
        "tracklets": _inside(clip_root / "tracking" / "v1" / "tracklets" / "tracklets.jsonl", clip_root),
        "assignments": _inside(state_root / "final_tracklet_assignments.jsonl", clip_root),
        "identities": _inside(state_root / "final_identities.jsonl", clip_root),
    }
    if any(not path.is_file() for path in paths.values()):
        raise _error(404, "Tracking V1 overlay artifacts are incomplete.", "tracking_identity_overlay_missing")

    identities: set[str] = set()
    for row in _read_jsonl(paths["identities"]):
        identity_id = row.get("identity_id")
        if not isinstance(identity_id, str) or not identity_id or identity_id in identities:
            raise _error(422, "Tracking V1 overlay identities are invalid.", "tracking_identity_overlay_invalid")
        identities.add(identity_id)

    assignments: dict[str, str] = {}
    for row in _read_jsonl(paths["assignments"]):
        tracklet_id, identity_id = row.get("tracklet_id"), row.get("identity_id")
        if not isinstance(tracklet_id, str) or not tracklet_id or not isinstance(identity_id, str) or identity_id not in identities or tracklet_id in assignments:
            raise _error(422, "Tracking V1 overlay assignments are invalid.", "tracking_identity_overlay_invalid")
        assignments[tracklet_id] = identity_id

    detections: dict[str, dict[str, Any]] = {}
    for frame in _read_jsonl(paths["frames"]):
        values = frame.get("detections")
        if not isinstance(values, list):
            raise _error(422, "Tracking V1 overlay detections are invalid.", "tracking_identity_overlay_invalid")
        for detection in values:
            detection_id = detection.get("detection_id") if isinstance(detection, dict) else None
            timestamp = detection.get("timestamp_s") if isinstance(detection, dict) else None
            if not isinstance(detection_id, str) or not detection_id or detection_id in detections or _finite_number(timestamp) is None:
                raise _error(422, "Tracking V1 overlay detections are invalid.", "tracking_identity_overlay_invalid")
            bbox = _bbox(detection.get("source_bbox_xyxy"))
            if bbox is not None:
                detections[detection_id] = {"timestamp_s": float(timestamp), "bbox": bbox, "source_lens": detection.get("source_lens") if isinstance(detection.get("source_lens"), str) else None}
            else:
                detections[detection_id] = {"timestamp_s": float(timestamp), "bbox": None, "source_lens": None}

    tracklet_detections: dict[str, list[str]] = {}
    detection_owner: set[str] = set()
    for row in _read_jsonl(paths["tracklets"]):
        tracklet_id, observations = row.get("tracklet_id"), row.get("observations")
        if not isinstance(tracklet_id, str) or not tracklet_id or tracklet_id in tracklet_detections or not isinstance(observations, list):
            raise _error(422, "Tracking V1 overlay tracklets are invalid.", "tracking_identity_overlay_invalid")
        ids: list[str] = []
        seen: set[str] = set()
        for observation in observations:
            detection_id = observation.get("detection_id") if isinstance(observation, dict) else None
            if not isinstance(detection_id, str) or detection_id not in detections or detection_id in seen or detection_id in detection_owner:
                raise _error(422, "Tracking V1 overlay tracklets are invalid.", "tracking_identity_overlay_invalid")
            seen.add(detection_id)
            detection_owner.add(detection_id)
            ids.append(detection_id)
        tracklet_detections[tracklet_id] = ids

    if set(assignments) - set(tracklet_detections):
        raise _error(422, "Tracking V1 overlay assignments reference missing tracklets.", "tracking_identity_overlay_invalid")

    tracks: list[dict[str, Any]] = []
    for tracklet_id, identity_id in sorted(assignments.items()):
        observations = [detections[detection_id] for detection_id in tracklet_detections[tracklet_id] if detections[detection_id]["bbox"] is not None]
        observations.sort(key=lambda item: item["timestamp_s"])
        if observations:
            tracks.append({"tracklet_id": tracklet_id, "identity_id": identity_id, "observations": observations})
    return {"state_id": state_id, "provisional": True, "identity_count": len(identities), "source_width": source_width, "source_height": source_height, "tracks": tracks}


def tracking_identity_overlay_router(store_provider: Callable[[], Any]) -> APIRouter:
    router = APIRouter(prefix="/api/tracking/matches/{match_id}/clips/{clip_id}/identity-overlay")

    @router.get("")
    def identity_overlay(match_id: str, clip_id: str, state: str = "latest") -> dict[str, Any]:
        store = store_provider()
        if store.match(match_id) is None:
            raise _error(404, "Workspace not found.", "match_not_found")
        clip = store.clip(match_id, clip_id)
        if clip is None:
            raise _error(404, "Clip not found.", "clip_not_found")
        clips_root = store.asset_dir(match_id, "clips").resolve()
        clip_root = _inside(clips_root / clip_id, clips_root)
        state_id, state_root = _state_root(clip_root, state)
        media_id = clip.get("clip_media_id")
        media_result = store.media(media_id) if isinstance(media_id, str) else None
        media = media_result[1] if media_result else None
        width, height = (media or {}).get("width"), (media or {}).get("height")
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise _error(422, "Physical clip dimensions are unavailable for the Tracking V1 overlay.", "tracking_identity_overlay_invalid")
        return build_identity_overlay(clip_root, state_id, state_root, width, height)

    return router
