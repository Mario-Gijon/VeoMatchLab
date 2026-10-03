"""Read-only HTTP boundary for the provisional actor trajectory artifact."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

from fastapi import APIRouter

from .tracking_final_api import _error


TRAJECTORY_VERSION = "actor-trajectory-reconstruction-v1"
REQUIRED_FILES = ("trajectories.jsonl", "segments.jsonl", "gap_events.jsonl", "summary.json", "index.json")


def _missing(message: str = "Actor trajectories are not available for this clip."):
    return _error(404, message, "actor_trajectory_missing")


def _invalid(message: str = "Actor trajectory artifact is invalid."):
    return _error(422, message, "actor_trajectory_invalid")


def _inside(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise _missing()
    return resolved


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _bbox(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise _invalid("Actor trajectory visual evidence contains an invalid bounding box.")
    parsed = [_finite(item) for item in value]
    if any(item is None for item in parsed) or parsed[0] >= parsed[2] or parsed[1] >= parsed[3]:
        raise _invalid("Actor trajectory visual evidence contains an invalid bounding box.")
    return [float(item) for item in parsed]


def _visual_endpoint(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _invalid("Actor trajectory visual evidence is invalid.")
    view = value.get("representative_view")
    frame_index, timestamp = _integer(value.get("frame_index")), _finite(value.get("timestamp_s"))
    if not isinstance(view, str) or not view or frame_index is None or frame_index < 0 or timestamp is None or timestamp < 0:
        raise _invalid("Actor trajectory visual evidence is invalid.")
    virtual_bbox = _bbox(value.get("virtual_bbox_xyxy"))
    if virtual_bbox is None:
        raise _invalid("Actor trajectory visual evidence is missing its virtual bounding box.")
    source_lens = value.get("source_lens")
    if source_lens is not None and (not isinstance(source_lens, str) or not source_lens):
        raise _invalid("Actor trajectory visual evidence contains an invalid source lens.")
    return {
        "detection_id": value.get("detection_id") if isinstance(value.get("detection_id"), str) else None,
        "frame_index": frame_index,
        "timestamp_s": float(timestamp),
        "representative_view": view,
        "virtual_bbox_xyxy": virtual_bbox,
        "source_lens": source_lens,
        "source_bbox_xyxy": _bbox(value.get("source_bbox_xyxy")),
    }


def _visual_interpolation(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise _invalid("Actor trajectory visual interpolation is invalid.")
    left, right = _visual_endpoint(value.get("left")), _visual_endpoint(value.get("right"))
    alpha = _finite(value.get("alpha"))
    allowed, source_allowed = value.get("visual_interpolation_allowed"), value.get("source_bbox_interpolation_allowed")
    if left is None or right is None or alpha is None or not 0 <= alpha <= 1 or not isinstance(allowed, bool) or not isinstance(source_allowed, bool):
        raise _invalid("Actor trajectory visual interpolation is invalid.")
    if source_allowed and (left["source_bbox_xyxy"] is None or right["source_bbox_xyxy"] is None or not left["source_lens"] or left["source_lens"] != right["source_lens"]):
        raise _invalid("Actor trajectory source bounding-box interpolation is invalid.")
    return {"left": left, "right": right, "alpha": float(alpha), "visual_interpolation_allowed": allowed, "source_bbox_interpolation_allowed": source_allowed}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as source:
            for number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"line {number} is not an object")
                rows.append(row)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise _invalid() from exc
    return rows


def _artifact_root(clip_root: Path) -> Path:
    root = _inside(clip_root / "tracking" / "v1" / "diagnostics" / "actor_trajectory_reconstruction" / TRAJECTORY_VERSION, clip_root)
    if not root.is_dir() or any(not _inside(root / filename, clip_root).is_file() for filename in REQUIRED_FILES):
        raise _missing()
    return root


def _point(value: Any, previous_frame: int | None, previous_time: float | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise _invalid("Actor trajectory point is invalid.")
    frame_index, timestamp = _integer(value.get("frame_index")), _finite(value.get("timestamp_s"))
    x, z = _finite(value.get("x")), _finite(value.get("z"))
    origin = value.get("origin")
    if (frame_index is None or frame_index < 0 or timestamp is None or timestamp < 0 or x is None or z is None
            or origin not in {"observed", "interpolated"} or (previous_frame is not None and frame_index <= previous_frame)
            or (previous_time is not None and timestamp <= previous_time)):
        raise _invalid("Actor trajectory point order or position is invalid.")
    anchor, interpolation = _visual_endpoint(value.get("visual_anchor")), _visual_interpolation(value.get("visual_interpolation"))
    if origin == "observed" and interpolation is not None:
        raise _invalid("Observed actor trajectory point contains interpolation evidence.")
    if origin == "interpolated" and anchor is not None:
        raise _invalid("Interpolated actor trajectory point contains observed evidence.")
    tracklet_ids = value.get("source_tracklet_ids")
    if tracklet_ids is not None and (not isinstance(tracklet_ids, list) or not all(isinstance(item, str) and item for item in tracklet_ids)):
        raise _invalid("Actor trajectory source tracklets are invalid.")
    return {
        "frame_index": frame_index, "timestamp_s": float(timestamp), "x": float(x), "z": float(z),
        "vx_mps": _finite(value.get("vx_mps")), "vz_mps": _finite(value.get("vz_mps")), "speed_mps": _finite(value.get("speed_mps")),
        "origin": origin, "segment_id": value.get("segment_id") if isinstance(value.get("segment_id"), str) else None,
        "segment_index": _integer(value.get("segment_index")), "source_tracklet_ids": tracklet_ids or [],
        "visual_anchor": anchor, "visual_interpolation": interpolation,
    }


def build_actor_trajectories(clip_root: Path, root: Path) -> dict[str, Any]:
    root = _inside(root, clip_root)
    try:
        summary = json.loads(_inside(root / "summary.json", clip_root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise _invalid() from exc
    if not isinstance(summary, dict) or summary.get("trajectory_version") != TRAJECTORY_VERSION:
        raise _invalid("Actor trajectory version is invalid.")
    actor_count = _integer(summary.get("actor_count"))
    if actor_count is None or actor_count < 0:
        raise _invalid("Actor trajectory actor count is invalid.")
    actors: list[dict[str, Any]] = []
    actor_ids: set[str] = set()
    observed_count = interpolated_count = 0
    for row in _read_jsonl(_inside(root / "trajectories.jsonl", clip_root)):
        actor_id, tracklet_ids, points = row.get("actor_id"), row.get("tracklet_ids"), row.get("points")
        if (not isinstance(actor_id, str) or not actor_id or actor_id in {"UNASSIGNED", "NON_ROSTER"} or actor_id in actor_ids or not isinstance(tracklet_ids, list)
                or not all(isinstance(item, str) and item for item in tracklet_ids) or not isinstance(points, list)):
            raise _invalid("Actor trajectory row is invalid.")
        actor_ids.add(actor_id)
        parsed_points: list[dict[str, Any]] = []
        previous_frame: int | None = None
        previous_time: float | None = None
        for point in points:
            parsed = _point(point, previous_frame, previous_time)
            previous_frame, previous_time = parsed["frame_index"], parsed["timestamp_s"]
            observed_count += parsed["origin"] == "observed"
            interpolated_count += parsed["origin"] == "interpolated"
            parsed_points.append(parsed)
        actors.append({"actor_id": actor_id, "tracklet_ids": tracklet_ids, "segment_count": _integer(row.get("segment_count")) or 0, "points": parsed_points})
    if len(actors) != actor_count:
        raise _invalid("Actor trajectory actor count does not match its rows.")
    smoothing = summary.get("smoothing") if isinstance(summary.get("smoothing"), dict) else {}
    cadence = summary.get("sampling_cadence") if isinstance(summary.get("sampling_cadence"), dict) else smoothing.get("sampling_cadence") if isinstance(smoothing.get("sampling_cadence"), dict) else {}
    display_count = _integer(summary.get("display_point_count"))
    summary_observed, summary_interpolated = _integer(summary.get("observed_display_point_count")), _integer(summary.get("interpolated_display_point_count"))
    if display_count != observed_count + interpolated_count or summary_observed != observed_count or summary_interpolated != interpolated_count:
        raise _invalid("Actor trajectory point counts do not match its summary.")
    return {
        "trajectory_version": TRAJECTORY_VERSION, "provisional": True, "actor_count": actor_count,
        "display_point_count": display_count, "observed_point_count": observed_count, "interpolated_point_count": interpolated_count,
        "segment_count": _integer(summary.get("segment_count")) or 0,
        "estimated_fps": _finite(cadence.get("estimated_fps")), "frame_step": _integer(cadence.get("frame_step")), "time_step_s": _finite(cadence.get("time_step_s")),
        "actors": sorted(actors, key=lambda actor: actor["actor_id"]),
    }


def tracking_actor_trajectory_router(store_provider: Callable[[], Any], camera_specs_provider: Callable[[dict[str, Any]], dict[str, Any]]) -> APIRouter:
    router = APIRouter(prefix="/api/tracking/matches/{match_id}/clips/{clip_id}/actor-trajectories")

    @router.get("")
    def actor_trajectories(match_id: str, clip_id: str) -> dict[str, Any]:
        store = store_provider()
        match = store.match(match_id)
        if match is None:
            raise _error(404, "Workspace not found.", "match_not_found")
        if store.clip(match_id, clip_id) is None:
            raise _error(404, "Clip not found.", "clip_not_found")
        clips_root = store.asset_dir(match_id, "clips").resolve()
        clip_root = _inside(clips_root / clip_id, clips_root)
        payload = build_actor_trajectories(clip_root, _artifact_root(clip_root))
        payload["camera_specs"] = camera_specs_provider(match)
        return payload

    return router
