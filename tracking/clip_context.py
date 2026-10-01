"""Resolve the validated, materialized inputs for one Tracking V1 clip.

This module deliberately owns only workspace lookup and validation.  It does
not load tracking models or alter any Tracking V1 algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping


_MATCH_ID_RE = re.compile(r"match_[A-Za-z0-9_-]+")
_CLIP_ID_RE = re.compile(r"clip_[A-Za-z0-9_-]+")


@dataclass(frozen=True, slots=True)
class TrackingClipContext:
    """Canonical, validated input paths and timing for a materialized clip."""

    data_root: Path
    match_id: str
    clip_id: str
    match_dir: Path
    clip_dir: Path
    clip_video: Path
    calibration: Path
    tracking_root: Path
    duration_s: float
    fps: float
    frame_count: int
    width: int
    height: int


def _validate_identifier(value: str, pattern: re.Pattern[str], label: str) -> None:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        raise ValueError(f"invalid {label}: {value!r}")


def _require_directory(path: Path, label: str) -> Path:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    return path.resolve(strict=True)


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {path}")
    return path.resolve(strict=True)


def _managed_child(parent: Path, name: str, label: str, *, directory: bool) -> Path:
    candidate = parent / name
    resolved = _require_directory(candidate, label) if directory else _require_file(candidate, label)
    if resolved.parent != parent:
        raise ValueError(f"{label} is outside its managed parent: {candidate}")
    return resolved


def _optional_managed_directory(parent: Path, name: str, label: str) -> Path:
    candidate = parent / name
    if candidate.exists():
        return _managed_child(parent, name, label, directory=True)
    resolved = candidate.resolve(strict=False)
    if resolved.parent != parent:
        raise ValueError(f"{label} is outside its managed parent: {candidate}")
    return resolved


def _load_match_metadata(match_dir: Path, match_id: str) -> Mapping[str, Any] | None:
    path = match_dir / "match.json"
    if not path.exists():
        return None
    path = _managed_child(match_dir, "match.json", "match metadata", directory=False)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"match metadata is invalid JSON: {path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("match metadata must be a JSON object")
    if payload.get("id") != match_id:
        raise ValueError("match metadata identifier does not match its directory")
    return payload


def _clip_metadata(payload: Mapping[str, Any] | None, clip_id: str) -> Mapping[str, Any] | None:
    if payload is None:
        return None
    clips = payload.get("clips")
    if clips is None:
        return None
    if not isinstance(clips, list):
        raise ValueError("match metadata clips must be a list")
    for item in clips:
        if isinstance(item, Mapping) and item.get("id") == clip_id:
            return item
    raise FileNotFoundError(f"clip metadata does not exist for {clip_id!r}")


def _clip_media_metadata(
    payload: Mapping[str, Any] | None,
    clip: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if payload is None or clip is None:
        return None
    media_id = clip.get("clip_media_id")
    if media_id is None:
        return None
    media = payload.get("media")
    if not isinstance(media, list):
        raise ValueError("match metadata media must be a list")
    for item in media:
        if isinstance(item, Mapping) and item.get("id") == media_id:
            return item
    raise ValueError("clip_media_id is not present in match metadata")


def _metadata_path(value: Any, data_root: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} path is invalid")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = data_root / candidate
    return candidate.resolve(strict=False)


def _validate_metadata_path(
    record: Mapping[str, Any] | None,
    expected: Path,
    data_root: Path,
    label: str,
) -> None:
    if record is None:
        return
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} metadata must be an object")
    if record.get("path") is None:
        return
    path = _metadata_path(record["path"], data_root, label)
    if path != expected:
        raise ValueError(f"{label} metadata path is outside the managed clip context")


def _positive_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number")
    return number


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if number <= 0 or number != value:
        raise ValueError(f"{label} must be a positive integer")
    return number


def _metadata_value(
    media: Mapping[str, Any] | None,
    clip: Mapping[str, Any] | None,
    key: str,
) -> Any | None:
    for record in (media, clip):
        if record is not None and record.get(key) is not None:
            return record[key]
    return None


def _probe_video(video: Path) -> tuple[float, int, int, int]:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("OpenCV is required to validate a Tracking V1 clip video") from exc

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError(f"clip video could not be opened: {video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        capture.release()

    _positive_float(fps, "video fps")
    _positive_int(frame_count, "video frame_count")
    _positive_int(width, "video width")
    _positive_int(height, "video height")
    return fps, frame_count, width, height


def _validate_close(metadata: float, actual: float, label: str) -> None:
    if not math.isclose(metadata, actual, rel_tol=0.01, abs_tol=0.5):
        raise ValueError(f"canonical {label} does not match the materialized clip video")


def resolve_tracking_clip_context(
    data_root: str | Path,
    match_id: str,
    clip_id: str,
) -> TrackingClipContext:
    """Resolve a materialized clip without reading experiment-specific config.

    Metadata in ``match.json`` is preferred when available, but the real
    ``source.mp4`` is always opened to reject corrupt or inconsistent inputs.
    All resolved paths remain beneath ``data/matches/<match_id>``.
    """

    _validate_identifier(match_id, _MATCH_ID_RE, "match_id")
    _validate_identifier(clip_id, _CLIP_ID_RE, "clip_id")

    root_input = Path(data_root).expanduser()
    root = _require_directory(root_input, "data root")
    matches_root = _managed_child(root, "matches", "matches directory", directory=True)
    match_dir = _managed_child(matches_root, match_id, "match directory", directory=True)
    clips_dir = _managed_child(match_dir, "clips", "clips directory", directory=True)
    clip_dir = _managed_child(clips_dir, clip_id, "clip directory", directory=True)
    calibration_dir = _managed_child(match_dir, "calibration", "calibration directory", directory=True)
    calibration = _managed_child(calibration_dir, "camera_alignment.veo", "calibration", directory=False)
    clip_video = _managed_child(clip_dir, "source.mp4", "clip video", directory=False)
    tracking_dir = _optional_managed_directory(clip_dir, "tracking", "tracking directory")
    tracking_root = _optional_managed_directory(tracking_dir, "v1", "Tracking V1 root")

    metadata = _load_match_metadata(match_dir, match_id)
    clip = _clip_metadata(metadata, clip_id)
    clip_media = _clip_media_metadata(metadata, clip)
    _validate_metadata_path(clip_media, clip_video, root, "clip video")
    _validate_metadata_path(metadata.get("calibration") if metadata else None, calibration, root, "calibration")

    video_fps, video_frame_count, video_width, video_height = _probe_video(clip_video)
    video_duration_s = video_frame_count / video_fps

    metadata_fps = _metadata_value(clip_media, clip, "fps")
    metadata_frame_count = _metadata_value(clip_media, clip, "frame_count")
    metadata_width = _metadata_value(clip_media, clip, "width")
    metadata_height = _metadata_value(clip_media, clip, "height")
    metadata_duration = _metadata_value(clip_media, clip, "duration_s")

    fps = _positive_float(metadata_fps, "canonical fps") if metadata_fps is not None else video_fps
    frame_count = _positive_int(metadata_frame_count, "canonical frame_count") if metadata_frame_count is not None else video_frame_count
    width = _positive_int(metadata_width, "canonical width") if metadata_width is not None else video_width
    height = _positive_int(metadata_height, "canonical height") if metadata_height is not None else video_height
    duration_s = _positive_float(metadata_duration, "canonical duration_s") if metadata_duration is not None else video_duration_s

    _validate_close(fps, video_fps, "fps")
    _validate_close(float(frame_count), float(video_frame_count), "frame_count")
    if width != video_width or height != video_height:
        raise ValueError("canonical video dimensions do not match the materialized clip video")
    _validate_close(duration_s, video_duration_s, "duration_s")

    return TrackingClipContext(
        data_root=root,
        match_id=match_id,
        clip_id=clip_id,
        match_dir=match_dir,
        clip_dir=clip_dir,
        clip_video=clip_video,
        calibration=calibration,
        tracking_root=tracking_root,
        duration_s=duration_s,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
    )
