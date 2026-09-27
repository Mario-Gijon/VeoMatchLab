"""Mechanical backend bridge from forward tracking rows to Tracking V2 output."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from enum import Enum
from numbers import Real
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.identity_resolution import resolve_identities
from tracking.interpolation import interpolate_trajectories
from tracking.models import PositionKind, TrackLifecycle, TrackSample
from tracking.participant_validation import filter_participants, validate_participants
from tracking.smoothing import smooth_trajectories


FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


def raw_rows_to_trajectories(raw_rows: Iterable[Mapping[str, Any]]) -> dict[int, tuple[TrackSample, ...]]:
    """Reconstruct tracker trajectories from frame-oriented forward rows."""
    trajectories: dict[int, list[TrackSample]] = defaultdict(list)
    for row in raw_rows:
        timestamp = float(row["timestamp"] if "timestamp" in row else row["timestamp_s"])
        frame_index = int(row["frame_index"])
        for track in row.get("tracks", []):
            track_id = int(track["track_id"])
            trajectories[track_id].append(TrackSample(
                track_id=track_id,
                frame_index=frame_index,
                timestamp=timestamp,
                x=float(track["x"]),
                z=float(track["z"]),
                vx=float(track["vx"]),
                vz=float(track["vz"]),
                lifecycle=TrackLifecycle(track["lifecycle"]),
                position_kind=PositionKind(track["position_kind"]),
                measurement_id=track.get("measurement_id"),
                # The forward JSONL diagnostic does not carry snapshot confidence.
                confidence=track.get("confidence"),
            ))
    return {
        track_id: tuple(sorted(samples, key=lambda sample: (sample.frame_index, sample.timestamp)))
        for track_id, samples in sorted(trajectories.items())
    }


def final_trajectories_to_rows(trajectories: Mapping[int, tuple[TrackSample, ...]]) -> list[dict[str, Any]]:
    """Convert final trajectories back into timestamp-sorted artifact rows."""
    grouped: dict[int, list[TrackSample]] = defaultdict(list)
    for track_id, samples in trajectories.items():
        for sample in samples:
            if sample.track_id != track_id:
                raise ValueError(f"Trajectory key {track_id} does not match sample track_id {sample.track_id}")
            grouped[sample.frame_index].append(sample)

    rows: list[dict[str, Any]] = []
    for frame_index in sorted(grouped):
        samples = sorted(grouped[frame_index], key=lambda sample: sample.track_id)
        timestamp = samples[0].timestamp
        if any(not math.isclose(sample.timestamp, timestamp, abs_tol=1e-9) for sample in samples[1:]):
            raise ValueError(f"Frame {frame_index} has inconsistent final timestamps")
        rows.append({
            "frame_index": frame_index,
            "timestamp": timestamp,
            "timestamp_s": timestamp,
            "tracks": [{
                "track_id": sample.track_id,
                "x": sample.x,
                "z": sample.z,
                "vx": sample.vx,
                "vz": sample.vz,
                "lifecycle": sample.lifecycle.value,
                "position_kind": sample.position_kind.value,
                "measurement_id": sample.measurement_id,
            } for sample in samples],
        })
    return rows


def postprocess_tracking_rows(raw_rows: Iterable[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the fixed Tracking V2 offline pipeline over raw forward rows."""
    trajectories = raw_rows_to_trajectories(raw_rows)
    resolved, identity_report = resolve_identities(
        trajectories, field_length=FIELD_LENGTH_M, field_width=FIELD_WIDTH_M,
    )
    participant_report = validate_participants(
        resolved, field_length=FIELD_LENGTH_M, field_width=FIELD_WIDTH_M,
    )
    participants = filter_participants(resolved, participant_report)
    smoothed, smoothing_report = smooth_trajectories(participants)
    final, interpolation_report = interpolate_trajectories(smoothed)
    rows = final_trajectories_to_rows(final)
    report = {
        "identity_resolution": identity_report.to_dict(),
        "participant_validation": participant_report.to_dict(),
        "smoothing": smoothing_report.to_dict(),
        "interpolation": interpolation_report.to_dict(),
        "summary": {
            "raw_track_count": len(trajectories),
            "resolved_track_count": len(resolved),
            "participant_track_count": len(participants),
            "final_track_count": len(final),
            "final_frame_count": len(rows),
        },
    }
    return rows, json_safe(report)


def json_safe(value: Any) -> Any:
    """Recursively make diagnostic data valid strict JSON (not NaN/Infinity)."""
    if isinstance(value, Real) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Enum):
        return json_safe(value.value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_postprocess_outputs(output_dir: Path, rows: Iterable[Mapping[str, Any]], report: Mapping[str, Any]) -> None:
    """Persist V2 output separately from the raw forward-tracker diagnostic."""
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "tracking_frames_final.jsonl"
    final_path.write_text(
        "".join(json.dumps(json_safe(row), allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    (output_dir / "tracking_postprocess_report.json").write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
