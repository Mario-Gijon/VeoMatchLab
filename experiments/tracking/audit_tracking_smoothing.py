#!/usr/bin/env python3
"""Audit offline RTS smoothing on the final resolved participant trajectories.

Pipeline:
    tracking_frames.jsonl
        -> global identity resolution
        -> final participant validation
        -> keep KEEP + UNCERTAIN participants
        -> offline Kalman + RTS smoothing
        -> tracking_smoothing_report.json

This script is diagnostic only. It never rewrites tracking_frames.jsonl or
tracks.person.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.identity_resolution import resolve_identities  # noqa: E402
from tracking.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402
from tracking.participant_validation import (  # noqa: E402
    filter_participants,
    validate_participants,
)
from tracking.smoothing import smooth_trajectories  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_FIELD_LENGTH_M = 105.0
DEFAULT_FIELD_WIDTH_M = 68.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=None,
        help=(
            "Path to tracking_frames.jsonl. If omitted, use the newest file "
            "under data/matches/**/clips/**."
        ),
    )
    parser.add_argument(
        "--field-length",
        type=float,
        default=DEFAULT_FIELD_LENGTH_M,
    )
    parser.add_argument(
        "--field-width",
        type=float,
        default=DEFAULT_FIELD_WIDTH_M,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional report path. Defaults to "
            "tracking_smoothing_report.json beside the input."
        ),
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=0,
        help=(
            "Print up to N largest per-sample position corrections across all "
            "final participants (default: 0)."
        ),
    )
    return parser.parse_args()


def newest_tracking_frames(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    candidates = [
        path
        for path in (data_root / "matches").glob(
            "*/clips/*/*/tracking_frames.jsonl"
        )
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No tracking_frames.jsonl found under {data_root / 'matches'}"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_trajectories(
    path: Path,
) -> tuple[dict[int, tuple[TrackSample, ...]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    trajectories: dict[int, list[TrackSample]] = defaultdict(list)
    frame_count = 0
    first_timestamp: float | None = None
    last_timestamp: float | None = None

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                frame_index = int(row["frame_index"])
                timestamp = float(row.get("timestamp_s", row.get("timestamp")))
                raw_tracks = row["tracks"]
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed tracking frame"
                ) from exc

            if not isinstance(raw_tracks, list):
                raise ValueError(
                    f"{path}:{line_number}: expected 'tracks' list"
                )

            frame_count += 1
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

            for raw in raw_tracks:
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"{path}:{line_number}: invalid track snapshot"
                    )

                try:
                    track_id = int(raw["track_id"])
                    kind = PositionKind(str(raw["position_kind"]))
                    lifecycle = TrackLifecycle(str(raw["lifecycle"]))

                    confidence_raw = raw.get("confidence")
                    confidence = (
                        float(confidence_raw)
                        if confidence_raw is not None
                        else None
                    )

                    trajectories[track_id].append(
                        TrackSample(
                            track_id=track_id,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            x=float(raw["x"]),
                            z=float(raw["z"]),
                            vx=float(raw.get("vx", 0.0)),
                            vz=float(raw.get("vz", 0.0)),
                            lifecycle=lifecycle,
                            position_kind=kind,
                            measurement_id=raw.get("measurement_id"),
                            confidence=confidence,
                        )
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: malformed track snapshot {raw!r}"
                    ) from exc

    normalized = {
        track_id: tuple(
            sorted(
                samples,
                key=lambda item: (item.frame_index, item.timestamp),
            )
        )
        for track_id, samples in sorted(trajectories.items())
    }

    metadata = {
        "source": str(path.resolve()),
        "frames": frame_count,
        "raw_track_ids": len(normalized),
        "first_timestamp_s": first_timestamp,
        "last_timestamp_s": last_timestamp,
        "duration_s": (
            max(0.0, float(last_timestamp - first_timestamp))
            if first_timestamp is not None and last_timestamp is not None
            else 0.0
        ),
    }

    return normalized, metadata


def sample_corrections(
    original: dict[int, tuple[TrackSample, ...]],
    smoothed: dict[int, tuple[TrackSample, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for track_id in sorted(original):
        before = original[track_id]
        after = smoothed[track_id]

        if len(before) != len(after):
            raise RuntimeError(
                f"Track {track_id}: smoothing changed sample count"
            )

        for raw, smooth in zip(before, after):
            correction = math.hypot(
                smooth.x - raw.x,
                smooth.z - raw.z,
            )

            rows.append(
                {
                    "track_id": track_id,
                    "frame_index": raw.frame_index,
                    "timestamp_s": raw.timestamp,
                    "position_kind": raw.position_kind.value,
                    "correction_m": correction,
                    "raw_x": raw.x,
                    "raw_z": raw.z,
                    "smoothed_x": smooth.x,
                    "smoothed_z": smooth.z,
                    "raw_speed_m_s": math.hypot(raw.vx, raw.vz),
                    "smoothed_speed_m_s": math.hypot(
                        smooth.vx,
                        smooth.vz,
                    ),
                }
            )

    rows.sort(
        key=lambda row: (
            -float(row["correction_m"]),
            int(row["track_id"]),
            int(row["frame_index"]),
        )
    )
    return rows


def print_report(
    *,
    source: Path,
    participants: dict[int, tuple[TrackSample, ...]],
    smoothed: dict[int, tuple[TrackSample, ...]],
    identity_report,
    validation_report,
    smoothing_report,
    largest_corrections: list[dict[str, object]],
    show_samples: int,
) -> None:
    print()
    print("Tracking V2.4 RTS smoothing audit")
    print(f"source : {source}")
    print(
        "pipeline: "
        f"raw={identity_report.raw_track_count}  "
        f"resolved={identity_report.final_identity_count}  "
        f"participants={len(participants)}  "
        f"smoothed={len(smoothed)}"
    )
    print(
        "participant: "
        f"KEEP={len(validation_report.kept_track_ids)}  "
        f"UNCERTAIN={len(validation_report.uncertain_track_ids)}  "
        f"REJECT={len(validation_report.rejected_track_ids)}"
    )

    print()
    print("SMOOTHING METRICS")
    print(
        f"{'ID':>4} {'OBS':>5} {'PRED':>5} "
        f"{'OBS_RMS':>9} {'OBS_MED':>9} {'OBS_MAX':>9} "
        f"{'PRED_RMS':>10} {'PRED_MAX':>10} "
        f"{'VRAW':>8} {'VSMTH':>8}"
    )
    print("-" * 100)

    for track_id, metrics in sorted(smoothing_report.tracks.items()):
        print(
            f"{track_id:>4} "
            f"{metrics.observed_count:>5} "
            f"{metrics.predicted_count:>5} "
            f"{metrics.observed_rms_correction_m:>8.3f}m "
            f"{metrics.observed_median_correction_m:>8.3f}m "
            f"{metrics.observed_max_correction_m:>8.3f}m "
            f"{metrics.predicted_rms_correction_m:>9.3f}m "
            f"{metrics.predicted_max_correction_m:>9.3f}m "
            f"{metrics.raw_max_speed_m_s:>7.2f} "
            f"{metrics.smoothed_max_speed_m_s:>7.2f}"
        )

    global_observed = [
        row
        for row in largest_corrections
        if row["position_kind"] == PositionKind.OBSERVED.value
    ]
    global_predicted = [
        row
        for row in largest_corrections
        if row["position_kind"] == PositionKind.PREDICTED.value
    ]

    print()
    print("GLOBAL CORRECTION SUMMARY")
    print(
        "observed : "
        + _correction_summary(global_observed)
    )
    print(
        "predicted: "
        + _correction_summary(global_predicted)
    )

    speed_before = max(
        (
            math.hypot(sample.vx, sample.vz)
            for samples in participants.values()
            for sample in samples
        ),
        default=0.0,
    )
    speed_after = max(
        (
            math.hypot(sample.vx, sample.vz)
            for samples in smoothed.values()
            for sample in samples
        ),
        default=0.0,
    )

    print(
        f"max speed: raw={speed_before:.2f}m/s  "
        f"smoothed={speed_after:.2f}m/s"
    )

    if show_samples > 0:
        print()
        print(
            f"TOP {min(show_samples, len(largest_corrections))} "
            "POSITION CORRECTIONS"
        )
        print(
            f"{'ID':>4} {'FRAME':>6} {'TIME':>7} {'KIND':<12} "
            f"{'CORR':>8} {'VRAW':>8} {'VSMTH':>8}"
        )
        print("-" * 68)

        for row in largest_corrections[:show_samples]:
            print(
                f"{int(row['track_id']):>4} "
                f"{int(row['frame_index']):>6} "
                f"{float(row['timestamp_s']):>6.2f}s "
                f"{str(row['position_kind']):<12} "
                f"{float(row['correction_m']):>7.3f}m "
                f"{float(row['raw_speed_m_s']):>7.2f} "
                f"{float(row['smoothed_speed_m_s']):>7.2f}"
            )


def _correction_summary(
    rows: list[dict[str, object]],
) -> str:
    if not rows:
        return "none"

    values = sorted(
        float(row["correction_m"])
        for row in rows
    )

    rms = math.sqrt(
        sum(value * value for value in values)
        / len(values)
    )
    median_value = _percentile(values, 0.50)
    p90 = _percentile(values, 0.90)
    p99 = _percentile(values, 0.99)
    max_value = values[-1]

    return (
        f"n={len(values)} "
        f"rms={rms:.3f}m "
        f"median={median_value:.3f}m "
        f"p90={p90:.3f}m "
        f"p99={p99:.3f}m "
        f"max={max_value:.3f}m"
    )


def _percentile(
    sorted_values: list[float],
    q: float,
) -> float:
    if not sorted_values:
        return 0.0

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return (
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


def build_payload(
    *,
    source_metadata: dict[str, Any],
    field_length: float,
    field_width: float,
    identity_report,
    validation_report,
    smoothing_report,
    largest_corrections: list[dict[str, object]],
) -> dict[str, Any]:
    return {
        "version": "tracking-rts-smoothing-v2.4",
        "source": source_metadata,
        "field": {
            "length_m": field_length,
            "width_m": field_width,
        },
        "summary": {
            "raw_tracks": identity_report.raw_track_count,
            "resolved_identities": identity_report.final_identity_count,
            "final_participants": len(validation_report.accepted_track_ids),
            "smoothed_tracks": len(smoothing_report.track_ids),
        },
        "identity_resolution": identity_report.to_dict(),
        "participant_validation": validation_report.to_dict(),
        "smoothing": smoothing_report.to_dict(),
        "largest_corrections": largest_corrections[:100],
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]

    return value


def main() -> None:
    args = parse_args()

    if args.field_length <= 0.0 or args.field_width <= 0.0:
        raise SystemExit("--field-length and --field-width must be > 0")

    if args.show_samples < 0:
        raise SystemExit("--show-samples must be >= 0")

    try:
        tracking_path = (
            args.tracking_frames.expanduser().resolve()
            if args.tracking_frames is not None
            else newest_tracking_frames().resolve()
        )
        trajectories, source_metadata = load_trajectories(tracking_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not trajectories:
        raise SystemExit(f"No track samples found in {tracking_path}")

    resolved, identity_report = resolve_identities(
        trajectories,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    validation_report = validate_participants(
        resolved,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    participants = filter_participants(
        resolved,
        validation_report,
    )

    smoothed, smoothing_report = smooth_trajectories(
        participants,
    )

    largest_corrections = sample_corrections(
        participants,
        smoothed,
    )

    print_report(
        source=tracking_path,
        participants=participants,
        smoothed=smoothed,
        identity_report=identity_report,
        validation_report=validation_report,
        smoothing_report=smoothing_report,
        largest_corrections=largest_corrections,
        show_samples=args.show_samples,
    )

    payload = build_payload(
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
        identity_report=identity_report,
        validation_report=validation_report,
        smoothing_report=smoothing_report,
        largest_corrections=largest_corrections,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name(
            "tracking_smoothing_report.json"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nreport : {output_path}")


if __name__ == "__main__":
    main()
