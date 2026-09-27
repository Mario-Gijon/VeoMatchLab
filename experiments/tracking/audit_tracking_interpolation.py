#!/usr/bin/env python3
"""Audit final 30 FPS interpolation on smoothed participant trajectories.

Pipeline:
    tracking_frames.jsonl
        -> global identity resolution
        -> participant validation
        -> RTS smoothing
        -> cubic Hermite interpolation to source-frame cadence
        -> tracking_interpolation_report.json

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
from tracking.interpolation import interpolate_trajectories  # noqa: E402
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
            "tracking_interpolation_report.json beside the input."
        ),
    )
    parser.add_argument(
        "--show-samples",
        type=int,
        default=20,
        help=(
            "Print up to N interpolated samples with the largest Hermite-vs-"
            "linear deviation (default: 20)."
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
        "tracking_rows": frame_count,
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


def verify_original_samples_preserved(
    smoothed: dict[int, tuple[TrackSample, ...]],
    interpolated: dict[int, tuple[TrackSample, ...]],
) -> tuple[int, list[str]]:
    checked = 0
    failures: list[str] = []

    for track_id in sorted(smoothed):
        output_by_frame = {
            sample.frame_index: sample
            for sample in interpolated[track_id]
        }

        for original in smoothed[track_id]:
            checked += 1
            actual = output_by_frame.get(original.frame_index)

            if actual is None:
                failures.append(
                    f"track {track_id} frame {original.frame_index}: missing"
                )
                continue

            if actual != original:
                failures.append(
                    f"track {track_id} frame {original.frame_index}: modified"
                )

    return checked, failures


def inspect_interpolated_samples(
    smoothed: dict[int, tuple[TrackSample, ...]],
    interpolated: dict[int, tuple[TrackSample, ...]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for track_id in sorted(interpolated):
        originals = smoothed[track_id]
        original_by_frame = {
            sample.frame_index: sample
            for sample in originals
        }

        original_frames = sorted(original_by_frame)

        for sample in interpolated[track_id]:
            if sample.frame_index in original_by_frame:
                continue

            left_frame = max(
                frame
                for frame in original_frames
                if frame < sample.frame_index
            )
            right_frame = min(
                frame
                for frame in original_frames
                if frame > sample.frame_index
            )

            left = original_by_frame[left_frame]
            right = original_by_frame[right_frame]

            alpha = (
                (sample.frame_index - left.frame_index)
                / (right.frame_index - left.frame_index)
            )

            linear_x = left.x + alpha * (right.x - left.x)
            linear_z = left.z + alpha * (right.z - left.z)

            deviation = math.hypot(
                sample.x - linear_x,
                sample.z - linear_z,
            )

            rows.append(
                {
                    "track_id": track_id,
                    "frame_index": sample.frame_index,
                    "timestamp_s": sample.timestamp,
                    "left_frame": left.frame_index,
                    "right_frame": right.frame_index,
                    "x": sample.x,
                    "z": sample.z,
                    "vx": sample.vx,
                    "vz": sample.vz,
                    "speed_m_s": math.hypot(sample.vx, sample.vz),
                    "hermite_vs_linear_deviation_m": deviation,
                }
            )

    rows.sort(
        key=lambda row: (
            -float(row["hermite_vs_linear_deviation_m"]),
            int(row["track_id"]),
            int(row["frame_index"]),
        )
    )
    return rows


def contiguous_frame_check(
    trajectories: dict[int, tuple[TrackSample, ...]],
) -> tuple[int, list[str]]:
    checked_tracks = 0
    failures: list[str] = []

    for track_id, samples in sorted(trajectories.items()):
        checked_tracks += 1
        if not samples:
            continue

        frames = [sample.frame_index for sample in samples]
        expected = list(range(frames[0], frames[-1] + 1))

        if frames != expected:
            missing = sorted(set(expected) - set(frames))
            preview = ",".join(str(frame) for frame in missing[:10])
            failures.append(
                f"track {track_id}: {len(missing)} missing frame(s)"
                + (f" [{preview}]" if preview else "")
            )

    return checked_tracks, failures


def print_report(
    *,
    source: Path,
    participants: dict[int, tuple[TrackSample, ...]],
    smoothed: dict[int, tuple[TrackSample, ...]],
    interpolated: dict[int, tuple[TrackSample, ...]],
    identity_report,
    validation_report,
    smoothing_report,
    interpolation_report,
    preservation_checked: int,
    preservation_failures: list[str],
    contiguous_checked: int,
    contiguous_failures: list[str],
    sample_rows: list[dict[str, object]],
    show_samples: int,
) -> None:
    print()
    print("Tracking V2.5 30 FPS interpolation audit")
    print(f"source : {source}")
    print(
        "pipeline: "
        f"raw={identity_report.raw_track_count}  "
        f"resolved={identity_report.final_identity_count}  "
        f"participants={len(participants)}  "
        f"smoothed={len(smoothed)}  "
        f"interpolated={len(interpolated)}"
    )
    print(
        "participant: "
        f"KEEP={len(validation_report.kept_track_ids)}  "
        f"UNCERTAIN={len(validation_report.uncertain_track_ids)}  "
        f"REJECT={len(validation_report.rejected_track_ids)}"
    )
    print(
        f"inserted samples: {interpolation_report.total_inserted_samples}"
    )

    print()
    print("INTERPOLATION METRICS")
    print(
        f"{'ID':>4} {'IN':>5} {'OUT':>5} {'ADD':>5} "
        f"{'GAPS':>5} {'SKIP':>5} "
        f"{'MAXGAP':>8} {'VMAX':>8} {'CURVE':>8}"
    )
    print("-" * 72)

    for track_id, metrics in sorted(interpolation_report.tracks.items()):
        print(
            f"{track_id:>4} "
            f"{metrics.input_samples:>5} "
            f"{metrics.output_samples:>5} "
            f"{metrics.inserted_samples:>5} "
            f"{metrics.filled_gaps:>5} "
            f"{metrics.skipped_large_gaps:>5} "
            f"{metrics.max_filled_gap_seconds:>7.3f}s "
            f"{metrics.max_interpolated_speed_m_s:>7.2f} "
            f"{metrics.max_hermite_vs_linear_deviation_m:>7.3f}m"
        )

    print()
    print("INVARIANTS")
    print(
        f"original smoothed samples preserved: "
        f"{preservation_checked} checked, "
        f"{len(preservation_failures)} failure(s)"
    )
    if preservation_failures:
        for failure in preservation_failures[:20]:
            print(f"  {failure}")

    print(
        f"contiguous final frame ranges: "
        f"{contiguous_checked} tracks checked, "
        f"{len(contiguous_failures)} failure(s)"
    )
    if contiguous_failures:
        for failure in contiguous_failures[:20]:
            print(f"  {failure}")

    all_interpolated = [
        sample
        for samples in interpolated.values()
        for sample in samples
        if sample.position_kind is PositionKind.INTERPOLATED
    ]

    invalid_metadata = [
        sample
        for sample in all_interpolated
        if sample.measurement_id is not None
        or sample.confidence is not None
    ]

    max_speed = max(
        (
            math.hypot(sample.vx, sample.vz)
            for sample in all_interpolated
        ),
        default=0.0,
    )

    print(
        f"INTERPOLATED metadata clean: "
        f"{len(all_interpolated) - len(invalid_metadata)}/"
        f"{len(all_interpolated)}"
    )
    print(f"max interpolated speed: {max_speed:.2f}m/s")

    if sample_rows:
        deviations = sorted(
            float(row["hermite_vs_linear_deviation_m"])
            for row in sample_rows
        )
        print(
            "Hermite-vs-linear deviation: "
            f"median={_percentile(deviations, 0.50):.4f}m "
            f"p90={_percentile(deviations, 0.90):.4f}m "
            f"p99={_percentile(deviations, 0.99):.4f}m "
            f"max={deviations[-1]:.4f}m"
        )

    if show_samples > 0 and sample_rows:
        print()
        print(
            f"TOP {min(show_samples, len(sample_rows))} "
            "INTERPOLATED CURVE DEVIATIONS"
        )
        print(
            f"{'ID':>4} {'FRAME':>6} {'TIME':>7} "
            f"{'LEFT':>6} {'RIGHT':>6} "
            f"{'DEV':>8} {'SPEED':>8}"
        )
        print("-" * 58)

        for row in sample_rows[:show_samples]:
            print(
                f"{int(row['track_id']):>4} "
                f"{int(row['frame_index']):>6} "
                f"{float(row['timestamp_s']):>6.2f}s "
                f"{int(row['left_frame']):>6} "
                f"{int(row['right_frame']):>6} "
                f"{float(row['hermite_vs_linear_deviation_m']):>7.4f}m "
                f"{float(row['speed_m_s']):>7.2f}"
            )

    print()
    print("FINAL SAMPLE SUPPORT")
    print(
        f"{'ID':>4} {'TOTAL':>6} {'OBS':>5} {'PRED':>5} "
        f"{'INT':>5} {'FIRST':>6} {'LAST':>6}"
    )
    print("-" * 52)

    for track_id, samples in sorted(interpolated.items()):
        observed = sum(
            sample.position_kind is PositionKind.OBSERVED
            for sample in samples
        )
        predicted = sum(
            sample.position_kind is PositionKind.PREDICTED
            for sample in samples
        )
        inserted = sum(
            sample.position_kind is PositionKind.INTERPOLATED
            for sample in samples
        )

        print(
            f"{track_id:>4} "
            f"{len(samples):>6} "
            f"{observed:>5} "
            f"{predicted:>5} "
            f"{inserted:>5} "
            f"{samples[0].frame_index:>6} "
            f"{samples[-1].frame_index:>6}"
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
    interpolation_report,
    preservation_checked: int,
    preservation_failures: list[str],
    contiguous_checked: int,
    contiguous_failures: list[str],
    sample_rows: list[dict[str, object]],
) -> dict[str, Any]:
    return {
        "version": "tracking-30fps-interpolation-v2.5",
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
            "interpolated_tracks": len(interpolation_report.track_ids),
            "inserted_samples": interpolation_report.total_inserted_samples,
            "preservation_failures": len(preservation_failures),
            "contiguous_frame_failures": len(contiguous_failures),
        },
        "identity_resolution": identity_report.to_dict(),
        "participant_validation": validation_report.to_dict(),
        "smoothing": smoothing_report.to_dict(),
        "interpolation": interpolation_report.to_dict(),
        "invariants": {
            "original_samples_checked": preservation_checked,
            "original_sample_failures": preservation_failures,
            "contiguous_tracks_checked": contiguous_checked,
            "contiguous_frame_failures": contiguous_failures,
        },
        "largest_interpolated_curve_deviations": sample_rows[:100],
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

    interpolated, interpolation_report = interpolate_trajectories(
        smoothed,
    )

    preservation_checked, preservation_failures = (
        verify_original_samples_preserved(
            smoothed,
            interpolated,
        )
    )

    contiguous_checked, contiguous_failures = contiguous_frame_check(
        interpolated,
    )

    sample_rows = inspect_interpolated_samples(
        smoothed,
        interpolated,
    )

    print_report(
        source=tracking_path,
        participants=participants,
        smoothed=smoothed,
        interpolated=interpolated,
        identity_report=identity_report,
        validation_report=validation_report,
        smoothing_report=smoothing_report,
        interpolation_report=interpolation_report,
        preservation_checked=preservation_checked,
        preservation_failures=preservation_failures,
        contiguous_checked=contiguous_checked,
        contiguous_failures=contiguous_failures,
        sample_rows=sample_rows,
        show_samples=args.show_samples,
    )

    payload = build_payload(
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
        identity_report=identity_report,
        validation_report=validation_report,
        smoothing_report=smoothing_report,
        interpolation_report=interpolation_report,
        preservation_checked=preservation_checked,
        preservation_failures=preservation_failures,
        contiguous_checked=contiguous_checked,
        contiguous_failures=contiguous_failures,
        sample_rows=sample_rows,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name(
            "tracking_interpolation_report.json"
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
