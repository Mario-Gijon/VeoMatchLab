#!/usr/bin/env python3
"""Audit final participant validation after global identity resolution.

Pipeline:
    tracking_frames.jsonl
        -> Tracking V2.2 global identity resolution
        -> Tracking V2.3 participant validation
        -> keep KEEP + UNCERTAIN
        -> reject only high-confidence residual non-participants
        -> tracking_participant_validation_report.json

This script is diagnostic only. It never rewrites tracking_frames.jsonl.
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

from tracking_v0.identity_resolution import resolve_identities  # noqa: E402
from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402
from tracking_v0.participant_validation import (  # noqa: E402
    ParticipantDecision,
    filter_participants,
    validate_participants,
)


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
            "tracking_participant_validation_report.json beside the input."
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


def print_report(
    *,
    source: Path,
    resolved: dict[int, tuple[TrackSample, ...]],
    final: dict[int, tuple[TrackSample, ...]],
    identity_report,
    validation_report,
) -> None:
    print()
    print("Tracking V2.3 participant validation audit")
    print(f"source : {source}")

    print(
        "identity: "
        f"raw={identity_report.raw_track_count}  "
        f"accepted-before-resolution={identity_report.accepted_before_resolution}  "
        f"resolved={identity_report.final_identity_count}"
    )

    print(
        "participant: "
        f"KEEP={len(validation_report.kept_track_ids)}  "
        f"UNCERTAIN={len(validation_report.uncertain_track_ids)}  "
        f"REJECT={len(validation_report.rejected_track_ids)}  "
        f"FINAL={len(final)}"
    )

    print()
    print("PARTICIPANT DECISIONS")
    print(
        f"{'ID':>4} {'DECISION':<9} "
        f"{'OBS':>5} {'OBS%':>6} {'SPAN':>7} "
        f"{'STRICT':>7} {'Dmed':>8} {'Dmax':>8} "
        f"{'RUN':>7}  REASONS"
    )
    print("-" * 112)

    for track_id, result in sorted(validation_report.results.items()):
        metrics = result.metrics
        print(
            f"{track_id:>4} "
            f"{result.decision.value.upper():<9} "
            f"{metrics.observed_count:>5} "
            f"{100.0 * metrics.observed_fraction:>5.1f}% "
            f"{metrics.observed_span_seconds:>6.2f}s "
            f"{100.0 * metrics.strict_observed_fraction:>6.1f}% "
            f"{metrics.median_field_depth_m:>7.2f}m "
            f"{metrics.max_field_depth_m:>7.2f}m "
            f"{metrics.longest_observed_run_seconds:>6.2f}s  "
            + ", ".join(result.reasons)
        )

    print()
    print("REJECTED FINAL IDENTITIES")
    if not validation_report.rejected_track_ids:
        print("None.")
    else:
        for track_id in validation_report.rejected_track_ids:
            result = validation_report.results[track_id]
            metrics = result.metrics

            print(
                f"ID {track_id}: "
                f"{', '.join(result.reasons)} | "
                f"obs={metrics.observed_count} "
                f"obs%={100.0 * metrics.observed_fraction:.1f}% "
                f"strict={100.0 * metrics.strict_observed_fraction:.1f}% "
                f"Dmed={metrics.median_field_depth_m:.2f}m "
                f"Dmax={metrics.max_field_depth_m:.2f}m "
                f"run={metrics.longest_observed_run_seconds:.2f}s"
            )

    print()
    print("FINAL ACCEPTED IDS")
    print(", ".join(str(track_id) for track_id in sorted(final)))

    print()
    print("FINAL SUPPORT")
    print(
        f"{'ID':>4} {'SAMPLES':>7} {'OBS':>5} {'PRED':>5} "
        f"{'INT':>5} {'FIRST':>7} {'LAST':>7}"
    )
    print("-" * 52)

    for track_id, samples in sorted(final.items()):
        observed = sum(
            item.position_kind is PositionKind.OBSERVED
            for item in samples
        )
        predicted = sum(
            item.position_kind is PositionKind.PREDICTED
            for item in samples
        )
        interpolated = sum(
            item.position_kind is PositionKind.INTERPOLATED
            for item in samples
        )

        print(
            f"{track_id:>4} "
            f"{len(samples):>7} "
            f"{observed:>5} "
            f"{predicted:>5} "
            f"{interpolated:>5} "
            f"{samples[0].timestamp:>6.2f}s "
            f"{samples[-1].timestamp:>6.2f}s"
        )

    print()
    print(
        "count flow: "
        f"{identity_report.raw_track_count} raw "
        f"-> {identity_report.accepted_before_resolution} quality-accepted "
        f"-> {len(resolved)} resolved identities "
        f"-> {len(final)} final participants"
    )


def build_payload(
    *,
    source_metadata: dict[str, Any],
    field_length: float,
    field_width: float,
    resolved: dict[int, tuple[TrackSample, ...]],
    final: dict[int, tuple[TrackSample, ...]],
    identity_report,
    validation_report,
) -> dict[str, Any]:
    return {
        "version": "tracking-participant-validation-v2.3",
        "source": source_metadata,
        "field": {
            "length_m": field_length,
            "width_m": field_width,
        },
        "summary": {
            "raw_tracks": identity_report.raw_track_count,
            "quality_accepted": identity_report.accepted_before_resolution,
            "resolved_identities": len(resolved),
            "participant_keep": len(validation_report.kept_track_ids),
            "participant_uncertain": len(validation_report.uncertain_track_ids),
            "participant_reject": len(validation_report.rejected_track_ids),
            "final_participants": len(final),
        },
        "identity_resolution": identity_report.to_dict(),
        "participant_validation": validation_report.to_dict(),
        "final_tracks": {
            str(track_id): {
                "samples": len(samples),
                "observed_samples": sum(
                    item.position_kind is PositionKind.OBSERVED
                    for item in samples
                ),
                "predicted_samples": sum(
                    item.position_kind is PositionKind.PREDICTED
                    for item in samples
                ),
                "interpolated_samples": sum(
                    item.position_kind is PositionKind.INTERPOLATED
                    for item in samples
                ),
                "first_timestamp_s": (
                    samples[0].timestamp if samples else None
                ),
                "last_timestamp_s": (
                    samples[-1].timestamp if samples else None
                ),
            }
            for track_id, samples in sorted(final.items())
        },
    }


def json_safe(value: Any) -> Any:
    """Recursively replace non-standard JSON floats with null."""
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

    final = filter_participants(
        resolved,
        validation_report,
    )

    print_report(
        source=tracking_path,
        resolved=resolved,
        final=final,
        identity_report=identity_report,
        validation_report=validation_report,
    )

    payload = build_payload(
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
        resolved=resolved,
        final=final,
        identity_report=identity_report,
        validation_report=validation_report,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name(
            "tracking_participant_validation_report.json"
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
