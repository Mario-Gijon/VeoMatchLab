#!/usr/bin/env python3
"""Audit Tracking V2.1 quality decisions on an existing tracking JSONL run.

This script is intentionally read-only with respect to tracking output:
- it reconstructs trajectories from `tracking_frames.jsonl`;
- it runs `tracking.quality`;
- it prints a compact KEEP / UNCERTAIN / REJECT table;
- it writes `tracking_quality_report.json` next to the input unless overridden.

It does NOT alter RF-DETR, association, GlobalTracker, or persisted web artifacts.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402
from tracking.quality import (  # noqa: E402
    TrackQualityDecision,
    TrackQualityReport,
    analyze_trajectories,
)


DEFAULT_FIELD_LENGTH_M = 105.0
DEFAULT_FIELD_WIDTH_M = 68.0
DEFAULT_DATA_ROOT = ROOT / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=None,
        help=(
            "Path to tracking_frames.jsonl. If omitted, the newest file under "
            "data/matches/**/clips/** is used."
        ),
    )
    parser.add_argument(
        "--field-length",
        type=float,
        default=DEFAULT_FIELD_LENGTH_M,
        help=f"Field length in metres (default: {DEFAULT_FIELD_LENGTH_M:g}).",
    )
    parser.add_argument(
        "--field-width",
        type=float,
        default=DEFAULT_FIELD_WIDTH_M,
        help=f"Field width in metres (default: {DEFAULT_FIELD_WIDTH_M:g}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional report path. Defaults to tracking_quality_report.json "
            "next to tracking_frames.jsonl."
        ),
    )
    parser.add_argument(
        "--show",
        choices=("all", "reject", "uncertain", "keep"),
        default="all",
        help="Limit console rows to one decision class.",
    )
    return parser.parse_args()


def newest_tracking_frames(data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    """Return the newest clip tracking_frames.jsonl under the local data root."""
    matches_root = data_root / "matches"
    candidates = [
        path
        for path in matches_root.glob("*/clips/*/*/tracking_frames.jsonl")
        if path.is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            "No tracking_frames.jsonl found under "
            f"{matches_root}/<match>/clips/<clip>/<job>/"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_trajectories(path: Path) -> tuple[dict[int, tuple[TrackSample, ...]], dict[str, Any]]:
    """Rebuild per-ID temporal trajectories from tracker frame snapshots."""
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
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}: {exc}"
                ) from exc

            frame_index = _required_int(row, "frame_index", path, line_number)
            timestamp = _row_timestamp(row, path, line_number)
            frame_count += 1

            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

            tracks = row.get("tracks")
            if not isinstance(tracks, list):
                raise ValueError(
                    f"{path}:{line_number}: expected 'tracks' to be a list"
                )

            for raw_track in tracks:
                if not isinstance(raw_track, dict):
                    raise ValueError(
                        f"{path}:{line_number}: every track must be an object"
                    )
                track = _track_sample(
                    raw_track,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    path=path,
                    line_number=line_number,
                )
                trajectories[track.track_id].append(track)

    ordered: dict[int, tuple[TrackSample, ...]] = {}
    for track_id, samples in trajectories.items():
        ordered[track_id] = tuple(
            sorted(samples, key=lambda item: (item.timestamp, item.frame_index))
        )

    metadata = {
        "source": str(path.resolve()),
        "frames": frame_count,
        "track_count": len(ordered),
        "first_timestamp_s": first_timestamp,
        "last_timestamp_s": last_timestamp,
        "duration_s": (
            max(0.0, float(last_timestamp - first_timestamp))
            if first_timestamp is not None and last_timestamp is not None
            else 0.0
        ),
    }
    return ordered, metadata


def build_report_payload(
    report: TrackQualityReport,
    *,
    source_metadata: dict[str, Any],
    field_length: float,
    field_width: float,
) -> dict[str, Any]:
    return {
        "version": "tracking-quality-v2.1",
        "field": {
            "length_m": field_length,
            "width_m": field_width,
        },
        "source": source_metadata,
        "summary": {
            "tracks_analyzed": len(report.results),
            "keep": len(report.kept_track_ids),
            "uncertain": len(report.uncertain_track_ids),
            "reject": len(report.rejected_track_ids),
            "accepted": len(report.accepted_track_ids),
        },
        **report.to_dict(),
    }


def print_report(
    report: TrackQualityReport,
    *,
    source: Path,
    field_length: float,
    field_width: float,
    show: str,
) -> None:
    print()
    print("Tracking V2.1 quality audit")
    print(f"source : {source}")
    print(f"field  : {field_length:.2f} x {field_width:.2f} m")
    print(
        "summary: "
        f"KEEP={len(report.kept_track_ids)}  "
        f"UNCERTAIN={len(report.uncertain_track_ids)}  "
        f"REJECT={len(report.rejected_track_ids)}  "
        f"TOTAL={len(report.results)}"
    )
    print()

    header = (
        f"{'ID':>4}  {'DECISION':<9} "
        f"{'OBS':>4} {'OBS%':>6} {'SPAN':>7} "
        f"{'STRICT%':>8} {'EDGE%':>7} "
        f"{'Dmax':>7} {'Dp90':>7} {'Drun':>7}  REASONS"
    )
    print(header)
    print("-" * len(header))

    decision_filter = None if show == "all" else TrackQualityDecision(show)

    results = sorted(
        report.results.values(),
        key=lambda item: (
            _decision_order(item.decision),
            item.track_id,
        ),
    )

    for result in results:
        if decision_filter is not None and result.decision is not decision_filter:
            continue

        metrics = result.metrics
        print(
            f"{result.track_id:>4}  "
            f"{result.decision.value.upper():<9} "
            f"{metrics.observed_count:>4} "
            f"{100.0 * metrics.observed_fraction:>5.1f}% "
            f"{metrics.observed_span_seconds:>6.2f}s "
            f"{100.0 * metrics.strict_observed_fraction:>7.1f}% "
            f"{100.0 * metrics.edge_observed_fraction:>6.1f}% "
            f"{_fmt_depth(metrics.max_field_depth_m):>7} "
            f"{_fmt_depth(metrics.p90_field_depth_m):>7} "
            f"{metrics.longest_observed_run_seconds:>6.2f}s  "
            f"{','.join(result.reasons)}"
        )

    print()
    if report.rejected_track_ids:
        print(
            "REJECT candidates: "
            + ", ".join(str(track_id) for track_id in report.rejected_track_ids)
        )
    else:
        print("REJECT candidates: none")

    if report.uncertain_track_ids:
        print(
            "UNCERTAIN preserved: "
            + ", ".join(str(track_id) for track_id in report.uncertain_track_ids)
        )


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

    report = analyze_trajectories(
        trajectories,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    print_report(
        report,
        source=tracking_path,
        field_length=args.field_length,
        field_width=args.field_width,
        show=args.show,
    )

    payload = build_report_payload(
        report,
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name("tracking_quality_report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nreport : {output_path}")


def _track_sample(
    raw: dict[str, Any],
    *,
    frame_index: int,
    timestamp: float,
    path: Path,
    line_number: int,
) -> TrackSample:
    try:
        track_id = int(raw["track_id"])
        x = float(raw["x"])
        z = float(raw["z"])
        vx = float(raw.get("vx", 0.0))
        vz = float(raw.get("vz", 0.0))
        lifecycle = TrackLifecycle(str(raw["lifecycle"]))
        position_kind = PositionKind(str(raw["position_kind"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid track snapshot: {raw!r}"
        ) from exc

    confidence_value = raw.get("confidence")
    confidence = (
        float(confidence_value)
        if confidence_value is not None
        else None
    )

    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=timestamp,
        x=x,
        z=z,
        vx=vx,
        vz=vz,
        lifecycle=lifecycle,
        position_kind=position_kind,
        measurement_id=raw.get("measurement_id"),
        confidence=confidence,
    )


def _row_timestamp(row: dict[str, Any], path: Path, line_number: int) -> float:
    value = row.get("timestamp_s", row.get("timestamp"))
    if value is None:
        raise ValueError(
            f"{path}:{line_number}: missing timestamp_s/timestamp"
        )
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid timestamp {value!r}"
        ) from exc


def _required_int(
    row: dict[str, Any],
    key: str,
    path: Path,
    line_number: int,
) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"{path}:{line_number}: invalid or missing {key!r}"
        ) from exc


def _decision_order(decision: TrackQualityDecision) -> int:
    return {
        TrackQualityDecision.REJECT: 0,
        TrackQualityDecision.UNCERTAIN: 1,
        TrackQualityDecision.KEEP: 2,
    }[decision]


def _fmt_depth(value: float) -> str:
    if value == float("-inf"):
        return "—"
    return f"{value:.2f}m"


if __name__ == "__main__":
    main()
