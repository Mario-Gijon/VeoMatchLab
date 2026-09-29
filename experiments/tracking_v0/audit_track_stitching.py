#!/usr/bin/env python3
"""Audit Tracking V2.2 stitching on an existing tracking JSONL run.

Pipeline:
    tracking_frames.jsonl
        -> reconstruct forward trajectories
        -> Tracking V2.1 quality analysis
        -> drop only REJECT tracks
        -> Tracking V2.2 offline stitching
        -> print chosen links/chains
        -> write tracking_stitching_report.json

This script is diagnostic only. It does not modify persisted tracking artifacts,
RF-DETR, same-frame association, or the web application.
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

from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402
from tracking_v0.quality import analyze_trajectories  # noqa: E402
from tracking_v0.stitching import (  # noqa: E402
    StitchingConfig,
    StitchingReport,
    stitch_trajectories,
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
            "Optional JSON report path. Defaults to "
            "tracking_stitching_report.json beside the input."
        ),
    )
    parser.add_argument(
        "--max-gap-seconds",
        type=float,
        default=None,
        help="Override StitchingConfig.max_gap_seconds for diagnosis.",
    )
    parser.add_argument(
        "--show-singletons",
        action="store_true",
        help="Also print accepted tracks that were not stitched.",
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
            "No tracking_frames.jsonl found under "
            f"{data_root / 'matches'}/<match>/clips/<clip>/<job>/"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def load_trajectories(
    path: Path,
) -> tuple[dict[int, tuple[TrackSample, ...]], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    trajectories: dict[int, list[TrackSample]] = defaultdict(list)
    frames = 0
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

            try:
                frame_index = int(row["frame_index"])
                timestamp = float(
                    row.get("timestamp_s", row.get("timestamp"))
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}:{line_number}: invalid frame metadata"
                ) from exc

            frames += 1
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

            raw_tracks = row.get("tracks")
            if not isinstance(raw_tracks, list):
                raise ValueError(
                    f"{path}:{line_number}: expected 'tracks' list"
                )

            for raw in raw_tracks:
                if not isinstance(raw, dict):
                    raise ValueError(
                        f"{path}:{line_number}: invalid track object"
                    )

                try:
                    kind = PositionKind(str(raw["position_kind"]))
                    lifecycle = TrackLifecycle(str(raw["lifecycle"]))
                    track_id = int(raw["track_id"])
                    x = float(raw["x"])
                    z = float(raw["z"])
                    vx = float(raw.get("vx", 0.0))
                    vz = float(raw.get("vz", 0.0))
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: malformed track {raw!r}"
                    ) from exc

                confidence_value = raw.get("confidence")
                confidence = (
                    float(confidence_value)
                    if confidence_value is not None
                    else None
                )

                trajectories[track_id].append(
                    TrackSample(
                        track_id=track_id,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        x=x,
                        z=z,
                        vx=vx,
                        vz=vz,
                        lifecycle=lifecycle,
                        position_kind=kind,
                        measurement_id=raw.get("measurement_id"),
                        confidence=confidence,
                    )
                )

    normalized = {
        track_id: tuple(
            sorted(
                samples,
                key=lambda sample: (sample.timestamp, sample.frame_index),
            )
        )
        for track_id, samples in sorted(trajectories.items())
    }

    metadata = {
        "source": str(path.resolve()),
        "frames": frames,
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


def print_audit(
    *,
    source: Path,
    quality_report,
    stitching_report: StitchingReport,
    merged: dict[int, tuple[TrackSample, ...]],
    show_singletons: bool,
) -> None:
    print()
    print("Tracking V2.2 stitching audit")
    print(f"source : {source}")
    print(
        "quality: "
        f"KEEP={len(quality_report.kept_track_ids)}  "
        f"UNCERTAIN={len(quality_report.uncertain_track_ids)}  "
        f"REJECT={len(quality_report.rejected_track_ids)}"
    )
    print(
        "identity: "
        f"accepted-before={len(quality_report.accepted_track_ids)}  "
        f"after-stitching={len(merged)}  "
        f"links={len(stitching_report.links)}"
    )
    print()

    if quality_report.rejected_track_ids:
        print(
            "Rejected before stitching: "
            + ", ".join(
                str(track_id)
                for track_id in quality_report.rejected_track_ids
            )
        )
    else:
        print("Rejected before stitching: none")

    print()
    print("SELECTED LINKS")
    print(
        f"{'FROM':>5} {'TO':>5} {'GAP':>7} "
        f"{'FWD':>7} {'BACK':>7} {'DIST':>7} "
        f"{'VREQ':>7} {'DV':>7} {'GATE':>7} {'COST':>7}"
    )
    print("-" * 88)

    if not stitching_report.links:
        print("No stitch links selected.")
    else:
        for link in stitching_report.links:
            c = link.candidate
            print(
                f"{link.predecessor_id:>5} "
                f"{link.successor_id:>5} "
                f"{c.gap_seconds:>6.2f}s "
                f"{c.forward_prediction_error_m:>6.2f}m "
                f"{c.backward_prediction_error_m:>6.2f}m "
                f"{c.endpoint_distance_m:>6.2f}m "
                f"{c.implied_speed_m_s:>5.2f}m/s "
                f"{c.velocity_difference_m_s:>5.2f}m/s "
                f"{c.position_gate_m:>6.2f}m "
                f"{c.cost:>7.3f}"
            )

    print()
    print("MERGED CHAINS")
    merged_chains = [
        chain for chain in stitching_report.chains
        if len(chain) > 1
    ]

    if merged_chains:
        for chain in merged_chains:
            canonical = chain[0]
            print(
                f"ID {canonical}: "
                + " -> ".join(str(track_id) for track_id in chain)
            )
    else:
        print("No multi-track chains.")

    if show_singletons:
        print()
        print("SINGLETONS")
        for chain in stitching_report.chains:
            if len(chain) == 1:
                print(chain[0])

    print()
    print("ID REMAP")
    remapped = [
        (track_id, canonical)
        for track_id, canonical
        in sorted(stitching_report.canonical_id_by_track.items())
        if track_id != canonical
    ]
    if remapped:
        for track_id, canonical in remapped:
            print(f"{track_id} -> {canonical}")
    else:
        print("No IDs remapped.")


def report_payload(
    *,
    source_metadata: dict[str, Any],
    field_length: float,
    field_width: float,
    config: StitchingConfig,
    quality_report,
    stitching_report: StitchingReport,
    merged: dict[int, tuple[TrackSample, ...]],
) -> dict[str, Any]:
    return {
        "version": "tracking-stitching-v2.2",
        "source": source_metadata,
        "field": {
            "length_m": field_length,
            "width_m": field_width,
        },
        "config": {
            key: value
            for key, value in vars(config).items()
        }
        if hasattr(config, "__dict__")
        else {
            field_name: getattr(config, field_name)
            for field_name in config.__dataclass_fields__
        },
        "summary": {
            "raw_track_ids": source_metadata["raw_track_ids"],
            "quality_keep": len(quality_report.kept_track_ids),
            "quality_uncertain": len(quality_report.uncertain_track_ids),
            "quality_reject": len(quality_report.rejected_track_ids),
            "accepted_before_stitching": len(
                quality_report.accepted_track_ids
            ),
            "canonical_after_stitching": len(merged),
            "stitch_links": len(stitching_report.links),
            "merged_chains": sum(
                len(chain) > 1 for chain in stitching_report.chains
            ),
        },
        "quality": quality_report.to_dict(),
        "stitching": stitching_report.to_dict(),
        "canonical_tracks": {
            str(track_id): {
                "samples": len(samples),
                "first_timestamp_s": (
                    samples[0].timestamp if samples else None
                ),
                "last_timestamp_s": (
                    samples[-1].timestamp if samples else None
                ),
                "observed_samples": sum(
                    sample.position_kind is PositionKind.OBSERVED
                    for sample in samples
                ),
                "predicted_samples": sum(
                    sample.position_kind is PositionKind.PREDICTED
                    for sample in samples
                ),
                "interpolated_samples": sum(
                    sample.position_kind is PositionKind.INTERPOLATED
                    for sample in samples
                ),
            }
            for track_id, samples in sorted(merged.items())
        },
    }


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

    quality_report = analyze_trajectories(
        trajectories,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    config = StitchingConfig()
    if args.max_gap_seconds is not None:
        if args.max_gap_seconds <= 0.0:
            raise SystemExit("--max-gap-seconds must be > 0")
        config = StitchingConfig(
            **{
                field_name: (
                    args.max_gap_seconds
                    if field_name == "max_gap_seconds"
                    else getattr(config, field_name)
                )
                for field_name in config.__dataclass_fields__
            }
        )

    merged, stitching_report = stitch_trajectories(
        trajectories,
        config=config,
        allowed_track_ids=quality_report.accepted_track_ids,
    )

    print_audit(
        source=tracking_path,
        quality_report=quality_report,
        stitching_report=stitching_report,
        merged=merged,
        show_singletons=args.show_singletons,
    )

    payload = report_payload(
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
        config=config,
        quality_report=quality_report,
        stitching_report=stitching_report,
        merged=merged,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name("tracking_stitching_report.json")
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nreport : {output_path}")


if __name__ == "__main__":
    main()
