#!/usr/bin/env python3
"""Audit Tracking V2.2 global identity resolution on an existing tracking run.

Pipeline:
    tracking_frames.jsonl
        -> quality filtering
        -> handover consolidation on raw accepted tracks
        -> ordinary stitching on raw accepted tracks
        -> global transitive identity graph
        -> component conflict checks
        -> canonical final trajectories
        -> tracking_identity_resolution_report.json

This script is diagnostic only. It never rewrites source tracking artifacts.
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

from tracking.identity_resolution import (  # noqa: E402
    IdentityRelationKind,
    resolve_identities,
)
from tracking.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402


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
            "tracking_identity_resolution_report.json beside the input."
        ),
    )
    parser.add_argument(
        "--show-singletons",
        action="store_true",
        help="Print single-track final components too.",
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
    final: dict[int, tuple[TrackSample, ...]],
    report,
    show_singletons: bool,
) -> None:
    print()
    print("Tracking V2.2 global identity resolution audit")
    print(f"source : {source}")
    print(
        "quality: "
        f"KEEP={len(report.quality.kept_track_ids)}  "
        f"UNCERTAIN={len(report.quality.uncertain_track_ids)}  "
        f"REJECT={len(report.quality.rejected_track_ids)}"
    )
    print(
        "relations: "
        f"HANDOVER={len(report.consolidation.merges)}  "
        f"STITCH={len(report.stitching.links)}  "
        f"ACCEPTED={len(report.accepted_relations)}  "
        f"REJECTED={len(report.rejected_relations)}"
    )
    print(
        "identity: "
        f"raw={report.raw_track_count}  "
        f"accepted-before={report.accepted_before_resolution}  "
        f"final={report.final_identity_count}"
    )
    print()

    if report.quality.rejected_track_ids:
        print(
            "Quality REJECT: "
            + ", ".join(
                str(track_id)
                for track_id in report.quality.rejected_track_ids
            )
        )
    else:
        print("Quality REJECT: none")

    print()
    print("ACCEPTED IDENTITY RELATIONS")
    print(
        f"{'KIND':<9} {'LEFT':>5} {'RIGHT':>5} "
        f"{'STRENGTH':>9}  DETAIL"
    )
    print("-" * 92)

    if not report.accepted_relations:
        print("No accepted relations.")
    else:
        for relation in report.accepted_relations:
            if relation.kind is IdentityRelationKind.HANDOVER:
                detail = (
                    f"score={relation.detail.get('score')} "
                    f"hand={_pct(relation.detail.get('handover_fraction'))} "
                    f"cobs={_pct(relation.detail.get('simultaneous_observed_fraction'))} "
                    f"dmed={_m(relation.detail.get('median_distance_m'))}"
                )
            else:
                detail = (
                    f"cost={relation.detail.get('cost')} "
                    f"gap={_s(relation.detail.get('gap_seconds'))} "
                    f"fwd={_m(relation.detail.get('forward_prediction_error_m'))} "
                    f"back={_m(relation.detail.get('backward_prediction_error_m'))}"
                )

            print(
                f"{relation.kind.value.upper():<9} "
                f"{relation.left_id:>5} "
                f"{relation.right_id:>5} "
                f"{relation.strength:>9.3f}  "
                f"{detail}"
            )

    print()
    print("REJECTED GRAPH RELATIONS")
    if not report.rejected_relations:
        print("None.")
    else:
        for item in report.rejected_relations:
            relation = item.relation
            print(
                f"{relation.kind.value.upper():<9} "
                f"{relation.left_id} -> {relation.right_id}: "
                f"{item.reason}"
            )

    print()
    print("FINAL IDENTITY COMPONENTS")

    multi_components = [
        component
        for component in report.components
        if len(component) > 1
    ]

    if not multi_components:
        print("No merged components.")
    else:
        for component in multi_components:
            canonical = report.canonical_id_by_track[component[0]]
            mapping = " + ".join(str(track_id) for track_id in component)
            print(f"ID {canonical}: {mapping}")

    if show_singletons:
        print()
        print("SINGLETON COMPONENTS")
        for component in report.components:
            if len(component) == 1:
                print(component[0])

    print()
    print("ID REMAP")
    remapped = [
        (track_id, canonical_id)
        for track_id, canonical_id
        in sorted(report.canonical_id_by_track.items())
        if track_id != canonical_id
    ]

    if not remapped:
        print("No IDs remapped.")
    else:
        for track_id, canonical_id in remapped:
            print(f"{track_id} -> {canonical_id}")

    print()
    print("FINAL CANONICAL IDS")
    print(", ".join(str(track_id) for track_id in sorted(final)))

    print()
    print("FINAL TRACK SUPPORT")
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


def build_payload(
    *,
    source_metadata: dict[str, Any],
    field_length: float,
    field_width: float,
    final: dict[int, tuple[TrackSample, ...]],
    report,
) -> dict[str, Any]:
    return {
        "version": "tracking-global-identity-resolution-v2.2",
        "source": source_metadata,
        "field": {
            "length_m": field_length,
            "width_m": field_width,
        },
        **report.to_dict(),
        "final_tracks": {
            str(track_id): {
                "samples": len(samples),
                "first_timestamp_s": (
                    samples[0].timestamp if samples else None
                ),
                "last_timestamp_s": (
                    samples[-1].timestamp if samples else None
                ),
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
            }
            for track_id, samples in sorted(final.items())
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

    final, report = resolve_identities(
        trajectories,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    print_report(
        source=tracking_path,
        final=final,
        report=report,
        show_singletons=args.show_singletons,
    )

    payload = build_payload(
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
        final=final,
        report=report,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name(
            "tracking_identity_resolution_report.json"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nreport : {output_path}")


def _pct(value: object) -> str:
    if value is None:
        return "—"
    return f"{100.0 * float(value):.1f}%"


def _m(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}m"


def _s(value: object) -> str:
    if value is None:
        return "—"
    return f"{float(value):.2f}s"


if __name__ == "__main__":
    main()
