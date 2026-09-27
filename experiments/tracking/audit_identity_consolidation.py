#!/usr/bin/env python3
"""Audit Tracking V2.2 identity consolidation on an existing tracking run.

Pipeline:
    tracking_frames.jsonl
        -> reconstruct forward trajectories
        -> Tracking V2.1 quality analysis
        -> remove only REJECT tracks
        -> Tracking V2.2 identity consolidation
        -> optional non-overlapping stitching on consolidated identities
        -> write tracking_identity_consolidation_report.json

This script is diagnostic only. It never rewrites the source tracking artifact.
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

from tracking.identity_consolidation import (  # noqa: E402
    IdentityConsolidationReport,
    consolidate_identities,
)
from tracking.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402
from tracking.quality import analyze_trajectories  # noqa: E402
from tracking.stitching import StitchingReport, stitch_trajectories  # noqa: E402


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
            "tracking_identity_consolidation_report.json next to the input."
        ),
    )
    parser.add_argument(
        "--no-stitching",
        action="store_true",
        help="Stop after identity consolidation and skip ordinary stitching.",
    )
    parser.add_argument(
        "--top-candidates",
        type=int,
        default=3,
        help="Print this many candidate identities per fragment (default: 3).",
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


def print_identity_report(
    *,
    source: Path,
    quality_report,
    identity_report: IdentityConsolidationReport,
    consolidated: dict[int, tuple[TrackSample, ...]],
    stitching_report: StitchingReport | None,
    stitched: dict[int, tuple[TrackSample, ...]] | None,
    top_candidates: int,
) -> None:
    print()
    print("Tracking V2.2 identity consolidation audit")
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
        f"after-consolidation={len(consolidated)}  "
        f"merges={len(identity_report.merges)}"
    )

    if stitching_report is not None and stitched is not None:
        print(
            "stitching: "
            f"before={len(consolidated)}  "
            f"after={len(stitched)}  "
            f"links={len(stitching_report.links)}"
        )

    print()

    if quality_report.rejected_track_ids:
        print(
            "Quality REJECT: "
            + ", ".join(
                str(track_id)
                for track_id in quality_report.rejected_track_ids
            )
        )
    else:
        print("Quality REJECT: none")

    print()
    print("IDENTITY MERGES")
    print(
        f"{'FRAG':>5} {'->':>2} {'ID':>5} "
        f"{'SCORE':>7} {'RATIO':>7} {'MARGIN':>7} "
        f"{'COV':>6} {'HAND':>6} {'COBS':>6} "
        f"{'Dmed':>7} {'D90':>7} {'dV':>7}"
    )
    print("-" * 90)

    if not identity_report.merges:
        print("No identity merges selected.")
    else:
        for merge in identity_report.merges:
            candidate = merge.candidate
            ratio = (
                "inf"
                if merge.score_ratio == float("inf")
                else f"{merge.score_ratio:.2f}"
            )
            print(
                f"{merge.fragment_id:>5} -> "
                f"{merge.canonical_id:>5} "
                f"{merge.score:>7.3f} "
                f"{ratio:>7} "
                f"{merge.score_margin:>7.3f} "
                f"{100.0 * candidate.temporal_coverage:>5.1f}% "
                f"{100.0 * candidate.handover_fraction:>5.1f}% "
                f"{100.0 * candidate.simultaneous_observed_fraction:>5.1f}% "
                f"{candidate.median_distance_m:>6.2f}m "
                f"{candidate.p90_distance_m:>6.2f}m "
                f"{candidate.median_velocity_difference_m_s:>6.2f}"
            )

    print()
    print("TOP CANDIDATES / REJECTION REASONS")

    interesting_ids = sorted(
        set(identity_report.rejected_proposals)
        | {merge.fragment_id for merge in identity_report.merges}
    )

    for fragment_id in interesting_ids:
        merged_to = identity_report.canonical_id_by_track.get(fragment_id)
        status = (
            f"MERGED->{merged_to}"
            if merged_to is not None and merged_to != fragment_id
            else identity_report.rejected_proposals.get(
                fragment_id,
                "no-decision",
            )
        )

        print(f"\nTrack {fragment_id}: {status}")

        ranking = identity_report.candidate_rankings.get(fragment_id, ())
        if not ranking:
            print("  no candidate identities")
            continue

        print(
            "  "
            f"{'ID':>4} {'SCORE':>7} {'COV':>6} {'HAND':>6} "
            f"{'COBS':>6} {'Dmed':>7} {'D90':>7} "
            f"{'dV':>7} {'PRIOR':>6} {'LATER':>6}"
        )

        for candidate in ranking[:top_candidates]:
            print(
                "  "
                f"{candidate.candidate_id:>4} "
                f"{candidate.score:>7.3f} "
                f"{100.0 * candidate.temporal_coverage:>5.1f}% "
                f"{100.0 * candidate.handover_fraction:>5.1f}% "
                f"{100.0 * candidate.simultaneous_observed_fraction:>5.1f}% "
                f"{candidate.median_distance_m:>6.2f}m "
                f"{candidate.p90_distance_m:>6.2f}m "
                f"{candidate.median_velocity_difference_m_s:>6.2f} "
                f"{candidate.candidate_prior_observed_samples:>6} "
                f"{candidate.candidate_later_observed_samples:>6}"
            )

    if stitching_report is not None:
        print()
        print("POST-CONSOLIDATION STITCHING")

        multi = [
            chain
            for chain in stitching_report.chains
            if len(chain) > 1
        ]

        if not multi:
            print("No additional non-overlapping stitch chains.")
        else:
            for chain in multi:
                print("  " + " -> ".join(str(track_id) for track_id in chain))

    print()
    print("FINAL CANONICAL IDS")
    final_ids = (
        tuple(sorted(stitched))
        if stitched is not None
        else tuple(sorted(consolidated))
    )
    print(", ".join(str(track_id) for track_id in final_ids))


def build_payload(
    *,
    source_metadata: dict[str, Any],
    field_length: float,
    field_width: float,
    quality_report,
    identity_report: IdentityConsolidationReport,
    consolidated: dict[int, tuple[TrackSample, ...]],
    stitching_report: StitchingReport | None,
    stitched: dict[int, tuple[TrackSample, ...]] | None,
) -> dict[str, Any]:
    final = stitched if stitched is not None else consolidated

    payload: dict[str, Any] = {
        "version": "tracking-identity-consolidation-v2.2",
        "source": source_metadata,
        "field": {
            "length_m": field_length,
            "width_m": field_width,
        },
        "summary": {
            "raw_track_ids": source_metadata["raw_track_ids"],
            "quality_keep": len(quality_report.kept_track_ids),
            "quality_uncertain": len(quality_report.uncertain_track_ids),
            "quality_reject": len(quality_report.rejected_track_ids),
            "accepted_before_consolidation": len(
                quality_report.accepted_track_ids
            ),
            "after_identity_consolidation": len(consolidated),
            "identity_merges": len(identity_report.merges),
            "after_optional_stitching": len(final),
            "stitch_links": (
                len(stitching_report.links)
                if stitching_report is not None
                else 0
            ),
        },
        "quality": quality_report.to_dict(),
        "identity_consolidation": identity_report.to_dict(),
        "final_tracks": {
            str(track_id): {
                "samples": len(samples),
                "first_timestamp_s": samples[0].timestamp if samples else None,
                "last_timestamp_s": samples[-1].timestamp if samples else None,
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

    if stitching_report is not None:
        payload["post_consolidation_stitching"] = stitching_report.to_dict()

    return payload


def main() -> None:
    args = parse_args()

    if args.field_length <= 0.0 or args.field_width <= 0.0:
        raise SystemExit("--field-length and --field-width must be > 0")
    if args.top_candidates < 1:
        raise SystemExit("--top-candidates must be >= 1")

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

    consolidated, identity_report = consolidate_identities(
        trajectories,
        quality_report=quality_report,
    )

    stitching_report: StitchingReport | None = None
    stitched: dict[int, tuple[TrackSample, ...]] | None = None

    if not args.no_stitching:
        stitched, stitching_report = stitch_trajectories(
            consolidated,
        )

    print_identity_report(
        source=tracking_path,
        quality_report=quality_report,
        identity_report=identity_report,
        consolidated=consolidated,
        stitching_report=stitching_report,
        stitched=stitched,
        top_candidates=args.top_candidates,
    )

    payload = build_payload(
        source_metadata=source_metadata,
        field_length=args.field_length,
        field_width=args.field_width,
        quality_report=quality_report,
        identity_report=identity_report,
        consolidated=consolidated,
        stitching_report=stitching_report,
        stitched=stitched,
    )

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name(
            "tracking_identity_consolidation_report.json"
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nreport : {output_path}")


if __name__ == "__main__":
    main()
