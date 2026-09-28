#!/usr/bin/env python3
"""Audit Tracking V2.6.2 exact overlap-aware fixed-roster identity clustering.

Diagnostic pipeline:

    raw tracking_frames.jsonl
        -> existing resolve_identities(...)
        -> V2.6.2 overlap-aware pair evidence
        -> global exact-K identity-tree MILP
        -> report only

This script never rewrites tracking_frames.jsonl and never publishes
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

from tracking.fixed_roster import (  # noqa: E402
    FixedRosterConfig,
    FixedRosterInfeasibleError,
    resolve_fixed_roster,
)
from tracking.identity_resolution import resolve_identities  # noqa: E402
from tracking.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data"
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-frames", type=Path, default=None)
    parser.add_argument("--roster-size", type=int, default=23)
    parser.add_argument(
        "--raw-input",
        action="store_true",
        help="Skip existing resolve_identities and feed raw tracker IDs directly.",
    )
    parser.add_argument("--show-links", type=int, default=40)
    parser.add_argument("--show-candidates", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
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
    rows = 0
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
                timestamp = float(
                    row.get("timestamp_s", row.get("timestamp"))
                )
                raw_tracks = row["tracks"]
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed tracking frame"
                ) from exc

            rows += 1
            if first_timestamp is None:
                first_timestamp = timestamp
            last_timestamp = timestamp

            for raw in raw_tracks:
                try:
                    track_id = int(raw["track_id"])
                    trajectories[track_id].append(
                        TrackSample(
                            track_id=track_id,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            x=float(raw["x"]),
                            z=float(raw["z"]),
                            vx=float(raw.get("vx", 0.0)),
                            vz=float(raw.get("vz", 0.0)),
                            lifecycle=TrackLifecycle(
                                str(raw["lifecycle"])
                            ),
                            position_kind=PositionKind(
                                str(raw["position_kind"])
                            ),
                            measurement_id=raw.get("measurement_id"),
                            confidence=(
                                float(raw["confidence"])
                                if raw.get("confidence") is not None
                                else None
                            ),
                        )
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: malformed track {raw!r}"
                    ) from exc

    normalized = {
        track_id: tuple(
            sorted(
                samples,
                key=lambda sample: (
                    sample.frame_index,
                    sample.timestamp,
                ),
            )
        )
        for track_id, samples in sorted(trajectories.items())
    }

    return normalized, {
        "source": str(path.resolve()),
        "rows": rows,
        "raw_tracks": len(normalized),
        "first_timestamp_s": first_timestamp,
        "last_timestamp_s": last_timestamp,
    }


def print_report(
    *,
    source: Path,
    raw_count: int,
    input_name: str,
    input_count: int,
    final,
    report,
    show_links: int,
    show_candidates: int,
) -> None:
    print()
    print("Tracking V2.6.2 overlap-aware fixed-roster audit")
    print(f"source              : {source}")
    print(f"raw tracks          : {raw_count}")
    print(f"input stage         : {input_name}")
    print(f"resolver input      : {input_count}")
    print(f"target roster       : {report.roster_size}")
    print(f"candidate parents   : {report.candidate_parent_links}")
    print(f"hard conflict pairs : {report.hard_conflict_pairs}")
    print(
        "forced roots        : "
        f"{len(report.forced_roots_without_parent_candidate)} "
        f"{list(report.forced_roots_without_parent_candidate)}"
    )
    print(f"final identities    : {report.final_identity_count}")
    print(
        "link confidence     : "
        f"HIGH={len(report.high_confidence_links)}  "
        f"MEDIUM={len(report.medium_confidence_links)}  "
        f"LOW={len(report.low_confidence_links)}"
    )

    print()
    print("GLOBALLY SELECTED PARENT LINKS")
    print(
        f"{'P':>4} {'->':>2} {'C':>4} "
        f"{'SCORE':>7} {'CONF':>7} {'MODE':>15} "
        f"{'HAND':>6} {'Dmed':>7} {'GAP':>7} {'DIST':>7}"
    )
    print("-" * 86)

    ordered_links = sorted(
        report.selected_parent_links,
        key=lambda link: (
            -link.score,
            link.parent_id,
            link.child_id,
        ),
    )

    for link in ordered_links[:show_links]:
        e = link.evidence
        dmed = (
            f"{e.handover_median_distance_m:.2f}"
            if e.handover_median_distance_m is not None
            else (
                f"{e.simultaneous_median_distance_m:.2f}"
                if e.simultaneous_median_distance_m is not None
                else "-"
            )
        )
        gap = (
            f"{e.temporal_gap_seconds:.2f}"
            if e.temporal_gap_seconds is not None
            else "-"
        )
        dist = (
            f"{e.temporal_endpoint_distance_m:.2f}"
            if e.temporal_endpoint_distance_m is not None
            else "-"
        )

        print(
            f"{link.parent_id:>4} -> {link.child_id:>4} "
            f"{link.score:>7.3f} "
            f"{link.confidence:>7} "
            f"{e.mode.value:>15} "
            f"{e.handover_fraction:>6.2f} "
            f"{dmed:>7} "
            f"{gap:>7} "
            f"{dist:>7}"
        )

    print()
    print("FINAL IDENTITY COMPONENTS")
    for component in report.components:
        representative = report.representative_by_track[component[0]]
        samples = final[representative]
        observed = sum(
            sample.position_kind is PositionKind.OBSERVED
            for sample in samples
        )
        print(
            f"  canonical {representative:>3}: "
            + " + ".join(str(track_id) for track_id in component)
            + f"    observed={observed}"
        )

    weak = sorted(
        report.low_confidence_links,
        key=lambda link: link.score,
    )

    if weak:
        print()
        print("LOW-CONFIDENCE LINKS — inspect these first")
        for link in weak:
            e = link.evidence
            print(
                f"  {link.parent_id} -> {link.child_id}: "
                f"{link.score:.3f} ({e.mode.value}), "
                f"hand_cov={e.handover_coverage:.2f}, "
                f"hand={e.handover_fraction:.2f}, "
                f"hand_dmed={e.handover_median_distance_m}, "
                f"gap={e.temporal_gap_seconds}, "
                f"temporal_dist={e.temporal_endpoint_distance_m}"
            )

    interesting_ids = {
        track_id
        for link in weak
        for track_id in (
            link.parent_id,
            link.child_id,
        )
    }

    if not interesting_ids:
        interesting_ids = {
            track_id
            for component in report.components
            if len(component) > 1
            for track_id in component
        }

    if show_candidates > 0 and interesting_ids:
        print()
        print("TOP PAIR CANDIDATES")
        for track_id in sorted(interesting_ids):
            print(f"\nTrack {track_id}:")
            for e in report.candidate_rankings.get(
                track_id,
                (),
            )[:show_candidates]:
                other = (
                    e.later_id
                    if e.earlier_id == track_id
                    else e.earlier_id
                )
                conflict = " CONFLICT" if e.hard_conflict else ""
                print(
                    f"  <-> {other:>3} "
                    f"score={e.score:.3f} "
                    f"mode={e.mode.value}"
                    f"{conflict} "
                    f"sim={e.simultaneous_observed_fraction:.2f} "
                    f"simD={e.simultaneous_median_distance_m}"
                )


def json_safe(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            json_safe(item)
            for item in value
        ]
    return value


def main() -> None:
    args = parse_args()

    source = (
        args.tracking_frames.expanduser().resolve()
        if args.tracking_frames is not None
        else newest_tracking_frames().resolve()
    )

    try:
        raw, metadata = load_trajectories(source)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    input_trajectories = raw
    input_name = "raw GlobalTracker"
    identity_payload = None

    if not args.raw_input:
        input_trajectories, identity_report = resolve_identities(
            raw,
            field_length=FIELD_LENGTH_M,
            field_width=FIELD_WIDTH_M,
        )
        input_name = (
            "existing identity resolution "
            f"({identity_report.final_identity_count} identities)"
        )
        identity_payload = identity_report.to_dict()

    try:
        final, report = resolve_fixed_roster(
            input_trajectories,
            config=FixedRosterConfig(
                roster_size=args.roster_size,
            ),
        )
    except FixedRosterInfeasibleError as exc:
        raise SystemExit(
            "Fixed-roster V2.6.2 is infeasible with the current evidence "
            f"model: {exc}"
        ) from exc

    print_report(
        source=source,
        raw_count=len(raw),
        input_name=input_name,
        input_count=len(input_trajectories),
        final=final,
        report=report,
        show_links=args.show_links,
        show_candidates=args.show_candidates,
    )

    payload = {
        "version": "tracking-fixed-roster-v2.6.2",
        "source": metadata,
        "input_stage": input_name,
        "identity_resolution": identity_payload,
        "fixed_roster": report.to_dict(),
        "final_tracks": {
            str(track_id): {
                "samples": len(samples),
                "observed_samples": sum(
                    sample.position_kind is PositionKind.OBSERVED
                    for sample in samples
                ),
                "first_frame": samples[0].frame_index,
                "last_frame": samples[-1].frame_index,
                "first_timestamp_s": samples[0].timestamp,
                "last_timestamp_s": samples[-1].timestamp,
            }
            for track_id, samples in sorted(final.items())
        },
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(
            "tracking_fixed_roster_report.json"
        )
    )
    output.write_text(
        json.dumps(
            json_safe(payload),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\nreport : {output}")


if __name__ == "__main__":
    main()
