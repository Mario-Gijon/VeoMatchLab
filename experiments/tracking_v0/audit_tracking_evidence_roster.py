#!/usr/bin/env python3
"""Audit Tracking V2.7.2 evidence-scored fixed-roster assignment.

Default pipeline:

    raw GlobalTracker tracking_frames.jsonl
        -> per-track evidence score
        -> pairwise same-identity score
        -> exact 23-root MILP with optional JUNK tracks
        -> trim unsupported predicted edges
        -> report only

The auditor does NOT write tracks.person and does NOT alter the analysis job.

Use --after-resolution only as an A/B diagnostic. V2.7 is intentionally able
to work directly from raw GlobalTracker tracklets.
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

from tracking_v0.evidence_roster import (  # noqa: E402
    EvidenceRosterConfig,
    EvidenceRosterInfeasibleError,
    TrackAssignmentKind,
    resolve_evidence_roster,
)
from tracking_v0.identity_resolution import resolve_identities  # noqa: E402
from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402


DEFAULT_DATA_ROOT = ROOT / "data"
FIELD_LENGTH_M = 105.0
FIELD_WIDTH_M = 68.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-frames", type=Path, default=None)
    parser.add_argument("--roster-size", type=int, default=23)
    parser.add_argument(
        "--after-resolution",
        action="store_true",
        help=(
            "A/B diagnostic: run existing resolve_identities first. "
            "Default is to score raw GlobalTracker tracklets directly."
        ),
    )
    parser.add_argument("--show-tracks", type=int, default=60)
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
    input_name: str,
    raw_count: int,
    final,
    report,
    show_tracks: int,
    show_links: int,
    show_candidates: int,
) -> None:
    print()
    print("Tracking V2.7.2 evidence-scored roster audit")
    print(f"source            : {source}")
    print(f"input stage       : {input_name}")
    print(f"raw tracks        : {raw_count}")
    print(f"scored input      : {report.input_track_count}")
    print(f"target roster     : {report.roster_size}")
    print(f"selected tracks   : {report.selected_track_count}")
    print(f"junk tracks       : {report.junk_track_count} {list(report.junk_track_ids)}")
    print(f"root tracks       : {len(report.root_track_ids)} {list(report.root_track_ids)}")
    print(f"fragment tracks   : {len(report.fragment_track_ids)} {list(report.fragment_track_ids)}")
    print(f"final identities  : {len(final)}")
    print("pair policy       : local-support handover >= 0.70 may override moderate OBS overlap")

    print()
    print("TRACK EVIDENCE / ASSIGNMENT")
    print(
        f"{'ID':>4} {'DECISION':>9} {'REAL':>6} {'ANCH':>6} {'BASE':>6} {'GHOST':>6} "
        f"{'OBS':>5} {'FRAC':>6} {'SPAN':>7} {'RUN':>7} "
        f"{'TAIL':>7} {'REP':>5} {'PARENT':>7} {'Pscore':>7}"
    )
    print("-" * 112)

    rows = sorted(
        report.track_evidence.values(),
        key=lambda row: (
            report.assignments[row.track_id].kind
            is not TrackAssignmentKind.JUNK,
            row.real_score,
            row.track_id,
        ),
    )

    for evidence in rows[:show_tracks]:
        assignment = report.assignments[evidence.track_id]
        print(
            f"{evidence.track_id:>4} "
            f"{assignment.kind.value:>9} "
            f"{evidence.real_score:>6.3f} "
            f"{evidence.anchor_score:>6.3f} "
            f"{evidence.base_real_score:>6.3f} "
            f"{evidence.ghost_score:>6.3f} "
            f"{evidence.observed_count:>5} "
            f"{evidence.observed_fraction:>6.3f} "
            f"{evidence.observed_span_seconds:>6.2f}s "
            f"{evidence.longest_observed_run_seconds:>6.2f}s "
            f"{evidence.trailing_prediction_seconds:>6.2f}s "
            f"{str(assignment.representative_id):>5} "
            f"{str(assignment.parent_id):>7} "
            f"{('-' if assignment.parent_score is None else f'{assignment.parent_score:.3f}'):>7}"
        )

    print()
    print("SELECTED IDENTITY LINKS")
    print(
        f"{'P':>4} {'->':>2} {'C':>4} {'SCORE':>7} {'MODE':>15} "
        f"{'HAND':>6} {'Dmed':>7} {'GAP':>7} {'DIST':>7}"
    )
    print("-" * 82)

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
        distance = (
            f"{e.temporal_endpoint_distance_m:.2f}"
            if e.temporal_endpoint_distance_m is not None
            else "-"
        )
        print(
            f"{link.parent_id:>4} -> {link.child_id:>4} "
            f"{link.score:>7.3f} "
            f"{link.mode.value:>15} "
            f"{e.handover_fraction:>6.2f} "
            f"{dmed:>7} "
            f"{gap:>7} "
            f"{distance:>7}"
        )

    print()
    print("FINAL COMPONENTS")
    for component in report.components:
        representative = report.representative_by_track[component[0]]
        if representative is None:
            continue
        print(
            f"  canonical {representative:>3}: "
            + " + ".join(str(track_id) for track_id in component)
        )

    if report.junk_track_ids and show_candidates > 0:
        print()
        print("TOP IDENTITY CANDIDATES FOR JUNK TRACKS")
        for track_id in report.junk_track_ids:
            print(f"\nTrack {track_id}:")
            for pair in report.pair_rankings.get(track_id, ())[:show_candidates]:
                other = (
                    pair.later_id
                    if pair.earlier_id == track_id
                    else pair.earlier_id
                )
                suffix = " CONFLICT" if pair.hard_conflict else ""
                print(
                    f"  <-> {other:>3} "
                    f"score={pair.score:.3f} "
                    f"mode={pair.mode.value}"
                    f"{suffix}"
                )

    trimmed = sum(
        metrics.removed_leading_samples
        + metrics.removed_trailing_samples
        for metrics in report.edge_trim.values()
    )
    print()
    print(f"unsupported edge-prediction samples trimmed: {trimmed}")


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

    if args.after_resolution:
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
        final, report = resolve_evidence_roster(
            input_trajectories,
            config=EvidenceRosterConfig(
                roster_size=args.roster_size,
            ),
        )
    except EvidenceRosterInfeasibleError as exc:
        raise SystemExit(
            f"Evidence-roster V2.7 is infeasible: {exc}"
        ) from exc

    print_report(
        source=source,
        input_name=input_name,
        raw_count=len(raw),
        final=final,
        report=report,
        show_tracks=args.show_tracks,
        show_links=args.show_links,
        show_candidates=args.show_candidates,
    )

    payload = {
        "version": "tracking-evidence-roster-v2.7.2",
        "source": metadata,
        "input_stage": input_name,
        "identity_resolution": identity_payload,
        "report": report.to_dict(),
        "final_tracks": {
            str(track_id): {
                "samples": len(samples),
                "observed_samples": sum(
                    sample.position_kind is PositionKind.OBSERVED
                    for sample in samples
                ),
                "predicted_samples": sum(
                    sample.position_kind is PositionKind.PREDICTED
                    for sample in samples
                ),
                "first_frame": samples[0].frame_index,
                "last_frame": samples[-1].frame_index,
            }
            for track_id, samples in sorted(final.items())
        },
    }

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else source.with_name(
            "tracking_evidence_roster_report.json"
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
