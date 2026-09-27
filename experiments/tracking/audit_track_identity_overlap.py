#!/usr/bin/env python3
"""Audit temporal overlap between tracker IDs for offline identity consolidation.

Why this exists:
The forward tracker can produce identity fragments that cannot be handled by
simple non-overlapping tracklet stitching. Example:

    ID 5 observed  -> ID 27 observed -> ID 5 observed again

The observed *spans* overlap, so a simple A->B stitch is illegal, but the two IDs
may still represent the same player if their real observations alternate and
their positions agree.

This diagnostic does not merge or delete anything. It measures, for each
non-stable/suspicious track, how its OBSERVED samples relate to every other
accepted track:

- how often the other track exists at the same frame;
- how often both are OBSERVED simultaneously;
- how often suspect OBSERVED coincides with candidate PREDICTED;
- spatial distance at those frames;
- near-distance fractions.

It writes tracking_identity_overlap_report.json beside the input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.models import PositionKind, TrackLifecycle, TrackSample  # noqa: E402
from tracking.quality import (  # noqa: E402
    TrackQualityDecision,
    analyze_trajectories,
)


DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_FIELD_LENGTH_M = 105.0
DEFAULT_FIELD_WIDTH_M = 68.0


@dataclass(slots=True, frozen=True)
class PairOverlap:
    suspect_id: int
    candidate_id: int

    suspect_observed_samples: int
    candidate_observed_samples: int

    compared_frames: int
    coverage_fraction: float

    both_observed_frames: int
    candidate_predicted_frames: int
    candidate_interpolated_frames: int

    both_observed_fraction_of_compared: float
    handover_fraction_of_compared: float

    min_distance_m: float
    median_distance_m: float
    p90_distance_m: float
    max_distance_m: float

    within_0_75m_fraction: float
    within_1_50m_fraction: float
    within_2_50m_fraction: float

    median_velocity_difference_m_s: float

    diagnostic_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=None,
        help=(
            "Path to tracking_frames.jsonl. If omitted, use newest file under "
            "data/matches/**/clips/**."
        ),
    )
    parser.add_argument("--field-length", type=float, default=DEFAULT_FIELD_LENGTH_M)
    parser.add_argument("--field-width", type=float, default=DEFAULT_FIELD_WIDTH_M)
    parser.add_argument(
        "--track-id",
        type=int,
        action="append",
        default=None,
        help=(
            "Inspect only this suspect track ID. Repeat for multiple IDs. "
            "Default: automatically inspect tracks without long stable evidence."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        help="Number of candidate identities printed per suspect (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional JSON report path. Defaults to "
            "tracking_identity_overlap_report.json next to input."
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
) -> dict[int, tuple[TrackSample, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    trajectories: dict[int, list[TrackSample]] = defaultdict(list)

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
                    f"{path}:{line_number}: 'tracks' must be a list"
                )

            for raw in raw_tracks:
                try:
                    track_id = int(raw["track_id"])
                    kind = PositionKind(str(raw["position_kind"]))
                    lifecycle = TrackLifecycle(str(raw["lifecycle"]))
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
                        f"{path}:{line_number}: malformed track snapshot"
                    ) from exc

    return {
        track_id: tuple(
            sorted(
                samples,
                key=lambda sample: (sample.frame_index, sample.timestamp),
            )
        )
        for track_id, samples in sorted(trajectories.items())
    }


def pair_overlap(
    suspect: Sequence[TrackSample],
    candidate: Sequence[TrackSample],
) -> PairOverlap | None:
    suspect_observed = [
        sample
        for sample in suspect
        if sample.position_kind is PositionKind.OBSERVED
    ]
    candidate_observed_count = sum(
        sample.position_kind is PositionKind.OBSERVED
        for sample in candidate
    )

    if not suspect_observed:
        return None

    candidate_by_frame = {
        sample.frame_index: sample
        for sample in candidate
    }

    distances: list[float] = []
    velocity_differences: list[float] = []

    both_observed = 0
    candidate_predicted = 0
    candidate_interpolated = 0

    for sample in suspect_observed:
        other = candidate_by_frame.get(sample.frame_index)
        if other is None:
            continue

        distance = math.hypot(
            sample.x - other.x,
            sample.z - other.z,
        )
        velocity_difference = math.hypot(
            sample.vx - other.vx,
            sample.vz - other.vz,
        )

        if not math.isfinite(distance) or not math.isfinite(velocity_difference):
            continue

        distances.append(distance)
        velocity_differences.append(velocity_difference)

        if other.position_kind is PositionKind.OBSERVED:
            both_observed += 1
        elif other.position_kind is PositionKind.PREDICTED:
            candidate_predicted += 1
        elif other.position_kind is PositionKind.INTERPOLATED:
            candidate_interpolated += 1

    if not distances:
        return None

    compared = len(distances)
    coverage = compared / len(suspect_observed)

    sorted_distances = sorted(distances)
    median_distance = float(median(sorted_distances))
    p90_distance = _percentile(sorted_distances, 0.90)

    within_075 = sum(value <= 0.75 for value in distances) / compared
    within_150 = sum(value <= 1.50 for value in distances) / compared
    within_250 = sum(value <= 2.50 for value in distances) / compared

    both_observed_fraction = both_observed / compared
    handover_fraction = candidate_predicted / compared

    median_velocity_difference = float(median(velocity_differences))

    # Diagnostic ranking only; this is NOT an identity decision.
    #
    # Reward:
    # - temporal coverage of suspect's real observations;
    # - repeated close spatial agreement;
    # - candidate being in prediction when suspect becomes observed.
    #
    # Penalize:
    # - large median distance;
    # - large velocity disagreement;
    # - many simultaneous real observations, because distinct people can run
    #   near each other and both being observed is evidence against a handover.
    proximity = 1.0 / (1.0 + median_distance)
    velocity_term = 1.0 / (1.0 + 0.25 * median_velocity_difference)
    diagnostic_score = (
        coverage
        * proximity
        * velocity_term
        * (
            1.0
            + 0.65 * handover_fraction
            + 0.35 * within_150
            - 0.45 * both_observed_fraction
        )
    )

    return PairOverlap(
        suspect_id=suspect_observed[0].track_id,
        candidate_id=candidate[0].track_id,
        suspect_observed_samples=len(suspect_observed),
        candidate_observed_samples=candidate_observed_count,
        compared_frames=compared,
        coverage_fraction=coverage,
        both_observed_frames=both_observed,
        candidate_predicted_frames=candidate_predicted,
        candidate_interpolated_frames=candidate_interpolated,
        both_observed_fraction_of_compared=both_observed_fraction,
        handover_fraction_of_compared=handover_fraction,
        min_distance_m=float(sorted_distances[0]),
        median_distance_m=median_distance,
        p90_distance_m=p90_distance,
        max_distance_m=float(sorted_distances[-1]),
        within_0_75m_fraction=within_075,
        within_1_50m_fraction=within_150,
        within_2_50m_fraction=within_250,
        median_velocity_difference_m_s=median_velocity_difference,
        diagnostic_score=float(diagnostic_score),
    )


def automatic_suspects(
    quality_report,
) -> tuple[int, ...]:
    """
    Inspect every accepted track that lacks strong long-stable participation.

    This intentionally includes short KEEP tracks as well as UNCERTAIN tracks.
    """
    result: list[int] = []

    for track_id, quality in sorted(quality_report.results.items()):
        if quality.decision is TrackQualityDecision.REJECT:
            continue

        long_stable = (
            "long_stable_strict_field_presence" in quality.reasons
        )

        if not long_stable:
            result.append(track_id)

    return tuple(result)


def print_report(
    *,
    source: Path,
    suspects: Sequence[int],
    pairings: Mapping[int, Sequence[PairOverlap]],
    quality_report,
    top: int,
) -> None:
    print()
    print("Tracking identity-overlap audit")
    print(f"source   : {source}")
    print(
        "quality  : "
        f"KEEP={len(quality_report.kept_track_ids)}  "
        f"UNCERTAIN={len(quality_report.uncertain_track_ids)}  "
        f"REJECT={len(quality_report.rejected_track_ids)}"
    )
    print("suspects : " + ", ".join(str(value) for value in suspects))
    print()

    for suspect_id in suspects:
        quality = quality_report.results.get(suspect_id)
        if quality is None:
            continue

        print(
            f"TRACK {suspect_id} "
            f"[{quality.decision.value.upper()}] "
            f"obs={quality.metrics.observed_count} "
            f"obs%={100 * quality.metrics.observed_fraction:.1f}% "
            f"span={quality.metrics.observed_span_seconds:.2f}s"
        )

        rows = list(pairings.get(suspect_id, ()))[:top]
        if not rows:
            print("  no temporally overlapping candidate tracks")
            print()
            continue

        print(
            "  "
            f"{'CAND':>4} {'COV':>6} {'COBS':>5} {'HAND':>5} "
            f"{'Dmed':>7} {'D90':>7} "
            f"{'<.75':>6} {'<1.5':>6} {'<2.5':>6} "
            f"{'dVmed':>7} {'SCORE':>7}"
        )

        for row in rows:
            print(
                "  "
                f"{row.candidate_id:>4} "
                f"{100 * row.coverage_fraction:>5.1f}% "
                f"{row.both_observed_frames:>5} "
                f"{row.candidate_predicted_frames:>5} "
                f"{row.median_distance_m:>6.2f}m "
                f"{row.p90_distance_m:>6.2f}m "
                f"{100 * row.within_0_75m_fraction:>5.1f}% "
                f"{100 * row.within_1_50m_fraction:>5.1f}% "
                f"{100 * row.within_2_50m_fraction:>5.1f}% "
                f"{row.median_velocity_difference_m_s:>6.2f} "
                f"{row.diagnostic_score:>7.3f}"
            )

        print()


def main() -> None:
    args = parse_args()

    if args.field_length <= 0.0 or args.field_width <= 0.0:
        raise SystemExit("--field-length and --field-width must be > 0")
    if args.top < 1:
        raise SystemExit("--top must be >= 1")

    try:
        tracking_path = (
            args.tracking_frames.expanduser().resolve()
            if args.tracking_frames is not None
            else newest_tracking_frames().resolve()
        )
        trajectories = load_trajectories(tracking_path)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    quality_report = analyze_trajectories(
        trajectories,
        field_length=args.field_length,
        field_width=args.field_width,
    )

    if args.track_id:
        suspects = tuple(
            track_id
            for track_id in args.track_id
            if track_id in trajectories
            and quality_report.results[track_id].decision
            is not TrackQualityDecision.REJECT
        )
    else:
        suspects = automatic_suspects(quality_report)

    accepted_ids = set(quality_report.accepted_track_ids)

    pairings: dict[int, list[PairOverlap]] = {}

    for suspect_id in suspects:
        rows: list[PairOverlap] = []

        for candidate_id in sorted(accepted_ids):
            if candidate_id == suspect_id:
                continue

            overlap = pair_overlap(
                trajectories[suspect_id],
                trajectories[candidate_id],
            )
            if overlap is not None:
                rows.append(overlap)

        rows.sort(
            key=lambda row: (
                -row.diagnostic_score,
                row.median_distance_m,
                -row.coverage_fraction,
                row.candidate_id,
            )
        )
        pairings[suspect_id] = rows

    print_report(
        source=tracking_path,
        suspects=suspects,
        pairings=pairings,
        quality_report=quality_report,
        top=args.top,
    )

    payload = {
        "version": "tracking-identity-overlap-audit-v1",
        "source": str(tracking_path),
        "suspects": list(suspects),
        "quality": quality_report.to_dict(),
        "pairings": {
            str(track_id): [
                row.to_dict()
                for row in pairings.get(track_id, ())
            ]
            for track_id in suspects
        },
    }

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else tracking_path.with_name(
            "tracking_identity_overlap_report.json"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(f"report : {output_path}")


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("percentile requires data")
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = q * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower

    return float(
        sorted_values[lower] * (1.0 - fraction)
        + sorted_values[upper] * fraction
    )


if __name__ == "__main__":
    main()
