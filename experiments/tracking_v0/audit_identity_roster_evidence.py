#!/usr/bin/env python3
"""V2.8A-V5.1 roster/occupancy audit.

This is diagnostic only.  It does not change the V5 solver.

It answers the ambiguity exposed by counterfactuals:
  * is a high-realness JUNK unit already spatially represented by one selected
    canonical identity (duplicate/shadow)?
  * or does it occupy a distinct physical slot that the current roster is
    missing?
  * are two currently selected canonical identities plausible duplicates that
    could merge and free one of the 23 roster slots?

Focus defaults:
    11:0, 16:0, 13:1, 27:0, 14:1, 31:0
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking_v0.identity_evidence import ObservedTrackPoint  # noqa: E402
from tracking_v0.identity_roster_evidence import (  # noqa: E402
    assign_observations_to_units,
    build_canonical_membership,
    build_frame_canonical_points,
    compute_canonical_pair_evidence,
    compute_unit_occupancy_evidence,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=Path(
            "experiments/tracking_v0/reid_crop_debug/"
            "_tracking_tmp/tracking_frames.jsonl"
        ),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "experiments/tracking_v0/reid_similarity_debug/"
            "identity_graph_solution_v28a_v5.json"
        ),
    )
    parser.add_argument(
        "--roster-size",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--focus",
        nargs="*",
        default=[
            "11:0",
            "16:0",
            "13:1",
            "27:0",
            "14:1",
            "31:0",
        ],
    )
    return parser.parse_args()


def load_raw_observations(path):
    grouped = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line in stream:
            if not line.strip():
                continue

            row = json.loads(
                line
            )
            frame_index = int(
                row["frame_index"]
            )
            timestamp = float(
                row.get(
                    "timestamp_s",
                    row.get("timestamp"),
                )
            )

            for raw in row["tracks"]:
                if str(
                    raw.get(
                        "position_kind",
                        "",
                    )
                ) != "observed":
                    continue

                track_id = int(
                    raw["track_id"]
                )

                grouped[
                    track_id
                ].append(
                    ObservedTrackPoint(
                        track_id=track_id,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        x=float(raw["x"]),
                        z=float(raw["z"]),
                    )
                )

    return {
        track_id: tuple(
            sorted(
                rows,
                key=lambda point: (
                    point.frame_index,
                    point.timestamp,
                ),
            )
        )
        for track_id, rows in grouped.items()
    }


def normalize_pair_map(raw):
    output = {}

    for key, row in raw.items():
        if "|" in key:
            left, right = key.split("|", 1)
        else:
            left, right = key.split("->", 1)

        output[
            tuple(
                sorted(
                    (
                        left,
                        right,
                    )
                )
            )
        ] = row

    return output


def main():
    args = parse_args()

    payload = json.loads(
        args.report.read_text(
            encoding="utf-8",
        )
    )

    raw_observations = load_raw_observations(
        args.tracking_frames
    )

    units = payload["units"]
    pair_evidence = normalize_pair_map(
        payload["pair_evidence"]
    )
    hybrid_edges = normalize_pair_map(
        payload["hybrid_edges"]
    )

    identities = payload[
        "solution"
    ][
        "identities"
    ]

    unit_points = assign_observations_to_units(
        raw_observations,
        units,
    )

    (
        unit_to_canonical,
        canonical_to_units,
    ) = build_canonical_membership(
        identities
    )

    frame_points = build_frame_canonical_points(
        unit_points=unit_points,
        unit_to_canonical=unit_to_canonical,
    )

    print()
    print("V2.8A-V5.1 ROSTER / OCCUPANCY AUDIT")
    print(f"canonical identities : {len(identities)}")
    print(f"unit count           : {len(units)}")
    print(f"roster size          : {args.roster_size}")

    print()
    print("FOCUS UNIT OCCUPANCY")
    print(
        f"{'UNIT':>7} {'Q':>5} {'OBS':>5} "
        f"{'REPR%':>7} {'UNIQ%':>7} {'FILL%':>7} {'OVER%':>7} "
        f"{'MED-N':>6} {'SHADOW':>8} {'S-OV%':>6} {'S-DIST':>7} "
        f"{'HYBRID':>8} {'H-SCORE':>7}"
    )
    print("-" * 118)

    focus_evidence = {}

    for key in args.focus:
        if key not in units:
            continue

        evidence = compute_unit_occupancy_evidence(
            key,
            unit_points=unit_points,
            frame_canonical_points=frame_points,
            unit_to_canonical=unit_to_canonical,
            pair_evidence=pair_evidence,
            hybrid_edges=hybrid_edges,
            roster_size=args.roster_size,
        )

        focus_evidence[
            key
        ] = evidence

        row = units[key]

        print(
            f"{key:>7} "
            f"{float(row['quality']):>5.2f} "
            f"{evidence.observed_frames:>5} "
            f"{100.0 * evidence.represented_fraction:>6.1f}% "
            f"{100.0 * evidence.unique_fraction:>6.1f}% "
            f"{100.0 * evidence.fills_open_slot_fraction:>6.1f}% "
            f"{100.0 * evidence.overflow_if_added_fraction:>6.1f}% "
            f"{evidence.median_selected_identity_count:>6.1f} "
            f"{str(evidence.best_shadow_identity):>8} "
            f"{100.0 * evidence.best_shadow_overlap_fraction:>5.1f}% "
            f"{(evidence.best_shadow_median_distance_m if evidence.best_shadow_median_distance_m is not None else -1.0):>7.2f} "
            f"{str(evidence.best_hybrid_identity):>8} "
            f"{(evidence.best_hybrid_score if evidence.best_hybrid_score is not None else -1.0):>7.3f}"
        )

    print()
    print("FOCUS INTERPRETATION")
    for key, evidence in focus_evidence.items():
        print()
        print(
            f"{key}:"
        )
        print(
            f"  represented by selected ID : "
            f"{evidence.represented_fraction:.3f}"
        )
        print(
            f"  unique physical frames     : "
            f"{evidence.unique_fraction:.3f}"
        )
        print(
            f"  fills an open roster slot  : "
            f"{evidence.fills_open_slot_fraction:.3f}"
        )
        print(
            f"  would exceed roster        : "
            f"{evidence.overflow_if_added_fraction:.3f}"
        )
        print(
            f"  best simultaneous shadow   : "
            f"{evidence.best_shadow_identity}"
        )
        print(
            f"  best hybrid continuation   : "
            f"{evidence.best_hybrid_identity} "
            f"score={evidence.best_hybrid_score}"
        )

    print()
    print("SELECTED CANONICAL PAIRS — POSSIBLE MERGES")
    print(
        f"{'PAIR':>10} {'SIM':>5} {'SIM%':>6} "
        f"{'MED-D':>7} {'Q90-D':>7} {'CONF':>6} "
        f"{'ID-SUP':>7} {'HYBRID':>7} {'H-PAIR':>15}"
    )
    print("-" * 92)

    canonical_ids = sorted(
        canonical_to_units
    )

    canonical_pairs = []

    for left, right in itertools.combinations(
        canonical_ids,
        2,
    ):
        evidence = compute_canonical_pair_evidence(
            left,
            right,
            canonical_to_units=canonical_to_units,
            unit_points=unit_points,
            pair_evidence=pair_evidence,
            hybrid_edges=hybrid_edges,
        )

        canonical_pairs.append(
            evidence
        )

    # Show pairs with actual hybrid evidence first, then close simultaneous
    # identities without a known conflict.
    ranked = sorted(
        canonical_pairs,
        key=lambda row: (
            -row.best_hybrid_score,
            row.has_identity_conflict,
            (
                row.median_distance_m
                if row.median_distance_m is not None
                else 999.0
            ),
            -row.simultaneous_fraction,
        ),
    )

    shown = 0

    for row in ranked:
        interesting = (
            row.best_hybrid_score > 0.0
            or (
                not row.has_identity_conflict
                and row.simultaneous_fraction >= 0.10
                and row.median_distance_m is not None
                and row.median_distance_m <= 4.0
            )
        )

        if not interesting:
            continue

        print(
            f"{str(row.canonical_a)+'↔'+str(row.canonical_b):>10} "
            f"{row.simultaneous_frames:>5} "
            f"{100.0 * row.simultaneous_fraction:>5.1f}% "
            f"{(row.median_distance_m if row.median_distance_m is not None else -1.0):>7.2f} "
            f"{(row.q90_distance_m if row.q90_distance_m is not None else -1.0):>7.2f} "
            f"{str(row.has_identity_conflict):>6} "
            f"{row.best_identity_support:>7.3f} "
            f"{row.best_hybrid_score:>7.3f} "
            f"{str(row.best_hybrid_pair):>15}"
        )

        shown += 1
        if shown >= 25:
            break

    print()
    print("PER-FRAME SELECTED ROSTER OCCUPANCY")

    counts = [
        len(by_canonical)
        for by_canonical in frame_points.values()
    ]

    if counts:
        counts_sorted = sorted(
            counts
        )

        print(
            f"frames                    : {len(counts)}"
        )
        print(
            f"median selected observed  : "
            f"{counts_sorted[len(counts_sorted)//2]}"
        )
        print(
            f"max selected observed     : {max(counts)}"
        )
        print(
            f"frames with >=23 observed : "
            f"{sum(value >= args.roster_size for value in counts)}"
        )
        print(
            f"frames with <=22 observed : "
            f"{sum(value < args.roster_size for value in counts)}"
        )


if __name__ == "__main__":
    main()
