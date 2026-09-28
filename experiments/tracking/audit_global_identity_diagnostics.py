#!/usr/bin/env python3
"""V2.8A diagnostic audit for a previously solved global identity report.

This script changes NOTHING.  It explains why apparently real/high-quality units
were sent to JUNK and where the solver may be wasting ROOT slots.

Reads:
  global_identity_solution_v28a.json

Prints:
  * high-quality JUNK units + their strongest candidate relations,
  * compatible ROOT<->ROOT pairs that could potentially free a root slot,
  * weak/suspicious fragment assignments,
  * raw tracks represented by multiple roots,
  * raw tracks whose units are entirely junk.

This is intended to diagnose the objective/constraints before V2.8B.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug/"
            "global_identity_solution_v28a.json"
        ),
    )
    parser.add_argument(
        "--high-quality-junk",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=8,
    )
    return parser.parse_args()


def pair_key(a: str, b: str) -> str:
    left, right = sorted((a, b))
    return f"{left}|{right}"


def main():
    args = parse_args()
    payload = json.loads(
        args.report.read_text(encoding="utf-8")
    )

    units = payload["units"]
    pairs = payload["pair_evidence"]
    solution = payload["solution"]
    assignments = solution["assignments"]
    config = payload["solver_config"]

    roots = {
        key
        for key, row in assignments.items()
        if row["state"] == "root"
    }
    fragments = {
        key
        for key, row in assignments.items()
        if row["state"] == "fragment"
    }
    junk = {
        key
        for key, row in assignments.items()
        if row["state"] == "junk"
    }

    def relation(a: str, b: str):
        return pairs.get(pair_key(a, b))

    print()
    print("V2.8A DIAGNOSTICS")
    print(f"units        : {len(units)}")
    print(f"roots        : {len(roots)}")
    print(f"fragments    : {len(fragments)}")
    print(f"junk         : {len(junk)}")
    print(
        "fragment min : "
        f"{config['min_fragment_support']:.3f}"
    )
    print(
        "root Q tol   : "
        f"{config['root_quality_tolerance']:.3f}"
    )

    print()
    print("HIGH-QUALITY JUNK")
    print("=" * 100)

    high_junk = sorted(
        (
            key
            for key in junk
            if float(units[key]["quality"])
            >= args.high_quality_junk
        ),
        key=lambda key: (
            -float(units[key]["quality"]),
            key,
        ),
    )

    if not high_junk:
        print("(none)")
    else:
        for key in high_junk:
            unit = units[key]
            print()
            print(
                f"{key}  Q={unit['quality']:.3f}  "
                f"obs={unit['observed_count']}  "
                f"span={unit['observed_span_s']:.2f}s  "
                f"team={unit['team_label']}  "
                f"appearance={unit['appearance_label']}"
            )

            candidates = []
            for other in units:
                if other == key:
                    continue
                rel = relation(key, other)
                if rel is None:
                    continue

                other_assignment = assignments[other]
                other_root = (
                    other
                    if other_assignment["state"] == "root"
                    else other_assignment.get("root_key")
                )

                root_q_ok = (
                    float(units[other]["quality"])
                    + float(config["root_quality_tolerance"])
                    >= float(unit["quality"])
                )
                support_ok = (
                    float(rel["fused_support_score"])
                    >= float(config["min_fragment_support"])
                )
                conflict_ok = (
                    rel["conflict"] == "none"
                )

                candidates.append(
                    (
                        float(rel["fused_support_score"]),
                        other,
                        rel,
                        root_q_ok,
                        support_ok,
                        conflict_ok,
                        other_assignment,
                        other_root,
                    )
                )

            candidates.sort(
                key=lambda row: (
                    -row[0],
                    row[1],
                )
            )

            print(
                f"{'CAND':>8} {'STATE':>9} {'ROOT':>8} "
                f"{'SUP':>6} {'REID':>6} {'GEO':>6} "
                f"{'OVLP':>6} {'CONF':>7} "
                f"{'S?':>3} {'Q?':>3} {'C?':>3}"
            )
            print("-" * 94)

            for (
                support,
                other,
                rel,
                root_q_ok,
                support_ok,
                conflict_ok,
                other_assignment,
                other_root,
            ) in candidates[:args.top]:
                print(
                    f"{other:>8} "
                    f"{other_assignment['state']:>9} "
                    f"{str(other_root):>8} "
                    f"{support:>6.3f} "
                    f"{float(rel['reid_score']):>6.3f} "
                    f"{float(rel['geometry_score']):>6.3f} "
                    f"{float(rel['simultaneous_fraction']):>6.2f} "
                    f"{rel['conflict']:>7} "
                    f"{str(support_ok):>3} "
                    f"{str(root_q_ok):>3} "
                    f"{str(conflict_ok):>3}"
                )

    print()
    print("STRONGEST COMPATIBLE ROOT ↔ ROOT RELATIONS")
    print("=" * 100)

    root_pairs = []
    root_list = sorted(roots)
    for index, left in enumerate(root_list):
        for right in root_list[index + 1:]:
            rel = relation(left, right)
            if rel is None or rel["conflict"] != "none":
                continue
            root_pairs.append(
                (
                    float(rel["fused_support_score"]),
                    left,
                    right,
                    rel,
                )
            )

    root_pairs.sort(
        key=lambda row: (
            -row[0],
            row[1],
            row[2],
        )
    )

    print(
        f"{'PAIR':>18} {'SUP':>6} {'REID':>6} "
        f"{'GEO':>6} {'OVLP':>6} {'TEAM':>13}"
    )
    print("-" * 70)

    for support, left, right, rel in root_pairs[:args.top * 2]:
        print(
            f"{left+'↔'+right:>18} "
            f"{support:>6.3f} "
            f"{float(rel['reid_score']):>6.3f} "
            f"{float(rel['geometry_score']):>6.3f} "
            f"{float(rel['simultaneous_fraction']):>6.2f} "
            f"{rel['team_compatibility']:>13}"
        )

    print()
    print("WEAKEST CURRENT FRAGMENT ASSIGNMENTS")
    print("=" * 100)

    fragment_rows = []
    for key in fragments:
        assignment = assignments[key]
        root = assignment["root_key"]
        rel = relation(key, root)
        fragment_rows.append(
            (
                float(assignment.get("pair_support") or 0.0),
                key,
                root,
                float(units[key]["quality"]),
                float(units[root]["quality"]),
                rel,
            )
        )

    fragment_rows.sort(
        key=lambda row: (
            row[0],
            row[1],
        )
    )

    print(
        f"{'UNIT':>8} {'ROOT':>8} {'SUP':>6} "
        f"{'Qunit':>6} {'Qroot':>6} {'REID':>6} "
        f"{'GEO':>6} {'OVLP':>6}"
    )
    print("-" * 72)

    for support, key, root, q_unit, q_root, rel in fragment_rows:
        print(
            f"{key:>8} {root:>8} "
            f"{support:>6.3f} "
            f"{q_unit:>6.3f} "
            f"{q_root:>6.3f} "
            f"{float(rel['reid_score']):>6.3f} "
            f"{float(rel['geometry_score']):>6.3f} "
            f"{float(rel['simultaneous_fraction']):>6.2f}"
        )

    print()
    print("MULTI-ROOT RAW TRACKS")
    print("=" * 100)

    roots_by_track = defaultdict(list)
    for key in roots:
        roots_by_track[int(units[key]["track_id"])].append(key)

    multi = {
        track_id: sorted(keys)
        for track_id, keys in roots_by_track.items()
        if len(keys) > 1
    }

    if not multi:
        print("(none)")
    else:
        for track_id, keys in sorted(multi.items()):
            print(
                f"track {track_id:>3}: "
                + ", ".join(keys)
            )
            for index, left in enumerate(keys):
                for right in keys[index + 1:]:
                    rel = relation(left, right)
                    if rel is None:
                        continue
                    print(
                        f"    {left}↔{right}: "
                        f"support={float(rel['fused_support_score']):.3f}, "
                        f"conflict={rel['conflict']}, "
                        f"reid={float(rel['reid_score']):.3f}, "
                        f"geo={float(rel['geometry_score']):.3f}"
                    )

    print()
    print("RAW TRACKS ENTIRELY JUNK")
    print("=" * 100)

    by_track = defaultdict(list)
    for key, unit in units.items():
        by_track[int(unit["track_id"])].append(key)

    all_junk_tracks = []
    for track_id, keys in sorted(by_track.items()):
        if all(key in junk for key in keys):
            all_junk_tracks.append((track_id, keys))

    if not all_junk_tracks:
        print("(none)")
    else:
        for track_id, keys in all_junk_tracks:
            details = ", ".join(
                f"{key}(Q={float(units[key]['quality']):.3f},"
                f"obs={units[key]['observed_count']})"
                for key in sorted(keys)
            )
            print(
                f"track {track_id:>3}: {details}"
            )

    print()
    print(
        "Diagnostic only. Nothing was re-solved or written."
    )


if __name__ == "__main__":
    main()
