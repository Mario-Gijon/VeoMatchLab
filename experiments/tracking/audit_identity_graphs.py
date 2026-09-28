#!/usr/bin/env python3
"""V2.8A-V5 audit: hybrid identity graphs.

This reuses the same cached units, pair identity evidence, and V4.1 temporal
tri-state evidence, but removes the strict linear-path requirement.

A canonical identity is now a connected graph:
  * identity conflicts veto,
  * temporal contradictions veto,
  * temporal support adds confidence,
  * temporal neutral leaves strong identity evidence usable.

No detector/tracker/OSNet rerun. Audit only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Reuse the V4.1 audit data construction so there is no duplicate interpretation
# of caches/schemas.
from experiments.tracking.audit_identity_paths import (  # noqa: E402
    build_pair_evidence,
    build_temporal_transitions,
    build_units,
    load_appearance_labels,
    load_embeddings,
    load_observed_points,
    load_semantics,
)
from tracking.appearance_bank import (  # noqa: E402
    AppearanceBankConfig,
    build_appearance_banks,
)
from tracking.appearance_segments import (  # noqa: E402
    AppearanceSegmentationConfig,
    segment_all_banks,
)
from tracking.identity_graph_solver import (  # noqa: E402
    IdentityGraphSolverConfig,
    build_hybrid_identity_edges,
    generate_candidate_identity_graphs,
    solve_identity_graphs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug/"
            "reid_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=Path(
            "experiments/tracking/reid_crop_debug/"
            "_tracking_tmp/tracking_frames.jsonl"
        ),
    )
    parser.add_argument(
        "--semantic-report",
        type=Path,
        default=Path(
            "experiments/tracking/team_appearance_debug/"
            "semantic_constraints_report.json"
        ),
    )
    parser.add_argument(
        "--appearance-report",
        type=Path,
        default=Path(
            "experiments/tracking/team_appearance_debug/"
            "team_appearance_report.json"
        ),
    )
    parser.add_argument(
        "--identity-count",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug/"
            "identity_graph_solution_v28a_v5.json"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    embeddings = load_embeddings(
        args.embedding_cache
    )
    observed_points = load_observed_points(
        args.tracking_frames
    )
    appearance_labels = load_appearance_labels(
        args.appearance_report
    )
    (
        team_labels,
        role_anchors,
    ) = load_semantics(
        args.semantic_report
    )

    bank_config = AppearanceBankConfig(
        min_quality=0.10,
        max_embeddings_per_track=10,
        strong_similarity_threshold=0.75,
        unique_top_k=3,
    )
    banks = build_appearance_banks(
        embeddings,
        config=bank_config,
    )

    segment_config = AppearanceSegmentationConfig()

    segmentations = segment_all_banks(
        banks,
        source_labels=appearance_labels,
        config=segment_config,
    )

    (
        units,
        unit_points,
        unit_segment,
        clip_span_s,
        max_observed_count,
    ) = build_units(
        segmentations=segmentations,
        observed_points=observed_points,
        team_labels=team_labels,
        role_anchors=role_anchors,
    )

    pair_evidence = build_pair_evidence(
        units=units,
        unit_points=unit_points,
        unit_segment=unit_segment,
        bank_config=bank_config,
    )

    transitions = build_temporal_transitions(
        units=units,
        unit_points=unit_points,
    )

    solver_config = IdentityGraphSolverConfig(
        identity_count=args.identity_count,
    )

    hybrid_edges = build_hybrid_identity_edges(
        units,
        pair_evidence,
        transitions,
        config=solver_config,
    )

    solution = solve_identity_graphs(
        units,
        pair_evidence,
        transitions,
        config=solver_config,
    )

    candidates = generate_candidate_identity_graphs(
        units,
        pair_evidence,
        transitions,
        config=solver_config,
    )

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    print()
    print("V2.8A-V5 HYBRID IDENTITY GRAPH AUDIT")
    print(f"raw tracks             : {len(segmentations)}")
    print(f"identity units         : {len(units)}")
    print(f"pair relations         : {len(pair_evidence)}")
    print(f"temporal transitions   : {len(transitions)}")
    print(f"hybrid graph edges     : {len(hybrid_edges)}")
    print(f"candidate graphs       : {len(candidates)}")
    print(f"required identities    : {args.identity_count}")
    print(f"clip span              : {clip_span_s:.2f}s")

    print()
    print("TOP HYBRID EDGES")
    print(
        f"{'PAIR':>17} {'ID':>6} {'TEMP':>6} {'TV':>9} "
        f"{'HYBRID':>7} REASON"
    )
    print("-" * 82)

    for edge in sorted(
        hybrid_edges.values(),
        key=lambda row: (
            -row.hybrid_score,
            row.unit_a,
            row.unit_b,
        ),
    )[:30]:
        print(
            f"{edge.unit_a+'↔'+edge.unit_b:>17} "
            f"{edge.identity_support:>6.3f} "
            f"{edge.temporal_score:>6.3f} "
            f"{edge.temporal_verdict:>9} "
            f"{edge.hybrid_score:>7.3f} "
            f"{edge.reason}"
        )

    print()
    print("CANONICAL IDENTITY GRAPHS")
    print(
        f"{'#':>3} {'ROOT':>7} {'Q':>5} {'TEAM':>8} "
        f"{'MEMBERS':>38} {'MST-MIN':>7} {'MST-MEAN':>8}"
    )
    print("-" * 100)

    for identity in solution.identities:
        print(
            f"{identity.canonical_id:>3} "
            f"{identity.root_key:>7} "
            f"{identity.root_quality:>5.2f} "
            f"{identity.team_label.value:>8} "
            f"{','.join(identity.member_keys):>38} "
            f"{identity.mst_min_score:>7.3f} "
            f"{identity.mst_mean_score:>8.3f}"
        )

    print()
    print("JUNK")
    if solution.junk_keys:
        for key in solution.junk_keys:
            unit = unit_by_key[key]
            print(
                f"{key:>7} "
                f"Q={unit.quality:.3f} "
                f"obs={unit.observed_count} "
                f"span={unit.observed_span_s:.2f}s "
                f"team={unit.team_label.value}"
            )
    else:
        print("(none)")

    identity_of = {}
    for identity in solution.identities:
        for key in identity.member_keys:
            identity_of[key] = identity.canonical_id

    known_pairs = (
        ("1:0", "29:0"),
        ("9:0", "38:0"),
        ("9:1", "38:0"),
        ("10:1", "41:0"),
        ("3:0", "43:0"),
        ("3:0", "4:0"),
        ("11:0", "25:0"),
        ("27:0", "28:0"),
        ("13:0", "13:1"),
        ("13:0", "27:0"),
        ("14:0", "14:1"),
        ("14:0", "31:0"),
    )

    print()
    print("KNOWN / DIAGNOSTIC CHECK")
    print(
        f"{'PAIR':>13} {'ID':>6} {'ICONF':>7} "
        f"{'TV':>9} {'HEDGE':>6} {'SAME':>6}"
    )
    print("-" * 72)

    for left, right in known_pairs:
        relation = pair_evidence.get(
            tuple(sorted((left, right)))
        )
        if relation is None:
            continue

        transition = (
            transitions.get((left, right))
            or transitions.get((right, left))
        )
        verdict = (
            getattr(
                transition,
                "verdict",
                "none",
            )
            if transition is not None
            else "none"
        )

        same = (
            left in identity_of
            and right in identity_of
            and identity_of[left] == identity_of[right]
        )

        print(
            f"{left+'↔'+right:>13} "
            f"{relation.fused_support_score:>6.3f} "
            f"{relation.conflict.value:>7} "
            f"{verdict:>9} "
            f"{str(tuple(sorted((left, right))) in hybrid_edges):>6} "
            f"{str(same):>6}"
        )

    team_counts = {}
    for identity in solution.identities:
        team_counts[
            identity.team_label.value
        ] = (
            team_counts.get(
                identity.team_label.value,
                0,
            )
            + 1
        )

    print()
    print(
        "IDENTITY TEAM COUNTS: "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(team_counts.items())
        )
    )

    high_real_junk = [
        key
        for key in solution.junk_keys
        if unit_by_key[key].quality >= 0.60
    ]

    print(
        "HIGH-REALNESS JUNK (Q>=.60): "
        + (
            ", ".join(high_real_junk)
            if high_real_junk
            else "(none)"
        )
    )

    payload = {
        "version": "tracking-identity-graph-v28a-v5",
        "solver_config": {
            field: getattr(solver_config, field)
            for field in solver_config.__dataclass_fields__
        },
        "clip_span_s": clip_span_s,
        "max_observed_count": max_observed_count,
        "units": {
            unit.key: unit.to_dict()
            for unit in units
        },
        "pair_evidence": {
            f"{key[0]}|{key[1]}": value.to_dict()
            for key, value in sorted(pair_evidence.items())
        },
        "temporal_transitions": {
            f"{key[0]}->{key[1]}": value.to_dict()
            for key, value in sorted(transitions.items())
        },
        "hybrid_edges": {
            f"{key[0]}|{key[1]}": value.to_dict()
            for key, value in sorted(hybrid_edges.items())
        },
        "solution": solution.to_dict(),
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"objective utility : {solution.objective_utility:.4f}")
    print(f"report            : {args.output.resolve()}")
    print(
        "Audit only: no tracking frame rewrite / tracks.person publication."
    )


if __name__ == "__main__":
    main()
