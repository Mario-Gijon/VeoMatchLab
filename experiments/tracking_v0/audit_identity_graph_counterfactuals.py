#!/usr/bin/env python3
"""Counterfactual audit for V2.8A-V5.

This does NOT change the solver. It asks which current identity must give way if
a suspicious high-realness JUNK unit is forced to be represented.

Default hypotheses:
  * 11:0 must not be junk
  * 13:1 must not be junk
  * 14:1 must not be junk
  * 11:0 + 13:1 both must not be junk
  * 11:0 + 14:1 both must not be junk
  * force same: 13:0 <-> 13:1
  * force same: 14:0 <-> 14:1

The objective delta tells us whether the baseline is strongly preferred or
whether several explanations are nearly equivalent.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tracking_v0.audit_identity_paths import (  # noqa: E402
    build_pair_evidence,
    build_temporal_transitions,
    build_units,
    load_appearance_labels,
    load_embeddings,
    load_observed_points,
    load_semantics,
)
from tracking_v0.appearance_bank import (  # noqa: E402
    AppearanceBankConfig,
    build_appearance_banks,
)
from tracking_v0.appearance_segments import (  # noqa: E402
    AppearanceSegmentationConfig,
    segment_all_banks,
)
from tracking_v0.identity_graph_counterfactual import (  # noqa: E402
    CounterfactualConstraints,
    solve_counterfactual_partition,
)
from tracking_v0.identity_graph_solver import (  # noqa: E402
    IdentityGraphSolverConfig,
    generate_candidate_identity_graphs,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=Path(
            "experiments/tracking_v0/reid_similarity_debug/"
            "reid_embeddings.npz"
        ),
    )
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=Path(
            "experiments/tracking_v0/reid_crop_debug/"
            "_tracking_tmp/tracking_frames.jsonl"
        ),
    )
    parser.add_argument(
        "--semantic-report",
        type=Path,
        default=Path(
            "experiments/tracking_v0/team_appearance_debug/"
            "semantic_constraints_report.json"
        ),
    )
    parser.add_argument(
        "--appearance-report",
        type=Path,
        default=Path(
            "experiments/tracking_v0/team_appearance_debug/"
            "team_appearance_report.json"
        ),
    )
    parser.add_argument(
        "--identity-count",
        type=int,
        default=23,
    )
    return parser.parse_args()


def group_set(solution):
    return {
        tuple(group)
        for group in solution.selected_groups
    }


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
    team_labels, role_anchors = load_semantics(
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

    segmentations = segment_all_banks(
        banks,
        source_labels=appearance_labels,
        config=AppearanceSegmentationConfig(),
    )

    (
        units,
        unit_points,
        unit_segment,
        _clip_span_s,
        _max_observed_count,
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

    config = IdentityGraphSolverConfig(
        identity_count=args.identity_count,
    )

    candidates = generate_candidate_identity_graphs(
        units,
        pair_evidence,
        transitions,
        config=config,
    )

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    baseline = solve_counterfactual_partition(
        units,
        candidates,
        config=config,
    )

    if not baseline.feasible:
        raise RuntimeError(
            "baseline counterfactual partition is infeasible"
        )

    print()
    print("V2.8A-V5 COUNTERFACTUAL IDENTITY AUDIT")
    print(f"units            : {len(units)}")
    print(f"candidate graphs : {len(candidates)}")
    print(f"baseline utility : {baseline.objective_utility:.6f}")
    print(
        "baseline high-real junk: "
        + ", ".join(
            key
            for key in baseline.junk_keys
            if unit_by_key[key].quality >= 0.60
        )
    )

    scenarios = [
        (
            "FORCE 11:0 REAL",
            CounterfactualConstraints(
                required_nonjunk_keys=("11:0",),
            ),
        ),
        (
            "FORCE 13:1 REAL",
            CounterfactualConstraints(
                required_nonjunk_keys=("13:1",),
            ),
        ),
        (
            "FORCE 14:1 REAL",
            CounterfactualConstraints(
                required_nonjunk_keys=("14:1",),
            ),
        ),
        (
            "FORCE 11:0 + 13:1 REAL",
            CounterfactualConstraints(
                required_nonjunk_keys=(
                    "11:0",
                    "13:1",
                ),
            ),
        ),
        (
            "FORCE 11:0 + 14:1 REAL",
            CounterfactualConstraints(
                required_nonjunk_keys=(
                    "11:0",
                    "14:1",
                ),
            ),
        ),
        (
            "FORCE 13:0 SAME 13:1",
            CounterfactualConstraints(
                required_same_pairs=(
                    ("13:0", "13:1"),
                ),
            ),
        ),
        (
            "FORCE 14:0 SAME 14:1",
            CounterfactualConstraints(
                required_same_pairs=(
                    ("14:0", "14:1"),
                ),
            ),
        ),
    ]

    baseline_groups = group_set(
        baseline
    )

    for title, constraints in scenarios:
        solved = solve_counterfactual_partition(
            units,
            candidates,
            config=config,
            constraints=constraints,
        )

        print()
        print("=" * 100)
        print(title)

        if not solved.feasible:
            print(
                "INFEASIBLE: "
                + solved.solver_message
            )
            continue

        delta = (
            solved.objective_utility
            - baseline.objective_utility
        )

        print(
            f"utility       : {solved.objective_utility:.6f}"
        )
        print(
            f"delta vs base : {delta:+.6f}"
        )

        new_groups = (
            group_set(solved)
            - baseline_groups
        )
        removed_groups = (
            baseline_groups
            - group_set(solved)
        )

        print("NEW / CHANGED GROUPS")
        if new_groups:
            for group in sorted(
                new_groups,
                key=lambda row: (
                    row[0],
                    len(row),
                    row,
                ),
            ):
                qualities = ",".join(
                    f"{key}:{unit_by_key[key].quality:.2f}"
                    for key in group
                )
                print(
                    "  "
                    + ",".join(group)
                    + "   Q["
                    + qualities
                    + "]"
                )
        else:
            print("  (none)")

        print("REMOVED BASELINE GROUPS")
        if removed_groups:
            for group in sorted(
                removed_groups,
                key=lambda row: (
                    row[0],
                    len(row),
                    row,
                ),
            ):
                print(
                    "  "
                    + ",".join(group)
                )
        else:
            print("  (none)")

        new_junk = (
            set(solved.junk_keys)
            - set(baseline.junk_keys)
        )
        rescued = (
            set(baseline.junk_keys)
            - set(solved.junk_keys)
        )

        print(
            "RESCUED FROM JUNK : "
            + (
                ", ".join(sorted(rescued))
                if rescued
                else "(none)"
            )
        )
        print(
            "NEWLY JUNK        : "
            + (
                ", ".join(sorted(new_junk))
                if new_junk
                else "(none)"
            )
        )

    print()
    print("=" * 100)
    print("TOP CANDIDATE GRAPHS FOR FOCUS UNITS")

    for key in (
        "11:0",
        "13:1",
        "14:1",
        "16:0",
        "31:0",
    ):
        print()
        print(
            f"{key}  Q={unit_by_key[key].quality:.3f}"
        )

        rows = [
            candidate
            for candidate in candidates
            if key in candidate.member_keys
        ]

        for candidate in sorted(
            rows,
            key=lambda row: (
                -row.utility,
                -len(row.member_keys),
                row.member_keys,
            ),
        )[:10]:
            print(
                f"  util={candidate.utility:.4f} "
                f"mst={candidate.mst_min_score:.3f}/"
                f"{candidate.mst_mean_score:.3f} "
                f"members={','.join(candidate.member_keys)}"
            )


if __name__ == "__main__":
    main()
