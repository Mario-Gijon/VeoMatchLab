#!/usr/bin/env python3
"""V2.8A-V4 audit: identity evidence + temporal before/after paths.

No detector, tracker or OSNet rerun.

Pipeline:
    cached ReID + team semantics + C.2 visual segments
        -> identity units
        -> pair identity evidence
        -> explicit temporal transition evidence
        -> directed identity-path candidates
        -> exactly 23 canonical paths + JUNK

This is the first audit where "same person" must make sense BOTH visually/
semantically and as a temporal before/after trajectory.

TEMP_OUTSIDE support is implemented in tracking_v0.temporal_identity but is NOT
active in this audit because the current cached observation file intentionally
contains only accepted field observations.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking_v0.appearance_bank import (  # noqa: E402
    AppearanceBankConfig,
    ReIDEmbedding,
    build_appearance_banks,
    compare_track_banks,
)
from tracking_v0.appearance_segments import (  # noqa: E402
    AppearanceSegmentationConfig,
    segment_all_banks,
)
from tracking_v0.global_identity_solver import (  # noqa: E402
    IdentityUnit,
    PairConflict,
    UnitPairEvidence,
    compute_unit_quality,
)
from tracking_v0.identity_evidence import (  # noqa: E402
    IdentityEvidenceConfig,
    ObservedTrackPoint,
    compute_geometry_pair_evidence_from_points,
)
from tracking_v0.identity_path_solver import (  # noqa: E402
    IdentityPathSolverConfig,
    build_temporal_identity_edges,
    solve_identity_paths,
)
from tracking_v0.team_appearance import AppearanceLabel  # noqa: E402
from tracking_v0.team_similarity import RelativeTeamLabel  # noqa: E402
from tracking_v0.temporal_identity import (  # noqa: E402
    TemporalTransitionConfig,
    compute_temporal_transition,
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
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/tracking_v0/reid_similarity_debug/"
            "identity_path_solution_v28a_v4_1.json"
        ),
    )
    return parser.parse_args()


def load_embeddings(path):
    payload = np.load(
        path,
        allow_pickle=False,
    )

    return [
        ReIDEmbedding(
            track_id=int(track_id),
            frame_index=int(frame_index),
            quality=float(quality),
            vector=vector,
            source_path=str(source_path),
        )
        for (
            vector,
            track_id,
            frame_index,
            quality,
            source_path,
        ) in zip(
            payload["vectors"],
            payload["track_ids"],
            payload["frame_indices"],
            payload["qualities"],
            payload["source_paths"],
        )
    ]


def load_observed_points(path):
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
                if (
                    str(
                        raw.get(
                            "position_kind",
                            "",
                        )
                    )
                    != "observed"
                ):
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
                        x=float(
                            raw["x"]
                        ),
                        z=float(
                            raw["z"]
                        ),
                    )
                )

    return {
        track_id: tuple(
            sorted(
                rows,
                key=lambda point: (
                    point.timestamp,
                    point.frame_index,
                ),
            )
        )
        for track_id, rows in sorted(
            grouped.items()
        )
    }


def load_appearance_labels(path):
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    result = {}

    for raw_id, row in payload.get(
        "track_decisions",
        {},
    ).items():
        try:
            label = AppearanceLabel(
                str(
                    row.get(
                        "label",
                        "unknown",
                    )
                )
            )
        except ValueError:
            label = AppearanceLabel.UNKNOWN

        result[
            int(raw_id)
        ] = label

    return result


def load_semantics(path):
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    teams = {}

    for raw_id, row in payload.get(
        "team_similarity",
        {},
    ).items():
        try:
            label = RelativeTeamLabel(
                str(
                    row.get(
                        "label",
                        "unknown",
                    )
                )
            )
        except ValueError:
            label = RelativeTeamLabel.UNKNOWN

        teams[
            int(raw_id)
        ] = label

    role_anchors = set()

    for row in payload.get(
        "global_roles",
        {},
    ).values():
        if bool(
            row.get(
                "selected",
                False,
            )
        ):
            track_id = row.get(
                "track_id"
            )
            if track_id is not None:
                role_anchors.add(
                    int(
                        track_id
                    )
                )

    return (
        teams,
        role_anchors,
    )


def build_units(
    *,
    segmentations,
    observed_points,
    team_labels,
    role_anchors,
):
    all_points = [
        point
        for rows in observed_points.values()
        for point in rows
    ]

    if not all_points:
        raise RuntimeError(
            "no observed points"
        )

    clip_start = min(
        point.timestamp
        for point in all_points
    )
    clip_end = max(
        point.timestamp
        for point in all_points
    )
    clip_span_s = max(
        1e-6,
        clip_end
        - clip_start,
    )

    max_observed_count = max(
        len(rows)
        for rows in observed_points.values()
    )

    units = []
    unit_points = {}
    unit_segment = {}

    for track_id, segmentation in sorted(
        segmentations.items()
    ):
        raw_points = observed_points.get(
            track_id,
            (),
        )
        if not raw_points:
            continue

        segments = segmentation.segments

        boundaries = [
            (
                left.end_frame
                + right.start_frame
            )
            // 2
            for left, right in zip(
                segments,
                segments[1:],
            )
        ]

        raw_start = raw_points[
            0
        ].frame_index
        raw_end = raw_points[
            -1
        ].frame_index

        for index, segment in enumerate(
            segments
        ):
            start_frame = (
                raw_start
                if index == 0
                else boundaries[
                    index - 1
                ]
                + 1
            )
            end_frame = (
                raw_end
                if index
                == len(segments) - 1
                else boundaries[
                    index
                ]
            )

            points = tuple(
                point
                for point in raw_points
                if (
                    start_frame
                    <= point.frame_index
                    <= end_frame
                )
            )

            if not points:
                continue

            span_s = max(
                0.0,
                points[-1].timestamp
                - points[0].timestamp,
            )

            team = team_labels.get(
                track_id,
                RelativeTeamLabel.UNKNOWN,
            )

            role_anchor = (
                track_id
                in role_anchors
            )

            quality = compute_unit_quality(
                observed_count=len(
                    points
                ),
                observed_span_s=span_s,
                appearance_coherence=(
                    segment.median_internal_similarity
                ),
                appearance_q10=(
                    segment.q10_internal_similarity
                ),
                team_label=team,
                clip_observed_count_reference=(
                    max_observed_count
                ),
                clip_span_s=clip_span_s,
                role_anchor=role_anchor,
            )

            key = segment.key

            units.append(
                IdentityUnit(
                    key=key,
                    track_id=track_id,
                    segment_index=(
                        segment.segment_index
                    ),
                    start_frame=(
                        start_frame
                    ),
                    end_frame=end_frame,
                    observed_count=len(
                        points
                    ),
                    observed_span_s=(
                        span_s
                    ),
                    appearance_label=(
                        segmentation.source_label
                    ),
                    team_label=team,
                    appearance_coherence=(
                        segment.median_internal_similarity
                    ),
                    appearance_q10=(
                        segment.q10_internal_similarity
                    ),
                    embedding_count=(
                        segment.embedding_count
                    ),
                    quality=quality,
                    role_anchor=role_anchor,
                )
            )

            unit_points[
                key
            ] = points
            unit_segment[
                key
            ] = segment

    return (
        units,
        unit_points,
        unit_segment,
        clip_span_s,
        max_observed_count,
    )


def build_pair_evidence(
    *,
    units,
    unit_points,
    unit_segment,
    bank_config,
):
    identity_cfg = (
        IdentityEvidenceConfig()
    )

    result = {}

    for left_index, left in enumerate(
        units
    ):
        for right in units[
            left_index + 1:
        ]:
            key = tuple(
                sorted(
                    (
                        left.key,
                        right.key,
                    )
                )
            )

            reid = compare_track_banks(
                unit_segment[
                    left.key
                ].as_bank(),
                unit_segment[
                    right.key
                ].as_bank(),
                config=bank_config,
            )

            reid_raw = (
                0.50
                * reid.bidirectional_best_median
                + 0.35
                * reid.unique_topk_similarity
                + 0.15
                * reid.all_pairs_q90
            )

            reid_raw = min(
                1.0,
                max(
                    0.0,
                    reid_raw,
                ),
            )

            visual_reliability = min(
                1.0,
                min(
                    left.embedding_count,
                    right.embedding_count,
                )
                / 3.0,
            )

            reid_score = (
                0.5
                + visual_reliability
                * (
                    reid_raw
                    - 0.5
                )
            )

            geometry = (
                compute_geometry_pair_evidence_from_points(
                    left.track_id,
                    right.track_id,
                    rows_a=unit_points[
                        left.key
                    ],
                    rows_b=unit_points[
                        right.key
                    ],
                    config=identity_cfg,
                )
            )

            known = {
                RelativeTeamLabel.TEAM_A,
                RelativeTeamLabel.TEAM_B,
            }

            if (
                left.team_label in known
                and right.team_label in known
            ):
                if (
                    left.team_label
                    is right.team_label
                ):
                    team_compatibility = (
                        "same_team"
                    )
                    team_score = 1.0
                else:
                    team_compatibility = (
                        "team_conflict"
                    )
                    team_score = 0.0
            else:
                team_compatibility = (
                    "unknown"
                )
                team_score = 0.5

            fused = (
                identity_cfg.reid_weight
                * reid_score
                + identity_cfg.geometry_weight
                * geometry.transition_geometry_score
                + identity_cfg.team_weight
                * team_score
            )

            same_raw_split = (
                left.track_id
                == right.track_id
                and left.segment_index
                != right.segment_index
            )

            if (
                team_compatibility
                == "team_conflict"
            ):
                conflict = (
                    PairConflict.HARD
                )
                reason = (
                    "reliable_team_conflict"
                )
            elif (
                geometry.simultaneous_conflict
                and not same_raw_split
            ):
                conflict = (
                    PairConflict.STRONG
                )
                reason = (
                    "sustained_simultaneous_observed_conflict"
                )
            else:
                conflict = (
                    PairConflict.NONE
                )
                reason = (
                    "visual_split_soft_disagreement"
                    if same_raw_split
                    else "compatible_pair"
                )

            soft_penalty = (
                0.06
                + 0.12
                * max(
                    0.0,
                    0.75
                    - reid_score,
                )
                if same_raw_split
                else 0.0
            )

            result[
                key
            ] = UnitPairEvidence(
                unit_a=left.key,
                unit_b=right.key,
                reid_score=float(
                    reid_score
                ),
                geometry_score=float(
                    geometry.transition_geometry_score
                ),
                team_score=float(
                    team_score
                ),
                fused_support_score=float(
                    fused
                ),
                team_compatibility=(
                    team_compatibility
                ),
                simultaneous_count=(
                    geometry.simultaneous_count
                ),
                simultaneous_fraction=(
                    geometry.simultaneous_fraction
                ),
                simultaneous_median_distance_m=(
                    geometry.simultaneous_median_distance_m
                ),
                conflict=conflict,
                reason=reason,
                soft_same_identity_penalty=float(
                    soft_penalty
                ),
            )

    return result


def build_temporal_transitions(
    *,
    units,
    unit_points,
):
    cfg = (
        TemporalTransitionConfig()
    )

    result = {}

    for left_index, left in enumerate(
        units
    ):
        for right in units[
            left_index + 1:
        ]:
            evidence = (
                compute_temporal_transition(
                    left.key,
                    right.key,
                    rows_a=unit_points[
                        left.key
                    ],
                    rows_b=unit_points[
                        right.key
                    ],
                    config=cfg,
                )
            )

            result[
                (
                    evidence.source_key,
                    evidence.target_key,
                )
            ] = evidence

    return result


def main():
    args = parse_args()

    embeddings = load_embeddings(
        args.embedding_cache
    )
    observed_points = load_observed_points(
        args.tracking_frames
    )
    appearance_labels = (
        load_appearance_labels(
            args.appearance_report
        )
    )
    (
        team_labels,
        role_anchors,
    ) = load_semantics(
        args.semantic_report
    )

    bank_config = (
        AppearanceBankConfig(
            min_quality=0.10,
            max_embeddings_per_track=10,
            strong_similarity_threshold=0.75,
            unique_top_k=3,
        )
    )

    banks = build_appearance_banks(
        embeddings,
        config=bank_config,
    )

    segment_config = (
        AppearanceSegmentationConfig()
    )

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

    transitions = (
        build_temporal_transitions(
            units=units,
            unit_points=unit_points,
        )
    )

    solver_config = (
        IdentityPathSolverConfig(
            identity_count=(
                args.identity_count
            ),
        )
    )

    accepted_edges = (
        build_temporal_identity_edges(
            units,
            pair_evidence,
            transitions,
            config=solver_config,
        )
    )

    print()
    print(
        "V2.8A-V4.1 HYBRID IDENTITY + TEMPORAL PATH AUDIT"
    )
    print(
        f"raw tracks             : {len(segmentations)}"
    )
    print(
        f"identity units         : {len(units)}"
    )
    print(
        f"pair relations         : {len(pair_evidence)}"
    )
    print(
        f"temporal transitions   : {len(transitions)}"
    )
    print(
        f"accepted path edges    : {len(accepted_edges)}"
    )
    print(
        f"required identities    : {args.identity_count}"
    )
    print(
        f"clip span              : {clip_span_s:.2f}s"
    )
    print(
        f"max OBSERVED / raw ID  : {max_observed_count}"
    )

    solution = solve_identity_paths(
        units,
        pair_evidence,
        transitions,
        config=solver_config,
    )

    unit_by_key = {
        unit.key: unit
        for unit in units
    }

    print(
        f"candidate paths        : {solution.candidate_count}"
    )

    print()
    print("TOP TEMPORAL IDENTITY EDGES")
    print(
        f"{'EDGE':>17} {'TEMP':>6} {'ID':>6} "
        f"{'COMB':>6} {'VERDICT':>9} {'REGIME':>18} "
        f"{'GAP':>6} {'OVLP':>6}"
    )
    print("-" * 72)

    for edge in sorted(
        accepted_edges.values(),
        key=lambda row: (
            -row.combined_score,
            row.source_key,
            row.target_key,
        ),
    )[:25]:
        print(
            f"{edge.source_key+'→'+edge.target_key:>17} "
            f"{edge.temporal_score:>6.3f} "
            f"{edge.identity_support:>6.3f} "
            f"{edge.combined_score:>6.3f} "
            f"{edge.temporal_verdict:>9} "
            f"{edge.regime:>18} "
            f"{edge.gap_s:>6.2f} "
            f"{edge.overlap_s:>6.2f}"
        )

    print()
    print("CANONICAL TEMPORAL PATHS")
    print(
        f"{'#':>3} {'ROOT':>7} {'Q':>5} {'TEAM':>8} "
        f"{'PATH':>42} {'MIN-E':>6} {'MEAN-E':>7}"
    )
    print("-" * 95)

    for identity in solution.identities:
        path = " → ".join(
            identity.member_keys
        )
        print(
            f"{identity.canonical_id:>3} "
            f"{identity.root_key:>7} "
            f"{identity.root_quality:>5.2f} "
            f"{identity.team_label.value:>8} "
            f"{path:>42} "
            f"{identity.minimum_edge_score:>6.3f} "
            f"{identity.mean_edge_score:>7.3f}"
        )

    print()
    print("JUNK")
    if solution.junk_keys:
        for key in solution.junk_keys:
            unit = unit_by_key[
                key
            ]
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
            identity_of[
                key
            ] = identity.canonical_id

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
        ("14:0", "14:1"),
    )

    print()
    print(
        "KNOWN / DIAGNOSTIC PAIR + TEMPORAL CHECK"
    )
    print(
        f"{'PAIR':>13} {'ID-SUP':>6} {'CONFLICT':>9} "
        f"{'TEMP':>6} {'TV':>9} {'EDGE':>6} {'SAME':>6}"
    )
    print("-" * 72)

    for left, right in known_pairs:
        relation = pair_evidence.get(
            tuple(
                sorted(
                    (
                        left,
                        right,
                    )
                )
            )
        )

        transition = None

        for key in (
            (left, right),
            (right, left),
        ):
            if key in transitions:
                transition = transitions[
                    key
                ]
                break

        if relation is None:
            continue

        temporal_score = (
            transition.temporal_score
            if transition is not None
            else 0.0
        )

        edge_exists = any(
            key in accepted_edges
            for key in (
                (left, right),
                (right, left),
            )
        )

        same = (
            left in identity_of
            and right in identity_of
            and identity_of[left]
            == identity_of[right]
        )

        temporal_verdict = (
            getattr(
                transition,
                "verdict",
                "none",
            )
            if transition is not None
            else "none"
        )

        print(
            f"{left+'↔'+right:>13} "
            f"{relation.fused_support_score:>6.3f} "
            f"{relation.conflict.value:>9} "
            f"{temporal_score:>6.3f} "
            f"{temporal_verdict:>9} "
            f"{str(edge_exists):>6} "
            f"{str(same):>6}"
        )

    high_real_junk = [
        key
        for key in solution.junk_keys
        if unit_by_key[
            key
        ].quality >= 0.60
    ]

    print()
    print(
        "HIGH-REALNESS JUNK (Q>=.60): "
        + (
            ", ".join(
                high_real_junk
            )
            if high_real_junk
            else "(none)"
        )
    )

    if high_real_junk:
        print()
        print(
            "BEST TEMPORAL OPTIONS FOR HIGH-REALNESS JUNK"
        )
        print(
            f"{'UNIT':>7} {'EDGE':>17} {'TEMP':>6} "
            f"{'TV':>9} {'ID':>6} {'COMB':>6} {'ACCEPT':>7}"
        )
        print("-" * 72)

        for key in high_real_junk:
            candidates = []

            for transition in transitions.values():
                if (
                    transition.source_key != key
                    and transition.target_key != key
                ):
                    continue

                other = (
                    transition.target_key
                    if transition.source_key
                    == key
                    else transition.source_key
                )

                relation = pair_evidence.get(
                    tuple(
                        sorted(
                            (
                                key,
                                other,
                            )
                        )
                    )
                )
                if relation is None:
                    continue

                combined = (
                    solver_config.temporal_weight
                    * transition.temporal_score
                    + solver_config.identity_weight
                    * relation.fused_support_score
                )

                edge_key = (
                    transition.source_key,
                    transition.target_key,
                )

                candidates.append(
                    (
                        combined,
                        transition,
                        relation,
                        edge_key
                        in accepted_edges,
                    )
                )

            for (
                combined,
                transition,
                relation,
                accepted,
            ) in sorted(
                candidates,
                key=lambda row: -row[0],
            )[:5]:
                print(
                    f"{key:>7} "
                    f"{transition.source_key+'→'+transition.target_key:>17} "
                    f"{transition.temporal_score:>6.3f} "
                    f"{getattr(transition, 'verdict', 'none'):>9} "
                    f"{relation.fused_support_score:>6.3f} "
                    f"{combined:>6.3f} "
                    f"{str(accepted):>7}"
                )

    payload = {
        "version": "tracking-identity-path-v28a-v4.1",
        "solver_config": {
            field: getattr(
                solver_config,
                field,
            )
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
            for key, value in sorted(
                pair_evidence.items()
            )
        },
        "temporal_transitions": {
            f"{key[0]}->{key[1]}": value.to_dict()
            for key, value in sorted(
                transitions.items()
            )
        },
        "accepted_edges": {
            f"{key[0]}->{key[1]}": value.to_dict()
            for key, value in sorted(
                accepted_edges.items()
            )
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
    print(
        f"objective utility : {solution.objective_utility:.4f}"
    )
    print(
        f"report            : {args.output.resolve()}"
    )
    print(
        "Audit only: no tracking frame rewrite / tracks.person publication."
    )


if __name__ == "__main__":
    main()
