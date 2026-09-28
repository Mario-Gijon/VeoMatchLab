#!/usr/bin/env python3
"""Audit V6 full-match football roster identity solver.

V6 uses the complete clip before finalizing identity. There are no mandatory
bootstrap/root raw tracks: the first seconds are evidence, not authority.

The solver chooses exactly one path per canonical participant over all raw
tracklets, with the configured football composition participating directly in
the optimization (11 TEAM_A + 11 TEAM_B + 1 REFEREE for the current clip).

The audit reports:
- full-context solver status/objective,
- selected raw coverage and unresolved inside tracks,
- exact roster composition,
- selected path transitions and observation gaps,
- candidate transition counts / hard pair pruning,
- ReID comparisons used only on physically/team-plausible transitions.

Several seconds without observation are allowed. No positions are fabricated
inside those gaps, and nothing is written to production tracks or the preview.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.appearance_bank import (  # noqa: E402
    AppearanceBankConfig,
    ReIDEmbedding,
    build_appearance_banks,
    compare_track_banks,
)
from tracking.roster_recovery import (  # noqa: E402
    RawTracklet,
    RosterObservation,
    RosterRecoveryConfig,
    recover_roster_identities,
)
from tracking.team_similarity import RelativeTeamLabel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--semantic-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--role-report",
        type=Path,
        required=True,
        help=(
            "B.2 role_inference_report.json; only its per-track referee "
            "evidence is used for the global 11+11+1 composition."
        ),
    )
    parser.add_argument(
        "--roster-size",
        type=int,
        default=23,
    )
    parser.add_argument(
        "--team-a-count",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--team-b-count",
        type=int,
        default=11,
    )
    parser.add_argument(
        "--referee-count",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--field-length",
        type=float,
        default=105.0,
    )
    parser.add_argument(
        "--field-width",
        type=float,
        default=68.0,
    )
    parser.add_argument(
        "--continuation-margin",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--seed-search-seconds",
        type=float,
        default=8.0,
        help=(
            "Retained for config/backward compatibility. V6 does not lock "
            "roster roots from this window; final identity uses the full clip."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/tracking/"
            "reid_similarity_debug_margin15/"
            "roster_recovery_v6_report.json"
        ),
    )
    return parser.parse_args()


def load_semantic_evidence(
    semantic_path: Path,
    role_path: Path,
) -> dict[int, dict[str, object]]:
    semantic = json.loads(
        semantic_path.read_text(
            encoding="utf-8",
        )
    )
    roles = json.loads(
        role_path.read_text(
            encoding="utf-8",
        )
    )

    result: dict[
        int,
        dict[str, object],
    ] = {}

    for raw_id, row in semantic.get(
        "team_similarity",
        {},
    ).items():
        track_id = int(
            raw_id
        )
        value = str(
            row.get(
                "label",
                "unknown",
            )
        )

        try:
            label = RelativeTeamLabel(
                value
            )
        except ValueError:
            label = RelativeTeamLabel.UNKNOWN

        result[
            track_id
        ] = {
            "team_label": label,
            "team_a_similarity": (
                float(
                    row[
                        "team_a_similarity"
                    ]
                )
                if row.get(
                    "team_a_similarity"
                )
                is not None
                else None
            ),
            "team_b_similarity": (
                float(
                    row[
                        "team_b_similarity"
                    ]
                )
                if row.get(
                    "team_b_similarity"
                )
                is not None
                else None
            ),
            "referee_score": None,
        }

    for raw_id, row in roles.get(
        "decisions",
        {},
    ).items():
        track_id = int(
            raw_id
        )
        evidence = result.setdefault(
            track_id,
            {
                "team_label": (
                    RelativeTeamLabel.UNKNOWN
                ),
                "team_a_similarity": None,
                "team_b_similarity": None,
                "referee_score": None,
            },
        )

        scores = row.get(
            "scores",
            {},
        )
        if scores.get(
            "referee"
        ) is not None:
            evidence[
                "referee_score"
            ] = float(
                scores[
                    "referee"
                ]
            )

    return result



def load_embeddings(
    path: Path,
) -> list[ReIDEmbedding]:
    payload = np.load(
        path,
        allow_pickle=False,
    )

    vectors = payload["vectors"]
    track_ids = payload["track_ids"]
    frame_indices = payload["frame_indices"]
    qualities = payload["qualities"]
    source_paths = payload["source_paths"]

    return [
        ReIDEmbedding(
            track_id=int(
                track_ids[index]
            ),
            frame_index=int(
                frame_indices[index]
            ),
            quality=float(
                qualities[index]
            ),
            vector=vectors[index],
            source_path=str(
                source_paths[index]
            ),
        )
        for index in range(
            len(vectors)
        )
    ]


def load_tracklets(
    path: Path,
    *,
    semantic_evidence: dict[
        int,
        dict[str, object],
    ],
    field_length: float,
    field_width: float,
    continuation_margin: float,
) -> tuple[
    RawTracklet,
    ...,
]:
    grouped: dict[
        int,
        list[RosterObservation],
    ] = defaultdict(list)

    half_length = field_length / 2.0
    half_width = field_width / 2.0

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
                    row.get(
                        "timestamp",
                    ),
                )
            )

            for track in row.get(
                "tracks",
                [],
            ):
                if (
                    str(
                        track.get(
                            "position_kind",
                            "",
                        )
                    )
                    != "observed"
                ):
                    continue

                raw_id = int(
                    track[
                        "track_id"
                    ]
                )

                grouped[
                    raw_id
                ].append(
                    RosterObservation(
                        raw_track_id=raw_id,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        x=float(
                            track["x"]
                        ),
                        z=float(
                            track["z"]
                        ),
                        measurement_id=(
                            str(
                                track[
                                    "measurement_id"
                                ]
                            )
                            if track.get(
                                "measurement_id"
                            )
                            is not None
                            else None
                        ),
                        confidence=(
                            float(
                                track[
                                    "confidence"
                                ]
                            )
                            if track.get(
                                "confidence"
                            )
                            is not None
                            else None
                        ),
                    )
                )

    result = []

    for raw_id, observations in sorted(
        grouped.items()
    ):
        observations.sort(
            key=lambda item: (
                item.timestamp,
                item.frame_index,
            )
        )

        strict = sum(
            -half_length
            <= row.x
            <= half_length
            and -half_width
            <= row.z
            <= half_width
            for row in observations
        )

        allowed = sum(
            -half_length
            - continuation_margin
            <= row.x
            <= half_length
            + continuation_margin
            and -half_width
            - continuation_margin
            <= row.z
            <= half_width
            + continuation_margin
            for row in observations
        )

        result.append(
            RawTracklet(
                raw_track_id=raw_id,
                observations=tuple(
                    observations
                ),
                team_label=(
                    semantic_evidence.get(
                        raw_id,
                        {},
                    ).get(
                        "team_label",
                        RelativeTeamLabel.UNKNOWN,
                    )
                ),
                team_a_similarity=(
                    semantic_evidence.get(
                        raw_id,
                        {},
                    ).get(
                        "team_a_similarity"
                    )
                ),
                team_b_similarity=(
                    semantic_evidence.get(
                        raw_id,
                        {},
                    ).get(
                        "team_b_similarity"
                    )
                ),
                referee_score=(
                    semantic_evidence.get(
                        raw_id,
                        {},
                    ).get(
                        "referee_score"
                    )
                ),
                inside_observed_count=int(
                    strict
                ),
                continuation_observed_count=int(
                    allowed
                ),
            )
        )

    return tuple(
        result
    )


def make_appearance_similarity(
    embeddings: list[
        ReIDEmbedding
    ],
):
    config = AppearanceBankConfig(
        min_quality=0.10,
        max_embeddings_per_track=10,
        strong_similarity_threshold=0.75,
        unique_top_k=3,
    )

    banks = build_appearance_banks(
        embeddings,
        config=config,
    )
    cache = {}

    def similarity(
        left: int,
        right: int,
    ) -> float | None:
        key = tuple(
            sorted(
                (
                    left,
                    right,
                )
            )
        )

        if key in cache:
            return cache[
                key
            ]

        if (
            left not in banks
            or right not in banks
        ):
            cache[
                key
            ] = None
            return None

        metrics = compare_track_banks(
            banks[left],
            banks[right],
            config=config,
        )

        score = (
            0.50
            * metrics.bidirectional_best_median
            + 0.35
            * metrics.unique_topk_similarity
            + 0.15
            * metrics.all_pairs_q90
        )

        score = float(
            min(
                1.0,
                max(
                    0.0,
                    score,
                ),
            )
        )

        cache[
            key
        ] = score
        return score

    return similarity, banks


def main() -> None:
    args = parse_args()

    semantic_evidence = load_semantic_evidence(
        args.semantic_report,
        args.role_report,
    )
    embeddings = load_embeddings(
        args.embedding_cache
    )

    tracklets = load_tracklets(
        args.tracking_frames,
        semantic_evidence=semantic_evidence,
        field_length=args.field_length,
        field_width=args.field_width,
        continuation_margin=(
            args.continuation_margin
        ),
    )

    appearance_similarity, banks = (
        make_appearance_similarity(
            embeddings
        )
    )

    config = RosterRecoveryConfig(
        roster_size=args.roster_size,
        field_length_m=args.field_length,
        field_width_m=args.field_width,
        continuation_margin_m=(
            args.continuation_margin
        ),
        seed_search_seconds=(
            args.seed_search_seconds
        ),
        team_a_count=args.team_a_count,
        team_b_count=args.team_b_count,
        referee_count=args.referee_count,
    )

    result = recover_roster_identities(
        tracklets,
        appearance_similarity=(
            appearance_similarity
        ),
        config=config,
    )

    tracklet_by_id = {
        row.raw_track_id: row
        for row in tracklets
    }

    canonical_teams = Counter(
        identity.team_label.value
        for identity in result.identities
    )
    roster_classes = Counter(
        identity.roster_class.value
        for identity in result.identities
    )

    total_observed = sum(
        row.observed_count
        for row in tracklets
    )
    assigned_observed = sum(
        tracklet_by_id[
            raw_id
        ].observed_count
        for raw_id in result.raw_to_canonical
    )

    strict_inside_total = sum(
        row.inside_observed_count
        for row in tracklets
    )
    strict_inside_assigned = sum(
        tracklet_by_id[
            raw_id
        ].inside_observed_count
        for raw_id in result.raw_to_canonical
    )

    accepted_reason_counts = Counter(
        row.reason
        for row in result.accepted_recoveries
    )

    print()
    print(
        "V6 FULL-MATCH ROSTER SOLVER AUDIT"
    )
    print(
        f"raw tracklets              : {len(tracklets)}"
    )
    print(
        f"tracklets with ReID bank   : {len(banks)}"
    )
    print(
        f"first raw frame            : {result.seed_frame_index}"
    )
    print(
        f"first raw time             : {result.seed_timestamp:.3f}s"
    )
    print(
        f"all 23 paths first seen by : {result.roster_completed_timestamp:.3f}s"
    )
    print(
        "full context end time     : "
        + (
            f"{result.global_context_end_timestamp:.3f}s"
            if result.global_context_end_timestamp is not None
            else "n/a"
        )
    )
    print(
        f"canonical roster           : {len(result.identities)}"
    )
    print(
        "raw-label aggregate       : "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                canonical_teams.items()
            )
        )
    )
    print(
        "fixed roster composition  : "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                roster_classes.items()
            )
        )
    )
    print(
        "mandatory bootstrap roots : none (V6 full-match)"
    )
    print(
        f"selected raw tracklets     : {result.global_selected_track_count}"
    )
    print(
        f"unassigned raw fragments   : {len(result.unassigned_fragment_ids)}"
    )
    print(
        f"appearance comparisons     : {result.appearance_query_count}"
    )
    print(
        "composition score         : "
        + (
            f"{result.composition_score:.3f}"
            if result.composition_score is not None
            else "n/a"
        )
    )
    print(
        f"global solver status       : {result.global_solver_status}"
    )
    print(
        "global objective           : "
        + (
            f"{result.global_objective:.3f}"
            if result.global_objective is not None
            else "n/a"
        )
    )
    print(
        f"global candidate edges     : {result.global_candidate_edge_count}"
    )
    print(
        f"global path transitions    : {result.global_selected_transition_count}"
    )
    print(
        f"hard-team pair prunes      : {result.global_hard_team_pair_prunes}"
    )
    print(
        f"temporal/overlap prunes    : {result.global_temporal_pair_prunes}"
    )
    print(
        "accepted by mode          : "
        + (
            ", ".join(
                f"{key}={value}"
                for key, value in sorted(
                    accepted_reason_counts.items()
                )
            )
            if accepted_reason_counts
            else "(none)"
        )
    )
    print(
        f"observed coverage          : "
        f"{assigned_observed}/{total_observed} "
        f"({assigned_observed / max(total_observed, 1):.1%})"
    )
    print(
        f"strict-inside coverage     : "
        f"{strict_inside_assigned}/{strict_inside_total} "
        f"({strict_inside_assigned / max(strict_inside_total, 1):.1%})"
    )

    print()
    print("CANONICAL ROSTER")
    print(
        f"{'#':>3} {'CLASS':>8} {'RAWTEAM':>8} "
        f"{'FIRST':>5} MEMBERS"
    )
    print("-" * 72)

    for identity in result.identities:
        members = ",".join(
            str(value)
            for value in identity.member_raw_track_ids
        )
        print(
            f"{identity.canonical_id:>3} "
            f"{identity.roster_class.value:>8} "
            f"{identity.team_label.value:>8} "
            f"{identity.seed_raw_track_id:>5} "
            f"{members}"
        )

    print()
    print("SELECTED GLOBAL PATH TRANSITIONS")
    if result.accepted_recoveries:
        print(
            f"{'TO':>4} {'PATH':>5} {'FROM':>5} "
            f"{'SCORE':>6} {'CONT':>6} {'REID':>6} "
            f"{'TEAM':>6} {'ROST':>6} {'GAP':>6} {'MODE':>25}"
        )
        print("-" * 100)

        for row in result.accepted_recoveries:
            print(
                f"{row.fragment_raw_track_id:>4} "
                f"{row.canonical_id:>5} "
                f"{row.predecessor_raw_track_id:>5} "
                f"{row.total_score:>6.3f} "
                f"{row.continuity_score:>6.3f} "
                f"{row.appearance_score:>6.3f} "
                f"{row.team_score:>6.2f} "
                f"{row.roster_prior_score:>6.2f} "
                f"{row.gap_s:>6.2f} "
                f"{row.reason:>25}"
            )
    else:
        print("(none)")

    print()
    print("UNASSIGNED FRAGMENTS")
    if result.unassigned_fragment_ids:
        print(
            f"{'RAW':>4} {'TEAM':>8} {'OBS':>5} "
            f"{'IN':>5} {'ALLOW':>5} {'SPAN':>6}"
        )
        print("-" * 48)

        for raw_id in sorted(
            result.unassigned_fragment_ids,
            key=lambda value: (
                -tracklet_by_id[
                    value
                ].observed_count,
                value,
            ),
        ):
            row = tracklet_by_id[
                raw_id
            ]
            print(
                f"{raw_id:>4} "
                f"{row.team_label.value:>8} "
                f"{row.observed_count:>5} "
                f"{row.inside_observed_count:>5} "
                f"{row.continuation_observed_count:>5} "
                f"{row.observed_span_s:>6.2f}"
            )
    else:
        print("(none)")

    payload = {
        "version": "tracking-roster-recovery-v6",
        "config": {
            field: getattr(
                config,
                field,
            )
            for field in config.__dataclass_fields__
        },
        "summary": {
            "raw_tracklets": len(
                tracklets
            ),
            "reid_banks": len(
                banks
            ),
            "observed_total": (
                total_observed
            ),
            "observed_assigned": (
                assigned_observed
            ),
            "strict_inside_total": (
                strict_inside_total
            ),
            "strict_inside_assigned": (
                strict_inside_assigned
            ),
            "canonical_team_counts": dict(
                canonical_teams
            ),
            "roster_class_counts": dict(
                roster_classes
            ),
            "accepted_reason_counts": dict(
                accepted_reason_counts
            ),
            "global_solver_status": (
                result.global_solver_status
            ),
            "global_objective": (
                result.global_objective
            ),
            "global_candidate_edge_count": (
                result.global_candidate_edge_count
            ),
            "global_selected_transition_count": (
                result.global_selected_transition_count
            ),
            "global_hard_team_pair_prunes": (
                result.global_hard_team_pair_prunes
            ),
            "global_temporal_pair_prunes": (
                result.global_temporal_pair_prunes
            ),
            "global_context_end_timestamp": (
                result.global_context_end_timestamp
            ),
            "global_selected_track_count": (
                result.global_selected_track_count
            ),
        },
        "tracklets": {
            str(row.raw_track_id): {
                "team_label": (
                    row.team_label.value
                ),
                "team_a_similarity": (
                    row.team_a_similarity
                ),
                "team_b_similarity": (
                    row.team_b_similarity
                ),
                "referee_score": (
                    row.referee_score
                ),
                "start_frame": (
                    row.start_frame
                ),
                "end_frame": (
                    row.end_frame
                ),
                "start_timestamp": (
                    row.start_timestamp
                ),
                "end_timestamp": (
                    row.end_timestamp
                ),
                "observed_count": (
                    row.observed_count
                ),
                "inside_observed_count": (
                    row.inside_observed_count
                ),
                "continuation_observed_count": (
                    row.continuation_observed_count
                ),
            }
            for row in tracklets
        },
        "result": result.to_dict(),
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
        f"report : {args.output.resolve()}"
    )
    print(
        "Audit only: raw detections are preserved; "
        "no production tracking or preview was rewritten."
    )


if __name__ == "__main__":
    main()
