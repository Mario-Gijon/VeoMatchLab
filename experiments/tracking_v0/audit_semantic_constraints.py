#!/usr/bin/env python3
"""Audit B.2.1 global role selection + relative team similarity.

Uses existing Sprint A/B outputs only.  No detector rerun.

Inputs:
  --crop-dir              Sprint A reid_crop_debug
  --appearance-report     Sprint B.1 team_appearance_report.json
  --role-report           Sprint B.2 role_inference_report.json

Outputs:
  semantic_constraints_report.json

This audit is diagnostic only and does not touch tracks.person or V2.7.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking_v0.global_role_selection import (  # noqa: E402
    GlobalRoleSelectionConfig,
    select_global_unique_roles,
)
from tracking_v0.role_inference import (  # noqa: E402
    FootballRole,
    RoleInferenceResult,
    RoleScores,
    TrackAppearanceEvidence,
    TrackRoleDecision,
    TrackRoleGeometry,
)
from tracking_v0.team_appearance import (  # noqa: E402
    AppearanceCluster,
    AppearanceLabel,
    CropAppearance,
    KitAppearanceResult,
    TrackAppearanceDecision,
    extract_crop_appearance,
)
from tracking_v0.team_similarity import (  # noqa: E402
    TeamSimilarityConfig,
    compute_team_similarity,
)


BEST_RE = re.compile(
    r"best_\d+_frame_(?P<frame>\d+)_q_(?P<quality>[0-9.]+)\.jpg$"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path("experiments/tracking_v0/reid_crop_debug"),
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
        "--role-report",
        type=Path,
        default=Path(
            "experiments/tracking_v0/team_appearance_debug/"
            "role_inference_report.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/tracking_v0/team_appearance_debug/"
            "semantic_constraints_report.json"
        ),
    )
    return parser.parse_args()


def load_appearance_result(path: Path) -> KitAppearanceResult:
    payload = json.loads(path.read_text(encoding="utf-8"))

    clusters = tuple(
        AppearanceCluster(
            cluster_id=int(row["cluster_id"]),
            crop_count=int(row["crop_count"]),
            distinct_track_count=int(row["distinct_track_count"]),
            support_weight=float(row["support_weight"]),
            representative_rgb=tuple(row["representative_rgb"]),
            ui_color_hex=str(row["ui_color_hex"]),
            semantic_label=AppearanceLabel(row["semantic_label"]),
        )
        for row in payload["clusters"]
    )

    decisions = {}
    for raw_id, row in payload["track_decisions"].items():
        track_id = int(raw_id)
        decisions[track_id] = TrackAppearanceDecision(
            track_id=track_id,
            label=AppearanceLabel(row["label"]),
            cluster_id=(
                int(row["cluster_id"])
                if row["cluster_id"] is not None
                else None
            ),
            crop_count=int(row["crop_count"]),
            support_weight=float(row["support_weight"]),
            top_share=float(row["top_share"]),
            second_share=float(row["second_share"]),
            representative_rgb=(
                tuple(row["representative_rgb"])
                if row["representative_rgb"] is not None
                else None
            ),
            ui_color_hex=row["ui_color_hex"],
            cluster_shares={
                int(k): float(v)
                for k, v in row["cluster_shares"].items()
            },
        )

    # Reconstruct crop-cluster sequence by assigning each best crop to the
    # nearest semantic cluster through the source track decision.  For this
    # diagnostic we only need team prototypes from clearly TEAM_A/B tracks.
    return KitAppearanceResult(
        chosen_k=int(payload["chosen_k"]),
        silhouette_score=float(payload["silhouette_score"]),
        crop_cluster_ids=(),
        clusters=clusters,
        track_decisions=decisions,
    )


def load_crops(crop_dir: Path, appearance_result: KitAppearanceResult):
    rows = []
    cluster_ids = []

    label_to_cluster = {
        cluster.semantic_label: cluster.cluster_id
        for cluster in appearance_result.clusters
        if cluster.semantic_label in {
            AppearanceLabel.TEAM_A,
            AppearanceLabel.TEAM_B,
        }
    }

    for track_dir in sorted(crop_dir.glob("track_*")):
        if not track_dir.is_dir():
            continue
        try:
            track_id = int(track_dir.name.split("_", 1)[1])
        except (ValueError, IndexError):
            continue

        decision = appearance_result.track_decisions.get(track_id)
        if decision is None:
            continue

        for path in sorted(track_dir.glob("best_*.jpg")):
            match = BEST_RE.match(path.name)
            if match is None:
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue

            frame = int(match.group("frame"))
            quality = float(match.group("quality"))
            try:
                item = extract_crop_appearance(
                    image,
                    track_id=track_id,
                    frame_index=frame,
                    quality=quality,
                )
            except ValueError:
                continue

            rows.append(item)

            # Team prototype construction is intentionally restricted to
            # B.1-pure team tracks. Mixed/SPECIAL crops do not shape prototypes.
            if decision.label in label_to_cluster:
                cluster_ids.append(label_to_cluster[decision.label])
            else:
                # Placeholder; replaced below with the nearest team only after
                # prototypes are learned from pure tracks.
                cluster_ids.append(-1)

    pure_rows = [
        row
        for row, cluster_id in zip(rows, cluster_ids)
        if cluster_id >= 0
    ]
    pure_cluster_ids = [
        cluster_id
        for cluster_id in cluster_ids
        if cluster_id >= 0
    ]

    prototype_result = KitAppearanceResult(
        chosen_k=appearance_result.chosen_k,
        silhouette_score=appearance_result.silhouette_score,
        crop_cluster_ids=tuple(pure_cluster_ids),
        clusters=appearance_result.clusters,
        track_decisions=appearance_result.track_decisions,
    )
    return rows, pure_rows, prototype_result


def load_role_result(path: Path) -> RoleInferenceResult:
    payload = json.loads(path.read_text(encoding="utf-8"))
    decisions = {}

    for raw_id, row in payload["decisions"].items():
        track_id = int(raw_id)
        g = row["geometry"]
        geometry = TrackRoleGeometry(
            **{
                key: g[key]
                for key in TrackRoleGeometry.__dataclass_fields__
            }
        )
        s = row["scores"]
        scores = RoleScores(
            goalkeeper_left=float(s["goalkeeper_left"]),
            goalkeeper_right=float(s["goalkeeper_right"]),
            referee=float(s["referee"]),
            outfield=float(s["outfield"]),
        )
        decisions[track_id] = TrackRoleDecision(
            track_id=track_id,
            role=FootballRole(row["role"]),
            role_score=float(row["role_score"]),
            runner_up_role=(
                FootballRole(row["runner_up_role"])
                if row["runner_up_role"] is not None
                else None
            ),
            runner_up_score=float(row["runner_up_score"]),
            margin=float(row["margin"]),
            appearance_label=AppearanceLabel(row["appearance_label"]),
            appearance_purity=float(row["appearance_purity"]),
            ui_color_hex=row["ui_color_hex"],
            geometry=geometry,
            scores=scores,
            reason=str(row["reason"]),
        )

    def ranking(name):
        return tuple(
            int(value)
            for value in payload["rankings"][name]
        )

    return RoleInferenceResult(
        decisions=decisions,
        left_goalkeeper_ranking=ranking("goalkeeper_left"),
        right_goalkeeper_ranking=ranking("goalkeeper_right"),
        referee_ranking=ranking("referee"),
    )


def main():
    args = parse_args()

    appearance = load_appearance_result(args.appearance_report)
    all_crops, pure_crops, prototype_result = load_crops(
        args.crop_dir,
        appearance,
    )

    # Build prototypes from pure team crops first.
    from tracking_v0.team_similarity import build_team_prototypes, classify_crop_relative_to_teams, aggregate_track_team_similarity, TeamSimilarityResult

    prototypes = build_team_prototypes(
        pure_crops,
        prototype_result,
    )

    source_labels = {
        track_id: decision.label
        for track_id, decision in appearance.track_decisions.items()
    }

    scored_crops = [
        classify_crop_relative_to_teams(
            crop,
            prototypes,
            source_appearance_label=source_labels.get(
                crop.track_id,
                AppearanceLabel.UNKNOWN,
            ),
            config=TeamSimilarityConfig(),
        )
        for crop in all_crops
    ]

    grouped = {}
    for row in scored_crops:
        grouped.setdefault(row.track_id, []).append(row)

    team_tracks = {
        track_id: aggregate_track_team_similarity(
            track_id,
            rows,
            source_appearance_label=source_labels.get(
                track_id,
                AppearanceLabel.UNKNOWN,
            ),
            config=TeamSimilarityConfig(),
        )
        for track_id, rows in sorted(grouped.items())
    }

    role_result = load_role_result(args.role_report)
    global_roles = select_global_unique_roles(
        role_result,
        config=GlobalRoleSelectionConfig(),
    )

    print()
    print("Sprint B.2.1 semantic constraints audit")
    print()
    print("GLOBAL UNIQUE ROLES")
    for role, winner in global_roles.winners_by_role().items():
        print(
            f"{role.value:>18} : "
            f"{str(winner.track_id):>4} "
            f"score={winner.score:.3f} "
            f"runner={winner.runner_up_track_id} "
            f"runner_score={winner.runner_up_score:.3f} "
            f"margin={winner.margin:.3f} "
            f"[{winner.reason}]"
        )

    print()
    print("RELATIVE TEAM SIMILARITY")
    print(
        f"{'ID':>4} {'B1':>9} {'RELATIVE':>9} "
        f"{'simA':>6} {'simB':>6} {'MARGIN':>7} "
        f"{'voteA':>6} {'voteB':>6} REASON"
    )
    print("-" * 90)

    for track_id, row in sorted(team_tracks.items()):
        print(
            f"{track_id:>4} "
            f"{row.source_appearance_label.value:>9} "
            f"{row.label.value:>9} "
            f"{row.team_a_similarity:>6.3f} "
            f"{row.team_b_similarity:>6.3f} "
            f"{row.margin:>7.3f} "
            f"{row.team_a_vote_share:>6.2f} "
            f"{row.team_b_vote_share:>6.2f} "
            f"{row.reason}"
        )

    payload = {
        "version": "tracking-semantic-constraints-b2-1-v1",
        "global_roles": global_roles.to_dict(),
        "team_similarity": {
            str(track_id): row.to_dict()
            for track_id, row in sorted(team_tracks.items())
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print()
    print(f"report : {args.output.resolve()}")
    print("Diagnostic only; no tracks.person / V2.7 changes were made.")


if __name__ == "__main__":
    main()
