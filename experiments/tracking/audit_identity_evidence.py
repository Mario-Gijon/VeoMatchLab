#!/usr/bin/env python3
"""Sprint C.2 audit: appearance segmentation + multi-signal identity evidence.

Consumes existing outputs only:
  * Sprint C.1 reid_embeddings.npz
  * raw tracking_frames.jsonl
  * Sprint B.2.1 semantic_constraints_report.json

No detector or OSNet rerun is needed.

Default benchmark:
  expected same:
      1-29, 9-38, 10-41, 3-43
  expected different:
      3-4, 11-25, 27-28

Output:
  experiments/tracking/reid_similarity_debug/
      identity_evidence_report.json
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.appearance_bank import (  # noqa: E402
    AppearanceBankConfig,
    ReIDEmbedding,
    build_appearance_banks,
)
from tracking.appearance_segments import (  # noqa: E402
    AppearanceSegmentationConfig,
    segment_all_banks,
)
from tracking.identity_evidence import (  # noqa: E402
    IdentityEvidenceConfig,
    ObservedTrackPoint,
    build_identity_pair_evidence,
)
from tracking.team_appearance import AppearanceLabel  # noqa: E402
from tracking.team_similarity import RelativeTeamLabel  # noqa: E402


DEFAULT_SAME = (
    (1, 29),
    (9, 38),
    (10, 41),
    (3, 43),
)
DEFAULT_DIFFERENT = (
    (3, 4),
    (11, 25),
    (27, 28),
)


def parse_args() -> argparse.Namespace:
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
        required=True,
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
        "--output",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug/"
            "identity_evidence_report.json"
        ),
    )
    return parser.parse_args()


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

    rows = []

    for index in range(len(vectors)):
        rows.append(
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
        )

    return rows


def load_observed_points(
    path: Path,
) -> dict[int, tuple[ObservedTrackPoint, ...]]:
    grouped: dict[
        int,
        list[ObservedTrackPoint],
    ] = defaultdict(list)

    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line_number, line in enumerate(
            stream,
            start=1,
        ):
            if not line.strip():
                continue

            row = json.loads(line)
            frame_index = int(
                row["frame_index"]
            )
            timestamp = float(
                row.get(
                    "timestamp_s",
                    row.get("timestamp"),
                )
            )

            for track in row["tracks"]:
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

                track_id = int(
                    track["track_id"]
                )

                grouped[
                    track_id
                ].append(
                    ObservedTrackPoint(
                        track_id=track_id,
                        frame_index=frame_index,
                        timestamp=timestamp,
                        x=float(
                            track["x"]
                        ),
                        z=float(
                            track["z"]
                        ),
                    )
                )

    return {
        track_id: tuple(
            sorted(
                rows,
                key=lambda item: (
                    item.frame_index,
                    item.timestamp,
                ),
            )
        )
        for track_id, rows in sorted(
            grouped.items()
        )
    }


def load_team_labels(
    path: Path,
) -> dict[int, RelativeTeamLabel]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    result = {}

    for raw_id, row in payload.get(
        "team_similarity",
        {},
    ).items():
        label = str(
            row.get(
                "label",
                "unknown",
            )
        )
        try:
            result[
                int(raw_id)
            ] = RelativeTeamLabel(
                label
            )
        except ValueError:
            result[
                int(raw_id)
            ] = RelativeTeamLabel.UNKNOWN

    return result


def load_appearance_labels(
    path: Path,
) -> dict[int, AppearanceLabel]:
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
            result[
                int(raw_id)
            ] = AppearanceLabel(
                str(
                    row.get(
                        "label",
                        "unknown",
                    )
                )
            )
        except ValueError:
            result[
                int(raw_id)
            ] = AppearanceLabel.UNKNOWN

    return result


def main() -> None:
    args = parse_args()

    embeddings = load_embeddings(
        args.embedding_cache
    )
    observed_points = load_observed_points(
        args.tracking_frames
    )
    team_labels = load_team_labels(
        args.semantic_report
    )
    appearance_labels = load_appearance_labels(
        args.appearance_report
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

    print()
    print("Sprint C.2 appearance segmentation")
    print(
        f"tracks with banks    : {len(banks)}"
    )
    print(
        "segmented raw tracks : "
        f"{sum(row.segmented for row in segmentations.values())}"
    )

    print()
    print("SEGMENTED / SUSPICIOUS TRACKS")
    print(
        f"{'ID':>4} {'B1':>9} {'SEGS':>4} "
        f"{'BOUNDARIES':>18} {'GAIN':>8} {'CROSS':>8}"
    )
    print("-" * 70)

    suspicious = sorted(
        (
            row
            for row in segmentations.values()
            if (
                row.segmented
                or row.best_rejected_split is not None
            )
        ),
        key=lambda row: (
            not row.segmented,
            row.track_id,
        ),
    )

    for row in suspicious:
        if row.accepted_splits:
            best = max(
                row.accepted_splits,
                key=lambda item: item.split_gain,
            )
            boundaries = ",".join(
                str(item.boundary_after_frame)
                for item in row.accepted_splits
            )
        else:
            best = row.best_rejected_split
            boundaries = "-"

        print(
            f"{row.track_id:>4} "
            f"{row.source_label.value:>9} "
            f"{len(row.segments):>4} "
            f"{boundaries:>18} "
            f"{(best.split_gain if best else 0.0):>8.3f} "
            f"{(best.cross_similarity if best else 0.0):>8.3f}"
        )

    evidence_config = IdentityEvidenceConfig()

    pair_rows: list[
        tuple[str, Any]
    ] = []

    for expected, pairs in (
        ("same", DEFAULT_SAME),
        ("different", DEFAULT_DIFFERENT),
    ):
        for track_a, track_b in pairs:
            if (
                track_a not in segmentations
                or track_b not in segmentations
                or track_a not in observed_points
                or track_b not in observed_points
            ):
                pair_rows.append(
                    (
                        expected,
                        {
                            "track_a": track_a,
                            "track_b": track_b,
                            "status": "missing_data",
                        },
                    )
                )
                continue

            evidence = build_identity_pair_evidence(
                track_a,
                track_b,
                segmentations=segmentations,
                observed_points=observed_points,
                team_labels=team_labels,
                bank_config=bank_config,
                config=evidence_config,
            )
            pair_rows.append(
                (
                    expected,
                    evidence,
                )
            )

    print()
    print("FUSION BENCHMARK")
    print(
        f"{'PAIR':>9} {'EXPECT':>9} {'TEAM':>13} "
        f"{'REID':>6} {'GEO':>6} {'OVLP':>6} "
        f"{'Dmed':>6} {'FUSED':>7} {'RESULT':>16} "
        f"{'SEGS':>9}"
    )
    print("-" * 110)

    report_pairs = []

    for expected, item in pair_rows:
        if isinstance(item, dict):
            print(
                f"{item['track_a']}↔{item['track_b']} "
                f"{expected:>9} missing"
            )
            report_pairs.append(
                {
                    **item,
                    "expected": expected,
                }
            )
            continue

        geometry = item.geometry
        segment_pair = (
            f"{item.best_segment_pair.segment_a}"
            f"↔{item.best_segment_pair.segment_b}"
        )

        pair = (
            f"{item.track_a}"
            f"↔{item.track_b}"
        )

        dmed = (
            geometry.simultaneous_median_distance_m
        )

        print(
            f"{pair:>9} "
            f"{expected:>9} "
            f"{item.team_compatibility:>13} "
            f"{item.reid_score:>6.3f} "
            f"{item.geometry_score:>6.3f} "
            f"{geometry.simultaneous_fraction:>6.2f} "
            f"{(dmed if dmed is not None else -1.0):>6.2f} "
            f"{item.fused_support_score:>7.3f} "
            f"{item.label.value:>16} "
            f"{segment_pair:>9}"
        )

        report_pairs.append(
            {
                **item.to_dict(),
                "expected": expected,
            }
        )

    payload = {
        "version": "tracking-identity-evidence-c2-v1",
        "segmentation_config": {
            field: getattr(
                segment_config,
                field,
            )
            for field in segment_config.__dataclass_fields__
        },
        "identity_evidence_config": {
            field: getattr(
                evidence_config,
                field,
            )
            for field in evidence_config.__dataclass_fields__
        },
        "segmentations": {
            str(track_id): row.to_dict()
            for track_id, row in sorted(
                segmentations.items()
            )
        },
        "pair_benchmark": report_pairs,
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
        "Audit only: no raw-track merge, no V2.7 changes, "
        "no canonical identity publication."
    )


if __name__ == "__main__":
    main()
