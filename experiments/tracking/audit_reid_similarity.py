#!/usr/bin/env python3
"""Sprint C.1 audit: OSNet embeddings on rectified Sprint A crops.

This script does NOT change identity resolution or tracks.person.

It:
  1. loads Sprint A best_*.jpg crops,
  2. extracts official Torchreid OSNet embeddings,
  3. builds multi-embedding appearance banks per raw track,
  4. reports within-track coherence,
  5. evaluates known positive/negative raw-track pairs,
  6. optionally reads semantic_constraints_report.json for team compatibility,
  7. writes a reusable .npz embedding cache + JSON report.

Default benchmark pairs for the current difficult clip:
  expected same:
      1-29, 9-38, 10-41, 3-43
  expected different:
      3-4, 11-25, 27-28
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.appearance_bank import (  # noqa: E402
    AppearanceBankConfig,
    ReIDEmbedding,
    build_appearance_banks,
    compare_track_banks,
    within_track_coherence,
)
from tracking.reid_encoder import (  # noqa: E402
    ReIDEncoderConfig,
    TorchreidEncoder,
)


BEST_RE = re.compile(
    r"best_\d+_frame_(?P<frame>\d+)_q_(?P<quality>[0-9.]+)\.jpg$"
)

DEFAULT_POSITIVE_PAIRS = (
    (1, 29),
    (9, 38),
    (10, 41),
    (3, 43),
)

DEFAULT_NEGATIVE_PAIRS = (
    (3, 4),
    (11, 25),
    (27, 28),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=Path(
            "experiments/tracking/reid_crop_debug"
        ),
    )
    parser.add_argument(
        "--weights",
        type=Path,
        required=True,
        help=(
            "Official Torchreid ReID checkpoint, e.g. "
            "osnet_ain_x1_0 MSMT17 combineall."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="osnet_ain_x1_0",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--semantic-report",
        type=Path,
        default=Path(
            "experiments/tracking/"
            "team_appearance_debug/"
            "semantic_constraints_report.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/tracking/"
            "reid_similarity_debug"
        ),
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--strong-similarity-threshold",
        type=float,
        default=0.75,
        help=(
            "Audit-only coverage threshold; not an identity decision threshold."
        ),
    )
    parser.add_argument(
        "--positive-pair",
        nargs=2,
        type=int,
        action="append",
        default=None,
        metavar=("A", "B"),
    )
    parser.add_argument(
        "--negative-pair",
        nargs=2,
        type=int,
        action="append",
        default=None,
        metavar=("A", "B"),
    )
    return parser.parse_args()


def load_crop_rows(
    crop_dir: Path,
) -> tuple[
    list[np.ndarray],
    list[dict[str, Any]],
]:
    images: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for track_dir in sorted(
        crop_dir.glob("track_*")
    ):
        if not track_dir.is_dir():
            continue

        try:
            track_id = int(
                track_dir.name.split("_", 1)[1]
            )
        except (IndexError, ValueError):
            continue

        for path in sorted(
            track_dir.glob("best_*.jpg")
        ):
            match = BEST_RE.match(
                path.name
            )
            if match is None:
                continue

            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue

            images.append(image)
            metadata.append(
                {
                    "track_id": track_id,
                    "frame_index": int(
                        match.group("frame")
                    ),
                    "quality": float(
                        match.group("quality")
                    ),
                    "source_path": str(
                        path.resolve()
                    ),
                }
            )

    if not images:
        raise RuntimeError(
            f"No best_*.jpg crops found under {crop_dir}"
        )

    return images, metadata


def save_embedding_cache(
    path: Path,
    embeddings: list[ReIDEmbedding],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    vectors = np.stack(
        [item.vector for item in embeddings],
        axis=0,
    )

    np.savez_compressed(
        path,
        vectors=vectors,
        track_ids=np.asarray(
            [item.track_id for item in embeddings],
            dtype=np.int32,
        ),
        frame_indices=np.asarray(
            [item.frame_index for item in embeddings],
            dtype=np.int32,
        ),
        qualities=np.asarray(
            [item.quality for item in embeddings],
            dtype=np.float32,
        ),
        source_paths=np.asarray(
            [
                item.source_path or ""
                for item in embeddings
            ],
            dtype=str,
        ),
    )


def load_team_labels(
    path: Path,
) -> dict[int, str]:
    if not path.is_file():
        return {}

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )

    result: dict[int, str] = {}

    for raw_id, row in payload.get(
        "team_similarity",
        {},
    ).items():
        result[int(raw_id)] = str(
            row.get("label", "unknown")
        )

    return result


def team_compatibility(
    track_a: int,
    track_b: int,
    team_labels: dict[int, str],
) -> str:
    a = team_labels.get(
        track_a,
        "unknown",
    )
    b = team_labels.get(
        track_b,
        "unknown",
    )

    known = {"team_a", "team_b"}

    if a in known and b in known:
        if a == b:
            return "same_team"
        return "team_conflict"

    return "unknown"


def pair_payload(
    pair,
    *,
    expected: str,
    banks,
    config,
    team_labels,
) -> dict[str, Any]:
    track_a, track_b = pair

    if track_a not in banks or track_b not in banks:
        return {
            "track_a": track_a,
            "track_b": track_b,
            "expected": expected,
            "status": "missing_bank",
        }

    metrics = compare_track_banks(
        banks[track_a],
        banks[track_b],
        config=config,
    )

    return {
        **metrics.to_dict(),
        "expected": expected,
        "status": "ok",
        "team_compatibility": team_compatibility(
            track_a,
            track_b,
            team_labels,
        ),
    }


def print_pair_table(
    rows: list[dict[str, Any]],
) -> None:
    print()
    print("PAIR BENCHMARK")
    print(
        f"{'PAIR':>9} {'EXPECT':>9} {'TEAM':>13} "
        f"{'ALLmed':>7} {'Q90':>7} {'BEST':>7} "
        f"{'BIDIR':>7} {'TOPK':>7} {'COVER':>7}"
    )
    print("-" * 92)

    for row in rows:
        if row.get("status") != "ok":
            print(
                f"{row['track_a']}-{row['track_b']:>5} "
                f"{row['expected']:>9} "
                f"{'missing':>13}"
            )
            continue

        pair = (
            f"{row['track_a']}"
            f"↔{row['track_b']}"
        )
        print(
            f"{pair:>9} "
            f"{row['expected']:>9} "
            f"{row['team_compatibility']:>13} "
            f"{row['all_pairs_median']:>7.3f} "
            f"{row['all_pairs_q90']:>7.3f} "
            f"{row['best_similarity']:>7.3f} "
            f"{row['bidirectional_best_median']:>7.3f} "
            f"{row['unique_topk_similarity']:>7.3f} "
            f"{row['strong_match_coverage']:>7.2f}"
        )


def main() -> None:
    args = parse_args()

    crop_dir = (
        args.crop_dir
        .expanduser()
        .resolve()
    )
    weights = (
        args.weights
        .expanduser()
        .resolve()
    )
    output_dir = (
        args.output_dir
        .expanduser()
        .resolve()
    )

    if not crop_dir.is_dir():
        raise FileNotFoundError(crop_dir)
    if not weights.is_file():
        raise FileNotFoundError(weights)

    images, metadata = load_crop_rows(
        crop_dir
    )

    print(
        f"Loaded {len(images)} rectified crop(s) "
        f"from {len(set(row['track_id'] for row in metadata))} track(s)"
    )

    encoder = TorchreidEncoder(
        weights,
        config=ReIDEncoderConfig(
            model_name=args.model_name,
            device=args.device,
            batch_size=args.batch_size,
            verbose=False,
        ),
    )

    print(
        f"Extracting {args.model_name} embeddings on {args.device}..."
    )
    vectors = encoder.encode_bgr(
        images
    )

    embeddings = [
        ReIDEmbedding(
            track_id=row["track_id"],
            frame_index=row["frame_index"],
            quality=row["quality"],
            vector=vectors[index],
            source_path=row["source_path"],
        )
        for index, row in enumerate(metadata)
    ]

    bank_config = AppearanceBankConfig(
        min_quality=args.min_quality,
        strong_similarity_threshold=(
            args.strong_similarity_threshold
        ),
    )
    banks = build_appearance_banks(
        embeddings,
        config=bank_config,
    )

    coherence = [
        within_track_coherence(
            bank
        )
        for _, bank in sorted(
            banks.items()
        )
    ]
    coherence_sorted = sorted(
        coherence,
        key=lambda row: (
            row.q10_similarity,
            row.median_similarity,
            row.track_id,
        ),
    )

    print()
    print("LOWEST WITHIN-TRACK COHERENCE")
    print(
        f"{'ID':>4} {'N':>3} {'Q10':>7} "
        f"{'MEDIAN':>7} {'Q90':>7} "
        f"{'MIN':>7} {'MAX':>7}"
    )
    print("-" * 54)

    for row in coherence_sorted[:15]:
        print(
            f"{row.track_id:>4} "
            f"{row.embedding_count:>3} "
            f"{row.q10_similarity:>7.3f} "
            f"{row.median_similarity:>7.3f} "
            f"{row.q90_similarity:>7.3f} "
            f"{row.minimum_similarity:>7.3f} "
            f"{row.maximum_similarity:>7.3f}"
        )

    positive_pairs = tuple(
        tuple(pair)
        for pair in (
            args.positive_pair
            or DEFAULT_POSITIVE_PAIRS
        )
    )
    negative_pairs = tuple(
        tuple(pair)
        for pair in (
            args.negative_pair
            or DEFAULT_NEGATIVE_PAIRS
        )
    )

    team_labels = load_team_labels(
        args.semantic_report
    )

    pair_rows = [
        pair_payload(
            pair,
            expected="same",
            banks=banks,
            config=bank_config,
            team_labels=team_labels,
        )
        for pair in positive_pairs
    ]
    pair_rows.extend(
        pair_payload(
            pair,
            expected="different",
            banks=banks,
            config=bank_config,
            team_labels=team_labels,
        )
        for pair in negative_pairs
    )

    print_pair_table(
        pair_rows
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_path = (
        output_dir
        / "reid_embeddings.npz"
    )
    save_embedding_cache(
        cache_path,
        embeddings,
    )

    report = {
        "version": "tracking-reid-similarity-c1-v1",
        "model_name": args.model_name,
        "weights": str(weights),
        "device": args.device,
        "crop_count": len(images),
        "track_count": len(banks),
        "embedding_dimension": int(
            vectors.shape[1]
        ),
        "bank_config": {
            field: getattr(
                bank_config,
                field,
            )
            for field in bank_config.__dataclass_fields__
        },
        "within_track_coherence": {
            str(row.track_id): row.to_dict()
            for row in coherence
        },
        "pair_benchmark": pair_rows,
        "embedding_cache": str(
            cache_path
        ),
    }

    report_path = (
        output_dir
        / "reid_similarity_report.json"
    )
    report_path.write_text(
        json.dumps(
            report,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"cache  : {cache_path}")
    print(f"report : {report_path}")
    print(
        "Audit only: no identity merge / V2.7 / tracks.person changes."
    )


if __name__ == "__main__":
    main()
