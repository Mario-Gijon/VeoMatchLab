#!/usr/bin/env python3
"""Audit Sprint B.1 unsupervised kit appearance on Sprint A ReID crops.

Input:
    experiments/tracking/reid_crop_debug/
        report.json
        track_001/best_*.jpg
        ...

Output:
    <out>/team_appearance_report.json
    <out>/cluster_XX_<label>/contact_sheet.jpg
    <out>/mixed_tracks/track_XXX.jpg

No ReID identity fusion is performed.
No HOME/AWAY semantic guess is made.
No goalkeeper/referee semantic role is forced.

The two dominant outfield-looking appearance groups are called TEAM_A and
TEAM_B.  Smaller visual groups are SPECIAL.  A raw track whose crops vote for
multiple appearance groups becomes MIXED; insufficient evidence becomes UNKNOWN.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.team_appearance import (  # noqa: E402
    AppearanceLabel,
    CropAppearance,
    KitAppearanceConfig,
    extract_crop_appearance,
    fit_kit_appearance,
)


DEFAULT_CROP_DIR = Path(
    "experiments/tracking/reid_crop_debug"
)
DEFAULT_OUT_DIR = Path(
    "experiments/tracking/team_appearance_debug"
)

BEST_RE = re.compile(
    r"best_\d+_frame_(?P<frame>\d+)_q_(?P<quality>[0-9.]+)\.jpg$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--crop-dir",
        type=Path,
        default=DEFAULT_CROP_DIR,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--min-track-crops",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--min-track-purity",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--mixed-second-share",
        type=float,
        default=0.22,
    )
    return parser.parse_args()


def load_crops(
    crop_dir: Path,
    *,
    config: KitAppearanceConfig,
) -> tuple[
    list[CropAppearance],
    list[dict[str, Any]],
]:
    report_path = crop_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(
            f"Sprint A report not found: {report_path}"
        )

    report = json.loads(
        report_path.read_text(
            encoding="utf-8",
        )
    )

    appearances: list[CropAppearance] = []
    debug_rows: list[dict[str, Any]] = []

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

        track_report = report.get(
            "tracks",
            {},
        ).get(
            str(track_id),
            {},
        )

        quality_by_frame = {
            int(row["frame_index"]): float(
                row["quality"]
            )
            for row in track_report.get(
                "selected",
                [],
            )
            if (
                row.get("frame_index") is not None
                and row.get("quality") is not None
            )
        }

        for path in sorted(
            track_dir.glob("best_*.jpg")
        ):
            match = BEST_RE.match(
                path.name
            )
            if match is None:
                continue

            frame_index = int(
                match.group("frame")
            )
            quality = quality_by_frame.get(
                frame_index,
                float(
                    match.group("quality")
                ),
            )

            image = cv2.imread(
                str(path),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue

            try:
                appearance = extract_crop_appearance(
                    image,
                    track_id=track_id,
                    frame_index=frame_index,
                    quality=quality,
                    config=config,
                )
            except ValueError:
                continue

            debug_index = len(debug_rows)
            appearances.append(
                appearance
            )
            debug_rows.append(
                {
                    "index": debug_index,
                    "track_id": track_id,
                    "frame_index": frame_index,
                    "quality": quality,
                    "path": str(
                        path.resolve()
                    ),
                    "image": image,
                }
            )

    if not appearances:
        raise RuntimeError(
            f"No Sprint A best_*.jpg crops found under {crop_dir}"
        )

    return appearances, debug_rows


def _tile(
    image: np.ndarray,
    *,
    caption: str,
) -> np.ndarray:
    target_h = 256
    target_w = 128

    if image.shape[:2] != (
        target_h,
        target_w,
    ):
        image = cv2.resize(
            image,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )

    caption_h = 42
    output = np.zeros(
        (
            target_h + caption_h,
            target_w,
            3,
        ),
        dtype=np.uint8,
    )
    output[:target_h] = image

    cv2.putText(
        output,
        caption,
        (3, target_h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def contact_sheet(
    rows: list[tuple[np.ndarray, str]],
    *,
    columns: int = 6,
) -> np.ndarray:
    if not rows:
        raise ValueError(
            "contact_sheet requires rows"
        )

    tiles = [
        _tile(
            image,
            caption=caption,
        )
        for image, caption in rows
    ]

    h, w = tiles[0].shape[:2]
    n_rows = int(
        math.ceil(
            len(tiles) / columns
        )
    )

    sheet = np.zeros(
        (
            n_rows * h,
            columns * w,
            3,
        ),
        dtype=np.uint8,
    )

    for index, tile in enumerate(
        tiles
    ):
        row, col = divmod(
            index,
            columns,
        )
        sheet[
            row * h:(row + 1) * h,
            col * w:(col + 1) * w,
        ] = tile

    return sheet


def persist_debug(
    *,
    out_dir: Path,
    appearances: list[CropAppearance],
    debug_rows: list[dict[str, Any]],
    result,
) -> None:
    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cluster_by_id = {
        cluster.cluster_id: cluster
        for cluster in result.clusters
    }

    rows_by_cluster: dict[
        int,
        list[tuple[np.ndarray, str]],
    ] = {}

    for appearance, row, cluster_id in zip(
        appearances,
        debug_rows,
        result.crop_cluster_ids,
    ):
        rows_by_cluster.setdefault(
            cluster_id,
            [],
        ).append(
            (
                row["image"],
                (
                    f"t{appearance.track_id} "
                    f"f{appearance.frame_index} "
                    f"q{appearance.quality:.2f}"
                ),
            )
        )

    for cluster_id, rows in sorted(
        rows_by_cluster.items()
    ):
        cluster = cluster_by_id[
            cluster_id
        ]
        cluster_dir = out_dir / (
            f"cluster_{cluster_id:02d}_"
            f"{cluster.semantic_label.value}"
        )
        cluster_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        cv2.imwrite(
            str(
                cluster_dir
                / "contact_sheet.jpg"
            ),
            contact_sheet(
                rows,
            ),
        )

    mixed_dir = out_dir / "mixed_tracks"
    mixed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    indices_by_track: dict[
        int,
        list[int],
    ] = {}

    for index, appearance in enumerate(
        appearances
    ):
        indices_by_track.setdefault(
            appearance.track_id,
            [],
        ).append(index)

    for track_id, decision in sorted(
        result.track_decisions.items()
    ):
        if decision.label not in {
            AppearanceLabel.MIXED,
            AppearanceLabel.UNKNOWN,
        }:
            continue

        rows: list[
            tuple[np.ndarray, str]
        ] = []

        for index in indices_by_track.get(
            track_id,
            [],
        ):
            appearance = appearances[
                index
            ]
            cluster_id = result.crop_cluster_ids[
                index
            ]
            rows.append(
                (
                    debug_rows[index]["image"],
                    (
                        f"f{appearance.frame_index} "
                        f"c{cluster_id} "
                        f"q{appearance.quality:.2f}"
                    ),
                )
            )

        if rows:
            cv2.imwrite(
                str(
                    mixed_dir
                    / f"track_{track_id:03d}.jpg"
                ),
                contact_sheet(
                    rows,
                    columns=5,
                ),
            )


def print_report(
    result,
) -> None:
    print()
    print("Sprint B.1 kit appearance audit")
    print(
        f"chosen clusters : {result.chosen_k}"
    )
    print(
        "silhouette      : "
        f"{result.silhouette_score:.3f}"
    )

    print()
    print("CLUSTERS")
    print(
        f"{'ID':>3} {'LABEL':>9} "
        f"{'CROPS':>6} {'TRACKS':>6} "
        f"{'SUPPORT':>8} {'COLOR':>9}"
    )
    print("-" * 52)

    for cluster in result.clusters:
        print(
            f"{cluster.cluster_id:>3} "
            f"{cluster.semantic_label.value:>9} "
            f"{cluster.crop_count:>6} "
            f"{cluster.distinct_track_count:>6} "
            f"{cluster.support_weight:>8.2f} "
            f"{cluster.ui_color_hex:>9}"
        )

    print()
    print("TRACK DECISIONS")
    print(
        f"{'ID':>4} {'LABEL':>9} "
        f"{'CL':>3} {'CROPS':>5} "
        f"{'TOP':>6} {'SECOND':>6} "
        f"{'COLOR':>9}"
    )
    print("-" * 56)

    for track_id, decision in sorted(
        result.track_decisions.items()
    ):
        print(
            f"{track_id:>4} "
            f"{decision.label.value:>9} "
            f"{str(decision.cluster_id):>3} "
            f"{decision.crop_count:>5} "
            f"{decision.top_share:>6.2f} "
            f"{decision.second_share:>6.2f} "
            f"{str(decision.ui_color_hex):>9}"
        )


def main() -> None:
    args = parse_args()

    crop_dir = (
        args.crop_dir
        .expanduser()
        .resolve()
    )
    out_dir = (
        args.out_dir
        .expanduser()
        .resolve()
    )

    config = KitAppearanceConfig(
        min_clusters=args.min_clusters,
        max_clusters=args.max_clusters,
        min_track_crops=args.min_track_crops,
        min_track_purity=args.min_track_purity,
        mixed_second_share=args.mixed_second_share,
    )

    appearances, debug_rows = load_crops(
        crop_dir,
        config=config,
    )
    result = fit_kit_appearance(
        appearances,
        config=config,
    )

    print_report(
        result,
    )

    persist_debug(
        out_dir=out_dir,
        appearances=appearances,
        debug_rows=debug_rows,
        result=result,
    )

    payload = {
        "version": "tracking-kit-appearance-sprint-b1-v1",
        "crop_dir": str(
            crop_dir
        ),
        "config": {
            field: getattr(
                config,
                field,
            )
            for field in config.__dataclass_fields__
        },
        **result.to_dict(),
    }

    report_path = (
        out_dir
        / "team_appearance_report.json"
    )
    report_path.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"debug  : {out_dir}")
    print(f"report : {report_path}")
    print(
        "TEAM_A/TEAM_B are visual groups only; "
        "HOME/AWAY/referee/goalkeeper semantics are not forced yet."
    )


if __name__ == "__main__":
    main()
