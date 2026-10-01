#!/usr/bin/env python3
"""Prepare Hito 7C.2D-D1 Veo jersey-legibility labeling dataset.

This stage does NOT train a model and does NOT recognize jersey numbers.
It builds a compact, tracklet-diverse labeling set from the already-selected
7C.2A jersey views.

Label semantics:
    0 = useless
        No useful posterior torso/jersey evidence for number recognition.
    1 = back-visible
        Posterior torso/back is visible, but the number is not human-readable.
    2 = number-readable
        A human can reasonably read/discriminate the jersey number.

Design goal:
    High precision for class 2. Jersey evidence is supplemental identity
    evidence, not a requirement to identify every tracklet.

Optional reference predictions from the research-only SoccerNet benchmark are
used ONLY to stratify/select examples. They never become labels automatically.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


DATASET_VERSION = "jersey-legibility-dataset-v1"
LABEL_NAMES = {
    "0": "useless",
    "1": "back-visible",
    "2": "number-readable",
}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
                )
            yield payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _safe_token(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))
    return text.strip("-") or "item"


def _load_reference_predictions(
    path: Path | None,
) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    if not path.is_file():
        raise FileNotFoundError(path)

    result: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(path):
        sample_id = str(row["sample_id"])
        result[sample_id] = {
            "reference_legibility_score": (
                float(row["legibility_score"])
                if row.get("legibility_score") is not None
                else None
            ),
            "reference_legibility_accepted": bool(
                row.get("legibility_accepted", False)
            ),
            "reference_pose_crop_available": bool(
                row.get("pose_crop_available", False)
            ),
            "reference_parsed_number": (
                int(row["parsed_number"])
                if row.get("parsed_number") is not None
                else None
            ),
        }
    return result


def _readiness_thresholds(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float, float]:
    if not rows:
        return 0.0, 0.0
    values = np.asarray(
        [float(row.get("readiness_score", 0.0)) for row in rows],
        dtype=np.float64,
    )
    return (
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    )


def _readiness_bucket(
    score: float,
    *,
    low_threshold: float,
    high_threshold: float,
) -> str:
    value = float(score)
    if value >= high_threshold:
        return "high"
    if value <= low_threshold:
        return "low"
    return "mid"


def _decorate_rows(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    low_threshold, high_threshold = _readiness_thresholds(rows)
    decorated: list[dict[str, Any]] = []

    for row in rows:
        item = dict(row)
        item.update(
            reference.get(
                str(row["sample_id"]),
                {
                    "reference_legibility_score": None,
                    "reference_legibility_accepted": False,
                    "reference_pose_crop_available": False,
                    "reference_parsed_number": None,
                },
            )
        )
        item["sampling_bucket"] = _readiness_bucket(
            float(row.get("readiness_score", 0.0)),
            low_threshold=low_threshold,
            high_threshold=high_threshold,
        )
        decorated.append(item)

    return decorated


def _quota_by_bucket(target_count: int) -> dict[str, int]:
    target = max(0, int(target_count))
    high = int(round(target * 0.45))
    mid = int(round(target * 0.35))
    low = max(0, target - high - mid)
    return {
        "high": high,
        "mid": mid,
        "low": low,
    }


def _sample_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    target_count: int,
    seed: int,
    max_per_tracklet: int,
) -> list[dict[str, Any]]:
    """Sample a deterministic, tracklet-diverse labeling set.

    Research-benchmark positives are prioritized but never converted to labels.
    Within the remaining pool we preserve high/mid/low readiness coverage.

    Tracklet diversity is deliberate: jersey evidence is for identity
    confirmation, so 20 near-duplicate frames of one player are less valuable
    than one or two views from many different fragments.
    """
    if target_count <= 0 or not rows:
        return []
    if max_per_tracklet <= 0:
        raise ValueError("max_per_tracklet must be > 0")

    rng = random.Random(int(seed))
    target = min(int(target_count), len(rows))
    quotas = _quota_by_bucket(target)

    unique: dict[str, dict[str, Any]] = {}
    for raw in rows:
        sample_id = str(raw["sample_id"])
        if sample_id not in unique:
            unique[sample_id] = dict(raw)
    pool = list(unique.values())

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    per_tracklet: Counter[str] = Counter()
    bucket_counts: Counter[str] = Counter()

    def can_take(row: Mapping[str, Any]) -> bool:
        return (
            str(row["sample_id"]) not in selected_ids
            and per_tracklet[str(row["tracklet_id"])] < max_per_tracklet
        )

    def take(row: Mapping[str, Any], reason: str) -> bool:
        if not can_take(row):
            return False
        item = dict(row)
        item["sampling_reason"] = reason
        selected.append(item)
        selected_ids.add(str(item["sample_id"]))
        tracklet_id = str(item["tracklet_id"])
        per_tracklet[tracklet_id] += 1
        bucket_counts[str(item["sampling_bucket"])] += 1
        return True

    # First: reference-positive examples, ordered by reference confidence.
    forced = [
        row
        for row in pool
        if bool(row.get("reference_legibility_accepted", False))
    ]
    forced.sort(
        key=lambda row: (
            -float(row.get("reference_legibility_score") or 0.0),
            -float(row.get("readiness_score", 0.0)),
            str(row["tracklet_id"]),
            int(row.get("frame_index", 0)),
        )
    )
    for row in forced:
        if len(selected) >= target:
            break
        take(row, "reference-positive")

    # Then fill each readiness stratum. Prefer one view per tracklet first,
    # then allow a second view if the quota needs it.
    for bucket in ("high", "mid", "low"):
        candidates = [
            row
            for row in pool
            if str(row["sampling_bucket"]) == bucket
            and str(row["sample_id"]) not in selected_ids
        ]
        rng.shuffle(candidates)

        for pass_limit in (1, max_per_tracklet):
            if bucket_counts[bucket] >= quotas[bucket]:
                break
            for row in candidates:
                if len(selected) >= target:
                    break
                tracklet_id = str(row["tracklet_id"])
                if per_tracklet[tracklet_id] >= pass_limit:
                    continue
                if take(row, f"readiness-{bucket}"):
                    if bucket_counts[bucket] >= quotas[bucket]:
                        break

    # If forcing reference positives or the per-bucket cap made a quota
    # impossible, fill globally while preserving the tracklet cap.
    remaining = [
        row
        for row in pool
        if str(row["sample_id"]) not in selected_ids
    ]
    rng.shuffle(remaining)

    for pass_limit in (1, max_per_tracklet):
        if len(selected) >= target:
            break
        for row in remaining:
            if len(selected) >= target:
                break
            tracklet_id = str(row["tracklet_id"])
            if per_tracklet[tracklet_id] >= pass_limit:
                continue
            take(row, "diversity-fill")

    selected.sort(
        key=lambda row: (
            str(row["sampling_bucket"]),
            str(row["sampling_reason"]),
            str(row["tracklet_id"]),
            int(row.get("frame_index", 0)),
        )
    )
    return selected[:target]


def _fit_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _contact_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    columns: int = 5,
    panel_width: int = 240,
    panel_height: int = 250,
) -> np.ndarray | None:
    if not items:
        return None

    rows = math.ceil(len(items) / columns)
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 68

    for position, item in enumerate(items):
        image = item["image"]
        panel = np.zeros(
            (panel_height, panel_width, 3),
            dtype=np.uint8,
        )
        panel[:image_height] = _fit_panel(
            image,
            width=panel_width,
            height=image_height,
        )

        ref_score = item.get("reference_legibility_score")
        ref_text = (
            "ref=-"
            if ref_score is None
            else f"ref={float(ref_score):.3f}"
        )

        lines = (
            f"#{int(item['dataset_index']):04d} "
            f"{item['tracklet_id']}",
            (
                f"q={float(item['readiness_score']):.3f} "
                f"r={int(item['rank_within_tracklet'])} "
                f"{ref_text}"
            ),
            (
                f"{item['sampling_bucket']} "
                f"{item['sampling_reason']}"
            ),
        )

        for line_index, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (4, image_height + 17 + 19 * line_index),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        row_index = position // columns
        column_index = position % columns
        y = row_index * panel_height
        x = column_index * panel_width
        sheet[
            y:y + panel_height,
            x:x + panel_width,
        ] = panel

    return sheet


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _labeling_guide() -> str:
    return """# Veo jersey-legibility labels

This dataset is for **supplemental identity evidence**. It is not necessary to
read every player's number in every clip.

Use exactly one of:

- `0` — **useless**: front/side view, posterior torso hidden, too small,
  extreme blur/occlusion, staff/non-player, or otherwise not useful for
  jersey-number recognition.
- `1` — **back-visible**: posterior torso/back is visible, but the number is
  not readable enough to trust.
- `2` — **number-readable**: a human can reasonably read or discriminate the
  jersey number from this image.

Important rule: when in doubt between `1` and `2`, choose `1`.
A false class-2 example is more harmful than missing a difficult number.

Do not infer a number from context, player identity, previous frames, or the
reference model. Judge only the image shown.

The `reference_legibility_*` columns are diagnostic research-benchmark outputs
and are never ground-truth labels.
"""


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--reference-predictions",
        help=(
            "Optional predictions.jsonl from the research-only SoccerNet "
            "benchmark. Used only for sampling diagnostics."
        ),
    )
    parser.add_argument("--target-count", type=int, default=320)
    parser.add_argument("--max-per-tracklet", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--sheet-size", type=int, default=20)
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.target_count <= 0:
        raise ValueError("--target-count must be > 0")
    if args.sheet_size <= 0:
        raise ValueError("--sheet-size must be > 0")

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    selected_path = (
        root
        / "diagnostics"
        / "jersey_view"
        / "jersey-view-v1"
        / "selected.jsonl"
    )
    if not selected_path.is_file():
        raise FileNotFoundError(selected_path)

    source_rows = list(_iter_jsonl(selected_path))
    reference_path = (
        Path(args.reference_predictions).expanduser()
        if args.reference_predictions
        else None
    )
    reference = _load_reference_predictions(reference_path)
    decorated = _decorate_rows(source_rows, reference)

    sampled = _sample_rows(
        decorated,
        target_count=args.target_count,
        seed=args.seed,
        max_per_tracklet=args.max_per_tracklet,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            root
            / "diagnostics"
            / "jersey_legibility_dataset"
            / DATASET_VERSION
        )
    )
    images_dir = output_dir / "images"
    sheets_dir = output_dir / "sheets"

    if output_dir.exists():
        shutil.rmtree(output_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    sheets_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    sheet_items: list[dict[str, Any]] = []

    for dataset_index, row in enumerate(sampled, start=1):
        source = root / str(row["number_roi_ref"])
        if not source.is_file():
            raise FileNotFoundError(source)

        filename = (
            f"{dataset_index:04d}__"
            f"{_safe_token(row['tracklet_id'])}__"
            f"{_safe_token(row['sample_id'])}.jpg"
        )
        destination = images_dir / filename
        shutil.copy2(source, destination)

        record = {
            "dataset_index": dataset_index,
            "label": "",
            "label_name": "",
            "notes": "",
            "image": f"images/{filename}",
            "sample_id": str(row["sample_id"]),
            "tracklet_id": str(row["tracklet_id"]),
            "detection_id": str(row["detection_id"]),
            "frame_index": int(row["frame_index"]),
            "timestamp_s": float(row["timestamp_s"]),
            "readiness_score": float(row["readiness_score"]),
            "rank_within_tracklet": int(row["rank_within_tracklet"]),
            "sampling_bucket": str(row["sampling_bucket"]),
            "sampling_reason": str(row["sampling_reason"]),
            "reference_legibility_score": (
                ""
                if row.get("reference_legibility_score") is None
                else float(row["reference_legibility_score"])
            ),
            "reference_legibility_accepted": bool(
                row.get("reference_legibility_accepted", False)
            ),
            "reference_pose_crop_available": bool(
                row.get("reference_pose_crop_available", False)
            ),
            "reference_parsed_number": (
                ""
                if row.get("reference_parsed_number") is None
                else int(row["reference_parsed_number"])
            ),
            "source_number_roi_ref": str(row["number_roi_ref"]),
        }
        manifest_rows.append(record)

        image = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(destination)
        sheet_items.append(
            {
                **record,
                "reference_legibility_score": (
                    None
                    if record["reference_legibility_score"] == ""
                    else record["reference_legibility_score"]
                ),
                "image": image,
            }
        )

    fields = (
        "dataset_index",
        "label",
        "label_name",
        "notes",
        "image",
        "sample_id",
        "tracklet_id",
        "detection_id",
        "frame_index",
        "timestamp_s",
        "readiness_score",
        "rank_within_tracklet",
        "sampling_bucket",
        "sampling_reason",
        "reference_legibility_score",
        "reference_legibility_accepted",
        "reference_pose_crop_available",
        "reference_parsed_number",
        "source_number_roi_ref",
    )
    manifest_path = output_dir / "manifest.csv"
    _write_csv(manifest_path, manifest_rows, fields)

    labels_path = output_dir / "labels.csv"
    _write_csv(
        labels_path,
        [
            {
                "dataset_index": row["dataset_index"],
                "label": "",
                "notes": "",
            }
            for row in manifest_rows
        ],
        ("dataset_index", "label", "notes"),
    )

    sheet_size = int(args.sheet_size)
    sheet_paths: list[str] = []
    for start in range(0, len(sheet_items), sheet_size):
        batch = sheet_items[start:start + sheet_size]
        sheet = _contact_sheet(batch)
        if sheet is None:
            continue
        sheet_number = start // sheet_size + 1
        path = sheets_dir / f"sheet_{sheet_number:03d}.jpg"
        if not cv2.imwrite(
            str(path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        ):
            raise RuntimeError(f"could not write {path}")
        sheet_paths.append(str(path.relative_to(output_dir)))

    (output_dir / "README_LABELS.md").write_text(
        _labeling_guide(),
        encoding="utf-8",
    )

    bucket_counts = Counter(
        row["sampling_bucket"] for row in manifest_rows
    )
    reason_counts = Counter(
        row["sampling_reason"] for row in manifest_rows
    )
    tracklet_counts = Counter(
        row["tracklet_id"] for row in manifest_rows
    )

    summary = {
        "schema_version": 1,
        "dataset_version": DATASET_VERSION,
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_selected_views": len(source_rows),
        "sampled_views": len(manifest_rows),
        "sampled_tracklets": len(tracklet_counts),
        "max_views_for_one_tracklet": (
            max(tracklet_counts.values()) if tracklet_counts else 0
        ),
        "target_count": int(args.target_count),
        "max_per_tracklet": int(args.max_per_tracklet),
        "seed": int(args.seed),
        "bucket_counts": dict(sorted(bucket_counts.items())),
        "sampling_reason_counts": dict(sorted(reason_counts.items())),
        "reference_predictions": (
            str(reference_path) if reference_path is not None else None
        ),
        "reference_rows_matched": sum(
            1
            for row in decorated
            if str(row["sample_id"]) in reference
        ),
        "reference_positive_sampled": sum(
            1
            for row in manifest_rows
            if bool(row["reference_legibility_accepted"])
        ),
        "manifest": manifest_path.name,
        "labels": labels_path.name,
        "labeling_guide": "README_LABELS.md",
        "sheets": sheet_paths,
        "label_names": LABEL_NAMES,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    print(
        "[jersey-legibility-D1] "
        f"source_views={len(source_rows)} "
        f"sampled={len(manifest_rows)} "
        f"tracklets={len(tracklet_counts)}"
    )
    print(
        "[jersey-legibility-D1] "
        f"buckets={json.dumps(summary['bucket_counts'], sort_keys=True)}"
    )
    print(
        "[jersey-legibility-D1] "
        f"reference_positive_sampled="
        f"{summary['reference_positive_sampled']}"
    )
    print(
        "[jersey-legibility-D1] "
        f"sheets={len(sheet_paths)}"
    )
    print(f"[jersey-legibility-D1] artifact={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
