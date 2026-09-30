#!/usr/bin/env python3
"""Evaluate actual saved Tracking V1 crops for future ReID use.

This runner does not reopen the video and does not run a ReID network.  It
reads the Hito 2D crop index + JPEGs, computes actual-pixel quality, and writes:

    tracking/v1/diagnostics/reid_quality/
        quality.json
        strongest_reid.jpg
        weakest_reid.jpg
        rejected_reid.jpg

The existing appearance crop artifacts are left untouched.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DEV_CONFIG = ROOT / "experiments" / "tracking" / "dev_clip.json"

from tracking.appearance.quality import (  # noqa: E402
    ReIDCropQualityConfig,
    score_reid_crop,
)


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must encode a JSON object")
    return payload


def tracking_root(
    data_root: Path,
    *,
    match_id: str,
    clip_id: str,
) -> Path:
    return (
        data_root
        / "matches"
        / match_id
        / "clips"
        / clip_id
        / "tracking"
        / "v1"
    )


def _selection_quality(record: Mapping[str, Any]) -> float:
    if "selection_quality" in record:
        return float(record["selection_quality"])
    if "quality" in record:
        return float(record["quality"])
    raise ValueError("crop record has no selection quality")


def score_saved_records(
    *,
    tracking_root_path: Path,
    crop_index: Mapping[str, Any],
    config: ReIDCropQualityConfig,
) -> list[dict[str, Any]]:
    raw_records = crop_index.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("crop index records must be a list")

    results: list[dict[str, Any]] = []

    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("crop record must be a JSON object")

        if raw.get("status") != "saved":
            continue

        crop_ref = raw.get("crop_ref")
        bbox = raw.get("virtual_bbox_xyxy")
        if not isinstance(crop_ref, str) or not crop_ref:
            raise ValueError("saved crop record is missing crop_ref")
        if not isinstance(bbox, (list, tuple)):
            raise ValueError("saved crop record is missing virtual_bbox_xyxy")

        path = tracking_root_path / crop_ref
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read crop: {path}")

        quality = score_reid_crop(
            crop=image,
            virtual_bbox_xyxy=bbox,
            selection_quality=_selection_quality(raw),
            crop_clipped=bool(raw.get("crop_clipped", False)),
            config=config,
        )

        result = {
            "tracklet_id": raw["tracklet_id"],
            "detection_id": raw["detection_id"],
            "frame_index": raw["frame_index"],
            "timestamp_s": raw["timestamp_s"],
            "representative_view": raw.get("representative_view"),
            "crop_ref": crop_ref,
            **quality.to_dict(),
            "_image": image,
        }
        results.append(result)

    return results


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return (
        ordered[low] * (1.0 - fraction)
        + ordered[high] * fraction
    )


def summary(
    records: Sequence[Mapping[str, Any]],
    config: ReIDCropQualityConfig,
) -> dict[str, Any]:
    qualities = [float(item["reid_quality"]) for item in records]
    heights = [float(item["bbox_height_px"]) for item in records]
    widths = [float(item["bbox_width_px"]) for item in records]

    tier_counts = Counter(str(item["tier"]) for item in records)
    reason_counts: Counter[str] = Counter()
    for item in records:
        for reason in item["rejection_reasons"]:
            reason_counts[str(reason)] += 1

    eligible_count = sum(bool(item["reid_eligible"]) for item in records)

    return {
        "crop_count": len(records),
        "reid_eligible_count": eligible_count,
        "reid_rejected_count": len(records) - eligible_count,
        "eligible_fraction": (
            eligible_count / len(records)
            if records
            else None
        ),
        "tier_counts": {
            key: tier_counts[key]
            for key in sorted(tier_counts)
        },
        "rejection_reason_counts": {
            key: reason_counts[key]
            for key in sorted(reason_counts)
        },
        "reid_quality": {
            "min": min(qualities, default=None),
            "p10": _quantile(qualities, 0.10),
            "median": _quantile(qualities, 0.50),
            "p90": _quantile(qualities, 0.90),
            "max": max(qualities, default=None),
            "mean": statistics.fmean(qualities) if qualities else None,
        },
        "bbox_height_px": {
            "min": min(heights, default=None),
            "median": _quantile(heights, 0.50),
            "max": max(heights, default=None),
        },
        "bbox_width_px": {
            "min": min(widths, default=None),
            "median": _quantile(widths, 0.50),
            "max": max(widths, default=None),
        },
        "config": config.to_dict(),
    }


def _fit_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(
        width / image.shape[1],
        height / image.shape[0],
    )
    resized_w = max(1, int(round(image.shape[1] * scale)))
    resized_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def make_contact_sheet(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: int = 8,
    panel_width: int = 170,
    panel_height: int = 250,
) -> np.ndarray | None:
    if not records:
        return None

    rows = (len(records) + columns - 1) // columns
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 62

    for index, item in enumerate(records):
        image = item.get("_image")
        if not isinstance(image, np.ndarray):
            continue

        panel = np.zeros(
            (panel_height, panel_width, 3),
            dtype=np.uint8,
        )
        panel[:image_height] = _fit_panel(
            image,
            width=panel_width,
            height=image_height,
        )

        cv2.putText(
            panel,
            f"{item['tracklet_id']} {item['tier']}",
            (4, image_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.32,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"sel={item['selection_quality']:.2f} "
                f"reid={item['reid_quality']:.2f}"
            ),
            (4, image_height + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.31,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"{item['bbox_width_px']:.0f}x"
                f"{item['bbox_height_px']:.0f}px "
                f"lap={item['laplacian_variance']:.0f}"
            ),
            (4, image_height + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"t={item['timestamp_s']:.2f}s",
            (4, image_height + 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = index // columns
        column = index % columns
        y = row * panel_height
        x = column * panel_width
        sheet[y:y + panel_height, x:x + panel_width] = panel

    return sheet


def _public(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-config",
        default=str(DEFAULT_DEV_CONFIG),
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
    )
    parser.add_argument(
        "--crop-index",
        default=None,
        help=(
            "Default: <tracking-v1>/diagnostics/"
            "appearance_crops/index.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Default: <tracking-v1>/diagnostics/reid_quality",
    )
    parser.add_argument("--audit-count", type=int, default=40)
    parser.add_argument(
        "--min-bbox-height",
        type=float,
        default=48.0,
    )
    parser.add_argument(
        "--min-bbox-width",
        type=float,
        default=12.0,
    )
    parser.add_argument(
        "--min-reid-quality",
        type=float,
        default=0.55,
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)

    dev = load_json(_repo_path(args.dev_config))
    data_root = _repo_path(args.data_root)
    root = tracking_root(
        data_root,
        match_id=dev["match_id"],
        clip_id=dev["clip_id"],
    )

    crop_index_path = (
        _repo_path(args.crop_index)
        if args.crop_index is not None
        else root / "diagnostics" / "appearance_crops" / "index.json"
    )
    output_dir = (
        _repo_path(args.output_dir)
        if args.output_dir is not None
        else root / "diagnostics" / "reid_quality"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = ReIDCropQualityConfig(
        hard_min_bbox_height_px=args.min_bbox_height,
        hard_min_bbox_width_px=args.min_bbox_width,
        min_reid_quality=args.min_reid_quality,
    )

    crop_index = load_json(crop_index_path)
    records = score_saved_records(
        tracking_root_path=root,
        crop_index=crop_index,
        config=config,
    )
    report = summary(records, config)

    audit_count = max(1, int(args.audit_count))
    ranked = sorted(
        records,
        key=lambda item: (
            float(item["reid_quality"]),
            item["tracklet_id"],
            int(item["frame_index"]),
        ),
    )
    weakest = ranked[:audit_count]
    strongest = list(reversed(ranked[-audit_count:]))
    rejected = sorted(
        [
            item
            for item in records
            if not item["reid_eligible"]
        ],
        key=lambda item: (
            float(item["bbox_height_px"]),
            float(item["reid_quality"]),
            item["tracklet_id"],
        ),
    )[:audit_count]

    sheet_specs = (
        ("weakest_reid.jpg", weakest),
        ("strongest_reid.jpg", strongest),
        ("rejected_reid.jpg", rejected),
    )
    sheet_refs: dict[str, str | None] = {}

    for filename, items in sheet_specs:
        sheet = make_contact_sheet(items)
        path = output_dir / filename
        if sheet is None:
            sheet_refs[filename] = None
            continue
        if not cv2.imwrite(str(path), sheet):
            raise RuntimeError(f"could not write {path}")
        sheet_refs[filename] = str(path.relative_to(root))

    payload = {
        "match_id": dev["match_id"],
        "clip_id": dev["clip_id"],
        "crop_index": str(crop_index_path.relative_to(root)),
        "summary": report,
        "contact_sheets": sheet_refs,
        "records": [_public(item) for item in records],
    }

    output_path = output_dir / "quality.json"
    output_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("[reid-quality] completed")
    print(json.dumps(report, indent=2))
    for filename, ref in sheet_refs.items():
        if ref is not None:
            print(f"[reid-quality] {filename}={root / ref}")
    print(f"[reid-quality] output={output_path}")

    return payload


if __name__ == "__main__":
    run()
