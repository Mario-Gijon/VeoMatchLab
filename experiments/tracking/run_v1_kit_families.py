#!/usr/bin/env python3
"""Run Hito 7C.1D football-constrained kit-family diagnostics."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.appearance.kit_families import (  # noqa: E402
    KIT_FAMILY_VERSION,
    KitFamilyConfig,
    build_kit_family_profiles,
    cluster_kit_families,
    extract_kit_family_sample,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                yield json.loads(raw)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ) + "\n"
            )
            count += 1
    return count


def _fit_panel(image, width=150, height=195):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _sheet(items, columns=8, panel_width=150, panel_height=230):
    if not items:
        return None
    rows = (len(items) + columns - 1) // columns
    result = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 35
    for index, item in enumerate(items):
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:image_height] = _fit_panel(
            item["image"], panel_width, image_height
        )
        cv2.putText(
            panel,
            item["tracklet_id"],
            (3, image_height + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"cos={item['best_cosine']:.3f} "
                f"r={item['profile_reliability']:.2f}"
            ),
            (3, image_height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row = index // columns
        col = index % columns
        result[
            row * panel_height:(row + 1) * panel_height,
            col * panel_width:(col + 1) * panel_width,
        ] = panel
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--sheet-limit", type=int, default=48)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    old_samples_path = (
        root / "diagnostics" / "kit_evidence"
        / "kit-appearance-v1" / "samples.jsonl"
    )
    if not old_samples_path.is_file():
        raise FileNotFoundError(old_samples_path)

    config = KitFamilyConfig()
    samples = []
    rejected = 0
    best_crop_by_tracklet = {}

    for row in _iter_jsonl(old_samples_path):
        crop_path = root / str(row["crop_ref"])
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected += 1
            continue
        sample = extract_kit_family_sample(
            image,
            tracklet_id=str(row["tracklet_id"]),
            detection_id=str(row["detection_id"]),
            frame_index=int(row["frame_index"]),
            timestamp_s=float(row["timestamp_s"]),
            crop_ref=str(row["crop_ref"]),
            quality=float(row["quality"]),
            config=config,
        )
        if sample is None:
            rejected += 1
            continue
        samples.append(sample)

        current = best_crop_by_tracklet.get(sample.tracklet_id)
        if current is None or sample.quality > current["quality"]:
            best_crop_by_tracklet[sample.tracklet_id] = {
                "quality": sample.quality,
                "crop_ref": sample.crop_ref,
            }

    profiles = build_kit_family_profiles(samples)
    result = cluster_kit_families(profiles, config)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "kit_families" / KIT_FAMILY_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        output_dir / "samples.jsonl",
        (row.to_dict() for row in samples),
    )
    _write_jsonl(
        output_dir / "profiles.jsonl",
        (row.to_dict() for row in profiles),
    )
    _write_jsonl(
        output_dir / "assignments.jsonl",
        (row.to_dict() for row in result.assignments),
    )
    _write_jsonl(
        output_dir / "clusters.jsonl",
        (row.to_dict() for row in result.clusters),
    )

    by_key = defaultdict(list)
    for row in result.assignments:
        key = (
            f"{row.family_type}_{row.family_id:02d}"
            if row.family_id is not None
            else "unknown"
        )
        by_key[key].append(row)

    contact_sheets = {}
    for key, rows in sorted(by_key.items()):
        ranked = sorted(
            rows,
            key=lambda item: (
                -item.usable_for_tracking,
                -item.best_cosine,
                -item.profile_reliability,
                item.tracklet_id,
            ),
        )
        items = []
        for assignment in ranked:
            crop = best_crop_by_tracklet.get(assignment.tracklet_id)
            if crop is None:
                continue
            image = cv2.imread(
                str(root / crop["crop_ref"]),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue
            items.append({
                "tracklet_id": assignment.tracklet_id,
                "best_cosine": assignment.best_cosine,
                "profile_reliability": assignment.profile_reliability,
                "image": image,
            })
            if len(items) >= max(1, args.sheet_limit):
                break

        sheet = _sheet(items)
        if sheet is None:
            contact_sheets[key] = None
            continue
        path = output_dir / f"{key}.jpg"
        if not cv2.imwrite(
            str(path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"cannot write {path}")
        contact_sheets[key] = str(path)

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "sample_count": len(samples),
        "rejected_sample_count": rejected,
        "source_old_kit_samples": str(old_samples_path),
        "contact_sheets": contact_sheets,
    })
    _write_json(output_dir / "summary.json", summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "kit_family_version": KIT_FAMILY_VERSION,
            "samples": "samples.jsonl",
            "profiles": "profiles.jsonl",
            "assignments": "assignments.jsonl",
            "clusters": "clusters.jsonl",
            "summary": "summary.json",
            "primary_family_count": 2,
            "special_family_count": result.special_family_count,
            "identity_decisions_changed": False,
        },
    )

    print(
        "[kit-family-7C.1D] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[kit-family-7C.1D] "
        f"samples={len(samples)} rejected={rejected} "
        f"profiles={len(profiles)}"
    )
    print(
        "[kit-family-7C.1D] "
        f"primary_families=2 "
        f"special_families={result.special_family_count}"
    )
    print(
        "[kit-family-7C.1D] states="
        + json.dumps(
            summary["assignment_state_counts"],
            sort_keys=True,
        )
    )
    print(
        "[kit-family-7C.1D] types="
        + json.dumps(
            summary["assignment_type_counts"],
            sort_keys=True,
        )
    )
    print(
        "[kit-family-7C.1D] primary_similarity_thresholds="
        + json.dumps(summary["primary_similarity_thresholds"])
    )
    print(
        "[kit-family-7C.1D] primary_margin_thresholds="
        + json.dumps(summary["primary_margin_thresholds"])
    )
    print(
        "[kit-family-7C.1D] special_models="
        + json.dumps(summary["special_model_scores"], sort_keys=True)
    )
    print(
        "[kit-family-7C.1D] clusters="
        + json.dumps(summary["clusters"], sort_keys=True)
    )
    for key, path in contact_sheets.items():
        if path:
            print(f"[kit-family-7C.1D] {key}_sheet={path}")
    print(f"[kit-family-7C.1D] artifact={output_dir / 'index.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
