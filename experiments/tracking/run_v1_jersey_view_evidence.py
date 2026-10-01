#!/usr/bin/env python3
"""Run Hito 7C.2A jersey-number view/readiness diagnostics.

Consumes already-saved Tracking V1 appearance crops.  It does NOT reopen the
video, run RF-DETR, run PRTReID or recognize any number.

Outputs:
  diagnostics/jersey_view/jersey-view-v1/
      observations.jsonl
      rejected.jsonl
      selected.jsonl
      rois/<tracklet>/*.jpg
      strongest.jpg
      weakest_selected.jpg
      summary.json
      index.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.appearance.crops import safe_filename_token  # noqa: E402
from tracking.appearance.jersey_view import (  # noqa: E402
    JERSEY_VIEW_VERSION,
    JerseyViewConfig,
    JerseyViewObservation,
    RejectedJerseyView,
    crop_jersey_roi,
    jersey_view_summary,
    score_jersey_view,
    select_jersey_views,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402
from tracking.tracklets.models import Tracklet  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must encode a JSON object")
    return payload


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
                )
                + "\n"
            )
            count += 1
    return count


def _load_tracklets(path: Path) -> tuple[Tracklet, ...]:
    return tuple(
        Tracklet.from_dict(payload)
        for payload in _iter_jsonl(path)
    )


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
    canvas[
        y:y + resized.shape[0],
        x:x + resized.shape[1],
    ] = resized
    return canvas


def _contact_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    columns: int = 8,
    panel_width: int = 170,
    panel_height: int = 180,
) -> np.ndarray | None:
    if not items:
        return None

    rows = (len(items) + columns - 1) // columns
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 45

    for index, item in enumerate(items):
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

        cv2.putText(
            panel,
            str(item["tracklet_id"]),
            (3, image_height + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"q={float(item['readiness_score']):.3f} "
                f"r={int(item['rank_within_tracklet'])}"
            ),
            (3, image_height + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = index // columns
        column = index % columns
        y = row * panel_height
        x = column * panel_width
        sheet[
            y:y + panel_height,
            x:x + panel_width,
        ] = panel

    return sheet


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")

    parser.add_argument(
        "--max-selected-per-tracklet",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--min-selected-score",
        type=float,
        default=0.28,
    )
    parser.add_argument(
        "--min-temporal-spacing-s",
        type=float,
        default=0.35,
    )
    parser.add_argument("--audit-count", type=int, default=48)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    tracklets_path = root / "tracklets" / "tracklets.jsonl"
    crops_index_path = (
        root / "diagnostics" / "appearance_crops" / "index.json"
    )
    quality_path = (
        root / "diagnostics" / "reid_quality" / "quality.json"
    )

    for required in (
        tracklets_path,
        crops_index_path,
        quality_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    tracklets = _load_tracklets(tracklets_path)
    known_tracklet_ids = {
        item.tracklet_id for item in tracklets
    }

    crop_payload = _load_json(crops_index_path)
    crop_records = crop_payload.get("records")
    if not isinstance(crop_records, list):
        raise ValueError("appearance crop index has no records list")

    quality_payload = _load_json(quality_path)
    quality_records = quality_payload.get("records")
    if not isinstance(quality_records, list):
        raise ValueError("reid quality artifact has no records list")

    quality_by_detection: dict[str, Mapping[str, Any]] = {}
    for item in quality_records:
        if isinstance(item, Mapping):
            quality_by_detection[str(item["detection_id"])] = item

    config = JerseyViewConfig(
        max_selected_per_tracklet=args.max_selected_per_tracklet,
        min_selected_score=args.min_selected_score,
        min_temporal_spacing_s=args.min_temporal_spacing_s,
    )

    observations: list[JerseyViewObservation] = []
    rejected: list[RejectedJerseyView] = []
    image_by_sample: dict[str, np.ndarray] = {}

    for raw in crop_records:
        if not isinstance(raw, Mapping):
            raise ValueError("crop record must be an object")
        if raw.get("status") != "saved":
            continue

        tracklet_id = str(raw["tracklet_id"])
        if tracklet_id not in known_tracklet_ids:
            raise ValueError(
                f"crop references unknown tracklet {tracklet_id!r}"
            )

        detection_id = str(raw["detection_id"])
        crop_ref = str(raw["crop_ref"])
        crop_path = root / crop_ref
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"could not read appearance crop: {crop_path}"
            )

        quality_record = quality_by_detection.get(detection_id)
        reid_quality = (
            float(quality_record["reid_quality"])
            if quality_record is not None
            and quality_record.get("reid_quality") is not None
            else None
        )
        selection_quality = float(
            raw.get(
                "selection_quality",
                raw.get("quality", 0.0),
            )
        )

        result = score_jersey_view(
            image,
            tracklet_id=tracklet_id,
            detection_id=detection_id,
            frame_index=int(raw["frame_index"]),
            timestamp_s=float(raw["timestamp_s"]),
            crop_ref=crop_ref,
            selection_quality=selection_quality,
            reid_quality=reid_quality,
            config=config,
        )
        if isinstance(result, JerseyViewObservation):
            observations.append(result)
            image_by_sample[result.sample_id] = image
        else:
            rejected.append(result)

    selected = select_jersey_views(
        observations,
        config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "jersey_view"
        / JERSEY_VIEW_VERSION
    )
    roi_root = output_dir / "rois"

    selected_rows: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []

    for item in selected:
        observation = item.observation
        image = image_by_sample[observation.sample_id]
        roi = crop_jersey_roi(
            image,
            observation.roi_xyxy,
        )

        tracklet_dir = (
            roi_root
            / safe_filename_token(observation.tracklet_id)
        )
        tracklet_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"r{item.rank_within_tracklet:02d}_"
            f"f{observation.frame_index:06d}_"
            f"{safe_filename_token(observation.detection_id)}.jpg"
        )
        path = tracklet_dir / filename

        if not cv2.imwrite(
            str(path),
            roi,
            [int(cv2.IMWRITE_JPEG_QUALITY), 97],
        ):
            raise RuntimeError(f"could not write {path}")

        row = item.to_dict()
        row["number_roi_ref"] = str(
            path.relative_to(root)
        )
        selected_rows.append(row)
        audit_items.append(
            {
                **row,
                "image": roi,
            }
        )

    observations_path = output_dir / "observations.jsonl"
    rejected_path = output_dir / "rejected.jsonl"
    selected_path = output_dir / "selected.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    observation_count = _write_jsonl(
        observations_path,
        (item.to_dict() for item in observations),
    )
    rejected_count = _write_jsonl(
        rejected_path,
        (item.to_dict() for item in rejected),
    )
    selected_count = _write_jsonl(
        selected_path,
        selected_rows,
    )

    rejection_reasons = Counter()
    for item in rejected:
        for reason in item.reasons:
            rejection_reasons[reason] += 1

    summary = jersey_view_summary(
        observations,
        rejected,
        selected,
        total_tracklet_count=len(tracklets),
    )
    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "saved_crop_record_count": sum(
                1
                for item in crop_records
                if isinstance(item, Mapping)
                and item.get("status") == "saved"
            ),
            "rejection_reason_counts": dict(
                sorted(rejection_reasons.items())
            ),
            "config": config.to_dict(),
            "source_crop_index": str(crops_index_path),
            "source_reid_quality": str(quality_path),
        }
    )

    ranked = sorted(
        audit_items,
        key=lambda item: (
            float(item["readiness_score"]),
            str(item["tracklet_id"]),
            int(item["frame_index"]),
        ),
    )
    audit_count = max(1, int(args.audit_count))
    weakest = ranked[:audit_count]
    strongest = list(reversed(ranked[-audit_count:]))

    for filename, items in (
        ("weakest_selected.jpg", weakest),
        ("strongest.jpg", strongest),
    ):
        sheet = _contact_sheet(items)
        if sheet is not None:
            path = output_dir / filename
            output_dir.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), sheet):
                raise RuntimeError(f"could not write {path}")
            summary[filename] = str(path)

    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "jersey_view_version": JERSEY_VIEW_VERSION,
            "observations": observations_path.name,
            "rejected": rejected_path.name,
            "selected": selected_path.name,
            "summary": summary_path.name,
            "roi_root": "rois",
            "observation_count": observation_count,
            "rejected_count": rejected_count,
            "selected_count": selected_count,
            "number_recognition_run": False,
        },
    )

    print(
        "[jersey-view-7C.2A] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[jersey-view-7C.2A] "
        f"saved_crops={summary['saved_crop_record_count']} "
        f"observations={observation_count} "
        f"rejected={rejected_count}"
    )
    print(
        "[jersey-view-7C.2A] "
        f"selected={selected_count} "
        f"selected_tracklets={summary['selected_tracklet_count']}/"
        f"{len(tracklets)} "
        f"coverage={summary['selected_tracklet_coverage']:.6f}"
    )
    print(
        "[jersey-view-7C.2A] "
        f"readiness={json.dumps(summary['readiness_score'], sort_keys=True)}"
    )
    print(
        "[jersey-view-7C.2A] "
        f"selected_readiness="
        f"{json.dumps(summary['selected_readiness_score'], sort_keys=True)}"
    )
    print(
        "[jersey-view-7C.2A] "
        f"selected_per_tracklet="
        f"{json.dumps(summary['selected_per_tracklet'], sort_keys=True)}"
    )
    if summary.get("strongest.jpg"):
        print(
            "[jersey-view-7C.2A] "
            f"strongest={summary['strongest.jpg']}"
        )
    if summary.get("weakest_selected.jpg"):
        print(
            "[jersey-view-7C.2A] "
            f"weakest_selected={summary['weakest_selected.jpg']}"
        )
    print(f"[jersey-view-7C.2A] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
