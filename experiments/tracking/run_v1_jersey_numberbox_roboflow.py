#!/usr/bin/env python3
"""7C.2D-E1: benchmark a public Roboflow jersey-number-box detector on Veo crops.

Research/benchmark only. This stage detects whether a plausible jersey-number
region exists. It does NOT recognize the number and does NOT create identity
evidence.

Default public model:
    football-shirt-number-tlgel/1
    class: numberBox

Input:
    7C.2A selected jersey views.

Output:
    predictions.jsonl
    summary.json
    detections_contact_sheet.jpg
    no_detections_contact_sheet.jpg
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.tracking.run_v1_jersey_number_ocr import (  # noqa: E402
    _iter_jsonl,
    _select_rows,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


DEFAULT_MODEL_ID = "football-shirt-number-tlgel/1"
DEFAULT_CLASS = "numberBox"


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


def _prediction_box(
    raw: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
) -> dict[str, Any] | None:
    """Normalize a Roboflow center-x/y box and clamp it to the image."""
    try:
        x = float(raw["x"])
        y = float(raw["y"])
        width = float(raw["width"])
        height = float(raw["height"])
        confidence = float(raw["confidence"])
    except (KeyError, TypeError, ValueError):
        return None

    if width <= 0.0 or height <= 0.0:
        return None

    x1 = max(0.0, x - width / 2.0)
    y1 = max(0.0, y - height / 2.0)
    x2 = min(float(image_width), x + width / 2.0)
    y2 = min(float(image_height), y + height / 2.0)

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "class": str(raw.get("class", "")),
        "class_id": raw.get("class_id"),
        "confidence": max(0.0, min(1.0, confidence)),
        "xyxy": [x1, y1, x2, y2],
        "center_xywh": [x, y, width, height],
    }


def _extract_boxes(
    result: Mapping[str, Any],
    *,
    image_width: int,
    image_height: int,
    class_name: str,
) -> list[dict[str, Any]]:
    raw_predictions = result.get("predictions", [])
    if not isinstance(raw_predictions, list):
        return []

    boxes = []
    for raw in raw_predictions:
        if not isinstance(raw, Mapping):
            continue
        if str(raw.get("class", "")) != class_name:
            continue
        box = _prediction_box(
            raw,
            image_width=image_width,
            image_height=image_height,
        )
        if box is not None:
            boxes.append(box)

    boxes.sort(
        key=lambda item: float(item["confidence"]),
        reverse=True,
    )
    return boxes


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


def _draw_boxes(
    image: np.ndarray,
    boxes: Sequence[Mapping[str, Any]],
) -> np.ndarray:
    result = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = [
            int(round(value))
            for value in box["xyxy"]
        ]
        confidence = float(box["confidence"])

        cv2.rectangle(
            result,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )
        label = f"numberBox {confidence:.2f}"
        text_y = max(14, y1 - 5)
        cv2.putText(
            result,
            label,
            (max(2, x1), text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def _contact_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    columns: int = 5,
    panel_width: int = 250,
    panel_height: int = 265,
) -> np.ndarray | None:
    if not items:
        return None

    rows = (len(items) + columns - 1) // columns
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 62

    for index, item in enumerate(items):
        panel = np.zeros(
            (panel_height, panel_width, 3),
            dtype=np.uint8,
        )
        shown = _draw_boxes(
            item["image"],
            item["boxes"],
        )
        panel[:image_height] = _fit_panel(
            shown,
            width=panel_width,
            height=image_height,
        )

        boxes = item["boxes"]
        best_conf = (
            float(boxes[0]["confidence"])
            if boxes
            else 0.0
        )

        lines = (
            f"{item['tracklet_id']} q={item['readiness_score']:.3f}",
            (
                f"boxes={len(boxes)} "
                f"best={best_conf:.3f}"
            ),
            (
                f"rank={item['rank_within_tracklet']} "
                f"frame={item['frame_index']}"
            ),
        )

        for line_index, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (4, image_height + 17 + line_index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
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


def _create_client(
    *,
    api_key: str,
    confidence: float,
    max_batch_size: int,
    class_name: str,
):
    try:
        from inference_sdk import (
            InferenceConfiguration,
            InferenceHTTPClient,
        )
    except ImportError as exc:
        raise RuntimeError(
            "inference-sdk is not installed. "
            "Install it in the jersey environment with: "
            ".venv-jersey/bin/python -m pip install inference-sdk"
        ) from exc

    client = InferenceHTTPClient(
        api_url="https://serverless.roboflow.com",
        api_key=api_key,
    )
    return client.configure(
        InferenceConfiguration(
            api_key_transport="header",
            confidence_threshold=float(confidence),
            class_filter=[class_name],
            max_batch_size=max(1, int(max_batch_size)),
            max_concurrent_requests=4,
        )
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")

    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
    )
    parser.add_argument(
        "--class-name",
        default=DEFAULT_CLASS,
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.10,
        help=(
            "Low audit threshold on purpose. We want to see detector "
            "behavior before choosing a production threshold."
        ),
    )
    parser.add_argument(
        "--selection",
        choices=("input", "strongest"),
        default="strongest",
    )
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--audit-count", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--audit-dir")
    parser.add_argument(
        "--api-key-env",
        default="ROBOFLOW_API_KEY",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not (0.0 <= args.confidence <= 1.0):
        raise ValueError("--confidence must be between 0 and 1")

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise RuntimeError(
            f"Roboflow API key not found in ${args.api_key_env}. "
            "Export it locally; do not put it in source code."
        )

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

    rows = _select_rows(
        list(_iter_jsonl(selected_path)),
        selection=args.selection,
        limit=args.limit,
    )

    loaded: list[dict[str, Any]] = []
    for row in rows:
        image_path = root / str(row["number_roi_ref"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)

        loaded.append(
            {
                "row": row,
                "image": image,
                "image_path": image_path,
            }
        )

    client = _create_client(
        api_key=api_key,
        confidence=args.confidence,
        max_batch_size=args.batch_size,
        class_name=args.class_name,
    )

    predictions: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []

    total_boxes = 0
    views_with_box = 0

    for start in range(0, len(loaded), args.batch_size):
        batch = loaded[start:start + args.batch_size]
        batch_paths = [
            str(item["image_path"])
            for item in batch
        ]

        result = client.infer(
            batch_paths,
            model_id=args.model_id,
        )
        if isinstance(result, Mapping):
            results = [result]
        elif isinstance(result, list):
            results = result
        else:
            raise RuntimeError(
                f"unexpected Roboflow response type: {type(result)!r}"
            )

        if len(results) != len(batch):
            raise RuntimeError(
                "Roboflow batch response length does not match input length"
            )

        for item, response in zip(batch, results, strict=True):
            if not isinstance(response, Mapping):
                raise RuntimeError(
                    "Roboflow prediction is not a JSON object"
                )

            image = item["image"]
            row = item["row"]
            boxes = _extract_boxes(
                response,
                image_width=image.shape[1],
                image_height=image.shape[0],
                class_name=args.class_name,
            )

            if boxes:
                views_with_box += 1
                total_boxes += len(boxes)

            best_confidence = (
                float(boxes[0]["confidence"])
                if boxes
                else None
            )

            prediction = {
                "schema_version": 1,
                "backend": "roboflow-serverless",
                "model_id": args.model_id,
                "class_name": args.class_name,
                "confidence_threshold": float(args.confidence),
                "tracklet_id": str(row["tracklet_id"]),
                "detection_id": str(row["detection_id"]),
                "sample_id": str(row["sample_id"]),
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["timestamp_s"]),
                "number_roi_ref": str(row["number_roi_ref"]),
                "readiness_score": float(row["readiness_score"]),
                "rank_within_tracklet": int(row["rank_within_tracklet"]),
                "numberbox_detected": bool(boxes),
                "numberbox_count": len(boxes),
                "best_confidence": best_confidence,
                "boxes": boxes,
            }
            predictions.append(prediction)
            audit_items.append(
                {
                    **prediction,
                    "image": image,
                }
            )

        print(
            "[numberbox-E1] "
            f"processed={min(start + len(batch), len(loaded))}/"
            f"{len(loaded)} "
            f"views_with_box={views_with_box} "
            f"boxes={total_boxes}"
        )

    output_path = (
        Path(args.output)
        if args.output
        else (
            root
            / "diagnostics"
            / "jersey_numberbox"
            / "roboflow-football-shirt-number"
            / "predictions.jsonl"
        )
    )
    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else output_path.parent
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(output_path, predictions)

    detected = [
        item
        for item in audit_items
        if item["numberbox_detected"]
    ]
    missing = [
        item
        for item in audit_items
        if not item["numberbox_detected"]
    ]

    detected.sort(
        key=lambda item: (
            -float(item["best_confidence"] or 0.0),
            -float(item["readiness_score"]),
        )
    )
    missing.sort(
        key=lambda item: (
            -float(item["readiness_score"]),
            str(item["tracklet_id"]),
        )
    )

    audit_count = max(1, int(args.audit_count))
    detected_sheet = _contact_sheet(
        detected[:audit_count]
    )
    missing_sheet = _contact_sheet(
        missing[:audit_count]
    )

    detected_path = (
        audit_dir / "detections_contact_sheet.jpg"
    )
    missing_path = (
        audit_dir / "no_detections_contact_sheet.jpg"
    )

    if detected_sheet is not None:
        cv2.imwrite(str(detected_path), detected_sheet)
    if missing_sheet is not None:
        cv2.imwrite(str(missing_path), missing_sheet)

    summary = {
        "backend": "roboflow-serverless",
        "model_id": args.model_id,
        "class_name": args.class_name,
        "selection": args.selection,
        "confidence_threshold": float(args.confidence),
        "views": len(predictions),
        "views_with_numberbox": views_with_box,
        "view_detection_fraction": (
            views_with_box / len(predictions)
            if predictions
            else 0.0
        ),
        "total_numberboxes": total_boxes,
        "mean_boxes_per_detected_view": (
            total_boxes / views_with_box
            if views_with_box
            else 0.0
        ),
        "best_confidence": {
            "min": (
                min(
                    float(item["best_confidence"])
                    for item in predictions
                    if item["best_confidence"] is not None
                )
                if views_with_box
                else None
            ),
            "max": (
                max(
                    float(item["best_confidence"])
                    for item in predictions
                    if item["best_confidence"] is not None
                )
                if views_with_box
                else None
            ),
        },
    }
    summary_path = audit_dir / "summary.json"
    _write_json(summary_path, summary)

    print(
        "[numberbox-E1] "
        f"views={len(predictions)} "
        f"views_with_numberbox={views_with_box} "
        f"fraction={summary['view_detection_fraction']:.6f} "
        f"boxes={total_boxes}"
    )
    print(f"[numberbox-E1] artifact={output_path}")
    print(f"[numberbox-E1] summary={summary_path}")
    if detected_sheet is not None:
        print(f"[numberbox-E1] detections_sheet={detected_path}")
    if missing_sheet is not None:
        print(f"[numberbox-E1] no_detections_sheet={missing_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
