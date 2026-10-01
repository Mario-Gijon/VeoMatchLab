#!/usr/bin/env python3
"""7C.2D-E2: Roboflow numberBox -> SoccerNet PARSeq benchmark.

Research-only diagnostic stage.

Consumes E1 numberBox predictions already saved locally, crops each detector
box above a configurable threshold, and sends ONLY those tight regions to the
SoccerNet-finetuned PARSeq reference recognizer.

Important semantics:
* detector confidence is NOT jersey-number identity evidence;
* PARSeq confidence is NOT proof that a number was visually present;
* the diagnostic joint score is NOT an acceptance threshold;
* no tracklet/player identity is assigned here;
* invalid OCR text is never exposed as public raw_text, preventing tolerant
  downstream parsers from resurrecting rejected strings.

The external jersey-number reference model/code is benchmark-only and must not
become a commercial production dependency without separate license review.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


STRICT_NUMBER_RE = re.compile(r"^[0-9]{1,2}$")


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


def _strict_parse_number(text: Any) -> int | None:
    if text is None:
        return None
    value = str(text).strip()
    if STRICT_NUMBER_RE.fullmatch(value) is None:
        return None
    number = int(value)
    if not 1 <= number <= 99:
        return None
    return number


def _flatten_numeric(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        result = float(value)
        return [result] if math.isfinite(result) else []
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_flatten_numeric(item))
        return values
    return []


def _sequence_confidence(value: Any) -> float:
    """Convert PARSeq's scalar/list decoder confidence to one sequence score."""
    values = _flatten_numeric(value)
    if not values:
        return 0.0

    result = 1.0
    for item in values:
        result *= max(0.0, min(1.0, float(item)))
    return max(0.0, min(1.0, result))


def _expanded_xyxy(
    xyxy: Sequence[Any],
    *,
    image_width: int,
    image_height: int,
    pad_x: float,
    pad_y: float,
) -> tuple[int, int, int, int] | None:
    if len(xyxy) != 4:
        return None

    try:
        x1, y1, x2, y2 = [float(value) for value in xyxy]
    except (TypeError, ValueError):
        return None

    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    if x2 <= x1 or y2 <= y1:
        return None

    width = x2 - x1
    height = y2 - y1

    x1 -= width * float(pad_x)
    x2 += width * float(pad_x)
    y1 -= height * float(pad_y)
    y2 += height * float(pad_y)

    ix1 = max(0, min(image_width - 1, int(math.floor(x1))))
    iy1 = max(0, min(image_height - 1, int(math.floor(y1))))
    ix2 = max(ix1 + 1, min(image_width, int(math.ceil(x2))))
    iy2 = max(iy1 + 1, min(image_height, int(math.ceil(y2))))

    if ix2 <= ix1 or iy2 <= iy1:
        return None

    return ix1, iy1, ix2, iy2


def _crop_candidate(
    image: np.ndarray,
    xyxy: Sequence[Any],
    *,
    pad_x: float,
    pad_y: float,
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    bounds = _expanded_xyxy(
        xyxy,
        image_width=image.shape[1],
        image_height=image.shape[0],
        pad_x=pad_x,
        pad_y=pad_y,
    )
    if bounds is None:
        return None

    x1, y1, x2, y2 = bounds
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, bounds


def _safe_token(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value))
    return text.strip("-") or "item"


def _load_str_results(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("PARSeq result file must contain a JSON object")

    result: dict[str, Mapping[str, Any]] = {}
    for filename, row in payload.items():
        if isinstance(row, Mapping):
            result[str(filename)] = row
    return result


def _run_parseq(
    *,
    reference_root: Path,
    checkpoint: Path,
    conda_exe: Path,
    env_name: str,
    str_data_root: Path,
    result_file: Path,
    device: str,
) -> None:
    str_script = reference_root / "str.py"
    if not str_script.is_file():
        raise FileNotFoundError(str_script)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not conda_exe.is_file():
        raise FileNotFoundError(conda_exe)

    command = [
        str(conda_exe),
        "run",
        "-n",
        env_name,
        "python3",
        str(str_script),
        str(checkpoint),
        f"--data_root={str_data_root}",
        "--batch_size=1",
        "--inference",
        f"--result_file={result_file}",
        f"--device={device}",
    ]

    print(
        "[numberbox-parseq-E2] exec="
        + " ".join(command)
    )

    subprocess.run(
        command,
        cwd=str(reference_root),
        env=os.environ.copy(),
        check=True,
    )


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


def _draw_one_box(
    image: np.ndarray,
    bounds: Sequence[int],
) -> np.ndarray:
    result = image.copy()
    x1, y1, x2, y2 = [int(v) for v in bounds]
    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (255, 255, 255),
        2,
    )
    return result


def _candidate_contact_sheet(
    items: Sequence[Mapping[str, Any]],
    *,
    columns: int = 3,
    panel_width: int = 420,
    panel_height: int = 270,
) -> np.ndarray | None:
    if not items:
        return None

    rows = (len(items) + columns - 1) // columns
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )

    visual_height = 205
    left_width = 250
    right_width = panel_width - left_width

    for index, item in enumerate(items):
        panel = np.zeros(
            (panel_height, panel_width, 3),
            dtype=np.uint8,
        )

        original = _draw_one_box(
            item["image"],
            item["expanded_xyxy"],
        )
        panel[:visual_height, :left_width] = _fit_panel(
            original,
            width=left_width,
            height=visual_height,
        )
        panel[:visual_height, left_width:] = _fit_panel(
            item["crop"],
            width=right_width,
            height=visual_height,
        )

        number = item.get("parsed_number")
        ocr_text = (
            "?"
            if number is None
            else str(number)
        )

        lines = (
            (
                f"{item['tracklet_id']} "
                f"box#{item['box_index']}"
            ),
            (
                f"det={float(item['detector_confidence']):.3f} "
                f"OCR={ocr_text} "
                f"ocr_c={float(item['ocr_confidence']):.3f}"
            ),
            (
                f"joint={float(item['diagnostic_joint_score']):.3f} "
                f"raw={item.get('audit_raw_text')!r}"
            ),
        )

        for line_index, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (4, visual_height + 18 + line_index * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")

    parser.add_argument(
        "--numberbox-predictions",
        required=True,
        help="E1 predictions.jsonl, already produced locally.",
    )
    parser.add_argument(
        "--detector-threshold",
        type=float,
        default=0.30,
    )
    parser.add_argument("--pad-x", type=float, default=0.12)
    parser.add_argument("--pad-y", type=float, default=0.18)

    parser.add_argument(
        "--reference-root",
        default="~/Tools/jersey-number-pipeline",
    )
    parser.add_argument(
        "--checkpoint",
        default=(
            "~/Tools/jersey-number-pipeline/models/"
            "parseq_epoch=24-step=2575-val_accuracy=95.6044-"
            "val_NED=96.3255.ckpt"
        ),
    )
    parser.add_argument(
        "--conda-exe",
        default="~/.local/bin/micromamba",
    )
    parser.add_argument("--str-env", default="parseq2")
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--output")
    parser.add_argument("--audit-dir")
    parser.add_argument("--audit-count", type=int, default=100)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not 0.0 <= args.detector_threshold <= 1.0:
        raise ValueError("--detector-threshold must be between 0 and 1")
    if args.pad_x < 0.0 or args.pad_y < 0.0:
        raise ValueError("padding must be >= 0")

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    tracking_root = context.tracking_root

    predictions_path = Path(
        args.numberbox_predictions
    ).expanduser().resolve()
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    reference_root = Path(
        args.reference_root
    ).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    conda_exe = Path(args.conda_exe).expanduser().resolve()

    work_dir = Path(args.work_dir).expanduser().resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    str_imgs = work_dir / "str_data" / "imgs"
    str_imgs.mkdir(parents=True, exist_ok=True)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else work_dir / "candidates.jsonl"
    )
    audit_dir = (
        Path(args.audit_dir).expanduser().resolve()
        if args.audit_dir
        else work_dir
    )
    audit_dir.mkdir(parents=True, exist_ok=True)

    source_rows = list(_iter_jsonl(predictions_path))

    candidates: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    images_by_candidate: dict[str, np.ndarray] = {}
    crops_by_candidate: dict[str, np.ndarray] = {}

    candidate_counter = 0

    for view_index, row in enumerate(source_rows):
        image_path = tracking_root / str(row["number_roi_ref"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)

        raw_boxes = row.get("boxes", [])
        if not isinstance(raw_boxes, list):
            raw_boxes = []

        filtered_boxes = []
        for box_index, box in enumerate(raw_boxes):
            if not isinstance(box, Mapping):
                continue
            detector_confidence = float(
                box.get("confidence", 0.0)
            )
            if detector_confidence < args.detector_threshold:
                continue

            result = _crop_candidate(
                image,
                box.get("xyxy", []),
                pad_x=args.pad_x,
                pad_y=args.pad_y,
            )
            if result is None:
                continue

            crop, expanded_xyxy = result
            candidate_id = f"cand-{candidate_counter:05d}"
            filename = (
                f"{candidate_counter:05d}__"
                f"{_safe_token(row['tracklet_id'])}__"
                f"b{box_index:02d}.jpg"
            )
            crop_path = str_imgs / filename

            if not cv2.imwrite(
                str(crop_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), 98],
            ):
                raise RuntimeError(f"could not write {crop_path}")

            candidate = {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "source_view_index": view_index,
                "box_index": box_index,
                "tracklet_id": str(row["tracklet_id"]),
                "detection_id": str(row["detection_id"]),
                "sample_id": str(row["sample_id"]),
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["timestamp_s"]),
                "number_roi_ref": str(row["number_roi_ref"]),
                "readiness_score": float(
                    row.get("readiness_score", 0.0)
                ),
                "rank_within_tracklet": int(
                    row.get("rank_within_tracklet", 0)
                ),
                "detector_confidence": detector_confidence,
                "detector_xyxy": [
                    float(v) for v in box["xyxy"]
                ],
                "expanded_xyxy": list(expanded_xyxy),
                "str_filename": filename,
            }
            candidates.append(candidate)
            filtered_boxes.append(candidate_id)
            images_by_candidate[candidate_id] = image
            crops_by_candidate[candidate_id] = crop
            candidate_counter += 1

        view_rows.append(
            {
                "schema_version": 1,
                "source_view_index": view_index,
                "tracklet_id": str(row["tracklet_id"]),
                "detection_id": str(row["detection_id"]),
                "sample_id": str(row["sample_id"]),
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["timestamp_s"]),
                "number_roi_ref": str(row["number_roi_ref"]),
                "readiness_score": float(
                    row.get("readiness_score", 0.0)
                ),
                "candidate_ids": filtered_boxes,
                "candidate_count": len(filtered_boxes),
            }
        )

    print(
        "[numberbox-parseq-E2] "
        f"views={len(source_rows)} "
        f"candidate_boxes={len(candidates)} "
        f"detector_threshold={args.detector_threshold:.3f}"
    )

    str_results_path = work_dir / "str_results.json"

    if candidates:
        _run_parseq(
            reference_root=reference_root,
            checkpoint=checkpoint,
            conda_exe=conda_exe,
            env_name=args.str_env,
            str_data_root=work_dir / "str_data",
            result_file=str_results_path,
            device=args.device,
        )
        str_results = _load_str_results(str_results_path)
    else:
        _write_json(str_results_path, {})
        str_results = {}

    enriched_candidates: list[dict[str, Any]] = []
    by_view: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for candidate in candidates:
        filename = str(candidate["str_filename"])
        result = str_results.get(filename, {})

        audit_raw_text = (
            str(result.get("label", "")).strip()
            if result.get("label") is not None
            else ""
        )
        parsed_number = _strict_parse_number(audit_raw_text)
        ocr_confidence_raw = result.get("confidence")
        ocr_confidence = _sequence_confidence(
            ocr_confidence_raw
        )

        # Diagnostic only. Never interpret this product as calibrated
        # probability or identity evidence.
        joint_score = (
            float(candidate["detector_confidence"])
            * ocr_confidence
        )

        enriched = {
            **candidate,
            "ocr_backend": "soccer-reference-parseq",
            "raw_text": (
                audit_raw_text
                if parsed_number is not None
                else None
            ),
            "audit_raw_text": audit_raw_text,
            "parsed_number": parsed_number,
            "ocr_confidence": ocr_confidence,
            "ocr_confidence_raw": ocr_confidence_raw,
            "diagnostic_joint_score": joint_score,
            "research_only_noncommercial_reference": True,
        }
        enriched_candidates.append(enriched)
        by_view[int(candidate["source_view_index"])].append(
            enriched
        )

    enriched_views: list[dict[str, Any]] = []
    conflict_views = 0
    views_with_valid_ocr = 0

    for view in view_rows:
        items = by_view.get(
            int(view["source_view_index"]),
            [],
        )
        valid = [
            item
            for item in items
            if item["parsed_number"] is not None
        ]
        distinct_numbers = sorted(
            {
                int(item["parsed_number"])
                for item in valid
            }
        )
        if valid:
            views_with_valid_ocr += 1
        if len(distinct_numbers) > 1:
            conflict_views += 1

        best = (
            max(
                valid,
                key=lambda item: (
                    float(item["diagnostic_joint_score"]),
                    float(item["detector_confidence"]),
                    float(item["ocr_confidence"]),
                ),
            )
            if valid
            else None
        )

        enriched_views.append(
            {
                **view,
                "valid_ocr_candidate_count": len(valid),
                "distinct_numbers": distinct_numbers,
                "has_conflict": len(distinct_numbers) > 1,
                "diagnostic_best_number": (
                    int(best["parsed_number"])
                    if best is not None
                    else None
                ),
                "diagnostic_best_joint_score": (
                    float(best["diagnostic_joint_score"])
                    if best is not None
                    else None
                ),
                "identity_evidence_created": False,
            }
        )

    _write_jsonl(output_path, enriched_candidates)
    views_path = audit_dir / "views.jsonl"
    _write_jsonl(views_path, enriched_views)

    number_counts = Counter(
        int(item["parsed_number"])
        for item in enriched_candidates
        if item["parsed_number"] is not None
    )

    summary = {
        "schema_version": 1,
        "stage": "7C.2D-E2",
        "research_only_noncommercial_reference": True,
        "source_numberbox_predictions": str(predictions_path),
        "detector_threshold": float(args.detector_threshold),
        "pad_x": float(args.pad_x),
        "pad_y": float(args.pad_y),
        "views": len(source_rows),
        "views_with_candidate_boxes": sum(
            1
            for row in enriched_views
            if row["candidate_count"] > 0
        ),
        "candidate_boxes": len(enriched_candidates),
        "ocr_valid_candidates": sum(
            1
            for item in enriched_candidates
            if item["parsed_number"] is not None
        ),
        "views_with_valid_ocr": views_with_valid_ocr,
        "views_with_conflicting_numbers": conflict_views,
        "predicted_number_counts": {
            str(number): count
            for number, count in sorted(number_counts.items())
        },
        "identity_evidence_created": False,
        "note": (
            "All OCR/joint scores are diagnostic only. "
            "No jersey number is accepted as identity evidence in E2."
        ),
    }
    summary_path = audit_dir / "summary.json"
    _write_json(summary_path, summary)

    audit_items: list[dict[str, Any]] = []
    for item in enriched_candidates:
        candidate_id = str(item["candidate_id"])
        audit_items.append(
            {
                **item,
                "image": images_by_candidate[candidate_id],
                "crop": crops_by_candidate[candidate_id],
            }
        )

    audit_items.sort(
        key=lambda item: (
            -float(item["detector_confidence"]),
            -float(item["ocr_confidence"]),
        )
    )
    audit_count = max(1, int(args.audit_count))
    sheet = _candidate_contact_sheet(
        audit_items[:audit_count]
    )
    sheet_path = audit_dir / "candidates_contact_sheet.jpg"
    if sheet is not None:
        if not cv2.imwrite(
            str(sheet_path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 95],
        ):
            raise RuntimeError(f"could not write {sheet_path}")

    print(
        "[numberbox-parseq-E2] "
        f"candidate_boxes={summary['candidate_boxes']} "
        f"ocr_valid={summary['ocr_valid_candidates']} "
        f"views_with_valid_ocr={summary['views_with_valid_ocr']} "
        f"conflict_views={summary['views_with_conflicting_numbers']}"
    )
    print(f"[numberbox-parseq-E2] artifact={output_path}")
    print(f"[numberbox-parseq-E2] views={views_path}")
    print(f"[numberbox-parseq-E2] summary={summary_path}")
    if sheet is not None:
        print(
            "[numberbox-parseq-E2] sheet="
            f"{sheet_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
