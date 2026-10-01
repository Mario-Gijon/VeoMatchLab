#!/usr/bin/env python3
"""Run OCR over Hito 7C.2A selected jersey-number ROIs.

Revision V1 adds a fair OCR benchmark mode:

* global strongest-first selection by Hito 7C.2A readiness;
* several deterministic image-preprocessing variants per ROI;
* within-view variant consensus so correlated preprocessing variants do NOT
  become independent jersey observations;
* visual contact-sheet audit of recognized and unknown views.

Backends
--------
easyocr:
    Lightweight baseline.

mmocr:
    SoccerNet-style DBNet text detection + SAR text recognition.

This script writes per-view OCR predictions only. Tracklet aggregation remains
a separate deterministic stage (`run_v1_jersey_number_evidence.py`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.appearance.jersey_number import (  # noqa: E402
    parse_jersey_number,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


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


def _best_digit_result(
    results: Sequence[tuple[str, float]],
) -> tuple[str | None, int | None, float]:
    valid = []
    for text, confidence in results:
        number = parse_jersey_number(text)
        if number is None:
            continue
        valid.append(
            (str(text), int(number), float(confidence))
        )
    if not valid:
        return None, None, 0.0

    valid.sort(
        key=lambda item: (
            -item[2],
            len(
                "".join(
                    ch for ch in item[0] if ch.isdigit()
                )
            ),
            item[1],
        )
    )
    return valid[0]


def _select_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    selection: str,
    limit: int | None,
) -> list[Mapping[str, Any]]:
    if selection not in {"input", "strongest"}:
        raise ValueError(f"unsupported selection mode: {selection}")

    if selection == "strongest":
        ordered = sorted(
            rows,
            key=lambda row: (
                -float(row.get("readiness_score", 0.0)),
                int(row.get("rank_within_tracklet", 999)),
                str(row.get("tracklet_id", "")),
                int(row.get("frame_index", 0)),
            ),
        )
    else:
        ordered = list(rows)

    if limit is not None:
        ordered = ordered[: max(0, int(limit))]
    return ordered


def _resize(
    image: np.ndarray,
    scale: int,
) -> np.ndarray:
    return cv2.resize(
        image,
        (
            int(image.shape[1] * scale),
            int(image.shape[0] * scale),
        ),
        interpolation=cv2.INTER_CUBIC,
    )


def _build_preprocess_variants(
    image: np.ndarray,
    *,
    profile: str,
) -> tuple[tuple[str, np.ndarray], ...]:
    """Create deterministic variants.

    Variants are alternate views of ONE observation and must never be treated
    as independent temporal evidence by the later tracklet aggregator.
    """

    if profile == "none":
        return (("original", image),)
    if profile != "jersey":
        raise ValueError(
            f"unsupported preprocessing profile: {profile}"
        )

    x2 = _resize(image, 2)
    x4 = _resize(image, 4)

    gray = cv2.cvtColor(x4, cv2.COLOR_BGR2GRAY)
    clahe_gray = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(4, 4),
    ).apply(gray)
    clahe = cv2.cvtColor(
        clahe_gray,
        cv2.COLOR_GRAY2BGR,
    )

    blurred = cv2.GaussianBlur(
        x4,
        (0, 0),
        sigmaX=1.2,
        sigmaY=1.2,
    )
    sharpened = cv2.addWeighted(
        x4,
        1.7,
        blurred,
        -0.7,
        0.0,
    )

    return (
        ("original", image),
        ("x2", x2),
        ("x4", x4),
        ("x4_clahe", clahe),
        ("x4_sharpen", sharpened),
    )


def _choose_variant_consensus(
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collapse correlated preprocessing variants into ONE view prediction."""

    groups: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        number = prediction.get("parsed_number")
        if number is None:
            continue
        groups[int(number)].append(prediction)

    if not groups:
        return {
            "raw_text": None,
            "parsed_number": None,
            "ocr_confidence": 0.0,
            "chosen_variant": None,
            "variant_support_count": 0,
            "variant_consensus_score": 0.0,
        }

    candidates = []
    for number, items in groups.items():
        best = max(
            items,
            key=lambda item: float(
                item.get("ocr_confidence", 0.0)
            ),
        )
        max_conf = float(best["ocr_confidence"])

        # Variants are strongly correlated, so agreement is only a modest
        # tie-break/bonus.  We do NOT sum confidences as independent evidence.
        support_count = len(items)
        consensus_score = min(
            1.0,
            max_conf
            + 0.06 * math.log1p(max(0, support_count - 1)),
        )
        candidates.append(
            (
                consensus_score,
                support_count,
                max_conf,
                -number,
                number,
                best,
            )
        )

    _, support_count, max_conf, _, number, best = max(candidates)
    return {
        "raw_text": best.get("raw_text"),
        "parsed_number": int(number),
        "ocr_confidence": float(max_conf),
        "chosen_variant": str(best["variant"]),
        "variant_support_count": int(support_count),
        "variant_consensus_score": float(
            max(
                item[0]
                for item in candidates
                if item[4] == number
            )
        ),
    }


class EasyOCRBackend:
    name = "easyocr"

    def __init__(self, *, gpu: bool):
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed. Install it in the isolated "
                "jersey OCR environment."
            ) from exc

        self.reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
        )

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[str | None, int | None, float]:
        raw = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="0123456789",
            text_threshold=0.5,
            link_threshold=0.3,
        )
        candidates: list[tuple[str, float]] = []
        for item in raw:
            if len(item) < 3:
                continue
            candidates.append(
                (str(item[1]), float(item[2]))
            )
        return _best_digit_result(candidates)


class MMOCRBackend:
    name = "mmocr-dbnet-sar"

    def __init__(
        self,
        *,
        device: str,
        batch_size: int,
    ):
        try:
            from mmocr.apis import (
                TextDetInferencer,
                TextRecInferencer,
            )
            from mmocr.utils import (
                bbox2poly,
                crop_img,
                poly2bbox,
            )
        except ImportError as exc:
            raise RuntimeError(
                "MMOCR is not installed. Keep it in an isolated "
                "environment because its dependency stack is version-sensitive."
            ) from exc

        self._bbox2poly = bbox2poly
        self._crop_img = crop_img
        self._poly2bbox = poly2bbox
        self.batch_size = int(batch_size)
        self.detector = TextDetInferencer(
            "dbnet_resnet18_fpnc_1200e_icdar2015",
            device=device,
        )
        self.recognizer = TextRecInferencer(
            "SAR",
            device=device,
        )

    def predict(
        self,
        image: np.ndarray,
    ) -> tuple[str | None, int | None, float]:
        detection = self.detector(
            [image],
            return_datasamples=True,
            batch_size=1,
            progress_bar=False,
        )["predictions"][0]

        rec_inputs = []
        for polygon in detection.pred_instances["polygons"]:
            quad = self._bbox2poly(
                self._poly2bbox(polygon)
            ).tolist()
            crop = self._crop_img(
                image,
                quad,
            )
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                rec_inputs.append(crop)

        if not rec_inputs:
            return None, None, 0.0

        predictions = self.recognizer(
            rec_inputs,
            return_datasamples=True,
            batch_size=self.batch_size,
            progress_bar=False,
        )["predictions"]

        candidates: list[tuple[str, float]] = []
        for prediction in predictions:
            payload = self.recognizer.pred2dict(prediction)
            text = str(payload.get("text", ""))
            scores = payload.get("scores", 0.0)
            if isinstance(
                scores,
                (list, tuple, np.ndarray),
            ):
                confidence = (
                    float(np.mean(scores))
                    if len(scores)
                    else 0.0
                )
            else:
                confidence = float(scores)
            candidates.append(
                (text, confidence)
            )

        return _best_digit_result(candidates)


def _fit_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )
    scale = min(
        width / image.shape[1],
        height / image.shape[0],
    )
    resized = cv2.resize(
        image,
        (
            max(
                1,
                int(round(image.shape[1] * scale)),
            ),
            max(
                1,
                int(round(image.shape[0] * scale)),
            ),
        ),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
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
    panel_width: int = 180,
    panel_height: int = 190,
) -> np.ndarray | None:
    if not items:
        return None

    rows = (
        len(items) + columns - 1
    ) // columns
    sheet = np.zeros(
        (
            rows * panel_height,
            columns * panel_width,
            3,
        ),
        dtype=np.uint8,
    )
    image_height = panel_height - 55

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
            (
                f"{item['tracklet_id']} "
                f"r={item['readiness_score']:.3f}"
            ),
            (3, image_height + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        if item["parsed_number"] is None:
            label = "OCR=?"
        else:
            label = (
                f"OCR={item['parsed_number']} "
                f"c={item['ocr_confidence']:.2f} "
                f"v={item['variant_support_count']}"
            )
        cv2.putText(
            panel,
            label,
            (3, image_height + 31),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            str(item.get("chosen_variant") or ""),
            (3, image_height + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
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
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--match-id",
        required=True,
    )
    parser.add_argument(
        "--clip-id",
        required=True,
    )
    parser.add_argument(
        "--data-root",
        default="data",
    )
    parser.add_argument(
        "--backend",
        choices=("easyocr", "mmocr"),
        default="easyocr",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="MMOCR device, e.g. cuda, cuda:0 or cpu.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for EasyOCR.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--selection",
        choices=("input", "strongest"),
        default="strongest",
        help=(
            "strongest sorts globally by jersey-view readiness "
            "before applying --limit."
        ),
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=("none", "jersey"),
        default="jersey",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit after selection/sorting.",
    )
    parser.add_argument(
        "--audit-count",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--output",
    )
    parser.add_argument(
        "--audit-dir",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
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
        raise FileNotFoundError(
            selected_path
        )

    source_rows = list(
        _iter_jsonl(selected_path)
    )
    selected_rows = _select_rows(
        source_rows,
        selection=args.selection,
        limit=args.limit,
    )

    if args.backend == "easyocr":
        backend = EasyOCRBackend(
            gpu=not args.cpu,
        )
    else:
        backend = MMOCRBackend(
            device=args.device,
            batch_size=args.batch_size,
        )

    output_path = (
        Path(args.output)
        if args.output
        else (
            root
            / "diagnostics"
            / "jersey_number_ocr"
            / backend.name
            / "predictions.jsonl"
        )
    )
    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else output_path.parent
    )

    predictions: list[dict[str, Any]] = []
    audit_items: list[dict[str, Any]] = []
    recognized = 0

    for index, row in enumerate(
        selected_rows,
        start=1,
    ):
        roi_ref = str(
            row["number_roi_ref"]
        )
        image_path = root / roi_ref
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            raise FileNotFoundError(
                f"could not read jersey ROI: {image_path}"
            )

        variant_predictions: list[
            dict[str, Any]
        ] = []
        for variant_name, variant_image in (
            _build_preprocess_variants(
                image,
                profile=args.preprocess_profile,
            )
        ):
            raw_text, parsed_number, confidence = (
                backend.predict(
                    variant_image
                )
            )
            variant_predictions.append(
                {
                    "variant": variant_name,
                    "raw_text": raw_text,
                    "parsed_number": parsed_number,
                    "ocr_confidence": float(
                        max(
                            0.0,
                            min(
                                1.0,
                                confidence,
                            ),
                        )
                    ),
                }
            )

        chosen = _choose_variant_consensus(
            variant_predictions
        )
        if chosen["parsed_number"] is not None:
            recognized += 1

        prediction = {
            "schema_version": 2,
            "backend": backend.name,
            "selection": args.selection,
            "preprocess_profile": args.preprocess_profile,
            "tracklet_id": str(
                row["tracklet_id"]
            ),
            "detection_id": str(
                row["detection_id"]
            ),
            "sample_id": str(
                row["sample_id"]
            ),
            "frame_index": int(
                row["frame_index"]
            ),
            "timestamp_s": float(
                row["timestamp_s"]
            ),
            "number_roi_ref": roi_ref,
            "readiness_score": float(
                row["readiness_score"]
            ),
            "rank_within_tracklet": int(
                row["rank_within_tracklet"]
            ),
            **chosen,
            "variant_predictions": (
                variant_predictions
            ),
        }
        predictions.append(
            prediction
        )
        audit_items.append(
            {
                **prediction,
                "image": image,
            }
        )

        if (
            index % 25 == 0
            or index == len(selected_rows)
        ):
            print(
                "[jersey-ocr-7C.2B-v1] "
                f"processed={index}/"
                f"{len(selected_rows)} "
                f"recognized={recognized}"
            )

    _write_jsonl(
        output_path,
        predictions,
    )

    audit_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    audit_count = max(
        1,
        int(args.audit_count),
    )
    recognized_items = [
        item
        for item in audit_items
        if item["parsed_number"] is not None
    ]
    unknown_items = [
        item
        for item in audit_items
        if item["parsed_number"] is None
    ]

    recognized_items.sort(
        key=lambda item: (
            -float(
                item[
                    "variant_consensus_score"
                ]
            ),
            -float(
                item["ocr_confidence"]
            ),
            -float(
                item["readiness_score"]
            ),
        )
    )
    unknown_items.sort(
        key=lambda item: (
            -float(
                item["readiness_score"]
            ),
            str(
                item["tracklet_id"]
            ),
        )
    )

    recognized_sheet = _contact_sheet(
        recognized_items[:audit_count]
    )
    unknown_sheet = _contact_sheet(
        unknown_items[:audit_count]
    )

    recognized_path = (
        audit_dir
        / "recognized_contact_sheet.jpg"
    )
    unknown_path = (
        audit_dir
        / "unknown_contact_sheet.jpg"
    )

    if recognized_sheet is not None:
        if not cv2.imwrite(
            str(recognized_path),
            recognized_sheet,
        ):
            raise RuntimeError(
                f"could not write {recognized_path}"
            )
    if unknown_sheet is not None:
        if not cv2.imwrite(
            str(unknown_path),
            unknown_sheet,
        ):
            raise RuntimeError(
                f"could not write {unknown_path}"
            )

    fraction = (
        recognized / len(predictions)
        if predictions
        else 0.0
    )
    print(
        "[jersey-ocr-7C.2B-v1] "
        f"backend={backend.name} "
        f"selection={args.selection} "
        f"preprocess={args.preprocess_profile} "
        f"views={len(predictions)} "
        f"recognized={recognized} "
        f"fraction={fraction:.6f}"
    )
    print(
        "[jersey-ocr-7C.2B-v1] "
        f"artifact={output_path}"
    )
    if recognized_sheet is not None:
        print(
            "[jersey-ocr-7C.2B-v1] "
            f"recognized_sheet="
            f"{recognized_path}"
        )
    if unknown_sheet is not None:
        print(
            "[jersey-ocr-7C.2B-v1] "
            f"unknown_sheet={unknown_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
