#!/usr/bin/env python3
"""Hito 7C.2C-A2: EasyOCR text detector + generic PARSeq recognizer.

The experiment preserves the existing jersey-number benchmark contract:

* same Hito 7C.2A selected views;
* same strongest-first ordering;
* same preprocessing variants;
* same within-view variant consensus;
* same schema-v2 predictions.jsonl;
* same contact-sheet audit.

The only change over 7C.2C-A is that PARSeq no longer sees the complete
player/torso ROI. EasyOCR's detector first proposes text regions and PARSeq
recognizes only those tight crops.

EasyOCR recognition is NOT used here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.tracking.run_v1_jersey_number_ocr import (  # noqa: E402
    _build_preprocess_variants,
    _choose_variant_consensus,
    _contact_sheet,
    _iter_jsonl,
    _select_rows,
    _write_jsonl,
)
from experiments.tracking.run_v1_jersey_number_parseq import (  # noqa: E402
    PARSeqBackend,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _horizontal_crop(
    image: np.ndarray,
    box: Sequence[float],
) -> np.ndarray | None:
    if len(box) != 4:
        return None

    x1, x2, y1, y2 = (
        int(round(float(value)))
        for value in box
    )

    height, width = image.shape[:2]
    x1 = max(0, min(width, x1))
    x2 = max(0, min(width, x2))
    y1 = max(0, min(height, y1))
    y2 = max(0, min(height, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop


def _four_point_crop(
    image: np.ndarray,
    points: Sequence[Sequence[float]],
) -> np.ndarray | None:
    """Perspective-normalize one EasyOCR free-form quadrilateral."""

    if len(points) != 4:
        return None

    pts = np.asarray(
        points,
        dtype=np.float32,
    )
    if pts.shape != (4, 2):
        return None

    # EasyOCR's free boxes are ordered:
    # top-left, top-right, bottom-right, bottom-left.
    tl, tr, br, bl = pts

    width_top = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    target_width = int(
        round(max(width_top, width_bottom))
    )

    height_left = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)
    target_height = int(
        round(max(height_left, height_right))
    )

    if target_width < 1 or target_height < 1:
        return None

    destination = np.asarray(
        [
            [0.0, 0.0],
            [target_width - 1.0, 0.0],
            [
                target_width - 1.0,
                target_height - 1.0,
            ],
            [0.0, target_height - 1.0],
        ],
        dtype=np.float32,
    )

    matrix = cv2.getPerspectiveTransform(
        pts,
        destination,
    )
    crop = cv2.warpPerspective(
        image,
        matrix,
        (
            target_width,
            target_height,
        ),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if crop.size == 0:
        return None
    return crop


def _detected_crops(
    image: np.ndarray,
    *,
    horizontal: Sequence[Sequence[float]],
    free: Sequence[Sequence[Sequence[float]]],
) -> list[np.ndarray]:
    crops: list[np.ndarray] = []

    for box in horizontal:
        crop = _horizontal_crop(
            image,
            box,
        )
        if crop is not None:
            crops.append(crop)

    for points in free:
        crop = _four_point_crop(
            image,
            points,
        )
        if crop is not None:
            crops.append(crop)

    return crops


def _best_parseq_candidate(
    results: Sequence[
        tuple[str | None, int | None, float]
    ],
) -> tuple[
    str | None,
    int | None,
    float,
    int | None,
]:
    """Choose the strongest valid jersey candidate from detected text crops."""

    if not results:
        return None, None, 0.0, None

    valid = [
        (
            index,
            raw_text,
            parsed_number,
            float(confidence),
        )
        for index, (
            raw_text,
            parsed_number,
            confidence,
        ) in enumerate(results)
        if parsed_number is not None
    ]

    if valid:
        valid.sort(
            key=lambda item: (
                -item[3],
                int(item[2]),
                item[0],
            )
        )
        index, raw_text, number, confidence = valid[0]
        return (
            raw_text,
            int(number),
            confidence,
            int(index),
        )

    # Preserve the strongest raw PARSeq output for diagnostics even when
    # strict 1-2 digit parsing rejects every candidate.
    strongest = max(
        enumerate(results),
        key=lambda item: (
            float(item[1][2]),
            -item[0],
        ),
    )
    index, (
        raw_text,
        _,
        confidence,
    ) = strongest

    return (
        raw_text,
        None,
        float(confidence),
        int(index),
    )


class EasyOCRTextDetector:
    """CRAFT detector only. EasyOCR recognizer is deliberately disabled."""

    name = "easyocr-craft"

    def __init__(
        self,
        *,
        gpu: bool,
    ) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "EasyOCR is not installed in .venv-jersey."
            ) from exc

        self.reader = easyocr.Reader(
            ["en"],
            gpu=gpu,
            detector=True,
            recognizer=False,
            verbose=False,
        )

    def detect(
        self,
        image: np.ndarray,
    ) -> tuple[
        Sequence[Sequence[float]],
        Sequence[Sequence[Sequence[float]]],
    ]:
        horizontal_agg, free_agg = (
            self.reader.detect(
                image,
                min_size=20,
                text_threshold=0.5,
                low_text=0.4,
                link_threshold=0.3,
                canvas_size=2560,
                mag_ratio=1.0,
                slope_ths=0.1,
                ycenter_ths=0.5,
                height_ths=0.5,
                width_ths=0.5,
                add_margin=0.1,
                threshold=0.2,
                bbox_min_score=0.2,
                bbox_min_size=3,
                max_candidates=0,
            )
        )

        horizontal = (
            horizontal_agg[0]
            if horizontal_agg
            else []
        )
        free = (
            free_agg[0]
            if free_agg
            else []
        )
        return horizontal, free


def _prediction_row(
    source: Mapping[str, Any],
    *,
    selection: str,
    preprocess_profile: str,
    chosen: Mapping[str, Any],
    variant_predictions: Sequence[
        Mapping[str, Any]
    ],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "backend": (
            "easyocr-craft+parseq-generic"
        ),
        "selection": selection,
        "preprocess_profile": (
            preprocess_profile
        ),
        "tracklet_id": str(
            source["tracklet_id"]
        ),
        "detection_id": str(
            source["detection_id"]
        ),
        "sample_id": str(
            source["sample_id"]
        ),
        "frame_index": int(
            source["frame_index"]
        ),
        "timestamp_s": float(
            source["timestamp_s"]
        ),
        "number_roi_ref": str(
            source["number_roi_ref"]
        ),
        "readiness_score": float(
            source["readiness_score"]
        ),
        "rank_within_tracklet": int(
            source["rank_within_tracklet"]
        ),
        **dict(chosen),
        "variant_predictions": [
            dict(item)
            for item in variant_predictions
        ],
    }


def parse_args(
    argv=None,
) -> argparse.Namespace:
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
        "--device",
        default="auto",
    )
    parser.add_argument(
        "--model",
        choices=(
            "parseq",
            "parseq_tiny",
        ),
        default="parseq",
    )
    parser.add_argument(
        "--selection",
        choices=(
            "input",
            "strongest",
        ),
        default="strongest",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=(
            "none",
            "jersey",
        ),
        default="jersey",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
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
    parser.add_argument(
        "--detector-cpu",
        action="store_true",
        help=(
            "Run EasyOCR CRAFT detector on CPU. "
            "PARSeq device is controlled independently "
            "with --device."
        ),
    )
    return parser.parse_args(argv)


def main(
    argv=None,
) -> int:
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

    detector = EasyOCRTextDetector(
        gpu=not args.detector_cpu,
    )
    recognizer = PARSeqBackend(
        device=args.device,
        model_name=args.model,
    )

    output_path = (
        Path(args.output)
        if args.output
        else (
            root
            / "diagnostics"
            / "jersey_number_ocr"
            / "easyocr-craft+parseq-generic"
            / "predictions.jsonl"
        )
    )

    audit_dir = (
        Path(args.audit_dir)
        if args.audit_dir
        else output_path.parent
    )

    predictions: list[
        dict[str, Any]
    ] = []
    audit_items: list[
        dict[str, Any]
    ] = []

    recognized = 0
    detector_hits = 0
    detected_crop_count = 0

    print(
        "[jersey-hybrid-7C.2C-A2] "
        f"model={args.model} "
        f"parseq_device={recognizer.device} "
        f"views={len(selected_rows)}"
    )

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
                f"could not read jersey ROI: "
                f"{image_path}"
            )

        variants = (
            _build_preprocess_variants(
                image,
                profile=(
                    args.preprocess_profile
                ),
            )
        )

        variant_predictions: list[
            dict[str, Any]
        ] = []

        view_has_detection = False

        for (
            variant_name,
            variant_image,
        ) in variants:
            horizontal, free = (
                detector.detect(
                    variant_image
                )
            )

            crops = _detected_crops(
                variant_image,
                horizontal=horizontal,
                free=free,
            )

            if crops:
                view_has_detection = True

            detected_crop_count += len(
                crops
            )

            if crops:
                results = (
                    recognizer.predict_many(
                        crops
                    )
                )
            else:
                results = []

            (
                raw_text,
                parsed_number,
                confidence,
                chosen_crop_index,
            ) = _best_parseq_candidate(
                results
            )

            variant_predictions.append(
                {
                    "variant": (
                        variant_name
                    ),
                    "raw_text": raw_text,
                    "parsed_number": (
                        parsed_number
                    ),
                    "ocr_confidence": (
                        float(
                            max(
                                0.0,
                                min(
                                    1.0,
                                    confidence,
                                ),
                            )
                        )
                    ),
                    "detected_text_count": (
                        len(crops)
                    ),
                    "chosen_crop_index": (
                        chosen_crop_index
                    ),
                }
            )

        if view_has_detection:
            detector_hits += 1

        chosen = (
            _choose_variant_consensus(
                variant_predictions
            )
        )

        if (
            chosen["parsed_number"]
            is not None
        ):
            recognized += 1

        prediction = _prediction_row(
            row,
            selection=args.selection,
            preprocess_profile=(
                args.preprocess_profile
            ),
            chosen=chosen,
            variant_predictions=(
                variant_predictions
            ),
        )
        prediction[
            "detector_found_text"
        ] = bool(view_has_detection)
        prediction[
            "detected_crop_count"
        ] = int(
            sum(
                int(
                    item[
                        "detected_text_count"
                    ]
                )
                for item
                in variant_predictions
            )
        )

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
            or index
            == len(selected_rows)
        ):
            print(
                "[jersey-hybrid-7C.2C-A2] "
                f"processed={index}/"
                f"{len(selected_rows)} "
                f"detector_hits="
                f"{detector_hits} "
                f"recognized="
                f"{recognized}"
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
        if (
            item["parsed_number"]
            is not None
        )
    ]
    unknown_items = [
        item
        for item in audit_items
        if (
            item["parsed_number"]
            is None
        )
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

    recognized_sheet = (
        _contact_sheet(
            recognized_items[
                :audit_count
            ]
        )
    )
    unknown_sheet = (
        _contact_sheet(
            unknown_items[
                :audit_count
            ]
        )
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
                f"could not write "
                f"{recognized_path}"
            )

    if unknown_sheet is not None:
        if not cv2.imwrite(
            str(unknown_path),
            unknown_sheet,
        ):
            raise RuntimeError(
                f"could not write "
                f"{unknown_path}"
            )

    view_count = len(predictions)

    recognized_fraction = (
        recognized / view_count
        if view_count
        else 0.0
    )
    detector_fraction = (
        detector_hits / view_count
        if view_count
        else 0.0
    )

    print(
        "[jersey-hybrid-7C.2C-A2] "
        "backend="
        "easyocr-craft+parseq-generic "
        f"selection={args.selection} "
        f"preprocess="
        f"{args.preprocess_profile} "
        f"views={view_count} "
        f"detector_hits="
        f"{detector_hits} "
        f"detector_fraction="
        f"{detector_fraction:.6f} "
        f"recognized={recognized} "
        f"fraction="
        f"{recognized_fraction:.6f}"
    )

    print(
        "[jersey-hybrid-7C.2C-A2] "
        f"detected_crops="
        f"{detected_crop_count}"
    )
    print(
        "[jersey-hybrid-7C.2C-A2] "
        f"artifact={output_path}"
    )

    if recognized_sheet is not None:
        print(
            "[jersey-hybrid-7C.2C-A2] "
            f"recognized_sheet="
            f"{recognized_path}"
        )

    if unknown_sheet is not None:
        print(
            "[jersey-hybrid-7C.2C-A2] "
            f"unknown_sheet="
            f"{unknown_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
