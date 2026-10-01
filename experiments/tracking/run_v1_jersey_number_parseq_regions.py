#!/usr/bin/env python3
"""Hito 7C.2C-A3: geometric torso regions + generic PARSeq.

Motivation
----------
7C.2C-A (PARSeq over the whole jersey-view ROI) had high apparent coverage but
too many hallucinated digits. 7C.2C-A2 (CRAFT detector -> PARSeq) greatly
improved visual precision, but CRAFT only found text in a small fraction of
otherwise useful soccer views.

A3 is detector-free but does not show PARSeq the whole player. It proposes a
small deterministic family of overlapping upper/mid-torso regions, evaluates
the same preprocessing variants used by the EasyOCR benchmark, collapses
variants inside each region, then requires conservative agreement across
regions before publishing a jersey number.

Important:
* 7C.2A remains unchanged.
* 7C.2B remains unchanged.
* same strongest-first top-N benchmark selection.
* schema-v2 predictions.jsonl remains consumable by existing evidence code.
* rejected/alphanumeric PARSeq text is audit-only and cannot be silently
  reparsed downstream as a jersey number.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
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


@dataclass(frozen=True, slots=True)
class RegionSpec:
    name: str
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if not (
            0.0 <= self.x1 < self.x2 <= 1.0
            and 0.0 <= self.y1 < self.y2 <= 1.0
        ):
            raise ValueError(f"invalid fractional region: {self}")


# Deliberately few and heavily overlapping. These are alternate observations of
# the SAME frame, not independent temporal evidence.
DEFAULT_REGIONS: tuple[RegionSpec, ...] = (
    RegionSpec("torso_wide", 0.05, 0.08, 0.95, 0.62),
    RegionSpec("torso_core", 0.16, 0.10, 0.84, 0.58),
    RegionSpec("torso_upper", 0.08, 0.04, 0.92, 0.50),
    RegionSpec("torso_lower", 0.08, 0.18, 0.92, 0.68),
    RegionSpec("torso_left", 0.00, 0.08, 0.72, 0.62),
    RegionSpec("torso_right", 0.28, 0.08, 1.00, 0.62),
)


def _crop_fractional(
    image: np.ndarray,
    region: RegionSpec,
) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("expected BGR HxWx3 image")

    height, width = image.shape[:2]
    x1 = max(0, min(width - 1, int(math.floor(region.x1 * width))))
    y1 = max(0, min(height - 1, int(math.floor(region.y1 * height))))
    x2 = max(x1 + 1, min(width, int(math.ceil(region.x2 * width))))
    y2 = max(y1 + 1, min(height, int(math.ceil(region.y2 * height))))

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"empty crop for region {region.name}")
    return crop


def _region_candidate(
    variant_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    chosen = _choose_variant_consensus(variant_predictions)
    return {
        **chosen,
        "accepted_within_region": (
            chosen["parsed_number"] is not None
            and (
                int(chosen["variant_support_count"]) >= 2
                or float(chosen["ocr_confidence"]) >= 0.90
            )
        ),
    }


def _choose_region_consensus(
    region_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Collapse overlapping region hypotheses into one conservative view read."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in region_predictions:
        number = row.get("parsed_number")
        if number is None or not bool(row.get("accepted_within_region", False)):
            continue
        grouped[int(number)].append(row)

    if not grouped:
        return {
            "raw_text": None,
            "parsed_number": None,
            "ocr_confidence": 0.0,
            "chosen_variant": None,
            "variant_support_count": 0,
            "variant_consensus_score": 0.0,
            "chosen_region": None,
            "region_support_count": 0,
            "region_margin": None,
            "region_consensus_accepted": False,
            "rejection_reason": "no_region_candidate",
        }

    candidates: list[dict[str, Any]] = []
    for number, rows in grouped.items():
        best = max(
            rows,
            key=lambda row: (
                float(row["ocr_confidence"]),
                int(row["variant_support_count"]),
                str(row["region"]),
            ),
        )
        best_conf = float(best["ocr_confidence"])
        region_support = len(rows)
        total_variant_support = sum(
            int(row["variant_support_count"])
            for row in rows
        )

        # Regions overlap strongly, therefore support is only a modest bonus.
        score = min(
            1.0,
            best_conf
            + 0.045 * math.log1p(max(0, region_support - 1))
            + 0.020 * math.log1p(max(0, total_variant_support - 1)),
        )
        candidates.append(
            {
                "number": int(number),
                "rows": rows,
                "best": best,
                "best_conf": best_conf,
                "region_support": region_support,
                "total_variant_support": total_variant_support,
                "score": score,
            }
        )

    candidates.sort(
        key=lambda item: (
            -float(item["score"]),
            -int(item["region_support"]),
            -int(item["total_variant_support"]),
            -float(item["best_conf"]),
            int(item["number"]),
        )
    )
    winner = candidates[0]
    runner_up_score = (
        float(candidates[1]["score"])
        if len(candidates) > 1
        else 0.0
    )
    margin = float(winner["score"]) - runner_up_score

    # Main path: repeated support from at least two different geometric regions.
    repeated_accept = (
        int(winner["region_support"]) >= 2
        and int(winner["total_variant_support"]) >= 3
        and float(winner["best_conf"]) >= 0.68
        and margin >= 0.06
    )

    # Escape hatch for an exceptionally stable single region. This is useful
    # for side views where only one proposal actually contains the number.
    best_row = winner["best"]
    exceptional_single_accept = (
        int(winner["region_support"]) == 1
        and int(best_row["variant_support_count"]) >= 4
        and float(best_row["ocr_confidence"]) >= 0.96
        and float(best_row["variant_consensus_score"]) >= 0.98
        and margin >= 0.10
    )

    accepted = repeated_accept or exceptional_single_accept
    if not accepted:
        return {
            "raw_text": None,
            "parsed_number": None,
            "ocr_confidence": 0.0,
            "chosen_variant": None,
            "variant_support_count": 0,
            "variant_consensus_score": float(winner["score"]),
            "chosen_region": None,
            "region_support_count": int(winner["region_support"]),
            "region_margin": margin,
            "region_consensus_accepted": False,
            "rejection_reason": "insufficient_region_consensus",
            "rejected_best_number": int(winner["number"]),
            "rejected_best_confidence": float(winner["best_conf"]),
        }

    # Publish only exact numeric raw text. Invalid raw recognizer outputs stay
    # inside region_predictions for audit and can never be reparsed downstream.
    number = int(winner["number"])
    return {
        "raw_text": str(number),
        "parsed_number": number,
        "ocr_confidence": float(winner["best_conf"]),
        "chosen_variant": str(best_row.get("chosen_variant") or ""),
        "variant_support_count": int(winner["total_variant_support"]),
        "variant_consensus_score": float(winner["score"]),
        "chosen_region": str(best_row["region"]),
        "region_support_count": int(winner["region_support"]),
        "region_margin": margin,
        "region_consensus_accepted": True,
        "rejection_reason": None,
    }


def _prediction_row(
    source: Mapping[str, Any],
    *,
    selection: str,
    preprocess_profile: str,
    chosen: Mapping[str, Any],
    flat_variant_predictions: Sequence[Mapping[str, Any]],
    region_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "backend": "parseq-generic-regions",
        "selection": selection,
        "preprocess_profile": preprocess_profile,
        "tracklet_id": str(source["tracklet_id"]),
        "detection_id": str(source["detection_id"]),
        "sample_id": str(source["sample_id"]),
        "frame_index": int(source["frame_index"]),
        "timestamp_s": float(source["timestamp_s"]),
        "number_roi_ref": str(source["number_roi_ref"]),
        "readiness_score": float(source["readiness_score"]),
        "rank_within_tracklet": int(source["rank_within_tracklet"]),
        **dict(chosen),
        "variant_predictions": [
            dict(item) for item in flat_variant_predictions
        ],
        "region_predictions": [
            dict(item) for item in region_predictions
        ],
    }


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--model",
        choices=("parseq", "parseq_tiny"),
        default="parseq",
    )
    parser.add_argument(
        "--selection",
        choices=("input", "strongest"),
        default="strongest",
    )
    parser.add_argument(
        "--preprocess-profile",
        choices=("none", "jersey"),
        default="jersey",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--audit-count", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--audit-dir")
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
        raise FileNotFoundError(selected_path)

    source_rows = list(_iter_jsonl(selected_path))
    selected_rows = _select_rows(
        source_rows,
        selection=args.selection,
        limit=args.limit,
    )

    backend = PARSeqBackend(
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
            / "parseq-generic-regions"
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
    region_candidate_views = 0

    print(
        "[jersey-regions-7C.2C-A3] "
        f"model={args.model} device={backend.device} "
        f"views={len(selected_rows)} regions={len(DEFAULT_REGIONS)}"
    )

    for index, row in enumerate(selected_rows, start=1):
        roi_ref = str(row["number_roi_ref"])
        image_path = root / roi_ref
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"could not read jersey ROI: {image_path}"
            )

        batched_images: list[np.ndarray] = []
        descriptors: list[tuple[str, str]] = []

        for region in DEFAULT_REGIONS:
            crop = _crop_fractional(image, region)
            for variant_name, variant_image in _build_preprocess_variants(
                crop,
                profile=args.preprocess_profile,
            ):
                descriptors.append((region.name, variant_name))
                batched_images.append(variant_image)

        results = backend.predict_many(batched_images)
        if len(results) != len(descriptors):
            raise RuntimeError(
                "PARSeq result count does not match region-variant input count"
            )

        variants_by_region: dict[str, list[dict[str, Any]]] = defaultdict(list)
        flat_variant_predictions: list[dict[str, Any]] = []

        for (
            (region_name, variant_name),
            (raw_text, parsed_number, confidence),
        ) in zip(descriptors, results, strict=True):
            item = {
                "variant": variant_name,
                "region": region_name,
                "raw_text": raw_text,
                "parsed_number": parsed_number,
                "ocr_confidence": float(
                    max(0.0, min(1.0, confidence))
                ),
            }
            variants_by_region[region_name].append(item)
            flat_variant_predictions.append(
                {
                    **item,
                    # Keep variant names globally unique for audit.
                    "variant": f"{region_name}/{variant_name}",
                }
            )

        region_predictions: list[dict[str, Any]] = []
        for region in DEFAULT_REGIONS:
            candidate = _region_candidate(
                variants_by_region[region.name]
            )
            region_predictions.append(
                {
                    "region": region.name,
                    **candidate,
                    "variant_predictions": [
                        dict(item)
                        for item in variants_by_region[region.name]
                    ],
                }
            )

        if any(
            bool(item["accepted_within_region"])
            for item in region_predictions
        ):
            region_candidate_views += 1

        chosen = _choose_region_consensus(region_predictions)
        if chosen["parsed_number"] is not None:
            recognized += 1

        prediction = _prediction_row(
            row,
            selection=args.selection,
            preprocess_profile=args.preprocess_profile,
            chosen=chosen,
            flat_variant_predictions=flat_variant_predictions,
            region_predictions=region_predictions,
        )
        predictions.append(prediction)
        audit_items.append(
            {
                **prediction,
                "image": image,
            }
        )

        if index % 25 == 0 or index == len(selected_rows):
            print(
                "[jersey-regions-7C.2C-A3] "
                f"processed={index}/{len(selected_rows)} "
                f"region_candidate_views={region_candidate_views} "
                f"recognized={recognized}"
            )

    _write_jsonl(output_path, predictions)

    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_count = max(1, int(args.audit_count))

    recognized_items = [
        item for item in audit_items
        if item["parsed_number"] is not None
    ]
    unknown_items = [
        item for item in audit_items
        if item["parsed_number"] is None
    ]

    recognized_items.sort(
        key=lambda item: (
            -float(item["variant_consensus_score"]),
            -float(item["ocr_confidence"]),
            -int(item.get("region_support_count", 0)),
            -float(item["readiness_score"]),
        )
    )
    unknown_items.sort(
        key=lambda item: (
            -float(item.get("variant_consensus_score", 0.0)),
            -float(item["readiness_score"]),
            str(item["tracklet_id"]),
        )
    )

    recognized_sheet = _contact_sheet(
        recognized_items[:audit_count]
    )
    unknown_sheet = _contact_sheet(
        unknown_items[:audit_count]
    )

    recognized_path = audit_dir / "recognized_contact_sheet.jpg"
    unknown_path = audit_dir / "unknown_contact_sheet.jpg"

    if recognized_sheet is not None:
        if not cv2.imwrite(str(recognized_path), recognized_sheet):
            raise RuntimeError(f"could not write {recognized_path}")
    if unknown_sheet is not None:
        if not cv2.imwrite(str(unknown_path), unknown_sheet):
            raise RuntimeError(f"could not write {unknown_path}")

    view_count = len(predictions)
    fraction = recognized / view_count if view_count else 0.0
    candidate_fraction = (
        region_candidate_views / view_count
        if view_count
        else 0.0
    )

    summary = {
        "backend": "parseq-generic-regions",
        "model": args.model,
        "selection": args.selection,
        "preprocess_profile": args.preprocess_profile,
        "views": view_count,
        "region_candidate_views": region_candidate_views,
        "region_candidate_fraction": candidate_fraction,
        "recognized": recognized,
        "recognized_fraction": fraction,
        "region_count": len(DEFAULT_REGIONS),
    }
    summary_path = audit_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    print(
        "[jersey-regions-7C.2C-A3] "
        f"backend=parseq-generic-regions "
        f"views={view_count} "
        f"region_candidate_views={region_candidate_views} "
        f"candidate_fraction={candidate_fraction:.6f} "
        f"recognized={recognized} "
        f"fraction={fraction:.6f}"
    )
    print(f"[jersey-regions-7C.2C-A3] artifact={output_path}")
    print(f"[jersey-regions-7C.2C-A3] summary={summary_path}")
    if recognized_sheet is not None:
        print(
            "[jersey-regions-7C.2C-A3] "
            f"recognized_sheet={recognized_path}"
        )
    if unknown_sheet is not None:
        print(
            "[jersey-regions-7C.2C-A3] "
            f"unknown_sheet={unknown_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
