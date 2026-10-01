#!/usr/bin/env python3
"""Benchmark generic pretrained PARSeq on Hito 7C.2A jersey-number ROIs.

Hito 7C.2C-A changes only the recognizer sensor:

* same 7C.2A selected jersey views;
* same global strongest-first selection;
* same deterministic preprocessing variants as the EasyOCR baseline;
* same within-view variant consensus;
* same schema-v2 predictions.jsonl contract;
* same recognized/unknown contact-sheet audit.

PARSeq is used as a generic scene-text recognizer. The domain constraint is
applied conservatively: only outputs that are exactly one or two ASCII digits
are accepted as jersey-number candidates. We do not salvage digits from mixed
alphanumeric strings (e.g. "S9" stays unknown).

Tracklet aggregation remains a separate deterministic stage:
`run_v1_jersey_number_evidence.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
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
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


_STRICT_JERSEY_RE = re.compile(r"^[0-9]{1,2}$")


def _strict_parse_jersey_number(text: Any) -> int | None:
    """Accept only an exact 1-2 digit ASCII string.

    This deliberately differs from the more tolerant downstream parser.
    Generic PARSeq can emit letters, so benchmark-time filtering must not turn
    outputs such as ``S9`` or ``I0`` into apparently correct jersey numbers.
    """

    if text is None:
        return None
    value = str(text).strip()
    if _STRICT_JERSEY_RE.fullmatch(value) is None:
        return None
    number = int(value)
    return number if 0 <= number <= 99 else None


def _sequence_confidence(token_probabilities: Any) -> float:
    """Return PARSeq sequence confidence as product of decoded token probs."""

    try:
        count = int(token_probabilities.numel())
    except (AttributeError, TypeError, ValueError):
        values = np.asarray(token_probabilities, dtype=np.float64).reshape(-1)
        if values.size == 0:
            return 0.0
        return float(np.prod(values))

    if count == 0:
        return 0.0
    return float(token_probabilities.prod().item())


def _resolve_device(requested: str, torch_module: Any) -> str:
    value = str(requested).strip().lower()
    if value == "auto":
        return "cuda" if torch_module.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch_module.cuda.is_available():
        raise RuntimeError(
            f"PARSeq device {requested!r} requested but CUDA is unavailable. "
            "Use --device cpu or --device auto."
        )
    return str(requested)


class PARSeqBackend:
    """Generic pretrained PARSeq recognizer with jersey-safe output parsing."""

    name = "parseq-generic"

    def __init__(
        self,
        *,
        device: str = "auto",
        model_name: str = "parseq",
    ) -> None:
        if model_name not in {"parseq", "parseq_tiny"}:
            raise ValueError(
                "model_name must be 'parseq' or 'parseq_tiny'"
            )

        try:
            import torch
            from PIL import Image
            from torchvision import transforms as T
        except ImportError as exc:
            raise RuntimeError(
                "PARSeq runtime dependencies are missing. Install the "
                "PARSeq inference dependencies in .venv-jersey."
            ) from exc

        self._torch = torch
        self._Image = Image
        self.device = _resolve_device(device, torch)
        self.model_name = model_name

        try:
            self.model = torch.hub.load(
                "baudm/parseq",
                model_name,
                pretrained=True,
                trust_repo=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load pretrained PARSeq from baudm/parseq. "
                "Check network access and install pytorch-lightning, timm, "
                "PyYAML and nltk in .venv-jersey."
            ) from exc

        self.model = self.model.eval().to(self.device)

        # Match the official PARSeq SceneTextDataModule.get_transform():
        # Resize(img_size, bicubic) -> ToTensor -> Normalize(0.5, 0.5).
        img_size = tuple(
            int(value)
            for value in self.model.hparams.img_size
        )
        self.transform = T.Compose(
            [
                T.Resize(
                    img_size,
                    interpolation=T.InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(0.5, 0.5),
            ]
        )

    def _prepare(self, image: np.ndarray):
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                "PARSeq expects a BGR HxWx3 image"
            )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = self._Image.fromarray(rgb)
        return self.transform(pil)

    def predict_many(
        self,
        images: Sequence[np.ndarray],
    ) -> list[tuple[str | None, int | None, float]]:
        """Recognize a small image batch.

        ``max_length=2`` encodes the jersey-domain fact that only one/two
        characters are useful. Mixed alphanumeric strings are still rejected
        by `_strict_parse_jersey_number`.
        """

        if not images:
            return []

        batch = self._torch.stack(
            [self._prepare(image) for image in images],
            dim=0,
        ).to(self.device)

        with self._torch.inference_mode():
            logits = self.model(
                batch,
                max_length=2,
            )
            probabilities = logits.softmax(-1)
            labels, confidences = (
                self.model.tokenizer.decode(
                    probabilities
                )
            )

        results: list[
            tuple[str | None, int | None, float]
        ] = []
        for label, token_probs in zip(
            labels,
            confidences,
        ):
            raw_text = str(label)
            parsed_number = (
                _strict_parse_jersey_number(
                    raw_text
                )
            )
            confidence = (
                _sequence_confidence(
                    token_probs
                )
            )
            results.append(
                (
                    raw_text,
                    parsed_number,
                    float(
                        max(
                            0.0,
                            min(1.0, confidence),
                        )
                    ),
                )
            )
        return results


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
        "--device",
        default="auto",
        help=(
            "PARSeq inference device: auto, cpu, cuda, cuda:0, ..."
        ),
    )
    parser.add_argument(
        "--model",
        choices=("parseq", "parseq_tiny"),
        default="parseq",
        help="Generic pretrained PARSeq model.",
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
    return parser.parse_args(argv)


def _prediction_row(
    source: Mapping[str, Any],
    *,
    selection: str,
    preprocess_profile: str,
    backend_name: str,
    chosen: Mapping[str, Any],
    variant_predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build the schema-v2 row used by the existing EasyOCR runner."""

    return {
        "schema_version": 2,
        "backend": backend_name,
        "selection": selection,
        "preprocess_profile": preprocess_profile,
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

    print(
        "[jersey-parseq-7C.2C-A] "
        f"model={args.model} "
        f"device={backend.device} "
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
                f"could not read jersey ROI: {image_path}"
            )

        variants = _build_preprocess_variants(
            image,
            profile=args.preprocess_profile,
        )
        variant_results = backend.predict_many(
            [variant_image for _, variant_image in variants]
        )

        variant_predictions: list[
            dict[str, Any]
        ] = []
        for (
            (variant_name, _),
            (
                raw_text,
                parsed_number,
                confidence,
            ),
        ) in zip(
            variants,
            variant_results,
            strict=True,
        ):
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

        prediction = _prediction_row(
            row,
            selection=args.selection,
            preprocess_profile=(
                args.preprocess_profile
            ),
            backend_name=backend.name,
            chosen=chosen,
            variant_predictions=(
                variant_predictions
            ),
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
            or index == len(selected_rows)
        ):
            print(
                "[jersey-parseq-7C.2C-A] "
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
                item["variant_consensus_score"]
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
        "[jersey-parseq-7C.2C-A] "
        f"backend={backend.name} "
        f"model={args.model} "
        f"selection={args.selection} "
        f"preprocess={args.preprocess_profile} "
        f"views={len(predictions)} "
        f"recognized={recognized} "
        f"fraction={fraction:.6f}"
    )
    print(
        "[jersey-parseq-7C.2C-A] "
        f"artifact={output_path}"
    )
    if recognized_sheet is not None:
        print(
            "[jersey-parseq-7C.2C-A] "
            f"recognized_sheet={recognized_path}"
        )
    if unknown_sheet is not None:
        print(
            "[jersey-parseq-7C.2C-A] "
            f"unknown_sheet={unknown_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
