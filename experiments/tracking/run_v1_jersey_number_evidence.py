#!/usr/bin/env python3
"""Aggregate per-view jersey OCR into conservative tracklet-level evidence."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.appearance.jersey_number import (  # noqa: E402
    JERSEY_NUMBER_VERSION,
    JerseyNumberAggregationConfig,
    aggregate_all_tracklets,
    jersey_number_summary,
    observation_from_prediction_row,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402
from tracking.tracklets.models import Tracklet  # noqa: E402


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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--backend",
        default="easyocr",
        help="OCR artifact backend directory, e.g. easyocr or mmocr-dbnet-sar.",
    )
    parser.add_argument("--predictions")
    parser.add_argument("--output-dir")
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
    predictions_path = (
        Path(args.predictions)
        if args.predictions
        else root
        / "diagnostics"
        / "jersey_number_ocr"
        / args.backend
        / "predictions.jsonl"
    )
    if not tracklets_path.is_file():
        raise FileNotFoundError(tracklets_path)
    if not predictions_path.is_file():
        raise FileNotFoundError(predictions_path)

    tracklets = _load_tracklets(tracklets_path)
    known_tracklet_ids = {
        item.tracklet_id for item in tracklets
    }

    config = JerseyNumberAggregationConfig()
    observations = tuple(
        observation_from_prediction_row(row, config)
        for row in _iter_jsonl(predictions_path)
    )

    unknown_tracklets = sorted(
        {
            item.tracklet_id
            for item in observations
            if item.tracklet_id not in known_tracklet_ids
        }
    )
    if unknown_tracklets:
        raise ValueError(
            "OCR predictions reference unknown tracklets: "
            f"{unknown_tracklets[:10]}"
        )

    profiles = aggregate_all_tracklets(
        observations,
        config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "jersey_number_evidence"
        / JERSEY_NUMBER_VERSION
        / args.backend
    )
    observations_path = output_dir / "observations.jsonl"
    profiles_path = output_dir / "profiles.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    observation_count = _write_jsonl(
        observations_path,
        (item.to_dict() for item in observations),
    )
    profile_count = _write_jsonl(
        profiles_path,
        (item.to_dict() for item in profiles),
    )

    summary = jersey_number_summary(
        observations,
        profiles,
        total_tracklet_count=len(tracklets),
    )
    best_number_counts = Counter(
        profile.best_number
        for profile in profiles
        if profile.best_number is not None
    )
    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "backend": args.backend,
            "source_predictions": str(predictions_path),
            "best_number_counts": {
                str(number): count
                for number, count in sorted(best_number_counts.items())
            },
            "config": config.to_dict(),
        }
    )
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "jersey_number_version": JERSEY_NUMBER_VERSION,
            "backend": args.backend,
            "observations": observations_path.name,
            "profiles": profiles_path.name,
            "summary": summary_path.name,
            "observation_count": observation_count,
            "profile_count": profile_count,
        },
    )

    print(
        "[jersey-evidence-7C.2B] "
        f"match={context.match_id} clip={context.clip_id} "
        f"backend={args.backend}"
    )
    print(
        "[jersey-evidence-7C.2B] "
        f"views={summary['observation_count']} "
        f"recognized={summary['recognized_observation_count']} "
        f"fraction={summary['recognized_observation_fraction']}"
    )
    print(
        "[jersey-evidence-7C.2B] "
        f"profiles={summary['tracklet_profile_count']}/"
        f"{len(tracklets)} "
        f"known_number_coverage="
        f"{summary['known_number_tracklet_coverage']:.6f} "
        f"strong_coverage="
        f"{summary['strong_tracklet_coverage']:.6f}"
    )
    print(
        "[jersey-evidence-7C.2B] "
        f"strengths="
        f"{json.dumps(summary['tracklet_strength_counts'], sort_keys=True)}"
    )
    print(
        "[jersey-evidence-7C.2B] "
        f"ocr_confidence="
        f"{json.dumps(summary['ocr_confidence_recognized'], sort_keys=True)}"
    )
    print(
        "[jersey-evidence-7C.2B] "
        f"best_fraction="
        f"{json.dumps(summary['best_fraction'], sort_keys=True)}"
    )
    print(
        "[jersey-evidence-7C.2B] "
        f"margin={json.dumps(summary['margin'], sort_keys=True)}"
    )
    print(f"[jersey-evidence-7C.2B] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
