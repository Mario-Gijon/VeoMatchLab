#!/usr/bin/env python3
"""Run Tracking V1 Hito 7D robust overlapping ReID calibration.

Consumes existing V1 tracklets and the already-computed PRTReID embeddings.
It does NOT run detection, crop extraction, ReID inference, identity solving,
persistent tracking, team classification or jersey-number recognition.

Output:
tracking/v1/diagnostics/reid_calibration/reid-overlap-calibration-v1/
    split_positive.jsonl
    simultaneous_negative.jsonl
    close_overlap.jsonl
    summary.json
    index.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.reid_calibration import (  # noqa: E402
    REID_CALIBRATION_VERSION,
    ReIDCalibrationConfig,
    build_reid_overlap_calibration,
)
from tracking.appearance.evidence import (  # noqa: E402
    AppearanceEvidenceConfig,
    iter_embedding_jsonl,
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


def _load_tracklets(path: Path) -> tuple[Tracklet, ...]:
    return tuple(
        Tracklet.from_dict(payload)
        for payload in _iter_jsonl(path)
    )


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
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


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--min-split-samples",
        type=int,
        default=4,
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

    tracklets_path = (
        root / "tracklets" / "tracklets.jsonl"
    )
    embeddings_path = (
        root
        / "diagnostics"
        / "reid_benchmark"
        / "prtreid-soccernet-baseline-global-v1"
        / "embeddings.jsonl"
    )

    if not tracklets_path.is_file():
        raise FileNotFoundError(tracklets_path)
    if not embeddings_path.is_file():
        raise FileNotFoundError(embeddings_path)

    tracklets = _load_tracklets(tracklets_path)

    evidence_config = AppearanceEvidenceConfig()
    samples = tuple(
        iter_embedding_jsonl(
            embeddings_path,
            evidence_config,
        )
    )

    calibration = build_reid_overlap_calibration(
        tracklets=tracklets,
        samples=samples,
        evidence_config=evidence_config,
        calibration_config=ReIDCalibrationConfig(
            min_split_samples=args.min_split_samples,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            root
            / "diagnostics"
            / "reid_calibration"
            / REID_CALIBRATION_VERSION
        )
    )

    split_path = output_dir / "split_positive.jsonl"
    negative_path = (
        output_dir / "simultaneous_negative.jsonl"
    )
    close_path = output_dir / "close_overlap.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    split_count = _write_jsonl(
        split_path,
        (
            item.to_dict()
            for item in calibration.split_positive
        ),
    )
    negative_count = _write_jsonl(
        negative_path,
        (
            item.to_dict()
            for item
            in calibration.simultaneous_negative
        ),
    )
    close_count = _write_jsonl(
        close_path,
        (
            item.to_dict()
            for item in calibration.close_overlap
        ),
    )

    summary = calibration.summary()
    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "tracklet_count": len(tracklets),
            "appearance_sample_count": len(samples),
            "source_tracklets": str(tracklets_path),
            "source_embeddings": str(embeddings_path),
        }
    )
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "calibration_version": (
                REID_CALIBRATION_VERSION
            ),
            "split_positive": split_path.name,
            "simultaneous_negative": negative_path.name,
            "close_overlap": close_path.name,
            "summary": summary_path.name,
            "split_positive_count": split_count,
            "simultaneous_negative_count": (
                negative_count
            ),
            "close_overlap_count": close_count,
            "identity_decisions": False,
        },
    )

    anchors = summary["anchors"]
    print(
        "[reid-calibration-7D] "
        f"match={context.match_id} "
        f"clip={context.clip_id}"
    )
    print(
        "[reid-calibration-7D] "
        f"tracklets={len(tracklets)} "
        f"profiles={summary['profiled_tracklet_count']} "
        f"unprofiled={summary['unprofiled_tracklet_count']} "
        f"samples={len(samples)}"
    )
    print(
        "[reid-calibration-7D] "
        f"split_positive={split_count} "
        f"simultaneous_negative={negative_count} "
        f"close_overlap={close_count}"
    )
    print(
        "[reid-calibration-7D] "
        f"positive="
        f"{json.dumps(summary['split_positive'], sort_keys=True)}"
    )
    print(
        "[reid-calibration-7D] "
        f"negative="
        f"{json.dumps(summary['simultaneous_negative'], sort_keys=True)}"
    )
    print(
        "[reid-calibration-7D] "
        f"close_overlap="
        f"{json.dumps(summary['close_overlap_diagnostic'], sort_keys=True)}"
    )
    print(
        "[reid-calibration-7D] "
        f"roc_auc={summary['roc_auc']} "
        f"negative_p95={anchors['negative_p95']} "
        f"positive_p10={anchors['positive_p10']} "
        f"margin={anchors['p10_positive_minus_p95_negative']} "
        f"ordered={anchors['ordered']}"
    )
    print(
        "[reid-calibration-7D] "
        f"artifact={index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
