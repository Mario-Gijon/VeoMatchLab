#!/usr/bin/env python3
"""Run the Tracking V1 ReID proxy benchmark.

Hito 2E-A intentionally ships with a cheap HSV torso-histogram backend only.
The benchmark and artifact format are model-independent; a future PRTReID/KPR
backend will run through the same path and can be compared directly.

This runner:
1. reads the existing ``reid_quality/quality.json``;
2. keeps only ``reid_eligible`` crops;
3. computes embeddings in batches;
4. materializes model-agnostic ``AppearanceSample`` records;
5. evaluates same-tracklet positives and concurrent-player negatives;
6. writes diagnostics only -- the appearance stage is still not published.

Outputs:
    tracking/v1/diagnostics/reid_benchmark/<model_id>/
        embeddings.jsonl
        summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DEV_CONFIG = ROOT / "experiments" / "tracking" / "dev_clip.json"

from tracking.appearance.models import (  # noqa: E402
    AppearanceDescriptor,
    AppearanceSample,
)
from tracking.appearance.reid import (  # noqa: E402
    ReIDBackend,
    create_reid_backend,
    embed_validated,
)
from tracking.appearance.reid_benchmark import (  # noqa: E402
    ReIDBenchmarkConfig,
    benchmark_reid,
)
from tracking.tracklets import Tracklet, TrackletSet  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must encode a JSON object")
    return payload


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}"
                ) from exc
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}: line {line_number} must encode an object"
                )
            yield payload


def tracking_root(
    data_root: Path,
    *,
    match_id: str,
    clip_id: str,
) -> Path:
    return (
        data_root
        / "matches"
        / match_id
        / "clips"
        / clip_id
        / "tracking"
        / "v1"
    )


def load_tracklets(path: Path) -> TrackletSet:
    if not path.is_file():
        raise FileNotFoundError(path)
    return TrackletSet(
        tracklets=tuple(
            Tracklet.from_dict(payload)
            for payload in _iter_jsonl(path)
        )
    )


def eligible_quality_records(
    quality_payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_records = quality_payload.get("records")
    if not isinstance(raw_records, list):
        raise ValueError("quality payload records must be a list")

    result: list[dict[str, Any]] = []
    seen_detection_ids: set[str] = set()

    for raw in raw_records:
        if not isinstance(raw, Mapping):
            raise ValueError("quality record must be an object")
        if not bool(raw.get("reid_eligible", False)):
            continue

        required = (
            "tracklet_id",
            "detection_id",
            "frame_index",
            "timestamp_s",
            "crop_ref",
            "reid_quality",
        )
        missing = [
            key
            for key in required
            if key not in raw
        ]
        if missing:
            raise ValueError(
                f"eligible quality record is missing {missing}"
            )

        detection_id = str(raw["detection_id"])
        if detection_id in seen_detection_ids:
            raise ValueError(
                f"duplicate eligible detection_id {detection_id!r}"
            )
        seen_detection_ids.add(detection_id)

        result.append(dict(raw))

    result.sort(
        key=lambda item: (
            int(item["frame_index"]),
            str(item["tracklet_id"]),
            str(item["detection_id"]),
        )
    )
    return result


def embed_quality_records(
    *,
    records: Sequence[Mapping[str, Any]],
    tracking_root_path: Path,
    backend: ReIDBackend,
    batch_size: int,
) -> tuple[AppearanceSample, ...]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    samples: list[AppearanceSample] = []

    for start in range(0, len(records), batch_size):
        batch_records = records[start:start + batch_size]
        images = []

        for record in batch_records:
            crop_ref = str(record["crop_ref"])
            path = tracking_root_path / crop_ref
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(
                    f"could not read crop: {path}"
                )
            images.append(image)

        matrix = embed_validated(
            backend,
            images,
        )

        for record, embedding in zip(
            batch_records,
            matrix,
            strict=True,
        ):
            detection_id = str(record["detection_id"])
            sample_id = (
                f"reid-{backend.model_id}-{detection_id}"
            )

            descriptor = AppearanceDescriptor(
                name="reid",
                model_id=backend.model_id,
                values=tuple(float(value) for value in embedding),
                metric="cosine",
                metadata={
                    "source": "reid-benchmark",
                },
            )
            sample = AppearanceSample(
                sample_id=sample_id,
                tracklet_id=str(record["tracklet_id"]),
                detection_id=detection_id,
                frame_index=int(record["frame_index"]),
                timestamp_s=float(record["timestamp_s"]),
                quality=float(record["reid_quality"]),
                descriptors=(descriptor,),
                view_name=(
                    str(record["representative_view"])
                    if record.get("representative_view")
                    is not None
                    else None
                ),
                crop_ref=str(record["crop_ref"]),
                metadata={
                    "selection_quality": float(
                        record.get(
                            "selection_quality",
                            record.get("quality", 0.0),
                        )
                    ),
                    "reid_quality": float(
                        record["reid_quality"]
                    ),
                    "reid_tier": str(
                        record.get("tier", "unknown")
                    ),
                },
            )
            samples.append(sample)

    return tuple(samples)


def _safe_model_dir(model_id: str) -> str:
    result = "".join(
        character
        if character.isalnum() or character in ("-", "_")
        else "_"
        for character in model_id
    ).strip("_")
    if not result:
        raise ValueError("model_id has no safe path characters")
    return result


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-config",
        default=str(DEFAULT_DEV_CONFIG),
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
    )
    parser.add_argument(
        "--quality-json",
        default=None,
        help=(
            "Default: <tracking-v1>/diagnostics/"
            "reid_quality/quality.json"
        ),
    )
    parser.add_argument(
        "--backend",
        default="hsv-hist",
        help="Hito 2E-A: hsv-hist",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--negative-min-separation-m",
        type=float,
        default=1.50,
    )
    parser.add_argument(
        "--min-split-samples",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--hard-example-count",
        type=int,
        default=20,
    )
    return parser.parse_args(argv)


def run(
    argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    args = parse_args(argv)

    dev = load_json(_repo_path(args.dev_config))
    data_root = _repo_path(args.data_root)
    root = tracking_root(
        data_root,
        match_id=dev["match_id"],
        clip_id=dev["clip_id"],
    )

    quality_path = (
        _repo_path(args.quality_json)
        if args.quality_json is not None
        else root
        / "diagnostics"
        / "reid_quality"
        / "quality.json"
    )
    tracklets_path = (
        root
        / "tracklets"
        / "tracklets.jsonl"
    )

    quality_payload = load_json(quality_path)
    records = eligible_quality_records(
        quality_payload
    )
    tracklets = load_tracklets(tracklets_path)

    backend = create_reid_backend(args.backend)

    print(
        "[reid-benchmark] "
        f"backend={backend.model_id} "
        f"eligible_crops={len(records)} "
        f"dim={backend.embedding_dim}"
    )

    samples = embed_quality_records(
        records=records,
        tracking_root_path=root,
        backend=backend,
        batch_size=args.batch_size,
    )

    benchmark_config = ReIDBenchmarkConfig(
        min_split_samples=args.min_split_samples,
        negative_min_separation_m=(
            args.negative_min_separation_m
        ),
        hard_example_count=args.hard_example_count,
    )
    report = benchmark_reid(
        samples,
        tracklets,
        model_id=backend.model_id,
        config=benchmark_config,
    )

    output_dir = (
        root
        / "diagnostics"
        / "reid_benchmark"
        / _safe_model_dir(backend.model_id)
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings_path = (
        output_dir
        / "embeddings.jsonl"
    )
    with embeddings_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for sample in samples:
            handle.write(
                json.dumps(
                    sample.to_dict(),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    summary_payload = {
        "match_id": dev["match_id"],
        "clip_id": dev["clip_id"],
        "quality_json": str(
            quality_path.relative_to(root)
        ),
        "tracklets_artifact": (
            "tracklets/tracklets.jsonl"
        ),
        "backend": {
            "model_id": backend.model_id,
            "embedding_dim": backend.embedding_dim,
        },
        "report": report,
    }

    summary_path = (
        output_dir
        / "summary.json"
    )
    summary_path.write_text(
        json.dumps(
            summary_payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    sample_level = report["sample_level"]
    tracklet_level = report["tracklet_level"]

    print("[reid-benchmark] completed")
    print(
        "[reid-benchmark] sample "
        f"pos={sample_level['same_tracklet_positive']['count']} "
        f"neg={sample_level['same_frame_negative']['count']} "
        f"auc={sample_level['roc_auc']}"
    )
    print(
        "[reid-benchmark] tracklet "
        f"split_pos={tracklet_level['split_positive']['count']} "
        f"concurrent_neg="
        f"{tracklet_level['concurrent_negative']['count']} "
        f"auc={tracklet_level['roc_auc']} "
        f"margin="
        f"{tracklet_level['p10_positive_minus_p95_negative']}"
    )
    print(
        f"[reid-benchmark] embeddings={embeddings_path}"
    )
    print(
        f"[reid-benchmark] summary={summary_path}"
    )

    return summary_payload


if __name__ == "__main__":
    run()
