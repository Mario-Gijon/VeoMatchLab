#!/usr/bin/env python3
"""Run PRTReID-SoccerNet through the existing Tracking V1 benchmark.

PRTReID itself runs in ``.venv-prtreid``.  This Python 3.12 process only
prepares the request, invokes the isolated worker, converts the returned
256-D rows into AppearanceSample objects, and calls the exact same benchmark
used by the HSV baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DEV_CONFIG = ROOT / "experiments" / "tracking" / "dev_clip.json"

from tracking.appearance.prtreid_bridge import (  # noqa: E402
    PRTREID_MODEL_ID,
    build_prtreid_request,
    load_prtreid_npz,
    run_prtreid_worker,
    samples_from_prtreid,
    write_prtreid_request,
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
            payload = json.loads(line)
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

    result = [
        dict(item)
        for item in raw_records
        if isinstance(item, Mapping)
        and bool(item.get("reid_eligible", False))
    ]
    result.sort(
        key=lambda item: (
            int(item["frame_index"]),
            str(item["tracklet_id"]),
            str(item["detection_id"]),
        )
    )
    return result


def _safe_dir(value: str) -> str:
    token = "".join(
        char if char.isalnum() or char in "-_" else "_"
        for char in value
    ).strip("_")
    if not token:
        raise ValueError("model id has no safe path characters")
    return token


def parse_args(argv: Sequence[str] | None = None):
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
        "--prtreid-python",
        default=str(ROOT / ".venv-prtreid" / "bin" / "python"),
    )
    parser.add_argument(
        "--worker",
        default=str(
            ROOT
            / "experiments"
            / "tracking"
            / "prtreid_worker.py"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            ROOT
            / "models"
            / "prtreid"
            / "prtreid-soccernet-baseline.pth.tar"
        ),
    )
    parser.add_argument(
        "--hrnet-dir",
        default=str(ROOT / "models" / "prtreid"),
    )
    parser.add_argument("--device", default="cuda")
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


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)

    dev = load_json(_repo_path(args.dev_config))
    root = tracking_root(
        _repo_path(args.data_root),
        match_id=dev["match_id"],
        clip_id=dev["clip_id"],
    )

    quality_path = (
        root / "diagnostics" / "reid_quality" / "quality.json"
    )
    tracklets_path = root / "tracklets" / "tracklets.jsonl"

    records = eligible_quality_records(load_json(quality_path))
    tracklets = load_tracklets(tracklets_path)

    model_dir = (
        root
        / "diagnostics"
        / "reid_benchmark"
        / _safe_dir(PRTREID_MODEL_ID)
    )
    workspace = model_dir / "workspace"
    request_path = workspace / "request.json"
    output_npz = workspace / "embeddings.npz"

    request = build_prtreid_request(
        records,
        tracking_root=root,
    )
    write_prtreid_request(
        request_path,
        request,
    )

    print(
        "[prtreid-benchmark] "
        f"eligible_crops={len(records)} "
        f"model={PRTREID_MODEL_ID}"
    )

    completed = run_prtreid_worker(
        python_executable=_repo_path(args.prtreid_python),
        worker_script=_repo_path(args.worker),
        request_json=request_path,
        output_npz=output_npz,
        checkpoint=_repo_path(args.checkpoint),
        hrnet_dir=_repo_path(args.hrnet_dir),
        device=args.device,
        batch_size=args.batch_size,
    )
    if completed.stdout.strip():
        print(completed.stdout.strip())
    if completed.stderr.strip():
        print(completed.stderr.strip(), file=sys.stderr)

    embeddings, worker_metadata = load_prtreid_npz(
        output_npz,
        expected_count=len(records),
    )
    samples = samples_from_prtreid(
        records,
        embeddings,
        worker_metadata=worker_metadata,
    )

    report = benchmark_reid(
        samples,
        tracklets,
        model_id=PRTREID_MODEL_ID,
        config=ReIDBenchmarkConfig(
            min_split_samples=args.min_split_samples,
            negative_min_separation_m=(
                args.negative_min_separation_m
            ),
            hard_example_count=args.hard_example_count,
        ),
    )

    model_dir.mkdir(parents=True, exist_ok=True)

    embeddings_jsonl = model_dir / "embeddings.jsonl"
    with embeddings_jsonl.open("w", encoding="utf-8") as handle:
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
        "backend": {
            "model_id": PRTREID_MODEL_ID,
            "embedding_dim": int(
                worker_metadata["embedding_dim"]
            ),
            "checkpoint": worker_metadata.get(
                "checkpoint_name"
            ),
            "test_embeddings": worker_metadata.get(
                "test_embeddings"
            ),
            "device": worker_metadata.get("device"),
        },
        "report": report,
    }

    summary_path = model_dir / "summary.json"
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

    print("[prtreid-benchmark] completed")
    print(
        "[prtreid-benchmark] sample "
        f"pos={sample_level['same_tracklet_positive']['count']} "
        f"neg={sample_level['same_frame_negative']['count']} "
        f"auc={sample_level['roc_auc']}"
    )
    print(
        "[prtreid-benchmark] tracklet "
        f"split_pos={tracklet_level['split_positive']['count']} "
        f"concurrent_neg="
        f"{tracklet_level['concurrent_negative']['count']} "
        f"auc={tracklet_level['roc_auc']} "
        f"margin="
        f"{tracklet_level['p10_positive_minus_p95_negative']}"
    )
    print(f"[prtreid-benchmark] summary={summary_path}")

    return summary_payload


if __name__ == "__main__":
    run()
