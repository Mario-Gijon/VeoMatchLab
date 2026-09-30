"""Bridge helpers for running PRTReID in an isolated Python environment.

The production Tracking V1 environment currently uses Python 3.12.  The
SoccerNet/PRTReID stack is deliberately kept separate so dependency changes in
that research stack cannot destabilize the main tracker.

This module contains *no* PRTReID import.  It only:
- writes a deterministic extraction request,
- invokes an external Python worker,
- validates the returned NPZ,
- converts rows into the existing AppearanceSample contract.

The actual neural model is loaded by ``experiments/tracking/prtreid_worker.py``
inside ``.venv-prtreid``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np

from tracking.appearance.models import (
    AppearanceDescriptor,
    AppearanceSample,
)


PRTREID_MODEL_ID = "prtreid-soccernet-baseline-global-v1"
PRTREID_EXPECTED_DIM = 256


def _non_empty(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_quality(value: float) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ValueError("reid_quality must be finite and in [0, 1]")
    return parsed


def build_prtreid_request(
    records: Sequence[Mapping[str, Any]],
    *,
    tracking_root: Path,
) -> dict[str, Any]:
    """Build the JSON request consumed by the isolated worker."""
    items: list[dict[str, Any]] = []
    seen_detection_ids: set[str] = set()

    for index, record in enumerate(records):
        if not bool(record.get("reid_eligible", False)):
            raise ValueError(
                f"record {index} is not marked reid_eligible"
            )

        detection_id = _non_empty(
            "detection_id",
            str(record["detection_id"]),
        )
        if detection_id in seen_detection_ids:
            raise ValueError(
                f"duplicate detection_id in request: {detection_id!r}"
            )
        seen_detection_ids.add(detection_id)

        crop_ref = _non_empty(
            "crop_ref",
            str(record["crop_ref"]),
        )
        crop_path = tracking_root / crop_ref
        if not crop_path.is_file():
            raise FileNotFoundError(crop_path)

        items.append(
            {
                "row_index": index,
                "detection_id": detection_id,
                "crop_path": str(crop_path.resolve()),
            }
        )

    return {
        "schema_version": 1,
        "model_id": PRTREID_MODEL_ID,
        "expected_dim": PRTREID_EXPECTED_DIM,
        "items": items,
    }


def write_prtreid_request(
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


def run_prtreid_worker(
    *,
    python_executable: Path,
    worker_script: Path,
    request_json: Path,
    output_npz: Path,
    checkpoint: Path,
    hrnet_dir: Path,
    device: str = "cuda",
    batch_size: int = 64,
) -> subprocess.CompletedProcess[str]:
    """Run the isolated worker and surface stdout/stderr to the caller."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    required = (
        python_executable,
        worker_script,
        request_json,
        checkpoint,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if not hrnet_dir.is_dir():
        raise FileNotFoundError(hrnet_dir)

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(python_executable),
        str(worker_script),
        "--request",
        str(request_json),
        "--output",
        str(output_npz),
        "--checkpoint",
        str(checkpoint),
        "--hrnet-dir",
        str(hrnet_dir),
        "--device",
        device,
        "--batch-size",
        str(batch_size),
    ]

    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def load_prtreid_npz(
    path: Path,
    *,
    expected_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate worker output and return normalized embeddings + metadata."""
    if not path.is_file():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as payload:
        if "embeddings" not in payload or "metadata_json" not in payload:
            raise ValueError(
                "PRTReID NPZ must contain embeddings and metadata_json"
            )
        embeddings = np.asarray(
            payload["embeddings"],
            dtype=np.float32,
        )
        metadata_raw = payload["metadata_json"]

        if metadata_raw.ndim == 0:
            metadata_text = str(metadata_raw.item())
        else:
            raise ValueError("metadata_json must be a scalar string")

    metadata = json.loads(metadata_text)
    if not isinstance(metadata, dict):
        raise ValueError("PRTReID metadata must be a JSON object")

    if embeddings.shape != (
        expected_count,
        PRTREID_EXPECTED_DIM,
    ):
        raise ValueError(
            "PRTReID embedding shape mismatch: "
            f"{embeddings.shape} != "
            f"{(expected_count, PRTREID_EXPECTED_DIM)}"
        )
    if not np.isfinite(embeddings).all():
        raise ValueError("PRTReID embeddings contain non-finite values")

    if expected_count:
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(norms <= 1e-12):
            raise ValueError("PRTReID returned zero-norm embeddings")
        embeddings = embeddings / norms[:, None]

    if metadata.get("model_id") != PRTREID_MODEL_ID:
        raise ValueError(
            "unexpected PRTReID model_id: "
            f"{metadata.get('model_id')!r}"
        )
    if int(metadata.get("embedding_dim", -1)) != PRTREID_EXPECTED_DIM:
        raise ValueError("unexpected PRTReID embedding dimension")

    return embeddings, metadata


def samples_from_prtreid(
    records: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    *,
    worker_metadata: Mapping[str, Any],
) -> tuple[AppearanceSample, ...]:
    """Convert PRTReID rows to the common AppearanceSample contract."""
    matrix = np.asarray(embeddings, dtype=np.float32)
    if matrix.shape != (
        len(records),
        PRTREID_EXPECTED_DIM,
    ):
        raise ValueError(
            "records/embeddings shape mismatch: "
            f"{len(records)} records vs {matrix.shape}"
        )

    result: list[AppearanceSample] = []

    for record, embedding in zip(
        records,
        matrix,
        strict=True,
    ):
        detection_id = _non_empty(
            "detection_id",
            str(record["detection_id"]),
        )
        quality = _finite_quality(record["reid_quality"])

        descriptor = AppearanceDescriptor(
            name="reid",
            model_id=PRTREID_MODEL_ID,
            values=tuple(float(value) for value in embedding),
            metric="cosine",
            metadata={
                "source": "prtreid-isolated-worker",
                "checkpoint": worker_metadata.get(
                    "checkpoint_name"
                ),
                "test_embeddings": worker_metadata.get(
                    "test_embeddings"
                ),
            },
        )

        result.append(
            AppearanceSample(
                sample_id=(
                    f"reid-{PRTREID_MODEL_ID}-{detection_id}"
                ),
                tracklet_id=str(record["tracklet_id"]),
                detection_id=detection_id,
                frame_index=int(record["frame_index"]),
                timestamp_s=float(record["timestamp_s"]),
                quality=quality,
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
                    "reid_quality": quality,
                    "reid_tier": str(
                        record.get("tier", "unknown")
                    ),
                    "backend": "prtreid-isolated-worker",
                },
            )
        )

    return tuple(result)
