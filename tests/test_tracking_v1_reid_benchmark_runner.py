from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracking.appearance.reid import HSVHistogramReIDBackend


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "tracking"
    / "run_v1_reid_benchmark.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_v1_reid_benchmark",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def quality_record(
    *,
    tracklet_id="trk-1",
    detection_id="det-1",
    frame_index=0,
    crop_ref="diagnostics/crop.jpg",
    eligible=True,
):
    return {
        "tracklet_id": tracklet_id,
        "detection_id": detection_id,
        "frame_index": frame_index,
        "timestamp_s": frame_index / 30.0,
        "representative_view": "tight_far_c1",
        "crop_ref": crop_ref,
        "selection_quality": 0.9,
        "reid_quality": 0.8,
        "reid_eligible": eligible,
        "tier": "usable",
    }


def test_eligible_quality_records_filters_and_sorts():
    payload = {
        "records": [
            quality_record(
                tracklet_id="b",
                detection_id="b",
                frame_index=6,
            ),
            quality_record(
                tracklet_id="x",
                detection_id="x",
                frame_index=3,
                eligible=False,
            ),
            quality_record(
                tracklet_id="a",
                detection_id="a",
                frame_index=3,
            ),
        ]
    }

    result = runner.eligible_quality_records(payload)

    assert [item["detection_id"] for item in result] == ["a", "b"]


def test_eligible_quality_records_rejects_duplicate_detection():
    item = quality_record()

    with pytest.raises(ValueError, match="duplicate"):
        runner.eligible_quality_records(
            {"records": [item, dict(item)]}
        )


def test_embed_quality_records_creates_appearance_samples(tmp_path):
    crop_dir = tmp_path / "diagnostics"
    crop_dir.mkdir()
    crop = np.full((100, 50, 3), (0, 0, 255), dtype=np.uint8)
    path = crop_dir / "crop.jpg"
    assert cv2.imwrite(str(path), crop)

    records = [
        quality_record(
            crop_ref=str(path.relative_to(tmp_path))
        )
    ]
    backend = HSVHistogramReIDBackend()

    samples = runner.embed_quality_records(
        records=records,
        tracking_root_path=tmp_path,
        backend=backend,
        batch_size=16,
    )

    assert len(samples) == 1
    assert samples[0].quality == pytest.approx(0.8)
    descriptor = samples[0].descriptor(
        "reid",
        backend.model_id,
    )
    assert descriptor is not None
    assert descriptor.dimension == 128


def test_embed_quality_records_rejects_bad_batch_size(tmp_path):
    with pytest.raises(ValueError, match="batch_size"):
        runner.embed_quality_records(
            records=[],
            tracking_root_path=tmp_path,
            backend=HSVHistogramReIDBackend(),
            batch_size=0,
        )


def test_safe_model_dir():
    assert runner._safe_model_dir("foo/bar:v1") == "foo_bar_v1"


def test_parse_args_defaults():
    args = runner.parse_args([])

    assert args.backend == "hsv-hist"
    assert args.batch_size == 64
    assert args.negative_min_separation_m == pytest.approx(1.5)
    assert args.min_split_samples == 4
