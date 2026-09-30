from __future__ import annotations

import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from tracking.appearance.quality import ReIDCropQualityConfig


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "tracking"
    / "run_v1_reid_quality.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_v1_reid_quality",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def record(
    *,
    tracklet_id="trk-1",
    reid_quality=0.8,
    eligible=True,
    tier="usable",
    bbox_height=80.0,
    bbox_width=30.0,
    reasons=(),
):
    return {
        "tracklet_id": tracklet_id,
        "frame_index": 0,
        "timestamp_s": 0.0,
        "selection_quality": 0.9,
        "reid_quality": reid_quality,
        "reid_eligible": eligible,
        "tier": tier,
        "bbox_height_px": bbox_height,
        "bbox_width_px": bbox_width,
        "rejection_reasons": list(reasons),
        "_image": np.full((100, 40, 3), 100, dtype=np.uint8),
        "laplacian_variance": 20.0,
    }


def test_selection_quality_accepts_legacy_quality_key():
    assert runner._selection_quality({"quality": 0.7}) == pytest.approx(0.7)
    assert runner._selection_quality(
        {"selection_quality": 0.8}
    ) == pytest.approx(0.8)


def test_summary_counts_eligibility_tiers_and_reasons():
    records = [
        record(tracklet_id="a", reid_quality=0.9, tier="good"),
        record(
            tracklet_id="b",
            reid_quality=0.4,
            eligible=False,
            tier="reject",
            reasons=("bbox_height_below_min",),
        ),
    ]
    result = runner.summary(records, ReIDCropQualityConfig())

    assert result["crop_count"] == 2
    assert result["reid_eligible_count"] == 1
    assert result["reid_rejected_count"] == 1
    assert result["tier_counts"] == {
        "good": 1,
        "reject": 1,
    }
    assert result["rejection_reason_counts"] == {
        "bbox_height_below_min": 1,
    }


def test_quantile_interpolates():
    assert runner._quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)


def test_make_contact_sheet_returns_none_for_empty():
    assert runner.make_contact_sheet([]) is None


def test_make_contact_sheet_has_expected_shape():
    sheet = runner.make_contact_sheet(
        [record(tracklet_id="a"), record(tracklet_id="b")],
        columns=1,
        panel_width=100,
        panel_height=160,
    )

    assert sheet is not None
    assert sheet.shape == (320, 100, 3)


def test_public_drops_runtime_image():
    payload = runner._public(record())

    assert "_image" not in payload
    assert payload["tracklet_id"] == "trk-1"


def test_score_saved_records_reads_crop_and_scores_it(tmp_path):
    root = tmp_path
    crop_path = root / "diagnostics" / "appearance_crops" / "trk-1"
    crop_path.mkdir(parents=True)
    image_path = crop_path / "crop.jpg"

    yy, xx = np.indices((120, 60))
    gray = ((xx + yy) % 2 * 255).astype(np.uint8)
    image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    assert cv2.imwrite(str(image_path), image)

    index = {
        "records": [
            {
                "status": "saved",
                "tracklet_id": "trk-1",
                "detection_id": "det-1",
                "frame_index": 3,
                "timestamp_s": 0.1,
                "representative_view": "tight_far_c1",
                "quality": 0.95,
                "virtual_bbox_xyxy": [0, 0, 60, 120],
                "crop_ref": str(image_path.relative_to(root)),
                "crop_clipped": False,
            }
        ]
    }

    result = runner.score_saved_records(
        tracking_root_path=root,
        crop_index=index,
        config=ReIDCropQualityConfig(),
    )

    assert len(result) == 1
    assert result[0]["selection_quality"] == pytest.approx(0.95)
    assert result[0]["reid_eligible"]
    assert result[0]["reid_quality"] > 0.85


def test_parse_args_defaults():
    args = runner.parse_args([])

    assert args.min_bbox_height == pytest.approx(48.0)
    assert args.min_bbox_width == pytest.approx(12.0)
    assert args.min_reid_quality == pytest.approx(0.55)
    assert args.audit_count == 40
