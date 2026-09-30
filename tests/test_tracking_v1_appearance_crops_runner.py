from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tracking.appearance.selection import (
    AppearanceMoment,
    AppearanceSelectionConfig,
    AppearanceSelectionSet,
    TrackletMomentSelection,
)
from tracking.detections import CanonicalDetection, CanonicalDetectionFrame


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "tracking"
    / "run_v1_appearance_crops.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_v1_appearance_crops",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def detection(
    *,
    detection_id="det-1",
    frame_index=0,
    timestamp_s=0.0,
    representative_view="tight_far_c1",
    bbox=(10.0, 20.0, 30.0, 60.0),
):
    return CanonicalDetection(
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        confidence=0.9,
        x=0.0,
        z=0.0,
        representative_view=representative_view,
        virtual_bbox_xyxy=bbox,
    )


def moment(
    *,
    detection_id="det-1",
    tracklet_id="trk-1",
    frame_index=0,
    timestamp_s=0.0,
    quality=0.8,
):
    return AppearanceMoment(
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        quality=quality,
        representative_view="tight_far_c1",
        source_lens=None,
        confidence_score=0.8,
        visual_scale_score=0.8,
        edge_clearance_score=0.8,
        isolation_score=0.8,
        geometry_score=0.8,
        provenance_score=0.8,
    )


def selection_for(item):
    return AppearanceSelectionSet(
        selections=(
            TrackletMomentSelection(
                tracklet_id=item.tracklet_id,
                candidate_count=1,
                eligible_count=1,
                moments=(item,),
            ),
        ),
        config=AppearanceSelectionConfig(),
    )


def test_detection_render_size_reads_detection_stage_config():
    manifest = {
        "stages": {
            "detections": {
                "config": {
                    "width": 1280,
                    "height": 720,
                }
            }
        }
    }
    assert runner.detection_render_size(manifest) == (1280, 720)


def test_detection_render_size_rejects_missing_config():
    with pytest.raises(ValueError):
        runner.detection_render_size({"stages": {}})


def test_detection_index_enforces_global_unique_ids():
    first = detection()
    second = detection(
        detection_id="det-1",
        frame_index=3,
        timestamp_s=0.1,
    )
    frames = (
        CanonicalDetectionFrame(
            frame_index=0,
            timestamp_s=0.0,
            detections=(first,),
        ),
        CanonicalDetectionFrame(
            frame_index=3,
            timestamp_s=0.1,
            detections=(second,),
        ),
    )

    with pytest.raises(ValueError, match="duplicate"):
        runner.detection_index(frames)


def test_selected_records_resolves_renderable_detection():
    det = detection()
    item = moment()
    records = runner.selected_records(
        selection_for(item),
        {"det-1": det},
    )

    assert len(records) == 1
    assert records[0]["status"] == "pending"
    assert records[0]["representative_view"] == "tight_far_c1"
    assert records[0]["virtual_bbox_xyxy"] == [
        10.0,
        20.0,
        30.0,
        60.0,
    ]


def test_selected_records_marks_missing_view_as_skip():
    det = detection(representative_view=None)
    item = moment()
    records = runner.selected_records(
        selection_for(item),
        {"det-1": det},
    )

    assert records[0]["status"] == "skipped_no_representative_view"


def test_selected_records_marks_missing_bbox_as_skip():
    det = detection(bbox=None)
    item = moment()
    records = runner.selected_records(
        selection_for(item),
        {"det-1": det},
    )

    assert records[0]["status"] == "skipped_no_virtual_bbox"


def test_selected_records_rejects_missing_detection():
    item = moment()

    with pytest.raises(ValueError, match="missing"):
        runner.selected_records(
            selection_for(item),
            {},
        )


def test_ordered_tight_specs_orders_far_then_near():
    specs = [
        SimpleNamespace(
            name="tight_near_c2",
            mode="tight",
            cell=SimpleNamespace(band="near", index_in_band=1),
        ),
        SimpleNamespace(
            name="tight_far_c4",
            mode="tight",
            cell=SimpleNamespace(band="far", index_in_band=3),
        ),
        SimpleNamespace(
            name="tight_far_c2",
            mode="tight",
            cell=SimpleNamespace(band="far", index_in_band=1),
        ),
        SimpleNamespace(
            name="tight_near_c1",
            mode="tight",
            cell=SimpleNamespace(band="near", index_in_band=0),
        ),
        SimpleNamespace(
            name="tight_far_c1",
            mode="tight",
            cell=SimpleNamespace(band="far", index_in_band=0),
        ),
        SimpleNamespace(
            name="tight_far_c3",
            mode="tight",
            cell=SimpleNamespace(band="far", index_in_band=2),
        ),
        SimpleNamespace(
            name="context_far_c1",
            mode="context",
            cell=SimpleNamespace(band="far", index_in_band=0),
        ),
    ]

    names = [
        item.name
        for item in runner.ordered_tight_specs(specs)
    ]
    assert names == [
        "tight_far_c1",
        "tight_far_c2",
        "tight_far_c3",
        "tight_far_c4",
        "tight_near_c1",
        "tight_near_c2",
    ]


def test_ordered_tight_specs_rejects_wrong_camera_count():
    specs = [
        SimpleNamespace(
            name="tight_far_c1",
            mode="tight",
            cell=SimpleNamespace(band="far", index_in_band=0),
        )
    ]

    with pytest.raises(RuntimeError, match="4 FAR"):
        runner.ordered_tight_specs(specs)


def test_make_contact_sheet_returns_none_without_saved_records():
    assert runner.make_contact_sheet(
        [{"status": "failed"}]
    ) is None


def test_make_contact_sheet_builds_expected_grid():
    image = np.full((30, 20, 3), 255, dtype=np.uint8)
    records = [
        {
            "status": "saved",
            "_image": image,
            "tracklet_id": f"trk-{index}",
            "quality": 0.9,
            "timestamp_s": float(index),
        }
        for index in range(3)
    ]

    sheet = runner.make_contact_sheet(
        records,
        columns=2,
        panel_width=100,
        panel_height=140,
    )

    assert sheet is not None
    assert sheet.shape == (280, 200, 3)


def test_public_record_drops_private_runtime_image():
    payload = runner._public_record(
        {
            "status": "saved",
            "crop_ref": "x.jpg",
            "_image": np.zeros((2, 2, 3), dtype=np.uint8),
        }
    )

    assert payload == {
        "status": "saved",
        "crop_ref": "x.jpg",
    }


def test_parse_args_defaults():
    args = runner.parse_args([])

    assert args.max_samples == 8
    assert args.min_samples == 3
    assert args.min_quality == pytest.approx(0.42)
    assert args.padding_x == pytest.approx(0.08)
    assert args.padding_y == pytest.approx(0.06)
    assert args.audit_count == 40
