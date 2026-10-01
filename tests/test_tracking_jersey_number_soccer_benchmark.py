from __future__ import annotations

import numpy as np
import pytest

from experiments.tracking.run_v1_jersey_number_soccer_benchmark import (
    _checkpoint_issue,
    _env_run_prefix,
    _normalise_state_dict,
    _pose_input_payload,
    _prediction_row,
    _safe_filename,
    _sequence_confidence,
    _strict_number,
    _torso_box,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("1", 1),
        ("9", 9),
        ("10", 10),
        ("99", 99),
        (" 17 ", 17),
        ("0", None),
        ("00", None),
        ("100", None),
        ("S9", None),
        ("", None),
        (None, None),
    ],
)
def test_strict_number(text, expected):
    assert _strict_number(text) == expected


def test_sequence_confidence_multiplies_tokens():
    assert _sequence_confidence(
        [0.9, 0.8, 0.95]
    ) == pytest.approx(0.684)


def test_sequence_confidence_handles_scalar_and_empty():
    assert _sequence_confidence(0.7) == pytest.approx(0.7)
    assert _sequence_confidence([]) == 0.0
    assert _sequence_confidence(None) == 0.0


def test_safe_filename_is_sanitized():
    row = {
        "tracklet_id": "trk/12",
        "sample_id": "sample 3",
    }
    assert _safe_filename(7, row) == "0007__trk-12__sample-3.jpg"


def test_normalise_state_dict_strips_module_prefix():
    state = {
        "module.model_ft.fc.weight": "x",
        "module.model_ft.fc.bias": "y",
    }
    assert _normalise_state_dict(state) == {
        "model_ft.fc.weight": "x",
        "model_ft.fc.bias": "y",
    }


def test_torso_box_uses_shoulders_and_hips():
    keypoints = [[0.0, 0.0, 1.0] for _ in range(17)]
    keypoints[5] = [20.0, 10.0, 0.9]
    keypoints[6] = [40.0, 12.0, 0.9]
    keypoints[11] = [22.0, 50.0, 0.9]
    keypoints[12] = [42.0, 52.0, 0.9]

    assert _torso_box(
        (100, 80, 3),
        keypoints,
        confidence_threshold=0.4,
        padding_px=5,
    ) == (15, 5, 47, 52)


def test_torso_box_rejects_low_confidence_joint():
    keypoints = [[0.0, 0.0, 1.0] for _ in range(17)]
    keypoints[5] = [20.0, 10.0, 0.9]
    keypoints[6] = [40.0, 12.0, 0.2]
    keypoints[11] = [22.0, 50.0, 0.9]
    keypoints[12] = [42.0, 52.0, 0.9]

    assert _torso_box((100, 80, 3), keypoints) is None


def test_pose_input_filters_legibility(tmp_path):
    import cv2

    image = np.zeros((30, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "a.jpg"), image)
    cv2.imwrite(str(tmp_path / "b.jpg"), image)

    payload, legible = _pose_input_payload(
        [{"filename": "a.jpg"}, {"filename": "b.jpg"}],
        input_dir=tmp_path,
        scores={"a.jpg": 0.8, "b.jpg": 0.2},
        threshold=0.5,
    )

    assert legible == {"a.jpg"}
    assert len(payload["images"]) == 1
    assert len(payload["annotations"]) == 1


def _source():
    return {
        "tracklet_id": "trk-1",
        "detection_id": "det-1",
        "sample_id": "sample-1",
        "frame_index": 10,
        "timestamp_s": 2.5,
        "number_roi_ref": "roi.jpg",
        "readiness_score": 0.9,
        "rank_within_tracklet": 1,
    }


def test_prediction_accepts_valid_reference_output():
    row = _prediction_row(
        _source(),
        legibility_score=0.92,
        legible=True,
        pose_crop=True,
        str_row={
            "label": "10",
            "confidence": [0.9, 0.8, 0.95],
        },
    )

    assert row["parsed_number"] == 10
    assert row["raw_text"] == "10"
    assert row["benchmark_status"] == "recognized"
    assert row["ocr_confidence"] == pytest.approx(0.684)


def test_rejected_raw_text_cannot_leak_downstream():
    row = _prediction_row(
        _source(),
        legibility_score=0.92,
        legible=True,
        pose_crop=True,
        str_row={
            "label": "S9",
            "confidence": [0.99, 0.99],
        },
    )

    assert row["parsed_number"] is None
    assert row["raw_text"] is None
    assert row["audit_raw_text"] == "S9"
    assert row["benchmark_status"] == "str_invalid"


def test_prediction_marks_illegible():
    row = _prediction_row(
        _source(),
        legibility_score=0.2,
        legible=False,
        pose_crop=False,
        str_row=None,
    )

    assert row["parsed_number"] is None
    assert row["benchmark_status"] == "illegible"


def test_env_run_prefix_is_micromamba_compatible():
    assert _env_run_prefix(
        "/home/user/.local/bin/micromamba",
        "vitpose",
    ) == [
        "/home/user/.local/bin/micromamba",
        "run",
        "-n",
        "vitpose",
    ]


def test_checkpoint_issue_rejects_tiny_html_like_file(tmp_path):
    path = tmp_path / "vitpose-h.pth"
    path.write_text("<html>not a checkpoint</html>", encoding="utf-8")

    issue = _checkpoint_issue(
        path,
        min_bytes=1_000_000,
        require_zip=True,
    )

    assert issue is not None
    assert "too small" in issue


def test_checkpoint_issue_accepts_large_enough_zip(tmp_path):
    import zipfile

    path = tmp_path / "checkpoint.pth"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data.pkl", b"x" * 2000)

    assert _checkpoint_issue(
        path,
        min_bytes=100,
        require_zip=True,
    ) is None
