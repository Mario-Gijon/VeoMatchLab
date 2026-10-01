from __future__ import annotations

import numpy as np
import pytest

from experiments.tracking.run_v1_jersey_number_parseq import (
    _prediction_row,
    _resolve_device,
    _sequence_confidence,
    _strict_parse_jersey_number,
)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("0", 0),
        ("9", 9),
        ("10", 10),
        ("99", 99),
        (" 17 ", 17),
        ("S9", None),
        ("I0", None),
        ("A17", None),
        ("17A", None),
        ("104", None),
        ("", None),
        (" ", None),
        (None, None),
        ("9.0", None),
        ("٠٩", None),
    ],
)
def test_strict_parse_jersey_number(text, expected):
    assert _strict_parse_jersey_number(text) == expected


def test_sequence_confidence_multiplies_token_probabilities():
    probabilities = np.asarray(
        [0.9, 0.8, 0.95],
        dtype=np.float64,
    )
    assert _sequence_confidence(
        probabilities
    ) == pytest.approx(0.684)


def test_sequence_confidence_empty_is_zero():
    assert _sequence_confidence(
        np.asarray([], dtype=np.float64)
    ) == 0.0


class _FakeCuda:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    def __init__(self, available: bool):
        self.cuda = _FakeCuda(available)


def test_resolve_device_auto_prefers_cuda():
    assert _resolve_device(
        "auto",
        _FakeTorch(True),
    ) == "cuda"


def test_resolve_device_auto_falls_back_to_cpu():
    assert _resolve_device(
        "auto",
        _FakeTorch(False),
    ) == "cpu"


def test_resolve_device_rejects_missing_cuda():
    with pytest.raises(
        RuntimeError,
        match="CUDA is unavailable",
    ):
        _resolve_device(
            "cuda",
            _FakeTorch(False),
        )


def test_prediction_row_matches_existing_schema_v2_contract():
    source = {
        "tracklet_id": "trk-000123",
        "detection_id": "det-0042",
        "sample_id": "sample-1",
        "frame_index": 42,
        "timestamp_s": 12.5,
        "number_roi_ref": "diagnostics/jersey_view/roi.jpg",
        "readiness_score": 0.91,
        "rank_within_tracklet": 1,
    }
    chosen = {
        "raw_text": "10",
        "parsed_number": 10,
        "ocr_confidence": 0.82,
        "chosen_variant": "x4_clahe",
        "variant_support_count": 3,
        "variant_consensus_score": 0.88,
    }
    variants = [
        {
            "variant": "original",
            "raw_text": "10",
            "parsed_number": 10,
            "ocr_confidence": 0.72,
        }
    ]

    row = _prediction_row(
        source,
        selection="strongest",
        preprocess_profile="jersey",
        backend_name="parseq-generic",
        chosen=chosen,
        variant_predictions=variants,
    )

    assert row["schema_version"] == 2
    assert row["backend"] == "parseq-generic"
    assert row["selection"] == "strongest"
    assert row["preprocess_profile"] == "jersey"
    assert row["tracklet_id"] == "trk-000123"
    assert row["detection_id"] == "det-0042"
    assert row["sample_id"] == "sample-1"
    assert row["frame_index"] == 42
    assert row["timestamp_s"] == pytest.approx(12.5)
    assert row["parsed_number"] == 10
    assert row["variant_support_count"] == 3
    assert len(row["variant_predictions"]) == 1
