from __future__ import annotations

import numpy as np
import pytest

from experiments.tracking.run_v1_jersey_number_parseq_regions import (
    DEFAULT_REGIONS,
    RegionSpec,
    _choose_region_consensus,
    _crop_fractional,
    _region_candidate,
)


def test_default_regions_are_valid_and_unique():
    assert len(DEFAULT_REGIONS) >= 4
    assert len({item.name for item in DEFAULT_REGIONS}) == len(DEFAULT_REGIONS)


def test_crop_fractional_extracts_expected_shape():
    image = np.zeros((100, 80, 3), dtype=np.uint8)
    region = RegionSpec(
        "test",
        0.25,
        0.20,
        0.75,
        0.60,
    )

    crop = _crop_fractional(image, region)

    assert crop.shape == (40, 40, 3)


def test_region_candidate_accepts_variant_agreement():
    prediction = _region_candidate(
        [
            {
                "variant": "original",
                "raw_text": "10",
                "parsed_number": 10,
                "ocr_confidence": 0.75,
            },
            {
                "variant": "x2",
                "raw_text": "10",
                "parsed_number": 10,
                "ocr_confidence": 0.71,
            },
            {
                "variant": "x4",
                "raw_text": None,
                "parsed_number": None,
                "ocr_confidence": 0.0,
            },
        ]
    )

    assert prediction["parsed_number"] == 10
    assert prediction["variant_support_count"] == 2
    assert prediction["accepted_within_region"] is True


def test_region_candidate_allows_exceptionally_confident_single_variant():
    prediction = _region_candidate(
        [
            {
                "variant": "original",
                "raw_text": "9",
                "parsed_number": 9,
                "ocr_confidence": 0.95,
            }
        ]
    )

    assert prediction["accepted_within_region"] is True


def _region(
    *,
    name,
    number,
    confidence,
    variants,
    consensus=None,
    accepted=True,
):
    return {
        "region": name,
        "raw_text": None if number is None else str(number),
        "parsed_number": number,
        "ocr_confidence": confidence,
        "chosen_variant": "x4",
        "variant_support_count": variants,
        "variant_consensus_score": (
            confidence if consensus is None else consensus
        ),
        "accepted_within_region": accepted,
    }


def test_region_consensus_accepts_repeated_number():
    chosen = _choose_region_consensus(
        [
            _region(
                name="torso_wide",
                number=10,
                confidence=0.88,
                variants=2,
            ),
            _region(
                name="torso_core",
                number=10,
                confidence=0.81,
                variants=2,
            ),
            _region(
                name="torso_left",
                number=7,
                confidence=0.61,
                variants=2,
            ),
        ]
    )

    assert chosen["parsed_number"] == 10
    assert chosen["region_support_count"] == 2
    assert chosen["region_consensus_accepted"] is True
    assert chosen["raw_text"] == "10"


def test_region_consensus_rejects_weak_single_region():
    chosen = _choose_region_consensus(
        [
            _region(
                name="torso_wide",
                number=7,
                confidence=0.75,
                variants=2,
            )
        ]
    )

    assert chosen["parsed_number"] is None
    assert chosen["raw_text"] is None
    assert chosen["region_consensus_accepted"] is False


def test_region_consensus_can_accept_exceptional_single_region():
    chosen = _choose_region_consensus(
        [
            _region(
                name="torso_core",
                number=9,
                confidence=0.98,
                variants=5,
                consensus=1.0,
            )
        ]
    )

    assert chosen["parsed_number"] == 9
    assert chosen["raw_text"] == "9"


def test_rejected_candidate_cannot_leak_numeric_raw_text_downstream():
    chosen = _choose_region_consensus(
        [
            _region(
                name="torso_wide",
                number=22,
                confidence=0.80,
                variants=2,
            )
        ]
    )

    assert chosen["parsed_number"] is None
    assert chosen["raw_text"] is None
    assert chosen["rejected_best_number"] == 22


def test_no_region_candidate_is_explicit_unknown():
    chosen = _choose_region_consensus(
        [
            _region(
                name="torso_wide",
                number=None,
                confidence=0.0,
                variants=0,
                accepted=False,
            )
        ]
    )

    assert chosen["parsed_number"] is None
    assert chosen["raw_text"] is None
    assert chosen["rejection_reason"] == "no_region_candidate"
