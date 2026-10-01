from __future__ import annotations

from collections import Counter

from experiments.tracking.prepare_v1_jersey_legibility_dataset import (
    _decorate_rows,
    _quota_by_bucket,
    _readiness_bucket,
    _sample_rows,
)


def _row(index, tracklet, readiness, *, ref=False, ref_score=None):
    return {
        "sample_id": f"sample-{index}",
        "tracklet_id": tracklet,
        "detection_id": f"det-{index}",
        "frame_index": index,
        "timestamp_s": index / 10.0,
        "number_roi_ref": f"roi-{index}.jpg",
        "readiness_score": readiness,
        "rank_within_tracklet": 1,
        "reference_legibility_score": ref_score,
        "reference_legibility_accepted": ref,
        "reference_pose_crop_available": False,
        "reference_parsed_number": None,
        "sampling_bucket": (
            "high" if readiness >= 0.75
            else "low" if readiness <= 0.25
            else "mid"
        ),
    }


def test_readiness_bucket_boundaries():
    assert _readiness_bucket(
        0.8,
        low_threshold=0.25,
        high_threshold=0.75,
    ) == "high"
    assert _readiness_bucket(
        0.5,
        low_threshold=0.25,
        high_threshold=0.75,
    ) == "mid"
    assert _readiness_bucket(
        0.2,
        low_threshold=0.25,
        high_threshold=0.75,
    ) == "low"


def test_quota_sums_to_target():
    quota = _quota_by_bucket(320)
    assert sum(quota.values()) == 320
    assert quota["high"] > quota["mid"] > quota["low"]


def test_decorate_rows_never_turns_reference_into_label():
    rows = [
        {
            "sample_id": "s1",
            "tracklet_id": "t1",
            "detection_id": "d1",
            "frame_index": 1,
            "timestamp_s": 0.1,
            "number_roi_ref": "x.jpg",
            "readiness_score": 0.9,
            "rank_within_tracklet": 1,
        }
    ]
    reference = {
        "s1": {
            "reference_legibility_score": 0.99,
            "reference_legibility_accepted": True,
            "reference_pose_crop_available": True,
            "reference_parsed_number": 10,
        }
    }

    result = _decorate_rows(rows, reference)

    assert result[0]["reference_legibility_accepted"] is True
    assert "label" not in result[0]


def test_sample_is_deterministic_and_unique():
    rows = [
        _row(
            index,
            f"trk-{index // 2:03d}",
            readiness=(index % 10) / 10.0,
        )
        for index in range(80)
    ]

    a = _sample_rows(
        rows,
        target_count=30,
        seed=123,
        max_per_tracklet=2,
    )
    b = _sample_rows(
        rows,
        target_count=30,
        seed=123,
        max_per_tracklet=2,
    )

    assert [x["sample_id"] for x in a] == [
        x["sample_id"] for x in b
    ]
    assert len({x["sample_id"] for x in a}) == len(a)


def test_sample_respects_tracklet_cap():
    rows = []
    index = 0
    for tracklet in range(10):
        for view in range(5):
            rows.append(
                _row(
                    index,
                    f"trk-{tracklet}",
                    readiness=(view + 1) / 5.0,
                )
            )
            index += 1

    sampled = _sample_rows(
        rows,
        target_count=20,
        seed=7,
        max_per_tracklet=2,
    )
    counts = Counter(row["tracklet_id"] for row in sampled)

    assert len(sampled) == 20
    assert max(counts.values()) <= 2


def test_reference_positive_is_prioritized():
    rows = [
        _row(i, f"trk-{i}", readiness=0.5)
        for i in range(20)
    ]
    rows[17]["reference_legibility_accepted"] = True
    rows[17]["reference_legibility_score"] = 0.95

    sampled = _sample_rows(
        rows,
        target_count=5,
        seed=1,
        max_per_tracklet=2,
    )

    chosen = {row["sample_id"]: row for row in sampled}
    assert "sample-17" in chosen
    assert chosen["sample-17"]["sampling_reason"] == "reference-positive"


def test_sample_preserves_multiple_readiness_strata():
    rows = []
    for i in range(90):
        if i < 30:
            readiness = 0.1
            bucket = "low"
        elif i < 60:
            readiness = 0.5
            bucket = "mid"
        else:
            readiness = 0.9
            bucket = "high"
        row = _row(i, f"trk-{i}", readiness)
        row["sampling_bucket"] = bucket
        rows.append(row)

    sampled = _sample_rows(
        rows,
        target_count=30,
        seed=99,
        max_per_tracklet=2,
    )
    buckets = {row["sampling_bucket"] for row in sampled}

    assert buckets == {"high", "mid", "low"}
