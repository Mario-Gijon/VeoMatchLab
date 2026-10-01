from __future__ import annotations

import pytest

from tracking.actors.reid_calibration import (
    ReIDCalibrationConfig,
    _roc_auc,
    build_reid_overlap_calibration,
)
from tracking.appearance.evidence import (
    AppearanceEvidenceSample,
)
from tracking.tracklets.models import (
    Tracklet,
    TrackletObservation,
)


def _tracklet(
    tracklet_id: str,
    rows: list[tuple[int, float, float, float]],
) -> Tracklet:
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=tuple(
            TrackletObservation(
                detection_id=f"{tracklet_id}-det-{i}",
                frame_index=frame,
                timestamp_s=time,
                x=x,
                z=z,
                confidence=0.95,
            )
            for i, (frame, time, x, z)
            in enumerate(rows)
        ),
    )


def _sample(
    tracklet_id: str,
    index: int,
    embedding: tuple[float, ...],
    *,
    timestamp: float | None = None,
) -> AppearanceEvidenceSample:
    return AppearanceEvidenceSample(
        sample_id=f"{tracklet_id}-s{index}",
        tracklet_id=tracklet_id,
        detection_id=f"{tracklet_id}-d{index}",
        frame_index=index,
        timestamp_s=(
            float(index)
            if timestamp is None
            else float(timestamp)
        ),
        quality=0.95,
        embedding=embedding,
    )


def test_auc_perfect_separation():
    assert _roc_auc(
        [0.8, 0.9],
        [0.1, 0.2],
    ) == pytest.approx(1.0)


def test_auc_ties_are_half_credit():
    assert _roc_auc(
        [0.5],
        [0.5],
    ) == pytest.approx(0.5)


def test_split_positive_uses_same_tracklet_halves():
    tracklet = _tracklet(
        "a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 0.1, 0.0),
        ],
    )
    samples = tuple(
        _sample(
            "a",
            i,
            (1.0, 0.02 * i),
        )
        for i in range(4)
    )

    result = build_reid_overlap_calibration(
        tracklets=[tracklet],
        samples=samples,
        calibration_config=ReIDCalibrationConfig(
            min_split_samples=4,
        ),
    )

    assert len(result.split_positive) == 1
    example = result.split_positive[0]
    assert example.tracklet_a == "a"
    assert example.tracklet_b == "a"
    assert example.cosine > 0.99


def test_simultaneous_far_pair_is_negative_proxy():
    a = _tracklet(
        "a",
        [(0, 0.0, 0.0, 0.0)],
    )
    b = _tracklet(
        "b",
        [(0, 0.0, 6.0, 0.0)],
    )

    samples = []
    for i in range(4):
        samples.append(
            _sample("a", i, (1.0, 0.0))
        )
        samples.append(
            _sample("b", i, (0.0, 1.0))
        )

    result = build_reid_overlap_calibration(
        tracklets=[a, b],
        samples=tuple(samples),
    )

    assert len(result.simultaneous_negative) == 1
    negative = result.simultaneous_negative[0]
    assert negative.tracklet_a == "a"
    assert negative.tracklet_b == "b"
    assert negative.cosine == pytest.approx(0.0)


def test_close_overlap_is_diagnostic_not_positive():
    a = _tracklet(
        "a",
        [
            (0, 0.0, 0.0, 0.0),
            (3, 0.1, 0.1, 0.0),
        ],
    )
    b = _tracklet(
        "b",
        [
            (0, 0.0, 0.3, 0.0),
            (3, 0.1, 0.4, 0.0),
        ],
    )

    samples = []
    for i in range(4):
        samples.append(
            _sample("a", i, (1.0, 0.0))
        )
        samples.append(
            _sample("b", i, (0.99, 0.01))
        )

    result = build_reid_overlap_calibration(
        tracklets=[a, b],
        samples=tuple(samples),
    )

    assert len(result.close_overlap) == 1
    assert (
        result.close_overlap[0]
        .evidence["used_as_positive"]
        is False
    )


def test_summary_builds_empirical_anchors():
    a = _tracklet(
        "a",
        [(0, 0.0, 0.0, 0.0)],
    )
    b = _tracklet(
        "b",
        [(0, 0.0, 6.0, 0.0)],
    )

    samples = []
    for i in range(4):
        samples.append(
            _sample("a", i, (1.0, 0.0))
        )
        samples.append(
            _sample("b", i, (0.0, 1.0))
        )

    result = build_reid_overlap_calibration(
        tracklets=[a, b],
        samples=tuple(samples),
    )
    summary = result.summary()

    assert summary["anchors"]["negative_p95"] == pytest.approx(0.0)
    assert summary["anchors"]["positive_p10"] == pytest.approx(1.0)
    assert (
        summary["anchors"][
            "p10_positive_minus_p95_negative"
        ]
        > 0.0
    )
    assert summary["anchors"]["ordered"] is True
    assert summary["identity_decisions"] is False


def test_unprofiled_tracklets_are_counted_and_skipped():
    a = _tracklet(
        "a",
        [(0, 0.0, 0.0, 0.0)],
    )
    b = _tracklet(
        "b",
        [(0, 0.0, 6.0, 0.0)],
    )

    samples = tuple(
        _sample("a", i, (1.0, 0.0))
        for i in range(4)
    )

    result = build_reid_overlap_calibration(
        tracklets=[a, b],
        samples=samples,
    )

    assert result.profiled_tracklet_count == 1
    assert result.unprofiled_tracklet_count == 1
    assert result.simultaneous_negative == ()
