from __future__ import annotations

import numpy as np
import pytest

from tracking.appearance.models import (
    AppearanceDescriptor,
    AppearanceSample,
)
from tracking.appearance.reid_benchmark import (
    ReIDBenchmarkConfig,
    benchmark_reid,
    roc_auc,
)
from tracking.tracklets import (
    Tracklet,
    TrackletObservation,
    TrackletSet,
)


MODEL = "synthetic-reid-v1"


def obs(detection_id, frame, time_s, x, z):
    return TrackletObservation(
        detection_id=detection_id,
        frame_index=frame,
        timestamp_s=time_s,
        x=x,
        z=z,
        confidence=0.9,
    )


def trk(tracklet_id, prefix, xs):
    observations = tuple(
        obs(
            f"{prefix}-{index}",
            index * 3,
            float(index),
            x,
            0.0,
        )
        for index, x in enumerate(xs)
    )
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=observations,
    )


def sample(tracklet_id, detection_id, frame, time_s, vector, quality=0.9):
    return AppearanceSample(
        sample_id=f"sample-{detection_id}",
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        frame_index=frame,
        timestamp_s=time_s,
        quality=quality,
        descriptors=(
            AppearanceDescriptor(
                name="reid",
                model_id=MODEL,
                values=tuple(vector),
                metric="cosine",
            ),
        ),
        crop_ref=f"diagnostics/{detection_id}.jpg",
    )


def fixture():
    # A and B coexist at every frame and stay > 5m apart.
    tracklet_a = trk(
        "trk-a",
        "a",
        [0.0, 0.2, 0.4, 0.6],
    )
    tracklet_b = trk(
        "trk-b",
        "b",
        [10.0, 10.2, 10.4, 10.6],
    )
    tracklets = TrackletSet(
        tracklets=(tracklet_a, tracklet_b)
    )

    samples = []
    for index in range(4):
        # Same identity vectors stay close to their axis.
        samples.append(
            sample(
                "trk-a",
                f"a-{index}",
                index * 3,
                float(index),
                (1.0, 0.03 * index),
            )
        )
        samples.append(
            sample(
                "trk-b",
                f"b-{index}",
                index * 3,
                float(index),
                (0.03 * index, 1.0),
            )
        )

    return tuple(samples), tracklets


def test_roc_auc_perfect_and_tied():
    assert roc_auc([0.9, 1.0], [0.0, 0.1]) == pytest.approx(1.0)
    assert roc_auc([0.5], [0.5]) == pytest.approx(0.5)
    assert roc_auc([], [0.1]) is None


def test_benchmark_counts_proxy_pairs():
    samples, tracklets = fixture()
    result = benchmark_reid(
        samples,
        tracklets,
        model_id=MODEL,
    )

    # C(4, 2) per tracklet.
    assert (
        result["sample_level"]["same_tracklet_positive"]["count"]
        == 12
    )
    # One A/B pair per shared selected frame.
    assert (
        result["sample_level"]["same_frame_negative"]["count"]
        == 4
    )
    assert result["tracklet_level"]["split_positive"]["count"] == 2
    assert (
        result["tracklet_level"]["concurrent_negative"]["count"]
        == 1
    )


def test_benchmark_clean_fixture_has_high_auc_and_positive_margin():
    samples, tracklets = fixture()
    result = benchmark_reid(
        samples,
        tracklets,
        model_id=MODEL,
    )

    assert result["sample_level"]["roc_auc"] > 0.95
    assert result["tracklet_level"]["roc_auc"] > 0.95
    assert (
        result["tracklet_level"][
            "p10_positive_minus_p95_negative"
        ]
        > 0.5
    )


def test_benchmark_hard_examples_are_exposed():
    samples, tracklets = fixture()
    result = benchmark_reid(
        samples,
        tracklets,
        model_id=MODEL,
        config=ReIDBenchmarkConfig(
            hard_example_count=1,
        ),
    )

    assert len(
        result["sample_level"]["weakest_positive_examples"]
    ) == 1
    assert len(
        result["sample_level"]["hardest_negative_examples"]
    ) == 1


def test_benchmark_rejects_missing_tracklet():
    samples, tracklets = fixture()
    broken = AppearanceSample(
        sample_id="broken",
        tracklet_id="missing",
        detection_id="x",
        frame_index=0,
        timestamp_s=0.0,
        quality=1.0,
        descriptors=(
            AppearanceDescriptor(
                name="reid",
                model_id=MODEL,
                values=(1.0, 0.0),
            ),
        ),
    )

    with pytest.raises(ValueError, match="missing tracklet"):
        benchmark_reid(
            (*samples, broken),
            tracklets,
            model_id=MODEL,
        )


def test_negative_requires_spatial_separation():
    samples, tracklets = fixture()
    result = benchmark_reid(
        samples,
        tracklets,
        model_id=MODEL,
        config=ReIDBenchmarkConfig(
            negative_min_separation_m=20.0,
        ),
    )

    assert (
        result["sample_level"]["same_frame_negative"]["count"]
        == 0
    )
    assert (
        result["tracklet_level"]["concurrent_negative"]["count"]
        == 0
    )
