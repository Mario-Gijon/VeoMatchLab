from __future__ import annotations

import json

import pytest

from tracking.appearance import (
    AppearanceDescriptor,
    AppearanceSample,
    AppearanceSet,
    TrackletAppearance,
)


def descriptor(
    *,
    name="reid",
    model_id="reid-test-v1",
    values=(1.0, 0.0, 0.5),
    metric="cosine",
):
    return AppearanceDescriptor(
        name=name,
        model_id=model_id,
        values=values,
        metric=metric,
    )


def sample(
    sample_id="sample-1",
    tracklet_id="trk-000001",
    detection_id="det-00000003-0001",
    frame_index=3,
    timestamp_s=0.1,
    quality=0.9,
    descriptors=None,
    **kwargs,
):
    return AppearanceSample(
        sample_id=sample_id,
        tracklet_id=tracklet_id,
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        quality=quality,
        descriptors=(
            tuple(descriptors)
            if descriptors is not None
            else (descriptor(),)
        ),
        **kwargs,
    )


def test_descriptor_normalizes_and_exposes_dimension_and_key():
    item = AppearanceDescriptor(
        name=" reid ",
        model_id=" model-a ",
        values=[1, 2.5, -3],
        metric=" cosine ",
    )

    assert item.name == "reid"
    assert item.model_id == "model-a"
    assert item.values == (1.0, 2.5, -3.0)
    assert item.metric == "cosine"
    assert item.dimension == 3
    assert item.key == ("reid", "model-a")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": ""},
        {"model_id": " "},
        {"metric": ""},
        {"values": ()},
        {"values": (1.0, float("nan"))},
        {"values": (1.0, float("inf"))},
    ],
)
def test_descriptor_rejects_invalid_contract(kwargs):
    base = dict(
        name="reid",
        model_id="model-a",
        values=(1.0,),
        metric="cosine",
    )
    base.update(kwargs)

    with pytest.raises(ValueError):
        AppearanceDescriptor(**base)


def test_descriptor_roundtrip():
    original = AppearanceDescriptor(
        name="jersey_hsv_hist",
        model_id="hist-v1",
        values=(0.2, 0.8),
        metric="bhattacharyya",
        metadata={"bins": 2},
    )

    rebuilt = AppearanceDescriptor.from_dict(original.to_dict())

    assert rebuilt.to_dict() == original.to_dict()


def test_sample_accepts_multiple_descriptor_families():
    item = sample(
        descriptors=(
            descriptor(),
            descriptor(
                name="jersey_hsv_hist",
                model_id="hist-v1",
                values=(0.25, 0.75),
                metric="bhattacharyya",
            ),
        ),
        view_name="tight_far_c2",
        source_lens="right",
        crop_ref="appearance/crops/trk-000001/sample-1.jpg",
        metadata={"visible_fraction": 0.95},
    )

    assert item.quality == 0.9
    assert len(item.descriptors) == 2
    assert item.descriptor("reid", "reid-test-v1") is not None
    assert item.descriptor("missing", "x") is None


@pytest.mark.parametrize("quality", [-0.01, 1.01, float("nan")])
def test_sample_rejects_invalid_quality(quality):
    with pytest.raises(ValueError):
        sample(quality=quality)


@pytest.mark.parametrize("source_lens", ["center", "", "LEFT"])
def test_sample_rejects_invalid_source_lens(source_lens):
    with pytest.raises(ValueError):
        sample(source_lens=source_lens)


@pytest.mark.parametrize(
    "crop_ref",
    [
        "/tmp/crop.jpg",
        "../crop.jpg",
        "appearance/../crop.jpg",
    ],
)
def test_sample_rejects_crop_outside_artifact_root(crop_ref):
    with pytest.raises(ValueError):
        sample(crop_ref=crop_ref)


def test_sample_requires_at_least_one_descriptor():
    with pytest.raises(ValueError, match="at least one descriptor"):
        sample(descriptors=())


def test_sample_allows_multiple_views_for_same_detection_via_unique_sample_id():
    first = sample(sample_id="sample-left", view_name="tight_far_c2")
    second = sample(sample_id="sample-right", view_name="tight_near_c2")

    profile = TrackletAppearance(
        tracklet_id="trk-000001",
        samples=(first, second),
    )

    assert profile.sample_count == 2
    assert first.detection_id == second.detection_id


def test_sample_rejects_duplicate_descriptor_key():
    duplicate = descriptor()

    with pytest.raises(ValueError, match="duplicate descriptor"):
        sample(descriptors=(duplicate, duplicate))


def test_tracklet_profile_may_explicitly_have_no_appearance_evidence():
    profile = TrackletAppearance(tracklet_id="trk-000099")

    assert profile.sample_count == 0
    assert profile.aggregate_descriptor_count == 0
    assert not profile.has_evidence


def test_tracklet_profile_sorts_samples_deterministically():
    later = sample(
        sample_id="later",
        frame_index=30,
        timestamp_s=1.0,
    )
    earlier = sample(
        sample_id="earlier",
        frame_index=3,
        timestamp_s=0.1,
    )

    profile = TrackletAppearance(
        tracklet_id="trk-000001",
        samples=(later, earlier),
    )

    assert [item.sample_id for item in profile.samples] == [
        "earlier",
        "later",
    ]


def test_tracklet_profile_rejects_sample_from_other_tracklet():
    with pytest.raises(ValueError, match="belongs to"):
        TrackletAppearance(
            tracklet_id="trk-000001",
            samples=(sample(tracklet_id="trk-000002"),),
        )


def test_tracklet_profile_rejects_duplicate_sample_id():
    first = sample(sample_id="same")
    second = sample(
        sample_id="same",
        detection_id="det-other",
        frame_index=6,
        timestamp_s=0.2,
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        TrackletAppearance(
            tracklet_id="trk-000001",
            samples=(first, second),
        )


def test_tracklet_profile_requires_stable_dimension_per_descriptor_key():
    first = sample(
        sample_id="s1",
        descriptors=(descriptor(values=(1.0, 2.0)),),
    )
    second = sample(
        sample_id="s2",
        frame_index=6,
        timestamp_s=0.2,
        descriptors=(descriptor(values=(1.0, 2.0, 3.0)),),
    )

    with pytest.raises(ValueError, match="dimension changed"):
        TrackletAppearance(
            tracklet_id="trk-000001",
            samples=(first, second),
        )


def test_aggregate_descriptor_must_match_sample_dimension():
    evidence = sample(
        descriptors=(descriptor(values=(1.0, 2.0, 3.0)),),
    )

    with pytest.raises(ValueError, match="aggregate descriptor dimension"):
        TrackletAppearance(
            tracklet_id="trk-000001",
            samples=(evidence,),
            aggregate_descriptors=(
                descriptor(values=(1.0, 2.0)),
            ),
        )


def test_tracklet_profile_can_mix_descriptor_families_and_aggregate_one():
    evidence = sample(
        descriptors=(
            descriptor(values=(1.0, 0.0)),
            descriptor(
                name="jersey_hsv_hist",
                model_id="hist-v1",
                values=(0.3, 0.7),
                metric="bhattacharyya",
            ),
        )
    )
    aggregate = descriptor(values=(0.8, 0.2))

    profile = TrackletAppearance(
        tracklet_id="trk-000001",
        samples=(evidence,),
        aggregate_descriptors=(aggregate,),
    )

    assert profile.has_evidence
    assert profile.aggregate_descriptor(
        "reid",
        "reid-test-v1",
    ) == aggregate


def test_appearance_set_enforces_unique_tracklets():
    first = TrackletAppearance(tracklet_id="trk-000001")
    second = TrackletAppearance(tracklet_id="trk-000001")

    with pytest.raises(ValueError, match="duplicate tracklet_id"):
        AppearanceSet(profiles=(first, second))


def test_appearance_set_enforces_globally_unique_sample_ids():
    first = TrackletAppearance(
        tracklet_id="trk-000001",
        samples=(sample(sample_id="global"),),
    )
    second = TrackletAppearance(
        tracklet_id="trk-000002",
        samples=(
            sample(
                sample_id="global",
                tracklet_id="trk-000002",
            ),
        ),
    )

    with pytest.raises(ValueError, match="globally unique"):
        AppearanceSet(profiles=(first, second))


def test_appearance_set_counts_and_lookup():
    populated = TrackletAppearance(
        tracklet_id="trk-000002",
        samples=(sample(tracklet_id="trk-000002"),),
        aggregate_descriptors=(descriptor(),),
    )
    empty = TrackletAppearance(tracklet_id="trk-000001")

    result = AppearanceSet(
        profiles=(populated, empty),
        metadata={"model_policy": "test"},
    )

    assert [item.tracklet_id for item in result.profiles] == [
        "trk-000001",
        "trk-000002",
    ]
    assert result.tracklet_count == 2
    assert result.sample_count == 1
    assert result.empty_profile_count == 1
    assert result.aggregate_descriptor_count == 1
    assert result.profile("trk-000002") is populated
    assert result.profile("trk-missing") is None


def test_appearance_set_json_roundtrip():
    original = AppearanceSet(
        profiles=(
            TrackletAppearance(
                tracklet_id="trk-000001",
                samples=(
                    sample(
                        view_name="tight_far_c1",
                        source_lens="left",
                    ),
                ),
                aggregate_descriptors=(descriptor(),),
                metadata={"usable": True},
            ),
        ),
        metadata={"schema": 1},
    )

    rebuilt = AppearanceSet.from_json(original.to_json())

    assert rebuilt.to_dict() == original.to_dict()
    json.dumps(rebuilt.to_dict(), allow_nan=False)


def test_contract_does_not_assign_identity_team_or_shirt_number():
    payload = AppearanceSet(
        profiles=(
            TrackletAppearance(
                tracklet_id="trk-000001",
                samples=(sample(),),
            ),
        ),
    ).to_dict()

    raw = json.dumps(payload)

    assert '"player_id"' not in raw
    assert '"identity_id"' not in raw
    assert '"team_id"' not in raw
    assert '"shirt_number"' not in raw
