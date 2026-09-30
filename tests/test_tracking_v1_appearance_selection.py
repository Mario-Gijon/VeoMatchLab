from __future__ import annotations

import json
import math

import pytest

from tracking.appearance.selection import (
    AppearanceSelectionConfig,
    score_appearance_moment,
    select_appearance_moments,
)
from tracking.detections import CanonicalDetection, CanonicalDetectionFrame
from tracking.tracklets import Tracklet, TrackletObservation, TrackletSet


def detection(
    *,
    detection_id="det-1",
    frame_index=0,
    timestamp_s=0.0,
    confidence=0.95,
    x=0.0,
    z=0.0,
    bbox=(500.0, 200.0, 550.0, 320.0),
    source_height=110.0,
    spread=0.20,
    owner=True,
    multi=True,
):
    source_pixels = []
    if source_height is not None:
        source_pixels = [
            {
                "observation_id": 1,
                "view_name": "tight_far_c2",
                "lenses": [
                    {
                        "lens": "left",
                        "x_min": 100.0,
                        "y_min": 200.0,
                        "x_max": 130.0,
                        "y_max": 200.0 + source_height,
                        "anchor_x": 115.0,
                        "anchor_y": 200.0 + source_height,
                        "sample_count": 49,
                    }
                ],
            }
        ]

    return CanonicalDetection(
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        confidence=confidence,
        x=x,
        z=z,
        representative_view="tight_far_c2",
        virtual_bbox_xyxy=bbox,
        contributing_views=(
            ("tight_far_c2", "tight_near_c1")
            if multi
            else ("tight_far_c2",)
        ),
        metadata={
            "owner_supported": owner,
            "multi_view": multi,
            "position_spread_m": spread,
            "source_pixels": source_pixels,
        },
    )


def frame(
    selected,
    *others,
):
    return CanonicalDetectionFrame(
        frame_index=selected.frame_index,
        timestamp_s=selected.timestamp_s,
        detections=(selected, *others),
    )


def tracklet_from_detections(tracklet_id, detections):
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=tuple(
            TrackletObservation(
                detection_id=item.detection_id,
                frame_index=item.frame_index,
                timestamp_s=item.timestamp_s,
                x=item.x,
                z=item.z,
                confidence=item.confidence,
            )
            for item in detections
        ),
    )


def test_default_config_weights_sum_to_one():
    config = AppearanceSelectionConfig()
    total = (
        config.confidence_weight
        + config.visual_scale_weight
        + config.edge_clearance_weight
        + config.isolation_weight
        + config.geometry_weight
        + config.provenance_weight
    )
    assert total == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_samples_per_tracklet": 0},
        {"min_samples_if_available": 9},
        {"min_quality": 1.1},
        {"min_temporal_spacing_s": -0.1},
        {"confidence_good": 0.3},
        {"virtual_width_px": 0},
        {"visual_height_good_px": 10.0},
        {"provenance_weight": 0.10},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        AppearanceSelectionConfig(**kwargs)


def test_large_isolated_centered_detection_scores_high():
    target = detection()
    other = detection(
        detection_id="other",
        x=10.0,
        z=10.0,
    )

    result = score_appearance_moment(
        tracklet_id="trk-1",
        detection=target,
        frame=frame(target, other),
    )

    assert result.quality > 0.90
    assert result.visual_scale_score == 1.0
    assert result.isolation_score == 1.0
    assert result.geometry_score == 1.0
    assert result.provenance_score == 1.0


def test_small_crowded_edge_detection_scores_much_lower():
    target = detection(
        confidence=0.50,
        bbox=(0.0, 0.0, 10.0, 16.0),
        source_height=16.0,
        spread=2.0,
        owner=False,
        multi=False,
    )
    other = detection(
        detection_id="other",
        x=0.2,
        z=0.2,
    )

    result = score_appearance_moment(
        tracklet_id="trk-1",
        detection=target,
        frame=frame(target, other),
    )

    assert result.quality < 0.15
    assert result.visual_scale_score == 0.0
    assert result.edge_clearance_score == 0.0
    assert result.isolation_score == 0.0
    assert result.geometry_score == 0.0
    assert result.provenance_score == 0.0


def test_source_metadata_can_supply_visual_height_when_direct_source_bbox_is_none():
    target = detection(
        bbox=(500.0, 200.0, 510.0, 220.0),
        source_height=120.0,
    )

    result = score_appearance_moment(
        tracklet_id="trk-1",
        detection=target,
        frame=frame(target),
    )

    assert result.max_source_bbox_height_px == 120.0
    assert result.visual_scale_score == 1.0


def test_no_other_person_means_full_isolation_score():
    target = detection()

    result = score_appearance_moment(
        tracklet_id="trk-1",
        detection=target,
        frame=frame(target),
    )

    assert result.nearest_person_distance_m is None
    assert result.isolation_score == 1.0


def test_missing_spread_is_neutral_not_perfect():
    target = detection(spread=None)

    result = score_appearance_moment(
        tracklet_id="trk-1",
        detection=target,
        frame=frame(target),
    )

    assert result.geometry_score == 0.5


def sequence(count, *, dt=0.5):
    detections = []
    frames = []
    for index in range(count):
        item = detection(
            detection_id=f"det-{index:03d}",
            frame_index=index * 3,
            timestamp_s=index * dt,
            x=index * 0.1,
            z=0.0,
        )
        detections.append(item)
        frames.append(frame(item))
    return detections, frames


def test_selector_caps_samples_per_tracklet():
    detections, frames = sequence(30, dt=1.0)
    tracklet = tracklet_from_detections("trk-1", detections)

    result = select_appearance_moments(
        frames,
        TrackletSet(tracklets=(tracklet,)),
    )

    selection = result.selections[0]
    assert selection.candidate_count == 30
    assert selection.eligible_count == 30
    assert selection.selected_count == 8


def test_selector_temporally_spreads_long_tracklet():
    detections, frames = sequence(20, dt=1.0)
    tracklet = tracklet_from_detections("trk-1", detections)

    result = select_appearance_moments(
        frames,
        (tracklet,),
    )
    times = [
        moment.timestamp_s
        for moment in result.selections[0].moments
    ]

    assert len(times) == 8
    assert min(times) <= 2.0
    assert max(times) >= 16.0


def test_selector_respects_minimum_spacing_when_possible():
    detections, frames = sequence(12, dt=0.5)
    tracklet = tracklet_from_detections("trk-1", detections)

    config = AppearanceSelectionConfig(
        max_samples_per_tracklet=5,
        min_samples_if_available=3,
        min_temporal_spacing_s=1.0,
    )
    result = select_appearance_moments(
        frames,
        (tracklet,),
        config=config,
    )

    times = [
        moment.timestamp_s
        for moment in result.selections[0].moments
    ]
    assert len(times) == 5
    assert all(
        second - first >= 1.0 - 1e-9
        for first, second in zip(times, times[1:])
    )


def test_short_tracklet_fills_minimum_even_when_spacing_is_impossible():
    detections, frames = sequence(3, dt=0.1)
    tracklet = tracklet_from_detections("trk-1", detections)

    config = AppearanceSelectionConfig(
        min_temporal_spacing_s=1.0,
        min_samples_if_available=3,
    )
    result = select_appearance_moments(
        frames,
        (tracklet,),
        config=config,
    )

    assert result.selections[0].selected_count == 3


def test_singleton_can_produce_one_selected_moment():
    detections, frames = sequence(1)
    tracklet = tracklet_from_detections("trk-1", detections)

    result = select_appearance_moments(
        frames,
        (tracklet,),
    )

    assert result.selections[0].selected_count == 1


def test_low_quality_candidates_can_leave_explicit_empty_selection():
    bad = detection(
        confidence=0.40,
        bbox=(0.0, 0.0, 10.0, 12.0),
        source_height=12.0,
        spread=2.0,
        owner=False,
        multi=False,
    )
    crowded = detection(
        detection_id="other",
        x=0.1,
        z=0.1,
    )
    tracklet = tracklet_from_detections("trk-1", (bad,))

    result = select_appearance_moments(
        (frame(bad, crowded),),
        (tracklet,),
    )

    selection = result.selections[0]
    assert selection.candidate_count == 1
    assert selection.eligible_count == 0
    assert selection.selected_count == 0
    assert result.empty_selection_count == 1


def test_selector_ignores_canonical_detections_not_owned_by_tracklet():
    owned = detection(detection_id="owned")
    extra = detection(
        detection_id="extra",
        x=6.0,
        z=0.0,
    )
    tracklet = tracklet_from_detections("trk-1", (owned,))

    result = select_appearance_moments(
        (frame(owned, extra),),
        (tracklet,),
    )

    assert result.tracklet_count == 1
    assert result.candidate_count == 1


def test_selector_rejects_missing_detection_reference():
    owned = detection(detection_id="owned")
    missing = detection(detection_id="missing")
    tracklet = tracklet_from_detections("trk-1", (missing,))

    with pytest.raises(ValueError, match="references missing"):
        select_appearance_moments(
            (frame(owned),),
            (tracklet,),
        )


def test_selector_rejects_stale_tracklet_geometry():
    owned = detection(detection_id="owned")
    tracklet = Tracklet(
        tracklet_id="trk-1",
        observations=(
            TrackletObservation(
                detection_id="owned",
                frame_index=owned.frame_index,
                timestamp_s=owned.timestamp_s,
                x=owned.x + 1.0,
                z=owned.z,
                confidence=owned.confidence,
            ),
        ),
    )

    with pytest.raises(ValueError, match="X/Z does not match"):
        select_appearance_moments(
            (frame(owned),),
            (tracklet,),
        )


def test_selector_rejects_duplicate_detection_id_across_frames():
    first = detection(
        detection_id="same",
        frame_index=0,
        timestamp_s=0.0,
    )
    second = detection(
        detection_id="same",
        frame_index=3,
        timestamp_s=0.1,
    )

    with pytest.raises(ValueError, match="globally unique"):
        select_appearance_moments(
            (frame(first), frame(second)),
            (),
        )


def test_selector_rejects_non_monotonic_frames():
    first = detection(
        detection_id="a",
        frame_index=3,
        timestamp_s=0.1,
    )
    second = detection(
        detection_id="b",
        frame_index=0,
        timestamp_s=0.2,
    )

    with pytest.raises(ValueError, match="strictly frame ordered"):
        select_appearance_moments(
            (frame(first), frame(second)),
            (),
        )


def test_tracklet_selections_are_deterministically_sorted():
    a, frames_a = sequence(1)
    b = detection(
        detection_id="b",
        frame_index=3,
        timestamp_s=1.0,
    )
    frames = (
        frames_a[0],
        frame(b),
    )

    trk_b = tracklet_from_detections("trk-000002", (b,))
    trk_a = tracklet_from_detections("trk-000001", a)

    result = select_appearance_moments(
        frames,
        (trk_b, trk_a),
    )

    assert [
        item.tracklet_id
        for item in result.selections
    ] == ["trk-000001", "trk-000002"]


def test_selection_summary_is_json_safe_and_counts_everything():
    detections, frames = sequence(5, dt=1.0)
    tracklet = tracklet_from_detections("trk-1", detections)

    result = select_appearance_moments(
        frames,
        (tracklet,),
        config=AppearanceSelectionConfig(
            max_samples_per_tracklet=4,
        ),
    )
    summary = result.summary()

    assert summary["tracklet_count"] == 1
    assert summary["candidate_count"] == 5
    assert summary["selected_count"] == 4
    assert summary["empty_selection_count"] == 0
    assert summary["mean_selected_quality"] is not None
    json.dumps(result.to_dict(), allow_nan=False)


def test_selection_payload_has_no_identity_decisions():
    detections, frames = sequence(1)
    tracklet = tracklet_from_detections("trk-1", detections)

    payload = select_appearance_moments(
        frames,
        (tracklet,),
    ).to_dict()
    raw = json.dumps(payload)

    assert '"player_id"' not in raw
    assert '"identity_id"' not in raw
    assert '"team_id"' not in raw
    assert '"shirt_number"' not in raw
