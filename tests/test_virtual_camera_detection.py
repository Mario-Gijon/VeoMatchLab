"""Tests for robust same-frame six-camera association.

CHATGPT DELIVERY MARKER: TEST_VIRTUAL_CAMERA_DETECTION_V6

These tests do not load model weights or require CUDA.

Important invariants:
- X/Z proximity alone must never merge players.
- One-to-one matching must prevent many-to-one collapse.
- A clipped torso in one zoom may match a full-body observation in another
  using ORIGINAL Veo source containment, even when bbox-bottom X/Z is poor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.virtual_camera_coverage.association import (  # noqa: E402
    Observation,
    SourceLensEvidence,
    associate_observations,
    candidate_edge,
    observation_conflicting_measurements,
    observation_matching_measurements,
    observations_share_identity_evidence,
    promote_image_confirmed_rescues,
    one_to_one_matches_for_view_pair,
    rectangle_iou,
    rectangle_overlap_metrics,
)
from experiments.virtual_camera_coverage.coverage import (  # noqa: E402
    AsymmetricOwnershipLayout,
)

from experiments.virtual_camera_coverage.detect import (  # noqa: E402
    PersonDetection,
    _candidate_full_box,
    _clamped_crop_bounds,
    detection_metrics,
    predict_persons,
)


class _RawPrediction:
    def __init__(self):
        self.xyxy = np.asarray(
            [
                [10.0, 20.0, 30.0, 70.0],
                [1.0, 2.0, 1.0, 8.0],
                [40.0, 10.0, 60.0, 50.0],
            ],
            dtype=np.float64,
        )
        self.confidence = np.asarray(
            [0.91, 0.80, 0.75],
            dtype=np.float64,
        )


class _FakeBackend:
    def _predict_one(self, image_rgb, *, threshold):
        assert image_rgb.shape == (100, 120, 3)
        assert threshold == 0.30
        return _RawPrediction()

    @staticmethod
    def _class_names(raw):
        return ["person", "person", "ball"]


def _source(
    *,
    lens: str = "left",
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    anchor_x: float,
    anchor_y: float,
) -> SourceLensEvidence:
    return SourceLensEvidence(
        lens=lens,
        x_min=x1,
        y_min=y1,
        x_max=x2,
        y_max=y2,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
        sample_count=49,
    )


def _observation(
    observation_id: int,
    view_name: str,
    *,
    world_x: float,
    world_z: float,
    source: SourceLensEvidence,
    confidence: float = 0.9,
    truncated_top: bool = False,
    truncated_bottom: bool = False,
    truncated_left: bool = False,
    truncated_right: bool = False,
    is_owner: bool = False,
) -> Observation:
    band = "far" if "far" in view_name else "near"

    return Observation(
        observation_id=observation_id,
        frame_number=100,
        time_s=3.33,
        view_name=view_name,
        band=band,
        index_in_band=0,
        xyxy=(100.0, 100.0, 140.0, 200.0),
        confidence=confidence,
        foot_x=120.0,
        foot_y=199.5,
        world_x=world_x,
        world_z=world_z,
        inside_strict_field=True,
        inside_allowed_field=True,
        owner_cell=(
            view_name.removeprefix("tight_")
            if is_owner
            else None
        ),
        is_owner=is_owner,
        source_lenses=(source,),
        truncated_top=truncated_top,
        truncated_bottom=truncated_bottom,
        truncated_left=truncated_left,
        truncated_right=truncated_right,
        status="accepted",
    )


def _layout() -> AsymmetricOwnershipLayout:
    return AsymmetricOwnershipLayout(
        reference_lon=0.0,
        near_z=0.0,
        far_z=60.0,
        depth_sign=1.0,
        total_depth_m=60.0,
        split_depth_m=30.0,
        near_count=2,
        far_count=4,
        near_lon_edges=(-1.0, 0.0, 1.0),
        far_lon_edges=(-1.0, -0.5, 0.0, 0.5, 1.0),
    )


def test_predict_persons_filters_invalid_boxes_and_non_person_classes():
    backend = _FakeBackend()
    image = np.zeros(
        (100, 120, 3),
        dtype=np.uint8,
    )

    detections = predict_persons(
        backend,
        image,
        threshold=0.30,
    )

    assert len(detections) == 1
    assert detections[0].xyxy == (
        10.0,
        20.0,
        30.0,
        70.0,
    )
    assert detections[0].confidence == 0.91
    assert detections[0].width_px == 20.0
    assert detections[0].height_px == 50.0
    assert detections[0].area_px2 == 1000.0


def test_detection_metrics_empty():
    metrics = detection_metrics([])

    assert metrics["count"] == 0
    assert metrics["mean_confidence"] is None
    assert metrics["median_bbox_height_px"] is None


def test_detection_metrics_are_descriptive_per_view_metrics():
    detections = [
        PersonDetection(
            (0.0, 0.0, 10.0, 20.0),
            0.50,
        ),
        PersonDetection(
            (0.0, 0.0, 20.0, 40.0),
            0.90,
        ),
    ]

    metrics = detection_metrics(
        detections
    )

    assert metrics["count"] == 2
    assert metrics["mean_confidence"] == 0.70
    assert metrics["median_confidence"] == 0.70
    assert metrics["mean_bbox_width_px"] == 15.0
    assert metrics["median_bbox_width_px"] == 15.0
    assert metrics["mean_bbox_height_px"] == 30.0
    assert metrics["median_bbox_height_px"] == 30.0
    assert metrics["mean_bbox_area_px2"] == 500.0
    assert metrics["median_bbox_area_px2"] == 500.0


def test_original_source_iou_detects_same_physical_image_region():
    a = _source(
        x1=100,
        y1=200,
        x2=140,
        y2=300,
        anchor_x=120,
        anchor_y=298,
    )

    b = _source(
        x1=103,
        y1=202,
        x2=142,
        y2=301,
        anchor_x=121,
        anchor_y=299,
    )

    assert rectangle_iou(a, b) > 0.75


def test_overlap_over_minimum_handles_partial_inside_full_bbox():
    full = _source(
        x1=100,
        y1=180,
        x2=140,
        y2=320,
        anchor_x=120,
        anchor_y=318,
    )

    torso = _source(
        x1=103,
        y1=180,
        x2=138,
        y2=245,
        anchor_x=120,
        anchor_y=244,
    )

    iou, iom, horizontal_iom, vertical_iom, center_dx = (
        rectangle_overlap_metrics(
            full,
            torso,
        )
    )

    assert iou < 0.60
    assert iom > 0.90
    assert horizontal_iom > 0.90
    assert vertical_iom > 0.90
    assert center_dx < 0.10


def test_close_world_positions_do_not_merge_without_source_pixel_overlap():
    """Two real players may be 20 cm apart: proximity must NOT identify them."""
    a = _observation(
        1,
        "tight_far_c2",
        world_x=10.00,
        world_z=20.00,
        source=_source(
            x1=100,
            y1=200,
            x2=125,
            y2=300,
            anchor_x=112,
            anchor_y=298,
        ),
    )

    b = _observation(
        2,
        "tight_far_c3",
        world_x=10.20,
        world_z=20.05,
        source=_source(
            x1=145,
            y1=200,
            x2=170,
            y2=300,
            anchor_x=157,
            anchor_y=298,
        ),
    )

    assert candidate_edge(a, b) is None


def test_same_source_region_can_match_across_two_virtual_views():
    a = _observation(
        1,
        "tight_far_c2",
        world_x=10.00,
        world_z=20.00,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    b = _observation(
        2,
        "tight_far_c3",
        world_x=10.12,
        world_z=20.08,
        source=_source(
            x1=102,
            y1=201,
            x2=141,
            y2=301,
            anchor_x=121,
            anchor_y=299,
        ),
    )

    edge = candidate_edge(
        a,
        b,
    )

    assert edge is not None
    assert edge.source_iou > 0.80
    assert edge.world_distance_m < 0.20
    assert edge.match_mode == "strict_iou"


def test_truncated_torso_can_match_full_body_without_trusting_bad_footpoint():
    """Regression for FAR torso + NEAR full-body duplicate at zoom boundary."""
    full = _observation(
        1,
        "tight_near_c2",
        world_x=12.0,
        world_z=8.0,
        source=_source(
            x1=500,
            y1=250,
            x2=545,
            y2=405,
            anchor_x=522,
            anchor_y=402,
        ),
    )

    torso = _observation(
        2,
        "tight_far_c3",
        # Wrong because bbox-bottom is the torso bottom, not the feet.
        world_x=13.5,
        world_z=11.5,
        source=_source(
            x1=503,
            y1=252,
            x2=542,
            y2=325,
            anchor_x=522,
            anchor_y=324,
        ),
        truncated_top=True,
    )

    edge = candidate_edge(
        full,
        torso,
    )

    assert edge is not None
    assert edge.match_mode == "truncated_source_containment"
    assert edge.source_iom > 0.90
    assert edge.source_horizontal_iom > 0.90
    assert edge.world_distance_m > 1.75


def test_side_by_side_partial_players_do_not_match():
    """Containment rescue still requires strong horizontal source alignment."""
    a = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        source=_source(
            x1=100,
            y1=200,
            x2=135,
            y2=300,
            anchor_x=118,
            anchor_y=298,
        ),
        truncated_top=True,
    )

    b = _observation(
        2,
        "tight_far_c3",
        world_x=10.2,
        world_z=20.1,
        source=_source(
            x1=128,
            y1=205,
            x2=163,
            y2=305,
            anchor_x=146,
            anchor_y=303,
        ),
        truncated_top=True,
    )

    assert candidate_edge(a, b) is None


def test_one_to_one_matching_prevents_many_to_one_collapse():
    a1 = _observation(
        1,
        "tight_far_c2",
        world_x=10.00,
        world_z=20.00,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
        confidence=0.95,
    )

    a2 = _observation(
        2,
        "tight_far_c2",
        world_x=10.15,
        world_z=20.05,
        source=_source(
            x1=104,
            y1=202,
            x2=144,
            y2=302,
            anchor_x=124,
            anchor_y=300,
        ),
        confidence=0.90,
    )

    b1 = _observation(
        3,
        "tight_far_c3",
        world_x=10.05,
        world_z=20.02,
        source=_source(
            x1=101,
            y1=201,
            x2=141,
            y2=301,
            anchor_x=121,
            anchor_y=299,
        ),
    )

    selected = one_to_one_matches_for_view_pair(
        [a1, a2],
        [b1],
    )

    assert len(selected) == 1
    assert selected[0].observation_b == 3
    assert selected[0].observation_a in {1, 2}


def test_primary_seeds_person_and_matching_rescue_only_adds_evidence():
    primary = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        is_owner=True,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    rescue = _observation(
        2,
        "tight_far_c3",
        world_x=10.1,
        world_z=20.1,
        is_owner=False,
        source=_source(
            x1=102,
            y1=201,
            x2=141,
            y2=301,
            anchor_x=121,
            anchor_y=299,
        ),
    )

    result = associate_observations(
        [primary, rescue],
        _layout(),
    )

    assert len(result.measurements) == 1
    measurement = result.measurements[0]
    assert measurement.creation_source == "primary"
    assert measurement.owner_supported is True
    assert set(measurement.observation_ids) == {1, 2}
    assert result.deferred_rescue_observation_ids == ()


def test_single_unmatched_rescue_is_deferred_not_emitted_as_duplicate_person():
    rescue = _observation(
        1,
        "tight_far_c3",
        world_x=10.0,
        world_z=20.0,
        is_owner=False,
        source=_source(
            x1=500,
            y1=200,
            x2=540,
            y2=300,
            anchor_x=520,
            anchor_y=298,
        ),
    )

    result = associate_observations(
        [rescue],
        _layout(),
    )

    assert len(result.measurements) == 0
    assert result.deferred_rescue_observation_ids == (1,)
    assert result.rescue_observation_count == 1


def test_two_corroborated_rescues_can_be_promoted_when_owner_missed():
    rescue_a = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        is_owner=False,
        source=_source(
            x1=300,
            y1=200,
            x2=340,
            y2=300,
            anchor_x=320,
            anchor_y=298,
        ),
    )

    rescue_b = _observation(
        2,
        "tight_far_c3",
        world_x=10.1,
        world_z=20.1,
        is_owner=False,
        source=_source(
            x1=302,
            y1=201,
            x2=341,
            y2=301,
            anchor_x=321,
            anchor_y=299,
        ),
    )

    result = associate_observations(
        [rescue_a, rescue_b],
        _layout(),
    )

    assert len(result.measurements) == 1
    measurement = result.measurements[0]
    assert measurement.creation_source == "promoted_rescue"
    assert measurement.owner_supported is False
    assert set(measurement.observation_ids) == {1, 2}
    assert result.promoted_rescue_measurement_count == 1
    assert result.deferred_rescue_observation_ids == ()


def test_rescue_cannot_bridge_two_distinct_primary_people():
    primary_left = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        is_owner=True,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    rescue_middle = _observation(
        2,
        "tight_far_c3",
        world_x=10.1,
        world_z=20.1,
        is_owner=False,
        source=_source(
            x1=102,
            y1=201,
            x2=141,
            y2=301,
            anchor_x=121,
            anchor_y=299,
        ),
    )

    # Deliberately source-similar to the rescue too. The rescue may attach to
    # one seed, but must never merge the two PRIMARY components.
    primary_right = _observation(
        3,
        "tight_far_c4",
        world_x=10.2,
        world_z=20.2,
        is_owner=True,
        source=_source(
            x1=104,
            y1=202,
            x2=143,
            y2=302,
            anchor_x=123,
            anchor_y=300,
        ),
    )

    result = associate_observations(
        [
            primary_left,
            rescue_middle,
            primary_right,
        ],
        _layout(),
    )

    assert len(result.measurements) == 2
    assert all(
        measurement.creation_source == "primary"
        for measurement in result.measurements
    )

    represented = [
        set(measurement.observation_ids)
        for measurement in result.measurements
    ]

    # The rescue can belong to only one of the two people.
    assert sum(
        2 in ids
        for ids in represented
    ) == 1
    assert any(1 in ids for ids in represented)
    assert any(3 in ids for ids in represented)


def test_ambiguous_rescue_group_near_primary_is_deferred_not_promoted():
    primary = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        is_owner=True,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    # These two rescues corroborate each other but intentionally fail the
    # strict primary edge because their feet/XZ are poor. Their source region
    # still overlaps the primary enough to be ambiguous, so they must defer.
    rescue_a = _observation(
        2,
        "tight_near_c1",
        world_x=14.5,
        world_z=24.0,
        is_owner=False,
        truncated_top=True,
        source=_source(
            x1=105,
            y1=205,
            x2=138,
            y2=250,
            anchor_x=121,
            anchor_y=249,
        ),
    )

    rescue_b = _observation(
        3,
        "tight_near_c2",
        world_x=14.7,
        world_z=24.2,
        is_owner=False,
        truncated_top=True,
        source=_source(
            x1=106,
            y1=206,
            x2=139,
            y2=251,
            anchor_x=122,
            anchor_y=250,
        ),
    )

    result = associate_observations(
        [primary, rescue_a, rescue_b],
        _layout(),
    )

    assert len(result.measurements) == 1
    assert result.measurements[0].creation_source == "primary"
    assert set(result.deferred_rescue_observation_ids) == {2, 3}


def test_image_confirmed_promotion_turns_only_requested_deferred_into_person():
    rescue_a = _observation(
        1,
        "tight_far_c3",
        world_x=10.0,
        world_z=20.0,
        is_owner=False,
        source=_source(
            x1=300,
            y1=200,
            x2=340,
            y2=300,
            anchor_x=320,
            anchor_y=298,
        ),
    )

    rescue_b = _observation(
        2,
        "tight_near_c2",
        world_x=30.0,
        world_z=25.0,
        is_owner=False,
        source=_source(
            x1=700,
            y1=200,
            x2=740,
            y2=300,
            anchor_x=720,
            anchor_y=298,
        ),
    )

    result = associate_observations(
        [rescue_a, rescue_b],
        _layout(),
    )

    assert len(result.measurements) == 0
    assert set(result.deferred_rescue_observation_ids) == {1, 2}

    promoted = promote_image_confirmed_rescues(
        result,
        [2],
    )

    assert len(promoted.measurements) == 1
    measurement = promoted.measurements[0]
    assert measurement.creation_source == "image_confirmed_rescue"
    assert measurement.observation_ids == (2,)
    assert promoted.image_confirmed_rescue_measurement_count == 1
    assert promoted.deferred_rescue_observation_ids == (1,)


def test_image_confirmed_promotion_rejects_non_deferred_observation():
    primary = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        is_owner=True,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    result = associate_observations(
        [primary],
        _layout(),
    )

    try:
        promote_image_confirmed_rescues(
            result,
            [1],
        )
    except ValueError:
        pass
    else:
        raise AssertionError(
            "non-deferred PRIMARY must not be image-promoted"
        )


def test_source_conflict_guard_detects_rescue_duplicate_of_existing_person():
    primary = _observation(
        1,
        "tight_far_c2",
        world_x=10.0,
        world_z=20.0,
        is_owner=True,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    result = associate_observations(
        [primary],
        _layout(),
    )

    rescue = _observation(
        2,
        "tight_near_c1",
        world_x=12.5,
        world_z=24.0,
        is_owner=False,
        source=_source(
            x1=105,
            y1=205,
            x2=138,
            y2=260,
            anchor_x=121,
            anchor_y=259,
        ),
    )

    conflicts = observation_conflicting_measurements(
        rescue,
        result,
    )

    assert conflicts == (1,)


def test_targeted_crop_bounds_keep_expected_player_inside_image():
    image = np.zeros(
        (720, 1280, 3),
        dtype=np.uint8,
    )

    crop = _clamped_crop_bounds(
        image,
        center_x=1240.0,
        top_y=80.0,
        foot_y=130.0,
        expected_height_px=50.0,
    )

    assert crop is not None
    x1, y1, x2, y2 = crop

    assert 0 <= x1 < x2 <= 1280
    assert 0 <= y1 < y2 <= 720
    assert x1 <= 1240.0 <= x2
    assert y1 <= 80.0 <= y2
    assert y1 <= 130.0 <= y2


def test_candidate_full_box_maps_crop_coordinates_back_to_owner_view():
    detection = PersonDetection(
        xyxy=(10.0, 20.0, 30.0, 70.0),
        confidence=0.9,
    )

    assert _candidate_full_box(
        detection,
        (100, 200, 400, 500),
    ) == (
        110.0,
        220.0,
        130.0,
        270.0,
    )


def test_loose_ambiguity_does_not_automatically_mean_strict_identity():
    """Regression: loose source overlap must not veto owner-crop verification."""
    primary = _observation(
        1,
        "tight_far_c4",
        world_x=46.0,
        world_z=3.0,
        is_owner=True,
        source=_source(
            x1=100,
            y1=200,
            x2=140,
            y2=300,
            anchor_x=120,
            anchor_y=298,
        ),
    )

    result = associate_observations(
        [primary],
        _layout(),
    )

    ambiguous_rescue = _observation(
        2,
        "tight_near_c2",
        world_x=46.5,
        world_z=3.2,
        is_owner=False,
        source=_source(
            x1=116,
            y1=230,
            x2=156,
            y2=330,
            anchor_x=136,
            anchor_y=248,
        ),
    )

    # The deliberately loose ambiguity guard sees overlap...
    assert observation_conflicting_measurements(
        ambiguous_rescue,
        result,
    ) == (1,)

    # ...but normal identity-grade evidence does not establish that they are
    # the same person. Therefore the owner crop must still be allowed to run.
    assert not observations_share_identity_evidence(
        ambiguous_rescue,
        primary,
    )


def test_strict_duplicate_check_works_with_same_owner_view_crop_detection():
    """A crop re-detection must match a full-image PRIMARY from the same view."""
    primary = _observation(
        1,
        "tight_far_c4",
        world_x=46.0,
        world_z=3.0,
        is_owner=True,
        source=_source(
            x1=500,
            y1=200,
            x2=540,
            y2=310,
            anchor_x=520,
            anchor_y=308,
        ),
    )

    result = associate_observations(
        [primary],
        _layout(),
    )

    # Synthetic second-pass detection from a crop of the SAME owner view.
    crop_candidate = _observation(
        99,
        "tight_far_c4",
        world_x=46.08,
        world_z=3.04,
        is_owner=True,
        source=_source(
            x1=502,
            y1=201,
            x2=541,
            y2=311,
            anchor_x=521,
            anchor_y=309,
        ),
    )

    assert observations_share_identity_evidence(
        crop_candidate,
        primary,
    )

    assert observation_matching_measurements(
        crop_candidate,
        result,
    ) == (1,)
