from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np
import pytest

from tracking.reid_crops import (
    ReIDCrop,
    ReIDCropConfig,
    ReIDCropMetrics,
    angular_distance,
    candidate_from_detection,
    centred_spec_for_candidate,
    crop_quality,
    direction_angles,
    map_source_bbox_to_spec,
    normalize_person_crop,
    padded_crop_box,
    pixel_to_presentation_direction,
    presentation_direction_to_pixel,
    select_diverse_candidates,
    select_diverse_crops,
)


@dataclass(frozen=True)
class DummySpec:
    name: str = "tight_far_c1"
    longitude: float = 0.18
    latitude: float = -0.12
    vfov: float = math.radians(44.0)
    hfov: float = math.radians(70.0)


def test_pixel_ray_round_trip():
    spec = DummySpec()
    width, height = 1280, 720

    for x, y in (
        (100.0, 100.0),
        (640.0, 360.0),
        (1100.0, 600.0),
    ):
        ray = pixel_to_presentation_direction(
            spec,
            x,
            y,
            width=width,
            height=height,
        )
        projected = presentation_direction_to_pixel(
            spec,
            ray,
            width=width,
            height=height,
        )

        assert projected is not None
        assert projected[0] == pytest.approx(x, abs=1e-8)
        assert projected[1] == pytest.approx(y, abs=1e-8)


def test_direction_angles_returns_camera_centre_angles():
    spec = DummySpec()
    ray = pixel_to_presentation_direction(
        spec,
        639.5,
        359.5,
        width=1280,
        height=720,
    )

    longitude, latitude = direction_angles(ray)

    assert longitude == pytest.approx(spec.longitude, abs=1e-8)
    assert latitude == pytest.approx(spec.latitude, abs=1e-8)


def test_angular_distance_is_zero_for_same_ray():
    ray = np.asarray([0.2, -0.1, 1.0])
    assert angular_distance(ray, ray) == pytest.approx(0.0)


def test_candidate_scores_good_central_bbox_higher_than_edge_bbox():
    spec = DummySpec()
    cfg = ReIDCropConfig()

    central = candidate_from_detection(
        track_id=1,
        frame_index=10,
        timestamp=1.0,
        source_camera=spec.name,
        source_spec=spec,
        source_bbox_xyxy=(580, 250, 700, 650),
        detection_confidence=0.90,
        source_width=1280,
        source_height=720,
        config=cfg,
    )
    edge = candidate_from_detection(
        track_id=1,
        frame_index=20,
        timestamp=2.0,
        source_camera=spec.name,
        source_spec=spec,
        source_bbox_xyxy=(0, 250, 120, 650),
        detection_confidence=0.90,
        source_width=1280,
        source_height=720,
        config=cfg,
    )

    assert central.proposal_quality > edge.proposal_quality
    assert central.source_edge_score == pytest.approx(1.0)
    assert edge.source_edge_score == pytest.approx(0.0)


def test_centred_spec_uses_narrower_fov_for_small_player():
    spec = DummySpec()

    small = candidate_from_detection(
        track_id=1,
        frame_index=1,
        timestamp=0.0,
        source_camera=spec.name,
        source_spec=spec,
        source_bbox_xyxy=(620, 320, 660, 400),
        detection_confidence=0.9,
        source_width=1280,
        source_height=720,
    )
    big = candidate_from_detection(
        track_id=2,
        frame_index=2,
        timestamp=1.0,
        source_camera=spec.name,
        source_spec=spec,
        source_bbox_xyxy=(540, 160, 740, 650),
        detection_confidence=0.9,
        source_width=1280,
        source_height=720,
    )

    small_spec = centred_spec_for_candidate(
        spec,
        small,
        source_width=1280,
        source_height=720,
    )
    big_spec = centred_spec_for_candidate(
        spec,
        big,
        source_width=1280,
        source_height=720,
    )

    assert small_spec.vfov < big_spec.vfov
    assert math.degrees(small_spec.vfov) >= 3.5
    assert math.degrees(big_spec.vfov) <= 32.0


def test_mapped_bbox_is_near_centre_of_reid_view():
    spec = DummySpec()
    candidate = candidate_from_detection(
        track_id=1,
        frame_index=1,
        timestamp=0.0,
        source_camera=spec.name,
        source_spec=spec,
        source_bbox_xyxy=(570, 220, 710, 650),
        detection_confidence=0.9,
        source_width=1280,
        source_height=720,
    )
    target = centred_spec_for_candidate(
        spec,
        candidate,
        source_width=1280,
        source_height=720,
    )
    mapped = map_source_bbox_to_spec(
        spec,
        target,
        candidate.source_bbox_xyxy,
        source_width=1280,
        source_height=720,
        target_width=160,
        target_height=320,
    )

    cx = 0.5 * (mapped[0] + mapped[2])
    cy = 0.5 * (mapped[1] + mapped[3])

    assert cx == pytest.approx(79.5, abs=4.0)
    assert cy == pytest.approx(159.5, abs=8.0)


def test_padded_crop_box_reports_clipping():
    cfg = ReIDCropConfig()
    inside, inside_clip = padded_crop_box(
        (40, 40, 120, 280),
        width=160,
        height=320,
        config=cfg,
    )
    edge, edge_clip = padded_crop_box(
        (-10, 40, 70, 280),
        width=160,
        height=320,
        config=cfg,
    )

    assert inside_clip == pytest.approx(0.0)
    assert edge_clip > inside_clip
    assert inside[0] >= 0
    assert edge[0] == 0


def test_normalize_person_crop_preserves_requested_shape():
    image = np.full((180, 70, 3), 180, dtype=np.uint8)
    output = normalize_person_crop(
        image,
        output_width=128,
        output_height=256,
    )

    assert output.shape == (256, 128, 3)
    assert output.dtype == np.uint8
    assert output.max() == 180


def test_sharp_crop_scores_better_than_blurred_crop():
    checker = np.indices((200, 80)).sum(axis=0) % 2
    sharp = (checker * 255).astype(np.uint8)
    sharp = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)

    sharp_quality = crop_quality(
        sharp,
        player_height_px=180,
        clipped_fraction=0.0,
        valid_pixel_fraction=1.0,
        detection_confidence=0.9,
    )
    blur_quality = crop_quality(
        blurred,
        player_height_px=180,
        clipped_fraction=0.0,
        valid_pixel_fraction=1.0,
        detection_confidence=0.9,
    )

    assert sharp_quality[0] > blur_quality[0]
    assert sharp_quality[2] > blur_quality[2]


def test_candidate_selection_enforces_temporal_diversity():
    spec = DummySpec()
    candidates = []

    for frame, timestamp, confidence in (
        (0, 0.0, 0.99),
        (1, 0.1, 0.98),
        (15, 1.0, 0.90),
        (30, 2.0, 0.85),
    ):
        candidates.append(
            candidate_from_detection(
                track_id=1,
                frame_index=frame,
                timestamp=timestamp,
                source_camera=spec.name,
                source_spec=spec,
                source_bbox_xyxy=(580, 220, 700, 650),
                detection_confidence=confidence,
                source_width=1280,
                source_height=720,
            )
        )

    selected = select_diverse_candidates(
        candidates,
        max_crops=3,
        min_time_gap_seconds=0.75,
    )

    assert len(selected) == 3
    timestamps = [item.timestamp for item in selected]
    assert 0.0 in timestamps
    assert 0.1 not in timestamps
    assert 1.0 in timestamps
    assert 2.0 in timestamps


def _dummy_crop(frame: int, timestamp: float, quality: float) -> ReIDCrop:
    metrics = ReIDCropMetrics(
        track_id=1,
        frame_index=frame,
        timestamp=timestamp,
        source_camera="tight_far_c1",
        source_bbox_xyxy=(0, 0, 10, 20),
        detection_confidence=0.9,
        centred_longitude_deg=0.0,
        centred_latitude_deg=0.0,
        centred_vfov_deg=10.0,
        centred_hfov_deg=5.0,
        mapped_bbox_xyxy=(10, 10, 50, 200),
        crop_bbox_xyxy=(5, 5, 55, 205),
        player_height_px=190,
        clipped_fraction=0.0,
        valid_pixel_fraction=1.0,
        laplacian_variance=100.0,
        blur_score=0.8,
        size_score=1.0,
        clipping_score=1.0,
        quality=quality,
    )
    return ReIDCrop(
        image=np.zeros((256, 128, 3), dtype=np.uint8),
        metrics=metrics,
    )


def test_final_crop_selection_uses_final_quality_and_diversity():
    crops = (
        _dummy_crop(0, 0.0, 0.95),
        _dummy_crop(1, 0.1, 0.99),
        _dummy_crop(15, 1.0, 0.80),
        _dummy_crop(30, 2.0, 0.75),
    )

    selected = select_diverse_crops(
        crops,
        max_crops=3,
        min_time_gap_seconds=0.75,
    )

    frames = [item.metrics.frame_index for item in selected]
    assert 1 in frames
    assert 0 not in frames
    assert 15 in frames
    assert 30 in frames
