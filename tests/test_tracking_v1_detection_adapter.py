from __future__ import annotations

from dataclasses import dataclass

import pytest

from tracking.detections.adapter import (
    canonical_detection_from_measurement,
    canonical_frame_from_association,
    canonical_frames_from_associations,
)


@dataclass(frozen=True)
class Lens:
    lens: str
    x_min: float = 10.0
    y_min: float = 20.0
    x_max: float = 30.0
    y_max: float = 80.0
    anchor_x: float = 20.0
    anchor_y: float = 80.0
    sample_count: int = 25

    def to_dict(self):
        return {
            "lens": self.lens,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "anchor_x": self.anchor_x,
            "anchor_y": self.anchor_y,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True)
class Observation:
    observation_id: int
    frame_number: int = 42
    time_s: float = 1.4
    view_name: str = "far_left"
    band: str = "far"
    xyxy: tuple[float, float, float, float] = (100, 200, 300, 700)
    confidence: float = 0.9
    owner_cell: str | None = "far_left"
    ownership_role: str = "primary"
    source_lenses: tuple[Lens, ...] = (Lens("left"),)


@dataclass(frozen=True)
class Measurement:
    measurement_id: int = 7
    observation_ids: tuple[int, ...] = (1,)
    primary_observation_id: int = 1
    representative_world_x: float | None = 34.0
    representative_world_z: float | None = 17.0
    fused_world_x: float | None = 34.5
    fused_world_z: float | None = 17.25
    position_spread_m: float | None = 0.2
    confidence: float = 0.87
    owner_supported: bool = True
    multi_view: bool = False
    creation_source: str = "primary"


@dataclass(frozen=True)
class Association:
    observations: tuple[Observation, ...]
    measurements: tuple[Measurement, ...]


def test_adapter_prefers_fused_world_position():
    observation = Observation(1)
    detection = canonical_detection_from_measurement(
        Measurement(),
        frame_index=42,
        timestamp_s=1.4,
        observations_by_id={1: observation},
    )

    assert detection is not None
    assert detection.x == 34.5
    assert detection.z == 17.25
    assert detection.detection_id == "det-00000042-0007"


def test_adapter_falls_back_to_representative_world_position():
    observation = Observation(1)
    measurement = Measurement(fused_world_x=None, fused_world_z=None)

    detection = canonical_detection_from_measurement(
        measurement,
        frame_index=42,
        timestamp_s=1.4,
        observations_by_id={1: observation},
    )

    assert detection is not None
    assert (detection.x, detection.z) == (34.0, 17.0)


def test_adapter_skips_measurement_without_any_world_position():
    observation = Observation(1)
    measurement = Measurement(
        fused_world_x=None,
        fused_world_z=None,
        representative_world_x=None,
        representative_world_z=None,
    )

    assert canonical_detection_from_measurement(
        measurement,
        frame_index=42,
        timestamp_s=1.4,
        observations_by_id={1: observation},
    ) is None


def test_adapter_maps_unambiguous_primary_lens_and_virtual_geometry():
    observation = Observation(1)
    detection = canonical_detection_from_measurement(
        Measurement(),
        frame_index=42,
        timestamp_s=1.4,
        observations_by_id={1: observation},
    )

    assert detection is not None
    assert detection.source_lens == "left"
    assert detection.source_bbox_xyxy == (10.0, 20.0, 30.0, 80.0)
    assert detection.source_footpoint_xy == (20.0, 80.0)
    assert detection.virtual_bbox_xyxy == (100.0, 200.0, 300.0, 700.0)
    assert detection.representative_view == "far_left"


def test_adapter_keeps_multi_lens_provenance_without_fake_single_lens():
    observation = Observation(
        1,
        source_lenses=(Lens("left"), Lens("right")),
    )
    detection = canonical_detection_from_measurement(
        Measurement(),
        frame_index=42,
        timestamp_s=1.4,
        observations_by_id={1: observation},
    )

    assert detection is not None
    assert detection.source_lens is None
    assert detection.source_bbox_xyxy is None
    assert detection.source_footpoint_xy is None
    assert len(detection.metadata["source_pixels"][0]["lenses"]) == 2


def test_adapter_preserves_unique_contributing_view_order():
    first = Observation(1, view_name="far_left")
    second = Observation(
        2,
        view_name="near_center",
        ownership_role="rescue",
        source_lenses=(Lens("left"),),
    )
    third = Observation(
        3,
        view_name="far_left",
        ownership_role="rescue",
        source_lenses=(Lens("left"),),
    )
    measurement = Measurement(
        observation_ids=(1, 2, 3),
        primary_observation_id=1,
        multi_view=True,
    )

    detection = canonical_detection_from_measurement(
        measurement,
        frame_index=42,
        timestamp_s=1.4,
        observations_by_id={1: first, 2: second, 3: third},
    )

    assert detection is not None
    assert detection.contributing_views == ("far_left", "near_center")
    assert detection.metadata["observation_ids"] == [1, 2, 3]


def test_adapter_rejects_missing_primary_provenance():
    with pytest.raises(ValueError, match="primary observation"):
        canonical_detection_from_measurement(
            Measurement(),
            frame_index=42,
            timestamp_s=1.4,
            observations_by_id={},
        )


def test_adapter_rejects_primary_not_listed_in_measurement():
    first = Observation(1)
    second = Observation(2)
    measurement = Measurement(
        observation_ids=(2,),
        primary_observation_id=1,
    )

    with pytest.raises(ValueError, match="primary observation is not part"):
        canonical_detection_from_measurement(
            measurement,
            frame_index=42,
            timestamp_s=1.4,
            observations_by_id={1: first, 2: second},
        )


def test_adapter_rejects_missing_contributing_observation():
    measurement = Measurement(
        observation_ids=(1, 2),
        primary_observation_id=1,
    )

    with pytest.raises(ValueError, match="contributing observation"):
        canonical_detection_from_measurement(
            measurement,
            frame_index=42,
            timestamp_s=1.4,
            observations_by_id={1: Observation(1)},
        )


def test_adapter_rejects_wrong_frame_or_timestamp_provenance():
    with pytest.raises(ValueError, match="belongs to frame"):
        canonical_detection_from_measurement(
            Measurement(),
            frame_index=42,
            timestamp_s=1.4,
            observations_by_id={1: Observation(1, frame_number=41)},
        )

    with pytest.raises(ValueError, match="timestamp"):
        canonical_detection_from_measurement(
            Measurement(),
            frame_index=42,
            timestamp_s=1.4,
            observations_by_id={1: Observation(1, time_s=1.5)},
        )


def test_frame_adapter_is_sorted_and_skips_no_position_measurements():
    observation_a = Observation(1)
    observation_b = Observation(2, view_name="far_right")
    valid_a = Measurement(
        measurement_id=9,
        observation_ids=(1,),
        primary_observation_id=1,
    )
    valid_b = Measurement(
        measurement_id=3,
        observation_ids=(2,),
        primary_observation_id=2,
    )
    invalid = Measurement(
        measurement_id=5,
        observation_ids=(1,),
        primary_observation_id=1,
        fused_world_x=None,
        fused_world_z=None,
        representative_world_x=None,
        representative_world_z=None,
    )

    frame = canonical_frame_from_association(
        Association(
            observations=(observation_a, observation_b),
            measurements=(valid_a, invalid, valid_b),
        ),
        frame_index=42,
        timestamp_s=1.4,
    )

    assert [item.detection_id for item in frame.detections] == [
        "det-00000042-0003",
        "det-00000042-0009",
    ]


def test_frame_adapter_rejects_duplicate_observation_or_measurement_ids():
    with pytest.raises(ValueError, match="duplicate observation_id"):
        canonical_frame_from_association(
            Association(
                observations=(Observation(1), Observation(1)),
                measurements=(),
            ),
            frame_index=42,
            timestamp_s=1.4,
        )

    with pytest.raises(ValueError, match="duplicate measurement_id"):
        canonical_frame_from_association(
            Association(
                observations=(Observation(1),),
                measurements=(Measurement(), Measurement()),
            ),
            frame_index=42,
            timestamp_s=1.4,
        )


def test_lazy_frame_adapter_requires_monotonic_rows():
    association = Association(observations=(), measurements=())

    frames = list(
        canonical_frames_from_associations(
            [
                (10, 1.0, association),
                (11, 1.1, association),
            ]
        )
    )
    assert [frame.frame_index for frame in frames] == [10, 11]

    with pytest.raises(ValueError, match="frame_index"):
        list(
            canonical_frames_from_associations(
                [
                    (11, 1.0, association),
                    (10, 1.1, association),
                ]
            )
        )
