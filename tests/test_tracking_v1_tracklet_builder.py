import pytest

from tracking.detections import CanonicalDetection, CanonicalDetectionFrame
from tracking.tracklets.builder import (
    ConservativeTrackletBuilder,
    TrackletBuilderConfig,
    build_tracklets,
)


FPS = 30.0


def _detection(
    frame_index: int,
    detection_id: str,
    x: float,
    z: float = 20.0,
    confidence: float = 0.95,
) -> CanonicalDetection:
    return CanonicalDetection(
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_s=frame_index / FPS,
        confidence=confidence,
        x=x,
        z=z,
    )


def _frame(frame_index: int, *detections: CanonicalDetection) -> CanonicalDetectionFrame:
    return CanonicalDetectionFrame(
        frame_index=frame_index,
        timestamp_s=frame_index / FPS,
        detections=detections,
    )


def test_builder_links_simple_smooth_motion():
    frames = [
        _frame(0, _detection(0, "d0", 10.0)),
        _frame(1, _detection(1, "d1", 10.2)),
        _frame(2, _detection(2, "d2", 10.4)),
        _frame(3, _detection(3, "d3", 10.6)),
    ]

    result = build_tracklets(frames)

    assert result.tracklet_count == 1
    assert [item.detection_id for item in result.tracklets[0].observations] == [
        "d0",
        "d1",
        "d2",
        "d3",
    ]


def test_builder_keeps_singletons_instead_of_dropping_evidence():
    result = build_tracklets(
        [_frame(0, _detection(0, "d0", 10.0), _detection(0, "d1", 30.0))]
    )

    assert result.tracklet_count == 2
    assert result.observation_count == 2


def test_builder_reconnects_after_short_detector_miss():
    frames = [
        _frame(0, _detection(0, "d0", 10.0)),
        _frame(1, _detection(1, "d1", 10.2)),
        _frame(2),
        _frame(3, _detection(3, "d3", 10.6)),
    ]

    result = build_tracklets(frames)

    assert result.tracklet_count == 1
    assert [item.frame_index for item in result.tracklets[0].observations] == [
        0,
        1,
        3,
    ]
    assert result.tracklets[0].max_frame_gap == 1


def test_builder_splits_after_long_gap():
    config = TrackletBuilderConfig(max_gap_frames=1, max_gap_s=0.20)
    frames = [
        _frame(0, _detection(0, "d0", 10.0)),
        _frame(1, _detection(1, "d1", 10.2)),
        _frame(2),
        _frame(3),
        _frame(4, _detection(4, "d4", 10.8)),
    ]

    result = build_tracklets(frames, config)

    assert result.tracklet_count == 2
    assert [t.observation_count for t in result.tracklets] == [2, 1]


def test_builder_rejects_physically_impossible_jump():
    config = TrackletBuilderConfig(
        max_speed_mps=8.0,
        prediction_gate_m=0.25,
        uncertainty_growth_mps=0.5,
    )
    frames = [
        _frame(0, _detection(0, "d0", 10.0)),
        _frame(1, _detection(1, "d1", 20.0)),
    ]

    result = build_tracklets(frames, config)

    assert result.tracklet_count == 2
    assert [t.observation_count for t in result.tracklets] == [1, 1]


def test_builder_prefers_prediction_over_last_position():
    config = TrackletBuilderConfig(
        prediction_gate_m=0.35,
        uncertainty_growth_mps=0.5,
        ambiguity_margin=0.05,
    )
    frames = [
        _frame(0, _detection(0, "a0", 0.0)),
        _frame(1, _detection(1, "a1", 0.2)),
        _frame(
            2,
            _detection(2, "true", 0.4),
            _detection(2, "decoy", 0.05),
        ),
    ]

    result = build_tracklets(frames, config)

    first = result.tracklets[0]
    assert [item.detection_id for item in first.observations] == [
        "a0",
        "a1",
        "true",
    ]


def test_builder_fragments_when_track_has_ambiguous_candidates():
    config = TrackletBuilderConfig(
        prediction_gate_m=1.0,
        uncertainty_growth_mps=0.0,
        ambiguity_margin=0.20,
    )
    frames = [
        _frame(0, _detection(0, "d0", 10.0)),
        _frame(
            1,
            _detection(1, "left", 10.10),
            _detection(1, "right", 10.12),
        ),
    ]

    result = build_tracklets(frames, config)

    # The original trajectory is not forced onto either near-identical
    # candidate. Both detections start fresh fragments.
    assert result.tracklet_count == 3
    assert result.tracklets[0].observation_count == 1


def test_builder_fragments_when_detection_is_ambiguous_between_tracks():
    config = TrackletBuilderConfig(
        prediction_gate_m=1.0,
        uncertainty_growth_mps=0.0,
        ambiguity_margin=0.20,
    )
    frames = [
        _frame(
            0,
            _detection(0, "a0", 10.00),
            _detection(0, "b0", 10.04),
        ),
        _frame(1, _detection(1, "candidate", 10.02)),
    ]

    result = build_tracklets(frames, config)

    assert result.tracklet_count == 3
    assert sorted(t.observation_count for t in result.tracklets) == [1, 1, 1]


def test_builder_keeps_two_well_separated_players_stable():
    frames = [
        _frame(
            0,
            _detection(0, "a0", 10.0),
            _detection(0, "b0", 30.0),
        ),
        _frame(
            1,
            _detection(1, "a1", 10.2),
            _detection(1, "b1", 29.8),
        ),
        _frame(
            2,
            _detection(2, "a2", 10.4),
            _detection(2, "b2", 29.6),
        ),
    ]

    result = build_tracklets(frames)

    assert result.tracklet_count == 2
    assert [
        [obs.detection_id for obs in tracklet.observations]
        for tracklet in result.tracklets
    ] == [
        ["a0", "a1", "a2"],
        ["b0", "b1", "b2"],
    ]


def test_builder_is_deterministic():
    frames = [
        _frame(
            0,
            _detection(0, "a0", 10.0),
            _detection(0, "b0", 30.0),
        ),
        _frame(
            1,
            _detection(1, "a1", 10.2),
            _detection(1, "b1", 29.8),
        ),
    ]

    first = build_tracklets(frames)
    second = build_tracklets(frames)

    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize(
    "frames",
    [
        [
            _frame(1, _detection(1, "d1", 10.0)),
            _frame(0, _detection(0, "d0", 10.0)),
        ],
        [
            CanonicalDetectionFrame(0, 1.0, ()),
            CanonicalDetectionFrame(1, 0.9, ()),
        ],
    ],
)
def test_builder_rejects_non_monotonic_frames(frames):
    with pytest.raises(ValueError):
        build_tracklets(frames)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gap_frames": -1},
        {"max_gap_s": -0.1},
        {"max_speed_mps": 0.0},
        {"prediction_gate_m": 0.0},
        {"uncertainty_growth_mps": -1.0},
        {"ambiguity_margin": -0.1},
        {"confidence_penalty": -0.1},
        {"velocity_smoothing": 1.1},
    ],
)
def test_builder_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrackletBuilderConfig(**kwargs)


def test_builder_metadata_records_algorithm_and_config():
    config = TrackletBuilderConfig(max_gap_frames=4, ambiguity_margin=0.25)

    result = ConservativeTrackletBuilder(config).build([])

    assert result.metadata["builder"] == "conservative-local-v1"
    assert result.metadata["config"]["max_gap_frames"] == 4
    assert result.metadata["config"]["ambiguity_margin"] == 0.25
