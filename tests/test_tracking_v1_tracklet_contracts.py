import json
import math

import pytest

from tracking.tracklets import Tracklet, TrackletObservation, TrackletSet


def _observation(index: int, **overrides) -> TrackletObservation:
    payload = {
        "detection_id": f"det-{index}",
        "frame_index": index,
        "timestamp_s": index / 30,
        "x": 10.0 + index * 0.2,
        "z": 20.0,
        "confidence": 0.9,
    }
    payload.update(overrides)
    return TrackletObservation(**payload)


def _tracklet(tracklet_id: str = "trk-1", indices=(10, 11, 13), **overrides) -> Tracklet:
    payload = {
        "tracklet_id": tracklet_id,
        "observations": tuple(_observation(index) for index in indices),
        "metadata": {"source": "test"},
    }
    payload.update(overrides)
    return Tracklet(**payload)


def test_tracklet_observation_normalizes_valid_payload():
    observation = TrackletObservation(
        detection_id=" det-10 ",
        frame_index=10,
        timestamp_s=1 / 3,
        x=12,
        z=20.5,
        confidence=0.75,
    )

    assert observation.detection_id == "det-10"
    assert observation.x == 12.0
    assert observation.z == 20.5
    assert observation.confidence == 0.75


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detection_id", ""),
        ("frame_index", -1),
        ("frame_index", True),
        ("timestamp_s", -0.1),
        ("timestamp_s", math.nan),
        ("x", math.inf),
        ("z", -math.inf),
        ("confidence", -0.01),
        ("confidence", 1.01),
    ],
)
def test_tracklet_observation_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        _observation(10, **{field: value})


def test_tracklet_accepts_gaps_and_exposes_motion_summary():
    tracklet = _tracklet(indices=(10, 11, 13))

    assert tracklet.start_frame == 10
    assert tracklet.end_frame == 13
    assert tracklet.observation_count == 3
    assert tracklet.max_frame_gap == 1
    assert tracklet.duration_s == pytest.approx(3 / 30)
    assert tracklet.mean_confidence == pytest.approx(0.9)
    assert tracklet.displacement_m == pytest.approx(0.6)
    assert tracklet.path_length_m == pytest.approx(0.6)


def test_tracklet_single_observation_is_valid():
    tracklet = _tracklet(indices=(10,))

    assert tracklet.start_frame == tracklet.end_frame == 10
    assert tracklet.duration_s == 0.0
    assert tracklet.path_length_m == 0.0
    assert tracklet.displacement_m == 0.0
    assert tracklet.max_frame_gap == 0


@pytest.mark.parametrize(
    "observations",
    [
        (),
        (_observation(10), _observation(10, detection_id="other")),
        (_observation(11), _observation(10)),
        (
            _observation(10, timestamp_s=1.0),
            _observation(11, timestamp_s=0.9),
        ),
        (
            _observation(10, detection_id="same"),
            _observation(11, detection_id="same"),
        ),
    ],
)
def test_tracklet_rejects_invalid_sequences(observations):
    with pytest.raises(ValueError):
        Tracklet(tracklet_id="trk", observations=observations)


def test_tracklet_rejects_invalid_metadata():
    with pytest.raises(ValueError, match="metadata"):
        _tracklet(metadata={"unsupported": {1, 2}})


def test_tracklet_dict_and_json_round_trips():
    tracklet = _tracklet()

    assert Tracklet.from_dict(tracklet.to_dict()).to_dict() == tracklet.to_dict()
    assert Tracklet.from_json(tracklet.to_json()).to_dict() == tracklet.to_dict()
    assert json.loads(tracklet.to_json()) == tracklet.to_dict()


def test_tracklet_set_accepts_empty_or_multiple_tracklets():
    empty = TrackletSet()
    populated = TrackletSet(
        (
            _tracklet("trk-a", indices=(10, 11)),
            _tracklet("trk-b", indices=(20, 21)),
        )
    )

    assert empty.tracklet_count == 0
    assert empty.observation_count == 0
    assert populated.tracklet_count == 2
    assert populated.observation_count == 4


def test_tracklet_set_rejects_duplicate_tracklet_ids():
    with pytest.raises(ValueError, match="duplicate tracklet_id"):
        TrackletSet(
            (
                _tracklet("trk-a", indices=(10, 11)),
                _tracklet("trk-a", indices=(20, 21)),
            )
        )


def test_tracklet_set_rejects_detection_owned_by_two_tracklets():
    first = _tracklet("trk-a", indices=(10, 11))
    second = Tracklet(
        "trk-b",
        (
            _observation(20, detection_id="det-10"),
            _observation(21),
        ),
    )

    with pytest.raises(ValueError, match="belongs to both"):
        TrackletSet((first, second))


def test_tracklet_set_rejects_non_json_metadata():
    with pytest.raises(ValueError, match="metadata"):
        TrackletSet(metadata={"unsupported": {1}})


def test_tracklet_set_dict_and_json_round_trips():
    value = TrackletSet(
        (
            _tracklet("trk-a", indices=(10, 11)),
            _tracklet("trk-b", indices=(20, 22)),
        ),
        metadata={"builder": "strict-local-v1"},
    )

    assert TrackletSet.from_dict(value.to_dict()).to_dict() == value.to_dict()
    assert TrackletSet.from_json(value.to_json()).to_dict() == value.to_dict()
