from experiments.player_detection.pitch_tracking import (
    PitchTracker,
    ProjectedDetection,
    inside_field_detections,
    merge_cross_camera,
)
from experiments.player_detection.run_interval import bbox_bottom_center


def detection(lens: str, x: float, z: float) -> ProjectedDetection:
    return ProjectedDetection({"lens": lens, "world_x": x, "world_z": z, "inside_field": True}, lens, x, z)


def test_bottom_center_is_the_detection_footpoint():
    assert bbox_bottom_center([20, 30, 80, 150]) == (50.0, 150.0)


def test_close_left_right_detections_merge_to_one_both_observation():
    observations = merge_cross_camera([detection("left", 2.0, 3.0), detection("right", 2.6, 3.0)], 1.0)
    assert len(observations) == 1
    assert observations[0].source == "both"
    assert (observations[0].x, observations[0].z) == (2.3, 3.0)


def test_distant_left_right_detections_do_not_merge():
    observations = merge_cross_camera([detection("left", 0.0, 0.0), detection("right", 3.0, 0.0)], 1.0)
    assert [observation.source for observation in observations] == ["left", "right"]


def test_cross_camera_matching_is_one_to_one():
    observations = merge_cross_camera(
        [detection("left", 0.0, 0.0), detection("right", .1, 0.0), detection("right", .2, 0.0)], .5,
    )
    assert len(observations) == 2
    assert sum(observation.source == "both" for observation in observations) == 1


def test_same_camera_detections_never_merge():
    observations = merge_cross_camera([detection("left", 0.0, 0.0), detection("left", .1, 0.0)], 1.0)
    assert len(observations) == 2
    assert {observation.source for observation in observations} == {"left"}


def test_moving_observation_keeps_global_id():
    tracker = PitchTracker(1.0, 1)
    first = merge_cross_camera([detection("left", 0.0, 0.0)], 1.0)
    second = merge_cross_camera([detection("left", .6, 0.0)], 1.0)
    tracker.update(first, frame_index=10, timestamp_s=1.0)
    tracker.update(second, frame_index=11, timestamp_s=1.1)
    assert first[0].global_track_id == second[0].global_track_id == 1


def test_track_survives_configured_gap():
    tracker = PitchTracker(1.0, 2)
    first = merge_cross_camera([detection("left", 0.0, 0.0)], 1.0)
    tracker.update(first, frame_index=10, timestamp_s=1.0)
    tracker.update([], frame_index=11, timestamp_s=1.1)
    tracker.update([], frame_index=12, timestamp_s=1.2)
    returning = merge_cross_camera([detection("right", .3, 0.0)], 1.0)
    tracker.update(returning, frame_index=13, timestamp_s=1.3)
    assert returning[0].global_track_id == 1


def test_distant_observation_receives_new_global_id():
    tracker = PitchTracker(1.0, 1)
    first = merge_cross_camera([detection("left", 0.0, 0.0)], 1.0)
    later = merge_cross_camera([detection("left", 3.0, 0.0)], 1.0)
    tracker.update(first, frame_index=10, timestamp_s=1.0)
    tracker.update(later, frame_index=11, timestamp_s=1.1)
    assert (first[0].global_track_id, later[0].global_track_id) == (1, 2)


def test_both_lenses_create_one_global_track():
    records = [
        {"lens": "left", "world_x": 1.0, "world_z": 2.0, "inside_field": True},
        {"lens": "right", "world_x": 1.3, "world_z": 2.0, "inside_field": True},
        {"lens": "left", "world_x": None, "world_z": None, "inside_field": False},
    ]
    observations = merge_cross_camera(inside_field_detections(records), .5)
    tracker = PitchTracker(1.0, 1)
    tracker.update(observations, frame_index=10, timestamp_s=1.0)
    assert len(observations) == len(tracker.tracks) == 1
    assert observations[0].source == "both"
