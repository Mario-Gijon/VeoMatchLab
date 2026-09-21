"""Pitch-space duplicate merging and baseline global tracking for diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Any, Iterable


@dataclass
class ProjectedDetection:
    """One inside-field source detection in canonical geometry coordinates."""

    source_record: dict[str, Any]
    lens: str
    x: float
    z: float


@dataclass
class PitchObservation:
    """One physical-player candidate after optional left/right merging."""

    x: float
    z: float
    source: str
    detections: list[ProjectedDetection]
    global_track_id: int | None = None


@dataclass
class GlobalTrack:
    global_track_id: int
    last_position: tuple[float, float]
    previous_position: tuple[float, float] | None
    last_frame_index: int
    last_timestamp_s: float
    age: int = 1
    missed_frames: int = 0

    def predicted_position(self) -> tuple[float, float]:
        if self.previous_position is None:
            return self.last_position
        return (
            self.last_position[0] + self.last_position[0] - self.previous_position[0],
            self.last_position[1] + self.last_position[1] - self.previous_position[1],
        )


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return hypot(a[0] - b[0], a[1] - b[1])


def inside_field_detections(records: Iterable[dict[str, Any]]) -> list[ProjectedDetection]:
    """Adapt RAW experiment records without altering their geometry result."""
    return [
        ProjectedDetection(record, str(record["lens"]), float(record["world_x"]), float(record["world_z"]))
        for record in records
        if record.get("inside_field") is True
    ]


def merge_cross_camera(detections: Iterable[ProjectedDetection], maximum_distance: float) -> list[PitchObservation]:
    """Greedily apply one-to-one, minimum-distance left/right matching only."""
    if maximum_distance < 0:
        raise ValueError("maximum_distance must be non-negative")
    items = list(detections)
    left = [(index, item) for index, item in enumerate(items) if item.lens == "left"]
    right = [(index, item) for index, item in enumerate(items) if item.lens == "right"]
    pairs = sorted(
        (distance((left_item.x, left_item.z), (right_item.x, right_item.z)), left_index, right_index, left_item, right_item)
        for left_index, left_item in left
        for right_index, right_item in right
        if distance((left_item.x, left_item.z), (right_item.x, right_item.z)) <= maximum_distance
    )
    matched: set[int] = set()
    observations: list[PitchObservation] = []
    for _pair_distance, left_index, right_index, left_item, right_item in pairs:
        if left_index in matched or right_index in matched:
            continue
        matched.update((left_index, right_index))
        observations.append(PitchObservation(
            x=(left_item.x + right_item.x) / 2.0,
            z=(left_item.z + right_item.z) / 2.0,
            source="both",
            detections=[left_item, right_item],
        ))
    for index, item in enumerate(items):
        if index not in matched:
            observations.append(PitchObservation(item.x, item.z, item.lens, [item]))
    return observations


class PitchTracker:
    """Small deterministic global tracker; it intentionally uses no appearance data."""

    def __init__(self, maximum_distance: float, max_missed: int) -> None:
        if maximum_distance < 0 or max_missed < 0:
            raise ValueError("tracking distances and missed-frame count must be non-negative")
        self.maximum_distance = maximum_distance
        self.max_missed = max_missed
        self.tracks: dict[int, GlobalTrack] = {}
        self._next_id = 1

    def update(self, observations: list[PitchObservation], *, frame_index: int, timestamp_s: float) -> list[PitchObservation]:
        pairs = sorted(
            (distance(track.predicted_position(), (observation.x, observation.z)), track_id, observation_index)
            for track_id, track in self.tracks.items()
            for observation_index, observation in enumerate(observations)
            if distance(track.predicted_position(), (observation.x, observation.z)) <= self.maximum_distance
        )
        assigned_tracks: set[int] = set()
        assigned_observations: set[int] = set()
        for _pair_distance, track_id, observation_index in pairs:
            if track_id in assigned_tracks or observation_index in assigned_observations:
                continue
            track = self.tracks[track_id]
            observation = observations[observation_index]
            observation.global_track_id = track_id
            track.previous_position = track.last_position
            track.last_position = (observation.x, observation.z)
            track.last_frame_index = frame_index
            track.last_timestamp_s = timestamp_s
            track.age += 1
            track.missed_frames = 0
            assigned_tracks.add(track_id)
            assigned_observations.add(observation_index)
        for track_id, track in list(self.tracks.items()):
            if track_id not in assigned_tracks:
                track.missed_frames += 1
                if track.missed_frames > self.max_missed:
                    del self.tracks[track_id]
        for observation_index, observation in enumerate(observations):
            if observation_index in assigned_observations:
                continue
            track_id = self._next_id
            self._next_id += 1
            observation.global_track_id = track_id
            self.tracks[track_id] = GlobalTrack(track_id, (observation.x, observation.z), None, frame_index, timestamp_s)
        return observations
