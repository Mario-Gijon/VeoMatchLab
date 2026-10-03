"""Tracking V1 Hito 7I.7: actor-level trajectory reconstruction.

Consumes the 23-actor partition from 7I.6 and turns fragmented tracklet
observations into 23 renderable physical trajectories.

The key change is the unit of motion:

    tracklet -> actor trajectory

Small gaps are reconstructed at every missing SOURCE frame, then the already
validated Tracking V1 constant-velocity Kalman + RTS smoother is reused.
Long/physically implausible gaps remain hard LOST boundaries; no carry-forward
is invented.

Field rendering
---------------
Every emitted display point contains smoothed X/Z, so a 10 Hz detector can be
rendered at the source-video frame cadence without the 2D pitch "trembling".

Ultra / interactive rendering
-----------------------------
Observed points preserve the canonical detection visual evidence:
representative_view + virtual_bbox_xyxy.

Inserted points do NOT invent a projected X/Z pixel. Instead they carry the
left/right canonical visual anchors plus interpolation alpha. The frontend can
compute the same presentation rays used by Baseline/V8 for both endpoints and
interpolate those rays. This keeps image alignment tied to actual detections.

Double-lens rendering
---------------------
Observed source_bbox_xyxy is preserved. For a reconstructed point we only mark
source-bbox interpolation as allowed when BOTH endpoints have source bboxes on
the same physical lens.

No identity is changed here. NON_ROSTER tracklets are excluded entirely.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Mapping, Sequence

from tracking.final.models import (
    FusionMode,
    IdentityTrajectory,
    TrajectoryPoint,
    TrajectoryPointOrigin,
)
from tracking.final.reconstruction import infer_sampling_cadence
from tracking.final.smoothing import (
    TemporalSmoothingConfig,
    smooth_trajectories,
)


ACTOR_TRAJECTORY_VERSION = "actor-trajectory-reconstruction-v1"


@dataclass(frozen=True, slots=True)
class ActorTrajectoryConfig:
    expected_actor_slots: int = 23

    # Reconstruct only short visual gaps. A long LOST interval stays absent.
    max_interpolation_gap_s: float = 0.50
    max_endpoint_speed_mps: float = 12.5

    # Require timestamp/frame ordering to agree with the observed source video.
    min_dt_s: float = 1e-6

    # Existing Tracking V1 RTS parameters, exposed here so the artifact is
    # reproducible and the UI never has to smooth positions itself.
    process_acceleration_sigma_mps2: float = 12.0
    observed_position_sigma_m: float = 0.30
    interpolated_position_sigma_m: float = 0.90
    minimum_confidence: float = 0.15
    adjustment_review_threshold_m: float = 1.50

    def __post_init__(self) -> None:
        if self.expected_actor_slots < 1:
            raise ValueError("expected_actor_slots must be >= 1")
        for name in (
            "max_interpolation_gap_s",
            "max_endpoint_speed_mps",
            "min_dt_s",
            "process_acceleration_sigma_mps2",
            "observed_position_sigma_m",
            "interpolated_position_sigma_m",
            "adjustment_review_threshold_m",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 < self.minimum_confidence <= 1:
            raise ValueError("minimum_confidence must be in (0,1]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class _Observation:
    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float


@dataclass(frozen=True, slots=True)
class _Tracklet:
    tracklet_id: str
    observations: tuple[_Observation, ...]


@dataclass(frozen=True, slots=True)
class _Visual:
    detection_id: str
    frame_index: int
    timestamp_s: float
    representative_view: str | None
    virtual_bbox_xyxy: tuple[float, float, float, float] | None
    source_lens: str | None
    source_bbox_xyxy: tuple[float, float, float, float] | None
    source_footpoint_xy: tuple[float, float] | None

    @property
    def has_virtual_anchor(self) -> bool:
        return (
            self.representative_view is not None
            and self.virtual_bbox_xyxy is not None
        )

    @property
    def has_source_anchor(self) -> bool:
        return (
            self.source_lens is not None
            and self.source_bbox_xyxy is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "representative_view": self.representative_view,
            "virtual_bbox_xyxy": (
                list(self.virtual_bbox_xyxy)
                if self.virtual_bbox_xyxy is not None
                else None
            ),
            "source_lens": self.source_lens,
            "source_bbox_xyxy": (
                list(self.source_bbox_xyxy)
                if self.source_bbox_xyxy is not None
                else None
            ),
            "source_footpoint_xy": (
                list(self.source_footpoint_xy)
                if self.source_footpoint_xy is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class GapEvent:
    actor_id: str
    left_frame_index: int
    right_frame_index: int
    left_timestamp_s: float
    right_timestamp_s: float
    frame_delta: int
    duration_s: float
    endpoint_distance_m: float
    endpoint_speed_mps: float
    cross_tracklet: bool
    accepted: bool
    reason: str
    inserted_point_count: int
    visual_interpolation_allowed: bool
    source_bbox_interpolation_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "trajectory_version": ACTOR_TRAJECTORY_VERSION,
            **asdict(self),
        }


@dataclass(frozen=True, slots=True)
class ActorTrajectoryResult:
    raw_trajectories: tuple[IdentityTrajectory, ...]
    reconstructed_trajectories: tuple[IdentityTrajectory, ...]
    display_trajectories: tuple[dict[str, Any], ...]
    segment_rows: tuple[dict[str, Any], ...]
    gap_events: tuple[GapEvent, ...]
    smoothing_summary: Mapping[str, Any]
    config: ActorTrajectoryConfig

    def summary(self) -> dict[str, Any]:
        display_points = [
            point
            for trajectory in self.display_trajectories
            for point in trajectory["points"]
        ]
        observed = [
            point for point in display_points
            if point["origin"] == "observed"
        ]
        interpolated = [
            point for point in display_points
            if point["origin"] == "interpolated"
        ]
        visual_observed = [
            point for point in observed
            if point["visual_anchor"] is not None
        ]
        visual_interpolated = [
            point for point in interpolated
            if (
                point["visual_interpolation"] is not None
                and point["visual_interpolation"][
                    "visual_interpolation_allowed"
                ]
            )
        ]
        source_bbox_observed = [
            point for point in observed
            if (
                point["visual_anchor"] is not None
                and point["visual_anchor"]["source_bbox_xyxy"]
                is not None
            )
        ]
        source_bbox_interpolated = [
            point for point in interpolated
            if (
                point["visual_interpolation"] is not None
                and point["visual_interpolation"][
                    "source_bbox_interpolation_allowed"
                ]
            )
        ]

        return {
            "schema_version": 1,
            "trajectory_version": ACTOR_TRAJECTORY_VERSION,
            "actor_count": len(self.display_trajectories),
            "raw_observed_point_count": sum(
                len(row.points) for row in self.raw_trajectories
            ),
            "reconstructed_point_count": sum(
                len(row.points)
                for row in self.reconstructed_trajectories
            ),
            "display_point_count": len(display_points),
            "observed_display_point_count": len(observed),
            "interpolated_display_point_count": len(interpolated),
            "segment_count": len(self.segment_rows),
            "gap_event_count": len(self.gap_events),
            "filled_gap_count": sum(
                event.accepted for event in self.gap_events
            ),
            "preserved_gap_count": sum(
                not event.accepted for event in self.gap_events
            ),
            "gap_reason_counts": dict(sorted(Counter(
                event.reason for event in self.gap_events
            ).items())),
            "visual_observed_point_count": len(visual_observed),
            "visual_interpolated_point_count": len(
                visual_interpolated
            ),
            "source_bbox_observed_point_count": len(
                source_bbox_observed
            ),
            "source_bbox_interpolated_point_count": len(
                source_bbox_interpolated
            ),
            "long_gap_carry_forward": False,
            "non_roster_included": False,
            "field_positions_are_smoothed": True,
            "visual_positions_use_detection_rays": True,
            "smoothing": dict(self.smoothing_summary),
            "config": self.config.to_dict(),
        }


def _bbox(
    value: Any,
) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must have four values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("bbox contains non-finite values")
    return result


def _point2(
    value: Any,
) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("point must have two values")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError("point contains non-finite values")
    return result


def _parse_tracklet(row: Mapping[str, Any]) -> _Tracklet:
    tid = str(row["tracklet_id"])
    raw = row.get("observations")
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{tid} has no observations")

    observations = []
    previous_frame = None
    previous_time = None
    for item in raw:
        observation = _Observation(
            detection_id=str(item["detection_id"]),
            frame_index=int(item["frame_index"]),
            timestamp_s=float(item["timestamp_s"]),
            x=float(item["x"]),
            z=float(item["z"]),
            confidence=float(item.get("confidence", 1.0)),
        )
        if previous_frame is not None:
            if observation.frame_index <= previous_frame:
                raise ValueError(
                    f"{tid} frames are not strictly increasing"
                )
            if observation.timestamp_s <= previous_time:
                raise ValueError(
                    f"{tid} timestamps are not strictly increasing"
                )
        if not all(math.isfinite(value) for value in (
            observation.timestamp_s,
            observation.x,
            observation.z,
            observation.confidence,
        )):
            raise ValueError(f"{tid} contains non-finite values")
        previous_frame = observation.frame_index
        previous_time = observation.timestamp_s
        observations.append(observation)

    return _Tracklet(tid, tuple(observations))


def build_visual_index(
    detection_frame_rows: Sequence[Mapping[str, Any]],
) -> dict[str, _Visual]:
    result: dict[str, _Visual] = {}

    for frame_row in detection_frame_rows:
        frame_index = int(frame_row["frame_index"])
        timestamp_s = float(frame_row["timestamp_s"])
        raw_detections = frame_row.get("detections", ())
        if not isinstance(raw_detections, (list, tuple)):
            raise ValueError("frame detections must be a list")

        for item in raw_detections:
            detection_id = str(item["detection_id"])
            if detection_id in result:
                raise ValueError(
                    f"duplicate canonical detection {detection_id}"
                )
            detection_frame = int(item["frame_index"])
            detection_time = float(item["timestamp_s"])
            if detection_frame != frame_index:
                raise ValueError(
                    f"{detection_id} frame mismatch"
                )
            if not math.isclose(
                detection_time,
                timestamp_s,
                abs_tol=1e-6,
                rel_tol=0.0,
            ):
                raise ValueError(
                    f"{detection_id} timestamp mismatch"
                )

            result[detection_id] = _Visual(
                detection_id=detection_id,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
                representative_view=(
                    str(item["representative_view"])
                    if item.get("representative_view") is not None
                    else None
                ),
                virtual_bbox_xyxy=_bbox(
                    item.get("virtual_bbox_xyxy")
                ),
                source_lens=(
                    str(item["source_lens"])
                    if item.get("source_lens") is not None
                    else None
                ),
                source_bbox_xyxy=_bbox(
                    item.get("source_bbox_xyxy")
                ),
                source_footpoint_xy=_point2(
                    item.get("source_footpoint_xy")
                ),
            )

    return result


def _observed_metadata(
    *,
    tracklet_id: str,
    visual: _Visual | None,
) -> dict[str, Any]:
    return {
        "builder": ACTOR_TRAJECTORY_VERSION,
        "tracklet_id": tracklet_id,
        "render_kind": "observed_anchor",
        "visual_anchor": (
            visual.to_dict()
            if visual is not None
            else None
        ),
    }


def build_raw_actor_trajectories(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
    detection_frame_rows: Sequence[Mapping[str, Any]],
    expected_actor_slots: int = 23,
) -> tuple[IdentityTrajectory, ...]:
    tracklets = {
        parsed.tracklet_id: parsed
        for parsed in (_parse_tracklet(row) for row in tracklet_rows)
    }
    if len(tracklets) != len(tracklet_rows):
        raise ValueError("duplicate tracklet IDs")

    actor_ids = tuple(sorted(
        str(row["actor_id"]) for row in actor_rows
    ))
    if len(actor_ids) != expected_actor_slots:
        raise ValueError(
            f"expected {expected_actor_slots} actors, "
            f"got {len(actor_ids)}"
        )
    if len(set(actor_ids)) != len(actor_ids):
        raise ValueError("duplicate actor IDs")

    visual_index = build_visual_index(detection_frame_rows)

    member_ids: dict[str, list[str]] = {
        actor_id: [] for actor_id in actor_ids
    }
    for row in assignment_rows:
        tid = str(row["tracklet_id"])
        actor_id = str(row["actor_id"])
        if tid not in tracklets:
            raise ValueError(
                f"assignment references unknown tracklet {tid}"
            )
        if actor_id not in member_ids:
            raise ValueError(
                f"assignment references unknown actor {actor_id}"
            )
        member_ids[actor_id].append(tid)

    result = []
    for actor_id in actor_ids:
        tids = tuple(sorted(set(member_ids[actor_id])))
        if not tids:
            raise ValueError(f"{actor_id} has no assigned tracklets")

        frame_owner: dict[int, str] = {}
        points = []

        for tid in tids:
            for observation in tracklets[tid].observations:
                previous = frame_owner.get(observation.frame_index)
                if previous is not None and previous != tid:
                    raise ValueError(
                        f"{actor_id} exact-frame coexistence: "
                        f"{previous}, {tid} at {observation.frame_index}"
                    )
                frame_owner[observation.frame_index] = tid

                visual = visual_index.get(
                    observation.detection_id
                )
                if visual is not None:
                    if visual.frame_index != observation.frame_index:
                        raise ValueError("visual frame mismatch")
                    if not math.isclose(
                        visual.timestamp_s,
                        observation.timestamp_s,
                        abs_tol=1e-6,
                        rel_tol=0.0,
                    ):
                        raise ValueError("visual timestamp mismatch")

                points.append(
                    TrajectoryPoint(
                        identity_id=actor_id,
                        frame_index=observation.frame_index,
                        timestamp_s=observation.timestamp_s,
                        x=observation.x,
                        z=observation.z,
                        confidence=max(
                            0.0,
                            min(1.0, observation.confidence),
                        ),
                        origin=TrajectoryPointOrigin.OBSERVED,
                        fusion_mode=FusionMode.SINGLE,
                        source_tracklet_ids=(tid,),
                        source_detection_ids=(
                            observation.detection_id,
                        ),
                        source_count=1,
                        position_spread_m=0.0,
                        metadata=_observed_metadata(
                            tracklet_id=tid,
                            visual=visual,
                        ),
                    )
                )

        points.sort(
            key=lambda row: (
                row.frame_index,
                row.timestamp_s,
            )
        )
        result.append(
            IdentityTrajectory(
                identity_id=actor_id,
                tracklet_ids=tids,
                points=tuple(points),
                trajectory_version=(
                    "global-closed-roster-observations-v1"
                ),
                metadata={
                    "builder": ACTOR_TRAJECTORY_VERSION,
                    "source": (
                        "global-closed-roster-reassignment-v1"
                    ),
                    "non_roster_excluded": True,
                },
            )
        )

    return tuple(result)


def _visual_from_point(
    point: TrajectoryPoint,
) -> dict[str, Any] | None:
    value = point.metadata.get("visual_anchor")
    return dict(value) if isinstance(value, Mapping) else None


def _safe_gap(
    left: TrajectoryPoint,
    right: TrajectoryPoint,
    *,
    config: ActorTrajectoryConfig,
) -> tuple[bool, str, float, float, int]:
    frame_delta = right.frame_index - left.frame_index
    duration_s = right.timestamp_s - left.timestamp_s

    if frame_delta <= 1:
        return False, "not_a_gap", 0.0, 0.0, 0
    if duration_s <= config.min_dt_s:
        return False, "non_positive_time", 0.0, math.inf, 0

    distance = math.hypot(
        right.x - left.x,
        right.z - left.z,
    )
    speed = distance / duration_s

    if duration_s > config.max_interpolation_gap_s:
        return (
            False,
            "gap_too_long",
            distance,
            speed,
            0,
        )
    if speed > config.max_endpoint_speed_mps:
        return (
            False,
            "endpoint_speed_too_high",
            distance,
            speed,
            0,
        )

    return (
        True,
        "safe_actor_gap",
        distance,
        speed,
        frame_delta - 1,
    )


def reconstruct_short_actor_gaps(
    *,
    trajectories: Sequence[IdentityTrajectory],
    config: ActorTrajectoryConfig,
) -> tuple[
    tuple[IdentityTrajectory, ...],
    tuple[GapEvent, ...],
]:
    rebuilt = []
    events: list[GapEvent] = []

    for trajectory in sorted(
        trajectories,
        key=lambda row: row.identity_id,
    ):
        points = trajectory.points
        output = [points[0]]

        for left, right in zip(points, points[1:]):
            accepted, reason, distance, speed, inserted = _safe_gap(
                left,
                right,
                config=config,
            )

            left_visual = _visual_from_point(left)
            right_visual = _visual_from_point(right)

            visual_allowed = bool(
                accepted
                and left_visual is not None
                and right_visual is not None
                and left_visual.get("representative_view")
                is not None
                and right_visual.get("representative_view")
                is not None
                and left_visual.get("virtual_bbox_xyxy")
                is not None
                and right_visual.get("virtual_bbox_xyxy")
                is not None
            )

            source_bbox_allowed = bool(
                accepted
                and left_visual is not None
                and right_visual is not None
                and left_visual.get("source_bbox_xyxy")
                is not None
                and right_visual.get("source_bbox_xyxy")
                is not None
                and left_visual.get("source_lens") is not None
                and left_visual.get("source_lens")
                == right_visual.get("source_lens")
            )

            event = GapEvent(
                actor_id=trajectory.identity_id,
                left_frame_index=left.frame_index,
                right_frame_index=right.frame_index,
                left_timestamp_s=left.timestamp_s,
                right_timestamp_s=right.timestamp_s,
                frame_delta=right.frame_index - left.frame_index,
                duration_s=right.timestamp_s - left.timestamp_s,
                endpoint_distance_m=distance,
                endpoint_speed_mps=speed,
                cross_tracklet=(
                    left.source_tracklet_ids
                    != right.source_tracklet_ids
                ),
                accepted=accepted,
                reason=reason,
                inserted_point_count=inserted if accepted else 0,
                visual_interpolation_allowed=visual_allowed,
                source_bbox_interpolation_allowed=(
                    source_bbox_allowed
                ),
            )

            if right.frame_index - left.frame_index > 1:
                events.append(event)

            if accepted:
                frame_delta = (
                    right.frame_index - left.frame_index
                )
                tracklet_ids = tuple(sorted(set(
                    left.source_tracklet_ids
                    + right.source_tracklet_ids
                )))
                detection_ids = tuple(sorted(set(
                    left.source_detection_ids
                    + right.source_detection_ids
                )))

                for frame_index in range(
                    left.frame_index + 1,
                    right.frame_index,
                ):
                    alpha = (
                        (frame_index - left.frame_index)
                        / frame_delta
                    )
                    timestamp_s = (
                        left.timestamp_s
                        + alpha
                        * (
                            right.timestamp_s
                            - left.timestamp_s
                        )
                    )
                    x = (
                        left.x
                        + alpha * (right.x - left.x)
                    )
                    z = (
                        left.z
                        + alpha * (right.z - left.z)
                    )

                    output.append(
                        TrajectoryPoint(
                            identity_id=trajectory.identity_id,
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                            x=x,
                            z=z,
                            confidence=min(
                                left.confidence,
                                right.confidence,
                            ),
                            origin=(
                                TrajectoryPointOrigin.INTERPOLATED
                            ),
                            fusion_mode=(
                                FusionMode.NOT_APPLICABLE
                            ),
                            source_tracklet_ids=tracklet_ids,
                            source_detection_ids=detection_ids,
                            source_count=len(detection_ids),
                            position_spread_m=max(
                                left.position_spread_m,
                                right.position_spread_m,
                            ),
                            metadata={
                                "builder": (
                                    ACTOR_TRAJECTORY_VERSION
                                ),
                                "render_kind": (
                                    "actor_gap_interpolation"
                                ),
                                "visual_interpolation": {
                                    "left": left_visual,
                                    "right": right_visual,
                                    "alpha": alpha,
                                    "visual_interpolation_allowed": (
                                        visual_allowed
                                    ),
                                    "source_bbox_interpolation_allowed": (
                                        source_bbox_allowed
                                    ),
                                },
                                "gap": event.to_dict(),
                            },
                        )
                    )

            output.append(right)

        output.sort(
            key=lambda row: (
                row.frame_index,
                row.timestamp_s,
            )
        )

        metadata = dict(trajectory.metadata)
        metadata.update({
            "builder": ACTOR_TRAJECTORY_VERSION,
            "short_gaps_reconstructed": True,
            "max_interpolation_gap_s": (
                config.max_interpolation_gap_s
            ),
            "max_endpoint_speed_mps": (
                config.max_endpoint_speed_mps
            ),
        })

        rebuilt.append(
            IdentityTrajectory(
                identity_id=trajectory.identity_id,
                tracklet_ids=trajectory.tracklet_ids,
                points=tuple(output),
                trajectory_version=(
                    "actor-gap-reconstruction-v1"
                ),
                metadata=metadata,
            )
        )

    return tuple(rebuilt), tuple(events)


def _smoothed_display_rows(
    smoothed_trajectories: Sequence[Any],
) -> tuple[
    tuple[dict[str, Any], ...],
    tuple[dict[str, Any], ...],
]:
    trajectories = []
    segments = []

    for trajectory in sorted(
        smoothed_trajectories,
        key=lambda row: row.identity_id,
    ):
        points = []
        segment_groups: dict[str, list[Any]] = defaultdict(list)

        for point in trajectory.points:
            measurement_metadata = point.metadata.get(
                "measurement_metadata",
                {},
            )
            if not isinstance(
                measurement_metadata,
                Mapping,
            ):
                measurement_metadata = {}

            visual_anchor = measurement_metadata.get(
                "visual_anchor"
            )
            if not isinstance(visual_anchor, Mapping):
                visual_anchor = None
            else:
                visual_anchor = dict(visual_anchor)

            visual_interpolation = measurement_metadata.get(
                "visual_interpolation"
            )
            if not isinstance(
                visual_interpolation,
                Mapping,
            ):
                visual_interpolation = None
            else:
                visual_interpolation = dict(
                    visual_interpolation
                )

            points.append({
                "frame_index": point.frame_index,
                "timestamp_s": point.timestamp_s,
                "x": point.smoothed_x,
                "z": point.smoothed_z,
                "vx_mps": point.vx_mps,
                "vz_mps": point.vz_mps,
                "speed_mps": point.speed_mps,
                "origin": point.origin,
                "segment_id": point.segment_id,
                "segment_index": point.segment_index,
                "measurement_x": point.measurement_x,
                "measurement_z": point.measurement_z,
                "position_adjustment_m": (
                    point.position_adjustment_m
                ),
                "position_std_m": point.position_std_m,
                "confidence": point.measurement_confidence,
                "source_tracklet_ids": list(
                    point.source_tracklet_ids
                ),
                "source_detection_ids": list(
                    point.source_detection_ids
                ),
                "visual_anchor": visual_anchor,
                "visual_interpolation": (
                    visual_interpolation
                ),
                "render_position_kind": (
                    "observed"
                    if point.origin == "observed"
                    else "interpolated"
                ),
            })
            segment_groups[point.segment_id].append(point)

        for segment_id, segment_points in sorted(
            segment_groups.items()
        ):
            first = segment_points[0]
            last = segment_points[-1]
            segments.append({
                "schema_version": 1,
                "trajectory_version": (
                    ACTOR_TRAJECTORY_VERSION
                ),
                "actor_id": trajectory.identity_id,
                "segment_id": segment_id,
                "segment_index": first.segment_index,
                "start_frame": first.frame_index,
                "end_frame": last.frame_index,
                "start_timestamp_s": first.timestamp_s,
                "end_timestamp_s": last.timestamp_s,
                "point_count": len(segment_points),
                "observed_point_count": sum(
                    point.origin == "observed"
                    for point in segment_points
                ),
                "interpolated_point_count": sum(
                    point.origin == "interpolated"
                    for point in segment_points
                ),
                "kinematics_quality_counts": dict(
                    sorted(Counter(
                        point.kinematics_quality
                        for point in segment_points
                    ).items())
                ),
            })

        trajectories.append({
            "schema_version": 1,
            "trajectory_version": ACTOR_TRAJECTORY_VERSION,
            "actor_id": trajectory.identity_id,
            "tracklet_ids": list(
                trajectory.tracklet_ids
            ),
            "point_count": len(points),
            "segment_count": trajectory.segment_count,
            "points": points,
            "provisional": True,
        })

    return tuple(trajectories), tuple(segments)


def build_actor_trajectories(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    actor_rows: Sequence[Mapping[str, Any]],
    assignment_rows: Sequence[Mapping[str, Any]],
    detection_frame_rows: Sequence[Mapping[str, Any]],
    config: ActorTrajectoryConfig | None = None,
) -> ActorTrajectoryResult:
    selected = config or ActorTrajectoryConfig()

    raw = build_raw_actor_trajectories(
        tracklet_rows=tracklet_rows,
        actor_rows=actor_rows,
        assignment_rows=assignment_rows,
        detection_frame_rows=detection_frame_rows,
        expected_actor_slots=selected.expected_actor_slots,
    )

    rebuilt, gap_events = reconstruct_short_actor_gaps(
        trajectories=raw,
        config=selected,
    )

    # Because we inserted every safe missing SOURCE frame, the inferred cadence
    # becomes the actual video-frame cadence (~30 Hz in this match), not the
    # detector sampling cadence (~10 Hz).
    cadence = infer_sampling_cadence(rebuilt)

    smoothing_config = TemporalSmoothingConfig(
        process_acceleration_sigma_mps2=(
            selected.process_acceleration_sigma_mps2
        ),
        observed_position_sigma_m=(
            selected.observed_position_sigma_m
        ),
        interpolated_position_sigma_m=(
            selected.interpolated_position_sigma_m
        ),
        minimum_confidence=selected.minimum_confidence,
        adjustment_review_threshold_m=(
            selected.adjustment_review_threshold_m
        ),
    )

    smoothed, _, smoothing_summary = smooth_trajectories(
        trajectories=rebuilt,
        config=smoothing_config,
        cadence=cadence,
    )

    display, segments = _smoothed_display_rows(
        smoothed
    )

    smoothing_summary = dict(smoothing_summary)
    smoothing_summary["sampling_cadence"] = {
        "frame_step": cadence.frame_step,
        "time_step_s": cadence.time_step_s,
        "estimated_fps": (
            1.0 / cadence.time_step_s
            if cadence.time_step_s > 0
            else None
        ),
    }

    return ActorTrajectoryResult(
        raw_trajectories=raw,
        reconstructed_trajectories=rebuilt,
        display_trajectories=display,
        segment_rows=segments,
        gap_events=gap_events,
        smoothing_summary=smoothing_summary,
        config=selected,
    )
