"""Visual and transition audit for provisional Tracking V1 identities.

Hito 3D renders the provisional 3B-B V1 backbone partition in field coordinates and
produces transition diagnostics for human inspection before identities become
canonical.

Important:
- no interpolation is performed;
- a fixed deterministic color is assigned per identity;
- tracklet gaps remain visually empty;
- overlaps remain visible as simultaneous samples;
- this module does not modify the solver or identity assignments.
"""

from __future__ import annotations

import colorsys
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import cv2
import numpy as np


VISUAL_AUDIT_VERSION = "visual-backbone-audit-v1"


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("pair requires two different tracklets")
    return (a, b) if a < b else (b, a)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source}:{line_number} is not a JSON object"
                )
            yield payload


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True, slots=True)
class Observation:
    tracklet_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float


@dataclass(frozen=True, slots=True)
class TrackletTrajectory:
    tracklet_id: str
    observations: tuple[Observation, ...]

    @property
    def start_time_s(self) -> float:
        return self.observations[0].timestamp_s

    @property
    def end_time_s(self) -> float:
        return self.observations[-1].timestamp_s

    @property
    def start_position(self) -> tuple[float, float]:
        first = self.observations[0]
        return (first.x, first.z)

    @property
    def end_position(self) -> tuple[float, float]:
        last = self.observations[-1]
        return (last.x, last.z)

    @property
    def frame_set(self) -> frozenset[int]:
        return frozenset(obs.frame_index for obs in self.observations)


@dataclass(frozen=True, slots=True)
class IdentityAssignment:
    identity_id: str
    tracklet_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class VisualAuditConfig:
    field_length_m: float = 105.0
    field_width_m: float = 68.0
    canvas_width_px: int = 1280
    canvas_height_px: int = 840
    margin_px: int = 55
    output_fps: float = 10.0
    trail_seconds: float = 1.5
    observation_tolerance_s: float = 0.065
    point_radius_px: int = 7
    show_tracklet_label: bool = False

    def __post_init__(self) -> None:
        if self.field_length_m <= 0 or self.field_width_m <= 0:
            raise ValueError("field dimensions must be > 0")
        if self.canvas_width_px < 320 or self.canvas_height_px < 240:
            raise ValueError("canvas is too small")
        if self.margin_px < 0:
            raise ValueError("margin_px must be >= 0")
        if self.output_fps <= 0:
            raise ValueError("output_fps must be > 0")
        if self.trail_seconds < 0:
            raise ValueError("trail_seconds must be >= 0")
        if self.observation_tolerance_s < 0:
            raise ValueError("observation_tolerance_s must be >= 0")


def load_tracklets(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, TrackletTrajectory]:
    result: dict[str, TrackletTrajectory] = {}
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        raw_obs = row.get("observations")
        if not isinstance(raw_obs, list) or not raw_obs:
            raise ValueError(f"{tracklet_id} has no observations")

        observations = tuple(
            Observation(
                tracklet_id=tracklet_id,
                frame_index=int(obs["frame_index"]),
                timestamp_s=_finite_float(
                    "timestamp_s", obs["timestamp_s"]
                ),
                x=_finite_float("x", obs["x"]),
                z=_finite_float("z", obs["z"]),
                confidence=_finite_float(
                    "confidence", obs["confidence"]
                ),
            )
            for obs in raw_obs
        )
        previous_frame = -1
        previous_time = -1.0
        for obs in observations:
            if obs.frame_index <= previous_frame:
                raise ValueError(
                    f"{tracklet_id} frames must be strictly increasing"
                )
            if obs.timestamp_s <= previous_time:
                raise ValueError(
                    f"{tracklet_id} timestamps must be strictly increasing"
                )
            previous_frame = obs.frame_index
            previous_time = obs.timestamp_s

        if tracklet_id in result:
            raise ValueError(f"duplicate tracklet: {tracklet_id}")
        result[tracklet_id] = TrackletTrajectory(
            tracklet_id=tracklet_id,
            observations=observations,
        )
    return result


def load_identities(
    rows: Iterable[Mapping[str, Any]],
) -> list[IdentityAssignment]:
    result: list[IdentityAssignment] = []
    seen_ids: set[str] = set()
    seen_tracklets: set[str] = set()

    for row in rows:
        identity_id = str(row["identity_id"])
        tracklet_ids = tuple(
            sorted(str(value) for value in row["tracklet_ids"])
        )
        if not tracklet_ids:
            raise ValueError(f"{identity_id} has no tracklets")
        if identity_id in seen_ids:
            raise ValueError(f"duplicate identity: {identity_id}")
        for tracklet_id in tracklet_ids:
            if tracklet_id in seen_tracklets:
                raise ValueError(
                    f"tracklet assigned to multiple identities: {tracklet_id}"
                )
            seen_tracklets.add(tracklet_id)
        seen_ids.add(identity_id)
        result.append(
            IdentityAssignment(
                identity_id=identity_id,
                tracklet_ids=tracklet_ids,
            )
        )
    return sorted(result, key=lambda item: item.identity_id)


def load_identity_audits(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        identity_id = str(row["identity_id"])
        if identity_id in result:
            raise ValueError(f"duplicate identity audit: {identity_id}")
        result[identity_id] = dict(row)
    return result


def load_scored_links(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        pair = canonical_pair(
            str(row["tracklet_a"]),
            str(row["tracklet_b"]),
        )
        if pair in result:
            raise ValueError(f"duplicate scored link: {pair}")
        result[pair] = dict(row)
    return result


def deterministic_identity_color(identity_id: str) -> tuple[int, int, int]:
    """Return a deterministic bright BGR color without relying on hash()."""
    number = 0
    for char in identity_id:
        number = (number * 131 + ord(char)) % 1_000_003
    hue = ((number * 0.6180339887498949) % 1.0)
    saturation = 0.72
    value = 0.98
    r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
    return (
        int(round(b * 255)),
        int(round(g * 255)),
        int(round(r * 255)),
    )


def field_to_pixel(
    x: float,
    z: float,
    config: VisualAuditConfig,
) -> tuple[int, int]:
    usable_w = config.canvas_width_px - 2 * config.margin_px
    usable_h = config.canvas_height_px - 2 * config.margin_px

    left = -config.field_length_m / 2.0
    top = config.field_width_m / 2.0

    px = config.margin_px + (
        (x - left) / config.field_length_m
    ) * usable_w
    py = config.margin_px + (
        (top - z) / config.field_width_m
    ) * usable_h
    return int(round(px)), int(round(py))


def _draw_pitch(frame: np.ndarray, config: VisualAuditConfig) -> None:
    bgr_bg = (36, 92, 36)
    bgr_line = (235, 235, 235)
    frame[:] = bgr_bg

    m = config.margin_px
    w = config.canvas_width_px
    h = config.canvas_height_px
    left = m
    right = w - m
    top = m
    bottom = h - m
    cx = (left + right) // 2
    cy = (top + bottom) // 2

    cv2.rectangle(frame, (left, top), (right, bottom), bgr_line, 2)
    cv2.line(frame, (cx, top), (cx, bottom), bgr_line, 2)

    radius = int(round(
        (9.15 / config.field_length_m) * (right - left)
    ))
    cv2.circle(frame, (cx, cy), max(radius, 1), bgr_line, 2)
    cv2.circle(frame, (cx, cy), 3, bgr_line, -1)

    penalty_depth_px = int(round(
        (16.5 / config.field_length_m) * (right - left)
    ))
    penalty_width_px = int(round(
        (40.32 / config.field_width_m) * (bottom - top)
    ))
    six_depth_px = int(round(
        (5.5 / config.field_length_m) * (right - left)
    ))
    six_width_px = int(round(
        (18.32 / config.field_width_m) * (bottom - top)
    ))

    for side_x, sign in ((left, 1), (right, -1)):
        p_outer = side_x + sign * penalty_depth_px
        s_outer = side_x + sign * six_depth_px
        cv2.rectangle(
            frame,
            (min(side_x, p_outer), cy - penalty_width_px // 2),
            (max(side_x, p_outer), cy + penalty_width_px // 2),
            bgr_line,
            2,
        )
        cv2.rectangle(
            frame,
            (min(side_x, s_outer), cy - six_width_px // 2),
            (max(side_x, s_outer), cy + six_width_px // 2),
            bgr_line,
            2,
        )


def _identity_observations(
    identity: IdentityAssignment,
    tracklets: Mapping[str, TrackletTrajectory],
) -> list[Observation]:
    observations: list[Observation] = []
    for tracklet_id in identity.tracklet_ids:
        observations.extend(tracklets[tracklet_id].observations)
    return sorted(
        observations,
        key=lambda obs: (
            obs.timestamp_s,
            obs.frame_index,
            obs.tracklet_id,
        ),
    )


def build_timeline_index(
    identities: Sequence[IdentityAssignment],
    tracklets: Mapping[str, TrackletTrajectory],
) -> dict[str, list[Observation]]:
    return {
        identity.identity_id: _identity_observations(
            identity, tracklets
        )
        for identity in identities
    }


def _observations_in_window(
    observations: Sequence[Observation],
    start_s: float,
    end_s: float,
) -> list[Observation]:
    return [
        obs
        for obs in observations
        if start_s - 1e-12 <= obs.timestamp_s <= end_s + 1e-12
    ]


def render_topdown_frame(
    *,
    time_s: float,
    identities: Sequence[IdentityAssignment],
    timeline_index: Mapping[str, Sequence[Observation]],
    audits: Mapping[str, Mapping[str, Any]],
    config: VisualAuditConfig,
) -> np.ndarray:
    frame = np.zeros(
        (
            config.canvas_height_px,
            config.canvas_width_px,
            3,
        ),
        dtype=np.uint8,
    )
    _draw_pitch(frame, config)

    cv2.putText(
        frame,
        f"t={time_s:6.2f}s",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for identity in identities:
        identity_id = identity.identity_id
        observations = timeline_index[identity_id]
        color = deterministic_identity_color(identity_id)

        trail = _observations_in_window(
            observations,
            max(0.0, time_s - config.trail_seconds),
            time_s,
        )
        current = [
            obs
            for obs in observations
            if abs(obs.timestamp_s - time_s)
            <= config.observation_tolerance_s
        ]

        # Trail is intentionally drawn as observation dots, not segments:
        # this preserves the fact that there is no interpolated trajectory
        # between samples and keeps gaps visually empty.
        for obs in trail:
            point = field_to_pixel(obs.x, obs.z, config)
            age = max(0.0, time_s - obs.timestamp_s)
            if config.trail_seconds > 0:
                freshness = max(
                    0.2,
                    1.0 - age / config.trail_seconds,
                )
            else:
                freshness = 1.0
            radius = max(
                2,
                int(round(4 * freshness)),
            )
            cv2.circle(
                frame,
                point,
                radius,
                color,
                -1,
                cv2.LINE_AA,
            )

        for obs in current:
            point = field_to_pixel(obs.x, obs.z, config)
            cv2.circle(
                frame,
                point,
                config.point_radius_px,
                color,
                -1,
                cv2.LINE_AA,
            )
            cv2.circle(
                frame,
                point,
                config.point_radius_px + 2,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

            label = identity_id
            if config.show_tracklet_label:
                label += f"/{obs.tracklet_id}"

            flags = audits.get(identity_id, {}).get("flags", [])
            if flags:
                label += " *"

            cv2.putText(
                frame,
                label,
                (point[0] + 9, point[1] - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return frame


def _distance_m(
    a: tuple[float, float],
    b: tuple[float, float],
) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _backbone_pairs(
    audit: Mapping[str, Any] | None,
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    if not audit:
        return result
    backbone = audit.get("backbone")
    if not isinstance(backbone, Mapping):
        return result
    for row in backbone.get("edges", []):
        result.add(
            canonical_pair(
                str(row["tracklet_a"]),
                str(row["tracklet_b"]),
            )
        )
    return result


def _interval_components(
    trajectories: Sequence[TrackletTrajectory],
) -> list[list[TrackletTrajectory]]:
    """Group tracklets into components with continuous interval coverage.

    Overlapping/nested tracklets belong to the same component. A new component
    begins only when there is a genuine positive temporal gap in the identity's
    union coverage.
    """
    ordered = sorted(
        trajectories,
        key=lambda tracklet: (
            tracklet.start_time_s,
            tracklet.end_time_s,
            tracklet.tracklet_id,
        ),
    )
    components: list[list[TrackletTrajectory]] = []
    component_end = -math.inf

    for tracklet in ordered:
        if not components or tracklet.start_time_s > component_end + 1e-12:
            components.append([tracklet])
            component_end = tracklet.end_time_s
        else:
            components[-1].append(tracklet)
            component_end = max(component_end, tracklet.end_time_s)

    return components


def _overlap_record(
    *,
    identity: IdentityAssignment,
    left: TrackletTrajectory,
    right: TrackletTrajectory,
    backbone: set[tuple[str, str]],
    identity_flags: list[str],
    scored_links: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    pair = canonical_pair(left.tracklet_id, right.tracklet_id)
    scored = scored_links.get(pair)

    overlap_start = max(left.start_time_s, right.start_time_s)
    overlap_end = min(left.end_time_s, right.end_time_s)
    overlap_s = max(0.0, overlap_end - overlap_start)
    shared_frames = sorted(left.frame_set & right.frame_set)

    # Endpoint distance is not a physical separation during the overlap; keep
    # it only as an auxiliary diagnostic and name it explicitly as such.
    endpoint_distance = _distance_m(
        left.end_position,
        right.start_position,
    )

    review_flags = ["temporal_overlap"]
    if shared_frames:
        review_flags.append("shared_observation_frame")
    if pair not in backbone:
        review_flags.append("not_backbone_edge")

    return {
        "schema_version": 1,
        "visual_audit_version": VISUAL_AUDIT_VERSION,
        "identity_id": identity.identity_id,
        "tracklet_a": left.tracklet_id,
        "tracklet_b": right.tracklet_id,
        "relation": "overlap",
        "end_time_a_s": left.end_time_s,
        "start_time_b_s": right.start_time_s,
        "gap_s": -overlap_s,
        "overlap_s": overlap_s,
        "endpoint_distance_m": endpoint_distance,
        "implicit_speed_mps": None,
        "shared_frame_count": len(shared_frames),
        "first_shared_frame": (
            shared_frames[0] if shared_frames else None
        ),
        "is_backbone_edge": pair in backbone,
        "rank_score": (
            scored.get("rank_score")
            if scored is not None
            else None
        ),
        "evidence_confidence": (
            scored.get("evidence_confidence")
            if scored is not None
            else None
        ),
        "link_kind": (
            scored.get("link_kind")
            if scored is not None
            else None
        ),
        "identity_flags": identity_flags,
        "review_flags": review_flags,
    }


def _gap_record(
    *,
    identity: IdentityAssignment,
    left_component: Sequence[TrackletTrajectory],
    right_component: Sequence[TrackletTrajectory],
    backbone: set[tuple[str, str]],
    identity_flags: list[str],
    scored_links: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    # The true union-coverage gap is bounded by the latest-ending fragment of
    # the left component and earliest-starting fragment of the right component.
    left = max(
        left_component,
        key=lambda tracklet: (
            tracklet.end_time_s,
            tracklet.start_time_s,
            tracklet.tracklet_id,
        ),
    )
    right = min(
        right_component,
        key=lambda tracklet: (
            tracklet.start_time_s,
            tracklet.end_time_s,
            tracklet.tracklet_id,
        ),
    )

    pair = canonical_pair(left.tracklet_id, right.tracklet_id)
    scored = scored_links.get(pair)

    gap_s = right.start_time_s - left.end_time_s
    if gap_s < -1e-9:
        raise AssertionError("component handoff unexpectedly overlaps")
    gap_s = max(0.0, gap_s)

    distance = _distance_m(
        left.end_position,
        right.start_position,
    )
    implicit_speed = (
        distance / gap_s
        if gap_s > 1e-12
        else None
    )

    review_flags: list[str] = []
    if pair not in backbone:
        review_flags.append("not_backbone_edge")
    if gap_s > 1.0:
        review_flags.append("gap_gt_1s")
    if implicit_speed is not None and implicit_speed > 12.0:
        review_flags.append("raw_speed_gt_12mps")

    return {
        "schema_version": 1,
        "visual_audit_version": VISUAL_AUDIT_VERSION,
        "identity_id": identity.identity_id,
        "tracklet_a": left.tracklet_id,
        "tracklet_b": right.tracklet_id,
        "relation": "gap" if gap_s > 1e-12 else "touching",
        "end_time_a_s": left.end_time_s,
        "start_time_b_s": right.start_time_s,
        "gap_s": gap_s,
        "overlap_s": 0.0,
        "endpoint_distance_m": distance,
        "implicit_speed_mps": implicit_speed,
        "shared_frame_count": 0,
        "first_shared_frame": None,
        "is_backbone_edge": pair in backbone,
        "rank_score": (
            scored.get("rank_score")
            if scored is not None
            else None
        ),
        "evidence_confidence": (
            scored.get("evidence_confidence")
            if scored is not None
            else None
        ),
        "link_kind": (
            scored.get("link_kind")
            if scored is not None
            else None
        ),
        "identity_flags": identity_flags,
        "review_flags": review_flags,
    }


def build_transition_records(
    identities: Sequence[IdentityAssignment],
    tracklets: Mapping[str, TrackletTrajectory],
    audits: Mapping[str, Mapping[str, Any]],
    scored_links: Mapping[tuple[str, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build truthful temporal handoff + overlap diagnostics.

    Previous implementation paired tracklets merely by start-time adjacency.
    That is wrong when a short duplicate tracklet is nested inside a much
    longer one: it can manufacture a large fake gap to the next fragment.

    This version:
    1) reports every actual interval-overlap pair as an overlap diagnostic;
    2) builds union-coverage components;
    3) reports one genuine handoff between each adjacent coverage component,
       using the latest-ending left tracklet and earliest-starting right one.

    Therefore gap statistics match the identity timeline audit rather than the
    arbitrary order in which overlapping tracklets happen to start.
    """
    records: list[dict[str, Any]] = []

    for identity in identities:
        trajectories = [
            tracklets[tracklet_id]
            for tracklet_id in identity.tracklet_ids
        ]
        backbone = _backbone_pairs(audits.get(identity.identity_id))
        identity_flags = list(
            audits.get(identity.identity_id, {}).get("flags", [])
        )

        # Explicit overlap diagnostics.
        for left, right in itertools.combinations(
            sorted(
                trajectories,
                key=lambda tracklet: tracklet.tracklet_id,
            ),
            2,
        ):
            if (
                max(left.start_time_s, right.start_time_s)
                <= min(left.end_time_s, right.end_time_s) + 1e-12
            ):
                records.append(
                    _overlap_record(
                        identity=identity,
                        left=left,
                        right=right,
                        backbone=backbone,
                        identity_flags=identity_flags,
                        scored_links=scored_links,
                    )
                )

        # Genuine gaps in the union of all tracklet intervals.
        components = _interval_components(trajectories)
        for left_component, right_component in zip(
            components,
            components[1:],
        ):
            records.append(
                _gap_record(
                    identity=identity,
                    left_component=left_component,
                    right_component=right_component,
                    backbone=backbone,
                    identity_flags=identity_flags,
                    scored_links=scored_links,
                )
            )

    return sorted(
        records,
        key=lambda row: (
            row["identity_id"],
            min(
                row["end_time_a_s"],
                row["start_time_b_s"],
            ),
            row["relation"],
            row["tracklet_a"],
            row["tracklet_b"],
        ),
    )


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    n = len(ordered)

    def percentile(q: float) -> float:
        if n == 1:
            return ordered[0]
        pos = (n - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        alpha = pos - lo
        return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha

    return {
        "count": n,
        "min": ordered[0],
        "median": percentile(0.5),
        "p90": percentile(0.9),
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


def build_visual_summary(
    identities: Sequence[IdentityAssignment],
    tracklets: Mapping[str, TrackletTrajectory],
    audits: Mapping[str, Mapping[str, Any]],
    transitions: Sequence[Mapping[str, Any]],
    config: VisualAuditConfig,
) -> dict[str, Any]:
    start_time = min(
        trajectory.start_time_s
        for trajectory in tracklets.values()
    )
    end_time = max(
        trajectory.end_time_s
        for trajectory in tracklets.values()
    )

    gaps = [
        float(row["gap_s"])
        for row in transitions
        if row["gap_s"] is not None and row["gap_s"] > 0
    ]
    speeds = [
        float(row["implicit_speed_mps"])
        for row in transitions
        if row["implicit_speed_mps"] is not None
    ]
    distances = [
        float(row["endpoint_distance_m"])
        for row in transitions
    ]

    review_counts: dict[str, int] = {}
    relation_counts: dict[str, int] = {}
    for row in transitions:
        relation = str(row["relation"])
        relation_counts[relation] = relation_counts.get(relation, 0) + 1
        for flag in row["review_flags"]:
            review_counts[flag] = review_counts.get(flag, 0) + 1

    identity_flag_counts: dict[str, int] = {}
    for audit in audits.values():
        for flag in audit.get("flags", []):
            identity_flag_counts[flag] = (
                identity_flag_counts.get(flag, 0) + 1
            )

    return {
        "schema_version": 1,
        "visual_audit_version": VISUAL_AUDIT_VERSION,
        "provisional": True,
        "identity_stage_completed": False,
        "identity_count": len(identities),
        "tracklet_count": len(tracklets),
        "timeline_start_s": start_time,
        "timeline_end_s": end_time,
        "duration_s": end_time - start_time,
        "transition_count": len(transitions),
        "transition_relation_counts": dict(
            sorted(relation_counts.items())
        ),
        "backbone_transition_count": sum(
            bool(row["is_backbone_edge"])
            for row in transitions
        ),
        "transition_gap_s": _distribution(gaps),
        "transition_endpoint_distance_m": _distribution(distances),
        "transition_implicit_speed_mps": _distribution(speeds),
        "transition_review_flag_counts": dict(
            sorted(review_counts.items())
        ),
        "identity_flag_counts": dict(
            sorted(identity_flag_counts.items())
        ),
        "config": {
            "field_length_m": config.field_length_m,
            "field_width_m": config.field_width_m,
            "canvas_width_px": config.canvas_width_px,
            "canvas_height_px": config.canvas_height_px,
            "margin_px": config.margin_px,
            "output_fps": config.output_fps,
            "trail_seconds": config.trail_seconds,
            "observation_tolerance_s": config.observation_tolerance_s,
            "point_radius_px": config.point_radius_px,
            "show_tracklet_label": config.show_tracklet_label,
        },
    }


def render_topdown_video(
    *,
    output_path: Path,
    identities: Sequence[IdentityAssignment],
    tracklets: Mapping[str, TrackletTrajectory],
    audits: Mapping[str, Mapping[str, Any]],
    config: VisualAuditConfig,
) -> dict[str, Any]:
    timeline_index = build_timeline_index(
        identities, tracklets
    )
    start_time = min(
        trajectory.start_time_s
        for trajectory in tracklets.values()
    )
    end_time = max(
        trajectory.end_time_s
        for trajectory in tracklets.values()
    )

    frame_count = int(
        math.floor(
            (end_time - start_time) * config.output_fps
        )
    ) + 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        config.output_fps,
        (
            config.canvas_width_px,
            config.canvas_height_px,
        ),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"could not open video writer for {output_path}"
        )

    try:
        for frame_index in range(frame_count):
            time_s = start_time + (
                frame_index / config.output_fps
            )
            frame = render_topdown_frame(
                time_s=time_s,
                identities=identities,
                timeline_index=timeline_index,
                audits=audits,
                config=config,
            )
            writer.write(frame)
    finally:
        writer.release()

    return {
        "frame_count": frame_count,
        "fps": config.output_fps,
        "width": config.canvas_width_px,
        "height": config.canvas_height_px,
        "start_time_s": start_time,
        "end_time_s": end_time,
    }
