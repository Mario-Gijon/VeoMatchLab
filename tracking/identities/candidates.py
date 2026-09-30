"""Physical compatibility and candidate graph construction for Tracking V1.

Hito 3A turns local tracklets into a graph suitable for the future global
identity solver.

Principles
----------
* Tracklet != identity.
* Appearance is soft evidence only.
* Hard ``cannot_link`` constraints are produced only from time + field physics.
* A high ReID cosine can NEVER override a physical contradiction.
* Missing ReID is "unknown", not negative evidence.
* Overlapping tracklets are not automatically incompatible: two local
  tracklets can represent duplicate observations of the same physical player.
* Endpoint-velocity predictions are diagnostic/soft evidence only; hard
  reachability uses a deliberately conservative max-speed envelope.

Every unordered pair of tracklets is represented in the graph as either:
- ``candidate``: the future solver may consider the pair;
- ``cannot_link``: the future solver must not place both tracklets in one
  physical identity.

No must-link decision is made here.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np


GRAPH_VERSION = "physical-compatibility-v1"


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _position(x: float, z: float) -> np.ndarray:
    return np.asarray([x, z], dtype=np.float64)


def _distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("stats values must be finite and 1-D")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


@dataclass(frozen=True, slots=True)
class PhysicalCompatibilityConfig:
    """Conservative physics used only for hard candidate pruning.

    ``max_transition_speed_mps`` intentionally matches the local conservative
    tracklet builder's hard speed envelope.

    ``transition_position_slack_m`` absorbs field-projection noise and the fact
    that two endpoint observations are not exact physical player centers.

    Overlapping tracklets need robust spatial separation before we hard-ban
    them. Exact same-frame observations are used when available; otherwise we
    may compare positions by LOCAL interpolation only inside each tracklet and
    only across short observation brackets. This never fills tracking gaps or
    creates trajectory output: it is used solely to test whether two tracklets
    that coexist in time can physically be the same person.

    A single supported comparison uses a conservative hard threshold because
    one projected field position can be noisy. Multiple supported comparisons
    use the robust median separation threshold.
    """

    max_transition_speed_mps: float = 12.0
    transition_position_slack_m: float = 1.5
    overlap_close_median_separation_m: float = 1.5
    overlap_hard_median_separation_m: float = 3.0
    single_frame_hard_separation_m: float = 2.75
    overlap_interpolation_max_gap_s: float = 0.35
    velocity_window_s: float = 0.8
    velocity_min_samples: int = 3

    def __post_init__(self) -> None:
        if self.max_transition_speed_mps <= 0:
            raise ValueError("max_transition_speed_mps must be > 0")
        if self.transition_position_slack_m < 0:
            raise ValueError("transition_position_slack_m must be >= 0")
        if self.overlap_close_median_separation_m < 0:
            raise ValueError("overlap_close_median_separation_m must be >= 0")
        if (
            self.overlap_hard_median_separation_m
            <= self.overlap_close_median_separation_m
        ):
            raise ValueError(
                "overlap_hard_median_separation_m must exceed close threshold"
            )
        if (
            self.single_frame_hard_separation_m
            <= self.overlap_close_median_separation_m
        ):
            raise ValueError(
                "single_frame_hard_separation_m must exceed close threshold"
            )
        if self.overlap_interpolation_max_gap_s <= 0:
            raise ValueError(
                "overlap_interpolation_max_gap_s must be > 0"
            )
        if self.velocity_window_s <= 0:
            raise ValueError("velocity_window_s must be > 0")
        if self.velocity_min_samples < 2:
            raise ValueError("velocity_min_samples must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_transition_speed_mps": self.max_transition_speed_mps,
            "transition_position_slack_m": self.transition_position_slack_m,
            "overlap_close_median_separation_m": self.overlap_close_median_separation_m,
            "overlap_hard_median_separation_m": self.overlap_hard_median_separation_m,
            "single_frame_hard_separation_m": self.single_frame_hard_separation_m,
            "overlap_interpolation_max_gap_s": self.overlap_interpolation_max_gap_s,
            "velocity_window_s": self.velocity_window_s,
            "velocity_min_samples": self.velocity_min_samples,
        }


@dataclass(frozen=True, slots=True)
class TrackletObservation:
    detection_id: str
    frame_index: int
    timestamp_s: float
    x: float
    z: float
    confidence: float

    def __post_init__(self) -> None:
        if not self.detection_id:
            raise ValueError("detection_id must not be empty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if self.timestamp_s < 0:
            raise ValueError("timestamp_s must be >= 0")
        if not math.isfinite(self.x) or not math.isfinite(self.z):
            raise ValueError("x/z must be finite")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0,1]")

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.z)


@dataclass(frozen=True, slots=True)
class TrackletNode:
    tracklet_id: str
    observations: tuple[TrackletObservation, ...]
    start_velocity_mps: tuple[float, float] | None
    end_velocity_mps: tuple[float, float] | None
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not self.tracklet_id:
            raise ValueError("tracklet_id must not be empty")
        if not self.observations:
            raise ValueError("tracklet must contain observations")
        previous_frame = -1
        previous_time = -1.0
        for observation in self.observations:
            if observation.frame_index <= previous_frame:
                raise ValueError("tracklet frames must be strictly increasing")
            if observation.timestamp_s <= previous_time:
                raise ValueError("tracklet timestamps must be strictly increasing")
            previous_frame = observation.frame_index
            previous_time = observation.timestamp_s

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def first(self) -> TrackletObservation:
        return self.observations[0]

    @property
    def last(self) -> TrackletObservation:
        return self.observations[-1]

    @property
    def start_time_s(self) -> float:
        return self.first.timestamp_s

    @property
    def end_time_s(self) -> float:
        return self.last.timestamp_s

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([obs.confidence for obs in self.observations]))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "tracklet_id": self.tracklet_id,
            "observation_count": self.observation_count,
            "start_frame": self.first.frame_index,
            "end_frame": self.last.frame_index,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "start_position": {"x": self.first.x, "z": self.first.z},
            "end_position": {"x": self.last.x, "z": self.last.z},
            "start_velocity_mps": (
                {"vx": self.start_velocity_mps[0], "vz": self.start_velocity_mps[1]}
                if self.start_velocity_mps is not None
                else None
            ),
            "end_velocity_mps": (
                {"vx": self.end_velocity_mps[0], "vz": self.end_velocity_mps[1]}
                if self.end_velocity_mps is not None
                else None
            ),
            "mean_confidence": self.mean_confidence,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CandidateEdge:
    tracklet_a: str
    tracklet_b: str
    status: str
    reason: str
    temporal_relation: str
    physics: Mapping[str, Any]
    appearance: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("candidate edges require canonical sorted tracklet ids")
        if self.status not in {"candidate", "cannot_link"}:
            raise ValueError("status must be candidate or cannot_link")

    @property
    def hard_constraint(self) -> bool:
        return self.status == "cannot_link"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "graph_version": GRAPH_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "status": self.status,
            "reason": self.reason,
            "temporal_relation": self.temporal_relation,
            "hard_constraint": self.hard_constraint,
            "hard_constraint_source": "physics" if self.hard_constraint else None,
            "physics": dict(self.physics),
            "appearance": dict(self.appearance) if self.appearance is not None else None,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class CandidateGraph:
    nodes: tuple[TrackletNode, ...]
    edges: tuple[CandidateEdge, ...]

    def __post_init__(self) -> None:
        node_ids = [node.tracklet_id for node in self.nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("candidate graph contains duplicate node ids")
        expected = len(node_ids) * (len(node_ids) - 1) // 2
        if len(self.edges) != expected:
            raise ValueError(
                f"candidate graph must contain every unordered pair: "
                f"{len(self.edges)} != {expected}"
            )


def _weighted_velocity(
    observations: Sequence[TrackletObservation],
    *,
    side: str,
    config: PhysicalCompatibilityConfig,
) -> tuple[float, float] | None:
    if len(observations) < config.velocity_min_samples:
        return None

    if side == "start":
        anchor = observations[0].timestamp_s
        selected = [
            observation
            for observation in observations
            if observation.timestamp_s - anchor <= config.velocity_window_s + 1e-9
        ]
    elif side == "end":
        anchor = observations[-1].timestamp_s
        selected = [
            observation
            for observation in observations
            if anchor - observation.timestamp_s <= config.velocity_window_s + 1e-9
        ]
    else:
        raise ValueError("side must be start or end")

    if len(selected) < config.velocity_min_samples:
        return None

    times = np.asarray([obs.timestamp_s for obs in selected], dtype=np.float64)
    xs = np.asarray([obs.x for obs in selected], dtype=np.float64)
    zs = np.asarray([obs.z for obs in selected], dtype=np.float64)
    weights = np.asarray([max(obs.confidence, 1e-3) for obs in selected], dtype=np.float64)

    centered = times - float(np.average(times, weights=weights))
    denominator = float(np.sum(weights * centered * centered))
    if denominator <= 1e-12:
        return None

    x_center = float(np.average(xs, weights=weights))
    z_center = float(np.average(zs, weights=weights))
    vx = float(np.sum(weights * centered * (xs - x_center)) / denominator)
    vz = float(np.sum(weights * centered * (zs - z_center)) / denominator)

    if not math.isfinite(vx) or not math.isfinite(vz):
        return None
    return (vx, vz)


def tracklet_node_from_row(
    row: Mapping[str, Any],
    config: PhysicalCompatibilityConfig,
) -> TrackletNode:
    raw_observations = row.get("observations")
    if not isinstance(raw_observations, list) or not raw_observations:
        raise ValueError("tracklet row must contain a non-empty observations list")

    observations = tuple(
        TrackletObservation(
            detection_id=str(raw["detection_id"]),
            frame_index=int(raw["frame_index"]),
            timestamp_s=_finite_float("timestamp_s", raw["timestamp_s"]),
            x=_finite_float("x", raw["x"]),
            z=_finite_float("z", raw["z"]),
            confidence=_finite_float("confidence", raw["confidence"]),
        )
        for raw in raw_observations
    )
    return TrackletNode(
        tracklet_id=str(row["tracklet_id"]),
        observations=observations,
        start_velocity_mps=_weighted_velocity(observations, side="start", config=config),
        end_velocity_mps=_weighted_velocity(observations, side="end", config=config),
        metadata=dict(row.get("metadata") or {}),
    )


def iter_tracklet_nodes_jsonl(
    path: str | Path,
    config: PhysicalCompatibilityConfig,
) -> Iterator[TrackletNode]:
    source_path = Path(path)
    seen: set[str] = set()
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(f"{source_path}:{line_number} is not a JSON object")
            try:
                node = tracklet_node_from_row(payload, config)
            except Exception as exc:
                raise ValueError(f"{source_path}:{line_number}: {exc}") from exc
            if node.tracklet_id in seen:
                raise ValueError(f"duplicate tracklet_id: {node.tracklet_id}")
            seen.add(node.tracklet_id)
            yield node


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("appearance pair cannot reference one tracklet twice")
    return (a, b) if a < b else (b, a)


def load_appearance_pairwise_jsonl(
    path: str | Path,
) -> dict[tuple[str, str], dict[str, Any]]:
    """Load Hito 2F pairwise evidence without reinterpreting its scores."""
    source_path = Path(path)
    result: dict[tuple[str, str], dict[str, Any]] = {}
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"{source_path}:{line_number} is not a JSON object")
            key = _canonical_pair(
                str(payload["tracklet_a"]),
                str(payload["tracklet_b"]),
            )
            if key in result:
                raise ValueError(f"duplicate appearance pair: {key}")
            # Keep the complete Hito 2F evidence contract.  3A never converts
            # similarity into a hard physical decision.
            result[key] = payload
    return result


def _shared_frame_distances(
    a: TrackletNode,
    b: TrackletNode,
) -> tuple[list[int], list[float]]:
    by_frame_a = {obs.frame_index: obs for obs in a.observations}
    by_frame_b = {obs.frame_index: obs for obs in b.observations}
    shared_frames = sorted(set(by_frame_a) & set(by_frame_b))
    distances = [
        _distance(
            by_frame_a[frame].position,
            by_frame_b[frame].position,
        )
        for frame in shared_frames
    ]
    return shared_frames, distances



def _position_at_time(
    node: TrackletNode,
    timestamp_s: float,
    config: PhysicalCompatibilityConfig,
) -> tuple[tuple[float, float], str] | None:
    """Return a supported position at ``timestamp_s``.

    Exact observations are preferred. Otherwise interpolate only between the
    immediately bracketing observations and only when their temporal distance
    is <= ``overlap_interpolation_max_gap_s``.

    There is deliberately NO extrapolation. The helper therefore cannot bridge
    an unseen temporal gap and cannot invent tracking output.
    """
    observations = node.observations
    epsilon = 1e-9

    for observation in observations:
        if abs(observation.timestamp_s - timestamp_s) <= epsilon:
            return observation.position, "exact"

    previous = None
    following = None
    for observation in observations:
        if observation.timestamp_s < timestamp_s:
            previous = observation
            continue
        if observation.timestamp_s > timestamp_s:
            following = observation
            break

    if previous is None or following is None:
        return None

    span_s = following.timestamp_s - previous.timestamp_s
    if span_s <= 0:
        return None
    if span_s > config.overlap_interpolation_max_gap_s + epsilon:
        return None

    alpha = (timestamp_s - previous.timestamp_s) / span_s
    x = previous.x + alpha * (following.x - previous.x)
    z = previous.z + alpha * (following.z - previous.z)
    return (float(x), float(z)), "interpolated"


def _synchronized_overlap_distances(
    a: TrackletNode,
    b: TrackletNode,
    config: PhysicalCompatibilityConfig,
) -> dict[str, Any]:
    """Compare two overlapping tracklets at physically comparable times.

    Candidate timestamps are real observation times from either tracklet that
    fall inside the temporal intersection. A timestamp contributes only if both
    tracklets have an exact position or a short, locally interpolated position
    at that time.
    """
    overlap_start_s = max(a.start_time_s, b.start_time_s)
    overlap_end_s = min(a.end_time_s, b.end_time_s)
    if overlap_start_s > overlap_end_s + 1e-9:
        return {
            "interval_overlap_s": 0.0,
            "support_count": 0,
            "exact_exact_count": 0,
            "interpolated_support_count": 0,
            "support_times_s": [],
            "distances_m": [],
            "separation_stats": _stats([]),
        }

    candidate_times = sorted(
        {
            observation.timestamp_s
            for node in (a, b)
            for observation in node.observations
            if (
                overlap_start_s - 1e-9
                <= observation.timestamp_s
                <= overlap_end_s + 1e-9
            )
        }
    )

    distances: list[float] = []
    support_times: list[float] = []
    exact_exact_count = 0
    interpolated_support_count = 0

    for timestamp_s in candidate_times:
        position_a = _position_at_time(a, timestamp_s, config)
        position_b = _position_at_time(b, timestamp_s, config)
        if position_a is None or position_b is None:
            continue

        point_a, mode_a = position_a
        point_b, mode_b = position_b
        distances.append(_distance(point_a, point_b))
        support_times.append(float(timestamp_s))

        if mode_a == "exact" and mode_b == "exact":
            exact_exact_count += 1
        else:
            interpolated_support_count += 1

    return {
        "interval_overlap_s": max(
            0.0,
            overlap_end_s - overlap_start_s,
        ),
        "support_count": len(distances),
        "exact_exact_count": exact_exact_count,
        "interpolated_support_count": interpolated_support_count,
        "support_times_s": support_times,
        "distances_m": distances,
        "separation_stats": _stats(distances),
    }


def _velocity_payload(
    velocity: tuple[float, float] | None,
) -> dict[str, float] | None:
    if velocity is None:
        return None
    return {
        "vx": float(velocity[0]),
        "vz": float(velocity[1]),
        "speed_mps": float(math.hypot(*velocity)),
    }


def _transition_physics(
    predecessor: TrackletNode,
    successor: TrackletNode,
    config: PhysicalCompatibilityConfig,
) -> dict[str, Any]:
    gap_s = successor.start_time_s - predecessor.end_time_s
    if gap_s < -1e-9:
        raise ValueError("transition_physics requires non-overlapping ordered tracklets")
    gap_s = max(0.0, gap_s)

    endpoint_distance_m = _distance(
        predecessor.last.position,
        successor.first.position,
    )
    reachability_limit_m = (
        config.transition_position_slack_m
        + config.max_transition_speed_mps * gap_s
    )
    reachability_margin_m = reachability_limit_m - endpoint_distance_m

    if gap_s > 1e-12:
        raw_endpoint_speed_mps = endpoint_distance_m / gap_s
        required_speed_after_slack_mps = max(
            0.0,
            endpoint_distance_m - config.transition_position_slack_m,
        ) / gap_s
    else:
        raw_endpoint_speed_mps = None
        required_speed_after_slack_mps = (
            0.0
            if endpoint_distance_m <= config.transition_position_slack_m
            else None
        )

    forward_prediction_error_m = None
    if predecessor.end_velocity_mps is not None:
        predicted = (
            _position(predecessor.last.x, predecessor.last.z)
            + np.asarray(predecessor.end_velocity_mps) * gap_s
        )
        forward_prediction_error_m = _distance(
            predicted,
            successor.first.position,
        )

    backward_prediction_error_m = None
    if successor.start_velocity_mps is not None:
        predicted_previous = (
            _position(successor.first.x, successor.first.z)
            - np.asarray(successor.start_velocity_mps) * gap_s
        )
        backward_prediction_error_m = _distance(
            predicted_previous,
            predecessor.last.position,
        )

    prediction_errors = [
        value
        for value in (forward_prediction_error_m, backward_prediction_error_m)
        if value is not None
    ]

    direction_cosine = None
    if (
        predecessor.end_velocity_mps is not None
        and successor.start_velocity_mps is not None
    ):
        va = np.asarray(predecessor.end_velocity_mps, dtype=np.float64)
        vb = np.asarray(successor.start_velocity_mps, dtype=np.float64)
        norm = float(np.linalg.norm(va) * np.linalg.norm(vb))
        if norm > 1e-12:
            direction_cosine = float(np.clip(np.dot(va, vb) / norm, -1.0, 1.0))

    return {
        "predecessor": predecessor.tracklet_id,
        "successor": successor.tracklet_id,
        "gap_s": gap_s,
        "endpoint_distance_m": endpoint_distance_m,
        "raw_endpoint_speed_mps": raw_endpoint_speed_mps,
        "position_slack_m": config.transition_position_slack_m,
        "required_speed_after_slack_mps": required_speed_after_slack_mps,
        "max_transition_speed_mps": config.max_transition_speed_mps,
        "reachability_limit_m": reachability_limit_m,
        "reachability_margin_m": reachability_margin_m,
        "predecessor_end_velocity": _velocity_payload(
            predecessor.end_velocity_mps
        ),
        "successor_start_velocity": _velocity_payload(
            successor.start_velocity_mps
        ),
        "forward_prediction_error_m": forward_prediction_error_m,
        "backward_prediction_error_m": backward_prediction_error_m,
        "prediction_error_mean_m": (
            float(np.mean(prediction_errors)) if prediction_errors else None
        ),
        "velocity_direction_cosine": direction_cosine,
    }


def build_candidate_edge(
    a: TrackletNode,
    b: TrackletNode,
    config: PhysicalCompatibilityConfig,
    *,
    appearance: Mapping[str, Any] | None = None,
) -> CandidateEdge:
    if a.tracklet_id == b.tracklet_id:
        raise ValueError("candidate edge requires two different tracklets")

    if b.tracklet_id < a.tracklet_id:
        a, b = b, a

    shared_frames, shared_distances = _shared_frame_distances(a, b)
    intervals_overlap = (
        max(a.start_time_s, b.start_time_s)
        <= min(a.end_time_s, b.end_time_s) + 1e-9
    )

    if intervals_overlap:
        synchronized = _synchronized_overlap_distances(
            a,
            b,
            config,
        )
        support_count = int(synchronized["support_count"])
        separation = synchronized["separation_stats"]

        if support_count > 0:
            median = float(separation["median"])
            if support_count == 1:
                hard_separated = (
                    float(separation["max"])
                    > config.single_frame_hard_separation_m
                )
            else:
                hard_separated = (
                    median > config.overlap_hard_median_separation_m
                )

            if hard_separated:
                status = "cannot_link"
                reason = (
                    "simultaneous_spatial_separation"
                    if shared_frames
                    else "overlap_interpolated_spatial_separation"
                )
            elif median <= config.overlap_close_median_separation_m:
                status = "candidate"
                reason = "overlap_close"
            else:
                status = "candidate"
                reason = "overlap_ambiguous"

            temporal_relation = (
                "simultaneous_samples"
                if shared_frames
                else "overlapping_intervals_synchronized"
            )
            physics = {
                "shared_frame_count": len(shared_frames),
                "first_shared_frame": (
                    shared_frames[0] if shared_frames else None
                ),
                "last_shared_frame": (
                    shared_frames[-1] if shared_frames else None
                ),
                # Legacy name retained because 3B-A consumes this key. It now
                # represents all supported contemporaneous comparisons, exact
                # or locally interpolated.
                "shared_frame_separation_m": separation,
                "synchronized_separation_m": separation,
                "synchronized_support_count": support_count,
                "exact_exact_support_count": int(
                    synchronized["exact_exact_count"]
                ),
                "interpolated_support_count": int(
                    synchronized["interpolated_support_count"]
                ),
                "first_support_time_s": (
                    synchronized["support_times_s"][0]
                    if synchronized["support_times_s"]
                    else None
                ),
                "last_support_time_s": (
                    synchronized["support_times_s"][-1]
                    if synchronized["support_times_s"]
                    else None
                ),
                "interval_overlap_s": synchronized[
                    "interval_overlap_s"
                ],
                "close_median_threshold_m": (
                    config.overlap_close_median_separation_m
                ),
                "hard_median_threshold_m": (
                    config.overlap_hard_median_separation_m
                ),
                "single_frame_hard_threshold_m": (
                    config.single_frame_hard_separation_m
                ),
                "overlap_interpolation_max_gap_s": (
                    config.overlap_interpolation_max_gap_s
                ),
            }
        else:
            # Intervals overlap, but neither exact nor safely interpolated
            # contemporaneous positions are available. Do not manufacture a
            # hard constraint from unsupported geometry.
            status = "candidate"
            reason = "overlap_unsynchronized"
            temporal_relation = "overlapping_intervals_without_synchronized_support"
            physics = {
                "shared_frame_count": len(shared_frames),
                "first_shared_frame": (
                    shared_frames[0] if shared_frames else None
                ),
                "last_shared_frame": (
                    shared_frames[-1] if shared_frames else None
                ),
                "synchronized_support_count": 0,
                "exact_exact_support_count": 0,
                "interpolated_support_count": 0,
                "interval_overlap_s": synchronized[
                    "interval_overlap_s"
                ],
                "overlap_interpolation_max_gap_s": (
                    config.overlap_interpolation_max_gap_s
                ),
            }

    else:
        if a.end_time_s < b.start_time_s:
            predecessor, successor = a, b
        else:
            predecessor, successor = b, a

        transition = _transition_physics(predecessor, successor, config)
        reachable = transition["reachability_margin_m"] >= -1e-9

        status = "candidate" if reachable else "cannot_link"
        reason = "transition_reachable" if reachable else "transition_unreachable"
        temporal_relation = "non_overlapping"
        physics = transition

    return CandidateEdge(
        tracklet_a=a.tracklet_id,
        tracklet_b=b.tracklet_id,
        status=status,
        reason=reason,
        temporal_relation=temporal_relation,
        physics=physics,
        appearance=dict(appearance) if appearance is not None else None,
    )


def build_candidate_graph(
    nodes: Sequence[TrackletNode],
    config: PhysicalCompatibilityConfig,
    *,
    appearance_by_pair: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> CandidateGraph:
    ordered = tuple(sorted(nodes, key=lambda node: node.tracklet_id))
    appearance_by_pair = appearance_by_pair or {}

    node_ids = {node.tracklet_id for node in ordered}
    if len(node_ids) != len(ordered):
        raise ValueError("duplicate tracklet ids")

    unknown_appearance_nodes = {
        tracklet_id
        for pair in appearance_by_pair
        for tracklet_id in pair
        if tracklet_id not in node_ids
    }
    if unknown_appearance_nodes:
        raise ValueError(
            "appearance evidence references unknown tracklets: "
            f"{sorted(unknown_appearance_nodes)[:10]}"
        )

    edges: list[CandidateEdge] = []
    for a, b in itertools.combinations(ordered, 2):
        key = (a.tracklet_id, b.tracklet_id)
        edges.append(
            build_candidate_edge(
                a,
                b,
                config,
                appearance=appearance_by_pair.get(key),
            )
        )

    return CandidateGraph(nodes=ordered, edges=tuple(edges))


def build_candidate_graph_summary(
    graph: CandidateGraph,
    config: PhysicalCompatibilityConfig,
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}
    temporal_counts: dict[str, int] = {}

    appearance_count = 0
    candidate_with_appearance = 0
    cannot_link_with_appearance = 0

    for edge in graph.edges:
        reason_counts[edge.reason] = reason_counts.get(edge.reason, 0) + 1
        status_counts[edge.status] = status_counts.get(edge.status, 0) + 1
        temporal_counts[edge.temporal_relation] = (
            temporal_counts.get(edge.temporal_relation, 0) + 1
        )
        if edge.appearance is not None:
            appearance_count += 1
            if edge.status == "candidate":
                candidate_with_appearance += 1
            else:
                cannot_link_with_appearance += 1

    candidate_count = status_counts.get("candidate", 0)
    cannot_link_count = status_counts.get("cannot_link", 0)

    return {
        "schema_version": 1,
        "graph_version": GRAPH_VERSION,
        "identity_decisions": False,
        "hard_constraints_only_from_physics": True,
        "node_count": len(graph.nodes),
        "pair_count": len(graph.edges),
        "candidate_count": candidate_count,
        "cannot_link_count": cannot_link_count,
        "candidate_fraction": (
            candidate_count / len(graph.edges) if graph.edges else 0.0
        ),
        "cannot_link_fraction": (
            cannot_link_count / len(graph.edges) if graph.edges else 0.0
        ),
        "appearance_attached_count": appearance_count,
        "appearance_missing_count": len(graph.edges) - appearance_count,
        "candidate_with_appearance_count": candidate_with_appearance,
        "cannot_link_with_appearance_count": cannot_link_with_appearance,
        "status_counts": dict(sorted(status_counts.items())),
        "reason_counts": dict(sorted(reason_counts.items())),
        "temporal_relation_counts": dict(sorted(temporal_counts.items())),
        "config": config.to_dict(),
    }
