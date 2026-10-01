"""Hito 7B.1: sparse temporal/physical index over conservative tracklets.

This module is deliberately *not* an identity solver.

Its job is to turn the set of short conservative Tracking V1 tracklets into
cheap indices and physical evidence that later offline stages can use without
enumerating every unordered tracklet pair.

Football-specific design assumptions
------------------------------------
* A tracklet is not a player identity.
* Exact simultaneous observations are extremely valuable: two sufficiently
  separated people seen at the same instant cannot be the same actor.
* Local disappearance/reappearance hypotheses should be queried by time, not
  by scanning N choose 2 pairs.
* Physics can make an association impossible, but it cannot by itself prove an
  identity.
* Missing appearance evidence is irrelevant here. ReID/team/jersey evidence
  belongs to later Hito 7B/7C stages.
* The future solver is offline/global. Nothing in this module commits an
  identity decision.

The index exposes:
1. exact frame -> observed-tracklet membership;
2. accumulated exact co-observation evidence;
3. hard coexistence conflicts from robust field-space separation;
4. an ordered start-time index for local successor queries;
5. physically explainable endpoint-transition features.

No all-pairs appearance graph and no persistent actor IDs are produced here.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import statistics
from typing import Any, Iterable, Iterator, Mapping, Sequence

from tracking.tracklets.models import Tracklet, TrackletObservation


ACTOR_INDEX_VERSION = "sparse-tracklet-index-v1"


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise ValueError("pair requires two distinct tracklets")
    return (first, second) if first < second else (second, first)


def _distance(
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


@dataclass(frozen=True, slots=True)
class SparseTrackletIndexConfig:
    """Conservative physical/indexing parameters for Hito 7B.1.

    The speed/slack and coexistence thresholds intentionally match the already
    validated Tracking V1 physical candidate semantics. They are *feasibility*
    limits, not identity scores.

    ``local_gap_horizon_s`` is only an indexing/retrieval horizon. A pair
    outside this horizon is NOT a cannot-link: it may later be recovered by
    appearance, team, jersey-number or other global evidence.
    """

    local_gap_horizon_s: float = 1.0
    max_transition_speed_mps: float = 12.0
    transition_position_slack_m: float = 1.5
    single_frame_hard_separation_m: float = 2.75
    overlap_hard_median_separation_m: float = 3.0
    close_overlap_median_separation_m: float = 1.5
    velocity_window_s: float = 0.8
    velocity_min_samples: int = 3
    time_epsilon_s: float = 1e-9

    def __post_init__(self) -> None:
        positive = (
            "local_gap_horizon_s",
            "max_transition_speed_mps",
            "transition_position_slack_m",
            "single_frame_hard_separation_m",
            "overlap_hard_median_separation_m",
            "close_overlap_median_separation_m",
            "velocity_window_s",
        )
        for name in positive:
            value = _finite_float(name, getattr(self, name))
            if value <= 0.0:
                raise ValueError(f"{name} must be > 0")

        if self.velocity_min_samples < 2:
            raise ValueError("velocity_min_samples must be >= 2")
        if self.time_epsilon_s < 0.0:
            raise ValueError("time_epsilon_s must be >= 0")
        if (
            self.close_overlap_median_separation_m
            >= self.overlap_hard_median_separation_m
        ):
            raise ValueError(
                "close overlap threshold must be below hard overlap threshold"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EndpointVelocity:
    vx_mps: float
    vz_mps: float
    sample_count: int

    @property
    def speed_mps(self) -> float:
        return math.hypot(self.vx_mps, self.vz_mps)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vx_mps": self.vx_mps,
            "vz_mps": self.vz_mps,
            "speed_mps": self.speed_mps,
            "sample_count": self.sample_count,
        }


@dataclass(frozen=True, slots=True)
class SparseTrackletNode:
    """Cheap immutable summary of one conservative local tracklet."""

    tracklet_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    first_x: float
    first_z: float
    last_x: float
    last_z: float
    observation_count: int
    mean_confidence: float
    start_velocity: EndpointVelocity | None
    end_velocity: EndpointVelocity | None

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    @property
    def first_position(self) -> tuple[float, float]:
        return (self.first_x, self.first_z)

    @property
    def last_position(self) -> tuple[float, float]:
        return (self.last_x, self.last_z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index_version": ACTOR_INDEX_VERSION,
            "tracklet_id": self.tracklet_id,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "first_position": {"x": self.first_x, "z": self.first_z},
            "last_position": {"x": self.last_x, "z": self.last_z},
            "observation_count": self.observation_count,
            "mean_confidence": self.mean_confidence,
            "start_velocity": (
                self.start_velocity.to_dict()
                if self.start_velocity is not None
                else None
            ),
            "end_velocity": (
                self.end_velocity.to_dict()
                if self.end_velocity is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class CoobservedPairEvidence:
    """Exact simultaneous field-space evidence accumulated across frames."""

    tracklet_a: str
    tracklet_b: str
    shared_frame_count: int
    first_shared_frame: int
    last_shared_frame: int
    min_separation_m: float
    median_separation_m: float
    max_separation_m: float
    mean_separation_m: float

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("co-observed pair requires canonical ordering")
        if self.shared_frame_count < 1:
            raise ValueError("shared_frame_count must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index_version": ACTOR_INDEX_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "shared_frame_count": self.shared_frame_count,
            "first_shared_frame": self.first_shared_frame,
            "last_shared_frame": self.last_shared_frame,
            "separation_m": {
                "min": self.min_separation_m,
                "median": self.median_separation_m,
                "max": self.max_separation_m,
                "mean": self.mean_separation_m,
            },
        }


@dataclass(frozen=True, slots=True)
class HardConflict:
    """A physically impossible same-actor relation."""

    tracklet_a: str
    tracklet_b: str
    reason: str
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("hard conflict requires canonical ordering")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index_version": ACTOR_INDEX_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "reason": self.reason,
            "hard_constraint": True,
            "hard_constraint_source": "field_physics",
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class TransitionFeatures:
    """Explainable endpoint evidence for A -> B.

    ``physically_reachable`` is only a hard feasibility result. ``True`` does
    not mean the tracklets belong to the same player.
    """

    predecessor_tracklet_id: str
    successor_tracklet_id: str
    gap_s: float
    endpoint_distance_m: float
    reachability_limit_m: float
    reachability_margin_m: float
    raw_endpoint_speed_mps: float | None
    required_speed_after_slack_mps: float | None
    predecessor_end_velocity: EndpointVelocity | None
    successor_start_velocity: EndpointVelocity | None
    forward_prediction_error_m: float | None
    backward_prediction_error_m: float | None
    velocity_direction_cosine: float | None
    predecessor_end_confidence: float
    successor_start_confidence: float
    physically_reachable: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "index_version": ACTOR_INDEX_VERSION,
            "predecessor_tracklet_id": self.predecessor_tracklet_id,
            "successor_tracklet_id": self.successor_tracklet_id,
            "gap_s": self.gap_s,
            "endpoint_distance_m": self.endpoint_distance_m,
            "reachability_limit_m": self.reachability_limit_m,
            "reachability_margin_m": self.reachability_margin_m,
            "raw_endpoint_speed_mps": self.raw_endpoint_speed_mps,
            "required_speed_after_slack_mps": self.required_speed_after_slack_mps,
            "predecessor_end_velocity": (
                self.predecessor_end_velocity.to_dict()
                if self.predecessor_end_velocity is not None
                else None
            ),
            "successor_start_velocity": (
                self.successor_start_velocity.to_dict()
                if self.successor_start_velocity is not None
                else None
            ),
            "forward_prediction_error_m": self.forward_prediction_error_m,
            "backward_prediction_error_m": self.backward_prediction_error_m,
            "velocity_direction_cosine": self.velocity_direction_cosine,
            "predecessor_end_confidence": self.predecessor_end_confidence,
            "successor_start_confidence": self.successor_start_confidence,
            "physically_reachable": self.physically_reachable,
            "identity_decision": None,
        }


def _weighted_velocity(
    observations: Sequence[TrackletObservation],
    *,
    side: str,
    config: SparseTrackletIndexConfig,
) -> EndpointVelocity | None:
    """Confidence-weighted endpoint linear velocity, matching V1 semantics."""

    if len(observations) < config.velocity_min_samples:
        return None

    if side == "start":
        anchor = observations[0].timestamp_s
        selected = [
            observation
            for observation in observations
            if observation.timestamp_s - anchor
            <= config.velocity_window_s + config.time_epsilon_s
        ]
    elif side == "end":
        anchor = observations[-1].timestamp_s
        selected = [
            observation
            for observation in observations
            if anchor - observation.timestamp_s
            <= config.velocity_window_s + config.time_epsilon_s
        ]
    else:
        raise ValueError("side must be 'start' or 'end'")

    if len(selected) < config.velocity_min_samples:
        return None

    weights = [max(float(item.confidence), 1e-3) for item in selected]
    total_weight = sum(weights)
    mean_t = sum(
        weight * item.timestamp_s
        for weight, item in zip(weights, selected)
    ) / total_weight
    mean_x = sum(
        weight * item.x
        for weight, item in zip(weights, selected)
    ) / total_weight
    mean_z = sum(
        weight * item.z
        for weight, item in zip(weights, selected)
    ) / total_weight

    denominator = sum(
        weight * (item.timestamp_s - mean_t) ** 2
        for weight, item in zip(weights, selected)
    )
    if denominator <= 1e-12:
        return None

    vx = sum(
        weight * (item.timestamp_s - mean_t) * (item.x - mean_x)
        for weight, item in zip(weights, selected)
    ) / denominator
    vz = sum(
        weight * (item.timestamp_s - mean_t) * (item.z - mean_z)
        for weight, item in zip(weights, selected)
    ) / denominator

    if not math.isfinite(vx) or not math.isfinite(vz):
        return None

    return EndpointVelocity(
        vx_mps=float(vx),
        vz_mps=float(vz),
        sample_count=len(selected),
    )


def _node_from_tracklet(
    tracklet: Tracklet,
    config: SparseTrackletIndexConfig,
) -> SparseTrackletNode:
    first = tracklet.observations[0]
    last = tracklet.observations[-1]
    return SparseTrackletNode(
        tracklet_id=tracklet.tracklet_id,
        start_frame=tracklet.start_frame,
        end_frame=tracklet.end_frame,
        start_time_s=tracklet.start_timestamp_s,
        end_time_s=tracklet.end_timestamp_s,
        first_x=first.x,
        first_z=first.z,
        last_x=last.x,
        last_z=last.z,
        observation_count=tracklet.observation_count,
        mean_confidence=tracklet.mean_confidence,
        start_velocity=_weighted_velocity(
            tracklet.observations,
            side="start",
            config=config,
        ),
        end_velocity=_weighted_velocity(
            tracklet.observations,
            side="end",
            config=config,
        ),
    )


class SparseTrackletIndex:
    """Sparse exact-frame + start-time index for offline actor reconstruction."""

    def __init__(
        self,
        tracklets: Iterable[Tracklet],
        config: SparseTrackletIndexConfig | None = None,
    ) -> None:
        self.config = config or SparseTrackletIndexConfig()

        normalized = tuple(
            item if isinstance(item, Tracklet) else Tracklet.from_dict(item)
            for item in tracklets
        )
        ids = [item.tracklet_id for item in normalized]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate tracklet_id in sparse index")

        self._tracklets_by_id = {
            item.tracklet_id: item
            for item in normalized
        }
        self._nodes_by_id = {
            item.tracklet_id: _node_from_tracklet(item, self.config)
            for item in normalized
        }

        self._nodes_by_start = tuple(
            sorted(
                self._nodes_by_id.values(),
                key=lambda node: (
                    node.start_time_s,
                    node.start_frame,
                    node.tracklet_id,
                ),
            )
        )
        self._start_times = tuple(
            node.start_time_s
            for node in self._nodes_by_start
        )

        frame_members: dict[
            int,
            list[tuple[str, TrackletObservation]],
        ] = defaultdict(list)
        for tracklet in normalized:
            for observation in tracklet.observations:
                frame_members[observation.frame_index].append(
                    (tracklet.tracklet_id, observation)
                )

        self._frame_members = {
            frame: tuple(sorted(values, key=lambda item: item[0]))
            for frame, values in sorted(frame_members.items())
        }
        self._coobserved_pairs = self._build_coobserved_pairs()

    @property
    def tracklet_count(self) -> int:
        return len(self._nodes_by_id)

    @property
    def observation_count(self) -> int:
        return sum(node.observation_count for node in self._nodes_by_id.values())

    @property
    def observed_frame_count(self) -> int:
        return len(self._frame_members)

    @property
    def nodes(self) -> tuple[SparseTrackletNode, ...]:
        return tuple(self._nodes_by_id[key] for key in sorted(self._nodes_by_id))

    @property
    def coobserved_pairs(self) -> tuple[CoobservedPairEvidence, ...]:
        return self._coobserved_pairs

    def node(self, tracklet_id: str) -> SparseTrackletNode:
        try:
            return self._nodes_by_id[tracklet_id]
        except KeyError as exc:
            raise KeyError(f"unknown tracklet_id: {tracklet_id!r}") from exc

    def frame_tracklet_ids(self, frame_index: int) -> tuple[str, ...]:
        return tuple(
            tracklet_id
            for tracklet_id, _ in self._frame_members.get(frame_index, ())
        )

    def query_local_successors(
        self,
        predecessor_tracklet_id: str,
        *,
        horizon_s: float | None = None,
    ) -> tuple[SparseTrackletNode, ...]:
        """Return only temporally nearby possible successors.

        This is an indexing operation, not a physical/identity decision.
        Tracklets outside the horizon remain globally unresolved and may later
        be recovered by ReID/team/jersey/global context.
        """

        predecessor = self.node(predecessor_tracklet_id)
        horizon = (
            self.config.local_gap_horizon_s
            if horizon_s is None
            else _finite_float("horizon_s", horizon_s)
        )
        if horizon < 0.0:
            raise ValueError("horizon_s must be >= 0")

        epsilon = self.config.time_epsilon_s
        low = bisect_left(
            self._start_times,
            predecessor.end_time_s - epsilon,
        )
        high = bisect_right(
            self._start_times,
            predecessor.end_time_s + horizon + epsilon,
        )

        result = [
            node
            for node in self._nodes_by_start[low:high]
            if node.tracklet_id != predecessor.tracklet_id
            and node.start_time_s >= predecessor.end_time_s - epsilon
        ]
        result.sort(
            key=lambda node: (
                max(0.0, node.start_time_s - predecessor.end_time_s),
                node.start_time_s,
                node.tracklet_id,
            )
        )
        return tuple(result)

    def transition_features(
        self,
        predecessor_tracklet_id: str,
        successor_tracklet_id: str,
    ) -> TransitionFeatures:
        predecessor = self.node(predecessor_tracklet_id)
        successor = self.node(successor_tracklet_id)

        gap_s = successor.start_time_s - predecessor.end_time_s
        if gap_s < -self.config.time_epsilon_s:
            raise ValueError(
                "transition_features requires a non-overlapping ordered pair"
            )
        gap_s = max(0.0, gap_s)

        endpoint_distance_m = _distance(
            predecessor.last_position,
            successor.first_position,
        )
        reachability_limit_m = (
            self.config.transition_position_slack_m
            + self.config.max_transition_speed_mps * gap_s
        )
        margin = reachability_limit_m - endpoint_distance_m

        if gap_s > self.config.time_epsilon_s:
            raw_speed = endpoint_distance_m / gap_s
            required_speed = max(
                0.0,
                endpoint_distance_m - self.config.transition_position_slack_m,
            ) / gap_s
        else:
            raw_speed = None
            required_speed = (
                0.0
                if endpoint_distance_m <= self.config.transition_position_slack_m
                else None
            )

        forward_error = None
        if predecessor.end_velocity is not None:
            predicted = (
                predecessor.last_x + predecessor.end_velocity.vx_mps * gap_s,
                predecessor.last_z + predecessor.end_velocity.vz_mps * gap_s,
            )
            forward_error = _distance(predicted, successor.first_position)

        backward_error = None
        if successor.start_velocity is not None:
            predicted_previous = (
                successor.first_x - successor.start_velocity.vx_mps * gap_s,
                successor.first_z - successor.start_velocity.vz_mps * gap_s,
            )
            backward_error = _distance(
                predicted_previous,
                predecessor.last_position,
            )

        direction_cosine = None
        if (
            predecessor.end_velocity is not None
            and successor.start_velocity is not None
        ):
            ax = predecessor.end_velocity.vx_mps
            az = predecessor.end_velocity.vz_mps
            bx = successor.start_velocity.vx_mps
            bz = successor.start_velocity.vz_mps
            denominator = math.hypot(ax, az) * math.hypot(bx, bz)
            if denominator > 1e-12:
                direction_cosine = max(
                    -1.0,
                    min(1.0, (ax * bx + az * bz) / denominator),
                )

        predecessor_tracklet = self._tracklets_by_id[predecessor.tracklet_id]
        successor_tracklet = self._tracklets_by_id[successor.tracklet_id]

        return TransitionFeatures(
            predecessor_tracklet_id=predecessor.tracklet_id,
            successor_tracklet_id=successor.tracklet_id,
            gap_s=gap_s,
            endpoint_distance_m=endpoint_distance_m,
            reachability_limit_m=reachability_limit_m,
            reachability_margin_m=margin,
            raw_endpoint_speed_mps=raw_speed,
            required_speed_after_slack_mps=required_speed,
            predecessor_end_velocity=predecessor.end_velocity,
            successor_start_velocity=successor.start_velocity,
            forward_prediction_error_m=forward_error,
            backward_prediction_error_m=backward_error,
            velocity_direction_cosine=direction_cosine,
            predecessor_end_confidence=(
                predecessor_tracklet.observations[-1].confidence
            ),
            successor_start_confidence=(
                successor_tracklet.observations[0].confidence
            ),
            physically_reachable=margin >= -1e-9,
        )

    def iter_local_transition_features(
        self,
        *,
        horizon_s: float | None = None,
    ) -> Iterator[TransitionFeatures]:
        for predecessor in self.nodes:
            for successor in self.query_local_successors(
                predecessor.tracklet_id,
                horizon_s=horizon_s,
            ):
                yield self.transition_features(
                    predecessor.tracklet_id,
                    successor.tracklet_id,
                )

    def hard_coexistence_conflicts(self) -> tuple[HardConflict, ...]:
        result: list[HardConflict] = []

        for evidence in self._coobserved_pairs:
            if evidence.shared_frame_count == 1:
                hard = (
                    evidence.max_separation_m
                    > self.config.single_frame_hard_separation_m
                )
                threshold = self.config.single_frame_hard_separation_m
                rule = "single_exact_frame_separation"
            else:
                hard = (
                    evidence.median_separation_m
                    > self.config.overlap_hard_median_separation_m
                )
                threshold = self.config.overlap_hard_median_separation_m
                rule = "multi_exact_frame_median_separation"

            if not hard:
                continue

            result.append(
                HardConflict(
                    tracklet_a=evidence.tracklet_a,
                    tracklet_b=evidence.tracklet_b,
                    reason="simultaneous_spatial_separation",
                    evidence={
                        **evidence.to_dict(),
                        "decision_rule": rule,
                        "hard_threshold_m": threshold,
                    },
                )
            )

        return tuple(result)

    def close_coobserved_pairs(self) -> tuple[CoobservedPairEvidence, ...]:
        """Potential duplicate/fragment evidence for the next sub-hito."""

        hard_pairs = {
            (item.tracklet_a, item.tracklet_b)
            for item in self.hard_coexistence_conflicts()
        }
        return tuple(
            evidence
            for evidence in self._coobserved_pairs
            if (evidence.tracklet_a, evidence.tracklet_b) not in hard_pairs
            and evidence.median_separation_m
            <= self.config.close_overlap_median_separation_m
        )

    def _build_coobserved_pairs(
        self,
    ) -> tuple[CoobservedPairEvidence, ...]:
        samples: dict[
            tuple[str, str],
            list[tuple[int, float]],
        ] = defaultdict(list)

        for frame_index, members in self._frame_members.items():
            for left_index in range(len(members)):
                left_id, left_observation = members[left_index]
                for right_index in range(left_index + 1, len(members)):
                    right_id, right_observation = members[right_index]
                    pair = _canonical_pair(left_id, right_id)
                    separation = math.hypot(
                        left_observation.x - right_observation.x,
                        left_observation.z - right_observation.z,
                    )
                    samples[pair].append((frame_index, separation))

        result: list[CoobservedPairEvidence] = []
        for pair in sorted(samples):
            values = sorted(samples[pair], key=lambda item: item[0])
            frames = [item[0] for item in values]
            separations = [item[1] for item in values]
            result.append(
                CoobservedPairEvidence(
                    tracklet_a=pair[0],
                    tracklet_b=pair[1],
                    shared_frame_count=len(values),
                    first_shared_frame=frames[0],
                    last_shared_frame=frames[-1],
                    min_separation_m=min(separations),
                    median_separation_m=float(statistics.median(separations)),
                    max_separation_m=max(separations),
                    mean_separation_m=sum(separations) / len(separations),
                )
            )

        return tuple(result)


def build_sparse_tracklet_index(
    tracklets: Iterable[Tracklet],
    config: SparseTrackletIndexConfig | None = None,
) -> SparseTrackletIndex:
    return SparseTrackletIndex(tracklets, config)
