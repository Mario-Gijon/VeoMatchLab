from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import isfinite
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .models import PositionKind, TrackSample


_LARGE_COST = 1e9


@dataclass(slots=True, frozen=True)
class StitchingConfig:
    """
    Offline tracklet stitching parameters for Tracking V2.2.

    Stitching operates only after same-frame association and the forward tracker.
    It never creates observations. Its job is to connect temporally separated
    tracklets that are physically compatible with being the same person.

    The defaults are intentionally conservative:
    - only observed endpoints are used for deciding identity links;
    - tracklets with overlapping observed support are never linked;
    - impossible displacement/speed candidates are rejected before assignment;
    - a link must beat the explicit unmatched cost.
    """

    min_gap_seconds: float = 1.0 / 30.0
    max_gap_seconds: float = 4.0

    # Hard physical gates.
    base_position_gate_m: float = 2.0
    position_gate_growth_m_per_s: float = 4.0
    max_position_gate_m: float = 14.0
    max_endpoint_speed_m_s: float = 12.5
    max_velocity_difference_m_s: float = 9.0

    # Cost terms. Lower is better.
    position_cost_weight: float = 1.0
    backward_position_cost_weight: float = 0.65
    velocity_cost_weight: float = 0.18
    gap_cost_weight: float = 0.12

    # A predecessor may remain unmatched instead of accepting an expensive link.
    unmatched_cost: float = 5.0

    # If both tracklets have enough observed samples, require slightly more
    # conservative geometry because we are linking two established identities.
    established_observed_samples: int = 8
    established_gate_scale: float = 0.85

    def __post_init__(self) -> None:
        positive = (
            self.min_gap_seconds,
            self.max_gap_seconds,
            self.base_position_gate_m,
            self.position_gate_growth_m_per_s,
            self.max_position_gate_m,
            self.max_endpoint_speed_m_s,
            self.max_velocity_difference_m_s,
            self.position_cost_weight,
            self.backward_position_cost_weight,
            self.velocity_cost_weight,
            self.gap_cost_weight,
            self.unmatched_cost,
            self.established_gate_scale,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("StitchingConfig positive parameters must be > 0")
        if self.max_gap_seconds < self.min_gap_seconds:
            raise ValueError("max_gap_seconds must be >= min_gap_seconds")
        if self.max_position_gate_m < self.base_position_gate_m:
            raise ValueError(
                "max_position_gate_m must be >= base_position_gate_m"
            )
        if self.established_observed_samples < 1:
            raise ValueError("established_observed_samples must be >= 1")
        if not 0.0 < self.established_gate_scale <= 1.0:
            raise ValueError("established_gate_scale must be in (0, 1]")


@dataclass(slots=True, frozen=True)
class TrackletEndpoint:
    track_id: int
    first_observed: TrackSample
    last_observed: TrackSample
    observed_count: int

    @property
    def first_time(self) -> float:
        return self.first_observed.timestamp

    @property
    def last_time(self) -> float:
        return self.last_observed.timestamp


@dataclass(slots=True, frozen=True)
class StitchCandidate:
    predecessor_id: int
    successor_id: int
    gap_seconds: float

    forward_prediction_error_m: float
    backward_prediction_error_m: float
    endpoint_distance_m: float
    implied_speed_m_s: float
    velocity_difference_m_s: float
    position_gate_m: float

    cost: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class StitchLink:
    predecessor_id: int
    successor_id: int
    cost: float
    candidate: StitchCandidate

    def to_dict(self) -> dict[str, object]:
        return {
            "predecessor_id": self.predecessor_id,
            "successor_id": self.successor_id,
            "cost": self.cost,
            "candidate": self.candidate.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class StitchingReport:
    links: tuple[StitchLink, ...]
    chains: tuple[tuple[int, ...], ...]
    canonical_id_by_track: Mapping[int, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "links": [link.to_dict() for link in self.links],
            "chains": [list(chain) for chain in self.chains],
            "canonical_id_by_track": {
                str(track_id): canonical_id
                for track_id, canonical_id in sorted(
                    self.canonical_id_by_track.items()
                )
            },
        }


def stitch_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    config: StitchingConfig | None = None,
    allowed_track_ids: Sequence[int] | None = None,
) -> tuple[dict[int, tuple[TrackSample, ...]], StitchingReport]:
    """
    Globally connect compatible non-overlapping tracklets.

    Returns:
        merged_trajectories:
            One trajectory per canonical identity. The canonical ID is the
            earliest track ID in each temporal chain.
        report:
            Audit information with chosen links and ID remapping.

    `allowed_track_ids` can be used after quality filtering so tracks already
    classified REJECT are excluded from identity stitching.
    """
    cfg = config or StitchingConfig()
    allowed = (
        set(int(track_id) for track_id in allowed_track_ids)
        if allowed_track_ids is not None
        else None
    )

    normalized: dict[int, tuple[TrackSample, ...]] = {}
    endpoints: dict[int, TrackletEndpoint] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        if allowed is not None and track_id not in allowed:
            continue
        samples = _validated_samples(track_id, raw_samples)
        if not samples:
            continue
        endpoint = _endpoint(track_id, samples)
        if endpoint is None:
            continue
        normalized[track_id] = samples
        endpoints[track_id] = endpoint

    if not normalized:
        empty = StitchingReport(
            links=(),
            chains=(),
            canonical_id_by_track={},
        )
        return {}, empty

    candidates = _candidate_links(endpoints, cfg)
    links = _global_assignment(
        tuple(sorted(normalized)),
        candidates,
        cfg,
    )
    chains = _build_chains(tuple(sorted(normalized)), links, endpoints)

    canonical_id_by_track: dict[int, int] = {}
    merged: dict[int, tuple[TrackSample, ...]] = {}

    for chain in chains:
        canonical = chain[0]
        for track_id in chain:
            canonical_id_by_track[track_id] = canonical
        merged[canonical] = _merge_chain(
            chain,
            normalized,
            canonical_id=canonical,
        )

    report = StitchingReport(
        links=tuple(links),
        chains=tuple(chains),
        canonical_id_by_track=canonical_id_by_track,
    )
    return merged, report


def candidate_between(
    predecessor: Sequence[TrackSample],
    successor: Sequence[TrackSample],
    *,
    config: StitchingConfig | None = None,
) -> StitchCandidate | None:
    """Public helper useful for diagnostics/tests of one proposed identity link."""
    cfg = config or StitchingConfig()
    if not predecessor or not successor:
        return None

    pred_id = predecessor[0].track_id
    succ_id = successor[0].track_id
    pred = _endpoint(pred_id, _validated_samples(pred_id, predecessor))
    succ = _endpoint(succ_id, _validated_samples(succ_id, successor))

    if pred is None or succ is None:
        return None

    return _candidate(pred, succ, cfg)


def _candidate_links(
    endpoints: Mapping[int, TrackletEndpoint],
    config: StitchingConfig,
) -> dict[tuple[int, int], StitchCandidate]:
    result: dict[tuple[int, int], StitchCandidate] = {}

    for predecessor_id, predecessor in endpoints.items():
        for successor_id, successor in endpoints.items():
            if predecessor_id == successor_id:
                continue
            candidate = _candidate(predecessor, successor, config)
            if candidate is not None:
                result[(predecessor_id, successor_id)] = candidate

    return result


def _candidate(
    predecessor: TrackletEndpoint,
    successor: TrackletEndpoint,
    config: StitchingConfig,
) -> StitchCandidate | None:
    a = predecessor.last_observed
    b = successor.first_observed
    gap = float(b.timestamp - a.timestamp)

    # Same identity cannot have two independent observed tracklets at the same
    # instant, and very large gaps are not safe for geometry-only stitching.
    if gap < config.min_gap_seconds or gap > config.max_gap_seconds:
        return None

    a_position = np.array([a.x, a.z], dtype=np.float64)
    b_position = np.array([b.x, b.z], dtype=np.float64)
    a_velocity = np.array([a.vx, a.vz], dtype=np.float64)
    b_velocity = np.array([b.vx, b.vz], dtype=np.float64)

    if not all(
        np.isfinite(value).all()
        for value in (a_position, b_position, a_velocity, b_velocity)
    ):
        return None

    endpoint_delta = b_position - a_position
    endpoint_distance = float(np.linalg.norm(endpoint_delta))
    implied_speed = endpoint_distance / gap

    if implied_speed > config.max_endpoint_speed_m_s:
        return None

    forward_prediction = a_position + a_velocity * gap
    backward_prediction = b_position - b_velocity * gap

    forward_error = float(np.linalg.norm(b_position - forward_prediction))
    backward_error = float(np.linalg.norm(a_position - backward_prediction))
    velocity_difference = float(np.linalg.norm(b_velocity - a_velocity))

    gate = min(
        config.max_position_gate_m,
        config.base_position_gate_m
        + config.position_gate_growth_m_per_s * gap,
    )

    if (
        predecessor.observed_count >= config.established_observed_samples
        and successor.observed_count >= config.established_observed_samples
    ):
        gate *= config.established_gate_scale

    if forward_error > gate:
        return None
    if backward_error > gate:
        return None
    if velocity_difference > config.max_velocity_difference_m_s:
        return None

    # Normalize position terms by the gate so costs remain comparable across
    # different gap lengths.
    cost = (
        config.position_cost_weight * (forward_error / gate)
        + config.backward_position_cost_weight * (backward_error / gate)
        + config.velocity_cost_weight
        * (velocity_difference / config.max_velocity_difference_m_s)
        + config.gap_cost_weight * (gap / config.max_gap_seconds)
    )

    if not isfinite(cost) or cost >= config.unmatched_cost:
        return None

    return StitchCandidate(
        predecessor_id=predecessor.track_id,
        successor_id=successor.track_id,
        gap_seconds=gap,
        forward_prediction_error_m=forward_error,
        backward_prediction_error_m=backward_error,
        endpoint_distance_m=endpoint_distance,
        implied_speed_m_s=implied_speed,
        velocity_difference_m_s=velocity_difference,
        position_gate_m=gate,
        cost=float(cost),
    )


def _global_assignment(
    track_ids: tuple[int, ...],
    candidates: Mapping[tuple[int, int], StitchCandidate],
    config: StitchingConfig,
) -> list[StitchLink]:
    """
    One global predecessor->successor assignment with explicit unmatched slots.

    Rows are possible predecessors.
    First N columns are real successor IDs.
    Last N columns are row-specific unmatched choices.
    """
    count = len(track_ids)
    if count == 0 or not candidates:
        return []

    index_by_id = {track_id: index for index, track_id in enumerate(track_ids)}
    costs = np.full((count, count * 2), _LARGE_COST, dtype=np.float64)

    for (predecessor_id, successor_id), candidate in candidates.items():
        row = index_by_id.get(predecessor_id)
        col = index_by_id.get(successor_id)
        if row is None or col is None:
            continue
        costs[row, col] = candidate.cost

    # Unique unmatched column for every predecessor.
    for row in range(count):
        costs[row, count + row] = config.unmatched_cost

    row_indices, column_indices = linear_sum_assignment(costs)

    links: list[StitchLink] = []
    for row, col in zip(row_indices.tolist(), column_indices.tolist()):
        if col >= count:
            continue
        predecessor_id = track_ids[row]
        successor_id = track_ids[col]
        candidate = candidates.get((predecessor_id, successor_id))
        if candidate is None:
            continue
        if candidate.cost >= config.unmatched_cost:
            continue
        links.append(
            StitchLink(
                predecessor_id=predecessor_id,
                successor_id=successor_id,
                cost=candidate.cost,
                candidate=candidate,
            )
        )

    links.sort(
        key=lambda link: (
            link.candidate.gap_seconds,
            link.cost,
            link.predecessor_id,
            link.successor_id,
        )
    )
    return links


def _build_chains(
    track_ids: tuple[int, ...],
    links: Sequence[StitchLink],
    endpoints: Mapping[int, TrackletEndpoint],
) -> list[tuple[int, ...]]:
    successor_by_id = {
        link.predecessor_id: link.successor_id
        for link in links
    }
    predecessor_by_id = {
        link.successor_id: link.predecessor_id
        for link in links
    }

    # Time-directed candidates should already make cycles impossible. Keep an
    # explicit guard because a stitching report must never silently corrupt IDs.
    for start in track_ids:
        seen: set[int] = set()
        current = start
        while current in successor_by_id:
            if current in seen:
                raise RuntimeError("Cycle detected in track stitching links")
            seen.add(current)
            current = successor_by_id[current]

    starts = [
        track_id
        for track_id in track_ids
        if track_id not in predecessor_by_id
    ]
    starts.sort(
        key=lambda track_id: (
            endpoints[track_id].first_time,
            track_id,
        )
    )

    chains: list[tuple[int, ...]] = []
    visited: set[int] = set()

    for start in starts:
        chain: list[int] = []
        current = start

        while current not in visited:
            chain.append(current)
            visited.add(current)
            if current not in successor_by_id:
                break
            current = successor_by_id[current]

        chains.append(tuple(chain))

    # Defensive fallback for any isolated item not reached above.
    for track_id in track_ids:
        if track_id not in visited:
            chains.append((track_id,))

    return chains


def _merge_chain(
    chain: Sequence[int],
    trajectories: Mapping[int, tuple[TrackSample, ...]],
    *,
    canonical_id: int,
) -> tuple[TrackSample, ...]:
    """
    Merge samples by frame, preferring stronger evidence when temporal tails
    overlap (OBSERVED > INTERPOLATED > PREDICTED).
    """
    by_frame: dict[int, TrackSample] = {}

    for track_id in chain:
        for sample in trajectories[track_id]:
            rewritten = replace(sample, track_id=canonical_id)
            existing = by_frame.get(rewritten.frame_index)

            if existing is None:
                by_frame[rewritten.frame_index] = rewritten
                continue

            if _sample_priority(rewritten) > _sample_priority(existing):
                by_frame[rewritten.frame_index] = rewritten

    return tuple(
        sorted(
            by_frame.values(),
            key=lambda item: (item.timestamp, item.frame_index),
        )
    )


def _sample_priority(sample: TrackSample) -> int:
    return {
        PositionKind.OBSERVED: 3,
        PositionKind.INTERPOLATED: 2,
        PositionKind.PREDICTED: 1,
    }[sample.position_kind]


def _endpoint(
    track_id: int,
    samples: Sequence[TrackSample],
) -> TrackletEndpoint | None:
    observed = [
        sample
        for sample in samples
        if sample.position_kind is PositionKind.OBSERVED
    ]
    if not observed:
        return None

    return TrackletEndpoint(
        track_id=track_id,
        first_observed=observed[0],
        last_observed=observed[-1],
        observed_count=len(observed),
    )


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    if not samples:
        return ()

    ordered = tuple(
        sorted(samples, key=lambda item: (item.timestamp, item.frame_index))
    )

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match sample "
                f"track_id {sample.track_id}"
            )
        if not isfinite(sample.timestamp):
            raise ValueError("Track sample timestamp must be finite")
        if not all(
            isfinite(value)
            for value in (sample.x, sample.z, sample.vx, sample.vz)
        ):
            raise ValueError(
                f"Track {track_id} contains non-finite position/velocity"
            )

    return ordered
