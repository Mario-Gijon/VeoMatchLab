"""Hito 6A: candidate graph for persistent football actors.

Tracking V1 deliberately publishes conservative *physical identities*.  They
are clean fragments, not yet a promise that one person owns one ID throughout
an entire clip.  Hito 6A keeps those identities immutable and asks a different
question:

    Which canonical physical identities could plausibly belong to the same
    long-lived actor?

No merge is performed here.  The output is diagnostics/evidence for Hito 6B.

Generalisation principles
-------------------------
* The number of actors is inferred from stable simultaneous occupancy; 23 is
  never hard-coded.
* Existing V1 hard cannot-links remain hard.  Appearance cannot override them.
* Missing appearance is unknown, not negative evidence.
* Appearance calibration is derived from the clip when enough known-positive
  and known-negative tracklet pairs exist.
* Sequential links are scored from geometry, motion, time and appearance, but
  Hito 6A makes no identity decision and has no link threshold.
* Temporally overlapping but physically compatible fragments are preserved as
  a separate diagnostic relation instead of being forced into a path-cover
  assumption prematurely.
"""

from __future__ import annotations

import collections
import itertools
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


CANDIDATE_VERSION = "persistent-actor-candidates-v1.1"
IDENTITY_PUBLISH_VERSION = "identity-publish-v1"
FINAL_PUBLISH_VERSION = "final-publish-v1"
PHYSICAL_GRAPH_VERSION = "physical-compatibility-v1"


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("distribution values must be finite and one-dimensional")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise ValueError("pair requires two distinct ids")
    return (first, second) if first < second else (second, first)


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    if len(values) != len(weights) or not values:
        raise ValueError("weighted median requires equally-sized non-empty inputs")
    ordered = sorted(zip(values, weights), key=lambda item: item[0])
    total = sum(max(0.0, float(weight)) for _, weight in ordered)
    if total <= 1e-12:
        return float(np.median(np.asarray(values, dtype=np.float64)))
    threshold = total / 2.0
    cumulative = 0.0
    for value, weight in ordered:
        cumulative += max(0.0, float(weight))
        if cumulative >= threshold:
            return float(value)
    return float(ordered[-1][0])


@dataclass(frozen=True, slots=True)
class PersistentActorCandidateConfig:
    # Hard physics.  Same envelope already used by Tracking V1 3A.
    max_transition_speed_mps: float = 12.0
    transition_position_slack_m: float = 1.5

    # Soft continuation priors.  These are scales, never hard identity rules.
    comfortable_speed_mps: float = 5.5
    gap_decay_s: float = 2.5
    motion_error_base_m: float = 1.5
    motion_error_growth_mps: float = 1.5
    minimum_velocity_for_direction_mps: float = 0.75

    # Population estimator: choose a high occupancy only if it is genuinely
    # stable, not merely the single maximum frame.
    stable_population_support_fraction: float = 0.25

    # Empirical ReID calibration requirements.
    minimum_positive_appearance_pairs: int = 5
    minimum_negative_appearance_pairs: int = 20
    minimum_appearance_margin: float = 0.003

    # Candidate score composition.  Missing terms are renormalised away.
    appearance_weight: float = 0.45
    motion_weight: float = 0.20
    speed_weight: float = 0.20
    gap_weight: float = 0.10
    endpoint_confidence_weight: float = 0.05

    def __post_init__(self) -> None:
        positive = (
            "max_transition_speed_mps",
            "transition_position_slack_m",
            "comfortable_speed_mps",
            "gap_decay_s",
            "motion_error_base_m",
            "motion_error_growth_mps",
            "minimum_velocity_for_direction_mps",
        )
        for name in positive:
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if not 0 < self.stable_population_support_fraction <= 1:
            raise ValueError("stable_population_support_fraction must be in (0,1]")
        if self.minimum_positive_appearance_pairs < 1:
            raise ValueError("minimum_positive_appearance_pairs must be >= 1")
        if self.minimum_negative_appearance_pairs < 1:
            raise ValueError("minimum_negative_appearance_pairs must be >= 1")
        if self.minimum_appearance_margin <= 0:
            raise ValueError("minimum_appearance_margin must be > 0")
        weights = (
            self.appearance_weight,
            self.motion_weight,
            self.speed_weight,
            self.gap_weight,
            self.endpoint_confidence_weight,
        )
        if any(value < 0 for value in weights) or sum(weights) <= 0:
            raise ValueError("candidate score weights must be non-negative and non-zero")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True, slots=True)
class IdentityNode:
    identity_id: str
    tracklet_ids: tuple[str, ...]
    points: tuple[Mapping[str, Any], ...]
    segment_ids: tuple[str, ...]

    @property
    def first(self) -> Mapping[str, Any]:
        return self.points[0]

    @property
    def last(self) -> Mapping[str, Any]:
        return self.points[-1]

    @property
    def start_frame(self) -> int:
        return int(self.first["frame_index"])

    @property
    def end_frame(self) -> int:
        return int(self.last["frame_index"])

    @property
    def start_time_s(self) -> float:
        return float(self.first["timestamp_s"])

    @property
    def end_time_s(self) -> float:
        return float(self.last["timestamp_s"])

    def to_dict(self) -> dict[str, Any]:
        origin_counts = collections.Counter(str(point.get("origin", "unknown")) for point in self.points)
        return {
            "schema_version": 1,
            "candidate_version": CANDIDATE_VERSION,
            "identity_id": self.identity_id,
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": len(self.tracklet_ids),
            "point_count": len(self.points),
            "segment_count": len(self.segment_ids),
            "segment_ids": list(self.segment_ids),
            "start_frame_index": self.start_frame,
            "end_frame_index": self.end_frame,
            "start_timestamp_s": self.start_time_s,
            "end_timestamp_s": self.end_time_s,
            "duration_s": self.end_time_s - self.start_time_s,
            "first_position": _point_summary(self.first),
            "last_position": _point_summary(self.last),
            "origin_counts": dict(sorted(origin_counts.items())),
        }


@dataclass(frozen=True, slots=True)
class AppearanceCalibration:
    available: bool
    source: str
    negative_p95: float | None
    positive_p10: float | None
    positive_pair_count: int
    negative_pair_count: int
    positive_distribution: Mapping[str, Any]
    negative_distribution: Mapping[str, Any]

    def score(self, cosine: float | None) -> float | None:
        if not self.available or cosine is None:
            return None
        assert self.negative_p95 is not None
        assert self.positive_p10 is not None
        denominator = self.positive_p10 - self.negative_p95
        if denominator <= 0:
            return None
        return _clamp01((cosine - self.negative_p95) / denominator)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate_version": CANDIDATE_VERSION,
            "available": self.available,
            "source": self.source,
            "negative_p95": self.negative_p95,
            "positive_p10": self.positive_p10,
            "margin": (
                None
                if self.negative_p95 is None or self.positive_p10 is None
                else self.positive_p10 - self.negative_p95
            ),
            "positive_pair_count": self.positive_pair_count,
            "negative_pair_count": self.negative_pair_count,
            "positive_distribution": dict(self.positive_distribution),
            "negative_distribution": dict(self.negative_distribution),
        }


def _point_summary(point: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "frame_index": int(point["frame_index"]),
        "timestamp_s": float(point["timestamp_s"]),
        "x": float(point["smoothed_x"]),
        "z": float(point["smoothed_z"]),
        "vx_mps": _finite(point.get("vx_mps")),
        "vz_mps": _finite(point.get("vz_mps")),
        "measurement_confidence": _finite(point.get("measurement_confidence")),
        "position_std_m": _finite(point.get("position_std_m")),
        "segment_id": str(point["segment_id"]),
        "origin": str(point.get("origin", "unknown")),
    }


def _identity_assignment(identity_rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, tuple[str, ...]], dict[str, str]]:
    identities: dict[str, tuple[str, ...]] = {}
    assignment: dict[str, str] = {}
    for row in identity_rows:
        identity_id = str(row["identity_id"])
        if identity_id in identities:
            raise ValueError(f"duplicate identity_id: {identity_id}")
        if row.get("canonical") is not True:
            raise ValueError(f"{identity_id} is not canonical")
        members = tuple(sorted(str(value) for value in row["tracklet_ids"]))
        if not members or len(set(members)) != len(members):
            raise ValueError(f"{identity_id} has invalid tracklet membership")
        for tracklet_id in members:
            if tracklet_id in assignment:
                raise ValueError(f"tracklet assigned twice: {tracklet_id}")
            assignment[tracklet_id] = identity_id
        identities[identity_id] = members
    if not identities:
        raise ValueError("no canonical identities supplied")
    return identities, assignment


def build_identity_nodes(
    identity_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
) -> tuple[IdentityNode, ...]:
    memberships, _ = _identity_assignment(identity_rows)
    trajectories: dict[str, Mapping[str, Any]] = {}
    for row in trajectory_rows:
        identity_id = str(row["identity_id"])
        if identity_id in trajectories:
            raise ValueError(f"duplicate final trajectory: {identity_id}")
        if row.get("canonical") is not True:
            raise ValueError(f"{identity_id} final trajectory is not canonical")
        points = row.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(f"{identity_id} final trajectory has no points")
        trajectories[identity_id] = row
    if set(trajectories) != set(memberships):
        missing = sorted(set(memberships) - set(trajectories))
        extra = sorted(set(trajectories) - set(memberships))
        raise ValueError(f"identity/final population mismatch missing={missing} extra={extra}")

    result: list[IdentityNode] = []
    for identity_id in sorted(memberships):
        row = trajectories[identity_id]
        tracklet_ids = tuple(sorted(str(value) for value in row.get("tracklet_ids", ())))
        if tracklet_ids != memberships[identity_id]:
            raise ValueError(f"{identity_id} final membership differs from canonical identities")
        points = tuple(sorted((dict(point) for point in row["points"]), key=lambda point: int(point["frame_index"])))
        previous_frame = -1
        previous_time = -1.0
        segments: list[str] = []
        for point in points:
            frame = int(point["frame_index"])
            timestamp = float(point["timestamp_s"])
            if frame <= previous_frame or timestamp <= previous_time:
                raise ValueError(f"{identity_id} final points are not strictly ordered")
            for field_name in ("smoothed_x", "smoothed_z"):
                if _finite(point.get(field_name)) is None:
                    raise ValueError(f"{identity_id} has invalid {field_name}")
            segment_id = str(point["segment_id"])
            if not segments or segments[-1] != segment_id:
                segments.append(segment_id)
            previous_frame = frame
            previous_time = timestamp
        result.append(
            IdentityNode(
                identity_id=identity_id,
                tracklet_ids=tracklet_ids,
                points=points,
                segment_ids=tuple(segments),
            )
        )
    return tuple(result)



def _maximum_hard_conflict_clique(
    active_identity_ids: Sequence[str],
    hard_conflicts: Mapping[str, set[str]],
) -> tuple[str, ...]:
    """Return one maximum clique in the identity-level hard-conflict graph.

    A clique is a set of identities that are pairwise known to be distinct
    physical people.  Active populations in our football clips are small
    (typically around the low twenties), so an exact branch-and-bound search is
    both deterministic and cheap.  This is a lower bound on the true number of
    distinct people, never an upper bound.
    """

    vertices = tuple(sorted(set(active_identity_ids)))
    if not vertices:
        return ()

    adjacency = {
        vertex: set(hard_conflicts.get(vertex, set())) & set(vertices)
        for vertex in vertices
    }
    best: tuple[str, ...] = ()

    def expand(chosen: tuple[str, ...], candidates: tuple[str, ...]) -> None:
        nonlocal best
        if len(chosen) + len(candidates) <= len(best):
            return
        if not candidates:
            if len(chosen) > len(best):
                best = chosen
            return

        remaining = list(candidates)
        while remaining:
            if len(chosen) + len(remaining) <= len(best):
                return
            # Prefer highly-connected vertices first for tighter pruning, while
            # keeping a deterministic lexical tie-break.
            vertex = max(
                remaining,
                key=lambda item: (
                    len(adjacency[item] & set(remaining)),
                    item,
                ),
            )
            remaining.remove(vertex)
            next_candidates = tuple(
                item
                for item in remaining
                if item in adjacency[vertex]
            )
            expand(chosen + (vertex,), next_candidates)

        if len(chosen) > len(best):
            best = chosen

    expand((), vertices)
    return tuple(sorted(best))


def _identity_hard_conflicts(
    identity_rows: Sequence[Mapping[str, Any]],
    physical_candidate_rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    _, assignment = _identity_assignment(identity_rows)
    conflicts: dict[str, set[str]] = collections.defaultdict(set)
    for row in physical_candidate_rows:
        if row.get("status") != "cannot_link":
            continue
        identity_a = assignment.get(str(row["tracklet_a"]))
        identity_b = assignment.get(str(row["tracklet_b"]))
        if identity_a is None or identity_b is None or identity_a == identity_b:
            continue
        conflicts[identity_a].add(identity_b)
        conflicts[identity_b].add(identity_a)
    return {key: set(value) for key, value in conflicts.items()}


def estimate_population(
    trajectory_rows: Sequence[Mapping[str, Any]],
    config: PersistentActorCandidateConfig | None = None,
    *,
    identity_rows: Sequence[Mapping[str, Any]] | None = None,
    physical_candidate_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    selected = config or PersistentActorCandidateConfig()

    active_by_frame: dict[int, set[str]] = collections.defaultdict(set)
    for row in trajectory_rows:
        identity_id = str(row["identity_id"])
        seen_frames: set[int] = set()
        for point in row.get("points", []):
            frame = int(point["frame_index"])
            if frame in seen_frames:
                raise ValueError(f"{identity_id} repeats frame {frame}")
            seen_frames.add(frame)
            active_by_frame[frame].add(identity_id)
    if not active_by_frame:
        raise ValueError("cannot estimate population without final points")

    ordered_frames = sorted(active_by_frame)
    raw_counts = [len(active_by_frame[frame]) for frame in ordered_frames]
    raw_histogram = collections.Counter(raw_counts)
    highest_frequency = max(raw_histogram.values())
    # Missing observations are more common than stable duplicate fragments, so
    # on a mode tie keep the larger occupancy.  Crucially, we no longer push the
    # estimate upward merely because a higher raw occupancy occurs in 25% of
    # frames: duplicate identities can create exactly that pattern.
    raw_modal_population = max(
        count
        for count, frequency in raw_histogram.items()
        if frequency == highest_frequency
    )

    hard_distinct_available = (
        identity_rows is not None
        and physical_candidate_rows is not None
    )
    distinct_counts: list[int] = []
    distinct_cliques: dict[int, tuple[str, ...]] = {}
    if hard_distinct_available:
        hard_conflicts = _identity_hard_conflicts(
            identity_rows,
            physical_candidate_rows,
        )
        cache: dict[tuple[str, ...], tuple[str, ...]] = {}
        for frame in ordered_frames:
            active = tuple(sorted(active_by_frame[frame]))
            clique = cache.get(active)
            if clique is None:
                clique = _maximum_hard_conflict_clique(active, hard_conflicts)
                cache[active] = clique
            distinct_cliques[frame] = clique
            distinct_counts.append(len(clique))

    distinct_histogram = collections.Counter(distinct_counts)
    distinct_modal_population = None
    distinct_stable_high_population = None
    distinct_max_population = None
    distinct_exact_support = None
    distinct_at_least_support = None
    if distinct_counts:
        highest_distinct_frequency = max(distinct_histogram.values())
        distinct_modal_population = max(
            count
            for count, frequency in distinct_histogram.items()
            if frequency == highest_distinct_frequency
        )
        frame_count = len(distinct_counts)
        stable_candidates = [
            count
            for count in sorted(distinct_histogram)
            if sum(value >= count for value in distinct_counts) / frame_count
            >= selected.stable_population_support_fraction
        ]
        distinct_stable_high_population = (
            max(stable_candidates)
            if stable_candidates
            else distinct_modal_population
        )
        distinct_max_population = max(distinct_counts)
    else:
        frame_count = len(raw_counts)

    # Raw mode is the main occupancy hypothesis.  The maximum-clique signal is
    # a hard-physics lower bound that may raise the estimate when raw mode is
    # depressed by misses, but it can never inflate it because of duplicate
    # fragments.  We intentionally do not use a raw "stable high" quantile.
    estimate = raw_modal_population
    if distinct_stable_high_population is not None:
        estimate = max(estimate, distinct_stable_high_population)

    raw_exact_support = sum(value == estimate for value in raw_counts) / frame_count
    raw_at_least_support = sum(value >= estimate for value in raw_counts) / frame_count
    if distinct_counts:
        distinct_exact_support = (
            sum(value == estimate for value in distinct_counts) / frame_count
        )
        distinct_at_least_support = (
            sum(value >= estimate for value in distinct_counts) / frame_count
        )

    if (
        raw_modal_population == estimate
        and raw_exact_support >= 0.50
        and (
            distinct_stable_high_population is None
            or distinct_stable_high_population == estimate
        )
    ):
        confidence = "high"
    elif (
        distinct_stable_high_population == estimate
        and raw_modal_population == estimate
        and raw_exact_support >= 0.25
    ):
        confidence = "high"
    elif raw_modal_population == estimate and raw_exact_support >= 0.20:
        confidence = "medium"
    elif distinct_at_least_support is not None and distinct_at_least_support >= selected.stable_population_support_fraction:
        confidence = "medium"
    else:
        confidence = "low"

    raw_distribution = _distribution([float(value) for value in raw_counts])
    distinct_distribution = _distribution([float(value) for value in distinct_counts])
    exemplar_frame = None
    exemplar_clique: list[str] = []
    if distinct_cliques:
        candidates = [
            (frame, clique)
            for frame, clique in distinct_cliques.items()
            if len(clique) == (distinct_max_population or 0)
        ]
        if candidates:
            exemplar_frame, clique = min(candidates, key=lambda item: item[0])
            exemplar_clique = list(clique)

    return {
        "schema_version": 1,
        "candidate_version": CANDIDATE_VERSION,
        "estimator": "modal-plus-hard-distinct-lower-bound-v2",
        "estimated_population": int(estimate),
        "confidence": confidence,
        "frame_count": frame_count,
        "modal_population": int(raw_modal_population),
        "raw_modal_population": int(raw_modal_population),
        "raw_max_population": int(max(raw_counts)),
        "raw_min_population": int(min(raw_counts)),
        "raw_exact_support_fraction": float(raw_exact_support),
        "raw_at_least_support_fraction": float(raw_at_least_support),
        # Compatibility aliases used by the existing runner/UI diagnostics.
        "max_population": int(max(raw_counts)),
        "min_population": int(min(raw_counts)),
        "exact_support_fraction": float(raw_exact_support),
        "at_least_support_fraction": float(raw_at_least_support),
        "hard_distinct_available": bool(distinct_counts),
        "distinct_modal_population": (
            int(distinct_modal_population)
            if distinct_modal_population is not None
            else None
        ),
        "distinct_stable_high_population": (
            int(distinct_stable_high_population)
            if distinct_stable_high_population is not None
            else None
        ),
        "distinct_max_population": (
            int(distinct_max_population)
            if distinct_max_population is not None
            else None
        ),
        "distinct_exact_support_fraction": distinct_exact_support,
        "distinct_at_least_support_fraction": distinct_at_least_support,
        "stable_support_fraction_threshold": selected.stable_population_support_fraction,
        "occupancy_distribution": raw_distribution,
        "raw_occupancy_distribution": raw_distribution,
        "distinct_occupancy_distribution": distinct_distribution,
        "histogram": {str(key): raw_histogram[key] for key in sorted(raw_histogram)},
        "raw_histogram": {str(key): raw_histogram[key] for key in sorted(raw_histogram)},
        "distinct_histogram": {
            str(key): distinct_histogram[key]
            for key in sorted(distinct_histogram)
        },
        "distinct_exemplar_frame_index": exemplar_frame,
        "distinct_exemplar_identity_ids": exemplar_clique,
        "semantics": {
            "raw_occupancy": "upper observation count; may contain duplicate fragments",
            "hard_distinct_occupancy": "maximum pairwise hard-cannot-link clique; lower bound on distinct people",
            "estimated_population": "raw modal hypothesis, raised only by a stable hard-distinct lower bound",
        },
    }

def _appearance_rows_by_pair(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        pair = _canonical_pair(str(row["tracklet_a"]), str(row["tracklet_b"]))
        if pair in result:
            raise ValueError(f"duplicate appearance pair: {pair}")
        result[pair] = row
    return result


def _physical_rows_by_pair(
    rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        pair = _canonical_pair(str(row["tracklet_a"]), str(row["tracklet_b"]))
        if pair in result:
            raise ValueError(f"duplicate physical candidate pair: {pair}")
        status = row.get("status")
        if status not in {"candidate", "cannot_link"}:
            raise ValueError(f"invalid physical candidate status: {status!r}")
        result[pair] = row
    return result


def build_appearance_calibration(
    *,
    assignment: Mapping[str, str],
    appearance_rows: Sequence[Mapping[str, Any]],
    physical_rows: Sequence[Mapping[str, Any]],
    config: PersistentActorCandidateConfig,
    trusted_calibration: Mapping[str, Any] | None = None,
) -> AppearanceCalibration:
    """Build the ReID support mapping used by 6A.

    Prefer the already-validated Tracking V1 soft-link calibration when it is
    supplied.  That calibration comes from split-positive tracklet halves and
    concurrent physical negatives, which is much cleaner than treating every
    pair of tracklets inside a final identity as a positive example.

    We still compute the old empirical distributions below as diagnostics and
    as a fallback for synthetic/unit-test inputs that do not have a V1
    calibration artifact.
    """

    physical = _physical_rows_by_pair(physical_rows)
    positives: list[float] = []
    negatives: list[float] = []

    for row in appearance_rows:
        tracklet_a = str(row["tracklet_a"])
        tracklet_b = str(row["tracklet_b"])
        cosine = _finite(row.get("aggregate_cosine"))
        if cosine is None:
            continue
        identity_a = assignment.get(tracklet_a)
        identity_b = assignment.get(tracklet_b)
        if identity_a is None or identity_b is None:
            continue
        if identity_a == identity_b:
            positives.append(cosine)
            continue
        physical_row = physical.get(_canonical_pair(tracklet_a, tracklet_b))
        if physical_row is not None and physical_row.get("status") == "cannot_link":
            negatives.append(cosine)

    positive_distribution = _distribution(positives)
    negative_distribution = _distribution(negatives)

    if trusted_calibration is not None:
        appearance_block: Mapping[str, Any] = trusted_calibration
        if isinstance(trusted_calibration.get("appearance"), Mapping):
            appearance_block = trusted_calibration["appearance"]  # type: ignore[index]
        negative_anchor = appearance_block.get("negative_anchor")
        positive_anchor = appearance_block.get("positive_anchor")
        if isinstance(negative_anchor, Mapping) and isinstance(positive_anchor, Mapping):
            negative_p95 = _finite(negative_anchor.get("value"))
            positive_p10 = _finite(positive_anchor.get("value"))
            if (
                negative_p95 is not None
                and positive_p10 is not None
                and positive_p10 - negative_p95 >= config.minimum_appearance_margin
            ):
                return AppearanceCalibration(
                    available=True,
                    source="tracking-v1-soft-link-calibration",
                    negative_p95=negative_p95,
                    positive_p10=positive_p10,
                    positive_pair_count=len(positives),
                    negative_pair_count=len(negatives),
                    positive_distribution=positive_distribution,
                    negative_distribution=negative_distribution,
                )

    enough = (
        len(positives) >= config.minimum_positive_appearance_pairs
        and len(negatives) >= config.minimum_negative_appearance_pairs
    )
    positive_p10 = positive_distribution["p10"] if enough else None
    negative_p95 = negative_distribution["p95"] if enough else None
    margin_ok = (
        positive_p10 is not None
        and negative_p95 is not None
        and positive_p10 - negative_p95 >= config.minimum_appearance_margin
    )
    return AppearanceCalibration(
        available=bool(enough and margin_ok),
        source=(
            "fallback-empirical-final-identity-calibration"
            if enough and margin_ok
            else "unavailable-insufficient-separation"
        ),
        negative_p95=(float(negative_p95) if negative_p95 is not None else None),
        positive_p10=(float(positive_p10) if positive_p10 is not None else None),
        positive_pair_count=len(positives),
        negative_pair_count=len(negatives),
        positive_distribution=positive_distribution,
        negative_distribution=negative_distribution,
    )

def _identity_appearance(
    members_a: Sequence[str],
    members_b: Sequence[str],
    appearance_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
    calibration: AppearanceCalibration,
) -> dict[str, Any]:
    rows: list[Mapping[str, Any]] = []
    for tracklet_a in members_a:
        for tracklet_b in members_b:
            row = appearance_by_pair.get(_canonical_pair(tracklet_a, tracklet_b))
            if row is not None:
                rows.append(row)
    if not rows:
        return {
            "available": False,
            "pair_count": 0,
            "aggregate_cosine_weighted_mean": None,
            "aggregate_cosine_weighted_median": None,
            "aggregate_cosine_max": None,
            "reliability_weighted_mean": None,
            "calibrated_score": None,
        }

    cosines: list[float] = []
    weights: list[float] = []
    reliabilities: list[float] = []
    for row in rows:
        cosine = _finite(row.get("aggregate_cosine"))
        if cosine is None:
            continue
        reliability = _clamp01(_finite(row.get("reliability")) or 0.0)
        support_quality = _clamp01(_finite(row.get("support_quality")) or 0.0)
        weight = max(1e-6, reliability * support_quality)
        cosines.append(cosine)
        weights.append(weight)
        reliabilities.append(reliability)
    if not cosines:
        return {
            "available": False,
            "pair_count": 0,
            "aggregate_cosine_weighted_mean": None,
            "aggregate_cosine_weighted_median": None,
            "aggregate_cosine_max": None,
            "reliability_weighted_mean": None,
            "calibrated_score": None,
        }

    weight_sum = sum(weights)
    weighted_mean = sum(value * weight for value, weight in zip(cosines, weights)) / weight_sum
    reliability_mean = sum(value * weight for value, weight in zip(reliabilities, weights)) / weight_sum
    return {
        "available": True,
        "pair_count": len(cosines),
        "aggregate_cosine_weighted_mean": float(weighted_mean),
        "aggregate_cosine_weighted_median": _weighted_median(cosines, weights),
        "aggregate_cosine_max": float(max(cosines)),
        "reliability_weighted_mean": float(reliability_mean),
        "calibrated_score": calibration.score(float(weighted_mean)),
    }


def _hard_constraints_between(
    members_a: Sequence[str],
    members_b: Sequence[str],
    physical_by_pair: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    reasons: collections.Counter[str] = collections.Counter()
    pairs: list[list[str]] = []
    missing_pairs = 0
    for tracklet_a in members_a:
        for tracklet_b in members_b:
            row = physical_by_pair.get(_canonical_pair(tracklet_a, tracklet_b))
            if row is None:
                missing_pairs += 1
                continue
            if row.get("status") == "cannot_link":
                reasons[str(row.get("reason", "unknown"))] += 1
                pairs.append([tracklet_a, tracklet_b])
    return {
        "hard_constraint_count": len(pairs),
        "hard_constraint_reasons": dict(sorted(reasons.items())),
        "hard_constraint_tracklet_pairs": pairs,
        "missing_physical_pair_count": missing_pairs,
    }


def _point_xy(point: Mapping[str, Any]) -> np.ndarray:
    return np.asarray([float(point["smoothed_x"]), float(point["smoothed_z"])], dtype=np.float64)


def _velocity(point: Mapping[str, Any]) -> np.ndarray | None:
    vx = _finite(point.get("vx_mps"))
    vz = _finite(point.get("vz_mps"))
    if vx is None or vz is None:
        return None
    return np.asarray([vx, vz], dtype=np.float64)


def _endpoint_confidence(point: Mapping[str, Any]) -> float | None:
    value = _finite(point.get("measurement_confidence"))
    return None if value is None else _clamp01(value)


def _motion_components(
    predecessor: IdentityNode,
    successor: IdentityNode,
    config: PersistentActorCandidateConfig,
) -> tuple[dict[str, Any], dict[str, float | None]]:
    left = predecessor.last
    right = successor.first
    gap_s = float(right["timestamp_s"]) - float(left["timestamp_s"])
    if gap_s <= 0:
        raise ValueError("sequential motion requires a positive gap")

    left_xy = _point_xy(left)
    right_xy = _point_xy(right)
    distance = float(np.linalg.norm(right_xy - left_xy))
    raw_transition_speed = distance / gap_s
    hard_required_speed = (
        max(0.0, distance - config.transition_position_slack_m) / gap_s
    )
    # Soft ranking uses the actual endpoint displacement.  Positional slack is
    # reserved exclusively for the conservative hard impossibility envelope.
    speed_score = math.exp(
        -0.5 * (raw_transition_speed / config.comfortable_speed_mps) ** 2
    )
    gap_score = math.exp(-gap_s / config.gap_decay_s)

    left_velocity = _velocity(left)
    right_velocity = _velocity(right)
    forward_error = None
    backward_error = None
    errors: list[float] = []
    if left_velocity is not None:
        predicted = left_xy + left_velocity * gap_s
        forward_error = float(np.linalg.norm(predicted - right_xy))
        errors.append(forward_error)
    if right_velocity is not None:
        predicted = right_xy - right_velocity * gap_s
        backward_error = float(np.linalg.norm(predicted - left_xy))
        errors.append(backward_error)

    motion_score = None
    if errors:
        error = float(np.mean(errors))
        sigma = config.motion_error_base_m + config.motion_error_growth_mps * gap_s
        motion_score = math.exp(-0.5 * (error / sigma) ** 2)

    direction_cosine = None
    direction_score = None
    if left_velocity is not None and right_velocity is not None:
        left_speed = float(np.linalg.norm(left_velocity))
        right_speed = float(np.linalg.norm(right_velocity))
        if left_speed >= config.minimum_velocity_for_direction_mps and right_speed >= config.minimum_velocity_for_direction_mps:
            direction_cosine = float(np.clip(np.dot(left_velocity, right_velocity) / (left_speed * right_speed), -1.0, 1.0))
            direction_score = (direction_cosine + 1.0) / 2.0
            if motion_score is None:
                motion_score = direction_score
            else:
                # Direction is deliberately a minor modifier: footballers turn.
                motion_score = 0.85 * motion_score + 0.15 * direction_score

    left_conf = _endpoint_confidence(left)
    right_conf = _endpoint_confidence(right)
    confidence_score = None
    if left_conf is not None and right_conf is not None:
        confidence_score = math.sqrt(left_conf * right_conf)

    motion = {
        "gap_s": gap_s,
        "distance_m": distance,
        "raw_transition_speed_mps": raw_transition_speed,
        "hard_required_speed_mps": hard_required_speed,
        # Backward-compatible name: this is the conservative hard-envelope
        # speed after positional slack, not the soft ranking speed.
        "required_speed_mps": hard_required_speed,
        "max_transition_speed_mps": config.max_transition_speed_mps,
        "transition_position_slack_m": config.transition_position_slack_m,
        "forward_prediction_error_m": forward_error,
        "backward_prediction_error_m": backward_error,
        "direction_cosine": direction_cosine,
        "predecessor_endpoint": _point_summary(left),
        "successor_endpoint": _point_summary(right),
    }
    scores = {
        "speed": _clamp01(speed_score),
        "gap": _clamp01(gap_score),
        "motion": (None if motion_score is None else _clamp01(motion_score)),
        "endpoint_confidence": confidence_score,
    }
    return motion, scores


def _overlap_stats(first: IdentityNode, second: IdentityNode) -> dict[str, Any]:
    first_by_frame = {int(point["frame_index"]): point for point in first.points}
    second_by_frame = {int(point["frame_index"]): point for point in second.points}
    common = sorted(set(first_by_frame) & set(second_by_frame))
    separations = [
        float(np.linalg.norm(_point_xy(first_by_frame[frame]) - _point_xy(second_by_frame[frame])))
        for frame in common
    ]
    return {
        "common_frame_count": len(common),
        "first_common_frame": common[0] if common else None,
        "last_common_frame": common[-1] if common else None,
        "separation_m": _distribution(separations),
    }


def _score_components(
    *,
    appearance_score: float | None,
    motion_scores: Mapping[str, float | None],
    config: PersistentActorCandidateConfig,
) -> tuple[dict[str, float | None], float | None]:
    components = {
        "appearance": appearance_score,
        "motion": motion_scores.get("motion"),
        "speed": motion_scores.get("speed"),
        "gap": motion_scores.get("gap"),
        "endpoint_confidence": motion_scores.get("endpoint_confidence"),
    }
    weights = {
        "appearance": config.appearance_weight,
        "motion": config.motion_weight,
        "speed": config.speed_weight,
        "gap": config.gap_weight,
        "endpoint_confidence": config.endpoint_confidence_weight,
    }
    numerator = 0.0
    denominator = 0.0
    for name, value in components.items():
        if value is None:
            continue
        weight = weights[name]
        numerator += weight * _clamp01(value)
        denominator += weight
    return components, (None if denominator <= 0 else numerator / denominator)


def build_candidate_diagnostics(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    appearance_rows: Sequence[Mapping[str, Any]],
    physical_candidate_rows: Sequence[Mapping[str, Any]],
    trusted_appearance_calibration: Mapping[str, Any] | None = None,
    config: PersistentActorCandidateConfig | None = None,
) -> dict[str, Any]:
    selected = config or PersistentActorCandidateConfig()
    memberships, assignment = _identity_assignment(identity_rows)
    nodes = build_identity_nodes(identity_rows, trajectory_rows)
    node_by_id = {node.identity_id: node for node in nodes}
    appearance_by_pair = _appearance_rows_by_pair(appearance_rows)
    physical_by_pair = _physical_rows_by_pair(physical_candidate_rows)
    calibration = build_appearance_calibration(
        assignment=assignment,
        appearance_rows=appearance_rows,
        physical_rows=physical_candidate_rows,
        config=selected,
        trusted_calibration=trusted_appearance_calibration,
    )
    population = estimate_population(
        trajectory_rows,
        selected,
        identity_rows=identity_rows,
        physical_candidate_rows=physical_candidate_rows,
    )

    pair_rows: list[dict[str, Any]] = []
    for identity_a, identity_b in itertools.combinations(sorted(node_by_id), 2):
        first = node_by_id[identity_a]
        second = node_by_id[identity_b]
        hard = _hard_constraints_between(first.tracklet_ids, second.tracklet_ids, physical_by_pair)
        appearance = _identity_appearance(first.tracklet_ids, second.tracklet_ids, appearance_by_pair, calibration)
        overlap = _overlap_stats(first, second)

        status = "candidate"
        hard_reason = None
        relation = "unknown"
        solver_eligible = False
        predecessor: IdentityNode | None = None
        successor: IdentityNode | None = None
        motion: dict[str, Any] | None = None
        motion_scores: dict[str, float | None] = {
            "speed": None,
            "gap": None,
            "motion": None,
            "endpoint_confidence": None,
        }

        if hard["hard_constraint_count"] > 0:
            status = "cannot_link"
            hard_reason = "v1_hard_constraint"

        if first.end_time_s < second.start_time_s:
            relation = "sequential"
            predecessor, successor = first, second
        elif second.end_time_s < first.start_time_s:
            relation = "sequential"
            predecessor, successor = second, first
        elif overlap["common_frame_count"] > 0:
            relation = "overlap"
        else:
            relation = "interleaved_support"

        if relation == "sequential":
            assert predecessor is not None and successor is not None
            motion, motion_scores = _motion_components(predecessor, successor, selected)
            if motion["hard_required_speed_mps"] > selected.max_transition_speed_mps:
                status = "cannot_link"
                hard_reason = hard_reason or "endpoint_transition_impossible"
            solver_eligible = status == "candidate"
        elif relation == "overlap":
            # Compatible overlap can indicate duplicate fragments, but it is
            # intentionally kept out of the sequential path-cover solver until
            # Hito 6B explicitly handles duplicate-overlap components.
            solver_eligible = False
        else:
            solver_eligible = False

        components, link_score = _score_components(
            appearance_score=appearance.get("calibrated_score"),
            motion_scores=motion_scores,
            config=selected,
        )
        if status == "cannot_link":
            link_score = None

        pair_rows.append(
            {
                "schema_version": 1,
                "candidate_version": CANDIDATE_VERSION,
                "identity_a": identity_a,
                "identity_b": identity_b,
                "status": status,
                "relation": relation,
                "solver_eligible": solver_eligible,
                "hard_block_reason": hard_reason,
                "predecessor_identity_id": (
                    predecessor.identity_id if predecessor is not None else None
                ),
                "successor_identity_id": (
                    successor.identity_id if successor is not None else None
                ),
                "v1_hard_constraints": hard,
                "overlap": overlap,
                "motion": motion,
                "appearance": appearance,
                "score_components": components,
                "link_score": link_score,
                "identity_decision": None,
            }
        )

    # Per-predecessor rank is diagnostic only.  It is useful for reviewing
    # whether the visually obvious continuation (e.g. a goalkeeper fragment)
    # appears near the top before any solver is allowed to merge it.
    by_predecessor: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in pair_rows:
        if row["solver_eligible"] and row["link_score"] is not None:
            by_predecessor[str(row["predecessor_identity_id"])].append(row)
    for rows in by_predecessor.values():
        rows.sort(key=lambda row: (-float(row["link_score"]), str(row["successor_identity_id"])))
        for rank, row in enumerate(rows, start=1):
            row["predecessor_rank"] = rank
    for row in pair_rows:
        row.setdefault("predecessor_rank", None)

    candidate_rows = [row for row in pair_rows if row["status"] == "candidate"]
    solver_rows = [row for row in pair_rows if row["solver_eligible"]]
    blocked_rows = [row for row in pair_rows if row["status"] == "cannot_link"]
    relation_counts = collections.Counter(str(row["relation"]) for row in pair_rows)
    hard_reason_counts = collections.Counter(
        str(row["hard_block_reason"])
        for row in blocked_rows
        if row["hard_block_reason"] is not None
    )
    top_solver = sorted(
        solver_rows,
        key=lambda row: (-float(row["link_score"]), str(row["predecessor_identity_id"]), str(row["successor_identity_id"])),
    )[:25]

    summary = {
        "schema_version": 1,
        "candidate_version": CANDIDATE_VERSION,
        "identity_decisions": False,
        "canonical_tracking_v1_unchanged": True,
        "identity_count": len(nodes),
        "identity_pair_count": len(pair_rows),
        "candidate_pair_count": len(candidate_rows),
        "cannot_link_pair_count": len(blocked_rows),
        "solver_eligible_sequential_count": len(solver_rows),
        "relation_counts": dict(sorted(relation_counts.items())),
        "hard_block_reason_counts": dict(sorted(hard_reason_counts.items())),
        "population": population,
        "appearance_calibration": calibration.to_dict(),
        "link_score_distribution": _distribution([
            float(row["link_score"])
            for row in solver_rows
            if row["link_score"] is not None
        ]),
        "top_solver_candidates": [
            {
                "predecessor_identity_id": row["predecessor_identity_id"],
                "successor_identity_id": row["successor_identity_id"],
                "link_score": row["link_score"],
                "gap_s": row["motion"]["gap_s"] if row["motion"] else None,
                "distance_m": row["motion"]["distance_m"] if row["motion"] else None,
                "raw_transition_speed_mps": row["motion"]["raw_transition_speed_mps"] if row["motion"] else None,
                "hard_required_speed_mps": row["motion"]["hard_required_speed_mps"] if row["motion"] else None,
                "required_speed_mps": row["motion"]["required_speed_mps"] if row["motion"] else None,
                "appearance_cosine": row["appearance"].get("aggregate_cosine_weighted_mean"),
                "appearance_score": row["appearance"].get("calibrated_score"),
                "predecessor_rank": row["predecessor_rank"],
            }
            for row in top_solver
        ],
        "config": selected.to_dict(),
        "provenance": {
            "identity_publish_version": IDENTITY_PUBLISH_VERSION,
            "final_publish_version": FINAL_PUBLISH_VERSION,
            "physical_graph_version": PHYSICAL_GRAPH_VERSION,
        },
    }
    return {
        "nodes": tuple(node.to_dict() for node in nodes),
        "pairs": tuple(pair_rows),
        "population": population,
        "appearance_calibration": calibration.to_dict(),
        "summary": summary,
    }
