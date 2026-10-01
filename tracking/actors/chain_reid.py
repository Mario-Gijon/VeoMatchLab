"""Tracking V1 Hito 7E.3A: chain-level ReID evidence.

This stage operates on the conservative temporal paths produced by Hito 7E.2.
It does NOT merge actor paths.

Why chain-level?
----------------
Counting raw tracklet-to-tracklet appearance edges is misleading: long paths
produce many combinatorial pairs even when every individual edge is weak.
Instead, this module builds one robust appearance profile per provisional actor
path and performs sparse temporal retrieval between those chain profiles.

Calibration is also moved to the chain level:
* positive proxy: split one conservative temporal path into early/late halves;
* negative proxy: two provisional paths connected by any hard tracklet
  cannot-link;
* same/different anonymous kit cluster is used only to choose the appropriate
  negative distribution.

Only sequential path pairs are eligible for temporal re-identification.
Overlapping paths are diagnostic/non-candidates here.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from tracking.appearance.evidence import TrackletAppearanceProfile


CHAIN_REID_VERSION = "actor-chain-reid-v1"


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _unit(values: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} has zero norm")
    return vector / norm


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    a = _unit(first, label="first")
    b = _unit(second, label="second")
    if a.shape != b.shape:
        raise ValueError("embedding dimensions differ")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("geometric mean requires values")
    if np.any(~np.isfinite(array)):
        raise ValueError("geometric mean got non-finite value")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("geometric mean expects [0,1]")
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("pair requires distinct ids")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, slots=True)
class ChainReIDConfig:
    top_k_successors: int = 16
    top_k_predecessors: int = 16
    one_sided_rank_floor: float = 0.05
    time_epsilon_s: float = 1e-9

    def __post_init__(self) -> None:
        if self.top_k_successors < 1:
            raise ValueError("top_k_successors must be >= 1")
        if self.top_k_predecessors < 1:
            raise ValueError("top_k_predecessors must be >= 1")
        if not 0.0 < self.one_sided_rank_floor <= 1.0:
            raise ValueError("one_sided_rank_floor must be in (0,1]")
        if self.time_epsilon_s < 0.0:
            raise ValueError("time_epsilon_s must be >= 0")


@dataclass(frozen=True, slots=True)
class ChainProfile:
    actor_id: str
    tracklet_ids: tuple[str, ...]
    profiled_tracklet_ids: tuple[str, ...]
    aggregate_embedding: tuple[float, ...]
    reliability: float
    first_timestamp_s: float
    last_timestamp_s: float
    kit_cluster_id: int | None
    kit_dominance: float | None

    @property
    def profiled_tracklet_count(self) -> int:
        return len(self.profiled_tracklet_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "chain_reid_version": CHAIN_REID_VERSION,
            "actor_id": self.actor_id,
            "tracklet_ids": list(self.tracklet_ids),
            "profiled_tracklet_ids": list(self.profiled_tracklet_ids),
            "profiled_tracklet_count": self.profiled_tracklet_count,
            "aggregate_embedding": list(self.aggregate_embedding),
            "reliability": self.reliability,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
            "kit_cluster_id": self.kit_cluster_id,
            "kit_dominance": self.kit_dominance,
        }


@dataclass(frozen=True, slots=True)
class ChainCalibration:
    positive: tuple[float, ...]
    same_kit_negative: tuple[float, ...]
    different_kit_negative: tuple[float, ...]
    unknown_kit_negative: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.positive:
            raise ValueError("chain-level positive calibration is empty")
        all_negative = (
            len(self.same_kit_negative)
            + len(self.different_kit_negative)
            + len(self.unknown_kit_negative)
        )
        if all_negative == 0:
            raise ValueError("chain-level negative calibration is empty")

    def negatives(self, family: str) -> tuple[float, ...]:
        if family == "same_kit" and self.same_kit_negative:
            return self.same_kit_negative
        if family == "different_kit" and self.different_kit_negative:
            return self.different_kit_negative
        if family == "unknown_kit" and self.unknown_kit_negative:
            return self.unknown_kit_negative
        pooled = tuple(sorted(
            self.same_kit_negative
            + self.different_kit_negative
            + self.unknown_kit_negative
        ))
        return pooled

    @staticmethod
    def _cdf(values: Sequence[float], value: float) -> float:
        count = bisect_right(values, float(value))
        return float((count + 0.5) / (len(values) + 1.0))

    def support(self, cosine: float, *, family: str) -> dict[str, float]:
        negatives = self.negatives(family)
        negative_cdf = self._cdf(negatives, cosine)
        positive_cdf = self._cdf(self.positive, cosine)
        support = _geometric_mean([negative_cdf, positive_cdf])
        return {
            "negative_cdf": negative_cdf,
            "negative_tail_fraction": 1.0 - negative_cdf,
            "positive_cdf": positive_cdf,
            "support": support,
        }


@dataclass(frozen=True, slots=True)
class ChainCandidate:
    predecessor_actor_id: str
    successor_actor_id: str
    gap_s: float
    aggregate_cosine: float
    kit_family: str
    empirical_support: float
    profile_confidence: float
    forward_retrieval_rank: int | None
    backward_retrieval_rank: int | None
    retrieval_support: float
    rank_score: float
    outgoing_score_rank: int | None = None
    incoming_score_rank: int | None = None
    mutual_best: bool = False
    mutual_top3: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "chain_reid_version": CHAIN_REID_VERSION,
            "predecessor_actor_id": self.predecessor_actor_id,
            "successor_actor_id": self.successor_actor_id,
            "gap_s": self.gap_s,
            "aggregate_cosine": self.aggregate_cosine,
            "kit_family": self.kit_family,
            "empirical_support": self.empirical_support,
            "profile_confidence": self.profile_confidence,
            "forward_retrieval_rank": self.forward_retrieval_rank,
            "backward_retrieval_rank": self.backward_retrieval_rank,
            "retrieval_support": self.retrieval_support,
            "rank_score": self.rank_score,
            "rank_score_semantics": "relative_chain_reid_evidence_not_probability",
            "outgoing_score_rank": self.outgoing_score_rank,
            "incoming_score_rank": self.incoming_score_rank,
            "mutual_best": self.mutual_best,
            "mutual_top3": self.mutual_top3,
            "identity_decision": None,
        }


def _profile_weight(profile: TrackletAppearanceProfile) -> float:
    return max(
        1e-6,
        float(profile.reliability)
        * math.sqrt(max(1.0, float(profile.effective_sample_count))),
    )


def _aggregate_profiles(
    profiles: Sequence[TrackletAppearanceProfile],
) -> tuple[tuple[float, ...], float]:
    if not profiles:
        raise ValueError("cannot aggregate no profiles")
    vectors = np.vstack([
        _unit(profile.aggregate_embedding, label=profile.tracklet_id)
        for profile in profiles
    ])
    weights = np.asarray(
        [_profile_weight(profile) for profile in profiles],
        dtype=np.float64,
    )
    aggregate = np.sum(vectors * weights[:, None], axis=0)
    aggregate = _unit(aggregate, label="chain aggregate")
    mean_reliability = float(
        np.average(
            [profile.reliability for profile in profiles],
            weights=weights,
        )
    )
    support = 1.0 - math.exp(-len(profiles) / 2.0)
    reliability = _clamp01(mean_reliability * support)
    return (
        tuple(float(v) for v in aggregate),
        reliability,
    )


def _kit_summary(
    tracklet_ids: Sequence[str],
    kit_by_tracklet: Mapping[str, Mapping[str, Any]],
) -> tuple[int | None, float | None]:
    mass: dict[int, float] = defaultdict(float)
    for tracklet_id in tracklet_ids:
        row = kit_by_tracklet.get(tracklet_id)
        if row is None:
            continue
        cluster = int(row["cluster_id"])
        reliability = _clamp01(
            _finite(
                "kit profile_reliability",
                row.get("profile_reliability", 0.0),
            )
        )
        margin = max(
            0.0,
            _finite("kit margin", row.get("margin", 0.0)),
        )
        mass[cluster] += reliability * (0.05 + margin)

    if not mass:
        return None, None
    ordered = sorted(
        mass.items(),
        key=lambda item: (-item[1], item[0]),
    )
    total = sum(value for _, value in ordered)
    return ordered[0][0], float(ordered[0][1] / total)


def build_chain_profiles(
    *,
    actor_rows: Sequence[Mapping[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
    tracklet_profiles: Sequence[TrackletAppearanceProfile],
    kit_assignments: Sequence[Mapping[str, Any]],
) -> tuple[ChainProfile, ...]:
    actor_members = {
        str(row["actor_id"]): tuple(
            str(value) for value in row["tracklet_ids"]
        )
        for row in actor_rows
    }
    actor_windows = {
        str(row["actor_id"]): (
            _finite("start_time_s", row["start_time_s"]),
            _finite("end_time_s", row["end_time_s"]),
        )
        for row in actor_rows
    }
    assigned = {
        str(row["tracklet_id"]): str(row["actor_id"])
        for row in assignments
    }

    for actor_id, members in actor_members.items():
        for tracklet_id in members:
            if assigned.get(tracklet_id) != actor_id:
                raise ValueError("actor/assignment artifact mismatch")

    profile_by_tracklet = {
        profile.tracklet_id: profile
        for profile in tracklet_profiles
    }
    kit_by_tracklet = {
        str(row["tracklet_id"]): row
        for row in kit_assignments
    }

    result: list[ChainProfile] = []
    for actor_id in sorted(actor_members):
        members = actor_members[actor_id]
        profiles = [
            profile_by_tracklet[tracklet_id]
            for tracklet_id in members
            if tracklet_id in profile_by_tracklet
        ]
        if not profiles:
            continue

        profiles.sort(
            key=lambda profile: (
                profile.first_timestamp_s,
                profile.tracklet_id,
            )
        )
        aggregate, reliability = _aggregate_profiles(profiles)
        kit_cluster, kit_dominance = _kit_summary(
            members,
            kit_by_tracklet,
        )
        start, end = actor_windows[actor_id]

        result.append(
            ChainProfile(
                actor_id=actor_id,
                tracklet_ids=members,
                profiled_tracklet_ids=tuple(
                    profile.tracklet_id
                    for profile in profiles
                ),
                aggregate_embedding=aggregate,
                reliability=reliability,
                first_timestamp_s=start,
                last_timestamp_s=end,
                kit_cluster_id=kit_cluster,
                kit_dominance=kit_dominance,
            )
        )

    return tuple(result)


def _kit_family(a: ChainProfile, b: ChainProfile) -> str:
    if a.kit_cluster_id is None or b.kit_cluster_id is None:
        return "unknown_kit"
    return (
        "same_kit"
        if a.kit_cluster_id == b.kit_cluster_id
        else "different_kit"
    )


def fit_chain_calibration(
    *,
    chain_profiles: Sequence[ChainProfile],
    tracklet_profiles: Sequence[TrackletAppearanceProfile],
    assignments: Sequence[Mapping[str, Any]],
    hard_conflicts: Sequence[Mapping[str, Any]],
    trusted_positive_cosines: Sequence[float] | None = None,
) -> tuple[ChainCalibration, tuple[dict[str, Any], ...], tuple[dict[str, Any], ...]]:
    profile_by_tracklet = {
        profile.tracklet_id: profile
        for profile in tracklet_profiles
    }
    actor_by_tracklet = {
        str(row["tracklet_id"]): str(row["actor_id"])
        for row in assignments
    }
    chain_by_actor = {
        profile.actor_id: profile
        for profile in chain_profiles
    }

    positives: list[float] = []
    positive_rows: list[dict[str, Any]] = []

    for chain in chain_profiles:
        profiles = [
            profile_by_tracklet[tracklet_id]
            for tracklet_id in chain.profiled_tracklet_ids
        ]
        if len(profiles) < 2:
            continue
        midpoint = len(profiles) // 2
        left = profiles[:midpoint]
        right = profiles[midpoint:]
        if not left or not right:
            continue
        left_embedding, _ = _aggregate_profiles(left)
        right_embedding, _ = _aggregate_profiles(right)
        cosine = _cosine(left_embedding, right_embedding)
        positives.append(cosine)
        positive_rows.append({
            "actor_id": chain.actor_id,
            "left_tracklet_count": len(left),
            "right_tracklet_count": len(right),
            "cosine": cosine,
            "proxy": "same_temporal_backbone_split",
        })

    hard_actor_pairs: set[tuple[str, str]] = set()
    for row in hard_conflicts:
        ta = str(row["tracklet_a"])
        tb = str(row["tracklet_b"])
        aa = actor_by_tracklet.get(ta)
        ab = actor_by_tracklet.get(tb)
        if aa is None or ab is None:
            continue
        if aa == ab:
            raise ValueError(
                f"hard conflict exists inside temporal actor {aa}"
            )
        hard_actor_pairs.add(_canonical_pair(aa, ab))

    negative_groups: dict[str, list[float]] = {
        "same_kit": [],
        "different_kit": [],
        "unknown_kit": [],
    }
    negative_rows: list[dict[str, Any]] = []

    for actor_a, actor_b in sorted(hard_actor_pairs):
        a = chain_by_actor.get(actor_a)
        b = chain_by_actor.get(actor_b)
        if a is None or b is None:
            continue
        family = _kit_family(a, b)
        cosine = _cosine(
            a.aggregate_embedding,
            b.aggregate_embedding,
        )
        negative_groups[family].append(cosine)
        negative_rows.append({
            "actor_a": actor_a,
            "actor_b": actor_b,
            "kit_family": family,
            "cosine": cosine,
            "proxy": "cross_chain_hard_cannot_link",
        })

    trusted = (
        tuple(
            sorted(
                _finite("trusted positive cosine", value)
                for value in trusted_positive_cosines
            )
        )
        if trusted_positive_cosines is not None
        else tuple(sorted(positives))
    )
    if not trusted:
        raise ValueError("trusted positive calibration is empty")

    calibration = ChainCalibration(
        positive=trusted,
        same_kit_negative=tuple(sorted(negative_groups["same_kit"])),
        different_kit_negative=tuple(sorted(negative_groups["different_kit"])),
        unknown_kit_negative=tuple(sorted(negative_groups["unknown_kit"])),
    )
    return calibration, tuple(positive_rows), tuple(negative_rows)


def _top_k_indices(
    scores: np.ndarray,
    valid: np.ndarray,
    ids: Sequence[str],
    k: int,
) -> list[int]:
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return []
    ranked = sorted(
        (int(i) for i in indices),
        key=lambda i: (-float(scores[i]), ids[i]),
    )
    return ranked[:k]


def build_chain_candidates(
    *,
    chain_profiles: Sequence[ChainProfile],
    calibration: ChainCalibration,
    hard_actor_pairs: set[tuple[str, str]],
    config: ChainReIDConfig | None = None,
) -> tuple[ChainCandidate, ...]:
    selected = config or ChainReIDConfig()
    ordered = tuple(
        sorted(chain_profiles, key=lambda item: item.actor_id)
    )
    if not ordered:
        return ()

    ids = [item.actor_id for item in ordered]
    vectors = np.vstack([
        _unit(item.aggregate_embedding, label=item.actor_id)
        for item in ordered
    ])
    similarities = np.clip(vectors @ vectors.T, -1.0, 1.0)

    forward_rank: dict[tuple[int, int], int] = {}
    backward_rank: dict[tuple[int, int], int] = {}

    count = len(ordered)
    for i, predecessor in enumerate(ordered):
        valid = np.zeros(count, dtype=bool)
        for j, successor in enumerate(ordered):
            if i == j:
                continue
            if (
                predecessor.last_timestamp_s
                <= successor.first_timestamp_s
                + selected.time_epsilon_s
                and _canonical_pair(
                    predecessor.actor_id,
                    successor.actor_id,
                ) not in hard_actor_pairs
            ):
                valid[j] = True
        for rank, j in enumerate(
            _top_k_indices(
                similarities[i],
                valid,
                ids,
                selected.top_k_successors,
            ),
            start=1,
        ):
            forward_rank[(i, j)] = rank

    for j, successor in enumerate(ordered):
        valid = np.zeros(count, dtype=bool)
        for i, predecessor in enumerate(ordered):
            if i == j:
                continue
            if (
                predecessor.last_timestamp_s
                <= successor.first_timestamp_s
                + selected.time_epsilon_s
                and _canonical_pair(
                    predecessor.actor_id,
                    successor.actor_id,
                ) not in hard_actor_pairs
            ):
                valid[i] = True

        indices = _top_k_indices(
            similarities[:, j],
            valid,
            ids,
            selected.top_k_predecessors,
        )
        for rank, i in enumerate(indices, start=1):
            backward_rank[(i, j)] = rank

    pairs = sorted(
        set(forward_rank) | set(backward_rank)
    )

    def rank_support(rank: int | None) -> float:
        if rank is None:
            return selected.one_sided_rank_floor
        return 1.0 / math.sqrt(max(1, rank))

    result: list[ChainCandidate] = []
    for i, j in pairs:
        a = ordered[i]
        b = ordered[j]
        cosine = float(similarities[i, j])
        family = _kit_family(a, b)
        empirical = calibration.support(
            cosine,
            family=family,
        )
        forward = forward_rank.get((i, j))
        backward = backward_rank.get((i, j))
        retrieval_support = _geometric_mean([
            rank_support(forward),
            rank_support(backward),
        ])
        confidence = math.sqrt(
            max(0.0, a.reliability * b.reliability)
        )
        raw_support = _geometric_mean([
            empirical["support"],
            retrieval_support,
        ])
        score = _clamp01(
            0.5 + confidence * (raw_support - 0.5)
        )

        result.append(
            ChainCandidate(
                predecessor_actor_id=a.actor_id,
                successor_actor_id=b.actor_id,
                gap_s=max(
                    0.0,
                    b.first_timestamp_s - a.last_timestamp_s,
                ),
                aggregate_cosine=cosine,
                kit_family=family,
                empirical_support=empirical["support"],
                profile_confidence=confidence,
                forward_retrieval_rank=forward,
                backward_retrieval_rank=backward,
                retrieval_support=retrieval_support,
                rank_score=score,
            )
        )

    outgoing: dict[str, list[int]] = defaultdict(list)
    incoming: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(result):
        outgoing[row.predecessor_actor_id].append(index)
        incoming[row.successor_actor_id].append(index)

    outgoing_rank: dict[int, int] = {}
    incoming_rank: dict[int, int] = {}

    for indices in outgoing.values():
        indices.sort(
            key=lambda index: (
                -result[index].rank_score,
                result[index].successor_actor_id,
            )
        )
        for rank, index in enumerate(indices, start=1):
            outgoing_rank[index] = rank

    for indices in incoming.values():
        indices.sort(
            key=lambda index: (
                -result[index].rank_score,
                result[index].predecessor_actor_id,
            )
        )
        for rank, index in enumerate(indices, start=1):
            incoming_rank[index] = rank

    return tuple(
        replace(
            row,
            outgoing_score_rank=outgoing_rank[index],
            incoming_score_rank=incoming_rank[index],
            mutual_best=(
                outgoing_rank[index] == 1
                and incoming_rank[index] == 1
            ),
            mutual_top3=(
                outgoing_rank[index] <= 3
                and incoming_rank[index] <= 3
            ),
        )
        for index, row in enumerate(result)
    )


def summarize_chain_reid(
    *,
    chain_profiles: Sequence[ChainProfile],
    candidates: Sequence[ChainCandidate],
    calibration: ChainCalibration,
    total_actor_count: int,
    config: ChainReIDConfig,
) -> dict[str, Any]:
    bands = Counter()
    for row in candidates:
        for threshold in (0.60, 0.65, 0.70, 0.75, 0.80):
            if row.rank_score >= threshold:
                bands[f"ge_{threshold:.2f}"] += 1
                if row.mutual_best:
                    bands[f"mutual_best_ge_{threshold:.2f}"] += 1
                if row.mutual_top3:
                    bands[f"mutual_top3_ge_{threshold:.2f}"] += 1

    return {
        "schema_version": 1,
        "chain_reid_version": CHAIN_REID_VERSION,
        "actor_count": total_actor_count,
        "profiled_actor_count": len(chain_profiles),
        "unprofiled_actor_count": total_actor_count - len(chain_profiles),
        "candidate_count": len(candidates),
        "mutual_best_count": sum(row.mutual_best for row in candidates),
        "mutual_top3_count": sum(row.mutual_top3 for row in candidates),
        "rank_score": _stats([row.rank_score for row in candidates]),
        "aggregate_cosine": _stats(
            [row.aggregate_cosine for row in candidates]
        ),
        "gap_s": _stats([row.gap_s for row in candidates]),
        "score_band_counts": dict(sorted(bands.items())),
        "calibration": {
            "trusted_positive": _stats(list(calibration.positive)),
            "same_kit_negative": _stats(
                list(calibration.same_kit_negative)
            ),
            "different_kit_negative": _stats(
                list(calibration.different_kit_negative)
            ),
            "unknown_kit_negative": _stats(
                list(calibration.unknown_kit_negative)
            ),
        },
        "config": {
            "top_k_successors": config.top_k_successors,
            "top_k_predecessors": config.top_k_predecessors,
            "one_sided_rank_floor": config.one_sided_rank_floor,
            "time_epsilon_s": config.time_epsilon_s,
        },
        "identity_decisions": False,
        "raw_edge_multiplicity_used_as_evidence": False,
        "overlapping_chain_candidates_allowed": False,
        "trusted_positive_source": "7D same-tracklet chronological split",
        "provisional_chain_split_used_for_calibration": False,
    }
