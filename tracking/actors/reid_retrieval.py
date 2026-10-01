"""Hito 7B.2: sparse global ReID retrieval for offline football tracking.

The expensive mistake to avoid is materialising rich appearance evidence for
every unordered pair of tracklets.  For a 90 s clip with ~1.8k tracklets that
already means ~1.8M pairs.

This module uses only the robust aggregate 256-D ReID profile for retrieval:

    all profiled tracklets
        -> block cosine matrix multiplication
        -> top-K non-conflicting neighbours per tracklet
        -> symmetric sparse candidate union

Important semantics
-------------------
* Retrieval is GLOBAL over the clip by default.  A player may disappear for
  tens of seconds and still be a candidate later.
* There is NO cosine threshold here.  Top-K is a retrieval budget, not an
  identity decision.
* Exact physical hard conflicts always win over appearance.  A high ReID
  cosine can never resurrect a pair known to represent two simultaneous,
  spatially distinct people.
* Missing ReID means unknown.  Unprofiled tracklets are not rejected; they are
  simply absent from this appearance-only retrieval stage.
* No actor/player identity is assigned here.
* Rich sample-to-sample appearance evidence is intentionally deferred until
  after the sparse union is built.

The later global/offline solver can combine these candidates with local
physics, actor occupancy, team, jersey number and future evidence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tracking.appearance.evidence import TrackletAppearanceProfile


REID_RETRIEVAL_VERSION = "sparse-reid-retrieval-v1"


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise ValueError("pair requires two distinct tracklets")
    return (first, second) if first < second else (second, first)


def _unit_vector(values: Sequence[float], *, label: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{label} must be a non-empty 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{label} contains non-finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"{label} has zero norm")
    return vector / norm


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
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
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


@dataclass(frozen=True, slots=True)
class SparseReIDRetrievalConfig:
    """Retrieval budget, never an identity threshold."""

    top_k_per_tracklet: int = 64
    cosine_block_size: int = 256

    def __post_init__(self) -> None:
        if self.top_k_per_tracklet < 1:
            raise ValueError("top_k_per_tracklet must be >= 1")
        if self.cosine_block_size < 1:
            raise ValueError("cosine_block_size must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DirectedReIDNeighbor:
    """One query-tracklet -> neighbour retrieval result."""

    query_tracklet_id: str
    candidate_tracklet_id: str
    rank: int
    aggregate_cosine: float

    def __post_init__(self) -> None:
        if self.query_tracklet_id == self.candidate_tracklet_id:
            raise ValueError("a tracklet cannot retrieve itself")
        if self.rank < 1:
            raise ValueError("rank must be >= 1")
        if not math.isfinite(self.aggregate_cosine):
            raise ValueError("aggregate_cosine must be finite")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "retrieval_version": REID_RETRIEVAL_VERSION,
            "query_tracklet_id": self.query_tracklet_id,
            "candidate_tracklet_id": self.candidate_tracklet_id,
            "rank": self.rank,
            "aggregate_cosine": self.aggregate_cosine,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class SparseReIDCandidate:
    """Symmetric candidate pair created by one or both top-K directions."""

    tracklet_a: str
    tracklet_b: str
    aggregate_cosine: float
    rank_a_to_b: int | None
    rank_b_to_a: int | None

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("candidate pair requires canonical ordering")
        if self.rank_a_to_b is None and self.rank_b_to_a is None:
            raise ValueError("candidate must have at least one retrieval direction")
        if not math.isfinite(self.aggregate_cosine):
            raise ValueError("aggregate_cosine must be finite")

    @property
    def mutual(self) -> bool:
        return (
            self.rank_a_to_b is not None
            and self.rank_b_to_a is not None
        )

    @property
    def best_rank(self) -> int:
        return min(
            rank
            for rank in (self.rank_a_to_b, self.rank_b_to_a)
            if rank is not None
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "retrieval_version": REID_RETRIEVAL_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "candidate_reason": "appearance_topk",
            "aggregate_cosine": self.aggregate_cosine,
            "rank_a_to_b": self.rank_a_to_b,
            "rank_b_to_a": self.rank_b_to_a,
            "best_rank": self.best_rank,
            "mutual_topk": self.mutual,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class SparseReIDRetrieval:
    profiles: tuple[TrackletAppearanceProfile, ...]
    neighbors: tuple[DirectedReIDNeighbor, ...]
    candidates: tuple[SparseReIDCandidate, ...]
    hard_conflict_pair_count: int
    config: SparseReIDRetrievalConfig

    def summary(
        self,
        *,
        total_tracklet_count: int | None = None,
    ) -> dict[str, Any]:
        profile_count = len(self.profiles)
        possible_profile_pairs = (
            profile_count * (profile_count - 1) // 2
        )
        candidate_degrees = Counter()
        for candidate in self.candidates:
            candidate_degrees[candidate.tracklet_a] += 1
            candidate_degrees[candidate.tracklet_b] += 1

        degrees = [
            float(candidate_degrees.get(profile.tracklet_id, 0))
            for profile in self.profiles
        ]
        cosine_values = [
            candidate.aggregate_cosine
            for candidate in self.candidates
        ]
        mutual_count = sum(
            1 for candidate in self.candidates if candidate.mutual
        )

        payload: dict[str, Any] = {
            "schema_version": 1,
            "retrieval_version": REID_RETRIEVAL_VERSION,
            "profile_count": profile_count,
            "possible_profile_pair_count": possible_profile_pairs,
            "hard_conflict_pair_count": self.hard_conflict_pair_count,
            "directed_neighbor_count": len(self.neighbors),
            "candidate_pair_count": len(self.candidates),
            "mutual_candidate_count": mutual_count,
            "one_sided_candidate_count": (
                len(self.candidates) - mutual_count
            ),
            "candidate_fraction_of_profile_pairs": (
                len(self.candidates) / possible_profile_pairs
                if possible_profile_pairs
                else None
            ),
            "pair_reduction_fraction": (
                1.0 - len(self.candidates) / possible_profile_pairs
                if possible_profile_pairs
                else None
            ),
            "candidate_degree": _distribution(degrees),
            "candidate_aggregate_cosine": _distribution(
                cosine_values
            ),
            "config": self.config.to_dict(),
            "identity_decisions": False,
            "cosine_threshold_used": False,
            "global_temporal_retrieval": True,
            "hard_conflicts_override_appearance": True,
        }
        if total_tracklet_count is not None:
            if total_tracklet_count < profile_count:
                raise ValueError(
                    "total_tracklet_count cannot be below profile_count"
                )
            payload["total_tracklet_count"] = int(
                total_tracklet_count
            )
            payload["unprofiled_tracklet_count"] = int(
                total_tracklet_count - profile_count
            )
            payload["profile_coverage"] = (
                profile_count / total_tracklet_count
                if total_tracklet_count
                else None
            )
        return payload


def _normalize_profiles(
    profiles: Iterable[TrackletAppearanceProfile],
) -> tuple[TrackletAppearanceProfile, ...]:
    ordered = tuple(
        sorted(profiles, key=lambda item: item.tracklet_id)
    )
    seen: set[str] = set()
    dimensions: set[int] = set()
    models: set[tuple[str, str]] = set()

    for profile in ordered:
        if profile.tracklet_id in seen:
            raise ValueError(
                f"duplicate tracklet profile: {profile.tracklet_id}"
            )
        seen.add(profile.tracklet_id)
        dimensions.add(int(profile.embedding_dim))
        models.add(
            (profile.model_id, profile.descriptor_name)
        )

    if len(dimensions) > 1:
        raise ValueError(
            "all profiles must share one embedding dimension"
        )
    if len(models) > 1:
        raise ValueError(
            "all profiles must share one model/descriptor"
        )
    return ordered


def _hard_conflict_pairs(
    rows: Iterable[Any],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()

    for row in rows:
        if isinstance(row, Mapping):
            first = str(row["tracklet_a"])
            second = str(row["tracklet_b"])
        else:
            first = str(row.tracklet_a)
            second = str(row.tracklet_b)
        result.add(_canonical_pair(first, second))

    return result


def _top_k_indices(
    scores: np.ndarray,
    valid_mask: np.ndarray,
    *,
    ids: Sequence[str],
    k: int,
) -> list[int]:
    """Return deterministic top-K indices without sorting the whole row.

    Boundary ties are expanded before the final lexical tie-break so different
    BLAS/block layouts cannot silently change which equal-score item survives.
    """

    valid_indices = np.flatnonzero(valid_mask)
    if valid_indices.size == 0:
        return []

    valid_scores = scores[valid_indices]
    if valid_indices.size <= k:
        finalists = valid_indices
    else:
        cutoff = float(
            np.partition(valid_scores, -k)[-k]
        )
        finalists = valid_indices[
            valid_scores >= cutoff - 1e-12
        ]

    ranked = sorted(
        (int(index) for index in finalists),
        key=lambda index: (
            -float(scores[index]),
            ids[index],
        ),
    )
    return ranked[:k]


def build_sparse_reid_retrieval(
    profiles: Iterable[TrackletAppearanceProfile],
    *,
    hard_conflicts: Iterable[Any] = (),
    config: SparseReIDRetrievalConfig | None = None,
) -> SparseReIDRetrieval:
    """Build global top-K appearance candidates without rich all-pairs evidence."""

    selected = config or SparseReIDRetrievalConfig()
    ordered = _normalize_profiles(profiles)
    if not ordered:
        return SparseReIDRetrieval(
            profiles=(),
            neighbors=(),
            candidates=(),
            hard_conflict_pair_count=0,
            config=selected,
        )

    ids = [profile.tracklet_id for profile in ordered]
    id_to_index = {
        tracklet_id: index
        for index, tracklet_id in enumerate(ids)
    }

    vectors = np.vstack(
        [
            _unit_vector(
                profile.aggregate_embedding,
                label=profile.tracklet_id,
            )
            for profile in ordered
        ]
    ).astype(np.float32, copy=False)

    hard_pairs = _hard_conflict_pairs(hard_conflicts)
    hard_by_index: dict[int, set[int]] = {}
    relevant_hard_pairs = 0
    for first, second in hard_pairs:
        first_index = id_to_index.get(first)
        second_index = id_to_index.get(second)
        if first_index is None or second_index is None:
            continue
        hard_by_index.setdefault(first_index, set()).add(
            second_index
        )
        hard_by_index.setdefault(second_index, set()).add(
            first_index
        )
        relevant_hard_pairs += 1

    neighbors: list[DirectedReIDNeighbor] = []
    count = len(ordered)
    block = selected.cosine_block_size

    for start in range(0, count, block):
        stop = min(count, start + block)
        similarities = vectors[start:stop] @ vectors.T
        similarities = np.clip(
            similarities,
            -1.0,
            1.0,
        )

        for local_index in range(stop - start):
            query_index = start + local_index
            row = similarities[local_index]
            valid = np.ones(count, dtype=bool)
            valid[query_index] = False

            for blocked in hard_by_index.get(
                query_index,
                (),
            ):
                valid[blocked] = False

            ranked = _top_k_indices(
                row,
                valid,
                ids=ids,
                k=selected.top_k_per_tracklet,
            )
            for rank, candidate_index in enumerate(
                ranked,
                start=1,
            ):
                neighbors.append(
                    DirectedReIDNeighbor(
                        query_tracklet_id=ids[query_index],
                        candidate_tracklet_id=ids[
                            candidate_index
                        ],
                        rank=rank,
                        aggregate_cosine=float(
                            row[candidate_index]
                        ),
                    )
                )

    # Union the directed retrieval into canonical undirected candidate pairs.
    pair_rows: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    for neighbor in neighbors:
        pair = _canonical_pair(
            neighbor.query_tracklet_id,
            neighbor.candidate_tracklet_id,
        )
        row = pair_rows.setdefault(
            pair,
            {
                "aggregate_cosine": neighbor.aggregate_cosine,
                "rank_a_to_b": None,
                "rank_b_to_a": None,
            },
        )

        if not math.isclose(
            float(row["aggregate_cosine"]),
            neighbor.aggregate_cosine,
            rel_tol=0.0,
            abs_tol=2e-6,
        ):
            raise ValueError(
                f"asymmetric cosine for pair {pair}"
            )

        if neighbor.query_tracklet_id == pair[0]:
            current = row["rank_a_to_b"]
            row["rank_a_to_b"] = (
                neighbor.rank
                if current is None
                else min(current, neighbor.rank)
            )
        else:
            current = row["rank_b_to_a"]
            row["rank_b_to_a"] = (
                neighbor.rank
                if current is None
                else min(current, neighbor.rank)
            )

    candidates = tuple(
        SparseReIDCandidate(
            tracklet_a=pair[0],
            tracklet_b=pair[1],
            aggregate_cosine=float(
                payload["aggregate_cosine"]
            ),
            rank_a_to_b=payload["rank_a_to_b"],
            rank_b_to_a=payload["rank_b_to_a"],
        )
        for pair, payload in sorted(pair_rows.items())
    )

    # Belt-and-braces invariant: no known hard conflict may survive.
    for candidate in candidates:
        if (
            candidate.tracklet_a,
            candidate.tracklet_b,
        ) in hard_pairs:
            raise AssertionError(
                "hard conflict survived ReID retrieval"
            )

    neighbors.sort(
        key=lambda item: (
            item.query_tracklet_id,
            item.rank,
            item.candidate_tracklet_id,
        )
    )

    return SparseReIDRetrieval(
        profiles=ordered,
        neighbors=tuple(neighbors),
        candidates=candidates,
        hard_conflict_pair_count=relevant_hard_pairs,
        config=selected,
    )
