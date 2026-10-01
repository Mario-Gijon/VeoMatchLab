"""Tracking V1 Hito 7E.3B: conservative persistent chain solver.

Consumes:
* Hito 7E.2 temporal backbone actors/assignments;
* Hito 7E.3A chain-level ReID candidates;
* Hito 7B hard tracklet cannot-links.

This stage turns provisional temporal chains into provisional persistent
identities. It remains conservative:

* only sequential chain candidates are considered;
* each chain has at most one predecessor and one successor;
* default competition policy requires chain-level mutual-best;
* default minimum rank_score is 0.70;
* anonymous different-kit links are rejected by default;
* every proposed merge is checked against ALL hard cannot-links already inside
  both identity components;
* no target population (22 players, etc.) is imposed;
* isolated / uncertain chains remain separate identities.

The solver is deterministic and intentionally prefers fragmentation over
identity contamination.  It is a conservative global stitcher, not a claim of
globally optimal correlation clustering.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


PERSISTENT_CHAIN_SOLVER_VERSION = "persistent-chain-path-v1"


def _finite(name: str, value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _stats(values: Sequence[float]) -> dict[str, Any]:
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


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("pair requires distinct ids")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, slots=True)
class PersistentChainSolverConfig:
    min_rank_score: float = 0.70
    competition_mode: str = "mutual_best"
    allow_different_kit: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_rank_score <= 1.0:
            raise ValueError("min_rank_score must be in [0,1]")
        if self.competition_mode not in {
            "mutual_best",
            "mutual_top3",
        }:
            raise ValueError(
                "competition_mode must be mutual_best or mutual_top3"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SelectedChainLink:
    predecessor_actor_id: str
    successor_actor_id: str
    rank_score: float
    aggregate_cosine: float
    profile_confidence: float
    gap_s: float
    kit_family: str
    competition_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "solver_version": PERSISTENT_CHAIN_SOLVER_VERSION,
            "predecessor_actor_id": self.predecessor_actor_id,
            "successor_actor_id": self.successor_actor_id,
            "rank_score": self.rank_score,
            "aggregate_cosine": self.aggregate_cosine,
            "profile_confidence": self.profile_confidence,
            "gap_s": self.gap_s,
            "kit_family": self.kit_family,
            "competition_mode": self.competition_mode,
            "identity_decision": "provisional_persistent_link",
        }


@dataclass(frozen=True, slots=True)
class PersistentIdentity:
    identity_id: str
    actor_ids: tuple[str, ...]
    tracklet_ids: tuple[str, ...]
    start_time_s: float
    end_time_s: float

    @property
    def actor_count(self) -> int:
        return len(self.actor_ids)

    @property
    def tracklet_count(self) -> int:
        return len(self.tracklet_ids)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "solver_version": PERSISTENT_CHAIN_SOLVER_VERSION,
            "identity_id": self.identity_id,
            "actor_ids": list(self.actor_ids),
            "tracklet_ids": list(self.tracklet_ids),
            "actor_count": self.actor_count,
            "tracklet_count": self.tracklet_count,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "provisional": True,
        }


@dataclass(frozen=True, slots=True)
class PersistentChainSolverResult:
    identities: tuple[PersistentIdentity, ...]
    actor_assignments: tuple[Mapping[str, Any], ...]
    tracklet_assignments: tuple[Mapping[str, Any], ...]
    selected_links: tuple[SelectedChainLink, ...]
    rejection_counts: Mapping[str, int]
    eligible_candidate_count: int
    config: PersistentChainSolverConfig

    def summary(self) -> dict[str, Any]:
        actor_sizes = [identity.actor_count for identity in self.identities]
        tracklet_sizes = [
            identity.tracklet_count for identity in self.identities
        ]
        scores = [link.rank_score for link in self.selected_links]
        gaps = [link.gap_s for link in self.selected_links]

        return {
            "schema_version": 1,
            "solver_version": PERSISTENT_CHAIN_SOLVER_VERSION,
            "identity_count": len(self.identities),
            "non_singleton_identity_count": sum(
                value > 1 for value in actor_sizes
            ),
            "singleton_identity_count": sum(
                value == 1 for value in actor_sizes
            ),
            "largest_identity_actor_count": (
                max(actor_sizes) if actor_sizes else 0
            ),
            "largest_identity_tracklet_count": (
                max(tracklet_sizes) if tracklet_sizes else 0
            ),
            "identity_actor_count": _stats(
                [float(value) for value in actor_sizes]
            ),
            "identity_tracklet_count": _stats(
                [float(value) for value in tracklet_sizes]
            ),
            "eligible_candidate_count": self.eligible_candidate_count,
            "selected_link_count": len(self.selected_links),
            "selected_score": _stats(scores),
            "selected_gap_s": _stats(gaps),
            "selected_kit_family_counts": dict(
                sorted(
                    Counter(
                        link.kit_family
                        for link in self.selected_links
                    ).items()
                )
            ),
            "rejection_counts": dict(
                sorted(self.rejection_counts.items())
            ),
            "config": self.config.to_dict(),
            "population_forced": False,
            "hard_constraints_inviolable": True,
            "fragmentation_preferred_over_contamination": True,
            "provisional": True,
            "canonical": False,
        }


class _ComponentState:
    """Union-find plus explicit actor membership for hard-conflict checks."""

    def __init__(
        self,
        actor_ids: Sequence[str],
        hard_actor_neighbors: Mapping[str, set[str]],
    ) -> None:
        self.parent = {actor_id: actor_id for actor_id in actor_ids}
        self.members = {
            actor_id: {actor_id}
            for actor_id in actor_ids
        }
        self.hard_actor_neighbors = hard_actor_neighbors

    def find(self, actor_id: str) -> str:
        parent = self.parent[actor_id]
        if parent != actor_id:
            self.parent[actor_id] = self.find(parent)
        return self.parent[actor_id]

    def compatible(self, left_actor: str, right_actor: str) -> bool:
        left = self.members[self.find(left_actor)]
        right = self.members[self.find(right_actor)]

        if len(left) > len(right):
            left, right = right, left

        for actor_id in left:
            blocked = self.hard_actor_neighbors.get(actor_id, set())
            if blocked.intersection(right):
                return False
        return True

    def union(self, left_actor: str, right_actor: str) -> None:
        root_a = self.find(left_actor)
        root_b = self.find(right_actor)
        if root_a == root_b:
            return

        if len(self.members[root_a]) < len(self.members[root_b]):
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.members[root_a].update(self.members[root_b])
        del self.members[root_b]


def _competition_passes(
    candidate: Mapping[str, Any],
    mode: str,
) -> bool:
    if mode == "mutual_best":
        return bool(candidate.get("mutual_best", False))
    if mode == "mutual_top3":
        return bool(candidate.get("mutual_top3", False))
    raise ValueError(f"unsupported competition mode {mode!r}")


def _build_hard_actor_graph(
    *,
    actor_by_tracklet: Mapping[str, str],
    hard_conflicts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, set[str]], set[tuple[str, str]]]:
    neighbors: dict[str, set[str]] = defaultdict(set)
    pairs: set[tuple[str, str]] = set()

    for row in hard_conflicts:
        tracklet_a = str(row["tracklet_a"])
        tracklet_b = str(row["tracklet_b"])

        actor_a = actor_by_tracklet.get(tracklet_a)
        actor_b = actor_by_tracklet.get(tracklet_b)
        if actor_a is None or actor_b is None:
            raise ValueError(
                "hard conflict references a tracklet without backbone actor"
            )
        if actor_a == actor_b:
            raise ValueError(
                f"hard conflict already exists inside backbone actor {actor_a}"
            )

        pair = _canonical_pair(actor_a, actor_b)
        if pair in pairs:
            continue

        pairs.add(pair)
        neighbors[actor_a].add(actor_b)
        neighbors[actor_b].add(actor_a)

    return dict(neighbors), pairs


def solve_persistent_chain_paths(
    *,
    actor_rows: Sequence[Mapping[str, Any]],
    backbone_assignments: Sequence[Mapping[str, Any]],
    chain_candidates: Sequence[Mapping[str, Any]],
    hard_conflicts: Sequence[Mapping[str, Any]],
    config: PersistentChainSolverConfig | None = None,
) -> PersistentChainSolverResult:
    selected = config or PersistentChainSolverConfig()

    actors: dict[str, dict[str, Any]] = {}
    for row in actor_rows:
        actor_id = str(row["actor_id"])
        if actor_id in actors:
            raise ValueError(f"duplicate actor_id: {actor_id}")
        tracklet_ids = tuple(
            str(value)
            for value in row["tracklet_ids"]
        )
        if not tracklet_ids:
            raise ValueError("backbone actor has no tracklets")
        actors[actor_id] = {
            "tracklet_ids": tracklet_ids,
            "start_time_s": _finite(
                "start_time_s",
                row["start_time_s"],
            ),
            "end_time_s": _finite(
                "end_time_s",
                row["end_time_s"],
            ),
        }

    actor_by_tracklet: dict[str, str] = {}
    for row in backbone_assignments:
        tracklet_id = str(row["tracklet_id"])
        actor_id = str(row["actor_id"])
        if actor_id not in actors:
            raise ValueError("assignment references unknown actor")
        previous = actor_by_tracklet.get(tracklet_id)
        if previous is not None and previous != actor_id:
            raise ValueError("tracklet assigned to multiple actors")
        actor_by_tracklet[tracklet_id] = actor_id

    expected_tracklets = {
        tracklet_id
        for actor in actors.values()
        for tracklet_id in actor["tracklet_ids"]
    }
    if set(actor_by_tracklet) != expected_tracklets:
        raise ValueError("actor rows and assignments disagree")

    hard_neighbors, hard_actor_pairs = _build_hard_actor_graph(
        actor_by_tracklet=actor_by_tracklet,
        hard_conflicts=hard_conflicts,
    )

    rejections = Counter()
    eligible: list[Mapping[str, Any]] = []

    for row in chain_candidates:
        predecessor = str(row["predecessor_actor_id"])
        successor = str(row["successor_actor_id"])

        if predecessor not in actors or successor not in actors:
            raise ValueError("chain candidate references unknown actor")
        if predecessor == successor:
            raise ValueError("self chain candidate is invalid")

        # 7E.3A already generates only sequential candidates, but verify it.
        if (
            actors[predecessor]["end_time_s"]
            > actors[successor]["start_time_s"] + 1e-9
        ):
            rejections["not_sequential"] += 1
            continue

        score = _finite("rank_score", row["rank_score"])
        if score < selected.min_rank_score:
            rejections["score_below_floor"] += 1
            continue

        if not _competition_passes(
            row,
            selected.competition_mode,
        ):
            rejections["competition_filter"] += 1
            continue

        kit_family = str(row.get("kit_family", "unknown_kit"))
        if (
            kit_family == "different_kit"
            and not selected.allow_different_kit
        ):
            rejections["different_kit"] += 1
            continue

        if (
            _canonical_pair(predecessor, successor)
            in hard_actor_pairs
        ):
            rejections["direct_hard_conflict"] += 1
            continue

        eligible.append(row)

    eligible.sort(
        key=lambda row: (
            -_finite("rank_score", row["rank_score"]),
            -_finite(
                "profile_confidence",
                row["profile_confidence"],
            ),
            _finite("gap_s", row["gap_s"]),
            str(row["predecessor_actor_id"]),
            str(row["successor_actor_id"]),
        )
    )

    successor_of: dict[str, str] = {}
    predecessor_of: dict[str, str] = {}
    components = _ComponentState(
        sorted(actors),
        hard_neighbors,
    )
    selected_links: list[SelectedChainLink] = []

    for row in eligible:
        predecessor = str(row["predecessor_actor_id"])
        successor = str(row["successor_actor_id"])

        if predecessor in successor_of:
            rejections["predecessor_successor_taken"] += 1
            continue
        if successor in predecessor_of:
            rejections["successor_predecessor_taken"] += 1
            continue

        if components.find(predecessor) == components.find(successor):
            rejections["same_component"] += 1
            continue

        if not components.compatible(predecessor, successor):
            rejections["component_hard_conflict"] += 1
            continue

        successor_of[predecessor] = successor
        predecessor_of[successor] = predecessor
        components.union(predecessor, successor)

        selected_links.append(
            SelectedChainLink(
                predecessor_actor_id=predecessor,
                successor_actor_id=successor,
                rank_score=_finite(
                    "rank_score",
                    row["rank_score"],
                ),
                aggregate_cosine=_finite(
                    "aggregate_cosine",
                    row["aggregate_cosine"],
                ),
                profile_confidence=_finite(
                    "profile_confidence",
                    row["profile_confidence"],
                ),
                gap_s=_finite("gap_s", row["gap_s"]),
                kit_family=str(
                    row.get("kit_family", "unknown_kit")
                ),
                competition_mode=selected.competition_mode,
            )
        )

    roots = sorted(
        (
            actor_id
            for actor_id in actors
            if actor_id not in predecessor_of
        ),
        key=lambda actor_id: (
            actors[actor_id]["start_time_s"],
            actor_id,
        ),
    )

    actor_paths: list[tuple[str, ...]] = []
    visited: set[str] = set()

    for root in roots:
        path: list[str] = []
        local_seen: set[str] = set()
        current = root

        while True:
            if current in local_seen:
                raise RuntimeError("cycle detected in persistent chain paths")
            if current in visited:
                raise RuntimeError("actor appears in multiple identity paths")

            local_seen.add(current)
            visited.add(current)
            path.append(current)

            next_actor = successor_of.get(current)
            if next_actor is None:
                break
            current = next_actor

        actor_paths.append(tuple(path))

    if visited != set(actors):
        missing = sorted(set(actors) - visited)
        raise RuntimeError(
            f"persistent solver lost actors: {missing[:10]}"
        )

    identities: list[PersistentIdentity] = []
    actor_assignments: list[dict[str, Any]] = []
    tracklet_assignments: list[dict[str, Any]] = []

    for index, actor_path in enumerate(actor_paths, start=1):
        identity_id = f"person-{index:04d}"
        tracklets: list[str] = []

        for actor_order, actor_id in enumerate(actor_path):
            actor_assignments.append({
                "schema_version": 1,
                "solver_version": PERSISTENT_CHAIN_SOLVER_VERSION,
                "actor_id": actor_id,
                "identity_id": identity_id,
                "identity_actor_order": actor_order,
                "provisional": True,
            })

            for tracklet_id in actors[actor_id]["tracklet_ids"]:
                tracklets.append(tracklet_id)
                tracklet_assignments.append({
                    "schema_version": 1,
                    "solver_version": PERSISTENT_CHAIN_SOLVER_VERSION,
                    "tracklet_id": tracklet_id,
                    "source_actor_id": actor_id,
                    "identity_id": identity_id,
                    "provisional": True,
                })

        identity = PersistentIdentity(
            identity_id=identity_id,
            actor_ids=actor_path,
            tracklet_ids=tuple(tracklets),
            start_time_s=min(
                actors[actor_id]["start_time_s"]
                for actor_id in actor_path
            ),
            end_time_s=max(
                actors[actor_id]["end_time_s"]
                for actor_id in actor_path
            ),
        )
        identities.append(identity)

    identity_by_actor = {
        row["actor_id"]: row["identity_id"]
        for row in actor_assignments
    }

    # Defensive full hard-constraint validation at final identity level.
    for actor_a, blocked in hard_neighbors.items():
        identity_a = identity_by_actor[actor_a]
        for actor_b in blocked:
            if identity_a == identity_by_actor[actor_b]:
                raise RuntimeError(
                    "hard cannot-link survived inside persistent identity: "
                    f"{actor_a}, {actor_b}"
                )

    return PersistentChainSolverResult(
        identities=tuple(identities),
        actor_assignments=tuple(actor_assignments),
        tracklet_assignments=tuple(tracklet_assignments),
        selected_links=tuple(selected_links),
        rejection_counts=dict(rejections),
        eligible_candidate_count=len(eligible),
        config=selected,
    )
