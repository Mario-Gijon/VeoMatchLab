"""Tracking V1 Hito 7E.2: conservative temporal actor backbone.

This stage converts strong *local sequential* 7E.1 links into disjoint temporal
paths.  It deliberately does NOT perform long-gap appearance re-identification.

Why only local continuations here?
----------------------------------
7E.1 showed that high-confidence appearance rescues can span many seconds and
that some high-scoring appearance pairs overlap in time.  Those hypotheses are
better evaluated later between richer actor chains, not used to seed identity
contamination from isolated tracklets.

The backbone therefore accepts only links that:
* are ``local_continuation``;
* are sequential in time;
* pass a configurable evidence-score floor;
* are within each endpoint's top-N competing hypotheses;
* preserve max one predecessor / one successor per tracklet;
* never introduce any hard cannot-link anywhere inside the merged component.

The output is a set of provisional paths.  Singleton tracklets are preserved.
No population count is imposed and fragmentation is preferred over identity
contamination.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tracking.tracklets.models import Tracklet


TEMPORAL_BACKBONE_VERSION = "temporal-actor-backbone-v1"


def _finite(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


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
        raise ValueError("pair requires two distinct tracklets")
    return (a, b) if a < b else (b, a)


@dataclass(frozen=True, slots=True)
class TemporalBackboneConfig:
    min_rank_score: float = 0.70
    max_endpoint_rank: int = 3
    time_epsilon_s: float = 1e-9

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_rank_score <= 1.0:
            raise ValueError("min_rank_score must be in [0,1]")
        if self.max_endpoint_rank < 1:
            raise ValueError("max_endpoint_rank must be >= 1")
        if self.time_epsilon_s < 0.0:
            raise ValueError("time_epsilon_s must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TrackletWindow:
    tracklet_id: str
    start_time_s: float
    end_time_s: float
    observation_count: int
    mean_confidence: float

    @classmethod
    def from_tracklet(cls, item: Tracklet) -> "TrackletWindow":
        return cls(
            tracklet_id=item.tracklet_id,
            start_time_s=float(item.start_timestamp_s),
            end_time_s=float(item.end_timestamp_s),
            observation_count=int(item.observation_count),
            mean_confidence=float(item.mean_confidence),
        )


@dataclass(frozen=True, slots=True)
class DirectedBackboneEdge:
    predecessor_tracklet_id: str
    successor_tracklet_id: str
    rank_score: float
    evidence_confidence: float
    endpoint_rank_predecessor: int
    endpoint_rank_successor: int
    gap_s: float
    source_pair: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backbone_version": TEMPORAL_BACKBONE_VERSION,
            "predecessor_tracklet_id": self.predecessor_tracklet_id,
            "successor_tracklet_id": self.successor_tracklet_id,
            "rank_score": self.rank_score,
            "evidence_confidence": self.evidence_confidence,
            "endpoint_rank_predecessor": self.endpoint_rank_predecessor,
            "endpoint_rank_successor": self.endpoint_rank_successor,
            "gap_s": self.gap_s,
            "source_pair": list(self.source_pair),
            "identity_decision": "provisional_backbone_link",
        }


@dataclass(frozen=True, slots=True)
class TemporalActorPath:
    actor_id: str
    tracklet_ids: tuple[str, ...]
    start_time_s: float
    end_time_s: float
    observation_count: int
    mean_tracklet_confidence: float

    @property
    def tracklet_count(self) -> int:
        return len(self.tracklet_ids)

    @property
    def duration_s(self) -> float:
        return self.end_time_s - self.start_time_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "backbone_version": TEMPORAL_BACKBONE_VERSION,
            "actor_id": self.actor_id,
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": self.tracklet_count,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.duration_s,
            "observation_count": self.observation_count,
            "mean_tracklet_confidence": self.mean_tracklet_confidence,
            "provisional": True,
        }


@dataclass(frozen=True, slots=True)
class TemporalBackboneResult:
    paths: tuple[TemporalActorPath, ...]
    assignments: tuple[Mapping[str, Any], ...]
    accepted_edges: tuple[DirectedBackboneEdge, ...]
    rejection_counts: Mapping[str, int]
    eligible_edge_count: int
    config: TemporalBackboneConfig

    def summary(self) -> dict[str, Any]:
        sizes = [path.tracklet_count for path in self.paths]
        non_singleton = [value for value in sizes if value > 1]
        accepted_scores = [
            edge.rank_score for edge in self.accepted_edges
        ]
        accepted_gaps = [
            edge.gap_s for edge in self.accepted_edges
        ]
        return {
            "schema_version": 1,
            "backbone_version": TEMPORAL_BACKBONE_VERSION,
            "path_count": len(self.paths),
            "non_singleton_path_count": len(non_singleton),
            "singleton_path_count": sum(value == 1 for value in sizes),
            "largest_path_tracklet_count": max(sizes) if sizes else 0,
            "path_tracklet_count": _stats(
                [float(value) for value in sizes]
            ),
            "accepted_edge_count": len(self.accepted_edges),
            "eligible_edge_count": self.eligible_edge_count,
            "accepted_score": _stats(accepted_scores),
            "accepted_gap_s": _stats(accepted_gaps),
            "rejection_counts": dict(
                sorted(self.rejection_counts.items())
            ),
            "config": self.config.to_dict(),
            "population_forced": False,
            "appearance_rescue_used": False,
            "close_overlap_used": False,
            "hard_constraints_inviolable": True,
            "provisional": True,
        }


class _DisjointComponents:
    def __init__(
        self,
        tracklet_ids: Iterable[str],
        hard_by_node: Mapping[str, set[str]],
    ) -> None:
        ids = tuple(tracklet_ids)
        self.parent = {item: item for item in ids}
        self.members = {item: {item} for item in ids}
        self.hard_by_node = hard_by_node

    def find(self, item: str) -> str:
        parent = self.parent[item]
        if parent != item:
            self.parent[item] = self.find(parent)
        return self.parent[item]

    def compatible(self, first: str, second: str) -> bool:
        left = self.members[self.find(first)]
        right = self.members[self.find(second)]
        if len(left) > len(right):
            left, right = right, left

        for node in left:
            blocked = self.hard_by_node.get(node, set())
            if blocked.intersection(right):
                return False
        return True

    def union(self, first: str, second: str) -> bool:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return False

        if len(self.members[root_a]) < len(self.members[root_b]):
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.members[root_a].update(self.members[root_b])
        del self.members[root_b]
        return True


def _orient_sequential(
    a: TrackletWindow,
    b: TrackletWindow,
    *,
    epsilon: float,
) -> tuple[str, str, float] | None:
    if a.end_time_s <= b.start_time_s + epsilon:
        return (
            a.tracklet_id,
            b.tracklet_id,
            max(0.0, b.start_time_s - a.end_time_s),
        )
    if b.end_time_s <= a.start_time_s + epsilon:
        return (
            b.tracklet_id,
            a.tracklet_id,
            max(0.0, a.start_time_s - b.end_time_s),
        )
    return None


def _endpoint_rank(
    row: Mapping[str, Any],
    endpoint: str,
) -> int | None:
    if endpoint == str(row["tracklet_a"]):
        value = row.get("endpoint_rank_a")
    elif endpoint == str(row["tracklet_b"]):
        value = row.get("endpoint_rank_b")
    else:
        raise ValueError("endpoint is not part of link")
    return int(value) if value is not None else None


def solve_temporal_backbone(
    *,
    tracklets: Sequence[Tracklet],
    scored_links: Sequence[Mapping[str, Any]],
    hard_conflicts: Sequence[Mapping[str, Any]],
    config: TemporalBackboneConfig | None = None,
) -> TemporalBackboneResult:
    selected = config or TemporalBackboneConfig()

    windows = {
        tracklet.tracklet_id: TrackletWindow.from_tracklet(tracklet)
        for tracklet in tracklets
    }
    if len(windows) != len(tracklets):
        raise ValueError("duplicate tracklet_id")

    hard_by_node: dict[str, set[str]] = defaultdict(set)
    for row in hard_conflicts:
        a = str(row["tracklet_a"])
        b = str(row["tracklet_b"])
        if a not in windows or b not in windows:
            raise ValueError("hard conflict references unknown tracklet")
        hard_by_node[a].add(b)
        hard_by_node[b].add(a)

    rejections = Counter()
    eligible: list[DirectedBackboneEdge] = []

    for row in scored_links:
        a_id = str(row["tracklet_a"])
        b_id = str(row["tracklet_b"])
        if a_id not in windows or b_id not in windows:
            raise ValueError("scored link references unknown tracklet")

        if row.get("link_family") != "local_continuation":
            rejections["non_local_family"] += 1
            continue

        score = _finite("rank_score", row["rank_score"])
        if score < selected.min_rank_score:
            rejections["score_below_floor"] += 1
            continue

        oriented = _orient_sequential(
            windows[a_id],
            windows[b_id],
            epsilon=selected.time_epsilon_s,
        )
        if oriented is None:
            rejections["not_sequential"] += 1
            continue

        predecessor, successor, gap_s = oriented
        predecessor_rank = _endpoint_rank(row, predecessor)
        successor_rank = _endpoint_rank(row, successor)

        if predecessor_rank is None or successor_rank is None:
            rejections["missing_endpoint_rank"] += 1
            continue
        if (
            predecessor_rank > selected.max_endpoint_rank
            or successor_rank > selected.max_endpoint_rank
        ):
            rejections["endpoint_rank_above_limit"] += 1
            continue

        eligible.append(
            DirectedBackboneEdge(
                predecessor_tracklet_id=predecessor,
                successor_tracklet_id=successor,
                rank_score=score,
                evidence_confidence=_finite(
                    "evidence_confidence",
                    row["evidence_confidence"],
                ),
                endpoint_rank_predecessor=predecessor_rank,
                endpoint_rank_successor=successor_rank,
                gap_s=gap_s,
                source_pair=_canonical_pair(a_id, b_id),
            )
        )

    eligible.sort(
        key=lambda edge: (
            -edge.rank_score,
            -edge.evidence_confidence,
            edge.gap_s,
            edge.predecessor_tracklet_id,
            edge.successor_tracklet_id,
        )
    )

    successor_of: dict[str, str] = {}
    predecessor_of: dict[str, str] = {}
    components = _DisjointComponents(
        windows,
        hard_by_node,
    )

    accepted: list[DirectedBackboneEdge] = []

    for edge in eligible:
        predecessor = edge.predecessor_tracklet_id
        successor = edge.successor_tracklet_id

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

        components.union(predecessor, successor)
        successor_of[predecessor] = successor
        predecessor_of[successor] = predecessor
        accepted.append(edge)

    # Every selected directed edge must form a DAG path.  Since orientation is
    # chronological, a cycle would indicate invalid input or a logic bug.
    roots = sorted(
        tracklet_id
        for tracklet_id in windows
        if tracklet_id not in predecessor_of
    )

    raw_paths: list[tuple[str, ...]] = []
    visited: set[str] = set()

    for root in roots:
        path: list[str] = []
        current = root
        local_seen: set[str] = set()

        while True:
            if current in local_seen:
                raise RuntimeError("cycle detected in temporal backbone")
            local_seen.add(current)

            if current in visited:
                raise RuntimeError(
                    "tracklet appears in multiple backbone paths"
                )

            path.append(current)
            visited.add(current)

            next_id = successor_of.get(current)
            if next_id is None:
                break
            current = next_id

        raw_paths.append(tuple(path))

    if visited != set(windows):
        missing = sorted(set(windows) - visited)
        raise RuntimeError(
            f"backbone lost tracklets: {missing[:10]}"
        )

    # Stable IDs follow chronological start then lexical tracklet IDs, not
    # component size, so reruns remain deterministic when only later paths grow.
    raw_paths.sort(
        key=lambda path: (
            windows[path[0]].start_time_s,
            path[0],
        )
    )

    paths: list[TemporalActorPath] = []
    assignments: list[dict[str, Any]] = []

    for actor_index, tracklet_path in enumerate(raw_paths, start=1):
        actor_id = f"actor-{actor_index:04d}"
        path_windows = [
            windows[tracklet_id]
            for tracklet_id in tracklet_path
        ]
        actor = TemporalActorPath(
            actor_id=actor_id,
            tracklet_ids=tracklet_path,
            start_time_s=min(
                item.start_time_s for item in path_windows
            ),
            end_time_s=max(
                item.end_time_s for item in path_windows
            ),
            observation_count=sum(
                item.observation_count for item in path_windows
            ),
            mean_tracklet_confidence=float(
                np.mean(
                    [
                        item.mean_confidence
                        for item in path_windows
                    ]
                )
            ),
        )
        paths.append(actor)

        for order, tracklet_id in enumerate(tracklet_path):
            assignments.append(
                {
                    "schema_version": 1,
                    "backbone_version": TEMPORAL_BACKBONE_VERSION,
                    "tracklet_id": tracklet_id,
                    "actor_id": actor_id,
                    "path_order": order,
                    "provisional": True,
                }
            )

    # Defensive validation: accepted path components may never contain a hard
    # cannot-link, even transitively.
    actor_by_tracklet = {
        row["tracklet_id"]: row["actor_id"]
        for row in assignments
    }
    for a, blocked in hard_by_node.items():
        actor_a = actor_by_tracklet[a]
        for b in blocked:
            if actor_a == actor_by_tracklet[b]:
                raise RuntimeError(
                    f"hard conflict inside provisional actor: {a}, {b}"
                )

    return TemporalBackboneResult(
        paths=tuple(paths),
        assignments=tuple(assignments),
        accepted_edges=tuple(accepted),
        rejection_counts=dict(rejections),
        eligible_edge_count=len(eligible),
        config=selected,
    )
