"""Global physical-identity partition solver for Tracking V1.

Hito 3B-B consumes:
- Hito 3A hard physical constraints (`cannot_link`);
- Hito 3B-A soft scored candidate links.

It outputs a global partition of tracklets into provisional physical identities.

The solver is deliberately NOT an edge-greedy linker.  It optimizes one global
partition objective subject to all hard cannot-link constraints and performs
multi-start best-improvement local search with both whole-cluster merges and
single-tracklet relocations/splits.

Soft evidence
-------------
3B-A gives two independent values:
- rank_score: relative support in [0, 1], *not a probability*;
- evidence_confidence: how trustworthy/complete that evidence is.

We shrink rank_score toward neutral 0.5 according to confidence:

    effective_affinity = 0.5 + confidence * (rank_score - 0.5)

and use signed utility

    utility = 2 * (effective_affinity - 0.5)

So:
- high score + high confidence -> strong positive utility;
- low score + high confidence  -> strong negative utility;
- any score + low confidence   -> weak/near-neutral utility.

No appearance or soft score can override a Hito 3A `cannot_link`.

Partition objective
-------------------
For every pair placed in the same identity, add its signed candidate utility.
Pairs in different identities add zero.  Any partition containing a hard
cannot-link inside an identity is infeasible.

This is a constrained correlation-clustering style objective.  It does not
assume or pre-specify the number of players/identities.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


SOLVER_VERSION = "global-partition-v1"
BACKBONE_SOLVER_VERSION = "global-backbone-v1"
_EPS = 1e-12


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("pair requires two different tracklets")
    return (a, b) if a < b else (b, a)


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as source:
        for line_number, raw in enumerate(source, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source_path}:{line_number} is not a JSON object"
                )
            yield payload


@dataclass(frozen=True, slots=True)
class LinkUtility:
    tracklet_a: str
    tracklet_b: str
    rank_score: float
    evidence_confidence: float
    effective_affinity: float
    utility: float
    link_kind: str

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("LinkUtility requires canonical tracklet order")
        if not (0.0 <= self.rank_score <= 1.0):
            raise ValueError("rank_score must be in [0,1]")
        if not (0.0 <= self.evidence_confidence <= 1.0):
            raise ValueError("evidence_confidence must be in [0,1]")
        if not (0.0 <= self.effective_affinity <= 1.0):
            raise ValueError("effective_affinity must be in [0,1]")
        if not (-1.0 <= self.utility <= 1.0):
            raise ValueError("utility must be in [-1,1]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "rank_score": self.rank_score,
            "evidence_confidence": self.evidence_confidence,
            "effective_affinity": self.effective_affinity,
            "utility": self.utility,
            "link_kind": self.link_kind,
        }


@dataclass(frozen=True, slots=True)
class SolverInput:
    tracklet_ids: tuple[str, ...]
    hard_cannot_links: frozenset[tuple[str, str]]
    utilities: Mapping[tuple[str, str], LinkUtility]

    def __post_init__(self) -> None:
        if len(self.tracklet_ids) != len(set(self.tracklet_ids)):
            raise ValueError("duplicate tracklet id")
        nodes = set(self.tracklet_ids)
        for pair in self.hard_cannot_links:
            if pair[0] not in nodes or pair[1] not in nodes:
                raise ValueError("cannot-link references unknown tracklet")
        for pair, utility in self.utilities.items():
            if pair[0] not in nodes or pair[1] not in nodes:
                raise ValueError("utility references unknown tracklet")
            if pair in self.hard_cannot_links:
                raise ValueError(
                    "soft utility must not exist for a hard cannot-link"
                )
            if pair != (utility.tracklet_a, utility.tracklet_b):
                raise ValueError("utility key/payload mismatch")


@dataclass(frozen=True, slots=True)
class PartitionResult:
    clusters: tuple[tuple[str, ...], ...]
    objective: float
    start_name: str
    iterations: int
    trace: tuple[Mapping[str, Any], ...]

    @property
    def identity_count(self) -> int:
        return len(self.clusters)

    def assignment(self) -> dict[str, int]:
        result: dict[str, int] = {}
        for identity_index, cluster in enumerate(self.clusters):
            for tracklet_id in cluster:
                if tracklet_id in result:
                    raise ValueError("tracklet appears in multiple clusters")
                result[tracklet_id] = identity_index
        return result


def utility_from_scored_link(row: Mapping[str, Any]) -> LinkUtility:
    if row.get("identity_decision") not in (None,):
        raise ValueError("3B-A input unexpectedly contains identity decisions")

    a, b = canonical_pair(
        str(row["tracklet_a"]),
        str(row["tracklet_b"]),
    )
    rank_score = _clamp01(
        _finite_float("rank_score", row["rank_score"])
    )
    confidence = _clamp01(
        _finite_float(
            "evidence_confidence",
            row["evidence_confidence"],
        )
    )
    effective_affinity = _clamp01(
        0.5 + confidence * (rank_score - 0.5)
    )
    utility = 2.0 * (effective_affinity - 0.5)
    utility = max(-1.0, min(1.0, utility))

    return LinkUtility(
        tracklet_a=a,
        tracklet_b=b,
        rank_score=rank_score,
        evidence_confidence=confidence,
        effective_affinity=effective_affinity,
        utility=utility,
        link_kind=str(row["link_kind"]),
    )


def build_solver_input(
    candidate_edges: Sequence[Mapping[str, Any]],
    scored_links: Sequence[Mapping[str, Any]],
) -> SolverInput:
    tracklet_ids: set[str] = set()
    hard: set[tuple[str, str]] = set()
    candidate_pairs: set[tuple[str, str]] = set()

    for edge in candidate_edges:
        pair = canonical_pair(
            str(edge["tracklet_a"]),
            str(edge["tracklet_b"]),
        )
        tracklet_ids.update(pair)
        status = edge.get("status")
        if status == "cannot_link":
            hard.add(pair)
        elif status == "candidate":
            candidate_pairs.add(pair)
        else:
            raise ValueError(
                f"unexpected Hito 3A edge status: {status!r}"
            )

    utilities: dict[tuple[str, str], LinkUtility] = {}
    for row in scored_links:
        utility = utility_from_scored_link(row)
        pair = (utility.tracklet_a, utility.tracklet_b)
        if pair in utilities:
            raise ValueError(f"duplicate scored link: {pair}")
        if pair not in candidate_pairs:
            raise ValueError(
                f"scored link {pair} is not a Hito 3A candidate"
            )
        utilities[pair] = utility

    missing = candidate_pairs - set(utilities)
    extra = set(utilities) - candidate_pairs
    if missing:
        raise ValueError(
            f"{len(missing)} Hito 3A candidates lack 3B-A scores; "
            f"examples={sorted(missing)[:5]}"
        )
    if extra:
        raise ValueError(
            f"3B-A contains unknown candidate pairs: {sorted(extra)[:5]}"
        )

    expected_pair_count = (
        len(tracklet_ids) * (len(tracklet_ids) - 1) // 2
    )
    if len(candidate_pairs) + len(hard) != expected_pair_count:
        raise ValueError(
            "Hito 3A graph does not contain every unordered tracklet pair"
        )

    return SolverInput(
        tracklet_ids=tuple(sorted(tracklet_ids)),
        hard_cannot_links=frozenset(hard),
        utilities=utilities,
    )


def pair_utility(
    solver_input: SolverInput,
    a: str,
    b: str,
) -> float:
    pair = canonical_pair(a, b)
    if pair in solver_input.hard_cannot_links:
        raise ValueError(
            "pair_utility requested for hard cannot-link"
        )
    utility = solver_input.utilities.get(pair)
    # Defensive neutral fallback; build_solver_input normally guarantees every
    # non-hard pair has a score.
    return 0.0 if utility is None else utility.utility


def clusters_compatible(
    left: Sequence[str],
    right: Sequence[str],
    hard_cannot_links: frozenset[tuple[str, str]],
) -> bool:
    for a in left:
        for b in right:
            if canonical_pair(a, b) in hard_cannot_links:
                return False
    return True


def cluster_objective(
    cluster: Sequence[str],
    solver_input: SolverInput,
) -> float:
    total = 0.0
    for a, b in itertools.combinations(cluster, 2):
        pair = canonical_pair(a, b)
        if pair in solver_input.hard_cannot_links:
            return -math.inf
        total += pair_utility(solver_input, a, b)
    return total


def partition_objective(
    clusters: Sequence[Sequence[str]],
    solver_input: SolverInput,
) -> float:
    assigned: set[str] = set()
    total = 0.0
    for cluster in clusters:
        for tracklet_id in cluster:
            if tracklet_id in assigned:
                raise ValueError(
                    f"duplicate assignment for {tracklet_id}"
                )
            assigned.add(tracklet_id)
        value = cluster_objective(cluster, solver_input)
        if not math.isfinite(value):
            return -math.inf
        total += value

    if assigned != set(solver_input.tracklet_ids):
        missing = set(solver_input.tracklet_ids) - assigned
        extra = assigned - set(solver_input.tracklet_ids)
        raise ValueError(
            f"partition node mismatch missing={missing} extra={extra}"
        )
    return total


def _canonicalize_clusters(
    clusters: Iterable[Iterable[str]],
) -> tuple[tuple[str, ...], ...]:
    normalized = [
        tuple(sorted(cluster))
        for cluster in clusters
        if tuple(cluster)
    ]
    normalized.sort(key=lambda cluster: cluster)
    return tuple(normalized)


def _merge_delta(
    left: Sequence[str],
    right: Sequence[str],
    solver_input: SolverInput,
) -> float:
    if not clusters_compatible(
        left,
        right,
        solver_input.hard_cannot_links,
    ):
        return -math.inf
    return sum(
        pair_utility(solver_input, a, b)
        for a in left
        for b in right
    )


def _move_delta(
    node: str,
    source: Sequence[str],
    target: Sequence[str],
    solver_input: SolverInput,
) -> float:
    if node not in source:
        raise ValueError("node does not belong to source")
    if target and not clusters_compatible(
        [node],
        target,
        solver_input.hard_cannot_links,
    ):
        return -math.inf

    lost = sum(
        pair_utility(solver_input, node, other)
        for other in source
        if other != node
    )
    gained = sum(
        pair_utility(solver_input, node, other)
        for other in target
    )
    return gained - lost


def _apply_merge(
    clusters: list[list[str]],
    i: int,
    j: int,
) -> None:
    if i > j:
        i, j = j, i
    clusters[i] = sorted(clusters[i] + clusters[j])
    del clusters[j]


def _apply_move(
    clusters: list[list[str]],
    source_index: int,
    target_index: int | None,
    node: str,
) -> None:
    source = clusters[source_index]
    source.remove(node)

    if target_index is None:
        clusters.append([node])
    else:
        if not source and target_index > source_index:
            target_index -= 1
        clusters[target_index].append(node)
        clusters[target_index].sort()

    if not source:
        del clusters[source_index]


def _best_improvement(
    initial_clusters: Sequence[Sequence[str]],
    solver_input: SolverInput,
    *,
    start_name: str,
    max_iterations: int = 10000,
) -> PartitionResult:
    clusters = [
        list(cluster)
        for cluster in _canonicalize_clusters(initial_clusters)
    ]
    current = partition_objective(clusters, solver_input)
    if not math.isfinite(current):
        raise ValueError(
            f"initial partition {start_name!r} is infeasible"
        )

    trace: list[dict[str, Any]] = []
    iterations = 0

    while iterations < max_iterations:
        best_delta = 0.0
        best_action: tuple[Any, ...] | None = None

        # Whole-cluster merge moves.
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                delta = _merge_delta(
                    clusters[i],
                    clusters[j],
                    solver_input,
                )
                if delta > best_delta + _EPS:
                    best_delta = delta
                    best_action = ("merge", i, j)

        # Node relocation / implicit split moves.
        for source_index, source in enumerate(clusters):
            for node in tuple(source):
                # Move to an existing cluster.
                for target_index, target in enumerate(clusters):
                    if target_index == source_index:
                        continue
                    delta = _move_delta(
                        node,
                        source,
                        target,
                        solver_input,
                    )
                    if delta > best_delta + _EPS:
                        best_delta = delta
                        best_action = (
                            "move",
                            source_index,
                            target_index,
                            node,
                        )

                # Split this node to a singleton.  Only useful when source has
                # more than one node.
                if len(source) > 1:
                    delta = _move_delta(
                        node,
                        source,
                        [],
                        solver_input,
                    )
                    if delta > best_delta + _EPS:
                        best_delta = delta
                        best_action = (
                            "move",
                            source_index,
                            None,
                            node,
                        )

        if best_action is None:
            break

        before = current
        if best_action[0] == "merge":
            _, i, j = best_action
            left = tuple(clusters[i])
            right = tuple(clusters[j])
            _apply_merge(clusters, i, j)
            action_payload = {
                "type": "merge",
                "left": left,
                "right": right,
            }
        else:
            _, source_index, target_index, node = best_action
            source_before = tuple(clusters[source_index])
            target_before = (
                tuple(clusters[target_index])
                if target_index is not None
                else ()
            )
            _apply_move(
                clusters,
                source_index,
                target_index,
                node,
            )
            action_payload = {
                "type": "move",
                "node": node,
                "source": source_before,
                "target": target_before,
            }

        current = partition_objective(clusters, solver_input)
        if current + _EPS < before:
            raise AssertionError(
                "best-improvement action reduced objective"
            )
        trace.append(
            {
                "iteration": iterations + 1,
                "delta": best_delta,
                "objective_before": before,
                "objective_after": current,
                "action": action_payload,
            }
        )
        iterations += 1

    return PartitionResult(
        clusters=_canonicalize_clusters(clusters),
        objective=current,
        start_name=start_name,
        iterations=iterations,
        trace=tuple(trace),
    )


def _can_union_groups(
    left: set[str],
    right: set[str],
    hard: frozenset[tuple[str, str]],
) -> bool:
    return clusters_compatible(
        sorted(left),
        sorted(right),
        hard,
    )


def _threshold_seed(
    solver_input: SolverInput,
    *,
    minimum_utility: float,
    name: str,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    parent = {node: node for node in solver_input.tracklet_ids}
    groups = {node: {node} for node in solver_input.tracklet_ids}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    edges = sorted(
        (
            utility
            for utility in solver_input.utilities.values()
            if utility.utility >= minimum_utility
        ),
        key=lambda utility: (
            utility.utility,
            utility.evidence_confidence,
            utility.rank_score,
            utility.tracklet_a,
            utility.tracklet_b,
        ),
        reverse=True,
    )

    for utility in edges:
        ra = find(utility.tracklet_a)
        rb = find(utility.tracklet_b)
        if ra == rb:
            continue
        if not _can_union_groups(
            groups[ra],
            groups[rb],
            solver_input.hard_cannot_links,
        ):
            continue

        # Deterministic union.
        keep, drop = sorted((ra, rb))
        parent[drop] = keep
        groups[keep] = groups[keep] | groups[drop]
        del groups[drop]

    clusters = _canonicalize_clusters(groups.values())
    return name, clusters


def _mutual_best_seed(
    solver_input: SolverInput,
) -> tuple[str, tuple[tuple[str, ...], ...]]:
    best: dict[str, tuple[float, str]] = {}
    for utility in solver_input.utilities.values():
        if utility.utility <= 0:
            continue
        for node, other in (
            (utility.tracklet_a, utility.tracklet_b),
            (utility.tracklet_b, utility.tracklet_a),
        ):
            current = best.get(node)
            candidate = (utility.utility, other)
            if current is None or candidate > current:
                best[node] = candidate

    pairs: list[tuple[float, str, str]] = []
    for node, (value, other) in best.items():
        other_best = best.get(other)
        if (
            other_best is not None
            and other_best[1] == node
            and node < other
        ):
            pairs.append((value, node, other))

    clusters: list[set[str]] = [
        {node} for node in solver_input.tracklet_ids
    ]
    location = {
        node: index
        for index, node in enumerate(solver_input.tracklet_ids)
    }

    for _, a, b in sorted(pairs, reverse=True):
        ia = location[a]
        ib = location[b]
        if ia == ib:
            continue
        left = clusters[ia]
        right = clusters[ib]
        if not left or not right:
            continue
        if not _can_union_groups(
            left,
            right,
            solver_input.hard_cannot_links,
        ):
            continue
        merged = left | right
        clusters[ia] = merged
        clusters[ib] = set()
        for node in merged:
            location[node] = ia

    return (
        "mutual-best",
        _canonicalize_clusters(
            cluster for cluster in clusters if cluster
        ),
    )


def generate_initial_partitions(
    solver_input: SolverInput,
) -> list[tuple[str, tuple[tuple[str, ...], ...]]]:
    seeds: list[
        tuple[str, tuple[tuple[str, ...], ...]]
    ] = [
        (
            "singletons",
            tuple((node,) for node in solver_input.tracklet_ids),
        ),
        _mutual_best_seed(solver_input),
    ]

    for threshold in (0.15, 0.30, 0.45, 0.60):
        seeds.append(
            _threshold_seed(
                solver_input,
                minimum_utility=threshold,
                name=f"utility>={threshold:.2f}",
            )
        )

    # De-duplicate equal partitions while keeping deterministic order.
    unique: list[
        tuple[str, tuple[tuple[str, ...], ...]]
    ] = []
    seen: set[tuple[tuple[str, ...], ...]] = set()
    for name, clusters in seeds:
        key = _canonicalize_clusters(clusters)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, key))
    return unique


def solve_global_partition(
    solver_input: SolverInput,
) -> tuple[PartitionResult, tuple[Mapping[str, Any], ...]]:
    results: list[PartitionResult] = []

    for start_name, clusters in generate_initial_partitions(
        solver_input
    ):
        result = _best_improvement(
            clusters,
            solver_input,
            start_name=start_name,
        )
        results.append(result)

    if not results:
        raise RuntimeError("solver generated no starts")

    # Highest objective wins.  On an exact objective tie prefer the more
    # conservative partition with MORE identities (fewer unsupported merges).
    best = max(
        results,
        key=lambda result: (
            result.objective,
            result.identity_count,
            tuple(result.clusters),
        ),
    )

    start_results = tuple(
        {
            "start_name": result.start_name,
            "objective": result.objective,
            "identity_count": result.identity_count,
            "iterations": result.iterations,
            "selected": result is best,
        }
        for result in results
    )
    return best, start_results


def validate_partition(
    result: PartitionResult,
    solver_input: SolverInput,
) -> dict[str, Any]:
    assigned = result.assignment()
    if set(assigned) != set(solver_input.tracklet_ids):
        raise ValueError("partition does not assign every tracklet")

    violations: list[tuple[str, str]] = []
    for a, b in solver_input.hard_cannot_links:
        if assigned[a] == assigned[b]:
            violations.append((a, b))

    recomputed = partition_objective(
        result.clusters,
        solver_input,
    )

    return {
        "assignment_count": len(assigned),
        "identity_count": result.identity_count,
        "hard_cannot_link_count": len(
            solver_input.hard_cannot_links
        ),
        "hard_constraint_violation_count": len(violations),
        "hard_constraint_violations": violations,
        "objective": result.objective,
        "recomputed_objective": recomputed,
        "objective_matches": math.isclose(
            result.objective,
            recomputed,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ),
    }


def solution_diagnostics(
    result: PartitionResult,
    solver_input: SolverInput,
) -> dict[str, Any]:
    assignment = result.assignment()

    intra: list[LinkUtility] = []
    cut_positive: list[LinkUtility] = []
    positive_total = 0.0
    positive_inside = 0.0

    for utility in solver_input.utilities.values():
        if utility.utility > 0:
            positive_total += utility.utility
        same = (
            assignment[utility.tracklet_a]
            == assignment[utility.tracklet_b]
        )
        if same:
            intra.append(utility)
            if utility.utility > 0:
                positive_inside += utility.utility
        elif utility.utility > 0:
            cut_positive.append(utility)

    sizes = [len(cluster) for cluster in result.clusters]
    singleton_count = sum(1 for size in sizes if size == 1)

    return {
        "identity_count": result.identity_count,
        "singleton_identity_count": singleton_count,
        "multi_tracklet_identity_count": (
            result.identity_count - singleton_count
        ),
        "largest_identity_size": max(sizes) if sizes else 0,
        "tracklets_per_identity": _distribution(sizes),
        "objective": result.objective,
        "positive_utility_total": positive_total,
        "positive_utility_inside": positive_inside,
        "positive_utility_capture_fraction": (
            positive_inside / positive_total
            if positive_total > 0
            else None
        ),
        "intra_identity_link_count": len(intra),
        "intra_identity_negative_link_count": sum(
            1 for utility in intra if utility.utility < 0
        ),
        "cut_positive_link_count": len(cut_positive),
    }


def _distribution(
    values: Sequence[int | float],
) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    count = len(ordered)
    if count % 2:
        median = ordered[count // 2]
    else:
        median = (
            ordered[count // 2 - 1]
            + ordered[count // 2]
        ) / 2.0
    return {
        "count": count,
        "min": ordered[0],
        "median": median,
        "max": ordered[-1],
        "mean": sum(ordered) / count,
    }


def identity_records(
    result: PartitionResult,
    solver_input: SolverInput,
    *,
    solver_version: str = SOLVER_VERSION,
) -> list[dict[str, Any]]:
    utilities = solver_input.utilities
    records: list[dict[str, Any]] = []

    # Stable physical identity labels are based on sorted cluster content, not
    # solver iteration order.
    for index, cluster in enumerate(
        sorted(result.clusters, key=lambda c: c),
        start=1,
    ):
        pair_utilities: list[float] = []
        positive_support = 0.0
        negative_cost = 0.0
        kinds: dict[str, int] = {}

        for a, b in itertools.combinations(cluster, 2):
            utility = utilities.get(canonical_pair(a, b))
            if utility is None:
                continue
            pair_utilities.append(utility.utility)
            if utility.utility > 0:
                positive_support += utility.utility
            elif utility.utility < 0:
                negative_cost += -utility.utility
            kinds[utility.link_kind] = (
                kinds.get(utility.link_kind, 0) + 1
            )

        records.append(
            {
                "schema_version": 1,
                "solver_version": solver_version,
                "identity_id": f"identity-{index:03d}",
                "tracklet_ids": list(cluster),
                "tracklet_count": len(cluster),
                "cluster_objective": cluster_objective(
                    cluster,
                    solver_input,
                ),
                "positive_support": positive_support,
                "negative_cost": negative_cost,
                "intra_pair_utility": _distribution(
                    pair_utilities
                ),
                "intra_link_kind_counts": kinds,
                "provisional": True,
            }
        )
    return records


def assignment_records(
    identities: Sequence[Mapping[str, Any]],
    *,
    solver_version: str = SOLVER_VERSION,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in identities:
        identity_id = str(identity["identity_id"])
        for tracklet_id in identity["tracklet_ids"]:
            rows.append(
                {
                    "schema_version": 1,
                    "solver_version": solver_version,
                    "tracklet_id": str(tracklet_id),
                    "identity_id": identity_id,
                    "provisional": True,
                }
            )
    rows.sort(key=lambda row: row["tracklet_id"])
    return rows


def intra_identity_link_records(
    result: PartitionResult,
    solver_input: SolverInput,
    *,
    solver_version: str = SOLVER_VERSION,
) -> list[dict[str, Any]]:
    assignment = result.assignment()
    rows = []
    for pair, utility in solver_input.utilities.items():
        if assignment[pair[0]] != assignment[pair[1]]:
            continue
        rows.append(
            {
                "schema_version": 1,
                "solver_version": solver_version,
                **utility.to_dict(),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["utility"],
            row["tracklet_a"],
            row["tracklet_b"],
        )
    )
    return rows


def cut_positive_link_records(
    result: PartitionResult,
    solver_input: SolverInput,
    *,
    solver_version: str = SOLVER_VERSION,
) -> list[dict[str, Any]]:
    assignment = result.assignment()
    rows = []
    for pair, utility in solver_input.utilities.items():
        if utility.utility <= 0:
            continue
        if assignment[pair[0]] == assignment[pair[1]]:
            continue

        rows.append(
            {
                "schema_version": 1,
                "solver_version": solver_version,
                **utility.to_dict(),
                "identity_index_a": assignment[pair[0]],
                "identity_index_b": assignment[pair[1]],
            }
        )
    rows.sort(
        key=lambda row: (
            -row["utility"],
            row["tracklet_a"],
            row["tracklet_b"],
        )
    )
    return rows


# ============================================================================
# Hito 3B-B V1: clique-guarded backbone augmentation
# ============================================================================

@dataclass(frozen=True, slots=True)
class TrackletWindow:
    tracklet_id: str
    start_time_s: float
    end_time_s: float

    def __post_init__(self) -> None:
        if not self.tracklet_id:
            raise ValueError("tracklet_id must not be empty")
        if self.start_time_s < 0:
            raise ValueError("start_time_s must be >= 0")
        if self.end_time_s < self.start_time_s:
            raise ValueError("end_time_s must be >= start_time_s")


@dataclass(frozen=True, slots=True)
class BackboneAugmentationConfig:
    """Conservative second stage on top of the clique baseline.

    A pure maximum-spanning-tree objective is intentionally NOT used here:
    one isolated false-positive link can connect two otherwise unrelated
    compatible clusters.

    Instead we:
    1. solve the existing clique partition first;
    2. inspect only cross-cluster *continuation* links;
    3. derive a strong-link threshold from the positive cross-link tail;
    4. require the link to connect the temporal tail of one baseline identity
       to the temporal head of another;
    5. require the two full baseline identities to remain hard-compatible;
    6. require reciprocal-best boundary support;
    7. perform only those strong backbone augmentations.

    This repairs trajectory-vs-clique conflicts while preserving the clique
    solver as a contamination guard.
    """

    strong_positive_quantile: float = 0.95
    strong_utility_floor: float = 0.35
    boundary_tolerance_s: float = 0.25
    require_reciprocal_best: bool = True

    def __post_init__(self) -> None:
        if not (0.5 <= self.strong_positive_quantile <= 1.0):
            raise ValueError(
                "strong_positive_quantile must be in [0.5,1]"
            )
        if not (0.0 <= self.strong_utility_floor <= 1.0):
            raise ValueError(
                "strong_utility_floor must be in [0,1]"
            )
        if self.boundary_tolerance_s < 0:
            raise ValueError(
                "boundary_tolerance_s must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class BackboneAugmentationLink:
    source_cluster_index: int
    target_cluster_index: int
    predecessor_tracklet_id: str
    successor_tracklet_id: str
    utility: float
    rank_score: float
    evidence_confidence: float
    gap_s: float
    endpoint_distance_m: float
    source_tail_offset_s: float
    target_head_offset_s: float
    whole_clique_merge_delta: float
    strong_utility_threshold: float
    reciprocal_best: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "solver_version": BACKBONE_SOLVER_VERSION,
            "source_cluster_index": self.source_cluster_index,
            "target_cluster_index": self.target_cluster_index,
            "predecessor_tracklet_id": self.predecessor_tracklet_id,
            "successor_tracklet_id": self.successor_tracklet_id,
            "utility": self.utility,
            "rank_score": self.rank_score,
            "evidence_confidence": self.evidence_confidence,
            "gap_s": self.gap_s,
            "endpoint_distance_m": self.endpoint_distance_m,
            "source_tail_offset_s": self.source_tail_offset_s,
            "target_head_offset_s": self.target_head_offset_s,
            "whole_clique_merge_delta": self.whole_clique_merge_delta,
            "strong_utility_threshold": self.strong_utility_threshold,
            "reciprocal_best": self.reciprocal_best,
        }


def _percentile(
    values: Sequence[float],
    quantile: float,
) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    alpha = position - low
    return (
        ordered[low] * (1.0 - alpha)
        + ordered[high] * alpha
    )


def load_tracklet_windows(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, TrackletWindow]:
    result: dict[str, TrackletWindow] = {}
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id in result:
            raise ValueError(
                f"duplicate tracklet window: {tracklet_id}"
            )
        result[tracklet_id] = TrackletWindow(
            tracklet_id=tracklet_id,
            start_time_s=_finite_float(
                "start_time_s", row["start_time_s"]
            ),
            end_time_s=_finite_float(
                "end_time_s", row["end_time_s"]
            ),
        )
    return result


def _candidate_edge_map(
    candidate_edges: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    result: dict[
        tuple[str, str],
        Mapping[str, Any],
    ] = {}
    for edge in candidate_edges:
        pair = canonical_pair(
            str(edge["tracklet_a"]),
            str(edge["tracklet_b"]),
        )
        if pair in result:
            raise ValueError(
                f"duplicate candidate edge: {pair}"
            )
        result[pair] = edge
    return result


def _cluster_time_bounds(
    cluster: Sequence[str],
    tracklet_windows: Mapping[str, TrackletWindow],
) -> tuple[float, float]:
    return (
        min(
            tracklet_windows[tracklet_id].start_time_s
            for tracklet_id in cluster
        ),
        max(
            tracklet_windows[tracklet_id].end_time_s
            for tracklet_id in cluster
        ),
    )


def _cross_clique_delta(
    left: Sequence[str],
    right: Sequence[str],
    solver_input: SolverInput,
) -> float:
    if not clusters_compatible(
        left,
        right,
        solver_input.hard_cannot_links,
    ):
        return -math.inf
    return sum(
        pair_utility(
            solver_input,
            tracklet_a,
            tracklet_b,
        )
        for tracklet_a in left
        for tracklet_b in right
    )


def _cross_identity_continuation_rows(
    *,
    baseline: PartitionResult,
    solver_input: SolverInput,
    candidate_edges: Sequence[Mapping[str, Any]],
    tracklet_windows: Mapping[str, TrackletWindow],
) -> list[dict[str, Any]]:
    assignment = baseline.assignment()
    edge_map = _candidate_edge_map(candidate_edges)
    cluster_bounds = {
        index: _cluster_time_bounds(
            cluster,
            tracklet_windows,
        )
        for index, cluster in enumerate(
            baseline.clusters
        )
    }

    rows: list[dict[str, Any]] = []

    for pair, utility in solver_input.utilities.items():
        if utility.link_kind != "continuation":
            continue
        if utility.utility <= 0:
            continue

        cluster_a = assignment[pair[0]]
        cluster_b = assignment[pair[1]]
        if cluster_a == cluster_b:
            continue

        edge = edge_map.get(pair)
        if edge is None:
            continue
        if edge.get("status") != "candidate":
            continue
        if edge.get("reason") != "transition_reachable":
            continue

        physics = edge.get("physics")
        if not isinstance(physics, Mapping):
            raise ValueError(
                f"continuation edge {pair} lacks physics"
            )

        predecessor = str(physics["predecessor"])
        successor = str(physics["successor"])
        source_cluster = assignment[predecessor]
        target_cluster = assignment[successor]
        if source_cluster == target_cluster:
            continue

        source_members = baseline.clusters[
            source_cluster
        ]
        target_members = baseline.clusters[
            target_cluster
        ]
        feasible = clusters_compatible(
            source_members,
            target_members,
            solver_input.hard_cannot_links,
        )

        source_end = cluster_bounds[
            source_cluster
        ][1]
        target_start = cluster_bounds[
            target_cluster
        ][0]
        source_tail_offset_s = max(
            0.0,
            source_end
            - tracklet_windows[
                predecessor
            ].end_time_s,
        )
        target_head_offset_s = max(
            0.0,
            tracklet_windows[
                successor
            ].start_time_s
            - target_start,
        )

        rows.append(
            {
                "source_cluster_index": source_cluster,
                "target_cluster_index": target_cluster,
                "predecessor_tracklet_id": predecessor,
                "successor_tracklet_id": successor,
                "utility": utility.utility,
                "rank_score": utility.rank_score,
                "evidence_confidence": (
                    utility.evidence_confidence
                ),
                "gap_s": _finite_float(
                    "gap_s", physics["gap_s"]
                ),
                "endpoint_distance_m": _finite_float(
                    "endpoint_distance_m",
                    physics["endpoint_distance_m"],
                ),
                "source_tail_offset_s": (
                    source_tail_offset_s
                ),
                "target_head_offset_s": (
                    target_head_offset_s
                ),
                "boundary_aligned": False,
                "whole_cluster_hard_compatible": (
                    feasible
                ),
                "whole_clique_merge_delta": (
                    _cross_clique_delta(
                        source_members,
                        target_members,
                        solver_input,
                    )
                    if feasible
                    else None
                ),
            }
        )

    return rows


def select_backbone_augmentation_links(
    *,
    baseline: PartitionResult,
    solver_input: SolverInput,
    candidate_edges: Sequence[Mapping[str, Any]],
    tracklet_windows: Mapping[str, TrackletWindow],
    config: BackboneAugmentationConfig,
) -> tuple[
    tuple[BackboneAugmentationLink, ...],
    Mapping[str, Any],
]:
    rows = _cross_identity_continuation_rows(
        baseline=baseline,
        solver_input=solver_input,
        candidate_edges=candidate_edges,
        tracklet_windows=tracklet_windows,
    )

    positive_utilities = [
        float(row["utility"])
        for row in rows
        if row["utility"] > 0
    ]
    quantile_threshold = _percentile(
        positive_utilities,
        config.strong_positive_quantile,
    )
    strong_threshold = max(
        config.strong_utility_floor,
        (
            quantile_threshold
            if quantile_threshold is not None
            else config.strong_utility_floor
        ),
    )

    eligible: list[dict[str, Any]] = []
    for row in rows:
        boundary_aligned = (
            row["source_tail_offset_s"]
            <= config.boundary_tolerance_s + _EPS
            and row["target_head_offset_s"]
            <= config.boundary_tolerance_s + _EPS
        )
        row["boundary_aligned"] = boundary_aligned

        if row["utility"] < strong_threshold - _EPS:
            continue
        if not boundary_aligned:
            continue
        if not row["whole_cluster_hard_compatible"]:
            continue

        # The baseline clique solver is already locally stable to whole-cluster
        # merges.  We intentionally target the disagreement case where the
        # local trajectory handoff is strong while the all-pairs clique merge
        # is neutral/negative.
        clique_delta = row["whole_clique_merge_delta"]
        if clique_delta is None:
            continue
        if clique_delta > _EPS:
            continue

        eligible.append(row)

    best_outgoing: dict[int, dict[str, Any]] = {}
    best_incoming: dict[int, dict[str, Any]] = {}

    def better(
        candidate: Mapping[str, Any],
        incumbent: Mapping[str, Any] | None,
    ) -> bool:
        if incumbent is None:
            return True
        candidate_key = (
            candidate["utility"],
            candidate["evidence_confidence"],
            candidate["rank_score"],
            -candidate["gap_s"],
            candidate["predecessor_tracklet_id"],
            candidate["successor_tracklet_id"],
        )
        incumbent_key = (
            incumbent["utility"],
            incumbent["evidence_confidence"],
            incumbent["rank_score"],
            -incumbent["gap_s"],
            incumbent["predecessor_tracklet_id"],
            incumbent["successor_tracklet_id"],
        )
        return candidate_key > incumbent_key

    for row in eligible:
        source = int(row["source_cluster_index"])
        target = int(row["target_cluster_index"])
        if better(row, best_outgoing.get(source)):
            best_outgoing[source] = row
        if better(row, best_incoming.get(target)):
            best_incoming[target] = row

    reciprocal: list[dict[str, Any]] = []
    for row in eligible:
        source = int(row["source_cluster_index"])
        target = int(row["target_cluster_index"])
        reciprocal_best = (
            best_outgoing.get(source) is row
            and best_incoming.get(target) is row
        )
        row["reciprocal_best"] = reciprocal_best
        if (
            config.require_reciprocal_best
            and not reciprocal_best
        ):
            continue
        reciprocal.append(row)

    reciprocal.sort(
        key=lambda row: (
            row["utility"],
            row["evidence_confidence"],
            row["rank_score"],
            -row["gap_s"],
            -row["source_cluster_index"],
            -row["target_cluster_index"],
        ),
        reverse=True,
    )

    parent = {
        index: index
        for index in range(len(baseline.clusters))
    }
    groups = {
        index: set(cluster)
        for index, cluster in enumerate(
            baseline.clusters
        )
    }
    incoming_used: set[int] = set()
    outgoing_used: set[int] = set()
    selected: list[BackboneAugmentationLink] = []

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for row in reciprocal:
        source = int(row["source_cluster_index"])
        target = int(row["target_cluster_index"])

        # Keep the augmentation path-like and conservative: at most one
        # outgoing bridge from a baseline cluster and at most one incoming.
        if source in outgoing_used or target in incoming_used:
            continue

        source_root = find(source)
        target_root = find(target)
        if source_root == target_root:
            continue

        if not clusters_compatible(
            sorted(groups[source_root]),
            sorted(groups[target_root]),
            solver_input.hard_cannot_links,
        ):
            continue

        keep, drop = sorted(
            (source_root, target_root)
        )
        parent[drop] = keep
        groups[keep] = (
            groups[source_root]
            | groups[target_root]
        )
        if drop != keep:
            groups.pop(drop, None)
        if source_root != keep:
            groups.pop(source_root, None)
        if target_root != keep:
            groups.pop(target_root, None)

        outgoing_used.add(source)
        incoming_used.add(target)

        selected.append(
            BackboneAugmentationLink(
                source_cluster_index=source,
                target_cluster_index=target,
                predecessor_tracklet_id=str(
                    row["predecessor_tracklet_id"]
                ),
                successor_tracklet_id=str(
                    row["successor_tracklet_id"]
                ),
                utility=float(row["utility"]),
                rank_score=float(row["rank_score"]),
                evidence_confidence=float(
                    row["evidence_confidence"]
                ),
                gap_s=float(row["gap_s"]),
                endpoint_distance_m=float(
                    row["endpoint_distance_m"]
                ),
                source_tail_offset_s=float(
                    row["source_tail_offset_s"]
                ),
                target_head_offset_s=float(
                    row["target_head_offset_s"]
                ),
                whole_clique_merge_delta=float(
                    row["whole_clique_merge_delta"]
                ),
                strong_utility_threshold=float(
                    strong_threshold
                ),
                reciprocal_best=bool(
                    row["reciprocal_best"]
                ),
            )
        )

    diagnostics = {
        "cross_identity_positive_continuation_count": (
            len(rows)
        ),
        "strong_utility_threshold": (
            strong_threshold
        ),
        "strong_candidate_count": sum(
            row["utility"] >= strong_threshold - _EPS
            for row in rows
        ),
        "eligible_boundary_conflict_count": len(
            eligible
        ),
        "reciprocal_eligible_count": len(
            reciprocal
        ),
        "selected_backbone_link_count": len(
            selected
        ),
    }
    return tuple(selected), diagnostics


def apply_backbone_augmentations(
    baseline: PartitionResult,
    selected_links: Sequence[BackboneAugmentationLink],
    solver_input: SolverInput,
) -> PartitionResult:
    parent = {
        index: index
        for index in range(len(baseline.clusters))
    }
    groups = {
        index: set(cluster)
        for index, cluster in enumerate(
            baseline.clusters
        )
    }

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    trace: list[dict[str, Any]] = []

    for iteration, link in enumerate(
        selected_links,
        start=1,
    ):
        source_root = find(
            link.source_cluster_index
        )
        target_root = find(
            link.target_cluster_index
        )
        if source_root == target_root:
            continue

        if not clusters_compatible(
            sorted(groups[source_root]),
            sorted(groups[target_root]),
            solver_input.hard_cannot_links,
        ):
            raise AssertionError(
                "selected backbone augmentation became hard-incompatible"
            )

        left_before = tuple(
            sorted(groups[source_root])
        )
        right_before = tuple(
            sorted(groups[target_root])
        )

        keep, drop = sorted(
            (source_root, target_root)
        )
        merged = (
            groups[source_root]
            | groups[target_root]
        )
        parent[drop] = keep
        groups[keep] = merged
        if drop != keep:
            groups.pop(drop, None)
        if source_root != keep:
            groups.pop(source_root, None)
        if target_root != keep:
            groups.pop(target_root, None)

        trace.append(
            {
                "iteration": iteration,
                "type": "backbone_merge",
                "predecessor_tracklet_id": (
                    link.predecessor_tracklet_id
                ),
                "successor_tracklet_id": (
                    link.successor_tracklet_id
                ),
                "utility": link.utility,
                "left_cluster_before": (
                    left_before
                ),
                "right_cluster_before": (
                    right_before
                ),
                "merged_cluster_after": tuple(
                    sorted(merged)
                ),
            }
        )

    final_clusters = _canonicalize_clusters(
        groups.values()
    )
    final_clique_objective = partition_objective(
        final_clusters,
        solver_input,
    )

    return PartitionResult(
        clusters=final_clusters,
        objective=final_clique_objective,
        start_name="clique-guarded-backbone",
        iterations=len(trace),
        trace=tuple(trace),
    )


def solve_global_backbone_partition(
    solver_input: SolverInput,
    *,
    candidate_edges: Sequence[Mapping[str, Any]],
    tracklet_windows: Mapping[str, TrackletWindow],
    config: BackboneAugmentationConfig | None = None,
) -> tuple[
    PartitionResult,
    PartitionResult,
    tuple[BackboneAugmentationLink, ...],
    Mapping[str, Any],
    tuple[Mapping[str, Any], ...],
]:
    """Solve clique baseline, then conservatively repair strong fragmentation.

    Returns:
        final_result,
        baseline_result,
        selected_backbone_links,
        backbone_diagnostics,
        baseline_start_results
    """
    if config is None:
        config = BackboneAugmentationConfig()

    if set(tracklet_windows) != set(
        solver_input.tracklet_ids
    ):
        raise ValueError(
            "tracklet windows must cover exactly solver tracklets"
        )

    baseline, start_results = (
        solve_global_partition(
            solver_input
        )
    )

    selected, diagnostics = (
        select_backbone_augmentation_links(
            baseline=baseline,
            solver_input=solver_input,
            candidate_edges=candidate_edges,
            tracklet_windows=tracklet_windows,
            config=config,
        )
    )
    final_result = apply_backbone_augmentations(
        baseline,
        selected,
        solver_input,
    )

    return (
        final_result,
        baseline,
        selected,
        diagnostics,
        start_results,
    )


def backbone_solution_diagnostics(
    *,
    final_result: PartitionResult,
    baseline_result: PartitionResult,
    selected_links: Sequence[BackboneAugmentationLink],
    solver_input: SolverInput,
    backbone_diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_diag = solution_diagnostics(
        baseline_result,
        solver_input,
    )
    final_diag = solution_diagnostics(
        final_result,
        solver_input,
    )

    selected_utility_total = sum(
        link.utility
        for link in selected_links
    )

    return {
        "objective_kind": (
            "clique_guarded_backbone_augmentation"
        ),
        "baseline_solver_version": SOLVER_VERSION,
        "baseline_identity_count": (
            baseline_result.identity_count
        ),
        "final_identity_count": (
            final_result.identity_count
        ),
        "identity_count_delta": (
            final_result.identity_count
            - baseline_result.identity_count
        ),
        "baseline_singleton_identity_count": (
            baseline_diag[
                "singleton_identity_count"
            ]
        ),
        "final_singleton_identity_count": (
            final_diag[
                "singleton_identity_count"
            ]
        ),
        "baseline_clique_objective": (
            baseline_result.objective
        ),
        "final_clique_objective": (
            final_result.objective
        ),
        "selected_backbone_utility_total": (
            selected_utility_total
        ),
        "layered_backbone_objective": (
            baseline_result.objective
            + selected_utility_total
        ),
        "baseline_positive_utility_capture_fraction": (
            baseline_diag[
                "positive_utility_capture_fraction"
            ]
        ),
        "final_positive_utility_capture_fraction": (
            final_diag[
                "positive_utility_capture_fraction"
            ]
        ),
        **dict(backbone_diagnostics),
    }


def validate_backbone_partition(
    *,
    final_result: PartitionResult,
    baseline_result: PartitionResult,
    selected_links: Sequence[BackboneAugmentationLink],
    solver_input: SolverInput,
) -> dict[str, Any]:
    final_validation = validate_partition(
        final_result,
        solver_input,
    )
    baseline_assignment = (
        baseline_result.assignment()
    )
    final_assignment = final_result.assignment()

    baseline_cluster_preserved = True
    for cluster in baseline_result.clusters:
        final_ids = {
            final_assignment[node]
            for node in cluster
        }
        if len(final_ids) != 1:
            baseline_cluster_preserved = False
            break

    selected_link_violations = []
    for link in selected_links:
        if (
            final_assignment[
                link.predecessor_tracklet_id
            ]
            != final_assignment[
                link.successor_tracklet_id
            ]
        ):
            selected_link_violations.append(
                (
                    link.predecessor_tracklet_id,
                    link.successor_tracklet_id,
                )
            )

    return {
        **final_validation,
        "baseline_cluster_preserved": (
            baseline_cluster_preserved
        ),
        "selected_backbone_link_count": len(
            selected_links
        ),
        "selected_backbone_link_violation_count": (
            len(selected_link_violations)
        ),
        "selected_backbone_link_violations": (
            selected_link_violations
        ),
        "baseline_assignment_count": len(
            baseline_assignment
        ),
    }
