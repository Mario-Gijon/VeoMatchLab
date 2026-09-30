"""Fragmentation audit for provisional Tracking V1 identities.

Hito 3E asks the opposite question from the physical-contamination audits:

    "Did we leave one real player split across multiple identities?"

It does NOT merge identities and does NOT modify the solver.  It inspects
cross-identity continuation evidence, with special attention to boundary
handoffs and singleton identities.

A key diagnostic is the "trajectory-vs-clique conflict":

- a very strong continuation connects the temporal tail of identity A to the
  temporal head of identity B;
- no Hito 3A hard constraint forbids merging the two identities;
- but the complete all-pairs/clique merge delta is <= 0 because other distant
  cross-pairs contribute negative soft evidence.

That pattern can indicate that a path/trajectory objective would join a player
that a clique objective keeps fragmented.  It is a REVIEW SIGNAL, never an
automatic merge decision.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


FRAGMENTATION_AUDIT_VERSION = "fragmentation-backbone-audit-v1"


def canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("pair requires two distinct tracklets")
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


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * quantile
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ordered[lo]
    alpha = pos - lo
    return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "min": min(numeric),
        "p50": _percentile(numeric, 0.50),
        "p90": _percentile(numeric, 0.90),
        "p95": _percentile(numeric, 0.95),
        "max": max(numeric),
        "mean": sum(numeric) / len(numeric),
    }


@dataclass(frozen=True, slots=True)
class FragmentationAuditConfig:
    strong_positive_quantile: float = 0.95
    strong_utility_floor: float = 0.35
    boundary_tolerance_s: float = 0.25
    long_singleton_span_fraction: float = 0.80
    short_identity_span_fraction: float = 0.50
    strong_utility_threshold_override: float | None = None

    def __post_init__(self) -> None:
        if not (0.5 <= self.strong_positive_quantile <= 1.0):
            raise ValueError(
                "strong_positive_quantile must be in [0.5,1]"
            )
        if not (0.0 <= self.strong_utility_floor <= 1.0):
            raise ValueError("strong_utility_floor must be in [0,1]")
        if self.boundary_tolerance_s < 0:
            raise ValueError("boundary_tolerance_s must be >= 0")
        if not (0.0 < self.long_singleton_span_fraction <= 1.0):
            raise ValueError(
                "long_singleton_span_fraction must be in (0,1]"
            )
        if not (0.0 < self.short_identity_span_fraction < 1.0):
            raise ValueError(
                "short_identity_span_fraction must be in (0,1)"
            )
        if (
            self.strong_utility_threshold_override is not None
            and not (
                0.0 <= self.strong_utility_threshold_override <= 1.0
            )
        ):
            raise ValueError(
                "strong_utility_threshold_override must be in [0,1]"
            )


@dataclass(frozen=True, slots=True)
class TrackletMeta:
    tracklet_id: str
    start_time_s: float
    end_time_s: float
    observation_count: int

    @property
    def span_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


@dataclass(frozen=True, slots=True)
class IdentityMeta:
    identity_id: str
    tracklet_ids: tuple[str, ...]
    start_time_s: float
    end_time_s: float
    observation_count: int
    union_coverage_s: float

    @property
    def span_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)

    @property
    def tracklet_count(self) -> int:
        return len(self.tracklet_ids)


def load_tracklet_meta(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, TrackletMeta]:
    result: dict[str, TrackletMeta] = {}

    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        observations = row.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(f"{tracklet_id} has no observations")

        times = [
            _finite_float("timestamp_s", obs["timestamp_s"])
            for obs in observations
        ]
        if times != sorted(times):
            raise ValueError(
                f"{tracklet_id} timestamps are not sorted"
            )
        if tracklet_id in result:
            raise ValueError(f"duplicate tracklet: {tracklet_id}")

        result[tracklet_id] = TrackletMeta(
            tracklet_id=tracklet_id,
            start_time_s=times[0],
            end_time_s=times[-1],
            observation_count=len(observations),
        )

    return result


def _union_coverage_s(
    tracklet_ids: Sequence[str],
    tracklets: Mapping[str, TrackletMeta],
) -> float:
    intervals = sorted(
        (
            tracklets[tracklet_id].start_time_s,
            tracklets[tracklet_id].end_time_s,
        )
        for tracklet_id in tracklet_ids
    )
    merged: list[list[float]] = []

    for start, end in intervals:
        if not merged or start > merged[-1][1] + 1e-12:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    return sum(end - start for start, end in merged)


def load_identity_meta(
    rows: Iterable[Mapping[str, Any]],
    tracklets: Mapping[str, TrackletMeta],
) -> dict[str, IdentityMeta]:
    result: dict[str, IdentityMeta] = {}
    seen_tracklets: set[str] = set()

    for row in rows:
        identity_id = str(row["identity_id"])
        tracklet_ids = tuple(
            sorted(str(value) for value in row["tracklet_ids"])
        )
        if not tracklet_ids:
            raise ValueError(f"{identity_id} has no tracklets")
        if identity_id in result:
            raise ValueError(f"duplicate identity: {identity_id}")

        for tracklet_id in tracklet_ids:
            if tracklet_id not in tracklets:
                raise ValueError(
                    f"{identity_id} references unknown {tracklet_id}"
                )
            if tracklet_id in seen_tracklets:
                raise ValueError(
                    f"{tracklet_id} assigned to multiple identities"
                )
            seen_tracklets.add(tracklet_id)

        result[identity_id] = IdentityMeta(
            identity_id=identity_id,
            tracklet_ids=tracklet_ids,
            start_time_s=min(
                tracklets[tracklet_id].start_time_s
                for tracklet_id in tracklet_ids
            ),
            end_time_s=max(
                tracklets[tracklet_id].end_time_s
                for tracklet_id in tracklet_ids
            ),
            observation_count=sum(
                tracklets[tracklet_id].observation_count
                for tracklet_id in tracklet_ids
            ),
            union_coverage_s=_union_coverage_s(
                tracklet_ids,
                tracklets,
            ),
        )

    if seen_tracklets != set(tracklets):
        missing = sorted(set(tracklets) - seen_tracklets)
        raise ValueError(
            f"identity partition misses tracklets: {missing[:10]}"
        )

    return result


def load_candidate_graph(
    rows: Iterable[Mapping[str, Any]],
) -> tuple[
    dict[tuple[str, str], dict[str, Any]],
    frozenset[tuple[str, str]],
]:
    candidate_edges: dict[tuple[str, str], dict[str, Any]] = {}
    hard: set[tuple[str, str]] = set()

    for row in rows:
        pair = canonical_pair(
            str(row["tracklet_a"]),
            str(row["tracklet_b"]),
        )
        if pair in candidate_edges or pair in hard:
            raise ValueError(f"duplicate candidate-graph pair: {pair}")

        status = row.get("status")
        if status == "candidate":
            candidate_edges[pair] = dict(row)
        elif status == "cannot_link":
            hard.add(pair)
        else:
            raise ValueError(f"unexpected graph status: {status!r}")

    return candidate_edges, frozenset(hard)


def effective_utility(row: Mapping[str, Any]) -> float:
    rank_score = _clamp01(
        _finite_float("rank_score", row["rank_score"])
    )
    confidence = _clamp01(
        _finite_float(
            "evidence_confidence",
            row["evidence_confidence"],
        )
    )
    effective_affinity = (
        0.5 + confidence * (rank_score - 0.5)
    )
    return max(
        -1.0,
        min(1.0, 2.0 * (effective_affinity - 0.5)),
    )


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
        payload = dict(row)
        payload["utility"] = effective_utility(row)
        result[pair] = payload
    return result


def _assignment(
    identities: Mapping[str, IdentityMeta],
) -> dict[str, str]:
    return {
        tracklet_id: identity_id
        for identity_id, identity in identities.items()
        for tracklet_id in identity.tracklet_ids
    }


def _identity_pair_context(
    source_identity: str,
    target_identity: str,
    identities: Mapping[str, IdentityMeta],
    hard: frozenset[tuple[str, str]],
    scored_links: Mapping[tuple[str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    source = identities[source_identity]
    target = identities[target_identity]

    hard_conflicts = []
    compatible_utilities: list[float] = []

    for left in source.tracklet_ids:
        for right in target.tracklet_ids:
            pair = canonical_pair(left, right)
            if pair in hard:
                hard_conflicts.append(pair)
                continue
            scored = scored_links.get(pair)
            if scored is not None:
                compatible_utilities.append(
                    float(scored["utility"])
                )

    merge_feasible = not hard_conflicts
    merge_delta = (
        sum(compatible_utilities)
        if merge_feasible
        else None
    )

    interval_overlap_s = max(
        0.0,
        min(source.end_time_s, target.end_time_s)
        - max(source.start_time_s, target.start_time_s),
    )

    return {
        "whole_identity_merge_feasible": merge_feasible,
        "whole_identity_hard_conflict_count": len(hard_conflicts),
        "whole_identity_hard_conflict_examples": [
            list(pair) for pair in hard_conflicts[:8]
        ],
        "whole_identity_merge_delta": merge_delta,
        "compatible_cross_pair_utility_sum": sum(
            compatible_utilities
        ),
        "compatible_cross_pair_count": len(
            compatible_utilities
        ),
        "identity_interval_overlap_s": interval_overlap_s,
    }


def build_cross_identity_continuations(
    *,
    identities: Mapping[str, IdentityMeta],
    tracklets: Mapping[str, TrackletMeta],
    candidate_edges: Mapping[tuple[str, str], Mapping[str, Any]],
    hard: frozenset[tuple[str, str]],
    scored_links: Mapping[tuple[str, str], Mapping[str, Any]],
    config: FragmentationAuditConfig,
) -> tuple[list[dict[str, Any]], float]:
    assignment = _assignment(identities)
    provisional: list[dict[str, Any]] = []

    for pair, scored in scored_links.items():
        if scored.get("link_kind") != "continuation":
            continue

        left_identity = assignment[pair[0]]
        right_identity = assignment[pair[1]]
        if left_identity == right_identity:
            continue

        edge = candidate_edges.get(pair)
        if edge is None:
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
        source_identity = assignment[predecessor]
        target_identity = assignment[successor]

        if source_identity == target_identity:
            continue

        utility = float(scored["utility"])
        source = identities[source_identity]
        target = identities[target_identity]

        source_tail_offset_s = max(
            0.0,
            source.end_time_s
            - tracklets[predecessor].end_time_s,
        )
        target_head_offset_s = max(
            0.0,
            tracklets[successor].start_time_s
            - target.start_time_s,
        )
        boundary_aligned = (
            source_tail_offset_s
            <= config.boundary_tolerance_s + 1e-12
            and target_head_offset_s
            <= config.boundary_tolerance_s + 1e-12
        )

        context = _identity_pair_context(
            source_identity,
            target_identity,
            identities,
            hard,
            scored_links,
        )

        provisional.append(
            {
                "schema_version": 1,
                "fragmentation_audit_version": (
                    FRAGMENTATION_AUDIT_VERSION
                ),
                "source_identity_id": source_identity,
                "target_identity_id": target_identity,
                "predecessor_tracklet_id": predecessor,
                "successor_tracklet_id": successor,
                "utility": utility,
                "rank_score": float(scored["rank_score"]),
                "evidence_confidence": float(
                    scored["evidence_confidence"]
                ),
                "gap_s": float(physics["gap_s"]),
                "endpoint_distance_m": float(
                    physics["endpoint_distance_m"]
                ),
                "required_speed_after_slack_mps": (
                    float(
                        physics[
                            "required_speed_after_slack_mps"
                        ]
                    )
                    if physics.get(
                        "required_speed_after_slack_mps"
                    )
                    is not None
                    else None
                ),
                "prediction_error_mean_m": (
                    float(physics["prediction_error_mean_m"])
                    if physics.get("prediction_error_mean_m")
                    is not None
                    else None
                ),
                "velocity_direction_cosine": (
                    float(physics["velocity_direction_cosine"])
                    if physics.get("velocity_direction_cosine")
                    is not None
                    else None
                ),
                "source_tail_offset_s": (
                    source_tail_offset_s
                ),
                "target_head_offset_s": (
                    target_head_offset_s
                ),
                "boundary_aligned": boundary_aligned,
                **context,
            }
        )

    positive_utilities = [
        row["utility"]
        for row in provisional
        if row["utility"] > 0
    ]
    quantile_threshold = (
        _percentile(
            positive_utilities,
            config.strong_positive_quantile,
        )
        if positive_utilities
        else None
    )
    strong_threshold = (
        float(config.strong_utility_threshold_override)
        if config.strong_utility_threshold_override is not None
        else max(
            config.strong_utility_floor,
            (
                quantile_threshold
                if quantile_threshold is not None
                else config.strong_utility_floor
            ),
        )
    )

    result = []
    for row in provisional:
        strong_positive = (
            row["utility"] >= strong_threshold
        )
        strong_boundary = (
            strong_positive and row["boundary_aligned"]
        )
        trajectory_clique_conflict = (
            strong_boundary
            and row["whole_identity_merge_feasible"]
            and row["whole_identity_merge_delta"] is not None
            and row["whole_identity_merge_delta"] <= 0.0
        )

        review_flags: list[str] = []
        if strong_positive:
            review_flags.append("top_cross_identity_support")
        if strong_boundary:
            review_flags.append("strong_boundary_handoff")
        if trajectory_clique_conflict:
            review_flags.append(
                "trajectory_vs_clique_conflict"
            )
        if (
            strong_positive
            and not row["whole_identity_merge_feasible"]
        ):
            review_flags.append(
                "strong_link_but_hard_merge_conflict"
            )
        if row["gap_s"] > 1.5:
            review_flags.append("long_gap")

        result.append(
            {
                **row,
                "strong_utility_threshold": (
                    strong_threshold
                ),
                "strong_positive": strong_positive,
                "strong_boundary_handoff": strong_boundary,
                "trajectory_vs_clique_conflict": (
                    trajectory_clique_conflict
                ),
                "review_flags": review_flags,
            }
        )

    eligible = [
        row
        for row in result
        if row["strong_positive"]
        and row["boundary_aligned"]
        and row["whole_identity_merge_feasible"]
        and row["whole_identity_merge_delta"] is not None
        and row["whole_identity_merge_delta"] <= 0.0
    ]
    best_outgoing: dict[str, dict[str, Any]] = {}
    best_incoming: dict[str, dict[str, Any]] = {}

    def _better_boundary(
        candidate: Mapping[str, Any],
        incumbent: Mapping[str, Any] | None,
    ) -> bool:
        if incumbent is None:
            return True
        return (
            candidate["utility"],
            candidate["evidence_confidence"],
            candidate["rank_score"],
            -candidate["gap_s"],
            candidate["predecessor_tracklet_id"],
            candidate["successor_tracklet_id"],
        ) > (
            incumbent["utility"],
            incumbent["evidence_confidence"],
            incumbent["rank_score"],
            -incumbent["gap_s"],
            incumbent["predecessor_tracklet_id"],
            incumbent["successor_tracklet_id"],
        )

    for row in eligible:
        source = row["source_identity_id"]
        target = row["target_identity_id"]
        if _better_boundary(row, best_outgoing.get(source)):
            best_outgoing[source] = row
        if _better_boundary(row, best_incoming.get(target)):
            best_incoming[target] = row

    for row in result:
        row["reciprocal_best_boundary_handoff"] = bool(
            row in eligible
            and best_outgoing.get(row["source_identity_id"]) is row
            and best_incoming.get(row["target_identity_id"]) is row
        )
        if row["reciprocal_best_boundary_handoff"]:
            row["review_flags"].append(
                "reciprocal_strong_boundary_handoff"
            )

    result.sort(
        key=lambda row: (
            -row["utility"],
            row["source_identity_id"],
            row["target_identity_id"],
            row["predecessor_tracklet_id"],
            row["successor_tracklet_id"],
        )
    )
    return result, strong_threshold


def build_identity_fragmentation_records(
    *,
    identities: Mapping[str, IdentityMeta],
    boundary_links: Sequence[Mapping[str, Any]],
    clip_start_s: float,
    clip_end_s: float,
    config: FragmentationAuditConfig,
) -> list[dict[str, Any]]:
    clip_span_s = max(1e-12, clip_end_s - clip_start_s)
    links_by_identity: dict[str, list[Mapping[str, Any]]] = {
        identity_id: []
        for identity_id in identities
    }

    for row in boundary_links:
        links_by_identity[row["source_identity_id"]].append(row)
        links_by_identity[row["target_identity_id"]].append(row)

    records: list[dict[str, Any]] = []

    for identity_id in sorted(identities):
        identity = identities[identity_id]
        links = links_by_identity[identity_id]
        positive = [
            row for row in links if row["utility"] > 0
        ]
        best = (
            max(
                positive,
                key=lambda row: (
                    row["utility"],
                    row["evidence_confidence"],
                ),
            )
            if positive
            else None
        )
        strong = [
            row
            for row in links
            if row["strong_boundary_handoff"]
        ]
        trajectory_conflicts = [
            row
            for row in strong
            if row["trajectory_vs_clique_conflict"]
        ]

        span_fraction = identity.span_s / clip_span_s
        coverage_fraction = (
            identity.union_coverage_s / clip_span_s
        )
        singleton = identity.tracklet_count == 1

        if singleton and span_fraction >= (
            config.long_singleton_span_fraction
        ):
            review_class = "long_singleton"
        elif trajectory_conflicts:
            review_class = "trajectory_vs_clique_conflict"
        elif strong:
            review_class = "strong_fragmentation_candidate"
        elif (
            singleton
            and span_fraction
            < config.short_identity_span_fraction
            and positive
        ):
            review_class = "short_unresolved_fragment"
        elif (
            singleton
            and span_fraction
            < config.short_identity_span_fraction
        ):
            review_class = "isolated_short_fragment"
        else:
            review_class = "no_strong_fragmentation_signal"

        records.append(
            {
                "schema_version": 1,
                "fragmentation_audit_version": (
                    FRAGMENTATION_AUDIT_VERSION
                ),
                "identity_id": identity_id,
                "tracklet_ids": list(identity.tracklet_ids),
                "tracklet_count": identity.tracklet_count,
                "singleton": singleton,
                "start_time_s": identity.start_time_s,
                "end_time_s": identity.end_time_s,
                "span_s": identity.span_s,
                "span_fraction_of_clip": span_fraction,
                "union_coverage_s": (
                    identity.union_coverage_s
                ),
                "coverage_fraction_of_clip": (
                    coverage_fraction
                ),
                "observation_count": (
                    identity.observation_count
                ),
                "positive_cross_continuation_count": len(
                    positive
                ),
                "strong_boundary_handoff_count": len(
                    strong
                ),
                "trajectory_vs_clique_conflict_count": len(
                    trajectory_conflicts
                ),
                "best_cross_identity_link": (
                    {
                        "source_identity_id": (
                            best["source_identity_id"]
                        ),
                        "target_identity_id": (
                            best["target_identity_id"]
                        ),
                        "predecessor_tracklet_id": (
                            best["predecessor_tracklet_id"]
                        ),
                        "successor_tracklet_id": (
                            best["successor_tracklet_id"]
                        ),
                        "utility": best["utility"],
                        "rank_score": best["rank_score"],
                        "evidence_confidence": (
                            best["evidence_confidence"]
                        ),
                        "gap_s": best["gap_s"],
                        "boundary_aligned": (
                            best["boundary_aligned"]
                        ),
                        "whole_identity_merge_feasible": (
                            best[
                                "whole_identity_merge_feasible"
                            ]
                        ),
                        "whole_identity_merge_delta": (
                            best[
                                "whole_identity_merge_delta"
                            ]
                        ),
                    }
                    if best is not None
                    else None
                ),
                "review_class": review_class,
                "requires_review": review_class not in {
                    "long_singleton",
                    "no_strong_fragmentation_signal",
                },
                "identity_decision": None,
            }
        )

    return records


def build_fragmentation_summary(
    *,
    identities: Mapping[str, IdentityMeta],
    boundary_links: Sequence[Mapping[str, Any]],
    identity_records: Sequence[Mapping[str, Any]],
    strong_threshold: float,
    config: FragmentationAuditConfig,
) -> dict[str, Any]:
    class_counts: dict[str, int] = {}
    for row in identity_records:
        class_counts[row["review_class"]] = (
            class_counts.get(row["review_class"], 0) + 1
        )

    positive = [
        row for row in boundary_links if row["utility"] > 0
    ]
    strong = [
        row
        for row in boundary_links
        if row["strong_boundary_handoff"]
    ]
    trajectory_conflicts = [
        row
        for row in boundary_links
        if row["trajectory_vs_clique_conflict"]
    ]
    reciprocal_remaining = [
        row
        for row in boundary_links
        if row.get(
            "reciprocal_best_boundary_handoff",
            False,
        )
    ]

    return {
        "schema_version": 1,
        "fragmentation_audit_version": (
            FRAGMENTATION_AUDIT_VERSION
        ),
        "provisional": True,
        "identity_stage_completed": False,
        "identity_decisions": False,
        "identity_count": len(identities),
        "singleton_identity_count": sum(
            identity.tracklet_count == 1
            for identity in identities.values()
        ),
        "multi_tracklet_identity_count": sum(
            identity.tracklet_count > 1
            for identity in identities.values()
        ),
        "cross_identity_continuation_count": len(
            boundary_links
        ),
        "positive_cross_identity_continuation_count": (
            len(positive)
        ),
        "strong_utility_threshold": strong_threshold,
        "strong_boundary_handoff_count": len(strong),
        "trajectory_vs_clique_conflict_count": len(
            trajectory_conflicts
        ),
        "remaining_reciprocal_strong_boundary_handoff_count": len(
            reciprocal_remaining
        ),
        "strong_hard_blocked_handoff_count": sum(
            bool(
                row["strong_positive"]
                and not row[
                    "whole_identity_merge_feasible"
                ]
            )
            for row in boundary_links
        ),
        "positive_cross_utility": _distribution(
            [row["utility"] for row in positive]
        ),
        "identity_review_class_counts": dict(
            sorted(class_counts.items())
        ),
        "config": {
            "strong_positive_quantile": (
                config.strong_positive_quantile
            ),
            "strong_utility_floor": (
                config.strong_utility_floor
            ),
            "boundary_tolerance_s": (
                config.boundary_tolerance_s
            ),
            "long_singleton_span_fraction": (
                config.long_singleton_span_fraction
            ),
            "short_identity_span_fraction": (
                config.short_identity_span_fraction
            ),
            "strong_utility_threshold_override": (
                config.strong_utility_threshold_override
            ),
        },
    }


def build_review_queue(
    *,
    boundary_links: Sequence[Mapping[str, Any]],
    identity_records: Sequence[Mapping[str, Any]],
    top_n: int = 30,
) -> dict[str, Any]:
    strong = [
        dict(row)
        for row in boundary_links
        if row["strong_boundary_handoff"]
    ]
    trajectory_conflicts = [
        dict(row)
        for row in boundary_links
        if row["trajectory_vs_clique_conflict"]
    ]
    hard_blocked = [
        dict(row)
        for row in boundary_links
        if (
            row["strong_positive"]
            and not row["whole_identity_merge_feasible"]
        )
    ]
    singleton_review = [
        dict(row)
        for row in identity_records
        if row["singleton"]
    ]

    singleton_review.sort(
        key=lambda row: (
            row["review_class"] == "long_singleton",
            -(
                row["best_cross_identity_link"]["utility"]
                if row["best_cross_identity_link"]
                else -1.0
            ),
            row["identity_id"],
        )
    )

    return {
        "schema_version": 1,
        "fragmentation_audit_version": (
            FRAGMENTATION_AUDIT_VERSION
        ),
        "provisional": True,
        "remaining_reciprocal_strong_boundary_handoffs": [
            dict(row)
            for row in boundary_links
            if row.get(
                "reciprocal_best_boundary_handoff",
                False,
            )
        ][:top_n],
        "trajectory_vs_clique_conflicts": (
            trajectory_conflicts[:top_n]
        ),
        "strong_boundary_handoffs": strong[:top_n],
        "strong_links_blocked_by_hard_constraints": (
            hard_blocked[:top_n]
        ),
        "singleton_review": singleton_review[:top_n],
        "strongest_positive_cross_identity_links": [
            dict(row)
            for row in boundary_links
            if row["utility"] > 0
        ][:top_n],
    }
