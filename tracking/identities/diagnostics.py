"""Diagnostics for provisional Tracking V1 physical identities.

Hito 3C audits a provisional 3B-B global partition before it is allowed to
become the canonical ``TrackingStage.IDENTITIES`` artifact.

The audit is intentionally diagnostic, not another hidden solver. It checks:
- temporal coverage, gaps and overlaps inside each provisional identity;
- the maximum-spanning-tree "backbone" that holds a cluster together;
- weak member attachment;
- best single-tracklet relocation to another identity;
- best whole-identity merge;
- exact best two-way split for manageable cluster sizes.

The exact split audit is especially important because 3B-B uses single-node
moves and whole-cluster merges. A cluster can be locally stable to every
single-node move while still admitting a profitable *group* split. 3C detects
that situation explicitly.

All utilities are reconstructed from the 3B-A contract:

    effective_affinity = 0.5 + confidence * (rank_score - 0.5)
    utility = 2 * (effective_affinity - 0.5)

No new identity assignment is published by this module.
"""

from __future__ import annotations

import itertools
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


LEGACY_DIAGNOSTICS_VERSION = "identity-audit-v1"
DIAGNOSTICS_VERSION = "identity-backbone-audit-v1"
_EPS = 1e-9


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


def _distribution(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    ordered = sorted(float(value) for value in values)
    n = len(ordered)

    def percentile(q: float) -> float:
        if n == 1:
            return ordered[0]
        pos = (n - 1) * q
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        if lo == hi:
            return ordered[lo]
        alpha = pos - lo
        return ordered[lo] * (1.0 - alpha) + ordered[hi] * alpha

    return {
        "count": n,
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "max": ordered[-1],
        "mean": sum(ordered) / n,
    }


@dataclass(frozen=True, slots=True)
class TrackletSpan:
    tracklet_id: str
    start_frame: int
    end_frame: int
    start_time_s: float
    end_time_s: float
    observation_count: int
    frame_set: frozenset[int]

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_time_s - self.start_time_s)


@dataclass(frozen=True, slots=True)
class PairUtility:
    tracklet_a: str
    tracklet_b: str
    rank_score: float
    evidence_confidence: float
    effective_affinity: float
    utility: float
    link_kind: str

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


def utility_from_scored_link(row: Mapping[str, Any]) -> PairUtility:
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
    effective = _clamp01(
        0.5 + confidence * (rank_score - 0.5)
    )
    utility = max(-1.0, min(1.0, 2.0 * (effective - 0.5)))
    return PairUtility(
        tracklet_a=a,
        tracklet_b=b,
        rank_score=rank_score,
        evidence_confidence=confidence,
        effective_affinity=effective,
        utility=utility,
        link_kind=str(row["link_kind"]),
    )


def load_tracklet_spans(
    rows: Iterable[Mapping[str, Any]],
) -> dict[str, TrackletSpan]:
    result: dict[str, TrackletSpan] = {}
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        observations = row.get("observations")
        if not isinstance(observations, list) or not observations:
            raise ValueError(
                f"{tracklet_id} has no observations"
            )
        frames = [int(obs["frame_index"]) for obs in observations]
        times = [
            _finite_float("timestamp_s", obs["timestamp_s"])
            for obs in observations
        ]
        if frames != sorted(frames):
            raise ValueError(
                f"{tracklet_id} frames are not sorted"
            )
        if times != sorted(times):
            raise ValueError(
                f"{tracklet_id} timestamps are not sorted"
            )
        if tracklet_id in result:
            raise ValueError(
                f"duplicate tracklet: {tracklet_id}"
            )
        result[tracklet_id] = TrackletSpan(
            tracklet_id=tracklet_id,
            start_frame=frames[0],
            end_frame=frames[-1],
            start_time_s=times[0],
            end_time_s=times[-1],
            observation_count=len(observations),
            frame_set=frozenset(frames),
        )
    return result


def load_utilities(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], PairUtility]:
    result: dict[tuple[str, str], PairUtility] = {}
    for row in rows:
        utility = utility_from_scored_link(row)
        pair = (utility.tracklet_a, utility.tracklet_b)
        if pair in result:
            raise ValueError(f"duplicate scored pair: {pair}")
        result[pair] = utility
    return result


def load_hard_cannot_links(
    rows: Iterable[Mapping[str, Any]],
) -> frozenset[tuple[str, str]]:
    hard: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("status") != "cannot_link":
            continue
        hard.add(
            canonical_pair(
                str(row["tracklet_a"]),
                str(row["tracklet_b"]),
            )
        )
    return frozenset(hard)


def pair_utility(
    utilities: Mapping[tuple[str, str], PairUtility],
    a: str,
    b: str,
) -> float:
    pair = canonical_pair(a, b)
    utility = utilities.get(pair)
    if utility is None:
        raise KeyError(
            f"no soft utility for compatible pair {pair}"
        )
    return utility.utility


def _cluster_objective(
    members: Sequence[str],
    utilities: Mapping[tuple[str, str], PairUtility],
) -> float:
    return sum(
        pair_utility(utilities, a, b)
        for a, b in itertools.combinations(members, 2)
    )


def _timeline_audit(
    members: Sequence[str],
    spans: Mapping[str, TrackletSpan],
) -> dict[str, Any]:
    member_spans = [spans[member] for member in members]
    start = min(span.start_time_s for span in member_spans)
    end = max(span.end_time_s for span in member_spans)
    intervals = sorted(
        (
            span.start_time_s,
            span.end_time_s,
            span.tracklet_id,
        )
        for span in member_spans
    )

    merged: list[list[float]] = []
    for interval_start, interval_end, _ in intervals:
        if not merged or interval_start > merged[-1][1] + 1e-12:
            merged.append([interval_start, interval_end])
        else:
            merged[-1][1] = max(
                merged[-1][1],
                interval_end,
            )

    gaps = [
        merged[index + 1][0] - merged[index][1]
        for index in range(len(merged) - 1)
        if merged[index + 1][0] > merged[index][1]
    ]
    union_duration = sum(
        interval_end - interval_start
        for interval_start, interval_end in merged
    )
    span_duration = max(0.0, end - start)

    overlap_pair_count = 0
    shared_frame_pair_count = 0
    for left, right in itertools.combinations(
        member_spans,
        2,
    ):
        if (
            max(left.start_time_s, right.start_time_s)
            <= min(left.end_time_s, right.end_time_s) + 1e-12
        ):
            overlap_pair_count += 1
        if left.frame_set & right.frame_set:
            shared_frame_pair_count += 1

    frame_counts: dict[int, int] = {}
    for span in member_spans:
        for frame in span.frame_set:
            frame_counts[frame] = (
                frame_counts.get(frame, 0) + 1
            )

    return {
        "start_time_s": start,
        "end_time_s": end,
        "span_s": span_duration,
        "union_coverage_s": union_duration,
        "coverage_fraction": (
            union_duration / span_duration
            if span_duration > 0
            else 1.0
        ),
        "gap_count": len(gaps),
        "total_gap_s": sum(gaps),
        "max_gap_s": max(gaps) if gaps else 0.0,
        "gaps_s": gaps,
        "overlapping_interval_pair_count": overlap_pair_count,
        "shared_observation_frame_pair_count": (
            shared_frame_pair_count
        ),
        "max_exact_observation_concurrency": (
            max(frame_counts.values())
            if frame_counts
            else 0
        ),
        "observation_count": sum(
            span.observation_count
            for span in member_spans
        ),
    }


def _maximum_spanning_tree(
    members: Sequence[str],
    utilities: Mapping[tuple[str, str], PairUtility],
) -> dict[str, Any]:
    if len(members) <= 1:
        return {
            "edge_count": 0,
            "total_utility": 0.0,
            "weakest_utility": None,
            "median_utility": None,
            "mean_utility": None,
            "negative_edge_count": 0,
            "edges": [],
        }

    parent = {member: member for member in members}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    weighted_edges: list[
        tuple[float, float, str, str, PairUtility]
    ] = []
    for a, b in itertools.combinations(
        sorted(members),
        2,
    ):
        utility = utilities[canonical_pair(a, b)]
        weighted_edges.append(
            (
                utility.utility,
                utility.evidence_confidence,
                a,
                b,
                utility,
            )
        )
    weighted_edges.sort(reverse=True)

    selected: list[PairUtility] = []
    for _, _, a, b, utility in weighted_edges:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            continue
        keep, drop = sorted((ra, rb))
        parent[drop] = keep
        selected.append(utility)
        if len(selected) == len(members) - 1:
            break

    if len(selected) != len(members) - 1:
        raise ValueError(
            "identity soft graph is unexpectedly disconnected"
        )

    values = [utility.utility for utility in selected]
    return {
        "edge_count": len(selected),
        "total_utility": sum(values),
        "weakest_utility": min(values),
        "median_utility": statistics.median(values),
        "mean_utility": sum(values) / len(values),
        "negative_edge_count": sum(
            value < 0 for value in values
        ),
        "edges": [
            utility.to_dict()
            for utility in selected
        ],
    }


def _best_exact_two_way_split(
    members: Sequence[str],
    utilities: Mapping[tuple[str, str], PairUtility],
    *,
    max_exact_size: int,
) -> dict[str, Any]:
    ordered = tuple(sorted(members))
    n = len(ordered)
    if n <= 1:
        return {
            "evaluated": False,
            "reason": "singleton",
            "best_delta": None,
            "left": None,
            "right": None,
        }
    if n > max_exact_size:
        return {
            "evaluated": False,
            "reason": "cluster_too_large",
            "best_delta": None,
            "left": None,
            "right": None,
        }

    first = ordered[0]
    remaining = ordered[1:]
    best_delta = -math.inf
    best_left: tuple[str, ...] | None = None
    best_right: tuple[str, ...] | None = None

    # Keep first in the left group to avoid mirrored partitions.
    for mask in range(0, 1 << len(remaining)):
        left = [first]
        right: list[str] = []
        for bit, member in enumerate(remaining):
            if mask & (1 << bit):
                left.append(member)
            else:
                right.append(member)
        if not right:
            continue

        cross_sum = sum(
            pair_utility(utilities, a, b)
            for a in left
            for b in right
        )
        delta = -cross_sum
        if delta > best_delta + 1e-12:
            best_delta = delta
            best_left = tuple(sorted(left))
            best_right = tuple(sorted(right))

    assert best_left is not None
    assert best_right is not None
    return {
        "evaluated": True,
        "reason": None,
        "best_delta": best_delta,
        "profitable": best_delta > _EPS,
        "left": list(best_left),
        "right": list(best_right),
    }


def _member_attachment(
    members: Sequence[str],
    utilities: Mapping[tuple[str, str], PairUtility],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for member in members:
        result[member] = sum(
            pair_utility(utilities, member, other)
            for other in members
            if other != member
        )
    return result


def _compatible_with_cluster(
    node: str,
    target: Sequence[str],
    hard: frozenset[tuple[str, str]],
) -> bool:
    return all(
        canonical_pair(node, other) not in hard
        for other in target
    )


def _best_member_moves(
    identity_id: str,
    members: Sequence[str],
    all_identities: Mapping[str, Sequence[str]],
    utilities: Mapping[tuple[str, str], PairUtility],
    hard: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    current_attachment = _member_attachment(
        members,
        utilities,
    )
    rows: list[dict[str, Any]] = []

    for member in sorted(members):
        best_delta = -math.inf
        best_target: str | None = None
        best_target_support: float | None = None

        for target_id, target_members in all_identities.items():
            if target_id == identity_id:
                continue
            if not _compatible_with_cluster(
                member,
                target_members,
                hard,
            ):
                continue
            target_support = sum(
                pair_utility(utilities, member, other)
                for other in target_members
            )
            delta = (
                target_support
                - current_attachment[member]
            )
            if delta > best_delta + 1e-12:
                best_delta = delta
                best_target = target_id
                best_target_support = target_support

        rows.append(
            {
                "tracklet_id": member,
                "current_attachment_utility": (
                    current_attachment[member]
                ),
                "best_alternative_identity_id": best_target,
                "best_alternative_support": (
                    best_target_support
                ),
                "best_move_delta": (
                    best_delta
                    if best_target is not None
                    else None
                ),
                "move_would_improve_objective": (
                    best_target is not None
                    and best_delta > _EPS
                ),
                "move_margin": (
                    -best_delta
                    if best_target is not None
                    else None
                ),
            }
        )
    return rows


def _best_merge_for_identity(
    identity_id: str,
    members: Sequence[str],
    all_identities: Mapping[str, Sequence[str]],
    utilities: Mapping[tuple[str, str], PairUtility],
    hard: frozenset[tuple[str, str]],
) -> dict[str, Any]:
    best_delta = -math.inf
    best_target: str | None = None

    for target_id, target_members in all_identities.items():
        if target_id == identity_id:
            continue
        compatible = all(
            canonical_pair(a, b) not in hard
            for a in members
            for b in target_members
        )
        if not compatible:
            continue

        delta = sum(
            pair_utility(utilities, a, b)
            for a in members
            for b in target_members
        )
        if delta > best_delta + 1e-12:
            best_delta = delta
            best_target = target_id

    return {
        "target_identity_id": best_target,
        "merge_delta": (
            best_delta
            if best_target is not None
            else None
        ),
        "merge_would_improve_objective": (
            best_target is not None
            and best_delta > _EPS
        ),
        "merge_margin": (
            -best_delta
            if best_target is not None
            else None
        ),
    }


def audit_partition(
    identities: Sequence[Mapping[str, Any]],
    spans: Mapping[str, TrackletSpan],
    utilities: Mapping[tuple[str, str], PairUtility],
    hard: frozenset[tuple[str, str]],
    *,
    max_exact_split_size: int = 16,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    identity_members: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()

    for identity in identities:
        identity_id = str(identity["identity_id"])
        members = tuple(
            sorted(str(value) for value in identity["tracklet_ids"])
        )
        if not members:
            raise ValueError(
                f"{identity_id} has no tracklets"
            )
        if identity_id in identity_members:
            raise ValueError(
                f"duplicate identity_id: {identity_id}"
            )
        for member in members:
            if member in assigned:
                raise ValueError(
                    f"tracklet {member} assigned twice"
                )
            if member not in spans:
                raise ValueError(
                    f"missing span for {member}"
                )
            assigned.add(member)
        identity_members[identity_id] = members

    if assigned != set(spans):
        raise ValueError(
            "identity partition does not cover exactly the tracklet set"
        )

    identity_audits: list[dict[str, Any]] = []
    tracklet_audits: list[dict[str, Any]] = []

    for identity_id in sorted(identity_members):
        members = identity_members[identity_id]
        objective = _cluster_objective(
            members,
            utilities,
        )
        timeline = _timeline_audit(
            members,
            spans,
        )
        backbone = _maximum_spanning_tree(
            members,
            utilities,
        )
        split = _best_exact_two_way_split(
            members,
            utilities,
            max_exact_size=max_exact_split_size,
        )
        moves = _best_member_moves(
            identity_id,
            members,
            identity_members,
            utilities,
            hard,
        )
        merge = _best_merge_for_identity(
            identity_id,
            members,
            identity_members,
            utilities,
            hard,
        )

        pair_values = [
            pair_utility(utilities, a, b)
            for a, b in itertools.combinations(members, 2)
        ]
        attachment_values = [
            row["current_attachment_utility"]
            for row in moves
        ]
        best_move_values = [
            row["best_move_delta"]
            for row in moves
            if row["best_move_delta"] is not None
        ]

        flags: list[str] = []
        if split.get("profitable"):
            flags.append("profitable_exact_two_way_split")
        if any(
            row["move_would_improve_objective"]
            for row in moves
        ):
            flags.append("profitable_single_tracklet_move")
        if merge["merge_would_improve_objective"]:
            flags.append("profitable_whole_identity_merge")
        if any(value < 0 for value in pair_values):
            flags.append("contains_negative_intra_identity_links")
        if timeline["gap_count"] > 0:
            flags.append("contains_temporal_gaps")
        if (
            timeline["max_exact_observation_concurrency"] > 1
        ):
            flags.append("contains_exact_frame_overlap")

        audit = {
            "schema_version": 1,
            "diagnostics_version": DIAGNOSTICS_VERSION,
            "identity_id": identity_id,
            "tracklet_ids": list(members),
            "tracklet_count": len(members),
            "cluster_objective": objective,
            "pair_utility": _distribution(pair_values),
            "negative_intra_pair_count": sum(
                value < 0 for value in pair_values
            ),
            "timeline": timeline,
            "backbone": backbone,
            "member_attachment": {
                "distribution": _distribution(
                    attachment_values
                ),
                "weakest_tracklet_id": (
                    min(
                        moves,
                        key=lambda row: (
                            row["current_attachment_utility"],
                            row["tracklet_id"],
                        ),
                    )["tracklet_id"]
                    if moves
                    else None
                ),
                "weakest_attachment_utility": (
                    min(attachment_values)
                    if attachment_values
                    else None
                ),
            },
            "best_exact_two_way_split": split,
            "best_whole_identity_merge": merge,
            "best_member_move_delta": (
                max(best_move_values)
                if best_move_values
                else None
            ),
            "flags": flags,
            "provisional": True,
        }
        identity_audits.append(audit)

        for move in moves:
            tracklet_audits.append(
                {
                    "schema_version": 1,
                    "diagnostics_version": DIAGNOSTICS_VERSION,
                    "identity_id": identity_id,
                    **move,
                }
            )

    profitable_splits = [
        audit
        for audit in identity_audits
        if audit["best_exact_two_way_split"].get(
            "profitable"
        )
    ]
    profitable_moves = [
        row
        for row in tracklet_audits
        if row["move_would_improve_objective"]
    ]
    profitable_merges = [
        audit
        for audit in identity_audits
        if audit["best_whole_identity_merge"][
            "merge_would_improve_objective"
        ]
    ]

    multi = [
        audit
        for audit in identity_audits
        if audit["tracklet_count"] > 1
    ]

    summary = {
        "schema_version": 1,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "provisional": True,
        "identity_count": len(identity_audits),
        "tracklet_count": len(assigned),
        "singleton_identity_count": sum(
            audit["tracklet_count"] == 1
            for audit in identity_audits
        ),
        "multi_tracklet_identity_count": len(multi),
        "exact_split_evaluated_identity_count": sum(
            bool(
                audit["best_exact_two_way_split"][
                    "evaluated"
                ]
            )
            for audit in identity_audits
        ),
        "profitable_exact_split_identity_count": len(
            profitable_splits
        ),
        "max_profitable_exact_split_delta": (
            max(
                audit["best_exact_two_way_split"][
                    "best_delta"
                ]
                for audit in profitable_splits
            )
            if profitable_splits
            else 0.0
        ),
        "profitable_single_tracklet_move_count": len(
            profitable_moves
        ),
        "profitable_whole_identity_merge_count": len(
            profitable_merges
        ),
        "hard_cannot_link_count": len(hard),
        "multi_identity_backbone_weakest_utility": (
            _distribution(
                [
                    audit["backbone"]["weakest_utility"]
                    for audit in multi
                ]
            )
            if multi
            else _distribution([])
        ),
        "multi_identity_temporal_coverage_fraction": (
            _distribution(
                [
                    audit["timeline"]["coverage_fraction"]
                    for audit in multi
                ]
            )
            if multi
            else _distribution([])
        ),
        "identity_temporal_gap_count": _distribution(
            [
                audit["timeline"]["gap_count"]
                for audit in identity_audits
            ]
        ),
        "negative_intra_pair_count_total": sum(
            audit["negative_intra_pair_count"]
            for audit in identity_audits
        ),
        "identity_flag_counts": _flag_counts(
            identity_audits
        ),
    }
    return identity_audits, tracklet_audits, summary


def _flag_counts(
    audits: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for audit in audits:
        for flag in audit.get("flags", []):
            counts[str(flag)] = counts.get(
                str(flag),
                0,
            ) + 1
    return dict(sorted(counts.items()))


def build_review_queue(
    identity_audits: Sequence[Mapping[str, Any]],
    tracklet_audits: Sequence[Mapping[str, Any]],
    utilities: Mapping[tuple[str, str], PairUtility],
    identities: Sequence[Mapping[str, Any]],
    *,
    top_n: int = 25,
) -> dict[str, Any]:
    assignment: dict[str, str] = {}
    for identity in identities:
        identity_id = str(identity["identity_id"])
        for tracklet_id in identity["tracklet_ids"]:
            assignment[str(tracklet_id)] = identity_id

    split_review = sorted(
        (
            audit
            for audit in identity_audits
            if audit["best_exact_two_way_split"]["evaluated"]
        ),
        key=lambda audit: (
            audit["best_exact_two_way_split"][
                "best_delta"
            ],
            audit["tracklet_count"],
        ),
        reverse=True,
    )[:top_n]

    weak_backbones = sorted(
        (
            audit
            for audit in identity_audits
            if audit["tracklet_count"] > 1
        ),
        key=lambda audit: (
            audit["backbone"]["weakest_utility"],
            audit["backbone"]["mean_utility"],
        ),
    )[:top_n]

    ambiguous_moves = sorted(
        (
            row
            for row in tracklet_audits
            if row["best_move_delta"] is not None
        ),
        key=lambda row: (
            -row["best_move_delta"],
            row["tracklet_id"],
        ),
    )[:top_n]

    cut_positive: list[dict[str, Any]] = []
    for utility in utilities.values():
        if utility.utility <= 0:
            continue
        if (
            assignment[utility.tracklet_a]
            == assignment[utility.tracklet_b]
        ):
            continue
        cut_positive.append(
            {
                **utility.to_dict(),
                "identity_a": assignment[
                    utility.tracklet_a
                ],
                "identity_b": assignment[
                    utility.tracklet_b
                ],
            }
        )
    cut_positive.sort(
        key=lambda row: (
            row["utility"],
            row["evidence_confidence"],
        ),
        reverse=True,
    )

    return {
        "schema_version": 1,
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "provisional": True,
        "highest_exact_split_pressure": [
            {
                "identity_id": audit["identity_id"],
                "tracklet_count": audit["tracklet_count"],
                **audit["best_exact_two_way_split"],
            }
            for audit in split_review
        ],
        "weakest_backbones": [
            {
                "identity_id": audit["identity_id"],
                "tracklet_count": audit["tracklet_count"],
                "backbone": audit["backbone"],
            }
            for audit in weak_backbones
        ],
        "most_ambiguous_tracklet_moves": ambiguous_moves,
        "strongest_cut_positive_links": (
            cut_positive[:top_n]
        ),
    }


# ============================================================================
# Hito 3C V1: backbone-aware diagnostics
# ============================================================================

@dataclass(frozen=True, slots=True)
class SelectedBackboneLink:
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

    @property
    def pair(self) -> tuple[str, str]:
        return canonical_pair(
            self.predecessor_tracklet_id,
            self.successor_tracklet_id,
        )


def load_selected_backbone_links(
    rows: Iterable[Mapping[str, Any]],
) -> list[SelectedBackboneLink]:
    result: list[SelectedBackboneLink] = []
    seen: set[tuple[str, str]] = set()

    for row in rows:
        link = SelectedBackboneLink(
            predecessor_tracklet_id=str(
                row["predecessor_tracklet_id"]
            ),
            successor_tracklet_id=str(
                row["successor_tracklet_id"]
            ),
            utility=_finite_float(
                "utility", row["utility"]
            ),
            rank_score=_clamp01(
                _finite_float(
                    "rank_score", row["rank_score"]
                )
            ),
            evidence_confidence=_clamp01(
                _finite_float(
                    "evidence_confidence",
                    row["evidence_confidence"],
                )
            ),
            gap_s=_finite_float(
                "gap_s", row["gap_s"]
            ),
            endpoint_distance_m=_finite_float(
                "endpoint_distance_m",
                row["endpoint_distance_m"],
            ),
            source_tail_offset_s=_finite_float(
                "source_tail_offset_s",
                row["source_tail_offset_s"],
            ),
            target_head_offset_s=_finite_float(
                "target_head_offset_s",
                row["target_head_offset_s"],
            ),
            whole_clique_merge_delta=_finite_float(
                "whole_clique_merge_delta",
                row["whole_clique_merge_delta"],
            ),
            strong_utility_threshold=_finite_float(
                "strong_utility_threshold",
                row["strong_utility_threshold"],
            ),
            reciprocal_best=bool(
                row["reciprocal_best"]
            ),
        )
        if link.pair in seen:
            raise ValueError(
                f"duplicate selected backbone link: {link.pair}"
            )
        seen.add(link.pair)
        result.append(link)

    return result


def load_candidate_edges_for_diagnostics(
    rows: Iterable[Mapping[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        pair = canonical_pair(
            str(row["tracklet_a"]),
            str(row["tracklet_b"]),
        )
        if pair in result:
            raise ValueError(
                f"duplicate candidate graph pair: {pair}"
            )
        result[pair] = dict(row)
    return result


def _identity_members_map(
    identities: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    assigned: set[str] = set()

    for row in identities:
        identity_id = str(row["identity_id"])
        members = tuple(
            sorted(
                str(value)
                for value in row["tracklet_ids"]
            )
        )
        if not members:
            raise ValueError(
                f"{identity_id} has no tracklets"
            )
        if identity_id in result:
            raise ValueError(
                f"duplicate identity: {identity_id}"
            )
        for member in members:
            if member in assigned:
                raise ValueError(
                    f"tracklet assigned twice: {member}"
                )
            assigned.add(member)
        result[identity_id] = members

    return result


def _assignment_from_members(
    identities: Mapping[str, Sequence[str]],
) -> dict[str, str]:
    return {
        tracklet_id: identity_id
        for identity_id, members in identities.items()
        for tracklet_id in members
    }


def _hard_violations_for_partition(
    identities: Mapping[str, Sequence[str]],
    hard: frozenset[tuple[str, str]],
) -> list[dict[str, Any]]:
    assignment = _assignment_from_members(identities)
    violations = []
    for a, b in sorted(hard):
        identity_a = assignment.get(a)
        identity_b = assignment.get(b)
        if (
            identity_a is not None
            and identity_a == identity_b
        ):
            violations.append(
                {
                    "tracklet_a": a,
                    "tracklet_b": b,
                    "identity_id": identity_a,
                }
            )
    return violations


def _baseline_block_mapping(
    *,
    baseline_members: Mapping[str, Sequence[str]],
    final_members: Mapping[str, Sequence[str]],
) -> tuple[
    dict[str, str],
    dict[str, list[str]],
    list[dict[str, Any]],
]:
    final_assignment = _assignment_from_members(
        final_members
    )
    baseline_to_final: dict[str, str] = {}
    final_to_baseline: dict[str, list[str]] = {
        identity_id: []
        for identity_id in final_members
    }
    violations: list[dict[str, Any]] = []

    for baseline_id, members in baseline_members.items():
        destinations = {
            final_assignment[tracklet_id]
            for tracklet_id in members
        }
        if len(destinations) != 1:
            violations.append(
                {
                    "baseline_identity_id": baseline_id,
                    "final_identity_ids": sorted(
                        destinations
                    ),
                    "tracklet_ids": list(members),
                }
            )
            continue
        final_id = next(iter(destinations))
        baseline_to_final[baseline_id] = final_id
        final_to_baseline[final_id].append(
            baseline_id
        )

    for baseline_ids in final_to_baseline.values():
        baseline_ids.sort()

    return (
        baseline_to_final,
        final_to_baseline,
        violations,
    )


def _cluster_hard_conflicts(
    left: Sequence[str],
    right: Sequence[str],
    hard: frozenset[tuple[str, str]],
) -> list[tuple[str, str]]:
    return [
        canonical_pair(a, b)
        for a in left
        for b in right
        if canonical_pair(a, b) in hard
    ]


def _candidate_transition_payload(
    candidate_edges: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
    pair: tuple[str, str],
) -> Mapping[str, Any] | None:
    row = candidate_edges.get(pair)
    if row is None:
        return None
    if row.get("status") != "candidate":
        return None
    if row.get("reason") != "transition_reachable":
        return None
    physics = row.get("physics")
    if not isinstance(physics, Mapping):
        return None
    return row


def _boundary_candidate_rows(
    *,
    baseline_members: Mapping[str, Sequence[str]],
    spans: Mapping[str, TrackletSpan],
    utilities: Mapping[tuple[str, str], PairUtility],
    hard: frozenset[tuple[str, str]],
    candidate_edges: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
    boundary_tolerance_s: float,
) -> list[dict[str, Any]]:
    assignment = _assignment_from_members(
        baseline_members
    )
    bounds = {
        identity_id: (
            min(
                spans[tracklet_id].start_time_s
                for tracklet_id in members
            ),
            max(
                spans[tracklet_id].end_time_s
                for tracklet_id in members
            ),
        )
        for identity_id, members
        in baseline_members.items()
    }

    result = []
    for pair, utility in utilities.items():
        if utility.link_kind != "continuation":
            continue
        if utility.utility <= 0:
            continue
        if assignment[pair[0]] == assignment[pair[1]]:
            continue

        edge = _candidate_transition_payload(
            candidate_edges,
            pair,
        )
        if edge is None:
            continue

        physics = edge["physics"]
        predecessor = str(physics["predecessor"])
        successor = str(physics["successor"])
        source_id = assignment[predecessor]
        target_id = assignment[successor]
        if source_id == target_id:
            continue

        source_members = baseline_members[source_id]
        target_members = baseline_members[target_id]
        hard_conflicts = _cluster_hard_conflicts(
            source_members,
            target_members,
            hard,
        )

        source_tail_offset_s = max(
            0.0,
            bounds[source_id][1]
            - spans[predecessor].end_time_s,
        )
        target_head_offset_s = max(
            0.0,
            spans[successor].start_time_s
            - bounds[target_id][0],
        )
        boundary_aligned = (
            source_tail_offset_s
            <= boundary_tolerance_s + _EPS
            and target_head_offset_s
            <= boundary_tolerance_s + _EPS
        )

        cross_clique_delta = None
        if not hard_conflicts:
            cross_clique_delta = sum(
                pair_utility(
                    utilities,
                    left,
                    right,
                )
                for left in source_members
                for right in target_members
            )

        result.append(
            {
                "source_baseline_identity_id": source_id,
                "target_baseline_identity_id": target_id,
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
                "boundary_aligned": boundary_aligned,
                "whole_cluster_hard_conflict_count": (
                    len(hard_conflicts)
                ),
                "whole_cluster_hard_conflicts": [
                    list(pair)
                    for pair in hard_conflicts[:10]
                ],
                "whole_clique_merge_delta": (
                    cross_clique_delta
                ),
            }
        )

    result.sort(
        key=lambda row: (
            row["utility"],
            row["evidence_confidence"],
            row["rank_score"],
            -row["gap_s"],
        ),
        reverse=True,
    )
    return result


def _best_alternative_margin(
    *,
    selected: Mapping[str, Any],
    candidates: Sequence[Mapping[str, Any]],
    side: str,
) -> tuple[
    Mapping[str, Any] | None,
    float | None,
]:
    selected_pair = canonical_pair(
        str(selected["predecessor_tracklet_id"]),
        str(selected["successor_tracklet_id"]),
    )

    if side == "outgoing":
        key_name = "source_baseline_identity_id"
        key_value = selected[key_name]
    elif side == "incoming":
        key_name = "target_baseline_identity_id"
        key_value = selected[key_name]
    else:
        raise ValueError(
            "side must be outgoing or incoming"
        )

    alternatives = [
        row
        for row in candidates
        if row[key_name] == key_value
        and row["boundary_aligned"]
        and row["whole_cluster_hard_conflict_count"] == 0
        and row["whole_clique_merge_delta"] is not None
        and row["whole_clique_merge_delta"] <= _EPS
        and canonical_pair(
            str(row["predecessor_tracklet_id"]),
            str(row["successor_tracklet_id"]),
        )
        != selected_pair
    ]
    if not alternatives:
        return None, None

    best = max(
        alternatives,
        key=lambda row: (
            row["utility"],
            row["evidence_confidence"],
            row["rank_score"],
            -row["gap_s"],
        ),
    )
    return (
        best,
        float(selected["utility"])
        - float(best["utility"]),
    )


def _selected_bridge_graph_components(
    *,
    baseline_ids: Sequence[str],
    bridge_pairs: Sequence[tuple[str, str]],
) -> tuple[
    list[list[str]],
    bool,
]:
    parent = {
        identity_id: identity_id
        for identity_id in baseline_ids
    }

    def find(identity_id: str) -> str:
        while parent[identity_id] != identity_id:
            parent[identity_id] = parent[
                parent[identity_id]
            ]
            identity_id = parent[identity_id]
        return identity_id

    acyclic = True
    for left, right in bridge_pairs:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            acyclic = False
            continue
        keep, drop = sorted(
            (root_left, root_right)
        )
        parent[drop] = keep

    groups: dict[str, list[str]] = {}
    for identity_id in baseline_ids:
        root = find(identity_id)
        groups.setdefault(root, []).append(
            identity_id
        )

    return (
        [
            sorted(group)
            for group in groups.values()
        ],
        acyclic,
    )


def audit_backbone_partition(
    *,
    final_identities: Sequence[Mapping[str, Any]],
    baseline_identities: Sequence[Mapping[str, Any]],
    selected_backbone_links: Sequence[SelectedBackboneLink],
    spans: Mapping[str, TrackletSpan],
    utilities: Mapping[tuple[str, str], PairUtility],
    hard: frozenset[tuple[str, str]],
    candidate_edges: Mapping[
        tuple[str, str],
        Mapping[str, Any],
    ],
    solver_summary: Mapping[str, Any],
    max_exact_split_size: int = 16,
    boundary_tolerance_s: float = 0.25,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
]:
    """Audit the backbone solution using its actual objective semantics.

    The old 3C clique diagnostics remain useful for the *baseline blocks*.
    However, a profitable all-pairs split of a backbone-augmented identity is
    not automatically an error: that is precisely the trajectory-vs-clique
    disagreement the V1 solver is designed to resolve.

    This audit therefore separates:
      - legacy clique stability of baseline blocks;
      - physical/topological safety of the final partition;
      - evidence quality of each selected backbone bridge;
      - omitted strong boundary handoffs.
    """
    baseline_members = _identity_members_map(
        baseline_identities
    )
    final_members = _identity_members_map(
        final_identities
    )

    if set(
        _assignment_from_members(baseline_members)
    ) != set(spans):
        raise ValueError(
            "baseline partition does not cover tracklets"
        )
    if set(
        _assignment_from_members(final_members)
    ) != set(spans):
        raise ValueError(
            "final partition does not cover tracklets"
        )

    (
        baseline_to_final,
        final_to_baseline,
        preservation_violations,
    ) = _baseline_block_mapping(
        baseline_members=baseline_members,
        final_members=final_members,
    )

    baseline_hard_violations = (
        _hard_violations_for_partition(
            baseline_members,
            hard,
        )
    )
    final_hard_violations = (
        _hard_violations_for_partition(
            final_members,
            hard,
        )
    )

    # The clique baseline is still expected to be stable under the old 3C
    # objective. This is a guardrail around the new augmentation layer.
    (
        baseline_identity_audits,
        baseline_tracklet_audits,
        baseline_summary,
    ) = audit_partition(
        baseline_identities,
        spans,
        utilities,
        hard,
        max_exact_split_size=max_exact_split_size,
    )

    baseline_assignment = (
        _assignment_from_members(
            baseline_members
        )
    )
    final_assignment = _assignment_from_members(
        final_members
    )

    boundary_candidates = _boundary_candidate_rows(
        baseline_members=baseline_members,
        spans=spans,
        utilities=utilities,
        hard=hard,
        candidate_edges=candidate_edges,
        boundary_tolerance_s=boundary_tolerance_s,
    )

    strong_threshold = _finite_float(
        "strong_utility_threshold",
        solver_summary[
            "strong_utility_threshold"
        ],
    )

    selected_pair_set = {
        link.pair
        for link in selected_backbone_links
    }

    # Recompute reciprocal-best semantics among eligible boundary
    # trajectory-vs-clique conflicts.
    eligible = [
        row
        for row in boundary_candidates
        if row["utility"] >= strong_threshold - _EPS
        and row["boundary_aligned"]
        and row[
            "whole_cluster_hard_conflict_count"
        ] == 0
        and row["whole_clique_merge_delta"] is not None
        and row["whole_clique_merge_delta"] <= _EPS
    ]

    best_outgoing: dict[str, Mapping[str, Any]] = {}
    best_incoming: dict[str, Mapping[str, Any]] = {}

    def candidate_key(
        row: Mapping[str, Any],
    ) -> tuple[Any, ...]:
        return (
            row["utility"],
            row["evidence_confidence"],
            row["rank_score"],
            -row["gap_s"],
            row["predecessor_tracklet_id"],
            row["successor_tracklet_id"],
        )

    for row in eligible:
        source = row[
            "source_baseline_identity_id"
        ]
        target = row[
            "target_baseline_identity_id"
        ]
        if (
            source not in best_outgoing
            or candidate_key(row)
            > candidate_key(
                best_outgoing[source]
            )
        ):
            best_outgoing[source] = row
        if (
            target not in best_incoming
            or candidate_key(row)
            > candidate_key(
                best_incoming[target]
            )
        ):
            best_incoming[target] = row

    reciprocal_eligible = []
    for row in eligible:
        source = row[
            "source_baseline_identity_id"
        ]
        target = row[
            "target_baseline_identity_id"
        ]
        if (
            best_outgoing.get(source) is row
            and best_incoming.get(target) is row
        ):
            reciprocal_eligible.append(row)

    bridge_audits: list[dict[str, Any]] = []
    selected_baseline_bridge_pairs: list[
        tuple[str, str]
    ] = []

    for index, link in enumerate(
        selected_backbone_links,
        start=1,
    ):
        pair = link.pair
        predecessor = (
            link.predecessor_tracklet_id
        )
        successor = (
            link.successor_tracklet_id
        )
        source_baseline_id = baseline_assignment[
            predecessor
        ]
        target_baseline_id = baseline_assignment[
            successor
        ]
        final_predecessor_id = final_assignment[
            predecessor
        ]
        final_successor_id = final_assignment[
            successor
        ]

        selected_baseline_bridge_pairs.append(
            (
                source_baseline_id,
                target_baseline_id,
            )
        )

        source_members = baseline_members[
            source_baseline_id
        ]
        target_members = baseline_members[
            target_baseline_id
        ]
        hard_conflicts = _cluster_hard_conflicts(
            source_members,
            target_members,
            hard,
        )

        edge = candidate_edges.get(pair)
        transition_reachable = bool(
            edge is not None
            and edge.get("status") == "candidate"
            and edge.get("reason")
            == "transition_reachable"
        )
        physics = (
            edge.get("physics")
            if edge is not None
            and isinstance(
                edge.get("physics"),
                Mapping,
            )
            else {}
        )

        scored = utilities.get(pair)
        if scored is None:
            raise ValueError(
                f"selected bridge lacks soft utility: {pair}"
            )

        source_end = max(
            spans[tracklet_id].end_time_s
            for tracklet_id in source_members
        )
        target_start = min(
            spans[tracklet_id].start_time_s
            for tracklet_id in target_members
        )
        source_tail_offset = max(
            0.0,
            source_end
            - spans[predecessor].end_time_s,
        )
        target_head_offset = max(
            0.0,
            spans[successor].start_time_s
            - target_start,
        )
        boundary_aligned = (
            source_tail_offset
            <= boundary_tolerance_s + _EPS
            and target_head_offset
            <= boundary_tolerance_s + _EPS
        )

        selected_row = next(
            (
                row
                for row in boundary_candidates
                if canonical_pair(
                    str(
                        row[
                            "predecessor_tracklet_id"
                        ]
                    ),
                    str(
                        row[
                            "successor_tracklet_id"
                        ]
                    ),
                )
                == pair
            ),
            None,
        )
        if selected_row is None:
            selected_row = {
                "source_baseline_identity_id": (
                    source_baseline_id
                ),
                "target_baseline_identity_id": (
                    target_baseline_id
                ),
                "predecessor_tracklet_id": predecessor,
                "successor_tracklet_id": successor,
                "utility": scored.utility,
                "rank_score": scored.rank_score,
                "evidence_confidence": (
                    scored.evidence_confidence
                ),
            }

        outgoing_alt, outgoing_margin = (
            _best_alternative_margin(
                selected=selected_row,
                candidates=boundary_candidates,
                side="outgoing",
            )
        )
        incoming_alt, incoming_margin = (
            _best_alternative_margin(
                selected=selected_row,
                candidates=boundary_candidates,
                side="incoming",
            )
        )

        reciprocal_best_recomputed = (
            best_outgoing.get(
                source_baseline_id
            )
            is selected_row
            and best_incoming.get(
                target_baseline_id
            )
            is selected_row
        )

        final_id = (
            final_predecessor_id
            if final_predecessor_id
            == final_successor_id
            else None
        )
        expected_union = set(source_members) | set(
            target_members
        )
        final_exact_union = bool(
            final_id is not None
            and set(final_members[final_id])
            == expected_union
        )

        utility_matches = math.isclose(
            scored.utility,
            link.utility,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        rank_matches = math.isclose(
            scored.rank_score,
            link.rank_score,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        confidence_matches = math.isclose(
            scored.evidence_confidence,
            link.evidence_confidence,
            rel_tol=0.0,
            abs_tol=1e-9,
        )

        valid = all(
            (
                source_baseline_id
                != target_baseline_id,
                final_id is not None,
                not hard_conflicts,
                transition_reachable,
                boundary_aligned,
                link.reciprocal_best,
                reciprocal_best_recomputed,
                link.utility
                >= strong_threshold - _EPS,
                utility_matches,
                rank_matches,
                confidence_matches,
                final_exact_union,
            )
        )

        flags = []
        if not valid:
            flags.append(
                "selected_backbone_bridge_validation_failed"
            )
        if outgoing_margin is not None:
            flags.append(
                "has_outgoing_boundary_alternative"
            )
        if incoming_margin is not None:
            flags.append(
                "has_incoming_boundary_alternative"
            )

        bridge_audits.append(
            {
                "schema_version": 1,
                "diagnostics_version": (
                    DIAGNOSTICS_VERSION
                ),
                "bridge_id": (
                    f"backbone-bridge-{index:03d}"
                ),
                "source_baseline_identity_id": (
                    source_baseline_id
                ),
                "target_baseline_identity_id": (
                    target_baseline_id
                ),
                "final_identity_id": final_id,
                "predecessor_tracklet_id": (
                    predecessor
                ),
                "successor_tracklet_id": (
                    successor
                ),
                "utility": link.utility,
                "rank_score": link.rank_score,
                "evidence_confidence": (
                    link.evidence_confidence
                ),
                "strong_utility_threshold": (
                    strong_threshold
                ),
                "utility_margin_over_threshold": (
                    link.utility
                    - strong_threshold
                ),
                "gap_s": link.gap_s,
                "endpoint_distance_m": (
                    link.endpoint_distance_m
                ),
                "required_speed_after_slack_mps": (
                    physics.get(
                        "required_speed_after_slack_mps"
                    )
                ),
                "prediction_error_mean_m": (
                    physics.get(
                        "prediction_error_mean_m"
                    )
                ),
                "velocity_direction_cosine": (
                    physics.get(
                        "velocity_direction_cosine"
                    )
                ),
                "source_tail_offset_s": (
                    source_tail_offset
                ),
                "target_head_offset_s": (
                    target_head_offset
                ),
                "boundary_aligned": (
                    boundary_aligned
                ),
                "whole_cluster_hard_conflict_count": (
                    len(hard_conflicts)
                ),
                "whole_cluster_hard_conflicts": [
                    list(pair)
                    for pair in hard_conflicts
                ],
                "whole_clique_merge_delta": (
                    link.whole_clique_merge_delta
                ),
                "reciprocal_best_artifact": (
                    link.reciprocal_best
                ),
                "reciprocal_best_recomputed": (
                    reciprocal_best_recomputed
                ),
                "outgoing_next_best": (
                    {
                        "target_baseline_identity_id": (
                            outgoing_alt[
                                "target_baseline_identity_id"
                            ]
                        ),
                        "predecessor_tracklet_id": (
                            outgoing_alt[
                                "predecessor_tracklet_id"
                            ]
                        ),
                        "successor_tracklet_id": (
                            outgoing_alt[
                                "successor_tracklet_id"
                            ]
                        ),
                        "utility": (
                            outgoing_alt["utility"]
                        ),
                    }
                    if outgoing_alt is not None
                    else None
                ),
                "outgoing_margin_to_next_best": (
                    outgoing_margin
                ),
                "incoming_next_best": (
                    {
                        "source_baseline_identity_id": (
                            incoming_alt[
                                "source_baseline_identity_id"
                            ]
                        ),
                        "predecessor_tracklet_id": (
                            incoming_alt[
                                "predecessor_tracklet_id"
                            ]
                        ),
                        "successor_tracklet_id": (
                            incoming_alt[
                                "successor_tracklet_id"
                            ]
                        ),
                        "utility": (
                            incoming_alt["utility"]
                        ),
                    }
                    if incoming_alt is not None
                    else None
                ),
                "incoming_margin_to_next_best": (
                    incoming_margin
                ),
                "soft_contract_matches": (
                    utility_matches
                    and rank_matches
                    and confidence_matches
                ),
                "transition_reachable": (
                    transition_reachable
                ),
                "final_identity_exactly_source_plus_target": (
                    final_exact_union
                ),
                "removing_bridge_restores_baseline_blocks": (
                    final_exact_union
                ),
                "bridge_valid": valid,
                "flags": flags,
                "provisional": True,
            }
        )

    # Check that the selected bridge graph itself is acyclic and that every
    # final identity corresponds exactly to one connected component of
    # baseline identities.
    bridge_components, bridge_graph_acyclic = (
        _selected_bridge_graph_components(
            baseline_ids=sorted(
                baseline_members
            ),
            bridge_pairs=(
                selected_baseline_bridge_pairs
            ),
        )
    )
    component_sets = {
        frozenset(component)
        for component in bridge_components
    }
    final_baseline_sets = {
        frozenset(baseline_ids)
        for baseline_ids
        in final_to_baseline.values()
        if baseline_ids
    }
    final_matches_bridge_components = (
        component_sets == final_baseline_sets
    )

    final_identity_audits = []
    for final_id in sorted(final_members):
        members = final_members[final_id]
        timeline = _timeline_audit(
            members,
            spans,
        )
        backbone = _maximum_spanning_tree(
            members,
            utilities,
        )
        pair_values = [
            pair_utility(
                utilities,
                a,
                b,
            )
            for a, b in itertools.combinations(
                members,
                2,
            )
        ]
        legacy_split = _best_exact_two_way_split(
            members,
            utilities,
            max_exact_size=max_exact_split_size,
        )
        baseline_ids = final_to_baseline[
            final_id
        ]
        augmented = len(baseline_ids) > 1

        split_matches_baseline_blocks = False
        if (
            augmented
            and len(baseline_ids) == 2
            and legacy_split.get("evaluated")
        ):
            expected = {
                frozenset(
                    baseline_members[
                        baseline_ids[0]
                    ]
                ),
                frozenset(
                    baseline_members[
                        baseline_ids[1]
                    ]
                ),
            }
            actual = {
                frozenset(
                    legacy_split["left"]
                ),
                frozenset(
                    legacy_split["right"]
                ),
            }
            split_matches_baseline_blocks = (
                expected == actual
            )

        flags: list[str] = []
        if any(
            value < 0
            for value in pair_values
        ):
            flags.append(
                "contains_negative_intra_identity_links"
            )
        if timeline["gap_count"] > 0:
            flags.append(
                "contains_temporal_gaps"
            )
        if (
            timeline[
                "max_exact_observation_concurrency"
            ]
            > 1
        ):
            flags.append(
                "contains_exact_frame_overlap"
            )
        if augmented:
            flags.append(
                "contains_selected_backbone_augmentation"
            )
        if (
            legacy_split.get("profitable")
            and augmented
            and split_matches_baseline_blocks
        ):
            flags.append(
                "legacy_clique_split_expected_by_backbone_objective"
            )
        elif legacy_split.get("profitable"):
            flags.append(
                "unexpected_legacy_clique_split_pressure"
            )

        final_identity_audits.append(
            {
                "schema_version": 1,
                "diagnostics_version": (
                    DIAGNOSTICS_VERSION
                ),
                "identity_id": final_id,
                "tracklet_ids": list(members),
                "tracklet_count": len(members),
                "baseline_identity_ids": (
                    baseline_ids
                ),
                "augmentation_kind": (
                    "selected_backbone_bridge"
                    if augmented
                    else "baseline_cluster"
                ),
                "legacy_clique_objective": (
                    _cluster_objective(
                        members,
                        utilities,
                    )
                ),
                "pair_utility": (
                    _distribution(
                        pair_values
                    )
                ),
                "negative_intra_pair_count": (
                    sum(
                        value < 0
                        for value in pair_values
                    )
                ),
                "timeline": timeline,
                "backbone": backbone,
                "legacy_clique_split_pressure": (
                    {
                        **legacy_split,
                        "matches_baseline_blocks": (
                            split_matches_baseline_blocks
                        ),
                        "treated_as_identity_error": (
                            bool(
                                legacy_split.get(
                                    "profitable"
                                )
                            )
                            and not (
                                augmented
                                and split_matches_baseline_blocks
                            )
                        ),
                    }
                ),
                "selected_backbone_bridge_count": (
                    sum(
                        bridge[
                            "final_identity_id"
                        ]
                        == final_id
                        for bridge in bridge_audits
                    )
                ),
                "flags": flags,
                "provisional": True,
            }
        )

    omitted_reciprocal = [
        row
        for row in reciprocal_eligible
        if canonical_pair(
            str(row["predecessor_tracklet_id"]),
            str(row["successor_tracklet_id"]),
        )
        not in selected_pair_set
    ]

    bridge_failures = [
        bridge
        for bridge in bridge_audits
        if not bridge["bridge_valid"]
    ]
    unexpected_split_pressure = [
        audit
        for audit in final_identity_audits
        if "unexpected_legacy_clique_split_pressure"
        in audit["flags"]
    ]

    summary = {
        "schema_version": 1,
        "diagnostics_version": (
            DIAGNOSTICS_VERSION
        ),
        "provisional": True,
        "identity_stage_completed": False,
        "identity_decisions": False,
        "objective_semantics": (
            "clique_baseline_plus_selected_backbone_bridges"
        ),
        "baseline_identity_count": len(
            baseline_members
        ),
        "final_identity_count": len(
            final_members
        ),
        "tracklet_count": len(spans),
        "selected_backbone_bridge_count": len(
            selected_backbone_links
        ),
        "augmented_final_identity_count": sum(
            len(baseline_ids) > 1
            for baseline_ids in final_to_baseline.values()
        ),
        "baseline_cluster_preservation_violation_count": (
            len(preservation_violations)
        ),
        "baseline_cluster_preservation_violations": (
            preservation_violations
        ),
        "baseline_hard_constraint_violation_count": (
            len(baseline_hard_violations)
        ),
        "final_hard_constraint_violation_count": (
            len(final_hard_violations)
        ),
        "final_hard_constraint_violations": (
            final_hard_violations
        ),
        "bridge_validation_failure_count": len(
            bridge_failures
        ),
        "bridge_graph_acyclic": (
            bridge_graph_acyclic
        ),
        "final_partition_matches_bridge_components": (
            final_matches_bridge_components
        ),
        "strong_utility_threshold": (
            strong_threshold
        ),
        "strong_boundary_conflict_candidate_count": (
            len(eligible)
        ),
        "reciprocal_eligible_boundary_handoff_count": (
            len(reciprocal_eligible)
        ),
        "unselected_reciprocal_eligible_boundary_handoff_count": (
            len(omitted_reciprocal)
        ),
        "unexpected_legacy_clique_split_pressure_count": (
            len(unexpected_split_pressure)
        ),
        "expected_legacy_clique_split_pressure_count": (
            sum(
                "legacy_clique_split_expected_by_backbone_objective"
                in audit["flags"]
                for audit in final_identity_audits
            )
        ),
        "baseline_clique_diagnostics": {
            "profitable_exact_split_identity_count": (
                baseline_summary[
                    "profitable_exact_split_identity_count"
                ]
            ),
            "profitable_single_tracklet_move_count": (
                baseline_summary[
                    "profitable_single_tracklet_move_count"
                ]
            ),
            "profitable_whole_identity_merge_count": (
                baseline_summary[
                    "profitable_whole_identity_merge_count"
                ]
            ),
        },
        "final_identity_flag_counts": (
            _flag_counts(
                final_identity_audits
            )
        ),
    }

    review_queue = {
        "schema_version": 1,
        "diagnostics_version": (
            DIAGNOSTICS_VERSION
        ),
        "provisional": True,
        "bridge_failures": bridge_failures,
        "selected_backbone_bridges": (
            bridge_audits
        ),
        "unselected_reciprocal_eligible_boundary_handoffs": (
            omitted_reciprocal
        ),
        "unexpected_legacy_clique_split_pressure": [
            {
                "identity_id": audit[
                    "identity_id"
                ],
                "tracklet_ids": audit[
                    "tracklet_ids"
                ],
                "baseline_identity_ids": audit[
                    "baseline_identity_ids"
                ],
                "legacy_clique_split_pressure": audit[
                    "legacy_clique_split_pressure"
                ],
            }
            for audit in unexpected_split_pressure
        ],
        "weakest_final_backbones": [
            {
                "identity_id": audit[
                    "identity_id"
                ],
                "baseline_identity_ids": audit[
                    "baseline_identity_ids"
                ],
                "backbone": audit[
                    "backbone"
                ],
            }
            for audit in sorted(
                (
                    row
                    for row in final_identity_audits
                    if row["tracklet_count"] > 1
                ),
                key=lambda row: (
                    row["backbone"][
                        "weakest_utility"
                    ],
                    row["backbone"][
                        "mean_utility"
                    ],
                ),
            )[:25]
        ],
    }

    return (
        final_identity_audits,
        bridge_audits,
        baseline_identity_audits,
        summary,
        review_queue,
    )
