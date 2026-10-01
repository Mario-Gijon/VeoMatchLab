"""Hito 7B.3: unified sparse candidate graph for offline football tracking.

This module joins the two complementary retrieval channels built in Hito 7B:

1. local field-space continuity for very short disappearances;
2. global ReID top-K retrieval for long/ambiguous disappearances.

It also carries exact coexistence conflicts and short-gap physical
impossibilities as hard constraints.

The result is still *not* an identity solution.  A candidate edge means only:

    "this pair deserves to remain available to the later global/offline solver"

and a hard conflict means:

    "these two tracklets cannot belong to the same physical actor under the
     validated field-space constraints."

Design principles
-----------------
* candidate recall is preferred over premature identity decisions;
* hard field physics overrides appearance;
* absence of a candidate is NOT a cannot-link;
* missing ReID is unknown, not negative;
* rich sample-to-sample ReID evidence is computed only after the sparse union;
* no team, jersey number, active/missing actor state or population forcing is
  performed here;
* all evidence is preserved so future offline solving can revise provisional
  associations using later observations.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import math
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from tracking.actors.index import (
    CoobservedPairEvidence,
    HardConflict,
    SparseTrackletIndex,
    TransitionFeatures,
)
from tracking.actors.reid_retrieval import (
    SparseReIDCandidate,
    SparseReIDRetrieval,
)
from tracking.appearance.evidence import (
    TrackletAppearanceProfile,
    TrackletPairAppearanceEvidence,
    compare_tracklet_profiles,
)


UNIFIED_CANDIDATE_VERSION = "sparse-actor-candidates-v1"


def _canonical_pair(first: str, second: str) -> tuple[str, str]:
    if first == second:
        raise ValueError("pair requires two distinct tracklets")
    return (first, second) if first < second else (second, first)


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
class UnifiedCandidateConfig:
    """Candidate-union policy.

    ``local_gap_horizon_s`` is intentionally short.  It is a retrieval horizon,
    never a statement that a player cannot reappear later.  Long reacquisition
    belongs to global ReID/team/jersey/actor-context evidence.
    """

    local_gap_horizon_s: float = 1.0
    include_close_overlap: bool = True
    add_short_gap_unreachable_as_hard: bool = True
    compute_rich_appearance: bool = True

    def __post_init__(self) -> None:
        if not math.isfinite(self.local_gap_horizon_s):
            raise ValueError("local_gap_horizon_s must be finite")
        if self.local_gap_horizon_s <= 0.0:
            raise ValueError("local_gap_horizon_s must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnifiedHardConflict:
    tracklet_a: str
    tracklet_b: str
    reasons: tuple[str, ...]
    evidence: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("hard conflict requires canonical ordering")
        if not self.reasons:
            raise ValueError("hard conflict requires at least one reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate_version": UNIFIED_CANDIDATE_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "reasons": list(self.reasons),
            "hard_constraint": True,
            "identity_decision": "cannot_link",
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class UnifiedCandidate:
    tracklet_a: str
    tracklet_b: str
    reasons: tuple[str, ...]
    local_transition: Mapping[str, Any] | None
    close_overlap: Mapping[str, Any] | None
    appearance_retrieval: Mapping[str, Any] | None
    appearance_evidence: Mapping[str, Any] | None

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("candidate requires canonical ordering")
        if not self.reasons:
            raise ValueError("candidate requires at least one reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "candidate_version": UNIFIED_CANDIDATE_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "reasons": list(self.reasons),
            "local_transition": (
                dict(self.local_transition)
                if self.local_transition is not None
                else None
            ),
            "close_overlap": (
                dict(self.close_overlap)
                if self.close_overlap is not None
                else None
            ),
            "appearance_retrieval": (
                dict(self.appearance_retrieval)
                if self.appearance_retrieval is not None
                else None
            ),
            "appearance_evidence": (
                dict(self.appearance_evidence)
                if self.appearance_evidence is not None
                else None
            ),
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class UnifiedCandidateGraph:
    candidates: tuple[UnifiedCandidate, ...]
    hard_conflicts: tuple[UnifiedHardConflict, ...]
    total_tracklet_count: int
    profiled_tracklet_count: int
    config: UnifiedCandidateConfig

    def summary(self) -> dict[str, Any]:
        total_possible = (
            self.total_tracklet_count
            * (self.total_tracklet_count - 1)
            // 2
        )
        degree = Counter()
        reason_counts = Counter()
        reason_combinations = Counter()
        appearance_evidence_count = 0

        for candidate in self.candidates:
            degree[candidate.tracklet_a] += 1
            degree[candidate.tracklet_b] += 1
            for reason in candidate.reasons:
                reason_counts[reason] += 1
            reason_combinations["+".join(candidate.reasons)] += 1
            if candidate.appearance_evidence is not None:
                appearance_evidence_count += 1

        hard_reason_counts = Counter()
        for conflict in self.hard_conflicts:
            for reason in conflict.reasons:
                hard_reason_counts[reason] += 1

        degrees = [
            float(degree.get(index, 0))
            for index in sorted(
                {
                    item.tracklet_a
                    for item in self.candidates
                }
                | {
                    item.tracklet_b
                    for item in self.candidates
                }
            )
        ]
        zero_degree = max(
            0,
            self.total_tracklet_count - len(degrees),
        )
        degrees.extend([0.0] * zero_degree)

        return {
            "schema_version": 1,
            "candidate_version": UNIFIED_CANDIDATE_VERSION,
            "tracklet_count": self.total_tracklet_count,
            "profiled_tracklet_count": self.profiled_tracklet_count,
            "unprofiled_tracklet_count": (
                self.total_tracklet_count
                - self.profiled_tracklet_count
            ),
            "total_possible_pair_count": total_possible,
            "hard_conflict_count": len(self.hard_conflicts),
            "candidate_count": len(self.candidates),
            "candidate_fraction_of_all_pairs": (
                len(self.candidates) / total_possible
                if total_possible
                else None
            ),
            "pair_reduction_fraction": (
                1.0 - len(self.candidates) / total_possible
                if total_possible
                else None
            ),
            "candidate_reason_counts": dict(
                sorted(reason_counts.items())
            ),
            "candidate_reason_combinations": dict(
                sorted(reason_combinations.items())
            ),
            "hard_conflict_reason_counts": dict(
                sorted(hard_reason_counts.items())
            ),
            "candidate_with_rich_appearance_count": (
                appearance_evidence_count
            ),
            "candidate_without_rich_appearance_count": (
                len(self.candidates) - appearance_evidence_count
            ),
            "candidate_degree": _distribution(degrees),
            "config": self.config.to_dict(),
            "identity_decisions": False,
            "absence_of_candidate_is_cannot_link": False,
            "hard_physics_overrides_appearance": True,
        }


def _merge_hard_conflict(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    tracklet_a: str,
    tracklet_b: str,
    reason: str,
    evidence_key: str,
    evidence: Mapping[str, Any],
) -> None:
    pair = _canonical_pair(tracklet_a, tracklet_b)
    row = rows.setdefault(
        pair,
        {
            "reasons": set(),
            "evidence": {},
        },
    )
    row["reasons"].add(reason)
    row["evidence"][evidence_key] = dict(evidence)


def _build_hard_conflicts(
    physical_index: SparseTrackletIndex,
    config: UnifiedCandidateConfig,
) -> tuple[
    tuple[UnifiedHardConflict, ...],
    set[tuple[str, str]],
]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    for conflict in physical_index.hard_coexistence_conflicts():
        _merge_hard_conflict(
            rows,
            tracklet_a=conflict.tracklet_a,
            tracklet_b=conflict.tracklet_b,
            reason="simultaneous_spatial_separation",
            evidence_key="coexistence",
            evidence=conflict.to_dict(),
        )

    if config.add_short_gap_unreachable_as_hard:
        for transition in physical_index.iter_local_transition_features(
            horizon_s=config.local_gap_horizon_s,
        ):
            if transition.physically_reachable:
                continue
            _merge_hard_conflict(
                rows,
                tracklet_a=transition.predecessor_tracklet_id,
                tracklet_b=transition.successor_tracklet_id,
                reason="short_gap_unreachable",
                evidence_key="short_gap_transition",
                evidence=transition.to_dict(),
            )

    conflicts = tuple(
        UnifiedHardConflict(
            tracklet_a=pair[0],
            tracklet_b=pair[1],
            reasons=tuple(sorted(payload["reasons"])),
            evidence=dict(payload["evidence"]),
        )
        for pair, payload in sorted(rows.items())
    )
    return conflicts, set(rows)


def build_unified_candidate_graph(
    *,
    physical_index: SparseTrackletIndex,
    reid_retrieval: SparseReIDRetrieval,
    profiles: Iterable[TrackletAppearanceProfile],
    config: UnifiedCandidateConfig | None = None,
    appearance_compare_fn: Callable[
        [TrackletAppearanceProfile, TrackletAppearanceProfile],
        TrackletPairAppearanceEvidence,
    ] = compare_tracklet_profiles,
) -> UnifiedCandidateGraph:
    selected = config or UnifiedCandidateConfig()

    ordered_profiles = tuple(profiles)
    profile_map = {
        profile.tracklet_id: profile
        for profile in ordered_profiles
    }
    if len(profile_map) != len(ordered_profiles):
        raise ValueError("duplicate tracklet appearance profile")

    physical_ids = {
        node.tracklet_id
        for node in physical_index.nodes
    }
    if not set(profile_map).issubset(physical_ids):
        unknown = sorted(set(profile_map) - physical_ids)
        raise ValueError(
            f"appearance profiles reference unknown tracklets: {unknown[:10]}"
        )

    conflicts, hard_pairs = _build_hard_conflicts(
        physical_index,
        selected,
    )

    candidate_rows: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {}

    def candidate_row(
        first: str,
        second: str,
    ) -> dict[str, Any] | None:
        pair = _canonical_pair(first, second)
        if pair in hard_pairs:
            return None
        return candidate_rows.setdefault(
            pair,
            {
                "reasons": set(),
                "local_transition": None,
                "close_overlap": None,
                "appearance_retrieval": None,
            },
        )

    # 1) very-short physical continuations
    for transition in physical_index.iter_local_transition_features(
        horizon_s=selected.local_gap_horizon_s,
    ):
        if not transition.physically_reachable:
            continue
        row = candidate_row(
            transition.predecessor_tracklet_id,
            transition.successor_tracklet_id,
        )
        if row is None:
            continue
        row["reasons"].add("local_physics")
        row["local_transition"] = transition.to_dict()

    # 2) exact close overlaps that were not hard-separated
    if selected.include_close_overlap:
        for overlap in physical_index.close_coobserved_pairs():
            row = candidate_row(
                overlap.tracklet_a,
                overlap.tracklet_b,
            )
            if row is None:
                continue
            row["reasons"].add("close_overlap")
            row["close_overlap"] = overlap.to_dict()

    # 3) global appearance top-K
    for appearance in reid_retrieval.candidates:
        row = candidate_row(
            appearance.tracklet_a,
            appearance.tracklet_b,
        )
        if row is None:
            continue
        row["reasons"].add("appearance_topk")
        row["appearance_retrieval"] = appearance.to_dict()

    candidates: list[UnifiedCandidate] = []
    for pair, payload in sorted(candidate_rows.items()):
        rich = None
        if selected.compute_rich_appearance:
            profile_a = profile_map.get(pair[0])
            profile_b = profile_map.get(pair[1])
            if profile_a is not None and profile_b is not None:
                rich = appearance_compare_fn(
                    profile_a,
                    profile_b,
                ).to_dict()

        candidates.append(
            UnifiedCandidate(
                tracklet_a=pair[0],
                tracklet_b=pair[1],
                reasons=tuple(sorted(payload["reasons"])),
                local_transition=payload["local_transition"],
                close_overlap=payload["close_overlap"],
                appearance_retrieval=payload[
                    "appearance_retrieval"
                ],
                appearance_evidence=rich,
            )
        )

    candidate_pairs = {
        (item.tracklet_a, item.tracklet_b)
        for item in candidates
    }
    overlap = candidate_pairs & hard_pairs
    if overlap:
        raise AssertionError(
            "hard conflict survived unified candidate construction"
        )

    return UnifiedCandidateGraph(
        candidates=tuple(candidates),
        hard_conflicts=conflicts,
        total_tracklet_count=physical_index.tracklet_count,
        profiled_tracklet_count=len(profile_map),
        config=selected,
    )
