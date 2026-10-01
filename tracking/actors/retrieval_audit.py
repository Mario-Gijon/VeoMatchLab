"""Hito 7B.4: recall-oriented audit for sparse actor candidate retrieval.

Sparse retrieval is only useful if it keeps plausible true continuations.
Because VeoMatchLab currently has no hand-labelled cross-tracklet player IDs,
this audit uses two conservative proxy-positive families already present in the
data:

1. LOCAL PHYSICS PROXY
   Consecutive observations inside the same conservative local tracklet are
   treated as known local continuity.  We ask whether the 7B.1 short-gap
   physical envelope would have kept each adjacent transition.

2. GLOBAL ReID SPLIT PROXY
   A sufficiently sampled tracklet is split into early and late appearance
   halves.  One half is queried against the matching other half plus every
   other tracklet's full aggregate profile.  The rank of the true mate measures
   whether a top-K sparse retrieval budget can preserve a known same-person
   relation without using a cosine threshold.

These are diagnostics, not formal player-identity ground truth.  The purpose is
to tune retrieval recall before the global/offline actor solver is allowed to
depend on the sparse graph.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tracking.appearance.evidence import (
    AppearanceEvidenceConfig,
    AppearanceEvidenceSample,
    TrackletAppearanceProfile,
    build_tracklet_profile,
)
from tracking.tracklets.models import Tracklet


RETRIEVAL_AUDIT_VERSION = "sparse-candidate-recall-audit-v1"


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


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if vector.ndim != 1 or vector.size == 0 or norm <= 1e-12:
        raise ValueError("embedding must be a non-zero 1-D vector")
    if not np.all(np.isfinite(vector)):
        raise ValueError("embedding contains non-finite values")
    return vector / norm


@dataclass(frozen=True, slots=True)
class RetrievalAuditConfig:
    local_gap_horizon_s: float = 1.0
    max_transition_speed_mps: float = 12.0
    transition_position_slack_m: float = 1.5
    min_split_samples: int = 4
    rank_cutoffs: tuple[int, ...] = (1, 5, 10, 20, 32, 64)
    worst_example_count: int = 20

    def __post_init__(self) -> None:
        if self.local_gap_horizon_s <= 0.0:
            raise ValueError("local_gap_horizon_s must be > 0")
        if self.max_transition_speed_mps <= 0.0:
            raise ValueError("max_transition_speed_mps must be > 0")
        if self.transition_position_slack_m <= 0.0:
            raise ValueError("transition_position_slack_m must be > 0")
        if self.min_split_samples < 4:
            raise ValueError("min_split_samples must be >= 4")
        if not self.rank_cutoffs:
            raise ValueError("rank_cutoffs must not be empty")
        if any(value < 1 for value in self.rank_cutoffs):
            raise ValueError("rank cutoffs must be >= 1")
        if tuple(sorted(set(self.rank_cutoffs))) != self.rank_cutoffs:
            raise ValueError(
                "rank_cutoffs must be unique and strictly increasing"
            )
        if self.worst_example_count < 1:
            raise ValueError("worst_example_count must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LocalPhysicsAudit:
    eligible_adjacent_transition_count: int
    recovered_transition_count: int
    missed_transition_count: int
    recall: float | None
    gap_s: Mapping[str, Any]
    endpoint_distance_m: Mapping[str, Any]
    required_speed_after_slack_mps: Mapping[str, Any]
    worst_misses: tuple[Mapping[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_adjacent_transition_count": (
                self.eligible_adjacent_transition_count
            ),
            "recovered_transition_count": self.recovered_transition_count,
            "missed_transition_count": self.missed_transition_count,
            "recall": self.recall,
            "gap_s": dict(self.gap_s),
            "endpoint_distance_m": dict(self.endpoint_distance_m),
            "required_speed_after_slack_mps": dict(
                self.required_speed_after_slack_mps
            ),
            "worst_misses": [
                dict(item) for item in self.worst_misses
            ],
        }


@dataclass(frozen=True, slots=True)
class ReIDSplitQuery:
    tracklet_id: str
    query_half: str
    mate_rank: int
    mate_cosine: float
    gallery_size: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "query_half": self.query_half,
            "mate_rank": self.mate_rank,
            "mate_cosine": self.mate_cosine,
            "gallery_size": self.gallery_size,
        }


@dataclass(frozen=True, slots=True)
class ReIDSplitAudit:
    eligible_tracklet_count: int
    query_count: int
    recall_at: Mapping[str, float | None]
    mean_reciprocal_rank: float | None
    rank: Mapping[str, Any]
    mate_cosine: Mapping[str, Any]
    worst_queries: tuple[ReIDSplitQuery, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible_tracklet_count": self.eligible_tracklet_count,
            "query_count": self.query_count,
            "recall_at": dict(self.recall_at),
            "mean_reciprocal_rank": self.mean_reciprocal_rank,
            "rank": dict(self.rank),
            "mate_cosine": dict(self.mate_cosine),
            "worst_queries": [
                item.to_dict() for item in self.worst_queries
            ],
        }


@dataclass(frozen=True, slots=True)
class SparseCandidateRecallAudit:
    local_physics: LocalPhysicsAudit
    reid_split: ReIDSplitAudit
    config: RetrievalAuditConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "audit_version": RETRIEVAL_AUDIT_VERSION,
            "local_physics": self.local_physics.to_dict(),
            "reid_split": self.reid_split.to_dict(),
            "config": self.config.to_dict(),
            "diagnostic_only": True,
            "formal_player_ground_truth": False,
        }


def audit_local_physics(
    tracklets: Iterable[Tracklet],
    config: RetrievalAuditConfig,
) -> LocalPhysicsAudit:
    gaps: list[float] = []
    distances: list[float] = []
    required_speeds: list[float] = []
    misses: list[dict[str, Any]] = []
    recovered = 0
    eligible = 0

    for tracklet in tracklets:
        for first, second in zip(
            tracklet.observations,
            tracklet.observations[1:],
        ):
            gap_s = second.timestamp_s - first.timestamp_s
            if (
                gap_s < -1e-9
                or gap_s > config.local_gap_horizon_s + 1e-9
            ):
                continue
            gap_s = max(0.0, gap_s)
            distance_m = math.hypot(
                second.x - first.x,
                second.z - first.z,
            )
            limit_m = (
                config.transition_position_slack_m
                + config.max_transition_speed_mps * gap_s
            )
            if gap_s > 1e-12:
                required_speed = max(
                    0.0,
                    distance_m
                    - config.transition_position_slack_m,
                ) / gap_s
            else:
                required_speed = (
                    0.0
                    if distance_m
                    <= config.transition_position_slack_m
                    else math.inf
                )

            eligible += 1
            gaps.append(gap_s)
            distances.append(distance_m)
            if math.isfinite(required_speed):
                required_speeds.append(required_speed)

            if distance_m <= limit_m + 1e-9:
                recovered += 1
            else:
                misses.append(
                    {
                        "tracklet_id": tracklet.tracklet_id,
                        "first_frame": first.frame_index,
                        "second_frame": second.frame_index,
                        "gap_s": gap_s,
                        "endpoint_distance_m": distance_m,
                        "reachability_limit_m": limit_m,
                        "required_speed_after_slack_mps": (
                            required_speed
                            if math.isfinite(required_speed)
                            else None
                        ),
                        "first_confidence": first.confidence,
                        "second_confidence": second.confidence,
                        "margin_m": limit_m - distance_m,
                    }
                )

    misses.sort(
        key=lambda item: (
            item["margin_m"],
            item["tracklet_id"],
            item["first_frame"],
        )
    )
    return LocalPhysicsAudit(
        eligible_adjacent_transition_count=eligible,
        recovered_transition_count=recovered,
        missed_transition_count=len(misses),
        recall=(
            recovered / eligible
            if eligible
            else None
        ),
        gap_s=_distribution(gaps),
        endpoint_distance_m=_distribution(distances),
        required_speed_after_slack_mps=_distribution(required_speeds),
        worst_misses=tuple(
            misses[: config.worst_example_count]
        ),
    )


def _clone_samples_for_half(
    samples: Sequence[AppearanceEvidenceSample],
    *,
    pseudo_tracklet_id: str,
) -> tuple[AppearanceEvidenceSample, ...]:
    return tuple(
        AppearanceEvidenceSample(
            sample_id=f"{pseudo_tracklet_id}:{sample.sample_id}",
            tracklet_id=pseudo_tracklet_id,
            detection_id=sample.detection_id,
            frame_index=sample.frame_index,
            timestamp_s=sample.timestamp_s,
            quality=sample.quality,
            embedding=sample.embedding,
            crop_ref=sample.crop_ref,
            view_name=sample.view_name,
        )
        for sample in samples
    )


def _rank_true_mate(
    query: TrackletAppearanceProfile,
    mate: TrackletAppearanceProfile,
    distractors: Sequence[TrackletAppearanceProfile],
) -> tuple[int, float, int]:
    query_vector = _unit(query.aggregate_embedding)
    rows: list[tuple[str, float, bool]] = []

    mate_cosine = float(
        np.clip(
            np.dot(query_vector, _unit(mate.aggregate_embedding)),
            -1.0,
            1.0,
        )
    )
    rows.append((mate.tracklet_id, mate_cosine, True))

    for distractor in distractors:
        cosine = float(
            np.clip(
                np.dot(
                    query_vector,
                    _unit(distractor.aggregate_embedding),
                ),
                -1.0,
                1.0,
            )
        )
        rows.append((distractor.tracklet_id, cosine, False))

    rows.sort(key=lambda row: (-row[1], row[0]))
    for rank, (_, _, is_mate) in enumerate(rows, start=1):
        if is_mate:
            return rank, mate_cosine, len(rows)

    raise AssertionError("true mate vanished from gallery")


def audit_reid_split_retrieval(
    samples: Iterable[AppearanceEvidenceSample],
    full_profiles: Iterable[TrackletAppearanceProfile],
    evidence_config: AppearanceEvidenceConfig,
    audit_config: RetrievalAuditConfig,
) -> ReIDSplitAudit:
    grouped: dict[str, list[AppearanceEvidenceSample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.tracklet_id].append(sample)

    full_profile_map = {
        profile.tracklet_id: profile
        for profile in full_profiles
    }

    eligible_ids = sorted(
        tracklet_id
        for tracklet_id, values in grouped.items()
        if (
            len(values) >= audit_config.min_split_samples
            and tracklet_id in full_profile_map
        )
    )

    queries: list[ReIDSplitQuery] = []

    for tracklet_id in eligible_ids:
        ordered = sorted(
            grouped[tracklet_id],
            key=lambda item: (
                item.timestamp_s,
                item.frame_index,
                item.sample_id,
            ),
        )
        midpoint = len(ordered) // 2
        early_samples = ordered[:midpoint]
        late_samples = ordered[midpoint:]
        if len(early_samples) < 2 or len(late_samples) < 2:
            continue

        early_id = f"{tracklet_id}::early"
        late_id = f"{tracklet_id}::late"
        early_profile = build_tracklet_profile(
            _clone_samples_for_half(
                early_samples,
                pseudo_tracklet_id=early_id,
            ),
            evidence_config,
        )
        late_profile = build_tracklet_profile(
            _clone_samples_for_half(
                late_samples,
                pseudo_tracklet_id=late_id,
            ),
            evidence_config,
        )

        distractors = [
            profile
            for other_id, profile in full_profile_map.items()
            if other_id != tracklet_id
        ]

        early_rank, early_cosine, early_gallery = _rank_true_mate(
            early_profile,
            late_profile,
            distractors,
        )
        late_rank, late_cosine, late_gallery = _rank_true_mate(
            late_profile,
            early_profile,
            distractors,
        )

        queries.append(
            ReIDSplitQuery(
                tracklet_id=tracklet_id,
                query_half="early",
                mate_rank=early_rank,
                mate_cosine=early_cosine,
                gallery_size=early_gallery,
            )
        )
        queries.append(
            ReIDSplitQuery(
                tracklet_id=tracklet_id,
                query_half="late",
                mate_rank=late_rank,
                mate_cosine=late_cosine,
                gallery_size=late_gallery,
            )
        )

    ranks = [float(item.mate_rank) for item in queries]
    cosines = [item.mate_cosine for item in queries]

    recall_at: dict[str, float | None] = {}
    for cutoff in audit_config.rank_cutoffs:
        recall_at[str(cutoff)] = (
            sum(
                1
                for item in queries
                if item.mate_rank <= cutoff
            )
            / len(queries)
            if queries
            else None
        )

    worst = sorted(
        queries,
        key=lambda item: (
            -item.mate_rank,
            item.mate_cosine,
            item.tracklet_id,
            item.query_half,
        ),
    )[: audit_config.worst_example_count]

    return ReIDSplitAudit(
        eligible_tracklet_count=len(queries) // 2,
        query_count=len(queries),
        recall_at=recall_at,
        mean_reciprocal_rank=(
            float(
                np.mean(
                    [
                        1.0 / item.mate_rank
                        for item in queries
                    ]
                )
            )
            if queries
            else None
        ),
        rank=_distribution(ranks),
        mate_cosine=_distribution(cosines),
        worst_queries=tuple(worst),
    )


def run_sparse_candidate_recall_audit(
    *,
    tracklets: Iterable[Tracklet],
    samples: Iterable[AppearanceEvidenceSample],
    full_profiles: Iterable[TrackletAppearanceProfile],
    evidence_config: AppearanceEvidenceConfig,
    audit_config: RetrievalAuditConfig | None = None,
) -> SparseCandidateRecallAudit:
    selected = audit_config or RetrievalAuditConfig()
    tracklet_rows = tuple(tracklets)
    sample_rows = tuple(samples)
    profile_rows = tuple(full_profiles)

    return SparseCandidateRecallAudit(
        local_physics=audit_local_physics(
            tracklet_rows,
            selected,
        ),
        reid_split=audit_reid_split_retrieval(
            sample_rows,
            profile_rows,
            evidence_config,
            selected,
        ),
        config=selected,
    )
