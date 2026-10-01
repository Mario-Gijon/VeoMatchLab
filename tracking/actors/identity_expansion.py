"""Tracking V1 Hito 7E.4: seeded persistent-identity expansion.

This stage grows the conservative persistent identities created by Hito 7E.3B.

Core idea
---------
The previous solver intentionally accepted only a very small mutual-best core.
Now we aggregate all chain-level ReID evidence already contained in each
provisional identity and let *established* identities absorb additional
sequential fragments.

An identity is considered an initial seed when it has ReID evidence from at
least ``min_seed_profiled_tracklets`` distinct conservative tracklets.  The
solver may then grow that seeded component outward through strong candidates.
Two completely unseeded components are never joined in this stage.

Safety rules remain unchanged:
* no forced population count;
* only sequential candidates;
* maximum one predecessor and one successor per source identity;
* different anonymous-kit identities rejected by default;
* any hard tracklet cannot-link blocks a component merge transitively;
* trusted positives come from Hito 7D same-tracklet splits;
* fragmentation is preferred over contamination.

The output is still provisional and lives under diagnostics.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


IDENTITY_EXPANSION_VERSION = "seeded-identity-expansion-v3-family-guard"


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
    a = _unit(first, label="first embedding")
    b = _unit(second, label="second embedding")
    if a.shape != b.shape:
        raise ValueError("embedding dimensions differ")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _geometric_mean(values: Sequence[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("geometric mean requires values")
    if np.any(~np.isfinite(array)):
        raise ValueError("geometric mean got non-finite input")
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
class IdentityExpansionConfig:
    top_k_successors: int = 16
    top_k_predecessors: int = 16
    one_sided_rank_floor: float = 0.05
    min_rank_score: float = 0.70
    competition_mode: str = "mutual_best"
    min_seed_profiled_tracklets: int = 2
    allow_different_kit: bool = False
    same_primary_bonus: float = 0.035
    different_primary_penalty: float = 0.300
    probable_different_primary_penalty: float = 0.200
    probable_primary_min_dominance: float = 0.60
    probable_primary_min_confidence: float = 0.20
    probable_primary_min_evidence_coverage: float = 0.15
    time_epsilon_s: float = 1e-9

    def __post_init__(self) -> None:
        if self.top_k_successors < 1 or self.top_k_predecessors < 1:
            raise ValueError("top-k values must be >= 1")
        if not 0.0 < self.one_sided_rank_floor <= 1.0:
            raise ValueError("one_sided_rank_floor must be in (0,1]")
        if not 0.0 <= self.min_rank_score <= 1.0:
            raise ValueError("min_rank_score must be in [0,1]")
        if self.competition_mode not in {
            "mutual_best",
            "mutual_top3",
        }:
            raise ValueError(
                "competition_mode must be mutual_best or mutual_top3"
            )
        if self.min_seed_profiled_tracklets < 1:
            raise ValueError(
                "min_seed_profiled_tracklets must be >= 1"
            )
        if not 0.0 <= self.same_primary_bonus <= 0.25:
            raise ValueError(
                "same_primary_bonus must be in [0,0.25]"
            )
        if not 0.0 <= self.different_primary_penalty <= 0.50:
            raise ValueError(
                "different_primary_penalty must be in [0,0.50]"
            )
        if not 0.0 <= self.probable_different_primary_penalty <= 0.50:
            raise ValueError(
                "probable_different_primary_penalty must be in [0,0.50]"
            )
        for name in (
            "probable_primary_min_dominance",
            "probable_primary_min_confidence",
            "probable_primary_min_evidence_coverage",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.time_epsilon_s < 0.0:
            raise ValueError("time_epsilon_s must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class IdentityProfile:
    identity_id: str
    source_actor_ids: tuple[str, ...]
    tracklet_ids: tuple[str, ...]
    profiled_actor_ids: tuple[str, ...]
    profiled_tracklet_count: int
    aggregate_embedding: tuple[float, ...]
    reliability: float
    start_time_s: float
    end_time_s: float
    kit_cluster_id: int | None
    kit_dominance: float | None
    initial_seed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "expansion_version": IDENTITY_EXPANSION_VERSION,
            "identity_id": self.identity_id,
            "source_actor_ids": list(self.source_actor_ids),
            "tracklet_ids": list(self.tracklet_ids),
            "profiled_actor_ids": list(self.profiled_actor_ids),
            "profiled_tracklet_count": self.profiled_tracklet_count,
            "aggregate_embedding": list(self.aggregate_embedding),
            "reliability": self.reliability,
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "kit_cluster_id": self.kit_cluster_id,
            "kit_dominance": self.kit_dominance,
            "initial_seed": self.initial_seed,
        }


@dataclass(frozen=True, slots=True)
class IdentityFamilyEvidence:
    identity_id: str
    state: str
    family_type: str
    family_id: int | None
    candidate_family_type: str | None
    candidate_family_id: int | None
    confidence: float
    dominance: float
    evidence_coverage: float
    usable_for_tracking: bool
    family_distribution: Mapping[str, float]

    @property
    def family_key(self) -> str | None:
        if self.family_type == "unknown" or self.family_id is None:
            return None
        return f"{self.family_type}-{self.family_id}"

    @property
    def candidate_family_key(self) -> str | None:
        if (
            self.candidate_family_type is None
            or self.candidate_family_id is None
        ):
            return None
        return (
            f"{self.candidate_family_type}-{self.candidate_family_id}"
        )

    @property
    def confident_primary(self) -> bool:
        return (
            self.usable_for_tracking
            and self.state == "confident_primary"
            and self.family_type == "primary"
            and self.family_id is not None
        )

    def probable_primary_id(
        self,
        config: "IdentityExpansionConfig",
    ) -> int | None:
        """Return a conservative primary-family candidate, including weak evidence.

        ``weak_primary`` is not promoted to a team label.  It is only allowed
        to express a *contradiction* when both identities independently point
        to different primary families with enough aggregate support.
        """
        if self.state not in {"confident_primary", "weak_primary"}:
            return None
        if (
            self.candidate_family_type != "primary"
            or self.candidate_family_id is None
        ):
            return None
        if self.dominance < config.probable_primary_min_dominance:
            return None
        if self.confidence < config.probable_primary_min_confidence:
            return None
        if (
            self.evidence_coverage
            < config.probable_primary_min_evidence_coverage
        ):
            return None
        return self.candidate_family_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "identity_id": self.identity_id,
            "state": self.state,
            "family_type": self.family_type,
            "family_id": self.family_id,
            "family_key": self.family_key,
            "candidate_family_type": self.candidate_family_type,
            "candidate_family_id": self.candidate_family_id,
            "candidate_family_key": self.candidate_family_key,
            "confidence": self.confidence,
            "dominance": self.dominance,
            "evidence_coverage": self.evidence_coverage,
            "usable_for_tracking": self.usable_for_tracking,
            "family_distribution": dict(self.family_distribution),
        }


@dataclass(frozen=True, slots=True)
class IdentityCalibration:
    positive: tuple[float, ...]
    same_kit_negative: tuple[float, ...]
    different_kit_negative: tuple[float, ...]
    unknown_kit_negative: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.positive:
            raise ValueError("trusted positive calibration is empty")
        if (
            len(self.same_kit_negative)
            + len(self.different_kit_negative)
            + len(self.unknown_kit_negative)
            == 0
        ):
            raise ValueError("identity-level negative calibration is empty")

    def negatives(self, family: str) -> tuple[float, ...]:
        if family == "same_kit" and self.same_kit_negative:
            return self.same_kit_negative
        if family == "different_kit" and self.different_kit_negative:
            return self.different_kit_negative
        if family == "unknown_kit" and self.unknown_kit_negative:
            return self.unknown_kit_negative
        return tuple(sorted(
            self.same_kit_negative
            + self.different_kit_negative
            + self.unknown_kit_negative
        ))

    @staticmethod
    def _cdf(values: Sequence[float], value: float) -> float:
        count = bisect_right(values, float(value))
        return float((count + 0.5) / (len(values) + 1.0))

    def support(self, cosine: float, *, family: str) -> dict[str, float]:
        negatives = self.negatives(family)
        negative_cdf = self._cdf(negatives, cosine)
        positive_cdf = self._cdf(self.positive, cosine)
        support = _geometric_mean(
            [negative_cdf, positive_cdf]
        )
        return {
            "negative_cdf": negative_cdf,
            "negative_tail_fraction": 1.0 - negative_cdf,
            "positive_cdf": positive_cdf,
            "support": support,
        }


@dataclass(frozen=True, slots=True)
class ExpansionCandidate:
    predecessor_identity_id: str
    successor_identity_id: str
    gap_s: float
    aggregate_cosine: float
    kit_family: str
    empirical_support: float
    profile_confidence: float
    forward_retrieval_rank: int | None
    backward_retrieval_rank: int | None
    retrieval_support: float
    base_rank_score: float = 0.0
    family_relation: str = "neutral"
    family_pair_strength: float = 0.0
    family_adjustment: float = 0.0
    family_evidence_source: str = "legacy_chain_kit"
    rank_score: float = 0.0
    outgoing_score_rank: int | None = None
    incoming_score_rank: int | None = None
    mutual_best: bool = False
    mutual_top3: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "expansion_version": IDENTITY_EXPANSION_VERSION,
            "predecessor_identity_id": self.predecessor_identity_id,
            "successor_identity_id": self.successor_identity_id,
            "gap_s": self.gap_s,
            "aggregate_cosine": self.aggregate_cosine,
            "kit_family": self.kit_family,
            "empirical_support": self.empirical_support,
            "profile_confidence": self.profile_confidence,
            "forward_retrieval_rank": self.forward_retrieval_rank,
            "backward_retrieval_rank": self.backward_retrieval_rank,
            "retrieval_support": self.retrieval_support,
            "base_rank_score": self.base_rank_score,
            "family_relation": self.family_relation,
            "family_pair_strength": self.family_pair_strength,
            "family_adjustment": self.family_adjustment,
            "family_evidence_source": self.family_evidence_source,
            "rank_score": self.rank_score,
            "rank_score_semantics": (
                "relative_identity_reid_plus_soft_family_evidence_not_probability"
            ),
            "outgoing_score_rank": self.outgoing_score_rank,
            "incoming_score_rank": self.incoming_score_rank,
            "mutual_best": self.mutual_best,
            "mutual_top3": self.mutual_top3,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class ExpandedIdentity:
    identity_id: str
    source_identity_ids: tuple[str, ...]
    actor_ids: tuple[str, ...]
    tracklet_ids: tuple[str, ...]
    start_time_s: float
    end_time_s: float
    seeded: bool

    @property
    def source_identity_count(self) -> int:
        return len(self.source_identity_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "expansion_version": IDENTITY_EXPANSION_VERSION,
            "identity_id": self.identity_id,
            "source_identity_ids": list(self.source_identity_ids),
            "source_identity_count": self.source_identity_count,
            "actor_ids": list(self.actor_ids),
            "actor_count": len(self.actor_ids),
            "tracklet_ids": list(self.tracklet_ids),
            "tracklet_count": len(self.tracklet_ids),
            "start_time_s": self.start_time_s,
            "end_time_s": self.end_time_s,
            "duration_s": self.end_time_s - self.start_time_s,
            "seeded": self.seeded,
            "provisional": True,
        }


@dataclass(frozen=True, slots=True)
class IdentityExpansionResult:
    profiles: tuple[IdentityProfile, ...]
    candidates: tuple[ExpansionCandidate, ...]
    selected_candidates: tuple[ExpansionCandidate, ...]
    expanded_identities: tuple[ExpandedIdentity, ...]
    identity_assignments: tuple[Mapping[str, Any], ...]
    actor_assignments: tuple[Mapping[str, Any], ...]
    tracklet_assignments: tuple[Mapping[str, Any], ...]
    rejection_counts: Mapping[str, int]
    hard_identity_pair_count: int
    calibration: IdentityCalibration
    config: IdentityExpansionConfig

    def summary(self) -> dict[str, Any]:
        source_counts = [
            item.source_identity_count
            for item in self.expanded_identities
        ]
        scores = [
            item.rank_score
            for item in self.selected_candidates
        ]
        gaps = [
            item.gap_s
            for item in self.selected_candidates
        ]
        score_bands = Counter()
        for row in self.candidates:
            for threshold in (0.60, 0.65, 0.70, 0.75, 0.80):
                if row.rank_score >= threshold:
                    score_bands[f"ge_{threshold:.2f}"] += 1
                    if row.mutual_best:
                        score_bands[
                            f"mutual_best_ge_{threshold:.2f}"
                        ] += 1
                    if row.mutual_top3:
                        score_bands[
                            f"mutual_top3_ge_{threshold:.2f}"
                        ] += 1

        return {
            "schema_version": 1,
            "expansion_version": IDENTITY_EXPANSION_VERSION,
            "profiled_identity_count": len(self.profiles),
            "initial_seed_count": sum(
                profile.initial_seed
                for profile in self.profiles
            ),
            "candidate_count": len(self.candidates),
            "mutual_best_count": sum(
                row.mutual_best for row in self.candidates
            ),
            "mutual_top3_count": sum(
                row.mutual_top3 for row in self.candidates
            ),
            "score_band_counts": dict(sorted(score_bands.items())),
            "selected_candidate_count": len(
                self.selected_candidates
            ),
            "expanded_identity_count": len(
                self.expanded_identities
            ),
            "non_singleton_expanded_identity_count": sum(
                value > 1 for value in source_counts
            ),
            "largest_source_identity_count": (
                max(source_counts) if source_counts else 0
            ),
            "expanded_source_identity_count": _stats(
                [float(value) for value in source_counts]
            ),
            "selected_score": _stats(scores),
            "selected_gap_s": _stats(gaps),
            "selected_kit_family_counts": dict(sorted(
                Counter(
                    row.kit_family
                    for row in self.selected_candidates
                ).items()
            )),
            "family_relation_counts": dict(sorted(
                Counter(
                    row.family_relation
                    for row in self.candidates
                ).items()
            )),
            "selected_family_relation_counts": dict(sorted(
                Counter(
                    row.family_relation
                    for row in self.selected_candidates
                ).items()
            )),
            "family_adjustment": {
                relation: _stats([
                    row.family_adjustment
                    for row in self.candidates
                    if row.family_relation == relation
                ])
                for relation in sorted({
                    row.family_relation for row in self.candidates
                })
            },
            "rejection_counts": dict(
                sorted(self.rejection_counts.items())
            ),
            "hard_identity_pair_count": self.hard_identity_pair_count,
            "calibration": {
                "trusted_positive": _stats(
                    list(self.calibration.positive)
                ),
                "same_kit_negative": _stats(
                    list(self.calibration.same_kit_negative)
                ),
                "different_kit_negative": _stats(
                    list(self.calibration.different_kit_negative)
                ),
                "unknown_kit_negative": _stats(
                    list(self.calibration.unknown_kit_negative)
                ),
            },
            "config": self.config.to_dict(),
            "population_forced": False,
            "hard_constraints_inviolable": True,
            "different_kit_hard_constraint": False,
            "family_evidence_policy": {
                "same_primary": "small_positive_score_adjustment",
                "different_primary": "strong_negative_score_adjustment",
                "probable_different_primary": (
                    "weak_primary_contradiction_negative_adjustment"
                ),
                "weak_same_primary_or_special": "neutral",
                "hard_constraint": False,
            },
            "family_aware_reid_calibration": (
                "same_primary_hard_negatives_when_family_evidence_available"
            ),
            "identity_reliability_aggregation": (
                "weighted_mean_of_calibrated_chain_reliabilities"
            ),
            "chain_support_penalty_reapplied": False,
            "fragmentation_preferred_over_contamination": True,
            "provisional": True,
            "canonical": False,
        }



def build_identity_family_evidence(
    rows: Sequence[Mapping[str, Any]] | None,
) -> dict[str, IdentityFamilyEvidence]:
    if not rows:
        return {}

    result: dict[str, IdentityFamilyEvidence] = {}
    for row in rows:
        identity_id_value = row.get("identity_id", row.get("entity_id"))
        if identity_id_value is None:
            raise ValueError("identity family row has no identity/entity id")
        identity_id = str(identity_id_value)
        if identity_id in result:
            raise ValueError(
                f"duplicate identity family evidence {identity_id}"
            )

        family_type = str(row.get("family_type", "unknown"))
        family_id_raw = row.get("family_id")
        family_id = (
            None if family_id_raw is None else int(family_id_raw)
        )

        candidate_type_raw = row.get("candidate_family_type")
        candidate_family_type = (
            str(candidate_type_raw)
            if candidate_type_raw is not None
            else (
                family_type
                if family_type in {"primary", "special"}
                else None
            )
        )
        candidate_id_raw = row.get("candidate_family_id")
        candidate_family_id = (
            int(candidate_id_raw)
            if candidate_id_raw is not None
            else (
                family_id
                if candidate_family_type in {"primary", "special"}
                else None
            )
        )

        distribution_raw = row.get("family_distribution", {})
        if not isinstance(distribution_raw, Mapping):
            raise ValueError(
                f"family_distribution for {identity_id} must be a mapping"
            )
        family_distribution = {
            str(key): _clamp01(_finite(
                f"family_distribution[{key}]",
                value,
            ))
            for key, value in distribution_raw.items()
        }

        state = str(row.get("state", "unknown"))
        usable = bool(row.get(
            "usable_for_tracking",
            state in {"confident_primary", "confident_special"},
        ))

        result[identity_id] = IdentityFamilyEvidence(
            identity_id=identity_id,
            state=state,
            family_type=family_type,
            family_id=family_id,
            candidate_family_type=candidate_family_type,
            candidate_family_id=candidate_family_id,
            confidence=_clamp01(_finite(
                "family confidence", row.get("confidence", 0.0)
            )),
            dominance=_clamp01(_finite(
                "family dominance", row.get("dominance", 0.0)
            )),
            evidence_coverage=_clamp01(_finite(
                "family evidence_coverage",
                row.get("evidence_coverage", 0.0),
            )),
            usable_for_tracking=usable,
            family_distribution=family_distribution,
        )
    return result


def family_pair_relation(
    first_identity_id: str,
    second_identity_id: str,
    evidence: Mapping[str, IdentityFamilyEvidence],
    *,
    config: IdentityExpansionConfig | None = None,
) -> tuple[str, float]:
    first = evidence.get(first_identity_id)
    second = evidence.get(second_identity_id)
    if first is None or second is None:
        return "neutral", 0.0

    selected = config or IdentityExpansionConfig()

    pair_strength = _clamp01(
        math.sqrt(first.confidence * second.confidence)
        * math.sqrt(first.dominance * second.dominance)
    )

    # High-confidence primary evidence keeps the original 7E.5 semantics.
    if first.confident_primary and second.confident_primary:
        relation = (
            "same_primary"
            if first.family_id == second.family_id
            else "different_primary"
        )
        return relation, pair_strength

    # 7F.B guard: weak_primary is still NOT a positive team label.  However,
    # when both aggregate identities independently carry enough primary-family
    # evidence and point to *different* anonymous primary families, treating
    # the pair as totally neutral throws away useful contradiction evidence.
    #
    # We only use this weak evidence negatively.  Weak agreement remains
    # neutral, so a noisy colour estimate can never create a merge bonus.
    first_probable = first.probable_primary_id(selected)
    second_probable = second.probable_primary_id(selected)
    if (
        first_probable is not None
        and second_probable is not None
        and first_probable != second_probable
    ):
        return "probable_different_primary", pair_strength

    return "neutral", 0.0


def family_score_adjustment(
    *,
    relation: str,
    pair_strength: float,
    config: IdentityExpansionConfig,
) -> float:
    strength = _clamp01(pair_strength)
    if relation == "same_primary":
        return config.same_primary_bonus * strength
    if relation == "different_primary":
        return -config.different_primary_penalty * strength
    if relation == "probable_different_primary":
        return -config.probable_different_primary_penalty * strength
    return 0.0


def _family_calibration_bucket(
    first: IdentityProfile,
    second: IdentityProfile,
    evidence: Mapping[str, IdentityFamilyEvidence],
) -> str:
    if evidence:
        relation, _strength = family_pair_relation(
            first.identity_id,
            second.identity_id,
            evidence,
        )
        if relation == "same_primary":
            return "same_kit"
        if relation == "different_primary":
            return "different_kit"
        return "unknown_kit"
    return _kit_family(first, second)

def _kit_family(
    first: IdentityProfile,
    second: IdentityProfile,
) -> str:
    if (
        first.kit_cluster_id is None
        or second.kit_cluster_id is None
    ):
        return "unknown_kit"
    return (
        "same_kit"
        if first.kit_cluster_id == second.kit_cluster_id
        else "different_kit"
    )


def build_identity_profiles(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    chain_profile_rows: Sequence[Mapping[str, Any]],
    config: IdentityExpansionConfig,
) -> tuple[IdentityProfile, ...]:
    chain_profiles = {
        str(row["actor_id"]): row
        for row in chain_profile_rows
    }

    result: list[IdentityProfile] = []

    for identity in identity_rows:
        identity_id = str(identity["identity_id"])
        actor_ids = tuple(
            str(value) for value in identity["actor_ids"]
        )
        tracklet_ids = tuple(
            str(value) for value in identity["tracklet_ids"]
        )

        available = [
            chain_profiles[actor_id]
            for actor_id in actor_ids
            if actor_id in chain_profiles
        ]
        if not available:
            continue

        vectors: list[np.ndarray] = []
        weights: list[float] = []
        profiled_tracklet_count = 0
        kit_mass: dict[int, float] = defaultdict(float)

        for row in available:
            embedding = _unit(
                row["aggregate_embedding"],
                label=str(row["actor_id"]),
            )
            reliability = _clamp01(
                _finite("chain reliability", row["reliability"])
            )
            count = int(row["profiled_tracklet_count"])
            if count < 1:
                raise ValueError(
                    "chain profile has invalid profiled_tracklet_count"
                )
            profiled_tracklet_count += count
            weight = max(
                1e-6,
                reliability * math.sqrt(float(count)),
            )
            vectors.append(embedding)
            weights.append(weight)

            cluster = row.get("kit_cluster_id")
            dominance = row.get("kit_dominance")
            if cluster is not None and dominance is not None:
                kit_mass[int(cluster)] += (
                    weight
                    * _clamp01(
                        _finite("kit_dominance", dominance)
                    )
                )

        matrix = np.vstack(vectors)
        weight_array = np.asarray(weights, dtype=np.float64)
        aggregate = np.sum(
            matrix * weight_array[:, None],
            axis=0,
        )
        aggregate = _unit(
            aggregate,
            label=f"identity {identity_id}",
        )

        # ChainProfile.reliability already contains the evidence-support
        # penalty associated with the number/quality of conservative tracklet
        # profiles used to build that chain.  Applying another support factor
        # here would double-penalize singleton identities and collapse scores
        # toward 0.5.  Preserve the calibrated chain scale and aggregate only
        # across independent source chains.
        reliability = _clamp01(
            float(
                np.average(
                    [
                        _clamp01(
                            _finite(
                                "chain reliability",
                                row["reliability"],
                            )
                        )
                        for row in available
                    ],
                    weights=weight_array,
                )
            )
        )

        if kit_mass:
            ordered_kit = sorted(
                kit_mass.items(),
                key=lambda item: (-item[1], item[0]),
            )
            kit_cluster = ordered_kit[0][0]
            total_kit_mass = sum(
                value for _, value in ordered_kit
            )
            kit_dominance = float(
                ordered_kit[0][1] / total_kit_mass
            )
        else:
            kit_cluster = None
            kit_dominance = None

        result.append(
            IdentityProfile(
                identity_id=identity_id,
                source_actor_ids=actor_ids,
                tracklet_ids=tracklet_ids,
                profiled_actor_ids=tuple(
                    str(row["actor_id"])
                    for row in available
                ),
                profiled_tracklet_count=profiled_tracklet_count,
                aggregate_embedding=tuple(
                    float(value) for value in aggregate
                ),
                reliability=reliability,
                start_time_s=_finite(
                    "identity start_time_s",
                    identity["start_time_s"],
                ),
                end_time_s=_finite(
                    "identity end_time_s",
                    identity["end_time_s"],
                ),
                kit_cluster_id=kit_cluster,
                kit_dominance=kit_dominance,
                initial_seed=(
                    profiled_tracklet_count
                    >= config.min_seed_profiled_tracklets
                ),
            )
        )

    return tuple(sorted(
        result,
        key=lambda row: row.identity_id,
    ))


def build_hard_identity_pairs(
    *,
    tracklet_assignment_rows: Sequence[Mapping[str, Any]],
    hard_conflicts: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    identity_by_tracklet = {
        str(row["tracklet_id"]): str(row["identity_id"])
        for row in tracklet_assignment_rows
    }
    pairs: set[tuple[str, str]] = set()

    for row in hard_conflicts:
        ta = str(row["tracklet_a"])
        tb = str(row["tracklet_b"])
        ia = identity_by_tracklet.get(ta)
        ib = identity_by_tracklet.get(tb)
        if ia is None or ib is None:
            raise ValueError(
                "hard conflict references unassigned tracklet"
            )
        if ia == ib:
            raise ValueError(
                f"hard conflict already inside persistent identity {ia}"
            )
        pairs.add(_canonical_pair(ia, ib))

    return pairs


def fit_identity_calibration(
    *,
    profiles: Sequence[IdentityProfile],
    hard_identity_pairs: set[tuple[str, str]],
    trusted_positive_cosines: Sequence[float],
    identity_family_evidence: Mapping[
        str, IdentityFamilyEvidence
    ] | None = None,
) -> IdentityCalibration:
    by_id = {
        profile.identity_id: profile
        for profile in profiles
    }

    negative_groups: dict[str, list[float]] = {
        "same_kit": [],
        "different_kit": [],
        "unknown_kit": [],
    }

    for identity_a, identity_b in sorted(hard_identity_pairs):
        first = by_id.get(identity_a)
        second = by_id.get(identity_b)
        if first is None or second is None:
            continue
        family = _family_calibration_bucket(
            first,
            second,
            identity_family_evidence or {},
        )
        negative_groups[family].append(
            _cosine(
                first.aggregate_embedding,
                second.aggregate_embedding,
            )
        )

    return IdentityCalibration(
        positive=tuple(sorted(
            _finite("trusted positive cosine", value)
            for value in trusted_positive_cosines
        )),
        same_kit_negative=tuple(sorted(
            negative_groups["same_kit"]
        )),
        different_kit_negative=tuple(sorted(
            negative_groups["different_kit"]
        )),
        unknown_kit_negative=tuple(sorted(
            negative_groups["unknown_kit"]
        )),
    )


def _top_k_indices(
    scores: np.ndarray,
    valid: np.ndarray,
    ids: Sequence[str],
    k: int,
) -> list[int]:
    indices = np.flatnonzero(valid)
    ranked = sorted(
        (int(index) for index in indices),
        key=lambda index: (
            -float(scores[index]),
            ids[index],
        ),
    )
    return ranked[:k]


def build_expansion_candidates(
    *,
    profiles: Sequence[IdentityProfile],
    calibration: IdentityCalibration,
    hard_identity_pairs: set[tuple[str, str]],
    config: IdentityExpansionConfig,
    identity_family_evidence: Mapping[
        str, IdentityFamilyEvidence
    ] | None = None,
) -> tuple[ExpansionCandidate, ...]:
    ordered = tuple(
        sorted(profiles, key=lambda row: row.identity_id)
    )
    if not ordered:
        return ()

    ids = [row.identity_id for row in ordered]
    vectors = np.vstack([
        _unit(row.aggregate_embedding, label=row.identity_id)
        for row in ordered
    ])
    similarities = np.clip(
        vectors @ vectors.T,
        -1.0,
        1.0,
    )

    forward_rank: dict[tuple[int, int], int] = {}
    backward_rank: dict[tuple[int, int], int] = {}
    count = len(ordered)

    for i, predecessor in enumerate(ordered):
        valid = np.zeros(count, dtype=bool)
        for j, successor in enumerate(ordered):
            if i == j:
                continue
            if (
                predecessor.end_time_s
                <= successor.start_time_s
                + config.time_epsilon_s
                and _canonical_pair(
                    predecessor.identity_id,
                    successor.identity_id,
                ) not in hard_identity_pairs
            ):
                valid[j] = True

        for rank, j in enumerate(
            _top_k_indices(
                similarities[i],
                valid,
                ids,
                config.top_k_successors,
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
                predecessor.end_time_s
                <= successor.start_time_s
                + config.time_epsilon_s
                and _canonical_pair(
                    predecessor.identity_id,
                    successor.identity_id,
                ) not in hard_identity_pairs
            ):
                valid[i] = True

        for rank, i in enumerate(
            _top_k_indices(
                similarities[:, j],
                valid,
                ids,
                config.top_k_predecessors,
            ),
            start=1,
        ):
            backward_rank[(i, j)] = rank

    pairs = sorted(set(forward_rank) | set(backward_rank))

    def retrieval_rank_support(rank: int | None) -> float:
        if rank is None:
            return config.one_sided_rank_floor
        return 1.0 / math.sqrt(max(1, rank))

    raw: list[ExpansionCandidate] = []

    for i, j in pairs:
        predecessor = ordered[i]
        successor = ordered[j]
        cosine = float(similarities[i, j])
        family_evidence = identity_family_evidence or {}
        if family_evidence:
            relation, pair_strength = family_pair_relation(
                predecessor.identity_id,
                successor.identity_id,
                family_evidence,
                config=config,
            )
            if relation == "same_primary":
                family = "same_kit"
            elif relation in {
                "different_primary",
                "probable_different_primary",
            }:
                family = "different_kit"
            else:
                family = "unknown_kit"
            # ReID calibration and kit-family compatibility are deliberately
            # separated.  Use the hardest same-primary negative population
            # when available, then apply family evidence as a soft score delta.
            empirical_family = "same_kit"
            evidence_source = "entity_consensus"
        else:
            relation = (
                "legacy_same_kit"
                if _kit_family(predecessor, successor) == "same_kit"
                else "legacy_different_kit"
                if _kit_family(predecessor, successor) == "different_kit"
                else "legacy_unknown_kit"
            )
            pair_strength = 0.0
            family = _kit_family(predecessor, successor)
            empirical_family = family
            evidence_source = "legacy_chain_kit"

        empirical = calibration.support(
            cosine,
            family=empirical_family,
        )
        forward = forward_rank.get((i, j))
        backward = backward_rank.get((i, j))
        retrieval_support = _geometric_mean([
            retrieval_rank_support(forward),
            retrieval_rank_support(backward),
        ])
        confidence = math.sqrt(
            max(
                0.0,
                predecessor.reliability
                * successor.reliability,
            )
        )
        raw_support = _geometric_mean([
            empirical["support"],
            retrieval_support,
        ])
        base_score = _clamp01(
            0.5 + confidence * (raw_support - 0.5)
        )
        adjustment = family_score_adjustment(
            relation=relation,
            pair_strength=pair_strength,
            config=config,
        )
        score = _clamp01(base_score + adjustment)

        raw.append(
            ExpansionCandidate(
                predecessor_identity_id=(
                    predecessor.identity_id
                ),
                successor_identity_id=successor.identity_id,
                gap_s=max(
                    0.0,
                    successor.start_time_s
                    - predecessor.end_time_s,
                ),
                aggregate_cosine=cosine,
                kit_family=family,
                empirical_support=empirical["support"],
                profile_confidence=confidence,
                forward_retrieval_rank=forward,
                backward_retrieval_rank=backward,
                retrieval_support=retrieval_support,
                base_rank_score=base_score,
                family_relation=relation,
                family_pair_strength=pair_strength,
                family_adjustment=adjustment,
                family_evidence_source=evidence_source,
                rank_score=score,
            )
        )

    outgoing: dict[str, list[int]] = defaultdict(list)
    incoming: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(raw):
        outgoing[row.predecessor_identity_id].append(index)
        incoming[row.successor_identity_id].append(index)

    out_rank: dict[int, int] = {}
    in_rank: dict[int, int] = {}

    for indices in outgoing.values():
        indices.sort(
            key=lambda index: (
                -raw[index].rank_score,
                raw[index].successor_identity_id,
            )
        )
        for rank, index in enumerate(indices, start=1):
            out_rank[index] = rank

    for indices in incoming.values():
        indices.sort(
            key=lambda index: (
                -raw[index].rank_score,
                raw[index].predecessor_identity_id,
            )
        )
        for rank, index in enumerate(indices, start=1):
            in_rank[index] = rank

    result: list[ExpansionCandidate] = []
    for index, row in enumerate(raw):
        result.append(
            ExpansionCandidate(
                predecessor_identity_id=row.predecessor_identity_id,
                successor_identity_id=row.successor_identity_id,
                gap_s=row.gap_s,
                aggregate_cosine=row.aggregate_cosine,
                kit_family=row.kit_family,
                empirical_support=row.empirical_support,
                profile_confidence=row.profile_confidence,
                forward_retrieval_rank=row.forward_retrieval_rank,
                backward_retrieval_rank=row.backward_retrieval_rank,
                retrieval_support=row.retrieval_support,
                base_rank_score=row.base_rank_score,
                family_relation=row.family_relation,
                family_pair_strength=row.family_pair_strength,
                family_adjustment=row.family_adjustment,
                family_evidence_source=row.family_evidence_source,
                rank_score=row.rank_score,
                outgoing_score_rank=out_rank[index],
                incoming_score_rank=in_rank[index],
                mutual_best=(
                    out_rank[index] == 1
                    and in_rank[index] == 1
                ),
                mutual_top3=(
                    out_rank[index] <= 3
                    and in_rank[index] <= 3
                ),
            )
        )

    return tuple(result)


class _SeededComponents:
    def __init__(
        self,
        *,
        identity_ids: Sequence[str],
        initial_seed: Mapping[str, bool],
        hard_neighbors: Mapping[str, set[str]],
    ) -> None:
        self.parent = {
            identity_id: identity_id
            for identity_id in identity_ids
        }
        self.members = {
            identity_id: {identity_id}
            for identity_id in identity_ids
        }
        self.seeded = {
            identity_id: bool(initial_seed.get(identity_id, False))
            for identity_id in identity_ids
        }
        self.hard_neighbors = hard_neighbors

    def find(self, identity_id: str) -> str:
        parent = self.parent[identity_id]
        if parent != identity_id:
            self.parent[identity_id] = self.find(parent)
        return self.parent[identity_id]

    def component_seeded(self, identity_id: str) -> bool:
        return self.seeded[self.find(identity_id)]

    def compatible(self, first: str, second: str) -> bool:
        left = self.members[self.find(first)]
        right = self.members[self.find(second)]
        if len(left) > len(right):
            left, right = right, left
        for identity_id in left:
            if self.hard_neighbors.get(
                identity_id,
                set(),
            ).intersection(right):
                return False
        return True

    def union(self, first: str, second: str) -> None:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return
        if len(self.members[root_a]) < len(self.members[root_b]):
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.members[root_a].update(self.members[root_b])
        self.seeded[root_a] = (
            self.seeded[root_a]
            or self.seeded[root_b]
        )
        del self.members[root_b]
        del self.seeded[root_b]


def _competition_passes(
    candidate: ExpansionCandidate,
    mode: str,
) -> bool:
    if mode == "mutual_best":
        return candidate.mutual_best
    if mode == "mutual_top3":
        return candidate.mutual_top3
    raise ValueError(f"unsupported competition mode {mode!r}")


def solve_seeded_expansion(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    persistent_actor_assignments: Sequence[Mapping[str, Any]],
    persistent_tracklet_assignments: Sequence[Mapping[str, Any]],
    profiles: Sequence[IdentityProfile],
    candidates: Sequence[ExpansionCandidate],
    hard_identity_pairs: set[tuple[str, str]],
    config: IdentityExpansionConfig,
) -> tuple[
    tuple[ExpansionCandidate, ...],
    tuple[ExpandedIdentity, ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    dict[str, int],
]:
    source_identity = {
        str(row["identity_id"]): row
        for row in identity_rows
    }
    identity_ids = tuple(sorted(source_identity))

    actor_ids_by_identity: dict[str, tuple[str, ...]] = {
        identity_id: tuple(
            str(value)
            for value in row["actor_ids"]
        )
        for identity_id, row in source_identity.items()
    }
    tracklet_ids_by_identity: dict[str, tuple[str, ...]] = {
        identity_id: tuple(
            str(value)
            for value in row["tracklet_ids"]
        )
        for identity_id, row in source_identity.items()
    }

    # Cross-check the assignment artifacts before changing anything.
    actor_assignment_map = {
        str(row["actor_id"]): str(row["identity_id"])
        for row in persistent_actor_assignments
    }
    tracklet_assignment_map = {
        str(row["tracklet_id"]): str(row["identity_id"])
        for row in persistent_tracklet_assignments
    }
    for identity_id in identity_ids:
        for actor_id in actor_ids_by_identity[identity_id]:
            if actor_assignment_map.get(actor_id) != identity_id:
                raise ValueError(
                    "persistent actor assignments disagree with identities"
                )
        for tracklet_id in tracklet_ids_by_identity[identity_id]:
            if tracklet_assignment_map.get(tracklet_id) != identity_id:
                raise ValueError(
                    "persistent tracklet assignments disagree with identities"
                )

    profile_by_id = {
        profile.identity_id: profile
        for profile in profiles
    }
    initial_seed = {
        identity_id: (
            profile_by_id[identity_id].initial_seed
            if identity_id in profile_by_id
            else False
        )
        for identity_id in identity_ids
    }

    hard_neighbors: dict[str, set[str]] = defaultdict(set)
    for first, second in hard_identity_pairs:
        hard_neighbors[first].add(second)
        hard_neighbors[second].add(first)

    components = _SeededComponents(
        identity_ids=identity_ids,
        initial_seed=initial_seed,
        hard_neighbors=hard_neighbors,
    )

    rejections = Counter()
    eligible: list[ExpansionCandidate] = []

    for candidate in candidates:
        if candidate.rank_score < config.min_rank_score:
            rejections["score_below_floor"] += 1
            continue
        if not _competition_passes(
            candidate,
            config.competition_mode,
        ):
            rejections["competition_filter"] += 1
            continue
        if (
            candidate.family_evidence_source == "legacy_chain_kit"
            and candidate.kit_family == "different_kit"
            and not config.allow_different_kit
        ):
            rejections["different_kit"] += 1
            continue
        eligible.append(candidate)

    eligible.sort(
        key=lambda row: (
            -row.rank_score,
            -row.profile_confidence,
            row.gap_s,
            row.predecessor_identity_id,
            row.successor_identity_id,
        )
    )

    successor_of: dict[str, str] = {}
    predecessor_of: dict[str, str] = {}
    selected: list[ExpansionCandidate] = []

    pending = list(eligible)

    while pending:
        deferred: list[ExpansionCandidate] = []
        merged_this_pass = 0

        for candidate in pending:
            predecessor = candidate.predecessor_identity_id
            successor = candidate.successor_identity_id

            if predecessor in successor_of:
                rejections["predecessor_successor_taken"] += 1
                continue
            if successor in predecessor_of:
                rejections["successor_predecessor_taken"] += 1
                continue

            root_a = components.find(predecessor)
            root_b = components.find(successor)

            if root_a == root_b:
                rejections["same_component"] += 1
                continue

            if not (
                components.component_seeded(predecessor)
                or components.component_seeded(successor)
            ):
                deferred.append(candidate)
                continue

            if not components.compatible(
                predecessor,
                successor,
            ):
                rejections["component_hard_conflict"] += 1
                continue

            successor_of[predecessor] = successor
            predecessor_of[successor] = predecessor
            components.union(predecessor, successor)
            selected.append(candidate)
            merged_this_pass += 1

        if merged_this_pass == 0:
            rejections["unseeded_components"] += len(deferred)
            break

        pending = deferred

    roots = sorted(
        (
            identity_id
            for identity_id in identity_ids
            if identity_id not in predecessor_of
        ),
        key=lambda identity_id: (
            _finite(
                "start_time_s",
                source_identity[identity_id]["start_time_s"],
            ),
            identity_id,
        ),
    )

    raw_paths: list[tuple[str, ...]] = []
    visited: set[str] = set()

    for root in roots:
        path: list[str] = []
        current = root
        local_seen: set[str] = set()

        while True:
            if current in local_seen:
                raise RuntimeError("cycle in seeded identity expansion")
            if current in visited:
                raise RuntimeError(
                    "source identity appears in multiple expanded identities"
                )
            local_seen.add(current)
            visited.add(current)
            path.append(current)

            next_id = successor_of.get(current)
            if next_id is None:
                break
            current = next_id

        raw_paths.append(tuple(path))

    if visited != set(identity_ids):
        raise RuntimeError("seeded expansion lost source identities")

    expanded: list[ExpandedIdentity] = []
    identity_assignments: list[dict[str, Any]] = []
    actor_assignments: list[dict[str, Any]] = []
    tracklet_assignments: list[dict[str, Any]] = []

    for index, path in enumerate(raw_paths, start=1):
        expanded_id = f"person-{index:04d}"
        actor_ids: list[str] = []
        tracklet_ids: list[str] = []

        seeded = any(
            initial_seed[identity_id]
            for identity_id in path
        )

        for order, source_id in enumerate(path):
            identity_assignments.append({
                "schema_version": 1,
                "expansion_version": IDENTITY_EXPANSION_VERSION,
                "source_identity_id": source_id,
                "identity_id": expanded_id,
                "identity_order": order,
                "provisional": True,
            })
            for actor_id in actor_ids_by_identity[source_id]:
                actor_ids.append(actor_id)
                actor_assignments.append({
                    "schema_version": 1,
                    "expansion_version": IDENTITY_EXPANSION_VERSION,
                    "actor_id": actor_id,
                    "source_identity_id": source_id,
                    "identity_id": expanded_id,
                    "provisional": True,
                })
            for tracklet_id in tracklet_ids_by_identity[source_id]:
                tracklet_ids.append(tracklet_id)
                tracklet_assignments.append({
                    "schema_version": 1,
                    "expansion_version": IDENTITY_EXPANSION_VERSION,
                    "tracklet_id": tracklet_id,
                    "source_identity_id": source_id,
                    "identity_id": expanded_id,
                    "provisional": True,
                })

        expanded.append(
            ExpandedIdentity(
                identity_id=expanded_id,
                source_identity_ids=path,
                actor_ids=tuple(actor_ids),
                tracklet_ids=tuple(tracklet_ids),
                start_time_s=min(
                    _finite(
                        "start_time_s",
                        source_identity[source_id]["start_time_s"],
                    )
                    for source_id in path
                ),
                end_time_s=max(
                    _finite(
                        "end_time_s",
                        source_identity[source_id]["end_time_s"],
                    )
                    for source_id in path
                ),
                seeded=seeded,
            )
        )

    expanded_by_source = {
        row["source_identity_id"]: row["identity_id"]
        for row in identity_assignments
    }

    for first, second in hard_identity_pairs:
        if expanded_by_source[first] == expanded_by_source[second]:
            raise RuntimeError(
                "hard cannot-link survived inside expanded identity: "
                f"{first}, {second}"
            )

    return (
        tuple(selected),
        tuple(expanded),
        tuple(identity_assignments),
        tuple(actor_assignments),
        tuple(tracklet_assignments),
        dict(rejections),
    )


def run_identity_expansion(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    persistent_actor_assignments: Sequence[Mapping[str, Any]],
    persistent_tracklet_assignments: Sequence[Mapping[str, Any]],
    chain_profile_rows: Sequence[Mapping[str, Any]],
    hard_conflicts: Sequence[Mapping[str, Any]],
    trusted_positive_cosines: Sequence[float],
    identity_family_rows: Sequence[Mapping[str, Any]] | None = None,
    config: IdentityExpansionConfig | None = None,
) -> IdentityExpansionResult:
    selected = config or IdentityExpansionConfig()
    family_evidence = build_identity_family_evidence(
        identity_family_rows
    )

    profiles = build_identity_profiles(
        identity_rows=identity_rows,
        chain_profile_rows=chain_profile_rows,
        config=selected,
    )

    hard_identity_pairs = build_hard_identity_pairs(
        tracklet_assignment_rows=persistent_tracklet_assignments,
        hard_conflicts=hard_conflicts,
    )

    calibration = fit_identity_calibration(
        profiles=profiles,
        hard_identity_pairs=hard_identity_pairs,
        trusted_positive_cosines=trusted_positive_cosines,
        identity_family_evidence=family_evidence,
    )

    candidates = build_expansion_candidates(
        profiles=profiles,
        calibration=calibration,
        hard_identity_pairs=hard_identity_pairs,
        config=selected,
        identity_family_evidence=family_evidence,
    )

    (
        selected_candidates,
        expanded,
        identity_assignments,
        actor_assignments,
        tracklet_assignments,
        rejections,
    ) = solve_seeded_expansion(
        identity_rows=identity_rows,
        persistent_actor_assignments=persistent_actor_assignments,
        persistent_tracklet_assignments=persistent_tracklet_assignments,
        profiles=profiles,
        candidates=candidates,
        hard_identity_pairs=hard_identity_pairs,
        config=selected,
    )

    return IdentityExpansionResult(
        profiles=profiles,
        candidates=candidates,
        selected_candidates=selected_candidates,
        expanded_identities=expanded,
        identity_assignments=identity_assignments,
        actor_assignments=actor_assignments,
        tracklet_assignments=tracklet_assignments,
        rejection_counts=rejections,
        hard_identity_pair_count=len(hard_identity_pairs),
        calibration=calibration,
        config=selected,
    )
