"""Hito 7C.1F: aggregate kit-family evidence to actors / identities.

The previous stage (7C.1E) decides kit-family evidence at the conservative
tracklet level.  Tracking, however, reasons about larger provisional actors
and persistent identities.  This module aggregates the tracklet evidence
without turning missing evidence into a contradiction.

Key rules
---------
* Two primary families remain anonymous.  ``primary-0``/``primary-1`` do not
  mean a particular club or colour.
* Special families remain anonymous and conservative.
* Confident tracklets vote at full strength; weak tracklets may contribute
  softly so several weak observations can become useful together.
* Unknown tracklets reduce evidence coverage but NEVER count as a vote for a
  different family.
* Conflicting known families reduce dominance.
* Family evidence is SOFT only.  This module creates no identity cannot-link.
* The same aggregator works for temporal actors or later persistent identities;
  the caller only supplies a tracklet -> entity mapping.

This is intentionally generic across clips and matches.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence


KIT_ENTITY_CONSENSUS_VERSION = "kit-family-entity-consensus-v1"


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


@dataclass(frozen=True, slots=True)
class KitEntityConsensusConfig:
    confident_primary_scale: float = 1.0
    confident_special_scale: float = 0.90
    weak_primary_scale: float = 0.35
    weak_special_scale: float = 0.15

    max_tracklet_base_mass: float = 3.0

    primary_min_dominance: float = 0.80
    primary_min_confidence: float = 0.35
    primary_min_support_tracklets: int = 1
    primary_min_support_samples: int = 2

    special_min_dominance: float = 0.90
    special_min_confidence: float = 0.45
    special_min_support_tracklets: int = 2
    special_min_support_samples: int = 4

    def __post_init__(self) -> None:
        for name in (
            "confident_primary_scale",
            "confident_special_scale",
            "weak_primary_scale",
            "weak_special_scale",
            "primary_min_dominance",
            "primary_min_confidence",
            "special_min_dominance",
            "special_min_confidence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.max_tracklet_base_mass <= 0.0:
            raise ValueError("max_tracklet_base_mass must be > 0")
        if self.primary_min_support_tracklets < 1:
            raise ValueError("primary_min_support_tracklets must be >= 1")
        if self.primary_min_support_samples < 1:
            raise ValueError("primary_min_support_samples must be >= 1")
        if self.special_min_support_tracklets < 1:
            raise ValueError("special_min_support_tracklets must be >= 1")
        if self.special_min_support_samples < 1:
            raise ValueError("special_min_support_samples must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EntityFamilyConsensus:
    entity_level: str
    entity_id: str
    family_type: str
    family_id: int | None
    candidate_family_type: str | None
    candidate_family_id: int | None
    state: str

    confidence: float
    dominance: float
    evidence_coverage: float
    conflict_fraction: float

    family_distribution: Mapping[str, float]
    family_evidence_mass: Mapping[str, float]
    total_family_evidence_mass: float
    total_possible_mass: float

    support_tracklets: int
    conflicting_tracklets: int
    unknown_tracklets: int
    missing_consensus_tracklets: int
    total_tracklets: int

    support_samples: int
    total_samples: int

    member_tracklet_ids: tuple[str, ...]

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
    def usable_for_tracking(self) -> bool:
        return self.state in {
            "confident_primary",
            "confident_special",
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "entity_consensus_version": KIT_ENTITY_CONSENSUS_VERSION,
            "entity_level": self.entity_level,
            "entity_id": self.entity_id,
            "family_type": self.family_type,
            "family_id": self.family_id,
            "family_key": self.family_key,
            "candidate_family_type": self.candidate_family_type,
            "candidate_family_id": self.candidate_family_id,
            "candidate_family_key": self.candidate_family_key,
            "state": self.state,
            "confidence": self.confidence,
            "dominance": self.dominance,
            "evidence_coverage": self.evidence_coverage,
            "conflict_fraction": self.conflict_fraction,
            "family_distribution": dict(self.family_distribution),
            "family_evidence_mass": dict(self.family_evidence_mass),
            "total_family_evidence_mass": self.total_family_evidence_mass,
            "total_possible_mass": self.total_possible_mass,
            "support_tracklets": self.support_tracklets,
            "conflicting_tracklets": self.conflicting_tracklets,
            "unknown_tracklets": self.unknown_tracklets,
            "missing_consensus_tracklets": self.missing_consensus_tracklets,
            "total_tracklets": self.total_tracklets,
            "support_samples": self.support_samples,
            "total_samples": self.total_samples,
            "member_tracklet_ids": list(self.member_tracklet_ids),
            "usable_for_tracking": self.usable_for_tracking,
            "team_label": None,
            "role_label": None,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class KitEntityConsensusResult:
    entity_level: str
    entities: tuple[EntityFamilyConsensus, ...]
    config: KitEntityConsensusConfig

    def summary(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        families: dict[str, int] = {}
        for row in self.entities:
            states[row.state] = states.get(row.state, 0) + 1
            key = row.family_key or "unknown"
            families[key] = families.get(key, 0) + 1

        confidences = [
            row.confidence
            for row in self.entities
            if row.candidate_family_key is not None
        ]
        coverages = [row.evidence_coverage for row in self.entities]

        return {
            "schema_version": 1,
            "entity_consensus_version": KIT_ENTITY_CONSENSUS_VERSION,
            "entity_level": self.entity_level,
            "entity_count": len(self.entities),
            "usable_for_tracking_count": sum(
                row.usable_for_tracking for row in self.entities
            ),
            "state_counts": dict(sorted(states.items())),
            "family_counts": dict(sorted(families.items())),
            "confidence": _stats(confidences),
            "evidence_coverage": _stats(coverages),
            "config": self.config.to_dict(),
            "missing_evidence_is_negative": False,
            "identity_hard_constraints_created": False,
            "semantic_team_labels_assigned": False,
            "semantic_role_labels_assigned": False,
        }


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
    ordered = sorted(float(v) for v in values)

    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = (len(ordered) - 1) * p
        lower = int(math.floor(position))
        upper = int(math.ceil(position))
        if lower == upper:
            return ordered[lower]
        fraction = position - lower
        return (
            ordered[lower] * (1.0 - fraction)
            + ordered[upper] * fraction
        )

    return {
        "count": len(ordered),
        "min": ordered[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
    }


def _candidate_key(row: Mapping[str, Any]) -> str | None:
    family_type = row.get("candidate_family_type")
    family_id = row.get("candidate_family_id")
    if family_type not in {"primary", "special"} or family_id is None:
        return None
    return f"{family_type}-{int(family_id)}"


def _vote_scale(
    row: Mapping[str, Any],
    config: KitEntityConsensusConfig,
) -> float:
    state = str(row.get("state", "unknown"))
    if state == "confident_primary":
        return config.confident_primary_scale
    if state == "confident_special":
        return config.confident_special_scale
    if state == "weak_primary":
        return config.weak_primary_scale
    if state == "weak_special":
        return config.weak_special_scale
    return 0.0


def _tracklet_possible_mass(
    row: Mapping[str, Any],
    config: KitEntityConsensusConfig,
) -> float:
    quality_mass = max(
        0.0,
        float(row.get("total_quality_mass", 0.0)),
    )
    total_samples = max(0, int(row.get("total_samples", 0)))
    if quality_mass <= 1e-12:
        quality_mass = float(total_samples)
    if quality_mass <= 1e-12:
        return 0.0
    return min(
        config.max_tracklet_base_mass,
        math.sqrt(quality_mass),
    )


def _tracklet_family_mass(
    row: Mapping[str, Any],
    config: KitEntityConsensusConfig,
) -> tuple[str | None, float, float]:
    possible = _tracklet_possible_mass(row, config)
    key = _candidate_key(row)
    scale = _vote_scale(row, config)

    if possible <= 1e-12 or key is None or scale <= 0.0:
        return None, 0.0, possible

    dominance = _clamp01(float(row.get("dominance", 0.0)))
    known_fraction = _clamp01(
        float(row.get("known_fraction", 0.0))
    )

    evidence = (
        possible
        * scale
        * dominance
        * known_fraction
    )
    return key, float(evidence), possible


def aggregate_entity_family_consensus(
    *,
    tracklet_consensus_rows: Sequence[Mapping[str, Any]],
    tracklet_entity_rows: Sequence[Mapping[str, Any]],
    entity_id_field: str,
    entity_level: str,
    config: KitEntityConsensusConfig | None = None,
) -> KitEntityConsensusResult:
    selected = config or KitEntityConsensusConfig()

    consensus_by_tracklet: dict[str, Mapping[str, Any]] = {}
    for row in tracklet_consensus_rows:
        tracklet_id = str(row["tracklet_id"])
        if tracklet_id in consensus_by_tracklet:
            raise ValueError(
                f"duplicate tracklet consensus {tracklet_id}"
            )
        consensus_by_tracklet[tracklet_id] = row

    tracklets_by_entity: dict[str, list[str]] = defaultdict(list)
    tracklet_owner: dict[str, str] = {}

    for row in tracklet_entity_rows:
        tracklet_id = str(row["tracklet_id"])
        entity_id = str(row[entity_id_field])
        previous = tracklet_owner.get(tracklet_id)
        if previous is not None and previous != entity_id:
            raise ValueError(
                f"tracklet {tracklet_id} maps to multiple entities"
            )
        tracklet_owner[tracklet_id] = entity_id
        tracklets_by_entity[entity_id].append(tracklet_id)

    entities: list[EntityFamilyConsensus] = []

    for entity_id in sorted(tracklets_by_entity):
        member_tracklets = tuple(sorted(
            set(tracklets_by_entity[entity_id])
        ))

        family_mass: dict[str, float] = defaultdict(float)
        tracklet_votes: dict[str, list[str]] = defaultdict(list)

        total_possible_mass = 0.0
        total_samples = 0
        missing_consensus = 0

        for tracklet_id in member_tracklets:
            row = consensus_by_tracklet.get(tracklet_id)
            if row is None:
                missing_consensus += 1
                continue

            total_samples += max(
                0,
                int(row.get("total_samples", 0)),
            )
            key, evidence, possible = _tracklet_family_mass(
                row,
                selected,
            )
            total_possible_mass += possible

            if key is not None and evidence > 0.0:
                family_mass[key] += evidence
                tracklet_votes[key].append(tracklet_id)

        total_family_mass = float(sum(family_mass.values()))
        if family_mass and total_family_mass > 1e-12:
            candidate_key = max(
                family_mass,
                key=lambda key: (
                    family_mass[key],
                    key,
                ),
            )
            dominance = float(
                family_mass[candidate_key] / total_family_mass
            )
            candidate_type, raw_id = candidate_key.rsplit("-", 1)
            candidate_id = int(raw_id)
        else:
            candidate_key = None
            candidate_type = None
            candidate_id = None
            dominance = 0.0

        evidence_coverage = (
            total_family_mass / total_possible_mass
            if total_possible_mass > 1e-12
            else 0.0
        )
        evidence_coverage = _clamp01(evidence_coverage)
        confidence = _clamp01(
            dominance * math.sqrt(evidence_coverage)
        )
        conflict_fraction = (
            1.0 - dominance
            if total_family_mass > 1e-12
            else 0.0
        )

        family_distribution = {
            key: (
                float(value / total_family_mass)
                if total_family_mass > 1e-12
                else 0.0
            )
            for key, value in sorted(family_mass.items())
        }

        if candidate_key is None:
            support_tracklets = 0
            support_samples = 0
            conflicting_tracklets = 0
            unknown_tracklets = (
                len(member_tracklets) - missing_consensus
            )
            final_type = "unknown"
            final_id = None
            state = "unknown"

        else:
            support_ids = set(tracklet_votes[candidate_key])
            support_tracklets = len(support_ids)

            support_samples = sum(
                max(
                    0,
                    int(
                        consensus_by_tracklet[tracklet_id].get(
                            "support_samples",
                            0,
                        )
                    ),
                )
                for tracklet_id in support_ids
            )

            known_vote_tracklets = set()
            for ids in tracklet_votes.values():
                known_vote_tracklets.update(ids)

            conflicting_tracklets = len(
                known_vote_tracklets - support_ids
            )
            unknown_tracklets = (
                len(member_tracklets)
                - len(known_vote_tracklets)
                - missing_consensus
            )

            final_type = "unknown"
            final_id = None

            if candidate_type == "primary":
                good = (
                    dominance >= selected.primary_min_dominance
                    and confidence >= selected.primary_min_confidence
                    and support_tracklets
                    >= selected.primary_min_support_tracklets
                    and support_samples
                    >= selected.primary_min_support_samples
                )
                state = (
                    "confident_primary"
                    if good
                    else "weak_primary"
                )
                if good:
                    final_type = "primary"
                    final_id = candidate_id

            elif candidate_type == "special":
                good = (
                    dominance >= selected.special_min_dominance
                    and confidence >= selected.special_min_confidence
                    and support_tracklets
                    >= selected.special_min_support_tracklets
                    and support_samples
                    >= selected.special_min_support_samples
                )
                state = (
                    "confident_special"
                    if good
                    else "weak_special"
                )
                if good:
                    final_type = "special"
                    final_id = candidate_id
            else:
                state = "unknown"

        entities.append(
            EntityFamilyConsensus(
                entity_level=str(entity_level),
                entity_id=entity_id,
                family_type=final_type,
                family_id=final_id,
                candidate_family_type=candidate_type,
                candidate_family_id=candidate_id,
                state=state,
                confidence=confidence,
                dominance=dominance,
                evidence_coverage=evidence_coverage,
                conflict_fraction=conflict_fraction,
                family_distribution=family_distribution,
                family_evidence_mass=dict(sorted(family_mass.items())),
                total_family_evidence_mass=total_family_mass,
                total_possible_mass=total_possible_mass,
                support_tracklets=support_tracklets,
                conflicting_tracklets=conflicting_tracklets,
                unknown_tracklets=unknown_tracklets,
                missing_consensus_tracklets=missing_consensus,
                total_tracklets=len(member_tracklets),
                support_samples=support_samples,
                total_samples=total_samples,
                member_tracklet_ids=member_tracklets,
            )
        )

    return KitEntityConsensusResult(
        entity_level=str(entity_level),
        entities=tuple(entities),
        config=selected,
    )
