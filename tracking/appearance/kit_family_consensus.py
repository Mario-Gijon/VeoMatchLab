"""Hito 7C.1E: temporal kit-family consensus per conservative tracklet.

This stage consumes the football-constrained colour families from Hito 7C.1D
and deliberately moves the decision unit from an individual crop to the whole
tracklet.

Why
---
A crop may contain an opponent, an official, blur, shadow or partial
occlusion.  A conservative tracklet, however, usually contains several
observations of the same physical person.  Kit evidence should therefore be
aggregated across the tracklet and a small number of contaminated crops should
not flip the family.

Design
------
* Family centroids come from 7C.1D: exactly 2 primary families plus 0..3
  anonymous special families.
* Sample-to-family thresholds are calibrated clip-wise from the already
  confident 7C.1D tracklet profiles.  No colour names or club-specific rules
  are used.
* Every saved sample independently votes for the closest family only when its
  cosine and competition margin are credible.
* Votes are quality/confidence weighted.
* The final tracklet family requires dominance, sufficient known evidence and
  support.  Special families require stricter repeated temporal support.
* Weak or mixed cases become ``unknown``.
* Family evidence remains SOFT.  This module creates no identity cannot-link.

The output is intended for later use by actor/identity scoring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections import defaultdict
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


KIT_FAMILY_CONSENSUS_VERSION = "kit-family-consensus-v1"


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


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _family_key(family_type: str, family_id: int) -> str:
    return f"{family_type}-{int(family_id)}"


def _parse_family_key(value: str) -> tuple[str, int]:
    family_type, raw_id = value.rsplit("-", 1)
    if family_type not in {"primary", "special"}:
        raise ValueError(f"unsupported family key {value!r}")
    return family_type, int(raw_id)


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


@dataclass(frozen=True, slots=True)
class KitFamilyConsensusConfig:
    calibration_cosine_quantile: float = 0.05
    calibration_margin_quantile: float = 0.05
    calibration_cosine_relaxation: float = 0.07
    calibration_margin_scale: float = 0.60
    min_calibration_profiles: int = 5
    fallback_cosine_threshold: float = 0.78
    fallback_margin_threshold: float = 0.025

    primary_min_dominance: float = 0.72
    primary_min_known_fraction: float = 0.55
    primary_min_support_samples: int = 2
    primary_max_conflict_fraction: float = 0.30

    special_min_dominance: float = 0.82
    special_min_known_fraction: float = 0.65
    special_min_support_samples: int = 2
    special_max_conflict_fraction: float = 0.20
    special_temporal_bin_s: float = 0.75
    special_min_temporal_bins: int = 2

    min_sample_quality: float = 0.10
    min_vote_weight: float = 0.05

    def __post_init__(self) -> None:
        for name in (
            "calibration_cosine_quantile",
            "calibration_margin_quantile",
            "primary_min_dominance",
            "primary_min_known_fraction",
            "primary_max_conflict_fraction",
            "special_min_dominance",
            "special_min_known_fraction",
            "special_max_conflict_fraction",
            "min_sample_quality",
            "min_vote_weight",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        if self.calibration_cosine_relaxation < 0.0:
            raise ValueError("calibration_cosine_relaxation must be >= 0")
        if not 0.0 <= self.calibration_margin_scale <= 1.0:
            raise ValueError("calibration_margin_scale must be in [0,1]")
        if self.min_calibration_profiles < 1:
            raise ValueError("min_calibration_profiles must be >= 1")
        if self.primary_min_support_samples < 1:
            raise ValueError("primary_min_support_samples must be >= 1")
        if self.special_min_support_samples < 1:
            raise ValueError("special_min_support_samples must be >= 1")
        if self.special_temporal_bin_s <= 0.0:
            raise ValueError("special_temporal_bin_s must be > 0")
        if self.special_min_temporal_bins < 1:
            raise ValueError("special_min_temporal_bins must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class FamilyCalibration:
    family_type: str
    family_id: int
    cosine_threshold: float
    margin_threshold: float
    calibration_profile_count: int
    source: str

    @property
    def family_key(self) -> str:
        return _family_key(self.family_type, self.family_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family_type": self.family_type,
            "family_id": self.family_id,
            "family_key": self.family_key,
            "cosine_threshold": self.cosine_threshold,
            "margin_threshold": self.margin_threshold,
            "calibration_profile_count": self.calibration_profile_count,
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class SampleFamilyEvidence:
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    crop_ref: str
    quality: float
    winner_family_type: str | None
    winner_family_id: int | None
    best_cosine: float
    second_best_cosine: float
    margin: float
    cosine_threshold: float | None
    margin_threshold: float | None
    vote_weight: float
    state: str

    @property
    def winner_family_key(self) -> str | None:
        if self.winner_family_type is None or self.winner_family_id is None:
            return None
        return _family_key(
            self.winner_family_type,
            self.winner_family_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "consensus_version": KIT_FAMILY_CONSENSUS_VERSION,
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "crop_ref": self.crop_ref,
            "quality": self.quality,
            "winner_family_type": self.winner_family_type,
            "winner_family_id": self.winner_family_id,
            "winner_family_key": self.winner_family_key,
            "best_cosine": self.best_cosine,
            "second_best_cosine": self.second_best_cosine,
            "margin": self.margin,
            "cosine_threshold": self.cosine_threshold,
            "margin_threshold": self.margin_threshold,
            "vote_weight": self.vote_weight,
            "state": self.state,
        }


@dataclass(frozen=True, slots=True)
class TrackletFamilyConsensus:
    tracklet_id: str
    family_type: str
    family_id: int | None
    candidate_family_type: str | None
    candidate_family_id: int | None
    state: str
    dominance: float
    known_fraction: float
    support_samples: int
    conflicting_samples: int
    unknown_samples: int
    total_samples: int
    support_temporal_bins: int
    weighted_family_mass: float
    total_known_mass: float
    total_quality_mass: float
    conflict_fraction: float
    mean_support_cosine: float | None
    mean_support_margin: float | None

    @property
    def family_key(self) -> str | None:
        if self.family_type == "unknown" or self.family_id is None:
            return None
        return _family_key(self.family_type, self.family_id)

    @property
    def candidate_family_key(self) -> str | None:
        if self.candidate_family_type is None or self.candidate_family_id is None:
            return None
        return _family_key(
            self.candidate_family_type,
            self.candidate_family_id,
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
            "consensus_version": KIT_FAMILY_CONSENSUS_VERSION,
            "tracklet_id": self.tracklet_id,
            "family_type": self.family_type,
            "family_id": self.family_id,
            "family_key": self.family_key,
            "candidate_family_type": self.candidate_family_type,
            "candidate_family_id": self.candidate_family_id,
            "candidate_family_key": self.candidate_family_key,
            "state": self.state,
            "dominance": self.dominance,
            "known_fraction": self.known_fraction,
            "support_samples": self.support_samples,
            "conflicting_samples": self.conflicting_samples,
            "unknown_samples": self.unknown_samples,
            "total_samples": self.total_samples,
            "support_temporal_bins": self.support_temporal_bins,
            "weighted_family_mass": self.weighted_family_mass,
            "total_known_mass": self.total_known_mass,
            "total_quality_mass": self.total_quality_mass,
            "conflict_fraction": self.conflict_fraction,
            "mean_support_cosine": self.mean_support_cosine,
            "mean_support_margin": self.mean_support_margin,
            "usable_for_tracking": self.usable_for_tracking,
            "team_label": None,
            "role_label": None,
            "identity_decision": None,
        }


@dataclass(frozen=True, slots=True)
class KitFamilyConsensusResult:
    calibrations: tuple[FamilyCalibration, ...]
    sample_evidence: tuple[SampleFamilyEvidence, ...]
    tracklets: tuple[TrackletFamilyConsensus, ...]
    config: KitFamilyConsensusConfig

    def summary(self) -> dict[str, Any]:
        state_counts: dict[str, int] = {}
        family_counts: dict[str, int] = {}
        for row in self.tracklets:
            state_counts[row.state] = state_counts.get(row.state, 0) + 1
            key = row.family_key or "unknown"
            family_counts[key] = family_counts.get(key, 0) + 1

        return {
            "schema_version": 1,
            "consensus_version": KIT_FAMILY_CONSENSUS_VERSION,
            "tracklet_count": len(self.tracklets),
            "sample_evidence_count": len(self.sample_evidence),
            "usable_for_tracking_count": sum(
                row.usable_for_tracking for row in self.tracklets
            ),
            "state_counts": dict(sorted(state_counts.items())),
            "family_counts": dict(sorted(family_counts.items())),
            "dominance": _stats([row.dominance for row in self.tracklets]),
            "known_fraction": _stats([
                row.known_fraction for row in self.tracklets
            ]),
            "support_samples": _stats([
                float(row.support_samples) for row in self.tracklets
            ]),
            "calibrations": [
                row.to_dict() for row in self.calibrations
            ],
            "config": self.config.to_dict(),
            "semantic_roles_assigned": False,
            "identity_hard_constraints_created": False,
            "decision_unit": "whole_conservative_tracklet",
            "contaminated_crop_tolerance": True,
        }


def build_family_calibrations(
    *,
    cluster_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    profile_assignment_rows: Sequence[Mapping[str, Any]],
    config: KitFamilyConsensusConfig,
) -> tuple[
    tuple[FamilyCalibration, ...],
    tuple[str, ...],
    np.ndarray,
]:
    cluster_by_key: dict[str, Mapping[str, Any]] = {}
    for row in cluster_rows:
        family_type = str(row["family_type"])
        family_id = int(row["family_id"])
        key = _family_key(family_type, family_id)
        cluster_by_key[key] = row

    if len(cluster_by_key) < 2:
        raise ValueError("at least two family centroids are required")

    family_keys = tuple(sorted(
        cluster_by_key,
        key=lambda key: (
            0 if key.startswith("primary-") else 1,
            int(key.rsplit("-", 1)[1]),
        ),
    ))
    centroids = np.vstack([
        _unit(
            cluster_by_key[key]["centroid"],
            label=f"centroid {key}",
        )
        for key in family_keys
    ])

    profile_by_id = {
        str(row["tracklet_id"]): row
        for row in profile_rows
    }

    values: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for assignment in profile_assignment_rows:
        if not bool(assignment.get("usable_for_tracking", False)):
            continue
        family_type = assignment.get("family_type")
        family_id = assignment.get("family_id")
        if family_type not in {"primary", "special"} or family_id is None:
            continue
        key = _family_key(str(family_type), int(family_id))
        if key not in cluster_by_key:
            continue

        tracklet_id = str(assignment["tracklet_id"])
        profile = profile_by_id.get(tracklet_id)
        if profile is None:
            continue

        vector = _unit(
            profile["signature"],
            label=f"profile {tracklet_id}",
        )
        similarities = vector @ centroids.T
        family_index = family_keys.index(key)
        own = float(similarities[family_index])
        competitors = np.delete(similarities, family_index)
        second = float(np.max(competitors))
        values[key].append((own, own - second))

    result: list[FamilyCalibration] = []
    for key in family_keys:
        family_type, family_id = _parse_family_key(key)
        rows = values.get(key, [])

        if len(rows) >= config.min_calibration_profiles:
            cosines = np.asarray(
                [row[0] for row in rows],
                dtype=np.float64,
            )
            margins = np.asarray(
                [row[1] for row in rows],
                dtype=np.float64,
            )
            cosine_threshold = max(
                -1.0,
                float(np.quantile(
                    cosines,
                    config.calibration_cosine_quantile,
                )) - config.calibration_cosine_relaxation,
            )
            margin_threshold = max(
                0.0,
                float(np.quantile(
                    margins,
                    config.calibration_margin_quantile,
                )) * config.calibration_margin_scale,
            )
            source = "clip_profile_calibration"
        else:
            cosine_threshold = config.fallback_cosine_threshold
            margin_threshold = config.fallback_margin_threshold
            source = "fallback"

        result.append(
            FamilyCalibration(
                family_type=family_type,
                family_id=family_id,
                cosine_threshold=cosine_threshold,
                margin_threshold=margin_threshold,
                calibration_profile_count=len(rows),
                source=source,
            )
        )

    return tuple(result), family_keys, centroids


def classify_sample_evidence(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    calibrations: Sequence[FamilyCalibration],
    family_keys: Sequence[str],
    centroids: np.ndarray,
    config: KitFamilyConsensusConfig,
) -> tuple[SampleFamilyEvidence, ...]:
    calibration_by_key = {
        row.family_key: row
        for row in calibrations
    }

    result: list[SampleFamilyEvidence] = []

    for sample in sample_rows:
        vector = _unit(
            sample["signature"],
            label=f"sample {sample['detection_id']}",
        )
        similarities = vector @ centroids.T
        order = np.argsort(similarities)[::-1]
        best_index = int(order[0])
        second_index = int(order[1])
        best_cosine = float(similarities[best_index])
        second_cosine = float(similarities[second_index])
        margin = best_cosine - second_cosine

        quality = _clamp01(float(sample["quality"]))
        best_key = str(family_keys[best_index])
        calibration = calibration_by_key[best_key]

        passes = (
            quality >= config.min_sample_quality
            and best_cosine >= calibration.cosine_threshold
            and margin >= calibration.margin_threshold
        )

        if passes:
            cosine_span = max(
                1e-6,
                1.0 - calibration.cosine_threshold,
            )
            cosine_strength = _clamp01(
                (best_cosine - calibration.cosine_threshold)
                / cosine_span
            )
            margin_scale = max(
                0.05,
                2.0 * calibration.margin_threshold,
            )
            margin_strength = _clamp01(
                margin / margin_scale
            )
            vote_weight = quality * (
                0.50 + 0.25 * cosine_strength + 0.25 * margin_strength
            )
            if vote_weight < config.min_vote_weight:
                passes = False
        else:
            vote_weight = 0.0

        if passes:
            family_type, family_id = _parse_family_key(best_key)
            state = "confident_vote"
        else:
            family_type = None
            family_id = None
            vote_weight = 0.0
            state = "unknown_vote"

        result.append(
            SampleFamilyEvidence(
                tracklet_id=str(sample["tracklet_id"]),
                detection_id=str(sample["detection_id"]),
                frame_index=int(sample["frame_index"]),
                timestamp_s=float(sample["timestamp_s"]),
                crop_ref=str(sample["crop_ref"]),
                quality=quality,
                winner_family_type=family_type,
                winner_family_id=family_id,
                best_cosine=best_cosine,
                second_best_cosine=second_cosine,
                margin=margin,
                cosine_threshold=calibration.cosine_threshold,
                margin_threshold=calibration.margin_threshold,
                vote_weight=float(vote_weight),
                state=state,
            )
        )

    return tuple(sorted(
        result,
        key=lambda row: (
            row.tracklet_id,
            row.timestamp_s,
            row.frame_index,
            row.detection_id,
        ),
    ))


def aggregate_tracklet_consensus(
    *,
    sample_evidence: Sequence[SampleFamilyEvidence],
    config: KitFamilyConsensusConfig,
) -> tuple[TrackletFamilyConsensus, ...]:
    by_tracklet: dict[str, list[SampleFamilyEvidence]] = defaultdict(list)
    for row in sample_evidence:
        by_tracklet[row.tracklet_id].append(row)

    output: list[TrackletFamilyConsensus] = []

    for tracklet_id in sorted(by_tracklet):
        rows = sorted(
            by_tracklet[tracklet_id],
            key=lambda row: (
                row.timestamp_s,
                row.frame_index,
                row.detection_id,
            ),
        )

        total_quality = float(sum(row.quality for row in rows))
        known_rows = [
            row for row in rows
            if row.winner_family_key is not None
        ]
        known_quality = float(sum(row.quality for row in known_rows))
        known_fraction = (
            known_quality / total_quality
            if total_quality > 1e-12
            else 0.0
        )

        mass: dict[str, float] = defaultdict(float)
        for row in known_rows:
            assert row.winner_family_key is not None
            mass[row.winner_family_key] += row.vote_weight

        total_known_mass = float(sum(mass.values()))

        if not mass or total_known_mass <= 1e-12:
            candidate_key = None
            dominance = 0.0
            support_rows: list[SampleFamilyEvidence] = []
            conflict_rows: list[SampleFamilyEvidence] = []
        else:
            candidate_key = max(
                mass,
                key=lambda key: (
                    mass[key],
                    key,
                ),
            )
            dominance = float(
                mass[candidate_key] / total_known_mass
            )
            support_rows = [
                row for row in known_rows
                if row.winner_family_key == candidate_key
            ]
            conflict_rows = [
                row for row in known_rows
                if row.winner_family_key != candidate_key
            ]

        support_samples = len(support_rows)
        conflicting_samples = len(conflict_rows)
        unknown_samples = len(rows) - len(known_rows)
        conflict_fraction = (
            conflicting_samples / len(known_rows)
            if known_rows
            else 0.0
        )

        if candidate_key is None:
            candidate_type = None
            candidate_id = None
            temporal_bins = 0
            weighted_family_mass = 0.0
            mean_support_cosine = None
            mean_support_margin = None
        else:
            candidate_type, candidate_id = _parse_family_key(candidate_key)
            first_timestamp = rows[0].timestamp_s
            temporal_bins = len({
                int(math.floor(
                    (row.timestamp_s - first_timestamp)
                    / config.special_temporal_bin_s
                ))
                for row in support_rows
            })
            weighted_family_mass = float(mass[candidate_key])
            mean_support_cosine = float(np.mean([
                row.best_cosine for row in support_rows
            ]))
            mean_support_margin = float(np.mean([
                row.margin for row in support_rows
            ]))

        final_type = "unknown"
        final_id: int | None = None

        if candidate_type == "primary":
            good = (
                dominance >= config.primary_min_dominance
                and known_fraction >= config.primary_min_known_fraction
                and support_samples >= config.primary_min_support_samples
                and conflict_fraction
                <= config.primary_max_conflict_fraction
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
                dominance >= config.special_min_dominance
                and known_fraction >= config.special_min_known_fraction
                and support_samples >= config.special_min_support_samples
                and temporal_bins >= config.special_min_temporal_bins
                and conflict_fraction
                <= config.special_max_conflict_fraction
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

        output.append(
            TrackletFamilyConsensus(
                tracklet_id=tracklet_id,
                family_type=final_type,
                family_id=final_id,
                candidate_family_type=candidate_type,
                candidate_family_id=candidate_id,
                state=state,
                dominance=dominance,
                known_fraction=known_fraction,
                support_samples=support_samples,
                conflicting_samples=conflicting_samples,
                unknown_samples=unknown_samples,
                total_samples=len(rows),
                support_temporal_bins=temporal_bins,
                weighted_family_mass=weighted_family_mass,
                total_known_mass=total_known_mass,
                total_quality_mass=total_quality,
                conflict_fraction=conflict_fraction,
                mean_support_cosine=mean_support_cosine,
                mean_support_margin=mean_support_margin,
            )
        )

    return tuple(output)


def build_kit_family_consensus(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    profile_assignment_rows: Sequence[Mapping[str, Any]],
    cluster_rows: Sequence[Mapping[str, Any]],
    config: KitFamilyConsensusConfig | None = None,
) -> KitFamilyConsensusResult:
    selected = config or KitFamilyConsensusConfig()

    calibrations, family_keys, centroids = build_family_calibrations(
        cluster_rows=cluster_rows,
        profile_rows=profile_rows,
        profile_assignment_rows=profile_assignment_rows,
        config=selected,
    )

    evidence = classify_sample_evidence(
        sample_rows=sample_rows,
        calibrations=calibrations,
        family_keys=family_keys,
        centroids=centroids,
        config=selected,
    )

    tracklets = aggregate_tracklet_consensus(
        sample_evidence=evidence,
        config=selected,
    )

    return KitFamilyConsensusResult(
        calibrations=calibrations,
        sample_evidence=evidence,
        tracklets=tracklets,
        config=selected,
    )
