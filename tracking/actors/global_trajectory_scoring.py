"""Tracking V1 Hito 7G.3: score global trajectory candidates.

Input
-----
7G.2 produces a deliberately broad set of physically reachable APPEND and
INSERT hypotheses.  This module ranks those hypotheses using the already
validated Tracking V1 ReID calibration plus geometry, short-horizon motion,
family evidence and local competition.

This stage still DOES NOT merge identities.

Key design rules
----------------
* Reuse the 7E/7F empirical ReID calibration; do not invent a second appearance
  model.
* Geometry is soft after the hard reachability gate from 7G.2.
* Endpoint velocity is only soft evidence and only when 7G.2 says the short
  prediction horizon was used.
* Missing appearance/velocity/family evidence is unknown, not negative.
* Weak family agreement never gives a positive bonus.
* Different/conflicting primary-family evidence remains a strong soft penalty,
  not a hard cannot-link.
* Scores are ranking/evidence scores, NOT probabilities.
* No target population is assumed and no identity decision is changed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Mapping, Sequence

import numpy as np

from tracking.actors.identity_expansion import (
    IdentityCalibration,
    IdentityExpansionConfig,
    IdentityFamilyEvidence,
    IdentityProfile,
    build_identity_family_evidence,
    family_pair_relation,
    family_score_adjustment,
)


GLOBAL_TRAJECTORY_SCORING_VERSION = "global-trajectory-scoring-v2-centered-geometry"


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("geometric mean requires values")
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError("non-finite evidence value")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError("evidence values must be in [0,1]")
    if np.any(array == 0.0):
        return 0.0
    return float(np.exp(np.mean(np.log(array))))


def _unit(values: Sequence[float]) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("embedding must be non-empty 1-D")
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("embedding norm must be finite and > 0")
    return vector / norm


def _cosine(first: Sequence[float], second: Sequence[float]) -> float:
    a = _unit(first)
    b = _unit(second)
    if a.shape != b.shape:
        raise ValueError("embedding dimensions differ")
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def _rank_support(rank: int | None, floor: float) -> float:
    if rank is None:
        return floor
    return 1.0 / math.sqrt(max(1, rank))


def _rank_map(
    rows: Sequence[dict[str, Any]],
    *,
    group_key,
    score_key,
) -> dict[int, int]:
    grouped: dict[Any, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row[score_key] is None:
            continue
        grouped[group_key(row)].append(index)

    result: dict[int, int] = {}
    for indices in grouped.values():
        indices.sort(
            key=lambda index: (
                -float(rows[index][score_key]),
                rows[index]["stable_tie_key"],
            )
        )
        for rank, index in enumerate(indices, start=1):
            result[index] = rank
    return result


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryScoringConfig:
    one_sided_rank_floor: float = 0.05
    physical_speed_scale_mps: float = 6.0
    geometry_time_scale_s: float = 2.5
    velocity_error_scale_m: float = 2.5
    minimum_velocity_reliability: float = 0.10
    appearance_weight: float = 0.70
    geometry_weight: float = 0.30

    same_primary_bonus: float = 0.035
    different_primary_penalty: float = 0.300
    probable_different_primary_penalty: float = 0.200
    probable_primary_min_dominance: float = 0.60
    probable_primary_min_confidence: float = 0.20
    probable_primary_min_evidence_coverage: float = 0.15

    def __post_init__(self) -> None:
        if not 0.0 < self.one_sided_rank_floor <= 1.0:
            raise ValueError("one_sided_rank_floor must be in (0,1]")
        if self.physical_speed_scale_mps <= 0.0:
            raise ValueError("physical_speed_scale_mps must be > 0")
        if self.geometry_time_scale_s <= 0.0:
            raise ValueError("geometry_time_scale_s must be > 0")
        if self.velocity_error_scale_m <= 0.0:
            raise ValueError("velocity_error_scale_m must be > 0")
        if not 0.0 <= self.minimum_velocity_reliability <= 1.0:
            raise ValueError(
                "minimum_velocity_reliability must be in [0,1]"
            )
        if not 0.0 <= self.appearance_weight <= 1.0:
            raise ValueError("appearance_weight must be in [0,1]")
        if not 0.0 <= self.geometry_weight <= 1.0:
            raise ValueError("geometry_weight must be in [0,1]")
        if abs(
            self.appearance_weight + self.geometry_weight - 1.0
        ) > 1e-9:
            raise ValueError(
                "appearance_weight + geometry_weight must equal 1"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def family_config(self) -> IdentityExpansionConfig:
        return IdentityExpansionConfig(
            same_primary_bonus=self.same_primary_bonus,
            different_primary_penalty=self.different_primary_penalty,
            probable_different_primary_penalty=(
                self.probable_different_primary_penalty
            ),
            probable_primary_min_dominance=(
                self.probable_primary_min_dominance
            ),
            probable_primary_min_confidence=(
                self.probable_primary_min_confidence
            ),
            probable_primary_min_evidence_coverage=(
                self.probable_primary_min_evidence_coverage
            ),
        )


@dataclass(frozen=True, slots=True)
class ScoredAppendCandidate:
    source: Mapping[str, Any]
    appearance_available: bool
    aggregate_cosine: float | None
    empirical_support: float | None
    negative_tail_fraction: float | None
    profile_confidence: float | None
    outgoing_reid_rank: int | None
    incoming_reid_rank: int | None
    reid_competition_support: float | None
    appearance_score: float | None
    physical_effort_support: float
    temporal_informativeness: float
    velocity_support: float | None
    geometry_support: float
    family_relation: str
    family_pair_strength: float
    family_adjustment: float
    trajectory_base_score: float | None
    trajectory_score: float | None
    outgoing_score_rank: int | None = None
    incoming_score_rank: int | None = None
    mutual_best: bool = False
    mutual_top3: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.source)
        payload.update({
            "schema_version": 1,
            "scoring_version": GLOBAL_TRAJECTORY_SCORING_VERSION,
            "appearance_available": self.appearance_available,
            "aggregate_cosine": self.aggregate_cosine,
            "empirical_support": self.empirical_support,
            "negative_tail_fraction": self.negative_tail_fraction,
            "profile_confidence": self.profile_confidence,
            "outgoing_reid_rank": self.outgoing_reid_rank,
            "incoming_reid_rank": self.incoming_reid_rank,
            "reid_competition_support": self.reid_competition_support,
            "appearance_score": self.appearance_score,
            "physical_effort_support": self.physical_effort_support,
            "temporal_informativeness": self.temporal_informativeness,
            "velocity_support": self.velocity_support,
            "geometry_support": self.geometry_support,
            "family_relation": self.family_relation,
            "family_pair_strength": self.family_pair_strength,
            "family_adjustment": self.family_adjustment,
            "trajectory_base_score": self.trajectory_base_score,
            "trajectory_score": self.trajectory_score,
            "trajectory_score_semantics": (
                "ranking_evidence_score_not_probability"
            ),
            "outgoing_score_rank": self.outgoing_score_rank,
            "incoming_score_rank": self.incoming_score_rank,
            "mutual_best": self.mutual_best,
            "mutual_top3": self.mutual_top3,
            "identity_decision": None,
        })
        return payload


@dataclass(frozen=True, slots=True)
class ScoredInsertCandidate:
    source: Mapping[str, Any]
    appearance_available: bool
    aggregate_cosine: float | None
    empirical_support: float | None
    negative_tail_fraction: float | None
    profile_confidence: float | None
    host_gap_reid_rank: int | None
    inserted_reid_rank: int | None
    reid_competition_support: float | None
    appearance_score: float | None
    left_physical_effort_support: float
    right_physical_effort_support: float
    left_temporal_informativeness: float
    right_temporal_informativeness: float
    left_velocity_support: float | None
    right_velocity_support: float | None
    geometry_support: float
    family_relation: str
    family_pair_strength: float
    family_adjustment: float
    trajectory_base_score: float | None
    trajectory_score: float | None
    host_gap_score_rank: int | None = None
    inserted_score_rank: int | None = None
    mutual_best: bool = False
    mutual_top3: bool = False

    def to_dict(self) -> dict[str, Any]:
        payload = dict(self.source)
        payload.update({
            "schema_version": 1,
            "scoring_version": GLOBAL_TRAJECTORY_SCORING_VERSION,
            "appearance_available": self.appearance_available,
            "aggregate_cosine": self.aggregate_cosine,
            "empirical_support": self.empirical_support,
            "negative_tail_fraction": self.negative_tail_fraction,
            "profile_confidence": self.profile_confidence,
            "host_gap_reid_rank": self.host_gap_reid_rank,
            "inserted_reid_rank": self.inserted_reid_rank,
            "reid_competition_support": self.reid_competition_support,
            "appearance_score": self.appearance_score,
            "left_physical_effort_support": (
                self.left_physical_effort_support
            ),
            "right_physical_effort_support": (
                self.right_physical_effort_support
            ),
            "left_temporal_informativeness": (
                self.left_temporal_informativeness
            ),
            "right_temporal_informativeness": (
                self.right_temporal_informativeness
            ),
            "left_velocity_support": self.left_velocity_support,
            "right_velocity_support": self.right_velocity_support,
            "geometry_support": self.geometry_support,
            "family_relation": self.family_relation,
            "family_pair_strength": self.family_pair_strength,
            "family_adjustment": self.family_adjustment,
            "trajectory_base_score": self.trajectory_base_score,
            "trajectory_score": self.trajectory_score,
            "trajectory_score_semantics": (
                "ranking_evidence_score_not_probability"
            ),
            "host_gap_score_rank": self.host_gap_score_rank,
            "inserted_score_rank": self.inserted_score_rank,
            "mutual_best": self.mutual_best,
            "mutual_top3": self.mutual_top3,
            "identity_decision": None,
        })
        return payload


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryScoringResult:
    append_candidates: tuple[ScoredAppendCandidate, ...]
    insert_candidates: tuple[ScoredInsertCandidate, ...]
    config: GlobalTrajectoryScoringConfig

    def summary(self) -> dict[str, Any]:
        def bands(rows):
            counter = Counter()
            for row in rows:
                score = row.trajectory_score
                if score is None:
                    counter["missing_score"] += 1
                    continue
                for threshold in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
                    if score >= threshold:
                        counter[f"ge_{threshold:.2f}"] += 1
                        if row.mutual_best:
                            counter[
                                f"mutual_best_ge_{threshold:.2f}"
                            ] += 1
                        if row.mutual_top3:
                            counter[
                                f"mutual_top3_ge_{threshold:.2f}"
                            ] += 1
            return dict(sorted(counter.items()))

        return {
            "schema_version": 1,
            "scoring_version": GLOBAL_TRAJECTORY_SCORING_VERSION,
            "append_candidate_count": len(self.append_candidates),
            "insert_candidate_count": len(self.insert_candidates),
            "append_appearance_available_count": sum(
                row.appearance_available for row in self.append_candidates
            ),
            "insert_appearance_available_count": sum(
                row.appearance_available for row in self.insert_candidates
            ),
            "append_mutual_best_count": sum(
                row.mutual_best for row in self.append_candidates
            ),
            "append_mutual_top3_count": sum(
                row.mutual_top3 for row in self.append_candidates
            ),
            "insert_mutual_best_count": sum(
                row.mutual_best for row in self.insert_candidates
            ),
            "insert_mutual_top3_count": sum(
                row.mutual_top3 for row in self.insert_candidates
            ),
            "append_score_bands": bands(self.append_candidates),
            "insert_score_bands": bands(self.insert_candidates),
            "append_family_relation_counts": dict(sorted(Counter(
                row.family_relation for row in self.append_candidates
            ).items())),
            "insert_family_relation_counts": dict(sorted(Counter(
                row.family_relation for row in self.insert_candidates
            ).items())),
            "config": self.config.to_dict(),
            "identity_decisions_changed": False,
            "population_forced": False,
            "score_is_probability": False,
        }


def identity_profiles_from_rows(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, IdentityProfile]:
    result: dict[str, IdentityProfile] = {}
    for row in rows:
        identity_id = str(row["identity_id"])
        if identity_id in result:
            raise ValueError(f"duplicate profile {identity_id}")
        result[identity_id] = IdentityProfile(
            identity_id=identity_id,
            source_actor_ids=tuple(
                str(value) for value in row.get("source_actor_ids", ())
            ),
            tracklet_ids=tuple(
                str(value) for value in row.get("tracklet_ids", ())
            ),
            profiled_actor_ids=tuple(
                str(value) for value in row.get("profiled_actor_ids", ())
            ),
            profiled_tracklet_count=int(
                row["profiled_tracklet_count"]
            ),
            aggregate_embedding=tuple(
                float(value) for value in row["aggregate_embedding"]
            ),
            reliability=_clamp01(float(row["reliability"])),
            start_time_s=float(row["start_time_s"]),
            end_time_s=float(row["end_time_s"]),
            kit_cluster_id=(
                None
                if row.get("kit_cluster_id") is None
                else int(row["kit_cluster_id"])
            ),
            kit_dominance=(
                None
                if row.get("kit_dominance") is None
                else float(row["kit_dominance"])
            ),
            initial_seed=bool(row.get("initial_seed", False)),
        )
    return result


def _temporal_informativeness(
    geometry: Mapping[str, Any],
    *,
    config: GlobalTrajectoryScoringConfig,
) -> float:
    """How informative geometry is at this temporal separation.

    A long-gap transition that is merely easy to reach is *not* strong identity
    evidence.  Informativeness therefore decays toward zero with gap duration.
    """
    gap_s = max(0.0, float(geometry.get("gap_s", 0.0)))
    return _clamp01(
        math.exp(-gap_s / config.geometry_time_scale_s)
    )


def _physical_effort_support(
    geometry: Mapping[str, Any],
    *,
    config: GlobalTrajectoryScoringConfig,
) -> float:
    """Centered physical support.

    0.5 is neutral.  At short gaps, an easy transition can approach 1 and a
    near-limit sprint can fall below 0.5.  At long gaps, both cases decay back
    toward 0.5 because mere reachability is weak identity evidence.
    """
    speed = geometry.get("required_speed_after_slack_mps")
    informativeness = _temporal_informativeness(
        geometry,
        config=config,
    )
    if speed is None:
        raw_ease = 0.5
    else:
        value = max(0.0, float(speed))
        scale = config.physical_speed_scale_mps
        raw_ease = _clamp01(
            math.exp(-0.5 * (value / scale) ** 2)
        )
    return _clamp01(
        0.5 + informativeness * (raw_ease - 0.5)
    )


def _velocity_support(
    geometry: Mapping[str, Any],
    *,
    config: GlobalTrajectoryScoringConfig,
) -> float | None:
    if not bool(geometry.get("velocity_prediction_used", False)):
        return None

    pieces: list[float] = []

    def error_piece(error_key: str, reliability_key: str) -> None:
        error = geometry.get(error_key)
        reliability = geometry.get(reliability_key)
        if error is None or reliability is None:
            return
        rel = _clamp01(float(reliability))
        if rel < config.minimum_velocity_reliability:
            return
        raw = math.exp(
            -max(0.0, float(error)) / config.velocity_error_scale_m
        )
        pieces.append(_clamp01(0.5 + rel * (raw - 0.5)))

    error_piece(
        "forward_prediction_error_m",
        "predecessor_velocity_reliability",
    )
    error_piece(
        "backward_prediction_error_m",
        "successor_velocity_reliability",
    )

    cosine = geometry.get("velocity_direction_cosine")
    pred_rel = geometry.get("predecessor_velocity_reliability")
    succ_rel = geometry.get("successor_velocity_reliability")
    if cosine is not None and pred_rel is not None and succ_rel is not None:
        rel = math.sqrt(
            _clamp01(float(pred_rel))
            * _clamp01(float(succ_rel))
        )
        if rel >= config.minimum_velocity_reliability:
            raw = _clamp01((float(cosine) + 1.0) / 2.0)
            pieces.append(_clamp01(0.5 + rel * (raw - 0.5)))

    if not pieces:
        return None
    return _geometric_mean(pieces)


def _geometry_support(
    geometries: Sequence[Mapping[str, Any]],
    *,
    config: GlobalTrajectoryScoringConfig,
) -> tuple[
    float,
    list[float],
    list[float],
    list[float | None],
]:
    effort = [
        _physical_effort_support(row, config=config)
        for row in geometries
    ]
    informativeness = [
        _temporal_informativeness(row, config=config)
        for row in geometries
    ]
    velocity = [
        _velocity_support(row, config=config)
        for row in geometries
    ]

    boundary_supports: list[float] = []
    for effort_value, velocity_value in zip(effort, velocity):
        if velocity_value is None:
            boundary_supports.append(effort_value)
        else:
            boundary_supports.append(
                _geometric_mean([effort_value, velocity_value])
            )

    # INSERT has two required boundaries.  One strong boundary must not hide an
    # uninformative/weak opposite boundary, so the weakest boundary governs.
    if len(boundary_supports) == 1:
        combined = boundary_supports[0]
    else:
        combined = min(boundary_supports)

    return (
        float(combined),
        effort,
        informativeness,
        velocity,
    )


def _combine_appearance_and_geometry(
    appearance: float,
    geometry: float,
    *,
    config: GlobalTrajectoryScoringConfig,
) -> float:
    """Combine centered evidence without moving neutral 0.5 upward.

    This fixes the v1 geometric-mean artefact where neutral appearance (0.5)
    plus perfect geometry (1.0) became sqrt(0.5) ~= 0.707.
    """
    return _clamp01(
        0.5
        + config.appearance_weight * (appearance - 0.5)
        + config.geometry_weight * (geometry - 0.5)
    )


def _family_features(
    left_id: str,
    right_id: str,
    family_evidence: Mapping[str, IdentityFamilyEvidence],
    *,
    config: GlobalTrajectoryScoringConfig,
) -> tuple[str, float, float]:
    family_config = config.family_config()
    if not family_evidence:
        return "neutral", 0.0, 0.0
    relation, strength = family_pair_relation(
        left_id,
        right_id,
        family_evidence,
        config=family_config,
    )
    adjustment = family_score_adjustment(
        relation=relation,
        pair_strength=strength,
        config=family_config,
    )
    return relation, float(strength), float(adjustment)


def _appearance_raw_rows(
    candidates: Sequence[Mapping[str, Any]],
    *,
    pair_ids,
    group_left,
    group_right,
    profiles: Mapping[str, IdentityProfile],
    calibration: IdentityCalibration,
    config: GlobalTrajectoryScoringConfig,
) -> tuple[list[dict[str, Any]], dict[int, int], dict[int, int]]:
    rows: list[dict[str, Any]] = []
    for source in candidates:
        left_id, right_id = pair_ids(source)
        first = profiles.get(left_id)
        second = profiles.get(right_id)

        cosine = None
        empirical = None
        negative_tail = None
        confidence = None
        if first is not None and second is not None:
            cosine = _cosine(
                first.aggregate_embedding,
                second.aggregate_embedding,
            )
            support = calibration.support(cosine, family="same_kit")
            empirical = float(support["support"])
            negative_tail = float(support["negative_tail_fraction"])
            confidence = math.sqrt(
                max(0.0, first.reliability * second.reliability)
            )

        rows.append({
            "source": dict(source),
            "left_id": left_id,
            "right_id": right_id,
            "cosine": cosine,
            "empirical": empirical,
            "negative_tail": negative_tail,
            "confidence": confidence,
            "stable_tie_key": repr(source),
        })

    left_rank = _rank_map(
        rows,
        group_key=group_left,
        score_key="cosine",
    )
    right_rank = _rank_map(
        rows,
        group_key=group_right,
        score_key="cosine",
    )
    return rows, left_rank, right_rank


def score_append_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, IdentityProfile],
    calibration: IdentityCalibration,
    family_evidence: Mapping[str, IdentityFamilyEvidence],
    config: GlobalTrajectoryScoringConfig,
) -> tuple[ScoredAppendCandidate, ...]:
    raw, outgoing_reid, incoming_reid = _appearance_raw_rows(
        candidates,
        pair_ids=lambda row: (
            str(row["predecessor_identity_id"]),
            str(row["successor_identity_id"]),
        ),
        group_left=lambda row: row["left_id"],
        group_right=lambda row: row["right_id"],
        profiles=profiles,
        calibration=calibration,
        config=config,
    )

    scored: list[ScoredAppendCandidate] = []
    for index, row in enumerate(raw):
        source = row["source"]
        out_rank = outgoing_reid.get(index)
        in_rank = incoming_reid.get(index)

        competition = None
        appearance = None
        if row["cosine"] is not None:
            competition = _geometric_mean([
                _rank_support(out_rank, config.one_sided_rank_floor),
                _rank_support(in_rank, config.one_sided_rank_floor),
            ])
            raw_support = _geometric_mean([
                float(row["empirical"]),
                competition,
            ])
            appearance = _clamp01(
                0.5
                + float(row["confidence"])
                * (raw_support - 0.5)
            )

        (
            geometry_support,
            effort,
            informativeness,
            velocity,
        ) = _geometry_support(
            [source["geometry"]],
            config=config,
        )

        relation, strength, adjustment = _family_features(
            row["left_id"],
            row["right_id"],
            family_evidence,
            config=config,
        )

        base = None
        score = None
        if appearance is not None:
            base = _combine_appearance_and_geometry(
                appearance,
                geometry_support,
                config=config,
            )
            score = _clamp01(base + adjustment)

        scored.append(
            ScoredAppendCandidate(
                source=source,
                appearance_available=appearance is not None,
                aggregate_cosine=row["cosine"],
                empirical_support=row["empirical"],
                negative_tail_fraction=row["negative_tail"],
                profile_confidence=row["confidence"],
                outgoing_reid_rank=out_rank,
                incoming_reid_rank=in_rank,
                reid_competition_support=competition,
                appearance_score=appearance,
                physical_effort_support=effort[0],
                temporal_informativeness=informativeness[0],
                velocity_support=velocity[0],
                geometry_support=geometry_support,
                family_relation=relation,
                family_pair_strength=strength,
                family_adjustment=adjustment,
                trajectory_base_score=base,
                trajectory_score=score,
            )
        )

    helper = [
        {
            "score": row.trajectory_score,
            "left": row.source["predecessor_identity_id"],
            "right": row.source["successor_identity_id"],
            "stable_tie_key": (
                str(row.source["predecessor_identity_id"])
                + "->"
                + str(row.source["successor_identity_id"])
            ),
        }
        for row in scored
    ]
    out_score = _rank_map(
        helper,
        group_key=lambda row: row["left"],
        score_key="score",
    )
    in_score = _rank_map(
        helper,
        group_key=lambda row: row["right"],
        score_key="score",
    )

    final = []
    for index, row in enumerate(scored):
        out_rank = out_score.get(index)
        in_rank = in_score.get(index)
        final.append(replace(
            row,
            outgoing_score_rank=out_rank,
            incoming_score_rank=in_rank,
            mutual_best=(out_rank == 1 and in_rank == 1),
            mutual_top3=(
                out_rank is not None
                and in_rank is not None
                and out_rank <= 3
                and in_rank <= 3
            ),
        ))
    return tuple(final)


def score_insert_candidates(
    *,
    candidates: Sequence[Mapping[str, Any]],
    profiles: Mapping[str, IdentityProfile],
    calibration: IdentityCalibration,
    family_evidence: Mapping[str, IdentityFamilyEvidence],
    config: GlobalTrajectoryScoringConfig,
) -> tuple[ScoredInsertCandidate, ...]:
    raw, host_gap_reid, inserted_reid = _appearance_raw_rows(
        candidates,
        pair_ids=lambda row: (
            str(row["host_identity_id"]),
            str(row["inserted_identity_id"]),
        ),
        group_left=lambda row: (
            row["left_id"],
            int(row["source"]["host_gap_index"]),
        ),
        group_right=lambda row: row["right_id"],
        profiles=profiles,
        calibration=calibration,
        config=config,
    )

    scored: list[ScoredInsertCandidate] = []
    for index, row in enumerate(raw):
        source = row["source"]
        host_rank = host_gap_reid.get(index)
        inserted_rank = inserted_reid.get(index)

        competition = None
        appearance = None
        if row["cosine"] is not None:
            competition = _geometric_mean([
                _rank_support(host_rank, config.one_sided_rank_floor),
                _rank_support(inserted_rank, config.one_sided_rank_floor),
            ])
            raw_support = _geometric_mean([
                float(row["empirical"]),
                competition,
            ])
            appearance = _clamp01(
                0.5
                + float(row["confidence"])
                * (raw_support - 0.5)
            )

        (
            geometry_support,
            effort,
            informativeness,
            velocity,
        ) = _geometry_support(
            [
                source["left_geometry"],
                source["right_geometry"],
            ],
            config=config,
        )
        relation, strength, adjustment = _family_features(
            row["left_id"],
            row["right_id"],
            family_evidence,
            config=config,
        )

        base = None
        score = None
        if appearance is not None:
            base = _combine_appearance_and_geometry(
                appearance,
                geometry_support,
                config=config,
            )
            score = _clamp01(base + adjustment)

        scored.append(
            ScoredInsertCandidate(
                source=source,
                appearance_available=appearance is not None,
                aggregate_cosine=row["cosine"],
                empirical_support=row["empirical"],
                negative_tail_fraction=row["negative_tail"],
                profile_confidence=row["confidence"],
                host_gap_reid_rank=host_rank,
                inserted_reid_rank=inserted_rank,
                reid_competition_support=competition,
                appearance_score=appearance,
                left_physical_effort_support=effort[0],
                right_physical_effort_support=effort[1],
                left_temporal_informativeness=informativeness[0],
                right_temporal_informativeness=informativeness[1],
                left_velocity_support=velocity[0],
                right_velocity_support=velocity[1],
                geometry_support=geometry_support,
                family_relation=relation,
                family_pair_strength=strength,
                family_adjustment=adjustment,
                trajectory_base_score=base,
                trajectory_score=score,
            )
        )

    helper = [
        {
            "score": row.trajectory_score,
            "host_gap": (
                str(row.source["host_identity_id"]),
                int(row.source["host_gap_index"]),
            ),
            "inserted": str(row.source["inserted_identity_id"]),
            "stable_tie_key": (
                str(row.source["host_identity_id"])
                + ":"
                + str(row.source["host_gap_index"])
                + "<-"
                + str(row.source["inserted_identity_id"])
            ),
        }
        for row in scored
    ]
    host_score = _rank_map(
        helper,
        group_key=lambda row: row["host_gap"],
        score_key="score",
    )
    inserted_score = _rank_map(
        helper,
        group_key=lambda row: row["inserted"],
        score_key="score",
    )

    final = []
    for index, row in enumerate(scored):
        host_rank = host_score.get(index)
        inserted_rank = inserted_score.get(index)
        final.append(replace(
            row,
            host_gap_score_rank=host_rank,
            inserted_score_rank=inserted_rank,
            mutual_best=(host_rank == 1 and inserted_rank == 1),
            mutual_top3=(
                host_rank is not None
                and inserted_rank is not None
                and host_rank <= 3
                and inserted_rank <= 3
            ),
        ))
    return tuple(final)


def score_global_trajectory_candidates(
    *,
    append_candidates: Sequence[Mapping[str, Any]],
    insert_candidates: Sequence[Mapping[str, Any]],
    profile_rows: Sequence[Mapping[str, Any]],
    calibration: IdentityCalibration,
    family_rows: Sequence[Mapping[str, Any]] = (),
    config: GlobalTrajectoryScoringConfig | None = None,
) -> GlobalTrajectoryScoringResult:
    selected = config or GlobalTrajectoryScoringConfig()
    profiles = identity_profiles_from_rows(profile_rows)
    family_evidence = build_identity_family_evidence(family_rows)

    append = score_append_candidates(
        candidates=append_candidates,
        profiles=profiles,
        calibration=calibration,
        family_evidence=family_evidence,
        config=selected,
    )
    insert = score_insert_candidates(
        candidates=insert_candidates,
        profiles=profiles,
        calibration=calibration,
        family_evidence=family_evidence,
        config=selected,
    )
    return GlobalTrajectoryScoringResult(
        append_candidates=append,
        insert_candidates=insert,
        config=selected,
    )
