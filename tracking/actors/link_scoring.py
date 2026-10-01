"""Tracking V1 Hito 7E.1: conservative sparse actor-link scoring.

Consumes the sparse 7B candidate graph plus 7C kit assignments and 7D ReID
calibration.  This module DOES NOT merge tracklets or assign identities.

Main idea
---------
A large cosine by itself is unsafe in football because same-kit players can be
extremely similar.  Therefore each candidate retains multiple independent
pieces of evidence:

* local field-space continuity, when available;
* ReID calibrated against empirical same/different-kit hard negatives;
* chronological same-tracklet split positives;
* exact directed retrieval ranks;
* rich appearance reliability/support quality;
* soft anonymous-kit compatibility;
* competition against alternative candidates incident to each endpoint.

The final ``rank_score`` is only a relative ranking heuristic.  It is NOT a
probability and NOT a merge threshold.  The future solver must still obey all
hard cannot-links and may prefer fragmentation over contamination.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
import math
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ACTOR_LINK_SCORING_VERSION = "actor-link-evidence-v1"


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


def _geometric_mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)):
        raise ValueError("non-finite value in geometric mean")
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


@dataclass(frozen=True, slots=True)
class RobustScale:
    p10: float
    p90: float
    higher_is_better: bool

    @classmethod
    def fit(
        cls,
        values: Sequence[float],
        *,
        higher_is_better: bool,
    ) -> "RobustScale | None":
        if not values:
            return None
        array = np.asarray(values, dtype=np.float64)
        return cls(
            p10=float(np.percentile(array, 10)),
            p90=float(np.percentile(array, 90)),
            higher_is_better=higher_is_better,
        )

    def map(self, value: float) -> float:
        value = _finite("scale value", value)
        if self.p90 <= self.p10 + 1e-12:
            return 0.5
        x = _clamp01((value - self.p10) / (self.p90 - self.p10))
        # smoothstep avoids sharp artificial corners.
        x = x * x * (3.0 - 2.0 * x)
        return x if self.higher_is_better else 1.0 - x

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class KitAssignment:
    tracklet_id: str
    cluster_id: int
    best_cosine: float
    margin: float
    profile_reliability: float

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "KitAssignment":
        return cls(
            tracklet_id=str(row["tracklet_id"]),
            cluster_id=int(row["cluster_id"]),
            best_cosine=_finite("best_cosine", row["best_cosine"]),
            margin=_finite("margin", row["margin"]),
            profile_reliability=_clamp01(
                _finite(
                    "profile_reliability",
                    row["profile_reliability"],
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class EmpiricalReIDCalibration:
    split_positive: tuple[float, ...]
    same_kit_negative: tuple[float, ...]
    different_kit_negative: tuple[float, ...]
    unknown_kit_negative: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.split_positive:
            raise ValueError("split_positive calibration is empty")
        if not self.same_kit_negative:
            raise ValueError("same-kit negative calibration is empty")

    def negatives_for(self, family: str) -> tuple[float, ...]:
        if family == "same_kit":
            return self.same_kit_negative
        if family == "different_kit":
            return (
                self.different_kit_negative
                if self.different_kit_negative
                else self.same_kit_negative
            )
        if family == "unknown_kit":
            return (
                self.unknown_kit_negative
                if self.unknown_kit_negative
                else self.same_kit_negative
            )
        raise ValueError(f"unknown kit family {family!r}")

    @staticmethod
    def _cdf(values: Sequence[float], value: float) -> float:
        ordered = values
        # Jeffreys-like smoothing keeps exact extremes away from 0/1.
        count_le = bisect_right(ordered, float(value))
        return float((count_le + 0.5) / (len(ordered) + 1.0))

    def reid_support(
        self,
        cosine: float,
        *,
        kit_family: str,
    ) -> dict[str, float]:
        cosine = _finite("cosine", cosine)
        negative_cdf = self._cdf(
            self.negatives_for(kit_family),
            cosine,
        )
        positive_cdf = self._cdf(
            self.split_positive,
            cosine,
        )

        # Strong evidence requires BOTH:
        #   1. unusually high similarity for physically impossible pairs; and
        #   2. similarity that looks plausible inside same-actor split pairs.
        combined = _geometric_mean(
            [negative_cdf, positive_cdf]
        )
        assert combined is not None

        return {
            "negative_cdf": negative_cdf,
            "negative_tail_fraction": 1.0 - negative_cdf,
            "positive_cdf": positive_cdf,
            "support": combined,
        }


@dataclass(frozen=True, slots=True)
class ActorLinkScoringConfig:
    one_sided_rank_floor: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.one_sided_rank_floor <= 1.0:
            raise ValueError(
                "one_sided_rank_floor must be in (0,1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ActorScoredLink:
    tracklet_a: str
    tracklet_b: str
    reasons: tuple[str, ...]
    link_family: str
    rank_score: float
    evidence_confidence: float
    components: Mapping[str, Any]
    endpoint_rank_a: int | None = None
    endpoint_rank_b: int | None = None
    endpoint_margin_a: float | None = None
    endpoint_margin_b: float | None = None
    mutual_best: bool = False
    mutual_top3: bool = False

    def __post_init__(self) -> None:
        if self.tracklet_a >= self.tracklet_b:
            raise ValueError("pair must be canonical")
        if not 0.0 <= self.rank_score <= 1.0:
            raise ValueError("rank_score must be in [0,1]")
        if not 0.0 <= self.evidence_confidence <= 1.0:
            raise ValueError(
                "evidence_confidence must be in [0,1]"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "scoring_version": ACTOR_LINK_SCORING_VERSION,
            "tracklet_a": self.tracklet_a,
            "tracklet_b": self.tracklet_b,
            "reasons": list(self.reasons),
            "link_family": self.link_family,
            "rank_score": self.rank_score,
            "rank_score_semantics": (
                "relative_evidence_ranking_not_probability"
            ),
            "evidence_confidence": self.evidence_confidence,
            "components": dict(self.components),
            "endpoint_rank_a": self.endpoint_rank_a,
            "endpoint_rank_b": self.endpoint_rank_b,
            "endpoint_margin_a": self.endpoint_margin_a,
            "endpoint_margin_b": self.endpoint_margin_b,
            "mutual_best": self.mutual_best,
            "mutual_top3": self.mutual_top3,
            "identity_decision": None,
        }


def _kit_family(
    first: KitAssignment | None,
    second: KitAssignment | None,
) -> str:
    if first is None or second is None:
        return "unknown_kit"
    return (
        "same_kit"
        if first.cluster_id == second.cluster_id
        else "different_kit"
    )


def _kit_pair_component(
    first: KitAssignment | None,
    second: KitAssignment | None,
    *,
    margin_scale: RobustScale | None,
    cosine_scale: RobustScale | None,
) -> dict[str, Any]:
    family = _kit_family(first, second)
    if first is None or second is None:
        return {
            "family": family,
            "available": False,
            "confidence": 0.0,
            "support": 0.5,
        }

    confidence_parts = [
        first.profile_reliability,
        second.profile_reliability,
    ]
    if margin_scale is not None:
        confidence_parts.extend(
            [
                margin_scale.map(first.margin),
                margin_scale.map(second.margin),
            ]
        )
    if cosine_scale is not None:
        confidence_parts.extend(
            [
                cosine_scale.map(first.best_cosine),
                cosine_scale.map(second.best_cosine),
            ]
        )

    confidence = _geometric_mean(confidence_parts)
    assert confidence is not None

    # Anonymous kit relation is soft. Same cluster nudges upward; different
    # cluster nudges downward. Low-confidence assignments stay near neutral.
    support = (
        0.5 + 0.5 * confidence
        if family == "same_kit"
        else 0.5 - 0.5 * confidence
    )

    return {
        "family": family,
        "available": True,
        "cluster_a": first.cluster_id,
        "cluster_b": second.cluster_id,
        "assignment_confidence": confidence,
        "support": _clamp01(support),
        "a": {
            "best_cosine": first.best_cosine,
            "margin": first.margin,
            "profile_reliability": first.profile_reliability,
        },
        "b": {
            "best_cosine": second.best_cosine,
            "margin": second.margin,
            "profile_reliability": second.profile_reliability,
        },
    }


def _rank_component(
    retrieval: Mapping[str, Any] | None,
    config: ActorLinkScoringConfig,
) -> dict[str, Any]:
    if retrieval is None:
        return {
            "available": False,
            "rank_a_to_b": None,
            "rank_b_to_a": None,
            "mutual_topk": False,
            "support": None,
        }

    rank_a = retrieval.get("rank_a_to_b")
    rank_b = retrieval.get("rank_b_to_a")
    rank_a = int(rank_a) if rank_a is not None else None
    rank_b = int(rank_b) if rank_b is not None else None

    def one(rank: int | None) -> float:
        if rank is None:
            return config.one_sided_rank_floor
        return 1.0 / math.sqrt(max(1, rank))

    support = _geometric_mean([one(rank_a), one(rank_b)])
    assert support is not None
    return {
        "available": True,
        "rank_a_to_b": rank_a,
        "rank_b_to_a": rank_b,
        "mutual_topk": rank_a is not None and rank_b is not None,
        "support": support,
    }


def _appearance_component(
    candidate: Mapping[str, Any],
    *,
    calibration: EmpiricalReIDCalibration,
    kit_family: str,
    config: ActorLinkScoringConfig,
) -> dict[str, Any]:
    retrieval = candidate.get("appearance_retrieval")
    rich = candidate.get("appearance_evidence")

    if not isinstance(retrieval, Mapping):
        return {
            "available": False,
            "support": None,
            "confidence": 0.0,
            "rank": _rank_component(None, config),
        }

    cosine = _finite(
        "appearance_retrieval.aggregate_cosine",
        retrieval["aggregate_cosine"],
    )
    empirical = calibration.reid_support(
        cosine,
        kit_family=kit_family,
    )
    rank = _rank_component(retrieval, config)

    confidence_parts: list[float] = []
    rich_payload: dict[str, Any] | None = None
    if isinstance(rich, Mapping):
        reliability = _clamp01(
            _finite(
                "appearance reliability",
                rich.get("reliability", 0.0),
            )
        )
        support_quality = _clamp01(
            _finite(
                "appearance support_quality",
                rich.get("support_quality", 0.0),
            )
        )
        confidence_parts.extend(
            [reliability, support_quality]
        )
        rich_payload = {
            "reliability": reliability,
            "support_quality": support_quality,
            "sample_count_a": rich.get("sample_count_a"),
            "sample_count_b": rich.get("sample_count_b"),
            "effective_sample_count_a": rich.get(
                "effective_sample_count_a"
            ),
            "effective_sample_count_b": rich.get(
                "effective_sample_count_b"
            ),
            "cross_similarity": rich.get("cross_similarity"),
            "bidirectional_best_similarity": rich.get(
                "bidirectional_best_similarity"
            ),
        }

    confidence = (
        _geometric_mean(confidence_parts)
        if confidence_parts
        else 0.35
    )
    assert confidence is not None

    support_inputs = [empirical["support"]]
    if rank["support"] is not None:
        support_inputs.append(float(rank["support"]))
    raw_support = _geometric_mean(support_inputs)
    assert raw_support is not None

    # Low-quality appearance is pulled toward neutral 0.5 rather than allowed
    # to strongly support OR strongly reject a link.
    effective_support = (
        0.5 + confidence * (raw_support - 0.5)
    )

    return {
        "available": True,
        "aggregate_cosine": cosine,
        "kit_conditioning_family": kit_family,
        "empirical": empirical,
        "rank": rank,
        "raw_support": raw_support,
        "confidence": confidence,
        "effective_support": _clamp01(effective_support),
        "rich": rich_payload,
    }


def _fit_local_scales(
    candidates: Sequence[Mapping[str, Any]],
) -> dict[str, RobustScale | None]:
    gaps: list[float] = []
    speeds: list[float] = []
    distances: list[float] = []
    prediction_errors: list[float] = []

    for candidate in candidates:
        local = candidate.get("local_transition")
        if not isinstance(local, Mapping):
            continue

        gaps.append(_finite("gap_s", local["gap_s"]))
        distances.append(
            _finite(
                "endpoint_distance_m",
                local["endpoint_distance_m"],
            )
        )
        speed = local.get("required_speed_after_slack_mps")
        if speed is not None:
            speeds.append(
                _finite(
                    "required_speed_after_slack_mps",
                    speed,
                )
            )

        for key in (
            "forward_prediction_error_m",
            "backward_prediction_error_m",
        ):
            value = local.get(key)
            if value is not None:
                prediction_errors.append(
                    _finite(key, value)
                )

    return {
        "gap_s": RobustScale.fit(
            gaps,
            higher_is_better=False,
        ),
        "required_speed_mps": RobustScale.fit(
            speeds,
            higher_is_better=False,
        ),
        "endpoint_distance_m": RobustScale.fit(
            distances,
            higher_is_better=False,
        ),
        "prediction_error_m": RobustScale.fit(
            prediction_errors,
            higher_is_better=False,
        ),
    }


def _local_component(
    local: Mapping[str, Any] | None,
    scales: Mapping[str, RobustScale | None],
) -> dict[str, Any]:
    if local is None:
        return {
            "available": False,
            "support": None,
            "confidence": 0.0,
        }

    values: dict[str, float | None] = {}
    supports: list[float] = []

    def add(
        name: str,
        raw: Any,
        scale_name: str,
    ) -> None:
        if raw is None:
            values[name] = None
            return
        value = _finite(name, raw)
        values[name] = value
        scale = scales.get(scale_name)
        if scale is not None:
            supports.append(scale.map(value))

    add("gap_s", local.get("gap_s"), "gap_s")
    add(
        "required_speed_after_slack_mps",
        local.get("required_speed_after_slack_mps"),
        "required_speed_mps",
    )
    add(
        "endpoint_distance_m",
        local.get("endpoint_distance_m"),
        "endpoint_distance_m",
    )
    add(
        "forward_prediction_error_m",
        local.get("forward_prediction_error_m"),
        "prediction_error_m",
    )
    add(
        "backward_prediction_error_m",
        local.get("backward_prediction_error_m"),
        "prediction_error_m",
    )

    direction = local.get("velocity_direction_cosine")
    if direction is not None:
        direction = _finite(
            "velocity_direction_cosine",
            direction,
        )
        values["velocity_direction_cosine"] = direction
        supports.append(
            _clamp01((direction + 1.0) * 0.5)
        )
    else:
        values["velocity_direction_cosine"] = None

    support = _geometric_mean(supports)
    if support is None:
        support = 0.5

    confidence_values = []
    for key in (
        "predecessor_end_confidence",
        "successor_start_confidence",
    ):
        if local.get(key) is not None:
            confidence_values.append(
                _clamp01(_finite(key, local[key]))
            )
    confidence = (
        _geometric_mean(confidence_values)
        if confidence_values
        else 0.5
    )
    assert confidence is not None

    return {
        "available": True,
        "support": support,
        "confidence": confidence,
        "features": values,
    }


def _link_family(
    candidate: Mapping[str, Any],
) -> str:
    reasons = set(str(v) for v in candidate.get("reasons", []))
    if "local_physics" in reasons:
        return "local_continuation"
    if "close_overlap" in reasons:
        return "close_overlap"
    return "appearance_rescue"


def _score_one(
    candidate: Mapping[str, Any],
    *,
    kit_by_id: Mapping[str, KitAssignment],
    calibration: EmpiricalReIDCalibration,
    kit_margin_scale: RobustScale | None,
    kit_cosine_scale: RobustScale | None,
    local_scales: Mapping[str, RobustScale | None],
    config: ActorLinkScoringConfig,
) -> ActorScoredLink:
    a = str(candidate["tracklet_a"])
    b = str(candidate["tracklet_b"])
    if a >= b:
        raise ValueError("candidate pair must be canonical")

    reasons = tuple(
        sorted(str(v) for v in candidate.get("reasons", []))
    )
    if not reasons:
        raise ValueError("candidate requires reasons")

    kit = _kit_pair_component(
        kit_by_id.get(a),
        kit_by_id.get(b),
        margin_scale=kit_margin_scale,
        cosine_scale=kit_cosine_scale,
    )
    appearance = _appearance_component(
        candidate,
        calibration=calibration,
        kit_family=str(kit["family"]),
        config=config,
    )
    local = _local_component(
        (
            candidate["local_transition"]
            if isinstance(
                candidate.get("local_transition"),
                Mapping,
            )
            else None
        ),
        local_scales,
    )

    family = _link_family(candidate)

    support_parts: list[float] = []
    confidence_parts: list[float] = []

    if local["available"]:
        support_parts.append(float(local["support"]))
        confidence_parts.append(float(local["confidence"]))

    if appearance["available"]:
        support_parts.append(
            float(appearance["effective_support"])
        )
        confidence_parts.append(
            float(appearance["confidence"])
        )

    # Kit is intentionally weak relative to physics/ReID. It can nudge but
    # cannot dominate a candidate by itself.
    kit_support = float(kit["support"])
    support_parts.append(
        0.5 + 0.50 * (kit_support - 0.5)
    )

    rank_score = _geometric_mean(support_parts)
    assert rank_score is not None

    evidence_confidence = (
        _geometric_mean(confidence_parts)
        if confidence_parts
        else 0.25
    )
    assert evidence_confidence is not None

    components = {
        "local": local,
        "appearance": appearance,
        "kit": kit,
        "source": {
            "reasons": list(reasons),
            "has_close_overlap": (
                isinstance(
                    candidate.get("close_overlap"),
                    Mapping,
                )
            ),
        },
    }

    return ActorScoredLink(
        tracklet_a=a,
        tracklet_b=b,
        reasons=reasons,
        link_family=family,
        rank_score=_clamp01(rank_score),
        evidence_confidence=_clamp01(evidence_confidence),
        components=components,
    )


def _with_competition(
    links: Sequence[ActorScoredLink],
) -> tuple[ActorScoredLink, ...]:
    incident: dict[str, list[tuple[float, str, int]]] = defaultdict(list)
    for index, link in enumerate(links):
        incident[link.tracklet_a].append(
            (link.rank_score, link.tracklet_b, index)
        )
        incident[link.tracklet_b].append(
            (link.rank_score, link.tracklet_a, index)
        )

    per_endpoint: dict[
        tuple[int, str],
        tuple[int, float | None],
    ] = {}

    for node, rows in incident.items():
        rows.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )
        for position, (score, _, link_index) in enumerate(
            rows,
            start=1,
        ):
            if len(rows) == 1:
                margin = None
            elif position == 1:
                margin = float(score - rows[1][0])
            else:
                margin = float(score - rows[0][0])
            per_endpoint[(link_index, node)] = (
                position,
                margin,
            )

    result: list[ActorScoredLink] = []
    for index, link in enumerate(links):
        rank_a, margin_a = per_endpoint[
            (index, link.tracklet_a)
        ]
        rank_b, margin_b = per_endpoint[
            (index, link.tracklet_b)
        ]
        result.append(
            replace(
                link,
                endpoint_rank_a=rank_a,
                endpoint_rank_b=rank_b,
                endpoint_margin_a=margin_a,
                endpoint_margin_b=margin_b,
                mutual_best=rank_a == 1 and rank_b == 1,
                mutual_top3=rank_a <= 3 and rank_b <= 3,
            )
        )

    return tuple(result)


def build_actor_link_scores(
    candidates: Sequence[Mapping[str, Any]],
    *,
    kit_assignments: Sequence[Mapping[str, Any]],
    split_positive_rows: Sequence[Mapping[str, Any]],
    simultaneous_negative_rows: Sequence[Mapping[str, Any]],
    config: ActorLinkScoringConfig | None = None,
) -> tuple[tuple[ActorScoredLink, ...], dict[str, Any]]:
    selected = config or ActorLinkScoringConfig()

    kit_by_id = {
        assignment.tracklet_id: assignment
        for assignment in (
            KitAssignment.from_row(row)
            for row in kit_assignments
        )
    }

    positive_values = sorted(
        _finite("split positive cosine", row["cosine"])
        for row in split_positive_rows
    )

    negative_groups: dict[str, list[float]] = {
        "same_kit": [],
        "different_kit": [],
        "unknown_kit": [],
    }
    for row in simultaneous_negative_rows:
        a = kit_by_id.get(str(row["tracklet_a"]))
        b = kit_by_id.get(str(row["tracklet_b"]))
        family = _kit_family(a, b)
        negative_groups[family].append(
            _finite("negative cosine", row["cosine"])
        )

    calibration = EmpiricalReIDCalibration(
        split_positive=tuple(positive_values),
        same_kit_negative=tuple(
            sorted(negative_groups["same_kit"])
        ),
        different_kit_negative=tuple(
            sorted(negative_groups["different_kit"])
        ),
        unknown_kit_negative=tuple(
            sorted(negative_groups["unknown_kit"])
        ),
    )

    assignment_values = list(kit_by_id.values())
    kit_margin_scale = RobustScale.fit(
        [item.margin for item in assignment_values],
        higher_is_better=True,
    )
    kit_cosine_scale = RobustScale.fit(
        [item.best_cosine for item in assignment_values],
        higher_is_better=True,
    )
    local_scales = _fit_local_scales(candidates)

    raw_links = tuple(
        _score_one(
            candidate,
            kit_by_id=kit_by_id,
            calibration=calibration,
            kit_margin_scale=kit_margin_scale,
            kit_cosine_scale=kit_cosine_scale,
            local_scales=local_scales,
            config=selected,
        )
        for candidate in candidates
    )
    links = _with_competition(raw_links)

    family_counts: dict[str, int] = defaultdict(int)
    for link in links:
        family_counts[link.link_family] += 1

    summary = {
        "schema_version": 1,
        "scoring_version": ACTOR_LINK_SCORING_VERSION,
        "candidate_count": len(links),
        "family_counts": dict(sorted(family_counts.items())),
        "rank_score": _stats(
            [link.rank_score for link in links]
        ),
        "evidence_confidence": _stats(
            [link.evidence_confidence for link in links]
        ),
        "mutual_best_count": sum(
            1 for link in links if link.mutual_best
        ),
        "mutual_top3_count": sum(
            1 for link in links if link.mutual_top3
        ),
        "mutual_best_by_family": {
            family: sum(
                1
                for link in links
                if (
                    link.link_family == family
                    and link.mutual_best
                )
            )
            for family in sorted(family_counts)
        },
        "calibration": {
            "split_positive": _stats(
                list(calibration.split_positive)
            ),
            "same_kit_negative": _stats(
                list(calibration.same_kit_negative)
            ),
            "different_kit_negative": _stats(
                list(calibration.different_kit_negative)
            ),
            "unknown_kit_negative": _stats(
                list(calibration.unknown_kit_negative)
            ),
        },
        "local_scales": {
            key: (
                value.to_dict()
                if value is not None
                else None
            )
            for key, value in local_scales.items()
        },
        "kit_margin_scale": (
            kit_margin_scale.to_dict()
            if kit_margin_scale is not None
            else None
        ),
        "kit_cosine_scale": (
            kit_cosine_scale.to_dict()
            if kit_cosine_scale is not None
            else None
        ),
        "config": selected.to_dict(),
        "identity_decisions": False,
        "score_is_probability": False,
        "fragmentation_preferred_over_contamination": True,
    }

    return links, summary
