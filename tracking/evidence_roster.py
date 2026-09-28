from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import exp, isfinite, log
from statistics import median
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .fixed_roster import (
    FixedRosterConfig,
    IdentityPairEvidence,
    PairEvidenceMode,
    identity_pair_evidence,
)
from .models import PositionKind, TrackSample


class EvidenceRosterInfeasibleError(RuntimeError):
    """The exact roster cannot be built from the available evidence."""


class TrackAssignmentKind(str, Enum):
    ROOT = "root"
    FRAGMENT = "fragment"
    JUNK = "junk"


@dataclass(slots=True, frozen=True)
class EvidenceRosterConfig:
    """
    Tracking V2.7.1: evidence-scored fixed-roster identity assignment.

    Key modelling change from V2.6:
    NOT every raw track must belong to a real participant.

    Every raw trajectory is assigned to exactly one of:
      ROOT      -> one of the K real participant identities;
      FRAGMENT  -> a later/duplicate fragment of one ROOT identity;
      JUNK      -> discarded evidence (short false detection, ghost track, etc.).

    The optimizer chooses all identities, fragments and junk jointly.

    `track_score` and pairwise `same_identity_score` live in [0, 1], but they
    are deterministic confidence scores rather than calibrated probabilities.
    """

    roster_size: int = 23

    # Per-track evidence.
    count_scale: float = 12.0
    span_scale_seconds: float = 1.25
    run_scale_seconds: float = 0.65

    count_weight: float = 0.28
    span_weight: float = 0.20
    run_weight: float = 0.22
    observed_fraction_weight: float = 0.25
    block_coherence_weight: float = 0.05

    # Root/anchor quality is intentionally separate from "is this real
    # evidence?". A short clean fragment can be very real but should not
    # displace a 30-second identity anchor from one of the fixed roster slots.
    anchor_real_weight: float = 0.15
    anchor_span_weight: float = 0.35
    anchor_global_observed_weight: float = 0.30
    anchor_longest_run_weight: float = 0.20

    # Explicit ghost evidence: very small amounts of real support followed by
    # mostly Kalman coasting should become cheap to discard.
    ghost_count_scale: float = 8.0
    ghost_run_scale_seconds: float = 0.45
    ghost_penalty_strength: float = 0.72

    # MILP objective. Smaller link weight makes it reasonable to preserve a
    # real-looking fragment when the identity match is good, while junk remains
    # strongly preferred for low-evidence tracks.
    link_cost_weight: float = 0.60
    score_epsilon: float = 1e-6

    # Final offline presentation: unsupported prediction before the first real
    # observation or after the last real observation is not evidence. Keep only
    # a tiny grace interval, never the whole GlobalTracker coast.
    prediction_edge_grace_seconds: float = 0.20

    ranking_size: int = 5

    def __post_init__(self) -> None:
        if self.roster_size < 1:
            raise ValueError("roster_size must be >= 1")

        positive = (
            self.count_scale,
            self.span_scale_seconds,
            self.run_scale_seconds,
            self.ghost_count_scale,
            self.ghost_run_scale_seconds,
            self.link_cost_weight,
            self.score_epsilon,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError("positive EvidenceRosterConfig values must be > 0")

        weights = (
            self.count_weight,
            self.span_weight,
            self.run_weight,
            self.observed_fraction_weight,
            self.block_coherence_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("track evidence weights must be >= 0")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("track evidence weights must sum to 1")

        anchor_weights = (
            self.anchor_real_weight,
            self.anchor_span_weight,
            self.anchor_global_observed_weight,
            self.anchor_longest_run_weight,
        )
        if any(value < 0.0 for value in anchor_weights):
            raise ValueError("anchor weights must be >= 0")
        if abs(sum(anchor_weights) - 1.0) > 1e-9:
            raise ValueError("anchor weights must sum to 1")

        if not 0.0 <= self.ghost_penalty_strength <= 1.0:
            raise ValueError("ghost_penalty_strength must be in [0, 1]")
        if self.prediction_edge_grace_seconds < 0.0:
            raise ValueError(
                "prediction_edge_grace_seconds must be >= 0"
            )
        if self.ranking_size < 1:
            raise ValueError("ranking_size must be >= 1")


@dataclass(slots=True, frozen=True)
class TrackEvidence:
    track_id: int

    sample_count: int
    observed_count: int
    predicted_count: int
    interpolated_count: int

    lifetime_seconds: float
    observed_span_seconds: float
    observed_fraction: float

    longest_observed_run_seconds: float
    observed_block_count: int
    median_sample_period_seconds: float

    leading_prediction_seconds: float
    trailing_prediction_seconds: float

    mean_observed_confidence: float | None

    count_score: float
    span_score: float
    run_score: float
    observed_fraction_score: float
    block_coherence_score: float

    base_real_score: float
    ghost_score: float
    real_score: float

    clip_span_seconds: float
    observed_span_coverage: float
    global_observed_coverage: float
    longest_run_coverage: float
    anchor_score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EvidenceParentLink:
    parent_id: int
    child_id: int
    score: float
    mode: PairEvidenceMode
    evidence: IdentityPairEvidence

    def to_dict(self) -> dict[str, object]:
        return {
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "score": self.score,
            "mode": self.mode.value,
            "evidence": self.evidence.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class TrackAssignment:
    track_id: int
    kind: TrackAssignmentKind
    representative_id: int | None
    real_score: float
    anchor_score: float
    parent_id: int | None
    parent_score: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "kind": self.kind.value,
            "representative_id": self.representative_id,
            "real_score": self.real_score,
            "anchor_score": self.anchor_score,
            "parent_id": self.parent_id,
            "parent_score": self.parent_score,
        }


@dataclass(slots=True, frozen=True)
class EdgeTrimMetrics:
    track_id: int
    input_samples: int
    output_samples: int
    removed_leading_samples: int
    removed_trailing_samples: int
    first_observed_timestamp: float
    last_observed_timestamp: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class EvidenceRosterReport:
    roster_size: int
    input_track_count: int
    selected_track_count: int
    junk_track_count: int

    track_evidence: Mapping[int, TrackEvidence]
    assignments: Mapping[int, TrackAssignment]

    selected_parent_links: tuple[EvidenceParentLink, ...]
    components: tuple[tuple[int, ...], ...]
    representative_by_track: Mapping[int, int | None]

    junk_track_ids: tuple[int, ...]
    root_track_ids: tuple[int, ...]
    fragment_track_ids: tuple[int, ...]

    pair_rankings: Mapping[int, tuple[IdentityPairEvidence, ...]]
    edge_trim: Mapping[int, EdgeTrimMetrics]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "roster_size": self.roster_size,
                "input_tracks": self.input_track_count,
                "selected_tracks": self.selected_track_count,
                "junk_tracks": self.junk_track_count,
                "root_tracks": len(self.root_track_ids),
                "fragment_tracks": len(self.fragment_track_ids),
                "selected_parent_links": len(self.selected_parent_links),
            },
            "root_track_ids": list(self.root_track_ids),
            "fragment_track_ids": list(self.fragment_track_ids),
            "junk_track_ids": list(self.junk_track_ids),
            "track_evidence": {
                str(track_id): evidence.to_dict()
                for track_id, evidence in sorted(
                    self.track_evidence.items()
                )
            },
            "assignments": {
                str(track_id): assignment.to_dict()
                for track_id, assignment in sorted(
                    self.assignments.items()
                )
            },
            "selected_parent_links": [
                link.to_dict()
                for link in self.selected_parent_links
            ],
            "components": [
                list(component)
                for component in self.components
            ],
            "representative_by_track": {
                str(track_id): representative
                for track_id, representative in sorted(
                    self.representative_by_track.items()
                )
            },
            "pair_rankings": {
                str(track_id): [
                    evidence.to_dict()
                    for evidence in ranking
                ]
                for track_id, ranking in sorted(
                    self.pair_rankings.items()
                )
            },
            "edge_trim": {
                str(track_id): metrics.to_dict()
                for track_id, metrics in sorted(
                    self.edge_trim.items()
                )
            },
        }


def analyze_track_evidence(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    config: EvidenceRosterConfig | None = None,
) -> dict[int, TrackEvidence]:
    cfg = config or EvidenceRosterConfig()

    normalized: dict[int, tuple[TrackSample, ...]] = {}
    first_times: list[float] = []
    last_times: list[float] = []

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(track_id, raw_samples)
        observed = [
            sample
            for sample in samples
            if sample.position_kind is PositionKind.OBSERVED
        ]
        if not samples or not observed:
            continue

        normalized[track_id] = samples
        first_times.append(observed[0].timestamp)
        last_times.append(observed[-1].timestamp)

    if not normalized:
        return {}

    clip_span_seconds = max(
        1.0 / 30.0,
        max(last_times) - min(first_times),
    )

    result: dict[int, TrackEvidence] = {}
    for track_id, samples in sorted(normalized.items()):
        result[track_id] = _track_evidence(
            samples,
            clip_span_seconds=clip_span_seconds,
            config=cfg,
        )

    return result


def resolve_evidence_roster(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    config: EvidenceRosterConfig | None = None,
    pair_config: FixedRosterConfig | None = None,
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    EvidenceRosterReport,
]:
    """
    Jointly choose K real identities, useful fragments, and junk tracks.

    MILP variables:
      assignment[i, r] = selected track i belongs to representative/root r
      junk[i]          = track i is discarded
      parent[e]        = positive same-identity edge e links child to earlier
                         parent inside the same representative component

    Invariants:
      * assignment OR junk = exactly one decision for every raw track;
      * exactly K representatives are selected;
      * every selected non-root has exactly one positive-evidence parent;
      * selected parent and child share one representative;
      * hard-conflicting tracks may never share a representative;
      * pair edges are oriented by first real observation, so parent chains are
        acyclic.

    Objective:
      -log(real_score) for selecting a track
      -log(1-real_score) for discarding it
      + link_cost_weight * -log(same_identity_score) for every parent link

    This makes high-evidence tracks expensive to discard and low-evidence ghost
    tracks cheap to discard, while still requiring exactly K real roots.
    """
    cfg = config or EvidenceRosterConfig()
    pair_cfg = pair_config or FixedRosterConfig(
        roster_size=cfg.roster_size,
    )

    normalized = _validated_trajectories(trajectories)
    if len(normalized) < cfg.roster_size:
        raise EvidenceRosterInfeasibleError(
            f"Cannot build {cfg.roster_size} identities from "
            f"{len(normalized)} input tracks"
        )

    track_evidence = analyze_track_evidence(
        normalized,
        config=cfg,
    )

    ordered_ids = tuple(
        sorted(
            normalized,
            key=lambda track_id: (
                _first_observed(normalized[track_id]).timestamp,
                track_id,
            ),
        )
    )
    index_by_id = {
        track_id: index
        for index, track_id in enumerate(ordered_ids)
    }

    parent_evidence: list[IdentityPairEvidence] = []
    hard_conflicts: set[tuple[int, int]] = set()
    rankings: dict[int, list[IdentityPairEvidence]] = {
        track_id: []
        for track_id in ordered_ids
    }

    for left_index, left_id in enumerate(ordered_ids):
        for right_id in ordered_ids[left_index + 1:]:
            evidence = identity_pair_evidence(
                normalized[left_id],
                normalized[right_id],
                config=pair_cfg,
            )

            rankings[left_id].append(evidence)
            rankings[right_id].append(evidence)

            pair = (
                min(left_id, right_id),
                max(left_id, right_id),
            )
            if evidence.hard_conflict:
                hard_conflicts.add(pair)
            elif evidence.score >= pair_cfg.minimum_parent_score:
                parent_evidence.append(evidence)

    pair_rankings = {
        track_id: tuple(
            sorted(
                rows,
                key=lambda item: (
                    -item.score,
                    item.hard_conflict,
                    item.earlier_id,
                    item.later_id,
                ),
            )[:cfg.ranking_size]
        )
        for track_id, rows in rankings.items()
    }

    (
        selected_edge_indices,
        representative_index,
        junk_indices,
    ) = _solve_evidence_milp(
        ordered_ids,
        track_evidence,
        parent_evidence,
        hard_conflicts,
        config=cfg,
    )

    representative_by_track: dict[int, int | None] = {}
    for item_index, track_id in enumerate(ordered_ids):
        if item_index in junk_indices:
            representative_by_track[track_id] = None
        else:
            representative_by_track[track_id] = ordered_ids[
                representative_index[item_index]
            ]

    selected_links = tuple(
        EvidenceParentLink(
            parent_id=parent_evidence[edge_index].earlier_id,
            child_id=parent_evidence[edge_index].later_id,
            score=parent_evidence[edge_index].score,
            mode=parent_evidence[edge_index].mode,
            evidence=parent_evidence[edge_index],
        )
        for edge_index in sorted(
            selected_edge_indices,
            key=lambda edge_index: (
                index_by_id[parent_evidence[edge_index].later_id],
                index_by_id[parent_evidence[edge_index].earlier_id],
            ),
        )
    )
    selected_parent_by_child = {
        link.child_id: link
        for link in selected_links
    }

    grouped: dict[int, list[int]] = {}
    for track_id in ordered_ids:
        representative = representative_by_track[track_id]
        if representative is None:
            continue
        grouped.setdefault(representative, []).append(track_id)

    components = tuple(
        tuple(
            sorted(
                members,
                key=lambda track_id: (
                    index_by_id[track_id],
                    track_id,
                ),
            )
        )
        for representative, members in sorted(
            grouped.items(),
            key=lambda item: index_by_id[item[0]],
        )
    )

    if len(components) != cfg.roster_size:
        raise RuntimeError(
            "Evidence-roster MILP invariant failed: "
            f"expected {cfg.roster_size} components, got {len(components)}"
        )

    merged: dict[int, tuple[TrackSample, ...]] = {}
    for component in components:
        representative = representative_by_track[component[0]]
        if representative is None:
            raise RuntimeError("Selected component has no representative")
        merged[representative] = _merge_component(
            representative,
            component,
            normalized,
        )

    trimmed, trim_report = trim_unsupported_prediction_edges(
        merged,
        grace_seconds=cfg.prediction_edge_grace_seconds,
    )

    assignments: dict[int, TrackAssignment] = {}
    roots: list[int] = []
    fragments: list[int] = []
    junk: list[int] = []

    for track_id in ordered_ids:
        representative = representative_by_track[track_id]
        evidence = track_evidence[track_id]

        if representative is None:
            kind = TrackAssignmentKind.JUNK
            junk.append(track_id)
            parent_id = None
            parent_score = None
        elif representative == track_id:
            kind = TrackAssignmentKind.ROOT
            roots.append(track_id)
            parent_id = None
            parent_score = None
        else:
            kind = TrackAssignmentKind.FRAGMENT
            fragments.append(track_id)
            parent_link = selected_parent_by_child.get(track_id)
            if parent_link is None:
                raise RuntimeError(
                    f"Selected fragment {track_id} has no parent link"
                )
            parent_id = parent_link.parent_id
            parent_score = parent_link.score

        assignments[track_id] = TrackAssignment(
            track_id=track_id,
            kind=kind,
            representative_id=representative,
            real_score=evidence.real_score,
            anchor_score=evidence.anchor_score,
            parent_id=parent_id,
            parent_score=parent_score,
        )

    report = EvidenceRosterReport(
        roster_size=cfg.roster_size,
        input_track_count=len(ordered_ids),
        selected_track_count=len(ordered_ids) - len(junk),
        junk_track_count=len(junk),
        track_evidence=track_evidence,
        assignments=assignments,
        selected_parent_links=selected_links,
        components=components,
        representative_by_track=representative_by_track,
        junk_track_ids=tuple(junk),
        root_track_ids=tuple(roots),
        fragment_track_ids=tuple(fragments),
        pair_rankings=pair_rankings,
        edge_trim=trim_report,
    )

    return trimmed, report


def trim_unsupported_prediction_edges(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    grace_seconds: float = 0.20,
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    dict[int, EdgeTrimMetrics],
]:
    """
    Remove unsupported leading/trailing prediction tails.

    Internal predicted gaps are preserved because they are bracketed by real
    observations and can later be corrected by RTS smoothing/interpolation.
    """
    if grace_seconds < 0.0:
        raise ValueError("grace_seconds must be >= 0")

    trimmed: dict[int, tuple[TrackSample, ...]] = {}
    report: dict[int, EdgeTrimMetrics] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(track_id, raw_samples)
        observed = [
            sample
            for sample in samples
            if sample.position_kind is PositionKind.OBSERVED
        ]
        if not observed:
            continue

        first_observed = observed[0].timestamp
        last_observed = observed[-1].timestamp
        keep_from = first_observed - grace_seconds
        keep_until = last_observed + grace_seconds

        output = tuple(
            sample
            for sample in samples
            if keep_from <= sample.timestamp <= keep_until
        )

        leading = sum(
            sample.timestamp < keep_from
            for sample in samples
        )
        trailing = sum(
            sample.timestamp > keep_until
            for sample in samples
        )

        trimmed[track_id] = output
        report[track_id] = EdgeTrimMetrics(
            track_id=track_id,
            input_samples=len(samples),
            output_samples=len(output),
            removed_leading_samples=leading,
            removed_trailing_samples=trailing,
            first_observed_timestamp=float(first_observed),
            last_observed_timestamp=float(last_observed),
        )

    return trimmed, report


def _track_evidence(
    samples: tuple[TrackSample, ...],
    *,
    clip_span_seconds: float,
    config: EvidenceRosterConfig,
) -> TrackEvidence:
    observed = tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.OBSERVED
    )
    predicted = tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.PREDICTED
    )
    interpolated = tuple(
        sample
        for sample in samples
        if sample.position_kind is PositionKind.INTERPOLATED
    )

    sample_count = len(samples)
    observed_count = len(observed)

    lifetime = (
        max(
            0.0,
            float(samples[-1].timestamp - samples[0].timestamp),
        )
        if samples
        else 0.0
    )
    observed_span = (
        max(
            0.0,
            float(observed[-1].timestamp - observed[0].timestamp),
        )
        if observed
        else 0.0
    )
    observed_fraction = (
        observed_count / sample_count
        if sample_count
        else 0.0
    )

    nominal_period = _median_positive_delta(samples)
    longest_run, block_count = _observed_run_metrics(
        observed,
        nominal_period=nominal_period,
    )

    leading_prediction_seconds = (
        max(
            0.0,
            float(observed[0].timestamp - samples[0].timestamp),
        )
        if observed
        else lifetime
    )
    trailing_prediction_seconds = (
        max(
            0.0,
            float(samples[-1].timestamp - observed[-1].timestamp),
        )
        if observed
        else lifetime
    )

    confidences = [
        float(sample.confidence)
        for sample in observed
        if (
            sample.confidence is not None
            and isfinite(float(sample.confidence))
        )
    ]
    mean_confidence = (
        sum(confidences) / len(confidences)
        if confidences
        else None
    )

    count_score = 1.0 - exp(
        -observed_count / config.count_scale
    )
    span_score = 1.0 - exp(
        -observed_span / config.span_scale_seconds
    )
    run_score = 1.0 - exp(
        -longest_run / config.run_scale_seconds
    )
    observed_fraction_score = (
        observed_fraction ** 0.75
        if observed_fraction > 0.0
        else 0.0
    )
    block_coherence_score = 1.0 / (
        1.0 + 0.10 * max(0, block_count - 1)
    )

    epsilon = 1e-9
    base_real_score = exp(
        config.count_weight
        * log(max(count_score, epsilon))
        + config.span_weight
        * log(max(span_score, epsilon))
        + config.run_weight
        * log(max(run_score, epsilon))
        + config.observed_fraction_weight
        * log(max(observed_fraction_score, epsilon))
        + config.block_coherence_weight
        * log(max(block_coherence_score, epsilon))
    )

    sparse_term = exp(
        -observed_count / config.ghost_count_scale
    )
    short_run_term = exp(
        -longest_run / config.ghost_run_scale_seconds
    )
    prediction_dominance = 1.0 - observed_fraction

    ghost_score = (
        sparse_term
        * short_run_term
        * prediction_dominance
    ) ** (1.0 / 3.0)

    real_score = base_real_score * (
        1.0 - config.ghost_penalty_strength * ghost_score
    )

    # Keep probabilities away from exact 0/1 so the MILP log costs remain
    # finite and diagnostically useful.
    real_score = float(
        min(
            1.0 - config.score_epsilon,
            max(config.score_epsilon, real_score),
        )
    )

    effective_period = max(
        nominal_period,
        1.0 / 30.0,
    )
    expected_observed_samples = max(
        1.0,
        clip_span_seconds / effective_period + 1.0,
    )

    observed_span_coverage = min(
        1.0,
        (observed_span + effective_period) / clip_span_seconds,
    )
    global_observed_coverage = min(
        1.0,
        observed_count / expected_observed_samples,
    )
    longest_run_coverage = min(
        1.0,
        longest_run / clip_span_seconds,
    )

    anchor_epsilon = 1e-9
    anchor_score = exp(
        config.anchor_real_weight
        * log(max(real_score, anchor_epsilon))
        + config.anchor_span_weight
        * log(max(observed_span_coverage, anchor_epsilon))
        + config.anchor_global_observed_weight
        * log(max(global_observed_coverage, anchor_epsilon))
        + config.anchor_longest_run_weight
        * log(max(longest_run_coverage, anchor_epsilon))
    )
    anchor_score = float(
        min(
            1.0 - config.score_epsilon,
            max(config.score_epsilon, anchor_score),
        )
    )

    return TrackEvidence(
        track_id=samples[0].track_id,
        sample_count=sample_count,
        observed_count=observed_count,
        predicted_count=len(predicted),
        interpolated_count=len(interpolated),
        lifetime_seconds=float(lifetime),
        observed_span_seconds=float(observed_span),
        observed_fraction=float(observed_fraction),
        longest_observed_run_seconds=float(longest_run),
        observed_block_count=int(block_count),
        median_sample_period_seconds=float(nominal_period),
        leading_prediction_seconds=float(leading_prediction_seconds),
        trailing_prediction_seconds=float(trailing_prediction_seconds),
        mean_observed_confidence=mean_confidence,
        count_score=float(count_score),
        span_score=float(span_score),
        run_score=float(run_score),
        observed_fraction_score=float(observed_fraction_score),
        block_coherence_score=float(block_coherence_score),
        base_real_score=float(base_real_score),
        ghost_score=float(ghost_score),
        real_score=real_score,
        clip_span_seconds=float(clip_span_seconds),
        observed_span_coverage=float(observed_span_coverage),
        global_observed_coverage=float(global_observed_coverage),
        longest_run_coverage=float(longest_run_coverage),
        anchor_score=anchor_score,
    )


def _solve_evidence_milp(
    ordered_ids: tuple[int, ...],
    track_evidence: Mapping[int, TrackEvidence],
    parent_evidence: Sequence[IdentityPairEvidence],
    hard_conflicts: set[tuple[int, int]],
    *,
    config: EvidenceRosterConfig,
) -> tuple[set[int], list[int], set[int]]:
    n = len(ordered_ids)
    edge_count = len(parent_evidence)
    index_by_id = {
        track_id: index
        for index, track_id in enumerate(ordered_ids)
    }

    assignment_count = n * n
    junk_offset = assignment_count
    parent_offset = junk_offset + n
    variable_count = parent_offset + edge_count

    def a_var(item_index: int, representative_index: int) -> int:
        return item_index * n + representative_index

    def j_var(item_index: int) -> int:
        return junk_offset + item_index

    def p_var(edge_index: int) -> int:
        return parent_offset + edge_index

    objective = np.zeros(variable_count, dtype=np.float64)
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.ones(variable_count, dtype=np.float64)

    epsilon = config.score_epsilon

    # Selection, root-anchor and junk likelihood costs.
    #
    # A short but clean track may have high real_score while being a poor
    # canonical anchor. Non-root assignment therefore uses real_score, whereas
    # self/root assignment uses anchor_score.
    for item_index, track_id in enumerate(ordered_ids):
        evidence = track_evidence[track_id]
        real_score = evidence.real_score
        anchor_score = evidence.anchor_score

        fragment_use_cost = -log(max(real_score, epsilon))
        root_cost = -log(max(anchor_score, epsilon))
        junk_cost = -log(max(1.0 - real_score, epsilon))

        for representative_index in range(item_index):
            objective[
                a_var(item_index, representative_index)
            ] = fragment_use_cost

        objective[
            a_var(item_index, item_index)
        ] = root_cost

        objective[j_var(item_index)] = junk_cost

        # A future track cannot represent an earlier track.
        for representative_index in range(item_index + 1, n):
            upper_bounds[
                a_var(item_index, representative_index)
            ] = 0.0

    for edge_index, evidence in enumerate(parent_evidence):
        objective[p_var(edge_index)] = (
            config.link_cost_weight
            * -log(max(evidence.score, epsilon))
        )

    rows: list[dict[int, float]] = []
    row_lower: list[float] = []
    row_upper: list[float] = []

    def add_constraint(
        coefficients: Mapping[int, float],
        lower: float,
        upper: float,
    ) -> None:
        rows.append(dict(coefficients))
        row_lower.append(lower)
        row_upper.append(upper)

    # Exactly one decision per track: selected into one identity OR junk.
    for item_index in range(n):
        coefficients = {
            a_var(item_index, representative_index): 1.0
            for representative_index in range(item_index + 1)
        }
        coefficients[j_var(item_index)] = 1.0
        add_constraint(coefficients, 1.0, 1.0)

    # If track i uses representative r, r itself must be a selected root.
    for item_index in range(n):
        for representative_index in range(item_index):
            add_constraint(
                {
                    a_var(item_index, representative_index): 1.0,
                    a_var(
                        representative_index,
                        representative_index,
                    ): -1.0,
                },
                -np.inf,
                0.0,
            )

    # Exactly K real identities.
    add_constraint(
        {
            a_var(item_index, item_index): 1.0
            for item_index in range(n)
        },
        float(config.roster_size),
        float(config.roster_size),
    )

    incoming_by_child: dict[int, list[int]] = {
        item_index: []
        for item_index in range(n)
    }
    for edge_index, evidence in enumerate(parent_evidence):
        child_index = index_by_id[evidence.later_id]
        incoming_by_child[child_index].append(edge_index)

    # Root OR exactly one parent OR junk.
    for item_index in range(n):
        coefficients = {
            a_var(item_index, item_index): 1.0,
            j_var(item_index): 1.0,
        }
        for edge_index in incoming_by_child[item_index]:
            coefficients[p_var(edge_index)] = 1.0
        add_constraint(coefficients, 1.0, 1.0)

    # Selected parent edge => parent and child share the same representative.
    for edge_index, evidence in enumerate(parent_evidence):
        parent_index = index_by_id[evidence.earlier_id]
        child_index = index_by_id[evidence.later_id]
        parent_variable = p_var(edge_index)

        for representative_index in range(n):
            add_constraint(
                {
                    parent_variable: 1.0,
                    a_var(parent_index, representative_index): 1.0,
                    a_var(child_index, representative_index): -1.0,
                },
                -np.inf,
                1.0,
            )
            add_constraint(
                {
                    parent_variable: 1.0,
                    a_var(parent_index, representative_index): -1.0,
                    a_var(child_index, representative_index): 1.0,
                },
                -np.inf,
                1.0,
            )

    # Strong simultaneous different-person evidence is transitive-safe:
    # conflicting tracks cannot share any representative.
    for left_id, right_id in hard_conflicts:
        left_index = index_by_id[left_id]
        right_index = index_by_id[right_id]
        common_limit = min(left_index, right_index)

        for representative_index in range(common_limit + 1):
            add_constraint(
                {
                    a_var(left_index, representative_index): 1.0,
                    a_var(right_index, representative_index): 1.0,
                },
                -np.inf,
                1.0,
            )

    data: list[float] = []
    row_indices: list[int] = []
    column_indices: list[int] = []

    for row_index, coefficients in enumerate(rows):
        for column_index, coefficient in coefficients.items():
            row_indices.append(row_index)
            column_indices.append(column_index)
            data.append(coefficient)

    matrix = coo_matrix(
        (
            np.asarray(data, dtype=np.float64),
            (
                np.asarray(row_indices, dtype=np.int32),
                np.asarray(column_indices, dtype=np.int32),
            ),
        ),
        shape=(len(rows), variable_count),
    ).tocsr()

    result = milp(
        c=objective,
        integrality=np.ones(variable_count, dtype=np.int32),
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(
            matrix,
            np.asarray(row_lower, dtype=np.float64),
            np.asarray(row_upper, dtype=np.float64),
        ),
        options={"presolve": True},
    )

    if not result.success or result.x is None:
        raise EvidenceRosterInfeasibleError(
            "Evidence-scored exact-roster MILP is infeasible "
            f"(status={result.status}, message={result.message!r})."
        )

    selected_edges = {
        edge_index
        for edge_index in range(edge_count)
        if result.x[p_var(edge_index)] >= 0.5
    }
    junk_indices = {
        item_index
        for item_index in range(n)
        if result.x[j_var(item_index)] >= 0.5
    }

    representative_index: list[int] = [-1] * n
    for item_index in range(n):
        if item_index in junk_indices:
            continue

        active = [
            rep_index
            for rep_index in range(item_index + 1)
            if result.x[
                a_var(item_index, rep_index)
            ] >= 0.5
        ]
        if len(active) != 1:
            raise RuntimeError(
                "MILP assignment invariant failed for "
                f"track index {item_index}: {active}"
            )
        representative_index[item_index] = active[0]

    roots = sum(
        1
        for item_index in range(n)
        if (
            item_index not in junk_indices
            and representative_index[item_index] == item_index
        )
    )
    if roots != config.roster_size:
        raise RuntimeError(
            "MILP root-count invariant failed: "
            f"expected {config.roster_size}, got {roots}"
        )

    selected_count = n - len(junk_indices)
    expected_edges = selected_count - config.roster_size
    if len(selected_edges) != expected_edges:
        raise RuntimeError(
            "MILP selected-parent invariant failed: "
            f"expected {expected_edges}, got {len(selected_edges)}"
        )

    return selected_edges, representative_index, junk_indices


def _merge_component(
    representative: int,
    component: Sequence[int],
    trajectories: Mapping[int, tuple[TrackSample, ...]],
) -> tuple[TrackSample, ...]:
    by_frame: dict[int, tuple[int, TrackSample]] = {}

    ordered_sources = [representative] + [
        track_id
        for track_id in component
        if track_id != representative
    ]

    for source_id in ordered_sources:
        for sample in trajectories[source_id]:
            rewritten = replace(
                sample,
                track_id=representative,
            )
            existing = by_frame.get(rewritten.frame_index)

            if existing is None:
                by_frame[rewritten.frame_index] = (
                    source_id,
                    rewritten,
                )
                continue

            existing_source, existing_sample = existing

            if _prefer_sample(
                new=rewritten,
                new_source=source_id,
                existing=existing_sample,
                existing_source=existing_source,
                representative=representative,
            ):
                by_frame[rewritten.frame_index] = (
                    source_id,
                    rewritten,
                )

    return tuple(
        sample
        for _, sample in sorted(
            by_frame.values(),
            key=lambda pair: (
                pair[1].frame_index,
                pair[1].timestamp,
            ),
        )
    )


def _prefer_sample(
    *,
    new: TrackSample,
    new_source: int,
    existing: TrackSample,
    existing_source: int,
    representative: int,
) -> bool:
    new_priority = _sample_priority(new)
    existing_priority = _sample_priority(existing)

    if new_priority != existing_priority:
        return new_priority > existing_priority

    if new.position_kind is PositionKind.OBSERVED:
        if (
            new.confidence is not None
            and existing.confidence is not None
        ):
            if new.confidence != existing.confidence:
                return new.confidence > existing.confidence
        elif new.confidence is not None:
            return True
        elif existing.confidence is not None:
            return False

    if existing_source == representative:
        return False
    if new_source == representative:
        return True

    return new_source < existing_source


def _sample_priority(sample: TrackSample) -> int:
    return {
        PositionKind.OBSERVED: 3,
        PositionKind.INTERPOLATED: 2,
        PositionKind.PREDICTED: 1,
    }[sample.position_kind]


def _observed_run_metrics(
    observed: Sequence[TrackSample],
    *,
    nominal_period: float,
) -> tuple[float, int]:
    if not observed:
        return 0.0, 0

    threshold = max(
        0.10,
        nominal_period * 1.60,
    )
    effective_period = max(
        nominal_period,
        1.0 / 30.0,
    )

    longest = effective_period
    blocks = 1
    run_start = observed[0].timestamp
    previous = observed[0].timestamp

    for sample in observed[1:]:
        gap = sample.timestamp - previous

        if gap <= threshold:
            longest = max(
                longest,
                sample.timestamp
                - run_start
                + effective_period,
            )
        else:
            blocks += 1
            run_start = sample.timestamp
            longest = max(
                longest,
                effective_period,
            )

        previous = sample.timestamp

    return float(longest), int(blocks)


def _median_positive_delta(
    samples: Sequence[TrackSample],
) -> float:
    deltas = [
        float(right.timestamp - left.timestamp)
        for left, right in zip(samples, samples[1:])
        if right.timestamp > left.timestamp
    ]
    return float(median(deltas)) if deltas else 0.0


def _first_observed(
    samples: Sequence[TrackSample],
) -> TrackSample:
    for sample in samples:
        if sample.position_kind is PositionKind.OBSERVED:
            return sample
    raise ValueError("Trajectory contains no observed samples")


def _validated_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
) -> dict[int, tuple[TrackSample, ...]]:
    result: dict[int, tuple[TrackSample, ...]] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        samples = _validated_samples(track_id, raw_samples)
        if any(
            sample.position_kind is PositionKind.OBSERVED
            for sample in samples
        ):
            result[track_id] = samples

    return result


def _validated_samples(
    track_id: int,
    samples: Sequence[TrackSample],
) -> tuple[TrackSample, ...]:
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.frame_index,
                sample.timestamp,
            ),
        )
    )

    previous_frame: int | None = None
    previous_timestamp: float | None = None

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match "
                f"sample track_id {sample.track_id}"
            )
        if sample.frame_index < 0:
            raise ValueError("Track sample frame_index must be >= 0")
        if not isfinite(sample.timestamp):
            raise ValueError("Track sample timestamp must be finite")
        if not all(
            isfinite(value)
            for value in (
                sample.x,
                sample.z,
                sample.vx,
                sample.vz,
            )
        ):
            raise ValueError(
                f"Track {track_id} contains non-finite state"
            )
        if (
            previous_frame is not None
            and sample.frame_index == previous_frame
        ):
            raise ValueError(
                f"Track {track_id} contains duplicate frame_index "
                f"{sample.frame_index}"
            )
        if (
            previous_timestamp is not None
            and sample.timestamp <= previous_timestamp
        ):
            raise ValueError(
                f"Track {track_id} timestamps must be strictly increasing"
            )

        previous_frame = sample.frame_index
        previous_timestamp = sample.timestamp

    return ordered
