"""VeoMatchLab post-link global identity refinement V5.1.

V5.1 sits *after* the V5 full-clip linker.  V5 already fixes the main
under-linking problem by considering short/medium continuations over the whole
clip.  The remaining failure mode is different: temporary duplicate branches
can survive as separate final identities, so one footballer may become two
points for a short interval.

This module resolves only conservative, evidence-backed duplicate branches.
It never merges two footballers merely because they are close.  A branch may be
suppressed only when:

- the two final identities overlap in time;
- their reconstructed motion is near-identical during that overlap;
- the shorter identity is substantially contained in the longer one;
- strong raw detections do not provide convincing independent evidence for two
  people; and
- one identity is clearly the weaker/shorter branch (or the two identities
  share proposal lineage).

The V5 analytical trajectories are otherwise preserved byte-for-byte at the
state level.  V5.1 does not smooth presentation output and does not force the
expected football population of 23.
"""

from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass, field
import math
import statistics
from typing import Any, Mapping, Sequence

import numpy as np

from tracking import global_identity as v3
from tracking import global_identity_V5 as v5


@dataclass(frozen=True)
class GlobalIdentityV5_1Config:
    base: v5.GlobalIdentityV5Config = field(default_factory=v5.make_config)

    # Pair must overlap enough to say anything meaningful.
    min_overlap_frames: int = 6
    min_overlap_s: float = 0.18
    evidence_stride: int = 2

    # Geometric/motion similarity for an actual duplicate branch.
    duplicate_median_distance_m: float = 0.90
    duplicate_p90_distance_m: float = 1.75
    duplicate_max_relative_speed_mps: float = 3.50

    # Independent STRONG detections are explicit evidence for two people.
    independent_evidence_radius_m: float = 0.95
    max_independent_evidence_ratio: float = 0.12
    max_both_strong_observed_ratio: float = 0.35

    # Branch topology.  The shorter identity must mostly live inside the
    # overlap and normally be short-lived.  Shared lineage permits a somewhat
    # longer branch because it is direct evidence that proposal splitting
    # produced two descendants from the same local source.
    min_shorter_overlap_fraction: float = 0.70
    max_branch_span_s: float = 6.0
    shared_lineage_max_branch_span_s: float = 10.0

    # Without shared lineage, the weaker branch must be clearly less supported.
    max_weaker_strength_ratio: float = 0.78

    def __post_init__(self) -> None:
        integer_positive = (
            "min_overlap_frames",
            "evidence_stride",
        )
        for name in integer_positive:
            if int(getattr(self, name)) < 1:
                raise ValueError(f"{name} must be >= 1")

        positive = (
            "min_overlap_s",
            "duplicate_median_distance_m",
            "duplicate_p90_distance_m",
            "duplicate_max_relative_speed_mps",
            "independent_evidence_radius_m",
            "max_branch_span_s",
            "shared_lineage_max_branch_span_s",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and > 0")

        bounded = (
            "max_independent_evidence_ratio",
            "max_both_strong_observed_ratio",
            "min_shorter_overlap_fraction",
            "max_weaker_strength_ratio",
        )
        for name in bounded:
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")

        if self.duplicate_p90_distance_m < self.duplicate_median_distance_m:
            raise ValueError(
                "duplicate_p90_distance_m must be >= duplicate_median_distance_m"
            )
        if self.shared_lineage_max_branch_span_s < self.max_branch_span_s:
            raise ValueError(
                "shared_lineage_max_branch_span_s must be >= max_branch_span_s"
            )


@dataclass(frozen=True)
class BranchCompetitionV5_1:
    identity_a: int
    identity_b: int

    overlap_frames: int
    overlap_duration_s: float
    shorter_identity_span_s: float
    shorter_overlap_fraction: float

    median_distance_m: float
    p90_distance_m: float
    median_relative_speed_mps: float

    informative_strong_frames: int
    independent_strong_frames: int
    independent_strong_ratio: float | None
    both_strong_observed_frames: int
    both_strong_observed_ratio: float

    shared_source_track_ids: tuple[int, ...]

    identity_a_strength: float
    identity_b_strength: float
    weaker_strength_ratio: float

    preferred_keep_identity_id: int
    preferred_suppress_identity_id: int

    duplicate_branch: bool
    reason: str
    score: float


@dataclass(frozen=True)
class GlobalIdentityV5_1Result:
    # Original V5 diagnostics/proposal result.
    duplicate_evidence: tuple[v3.DuplicateEvidence, ...]
    duplicate_components: tuple[tuple[int, ...], ...]
    blocked_duplicate_merges: tuple[tuple[int, int], ...]
    fragments: tuple[Any, ...]
    raw_candidate_count: int
    link_candidates: tuple[v5.IdentityLinkCandidateV5, ...]
    accepted_links: tuple[tuple[int, int], ...]

    # Post-link V5.1 diagnostics.
    pre_refinement_identities: tuple[v5.GlobalIdentityV5, ...]
    branch_competitions: tuple[BranchCompetitionV5_1, ...]
    suppressed_identity_ids: tuple[int, ...]

    # Final analytical result.
    identities: tuple[v5.GlobalIdentityV5, ...]
    frame_tracks: Mapping[int, tuple[v5.AnalyticalFrameTrack, ...]]


def make_config(
    *,
    weak_threshold: float = 0.20,
    strong_threshold: float = 0.40,
) -> GlobalIdentityV5_1Config:
    return GlobalIdentityV5_1Config(
        base=v5.make_config(
            weak_threshold=weak_threshold,
            strong_threshold=strong_threshold,
        )
    )


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty percentile input")
    position = q * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return (1.0 - alpha) * ordered[lower] + alpha * ordered[upper]


def _identity_span_s(identity: v5.GlobalIdentityV5) -> float:
    return max(
        0.0,
        float(identity.observed_end_time_s) - float(identity.observed_start_time_s),
    )


def _identity_strength(identity: v5.GlobalIdentityV5) -> float:
    """Deterministic support score used only to choose a duplicate winner.

    Long-lived STRONG evidence dominates.  Reconstructed states contribute a
    little so a continuous well-supported chain still beats a tiny duplicate
    branch, but inferred frames can never outweigh strong detector support.
    """

    weights = {
        "strong_observed": 1.00,
        "strong_bridge": 0.70,
        "weak_bridge": 0.35,
        "inferred": 0.10,
    }
    evidence_support = sum(
        weights.get(state.evidence, 0.0)
        for state in identity.states.values()
    )

    return float(
        evidence_support
        + 12.0 * identity.core_fragment_count
        + 3.0 * identity.bridge_fragment_count
        + 2.0 * identity.support_score
        + 8.0 * math.log1p(_identity_span_s(identity))
    )


def _raw_strong_positions(
    frames: Sequence[Any],
    *,
    strong_threshold: float,
) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for frame in frames:
        positions = [
            np.asarray(observation.world_position, dtype=np.float64)
            for observation in frame.observations
            if float(observation.confidence) >= strong_threshold
        ]
        result[int(frame.frame)] = (
            np.asarray(positions, dtype=np.float64)
            if positions
            else np.zeros((0, 2), dtype=np.float64)
        )
    return result


def _pair_overlap_frames(
    left: v5.GlobalIdentityV5,
    right: v5.GlobalIdentityV5,
) -> list[int]:
    # Iterate the smaller map to avoid materialising two large frame sets for a
    # future 45-minute half.
    if len(left.states) <= len(right.states):
        return sorted(
            frame_id
            for frame_id in left.states
            if frame_id in right.states
        )
    return sorted(
        frame_id
        for frame_id in right.states
        if frame_id in left.states
    )


def _preferred_winner(
    left: v5.GlobalIdentityV5,
    right: v5.GlobalIdentityV5,
    *,
    left_strength: float,
    right_strength: float,
) -> tuple[int, int]:
    left_key = (
        left_strength,
        _identity_span_s(left),
        left.core_fragment_count,
        len(left.states),
        -left.identity_id,
    )
    right_key = (
        right_strength,
        _identity_span_s(right),
        right.core_fragment_count,
        len(right.states),
        -right.identity_id,
    )
    if left_key >= right_key:
        return left.identity_id, right.identity_id
    return right.identity_id, left.identity_id


def evaluate_branch_competition(
    left: v5.GlobalIdentityV5,
    right: v5.GlobalIdentityV5,
    raw_strong: Mapping[int, np.ndarray],
    *,
    config: GlobalIdentityV5_1Config,
) -> BranchCompetitionV5_1 | None:
    if left.identity_id == right.identity_id:
        return None

    overlap = _pair_overlap_frames(left, right)
    if len(overlap) < config.min_overlap_frames:
        return None

    first = overlap[0]
    last = overlap[-1]
    overlap_duration = max(
        0.0,
        float(left.states[last].time_s) - float(left.states[first].time_s),
    )
    if overlap_duration < config.min_overlap_s:
        return None

    span_left = _identity_span_s(left)
    span_right = _identity_span_s(right)
    shorter_span = min(span_left, span_right)
    # A clip-boundary identity can be shorter than one frame in synthetic
    # tests.  Use one observed overlap duration as the denominator floor.
    denominator = max(shorter_span, overlap_duration, 1e-6)
    shorter_overlap_fraction = min(1.0, overlap_duration / denominator)

    distances: list[float] = []
    relative_speeds: list[float] = []
    both_strong = 0

    for frame_id in overlap:
        state_left = left.states[frame_id]
        state_right = right.states[frame_id]
        distances.append(
            float(np.linalg.norm(state_left.position - state_right.position))
        )
        relative_speeds.append(
            float(np.linalg.norm(state_left.velocity - state_right.velocity))
        )
        if (
            state_left.evidence == "strong_observed"
            and state_right.evidence == "strong_observed"
        ):
            both_strong += 1

    median_distance = float(statistics.median(distances))
    p90_distance = float(_percentile(distances, 0.90))
    median_relative_speed = float(statistics.median(relative_speeds))
    both_strong_ratio = both_strong / len(overlap)

    informative = 0
    independent = 0
    for frame_id in overlap[:: config.evidence_stride]:
        state_left = left.states[frame_id]
        state_right = right.states[frame_id]
        is_informative, is_independent = v3._distinct_observation_support(
            state_left.position,
            state_right.position,
            raw_strong.get(
                frame_id,
                np.zeros((0, 2), dtype=np.float64),
            ),
            radius_m=config.independent_evidence_radius_m,
        )
        if is_informative:
            informative += 1
            independent += int(is_independent)

    independent_ratio = (
        independent / informative
        if informative > 0
        else None
    )

    shared_sources = tuple(
        sorted(
            set(left.source_track_ids).intersection(right.source_track_ids)
        )
    )

    left_strength = _identity_strength(left)
    right_strength = _identity_strength(right)
    stronger = max(left_strength, right_strength)
    weaker = min(left_strength, right_strength)
    weaker_strength_ratio = (
        weaker / stronger
        if stronger > 1e-9
        else 1.0
    )

    keep_id, suppress_id = _preferred_winner(
        left,
        right,
        left_strength=left_strength,
        right_strength=right_strength,
    )

    duplicate = True
    reason = "duplicate_branch"

    if median_distance > config.duplicate_median_distance_m:
        duplicate = False
        reason = "median_distance"
    elif p90_distance > config.duplicate_p90_distance_m:
        duplicate = False
        reason = "p90_distance"
    elif median_relative_speed > config.duplicate_max_relative_speed_mps:
        duplicate = False
        reason = "relative_motion"
    elif shorter_overlap_fraction < config.min_shorter_overlap_fraction:
        duplicate = False
        reason = "not_branch_contained"
    elif (
        independent_ratio is not None
        and independent_ratio > config.max_independent_evidence_ratio
    ):
        duplicate = False
        reason = "independent_strong_evidence"
    elif both_strong_ratio > config.max_both_strong_observed_ratio:
        duplicate = False
        reason = "both_strong_observed"
    elif informative == 0 and not shared_sources:
        # Critical football-contact safeguard: proximity by itself is never
        # enough evidence that two people are one.
        duplicate = False
        reason = "no_independent_evidence_test"
    else:
        allowed_span = (
            config.shared_lineage_max_branch_span_s
            if shared_sources
            else config.max_branch_span_s
        )
        if shorter_span > allowed_span:
            duplicate = False
            reason = "branch_too_long"
        elif (
            not shared_sources
            and weaker_strength_ratio > config.max_weaker_strength_ratio
        ):
            duplicate = False
            reason = "support_too_similar"

    closeness = max(
        0.0,
        1.0 - median_distance / config.duplicate_median_distance_m,
    )
    tail_closeness = max(
        0.0,
        1.0 - p90_distance / config.duplicate_p90_distance_m,
    )
    motion_similarity = max(
        0.0,
        1.0
        - median_relative_speed / config.duplicate_max_relative_speed_mps,
    )
    evidence_term = (
        0.25
        if independent_ratio is None
        else max(0.0, 1.0 - independent_ratio)
    )
    asymmetry = max(0.0, 1.0 - weaker_strength_ratio)
    lineage_bonus = 0.75 if shared_sources else 0.0

    score = float(
        1.25 * closeness
        + 0.75 * tail_closeness
        + 0.50 * motion_similarity
        + 1.00 * shorter_overlap_fraction
        + 0.70 * evidence_term
        + 0.60 * asymmetry
        + lineage_bonus
        - 0.75 * both_strong_ratio
    )

    return BranchCompetitionV5_1(
        identity_a=int(left.identity_id),
        identity_b=int(right.identity_id),
        overlap_frames=len(overlap),
        overlap_duration_s=float(overlap_duration),
        shorter_identity_span_s=float(shorter_span),
        shorter_overlap_fraction=float(shorter_overlap_fraction),
        median_distance_m=median_distance,
        p90_distance_m=p90_distance,
        median_relative_speed_mps=median_relative_speed,
        informative_strong_frames=int(informative),
        independent_strong_frames=int(independent),
        independent_strong_ratio=(
            None if independent_ratio is None else float(independent_ratio)
        ),
        both_strong_observed_frames=int(both_strong),
        both_strong_observed_ratio=float(both_strong_ratio),
        shared_source_track_ids=shared_sources,
        identity_a_strength=float(left_strength),
        identity_b_strength=float(right_strength),
        weaker_strength_ratio=float(weaker_strength_ratio),
        preferred_keep_identity_id=int(keep_id),
        preferred_suppress_identity_id=int(suppress_id),
        duplicate_branch=bool(duplicate),
        reason=reason,
        score=score,
    )


def _time_indexed_identity_pairs(
    identities: Sequence[v5.GlobalIdentityV5],
) -> list[tuple[v5.GlobalIdentityV5, v5.GlobalIdentityV5]]:
    """Return only identity pairs whose analytical lifetimes overlap."""

    ordered = sorted(
        identities,
        key=lambda identity: (
            identity.observed_start_time_s,
            identity.identity_id,
        ),
    )
    starts = [identity.observed_start_time_s for identity in ordered]
    pairs: list[tuple[v5.GlobalIdentityV5, v5.GlobalIdentityV5]] = []

    for index, left in enumerate(ordered):
        # No later identity can overlap after left's observed end.  This keeps
        # the post-link pass scalable for a full half.
        stop = bisect_left(
            starts,
            left.observed_end_time_s + 1e-9,
            lo=index + 1,
        )
        for right in ordered[index + 1 : stop]:
            if right.observed_start_time_s > left.observed_end_time_s:
                break
            pairs.append((left, right))

    return pairs


def resolve_duplicate_branches(
    identities: Sequence[v5.GlobalIdentityV5],
    frames: Sequence[Any],
    *,
    config: GlobalIdentityV5_1Config,
) -> tuple[
    tuple[BranchCompetitionV5_1, ...],
    tuple[int, ...],
    tuple[v5.GlobalIdentityV5, ...],
]:
    identities = tuple(identities)
    raw_strong = _raw_strong_positions(
        frames,
        strong_threshold=config.base.strong_threshold,
    )

    competitions: list[BranchCompetitionV5_1] = []
    duplicate_candidates: list[BranchCompetitionV5_1] = []

    for left, right in _time_indexed_identity_pairs(identities):
        evidence = evaluate_branch_competition(
            left,
            right,
            raw_strong,
            config=config,
        )
        if evidence is None:
            continue
        competitions.append(evidence)
        if evidence.duplicate_branch:
            duplicate_candidates.append(evidence)

    # Strongest branch competitions first.  Never suppress an identity in
    # favour of a winner that has already itself been suppressed: this keeps
    # the refinement conservative and prevents transitive proximity collapse.
    duplicate_candidates.sort(
        key=lambda item: (
            -item.score,
            item.preferred_suppress_identity_id,
            item.preferred_keep_identity_id,
        )
    )

    suppressed: set[int] = set()
    for candidate in duplicate_candidates:
        keep = candidate.preferred_keep_identity_id
        remove = candidate.preferred_suppress_identity_id
        if keep in suppressed or remove in suppressed:
            continue
        suppressed.add(remove)

    survivors = tuple(
        identity
        for identity in identities
        if identity.identity_id not in suppressed
    )

    competitions.sort(
        key=lambda item: (
            not item.duplicate_branch,
            -item.score,
            item.identity_a,
            item.identity_b,
        )
    )

    return (
        tuple(competitions),
        tuple(sorted(suppressed)),
        survivors,
    )


def build_frame_tracks(
    frames: Sequence[Any],
    identities: Sequence[v5.GlobalIdentityV5],
) -> dict[int, tuple[v5.AnalyticalFrameTrack, ...]]:
    return v5.build_frame_tracks(frames, identities)


def solve_global_identities(
    frames: Sequence[Any],
    strong_frames: Sequence[Any],
    trajectories: Sequence[Any],
    *,
    field_length_m: float | None,
    field_width_m: float | None,
    config: GlobalIdentityV5_1Config | None = None,
) -> GlobalIdentityV5_1Result:
    cfg = config or make_config()

    base_result = v5.solve_global_identities(
        frames,
        strong_frames,
        trajectories,
        field_length_m=field_length_m,
        field_width_m=field_width_m,
        config=cfg.base,
    )

    competitions, suppressed_ids, identities = resolve_duplicate_branches(
        base_result.identities,
        frames,
        config=cfg,
    )

    frame_tracks = build_frame_tracks(frames, identities)

    return GlobalIdentityV5_1Result(
        duplicate_evidence=base_result.duplicate_evidence,
        duplicate_components=base_result.duplicate_components,
        blocked_duplicate_merges=base_result.blocked_duplicate_merges,
        fragments=base_result.fragments,
        raw_candidate_count=base_result.raw_candidate_count,
        link_candidates=base_result.link_candidates,
        accepted_links=base_result.accepted_links,
        pre_refinement_identities=base_result.identities,
        branch_competitions=competitions,
        suppressed_identity_ids=suppressed_ids,
        identities=identities,
        frame_tracks=frame_tracks,
    )


__all__ = [
    "BranchCompetitionV5_1",
    "GlobalIdentityV5_1Config",
    "GlobalIdentityV5_1Result",
    "build_frame_tracks",
    "evaluate_branch_competition",
    "make_config",
    "resolve_duplicate_branches",
    "solve_global_identities",
]
