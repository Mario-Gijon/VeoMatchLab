from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import hypot, isfinite
from statistics import median
from typing import Mapping, Sequence

from .models import PositionKind, TrackSample
from .quality import TrackQualityDecision, TrackQualityReport


@dataclass(slots=True, frozen=True)
class IdentityConsolidationConfig:
    """
    Conservative offline consolidation for overlapping identity fragments.

    This stage handles a failure mode that ordinary stitching cannot:

        canonical ID: OBSERVED -> PREDICTED -> OBSERVED
                                   ^
        fragment ID:              OBSERVED

    The fragment is merged only when temporal handover evidence is strong,
    spatial agreement is plausible, and one *eligible* candidate is clearly
    better than the other eligible alternatives.

    Important V1 detail:
    a runner-up that already fails hard handover/geometry gates is NOT allowed
    to make an otherwise strong merge look ambiguous.

    It does not use a target player count such as 23.
    """

    min_fragment_observed_samples: int = 8
    min_candidate_observed_samples: int = 30
    min_candidate_prior_observed_samples: int = 5

    min_temporal_coverage: float = 0.70
    min_handover_fraction: float = 0.70
    max_simultaneous_observed_fraction: float = 0.15

    max_median_distance_m: float = 4.50
    max_p90_distance_m: float = 10.0
    max_median_velocity_difference_m_s: float = 4.50

    min_score: float = 0.12
    min_score_ratio: float = 2.50
    min_score_margin: float = 0.08

    # Prevent two strong, simultaneous fragments from both replacing the same
    # canonical identity at the same time.
    max_fragment_conflict_fraction: float = 0.20

    def __post_init__(self) -> None:
        counts = (
            self.min_fragment_observed_samples,
            self.min_candidate_observed_samples,
            self.min_candidate_prior_observed_samples,
        )
        if any(value < 1 for value in counts):
            raise ValueError(
                "Identity consolidation sample thresholds must be >= 1"
            )

        fractions = (
            self.min_temporal_coverage,
            self.min_handover_fraction,
            self.max_simultaneous_observed_fraction,
            self.max_fragment_conflict_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError(
                "Identity consolidation fractions must be in [0, 1]"
            )

        positive = (
            self.max_median_distance_m,
            self.max_p90_distance_m,
            self.max_median_velocity_difference_m_s,
            self.min_score,
            self.min_score_ratio,
            self.min_score_margin,
        )
        if any(value <= 0.0 for value in positive):
            raise ValueError(
                "Identity consolidation positive thresholds must be > 0"
            )


@dataclass(slots=True, frozen=True)
class IdentityHandoverCandidate:
    fragment_id: int
    candidate_id: int

    fragment_observed_samples: int
    candidate_observed_samples: int
    candidate_prior_observed_samples: int
    candidate_later_observed_samples: int

    compared_frames: int
    temporal_coverage: float

    handover_frames: int
    handover_fraction: float

    simultaneous_observed_frames: int
    simultaneous_observed_fraction: float

    min_distance_m: float
    median_distance_m: float
    p90_distance_m: float
    max_distance_m: float

    within_0_75m_fraction: float
    within_1_50m_fraction: float
    within_2_50m_fraction: float

    median_velocity_difference_m_s: float
    score: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True, frozen=True)
class IdentityMerge:
    fragment_id: int
    canonical_id: int
    score: float
    score_ratio: float
    score_margin: float
    candidate: IdentityHandoverCandidate

    def to_dict(self) -> dict[str, object]:
        return {
            "fragment_id": self.fragment_id,
            "canonical_id": self.canonical_id,
            "score": self.score,
            # A unique eligible candidate legitimately produces an infinite
            # best/runner-up ratio. JSON has no standard representation for
            # infinity, so reports encode it as null instead of emitting NaN/
            # Infinity or crashing under allow_nan=False.
            "score_ratio": (
                self.score_ratio
                if isfinite(self.score_ratio)
                else None
            ),
            "score_margin": self.score_margin,
            "candidate": self.candidate.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class IdentityConsolidationReport:
    merges: tuple[IdentityMerge, ...]
    canonical_id_by_track: Mapping[int, int]
    rejected_proposals: Mapping[int, str]
    candidate_rankings: Mapping[int, tuple[IdentityHandoverCandidate, ...]]

    def to_dict(self) -> dict[str, object]:
        return {
            "merges": [merge.to_dict() for merge in self.merges],
            "canonical_id_by_track": {
                str(track_id): canonical_id
                for track_id, canonical_id in sorted(
                    self.canonical_id_by_track.items()
                )
            },
            "rejected_proposals": {
                str(track_id): reason
                for track_id, reason in sorted(
                    self.rejected_proposals.items()
                )
            },
            "candidate_rankings": {
                str(track_id): [
                    candidate.to_dict()
                    for candidate in ranking
                ]
                for track_id, ranking
                in sorted(self.candidate_rankings.items())
            },
        }


def consolidate_identities(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    quality_report: TrackQualityReport,
    config: IdentityConsolidationConfig | None = None,
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    IdentityConsolidationReport,
]:
    """
    Merge overlapping/intermittent identity fragments into earlier track IDs.

    Only quality-accepted tracks (KEEP + UNCERTAIN) participate. REJECT tracks
    are omitted completely.

    V1 decision procedure:
    1. rank all temporal candidates for diagnostics;
    2. discard candidates that fail hard handover/geometry gates;
    3. compare ambiguity only among candidates that remain eligible;
    4. merge only when the best eligible candidate is unique enough.

    This prevents a clearly different simultaneously-observed player from
    blocking a genuine handover merely because its soft score is second-best.
    """
    cfg = config or IdentityConsolidationConfig()
    accepted = set(quality_report.accepted_track_ids)

    normalized: dict[int, tuple[TrackSample, ...]] = {}
    observed_by_track: dict[int, tuple[TrackSample, ...]] = {}

    for track_id, raw_samples in sorted(trajectories.items()):
        if track_id not in accepted:
            continue

        samples = _validated_samples(track_id, raw_samples)
        if not samples:
            continue

        observed = tuple(
            sample
            for sample in samples
            if sample.position_kind is PositionKind.OBSERVED
        )
        if not observed:
            continue

        normalized[track_id] = samples
        observed_by_track[track_id] = observed

    rankings: dict[
        int,
        tuple[IdentityHandoverCandidate, ...],
    ] = {}
    rejected: dict[int, str] = {}
    proposals: list[IdentityMerge] = []

    for fragment_id in sorted(normalized):
        fragment_observed = observed_by_track[fragment_id]

        if len(fragment_observed) < cfg.min_fragment_observed_samples:
            rejected[fragment_id] = "too_few_fragment_observations"
            rankings[fragment_id] = ()
            continue

        quality = quality_report.results.get(fragment_id)
        if (
            quality is None
            or quality.decision is TrackQualityDecision.REJECT
        ):
            continue

        candidates: list[IdentityHandoverCandidate] = []

        for candidate_id in sorted(normalized):
            if candidate_id == fragment_id:
                continue

            candidate_observed = observed_by_track[candidate_id]

            # Canonical identity must already exist before the fragment starts.
            if (
                candidate_observed[0].timestamp
                >= fragment_observed[0].timestamp
            ):
                continue

            if (
                len(candidate_observed)
                < cfg.min_candidate_observed_samples
            ):
                continue

            handover = handover_candidate(
                fragment_observed,
                normalized[candidate_id],
            )
            if handover is None:
                continue

            if (
                handover.candidate_prior_observed_samples
                < cfg.min_candidate_prior_observed_samples
            ):
                continue

            candidates.append(handover)

        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.median_distance_m,
                -candidate.temporal_coverage,
                candidate.candidate_id,
            )
        )
        rankings[fragment_id] = tuple(candidates)

        if not candidates:
            rejected[fragment_id] = "no_preexisting_candidate"
            continue

        # CRITICAL V1 CHANGE:
        # ambiguity is measured only between candidates that are individually
        # plausible identity handovers.
        eligible_candidates = [
            candidate
            for candidate in candidates
            if _hard_candidate_failure_reason(
                candidate,
                config=cfg,
            )
            is None
        ]

        if not eligible_candidates:
            rejected[fragment_id] = _hard_candidate_failure_reason(
                candidates[0],
                config=cfg,
            ) or "no_eligible_candidate"
            continue

        best = eligible_candidates[0]
        second_score = (
            eligible_candidates[1].score
            if len(eligible_candidates) > 1
            else 0.0
        )

        score_ratio = (
            best.score / second_score
            if second_score > 0.0
            else float("inf")
        )
        score_margin = best.score - second_score

        uniqueness_failure = _uniqueness_failure_reason(
            score_ratio=score_ratio,
            score_margin=score_margin,
            config=cfg,
        )
        if uniqueness_failure is not None:
            rejected[fragment_id] = uniqueness_failure
            continue

        proposals.append(
            IdentityMerge(
                fragment_id=fragment_id,
                canonical_id=best.candidate_id,
                score=best.score,
                score_ratio=score_ratio,
                score_margin=score_margin,
                candidate=best,
            )
        )

    # Strongest proposals claim their canonical identity first. Additional
    # fragments may merge into the same identity only when their real observed
    # frames do not strongly conflict.
    proposals.sort(
        key=lambda merge: (
            -merge.score,
            -merge.score_margin,
            merge.fragment_id,
        )
    )

    accepted_merges: list[IdentityMerge] = []
    claimed_observed_frames: dict[int, set[int]] = {}

    for merge in proposals:
        root = _resolve_root(
            merge.canonical_id,
            {
                item.fragment_id: item.canonical_id
                for item in accepted_merges
            },
        )

        fragment_frames = {
            sample.frame_index
            for sample in observed_by_track[merge.fragment_id]
        }
        already_claimed = claimed_observed_frames.setdefault(
            root,
            set(),
        )

        conflict_fraction = (
            len(fragment_frames & already_claimed)
            / len(fragment_frames)
            if fragment_frames
            else 0.0
        )

        if (
            conflict_fraction
            > cfg.max_fragment_conflict_fraction
        ):
            rejected[
                merge.fragment_id
            ] = "conflicts_with_stronger_fragment"
            continue

        if root != merge.canonical_id:
            merge = replace(
                merge,
                canonical_id=root,
            )

        accepted_merges.append(merge)
        already_claimed.update(fragment_frames)

    direct_map = {
        merge.fragment_id: merge.canonical_id
        for merge in accepted_merges
    }

    canonical_id_by_track: dict[int, int] = {}
    for track_id in sorted(normalized):
        canonical_id_by_track[track_id] = _resolve_root(
            track_id,
            direct_map,
        )

    grouped: dict[int, list[int]] = {}
    for track_id, canonical_id in canonical_id_by_track.items():
        grouped.setdefault(
            canonical_id,
            [],
        ).append(track_id)

    consolidated: dict[int, tuple[TrackSample, ...]] = {}
    for canonical_id, members in sorted(grouped.items()):
        consolidated[canonical_id] = _merge_members(
            canonical_id,
            members,
            normalized,
        )

    report = IdentityConsolidationReport(
        merges=tuple(
            sorted(
                accepted_merges,
                key=lambda item: (
                    item.fragment_id,
                    item.canonical_id,
                ),
            )
        ),
        canonical_id_by_track=canonical_id_by_track,
        rejected_proposals=rejected,
        candidate_rankings=rankings,
    )
    return consolidated, report


def handover_candidate(
    fragment_observed: Sequence[TrackSample],
    candidate_trajectory: Sequence[TrackSample],
) -> IdentityHandoverCandidate | None:
    """
    Measure whether `fragment_observed` looks like a temporary replacement for
    `candidate_trajectory`.

    The fragment input must contain OBSERVED samples only. Candidate samples
    may be OBSERVED, PREDICTED, or INTERPOLATED.
    """
    if not fragment_observed or not candidate_trajectory:
        return None

    fragment_id = fragment_observed[0].track_id
    candidate_id = candidate_trajectory[0].track_id

    if fragment_id == candidate_id:
        return None

    if any(
        sample.track_id != fragment_id
        or sample.position_kind is not PositionKind.OBSERVED
        for sample in fragment_observed
    ):
        raise ValueError(
            "fragment_observed must contain OBSERVED samples from one track"
        )

    if any(
        sample.track_id != candidate_id
        for sample in candidate_trajectory
    ):
        raise ValueError(
            "candidate_trajectory must contain samples from one track"
        )

    candidate_by_frame = {
        sample.frame_index: sample
        for sample in candidate_trajectory
    }
    candidate_observed = [
        sample
        for sample in candidate_trajectory
        if sample.position_kind is PositionKind.OBSERVED
    ]

    first_fragment_time = fragment_observed[0].timestamp
    last_fragment_time = fragment_observed[-1].timestamp

    prior_observed = sum(
        sample.timestamp < first_fragment_time
        for sample in candidate_observed
    )
    later_observed = sum(
        sample.timestamp > last_fragment_time
        for sample in candidate_observed
    )

    distances: list[float] = []
    velocity_differences: list[float] = []
    handover_frames = 0
    simultaneous_observed_frames = 0

    for fragment in fragment_observed:
        candidate = candidate_by_frame.get(fragment.frame_index)
        if candidate is None:
            continue

        distance = hypot(
            fragment.x - candidate.x,
            fragment.z - candidate.z,
        )
        velocity_difference = hypot(
            fragment.vx - candidate.vx,
            fragment.vz - candidate.vz,
        )

        if (
            not isfinite(distance)
            or not isfinite(velocity_difference)
        ):
            continue

        distances.append(distance)
        velocity_differences.append(velocity_difference)

        if candidate.position_kind is PositionKind.OBSERVED:
            simultaneous_observed_frames += 1
        else:
            # PREDICTED is the current V1 handover state. INTERPOLATED is
            # included now so this remains valid after 30 FPS interpolation.
            handover_frames += 1

    if not distances:
        return None

    compared = len(distances)
    temporal_coverage = (
        compared / len(fragment_observed)
    )
    handover_fraction = (
        handover_frames / compared
    )
    simultaneous_fraction = (
        simultaneous_observed_frames / compared
    )

    sorted_distances = sorted(distances)
    median_distance = float(
        median(sorted_distances)
    )
    p90_distance = _percentile(
        sorted_distances,
        0.90,
    )
    median_velocity_difference = float(
        median(velocity_differences)
    )

    within_075 = (
        sum(value <= 0.75 for value in distances)
        / compared
    )
    within_150 = (
        sum(value <= 1.50 for value in distances)
        / compared
    )
    within_250 = (
        sum(value <= 2.50 for value in distances)
        / compared
    )

    # Geometry remains soft because far-player X/Z can contain metre-scale
    # errors. The hard temporal gates below carry the identity semantics.
    proximity = 1.0 / (1.0 + median_distance)
    velocity_term = (
        1.0
        / (
            1.0
            + 0.25
            * median_velocity_difference
        )
    )

    score = (
        temporal_coverage
        * proximity
        * velocity_term
        * (
            1.0
            + 0.65 * handover_fraction
            + 0.35 * within_150
            - 0.45 * simultaneous_fraction
        )
    )

    return IdentityHandoverCandidate(
        fragment_id=fragment_id,
        candidate_id=candidate_id,
        fragment_observed_samples=len(
            fragment_observed
        ),
        candidate_observed_samples=len(
            candidate_observed
        ),
        candidate_prior_observed_samples=prior_observed,
        candidate_later_observed_samples=later_observed,
        compared_frames=compared,
        temporal_coverage=temporal_coverage,
        handover_frames=handover_frames,
        handover_fraction=handover_fraction,
        simultaneous_observed_frames=(
            simultaneous_observed_frames
        ),
        simultaneous_observed_fraction=(
            simultaneous_fraction
        ),
        min_distance_m=float(
            sorted_distances[0]
        ),
        median_distance_m=median_distance,
        p90_distance_m=p90_distance,
        max_distance_m=float(
            sorted_distances[-1]
        ),
        within_0_75m_fraction=within_075,
        within_1_50m_fraction=within_150,
        within_2_50m_fraction=within_250,
        median_velocity_difference_m_s=(
            median_velocity_difference
        ),
        score=float(score),
    )


def _hard_candidate_failure_reason(
    candidate: IdentityHandoverCandidate,
    *,
    config: IdentityConsolidationConfig,
) -> str | None:
    """
    Test whether one candidate is individually plausible.

    These gates are applied BEFORE best-vs-runner-up ambiguity checks.
    """
    if (
        candidate.temporal_coverage
        < config.min_temporal_coverage
    ):
        return "insufficient_temporal_coverage"

    if (
        candidate.handover_fraction
        < config.min_handover_fraction
    ):
        return "not_a_temporal_handover"

    if (
        candidate.simultaneous_observed_fraction
        > config.max_simultaneous_observed_fraction
    ):
        return "too_many_simultaneous_real_observations"

    if (
        candidate.median_distance_m
        > config.max_median_distance_m
    ):
        return "median_distance_too_large"

    if (
        candidate.p90_distance_m
        > config.max_p90_distance_m
    ):
        return "p90_distance_too_large"

    if (
        candidate.median_velocity_difference_m_s
        > config.max_median_velocity_difference_m_s
    ):
        return "velocity_disagreement"

    if candidate.score < config.min_score:
        return "handover_score_too_low"

    return None


def _uniqueness_failure_reason(
    *,
    score_ratio: float,
    score_margin: float,
    config: IdentityConsolidationConfig,
) -> str | None:
    if score_ratio < config.min_score_ratio:
        return "best_candidate_not_unique_enough"

    if score_margin < config.min_score_margin:
        return "best_candidate_margin_too_small"

    return None


def _merge_members(
    canonical_id: int,
    members: Sequence[int],
    trajectories: Mapping[int, tuple[TrackSample, ...]],
) -> tuple[TrackSample, ...]:
    """
    Merge all member samples onto one canonical ID.

    Same-frame priority:
      OBSERVED > INTERPOLATED > PREDICTED

    If two OBSERVED samples collide, prefer higher confidence. When confidence
    is unavailable/equal, preserve the canonical track's own sample.
    """
    by_frame: dict[int, tuple[int, TrackSample]] = {}

    ordered_members = [canonical_id] + [
        track_id
        for track_id in sorted(members)
        if track_id != canonical_id
    ]

    for source_track_id in ordered_members:
        for sample in trajectories[source_track_id]:
            rewritten = replace(
                sample,
                track_id=canonical_id,
            )
            existing = by_frame.get(
                rewritten.frame_index
            )

            if existing is None:
                by_frame[
                    rewritten.frame_index
                ] = (
                    source_track_id,
                    rewritten,
                )
                continue

            existing_source, existing_sample = existing

            if _prefer_sample(
                new=rewritten,
                new_source=source_track_id,
                existing=existing_sample,
                existing_source=existing_source,
                canonical_id=canonical_id,
            ):
                by_frame[
                    rewritten.frame_index
                ] = (
                    source_track_id,
                    rewritten,
                )

    return tuple(
        sample
        for _, sample in sorted(
            by_frame.values(),
            key=lambda pair: (
                pair[1].timestamp,
                pair[1].frame_index,
            ),
        )
    )


def _prefer_sample(
    *,
    new: TrackSample,
    new_source: int,
    existing: TrackSample,
    existing_source: int,
    canonical_id: int,
) -> bool:
    new_priority = _sample_priority(new)
    existing_priority = _sample_priority(existing)

    if new_priority != existing_priority:
        return (
            new_priority > existing_priority
        )

    if new.position_kind is PositionKind.OBSERVED:
        new_conf = new.confidence
        existing_conf = existing.confidence

        if (
            new_conf is not None
            and existing_conf is not None
        ):
            if new_conf != existing_conf:
                return new_conf > existing_conf
        elif new_conf is not None:
            return True
        elif existing_conf is not None:
            return False

    if existing_source == canonical_id:
        return False
    if new_source == canonical_id:
        return True

    return new_source < existing_source


def _sample_priority(
    sample: TrackSample,
) -> int:
    return {
        PositionKind.OBSERVED: 3,
        PositionKind.INTERPOLATED: 2,
        PositionKind.PREDICTED: 1,
    }[sample.position_kind]


def _resolve_root(
    track_id: int,
    direct_map: Mapping[int, int],
) -> int:
    seen: set[int] = set()
    current = track_id

    while current in direct_map:
        if current in seen:
            raise RuntimeError(
                "Cycle detected in identity consolidation"
            )
        seen.add(current)
        current = direct_map[current]

    return current


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

    for sample in ordered:
        if sample.track_id != track_id:
            raise ValueError(
                f"Trajectory key {track_id} does not match "
                f"sample track_id {sample.track_id}"
            )

        if not isfinite(sample.timestamp):
            raise ValueError(
                "Track sample timestamp must be finite"
            )

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

    return ordered


def _percentile(
    sorted_values: Sequence[float],
    q: float,
) -> float:
    if not sorted_values:
        raise ValueError(
            "percentile requires data"
        )
    if not 0.0 <= q <= 1.0:
        raise ValueError(
            "q must be in [0, 1]"
        )
    if len(sorted_values) == 1:
        return float(sorted_values[0])

    position = q * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(
        lower + 1,
        len(sorted_values) - 1,
    )
    fraction = position - lower

    return float(
        sorted_values[lower]
        * (1.0 - fraction)
        + sorted_values[upper]
        * fraction
    )
