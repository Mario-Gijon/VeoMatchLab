from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from math import hypot, isfinite
from statistics import median
from typing import Mapping, Sequence

from .identity_consolidation import (
    IdentityConsolidationConfig,
    IdentityConsolidationReport,
    consolidate_identities,
)
from .models import PositionKind, TrackSample
from .quality import (
    TrackQualityConfig,
    TrackQualityReport,
    analyze_trajectories,
)
from .stitching import (
    StitchingConfig,
    StitchingReport,
    stitch_trajectories,
)


class IdentityRelationKind(str, Enum):
    HANDOVER = "handover"
    STITCH = "stitch"


@dataclass(slots=True, frozen=True)
class IdentityResolutionConfig:
    """
    Global relation-graph safeguards for Tracking V2.2.

    Identity consolidation and ordinary stitching detect different failure
    modes. Their selected relations are therefore computed independently on
    the same quality-accepted raw trajectories and then resolved transitively.

    Example:

        30 --stitch--> 34 --handover--> 2

    becomes one component {2, 30, 34}, whose canonical ID is the earliest
    established identity (normally 2).

    A proposed union is rejected when it would create a component containing
    two trajectories that are simultaneously OBSERVED for a substantial part
    of the shorter trajectory *and* spatially separated. This prevents an
    accidental chain of individually plausible links from collapsing two real
    people into one identity.
    """

    component_conflict_min_frames: int = 3
    max_component_simultaneous_observed_fraction: float = 0.20
    component_conflict_median_distance_m: float = 3.0

    def __post_init__(self) -> None:
        if self.component_conflict_min_frames < 1:
            raise ValueError("component_conflict_min_frames must be >= 1")
        if not 0.0 <= self.max_component_simultaneous_observed_fraction <= 1.0:
            raise ValueError(
                "max_component_simultaneous_observed_fraction must be in [0, 1]"
            )
        if self.component_conflict_median_distance_m <= 0.0:
            raise ValueError(
                "component_conflict_median_distance_m must be > 0"
            )


@dataclass(slots=True, frozen=True)
class IdentityRelation:
    left_id: int
    right_id: int
    kind: IdentityRelationKind

    # Higher strength means stronger evidence, but strengths are only used for
    # ordering within one relation kind. HANDOVER relations are intentionally
    # processed before STITCH relations because their temporal semantics are
    # more discriminative than endpoint-only geometry.
    strength: float

    # Human-readable source metrics for audit/debugging.
    detail: Mapping[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "kind": self.kind.value,
            "strength": self.strength,
            "detail": dict(self.detail),
        }


@dataclass(slots=True, frozen=True)
class RejectedIdentityRelation:
    relation: IdentityRelation
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "relation": self.relation.to_dict(),
            "reason": self.reason,
        }


@dataclass(slots=True, frozen=True)
class IdentityResolutionReport:
    quality: TrackQualityReport
    consolidation: IdentityConsolidationReport
    stitching: StitchingReport

    accepted_relations: tuple[IdentityRelation, ...]
    rejected_relations: tuple[RejectedIdentityRelation, ...]

    components: tuple[tuple[int, ...], ...]
    canonical_id_by_track: Mapping[int, int]

    @property
    def raw_track_count(self) -> int:
        return len(self.quality.results)

    @property
    def accepted_before_resolution(self) -> int:
        return len(self.quality.accepted_track_ids)

    @property
    def final_identity_count(self) -> int:
        return len(self.components)

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": {
                "raw_tracks": self.raw_track_count,
                "quality_keep": len(self.quality.kept_track_ids),
                "quality_uncertain": len(self.quality.uncertain_track_ids),
                "quality_reject": len(self.quality.rejected_track_ids),
                "accepted_before_resolution": self.accepted_before_resolution,
                "accepted_relations": len(self.accepted_relations),
                "rejected_relations": len(self.rejected_relations),
                "final_identities": self.final_identity_count,
            },
            "quality": self.quality.to_dict(),
            "consolidation": self.consolidation.to_dict(),
            "stitching": self.stitching.to_dict(),
            "accepted_relations": [
                relation.to_dict()
                for relation in self.accepted_relations
            ],
            "rejected_relations": [
                item.to_dict()
                for item in self.rejected_relations
            ],
            "components": [list(component) for component in self.components],
            "canonical_id_by_track": {
                str(track_id): canonical_id
                for track_id, canonical_id in sorted(
                    self.canonical_id_by_track.items()
                )
            },
        }


def resolve_identities(
    trajectories: Mapping[int, Sequence[TrackSample]],
    *,
    field_length: float,
    field_width: float,
    quality_config: TrackQualityConfig | None = None,
    consolidation_config: IdentityConsolidationConfig | None = None,
    stitching_config: StitchingConfig | None = None,
    resolution_config: IdentityResolutionConfig | None = None,
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    IdentityResolutionReport,
]:
    """
    Resolve quality filtering + handover consolidation + stitching globally.

    Critical design rule:
    consolidation and stitching are BOTH run against the original accepted raw
    trajectories. We do not feed one stage destructively into the other.
    Their selected identity relations are combined into a graph and resolved
    transitively afterwards.

    This preserves bridges such as:

        raw stitching:       30 -> 34
        raw consolidation:   34 -> 2
        global resolution:   {2, 30, 34} -> canonical ID 2

    No target count (23 or otherwise) is used.
    """
    cfg = resolution_config or IdentityResolutionConfig()
    normalized = _validated_trajectories(trajectories)

    quality_report = analyze_trajectories(
        normalized,
        field_length=field_length,
        field_width=field_width,
        config=quality_config,
    )

    accepted_ids = set(quality_report.accepted_track_ids)
    accepted_raw = {
        track_id: samples
        for track_id, samples in normalized.items()
        if track_id in accepted_ids
    }

    # Both algorithms inspect the same raw accepted evidence. Their trajectory
    # outputs are intentionally ignored here; only their selected relations are
    # used by the global graph resolver below.
    _, consolidation_report = consolidate_identities(
        normalized,
        quality_report=quality_report,
        config=consolidation_config,
    )

    _, stitching_report = stitch_trajectories(
        normalized,
        config=stitching_config,
        allowed_track_ids=quality_report.accepted_track_ids,
    )

    relations = _relations_from_reports(
        consolidation_report,
        stitching_report,
    )

    observed_by_track = {
        track_id: tuple(
            sample
            for sample in samples
            if sample.position_kind is PositionKind.OBSERVED
        )
        for track_id, samples in accepted_raw.items()
    }

    parent = {track_id: track_id for track_id in accepted_raw}
    members = {
        track_id: {track_id}
        for track_id in accepted_raw
    }

    accepted_relations: list[IdentityRelation] = []
    rejected_relations: list[RejectedIdentityRelation] = []

    for relation in relations:
        if relation.left_id not in parent or relation.right_id not in parent:
            rejected_relations.append(
                RejectedIdentityRelation(
                    relation=relation,
                    reason="relation_contains_quality_rejected_track",
                )
            )
            continue

        left_root = _find(parent, relation.left_id)
        right_root = _find(parent, relation.right_id)

        if left_root == right_root:
            # Redundant relation already implied transitively. It is useful
            # evidence, not an error, so keep it in the accepted audit trail.
            accepted_relations.append(relation)
            continue

        left_members = members[left_root]
        right_members = members[right_root]

        conflict = _component_conflict_reason(
            left_members,
            right_members,
            observed_by_track=observed_by_track,
            config=cfg,
        )

        if conflict is not None:
            rejected_relations.append(
                RejectedIdentityRelation(
                    relation=relation,
                    reason=conflict,
                )
            )
            continue

        new_root = _union(
            parent,
            members,
            left_root,
            right_root,
        )
        # Force path compression immediately so later relation checks see the
        # complete transitive component.
        for track_id in members[new_root]:
            parent[track_id] = new_root

        accepted_relations.append(relation)

    grouped: dict[int, set[int]] = {}
    for track_id in sorted(accepted_raw):
        root = _find(parent, track_id)
        grouped.setdefault(root, set()).add(track_id)

    canonical_id_by_track: dict[int, int] = {}
    final_trajectories: dict[int, tuple[TrackSample, ...]] = {}
    component_rows: list[tuple[int, ...]] = []

    for component_members in grouped.values():
        component = tuple(sorted(component_members))
        canonical_id = _choose_canonical_id(
            component,
            observed_by_track,
        )

        for track_id in component:
            canonical_id_by_track[track_id] = canonical_id

        final_trajectories[canonical_id] = _merge_component(
            canonical_id,
            component,
            accepted_raw,
        )
        component_rows.append(component)

    component_rows.sort(
        key=lambda component: (
            _choose_canonical_id(component, observed_by_track),
            component,
        )
    )

    report = IdentityResolutionReport(
        quality=quality_report,
        consolidation=consolidation_report,
        stitching=stitching_report,
        accepted_relations=tuple(accepted_relations),
        rejected_relations=tuple(rejected_relations),
        components=tuple(component_rows),
        canonical_id_by_track=canonical_id_by_track,
    )

    return final_trajectories, report


def _relations_from_reports(
    consolidation: IdentityConsolidationReport,
    stitching: StitchingReport,
) -> tuple[IdentityRelation, ...]:
    handovers = [
        IdentityRelation(
            left_id=merge.fragment_id,
            right_id=merge.canonical_id,
            kind=IdentityRelationKind.HANDOVER,
            strength=float(merge.score),
            detail={
                "score": merge.score,
                "score_ratio": (
                    merge.score_ratio
                    if isfinite(merge.score_ratio)
                    else None
                ),
                "score_margin": merge.score_margin,
                "handover_fraction": merge.candidate.handover_fraction,
                "simultaneous_observed_fraction": (
                    merge.candidate.simultaneous_observed_fraction
                ),
                "median_distance_m": merge.candidate.median_distance_m,
                "p90_distance_m": merge.candidate.p90_distance_m,
            },
        )
        for merge in consolidation.merges
    ]
    handovers.sort(
        key=lambda relation: (
            -relation.strength,
            relation.left_id,
            relation.right_id,
        )
    )

    stitches = [
        IdentityRelation(
            left_id=link.predecessor_id,
            right_id=link.successor_id,
            kind=IdentityRelationKind.STITCH,
            strength=1.0 / (1.0 + float(link.cost)),
            detail={
                "cost": link.cost,
                "gap_seconds": link.candidate.gap_seconds,
                "forward_prediction_error_m": (
                    link.candidate.forward_prediction_error_m
                ),
                "backward_prediction_error_m": (
                    link.candidate.backward_prediction_error_m
                ),
                "endpoint_distance_m": link.candidate.endpoint_distance_m,
                "implied_speed_m_s": link.candidate.implied_speed_m_s,
            },
        )
        for link in stitching.links
    ]
    stitches.sort(
        key=lambda relation: (
            -relation.strength,
            relation.left_id,
            relation.right_id,
        )
    )

    # Handover first by design; then endpoint stitching adds non-overlapping
    # bridges and may connect whole handover components transitively.
    return tuple(handovers + stitches)


def _component_conflict_reason(
    left_members: set[int],
    right_members: set[int],
    *,
    observed_by_track: Mapping[int, Sequence[TrackSample]],
    config: IdentityResolutionConfig,
) -> str | None:
    for left_id in sorted(left_members):
        for right_id in sorted(right_members):
            conflict = _pair_conflict(
                observed_by_track.get(left_id, ()),
                observed_by_track.get(right_id, ()),
                config=config,
            )
            if conflict is not None:
                return (
                    f"component_observation_conflict:{left_id}:{right_id}:"
                    f"{conflict}"
                )
    return None


def _pair_conflict(
    left_observed: Sequence[TrackSample],
    right_observed: Sequence[TrackSample],
    *,
    config: IdentityResolutionConfig,
) -> str | None:
    if not left_observed or not right_observed:
        return None

    right_by_frame = {
        sample.frame_index: sample
        for sample in right_observed
    }

    distances: list[float] = []
    for left in left_observed:
        right = right_by_frame.get(left.frame_index)
        if right is None:
            continue
        distance = hypot(
            left.x - right.x,
            left.z - right.z,
        )
        if isfinite(distance):
            distances.append(distance)

    overlap_frames = len(distances)
    if overlap_frames < config.component_conflict_min_frames:
        return None

    shorter_count = min(
        len(left_observed),
        len(right_observed),
    )
    overlap_fraction = overlap_frames / shorter_count

    if (
        overlap_fraction
        <= config.max_component_simultaneous_observed_fraction
    ):
        return None

    median_distance = float(median(distances))
    if (
        median_distance
        <= config.component_conflict_median_distance_m
    ):
        # Repeated simultaneous observations that are spatially close can come
        # from duplicate/competing tracker identities. Do not reject the union
        # solely because of temporal overlap.
        return None

    return (
        f"overlap={overlap_frames},"
        f"fraction={overlap_fraction:.3f},"
        f"median_distance_m={median_distance:.3f}"
    )


def _choose_canonical_id(
    component: Sequence[int],
    observed_by_track: Mapping[int, Sequence[TrackSample]],
) -> int:
    def key(track_id: int) -> tuple[float, int, int]:
        observed = observed_by_track.get(track_id, ())
        first_timestamp = (
            observed[0].timestamp
            if observed
            else float("inf")
        )
        # Prefer the identity that existed first. For exact ties, prefer the
        # one with more real support, then the smaller monotonic tracker ID.
        return (
            first_timestamp,
            -len(observed),
            track_id,
        )

    return min(component, key=key)


def _merge_component(
    canonical_id: int,
    component: Sequence[int],
    trajectories: Mapping[int, tuple[TrackSample, ...]],
) -> tuple[TrackSample, ...]:
    by_frame: dict[int, tuple[int, TrackSample]] = {}

    ordered_sources = [canonical_id] + [
        track_id
        for track_id in sorted(component)
        if track_id != canonical_id
    ]

    for source_id in ordered_sources:
        for sample in trajectories[source_id]:
            rewritten = replace(
                sample,
                track_id=canonical_id,
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
                canonical_id=canonical_id,
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
        return new_priority > existing_priority

    if new.position_kind is PositionKind.OBSERVED:
        new_conf = new.confidence
        existing_conf = existing.confidence

        if new_conf is not None and existing_conf is not None:
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


def _sample_priority(sample: TrackSample) -> int:
    return {
        PositionKind.OBSERVED: 3,
        PositionKind.INTERPOLATED: 2,
        PositionKind.PREDICTED: 1,
    }[sample.position_kind]


def _validated_trajectories(
    trajectories: Mapping[int, Sequence[TrackSample]],
) -> dict[int, tuple[TrackSample, ...]]:
    result: dict[int, tuple[TrackSample, ...]] = {}

    for track_id, samples in sorted(trajectories.items()):
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

        if ordered:
            result[track_id] = ordered

    return result


def _find(
    parent: dict[int, int],
    track_id: int,
) -> int:
    root = track_id
    while parent[root] != root:
        root = parent[root]

    while parent[track_id] != track_id:
        next_id = parent[track_id]
        parent[track_id] = root
        track_id = next_id

    return root


def _union(
    parent: dict[int, int],
    members: dict[int, set[int]],
    left_root: int,
    right_root: int,
) -> int:
    if left_root == right_root:
        return left_root

    # Root IDs here are implementation details only; final canonical identity
    # is chosen later from temporal evidence, not from union-find root order.
    if len(members[left_root]) < len(members[right_root]):
        left_root, right_root = right_root, left_root

    parent[right_root] = left_root
    members[left_root].update(members[right_root])
    del members[right_root]
    return left_root
