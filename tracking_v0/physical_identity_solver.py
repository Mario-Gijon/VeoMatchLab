"""V7 offline physical-identity solver for football tracking.

The solver reconstructs exactly N physical people from identity-consistent
tracker segments.  It intentionally has *no referee class* and does not force
11+11+1 composition.  TEAM_A / TEAM_B evidence is used only as a physical
compatibility constraint: a single person cannot belong confidently to both
teams.

The clip is solved offline with full past+future context.  Several seconds of
missing observations are allowed, but no position is fabricated inside a gap.
Long simultaneous/separated segments cannot be a temporal continuation of the
same physical person.

A future DUPLICATE relation type is represented explicitly in the API so that
simultaneous duplicate tracker IDs can later be fused without consuming two
physical identities.  V1 only audits duplicate hypotheses; it does not yet use
them as merge edges in the MILP.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .team_similarity import RelativeTeamLabel


_EPS = 1e-12


class PhysicalRelationKind(str, Enum):
    CONTINUATION = "continuation"
    DUPLICATE = "duplicate"


class PhysicalSegmentQuality(str, Enum):
    CLEAN = "clean"
    AMBIGUOUS = "ambiguous"
    CONTAMINATED = "contaminated"


@dataclass(frozen=True)
class PhysicalObservation:
    frame_index: int
    timestamp: float
    x: float
    z: float
    measurement_id: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not all(math.isfinite(value) for value in (self.timestamp, self.x, self.z)):
            raise ValueError("observation values must be finite")


@dataclass(frozen=True)
class PhysicalTrackSegment:
    segment_id: int
    raw_track_id: int
    segment_index: int
    observations: tuple[PhysicalObservation, ...]

    team_label: RelativeTeamLabel = RelativeTeamLabel.UNKNOWN
    team_confidence: float = 0.0
    reid_prototype: np.ndarray | None = None

    identity_quality: PhysicalSegmentQuality = PhysicalSegmentQuality.CLEAN
    identity_quality_score: float = 1.0

    inside_observed_count: int = 0
    continuation_observed_count: int = 0

    def __post_init__(self) -> None:
        if self.segment_id < 1:
            raise ValueError("segment_id must be >= 1")
        if self.raw_track_id < 1:
            raise ValueError("raw_track_id must be >= 1")
        if self.segment_index < 0:
            raise ValueError("segment_index must be >= 0")
        if not self.observations:
            raise ValueError("segment observations must not be empty")
        if not 0.0 <= float(self.team_confidence) <= 1.0:
            raise ValueError("team_confidence must be in [0, 1]")
        if not 0.0 <= float(self.identity_quality_score) <= 1.0:
            raise ValueError("identity_quality_score must be in [0, 1]")
        if self.inside_observed_count < 0 or self.continuation_observed_count < 0:
            raise ValueError("observation counts must be >= 0")

        previous = None
        for row in self.observations:
            key = (row.timestamp, row.frame_index)
            if previous is not None and key < previous:
                raise ValueError("observations must be sorted")
            previous = key

        if self.reid_prototype is not None:
            vector = np.asarray(self.reid_prototype, dtype=np.float32)
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError("reid_prototype must be a finite 1D vector")
            norm = float(np.linalg.norm(vector))
            if norm <= _EPS:
                raise ValueError("reid_prototype norm must be > 0")
            object.__setattr__(self, "reid_prototype", vector / norm)

    @property
    def key(self) -> str:
        return f"{self.raw_track_id}:{self.segment_index}"

    @property
    def start_frame(self) -> int:
        return self.observations[0].frame_index

    @property
    def end_frame(self) -> int:
        return self.observations[-1].frame_index

    @property
    def start_timestamp(self) -> float:
        return self.observations[0].timestamp

    @property
    def end_timestamp(self) -> float:
        return self.observations[-1].timestamp

    @property
    def observed_count(self) -> int:
        return len(self.observations)

    @property
    def observed_span_s(self) -> float:
        return max(0.0, self.end_timestamp - self.start_timestamp)

    @property
    def first_position(self) -> np.ndarray:
        row = self.observations[0]
        return np.asarray([row.x, row.z], dtype=np.float64)

    @property
    def last_position(self) -> np.ndarray:
        row = self.observations[-1]
        return np.asarray([row.x, row.z], dtype=np.float64)

    @property
    def established_inside(self) -> bool:
        return self.inside_observed_count > 0


@dataclass(frozen=True)
class PhysicalIdentitySolverConfig:
    roster_size: int = 23

    hard_team_confidence: float = 0.78

    # Several seconds without observation are normal in football footage.
    max_recovery_gap_s: float = 15.0
    base_recovery_distance_m: float = 3.0
    max_player_speed_m_s: float = 12.5

    # Only very short colocated overlap can be interpreted as tracker handover.
    max_continuation_overlap_s: float = 1.0
    max_short_overlap_distance_m: float = 2.25

    # Transition evidence.
    continuity_weight: float = 0.60
    reid_weight: float = 0.28
    team_weight: float = 0.12
    min_continuation_score: float = 0.25

    # Coverage drives the offline explanation.  Long strict-inside segments are
    # expensive to leave unexplained; zero-corridor outsiders have no utility.
    coverage_scale: float = 55.0
    fragment_penalty: float = 0.10
    transition_weight: float = 0.42
    min_reliable_observations: int = 8

    # Duplicate hypotheses are diagnostic in V1, not solver merge edges.
    duplicate_min_shared_frames: int = 8
    duplicate_max_median_distance_m: float = 12.0
    duplicate_min_movement_cosine: float = 0.80

    mip_time_limit_s: float = 30.0
    mip_relative_gap: float = 0.0

    def __post_init__(self) -> None:
        if self.roster_size < 1:
            raise ValueError("roster_size must be >= 1")
        if not 0.0 <= self.hard_team_confidence <= 1.0:
            raise ValueError("hard_team_confidence must be in [0, 1]")
        if self.max_recovery_gap_s <= 0.0:
            raise ValueError("max_recovery_gap_s must be > 0")
        if self.base_recovery_distance_m <= 0.0:
            raise ValueError("base_recovery_distance_m must be > 0")
        if self.max_player_speed_m_s <= 0.0:
            raise ValueError("max_player_speed_m_s must be > 0")
        if self.max_continuation_overlap_s < 0.0:
            raise ValueError("max_continuation_overlap_s must be >= 0")
        if self.max_short_overlap_distance_m <= 0.0:
            raise ValueError("max_short_overlap_distance_m must be > 0")

        weights = (self.continuity_weight, self.reid_weight, self.team_weight)
        if any(weight < 0.0 for weight in weights):
            raise ValueError("transition weights must be >= 0")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("transition weights must sum to 1")
        if not 0.0 <= self.min_continuation_score <= 1.0:
            raise ValueError("min_continuation_score must be in [0, 1]")
        if self.coverage_scale <= 0.0:
            raise ValueError("coverage_scale must be > 0")
        if self.fragment_penalty < 0.0:
            raise ValueError("fragment_penalty must be >= 0")
        if self.transition_weight < 0.0:
            raise ValueError("transition_weight must be >= 0")
        if self.min_reliable_observations < 1:
            raise ValueError("min_reliable_observations must be >= 1")
        if self.duplicate_min_shared_frames < 2:
            raise ValueError("duplicate_min_shared_frames must be >= 2")
        if self.duplicate_max_median_distance_m <= 0.0:
            raise ValueError("duplicate_max_median_distance_m must be > 0")
        if not -1.0 <= self.duplicate_min_movement_cosine <= 1.0:
            raise ValueError("duplicate_min_movement_cosine must be in [-1, 1]")
        if self.mip_time_limit_s <= 0.0:
            raise ValueError("mip_time_limit_s must be > 0")
        if not 0.0 <= self.mip_relative_gap <= 1.0:
            raise ValueError("mip_relative_gap must be in [0, 1]")


@dataclass(frozen=True)
class PhysicalRelationEvidence:
    left_segment_id: int
    right_segment_id: int
    kind: PhysicalRelationKind

    gap_s: float
    overlap_s: float
    endpoint_distance_m: float
    overlap_median_distance_m: float | None

    continuity_score: float
    reid_score: float
    team_score: float
    total_score: float

    shared_observed_frames: int = 0
    movement_cosine_median: float | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["kind"] = self.kind.value
        return payload


@dataclass(frozen=True)
class PhysicalIdentity:
    identity_id: int
    member_segment_ids: tuple[int, ...]
    member_keys: tuple[str, ...]
    raw_track_ids: tuple[int, ...]
    team_label: RelativeTeamLabel

    def to_dict(self) -> dict[str, object]:
        return {
            "identity_id": self.identity_id,
            "member_segment_ids": list(self.member_segment_ids),
            "member_keys": list(self.member_keys),
            "raw_track_ids": list(self.raw_track_ids),
            "team_label": self.team_label.value,
        }


@dataclass(frozen=True)
class PhysicalIdentityResult:
    identities: tuple[PhysicalIdentity, ...]
    segment_to_identity: Mapping[int, int]
    unassigned_segment_ids: tuple[int, ...]

    continuation_candidates: tuple[PhysicalRelationEvidence, ...]
    selected_continuations: tuple[PhysicalRelationEvidence, ...]
    duplicate_candidates: tuple[PhysicalRelationEvidence, ...]

    solver_status: str
    objective: float
    selected_segment_count: int
    candidate_continuation_count: int
    hard_team_prunes: int
    temporal_prunes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "identities": [identity.to_dict() for identity in self.identities],
            "segment_to_identity": {
                str(segment_id): identity_id
                for segment_id, identity_id in sorted(self.segment_to_identity.items())
            },
            "unassigned_segment_ids": list(self.unassigned_segment_ids),
            "continuation_candidates": [row.to_dict() for row in self.continuation_candidates],
            "selected_continuations": [row.to_dict() for row in self.selected_continuations],
            "duplicate_candidates": [row.to_dict() for row in self.duplicate_candidates],
            "solver_status": self.solver_status,
            "objective": self.objective,
            "selected_segment_count": self.selected_segment_count,
            "candidate_continuation_count": self.candidate_continuation_count,
            "hard_team_prunes": self.hard_team_prunes,
            "temporal_prunes": self.temporal_prunes,
        }


def solve_physical_identities(
    segments: Sequence[PhysicalTrackSegment],
    *,
    config: PhysicalIdentitySolverConfig | None = None,
) -> PhysicalIdentityResult:
    cfg = config or PhysicalIdentitySolverConfig()

    segment_by_id = {segment.segment_id: segment for segment in segments}
    if len(segment_by_id) != len(segments):
        raise ValueError("segment_id values must be unique")
    if len(segments) < cfg.roster_size:
        raise ValueError("not enough segments to form requested physical identities")

    (
        continuation_candidates,
        duplicate_candidates,
        hard_team_prunes,
        temporal_prunes,
    ) = _build_relations(segments, config=cfg)

    solution = _solve_milp(
        segments,
        continuation_candidates=continuation_candidates,
        config=cfg,
    )

    slot_paths: tuple[tuple[int, ...], ...] = solution["slot_paths"]  # type: ignore[assignment]
    selected_edges: tuple[tuple[int, int, int], ...] = solution["selected_edges"]  # type: ignore[assignment]

    ordered_paths = sorted(
        enumerate(slot_paths),
        key=lambda item: (
            min(segment_by_id[segment_id].start_timestamp for segment_id in item[1]),
            min(item[1]),
        ),
    )

    slot_to_identity: dict[int, int] = {}
    segment_to_identity: dict[int, int] = {}
    identities: list[PhysicalIdentity] = []

    for identity_id, (slot_id, member_ids) in enumerate(ordered_paths, start=1):
        slot_to_identity[slot_id] = identity_id
        ordered_members = tuple(
            sorted(
                member_ids,
                key=lambda segment_id: (
                    segment_by_id[segment_id].start_timestamp,
                    segment_by_id[segment_id].end_timestamp,
                    segment_id,
                ),
            )
        )
        for segment_id in ordered_members:
            segment_to_identity[segment_id] = identity_id

        member_segments = [segment_by_id[segment_id] for segment_id in ordered_members]
        identities.append(
            PhysicalIdentity(
                identity_id=identity_id,
                member_segment_ids=ordered_members,
                member_keys=tuple(segment.key for segment in member_segments),
                raw_track_ids=tuple(sorted({segment.raw_track_id for segment in member_segments})),
                team_label=_identity_team_label(member_segments, config=cfg),
            )
        )

    relation_by_pair = {
        (relation.left_segment_id, relation.right_segment_id): relation
        for relation in continuation_candidates
    }
    selected_relations = tuple(
        relation_by_pair[(left, right)]
        for slot_id, left, right in sorted(
            selected_edges,
            key=lambda item: (
                slot_to_identity[item[0]],
                segment_by_id[item[1]].start_timestamp,
                segment_by_id[item[2]].start_timestamp,
            ),
        )
    )

    unassigned = tuple(
        sorted(
            segment.segment_id
            for segment in segments
            if segment.segment_id not in segment_to_identity
        )
    )

    return PhysicalIdentityResult(
        identities=tuple(identities),
        segment_to_identity=segment_to_identity,
        unassigned_segment_ids=unassigned,
        continuation_candidates=tuple(continuation_candidates),
        selected_continuations=selected_relations,
        duplicate_candidates=tuple(duplicate_candidates),
        solver_status=str(solution["status"]),
        objective=float(solution["objective"]),
        selected_segment_count=len(segment_to_identity),
        candidate_continuation_count=len(continuation_candidates),
        hard_team_prunes=hard_team_prunes,
        temporal_prunes=temporal_prunes,
    )


def _hard_team(segment: PhysicalTrackSegment, *, config: PhysicalIdentitySolverConfig) -> RelativeTeamLabel:
    if segment.identity_quality is not PhysicalSegmentQuality.CLEAN:
        return RelativeTeamLabel.UNKNOWN
    if (
        segment.team_label in {RelativeTeamLabel.TEAM_A, RelativeTeamLabel.TEAM_B}
        and segment.team_confidence >= config.hard_team_confidence
    ):
        return segment.team_label
    return RelativeTeamLabel.UNKNOWN


def _identity_team_label(
    members: Sequence[PhysicalTrackSegment],
    *,
    config: PhysicalIdentitySolverConfig,
) -> RelativeTeamLabel:
    weights = {
        RelativeTeamLabel.TEAM_A: 0.0,
        RelativeTeamLabel.TEAM_B: 0.0,
    }
    for segment in members:
        label = _hard_team(segment, config=config)
        if label not in weights:
            continue
        weights[label] += min(180.0, float(segment.observed_count)) * max(0.1, segment.team_confidence)

    total = sum(weights.values())
    if total <= _EPS:
        return RelativeTeamLabel.UNKNOWN
    winner = max(weights, key=weights.get)
    if weights[winner] / total < 0.60:
        return RelativeTeamLabel.UNKNOWN
    return winner


def _build_relations(
    segments: Sequence[PhysicalTrackSegment],
    *,
    config: PhysicalIdentitySolverConfig,
) -> tuple[list[PhysicalRelationEvidence], list[PhysicalRelationEvidence], int, int]:
    ordered = sorted(
        segments,
        key=lambda segment: (
            segment.start_timestamp,
            segment.end_timestamp,
            segment.segment_id,
        ),
    )

    continuation: list[PhysicalRelationEvidence] = []
    duplicates: list[PhysicalRelationEvidence] = []
    hard_team_prunes = 0
    temporal_prunes = 0

    for i, left in enumerate(ordered):
        if left.identity_quality is PhysicalSegmentQuality.CONTAMINATED:
            continue

        left_team = _hard_team(left, config=config)

        for right in ordered[i + 1:]:
            if right.identity_quality is PhysicalSegmentQuality.CONTAMINATED:
                continue

            # AMBIGUOUS evidence may attach to the tail of an already clean
            # identity, but it may never bridge two identities.
            if left.identity_quality is PhysicalSegmentQuality.AMBIGUOUS:
                continue
            # If the segmentation stage explicitly split one raw tracker ID,
            # do not immediately glue those pieces back together here.  They
            # represent a detected internal identity discontinuity.
            if left.raw_track_id == right.raw_track_id:
                continue

            right_team = _hard_team(right, config=config)

            if (
                left_team is not RelativeTeamLabel.UNKNOWN
                and right_team is not RelativeTeamLabel.UNKNOWN
                and left_team is not right_team
            ):
                hard_team_prunes += 1
                continue

            duplicate = _duplicate_hypothesis(left, right, config=config)
            if duplicate is not None:
                duplicates.append(duplicate)

            continuity = _continuity_evidence(left, right, config=config)
            if continuity is None:
                temporal_prunes += 1
                continue

            (
                gap_s,
                overlap_s,
                endpoint_distance,
                overlap_median_distance,
                continuity_score,
                reason,
            ) = continuity

            reid_score = _reid_similarity(left, right)
            if left_team is not RelativeTeamLabel.UNKNOWN and left_team is right_team:
                team_score = 1.0
            else:
                team_score = 0.5

            total_score = (
                config.continuity_weight * continuity_score
                + config.reid_weight * reid_score
                + config.team_weight * team_score
            )
            if total_score < config.min_continuation_score:
                continue

            continuation.append(
                PhysicalRelationEvidence(
                    left_segment_id=left.segment_id,
                    right_segment_id=right.segment_id,
                    kind=PhysicalRelationKind.CONTINUATION,
                    gap_s=float(gap_s),
                    overlap_s=float(overlap_s),
                    endpoint_distance_m=float(endpoint_distance),
                    overlap_median_distance_m=overlap_median_distance,
                    continuity_score=float(continuity_score),
                    reid_score=float(reid_score),
                    team_score=float(team_score),
                    total_score=float(total_score),
                    reason=reason,
                )
            )

    return continuation, duplicates, hard_team_prunes, temporal_prunes


def _continuity_evidence(
    left: PhysicalTrackSegment,
    right: PhysicalTrackSegment,
    *,
    config: PhysicalIdentitySolverConfig,
) -> tuple[float, float, float, float | None, float, str] | None:
    gap_s = right.start_timestamp - left.end_timestamp
    overlap_s = max(
        0.0,
        min(left.end_timestamp, right.end_timestamp)
        - max(left.start_timestamp, right.start_timestamp),
    )
    endpoint_distance = float(np.linalg.norm(right.first_position - left.last_position))

    if gap_s < 0.0:
        if overlap_s > config.max_continuation_overlap_s + 1e-9:
            return None
        median_distance = _simultaneous_median_distance(left, right)
        if median_distance is None or median_distance > config.max_short_overlap_distance_m:
            return None
        closeness = max(0.0, 1.0 - median_distance / config.max_short_overlap_distance_m)
        score = 0.78 + 0.22 * closeness
        return (
            gap_s,
            overlap_s,
            endpoint_distance,
            float(median_distance),
            float(score),
            "short_colocated_overlap",
        )

    if gap_s > config.max_recovery_gap_s:
        return None

    reachable = config.base_recovery_distance_m + config.max_player_speed_m_s * gap_s
    if endpoint_distance > reachable + 1e-9:
        return None

    reachability = math.exp(
        -endpoint_distance
        / max(
            config.base_recovery_distance_m + 0.55 * config.max_player_speed_m_s * gap_s,
            _EPS,
        )
    )
    gap_score = math.exp(-gap_s / max(config.max_recovery_gap_s / 2.5, _EPS))
    score = 0.72 * reachability + 0.28 * gap_score

    return (
        gap_s,
        0.0,
        endpoint_distance,
        None,
        float(min(1.0, max(0.0, score))),
        "physically_reachable_gap",
    )


def _reid_similarity(left: PhysicalTrackSegment, right: PhysicalTrackSegment) -> float:
    if left.reid_prototype is None or right.reid_prototype is None:
        return 0.5
    cosine = float(np.dot(left.reid_prototype, right.reid_prototype))
    return float(min(1.0, max(0.0, cosine)))


def _simultaneous_median_distance(
    left: PhysicalTrackSegment,
    right: PhysicalTrackSegment,
) -> float | None:
    left_by_frame = {row.frame_index: row for row in left.observations}
    right_by_frame = {row.frame_index: row for row in right.observations}
    shared = sorted(set(left_by_frame) & set(right_by_frame))
    if not shared:
        return None
    distances = [
        math.hypot(
            left_by_frame[frame].x - right_by_frame[frame].x,
            left_by_frame[frame].z - right_by_frame[frame].z,
        )
        for frame in shared
    ]
    return float(np.median(np.asarray(distances, dtype=np.float64)))


def _duplicate_hypothesis(
    left: PhysicalTrackSegment,
    right: PhysicalTrackSegment,
    *,
    config: PhysicalIdentitySolverConfig,
) -> PhysicalRelationEvidence | None:
    left_team = _hard_team(left, config=config)
    right_team = _hard_team(right, config=config)
    if (
        left_team is not RelativeTeamLabel.UNKNOWN
        and right_team is not RelativeTeamLabel.UNKNOWN
        and left_team is not right_team
    ):
        return None

    left_by_frame = {row.frame_index: row for row in left.observations}
    right_by_frame = {row.frame_index: row for row in right.observations}
    shared = sorted(set(left_by_frame) & set(right_by_frame))
    if len(shared) < config.duplicate_min_shared_frames:
        return None

    distances = [
        math.hypot(
            left_by_frame[frame].x - right_by_frame[frame].x,
            left_by_frame[frame].z - right_by_frame[frame].z,
        )
        for frame in shared
    ]
    median_distance = float(np.median(np.asarray(distances, dtype=np.float64)))
    if median_distance > config.duplicate_max_median_distance_m:
        return None

    movement_cosines: list[float] = []
    previous = None
    for frame in shared:
        if previous is None:
            previous = frame
            continue
        if frame - previous > 3:
            previous = frame
            continue

        l0 = left_by_frame[previous]
        l1 = left_by_frame[frame]
        r0 = right_by_frame[previous]
        r1 = right_by_frame[frame]

        dl = np.asarray([l1.x - l0.x, l1.z - l0.z], dtype=np.float64)
        dr = np.asarray([r1.x - r0.x, r1.z - r0.z], dtype=np.float64)
        nl = float(np.linalg.norm(dl))
        nr = float(np.linalg.norm(dr))
        if nl > 0.05 and nr > 0.05:
            movement_cosines.append(float(np.dot(dl, dr) / (nl * nr)))
        previous = frame

    if not movement_cosines:
        return None
    movement_median = float(np.median(np.asarray(movement_cosines, dtype=np.float64)))
    if movement_median < config.duplicate_min_movement_cosine:
        return None

    reid_score = _reid_similarity(left, right)
    distance_score = max(0.0, 1.0 - median_distance / config.duplicate_max_median_distance_m)
    motion_score = (movement_median + 1.0) / 2.0
    total = 0.40 * distance_score + 0.35 * motion_score + 0.25 * reid_score

    return PhysicalRelationEvidence(
        left_segment_id=left.segment_id,
        right_segment_id=right.segment_id,
        kind=PhysicalRelationKind.DUPLICATE,
        gap_s=right.start_timestamp - left.end_timestamp,
        overlap_s=max(
            0.0,
            min(left.end_timestamp, right.end_timestamp)
            - max(left.start_timestamp, right.start_timestamp),
        ),
        endpoint_distance_m=float(np.linalg.norm(right.first_position - left.last_position)),
        overlap_median_distance_m=median_distance,
        continuity_score=motion_score,
        reid_score=reid_score,
        team_score=(1.0 if left_team is right_team and left_team is not RelativeTeamLabel.UNKNOWN else 0.5),
        total_score=float(total),
        shared_observed_frames=len(shared),
        movement_cosine_median=movement_median,
        reason="diagnostic_duplicate_hypothesis",
    )


def _node_utility(segment: PhysicalTrackSegment, *, config: PhysicalIdentitySolverConfig) -> float:
    if segment.inside_observed_count <= 0 and segment.continuation_observed_count <= 0:
        return -1e6
    if segment.identity_quality is PhysicalSegmentQuality.CONTAMINATED:
        return -1e6

    inside = float(segment.inside_observed_count)
    outer_only = float(
        max(0, segment.continuation_observed_count - segment.inside_observed_count)
    )
    coverage = (inside + 0.20 * outer_only) / config.coverage_scale
    reliability = min(
        1.0,
        segment.observed_count / float(config.min_reliable_observations),
    )

    quality_weight = (
        1.0
        if segment.identity_quality is PhysicalSegmentQuality.CLEAN
        else 0.25
    )
    quality_weight *= max(0.25, float(segment.identity_quality_score))

    return (
        reliability
        * coverage
        * quality_weight
        - config.fragment_penalty
    )


def _solve_milp(
    segments: Sequence[PhysicalTrackSegment],
    *,
    continuation_candidates: Sequence[PhysicalRelationEvidence],
    config: PhysicalIdentitySolverConfig,
) -> dict[str, object]:
    """Prize-collecting DAG path cover with exactly ``roster_size`` paths.

    V1 copied every segment/edge once per one of the 23 anonymous roster slots.
    That creates a very large amount of symmetry: swapping slot 3 with slot 17
    gives the exact same football solution, yet the MILP still has to explore
    those equivalent assignments.

    V2 removes the anonymous slot dimension entirely.  We solve one global DAG
    path-cover problem:

      selected segment x_i
      selected continuation y_ij
      source_i / sink_i

    with exactly ``roster_size`` sources and sinks.  Because continuation edges
    are acyclic, the selected graph decomposes into exactly that many physical
    identity paths.  The paths are numbered only AFTER optimization.

    TEAM_A / TEAM_B is propagated along selected edges with per-segment state,
    so a confident A segment cannot reach a confident B segment through an
    ambiguous middle segment.  No referee class exists here.
    """
    ordered = tuple(
        sorted(
            segments,
            key=lambda segment: segment.segment_id,
        )
    )
    segment_by_id = {
        segment.segment_id: segment
        for segment in ordered
    }
    segment_ids = tuple(
        segment.segment_id
        for segment in ordered
    )

    costs: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []

    def add_var(
        cost: float,
        *,
        lb: float = 0.0,
        ub: float = 1.0,
    ) -> int:
        index = len(costs)
        costs.append(float(cost))
        lower.append(float(lb))
        upper.append(float(ub))
        integrality.append(1)
        return index

    x: dict[int, int] = {}
    source: dict[int, int] = {}
    sink: dict[int, int] = {}
    team_a_state: dict[int, int] = {}
    team_b_state: dict[int, int] = {}

    for segment_id in segment_ids:
        segment = segment_by_id[segment_id]

        selectable = (
            segment.identity_quality
            is not PhysicalSegmentQuality.CONTAMINATED
            and not (
                segment.inside_observed_count <= 0
                and segment.continuation_observed_count <= 0
            )
        )

        x_ub = 1.0 if selectable else 0.0

        # Only CLEAN strict-inside evidence may establish one of the 23
        # physical people. AMBIGUOUS evidence can be a terminal attachment,
        # never a source/bridge. CONTAMINATED evidence is not selectable.
        source_ub = (
            1.0
            if (
                selectable
                and segment.inside_observed_count > 0
                and segment.identity_quality
                is PhysicalSegmentQuality.CLEAN
            )
            else 0.0
        )

        x[segment_id] = add_var(
            -_node_utility(
                segment,
                config=config,
            ),
            ub=x_ub,
        )
        source[segment_id] = add_var(
            0.0,
            ub=source_ub,
        )
        sink[segment_id] = add_var(
            0.0,
            ub=x_ub,
        )
        team_a_state[segment_id] = add_var(
            0.0,
            ub=x_ub,
        )
        team_b_state[segment_id] = add_var(
            0.0,
            ub=x_ub,
        )

    edge_by_pair = {
        (
            edge.left_segment_id,
            edge.right_segment_id,
        ): edge
        for edge in continuation_candidates
    }

    y: dict[tuple[int, int], int] = {}

    for (
        left_id,
        right_id,
    ), edge in edge_by_pair.items():
        if (
            upper[x[left_id]] <= 0.0
            or upper[x[right_id]] <= 0.0
        ):
            continue
        if (
            segment_by_id[left_id].identity_quality
            is PhysicalSegmentQuality.AMBIGUOUS
        ):
            continue

        reliability = min(
            1.0,
            segment_by_id[
                right_id
            ].observed_count
            / float(
                config.min_reliable_observations
            ),
        )

        y[
            (
                left_id,
                right_id,
            )
        ] = add_var(
            -(
                config.transition_weight
                * edge.total_score
                * reliability
            )
        )

    row_i: list[int] = []
    col_i: list[int] = []
    values: list[float] = []
    lbs: list[float] = []
    ubs: list[float] = []
    row_count = 0

    def add_constraint(
        terms: Iterable[
            tuple[int, float]
        ],
        *,
        lb: float,
        ub: float,
    ) -> None:
        nonlocal row_count

        for variable, coefficient in terms:
            row_i.append(row_count)
            col_i.append(variable)
            values.append(float(coefficient))

        lbs.append(float(lb))
        ubs.append(float(ub))
        row_count += 1

    incoming: dict[int, list[int]] = {
        segment_id: []
        for segment_id in segment_ids
    }
    outgoing: dict[int, list[int]] = {
        segment_id: []
        for segment_id in segment_ids
    }

    for (
        left_id,
        right_id,
    ), variable in y.items():
        outgoing[
            left_id
        ].append(variable)
        incoming[
            right_id
        ].append(variable)

    # Exactly 23 physical-person paths (or cfg.roster_size generally).
    add_constraint(
        (
            (
                source[segment_id],
                1.0,
            )
            for segment_id in segment_ids
        ),
        lb=float(config.roster_size),
        ub=float(config.roster_size),
    )
    add_constraint(
        (
            (
                sink[segment_id],
                1.0,
            )
            for segment_id in segment_ids
        ),
        lb=float(config.roster_size),
        ub=float(config.roster_size),
    )

    for segment_id in segment_ids:
        # source + one incoming continuation == selected
        in_terms = [
            (
                variable,
                1.0,
            )
            for variable in incoming[
                segment_id
            ]
        ]
        in_terms.extend(
            [
                (
                    source[
                        segment_id
                    ],
                    1.0,
                ),
                (
                    x[
                        segment_id
                    ],
                    -1.0,
                ),
            ]
        )
        add_constraint(
            in_terms,
            lb=0.0,
            ub=0.0,
        )

        # sink + one outgoing continuation == selected
        out_terms = [
            (
                variable,
                1.0,
            )
            for variable in outgoing[
                segment_id
            ]
        ]
        out_terms.extend(
            [
                (
                    sink[
                        segment_id
                    ],
                    1.0,
                ),
                (
                    x[
                        segment_id
                    ],
                    -1.0,
                ),
            ]
        )
        add_constraint(
            out_terms,
            lb=0.0,
            ub=0.0,
        )

        # Team states exist only for selected segments and are mutually
        # exclusive. Unknown/referee-like appearance may keep both at zero.
        add_constraint(
            (
                (
                    team_a_state[
                        segment_id
                    ],
                    1.0,
                ),
                (
                    team_b_state[
                        segment_id
                    ],
                    1.0,
                ),
                (
                    x[
                        segment_id
                    ],
                    -1.0,
                ),
            ),
            lb=-float("inf"),
            ub=0.0,
        )

        hard_team = _hard_team(
            segment_by_id[
                segment_id
            ],
            config=config,
        )

        if hard_team is RelativeTeamLabel.TEAM_A:
            # selected => TEAM_A state = 1
            add_constraint(
                (
                    (
                        team_a_state[
                            segment_id
                        ],
                        1.0,
                    ),
                    (
                        x[
                            segment_id
                        ],
                        -1.0,
                    ),
                ),
                lb=0.0,
                ub=0.0,
            )
        elif hard_team is RelativeTeamLabel.TEAM_B:
            add_constraint(
                (
                    (
                        team_b_state[
                            segment_id
                        ],
                        1.0,
                    ),
                    (
                        x[
                            segment_id
                        ],
                        -1.0,
                    ),
                ),
                lb=0.0,
                ub=0.0,
            )

    # If an edge is selected, both endpoints must carry the same propagated
    # team state. This prevents A -> UNKNOWN -> B from becoming one person.
    #
    # For binary y:
    #   state_left - state_right <= 1-y
    #   state_right - state_left <= 1-y
    for (
        left_id,
        right_id,
    ), edge_var in y.items():
        for state in (
            team_a_state,
            team_b_state,
        ):
            add_constraint(
                (
                    (
                        state[
                            left_id
                        ],
                        1.0,
                    ),
                    (
                        state[
                            right_id
                        ],
                        -1.0,
                    ),
                    (
                        edge_var,
                        1.0,
                    ),
                ),
                lb=-float("inf"),
                ub=1.0,
            )
            add_constraint(
                (
                    (
                        state[
                            right_id
                        ],
                        1.0,
                    ),
                    (
                        state[
                            left_id
                        ],
                        -1.0,
                    ),
                    (
                        edge_var,
                        1.0,
                    ),
                ),
                lb=-float("inf"),
                ub=1.0,
            )

    matrix = coo_matrix(
        (
            np.asarray(
                values,
                dtype=np.float64,
            ),
            (
                np.asarray(
                    row_i,
                    dtype=np.int32,
                ),
                np.asarray(
                    col_i,
                    dtype=np.int32,
                ),
            ),
        ),
        shape=(
            row_count,
            len(costs),
        ),
    ).tocsr()

    result = milp(
        c=np.asarray(
            costs,
            dtype=np.float64,
        ),
        integrality=np.asarray(
            integrality,
            dtype=np.int8,
        ),
        bounds=Bounds(
            np.asarray(
                lower,
                dtype=np.float64,
            ),
            np.asarray(
                upper,
                dtype=np.float64,
            ),
        ),
        constraints=LinearConstraint(
            matrix,
            np.asarray(
                lbs,
                dtype=np.float64,
            ),
            np.asarray(
                ubs,
                dtype=np.float64,
            ),
        ),
        options={
            "time_limit": float(
                config.mip_time_limit_s
            ),
            "mip_rel_gap": float(
                config.mip_relative_gap
            ),
            "presolve": True,
        },
    )

    if result.x is None:
        raise RuntimeError(
            "V7 physical identity MILP found no feasible solution: "
            f"status={result.status}, message={result.message}"
        )

    vector = np.asarray(
        result.x,
        dtype=np.float64,
    )

    selected_ids = {
        segment_id
        for segment_id in segment_ids
        if vector[
            x[
                segment_id
            ]
        ] > 0.5
    }

    selected_pair_edges = {
        (
            left_id,
            right_id,
        )
        for (
            left_id,
            right_id,
        ), variable in y.items()
        if vector[
            variable
        ] > 0.5
    }

    selected_sources = [
        segment_id
        for segment_id in segment_ids
        if vector[
            source[
                segment_id
            ]
        ] > 0.5
    ]

    if len(selected_sources) != config.roster_size:
        raise RuntimeError(
            "V7 path-cover source count mismatch: "
            f"{len(selected_sources)} != {config.roster_size}"
        )

    selected_outgoing: dict[
        int,
        int,
    ] = {}

    for (
        left_id,
        right_id,
    ) in selected_pair_edges:
        if left_id in selected_outgoing:
            raise RuntimeError(
                f"segment {left_id} has multiple selected outgoing edges"
            )
        selected_outgoing[
            left_id
        ] = right_id

    # Number the anonymous physical paths only after optimization.
    selected_sources.sort(
        key=lambda segment_id: (
            segment_by_id[
                segment_id
            ].start_timestamp,
            segment_by_id[
                segment_id
            ].start_frame,
            segment_id,
        )
    )

    slot_paths: list[
        tuple[int, ...]
    ] = []
    selected_edges: list[
        tuple[int, int, int]
    ] = []
    visited: set[int] = set()

    for synthetic_slot, source_id in enumerate(
        selected_sources
    ):
        path: list[int] = []
        current = source_id

        while True:
            if current in visited:
                raise RuntimeError(
                    "V7 selected path cover contains a cycle or shared node"
                )

            visited.add(current)
            path.append(current)

            next_id = selected_outgoing.get(
                current
            )
            if next_id is None:
                break

            selected_edges.append(
                (
                    synthetic_slot,
                    current,
                    next_id,
                )
            )
            current = next_id

        slot_paths.append(
            tuple(path)
        )

    if visited != selected_ids:
        missing = sorted(
            selected_ids
            - visited
        )
        raise RuntimeError(
            "V7 selected graph contains nodes outside the reconstructed "
            f"{config.roster_size} paths: {missing}"
        )

    status = (
        "optimal"
        if result.status == 0
        else f"status_{result.status}_feasible"
    )

    objective = (
        -float(result.fun)
        if result.fun is not None
        else float("nan")
    )

    return {
        "status": status,
        "objective": objective,
        "slot_paths": tuple(
            slot_paths
        ),
        "selected_edges": tuple(
            selected_edges
        ),
    }

