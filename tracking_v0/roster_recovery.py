"""V6 full-match roster identity solver for football tracking.

The raw temporal tracker remains the default source of continuity, but final
identity recovery is solved OFFLINE and GLOBALLY across the clip. Appearance/
ReID is consulted only for plausible track-to-track recovery transitions.

This module deliberately does NOT:
- run ReID per frame,
- delete raw detections,
- create identities for outside-only people,
- force a linear motion model during recovery,
- rewrite production ``tracks.person``.

The intended football-specific hierarchy is:

    raw detector/association
        -> raw temporal tracking
        -> keep the raw ID while continuity exists
        -> build plausible track-to-track recovery transitions
        -> hard simultaneous/coexistence and trusted-team constraints
        -> global fixed-roster path solver
        -> canonical roster identity

The canonical roster is fixed (23 for the current no-substitution clip), while
raw track IDs may fragment and be reattached over time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
from itertools import combinations
import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import coo_matrix

from .team_similarity import RelativeTeamLabel


_EPS = 1e-12
_LARGE_COST = 1e9


class RosterClass(str, Enum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    REFEREE = "referee"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class RosterObservation:
    raw_track_id: int
    frame_index: int
    timestamp: float
    x: float
    z: float
    measurement_id: str | None = None
    confidence: float | None = None

    def __post_init__(self) -> None:
        if self.raw_track_id < 1:
            raise ValueError("raw_track_id must be >= 1")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not all(
            math.isfinite(value)
            for value in (
                self.timestamp,
                self.x,
                self.z,
            )
        ):
            raise ValueError("observation values must be finite")


@dataclass(frozen=True)
class RawTracklet:
    raw_track_id: int
    observations: tuple[RosterObservation, ...]
    team_label: RelativeTeamLabel = RelativeTeamLabel.UNKNOWN
    team_a_similarity: float | None = None
    team_b_similarity: float | None = None
    referee_score: float | None = None
    inside_observed_count: int = 0
    continuation_observed_count: int = 0

    def __post_init__(self) -> None:
        if self.raw_track_id < 1:
            raise ValueError("raw_track_id must be >= 1")
        if not self.observations:
            raise ValueError("tracklet observations must not be empty")

        previous = None
        for row in self.observations:
            if row.raw_track_id != self.raw_track_id:
                raise ValueError("tracklet contains another raw_track_id")
            key = (row.timestamp, row.frame_index)
            if previous is not None and key < previous:
                raise ValueError("tracklet observations must be sorted")
            previous = key

        for name in (
            "team_a_similarity",
            "team_b_similarity",
            "referee_score",
        ):
            value = getattr(self, name)
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if self.inside_observed_count < 0:
            raise ValueError("inside_observed_count must be >= 0")
        if self.continuation_observed_count < 0:
            raise ValueError("continuation_observed_count must be >= 0")

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
        return max(
            0.0,
            self.end_timestamp - self.start_timestamp,
        )

    @property
    def first_position(self) -> np.ndarray:
        row = self.observations[0]
        return np.asarray(
            [row.x, row.z],
            dtype=np.float64,
        )

    @property
    def last_position(self) -> np.ndarray:
        row = self.observations[-1]
        return np.asarray(
            [row.x, row.z],
            dtype=np.float64,
        )

    @property
    def established_inside(self) -> bool:
        return self.inside_observed_count > 0


@dataclass(frozen=True)
class RosterRecoveryConfig:
    roster_size: int = 23
    field_length_m: float = 105.0
    field_width_m: float = 68.0
    continuation_margin_m: float = 1.5

    # Seed frame.  Search the early clip for the first frame containing at
    # least this many strict-inside raw observations.
    seed_search_seconds: float = 8.0
    initial_seed_min_fraction: float = 0.75

    # V3 waits for the complete bootstrap window before opening missing roster
    # slots. Candidate roots are selected globally from strong strict-inside
    # tracklets rather than greedily at their first frame.
    bootstrap_root_pool_size: int = 14
    bootstrap_root_recovery_guard: float = 0.60
    bootstrap_composition_weight: float = 0.35

    # Exact football composition is optional in the reusable core. The current
    # no-substitution match audit enables 11 + 11 + 1 explicitly.
    team_a_count: int | None = None
    team_b_count: int | None = None
    referee_count: int | None = None

    # Recovery is event-driven.  Tracklets whose starts are close together are
    # solved in one Hungarian batch so two new fragments cannot claim the same
    # canonical identity.
    recovery_batch_seconds: float = 0.30

    # Physical reachability only; no directional/linear extrapolation is
    # required.  The player merely has to be able to travel the endpoint
    # distance during the gap.
    base_recovery_distance_m: float = 3.0
    max_player_speed_m_s: float = 12.5
    max_recovery_gap_s: float = 15.0

    # A raw tracker may duplicate one physical player for a short overlap.
    # Such overlap is admissible only when the two observed trajectories are
    # physically colocated.
    max_overlap_seconds: float = 1.0
    max_duplicate_overlap_distance_m: float = 2.25

    # Score weights apply only at a recovery event. Trusted visual-team
    # conflict is a hard veto; the globally assigned 11+11+1 roster class is a
    # SOFT prior only, so a weak referee/team bootstrap decision cannot become
    # circular "proof" later.
    continuity_weight: float = 0.55
    appearance_weight: float = 0.25
    team_weight: float = 0.10
    roster_prior_weight: float = 0.10

    # Only sufficiently confident shirt/shorts evidence is allowed to hard-veto
    # the opposite team. The semantic report's explicit label is always trusted;
    # otherwise the relative prototype similarity needs this margin.
    trusted_team_similarity_margin: float = 0.065
    canonical_team_vote_share: float = 0.67

    # Recovery is iterative because each accepted handover changes which of the
    # 23 canonical IDs are occupied/free for the remaining fragments.
    max_recovery_passes: int = 3

    min_recovery_score: float = 0.62
    min_outside_only_recovery_score: float = 0.70
    min_assignment_margin: float = 0.050

    # If exclusion leaves exactly one structurally available canonical ID, we
    # can accept with a lower score. ReID remains supporting evidence rather
    # than having to distinguish eleven near-identical kits.
    min_structural_candidate_score: float = 0.40
    min_unique_candidate_score: float = 0.50
    min_unique_outside_candidate_score: float = 0.62

    # V5 global path solver. Missing observations are not identity breaks:
    # several seconds of absence are allowed and penalized softly by continuity.
    global_edge_min_score: float = 0.28
    global_transition_weight: float = 0.42
    global_class_weight: float = 0.30
    global_fragment_penalty: float = 0.10
    global_coverage_scale: float = 55.0
    global_hard_team_margin: float = 0.18
    global_min_reliable_observations: int = 8
    global_mip_time_limit_s: float = 30.0
    global_mip_relative_gap: float = 0.0

    # An outside-only fragment may continue an established player, but it may
    # never establish a canonical identity by itself.
    allow_outside_only_recovery: bool = True

    def __post_init__(self) -> None:
        if self.roster_size < 1:
            raise ValueError("roster_size must be >= 1")
        if self.field_length_m <= 0.0:
            raise ValueError("field_length_m must be > 0")
        if self.field_width_m <= 0.0:
            raise ValueError("field_width_m must be > 0")
        if self.continuation_margin_m < 0.0:
            raise ValueError("continuation_margin_m must be >= 0")
        if self.seed_search_seconds < 0.0:
            raise ValueError("seed_search_seconds must be >= 0")
        if not 0.0 < self.initial_seed_min_fraction <= 1.0:
            raise ValueError(
                "initial_seed_min_fraction must be in (0, 1]"
            )
        if self.bootstrap_root_pool_size < 1:
            raise ValueError("bootstrap_root_pool_size must be >= 1")
        if not 0.0 <= self.bootstrap_root_recovery_guard <= 1.0:
            raise ValueError(
                "bootstrap_root_recovery_guard must be in [0, 1]"
            )
        if self.bootstrap_composition_weight < 0.0:
            raise ValueError(
                "bootstrap_composition_weight must be >= 0"
            )

        composition = (
            self.team_a_count,
            self.team_b_count,
            self.referee_count,
        )
        if any(value is not None for value in composition):
            if any(value is None for value in composition):
                raise ValueError(
                    "team_a_count, team_b_count and referee_count "
                    "must be provided together"
                )
            assert self.team_a_count is not None
            assert self.team_b_count is not None
            assert self.referee_count is not None
            if min(
                self.team_a_count,
                self.team_b_count,
                self.referee_count,
            ) < 0:
                raise ValueError("roster composition counts must be >= 0")
            if (
                self.team_a_count
                + self.team_b_count
                + self.referee_count
                != self.roster_size
            ):
                raise ValueError(
                    "roster composition counts must sum to roster_size"
                )

        if self.recovery_batch_seconds < 0.0:
            raise ValueError("recovery_batch_seconds must be >= 0")
        if self.base_recovery_distance_m <= 0.0:
            raise ValueError("base_recovery_distance_m must be > 0")
        if self.max_player_speed_m_s <= 0.0:
            raise ValueError("max_player_speed_m_s must be > 0")
        if self.max_recovery_gap_s <= 0.0:
            raise ValueError("max_recovery_gap_s must be > 0")
        if self.max_overlap_seconds < 0.0:
            raise ValueError("max_overlap_seconds must be >= 0")
        if self.max_duplicate_overlap_distance_m <= 0.0:
            raise ValueError(
                "max_duplicate_overlap_distance_m must be > 0"
            )

        weights = (
            self.continuity_weight,
            self.appearance_weight,
            self.team_weight,
            self.roster_prior_weight,
        )
        if any(value < 0.0 for value in weights):
            raise ValueError("recovery weights must be >= 0")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("recovery weights must sum to 1")

        if not 0.0 <= self.trusted_team_similarity_margin <= 1.0:
            raise ValueError(
                "trusted_team_similarity_margin must be in [0, 1]"
            )
        if not 0.5 <= self.canonical_team_vote_share <= 1.0:
            raise ValueError(
                "canonical_team_vote_share must be in [0.5, 1]"
            )
        if self.max_recovery_passes < 1:
            raise ValueError("max_recovery_passes must be >= 1")

        if not 0.0 <= self.global_edge_min_score <= 1.0:
            raise ValueError("global_edge_min_score must be in [0, 1]")
        if self.global_transition_weight < 0.0:
            raise ValueError("global_transition_weight must be >= 0")
        if self.global_class_weight < 0.0:
            raise ValueError("global_class_weight must be >= 0")
        if self.global_fragment_penalty < 0.0:
            raise ValueError("global_fragment_penalty must be >= 0")
        if self.global_coverage_scale <= 0.0:
            raise ValueError("global_coverage_scale must be > 0")
        if not 0.0 <= self.global_hard_team_margin <= 1.0:
            raise ValueError("global_hard_team_margin must be in [0, 1]")
        if self.global_min_reliable_observations < 1:
            raise ValueError(
                "global_min_reliable_observations must be >= 1"
            )
        if self.global_mip_time_limit_s <= 0.0:
            raise ValueError("global_mip_time_limit_s must be > 0")
        if not 0.0 <= self.global_mip_relative_gap <= 1.0:
            raise ValueError("global_mip_relative_gap must be in [0, 1]")

        for name in (
            "min_recovery_score",
            "min_outside_only_recovery_score",
            "min_assignment_margin",
            "min_structural_candidate_score",
            "min_unique_candidate_score",
            "min_unique_outside_candidate_score",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RecoveryCandidate:
    canonical_id: int
    fragment_raw_track_id: int
    predecessor_raw_track_id: int

    gap_s: float
    endpoint_distance_m: float
    overlap_seconds: float
    overlap_median_distance_m: float | None

    continuity_score: float
    appearance_score: float
    team_score: float
    roster_prior_score: float
    total_score: float

    team_compatibility: str
    outside_only_fragment: bool
    admissible: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalRosterIdentity:
    canonical_id: int
    member_raw_track_ids: tuple[int, ...]
    team_label: RelativeTeamLabel
    roster_class: RosterClass
    seed_raw_track_id: int

    def to_dict(self) -> dict[str, object]:
        return {
            "canonical_id": self.canonical_id,
            "member_raw_track_ids": list(
                self.member_raw_track_ids
            ),
            "team_label": self.team_label.value,
            "roster_class": self.roster_class.value,
            "seed_raw_track_id": self.seed_raw_track_id,
        }


@dataclass(frozen=True)
class RosterRecoveryResult:
    seed_frame_index: int
    seed_timestamp: float
    roster_completed_frame_index: int
    roster_completed_timestamp: float
    bootstrap_decision_timestamp: float

    identities: tuple[CanonicalRosterIdentity, ...]
    raw_to_canonical: Mapping[int, int]

    bootstrap_added_root_ids: tuple[int, ...]
    bootstrap_recovered_fragment_ids: tuple[int, ...]
    post_bootstrap_recovered_fragment_ids: tuple[int, ...]
    assigned_fragment_ids: tuple[int, ...]
    unassigned_fragment_ids: tuple[int, ...]

    composition_score: float | None

    recovery_candidates: tuple[RecoveryCandidate, ...]
    accepted_recoveries: tuple[RecoveryCandidate, ...]

    appearance_query_count: int

    global_solver_status: str | None = None
    global_objective: float | None = None
    global_candidate_edge_count: int = 0
    global_selected_transition_count: int = 0
    global_hard_team_pair_prunes: int = 0
    global_temporal_pair_prunes: int = 0
    global_context_end_timestamp: float | None = None
    global_selected_track_count: int = 0

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_frame_index": self.seed_frame_index,
            "seed_timestamp": self.seed_timestamp,
            "roster_completed_frame_index": (
                self.roster_completed_frame_index
            ),
            "roster_completed_timestamp": (
                self.roster_completed_timestamp
            ),
            "bootstrap_decision_timestamp": (
                self.bootstrap_decision_timestamp
            ),
            "identities": [
                row.to_dict()
                for row in self.identities
            ],
            "raw_to_canonical": {
                str(raw_id): canonical_id
                for raw_id, canonical_id in sorted(
                    self.raw_to_canonical.items()
                )
            },
            "bootstrap_added_root_ids": list(
                self.bootstrap_added_root_ids
            ),
            "bootstrap_recovered_fragment_ids": list(
                self.bootstrap_recovered_fragment_ids
            ),
            "post_bootstrap_recovered_fragment_ids": list(
                self.post_bootstrap_recovered_fragment_ids
            ),
            "assigned_fragment_ids": list(
                self.assigned_fragment_ids
            ),
            "unassigned_fragment_ids": list(
                self.unassigned_fragment_ids
            ),
            "composition_score": self.composition_score,
            "recovery_candidates": [
                row.to_dict()
                for row in self.recovery_candidates
            ],
            "accepted_recoveries": [
                row.to_dict()
                for row in self.accepted_recoveries
            ],
            "appearance_query_count": (
                self.appearance_query_count
            ),
            "global_solver_status": self.global_solver_status,
            "global_objective": self.global_objective,
            "global_candidate_edge_count": (
                self.global_candidate_edge_count
            ),
            "global_selected_transition_count": (
                self.global_selected_transition_count
            ),
            "global_hard_team_pair_prunes": (
                self.global_hard_team_pair_prunes
            ),
            "global_temporal_pair_prunes": (
                self.global_temporal_pair_prunes
            ),
            "global_context_end_timestamp": (
                self.global_context_end_timestamp
            ),
            "global_selected_track_count": (
                self.global_selected_track_count
            ),
        }


AppearanceSimilarity = Callable[[int, int], float | None]


def recover_roster_identities(
    tracklets: Sequence[RawTracklet],
    *,
    appearance_similarity: AppearanceSimilarity | None = None,
    config: RosterRecoveryConfig | None = None,
) -> RosterRecoveryResult:
    """Solve the WHOLE clip as exactly ``roster_size`` canonical paths.

    V6 removes the final online bias left in V5: there are no mandatory
    bootstrap/root raw tracks. The first seconds are evidence, not authority.
    A canonical player may first become observable several seconds into the
    clip, and later evidence may change which early raw track belongs to which
    canonical path.

    With exact football composition configured, the MILP contains 11 TEAM_A
    path slots, 11 TEAM_B path slots and one REFEREE slot directly. The
    composition is therefore part of path selection, not a label correction
    performed afterwards.

    Stable raw tracks are still preferred as units. V6 does not split a raw
    track from a single noisy appearance crop. Internal segmentation remains a
    separate evidence-driven stage for genuine identity-switch cases.
    """
    cfg = config or RosterRecoveryConfig()

    tracklet_by_id = {
        row.raw_track_id: row
        for row in tracklets
    }
    if len(tracklet_by_id) != len(tracklets):
        raise ValueError("raw_track_id values must be unique")
    if len(tracklets) < cfg.roster_size:
        raise ValueError(
            "not enough raw tracklets to establish roster"
        )

    seed_frame = min(
        row.start_frame
        for row in tracklets
    )
    seed_timestamp = min(
        row.start_timestamp
        for row in tracklets
    )
    global_context_end = max(
        row.end_timestamp
        for row in tracklets
    )

    (
        pair_evidence,
        appearance_query_count,
        hard_team_prunes,
        temporal_prunes,
    ) = _build_global_pair_evidence(
        tracklets,
        appearance_similarity=appearance_similarity,
        config=cfg,
    )

    slot_classes = _global_roster_slot_classes(
        config=cfg,
    )
    class_scores = _global_track_class_scores(
        tracklets,
        config=cfg,
    )

    solution = _solve_full_match_roster_paths_milp(
        tracklets,
        slot_classes=slot_classes,
        class_scores=class_scores,
        pair_evidence=pair_evidence,
        config=cfg,
    )

    raw_paths = solution[
        "selected_paths"
    ]

    # Relabel solver slots deterministically. Same-class MILP slots are
    # exchangeable; canonical IDs should not change just because HiGHS picked
    # another symmetric slot number.
    ordered_paths = sorted(
        raw_paths,
        key=lambda item: (
            min(
                (
                    tracklet_by_id[
                        raw_id
                    ].start_timestamp
                    for raw_id in item[
                        "raw_ids"
                    ]
                ),
                default=float("inf"),
            ),
            min(
                item[
                    "raw_ids"
                ],
                default=10**9,
            ),
            item[
                "roster_class"
            ].value,
        ),
    )

    identities_out: list[
        CanonicalRosterIdentity
    ] = []
    raw_to_canonical: dict[
        int,
        int,
    ] = {}
    accepted_recoveries: list[
        RecoveryCandidate
    ] = []

    path_slot_to_canonical: dict[
        int,
        int,
    ] = {}

    for canonical_id, path in enumerate(
        ordered_paths,
        start=1,
    ):
        slot_id = int(
            path[
                "slot_id"
            ]
        )
        path_slot_to_canonical[
            slot_id
        ] = canonical_id

        members = tuple(
            sorted(
                path[
                    "raw_ids"
                ],
                key=lambda raw_id: (
                    tracklet_by_id[
                        raw_id
                    ].start_timestamp,
                    tracklet_by_id[
                        raw_id
                    ].end_timestamp,
                    raw_id,
                ),
            )
        )
        if not members:
            raise RuntimeError(
                "V6 solver returned an empty canonical path"
            )

        for raw_id in members:
            if raw_id in raw_to_canonical:
                raise RuntimeError(
                    f"raw track {raw_id} selected by multiple V6 paths"
                )
            raw_to_canonical[
                raw_id
            ] = canonical_id

        trusted_team = _global_canonical_team(
            [
                tracklet_by_id[
                    raw_id
                ]
                for raw_id in members
            ],
            config=cfg,
        )

        identities_out.append(
            CanonicalRosterIdentity(
                canonical_id=canonical_id,
                member_raw_track_ids=members,
                team_label=trusted_team,
                roster_class=path[
                    "roster_class"
                ],
                seed_raw_track_id=members[0],
            )
        )

    for (
        slot_id,
        predecessor_id,
        successor_id,
    ) in solution[
        "selected_edges"
    ]:
        canonical_id = (
            path_slot_to_canonical[
                int(
                    slot_id
                )
            ]
        )
        evidence = pair_evidence[
            (
                predecessor_id,
                successor_id,
            )
        ]
        roster_class = next(
            row.roster_class
            for row in identities_out
            if row.canonical_id
            == canonical_id
        )
        roster_prior = class_scores[
            successor_id
        ][
            roster_class
        ]
        total = _global_transition_score_for_canonical(
            evidence,
            roster_prior=roster_prior,
        )

        accepted_recoveries.append(
            RecoveryCandidate(
                canonical_id=canonical_id,
                fragment_raw_track_id=(
                    successor_id
                ),
                predecessor_raw_track_id=(
                    predecessor_id
                ),
                gap_s=float(
                    evidence[
                        "gap_s"
                    ]
                ),
                endpoint_distance_m=float(
                    evidence[
                        "endpoint_distance_m"
                    ]
                ),
                overlap_seconds=float(
                    evidence[
                        "overlap_seconds"
                    ]
                ),
                overlap_median_distance_m=(
                    evidence[
                        "overlap_median_distance_m"
                    ]
                ),
                continuity_score=float(
                    evidence[
                        "continuity_score"
                    ]
                ),
                appearance_score=float(
                    evidence[
                        "appearance_score"
                    ]
                ),
                team_score=float(
                    evidence[
                        "team_score"
                    ]
                ),
                roster_prior_score=float(
                    roster_prior
                ),
                total_score=float(
                    total
                ),
                team_compatibility=str(
                    evidence[
                        "team_compatibility"
                    ]
                ),
                outside_only_fragment=(
                    not tracklet_by_id[
                        successor_id
                    ].established_inside
                ),
                admissible=True,
                reason="accepted_full_match_path",
            )
        )

    unassigned = tuple(
        sorted(
            row.raw_track_id
            for row in tracklets
            if row.raw_track_id
            not in raw_to_canonical
        )
    )

    first_member_rows = [
        min(
            (
                tracklet_by_id[
                    raw_id
                ]
                for raw_id in identity.member_raw_track_ids
            ),
            key=lambda row: (
                row.start_timestamp,
                row.start_frame,
                row.raw_track_id,
            ),
        )
        for identity in identities_out
    ]
    roster_completed_frame = max(
        row.start_frame
        for row in first_member_rows
    )
    roster_completed_timestamp = max(
        row.start_timestamp
        for row in first_member_rows
    )

    # Score the exact class assignment that participated in optimization.
    path_class_scores = []
    for identity in identities_out:
        weights = 0.0
        total = 0.0
        for raw_id in identity.member_raw_track_ids:
            row = tracklet_by_id[
                raw_id
            ]
            weight = float(
                min(
                    max(
                        row.observed_count,
                        1,
                    ),
                    180,
                )
            )
            total += (
                weight
                * class_scores[
                    raw_id
                ][
                    identity.roster_class
                ]
            )
            weights += weight
        if weights > 0.0:
            path_class_scores.append(
                total / weights
            )

    composition_score = (
        float(
            sum(
                path_class_scores
            )
            / len(
                path_class_scores
            )
        )
        if path_class_scores
        else None
    )

    audit_candidates: list[
        RecoveryCandidate
    ] = []
    for (
        predecessor_id,
        successor_id,
    ), evidence in sorted(
        pair_evidence.items()
    ):
        audit_candidates.append(
            RecoveryCandidate(
                canonical_id=0,
                fragment_raw_track_id=(
                    successor_id
                ),
                predecessor_raw_track_id=(
                    predecessor_id
                ),
                gap_s=float(
                    evidence[
                        "gap_s"
                    ]
                ),
                endpoint_distance_m=float(
                    evidence[
                        "endpoint_distance_m"
                    ]
                ),
                overlap_seconds=float(
                    evidence[
                        "overlap_seconds"
                    ]
                ),
                overlap_median_distance_m=(
                    evidence[
                        "overlap_median_distance_m"
                    ]
                ),
                continuity_score=float(
                    evidence[
                        "continuity_score"
                    ]
                ),
                appearance_score=float(
                    evidence[
                        "appearance_score"
                    ]
                ),
                team_score=float(
                    evidence[
                        "team_score"
                    ]
                ),
                roster_prior_score=0.5,
                total_score=float(
                    evidence[
                        "base_score"
                    ]
                ),
                team_compatibility=str(
                    evidence[
                        "team_compatibility"
                    ]
                ),
                outside_only_fragment=(
                    not tracklet_by_id[
                        successor_id
                    ].established_inside
                ),
                admissible=True,
                reason="global_candidate_edge",
            )
        )

    class_counts = {
        RosterClass.TEAM_A: 0,
        RosterClass.TEAM_B: 0,
        RosterClass.REFEREE: 0,
        RosterClass.UNKNOWN: 0,
    }
    for identity in identities_out:
        class_counts[
            identity.roster_class
        ] += 1

    # A strict sanity check when exact composition was requested.
    if (
        cfg.team_a_count is not None
        and (
            class_counts[
                RosterClass.TEAM_A
            ]
            != cfg.team_a_count
            or class_counts[
                RosterClass.TEAM_B
            ]
            != cfg.team_b_count
            or class_counts[
                RosterClass.REFEREE
            ]
            != cfg.referee_count
        )
    ):
        raise RuntimeError(
            "V6 solver violated exact roster composition"
        )

    assigned_nonfirst = {
        raw_id
        for identity in identities_out
        for raw_id in identity.member_raw_track_ids[
            1:
        ]
    }

    return RosterRecoveryResult(
        seed_frame_index=seed_frame,
        seed_timestamp=seed_timestamp,
        roster_completed_frame_index=(
            roster_completed_frame
        ),
        roster_completed_timestamp=(
            roster_completed_timestamp
        ),
        # There is no bootstrap decision in V6. The final identity decision
        # intentionally uses the complete clip, so expose that timestamp here
        # for backwards-compatible consumers and explicitly below as context.
        bootstrap_decision_timestamp=(
            global_context_end
        ),
        identities=tuple(
            identities_out
        ),
        raw_to_canonical=raw_to_canonical,
        bootstrap_added_root_ids=(),
        bootstrap_recovered_fragment_ids=(),
        post_bootstrap_recovered_fragment_ids=tuple(
            sorted(
                assigned_nonfirst
            )
        ),
        assigned_fragment_ids=tuple(
            sorted(
                raw_to_canonical
            )
        ),
        unassigned_fragment_ids=unassigned,
        composition_score=composition_score,
        recovery_candidates=tuple(
            audit_candidates
        ),
        accepted_recoveries=tuple(
            sorted(
                accepted_recoveries,
                key=lambda row: (
                    row.canonical_id,
                    tracklet_by_id[
                        row.predecessor_raw_track_id
                    ].start_timestamp,
                    tracklet_by_id[
                        row.fragment_raw_track_id
                    ].start_timestamp,
                ),
            )
        ),
        appearance_query_count=(
            appearance_query_count
        ),
        global_solver_status=str(
            solution[
                "status"
            ]
        ),
        global_objective=float(
            solution[
                "objective"
            ]
        ),
        global_candidate_edge_count=len(
            pair_evidence
        ),
        global_selected_transition_count=len(
            solution[
                "selected_edges"
            ]
        ),
        global_hard_team_pair_prunes=(
            hard_team_prunes
        ),
        global_temporal_pair_prunes=(
            temporal_prunes
        ),
        global_context_end_timestamp=(
            global_context_end
        ),
        global_selected_track_count=len(
            raw_to_canonical
        ),
    )


def _global_trusted_team(
    tracklet: RawTracklet,
    *,
    config: RosterRecoveryConfig,
) -> RelativeTeamLabel:
    """Conservative team evidence for hard pair pruning."""
    if (
        tracklet.team_a_similarity is not None
        and tracklet.team_b_similarity is not None
    ):
        delta = (
            float(tracklet.team_a_similarity)
            - float(tracklet.team_b_similarity)
        )
        if abs(delta) < config.global_hard_team_margin:
            return RelativeTeamLabel.UNKNOWN
        return (
            RelativeTeamLabel.TEAM_A
            if delta > 0.0
            else RelativeTeamLabel.TEAM_B
        )

    if tracklet.team_label in {
        RelativeTeamLabel.TEAM_A,
        RelativeTeamLabel.TEAM_B,
    }:
        return tracklet.team_label

    return RelativeTeamLabel.UNKNOWN


def _global_canonical_team(
    tracklets: Sequence[RawTracklet],
    *,
    config: RosterRecoveryConfig,
) -> RelativeTeamLabel:
    votes = {
        RelativeTeamLabel.TEAM_A: 0.0,
        RelativeTeamLabel.TEAM_B: 0.0,
    }

    for row in tracklets:
        label = _global_trusted_team(
            row,
            config=config,
        )
        if label not in votes:
            continue

        votes[label] += float(
            min(
                max(row.observed_count, 1),
                180,
            )
        )

    total = sum(votes.values())
    if total <= 0.0:
        return RelativeTeamLabel.UNKNOWN

    winner = max(votes, key=votes.get)
    share = votes[winner] / total

    if share < 0.60:
        return RelativeTeamLabel.UNKNOWN

    return winner


def _global_roster_slot_classes(
    *,
    config: RosterRecoveryConfig,
) -> tuple[RosterClass, ...]:
    if (
        config.team_a_count is not None
        and config.team_b_count is not None
        and config.referee_count is not None
    ):
        slots = (
            [RosterClass.TEAM_A]
            * config.team_a_count
            + [RosterClass.TEAM_B]
            * config.team_b_count
            + [RosterClass.REFEREE]
            * config.referee_count
        )
    else:
        slots = [
            RosterClass.UNKNOWN
        ] * config.roster_size

    if len(slots) != config.roster_size:
        raise ValueError(
            "slot-class count must equal roster_size"
        )

    return tuple(
        slots
    )


def _global_track_class_scores(
    tracklets: Sequence[RawTracklet],
    *,
    config: RosterRecoveryConfig,
) -> dict[
    int,
    dict[
        RosterClass,
        float,
    ],
]:
    referee_values = [
        float(
            row.referee_score
        )
        for row in tracklets
        if row.referee_score
        is not None
    ]
    referee_min = (
        min(
            referee_values
        )
        if referee_values
        else 0.0
    )
    referee_max = (
        max(
            referee_values
        )
        if referee_values
        else 1.0
    )

    scores = {}
    for row in tracklets:
        if (
            row.team_a_similarity
            is not None
            and row.team_b_similarity
            is not None
        ):
            margin = (
                float(
                    row.team_a_similarity
                )
                - float(
                    row.team_b_similarity
                )
            )
            p_a = (
                1.0
                / (
                    1.0
                    + math.exp(
                        -margin
                        / 0.085
                    )
                )
            )
        elif (
            row.team_label
            is RelativeTeamLabel.TEAM_A
        ):
            p_a = 0.94
        elif (
            row.team_label
            is RelativeTeamLabel.TEAM_B
        ):
            p_a = 0.06
        else:
            p_a = 0.50

        p_b = (
            1.0
            - p_a
        )

        if row.referee_score is not None:
            denominator = max(
                referee_max
                - referee_min,
                1e-6,
            )
            ref_rank = (
                float(
                    row.referee_score
                )
                - referee_min
            ) / denominator
            ref_rank = min(
                1.0,
                max(
                    0.0,
                    ref_rank,
                ),
            )
        else:
            ref_rank = 0.0

        ambiguity = (
            1.0
            - abs(
                p_a - p_b
            )
        )
        ref_score = (
            0.72
            * ref_rank
            + 0.28
            * ambiguity
        )

        scores[
            row.raw_track_id
        ] = {
            RosterClass.TEAM_A: float(
                p_a
            ),
            RosterClass.TEAM_B: float(
                p_b
            ),
            RosterClass.REFEREE: float(
                ref_score
            ),
            RosterClass.UNKNOWN: 0.50,
        }

    return scores


def _slot_allows_track(
    tracklet: RawTracklet,
    *,
    roster_class: RosterClass,
    config: RosterRecoveryConfig,
) -> bool:
    # Entirely outside even the continuation corridor: never publish.
    if (
        tracklet.inside_observed_count <= 0
        and tracklet.continuation_observed_count <= 0
    ):
        return False

    if roster_class not in {
        RosterClass.TEAM_A,
        RosterClass.TEAM_B,
    }:
        return True

    if (
        tracklet.team_a_similarity
        is None
        or tracklet.team_b_similarity
        is None
    ):
        return True

    delta = (
        float(
            tracklet.team_a_similarity
        )
        - float(
            tracklet.team_b_similarity
        )
    )

    # Strong visual kit evidence can veto the opposite team. Ambiguous tracks
    # such as mixed/occluded crops remain available to either class.
    if (
        delta
        >= config.global_hard_team_margin
        and roster_class
        is RosterClass.TEAM_B
    ):
        return False
    if (
        delta
        <= -config.global_hard_team_margin
        and roster_class
        is RosterClass.TEAM_A
    ):
        return False

    return True


def _full_match_node_utility(
    tracklet: RawTracklet,
    *,
    roster_class: RosterClass,
    class_score: float,
    config: RosterRecoveryConfig,
) -> float:
    inside = float(
        tracklet.inside_observed_count
    )
    outer_only = float(
        max(
            0,
            tracklet.continuation_observed_count
            - tracklet.inside_observed_count,
        )
    )

    coverage = (
        inside
        + 0.20
        * outer_only
    ) / config.global_coverage_scale

    reliability = min(
        1.0,
        tracklet.observed_count
        / float(
            config.global_min_reliable_observations
        ),
    )

    return float(
        reliability
        * (
            coverage
            + config.global_class_weight
            * class_score
        )
        - config.global_fragment_penalty
    )


def _build_global_pair_evidence(
    tracklets: Sequence[RawTracklet],
    *,
    appearance_similarity: AppearanceSimilarity | None,
    config: RosterRecoveryConfig,
) -> tuple[
    dict[
        tuple[int, int],
        dict[str, object],
    ],
    int,
    int,
    int,
]:
    """Build only physically/team-plausible directed recovery edges."""
    ordered = sorted(
        tracklets,
        key=lambda row: (
            row.start_timestamp,
            row.end_timestamp,
            row.raw_track_id,
        ),
    )

    evidence: dict[
        tuple[int, int],
        dict[str, object],
    ] = {}
    appearance_queries = 0
    hard_team_prunes = 0
    temporal_prunes = 0

    for left in ordered:
        left_team = _global_trusted_team(
            left,
            config=config,
        )

        for right in ordered:
            if (
                left.raw_track_id
                == right.raw_track_id
            ):
                continue

            # Directed acyclic order. Equal starts are ordered by end/raw id,
            # but long simultaneous tracks will still fail continuity.
            left_key = (
                left.start_timestamp,
                left.end_timestamp,
                left.raw_track_id,
            )
            right_key = (
                right.start_timestamp,
                right.end_timestamp,
                right.raw_track_id,
            )
            if left_key >= right_key:
                continue

            right_team = _global_trusted_team(
                right,
                config=config,
            )

            if (
                left_team
                is not RelativeTeamLabel.UNKNOWN
                and right_team
                is not RelativeTeamLabel.UNKNOWN
                and left_team is not right_team
            ):
                hard_team_prunes += 1
                continue

            (
                admissible,
                _continuity_reason,
                gap_s,
                endpoint_distance,
                overlap_seconds,
                overlap_median_distance,
                continuity_score,
            ) = _continuity_evidence(
                left,
                right,
                config=config,
            )

            if not admissible:
                temporal_prunes += 1
                continue

            team_compatibility, team_score = (
                _team_compatibility(
                    left_team,
                    right_team,
                )
            )

            appearance_score = 0.5
            if appearance_similarity is not None:
                value = appearance_similarity(
                    left.raw_track_id,
                    right.raw_track_id,
                )
                appearance_queries += 1
                if (
                    value is not None
                    and math.isfinite(
                        value
                    )
                ):
                    appearance_score = float(
                        min(
                            1.0,
                            max(
                                0.0,
                                value,
                            ),
                        )
                    )

            base_score = (
                0.58
                * continuity_score
                + 0.27
                * appearance_score
                + 0.15
                * team_score
            )

            if (
                base_score
                < config.global_edge_min_score
            ):
                continue

            evidence[
                (
                    left.raw_track_id,
                    right.raw_track_id,
                )
            ] = {
                "gap_s": float(
                    gap_s
                ),
                "endpoint_distance_m": float(
                    endpoint_distance
                ),
                "overlap_seconds": float(
                    overlap_seconds
                ),
                "overlap_median_distance_m": (
                    overlap_median_distance
                ),
                "continuity_score": float(
                    continuity_score
                ),
                "appearance_score": float(
                    appearance_score
                ),
                "team_score": float(
                    team_score
                ),
                "team_compatibility": (
                    team_compatibility
                ),
                "base_score": float(
                    base_score
                ),
            }

    return (
        evidence,
        appearance_queries,
        hard_team_prunes,
        temporal_prunes,
    )


def _global_transition_score_for_canonical(
    evidence: Mapping[str, object],
    *,
    roster_prior: float,
) -> float:
    return float(
        0.90
        * float(
            evidence[
                "base_score"
            ]
        )
        + 0.10
        * roster_prior
    )



def _solve_full_match_roster_paths_milp(
    tracklets: Sequence[RawTracklet],
    *,
    slot_classes: Sequence[RosterClass],
    class_scores: Mapping[
        int,
        Mapping[
            RosterClass,
            float,
        ],
    ],
    pair_evidence: Mapping[
        tuple[int, int],
        Mapping[str, object],
    ],
    config: RosterRecoveryConfig,
) -> dict[str, object]:
    """Global exact-roster path cover with no mandatory roots."""
    ordered = tuple(
        sorted(
            tracklets,
            key=lambda row: row.raw_track_id,
        )
    )
    row_by_raw = {
        row.raw_track_id: row
        for row in ordered
    }
    raw_ids = tuple(
        row.raw_track_id
        for row in ordered
    )
    slot_ids = tuple(
        range(
            len(
                slot_classes
            )
        )
    )

    variable_cost: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    integrality: list[int] = []

    def add_var(
        cost: float,
        *,
        lb: float = 0.0,
        ub: float = 1.0,
    ) -> int:
        index = len(
            variable_cost
        )
        variable_cost.append(
            float(
                cost
            )
        )
        lower.append(
            float(
                lb
            )
        )
        upper.append(
            float(
                ub
            )
        )
        integrality.append(1)
        return index

    x: dict[
        tuple[int, int],
        int,
    ] = {}
    s: dict[
        tuple[int, int],
        int,
    ] = {}
    t: dict[
        tuple[int, int],
        int,
    ] = {}

    for slot_id, roster_class in enumerate(
        slot_classes
    ):
        for raw_id in raw_ids:
            tracklet = row_by_raw[
                raw_id
            ]
            allowed = _slot_allows_track(
                tracklet,
                roster_class=roster_class,
                config=config,
            )
            class_score = (
                class_scores[
                    raw_id
                ][
                    roster_class
                ]
            )
            utility = (
                _full_match_node_utility(
                    tracklet,
                    roster_class=roster_class,
                    class_score=class_score,
                    config=config,
                )
            )

            ub = (
                1.0
                if allowed
                else 0.0
            )

            x[
                (
                    raw_id,
                    slot_id,
                )
            ] = add_var(
                -utility,
                ub=ub,
            )
            s[
                (
                    raw_id,
                    slot_id,
                )
            ] = add_var(
                0.0,
                ub=ub,
            )
            t[
                (
                    raw_id,
                    slot_id,
                )
            ] = add_var(
                0.0,
                ub=ub,
            )

    y: dict[
        tuple[
            int,
            int,
            int,
        ],
        int,
    ] = {}

    for slot_id, roster_class in enumerate(
        slot_classes
    ):
        for (
            predecessor_id,
            successor_id,
        ), evidence in pair_evidence.items():
            if (
                upper[
                    x[
                        (
                            predecessor_id,
                            slot_id,
                        )
                    ]
                ]
                <= 0.0
                or upper[
                    x[
                        (
                            successor_id,
                            slot_id,
                        )
                    ]
                ]
                <= 0.0
            ):
                continue

            roster_prior = (
                class_scores[
                    successor_id
                ][
                    roster_class
                ]
            )
            transition_score = (
                _global_transition_score_for_canonical(
                    evidence,
                    roster_prior=roster_prior,
                )
            )
            reliability = min(
                1.0,
                row_by_raw[
                    successor_id
                ].observed_count
                / float(
                    config.global_min_reliable_observations
                ),
            )

            y[
                (
                    predecessor_id,
                    successor_id,
                    slot_id,
                )
            ] = add_var(
                -(
                    config.global_transition_weight
                    * transition_score
                    * reliability
                )
            )

    row_indices: list[int] = []
    col_indices: list[int] = []
    values: list[float] = []
    constraint_lb: list[float] = []
    constraint_ub: list[float] = []
    constraint_row = 0

    def add_constraint(
        terms: Iterable[
            tuple[int, float]
        ],
        *,
        lb: float,
        ub: float,
    ) -> None:
        nonlocal constraint_row
        for variable, coefficient in terms:
            row_indices.append(
                constraint_row
            )
            col_indices.append(
                variable
            )
            values.append(
                float(
                    coefficient
                )
            )
        constraint_lb.append(
            float(
                lb
            )
        )
        constraint_ub.append(
            float(
                ub
            )
        )
        constraint_row += 1

    # A raw track can explain at most one canonical participant.
    for raw_id in raw_ids:
        add_constraint(
            (
                (
                    x[
                        (
                            raw_id,
                            slot_id,
                        )
                    ],
                    1.0,
                )
                for slot_id in slot_ids
            ),
            lb=0.0,
            ub=1.0,
        )

    incoming = {
        (
            raw_id,
            slot_id,
        ): []
        for raw_id in raw_ids
        for slot_id in slot_ids
    }
    outgoing = {
        (
            raw_id,
            slot_id,
        ): []
        for raw_id in raw_ids
        for slot_id in slot_ids
    }

    for (
        predecessor_id,
        successor_id,
        slot_id,
    ), variable in y.items():
        outgoing[
            (
                predecessor_id,
                slot_id,
            )
        ].append(
            variable
        )
        incoming[
            (
                successor_id,
                slot_id,
            )
        ].append(
            variable
        )

    # Every roster slot is one non-empty connected path and must contain at
    # least one genuinely strict-inside observation somewhere in the full clip.
    for slot_id in slot_ids:
        add_constraint(
            (
                (
                    s[
                        (
                            raw_id,
                            slot_id,
                        )
                    ],
                    1.0,
                )
                for raw_id in raw_ids
            ),
            lb=1.0,
            ub=1.0,
        )
        add_constraint(
            (
                (
                    t[
                        (
                            raw_id,
                            slot_id,
                        )
                    ],
                    1.0,
                )
                for raw_id in raw_ids
            ),
            lb=1.0,
            ub=1.0,
        )
        add_constraint(
            (
                (
                    x[
                        (
                            raw_id,
                            slot_id,
                        )
                    ],
                    1.0,
                )
                for raw_id in raw_ids
                if row_by_raw[
                    raw_id
                ].inside_observed_count
                > 0
            ),
            lb=1.0,
            ub=float("inf"),
        )

        for raw_id in raw_ids:
            in_terms = [
                (
                    variable,
                    1.0,
                )
                for variable in incoming[
                    (
                        raw_id,
                        slot_id,
                    )
                ]
            ]
            in_terms.extend(
                [
                    (
                        s[
                            (
                                raw_id,
                                slot_id,
                            )
                        ],
                        1.0,
                    ),
                    (
                        x[
                            (
                                raw_id,
                                slot_id,
                            )
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

            out_terms = [
                (
                    variable,
                    1.0,
                )
                for variable in outgoing[
                    (
                        raw_id,
                        slot_id,
                    )
                ]
            ]
            out_terms.extend(
                [
                    (
                        t[
                            (
                                raw_id,
                                slot_id,
                            )
                        ],
                        1.0,
                    ),
                    (
                        x[
                            (
                                raw_id,
                                slot_id,
                            )
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

    # Symmetry breaking for exchangeable same-class slots: source times must be
    # nondecreasing. This does not change the feasible roster, only removes
    # equivalent slot permutations from the MILP search.
    source_rank = {
        raw_id: index
        for index, raw_id in enumerate(
            sorted(
                raw_ids,
                key=lambda raw_id: (
                    row_by_raw[
                        raw_id
                    ].start_timestamp,
                    row_by_raw[
                        raw_id
                    ].start_frame,
                    raw_id,
                ),
            )
        )
    }

    class_to_slots: dict[
        RosterClass,
        list[int],
    ] = {}
    for slot_id, roster_class in enumerate(
        slot_classes
    ):
        class_to_slots.setdefault(
            roster_class,
            [],
        ).append(
            slot_id
        )

    for same_class_slots in class_to_slots.values():
        for left_slot, right_slot in zip(
            same_class_slots,
            same_class_slots[
                1:
            ],
        ):
            terms = []
            for raw_id in raw_ids:
                rank = float(
                    source_rank[
                        raw_id
                    ]
                )
                terms.append(
                    (
                        s[
                            (
                                raw_id,
                                left_slot,
                            )
                        ],
                        rank,
                    )
                )
                terms.append(
                    (
                        s[
                            (
                                raw_id,
                                right_slot,
                            )
                        ],
                        -rank,
                    )
                )
            add_constraint(
                terms,
                lb=-float("inf"),
                ub=0.0,
            )

    matrix = coo_matrix(
        (
            np.asarray(
                values,
                dtype=np.float64,
            ),
            (
                np.asarray(
                    row_indices,
                    dtype=np.int32,
                ),
                np.asarray(
                    col_indices,
                    dtype=np.int32,
                ),
            ),
        ),
        shape=(
            constraint_row,
            len(
                variable_cost
            ),
        ),
    ).tocsr()

    result = milp(
        c=np.asarray(
            variable_cost,
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
                constraint_lb,
                dtype=np.float64,
            ),
            np.asarray(
                constraint_ub,
                dtype=np.float64,
            ),
        ),
        options={
            "time_limit": float(
                config.global_mip_time_limit_s
            ),
            "mip_rel_gap": float(
                config.global_mip_relative_gap
            ),
            "presolve": True,
        },
    )

    if result.x is None:
        raise RuntimeError(
            "V6 full-match roster MILP found no feasible solution: "
            f"status={result.status}, message={result.message}"
        )

    vector = np.asarray(
        result.x,
        dtype=np.float64,
    )

    selected_paths = []
    for slot_id, roster_class in enumerate(
        slot_classes
    ):
        selected = [
            raw_id
            for raw_id in raw_ids
            if (
                vector[
                    x[
                        (
                            raw_id,
                            slot_id,
                        )
                    ]
                ]
                > 0.5
            )
        ]
        if not selected:
            raise RuntimeError(
                f"V6 solver left slot {slot_id} empty"
            )

        selected_paths.append(
            {
                "slot_id": slot_id,
                "roster_class": roster_class,
                "raw_ids": tuple(
                    selected
                ),
            }
        )

    selected_edges = []
    for (
        predecessor_id,
        successor_id,
        slot_id,
    ), variable in y.items():
        if vector[
            variable
        ] > 0.5:
            selected_edges.append(
                (
                    slot_id,
                    predecessor_id,
                    successor_id,
                )
            )

    status = (
        "optimal"
        if result.status == 0
        else f"status_{result.status}"
    )
    objective = (
        -float(
            result.fun
        )
        if result.fun is not None
        else float("nan")
    )

    return {
        "status": status,
        "objective": objective,
        "selected_paths": tuple(
            selected_paths
        ),
        "selected_edges": tuple(
            selected_edges
        ),
    }



def _recover_fragment_batches(
    fragments: Sequence[RawTracklet],
    *,
    identities: dict[int, list[int]],
    canonical_team: dict[int, RelativeTeamLabel],
    raw_to_canonical: dict[int, int],
    tracklet_by_id: Mapping[int, RawTracklet],
    appearance_similarity: AppearanceSimilarity | None,
    roster_classes: Mapping[int, RosterClass] | None = None,
    config: RosterRecoveryConfig,
    all_candidates: list[RecoveryCandidate],
    accepted: list[RecoveryCandidate],
) -> tuple[
    set[int],
    dict[
        tuple[int, int],
        RecoveryCandidate,
    ],
    int,
]:
    """Recover unresolved fragments with iterative occupied/free exclusion.

    Each pass is chronological. Once a fragment is attached, the canonical
    identity's occupancy/history changes immediately for later batches. A new
    pass lets earlier unresolved fragments benefit from identities recovered
    later in the previous pass (offline future context / backfill).

    ReID is queried only after:
      - trusted-team conflict has been ruled out,
      - canonical occupancy has been ruled out,
      - physical temporal continuity is plausible.
    """
    recovered: set[int] = set()
    all_rows: dict[
        tuple[int, int],
        RecoveryCandidate,
    ] = {}
    queried_total = 0

    for _pass_index in range(
        config.max_recovery_passes
    ):
        pass_recovered = 0

        remaining = [
            row
            for row in fragments
            if row.raw_track_id
            not in raw_to_canonical
        ]
        remaining.sort(
            key=lambda row: (
                row.start_timestamp,
                row.raw_track_id,
            )
        )

        if not remaining:
            break

        for batch in _recovery_batches(
            remaining,
            window_s=config.recovery_batch_seconds,
        ):
            candidate_rows: dict[
                tuple[int, int],
                RecoveryCandidate,
            ] = {}

            for fragment in batch:
                if (
                    fragment.raw_track_id
                    in raw_to_canonical
                ):
                    continue

                for canonical_id in sorted(
                    identities
                ):
                    members = [
                        tracklet_by_id[
                            raw_id
                        ]
                        for raw_id in identities[
                            canonical_id
                        ]
                    ]

                    candidate, queried = (
                        _score_recovery_candidate(
                            canonical_id=canonical_id,
                            canonical_members=members,
                            canonical_team=canonical_team[
                                canonical_id
                            ],
                            canonical_roster_class=(
                                roster_classes.get(
                                    canonical_id,
                                    RosterClass.UNKNOWN,
                                )
                                if roster_classes
                                is not None
                                else RosterClass.UNKNOWN
                            ),
                            fragment=fragment,
                            appearance_similarity=(
                                appearance_similarity
                            ),
                            config=config,
                        )
                    )
                    queried_total += queried
                    all_candidates.append(
                        candidate
                    )
                    candidate_rows[
                        (
                            fragment.raw_track_id,
                            canonical_id,
                        )
                    ] = candidate
                    all_rows[
                        (
                            fragment.raw_track_id,
                            canonical_id,
                        )
                    ] = candidate

            live_batch = [
                row
                for row in batch
                if row.raw_track_id
                not in raw_to_canonical
            ]

            if not live_batch:
                continue

            assignments = _solve_batch_assignment(
                live_batch,
                canonical_ids=tuple(
                    sorted(
                        identities
                    )
                ),
                candidates=candidate_rows,
                config=config,
            )

            for (
                fragment,
                canonical_id,
                assignment_reason,
            ) in assignments:
                if (
                    fragment.raw_track_id
                    in raw_to_canonical
                ):
                    continue

                candidate = candidate_rows[
                    (
                        fragment.raw_track_id,
                        canonical_id,
                    )
                ]

                identities[
                    canonical_id
                ].append(
                    fragment.raw_track_id
                )
                identities[
                    canonical_id
                ].sort(
                    key=lambda raw_id: (
                        tracklet_by_id[
                            raw_id
                        ].start_timestamp,
                        raw_id,
                    )
                )
                raw_to_canonical[
                    fragment.raw_track_id
                ] = canonical_id
                recovered.add(
                    fragment.raw_track_id
                )
                pass_recovered += 1

                accepted.append(
                    replace(
                        candidate,
                        reason=assignment_reason,
                    )
                )

                canonical_team[
                    canonical_id
                ] = _trusted_canonical_team(
                    [
                        tracklet_by_id[
                            raw_id
                        ]
                        for raw_id in identities[
                            canonical_id
                        ]
                    ],
                    config=config,
                )

        if pass_recovered == 0:
            break

    return (
        recovered,
        all_rows,
        queried_total,
    )



def _select_bootstrap_roots(
    candidates: Sequence[RawTracklet],
    *,
    missing_count: int,
    identities: Mapping[int, Sequence[int]],
    tracklet_by_id: Mapping[int, RawTracklet],
    appearance_similarity: AppearanceSimilarity | None,
    bootstrap_deadline: float,
    config: RosterRecoveryConfig,
) -> tuple[
    tuple[int, ...],
    float | None,
    int,
]:
    if missing_count <= 0:
        return (), None, 0

    if len(candidates) < missing_count:
        return (), None, 0

    scored = []
    query_count = 0

    for candidate in candidates:
        (
            root_score,
            best_recovery,
            queries,
        ) = _bootstrap_root_score(
            candidate,
            identities=identities,
            tracklet_by_id=tracklet_by_id,
            appearance_similarity=appearance_similarity,
            bootstrap_deadline=bootstrap_deadline,
            config=config,
        )
        query_count += queries

        # A candidate that still strongly looks like a lost fragment of an
        # existing canonical should NOT open a new roster slot.
        if (
            best_recovery
            >= config.bootstrap_root_recovery_guard
        ):
            continue

        scored.append(
            (
                root_score,
                candidate,
            )
        )

    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].raw_track_id,
        )
    )

    pool = [
        row
        for _, row in scored[
            : config.bootstrap_root_pool_size
        ]
    ]

    if len(pool) < missing_count:
        return (), None, query_count

    best_combo: tuple[
        RawTracklet,
        ...,
    ] | None = None
    best_objective = -math.inf
    best_composition: float | None = None

    # Usually missing_count == 2 for this clip. The pool cap keeps exhaustive
    # evaluation small while allowing genuinely global bootstrap selection.
    for combo in combinations(
        pool,
        missing_count,
    ):
        duplicate_risk, duplicate_queries = (
            _bootstrap_combo_has_duplicate_identity_risk(
                combo,
                appearance_similarity=appearance_similarity,
                config=config,
            )
        )
        query_count += duplicate_queries

        if duplicate_risk:
            continue

        provisional = {
            canonical_id: list(
                members
            )
            for canonical_id, members in (
                identities.items()
            )
        }

        next_id = max(
            provisional,
            default=0,
        ) + 1

        for offset, row in enumerate(
            combo
        ):
            provisional[
                next_id + offset
            ] = [
                row.raw_track_id
            ]

        root_quality = sum(
            _bootstrap_root_intrinsic_quality(
                row,
                bootstrap_deadline=bootstrap_deadline,
                config=config,
            )
            for row in combo
        ) / max(
            len(combo),
            1,
        )

        _classes, composition_score = (
            _assign_roster_classes(
                provisional,
                tracklet_by_id=tracklet_by_id,
                config=config,
            )
        )

        objective = (
            root_quality
            + config.bootstrap_composition_weight
            * (
                composition_score
                if composition_score
                is not None
                else 0.0
            )
        )

        if objective > best_objective:
            best_objective = objective
            best_combo = combo
            best_composition = (
                composition_score
            )

    if best_combo is None:
        return (), None, query_count

    return (
        tuple(
            row.raw_track_id
            for row in best_combo
        ),
        best_composition,
        query_count,
    )


def _bootstrap_root_score(
    candidate: RawTracklet,
    *,
    identities: Mapping[int, Sequence[int]],
    tracklet_by_id: Mapping[int, RawTracklet],
    appearance_similarity: AppearanceSimilarity | None,
    bootstrap_deadline: float,
    config: RosterRecoveryConfig,
) -> tuple[
    float,
    float,
    int,
]:
    intrinsic = (
        _bootstrap_root_intrinsic_quality(
            candidate,
            bootstrap_deadline=bootstrap_deadline,
            config=config,
        )
    )

    distinct_votes = 0
    comparable = 0
    best_recovery = 0.0
    queries = 0

    for members in identities.values():
        member_rows = [
            tracklet_by_id[
                raw_id
            ]
            for raw_id in members
        ]

        predecessor = _best_predecessor(
            member_rows,
            candidate,
        )
        if predecessor is None:
            continue

        (
            admissible,
            _reason,
            _gap,
            _distance,
            overlap_seconds,
            overlap_median,
            continuity,
        ) = _continuity_evidence(
            predecessor,
            candidate,
            config=config,
        )

        # Simultaneous, spatially separated observations are powerful evidence
        # that this candidate is a genuinely different participant.
        if (
            overlap_seconds > 0.0
            and (
                overlap_median is None
                or overlap_median
                > config.max_duplicate_overlap_distance_m
            )
        ):
            distinct_votes += 1
            comparable += 1
            continue

        if admissible:
            comparable += 1
            appearance = 0.5

            if appearance_similarity is not None:
                value = appearance_similarity(
                    predecessor.raw_track_id,
                    candidate.raw_track_id,
                )
                queries += 1
                if (
                    value is not None
                    and math.isfinite(
                        value
                    )
                ):
                    appearance = min(
                        1.0,
                        max(
                            0.0,
                            float(value),
                        ),
                    )

            team_compatibility, team_score = (
                _team_compatibility(
                    _aggregate_team_label(
                        member_rows
                    ),
                    candidate.team_label,
                )
            )

            if team_compatibility == "conflict":
                recovery = 0.0
            else:
                recovery = (
                    config.continuity_weight
                    * continuity
                    + config.appearance_weight
                    * appearance
                    + config.team_weight
                    * team_score
                )

            best_recovery = max(
                best_recovery,
                recovery,
            )

    distinctness = (
        distinct_votes
        / comparable
        if comparable
        else 0.0
    )

    score = (
        0.72 * intrinsic
        + 0.28 * distinctness
    )

    return (
        float(score),
        float(best_recovery),
        queries,
    )


def _bootstrap_root_intrinsic_quality(
    tracklet: RawTracklet,
    *,
    bootstrap_deadline: float,
    config: RosterRecoveryConfig,
) -> float:
    inside_rows = [
        row
        for row in tracklet.observations
        if (
            row.timestamp
            <= bootstrap_deadline + 1e-9
            and _inside_strict_pitch(
                row,
                config=config,
            )
        )
    ]

    if not inside_rows:
        return 0.0

    count_score = min(
        1.0,
        len(inside_rows)
        / 45.0,
    )
    span = max(
        0.0,
        inside_rows[-1].timestamp
        - inside_rows[0].timestamp,
    )
    span_score = min(
        1.0,
        span
        / max(
            config.seed_search_seconds
            * 0.55,
            1e-6,
        ),
    )

    return float(
        0.58 * count_score
        + 0.42 * span_score
    )


def _bootstrap_combo_has_duplicate_identity_risk(
    combo: Sequence[RawTracklet],
    *,
    appearance_similarity: AppearanceSimilarity | None,
    config: RosterRecoveryConfig,
) -> tuple[bool, int]:
    queries = 0

    for left, right in combinations(
        combo,
        2,
    ):
        # Long/separated simultaneous overlap proves different people.
        (
            admissible,
            _reason,
            _gap,
            _distance,
            overlap_seconds,
            overlap_median,
            continuity,
        ) = _continuity_evidence(
            left,
            right,
            config=config,
        )

        if (
            overlap_seconds > 0.0
            and (
                overlap_median is None
                or overlap_median
                > config.max_duplicate_overlap_distance_m
            )
        ):
            continue

        if not admissible:
            continue

        team_compatibility, team_score = (
            _team_compatibility(
                left.team_label,
                right.team_label,
            )
        )
        if team_compatibility == "conflict":
            continue

        appearance = 0.5
        if appearance_similarity is not None:
            value = appearance_similarity(
                left.raw_track_id,
                right.raw_track_id,
            )
            queries += 1
            if (
                value is not None
                and math.isfinite(value)
            ):
                appearance = float(
                    min(
                        1.0,
                        max(
                            0.0,
                            value,
                        ),
                    )
                )

        same_score = (
            config.continuity_weight
            * continuity
            + config.appearance_weight
            * appearance
            + config.team_weight
            * team_score
        )

        if (
            same_score
            >= config.bootstrap_root_recovery_guard
        ):
            return True, queries

    return False, queries


def _aggregate_team_label(
    tracklets: Sequence[RawTracklet],
) -> RelativeTeamLabel:
    a = sum(
        min(
            row.observed_count,
            100,
        )
        for row in tracklets
        if row.team_label
        is RelativeTeamLabel.TEAM_A
    )
    b = sum(
        min(
            row.observed_count,
            100,
        )
        for row in tracklets
        if row.team_label
        is RelativeTeamLabel.TEAM_B
    )

    if a > b and a > 0:
        return RelativeTeamLabel.TEAM_A
    if b > a and b > 0:
        return RelativeTeamLabel.TEAM_B
    return RelativeTeamLabel.UNKNOWN


def _assign_roster_classes(
    identities: Mapping[
        int,
        Sequence[int],
    ],
    *,
    tracklet_by_id: Mapping[int, RawTracklet],
    config: RosterRecoveryConfig,
) -> tuple[
    dict[int, RosterClass],
    float | None,
]:
    canonical_ids = tuple(
        sorted(
            identities
        )
    )

    if (
        config.team_a_count is None
        or config.team_b_count is None
        or config.referee_count is None
    ):
        fallback = {}
        for canonical_id in canonical_ids:
            team = _aggregate_team_label(
                [
                    tracklet_by_id[
                        raw_id
                    ]
                    for raw_id in identities[
                        canonical_id
                    ]
                ]
            )
            if team is RelativeTeamLabel.TEAM_A:
                roster_class = RosterClass.TEAM_A
            elif team is RelativeTeamLabel.TEAM_B:
                roster_class = RosterClass.TEAM_B
            else:
                roster_class = RosterClass.UNKNOWN
            fallback[
                canonical_id
            ] = roster_class
        return fallback, None

    slots = (
        [RosterClass.TEAM_A]
        * config.team_a_count
        + [RosterClass.TEAM_B]
        * config.team_b_count
        + [RosterClass.REFEREE]
        * config.referee_count
    )

    if len(canonical_ids) != len(slots):
        raise ValueError(
            "exact roster composition requires a complete roster"
        )

    raw_referee_scores = [
        float(row.referee_score)
        for row in tracklet_by_id.values()
        if row.referee_score is not None
    ]
    referee_min = (
        min(raw_referee_scores)
        if raw_referee_scores
        else 0.0
    )
    referee_max = (
        max(raw_referee_scores)
        if raw_referee_scores
        else 1.0
    )

    score_matrix = np.zeros(
        (
            len(canonical_ids),
            len(slots),
        ),
        dtype=np.float64,
    )

    for row_index, canonical_id in enumerate(
        canonical_ids
    ):
        members = [
            tracklet_by_id[
                raw_id
            ]
            for raw_id in identities[
                canonical_id
            ]
        ]

        class_scores = (
            _identity_roster_class_scores(
                members,
                referee_min=referee_min,
                referee_max=referee_max,
            )
        )

        for col_index, roster_class in enumerate(
            slots
        ):
            score_matrix[
                row_index,
                col_index
            ] = class_scores[
                roster_class
            ]

    row_indices, col_indices = (
        linear_sum_assignment(
            -score_matrix
        )
    )

    result = {}
    assigned_scores = []

    for row_index, col_index in zip(
        row_indices.tolist(),
        col_indices.tolist(),
    ):
        canonical_id = canonical_ids[
            row_index
        ]
        roster_class = slots[
            col_index
        ]
        result[
            canonical_id
        ] = roster_class
        assigned_scores.append(
            float(
                score_matrix[
                    row_index,
                    col_index
                ]
            )
        )

    composition_score = (
        float(
            sum(
                assigned_scores
            )
            / max(
                len(
                    assigned_scores
                ),
                1,
            )
        )
    )

    return (
        result,
        composition_score,
    )


def _identity_roster_class_scores(
    members: Sequence[RawTracklet],
    *,
    referee_min: float,
    referee_max: float,
) -> dict[
    RosterClass,
    float,
]:
    total_weight = 0.0
    a_sum = 0.0
    b_sum = 0.0
    ref_sum = 0.0

    for row in members:
        weight = float(
            min(
                max(
                    row.observed_count,
                    1,
                ),
                120,
            )
        )

        if (
            row.team_a_similarity is not None
            and row.team_b_similarity is not None
        ):
            margin = (
                float(
                    row.team_a_similarity
                )
                - float(
                    row.team_b_similarity
                )
            )
            p_a = (
                1.0
                / (
                    1.0
                    + math.exp(
                        -margin
                        / 0.085
                    )
                )
            )
        elif (
            row.team_label
            is RelativeTeamLabel.TEAM_A
        ):
            p_a = 0.94
        elif (
            row.team_label
            is RelativeTeamLabel.TEAM_B
        ):
            p_a = 0.06
        else:
            p_a = 0.50

        p_b = 1.0 - p_a

        if row.referee_score is not None:
            denominator = max(
                referee_max
                - referee_min,
                1e-6,
            )
            ref_rank = (
                float(
                    row.referee_score
                )
                - referee_min
            ) / denominator
            ref_rank = min(
                1.0,
                max(
                    0.0,
                    ref_rank,
                ),
            )
        else:
            ref_rank = 0.0

        team_ambiguity = (
            1.0
            - abs(
                p_a - p_b
            )
        )

        ref_score = (
            0.72 * ref_rank
            + 0.28 * team_ambiguity
        )

        a_sum += (
            weight * p_a
        )
        b_sum += (
            weight * p_b
        )
        ref_sum += (
            weight * ref_score
        )
        total_weight += weight

    if total_weight <= 0.0:
        return {
            RosterClass.TEAM_A: 0.5,
            RosterClass.TEAM_B: 0.5,
            RosterClass.REFEREE: 0.5,
            RosterClass.UNKNOWN: 0.0,
        }

    return {
        RosterClass.TEAM_A: float(
            a_sum
            / total_weight
        ),
        RosterClass.TEAM_B: float(
            b_sum
            / total_weight
        ),
        RosterClass.REFEREE: float(
            ref_sum
            / total_weight
        ),
        RosterClass.UNKNOWN: 0.0,
    }



def _select_initial_seed_roster(
    tracklets: Sequence[RawTracklet],
    *,
    config: RosterRecoveryConfig,
) -> tuple[
    int,
    float,
    tuple[RawTracklet, ...],
    float,
]:
    """Select the strongest initial same-frame roster subset.

    Unlike the previous implementation, this function does NOT require all 23
    players in one frame.  It selects the early frame with the largest number
    of qualified participant tracklets, and the remaining canonical slots are
    established progressively by ``recover_roster_identities``.

    A tracklet is qualified only if it proves strict-inside presence somewhere
    during the bootstrap window.  At the actual seed frame it may be inside the
    1.5 m continuation corridor.  Outside-only staff can therefore never seed.
    """
    first_timestamp = min(
        row.start_timestamp
        for row in tracklets
    )
    bootstrap_deadline = (
        first_timestamp
        + config.seed_search_seconds
    )

    early_strict_ids: set[int] = set()

    for tracklet in tracklets:
        if _has_strict_inside_by(
            tracklet,
            deadline=bootstrap_deadline,
            config=config,
        ):
            early_strict_ids.add(
                tracklet.raw_track_id
            )

    frames: dict[
        int,
        list[
            tuple[
                RosterObservation,
                RawTracklet,
            ]
        ],
    ] = {}

    for tracklet in tracklets:
        if (
            tracklet.raw_track_id
            not in early_strict_ids
        ):
            continue

        for observation in tracklet.observations:
            if (
                observation.timestamp
                > bootstrap_deadline + 1e-9
            ):
                break

            if not _inside_allowed_pitch(
                observation,
                config=config,
            ):
                continue

            frames.setdefault(
                observation.frame_index,
                [],
            ).append(
                (
                    observation,
                    tracklet,
                )
            )

    best_frame: int | None = None
    best_rows: list[
        tuple[
            RosterObservation,
            RawTracklet,
        ]
    ] = []

    preferred_frame: int | None = None
    preferred_rows: list[
        tuple[
            RosterObservation,
            RawTracklet,
        ]
    ] = []

    minimum_initial = max(
        1,
        int(
            math.ceil(
                config.roster_size
                * config.initial_seed_min_fraction
            )
        ),
    )

    for frame_index in sorted(
        frames
    ):
        unique = {
            tracklet.raw_track_id: (
                observation,
                tracklet,
            )
            for observation, tracklet in frames[
                frame_index
            ]
        }

        rows = list(
            unique.values()
        )

        if (
            preferred_frame is None
            and len(rows)
            >= minimum_initial
        ):
            preferred_frame = (
                frame_index
            )
            preferred_rows = rows

        if (
            len(rows)
            > len(best_rows)
        ):
            best_frame = frame_index
            best_rows = rows

    if preferred_frame is not None:
        best_frame = preferred_frame
        best_rows = preferred_rows

    if (
        best_frame is None
        or not best_rows
    ):
        raise RuntimeError(
            "could not establish any initial roster seed "
            f"inside {config.seed_search_seconds:.2f}s bootstrap window"
        )

    best_rows.sort(
        key=lambda pair: (
            -pair[1].inside_observed_count,
            -pair[1].observed_count,
            -pair[1].observed_span_s,
            pair[1].raw_track_id,
        )
    )

    selected = tuple(
        pair[1]
        for pair in best_rows[
            : config.roster_size
        ]
    )

    timestamp = float(
        min(
            pair[0].timestamp
            for pair in best_rows
        )
    )

    return (
        best_frame,
        timestamp,
        selected,
        bootstrap_deadline,
    )



def _has_strict_inside_by(
    tracklet: RawTracklet,
    *,
    deadline: float,
    config: RosterRecoveryConfig,
) -> bool:
    return any(
        observation.timestamp
        <= deadline + 1e-9
        and _inside_strict_pitch(
            observation,
            config=config,
        )
        for observation in tracklet.observations
    )


def _inside_strict_pitch(
    observation: RosterObservation,
    *,
    config: RosterRecoveryConfig,
) -> bool:
    half_length = (
        config.field_length_m
        / 2.0
    )
    half_width = (
        config.field_width_m
        / 2.0
    )

    return (
        -half_length
        <= observation.x
        <= half_length
        and -half_width
        <= observation.z
        <= half_width
    )



def _inside_allowed_pitch(
    observation: RosterObservation,
    *,
    config: RosterRecoveryConfig,
) -> bool:
    half_length = (
        config.field_length_m
        / 2.0
        + config.continuation_margin_m
    )
    half_width = (
        config.field_width_m
        / 2.0
        + config.continuation_margin_m
    )

    return (
        -half_length
        <= observation.x
        <= half_length
        and -half_width
        <= observation.z
        <= half_width
    )

def _recovery_batches(
    fragments: Sequence[RawTracklet],
    *,
    window_s: float,
) -> Iterable[
    tuple[RawTracklet, ...]
]:
    if not fragments:
        return

    batch: list[RawTracklet] = []
    start = None

    for fragment in fragments:
        if not batch:
            batch = [fragment]
            start = fragment.start_timestamp
            continue

        assert start is not None

        if (
            fragment.start_timestamp
            - start
            <= window_s + 1e-9
        ):
            batch.append(
                fragment
            )
            continue

        yield tuple(
            batch
        )
        batch = [fragment]
        start = fragment.start_timestamp

    if batch:
        yield tuple(
            batch
        )


def _score_recovery_candidate(
    *,
    canonical_id: int,
    canonical_members: Sequence[RawTracklet],
    canonical_team: RelativeTeamLabel,
    canonical_roster_class: RosterClass = RosterClass.UNKNOWN,
    fragment: RawTracklet,
    appearance_similarity: AppearanceSimilarity | None,
    config: RosterRecoveryConfig,
) -> tuple[
    RecoveryCandidate,
    int,
]:
    trusted_canonical = _trusted_canonical_team(
        canonical_members,
        config=config,
        fallback=canonical_team,
    )
    trusted_fragment = _trusted_tracklet_team(
        fragment,
        config=config,
    )

    team_compatibility, team_score = (
        _team_compatibility(
            trusted_canonical,
            trusted_fragment,
        )
    )

    anchor = _best_temporal_anchor(
        canonical_members,
        fragment,
    )
    if anchor is None:
        anchor = canonical_members[-1]

    if team_compatibility == "conflict":
        return (
            _rejected_candidate(
                canonical_id,
                fragment,
                predecessor=anchor,
                reason="trusted_team_conflict",
                team_compatibility=(
                    team_compatibility
                ),
                roster_prior_score=(
                    _roster_prior_score(
                        canonical_roster_class,
                        fragment,
                        config=config,
                    )
                ),
            ),
            0,
        )

    available, availability_reason = (
        _canonical_available_for_fragment(
            canonical_members,
            fragment,
            config=config,
        )
    )
    if not available:
        return (
            _rejected_candidate(
                canonical_id,
                fragment,
                predecessor=anchor,
                reason=availability_reason,
                team_compatibility=(
                    team_compatibility
                ),
                roster_prior_score=(
                    _roster_prior_score(
                        canonical_roster_class,
                        fragment,
                        config=config,
                    )
                ),
            ),
            0,
        )

    if anchor is None:
        return (
            _rejected_candidate(
                canonical_id,
                fragment,
                predecessor=canonical_members[-1],
                reason="no_temporal_anchor",
                team_compatibility=(
                    team_compatibility
                ),
                roster_prior_score=0.5,
            ),
            0,
        )

    (
        admissible,
        continuity_reason,
        gap_s,
        endpoint_distance,
        overlap_seconds,
        overlap_median_distance,
        continuity_score,
    ) = _continuity_between(
        anchor,
        fragment,
        config=config,
    )

    roster_prior_score = (
        _roster_prior_score(
            canonical_roster_class,
            fragment,
            config=config,
        )
    )

    if not admissible:
        return (
            RecoveryCandidate(
                canonical_id=canonical_id,
                fragment_raw_track_id=(
                    fragment.raw_track_id
                ),
                predecessor_raw_track_id=(
                    anchor.raw_track_id
                ),
                gap_s=gap_s,
                endpoint_distance_m=(
                    endpoint_distance
                ),
                overlap_seconds=(
                    overlap_seconds
                ),
                overlap_median_distance_m=(
                    overlap_median_distance
                ),
                continuity_score=(
                    continuity_score
                ),
                appearance_score=0.5,
                team_score=team_score,
                roster_prior_score=(
                    roster_prior_score
                ),
                total_score=0.0,
                team_compatibility=(
                    team_compatibility
                ),
                outside_only_fragment=(
                    not fragment.established_inside
                ),
                admissible=False,
                reason=continuity_reason,
            ),
            0,
        )

    if (
        not fragment.established_inside
        and not config.allow_outside_only_recovery
    ):
        return (
            RecoveryCandidate(
                canonical_id=canonical_id,
                fragment_raw_track_id=(
                    fragment.raw_track_id
                ),
                predecessor_raw_track_id=(
                    anchor.raw_track_id
                ),
                gap_s=gap_s,
                endpoint_distance_m=(
                    endpoint_distance
                ),
                overlap_seconds=(
                    overlap_seconds
                ),
                overlap_median_distance_m=(
                    overlap_median_distance
                ),
                continuity_score=(
                    continuity_score
                ),
                appearance_score=0.5,
                team_score=team_score,
                roster_prior_score=(
                    roster_prior_score
                ),
                total_score=0.0,
                team_compatibility=(
                    team_compatibility
                ),
                outside_only_fragment=True,
                admissible=False,
                reason="outside_only_recovery_disabled",
            ),
            0,
        )

    appearance_score = 0.5
    queried = 0

    if appearance_similarity is not None:
        similarities = []

        # Recovery-only ReID. Query the temporally closest canonical history,
        # never every frame and never every visible player.
        ordered_members = sorted(
            canonical_members,
            key=lambda row: (
                _interval_gap_s(
                    row,
                    fragment,
                ),
                -row.observed_count,
                row.raw_track_id,
            ),
        )

        for member in ordered_members[:3]:
            value = appearance_similarity(
                member.raw_track_id,
                fragment.raw_track_id,
            )
            queried += 1

            if (
                value is not None
                and math.isfinite(
                    value
                )
            ):
                similarities.append(
                    min(
                        1.0,
                        max(
                            0.0,
                            float(
                                value
                            ),
                        ),
                    )
                )

        if similarities:
            similarities.sort(
                reverse=True
            )
            top = similarities[:2]
            appearance_score = float(
                sum(top)
                / len(top)
            )

    total = (
        config.continuity_weight
        * continuity_score
        + config.appearance_weight
        * appearance_score
        + config.team_weight
        * team_score
        + config.roster_prior_weight
        * roster_prior_score
    )

    return (
        RecoveryCandidate(
            canonical_id=canonical_id,
            fragment_raw_track_id=(
                fragment.raw_track_id
            ),
            predecessor_raw_track_id=(
                anchor.raw_track_id
            ),
            gap_s=gap_s,
            endpoint_distance_m=(
                endpoint_distance
            ),
            overlap_seconds=(
                overlap_seconds
            ),
            overlap_median_distance_m=(
                overlap_median_distance
            ),
            continuity_score=(
                continuity_score
            ),
            appearance_score=(
                appearance_score
            ),
            team_score=team_score,
            roster_prior_score=(
                roster_prior_score
            ),
            total_score=float(
                total
            ),
            team_compatibility=(
                team_compatibility
            ),
            outside_only_fragment=(
                not fragment.established_inside
            ),
            admissible=True,
            reason="compatible_recovery",
        ),
        queried,
    )


def _interval_gap_s(
    left: RawTracklet,
    right: RawTracklet,
) -> float:
    if left.end_timestamp < right.start_timestamp:
        return (
            right.start_timestamp
            - left.end_timestamp
        )
    if right.end_timestamp < left.start_timestamp:
        return (
            left.start_timestamp
            - right.end_timestamp
        )
    return 0.0


def _best_temporal_anchor(
    canonical_members: Sequence[RawTracklet],
    fragment: RawTracklet,
) -> RawTracklet | None:
    if not canonical_members:
        return None

    return min(
        canonical_members,
        key=lambda row: (
            _interval_gap_s(
                row,
                fragment,
            ),
            -row.observed_count,
            abs(
                row.end_timestamp
                - fragment.start_timestamp
            ),
            row.raw_track_id,
        ),
    )


def _continuity_between(
    anchor: RawTracklet,
    fragment: RawTracklet,
    *,
    config: RosterRecoveryConfig,
):
    # Orient non-overlapping pairs chronologically. This allows offline backfill:
    # a fragment that occurred before the canonical root can still attach to it.
    if (
        fragment.end_timestamp
        <= anchor.start_timestamp
    ):
        return _continuity_evidence(
            fragment,
            anchor,
            config=config,
        )

    return _continuity_evidence(
        anchor,
        fragment,
        config=config,
    )


def _canonical_available_for_fragment(
    canonical_members: Sequence[RawTracklet],
    fragment: RawTracklet,
    *,
    config: RosterRecoveryConfig,
) -> tuple[bool, str]:
    """Return whether this canonical ID is free for the fragment.

    A canonical occupied by a simultaneous, spatially distinct raw person is
    unavailable. A short colocated duplicate overlap remains admissible because
    raw trackers can briefly produce two IDs at handover.
    """
    for member in canonical_members:
        overlap_start = max(
            member.start_timestamp,
            fragment.start_timestamp,
        )
        overlap_end = min(
            member.end_timestamp,
            fragment.end_timestamp,
        )
        overlap_s = max(
            0.0,
            overlap_end - overlap_start,
        )

        if overlap_s <= 0.0:
            continue

        median_distance = (
            _simultaneous_median_distance(
                member,
                fragment,
            )
        )

        if median_distance is None:
            # Same tracker timebase normally gives shared frames. If it does not,
            # a meaningful interval overlap is still an occupancy conflict.
            if overlap_s > 0.15:
                return (
                    False,
                    "canonical_occupied_no_shared_frame",
                )
            continue

        if (
            median_distance
            > config.max_duplicate_overlap_distance_m
        ):
            return (
                False,
                "canonical_occupied_separated",
            )

        if (
            overlap_s
            > config.max_overlap_seconds
        ):
            return (
                False,
                "canonical_occupied_long_duplicate",
            )

    return True, "canonical_available"


def _trusted_tracklet_team(
    tracklet: RawTracklet,
    *,
    config: RosterRecoveryConfig,
) -> RelativeTeamLabel:
    known = {
        RelativeTeamLabel.TEAM_A,
        RelativeTeamLabel.TEAM_B,
    }

    if tracklet.team_label in known:
        return tracklet.team_label

    if (
        tracklet.team_a_similarity is None
        or tracklet.team_b_similarity is None
    ):
        return RelativeTeamLabel.UNKNOWN

    delta = (
        float(
            tracklet.team_a_similarity
        )
        - float(
            tracklet.team_b_similarity
        )
    )

    if (
        abs(delta)
        < config.trusted_team_similarity_margin
    ):
        return RelativeTeamLabel.UNKNOWN

    return (
        RelativeTeamLabel.TEAM_A
        if delta > 0.0
        else RelativeTeamLabel.TEAM_B
    )


def _trusted_canonical_team(
    tracklets: Sequence[RawTracklet],
    *,
    config: RosterRecoveryConfig,
    fallback: RelativeTeamLabel = RelativeTeamLabel.UNKNOWN,
) -> RelativeTeamLabel:
    votes = {
        RelativeTeamLabel.TEAM_A: 0.0,
        RelativeTeamLabel.TEAM_B: 0.0,
    }

    for row in tracklets:
        label = _trusted_tracklet_team(
            row,
            config=config,
        )
        if label not in votes:
            continue

        votes[
            label
        ] += float(
            min(
                max(
                    row.observed_count,
                    1,
                ),
                120,
            )
        )

    total = sum(
        votes.values()
    )
    if total <= 0.0:
        return fallback

    winner = max(
        votes,
        key=votes.get,
    )
    share = (
        votes[
            winner
        ]
        / total
    )

    if (
        share
        < config.canonical_team_vote_share
    ):
        return RelativeTeamLabel.UNKNOWN

    return winner


def _roster_prior_score(
    roster_class: RosterClass,
    fragment: RawTracklet,
    *,
    config: RosterRecoveryConfig,
) -> float:
    trusted = _trusted_tracklet_team(
        fragment,
        config=config,
    )

    if trusted is RelativeTeamLabel.TEAM_A:
        if roster_class is RosterClass.TEAM_A:
            return 1.0
        if roster_class is RosterClass.TEAM_B:
            return 0.15
        if roster_class is RosterClass.REFEREE:
            return 0.20
        return 0.50

    if trusted is RelativeTeamLabel.TEAM_B:
        if roster_class is RosterClass.TEAM_B:
            return 1.0
        if roster_class is RosterClass.TEAM_A:
            return 0.15
        if roster_class is RosterClass.REFEREE:
            return 0.20
        return 0.50

    if (
        roster_class is RosterClass.REFEREE
        and fragment.referee_score is not None
    ):
        return float(
            min(
                1.0,
                max(
                    0.0,
                    fragment.referee_score,
                ),
            )
        )

    return 0.50



def _best_predecessor(
    canonical_members: Sequence[RawTracklet],
    fragment: RawTracklet,
) -> RawTracklet | None:
    # Prefer the member ending nearest to the fragment start.  A small overlap
    # is allowed because raw trackers often duplicate the same physical player
    # briefly during an ID handover.
    ordered = sorted(
        canonical_members,
        key=lambda row: (
            abs(
                fragment.start_timestamp
                - row.end_timestamp
            ),
            -row.end_timestamp,
            row.raw_track_id,
        ),
    )

    return ordered[0] if ordered else None


def _continuity_evidence(
    predecessor: RawTracklet,
    fragment: RawTracklet,
    *,
    config: RosterRecoveryConfig,
) -> tuple[
    bool,
    str,
    float,
    float,
    float,
    float | None,
    float,
]:
    gap_s = (
        fragment.start_timestamp
        - predecessor.end_timestamp
    )

    overlap_seconds = max(
        0.0,
        min(
            predecessor.end_timestamp,
            fragment.end_timestamp,
        )
        - max(
            predecessor.start_timestamp,
            fragment.start_timestamp,
        ),
    )

    if gap_s < 0.0:
        if (
            overlap_seconds
            > config.max_overlap_seconds
            + 1e-9
        ):
            return (
                False,
                "overlap_too_long",
                gap_s,
                float(
                    np.linalg.norm(
                        fragment.first_position
                        - predecessor.last_position
                    )
                ),
                overlap_seconds,
                _simultaneous_median_distance(
                    predecessor,
                    fragment,
                ),
                0.0,
            )

        median_distance = (
            _simultaneous_median_distance(
                predecessor,
                fragment,
            )
        )

        if (
            median_distance is None
            or median_distance
            > config.max_duplicate_overlap_distance_m
        ):
            return (
                False,
                "simultaneous_overlap_conflict",
                gap_s,
                float(
                    np.linalg.norm(
                        fragment.first_position
                        - predecessor.last_position
                    )
                ),
                overlap_seconds,
                median_distance,
                0.0,
            )

        continuity = max(
            0.0,
            1.0
            - median_distance
            / config.max_duplicate_overlap_distance_m,
        )

        return (
            True,
            "short_colocated_overlap",
            gap_s,
            float(
                np.linalg.norm(
                    fragment.first_position
                    - predecessor.last_position
                )
            ),
            overlap_seconds,
            median_distance,
            float(
                0.78
                + 0.22
                * continuity
            ),
        )

    if gap_s > config.max_recovery_gap_s:
        return (
            False,
            "gap_too_large",
            gap_s,
            float(
                np.linalg.norm(
                    fragment.first_position
                    - predecessor.last_position
                )
            ),
            0.0,
            None,
            0.0,
        )

    endpoint_distance = float(
        np.linalg.norm(
            fragment.first_position
            - predecessor.last_position
        )
    )

    reachable_distance = (
        config.base_recovery_distance_m
        + config.max_player_speed_m_s
        * gap_s
    )

    if (
        endpoint_distance
        > reachable_distance
        + 1e-9
    ):
        return (
            False,
            "physically_unreachable",
            gap_s,
            endpoint_distance,
            0.0,
            None,
            0.0,
        )

    # No directional extrapolation.  This asks only whether the player could
    # physically have travelled between the two observed endpoints.
    reachability_score = math.exp(
        -endpoint_distance
        / max(
            config.base_recovery_distance_m
            + 0.55
            * config.max_player_speed_m_s
            * gap_s,
            _EPS,
        )
    )
    gap_score = math.exp(
        -gap_s
        / max(
            config.max_recovery_gap_s
            / 2.5,
            _EPS,
        )
    )

    continuity = (
        0.72
        * reachability_score
        + 0.28
        * gap_score
    )

    return (
        True,
        "physically_reachable_gap",
        gap_s,
        endpoint_distance,
        0.0,
        None,
        float(
            min(
                1.0,
                max(
                    0.0,
                    continuity,
                ),
            )
        ),
    )


def _simultaneous_median_distance(
    a: RawTracklet,
    b: RawTracklet,
) -> float | None:
    by_frame = {
        row.frame_index: row
        for row in a.observations
    }

    distances = []

    for row in b.observations:
        other = by_frame.get(
            row.frame_index
        )
        if other is None:
            continue

        distances.append(
            math.hypot(
                row.x - other.x,
                row.z - other.z,
            )
        )

    if not distances:
        return None

    return float(
        np.median(
            np.asarray(
                distances,
                dtype=np.float64,
            )
        )
    )


def _team_compatibility(
    canonical: RelativeTeamLabel,
    fragment: RelativeTeamLabel,
) -> tuple[str, float]:
    known = {
        RelativeTeamLabel.TEAM_A,
        RelativeTeamLabel.TEAM_B,
    }

    if (
        canonical in known
        and fragment in known
    ):
        if canonical is fragment:
            return "same_team", 1.0
        return "conflict", 0.0

    return "unknown", 0.5


def _updated_team_label(
    current: RelativeTeamLabel,
    fragment: RelativeTeamLabel,
) -> RelativeTeamLabel:
    if current is RelativeTeamLabel.UNKNOWN:
        return fragment
    return current


def _rejected_candidate(
    canonical_id: int,
    fragment: RawTracklet,
    *,
    predecessor: RawTracklet,
    reason: str,
    team_compatibility: str,
    roster_prior_score: float = 0.5,
) -> RecoveryCandidate:
    return RecoveryCandidate(
        canonical_id=canonical_id,
        fragment_raw_track_id=(
            fragment.raw_track_id
        ),
        predecessor_raw_track_id=(
            predecessor.raw_track_id
        ),
        gap_s=(
            fragment.start_timestamp
            - predecessor.end_timestamp
        ),
        endpoint_distance_m=float(
            np.linalg.norm(
                fragment.first_position
                - predecessor.last_position
            )
        ),
        overlap_seconds=0.0,
        overlap_median_distance_m=None,
        continuity_score=0.0,
        appearance_score=0.5,
        team_score=0.0,
        roster_prior_score=(
            roster_prior_score
        ),
        total_score=0.0,
        team_compatibility=(
            team_compatibility
        ),
        outside_only_fragment=(
            not fragment.established_inside
        ),
        admissible=False,
        reason=reason,
    )


def _solve_batch_assignment(
    fragments: Sequence[RawTracklet],
    *,
    canonical_ids: Sequence[int],
    candidates: Mapping[
        tuple[int, int],
        RecoveryCandidate,
    ],
    config: RosterRecoveryConfig,
) -> tuple[
    tuple[
        RawTracklet,
        int,
        str,
    ],
    ...,
]:
    if not fragments or not canonical_ids:
        return ()

    scores = np.full(
        (
            len(fragments),
            len(canonical_ids),
        ),
        -np.inf,
        dtype=np.float64,
    )
    unique_candidate_rows: set[int] = set()

    for row_index, fragment in enumerate(
        fragments
    ):
        structurally_viable = []

        for col_index, canonical_id in enumerate(
            canonical_ids
        ):
            candidate = candidates[
                (
                    fragment.raw_track_id,
                    canonical_id,
                )
            ]

            if (
                candidate.admissible
                and candidate.total_score
                >= config.min_structural_candidate_score
            ):
                structurally_viable.append(
                    (
                        col_index,
                        candidate,
                    )
                )

        unique_mode = (
            len(
                structurally_viable
            )
            == 1
        )

        if unique_mode:
            unique_candidate_rows.add(
                row_index
            )

        for col_index, candidate in (
            structurally_viable
        ):
            if unique_mode:
                threshold = (
                    config.min_unique_outside_candidate_score
                    if candidate.outside_only_fragment
                    else config.min_unique_candidate_score
                )
            else:
                threshold = (
                    config.min_outside_only_recovery_score
                    if candidate.outside_only_fragment
                    else config.min_recovery_score
                )

            if (
                candidate.total_score
                >= threshold
            ):
                scores[
                    row_index,
                    col_index
                ] = candidate.total_score

    if not np.isfinite(
        scores
    ).any():
        return ()

    costs = np.where(
        np.isfinite(
            scores
        ),
        -scores,
        _LARGE_COST,
    )

    row_indices, col_indices = (
        linear_sum_assignment(
            costs
        )
    )

    accepted = []

    for row_index, col_index in zip(
        row_indices.tolist(),
        col_indices.tolist(),
    ):
        score = scores[
            row_index,
            col_index
        ]

        if not np.isfinite(
            score
        ):
            continue

        fragment = fragments[
            row_index
        ]
        canonical_id = canonical_ids[
            col_index
        ]

        if row_index in unique_candidate_rows:
            accepted.append(
                (
                    fragment,
                    canonical_id,
                    "accepted_unique_available",
                )
            )
            continue

        row_scores = scores[
            row_index
        ]
        alternatives = [
            value
            for index, value in enumerate(
                row_scores.tolist()
            )
            if (
                index != col_index
                and math.isfinite(
                    value
                )
            )
        ]

        second_best = (
            max(
                alternatives
            )
            if alternatives
            else 0.0
        )

        if (
            score
            - second_best
            < config.min_assignment_margin
        ):
            continue

        accepted.append(
            (
                fragment,
                canonical_id,
                "accepted_score_margin",
            )
        )

    return tuple(
        accepted
    )
