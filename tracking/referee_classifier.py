"""V8 post-hoc referee classifier for offline football tracking.

This module deliberately runs *after* the physical-identity solver.

The physical solver's job is only:
    "reconstruct 23 different people from the whole clip"

It does not know about football roles.  Once those 23 physical identities are
stable, this classifier selects exactly one as the referee and treats the other
22 as players.

No raw-track ID is hard-coded.  The referee score combines:

1. TEAM STRUCTURE (strong evidence, not a tracking constraint)
   Removing the referee should leave a plausible 22-player roster.  For a
   normal no-substitution clip that means neither confidently-labelled team may
   have more than ``expected_team_size`` identities; UNKNOWN/special identities
   can later be resolved as goalkeepers or uncertain players.

2. KIT OUTLIER
   The referee's full-path kit appearance should generally be less similar to
   the two dominant player-kit populations.  This is computed as a relative
   full-roster outlier score, not a brittle absolute threshold.

3. GOALKEEPER AVOIDANCE
   A special/unknown identity living close to one goal is more likely to be a
   goalkeeper than the referee.  This prevents "different kit" from meaning
   "referee".

4. FULL-PATH SUPPORT / CENTRAL PRESENCE
   The referee should have enough strict-inside evidence and normally spend a
   meaningful fraction of time away from the immediate goal areas.

The classifier never changes physical identity membership.  It only labels one
of the already reconstructed physical people as REFEREE.  This keeps role
classification separate from identity recovery, which is important on tiny
football crops where appearance labels are noisy.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .physical_identity_solver import (
    PhysicalIdentity,
    PhysicalTrackSegment,
)
from .team_similarity import RelativeTeamLabel


_EPS = 1e-12


@dataclass(frozen=True)
class RefereeClassifierConfig:
    expected_team_size: int = 11

    # Geometry.  Coordinates follow the existing centered X/Z football field.
    goalkeeper_goal_radius_m: float = 20.0
    central_min_goal_distance_m: float = 24.0

    # Minimum useful path evidence.
    full_support_inside_observations: int = 60

    # Relative score weights.
    team_structure_weight: float = 0.46
    kit_outlier_weight: float = 0.30
    non_goalkeeper_weight: float = 0.14
    support_weight: float = 0.06
    central_presence_weight: float = 0.04

    def __post_init__(self) -> None:
        if self.expected_team_size < 1:
            raise ValueError("expected_team_size must be >= 1")
        if self.goalkeeper_goal_radius_m <= 0.0:
            raise ValueError("goalkeeper_goal_radius_m must be > 0")
        if self.central_min_goal_distance_m <= 0.0:
            raise ValueError("central_min_goal_distance_m must be > 0")
        if self.full_support_inside_observations < 1:
            raise ValueError("full_support_inside_observations must be >= 1")

        weights = (
            self.team_structure_weight,
            self.kit_outlier_weight,
            self.non_goalkeeper_weight,
            self.support_weight,
            self.central_presence_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("referee score weights must be >= 0")
        if abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("referee score weights must sum to 1")


@dataclass(frozen=True)
class RefereeCandidateScore:
    identity_id: int
    member_segment_ids: tuple[int, ...]
    member_keys: tuple[str, ...]
    team_label: RelativeTeamLabel

    total_score: float
    team_structure_score: float
    kit_outlier_score: float
    non_goalkeeper_score: float
    support_score: float
    central_presence_score: float

    kit_team_affinity: float | None
    goal_zone_fraction: float
    central_fraction: float
    strict_inside_observations: int

    team_a_after_removal: int
    team_b_after_removal: int
    unknown_after_removal: int
    roster_structure_feasible: bool

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["team_label"] = self.team_label.value
        payload["member_segment_ids"] = list(self.member_segment_ids)
        payload["member_keys"] = list(self.member_keys)
        return payload


@dataclass(frozen=True)
class RefereeClassificationResult:
    referee_identity_id: int
    player_identity_ids: tuple[int, ...]
    candidates: tuple[RefereeCandidateScore, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "referee_identity_id": self.referee_identity_id,
            "player_identity_ids": list(self.player_identity_ids),
            "candidates": [row.to_dict() for row in self.candidates],
        }


def classify_referee_identity(
    identities: Sequence[PhysicalIdentity],
    *,
    segment_by_id: Mapping[int, PhysicalTrackSegment],
    kit_prototype_by_segment_id: Mapping[int, np.ndarray] | None = None,
    field_length: float = 105.0,
    field_width: float = 68.0,
    config: RefereeClassifierConfig | None = None,
) -> RefereeClassificationResult:
    """Select exactly one referee from already reconstructed physical people."""

    cfg = config or RefereeClassifierConfig()

    if len(identities) < 3:
        raise ValueError("need at least three physical identities")
    if field_length <= 0.0 or field_width <= 0.0:
        raise ValueError("field dimensions must be > 0")

    identity_ids = [identity.identity_id for identity in identities]
    if len(identity_ids) != len(set(identity_ids)):
        raise ValueError("identity_id values must be unique")

    identity_by_id = {
        identity.identity_id: identity
        for identity in identities
    }

    for identity in identities:
        for segment_id in identity.member_segment_ids:
            if segment_id not in segment_by_id:
                raise ValueError(
                    f"identity {identity.identity_id} references unknown "
                    f"segment {segment_id}"
                )

    kit_map = kit_prototype_by_segment_id or {}

    identity_kit = {
        identity.identity_id: _identity_kit_prototype(
            identity,
            segment_by_id=segment_by_id,
            kit_prototype_by_segment_id=kit_map,
        )
        for identity in identities
    }

    # "How much does this identity look like either dominant player-kit group?"
    # Use leave-one-out peer medians so a referee misclassified as TEAM_A cannot
    # pull its own team reference towards itself.
    affinities = {
        identity.identity_id: _team_kit_affinity(
            identity,
            identities=identities,
            identity_kit=identity_kit,
        )
        for identity in identities
    }

    outlier_scores = _relative_outlier_scores(affinities)

    before_counts = _team_counts(identities)
    candidates: list[RefereeCandidateScore] = []

    for identity in identities:
        after = [
            other
            for other in identities
            if other.identity_id != identity.identity_id
        ]
        after_counts = _team_counts(after)

        structure_score, feasible = _team_structure_score(
            before_counts,
            after_counts,
            expected_team_size=cfg.expected_team_size,
        )

        (
            goal_zone_fraction,
            central_fraction,
            strict_inside_observations,
        ) = _identity_spatial_features(
            identity,
            segment_by_id=segment_by_id,
            field_length=field_length,
            field_width=field_width,
            goalkeeper_goal_radius_m=cfg.goalkeeper_goal_radius_m,
            central_min_goal_distance_m=cfg.central_min_goal_distance_m,
        )

        non_goalkeeper_score = float(
            np.clip(
                1.0 - goal_zone_fraction,
                0.0,
                1.0,
            )
        )
        support_score = float(
            min(
                1.0,
                strict_inside_observations
                / float(cfg.full_support_inside_observations),
            )
        )

        kit_outlier_score = outlier_scores.get(
            identity.identity_id,
            0.5,
        )

        total = (
            cfg.team_structure_weight * structure_score
            + cfg.kit_outlier_weight * kit_outlier_score
            + cfg.non_goalkeeper_weight * non_goalkeeper_score
            + cfg.support_weight * support_score
            + cfg.central_presence_weight * central_fraction
        )

        candidates.append(
            RefereeCandidateScore(
                identity_id=identity.identity_id,
                member_segment_ids=identity.member_segment_ids,
                member_keys=identity.member_keys,
                team_label=identity.team_label,
                total_score=float(total),
                team_structure_score=float(structure_score),
                kit_outlier_score=float(kit_outlier_score),
                non_goalkeeper_score=float(non_goalkeeper_score),
                support_score=float(support_score),
                central_presence_score=float(central_fraction),
                kit_team_affinity=affinities.get(identity.identity_id),
                goal_zone_fraction=float(goal_zone_fraction),
                central_fraction=float(central_fraction),
                strict_inside_observations=int(strict_inside_observations),
                team_a_after_removal=after_counts[RelativeTeamLabel.TEAM_A],
                team_b_after_removal=after_counts[RelativeTeamLabel.TEAM_B],
                unknown_after_removal=after_counts[RelativeTeamLabel.UNKNOWN],
                roster_structure_feasible=bool(feasible),
            )
        )

    candidates.sort(
        key=lambda row: (
            -row.total_score,
            -row.team_structure_score,
            -row.kit_outlier_score,
            -row.support_score,
            row.identity_id,
        )
    )

    referee = candidates[0]

    return RefereeClassificationResult(
        referee_identity_id=referee.identity_id,
        player_identity_ids=tuple(
            sorted(
                identity_id
                for identity_id in identity_by_id
                if identity_id != referee.identity_id
            )
        ),
        candidates=tuple(candidates),
    )


def _team_counts(
    identities: Sequence[PhysicalIdentity],
) -> dict[RelativeTeamLabel, int]:
    counts = {
        RelativeTeamLabel.TEAM_A: 0,
        RelativeTeamLabel.TEAM_B: 0,
        RelativeTeamLabel.UNKNOWN: 0,
    }
    for identity in identities:
        label = identity.team_label
        if label not in counts:
            label = RelativeTeamLabel.UNKNOWN
        counts[label] += 1
    return counts


def _team_structure_score(
    before: Mapping[RelativeTeamLabel, int],
    after: Mapping[RelativeTeamLabel, int],
    *,
    expected_team_size: int,
) -> tuple[float, bool]:
    a_before = before[RelativeTeamLabel.TEAM_A]
    b_before = before[RelativeTeamLabel.TEAM_B]
    a_after = after[RelativeTeamLabel.TEAM_A]
    b_after = after[RelativeTeamLabel.TEAM_B]
    unknown_after = after[RelativeTeamLabel.UNKNOWN]

    overflow_before = (
        max(0, a_before - expected_team_size)
        + max(0, b_before - expected_team_size)
    )
    overflow_after = (
        max(0, a_after - expected_team_size)
        + max(0, b_after - expected_team_size)
    )

    feasible = (
        a_after <= expected_team_size
        and b_after <= expected_team_size
    )

    if not feasible:
        return 0.0, False

    # With 22 remaining identities, UNKNOWN/special people can exactly occupy
    # unresolved player slots (for example goalkeepers) when neither labelled
    # team exceeds 11.
    required_unknown = (
        max(0, expected_team_size - a_after)
        + max(0, expected_team_size - b_after)
    )
    unknown_fit = (
        1.0
        if required_unknown == unknown_after
        else math.exp(
            -abs(required_unknown - unknown_after)
        )
    )

    if overflow_before > 0:
        overflow_gain = (
            overflow_before - overflow_after
        ) / float(overflow_before)
        score = (
            0.72 * max(0.0, overflow_gain)
            + 0.28 * unknown_fit
        )
    else:
        # No labelled-team overflow exists.  Structure remains useful but
        # should not dominate kit/spatial evidence.
        score = (
            0.55
            + 0.35 * unknown_fit
        )

    return float(np.clip(score, 0.0, 1.0)), True


def _identity_kit_prototype(
    identity: PhysicalIdentity,
    *,
    segment_by_id: Mapping[int, PhysicalTrackSegment],
    kit_prototype_by_segment_id: Mapping[int, np.ndarray],
) -> np.ndarray | None:
    vectors: list[np.ndarray] = []
    weights: list[float] = []

    for segment_id in identity.member_segment_ids:
        vector = kit_prototype_by_segment_id.get(segment_id)
        if vector is None:
            continue
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            continue
        norm = float(np.linalg.norm(vector))
        if norm <= _EPS:
            continue

        segment = segment_by_id[segment_id]
        vectors.append(vector / norm)
        weights.append(
            max(
                1.0,
                min(
                    180.0,
                    float(segment.observed_count),
                ),
            )
        )

    if not vectors:
        return None

    prototype = np.average(
        np.stack(vectors, axis=0),
        axis=0,
        weights=np.asarray(weights, dtype=np.float64),
    )
    norm = float(np.linalg.norm(prototype))
    if norm <= _EPS:
        return None
    return (prototype / norm).astype(np.float32)


def _team_kit_affinity(
    candidate: PhysicalIdentity,
    *,
    identities: Sequence[PhysicalIdentity],
    identity_kit: Mapping[int, np.ndarray | None],
) -> float | None:
    vector = identity_kit.get(candidate.identity_id)
    if vector is None:
        return None

    per_team: list[float] = []

    for team in (
        RelativeTeamLabel.TEAM_A,
        RelativeTeamLabel.TEAM_B,
    ):
        similarities = []
        for peer in identities:
            if peer.identity_id == candidate.identity_id:
                continue
            if peer.team_label is not team:
                continue
            peer_vector = identity_kit.get(peer.identity_id)
            if peer_vector is None:
                continue
            similarities.append(
                float(
                    np.clip(
                        np.dot(vector, peer_vector),
                        -1.0,
                        1.0,
                    )
                )
            )

        if similarities:
            per_team.append(
                float(
                    np.median(
                        np.asarray(
                            similarities,
                            dtype=np.float64,
                        )
                    )
                )
            )

    if not per_team:
        return None

    # A real player only needs to resemble one of the two player-kit groups.
    return max(per_team)


def _relative_outlier_scores(
    affinities: Mapping[int, float | None],
) -> dict[int, float]:
    available = [
        float(value)
        for value in affinities.values()
        if value is not None and math.isfinite(value)
    ]

    if len(available) < 2:
        return {
            identity_id: 0.5
            for identity_id in affinities
        }

    low = min(available)
    high = max(available)
    spread = high - low

    if spread <= 1e-6:
        return {
            identity_id: 0.5
            for identity_id in affinities
        }

    result: dict[int, float] = {}
    for identity_id, value in affinities.items():
        if value is None or not math.isfinite(value):
            # Missing appearance is mildly referee-like, but goalkeeper
            # avoidance / roster structure still have to support it.
            result[identity_id] = 0.60
            continue

        result[identity_id] = float(
            np.clip(
                (high - float(value)) / spread,
                0.0,
                1.0,
            )
        )

    return result


def _identity_spatial_features(
    identity: PhysicalIdentity,
    *,
    segment_by_id: Mapping[int, PhysicalTrackSegment],
    field_length: float,
    field_width: float,
    goalkeeper_goal_radius_m: float,
    central_min_goal_distance_m: float,
) -> tuple[float, float, int]:
    half_length = field_length / 2.0
    half_width = field_width / 2.0

    goal_zone = 0
    central = 0
    strict_inside = 0

    for segment_id in identity.member_segment_ids:
        segment = segment_by_id[segment_id]
        for row in segment.observations:
            if not (
                -half_length <= row.x <= half_length
                and -half_width <= row.z <= half_width
            ):
                continue

            strict_inside += 1

            left_goal_distance = math.hypot(
                row.x + half_length,
                row.z,
            )
            right_goal_distance = math.hypot(
                row.x - half_length,
                row.z,
            )
            nearest_goal = min(
                left_goal_distance,
                right_goal_distance,
            )

            if nearest_goal <= goalkeeper_goal_radius_m:
                goal_zone += 1
            if nearest_goal >= central_min_goal_distance_m:
                central += 1

    if strict_inside <= 0:
        return 0.0, 0.0, 0

    return (
        goal_zone / float(strict_inside),
        central / float(strict_inside),
        strict_inside,
    )
