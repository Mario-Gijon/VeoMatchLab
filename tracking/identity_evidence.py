"""Sprint C.2: interpretable identity-evidence fusion for audit.

This is deliberately NOT the V2.8 solver.

It combines:
  * segment-aware ReID,
  * relative team compatibility,
  * simultaneous-observation conflict,
  * transition geometry near a raw-track handover,
  * temporal gap/overlap.

The result is an audit label and component scores.  No tracks are merged and no
canonical IDs are produced.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

from .appearance_bank import (
    AppearanceBankConfig,
    TrackPairSimilarity,
    compare_track_banks,
)
from .appearance_segments import (
    AppearanceSegment,
    TrackAppearanceSegmentation,
)
from .team_similarity import RelativeTeamLabel

EPS = 1e-12


class IdentityEvidenceLabel(str, Enum):
    HARD_CONFLICT = "hard_conflict"
    STRONG_CONFLICT = "strong_conflict"
    STRONG_SUPPORT = "strong_support"
    MODERATE_SUPPORT = "moderate_support"
    AMBIGUOUS = "ambiguous"


@dataclass(frozen=True)
class ObservedTrackPoint:
    track_id: int
    frame_index: int
    timestamp: float
    x: float
    z: float


@dataclass(frozen=True)
class GeometryPairEvidence:
    track_a: int
    track_b: int

    observed_a: int
    observed_b: int

    simultaneous_count: int
    simultaneous_fraction: float
    simultaneous_median_distance_m: float | None
    simultaneous_q90_distance_m: float | None

    earlier_track: int
    later_track: int
    support_gap_s: float

    transition_distance_m: float | None
    transition_geometry_score: float

    simultaneous_conflict: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SegmentPairEvidence:
    segment_a: str
    segment_b: str
    reid: TrackPairSimilarity

    def to_dict(self) -> dict[str, object]:
        return {
            "segment_a": self.segment_a,
            "segment_b": self.segment_b,
            "reid": self.reid.to_dict(),
        }


@dataclass(frozen=True)
class IdentityPairEvidence:
    track_a: int
    track_b: int

    team_a: RelativeTeamLabel
    team_b: RelativeTeamLabel
    team_compatibility: str

    best_segment_pair: SegmentPairEvidence
    geometry: GeometryPairEvidence

    reid_score: float
    geometry_score: float
    team_score: float
    fused_support_score: float

    label: IdentityEvidenceLabel
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "track_a": self.track_a,
            "track_b": self.track_b,
            "team_a": self.team_a.value,
            "team_b": self.team_b.value,
            "team_compatibility": self.team_compatibility,
            "best_segment_pair": (
                self.best_segment_pair.to_dict()
            ),
            "geometry": self.geometry.to_dict(),
            "reid_score": self.reid_score,
            "geometry_score": self.geometry_score,
            "team_score": self.team_score,
            "fused_support_score": self.fused_support_score,
            "label": self.label.value,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class IdentityEvidenceConfig:
    simultaneous_conflict_min_count: int = 8
    simultaneous_conflict_min_fraction: float = 0.35
    simultaneous_conflict_min_distance_m: float = 1.0

    transition_window_s: float = 1.5
    transition_distance_sigma_m: float = 4.0

    # Long temporal gaps reduce geometric confidence, but do not make visual
    # identity impossible.
    temporal_gap_soft_s: float = 1.5
    temporal_gap_hard_s: float = 6.0

    # Audit-only fusion weights.  These are intentionally not production merge
    # thresholds.
    reid_weight: float = 0.55
    geometry_weight: float = 0.35
    team_weight: float = 0.10

    strong_support_threshold: float = 0.78
    moderate_support_threshold: float = 0.67

    def __post_init__(self) -> None:
        if self.simultaneous_conflict_min_count < 1:
            raise ValueError(
                "simultaneous_conflict_min_count must be >= 1"
            )
        if self.transition_window_s <= 0.0:
            raise ValueError(
                "transition_window_s must be > 0"
            )
        if self.transition_distance_sigma_m <= 0.0:
            raise ValueError(
                "transition_distance_sigma_m must be > 0"
            )
        if not (
            0.0
            <= self.temporal_gap_soft_s
            <= self.temporal_gap_hard_s
        ):
            raise ValueError(
                "temporal gap limits are invalid"
            )
        for name in (
            "simultaneous_conflict_min_fraction",
            "reid_weight",
            "geometry_weight",
            "team_weight",
            "strong_support_threshold",
            "moderate_support_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

        if abs(
            self.reid_weight
            + self.geometry_weight
            + self.team_weight
            - 1.0
        ) > 1e-9:
            raise ValueError(
                "fusion weights must sum to 1"
            )


def build_identity_pair_evidence(
    track_a: int,
    track_b: int,
    *,
    segmentations: Mapping[
        int,
        TrackAppearanceSegmentation,
    ],
    observed_points: Mapping[
        int,
        Sequence[ObservedTrackPoint],
    ],
    team_labels: Mapping[
        int,
        RelativeTeamLabel,
    ],
    bank_config: AppearanceBankConfig | None = None,
    config: IdentityEvidenceConfig | None = None,
) -> IdentityPairEvidence:
    cfg = config or IdentityEvidenceConfig()
    bank_cfg = bank_config or AppearanceBankConfig()

    segmentation_a = segmentations[
        track_a
    ]
    segmentation_b = segmentations[
        track_b
    ]

    best_segment = _best_segment_pair(
        segmentation_a.segments,
        segmentation_b.segments,
        bank_config=bank_cfg,
    )

    geometry = compute_geometry_pair_evidence(
        track_a,
        track_b,
        observed_points=observed_points,
        config=cfg,
    )

    team_a = team_labels.get(
        track_a,
        RelativeTeamLabel.UNKNOWN,
    )
    team_b = team_labels.get(
        track_b,
        RelativeTeamLabel.UNKNOWN,
    )

    team_compatibility = _team_compatibility(
        team_a,
        team_b,
    )

    reid_score = _reid_support_score(
        best_segment.reid
    )
    geometry_score = (
        geometry.transition_geometry_score
    )
    team_score = (
        1.0
        if team_compatibility == "same_team"
        else 0.5
        if team_compatibility == "unknown"
        else 0.0
    )

    fused = (
        cfg.reid_weight * reid_score
        + cfg.geometry_weight * geometry_score
        + cfg.team_weight * team_score
    )

    if team_compatibility == "team_conflict":
        label = IdentityEvidenceLabel.HARD_CONFLICT
        reason = "reliable_team_conflict"
    elif geometry.simultaneous_conflict:
        label = IdentityEvidenceLabel.STRONG_CONFLICT
        reason = "sustained_simultaneous_observed_conflict"
    elif fused >= cfg.strong_support_threshold:
        label = IdentityEvidenceLabel.STRONG_SUPPORT
        reason = "high_fused_support"
    elif fused >= cfg.moderate_support_threshold:
        label = IdentityEvidenceLabel.MODERATE_SUPPORT
        reason = "moderate_fused_support"
    else:
        label = IdentityEvidenceLabel.AMBIGUOUS
        reason = "insufficient_combined_evidence"

    return IdentityPairEvidence(
        track_a=track_a,
        track_b=track_b,
        team_a=team_a,
        team_b=team_b,
        team_compatibility=team_compatibility,
        best_segment_pair=best_segment,
        geometry=geometry,
        reid_score=float(reid_score),
        geometry_score=float(
            geometry_score
        ),
        team_score=float(team_score),
        fused_support_score=float(fused),
        label=label,
        reason=reason,
    )


def compute_geometry_pair_evidence(
    track_a: int,
    track_b: int,
    *,
    observed_points: Mapping[
        int,
        Sequence[ObservedTrackPoint],
    ],
    config: IdentityEvidenceConfig | None = None,
) -> GeometryPairEvidence:
    """Compute geometry evidence for two raw track IDs.

    This compatibility wrapper is appropriate when ``track_a`` and ``track_b``
    are distinct mapping keys.

    For two temporal segments that originate from the SAME raw track ID, use
    :func:`compute_geometry_pair_evidence_from_points` instead.  A mapping
    cannot contain two independent values under the same integer key.
    """

    rows_a = observed_points.get(
        track_a,
        (),
    )
    rows_b = observed_points.get(
        track_b,
        (),
    )

    return compute_geometry_pair_evidence_from_points(
        track_a,
        track_b,
        rows_a=rows_a,
        rows_b=rows_b,
        config=config,
    )


def compute_geometry_pair_evidence_from_points(
    track_a: int,
    track_b: int,
    *,
    rows_a: Sequence[ObservedTrackPoint],
    rows_b: Sequence[ObservedTrackPoint],
    config: IdentityEvidenceConfig | None = None,
) -> GeometryPairEvidence:
    """Compute geometry evidence from two explicit observation sequences.

    Unlike the mapping-based wrapper, this function safely compares two
    appearance/temporal segments that came from the same raw tracker ID.

    This matters for C.2/V2.8 appearance splits.  Code such as::

        {13: segment_0_points, 13: segment_1_points}

    silently keeps only the second value in Python.  The old audit path
    therefore compared a segment against itself and produced artificial
    ``geometry=1.0`` for same-raw visual splits.  Passing the two sequences
    explicitly avoids that aliasing.
    """

    cfg = config or IdentityEvidenceConfig()

    rows_a = tuple(
        sorted(
            rows_a,
            key=lambda row: (
                row.frame_index,
                row.timestamp,
            ),
        )
    )
    rows_b = tuple(
        sorted(
            rows_b,
            key=lambda row: (
                row.frame_index,
                row.timestamp,
            ),
        )
    )

    if not rows_a or not rows_b:
        raise ValueError(
            "geometry evidence requires observed points for both tracks"
        )

    by_frame_a = {
        row.frame_index: row
        for row in rows_a
    }
    by_frame_b = {
        row.frame_index: row
        for row in rows_b
    }

    common_frames = sorted(
        set(by_frame_a)
        & set(by_frame_b)
    )

    distances = np.asarray(
        [
            math.hypot(
                by_frame_a[frame].x
                - by_frame_b[frame].x,
                by_frame_a[frame].z
                - by_frame_b[frame].z,
            )
            for frame in common_frames
        ],
        dtype=np.float64,
    )

    simultaneous_count = len(
        common_frames
    )
    simultaneous_fraction = (
        simultaneous_count
        / max(
            1,
            min(
                len(rows_a),
                len(rows_b),
            ),
        )
    )

    simultaneous_median = (
        float(np.median(distances))
        if len(distances)
        else None
    )
    simultaneous_q90 = (
        float(
            np.quantile(
                distances,
                0.90,
            )
        )
        if len(distances)
        else None
    )

    simultaneous_conflict = (
        simultaneous_count
        >= cfg.simultaneous_conflict_min_count
        and simultaneous_fraction
        >= cfg.simultaneous_conflict_min_fraction
        and simultaneous_median is not None
        and simultaneous_median
        >= cfg.simultaneous_conflict_min_distance_m
    )

    # Define earlier/later by the end of each EXPLICIT sequence.  This remains
    # well-defined even when track_a == track_b.
    if (
        rows_a[-1].timestamp,
        rows_a[-1].frame_index,
    ) <= (
        rows_b[-1].timestamp,
        rows_b[-1].frame_index,
    ):
        earlier_track = track_a
        later_track = track_b
        earlier = rows_a
        later = rows_b
    else:
        earlier_track = track_b
        later_track = track_a
        earlier = rows_b
        later = rows_a

    support_gap_s = (
        later[0].timestamp
        - earlier[-1].timestamp
    )

    transition_distance = _transition_distance(
        earlier,
        later,
        window_s=cfg.transition_window_s,
    )

    if transition_distance is None:
        distance_score = 0.0
    else:
        distance_score = math.exp(
            -0.5
            * (
                transition_distance
                / cfg.transition_distance_sigma_m
            ) ** 2
        )

    gap_score = _temporal_gap_score(
        support_gap_s,
        config=cfg,
    )

    transition_geometry_score = (
        distance_score
        * gap_score
    )

    return GeometryPairEvidence(
        track_a=track_a,
        track_b=track_b,
        observed_a=len(rows_a),
        observed_b=len(rows_b),
        simultaneous_count=simultaneous_count,
        simultaneous_fraction=float(
            simultaneous_fraction
        ),
        simultaneous_median_distance_m=(
            simultaneous_median
        ),
        simultaneous_q90_distance_m=(
            simultaneous_q90
        ),
        earlier_track=earlier_track,
        later_track=later_track,
        support_gap_s=float(
            support_gap_s
        ),
        transition_distance_m=(
            float(transition_distance)
            if transition_distance is not None
            else None
        ),
        transition_geometry_score=float(
            transition_geometry_score
        ),
        simultaneous_conflict=bool(
            simultaneous_conflict
        ),
    )

def _best_segment_pair(
    segments_a: Sequence[
        AppearanceSegment
    ],
    segments_b: Sequence[
        AppearanceSegment
    ],
    *,
    bank_config: AppearanceBankConfig,
) -> SegmentPairEvidence:
    best: tuple[
        tuple[float, float, float],
        SegmentPairEvidence,
    ] | None = None

    for segment_a in segments_a:
        for segment_b in segments_b:
            metrics = compare_track_banks(
                segment_a.as_bank(),
                segment_b.as_bank(),
                config=bank_config,
            )

            evidence = SegmentPairEvidence(
                segment_a=segment_a.key,
                segment_b=segment_b.key,
                reid=metrics,
            )

            ranking = (
                metrics.bidirectional_best_median,
                metrics.unique_topk_similarity,
                metrics.all_pairs_q90,
            )

            if (
                best is None
                or ranking > best[0]
            ):
                best = (
                    ranking,
                    evidence,
                )

    if best is None:
        raise ValueError(
            "segment-aware comparison requires segments on both tracks"
        )

    return best[1]


def _reid_support_score(
    metrics: TrackPairSimilarity,
) -> float:
    # Use complementary robust metrics; BEST is deliberately excluded because
    # one lucky crop pair can be misleading for same-kit football players.
    value = (
        0.50
        * metrics.bidirectional_best_median
        + 0.35
        * metrics.unique_topk_similarity
        + 0.15
        * metrics.all_pairs_q90
    )

    # Cosine may technically be negative; map audit support to [0, 1].
    return float(
        min(
            1.0,
            max(
                0.0,
                value,
            ),
        )
    )


def _team_compatibility(
    team_a: RelativeTeamLabel,
    team_b: RelativeTeamLabel,
) -> str:
    known = {
        RelativeTeamLabel.TEAM_A,
        RelativeTeamLabel.TEAM_B,
    }

    if (
        team_a in known
        and team_b in known
    ):
        return (
            "same_team"
            if team_a is team_b
            else "team_conflict"
        )

    return "unknown"


def _transition_distance(
    earlier: Sequence[
        ObservedTrackPoint
    ],
    later: Sequence[
        ObservedTrackPoint
    ],
    *,
    window_s: float,
) -> float | None:
    transition_time = (
        earlier[-1].timestamp
    )

    earlier_rows = [
        row
        for row in earlier
        if (
            transition_time
            - window_s
            <= row.timestamp
            <= transition_time
        )
    ]
    later_rows = [
        row
        for row in later
        if (
            transition_time
            - window_s
            <= row.timestamp
            <= transition_time
            + window_s
        )
    ]

    if not earlier_rows or not later_rows:
        # Fallback to support endpoints for a non-overlap handover.
        return math.hypot(
            earlier[-1].x - later[0].x,
            earlier[-1].z - later[0].z,
        )

    later_by_frame = {
        row.frame_index: row
        for row in later_rows
    }

    distances: list[float] = []

    for row in earlier_rows:
        match = later_by_frame.get(
            row.frame_index
        )
        if match is not None:
            distances.append(
                math.hypot(
                    row.x - match.x,
                    row.z - match.z,
                )
            )

    if distances:
        return float(
            np.median(
                np.asarray(
                    distances,
                    dtype=np.float64,
                )
            )
        )

    # No exact shared frames in the transition window: closest endpoint pair.
    best = math.inf
    for left in earlier_rows[-4:]:
        for right in later_rows[:4]:
            best = min(
                best,
                math.hypot(
                    left.x - right.x,
                    left.z - right.z,
                ),
            )

    return (
        float(best)
        if math.isfinite(best)
        else None
    )


def _temporal_gap_score(
    support_gap_s: float,
    *,
    config: IdentityEvidenceConfig,
) -> float:
    # Overlap receives full temporal continuity here. Sustained overlap is
    # handled separately by simultaneous_conflict.
    if support_gap_s <= config.temporal_gap_soft_s:
        return 1.0

    if support_gap_s >= config.temporal_gap_hard_s:
        return 0.0

    span = (
        config.temporal_gap_hard_s
        - config.temporal_gap_soft_s
    )

    return float(
        1.0
        - (
            support_gap_s
            - config.temporal_gap_soft_s
        )
        / span
    )
