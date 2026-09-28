"""Relative TEAM_A / TEAM_B similarity from Sprint B.1 appearance descriptors.

This module deliberately avoids literal colour rules such as "white" or "red".
Instead it learns one robust descriptor prototype for each dominant team cluster
and asks a relative question for every crop/track:

    is this appearance closer to TEAM_A or TEAM_B, and by how much?

A decision is only emitted when both:
  * the best similarity is high enough, and
  * the margin over the other team is large enough.

Otherwise the answer is UNKNOWN.

SPECIAL / goalkeeper / referee-looking crops are never force-assigned to a team
from colour alone.  Team membership for those roles should be inferred later
from role + tactical side / roster context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

from .team_appearance import (
    AppearanceCluster,
    AppearanceLabel,
    CropAppearance,
    KitAppearanceResult,
)

EPS = 1e-9


class RelativeTeamLabel(str, Enum):
    TEAM_A = "team_a"
    TEAM_B = "team_b"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TeamSimilarityConfig:
    min_crop_similarity: float = 0.42
    min_crop_margin: float = 0.08

    min_track_crops: int = 3
    min_track_similarity: float = 0.48
    min_track_margin: float = 0.10
    min_track_vote_share: float = 0.65

    # If B.1 already called a crop/track SPECIAL, do not override that semantic
    # group with a forced team decision based only on relative proximity.
    preserve_special: bool = True

    def __post_init__(self) -> None:
        for name in (
            "min_crop_similarity",
            "min_crop_margin",
            "min_track_similarity",
            "min_track_margin",
            "min_track_vote_share",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.min_track_crops < 1:
            raise ValueError("min_track_crops must be >= 1")


@dataclass(frozen=True)
class TeamPrototypes:
    team_a: np.ndarray
    team_b: np.ndarray

    team_a_cluster_id: int
    team_b_cluster_id: int

    def __post_init__(self) -> None:
        a = np.asarray(self.team_a, dtype=np.float64)
        b = np.asarray(self.team_b, dtype=np.float64)

        if a.ndim != 1 or b.ndim != 1 or a.shape != b.shape:
            raise ValueError("team prototypes must be equal-length 1D vectors")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError("team prototypes must be finite")
        if np.linalg.norm(a) <= EPS or np.linalg.norm(b) <= EPS:
            raise ValueError("team prototypes must have nonzero norm")


@dataclass(frozen=True)
class CropTeamSimilarity:
    track_id: int
    frame_index: int

    team_a_similarity: float
    team_b_similarity: float
    best_similarity: float
    margin: float

    label: RelativeTeamLabel
    source_appearance_label: AppearanceLabel

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "frame_index": self.frame_index,
            "team_a_similarity": self.team_a_similarity,
            "team_b_similarity": self.team_b_similarity,
            "best_similarity": self.best_similarity,
            "margin": self.margin,
            "label": self.label.value,
            "source_appearance_label": self.source_appearance_label.value,
        }


@dataclass(frozen=True)
class TrackTeamSimilarity:
    track_id: int

    team_a_similarity: float
    team_b_similarity: float
    best_similarity: float
    margin: float

    team_a_vote_share: float
    team_b_vote_share: float

    crop_count: int
    label: RelativeTeamLabel
    source_appearance_label: AppearanceLabel

    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self) | {
            "label": self.label.value,
            "source_appearance_label": self.source_appearance_label.value,
        }


@dataclass(frozen=True)
class TeamSimilarityResult:
    prototypes: TeamPrototypes
    crops: tuple[CropTeamSimilarity, ...]
    tracks: Mapping[int, TrackTeamSimilarity]

    def to_dict(self) -> dict[str, object]:
        return {
            "team_a_cluster_id": self.prototypes.team_a_cluster_id,
            "team_b_cluster_id": self.prototypes.team_b_cluster_id,
            "tracks": {
                str(track_id): row.to_dict()
                for track_id, row in sorted(self.tracks.items())
            },
        }


def build_team_prototypes(
    crops: Sequence[CropAppearance],
    appearance_result: KitAppearanceResult,
) -> TeamPrototypes:
    """Build robust mean-normalized descriptors for B.1 TEAM_A / TEAM_B."""

    cluster_by_id = {
        cluster.cluster_id: cluster
        for cluster in appearance_result.clusters
    }

    team_a_cluster = _single_cluster_for_label(
        appearance_result.clusters,
        AppearanceLabel.TEAM_A,
    )
    team_b_cluster = _single_cluster_for_label(
        appearance_result.clusters,
        AppearanceLabel.TEAM_B,
    )

    if len(crops) != len(appearance_result.crop_cluster_ids):
        raise ValueError("crop count does not match appearance_result labels")

    team_a_rows = [
        crop
        for crop, cluster_id in zip(
            crops,
            appearance_result.crop_cluster_ids,
        )
        if cluster_id == team_a_cluster.cluster_id
    ]
    team_b_rows = [
        crop
        for crop, cluster_id in zip(
            crops,
            appearance_result.crop_cluster_ids,
        )
        if cluster_id == team_b_cluster.cluster_id
    ]

    if not team_a_rows or not team_b_rows:
        raise ValueError("both team clusters must contain crops")

    return TeamPrototypes(
        team_a=_robust_weighted_prototype(team_a_rows),
        team_b=_robust_weighted_prototype(team_b_rows),
        team_a_cluster_id=team_a_cluster.cluster_id,
        team_b_cluster_id=team_b_cluster.cluster_id,
    )


def compute_team_similarity(
    crops: Sequence[CropAppearance],
    appearance_result: KitAppearanceResult,
    *,
    config: TeamSimilarityConfig | None = None,
) -> TeamSimilarityResult:
    cfg = config or TeamSimilarityConfig()
    prototypes = build_team_prototypes(
        crops,
        appearance_result,
    )

    source_track_labels = {
        track_id: decision.label
        for track_id, decision in appearance_result.track_decisions.items()
    }

    crop_rows: list[CropTeamSimilarity] = []
    for crop in crops:
        source_label = source_track_labels.get(
            crop.track_id,
            AppearanceLabel.UNKNOWN,
        )
        crop_rows.append(
            classify_crop_relative_to_teams(
                crop,
                prototypes,
                source_appearance_label=source_label,
                config=cfg,
            )
        )

    grouped: dict[int, list[CropTeamSimilarity]] = {}
    for row in crop_rows:
        grouped.setdefault(
            row.track_id,
            [],
        ).append(row)

    tracks = {
        track_id: aggregate_track_team_similarity(
            track_id,
            rows,
            source_appearance_label=source_track_labels.get(
                track_id,
                AppearanceLabel.UNKNOWN,
            ),
            config=cfg,
        )
        for track_id, rows in sorted(grouped.items())
    }

    return TeamSimilarityResult(
        prototypes=prototypes,
        crops=tuple(crop_rows),
        tracks=tracks,
    )


def classify_crop_relative_to_teams(
    crop: CropAppearance,
    prototypes: TeamPrototypes,
    *,
    source_appearance_label: AppearanceLabel = AppearanceLabel.UNKNOWN,
    config: TeamSimilarityConfig | None = None,
) -> CropTeamSimilarity:
    cfg = config or TeamSimilarityConfig()

    a = _cosine01(crop.descriptor, prototypes.team_a)
    b = _cosine01(crop.descriptor, prototypes.team_b)

    if a >= b:
        best_label = RelativeTeamLabel.TEAM_A
        best = a
        second = b
    else:
        best_label = RelativeTeamLabel.TEAM_B
        best = b
        second = a

    margin = best - second

    if (
        cfg.preserve_special
        and source_appearance_label is AppearanceLabel.SPECIAL
    ):
        label = RelativeTeamLabel.UNKNOWN
    elif best < cfg.min_crop_similarity:
        label = RelativeTeamLabel.UNKNOWN
    elif margin < cfg.min_crop_margin:
        label = RelativeTeamLabel.UNKNOWN
    else:
        label = best_label

    return CropTeamSimilarity(
        track_id=crop.track_id,
        frame_index=crop.frame_index,
        team_a_similarity=float(a),
        team_b_similarity=float(b),
        best_similarity=float(best),
        margin=float(margin),
        label=label,
        source_appearance_label=source_appearance_label,
    )


def aggregate_track_team_similarity(
    track_id: int,
    rows: Sequence[CropTeamSimilarity],
    *,
    source_appearance_label: AppearanceLabel,
    config: TeamSimilarityConfig | None = None,
) -> TrackTeamSimilarity:
    cfg = config or TeamSimilarityConfig()

    if not rows:
        raise ValueError("track aggregation requires at least one crop")

    a_values = np.asarray(
        [row.team_a_similarity for row in rows],
        dtype=np.float64,
    )
    b_values = np.asarray(
        [row.team_b_similarity for row in rows],
        dtype=np.float64,
    )

    # Median is deliberately robust to one switched / occluded crop.
    a = float(np.median(a_values))
    b = float(np.median(b_values))

    if a >= b:
        best_label = RelativeTeamLabel.TEAM_A
        best = a
        second = b
    else:
        best_label = RelativeTeamLabel.TEAM_B
        best = b
        second = a

    margin = best - second

    team_a_votes = sum(
        row.label is RelativeTeamLabel.TEAM_A
        for row in rows
    )
    team_b_votes = sum(
        row.label is RelativeTeamLabel.TEAM_B
        for row in rows
    )
    total = len(rows)
    a_share = team_a_votes / total
    b_share = team_b_votes / total

    if cfg.preserve_special and source_appearance_label is AppearanceLabel.SPECIAL:
        label = RelativeTeamLabel.UNKNOWN
        reason = "special_appearance_preserved"
    elif total < cfg.min_track_crops:
        label = RelativeTeamLabel.UNKNOWN
        reason = f"insufficient_crops:{total}"
    elif best < cfg.min_track_similarity:
        label = RelativeTeamLabel.UNKNOWN
        reason = f"similarity_below_threshold:{best:.3f}"
    elif margin < cfg.min_track_margin:
        label = RelativeTeamLabel.UNKNOWN
        reason = f"team_margin_too_small:{margin:.3f}"
    else:
        vote_share = (
            a_share
            if best_label is RelativeTeamLabel.TEAM_A
            else b_share
        )
        if vote_share < cfg.min_track_vote_share:
            label = RelativeTeamLabel.UNKNOWN
            reason = f"vote_share_too_small:{vote_share:.3f}"
        else:
            label = best_label
            reason = "confident"

    return TrackTeamSimilarity(
        track_id=int(track_id),
        team_a_similarity=a,
        team_b_similarity=b,
        best_similarity=float(best),
        margin=float(margin),
        team_a_vote_share=float(a_share),
        team_b_vote_share=float(b_share),
        crop_count=total,
        label=label,
        source_appearance_label=source_appearance_label,
        reason=reason,
    )


def _single_cluster_for_label(
    clusters: Sequence[AppearanceCluster],
    label: AppearanceLabel,
) -> AppearanceCluster:
    matches = [
        cluster
        for cluster in clusters
        if cluster.semantic_label is label
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one {label.value} cluster, got {len(matches)}"
        )
    return matches[0]


def _robust_weighted_prototype(
    rows: Sequence[CropAppearance],
) -> np.ndarray:
    descriptors = np.stack(
        [row.descriptor for row in rows],
        axis=0,
    ).astype(np.float64)
    weights = np.asarray(
        [max(EPS, row.quality) for row in rows],
        dtype=np.float64,
    )

    # First pass weighted mean.
    mean = np.average(
        descriptors,
        axis=0,
        weights=weights,
    )
    mean = _normalize(mean)

    # Trim the most dissimilar 15% to stop one switched raw-track crop from
    # dragging the team prototype.
    similarities = np.asarray(
        [_cosine01(row, mean) for row in descriptors],
        dtype=np.float64,
    )
    keep_count = max(
        1,
        int(math.ceil(len(rows) * 0.85)),
    )
    keep = np.argsort(similarities)[-keep_count:]

    prototype = np.average(
        descriptors[keep],
        axis=0,
        weights=weights[keep],
    )
    return _normalize(prototype)


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or norm <= EPS:
        raise ValueError("cannot normalize appearance vector")
    return vector / norm


def _cosine01(a: np.ndarray, b: np.ndarray) -> float:
    aa = _normalize(np.asarray(a, dtype=np.float64))
    bb = _normalize(np.asarray(b, dtype=np.float64))
    cosine = float(np.dot(aa, bb))
    return float(min(1.0, max(0.0, cosine)))
