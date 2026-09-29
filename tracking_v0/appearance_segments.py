"""Sprint C.2: conservative temporal segmentation of raw-track appearance banks.

A raw tracker ID can contain more than one physical person.  ReID C.1 showed
that averaging an entire raw ID is therefore unsafe.

This module keeps the original raw track untouched and creates *visual evidence
segments* only.  A split is accepted when two temporally contiguous groups are
internally coherent and substantially less similar to one another than they are
within themselves.

The output is diagnostic evidence for later identity fusion; it is not a new
tracker and does not publish canonical identities.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence

import numpy as np

from .appearance_bank import ReIDEmbedding, TrackAppearanceBank
from .team_appearance import AppearanceLabel

EPS = 1e-12


@dataclass(frozen=True)
class AppearanceSegmentationConfig:
    min_embeddings_per_segment: int = 2
    max_segments_per_track: int = 3

    # Split evidence: internal similarity should exceed cross-boundary
    # similarity by at least this amount.
    min_split_gain: float = 0.10
    max_cross_similarity: float = 0.76
    min_child_coherence: float = 0.58

    # B.1 MIXED tracks are already suspicious, so allow a slightly smaller
    # ReID gain before accepting a split. This remains conservative: the
    # cross-similarity and child-coherence checks still apply.
    mixed_split_gain_scale: float = 0.78

    def __post_init__(self) -> None:
        if self.min_embeddings_per_segment < 2:
            raise ValueError(
                "min_embeddings_per_segment must be >= 2"
            )
        if self.max_segments_per_track < 1:
            raise ValueError(
                "max_segments_per_track must be >= 1"
            )
        for name in (
            "min_split_gain",
            "max_cross_similarity",
            "min_child_coherence",
            "mixed_split_gain_scale",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class AppearanceSegment:
    track_id: int
    segment_index: int

    start_frame: int
    end_frame: int
    embedding_count: int

    median_internal_similarity: float
    q10_internal_similarity: float

    prototype: np.ndarray
    embeddings: tuple[ReIDEmbedding, ...]

    def __post_init__(self) -> None:
        prototype = np.asarray(
            self.prototype,
            dtype=np.float32,
        )
        if prototype.ndim != 1:
            raise ValueError("segment prototype must be 1D")
        norm = float(np.linalg.norm(prototype))
        if not np.isfinite(prototype).all() or norm <= EPS:
            raise ValueError("invalid segment prototype")
        object.__setattr__(
            self,
            "prototype",
            prototype / norm,
        )

    @property
    def key(self) -> str:
        return f"{self.track_id}:{self.segment_index}"

    def as_bank(self) -> TrackAppearanceBank:
        return TrackAppearanceBank(
            track_id=self.track_id,
            embeddings=self.embeddings,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "segment_index": self.segment_index,
            "key": self.key,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "embedding_count": self.embedding_count,
            "median_internal_similarity": self.median_internal_similarity,
            "q10_internal_similarity": self.q10_internal_similarity,
        }


@dataclass(frozen=True)
class SplitEvidence:
    track_id: int
    boundary_after_frame: int

    left_count: int
    right_count: int

    left_coherence: float
    right_coherence: float
    cross_similarity: float
    split_gain: float

    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrackAppearanceSegmentation:
    track_id: int
    source_label: AppearanceLabel
    segments: tuple[AppearanceSegment, ...]
    accepted_splits: tuple[SplitEvidence, ...]
    best_rejected_split: SplitEvidence | None

    @property
    def segmented(self) -> bool:
        return len(self.segments) > 1

    def to_dict(self) -> dict[str, object]:
        return {
            "track_id": self.track_id,
            "source_label": self.source_label.value,
            "segmented": self.segmented,
            "segments": [
                segment.to_dict()
                for segment in self.segments
            ],
            "accepted_splits": [
                split.to_dict()
                for split in self.accepted_splits
            ],
            "best_rejected_split": (
                self.best_rejected_split.to_dict()
                if self.best_rejected_split is not None
                else None
            ),
        }


def segment_appearance_bank(
    bank: TrackAppearanceBank,
    *,
    source_label: AppearanceLabel = AppearanceLabel.UNKNOWN,
    config: AppearanceSegmentationConfig | None = None,
) -> TrackAppearanceSegmentation:
    cfg = config or AppearanceSegmentationConfig()

    embeddings = tuple(
        sorted(
            bank.embeddings,
            key=lambda item: (
                item.frame_index,
                -item.quality,
            ),
        )
    )

    if not embeddings:
        raise ValueError(
            "appearance segmentation requires a non-empty bank"
        )

    accepted: list[SplitEvidence] = []
    rejected: list[SplitEvidence] = []

    ranges: list[tuple[int, int]] = [
        (0, len(embeddings))
    ]

    while (
        len(ranges) < cfg.max_segments_per_track
    ):
        best_candidate: tuple[
            float,
            int,
            int,
            SplitEvidence,
        ] | None = None

        for range_index, (start, end) in enumerate(ranges):
            candidate = _best_split_for_range(
                bank.track_id,
                embeddings,
                start=start,
                end=end,
                source_label=source_label,
                config=cfg,
            )
            if candidate is None:
                continue

            cut, evidence = candidate

            if not evidence.accepted:
                rejected.append(evidence)
                continue

            ranking = (
                evidence.split_gain,
                -evidence.cross_similarity,
                -evidence.boundary_after_frame,
            )

            if (
                best_candidate is None
                or ranking
                > (
                    best_candidate[0],
                    best_candidate[1],
                    best_candidate[2],
                )
            ):
                best_candidate = (
                    ranking[0],
                    ranking[1],
                    ranking[2],
                    evidence,
                )
                best_range_index = range_index
                best_cut = cut

        if best_candidate is None:
            break

        start, end = ranges[
            best_range_index
        ]
        ranges[
            best_range_index:best_range_index + 1
        ] = [
            (start, best_cut),
            (best_cut, end),
        ]
        accepted.append(
            best_candidate[3]
        )

    segments = tuple(
        _make_segment(
            bank.track_id,
            segment_index,
            embeddings[start:end],
        )
        for segment_index, (start, end) in enumerate(
            sorted(
                ranges,
                key=lambda item: item[0],
            )
        )
    )

    best_rejected = (
        max(
            rejected,
            key=lambda item: (
                item.split_gain,
                -item.cross_similarity,
            ),
        )
        if rejected
        else None
    )

    return TrackAppearanceSegmentation(
        track_id=bank.track_id,
        source_label=source_label,
        segments=segments,
        accepted_splits=tuple(
            sorted(
                accepted,
                key=lambda item: item.boundary_after_frame,
            )
        ),
        best_rejected_split=best_rejected,
    )


def segment_all_banks(
    banks: Mapping[int, TrackAppearanceBank],
    *,
    source_labels: Mapping[int, AppearanceLabel] | None = None,
    config: AppearanceSegmentationConfig | None = None,
) -> dict[int, TrackAppearanceSegmentation]:
    source_labels = source_labels or {}

    return {
        track_id: segment_appearance_bank(
            bank,
            source_label=source_labels.get(
                track_id,
                AppearanceLabel.UNKNOWN,
            ),
            config=config,
        )
        for track_id, bank in sorted(
            banks.items()
        )
    }


def _best_split_for_range(
    track_id: int,
    embeddings: Sequence[ReIDEmbedding],
    *,
    start: int,
    end: int,
    source_label: AppearanceLabel,
    config: AppearanceSegmentationConfig,
) -> tuple[int, SplitEvidence] | None:
    minimum = config.min_embeddings_per_segment

    if end - start < 2 * minimum:
        return None

    gain_threshold = config.min_split_gain
    if source_label is AppearanceLabel.MIXED:
        gain_threshold *= config.mixed_split_gain_scale

    best: tuple[
        tuple[float, float, float],
        int,
        SplitEvidence,
    ] | None = None

    for cut in range(
        start + minimum,
        end - minimum + 1,
    ):
        left = embeddings[start:cut]
        right = embeddings[cut:end]

        left_coherence = _median_internal_similarity(
            left
        )
        right_coherence = _median_internal_similarity(
            right
        )
        cross_similarity = _median_cross_similarity(
            left,
            right,
        )

        weighted_within = (
            len(left) * left_coherence
            + len(right) * right_coherence
        ) / (len(left) + len(right))

        split_gain = (
            weighted_within
            - cross_similarity
        )

        accepted = (
            split_gain >= gain_threshold
            and cross_similarity
            <= config.max_cross_similarity
            and min(
                left_coherence,
                right_coherence,
            ) >= config.min_child_coherence
        )

        if split_gain < gain_threshold:
            reason = (
                "gain_below_threshold:"
                f"{split_gain:.3f}<{gain_threshold:.3f}"
            )
        elif (
            cross_similarity
            > config.max_cross_similarity
        ):
            reason = (
                "cross_similarity_too_high:"
                f"{cross_similarity:.3f}>"
                f"{config.max_cross_similarity:.3f}"
            )
        elif min(
            left_coherence,
            right_coherence,
        ) < config.min_child_coherence:
            reason = (
                "child_coherence_too_low:"
                f"{min(left_coherence, right_coherence):.3f}<"
                f"{config.min_child_coherence:.3f}"
            )
        else:
            reason = "accepted"

        evidence = SplitEvidence(
            track_id=track_id,
            boundary_after_frame=left[-1].frame_index,
            left_count=len(left),
            right_count=len(right),
            left_coherence=float(left_coherence),
            right_coherence=float(right_coherence),
            cross_similarity=float(
                cross_similarity
            ),
            split_gain=float(split_gain),
            accepted=accepted,
            reason=reason,
        )

        ranking = (
            split_gain,
            -cross_similarity,
            float(
                min(
                    left_coherence,
                    right_coherence,
                )
            ),
        )

        if (
            best is None
            or ranking > best[0]
        ):
            best = (
                ranking,
                cut,
                evidence,
            )

    assert best is not None
    return best[1], best[2]


def _make_segment(
    track_id: int,
    segment_index: int,
    embeddings: Sequence[ReIDEmbedding],
) -> AppearanceSegment:
    if not embeddings:
        raise ValueError("segment must not be empty")

    prototype = _quality_weighted_prototype(
        embeddings
    )
    internal = _pairwise_upper_similarities(
        embeddings
    )

    if len(internal) == 0:
        median_similarity = 1.0
        q10_similarity = 1.0
    else:
        median_similarity = float(
            np.median(internal)
        )
        q10_similarity = float(
            np.quantile(
                internal,
                0.10,
            )
        )

    return AppearanceSegment(
        track_id=track_id,
        segment_index=segment_index,
        start_frame=embeddings[0].frame_index,
        end_frame=embeddings[-1].frame_index,
        embedding_count=len(embeddings),
        median_internal_similarity=median_similarity,
        q10_internal_similarity=q10_similarity,
        prototype=prototype,
        embeddings=tuple(embeddings),
    )


def _quality_weighted_prototype(
    embeddings: Sequence[ReIDEmbedding],
) -> np.ndarray:
    matrix = np.stack(
        [item.vector for item in embeddings],
        axis=0,
    ).astype(np.float64)
    weights = np.asarray(
        [
            max(EPS, item.quality)
            for item in embeddings
        ],
        dtype=np.float64,
    )

    prototype = np.average(
        matrix,
        axis=0,
        weights=weights,
    )
    norm = float(
        np.linalg.norm(prototype)
    )
    if norm <= EPS:
        raise ValueError(
            "segment prototype has zero norm"
        )

    return (
        prototype / norm
    ).astype(np.float32)


def _median_internal_similarity(
    embeddings: Sequence[ReIDEmbedding],
) -> float:
    values = _pairwise_upper_similarities(
        embeddings
    )
    if len(values) == 0:
        return 1.0
    return float(
        np.median(values)
    )


def _pairwise_upper_similarities(
    embeddings: Sequence[ReIDEmbedding],
) -> np.ndarray:
    if len(embeddings) < 2:
        return np.empty(
            (0,),
            dtype=np.float64,
        )

    matrix = np.stack(
        [item.vector for item in embeddings],
        axis=0,
    ).astype(np.float64)

    similarity = matrix @ matrix.T
    upper = np.triu_indices(
        len(embeddings),
        k=1,
    )
    return np.clip(
        similarity[upper],
        -1.0,
        1.0,
    )


def _median_cross_similarity(
    left: Sequence[ReIDEmbedding],
    right: Sequence[ReIDEmbedding],
) -> float:
    a = np.stack(
        [item.vector for item in left],
        axis=0,
    ).astype(np.float64)
    b = np.stack(
        [item.vector for item in right],
        axis=0,
    ).astype(np.float64)

    values = (a @ b.T).ravel()

    return float(
        np.median(
            np.clip(
                values,
                -1.0,
                1.0,
            )
        )
    )
