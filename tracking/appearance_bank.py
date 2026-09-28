"""Sprint C.1: robust per-track banks of ReID embeddings.

A raw tracker ID may itself be contaminated by an identity switch.  Therefore
this module deliberately keeps multiple embeddings per track rather than
collapsing them into a single mean vector.

Pair comparison exposes several complementary metrics:
  * all-pairs median / q90 / best,
  * bidirectional best-match median,
  * greedy unique top-k matching,
  * fraction of crops that find a strong counterpart.

Within-track coherence is also reported so contaminated raw IDs can be detected
before identity fusion.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping, Sequence

import numpy as np


EPS = 1e-12


@dataclass(frozen=True)
class ReIDEmbedding:
    track_id: int
    frame_index: int
    quality: float
    vector: np.ndarray
    source_path: str | None = None

    def __post_init__(self) -> None:
        vector = np.asarray(
            self.vector,
            dtype=np.float32,
        )
        if vector.ndim != 1:
            raise ValueError("embedding vector must be 1D")
        if not np.isfinite(vector).all():
            raise ValueError("embedding contains non-finite values")
        norm = float(
            np.linalg.norm(vector)
        )
        if norm <= EPS:
            raise ValueError("embedding norm must be > 0")
        if not 0.0 <= self.quality <= 1.0:
            raise ValueError("quality must be in [0, 1]")

        object.__setattr__(
            self,
            "vector",
            vector / norm,
        )


@dataclass(frozen=True)
class AppearanceBankConfig:
    min_quality: float = 0.10
    max_embeddings_per_track: int = 10
    strong_similarity_threshold: float = 0.75
    unique_top_k: int = 3

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_quality <= 1.0:
            raise ValueError("min_quality must be in [0, 1]")
        if self.max_embeddings_per_track < 1:
            raise ValueError("max_embeddings_per_track must be >= 1")
        if not 0.0 <= self.strong_similarity_threshold <= 1.0:
            raise ValueError(
                "strong_similarity_threshold must be in [0, 1]"
            )
        if self.unique_top_k < 1:
            raise ValueError("unique_top_k must be >= 1")


@dataclass(frozen=True)
class TrackAppearanceBank:
    track_id: int
    embeddings: tuple[ReIDEmbedding, ...]

    @property
    def count(self) -> int:
        return len(self.embeddings)

    @property
    def dimension(self) -> int:
        return (
            len(self.embeddings[0].vector)
            if self.embeddings
            else 0
        )

    def matrix(self) -> np.ndarray:
        if not self.embeddings:
            return np.empty(
                (0, 0),
                dtype=np.float32,
            )
        return np.stack(
            [item.vector for item in self.embeddings],
            axis=0,
        )


@dataclass(frozen=True)
class WithinTrackCoherence:
    track_id: int
    embedding_count: int
    median_similarity: float
    q10_similarity: float
    q90_similarity: float
    minimum_similarity: float
    maximum_similarity: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class TrackPairSimilarity:
    track_a: int
    track_b: int
    count_a: int
    count_b: int

    all_pairs_median: float
    all_pairs_q90: float
    best_similarity: float

    row_best_median: float
    column_best_median: float
    bidirectional_best_median: float

    unique_topk_similarity: float
    strong_match_coverage: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_appearance_banks(
    embeddings: Sequence[ReIDEmbedding],
    *,
    config: AppearanceBankConfig | None = None,
) -> dict[int, TrackAppearanceBank]:
    cfg = config or AppearanceBankConfig()

    grouped: dict[
        int,
        list[ReIDEmbedding],
    ] = {}

    dimension: int | None = None

    for item in embeddings:
        if item.quality < cfg.min_quality:
            continue

        if dimension is None:
            dimension = len(item.vector)
        elif len(item.vector) != dimension:
            raise ValueError(
                "all ReID embeddings must share one dimension"
            )

        grouped.setdefault(
            item.track_id,
            [],
        ).append(item)

    banks: dict[int, TrackAppearanceBank] = {}

    for track_id, rows in sorted(
        grouped.items()
    ):
        # Quality-first selection, then chronological storage.
        chosen = sorted(
            rows,
            key=lambda item: (
                -item.quality,
                item.frame_index,
            ),
        )[:cfg.max_embeddings_per_track]

        chosen = sorted(
            chosen,
            key=lambda item: item.frame_index,
        )

        banks[track_id] = TrackAppearanceBank(
            track_id=track_id,
            embeddings=tuple(chosen),
        )

    return banks


def cosine_similarity_matrix(
    left: TrackAppearanceBank,
    right: TrackAppearanceBank,
) -> np.ndarray:
    a = left.matrix()
    b = right.matrix()

    if a.size == 0 or b.size == 0:
        return np.empty(
            (len(a), len(b)),
            dtype=np.float32,
        )

    if a.shape[1] != b.shape[1]:
        raise ValueError(
            "appearance banks use different embedding dimensions"
        )

    matrix = a @ b.T

    return np.clip(
        matrix,
        -1.0,
        1.0,
    )


def compare_track_banks(
    left: TrackAppearanceBank,
    right: TrackAppearanceBank,
    *,
    config: AppearanceBankConfig | None = None,
) -> TrackPairSimilarity:
    cfg = config or AppearanceBankConfig()

    if left.count == 0 or right.count == 0:
        raise ValueError(
            "track comparison requires non-empty appearance banks"
        )

    matrix = cosine_similarity_matrix(
        left,
        right,
    ).astype(np.float64)

    flattened = matrix.ravel()

    row_best = np.max(
        matrix,
        axis=1,
    )
    column_best = np.max(
        matrix,
        axis=0,
    )

    row_best_median = float(
        np.median(row_best)
    )
    column_best_median = float(
        np.median(column_best)
    )

    bidirectional = 0.5 * (
        row_best_median
        + column_best_median
    )

    unique_matches = _greedy_unique_matches(
        matrix,
    )

    top_k = min(
        cfg.unique_top_k,
        len(unique_matches),
    )
    unique_topk = (
        float(
            np.mean(
                [
                    score
                    for _, _, score in unique_matches[:top_k]
                ]
            )
        )
        if top_k > 0
        else -1.0
    )

    strong_a = float(
        np.mean(
            row_best
            >= cfg.strong_similarity_threshold
        )
    )
    strong_b = float(
        np.mean(
            column_best
            >= cfg.strong_similarity_threshold
        )
    )

    return TrackPairSimilarity(
        track_a=left.track_id,
        track_b=right.track_id,
        count_a=left.count,
        count_b=right.count,
        all_pairs_median=float(
            np.median(flattened)
        ),
        all_pairs_q90=float(
            np.quantile(flattened, 0.90)
        ),
        best_similarity=float(
            np.max(flattened)
        ),
        row_best_median=row_best_median,
        column_best_median=column_best_median,
        bidirectional_best_median=float(
            bidirectional
        ),
        unique_topk_similarity=unique_topk,
        strong_match_coverage=float(
            0.5 * (strong_a + strong_b)
        ),
    )


def within_track_coherence(
    bank: TrackAppearanceBank,
) -> WithinTrackCoherence:
    if bank.count == 0:
        raise ValueError(
            "within-track coherence requires a non-empty bank"
        )

    if bank.count == 1:
        values = np.asarray(
            [1.0],
            dtype=np.float64,
        )
    else:
        matrix = cosine_similarity_matrix(
            bank,
            bank,
        ).astype(np.float64)
        upper = np.triu_indices(
            bank.count,
            k=1,
        )
        values = matrix[upper]

    return WithinTrackCoherence(
        track_id=bank.track_id,
        embedding_count=bank.count,
        median_similarity=float(
            np.median(values)
        ),
        q10_similarity=float(
            np.quantile(values, 0.10)
        ),
        q90_similarity=float(
            np.quantile(values, 0.90)
        ),
        minimum_similarity=float(
            np.min(values)
        ),
        maximum_similarity=float(
            np.max(values)
        ),
    )


def compare_all_pairs(
    banks: Mapping[int, TrackAppearanceBank],
    pairs: Sequence[tuple[int, int]],
    *,
    config: AppearanceBankConfig | None = None,
) -> tuple[TrackPairSimilarity, ...]:
    rows: list[TrackPairSimilarity] = []

    for track_a, track_b in pairs:
        if track_a not in banks or track_b not in banks:
            continue

        rows.append(
            compare_track_banks(
                banks[track_a],
                banks[track_b],
                config=config,
            )
        )

    return tuple(rows)


def _greedy_unique_matches(
    matrix: np.ndarray,
) -> list[tuple[int, int, float]]:
    candidates = [
        (
            row,
            col,
            float(matrix[row, col]),
        )
        for row in range(matrix.shape[0])
        for col in range(matrix.shape[1])
    ]

    candidates.sort(
        key=lambda item: (
            -item[2],
            item[0],
            item[1],
        )
    )

    used_rows: set[int] = set()
    used_cols: set[int] = set()
    selected: list[
        tuple[int, int, float]
    ] = []

    for row, col, score in candidates:
        if row in used_rows or col in used_cols:
            continue

        selected.append(
            (row, col, score)
        )
        used_rows.add(row)
        used_cols.add(col)

        if (
            len(used_rows) == matrix.shape[0]
            or len(used_cols) == matrix.shape[1]
        ):
            break

    return selected
