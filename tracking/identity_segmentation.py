"""V4 purity-aware offline segmentation of raw tracker IDs into identity-consistent spans.

A temporal tracker ID is usually one physical person, but it can occasionally
jump from one person to another.  This module detects only *persistent* internal
identity switches.  It deliberately ignores isolated noisy crops.

Evidence hierarchy:
- persistent opposite-team semantic kit evidence is very strong,
- the raw crop-level kit descriptor is used directly so a persistent visual
  change can be detected even when global TEAM_A/TEAM_B clustering is noisy,
- ReID supports kit changes and can detect unusually strong same-team switches,
- one odd crop is never enough to split a track.

The module only returns temporal split boundaries and segment appearance
summaries.  It never changes detector/tracker output and never fabricates
positions between observations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Mapping, Sequence

import numpy as np

from .team_appearance import AppearanceLabel
from .team_similarity import RelativeTeamLabel


_EPS = 1e-12


class SegmentIdentityQuality(str, Enum):
    """How much a segment may be trusted as one physical person."""

    CLEAN = "clean"
    AMBIGUOUS = "ambiguous"
    CONTAMINATED = "contaminated"


@dataclass(frozen=True)
class IdentityAppearanceSample:
    raw_track_id: int
    frame_index: int
    quality: float = 1.0
    reid_vector: np.ndarray | None = None
    appearance_label: AppearanceLabel = AppearanceLabel.UNKNOWN
    # Normalized crop-level kit descriptor from team_appearance.extract_crop_appearance.
    # Unlike the semantic cluster label, this preserves the actual visual change
    # inside one raw track (important for cases such as raw35).
    kit_vector: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.raw_track_id < 1:
            raise ValueError("raw_track_id must be >= 1")
        if self.frame_index < 0:
            raise ValueError("frame_index must be >= 0")
        if not 0.0 <= float(self.quality) <= 1.0:
            raise ValueError("quality must be in [0, 1]")

        for name in ("reid_vector", "kit_vector"):
            value = getattr(self, name)
            if value is None:
                continue
            vector = np.asarray(value, dtype=np.float32)
            if vector.ndim != 1 or vector.size == 0:
                raise ValueError(f"{name} must be a non-empty 1D vector")
            if not np.isfinite(vector).all():
                raise ValueError(f"{name} must be finite")
            norm = float(np.linalg.norm(vector))
            if norm <= _EPS:
                raise ValueError(f"{name} norm must be > 0")
            object.__setattr__(self, name, vector / norm)


@dataclass(frozen=True)
class IdentitySegmentationConfig:
    min_samples_per_segment: int = 2
    max_segments_per_track: int = 4

    # Persistent semantic kit evidence.
    min_team_samples_per_side: int = 2
    min_team_purity: float = 0.75

    # V2: persistent change in the raw crop-level kit descriptor.  This catches
    # a visual identity switch even when global TEAM_A/TEAM_B clustering labels
    # both sides imperfectly.
    min_kit_samples_per_side: int = 3
    min_kit_split_gain: float = 0.075
    max_cross_kit_similarity: float = 0.80
    min_child_kit_coherence: float = 0.60
    boundary_window_samples: int = 2

    # V3: "persistent state replacement".  A raw track can be noisy/contaminated
    # before settling into a new persistent visual state.  Requiring both sides
    # to be semantically pure misses cases such as raw35:
    #
    #   B A A B | A A A A A A
    #
    # The right side is stable, while the left side is contaminated but still
    # contains repeated evidence of a different state.  V3 accepts this only
    # when raw kit descriptors AND ReID both support a real persistent change.
    stable_state_min_samples: int = 4
    stable_state_min_purity: float = 0.90
    replacement_min_opposite_samples: int = 2
    replacement_min_kit_gain: float = 0.08
    replacement_max_cross_kit_similarity: float = 0.90
    replacement_min_child_kit_coherence: float = 0.85
    replacement_max_cross_reid_similarity: float = 0.77
    replacement_min_child_reid_coherence: float = 0.68

    # A descriptor shift normally receives ReID support.  ReID is deliberately
    # permissive here: on tiny football crops it is useful to reject a clearly
    # contradictory "same person" signal, not to be the primary decision maker.
    kit_support_max_cross_reid_similarity: float = 0.84
    kit_support_min_child_reid_coherence: float = 0.50

    # Extremely strong persistent kit change can stand alone when ReID samples
    # are unavailable or unhelpful.
    very_strong_kit_gain: float = 0.14
    very_strong_max_cross_kit_similarity: float = 0.62

    # Combined semantic-team + ReID diagnostics (kept for audit compatibility).
    min_combined_reid_gain: float = 0.035
    combined_max_cross_similarity: float = 0.82
    combined_min_child_coherence: float = 0.54

    # ReID-only fallback for a same-team identity switch.
    min_strong_reid_gain: float = 0.12
    strong_max_cross_similarity: float = 0.72
    strong_min_child_coherence: float = 0.60

    # V4 segment-quality classification. Splitting and trusting are separate
    # questions: a range may not justify another cut yet still be too mixed to
    # establish or bridge a physical identity.
    clean_min_samples: int = 3
    clean_min_appearance_purity: float = 0.82
    clean_min_kit_coherence: float = 0.78
    clean_min_reid_coherence: float = 0.68

    contaminated_min_samples: int = 4
    contaminated_max_appearance_purity: float = 0.80
    contaminated_min_secondary_samples: int = 2
    contaminated_max_kit_coherence: float = 0.90
    contaminated_max_reid_coherence: float = 0.78
    contaminated_strong_vector_coherence: float = 0.68

    def __post_init__(self) -> None:
        if self.min_samples_per_segment < 2:
            raise ValueError("min_samples_per_segment must be >= 2")
        if self.max_segments_per_track < 1:
            raise ValueError("max_segments_per_track must be >= 1")
        if self.min_team_samples_per_side < 2:
            raise ValueError("min_team_samples_per_side must be >= 2")
        if self.min_kit_samples_per_side < 2:
            raise ValueError("min_kit_samples_per_side must be >= 2")
        if self.boundary_window_samples < 1:
            raise ValueError("boundary_window_samples must be >= 1")
        if self.stable_state_min_samples < 3:
            raise ValueError("stable_state_min_samples must be >= 3")
        if self.replacement_min_opposite_samples < 1:
            raise ValueError("replacement_min_opposite_samples must be >= 1")
        if self.clean_min_samples < 2:
            raise ValueError("clean_min_samples must be >= 2")
        if self.contaminated_min_samples < 3:
            raise ValueError("contaminated_min_samples must be >= 3")
        if self.contaminated_min_secondary_samples < 1:
            raise ValueError("contaminated_min_secondary_samples must be >= 1")

        for name in (
            "min_team_purity",
            "clean_min_appearance_purity",
            "clean_min_kit_coherence",
            "clean_min_reid_coherence",
            "contaminated_max_appearance_purity",
            "contaminated_max_kit_coherence",
            "contaminated_max_reid_coherence",
            "contaminated_strong_vector_coherence",
            "stable_state_min_purity",
            "replacement_min_kit_gain",
            "replacement_max_cross_kit_similarity",
            "replacement_min_child_kit_coherence",
            "replacement_max_cross_reid_similarity",
            "replacement_min_child_reid_coherence",
            "min_kit_split_gain",
            "max_cross_kit_similarity",
            "min_child_kit_coherence",
            "kit_support_max_cross_reid_similarity",
            "kit_support_min_child_reid_coherence",
            "very_strong_kit_gain",
            "very_strong_max_cross_kit_similarity",
            "min_combined_reid_gain",
            "combined_max_cross_similarity",
            "combined_min_child_coherence",
            "min_strong_reid_gain",
            "strong_max_cross_similarity",
            "strong_min_child_coherence",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class SegmentAppearanceSummary:
    raw_track_id: int
    segment_index: int
    start_frame: int
    end_frame: int
    sample_count: int

    appearance_label: AppearanceLabel
    appearance_confidence: float
    appearance_sample_count: int

    team_label: RelativeTeamLabel
    team_confidence: float

    reid_prototype: np.ndarray | None
    reid_sample_count: int
    reid_internal_similarity: float | None

    kit_sample_count: int
    kit_internal_similarity: float | None
    known_appearance_count: int
    secondary_appearance_count: int

    identity_quality: SegmentIdentityQuality
    identity_quality_score: float
    identity_quality_reason: str

    def __post_init__(self) -> None:
        if self.reid_prototype is not None:
            vector = np.asarray(self.reid_prototype, dtype=np.float32)
            norm = float(np.linalg.norm(vector))
            if vector.ndim != 1 or not np.isfinite(vector).all() or norm <= _EPS:
                raise ValueError("invalid reid_prototype")
            object.__setattr__(self, "reid_prototype", vector / norm)

    @property
    def key(self) -> str:
        return f"{self.raw_track_id}:{self.segment_index}"

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_track_id": self.raw_track_id,
            "segment_index": self.segment_index,
            "key": self.key,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "sample_count": self.sample_count,
            "appearance_label": self.appearance_label.value,
            "appearance_confidence": self.appearance_confidence,
            "appearance_sample_count": self.appearance_sample_count,
            "team_label": self.team_label.value,
            "team_confidence": self.team_confidence,
            "reid_sample_count": self.reid_sample_count,
            "reid_internal_similarity": self.reid_internal_similarity,
            "kit_sample_count": self.kit_sample_count,
            "kit_internal_similarity": self.kit_internal_similarity,
            "known_appearance_count": self.known_appearance_count,
            "secondary_appearance_count": self.secondary_appearance_count,
            "identity_quality": self.identity_quality.value,
            "identity_quality_score": self.identity_quality_score,
            "identity_quality_reason": self.identity_quality_reason,
        }


@dataclass(frozen=True)
class IdentitySplitEvidence:
    raw_track_id: int
    boundary_after_frame: int
    left_sample_count: int
    right_sample_count: int

    left_appearance: AppearanceLabel
    right_appearance: AppearanceLabel
    left_appearance_purity: float
    right_appearance_purity: float
    left_appearance_count: int
    right_appearance_count: int
    persistent_appearance_conflict: bool

    left_kit_coherence: float | None
    right_kit_coherence: float | None
    cross_kit_similarity: float | None
    kit_split_gain: float | None
    boundary_kit_similarity: float | None
    persistent_kit_shift: bool

    left_reid_coherence: float | None
    right_reid_coherence: float | None
    cross_reid_similarity: float | None
    reid_split_gain: float | None

    accepted: bool
    reason: str
    score: float

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["left_appearance"] = self.left_appearance.value
        payload["right_appearance"] = self.right_appearance.value
        return payload


@dataclass(frozen=True)
class TrackIdentitySegmentation:
    raw_track_id: int
    segments: tuple[SegmentAppearanceSummary, ...]
    accepted_splits: tuple[IdentitySplitEvidence, ...]
    best_rejected_split: IdentitySplitEvidence | None

    @property
    def segmented(self) -> bool:
        return len(self.segments) > 1

    @property
    def boundaries_after_frame(self) -> tuple[int, ...]:
        return tuple(
            split.boundary_after_frame
            for split in self.accepted_splits
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "raw_track_id": self.raw_track_id,
            "segmented": self.segmented,
            "boundaries_after_frame": list(self.boundaries_after_frame),
            "segments": [segment.to_dict() for segment in self.segments],
            "accepted_splits": [split.to_dict() for split in self.accepted_splits],
            "best_rejected_split": (
                self.best_rejected_split.to_dict()
                if self.best_rejected_split is not None
                else None
            ),
        }


def segment_track_identity(
    samples: Sequence[IdentityAppearanceSample],
    *,
    config: IdentitySegmentationConfig | None = None,
) -> TrackIdentitySegmentation:
    """Split one raw tracker ID only when persistent evidence supports it."""

    cfg = config or IdentitySegmentationConfig()
    if not samples:
        raise ValueError("identity segmentation requires at least one sample")

    raw_ids = {sample.raw_track_id for sample in samples}
    if len(raw_ids) != 1:
        raise ValueError("all samples must belong to the same raw_track_id")
    raw_track_id = next(iter(raw_ids))

    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (sample.frame_index, -sample.quality),
        )
    )

    ranges: list[tuple[int, int]] = [(0, len(ordered))]
    accepted: list[IdentitySplitEvidence] = []
    rejected: list[IdentitySplitEvidence] = []

    while len(ranges) < cfg.max_segments_per_track:
        best: tuple[float, int, int, IdentitySplitEvidence] | None = None

        for range_index, (start, end) in enumerate(ranges):
            candidate = _best_split_for_range(
                raw_track_id,
                ordered,
                start=start,
                end=end,
                config=cfg,
            )
            if candidate is None:
                continue

            cut, evidence = candidate
            if not evidence.accepted:
                rejected.append(evidence)
                continue

            ranking = evidence.score
            if best is None or ranking > best[0]:
                best = (ranking, range_index, cut, evidence)

        if best is None:
            break

        _, range_index, cut, evidence = best
        start, end = ranges[range_index]
        ranges[range_index:range_index + 1] = [
            (start, cut),
            (cut, end),
        ]
        accepted.append(evidence)

    ranges = sorted(ranges, key=lambda item: item[0])
    segments = tuple(
        _summarize_segment(
            raw_track_id,
            segment_index,
            ordered[start:end],
            config=cfg,
        )
        for segment_index, (start, end) in enumerate(ranges)
    )

    accepted_sorted = tuple(
        sorted(accepted, key=lambda item: item.boundary_after_frame)
    )
    best_rejected = (
        max(rejected, key=lambda item: item.score)
        if rejected
        else None
    )

    return TrackIdentitySegmentation(
        raw_track_id=raw_track_id,
        segments=segments,
        accepted_splits=accepted_sorted,
        best_rejected_split=best_rejected,
    )


def segment_all_tracks(
    samples_by_track: Mapping[int, Sequence[IdentityAppearanceSample]],
    *,
    config: IdentitySegmentationConfig | None = None,
) -> dict[int, TrackIdentitySegmentation]:
    result: dict[int, TrackIdentitySegmentation] = {}
    for raw_track_id, samples in sorted(samples_by_track.items()):
        if not samples:
            continue
        segmentation = segment_track_identity(samples, config=config)
        if segmentation.raw_track_id != raw_track_id:
            raise ValueError("samples_by_track key does not match sample raw_track_id")
        result[raw_track_id] = segmentation
    return result


def _persistent_state_context(
    left: Sequence[IdentityAppearanceSample],
    right: Sequence[IdentityAppearanceSample],
    *,
    left_appearance: AppearanceLabel,
    left_purity: float,
    right_appearance: AppearanceLabel,
    right_purity: float,
    config: IdentitySegmentationConfig,
) -> tuple[str | None, AppearanceLabel, int]:
    """Find a stable semantic state replacing a noisy earlier/later state.

    This does *not* accept a split by itself.  It merely identifies temporal
    persistence.  Raw-kit and ReID evidence must independently support the
    identity change before V3 accepts it.
    """

    known = {
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
        AppearanceLabel.SPECIAL,
    }

    def opposite_support(
        samples: Sequence[IdentityAppearanceSample],
        stable_label: AppearanceLabel,
    ) -> int:
        return sum(
            1
            for sample in samples
            if (
                sample.appearance_label in known
                and sample.appearance_label is not stable_label
            )
        )

    candidates: list[tuple[float, int, str, AppearanceLabel, int]] = []

    if (
        len(right) >= config.stable_state_min_samples
        and right_appearance in known
        and right_purity >= config.stable_state_min_purity
    ):
        support = opposite_support(left, right_appearance)
        if support >= config.replacement_min_opposite_samples:
            candidates.append(
                (
                    float(right_purity),
                    len(right),
                    "right",
                    right_appearance,
                    support,
                )
            )

    if (
        len(left) >= config.stable_state_min_samples
        and left_appearance in known
        and left_purity >= config.stable_state_min_purity
    ):
        support = opposite_support(right, left_appearance)
        if support >= config.replacement_min_opposite_samples:
            candidates.append(
                (
                    float(left_purity),
                    len(left),
                    "left",
                    left_appearance,
                    support,
                )
            )

    if not candidates:
        return None, AppearanceLabel.UNKNOWN, 0

    _, _, side, label, support = max(
        candidates,
        key=lambda item: (
            item[0],
            item[1],
            item[4],
        ),
    )
    return side, label, int(support)


def _best_split_for_range(
    raw_track_id: int,
    samples: Sequence[IdentityAppearanceSample],
    *,
    start: int,
    end: int,
    config: IdentitySegmentationConfig,
) -> tuple[int, IdentitySplitEvidence] | None:
    minimum = config.min_samples_per_segment
    if end - start < 2 * minimum:
        return None

    best: tuple[float, int, IdentitySplitEvidence] | None = None

    for cut in range(start + minimum, end - minimum + 1):
        left = samples[start:cut]
        right = samples[cut:end]

        left_appearance, left_purity, left_count = _dominant_appearance(left)
        right_appearance, right_purity, right_count = _dominant_appearance(right)

        persistent_appearance_conflict = (
            left_appearance in {
                AppearanceLabel.TEAM_A,
                AppearanceLabel.TEAM_B,
                AppearanceLabel.SPECIAL,
            }
            and right_appearance in {
                AppearanceLabel.TEAM_A,
                AppearanceLabel.TEAM_B,
                AppearanceLabel.SPECIAL,
            }
            and left_appearance is not right_appearance
            and left_count >= config.min_team_samples_per_side
            and right_count >= config.min_team_samples_per_side
            and left_purity >= config.min_team_purity
            and right_purity >= config.min_team_purity
        )

        (
            stable_state_side,
            stable_state_label,
            opposite_state_support,
        ) = _persistent_state_context(
            left,
            right,
            left_appearance=left_appearance,
            left_purity=left_purity,
            right_appearance=right_appearance,
            right_purity=right_purity,
            config=config,
        )

        kit = _vector_split_metrics(
            left,
            right,
            attribute="kit_vector",
            min_samples=config.min_kit_samples_per_side,
        )
        if kit is None:
            left_kit_coherence = None
            right_kit_coherence = None
            cross_kit_similarity = None
            kit_split_gain = None
            boundary_kit_similarity = None
            persistent_kit_shift = False
            very_strong_kit_shift = False
        else:
            (
                left_kit_coherence,
                right_kit_coherence,
                cross_kit_similarity,
                kit_split_gain,
            ) = kit
            boundary_kit_similarity = _local_boundary_similarity(
                left,
                right,
                attribute="kit_vector",
                window=config.boundary_window_samples,
            )
            persistent_kit_shift = (
                kit_split_gain >= config.min_kit_split_gain
                and cross_kit_similarity <= config.max_cross_kit_similarity
                and min(left_kit_coherence, right_kit_coherence)
                >= config.min_child_kit_coherence
            )
            very_strong_kit_shift = (
                kit_split_gain >= config.very_strong_kit_gain
                and cross_kit_similarity
                <= config.very_strong_max_cross_kit_similarity
                and min(left_kit_coherence, right_kit_coherence)
                >= config.min_child_kit_coherence
            )

        reid = _reid_split_metrics(left, right)
        if reid is None:
            left_coherence = None
            right_coherence = None
            cross_similarity = None
            split_gain = None
            combined_reid_support = False
            strong_reid_support = False
            kit_reid_support = False
        else:
            (
                left_coherence,
                right_coherence,
                cross_similarity,
                split_gain,
            ) = reid

            combined_reid_support = (
                split_gain >= config.min_combined_reid_gain
                and cross_similarity <= config.combined_max_cross_similarity
                and min(left_coherence, right_coherence)
                >= config.combined_min_child_coherence
            )
            strong_reid_support = (
                split_gain >= config.min_strong_reid_gain
                and cross_similarity <= config.strong_max_cross_similarity
                and min(left_coherence, right_coherence)
                >= config.strong_min_child_coherence
            )
            kit_reid_support = (
                cross_similarity
                <= config.kit_support_max_cross_reid_similarity
                and min(left_coherence, right_coherence)
                >= config.kit_support_min_child_reid_coherence
            )

        replacement_kit_support = (
            stable_state_side is not None
            and kit_split_gain is not None
            and cross_kit_similarity is not None
            and left_kit_coherence is not None
            and right_kit_coherence is not None
            and kit_split_gain >= config.replacement_min_kit_gain
            and cross_kit_similarity
            <= config.replacement_max_cross_kit_similarity
            and min(left_kit_coherence, right_kit_coherence)
            >= config.replacement_min_child_kit_coherence
        )

        replacement_reid_support = (
            stable_state_side is not None
            and cross_similarity is not None
            and left_coherence is not None
            and right_coherence is not None
            and cross_similarity
            <= config.replacement_max_cross_reid_similarity
            and min(left_coherence, right_coherence)
            >= config.replacement_min_child_reid_coherence
        )

        persistent_state_replacement = (
            replacement_kit_support
            and replacement_reid_support
        )

        # V3 decision hierarchy:
        # 1) persistent semantic team/special conflict is physically decisive;
        # 2) V3 stable-state replacement: one persistent semantic state replaces
        #    a noisy/contaminated state, with both kit + ReID support;
        # 3) persistent raw-kit descriptor change + non-contradictory ReID;
        # 4) exceptionally strong kit change even without useful ReID;
        # 5) unusually strong ReID-only discontinuity.
        accepted_kit_shift = (
            persistent_kit_shift and kit_reid_support
        ) or very_strong_kit_shift

        accepted = (
            persistent_appearance_conflict
            or persistent_state_replacement
            or accepted_kit_shift
            or strong_reid_support
        )

        if persistent_appearance_conflict and combined_reid_support:
            reason = "persistent_appearance_conflict+reid"
        elif persistent_appearance_conflict:
            reason = "persistent_appearance_conflict"
        elif persistent_state_replacement:
            reason = "persistent_state_replacement+kit+reid"
        elif accepted_kit_shift and kit_reid_support:
            reason = "persistent_kit_shift+reid"
        elif accepted_kit_shift:
            reason = "very_strong_persistent_kit_shift"
        elif strong_reid_support:
            reason = "strong_reid_discontinuity"
        else:
            reason = "insufficient_persistent_identity_change"

        semantic_strength = 0.0
        if persistent_appearance_conflict:
            semantic_strength = 0.55 + 0.45 * min(
                left_purity,
                right_purity,
            )

        kit_strength = 0.0
        if kit_split_gain is not None and cross_kit_similarity is not None:
            gain_strength = max(
                0.0,
                min(
                    1.0,
                    kit_split_gain
                    / max(config.very_strong_kit_gain, _EPS),
                ),
            )
            separation_strength = max(
                0.0,
                min(
                    1.0,
                    (
                        config.max_cross_kit_similarity
                        - cross_kit_similarity
                    )
                    / max(config.max_cross_kit_similarity, _EPS),
                ),
            )
            boundary_strength = (
                max(
                    0.0,
                    min(
                        1.0,
                        1.0 - boundary_kit_similarity,
                    ),
                )
                if boundary_kit_similarity is not None
                else 0.0
            )
            kit_strength = (
                0.55 * gain_strength
                + 0.20 * separation_strength
                + 0.25 * boundary_strength
            )

        reid_strength = 0.0
        if split_gain is not None:
            reid_strength = max(
                0.0,
                min(1.0, split_gain / 0.20),
            )

        replacement_strength = 0.0
        if persistent_state_replacement:
            stable_purity = (
                right_purity
                if stable_state_side == "right"
                else left_purity
            )
            stable_size = (
                len(right)
                if stable_state_side == "right"
                else len(left)
            )
            replacement_strength = min(
                1.0,
                0.45 * float(stable_purity)
                + 0.30 * min(
                    1.0,
                    stable_size / max(
                        float(config.stable_state_min_samples + 2),
                        1.0,
                    ),
                )
                + 0.25 * min(
                    1.0,
                    opposite_state_support
                    / max(
                        float(config.replacement_min_opposite_samples + 1),
                        1.0,
                    ),
                ),
            )

        score = (
            (1.0 if accepted else 0.0)
            + 0.48 * semantic_strength
            + 0.38 * kit_strength
            + 0.14 * reid_strength
            + 0.24 * replacement_strength
            # Tiny unsaturated terms are deliberately used only to localize
            # an already-supported change point rather than change acceptance.
            + 0.015 * max(0.0, float(kit_split_gain or 0.0))
            + 0.010 * max(0.0, float(split_gain or 0.0))
        )

        evidence = IdentitySplitEvidence(
            raw_track_id=raw_track_id,
            boundary_after_frame=left[-1].frame_index,
            left_sample_count=len(left),
            right_sample_count=len(right),
            left_appearance=left_appearance,
            right_appearance=right_appearance,
            left_appearance_purity=float(left_purity),
            right_appearance_purity=float(right_purity),
            left_appearance_count=int(left_count),
            right_appearance_count=int(right_count),
            persistent_appearance_conflict=bool(
                persistent_appearance_conflict
            ),
            left_kit_coherence=(
                float(left_kit_coherence)
                if left_kit_coherence is not None
                else None
            ),
            right_kit_coherence=(
                float(right_kit_coherence)
                if right_kit_coherence is not None
                else None
            ),
            cross_kit_similarity=(
                float(cross_kit_similarity)
                if cross_kit_similarity is not None
                else None
            ),
            kit_split_gain=(
                float(kit_split_gain)
                if kit_split_gain is not None
                else None
            ),
            boundary_kit_similarity=(
                float(boundary_kit_similarity)
                if boundary_kit_similarity is not None
                else None
            ),
            persistent_kit_shift=bool(
                persistent_kit_shift
            ),
            left_reid_coherence=(
                float(left_coherence)
                if left_coherence is not None
                else None
            ),
            right_reid_coherence=(
                float(right_coherence)
                if right_coherence is not None
                else None
            ),
            cross_reid_similarity=(
                float(cross_similarity)
                if cross_similarity is not None
                else None
            ),
            reid_split_gain=(
                float(split_gain)
                if split_gain is not None
                else None
            ),
            accepted=bool(accepted),
            reason=reason,
            score=float(score),
        )

        if best is None or score > best[0]:
            best = (score, cut, evidence)

    if best is None:
        return None
    return best[1], best[2]



def _dominant_appearance(
    samples: Sequence[IdentityAppearanceSample],
) -> tuple[AppearanceLabel, float, int]:
    labels = (
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
        AppearanceLabel.SPECIAL,
    )
    weights = {label: 0.0 for label in labels}
    counts = {label: 0 for label in labels}

    for sample in samples:
        if sample.appearance_label not in weights:
            continue
        weights[sample.appearance_label] += max(0.10, float(sample.quality))
        counts[sample.appearance_label] += 1

    total_weight = sum(weights.values())
    total_count = sum(counts.values())
    if total_weight <= _EPS or total_count == 0:
        return AppearanceLabel.UNKNOWN, 0.0, 0

    winner = max(weights, key=weights.get)
    purity = weights[winner] / total_weight
    return winner, float(purity), int(counts[winner])


def _appearance_to_team(label: AppearanceLabel) -> RelativeTeamLabel:
    if label is AppearanceLabel.TEAM_A:
        return RelativeTeamLabel.TEAM_A
    if label is AppearanceLabel.TEAM_B:
        return RelativeTeamLabel.TEAM_B
    return RelativeTeamLabel.UNKNOWN

def _reid_split_metrics(
    left: Sequence[IdentityAppearanceSample],
    right: Sequence[IdentityAppearanceSample],
) -> tuple[float, float, float, float] | None:
    return _vector_split_metrics(
        left,
        right,
        attribute="reid_vector",
        min_samples=2,
    )


def _vector_split_metrics(
    left: Sequence[IdentityAppearanceSample],
    right: Sequence[IdentityAppearanceSample],
    *,
    attribute: str,
    min_samples: int,
) -> tuple[float, float, float, float] | None:
    left_vectors = [
        getattr(sample, attribute)
        for sample in left
        if getattr(sample, attribute) is not None
    ]
    right_vectors = [
        getattr(sample, attribute)
        for sample in right
        if getattr(sample, attribute) is not None
    ]

    if (
        len(left_vectors) < min_samples
        or len(right_vectors) < min_samples
    ):
        return None

    left_matrix = np.stack(
        left_vectors,
        axis=0,
    ).astype(np.float64)
    right_matrix = np.stack(
        right_vectors,
        axis=0,
    ).astype(np.float64)

    left_coherence = _median_upper_cosine(left_matrix)
    right_coherence = _median_upper_cosine(right_matrix)
    cross_similarity = float(
        np.median(
            left_matrix @ right_matrix.T
        )
    )
    weighted_within = (
        len(left_vectors) * left_coherence
        + len(right_vectors) * right_coherence
    ) / (
        len(left_vectors)
        + len(right_vectors)
    )
    split_gain = (
        weighted_within
        - cross_similarity
    )

    return (
        float(left_coherence),
        float(right_coherence),
        float(cross_similarity),
        float(split_gain),
    )



def _local_boundary_similarity(
    left: Sequence[IdentityAppearanceSample],
    right: Sequence[IdentityAppearanceSample],
    *,
    attribute: str,
    window: int,
) -> float | None:
    left_vectors = [
        getattr(sample, attribute)
        for sample in left
        if getattr(sample, attribute) is not None
    ][-window:]
    right_vectors = [
        getattr(sample, attribute)
        for sample in right
        if getattr(sample, attribute) is not None
    ][:window]

    if not left_vectors or not right_vectors:
        return None

    def prototype(vectors: Sequence[np.ndarray]) -> np.ndarray | None:
        value = np.mean(
            np.stack(vectors, axis=0).astype(np.float64),
            axis=0,
        )
        norm = float(np.linalg.norm(value))
        if norm <= _EPS:
            return None
        return value / norm

    left_proto = prototype(left_vectors)
    right_proto = prototype(right_vectors)
    if left_proto is None or right_proto is None:
        return None

    return float(
        np.clip(
            np.dot(left_proto, right_proto),
            -1.0,
            1.0,
        )
    )


def _median_upper_cosine(matrix: np.ndarray) -> float:
    if len(matrix) < 2:
        return 1.0
    similarity = matrix @ matrix.T
    upper = similarity[np.triu_indices(len(matrix), k=1)]
    if upper.size == 0:
        return 1.0
    return float(np.median(upper))


def _segment_vector_coherence(
    samples: Sequence[IdentityAppearanceSample],
    *,
    attribute: str,
) -> tuple[int, float | None]:
    vectors = [
        getattr(sample, attribute)
        for sample in samples
        if getattr(sample, attribute) is not None
    ]
    if not vectors:
        return 0, None
    if len(vectors) == 1:
        return 1, 1.0
    matrix = np.stack(vectors, axis=0).astype(np.float64)
    return len(vectors), float(_median_upper_cosine(matrix))


def _appearance_support_counts(
    samples: Sequence[IdentityAppearanceSample],
) -> tuple[int, int]:
    labels = (
        AppearanceLabel.TEAM_A,
        AppearanceLabel.TEAM_B,
        AppearanceLabel.SPECIAL,
    )
    counts = {
        label: sum(
            1
            for sample in samples
            if sample.appearance_label is label
        )
        for label in labels
    }
    ordered = sorted(counts.values(), reverse=True)
    return sum(counts.values()), (ordered[1] if len(ordered) > 1 else 0)


def _classify_segment_quality(
    samples: Sequence[IdentityAppearanceSample],
    *,
    appearance_purity: float,
    appearance_count: int,
    kit_sample_count: int,
    kit_internal_similarity: float | None,
    reid_sample_count: int,
    reid_internal_similarity: float | None,
    config: IdentitySegmentationConfig,
) -> tuple[SegmentIdentityQuality, float, str, int, int]:
    """Classify whether one segment can safely represent a physical person.

    CLEAN can establish/bridge a physical identity.
    AMBIGUOUS is retained as evidence but should not establish or bridge one.
    CONTAMINATED contains persistent evidence of multiple visual states and
    should stay out of the identity solver until a later recovery stage.
    """

    known_count, secondary_count = _appearance_support_counts(samples)
    sample_count = len(samples)

    mixed_semantics = (
        sample_count >= config.contaminated_min_samples
        and known_count >= config.contaminated_min_samples
        and appearance_count >= config.contaminated_min_secondary_samples
        and secondary_count >= config.contaminated_min_secondary_samples
        and appearance_purity <= config.contaminated_max_appearance_purity
    )

    kit_disagrees = (
        kit_internal_similarity is not None
        and kit_sample_count >= config.contaminated_min_samples
        and kit_internal_similarity
        < config.contaminated_max_kit_coherence
    )
    reid_disagrees = (
        reid_internal_similarity is not None
        and reid_sample_count >= config.contaminated_min_samples
        and reid_internal_similarity
        < config.contaminated_max_reid_coherence
    )
    strong_vector_mixture = (
        sample_count >= config.contaminated_min_samples
        and kit_internal_similarity is not None
        and reid_internal_similarity is not None
        and kit_internal_similarity
        < config.contaminated_strong_vector_coherence
        and reid_internal_similarity
        < config.contaminated_strong_vector_coherence
    )

    if (
        (mixed_semantics and (kit_disagrees or reid_disagrees))
        or strong_vector_mixture
    ):
        score_parts = [
            max(0.0, 1.0 - appearance_purity),
        ]
        if kit_internal_similarity is not None:
            score_parts.append(
                max(0.0, 1.0 - kit_internal_similarity)
            )
        if reid_internal_similarity is not None:
            score_parts.append(
                max(0.0, 1.0 - reid_internal_similarity)
            )
        contamination = min(
            1.0,
            float(sum(score_parts) / max(len(score_parts), 1)),
        )
        return (
            SegmentIdentityQuality.CONTAMINATED,
            float(1.0 - contamination),
            "persistent_multi_state_contamination",
            known_count,
            secondary_count,
        )

    enough_samples = sample_count >= config.clean_min_samples
    appearance_ok = (
        known_count == 0
        or appearance_purity >= config.clean_min_appearance_purity
    )
    kit_ok = (
        kit_internal_similarity is None
        or kit_internal_similarity >= config.clean_min_kit_coherence
    )
    reid_ok = (
        reid_internal_similarity is None
        or reid_internal_similarity >= config.clean_min_reid_coherence
    )

    if enough_samples and appearance_ok and kit_ok and reid_ok:
        components = [
            min(1.0, sample_count / 6.0),
            appearance_purity if known_count else 0.75,
        ]
        if kit_internal_similarity is not None:
            components.append(kit_internal_similarity)
        if reid_internal_similarity is not None:
            components.append(reid_internal_similarity)
        score = float(
            np.clip(
                np.mean(components),
                0.0,
                1.0,
            )
        )
        return (
            SegmentIdentityQuality.CLEAN,
            score,
            "stable_single_state",
            known_count,
            secondary_count,
        )

    # Sparse or mildly inconsistent evidence is not thrown away; it simply
    # loses the right to establish/bridge an identity in the physical solver.
    components = [
        min(1.0, sample_count / max(float(config.clean_min_samples), 1.0)),
        appearance_purity if known_count else 0.5,
    ]
    if kit_internal_similarity is not None:
        components.append(kit_internal_similarity)
    if reid_internal_similarity is not None:
        components.append(reid_internal_similarity)
    score = float(
        np.clip(
            np.mean(components),
            0.0,
            1.0,
        )
    )
    return (
        SegmentIdentityQuality.AMBIGUOUS,
        score,
        "insufficient_clean_evidence",
        known_count,
        secondary_count,
    )


def _summarize_segment(
    raw_track_id: int,
    segment_index: int,
    samples: Sequence[IdentityAppearanceSample],
    *,
    config: IdentitySegmentationConfig,
) -> SegmentAppearanceSummary:
    if not samples:
        raise ValueError("segment samples must not be empty")

    appearance_label, appearance_purity, appearance_count = _dominant_appearance(samples)
    # Purity from a single crop must not become a hard lock. Scale confidence
    # by persistent support; two or more agreeing crops can reach full purity.
    appearance_confidence = appearance_purity * min(1.0, appearance_count / 2.0)
    team_label = _appearance_to_team(appearance_label)
    team_confidence = (
        appearance_confidence
        if team_label is not RelativeTeamLabel.UNKNOWN
        else 0.0
    )

    vectors = [
        sample.reid_vector
        for sample in samples
        if sample.reid_vector is not None
    ]
    vector_samples = [
        sample
        for sample in samples
        if sample.reid_vector is not None
    ]

    if vectors:
        matrix = np.stack(vectors, axis=0).astype(np.float64)
        weights = np.asarray(
            [max(0.05, sample.quality) for sample in vector_samples],
            dtype=np.float64,
        )
        prototype = np.average(matrix, axis=0, weights=weights)
        norm = float(np.linalg.norm(prototype))
        if norm <= _EPS:
            prototype_out = None
        else:
            prototype_out = (prototype / norm).astype(np.float32)
        internal = _median_upper_cosine(matrix) if len(matrix) >= 2 else 1.0
    else:
        prototype_out = None
        internal = None

    kit_sample_count, kit_internal = _segment_vector_coherence(
        samples,
        attribute="kit_vector",
    )

    (
        identity_quality,
        identity_quality_score,
        identity_quality_reason,
        known_appearance_count,
        secondary_appearance_count,
    ) = _classify_segment_quality(
        samples,
        appearance_purity=appearance_purity,
        appearance_count=appearance_count,
        kit_sample_count=kit_sample_count,
        kit_internal_similarity=kit_internal,
        reid_sample_count=len(vectors),
        reid_internal_similarity=internal,
        config=config,
    )

    return SegmentAppearanceSummary(
        raw_track_id=raw_track_id,
        segment_index=segment_index,
        start_frame=samples[0].frame_index,
        end_frame=samples[-1].frame_index,
        sample_count=len(samples),
        appearance_label=appearance_label,
        appearance_confidence=float(appearance_confidence),
        appearance_sample_count=int(appearance_count),
        team_label=team_label,
        team_confidence=float(team_confidence),
        reid_prototype=prototype_out,
        reid_sample_count=len(vectors),
        reid_internal_similarity=(
            float(internal)
            if internal is not None
            else None
        ),
        kit_sample_count=int(kit_sample_count),
        kit_internal_similarity=(
            float(kit_internal)
            if kit_internal is not None
            else None
        ),
        known_appearance_count=int(known_appearance_count),
        secondary_appearance_count=int(secondary_appearance_count),
        identity_quality=identity_quality,
        identity_quality_score=float(identity_quality_score),
        identity_quality_reason=identity_quality_reason,
    )

