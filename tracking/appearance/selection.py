"""High-quality visual-moment selection for Tracking V1 appearance.

This module does NOT run ReID, crop pixels or assign identities.  It chooses
which canonical detections are worth turning into visual appearance samples.

The selector joins:
    CanonicalDetectionFrame artifacts
        +
    Tracklet artifacts
        ->
    a small set of high-quality, temporally diverse moments per tracklet

Quality is intentionally based only on evidence already available before the
appearance model runs:

* detector confidence,
* player image size,
* distance from the representative virtual-view border,
* same-frame field-space isolation from nearby people,
* consistency of the multi-view X/Z estimate,
* owner/multi-view provenance.

The final score is a ranking heuristic, not identity evidence.  A low score
means "poor crop candidate", not "unlikely to be this player".
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from tracking.detections import CanonicalDetection, CanonicalDetectionFrame
from tracking.tracklets import Tracklet, TrackletSet


def _finite(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _unit_interval(name: str, value: float) -> float:
    parsed = _finite(name, value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _non_negative(name: str, value: float) -> float:
    parsed = _finite(name, value)
    if parsed < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return parsed


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer > 0")
    return value


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be an integer >= 0")
    return value


def _json_safe_dict(
    name: str,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain JSON-serializable values"
        ) from exc
    return result


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ramp(
    value: float,
    *,
    bad: float,
    good: float,
) -> float:
    """Map ``bad`` -> 0 and ``good`` -> 1 with clipping."""
    if not good > bad:
        raise ValueError("good threshold must be greater than bad threshold")
    return _clamp01((float(value) - bad) / (good - bad))


def _inverse_ramp(
    value: float,
    *,
    good: float,
    bad: float,
) -> float:
    """Map a low ``good`` value -> 1 and a high ``bad`` value -> 0."""
    if not bad > good:
        raise ValueError("bad threshold must be greater than good threshold")
    return 1.0 - _clamp01((float(value) - good) / (bad - good))


@dataclass(frozen=True, slots=True)
class AppearanceSelectionConfig:
    """Model-independent crop-candidate selection policy.

    Defaults are conservative first-pass values for the current 1280x720
    virtual-camera experiment.  They rank observations; they do not alter
    tracking or identity assignment.
    """

    max_samples_per_tracklet: int = 8
    min_samples_if_available: int = 3
    min_quality: float = 0.42
    min_temporal_spacing_s: float = 0.75
    temporal_diversity_weight: float = 0.25

    virtual_width_px: int = 1280
    virtual_height_px: int = 720

    confidence_floor: float = 0.40
    confidence_good: float = 0.95

    visual_height_bad_px: float = 18.0
    visual_height_good_px: float = 96.0

    edge_clearance_bad_px: float = 2.0
    edge_clearance_good_px: float = 20.0

    isolation_bad_m: float = 0.75
    isolation_good_m: float = 3.00

    spread_good_m: float = 0.25
    spread_bad_m: float = 1.50

    confidence_weight: float = 0.15
    visual_scale_weight: float = 0.30
    edge_clearance_weight: float = 0.15
    isolation_weight: float = 0.20
    geometry_weight: float = 0.15
    provenance_weight: float = 0.05

    def __post_init__(self) -> None:
        _positive_int(
            "max_samples_per_tracklet",
            self.max_samples_per_tracklet,
        )
        _positive_int(
            "min_samples_if_available",
            self.min_samples_if_available,
        )
        if self.min_samples_if_available > self.max_samples_per_tracklet:
            raise ValueError(
                "min_samples_if_available must be <= max_samples_per_tracklet"
            )

        _unit_interval("min_quality", self.min_quality)
        _non_negative(
            "min_temporal_spacing_s",
            self.min_temporal_spacing_s,
        )
        _unit_interval(
            "temporal_diversity_weight",
            self.temporal_diversity_weight,
        )

        _positive_int("virtual_width_px", self.virtual_width_px)
        _positive_int("virtual_height_px", self.virtual_height_px)

        for name in (
            "confidence_floor",
            "confidence_good",
        ):
            _unit_interval(name, getattr(self, name))
        if not self.confidence_good > self.confidence_floor:
            raise ValueError(
                "confidence_good must be greater than confidence_floor"
            )

        pairs = (
            (
                "visual_height_bad_px",
                self.visual_height_bad_px,
                "visual_height_good_px",
                self.visual_height_good_px,
            ),
            (
                "edge_clearance_bad_px",
                self.edge_clearance_bad_px,
                "edge_clearance_good_px",
                self.edge_clearance_good_px,
            ),
            (
                "isolation_bad_m",
                self.isolation_bad_m,
                "isolation_good_m",
                self.isolation_good_m,
            ),
            (
                "spread_good_m",
                self.spread_good_m,
                "spread_bad_m",
                self.spread_bad_m,
            ),
        )
        for low_name, low, high_name, high in pairs:
            _non_negative(low_name, low)
            _non_negative(high_name, high)
            if not high > low:
                raise ValueError(
                    f"{high_name} must be greater than {low_name}"
                )

        weights = (
            self.confidence_weight,
            self.visual_scale_weight,
            self.edge_clearance_weight,
            self.isolation_weight,
            self.geometry_weight,
            self.provenance_weight,
        )
        for index, value in enumerate(weights):
            _non_negative(f"weight[{index}]", value)
        if not math.isclose(sum(weights), 1.0, abs_tol=1e-9):
            raise ValueError("appearance quality weights must sum to 1.0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class AppearanceMoment:
    """One canonical detection selected as a future appearance sample source."""

    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    quality: float

    representative_view: str | None
    source_lens: str | None

    confidence_score: float
    visual_scale_score: float
    edge_clearance_score: float
    isolation_score: float
    geometry_score: float
    provenance_score: float

    virtual_bbox_height_px: float | None = None
    max_source_bbox_height_px: float | None = None
    nearest_person_distance_m: float | None = None
    position_spread_m: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.tracklet_id, str) or not self.tracklet_id.strip():
            raise ValueError("tracklet_id must be non-empty")
        if not isinstance(self.detection_id, str) or not self.detection_id.strip():
            raise ValueError("detection_id must be non-empty")
        _non_negative_int("frame_index", self.frame_index)
        if _finite("timestamp_s", self.timestamp_s) < 0.0:
            raise ValueError("timestamp_s must be >= 0")

        for name in (
            "quality",
            "confidence_score",
            "visual_scale_score",
            "edge_clearance_score",
            "isolation_score",
            "geometry_score",
            "provenance_score",
        ):
            _unit_interval(name, getattr(self, name))

        for name in (
            "virtual_bbox_height_px",
            "max_source_bbox_height_px",
            "nearest_person_distance_m",
            "position_spread_m",
        ):
            value = getattr(self, name)
            if value is not None:
                _non_negative(name, value)

        if self.representative_view is not None:
            if (
                not isinstance(self.representative_view, str)
                or not self.representative_view.strip()
            ):
                raise ValueError(
                    "representative_view must be non-empty or None"
                )

        if self.source_lens not in (None, "left", "right"):
            raise ValueError(
                "source_lens must be 'left', 'right', or None"
            )

        object.__setattr__(
            self,
            "metadata",
            _json_safe_dict("metadata", self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "quality": self.quality,
            "representative_view": self.representative_view,
            "source_lens": self.source_lens,
            "confidence_score": self.confidence_score,
            "visual_scale_score": self.visual_scale_score,
            "edge_clearance_score": self.edge_clearance_score,
            "isolation_score": self.isolation_score,
            "geometry_score": self.geometry_score,
            "provenance_score": self.provenance_score,
            "virtual_bbox_height_px": self.virtual_bbox_height_px,
            "max_source_bbox_height_px": self.max_source_bbox_height_px,
            "nearest_person_distance_m": self.nearest_person_distance_m,
            "position_spread_m": self.position_spread_m,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TrackletMomentSelection:
    """Selected appearance moments for one tracklet."""

    tracklet_id: str
    candidate_count: int
    eligible_count: int
    moments: tuple[AppearanceMoment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tracklet_id, str) or not self.tracklet_id.strip():
            raise ValueError("tracklet_id must be non-empty")
        _non_negative_int("candidate_count", self.candidate_count)
        _non_negative_int("eligible_count", self.eligible_count)

        if self.eligible_count > self.candidate_count:
            raise ValueError(
                "eligible_count must be <= candidate_count"
            )
        if len(self.moments) > self.eligible_count:
            raise ValueError(
                "selected moments must be <= eligible_count"
            )

        previous_time: float | None = None
        seen_ids: set[str] = set()
        for moment in self.moments:
            if moment.tracklet_id != self.tracklet_id:
                raise ValueError(
                    "moment belongs to a different tracklet"
                )
            if moment.detection_id in seen_ids:
                raise ValueError(
                    "duplicate detection_id in TrackletMomentSelection"
                )
            seen_ids.add(moment.detection_id)

            if (
                previous_time is not None
                and moment.timestamp_s <= previous_time
            ):
                raise ValueError(
                    "selected moments must be strictly time ordered"
                )
            previous_time = moment.timestamp_s

    @property
    def selected_count(self) -> int:
        return len(self.moments)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "moments": [moment.to_dict() for moment in self.moments],
        }


@dataclass(frozen=True, slots=True)
class AppearanceSelectionSet:
    """Selection diagnostics for all tracklets of one clip."""

    selections: tuple[TrackletMomentSelection, ...]
    config: AppearanceSelectionConfig
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        seen: set[str] = set()
        previous_id: str | None = None
        for selection in self.selections:
            if selection.tracklet_id in seen:
                raise ValueError(
                    f"duplicate tracklet selection: {selection.tracklet_id!r}"
                )
            seen.add(selection.tracklet_id)

            if (
                previous_id is not None
                and selection.tracklet_id <= previous_id
            ):
                raise ValueError(
                    "selections must be strictly ordered by tracklet_id"
                )
            previous_id = selection.tracklet_id

        object.__setattr__(
            self,
            "metadata",
            _json_safe_dict("metadata", self.metadata),
        )

    @property
    def tracklet_count(self) -> int:
        return len(self.selections)

    @property
    def candidate_count(self) -> int:
        return sum(item.candidate_count for item in self.selections)

    @property
    def eligible_count(self) -> int:
        return sum(item.eligible_count for item in self.selections)

    @property
    def selected_count(self) -> int:
        return sum(item.selected_count for item in self.selections)

    @property
    def empty_selection_count(self) -> int:
        return sum(item.selected_count == 0 for item in self.selections)

    @property
    def mean_selected_quality(self) -> float | None:
        values = [
            moment.quality
            for selection in self.selections
            for moment in selection.moments
        ]
        if not values:
            return None
        return sum(values) / len(values)

    def summary(self) -> dict[str, Any]:
        return {
            "tracklet_count": self.tracklet_count,
            "candidate_count": self.candidate_count,
            "eligible_count": self.eligible_count,
            "selected_count": self.selected_count,
            "empty_selection_count": self.empty_selection_count,
            "mean_selected_quality": self.mean_selected_quality,
            "config": self.config.to_dict(),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "selections": [
                selection.to_dict()
                for selection in self.selections
            ],
            "config": self.config.to_dict(),
            "metadata": dict(self.metadata),
        }


def _bbox_height(
    bbox: Sequence[float] | None,
) -> float | None:
    if bbox is None:
        return None
    if len(bbox) != 4:
        return None
    try:
        y1 = float(bbox[1])
        y2 = float(bbox[3])
    except (TypeError, ValueError):
        return None
    if not math.isfinite(y1) or not math.isfinite(y2):
        return None
    return max(0.0, y2 - y1)


def _max_source_bbox_height(
    detection: CanonicalDetection,
) -> float | None:
    heights: list[float] = []

    direct = _bbox_height(detection.source_bbox_xyxy)
    if direct is not None:
        heights.append(direct)

    raw_sources = detection.metadata.get("source_pixels")
    if isinstance(raw_sources, list):
        for observation in raw_sources:
            if not isinstance(observation, Mapping):
                continue
            lenses = observation.get("lenses")
            if not isinstance(lenses, list):
                continue
            for lens in lenses:
                if not isinstance(lens, Mapping):
                    continue
                try:
                    y_min = float(lens["y_min"])
                    y_max = float(lens["y_max"])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(y_min) and math.isfinite(y_max):
                    heights.append(max(0.0, y_max - y_min))

    return max(heights) if heights else None


def _virtual_edge_clearance(
    detection: CanonicalDetection,
    config: AppearanceSelectionConfig,
) -> float | None:
    bbox = detection.virtual_bbox_xyxy
    if bbox is None:
        return None

    x1, y1, x2, y2 = (float(item) for item in bbox)
    return max(
        0.0,
        min(
            x1,
            y1,
            float(config.virtual_width_px) - x2,
            float(config.virtual_height_px) - y2,
        ),
    )


def _position_spread(
    detection: CanonicalDetection,
) -> float | None:
    raw = detection.metadata.get("position_spread_m")
    if raw is None:
        return None
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value < 0.0:
        return None
    return value


def _nearest_person_distance(
    detection: CanonicalDetection,
    frame: CanonicalDetectionFrame,
) -> float | None:
    distances = [
        math.hypot(
            other.x - detection.x,
            other.z - detection.z,
        )
        for other in frame.detections
        if other.detection_id != detection.detection_id
    ]
    return min(distances) if distances else None


def _provenance_score(
    detection: CanonicalDetection,
) -> float:
    owner_supported = bool(
        detection.metadata.get("owner_supported", False)
    )
    multi_view = bool(
        detection.metadata.get(
            "multi_view",
            len(detection.contributing_views) > 1,
        )
    )

    # Owner-backed geometry is the stronger signal. Multi-view agreement adds
    # a smaller bonus; neither is required for a usable appearance crop.
    return (
        0.65 * float(owner_supported)
        + 0.35 * float(multi_view)
    )


def score_appearance_moment(
    *,
    tracklet_id: str,
    detection: CanonicalDetection,
    frame: CanonicalDetectionFrame,
    config: AppearanceSelectionConfig | None = None,
) -> AppearanceMoment:
    """Score one canonical detection as a future visual-sample moment."""
    selected = config or AppearanceSelectionConfig()

    confidence_score = _ramp(
        detection.confidence,
        bad=selected.confidence_floor,
        good=selected.confidence_good,
    )

    virtual_height = _bbox_height(
        detection.virtual_bbox_xyxy
    )
    source_height = _max_source_bbox_height(detection)

    visual_heights = [
        value
        for value in (virtual_height, source_height)
        if value is not None
    ]
    visual_scale_score = (
        max(
            _ramp(
                value,
                bad=selected.visual_height_bad_px,
                good=selected.visual_height_good_px,
            )
            for value in visual_heights
        )
        if visual_heights
        else 0.0
    )

    edge_clearance = _virtual_edge_clearance(
        detection,
        selected,
    )
    edge_clearance_score = (
        _ramp(
            edge_clearance,
            bad=selected.edge_clearance_bad_px,
            good=selected.edge_clearance_good_px,
        )
        if edge_clearance is not None
        else 0.50
    )

    nearest_distance = _nearest_person_distance(
        detection,
        frame,
    )
    isolation_score = (
        _ramp(
            nearest_distance,
            bad=selected.isolation_bad_m,
            good=selected.isolation_good_m,
        )
        if nearest_distance is not None
        else 1.0
    )

    spread = _position_spread(detection)
    geometry_score = (
        _inverse_ramp(
            spread,
            good=selected.spread_good_m,
            bad=selected.spread_bad_m,
        )
        if spread is not None
        else 0.50
    )

    provenance_score = _provenance_score(detection)

    quality = (
        selected.confidence_weight * confidence_score
        + selected.visual_scale_weight * visual_scale_score
        + selected.edge_clearance_weight * edge_clearance_score
        + selected.isolation_weight * isolation_score
        + selected.geometry_weight * geometry_score
        + selected.provenance_weight * provenance_score
    )
    quality = _clamp01(quality)

    return AppearanceMoment(
        tracklet_id=tracklet_id,
        detection_id=detection.detection_id,
        frame_index=detection.frame_index,
        timestamp_s=detection.timestamp_s,
        quality=quality,
        representative_view=detection.representative_view,
        source_lens=detection.source_lens,
        confidence_score=confidence_score,
        visual_scale_score=visual_scale_score,
        edge_clearance_score=edge_clearance_score,
        isolation_score=isolation_score,
        geometry_score=geometry_score,
        provenance_score=provenance_score,
        virtual_bbox_height_px=virtual_height,
        max_source_bbox_height_px=source_height,
        nearest_person_distance_m=nearest_distance,
        position_spread_m=spread,
        metadata={
            "contributing_view_count": len(
                detection.contributing_views
            ),
            "owner_supported": bool(
                detection.metadata.get("owner_supported", False)
            ),
            "multi_view": bool(
                detection.metadata.get(
                    "multi_view",
                    len(detection.contributing_views) > 1,
                )
            ),
        },
    )


def _validate_tracklet_detection(
    tracklet: Tracklet,
    observation,
    detection: CanonicalDetection,
) -> None:
    if detection.frame_index != observation.frame_index:
        raise ValueError(
            f"detection {detection.detection_id!r} frame does not match "
            f"tracklet {tracklet.tracklet_id!r}"
        )
    if not math.isclose(
        detection.timestamp_s,
        observation.timestamp_s,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"detection {detection.detection_id!r} timestamp does not match "
            f"tracklet {tracklet.tracklet_id!r}"
        )
    if not math.isclose(
        detection.x,
        observation.x,
        rel_tol=0.0,
        abs_tol=1e-6,
    ) or not math.isclose(
        detection.z,
        observation.z,
        rel_tol=0.0,
        abs_tol=1e-6,
    ):
        raise ValueError(
            f"detection {detection.detection_id!r} X/Z does not match "
            f"tracklet {tracklet.tracklet_id!r}"
        )


def _select_temporally_diverse(
    candidates: Sequence[AppearanceMoment],
    *,
    config: AppearanceSelectionConfig,
) -> tuple[AppearanceMoment, ...]:
    eligible = [
        item
        for item in candidates
        if item.quality >= config.min_quality
    ]
    if not eligible:
        return ()

    maximum = min(
        config.max_samples_per_tracklet,
        len(eligible),
    )
    minimum = min(
        config.min_samples_if_available,
        maximum,
    )

    # First pick: strongest visual evidence.
    selected: list[AppearanceMoment] = [
        max(
            eligible,
            key=lambda item: (
                item.quality,
                -item.timestamp_s,
                item.detection_id,
            ),
        )
    ]
    remaining = [
        item
        for item in eligible
        if item.detection_id != selected[0].detection_id
    ]

    times = [item.timestamp_s for item in eligible]
    time_span = max(times) - min(times)
    diversity_scale = max(
        time_span,
        config.min_temporal_spacing_s,
        1e-9,
    )

    while len(selected) < maximum:
        feasible = [
            item
            for item in remaining
            if all(
                abs(item.timestamp_s - chosen.timestamp_s)
                + 1e-12
                >= config.min_temporal_spacing_s
                for chosen in selected
            )
        ]
        if not feasible:
            break

        weight = config.temporal_diversity_weight

        def utility(item: AppearanceMoment) -> tuple[float, float, float, str]:
            nearest_selected = min(
                abs(item.timestamp_s - chosen.timestamp_s)
                for chosen in selected
            )
            diversity = _clamp01(
                nearest_selected / diversity_scale
            )
            value = (
                (1.0 - weight) * item.quality
                + weight * diversity
            )
            return (
                value,
                item.quality,
                -item.timestamp_s,
                item.detection_id,
            )

        chosen = max(feasible, key=utility)
        selected.append(chosen)
        remaining = [
            item
            for item in remaining
            if item.detection_id != chosen.detection_id
        ]

    # Very short tracklets cannot satisfy the spacing rule. Preserve a small
    # evidence floor by filling only up to ``minimum`` with the best remaining
    # eligible moments. We never fill all the way to ``maximum`` this way.
    if len(selected) < minimum:
        ranked_remaining = sorted(
            remaining,
            key=lambda item: (
                -item.quality,
                item.timestamp_s,
                item.detection_id,
            ),
        )
        for item in ranked_remaining:
            if len(selected) >= minimum:
                break
            selected.append(item)

    return tuple(
        sorted(
            selected,
            key=lambda item: (
                item.timestamp_s,
                item.detection_id,
            ),
        )
    )


def select_appearance_moments(
    frames: Iterable[
        CanonicalDetectionFrame | Mapping[str, Any]
    ],
    tracklets: TrackletSet | Iterable[Tracklet | Mapping[str, Any]],
    *,
    config: AppearanceSelectionConfig | None = None,
) -> AppearanceSelectionSet:
    """Select future appearance-sample moments for every tracklet.

    Every tracklet observation must resolve to exactly one canonical detection.
    This strict join prevents appearance from being extracted from stale or
    mismatched detection/tracklet artifacts.
    """
    selected_config = config or AppearanceSelectionConfig()

    detection_index: dict[
        str,
        tuple[CanonicalDetection, CanonicalDetectionFrame],
    ] = {}

    previous_frame_index: int | None = None
    previous_timestamp_s: float | None = None

    for raw_frame in frames:
        frame = (
            raw_frame
            if isinstance(raw_frame, CanonicalDetectionFrame)
            else CanonicalDetectionFrame.from_dict(raw_frame)
        )

        if (
            previous_frame_index is not None
            and frame.frame_index <= previous_frame_index
        ):
            raise ValueError(
                "canonical frames must be strictly frame ordered"
            )
        if (
            previous_timestamp_s is not None
            and frame.timestamp_s <= previous_timestamp_s
        ):
            raise ValueError(
                "canonical frames must be strictly time ordered"
            )

        for detection in frame.detections:
            if detection.detection_id in detection_index:
                raise ValueError(
                    "detection_id must be globally unique across frames: "
                    f"{detection.detection_id!r}"
                )
            detection_index[detection.detection_id] = (
                detection,
                frame,
            )

        previous_frame_index = frame.frame_index
        previous_timestamp_s = frame.timestamp_s

    if isinstance(tracklets, TrackletSet):
        normalized_tracklets = tracklets.tracklets
    else:
        normalized_tracklets = tuple(
            item
            if isinstance(item, Tracklet)
            else Tracklet.from_dict(item)
            for item in tracklets
        )
        # Reuse TrackletSet's cross-tracklet uniqueness validation.
        normalized_tracklets = TrackletSet(
            tracklets=normalized_tracklets
        ).tracklets

    selections: list[TrackletMomentSelection] = []

    for tracklet in sorted(
        normalized_tracklets,
        key=lambda item: item.tracklet_id,
    ):
        candidates: list[AppearanceMoment] = []

        for observation in tracklet.observations:
            resolved = detection_index.get(
                observation.detection_id
            )
            if resolved is None:
                raise ValueError(
                    f"tracklet {tracklet.tracklet_id!r} references missing "
                    f"detection {observation.detection_id!r}"
                )

            detection, frame = resolved
            _validate_tracklet_detection(
                tracklet,
                observation,
                detection,
            )

            candidates.append(
                score_appearance_moment(
                    tracklet_id=tracklet.tracklet_id,
                    detection=detection,
                    frame=frame,
                    config=selected_config,
                )
            )

        eligible_count = sum(
            item.quality >= selected_config.min_quality
            for item in candidates
        )
        moments = _select_temporally_diverse(
            candidates,
            config=selected_config,
        )

        selections.append(
            TrackletMomentSelection(
                tracklet_id=tracklet.tracklet_id,
                candidate_count=len(candidates),
                eligible_count=eligible_count,
                moments=moments,
            )
        )

    return AppearanceSelectionSet(
        selections=tuple(selections),
        config=selected_config,
        metadata={
            "selector": "appearance-moment-selector-v1",
            "detection_index_size": len(detection_index),
        },
    )
