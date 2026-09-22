"""Cross-lens observation fusion for Veo physical LEFT/RIGHT views.

Goal
----
The detector outputs observations, not final players.

A person visible in both physical lenses may therefore produce two detections.
This module turns those raw observations into ONE canonical field entity using:

1. calibrated bbox-footpoint -> world X/Z projection
2. strict field filtering (+0.50 m only at the far touchline)
3. mutual cross-lens reprojection checks
4. one-to-one LEFT/RIGHT assignment
5. deterministic ownership for unmatched overlap observations

The important invariant is:

    a matched LEFT/RIGHT pair creates exactly one canonical entity

and unmatched detections from the non-owner lens are never allowed into the
canonical state.

This intentionally does NOT merge same-lens detections by world distance.
Two real players standing close together must remain two players.

Temporal tracking/identity is a later layer. This file solves the same-frame
cross-lens duplication problem only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class Observation:
    """One detector observation projected onto the pitch."""

    observation_id: str
    source_view: str
    xyxy: tuple[float, float, float, float]
    confidence: float

    foot_x: float
    foot_y: float

    world_x: float
    world_z: float

    owner_view: str | None
    source_is_owner: bool

    visible_left: bool
    visible_right: bool

    source_off_axis_deg: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PairMetrics:
    """Geometric compatibility between one LEFT and one RIGHT observation."""

    left_id: str
    right_id: str

    world_distance_m: float

    left_world_in_right_normalized_error: float
    right_world_in_left_normalized_error: float

    left_world_in_right_pixel_error: float
    right_world_in_left_pixel_error: float

    score: float
    compatible: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CanonicalEntity:
    """One person hypothesis in the canonical pitch state."""

    entity_id: str
    world_x: float
    world_z: float

    confidence: float
    representative_view: str

    observation_ids: tuple[str, ...]
    matched_cross_lens: bool

    fusion_world_distance_m: float | None
    fusion_score: float | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FusionResult:
    observations: tuple[Observation, ...]
    matched_pairs: tuple[PairMetrics, ...]
    entities: tuple[CanonicalEntity, ...]

    rejected_outside_field: int
    rejected_non_owner_unmatched: int
    rejected_no_ground: int

    def to_dict(self) -> dict:
        return {
            "observations": [item.to_dict() for item in self.observations],
            "matched_pairs": [item.to_dict() for item in self.matched_pairs],
            "entities": [item.to_dict() for item in self.entities],
            "counts": {
                "observations": len(self.observations),
                "matched_pairs": len(self.matched_pairs),
                "canonical_entities": len(self.entities),
                "rejected_outside_field": self.rejected_outside_field,
                "rejected_non_owner_unmatched": self.rejected_non_owner_unmatched,
                "rejected_no_ground": self.rejected_no_ground,
            },
        }


class TwoLensFusion:
    """Same-frame LEFT/RIGHT canonicalization.

    Default gates are deliberately geometric rather than count-based.

    `max_world_distance_m`
        Foot-point X/Z agreement required between the two observations.

    `max_reprojection_error`
        Mutual reprojection error normalized by target bbox height. A value of
        1.0 means the projected foot-point is one target-box height away.

    Matching is one-to-one. We use Hungarian assignment when SciPy is present
    and a deterministic sorted greedy fallback otherwise.
    """

    def __init__(
        self,
        projector,
        *,
        max_world_distance_m: float = 1.50,
        max_reprojection_error: float = 0.85,
        world_weight: float = 1.0,
        reprojection_weight: float = 1.5,
    ) -> None:
        self.projector = projector

        self.max_world_distance_m = float(max_world_distance_m)
        self.max_reprojection_error = float(max_reprojection_error)
        self.world_weight = float(world_weight)
        self.reprojection_weight = float(reprojection_weight)

        if self.max_world_distance_m <= 0:
            raise ValueError("max_world_distance_m must be > 0")

        if self.max_reprojection_error <= 0:
            raise ValueError("max_reprojection_error must be > 0")

    @staticmethod
    def _bbox_foot(
        xyxy: Sequence[float],
    ) -> tuple[float, float]:
        x1, y1, x2, y2 = [float(value) for value in xyxy]

        if not np.isfinite([x1, y1, x2, y2]).all():
            raise ValueError("bbox coordinates must be finite")

        if x2 <= x1 or y2 <= y1:
            raise ValueError("bbox must satisfy x2>x1 and y2>y1")

        return 0.5 * (x1 + x2), y2 - 0.5

    @staticmethod
    def _bbox_height(
        observation: Observation,
    ) -> float:
        return max(
            float(observation.xyxy[3] - observation.xyxy[1]),
            8.0,
        )

    @staticmethod
    def _detection_fields(detection) -> tuple[tuple[float, float, float, float], float]:
        """Accept our backend dataclasses or simple dict-like detections."""
        if hasattr(detection, "xyxy"):
            xyxy = tuple(float(value) for value in detection.xyxy)
            confidence = float(detection.confidence)
            return xyxy, confidence

        if isinstance(detection, dict):
            xyxy = tuple(float(value) for value in detection["xyxy"])
            confidence = float(detection["confidence"])
            return xyxy, confidence

        raise TypeError(
            "Detection must expose `.xyxy`/`.confidence` or be a dict"
        )

    def _build_observation(
        self,
        view_name: str,
        detection,
        index: int,
    ) -> tuple[Observation | None, str | None]:
        xyxy, confidence = self._detection_fields(detection)
        foot_x, foot_y = self._bbox_foot(xyxy)

        projection = self.projector.project_bbox(
            view_name,
            *xyxy,
        )

        if (
            projection.world_x is None
            or projection.world_z is None
        ):
            return None, "no_ground"

        if not projection.inside_allowed_field:
            return None, "outside_field"

        source_visibility = self.projector.visibility_for_world_point(
            view_name,
            float(projection.world_x),
            float(projection.world_z),
        )

        return (
            Observation(
                observation_id=f"{view_name}_{index:03d}",
                source_view=view_name,
                xyxy=xyxy,
                confidence=confidence,
                foot_x=foot_x,
                foot_y=foot_y,
                world_x=float(projection.world_x),
                world_z=float(projection.world_z),
                owner_view=projection.owner_view,
                source_is_owner=bool(projection.is_owner),
                visible_left=bool(projection.left_visible),
                visible_right=bool(projection.right_visible),
                source_off_axis_deg=source_visibility.off_axis_angle_deg,
            ),
            None,
        )

    def build_observations(
        self,
        detections_by_view: dict[str, Iterable],
    ) -> tuple[list[Observation], dict[str, int]]:
        observations: list[Observation] = []
        rejected = {
            "outside_field": 0,
            "no_ground": 0,
        }

        for view_name in ("left", "right"):
            for index, detection in enumerate(
                detections_by_view.get(view_name, ())
            ):
                observation, reason = self._build_observation(
                    view_name,
                    detection,
                    index,
                )

                if observation is not None:
                    observations.append(observation)
                elif reason is not None:
                    rejected[reason] += 1

        return observations, rejected

    def pair_metrics(
        self,
        left: Observation,
        right: Observation,
    ) -> PairMetrics:
        if left.source_view != "left":
            raise ValueError("left observation must come from LEFT lens")

        if right.source_view != "right":
            raise ValueError("right observation must come from RIGHT lens")

        world_distance = math.hypot(
            left.world_x - right.world_x,
            left.world_z - right.world_z,
        )

        left_world_seen_right = self.projector.visibility_for_world_point(
            "right",
            left.world_x,
            left.world_z,
        )
        right_world_seen_left = self.projector.visibility_for_world_point(
            "left",
            right.world_x,
            right.world_z,
        )

        if not left_world_seen_right.visible:
            return self._incompatible_pair(left, right, world_distance)

        if not right_world_seen_left.visible:
            return self._incompatible_pair(left, right, world_distance)

        assert left_world_seen_right.pixel_x is not None
        assert left_world_seen_right.pixel_y is not None
        assert right_world_seen_left.pixel_x is not None
        assert right_world_seen_left.pixel_y is not None

        left_to_right_px = math.hypot(
            left_world_seen_right.pixel_x - right.foot_x,
            left_world_seen_right.pixel_y - right.foot_y,
        )
        right_to_left_px = math.hypot(
            right_world_seen_left.pixel_x - left.foot_x,
            right_world_seen_left.pixel_y - left.foot_y,
        )

        left_to_right_norm = (
            left_to_right_px / self._bbox_height(right)
        )
        right_to_left_norm = (
            right_to_left_px / self._bbox_height(left)
        )

        compatible = (
            world_distance <= self.max_world_distance_m
            and left_to_right_norm <= self.max_reprojection_error
            and right_to_left_norm <= self.max_reprojection_error
        )

        score = (
            self.world_weight
            * world_distance
            / self.max_world_distance_m
            + self.reprojection_weight
            * 0.5
            * (
                left_to_right_norm
                + right_to_left_norm
            )
        )

        return PairMetrics(
            left_id=left.observation_id,
            right_id=right.observation_id,
            world_distance_m=world_distance,
            left_world_in_right_normalized_error=left_to_right_norm,
            right_world_in_left_normalized_error=right_to_left_norm,
            left_world_in_right_pixel_error=left_to_right_px,
            right_world_in_left_pixel_error=right_to_left_px,
            score=score,
            compatible=compatible,
        )

    @staticmethod
    def _incompatible_pair(
        left: Observation,
        right: Observation,
        world_distance: float,
    ) -> PairMetrics:
        return PairMetrics(
            left_id=left.observation_id,
            right_id=right.observation_id,
            world_distance_m=world_distance,
            left_world_in_right_normalized_error=float("inf"),
            right_world_in_left_normalized_error=float("inf"),
            left_world_in_right_pixel_error=float("inf"),
            right_world_in_left_pixel_error=float("inf"),
            score=float("inf"),
            compatible=False,
        )

    def _assign_pairs(
        self,
        left: list[Observation],
        right: list[Observation],
    ) -> list[tuple[int, int, PairMetrics]]:
        if not left or not right:
            return []

        matrix = np.full(
            (len(left), len(right)),
            np.inf,
            dtype=np.float64,
        )
        metrics_by_index: dict[tuple[int, int], PairMetrics] = {}

        for i, left_observation in enumerate(left):
            for j, right_observation in enumerate(right):
                metrics = self.pair_metrics(
                    left_observation,
                    right_observation,
                )
                metrics_by_index[(i, j)] = metrics

                if metrics.compatible:
                    matrix[i, j] = metrics.score

        finite = np.isfinite(matrix)
        if not finite.any():
            return []

        # Use a large finite penalty because scipy's assignment does not need
        # to select invalid pairs just because the matrix is rectangular.
        penalty = 1e6
        working = np.where(finite, matrix, penalty)

        selected: list[tuple[int, int]] = []

        try:
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(working)
            selected = [
                (int(i), int(j))
                for i, j in zip(rows, cols, strict=True)
                if finite[i, j]
            ]

        except ImportError:
            # Deterministic fallback: sort all compatible edges by score and
            # greedily keep a one-to-one subset.
            edges = sorted(
                (
                    (matrix[i, j], i, j)
                    for i in range(len(left))
                    for j in range(len(right))
                    if finite[i, j]
                ),
                key=lambda item: (
                    float(item[0]),
                    int(item[1]),
                    int(item[2]),
                ),
            )

            used_left: set[int] = set()
            used_right: set[int] = set()

            for _, i, j in edges:
                if i in used_left or j in used_right:
                    continue
                used_left.add(i)
                used_right.add(j)
                selected.append((i, j))

        return [
            (i, j, metrics_by_index[(i, j)])
            for i, j in selected
        ]

    def _representative_observation(
        self,
        left: Observation,
        right: Observation,
    ) -> Observation:
        """Pick the geometrically preferred source for canonical X/Z."""
        midpoint_x = 0.5 * (left.world_x + right.world_x)
        midpoint_z = 0.5 * (left.world_z + right.world_z)

        owner, _, _ = self.projector.owner_for_world_point(
            midpoint_x,
            midpoint_z,
        )

        if owner == "left":
            return left

        if owner == "right":
            return right

        # Defensive fallback outside calibrated ownership.
        if left.source_off_axis_deg is not None and right.source_off_axis_deg is not None:
            if left.source_off_axis_deg < right.source_off_axis_deg:
                return left
            if right.source_off_axis_deg < left.source_off_axis_deg:
                return right

        return left if left.confidence >= right.confidence else right

    def fuse(
        self,
        detections_by_view: dict[str, Iterable],
    ) -> FusionResult:
        observations, rejected = self.build_observations(
            detections_by_view
        )

        left = [
            observation
            for observation in observations
            if observation.source_view == "left"
        ]
        right = [
            observation
            for observation in observations
            if observation.source_view == "right"
        ]

        assignments = self._assign_pairs(left, right)

        matched_left: set[str] = set()
        matched_right: set[str] = set()
        matched_metrics: list[PairMetrics] = []
        entities: list[CanonicalEntity] = []

        for left_index, right_index, metrics in assignments:
            left_observation = left[left_index]
            right_observation = right[right_index]

            matched_left.add(left_observation.observation_id)
            matched_right.add(right_observation.observation_id)
            matched_metrics.append(metrics)

            representative = self._representative_observation(
                left_observation,
                right_observation,
            )

            entity_id = f"entity_{len(entities):03d}"

            entities.append(
                CanonicalEntity(
                    entity_id=entity_id,
                    world_x=representative.world_x,
                    world_z=representative.world_z,
                    confidence=max(
                        left_observation.confidence,
                        right_observation.confidence,
                    ),
                    representative_view=representative.source_view,
                    observation_ids=(
                        left_observation.observation_id,
                        right_observation.observation_id,
                    ),
                    matched_cross_lens=True,
                    fusion_world_distance_m=metrics.world_distance_m,
                    fusion_score=metrics.score,
                )
            )

        rejected_non_owner = 0

        for observation in observations:
            if observation.source_view == "left":
                already_matched = (
                    observation.observation_id in matched_left
                )
            else:
                already_matched = (
                    observation.observation_id in matched_right
                )

            if already_matched:
                continue

            # Critical invariant: an unmatched overlap observation is allowed
            # into canonical state only from its deterministic owner lens.
            if not observation.source_is_owner:
                rejected_non_owner += 1
                continue

            entity_id = f"entity_{len(entities):03d}"

            entities.append(
                CanonicalEntity(
                    entity_id=entity_id,
                    world_x=observation.world_x,
                    world_z=observation.world_z,
                    confidence=observation.confidence,
                    representative_view=observation.source_view,
                    observation_ids=(observation.observation_id,),
                    matched_cross_lens=False,
                    fusion_world_distance_m=None,
                    fusion_score=None,
                )
            )

        # Sanity invariant: no raw observation can belong to two entities.
        used_observations: set[str] = set()

        for entity in entities:
            for observation_id in entity.observation_ids:
                if observation_id in used_observations:
                    raise RuntimeError(
                        "Fusion invariant violated: observation used twice: "
                        f"{observation_id}"
                    )
                used_observations.add(observation_id)

        return FusionResult(
            observations=tuple(observations),
            matched_pairs=tuple(matched_metrics),
            entities=tuple(entities),
            rejected_outside_field=rejected["outside_field"],
            rejected_non_owner_unmatched=rejected_non_owner,
            rejected_no_ground=rejected["no_ground"],
        )
