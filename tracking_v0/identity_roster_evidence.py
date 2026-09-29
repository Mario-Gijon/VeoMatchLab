"""Roster/occupancy diagnostics for canonical football identities.

This module does NOT solve identities.  It measures whether a unit looks like:

  * a temporal continuation of an already selected canonical identity,
  * a simultaneous duplicate/shadow of an already selected identity,
  * or a distinct physical occupant that is currently missing from the roster.

It also audits selected canonical identities against each other to find possible
duplicate identities that could be merged to free a roster slot.

The intended use is after V2.8A-V5 when the fixed 23-person prior creates close
counterfactual swaps such as 11:0 <-> 16:0.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Mapping, Sequence

import numpy as np

from .identity_evidence import ObservedTrackPoint


@dataclass(frozen=True)
class RosterEvidenceConfig:
    shadow_radius_m: float = 2.5
    shadow_identity_support: float = 0.60
    min_shadow_overlap_fraction: float = 0.12

    distinct_distance_m: float = 3.5

    def __post_init__(self) -> None:
        if self.shadow_radius_m <= 0.0:
            raise ValueError("shadow_radius_m must be > 0")
        if not 0.0 <= self.shadow_identity_support <= 1.0:
            raise ValueError(
                "shadow_identity_support must be in [0, 1]"
            )
        if not 0.0 <= self.min_shadow_overlap_fraction <= 1.0:
            raise ValueError(
                "min_shadow_overlap_fraction must be in [0, 1]"
            )
        if self.distinct_distance_m <= 0.0:
            raise ValueError("distinct_distance_m must be > 0")


@dataclass(frozen=True)
class UnitOccupancyEvidence:
    unit_key: str
    observed_frames: int

    represented_frames: int
    represented_fraction: float

    unique_frames: int
    unique_fraction: float

    fills_open_slot_frames: int
    fills_open_slot_fraction: float

    overflow_if_added_frames: int
    overflow_if_added_fraction: float

    median_selected_identity_count: float

    best_shadow_identity: int | None
    best_shadow_overlap_frames: int
    best_shadow_overlap_fraction: float
    best_shadow_median_distance_m: float | None
    best_shadow_identity_support: float | None

    best_hybrid_identity: int | None
    best_hybrid_score: float | None
    best_hybrid_pair: tuple[str, str] | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.best_hybrid_pair is not None:
            payload["best_hybrid_pair"] = list(
                self.best_hybrid_pair
            )
        return payload


@dataclass(frozen=True)
class CanonicalPairEvidence:
    canonical_a: int
    canonical_b: int

    simultaneous_frames: int
    simultaneous_fraction: float

    median_distance_m: float | None
    q90_distance_m: float | None

    has_identity_conflict: bool

    best_identity_support: float
    best_hybrid_score: float
    best_hybrid_pair: tuple[str, str] | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        if self.best_hybrid_pair is not None:
            payload["best_hybrid_pair"] = list(
                self.best_hybrid_pair
            )
        return payload


def assign_observations_to_units(
    raw_observations: Mapping[
        int,
        Sequence[ObservedTrackPoint],
    ],
    unit_rows: Mapping[
        str,
        Mapping[str, object],
    ],
) -> dict[
    str,
    tuple[ObservedTrackPoint, ...],
]:
    """Split raw-track observations into solved identity-unit frame ranges."""

    units_by_track: dict[
        int,
        list[
            tuple[str, int, int]
        ],
    ] = {}

    for key, row in unit_rows.items():
        track_id = int(
            row["track_id"]
        )
        start_frame = int(
            row["start_frame"]
        )
        end_frame = int(
            row["end_frame"]
        )

        units_by_track.setdefault(
            track_id,
            [],
        ).append(
            (
                key,
                start_frame,
                end_frame,
            )
        )

    for rows in units_by_track.values():
        rows.sort(
            key=lambda item: (
                item[1],
                item[2],
                item[0],
            )
        )

    output: dict[
        str,
        list[ObservedTrackPoint],
    ] = {
        key: []
        for key in unit_rows
    }

    for track_id, points in raw_observations.items():
        ranges = units_by_track.get(
            int(track_id),
            (),
        )

        for point in points:
            matches = [
                key
                for key, start, end in ranges
                if (
                    start
                    <= point.frame_index
                    <= end
                )
            ]

            if len(matches) > 1:
                raise ValueError(
                    f"frame {point.frame_index} for raw track "
                    f"{track_id} belongs to multiple identity units"
                )

            if matches:
                output[
                    matches[0]
                ].append(
                    point
                )

    return {
        key: tuple(
            sorted(
                points,
                key=lambda point: (
                    point.frame_index,
                    point.timestamp,
                ),
            )
        )
        for key, points in output.items()
    }


def build_canonical_membership(
    identities: Sequence[
        Mapping[str, object]
    ],
) -> tuple[
    dict[str, int],
    dict[int, tuple[str, ...]],
]:
    unit_to_canonical = {}
    canonical_to_units = {}

    for row in identities:
        canonical_id = int(
            row["canonical_id"]
        )
        members = tuple(
            str(key)
            for key in row["member_keys"]
        )

        canonical_to_units[
            canonical_id
        ] = members

        for key in members:
            if key in unit_to_canonical:
                raise ValueError(
                    f"unit {key} appears in multiple canonical identities"
                )
            unit_to_canonical[
                key
            ] = canonical_id

    return (
        unit_to_canonical,
        canonical_to_units,
    )


def build_frame_canonical_points(
    *,
    unit_points: Mapping[
        str,
        Sequence[ObservedTrackPoint],
    ],
    unit_to_canonical: Mapping[str, int],
) -> dict[
    int,
    dict[
        int,
        tuple[
            tuple[str, ObservedTrackPoint],
            ...,
        ],
    ],
]:
    frames: dict[
        int,
        dict[
            int,
            list[
                tuple[str, ObservedTrackPoint]
            ],
        ],
    ] = {}

    for unit_key, points in unit_points.items():
        canonical_id = unit_to_canonical.get(
            unit_key
        )
        if canonical_id is None:
            continue

        for point in points:
            frames.setdefault(
                point.frame_index,
                {},
            ).setdefault(
                canonical_id,
                [],
            ).append(
                (
                    unit_key,
                    point,
                )
            )

    return {
        frame: {
            canonical_id: tuple(rows)
            for canonical_id, rows in by_canonical.items()
        }
        for frame, by_canonical in frames.items()
    }


def compute_unit_occupancy_evidence(
    unit_key: str,
    *,
    unit_points: Mapping[
        str,
        Sequence[ObservedTrackPoint],
    ],
    frame_canonical_points: Mapping[
        int,
        Mapping[
            int,
            Sequence[
                tuple[
                    str,
                    ObservedTrackPoint,
                ]
            ],
        ],
    ],
    unit_to_canonical: Mapping[str, int],
    pair_evidence: Mapping[
        tuple[str, str],
        Mapping[str, object],
    ],
    hybrid_edges: Mapping[
        tuple[str, str],
        Mapping[str, object],
    ],
    roster_size: int,
    config: RosterEvidenceConfig | None = None,
) -> UnitOccupancyEvidence:
    cfg = config or RosterEvidenceConfig()

    points = tuple(
        unit_points.get(
            unit_key,
            (),
        )
    )

    if not points:
        raise ValueError(
            f"unit {unit_key} has no observations"
        )

    own_canonical = unit_to_canonical.get(
        unit_key
    )

    represented = 0
    unique = 0
    fills_slot = 0
    overflow = 0
    selected_counts = []

    shadow_rows: dict[
        int,
        list[
            tuple[
                float,
                float,
            ]
        ],
    ] = {}

    for point in points:
        by_canonical = frame_canonical_points.get(
            point.frame_index,
            {},
        )

        visible_canonical_ids = {
            canonical_id
            for canonical_id in by_canonical
            if canonical_id != own_canonical
        }

        selected_count = len(
            visible_canonical_ids
        )
        selected_counts.append(
            selected_count
        )

        frame_represented = False

        for canonical_id in visible_canonical_ids:
            rows = by_canonical[
                canonical_id
            ]

            best_distance = math.inf
            best_support = 0.0
            admissible = False

            for other_key, other_point in rows:
                relation = pair_evidence.get(
                    _pair_key(
                        unit_key,
                        other_key,
                    )
                )

                if relation is None:
                    continue

                if str(
                    relation.get(
                        "conflict",
                        "none",
                    )
                ) != "none":
                    continue

                support = float(
                    relation.get(
                        "fused_support_score",
                        0.0,
                    )
                )

                if (
                    support
                    < cfg.shadow_identity_support
                ):
                    continue

                distance = math.hypot(
                    point.x
                    - other_point.x,
                    point.z
                    - other_point.z,
                )

                if distance < best_distance:
                    best_distance = distance
                    best_support = support
                    admissible = True

            if not admissible:
                continue

            shadow_rows.setdefault(
                canonical_id,
                [],
            ).append(
                (
                    best_distance,
                    best_support,
                )
            )

            if (
                best_distance
                <= cfg.shadow_radius_m
            ):
                frame_represented = True

        if frame_represented:
            represented += 1
        else:
            unique += 1

            if selected_count < roster_size:
                fills_slot += 1

            if selected_count >= roster_size:
                overflow += 1

    observed_frames = len(
        points
    )

    best_shadow_identity = None
    best_shadow_overlap_frames = 0
    best_shadow_overlap_fraction = 0.0
    best_shadow_median_distance = None
    best_shadow_support = None

    ranked_shadows = []

    for canonical_id, rows in shadow_rows.items():
        close = [
            row
            for row in rows
            if row[0]
            <= cfg.shadow_radius_m
        ]

        overlap_frames = len(
            close
        )
        fraction = (
            overlap_frames
            / observed_frames
        )

        if not close:
            median_distance = math.inf
            median_support = 0.0
        else:
            median_distance = float(
                np.median(
                    np.asarray(
                        [
                            row[0]
                            for row in close
                        ],
                        dtype=np.float64,
                    )
                )
            )
            median_support = float(
                np.median(
                    np.asarray(
                        [
                            row[1]
                            for row in close
                        ],
                        dtype=np.float64,
                    )
                )
            )

        ranked_shadows.append(
            (
                fraction,
                overlap_frames,
                -median_distance,
                canonical_id,
                median_distance,
                median_support,
            )
        )

    if ranked_shadows:
        (
            best_shadow_overlap_fraction,
            best_shadow_overlap_frames,
            _negative_distance,
            best_shadow_identity,
            best_shadow_median_distance,
            best_shadow_support,
        ) = max(
            ranked_shadows
        )

        if (
            best_shadow_overlap_fraction
            < cfg.min_shadow_overlap_fraction
        ):
            best_shadow_identity = None

    best_hybrid_identity = None
    best_hybrid_score = None
    best_hybrid_pair = None

    for pair_key, edge in hybrid_edges.items():
        if unit_key not in pair_key:
            continue

        other_key = (
            pair_key[1]
            if pair_key[0] == unit_key
            else pair_key[0]
        )

        canonical_id = unit_to_canonical.get(
            other_key
        )

        if (
            canonical_id is None
            or canonical_id
            == own_canonical
        ):
            continue

        score = float(
            edge.get(
                "hybrid_score",
                0.0,
            )
        )

        if (
            best_hybrid_score is None
            or score
            > best_hybrid_score
        ):
            best_hybrid_identity = (
                canonical_id
            )
            best_hybrid_score = score
            best_hybrid_pair = pair_key

    return UnitOccupancyEvidence(
        unit_key=unit_key,
        observed_frames=observed_frames,
        represented_frames=represented,
        represented_fraction=(
            represented
            / observed_frames
        ),
        unique_frames=unique,
        unique_fraction=(
            unique
            / observed_frames
        ),
        fills_open_slot_frames=fills_slot,
        fills_open_slot_fraction=(
            fills_slot
            / observed_frames
        ),
        overflow_if_added_frames=overflow,
        overflow_if_added_fraction=(
            overflow
            / observed_frames
        ),
        median_selected_identity_count=float(
            np.median(
                np.asarray(
                    selected_counts,
                    dtype=np.float64,
                )
            )
        ),
        best_shadow_identity=best_shadow_identity,
        best_shadow_overlap_frames=(
            best_shadow_overlap_frames
        ),
        best_shadow_overlap_fraction=float(
            best_shadow_overlap_fraction
        ),
        best_shadow_median_distance_m=(
            None
            if (
                best_shadow_identity
                is None
                or best_shadow_median_distance
                is None
                or not math.isfinite(
                    best_shadow_median_distance
                )
            )
            else float(
                best_shadow_median_distance
            )
        ),
        best_shadow_identity_support=(
            None
            if best_shadow_identity is None
            else float(
                best_shadow_support
            )
        ),
        best_hybrid_identity=(
            best_hybrid_identity
        ),
        best_hybrid_score=(
            None
            if best_hybrid_score is None
            else float(
                best_hybrid_score
            )
        ),
        best_hybrid_pair=(
            best_hybrid_pair
        ),
    )


def compute_canonical_pair_evidence(
    canonical_a: int,
    canonical_b: int,
    *,
    canonical_to_units: Mapping[
        int,
        Sequence[str],
    ],
    unit_points: Mapping[
        str,
        Sequence[ObservedTrackPoint],
    ],
    pair_evidence: Mapping[
        tuple[str, str],
        Mapping[str, object],
    ],
    hybrid_edges: Mapping[
        tuple[str, str],
        Mapping[str, object],
    ],
) -> CanonicalPairEvidence:
    members_a = tuple(
        canonical_to_units[
            canonical_a
        ]
    )
    members_b = tuple(
        canonical_to_units[
            canonical_b
        ]
    )

    frame_a = _canonical_frame_points(
        members_a,
        unit_points=unit_points,
    )
    frame_b = _canonical_frame_points(
        members_b,
        unit_points=unit_points,
    )

    common_frames = sorted(
        set(
            frame_a
        )
        & set(
            frame_b
        )
    )

    distances = []

    for frame in common_frames:
        best = min(
            math.hypot(
                point_a.x
                - point_b.x,
                point_a.z
                - point_b.z,
            )
            for _key_a, point_a in frame_a[
                frame
            ]
            for _key_b, point_b in frame_b[
                frame
            ]
        )

        distances.append(
            best
        )

    denominator = max(
        1,
        min(
            len(frame_a),
            len(frame_b),
        ),
    )

    has_conflict = False
    best_identity_support = 0.0
    best_hybrid_score = 0.0
    best_hybrid_pair = None

    for left in members_a:
        for right in members_b:
            relation = pair_evidence.get(
                _pair_key(
                    left,
                    right,
                )
            )

            if relation is not None:
                if str(
                    relation.get(
                        "conflict",
                        "none",
                    )
                ) != "none":
                    has_conflict = True

                best_identity_support = max(
                    best_identity_support,
                    float(
                        relation.get(
                            "fused_support_score",
                            0.0,
                        )
                    ),
                )

            edge = hybrid_edges.get(
                _pair_key(
                    left,
                    right,
                )
            )

            if edge is not None:
                score = float(
                    edge.get(
                        "hybrid_score",
                        0.0,
                    )
                )

                if score > best_hybrid_score:
                    best_hybrid_score = score
                    best_hybrid_pair = (
                        left,
                        right,
                    )

    return CanonicalPairEvidence(
        canonical_a=canonical_a,
        canonical_b=canonical_b,
        simultaneous_frames=len(
            common_frames
        ),
        simultaneous_fraction=float(
            len(common_frames)
            / denominator
        ),
        median_distance_m=(
            float(
                np.median(
                    np.asarray(
                        distances,
                        dtype=np.float64,
                    )
                )
            )
            if distances
            else None
        ),
        q90_distance_m=(
            float(
                np.quantile(
                    np.asarray(
                        distances,
                        dtype=np.float64,
                    ),
                    0.90,
                )
            )
            if distances
            else None
        ),
        has_identity_conflict=(
            has_conflict
        ),
        best_identity_support=float(
            best_identity_support
        ),
        best_hybrid_score=float(
            best_hybrid_score
        ),
        best_hybrid_pair=(
            best_hybrid_pair
        ),
    )


def _canonical_frame_points(
    members: Sequence[str],
    *,
    unit_points: Mapping[
        str,
        Sequence[ObservedTrackPoint],
    ],
) -> dict[
    int,
    list[
        tuple[
            str,
            ObservedTrackPoint,
        ]
    ],
]:
    output = {}

    for key in members:
        for point in unit_points.get(
            key,
            (),
        ):
            output.setdefault(
                point.frame_index,
                [],
            ).append(
                (
                    key,
                    point,
                )
            )

    return output


def _pair_key(
    left: str,
    right: str,
) -> tuple[str, str]:
    return tuple(
        sorted(
            (
                left,
                right,
            )
        )
    )
