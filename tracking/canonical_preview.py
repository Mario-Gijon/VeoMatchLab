"""Non-destructive V2.8B canonical tracking preview.

This module takes:

  * raw forward-tracker frame rows (the immutable 15 Hz diagnostic), and
  * a V2.8A-V5 identity-graph solution,

and reconstructs exactly the selected canonical identities for visual review.

It does NOT modify the current production tracking artifact.

Pipeline:
    raw tracker rows
        -> keep only units selected by the V5 identity graph
        -> remap raw/segment units to canonical IDs
        -> resolve same-frame duplicate samples deterministically
        -> RTS smooth canonical trajectories
        -> interpolate short gaps to 30 FPS
        -> preview JSONL rows consumed by the existing 2D pitch

OBSERVED / PREDICTED / INTERPOLATED are preserved internally.  The frontend
does not need to render them differently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Sequence

from .interpolation import (
    InterpolationConfig,
    interpolate_trajectories,
)
from .models import (
    PositionKind,
    TrackLifecycle,
    TrackSample,
)
from .smoothing import (
    SmoothingConfig,
    smooth_trajectories,
)


@dataclass(frozen=True)
class CanonicalUnit:
    unit_key: str
    raw_track_id: int
    segment_index: int
    start_frame: int
    end_frame: int
    quality: float
    canonical_id: int
    is_root: bool


@dataclass(frozen=True)
class PreviewReconstructionReport:
    canonical_identity_count: int
    selected_unit_count: int
    junk_unit_count: int

    input_frame_count: int
    input_sample_count: int

    mapped_sample_count: int
    ignored_junk_sample_count: int
    ignored_unmapped_sample_count: int

    collision_frame_count: int
    collision_sample_count: int
    observed_collision_frame_count: int
    max_collision_distance_m: float

    pre_smoothing_track_count: int
    final_track_count: int
    final_frame_count: int
    final_sample_count: int
    inserted_sample_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PreviewBuildResult:
    rows: tuple[dict[str, Any], ...]
    report: PreviewReconstructionReport
    canonical_members: Mapping[int, tuple[str, ...]]
    raw_to_units: Mapping[int, tuple[str, ...]]

    def report_dict(self) -> dict[str, Any]:
        return {
            "summary": self.report.to_dict(),
            "canonical_members": {
                str(key): list(value)
                for key, value in sorted(
                    self.canonical_members.items()
                )
            },
            "raw_to_units": {
                str(key): list(value)
                for key, value in sorted(
                    self.raw_to_units.items()
                )
            },
        }


@dataclass(frozen=True)
class _Candidate:
    unit: CanonicalUnit
    sample: TrackSample


_POSITION_PRIORITY = {
    PositionKind.OBSERVED: 3,
    PositionKind.PREDICTED: 2,
    PositionKind.INTERPOLATED: 1,
}

_LIFECYCLE_PRIORITY = {
    TrackLifecycle.ACTIVE: 3,
    TrackLifecycle.TENTATIVE: 2,
    TrackLifecycle.COASTING: 1,
    TrackLifecycle.LOST: 0,
}


def build_canonical_preview(
    raw_rows: Iterable[Mapping[str, Any]],
    identity_solution: Mapping[str, Any],
    *,
    smoothing_config: SmoothingConfig | None = None,
    interpolation_config: InterpolationConfig | None = None,
) -> PreviewBuildResult:
    rows = tuple(
        sorted(
            (dict(row) for row in raw_rows),
            key=lambda row: (
                int(row["frame_index"]),
                float(
                    row.get(
                        "timestamp_s",
                        row.get("timestamp", 0.0),
                    )
                ),
            ),
        )
    )

    (
        units,
        canonical_members,
        raw_to_units,
        junk_keys,
    ) = _parse_solution(
        identity_solution
    )

    trajectories, reconstruction = _reconstruct_trajectories(
        rows,
        units=units,
        junk_keys=junk_keys,
    )

    smoothed, smoothing_report = smooth_trajectories(
        trajectories,
        config=smoothing_config,
    )
    final, interpolation_report = interpolate_trajectories(
        smoothed,
        config=interpolation_config,
    )

    preview_rows = _trajectories_to_rows(
        final
    )

    report = PreviewReconstructionReport(
        canonical_identity_count=len(
            canonical_members
        ),
        selected_unit_count=len(
            units
        ),
        junk_unit_count=len(
            junk_keys
        ),
        input_frame_count=len(
            rows
        ),
        input_sample_count=reconstruction[
            "input_sample_count"
        ],
        mapped_sample_count=reconstruction[
            "mapped_sample_count"
        ],
        ignored_junk_sample_count=reconstruction[
            "ignored_junk_sample_count"
        ],
        ignored_unmapped_sample_count=reconstruction[
            "ignored_unmapped_sample_count"
        ],
        collision_frame_count=reconstruction[
            "collision_frame_count"
        ],
        collision_sample_count=reconstruction[
            "collision_sample_count"
        ],
        observed_collision_frame_count=reconstruction[
            "observed_collision_frame_count"
        ],
        max_collision_distance_m=reconstruction[
            "max_collision_distance_m"
        ],
        pre_smoothing_track_count=len(
            trajectories
        ),
        final_track_count=len(
            final
        ),
        final_frame_count=len(
            preview_rows
        ),
        final_sample_count=sum(
            len(value)
            for value in final.values()
        ),
        inserted_sample_count=(
            interpolation_report.total_inserted_samples
        ),
    )

    # Keep these variables intentionally referenced: their reports are useful
    # during debugging even though the compact public report above is enough for
    # the app preview.
    _ = smoothing_report

    return PreviewBuildResult(
        rows=tuple(
            preview_rows
        ),
        report=report,
        canonical_members=canonical_members,
        raw_to_units=raw_to_units,
    )


def _parse_solution(
    payload: Mapping[str, Any],
) -> tuple[
    dict[str, CanonicalUnit],
    dict[int, tuple[str, ...]],
    dict[int, tuple[str, ...]],
    frozenset[str],
]:
    raw_units = payload.get(
        "units"
    )
    solution = payload.get(
        "solution"
    )

    if not isinstance(
        raw_units,
        Mapping,
    ):
        raise ValueError(
            "identity solution requires top-level units"
        )

    if not isinstance(
        solution,
        Mapping,
    ):
        raise ValueError(
            "identity solution requires top-level solution"
        )

    identities = solution.get(
        "identities"
    )
    if not isinstance(
        identities,
        Sequence,
    ):
        raise ValueError(
            "identity solution requires solution.identities"
        )

    expected_count = int(
        solution.get(
            "identity_count",
            len(identities),
        )
    )

    if len(
        identities
    ) != expected_count:
        raise ValueError(
            "identity solution does not contain the declared identity count"
        )

    selected_units: dict[
        str,
        CanonicalUnit
    ] = {}
    canonical_members: dict[
        int,
        tuple[str, ...]
    ] = {}

    for identity in identities:
        canonical_id = int(
            identity[
                "canonical_id"
            ]
        )
        root_key = str(
            identity[
                "root_key"
            ]
        )
        members = tuple(
            str(key)
            for key in identity[
                "member_keys"
            ]
        )

        if canonical_id in canonical_members:
            raise ValueError(
                f"duplicate canonical_id {canonical_id}"
            )

        if not members:
            raise ValueError(
                f"canonical {canonical_id} has no members"
            )

        canonical_members[
            canonical_id
        ] = members

        for key in members:
            raw = raw_units.get(
                key
            )
            if not isinstance(
                raw,
                Mapping,
            ):
                raise ValueError(
                    f"selected unit {key} is missing from units"
                )

            if key in selected_units:
                raise ValueError(
                    f"unit {key} belongs to multiple canonicals"
                )

            selected_units[
                key
            ] = CanonicalUnit(
                unit_key=key,
                raw_track_id=int(
                    raw["track_id"]
                ),
                segment_index=int(
                    raw["segment_index"]
                ),
                start_frame=int(
                    raw["start_frame"]
                ),
                end_frame=int(
                    raw["end_frame"]
                ),
                quality=float(
                    raw["quality"]
                ),
                canonical_id=canonical_id,
                is_root=(
                    key == root_key
                ),
            )

    if sorted(
        canonical_members
    ) != list(
        range(
            1,
            len(canonical_members)
            + 1,
        )
    ):
        raise ValueError(
            "preview expects dense canonical IDs 1..K"
        )

    junk_keys = frozenset(
        str(key)
        for key in solution.get(
            "junk_keys",
            ()
        )
    )

    overlap = (
        set(
            selected_units
        )
        & set(
            junk_keys
        )
    )
    if overlap:
        raise ValueError(
            "selected units cannot also be junk: "
            + ", ".join(
                sorted(
                    overlap
                )
            )
        )

    raw_to_units_dict: dict[
        int,
        list[str]
    ] = {}

    for unit in selected_units.values():
        raw_to_units_dict.setdefault(
            unit.raw_track_id,
            [],
        ).append(
            unit.unit_key
        )

    raw_to_units = {
        track_id: tuple(
            sorted(
                keys,
                key=lambda key: (
                    selected_units[
                        key
                    ].start_frame,
                    selected_units[
                        key
                    ].end_frame,
                    key,
                ),
            )
        )
        for track_id, keys in raw_to_units_dict.items()
    }

    return (
        selected_units,
        canonical_members,
        raw_to_units,
        junk_keys,
    )


def _reconstruct_trajectories(
    rows: Sequence[Mapping[str, Any]],
    *,
    units: Mapping[str, CanonicalUnit],
    junk_keys: frozenset[str],
) -> tuple[
    dict[int, tuple[TrackSample, ...]],
    dict[str, Any],
]:
    units_by_raw: dict[
        int,
        list[CanonicalUnit]
    ] = {}

    for unit in units.values():
        units_by_raw.setdefault(
            unit.raw_track_id,
            [],
        ).append(
            unit
        )

    for values in units_by_raw.values():
        values.sort(
            key=lambda item: (
                item.start_frame,
                item.end_frame,
                item.unit_key,
            )
        )

    # Junk raw/segment ranges are needed only for reporting why a sample was
    # discarded. They are reconstructed from the unit key's raw track ID when
    # possible later; selected ranges remain authoritative for preview mapping.
    junk_raw_ids = {
        _raw_track_id_from_unit_key(
            key
        )
        for key in junk_keys
    }

    candidates: dict[
        tuple[int, int],
        list[_Candidate]
    ] = {}

    input_sample_count = 0
    mapped_sample_count = 0
    ignored_junk_sample_count = 0
    ignored_unmapped_sample_count = 0

    for row in rows:
        frame_index = int(
            row["frame_index"]
        )
        timestamp = float(
            row.get(
                "timestamp_s",
                row.get("timestamp"),
            )
        )

        for raw in row.get(
            "tracks",
            (),
        ):
            input_sample_count += 1
            raw_track_id = int(
                raw["track_id"]
            )

            matching = [
                unit
                for unit in units_by_raw.get(
                    raw_track_id,
                    ()
                )
                if (
                    unit.start_frame
                    <= frame_index
                    <= unit.end_frame
                )
            ]

            if len(
                matching
            ) > 1:
                raise ValueError(
                    f"raw track {raw_track_id} frame {frame_index} "
                    "matches multiple selected identity units"
                )

            if not matching:
                if raw_track_id in junk_raw_ids:
                    ignored_junk_sample_count += 1
                else:
                    ignored_unmapped_sample_count += 1
                continue

            unit = matching[0]
            sample = _track_sample(
                raw,
                canonical_id=unit.canonical_id,
                frame_index=frame_index,
                timestamp=timestamp,
            )

            candidates.setdefault(
                (
                    unit.canonical_id,
                    frame_index,
                ),
                [],
            ).append(
                _Candidate(
                    unit=unit,
                    sample=sample,
                )
            )

            mapped_sample_count += 1

    trajectories: dict[
        int,
        list[TrackSample]
    ] = {}

    collision_frame_count = 0
    collision_sample_count = 0
    observed_collision_frame_count = 0
    max_collision_distance_m = 0.0

    for (
        canonical_id,
        _frame_index,
    ), rows_for_frame in sorted(
        candidates.items()
    ):
        if len(
            rows_for_frame
        ) > 1:
            collision_frame_count += 1
            collision_sample_count += (
                len(
                    rows_for_frame
                )
                - 1
            )

            observed = [
                item
                for item in rows_for_frame
                if (
                    item.sample.position_kind
                    is PositionKind.OBSERVED
                )
            ]

            if len(
                observed
            ) > 1:
                observed_collision_frame_count += 1

            for index, left in enumerate(
                rows_for_frame
            ):
                for right in rows_for_frame[
                    index + 1:
                ]:
                    max_collision_distance_m = max(
                        max_collision_distance_m,
                        math.hypot(
                            left.sample.x
                            - right.sample.x,
                            left.sample.z
                            - right.sample.z,
                        ),
                    )

        chosen = max(
            rows_for_frame,
            key=_candidate_rank,
        )

        trajectories.setdefault(
            canonical_id,
            [],
        ).append(
            chosen.sample
        )

    result = {
        canonical_id: tuple(
            sorted(
                samples,
                key=lambda sample: (
                    sample.frame_index,
                    sample.timestamp,
                ),
            )
        )
        for canonical_id, samples in sorted(
            trajectories.items()
        )
    }

    return (
        result,
        {
            "input_sample_count": input_sample_count,
            "mapped_sample_count": mapped_sample_count,
            "ignored_junk_sample_count": (
                ignored_junk_sample_count
            ),
            "ignored_unmapped_sample_count": (
                ignored_unmapped_sample_count
            ),
            "collision_frame_count": (
                collision_frame_count
            ),
            "collision_sample_count": (
                collision_sample_count
            ),
            "observed_collision_frame_count": (
                observed_collision_frame_count
            ),
            "max_collision_distance_m": float(
                max_collision_distance_m
            ),
        },
    )


def _candidate_rank(
    candidate: _Candidate,
) -> tuple[
    int,
    float,
    float,
    int,
    int,
    str,
]:
    sample = candidate.sample
    confidence = (
        float(
            sample.confidence
        )
        if (
            sample.confidence is not None
            and math.isfinite(
                float(
                    sample.confidence
                )
            )
        )
        else -1.0
    )

    return (
        _POSITION_PRIORITY[
            sample.position_kind
        ],
        candidate.unit.quality,
        confidence,
        1
        if candidate.unit.is_root
        else 0,
        _LIFECYCLE_PRIORITY[
            sample.lifecycle
        ],
        candidate.unit.unit_key,
    )


def _track_sample(
    raw: Mapping[str, Any],
    *,
    canonical_id: int,
    frame_index: int,
    timestamp: float,
) -> TrackSample:
    try:
        lifecycle = TrackLifecycle(
            str(
                raw[
                    "lifecycle"
                ]
            )
        )
    except (
        KeyError,
        ValueError,
    ) as exc:
        raise ValueError(
            "raw preview sample requires a valid lifecycle"
        ) from exc

    try:
        position_kind = PositionKind(
            str(
                raw[
                    "position_kind"
                ]
            )
        )
    except (
        KeyError,
        ValueError,
    ) as exc:
        raise ValueError(
            "raw preview sample requires a valid position_kind"
        ) from exc

    confidence_raw = raw.get(
        "confidence"
    )
    confidence = (
        float(
            confidence_raw
        )
        if (
            confidence_raw is not None
            and not isinstance(
                confidence_raw,
                bool,
            )
            and math.isfinite(
                float(
                    confidence_raw
                )
            )
        )
        else None
    )

    return TrackSample(
        track_id=canonical_id,
        frame_index=frame_index,
        timestamp=timestamp,
        x=float(
            raw["x"]
        ),
        z=float(
            raw["z"]
        ),
        vx=float(
            raw.get(
                "vx",
                0.0,
            )
        ),
        vz=float(
            raw.get(
                "vz",
                0.0,
            )
        ),
        lifecycle=lifecycle,
        position_kind=position_kind,
        measurement_id=raw.get(
            "measurement_id"
        ),
        confidence=confidence,
    )


def _trajectories_to_rows(
    trajectories: Mapping[
        int,
        Sequence[TrackSample],
    ],
) -> list[
    dict[str, Any]
]:
    grouped: dict[
        int,
        list[TrackSample]
    ] = {}

    for canonical_id, samples in trajectories.items():
        for sample in samples:
            if sample.track_id != canonical_id:
                raise ValueError(
                    f"canonical trajectory {canonical_id} contains "
                    f"track_id={sample.track_id}"
                )

            grouped.setdefault(
                sample.frame_index,
                [],
            ).append(
                sample
            )

    rows = []

    for frame_index in sorted(
        grouped
    ):
        samples = sorted(
            grouped[
                frame_index
            ],
            key=lambda sample: sample.track_id,
        )

        timestamp = float(
            samples[
                0
            ].timestamp
        )

        if any(
            not math.isclose(
                sample.timestamp,
                timestamp,
                abs_tol=1e-7,
            )
            for sample in samples[
                1:
            ]
        ):
            raise ValueError(
                f"frame {frame_index} has inconsistent timestamps"
            )

        rows.append(
            {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "timestamp_s": timestamp,
                "preview_mode": "v28b",
                "tracks": [
                    {
                        "track_id": sample.track_id,
                        "canonical_id": sample.track_id,
                        "x": sample.x,
                        "z": sample.z,
                        "vx": sample.vx,
                        "vz": sample.vz,
                        "lifecycle": sample.lifecycle.value,
                        "position_kind": sample.position_kind.value,
                        "measurement_id": sample.measurement_id,
                    }
                    for sample in samples
                ],
            }
        )

    return rows


def _raw_track_id_from_unit_key(
    key: str,
) -> int:
    try:
        return int(
            str(
                key
            ).split(
                ":",
                1,
            )[
                0
            ]
        )
    except (
        ValueError,
        IndexError,
    ) as exc:
        raise ValueError(
            f"invalid identity unit key {key!r}"
        ) from exc


def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        Enum,
    ):
        return json_safe(
            value.value
        )

    if isinstance(
        value,
        bool,
    ):
        return value

    if isinstance(
        value,
        int,
    ):
        return value

    if isinstance(
        value,
        float,
    ):
        return (
            value
            if math.isfinite(
                value
            )
            else None
        )

    if isinstance(
        value,
        Mapping,
    ):
        return {
            str(key): json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    return value
