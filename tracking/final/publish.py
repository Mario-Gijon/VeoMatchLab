"""Publish validated Tracking V1 final trajectories as the canonical FINAL stage.

Hito 4F is a promotion step only. It performs no interpolation, smoothing,
identity repair, or kinematic recomputation.

Publication is allowed only when:
- the canonical identity population matches the final trajectories exactly;
- `rts-cv-v1` counts are internally consistent;
- all preserved segment breaks match the actual trajectory segmentation;
- `final-trajectory-audit-v1` reports publishable=true and blocker_count=0;
- the audit population matches the smoothing population exactly.

Review flags are intentionally preserved. They are evidence for downstream
consumers, not hidden errors and not automatic mutation instructions.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


FINAL_PUBLISH_VERSION = "final-publish-v1"
IDENTITY_PUBLISH_VERSION = "identity-publish-v1"
OBSERVATION_FUSION_VERSION = "observation-fusion-v1"
GAP_RECONSTRUCTION_VERSION = "safe-gap-v1"
SMOOTHING_VERSION = "rts-cv-v1"
FINAL_DIAGNOSTICS_VERSION = "final-trajectory-audit-v1"


@dataclass(frozen=True, slots=True)
class FinalPublishInputs:
    trajectories: tuple[dict[str, Any], ...]
    segment_breaks: tuple[dict[str, Any], ...]
    review_queue: dict[str, Any]
    smoothing_summary: dict[str, Any]
    diagnostics_summary: dict[str, Any]
    identities: tuple[dict[str, Any], ...]
    identity_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class FinalPublishResult:
    trajectories: tuple[dict[str, Any], ...]
    segment_breaks: tuple[dict[str, Any], ...]
    review_queue: dict[str, Any]
    summary: dict[str, Any]
    provenance: dict[str, Any]


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(
                    f"{source}:{line_number} is not a JSON object"
                )
            yield payload


def load_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} is not a JSON object")
    return payload


def _require_equal(label: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        raise ValueError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _identity_membership(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    assigned_tracklets: set[str] = set()

    for row in rows:
        identity_id = str(row["identity_id"])
        if identity_id in result:
            raise ValueError(f"duplicate canonical identity: {identity_id}")
        if row.get("canonical") is not True:
            raise ValueError(
                f"{identity_id} is not marked canonical"
            )
        if row.get("provisional") is True:
            raise ValueError(
                f"{identity_id} is still marked provisional"
            )

        members = tuple(
            sorted(
                str(value)
                for value in row["tracklet_ids"]
            )
        )
        if not members:
            raise ValueError(
                f"{identity_id} contains no tracklets"
            )
        if len(set(members)) != len(members):
            raise ValueError(
                f"{identity_id} contains duplicate tracklets"
            )

        overlap = assigned_tracklets & set(members)
        if overlap:
            raise ValueError(
                "canonical identities share tracklets: "
                f"{sorted(overlap)}"
            )
        assigned_tracklets.update(members)
        result[identity_id] = members

    return result


def _trajectory_population(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, tuple[str, ...]],
    int,
    int,
    list[tuple[str, int, int]],
]:
    membership: dict[str, tuple[str, ...]] = {}
    point_count = 0
    segment_count = 0
    expected_breaks: list[
        tuple[str, int, int]
    ] = []

    for row in rows:
        identity_id = str(row["identity_id"])
        if identity_id in membership:
            raise ValueError(
                f"duplicate final trajectory: {identity_id}"
            )

        _require_equal(
            f"{identity_id} trajectory version",
            row.get("trajectory_version"),
            SMOOTHING_VERSION,
        )

        tracklet_ids = tuple(
            sorted(
                str(value)
                for value in row["tracklet_ids"]
            )
        )
        membership[identity_id] = tracklet_ids

        points = row.get("points")
        if not isinstance(points, list) or not points:
            raise ValueError(
                f"{identity_id} final trajectory has no points"
            )

        declared_point_count = row.get(
            "point_count",
            len(points),
        )
        _require_equal(
            f"{identity_id} point count",
            declared_point_count,
            len(points),
        )

        declared_segment_count = int(
            row.get("segment_count", 0)
        )
        if declared_segment_count < 1:
            raise ValueError(
                f"{identity_id} segment_count must be >= 1"
            )

        seen_frames: set[int] = set()
        segments: list[str] = []
        previous = None

        for point in points:
            if str(point["identity_id"]) != identity_id:
                raise ValueError(
                    f"{identity_id} contains a foreign trajectory point"
                )

            frame = int(point["frame_index"])
            timestamp = float(point["timestamp_s"])
            if frame in seen_frames:
                raise ValueError(
                    f"{identity_id} repeats frame {frame}"
                )
            seen_frames.add(frame)

            for name in (
                "measurement_x",
                "measurement_z",
                "smoothed_x",
                "smoothed_z",
                "position_adjustment_m",
                "position_std_m",
            ):
                if not _finite(point.get(name)):
                    raise ValueError(
                        f"{identity_id} frame {frame} has invalid {name}"
                    )

            segment_id = str(point["segment_id"])
            if not segments or segments[-1] != segment_id:
                segments.append(segment_id)

            if previous is not None:
                previous_frame = int(
                    previous["frame_index"]
                )
                previous_time = float(
                    previous["timestamp_s"]
                )
                if frame <= previous_frame:
                    raise ValueError(
                        f"{identity_id} frames are not strictly increasing"
                    )
                if timestamp <= previous_time:
                    raise ValueError(
                        f"{identity_id} timestamps are not strictly increasing"
                    )

                if (
                    str(previous["segment_id"])
                    != segment_id
                ):
                    expected_breaks.append(
                        (
                            identity_id,
                            previous_frame,
                            frame,
                        )
                    )

            previous = point

        _require_equal(
            f"{identity_id} segment count",
            declared_segment_count,
            len(segments),
        )

        point_count += len(points)
        segment_count += declared_segment_count

    return (
        membership,
        point_count,
        segment_count,
        expected_breaks,
    )


def _break_keys(
    rows: Sequence[Mapping[str, Any]],
) -> set[tuple[str, int, int]]:
    result: set[tuple[str, int, int]] = set()
    for row in rows:
        key = (
            str(row["identity_id"]),
            int(row["left_frame_index"]),
            int(row["right_frame_index"]),
        )
        if key in result:
            raise ValueError(
                f"duplicate segment break: {key}"
            )
        if (
            float(row["right_timestamp_s"])
            <= float(row["left_timestamp_s"])
        ):
            raise ValueError(
                f"non-positive segment break duration: {key}"
            )
        result.add(key)
    return result


def prepare_final_publication(
    inputs: FinalPublishInputs,
) -> FinalPublishResult:
    """Validate all final guardrails and prepare canonical records."""

    identity_membership = _identity_membership(
        inputs.identities
    )

    _require_equal(
        "identity publisher version",
        inputs.identity_summary.get(
            "publisher_version"
        ),
        IDENTITY_PUBLISH_VERSION,
    )
    _require_equal(
        "identity stage completed",
        inputs.identity_summary.get(
            "identity_stage_completed"
        ),
        True,
    )
    _require_equal(
        "identity summary canonical",
        inputs.identity_summary.get(
            "canonical"
        ),
        True,
    )
    _require_equal(
        "identity summary identity count",
        inputs.identity_summary.get(
            "identity_count"
        ),
        len(identity_membership),
    )

    (
        trajectory_membership,
        point_count,
        segment_count,
        expected_breaks,
    ) = _trajectory_population(
        inputs.trajectories
    )

    if trajectory_membership != identity_membership:
        identity_ids = set(identity_membership)
        trajectory_ids = set(
            trajectory_membership
        )
        missing = sorted(
            identity_ids - trajectory_ids
        )
        extra = sorted(
            trajectory_ids - identity_ids
        )
        mismatched = sorted(
            identity_id
            for identity_id
            in identity_ids & trajectory_ids
            if (
                identity_membership[identity_id]
                != trajectory_membership[
                    identity_id
                ]
            )
        )
        raise ValueError(
            "final trajectory population does not match canonical "
            f"identities; missing={missing} extra={extra} "
            f"tracklet_membership_mismatch={mismatched}"
        )

    expected_break_keys = set(
        expected_breaks
    )
    actual_break_keys = _break_keys(
        inputs.segment_breaks
    )
    if actual_break_keys != expected_break_keys:
        raise ValueError(
            "segment break artifact does not match final trajectory "
            f"segmentation; missing="
            f"{sorted(expected_break_keys - actual_break_keys)} "
            f"extra={sorted(actual_break_keys - expected_break_keys)}"
        )

    smoothing = inputs.smoothing_summary
    _require_equal(
        "smoothing version",
        smoothing.get(
            "smoothing_version"
        ),
        SMOOTHING_VERSION,
    )
    _require_equal(
        "smoothing final stage completed",
        smoothing.get(
            "final_stage_completed"
        ),
        False,
    )
    _require_equal(
        "smoothing trajectory count",
        smoothing.get(
            "trajectory_count"
        ),
        len(inputs.trajectories),
    )
    _require_equal(
        "smoothing output point count",
        smoothing.get(
            "output_point_count"
        ),
        point_count,
    )
    _require_equal(
        "smoothing segment count",
        smoothing.get(
            "segment_count"
        ),
        segment_count,
    )
    _require_equal(
        "smoothing preserved gap count",
        smoothing.get(
            "preserved_gap_break_count"
        ),
        len(actual_break_keys),
    )

    diagnostics = inputs.diagnostics_summary
    _require_equal(
        "final diagnostics version",
        diagnostics.get(
            "diagnostics_version"
        ),
        FINAL_DIAGNOSTICS_VERSION,
    )
    _require_equal(
        "final diagnostics smoothing source",
        diagnostics.get(
            "source_smoothing_version"
        ),
        SMOOTHING_VERSION,
    )
    _require_equal(
        "final diagnostics publishable",
        diagnostics.get(
            "publishable"
        ),
        True,
    )
    _require_equal(
        "final diagnostics blocker count",
        diagnostics.get(
            "blocker_count"
        ),
        0,
    )
    _require_equal(
        "final diagnostics trajectory count",
        diagnostics.get(
            "trajectory_count"
        ),
        len(inputs.trajectories),
    )
    _require_equal(
        "final diagnostics point count",
        diagnostics.get(
            "point_count"
        ),
        point_count,
    )
    _require_equal(
        "final diagnostics segment count",
        diagnostics.get(
            "segment_count"
        ),
        segment_count,
    )
    _require_equal(
        "final diagnostics segment break count",
        diagnostics.get(
            "segment_break_count"
        ),
        len(actual_break_keys),
    )

    queue = inputs.review_queue
    _require_equal(
        "review queue diagnostics version",
        queue.get(
            "diagnostics_version"
        ),
        FINAL_DIAGNOSTICS_VERSION,
    )
    _require_equal(
        "review queue publishable",
        queue.get("publishable"),
        True,
    )

    provenance = {
        "publisher_version": (
            FINAL_PUBLISH_VERSION
        ),
        "identity_publisher_version": (
            IDENTITY_PUBLISH_VERSION
        ),
        "observation_fusion_version": (
            OBSERVATION_FUSION_VERSION
        ),
        "gap_reconstruction_version": (
            GAP_RECONSTRUCTION_VERSION
        ),
        "smoothing_version": (
            SMOOTHING_VERSION
        ),
        "diagnostics_version": (
            FINAL_DIAGNOSTICS_VERSION
        ),
    }

    canonical_trajectories = []
    for row in sorted(
        inputs.trajectories,
        key=lambda item: str(
            item["identity_id"]
        ),
    ):
        canonical_trajectories.append(
            {
                **dict(row),
                "canonical": True,
                "provisional": False,
                "published_by": (
                    FINAL_PUBLISH_VERSION
                ),
            }
        )

    canonical_breaks = tuple(
        {
            **dict(row),
            "canonical": True,
            "provisional": False,
            "published_by": (
                FINAL_PUBLISH_VERSION
            ),
        }
        for row in sorted(
            inputs.segment_breaks,
            key=lambda item: (
                str(item["identity_id"]),
                int(
                    item[
                        "left_frame_index"
                    ]
                ),
                int(
                    item[
                        "right_frame_index"
                    ]
                ),
            ),
        )
    )

    canonical_review_queue = {
        **dict(inputs.review_queue),
        "canonical": True,
        "provisional": False,
        "final_stage_completed": True,
        "published_by": FINAL_PUBLISH_VERSION,
    }

    summary = {
        "schema_version": 1,
        "publisher_version": (
            FINAL_PUBLISH_VERSION
        ),
        "canonical": True,
        "final_stage_completed": True,
        "publishable": True,
        "trajectory_count": (
            len(canonical_trajectories)
        ),
        "identity_count": (
            len(identity_membership)
        ),
        "point_count": point_count,
        "segment_count": segment_count,
        "segment_break_count": (
            len(canonical_breaks)
        ),
        "review_point_count": (
            diagnostics.get(
                "review_point_count",
                0,
            )
        ),
        "review_segment_count": (
            diagnostics.get(
                "review_segment_count",
                0,
            )
        ),
        "review_identity_count": (
            diagnostics.get(
                "review_identity_count",
                0,
            )
        ),
        "short_segment_count": (
            diagnostics.get(
                "short_segment_count",
                0,
            )
        ),
        "review_flag_counts": dict(
            diagnostics.get(
                "review_flag_counts",
                {},
            )
        ),
        "blocker_count": 0,
        "point_origin_counts": dict(
            diagnostics.get(
                "point_origin_counts",
                {},
            )
        ),
        "speed_mps": dict(
            diagnostics.get(
                "speed_mps",
                {},
            )
        ),
        "acceleration_mps2": dict(
            diagnostics.get(
                "acceleration_mps2",
                {},
            )
        ),
        "position_adjustment_m": dict(
            diagnostics.get(
                "position_adjustment_m",
                {},
            )
        ),
        "position_std_m": dict(
            diagnostics.get(
                "position_std_m",
                {},
            )
        ),
        "velocity_std_mps": dict(
            diagnostics.get(
                "velocity_std_mps",
                {},
            )
        ),
        "sampling_cadence": dict(
            diagnostics.get(
                "sampling_cadence",
                {},
            )
        ),
        "provenance": provenance,
    }

    return FinalPublishResult(
        trajectories=tuple(
            canonical_trajectories
        ),
        segment_breaks=canonical_breaks,
        review_queue=canonical_review_queue,
        summary=summary,
        provenance=provenance,
    )
