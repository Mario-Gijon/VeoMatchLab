from __future__ import annotations

from tracking.canonical_preview import (
    PreviewFieldBounds,
    build_canonical_preview,
)
from tracking.interpolation import InterpolationConfig


def solution():
    return {
        "units": {
            "1:0": {
                "track_id": 1,
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": 4,
                "quality": 0.90,
            },
            "2:0": {
                "track_id": 2,
                "segment_index": 0,
                "start_frame": 4,
                "end_frame": 8,
                "quality": 0.80,
            },
            "3:0": {
                "track_id": 3,
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": 8,
                "quality": 0.40,
            },
        },
        "solution": {
            "identity_count": 1,
            "identities": [
                {
                    "canonical_id": 1,
                    "root_key": "1:0",
                    "member_keys": [
                        "1:0",
                        "2:0",
                    ],
                }
            ],
            "junk_keys": [
                "3:0",
            ],
        },
    }


def raw_track(
    track_id,
    x,
    *,
    kind="observed",
    confidence=None,
):
    row = {
        "track_id": track_id,
        "x": x,
        "z": 0.0,
        "vx": 1.0,
        "vz": 0.0,
        "lifecycle": "active",
        "position_kind": kind,
        "measurement_id": (
            f"m{track_id}"
            if kind == "observed"
            else None
        ),
    }

    if confidence is not None:
        row[
            "confidence"
        ] = confidence

    return row


def test_preview_remaps_members_and_drops_junk():
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "tracks": [
                raw_track(1, 0.0),
                raw_track(3, 30.0),
            ],
        },
        {
            "frame_index": 2,
            "timestamp_s": 2 / 30,
            "tracks": [
                raw_track(1, 0.2),
                raw_track(3, 30.2),
            ],
        },
        {
            "frame_index": 6,
            "timestamp_s": 6 / 30,
            "tracks": [
                raw_track(2, 0.6),
                raw_track(3, 30.6),
            ],
        },
        {
            "frame_index": 8,
            "timestamp_s": 8 / 30,
            "tracks": [
                raw_track(2, 0.8),
                raw_track(3, 30.8),
            ],
        },
    ]

    result = build_canonical_preview(
        rows,
        solution(),
        interpolation_config=InterpolationConfig(
            target_fps=30.0,
            max_gap_seconds=3.0,
        ),
    )

    assert result.report.canonical_identity_count == 1
    assert result.report.final_track_count == 1

    assert all(
        {
            track["track_id"]
            for track in row["tracks"]
        }
        <= {1}
        for row in result.rows
    )

    assert result.report.ignored_junk_sample_count == 4


def test_same_frame_collision_prefers_observed_then_higher_quality():
    payload = solution()

    rows = [
        {
            "frame_index": 4,
            "timestamp_s": 4 / 30,
            "tracks": [
                raw_track(
                    1,
                    4.0,
                    kind="predicted",
                ),
                raw_track(
                    2,
                    0.4,
                    kind="observed",
                ),
            ],
        }
    ]

    result = build_canonical_preview(
        rows,
        payload,
    )

    assert result.report.collision_frame_count == 1
    assert result.report.collision_sample_count == 1

    assert len(
        result.rows
    ) == 1

    track = result.rows[
        0
    ][
        "tracks"
    ][
        0
    ]

    assert track["track_id"] == 1
    assert abs(
        track["x"]
        - 0.4
    ) < 1e-9
    assert track["position_kind"] == "observed"


def test_preview_interpolates_under_canonical_id():
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "tracks": [
                raw_track(1, 0.0),
            ],
        },
        {
            "frame_index": 4,
            "timestamp_s": 4 / 30,
            "tracks": [
                raw_track(1, 0.4),
                raw_track(2, 0.4),
            ],
        },
        {
            "frame_index": 8,
            "timestamp_s": 8 / 30,
            "tracks": [
                raw_track(2, 0.8),
            ],
        },
    ]

    result = build_canonical_preview(
        rows,
        solution(),
        interpolation_config=InterpolationConfig(
            target_fps=30.0,
            max_gap_seconds=3.0,
        ),
    )

    frame_indices = [
        row["frame_index"]
        for row in result.rows
    ]

    assert frame_indices == list(
        range(
            0,
            9,
        )
    )

    assert result.report.inserted_sample_count >= 6

    for row in result.rows:
        assert [
            track["track_id"]
            for track in row["tracks"]
        ] == [1]


def test_established_player_is_kept_inside_continuation_corridor():
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "tracks": [
                raw_track(1, 0.0),
            ],
        },
        {
            "frame_index": 1,
            "timestamp_s": 1 / 30,
            "tracks": [
                {
                    **raw_track(1, 0.1),
                    "z": 35.0,
                },
            ],
        },
        {
            "frame_index": 2,
            "timestamp_s": 2 / 30,
            "tracks": [
                raw_track(1, 0.2),
            ],
        },
    ]

    result = build_canonical_preview(
        rows,
        solution(),
        field_bounds=PreviewFieldBounds(
            field_length_m=105.0,
            field_width_m=68.0,
            continuation_margin_m=1.5,
        ),
    )

    assert (
        result.report.ignored_outside_sample_count
        == 0
    )
    assert (
        result.report.continued_outside_sample_count
        == 1
    )

    # The sample must remain represented through the boundary transition.
    # RTS smoothing may move the displayed point slightly back inside the line,
    # so visibility/continuity is the invariant rather than z > 34 exactly.
    assert any(
        row["frame_index"] == 1
        and row["tracks"]
        for row in result.rows
    )


def test_samples_beyond_continuation_corridor_are_removed():
    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "tracks": [
                raw_track(1, 0.0),
            ],
        },
        {
            "frame_index": 1,
            "timestamp_s": 1 / 30,
            "tracks": [
                {
                    **raw_track(1, 0.1),
                    "z": 36.0,
                },
            ],
        },
        {
            "frame_index": 2,
            "timestamp_s": 2 / 30,
            "tracks": [
                raw_track(1, 0.2),
            ],
        },
    ]

    result = build_canonical_preview(
        rows,
        solution(),
        field_bounds=PreviewFieldBounds(
            field_length_m=105.0,
            field_width_m=68.0,
            continuation_margin_m=1.5,
        ),
    )

    assert (
        result.report.ignored_outside_sample_count
        == 1
    )

def test_outside_only_canonical_is_not_published_even_inside_corridor():
    payload = {
        "units": {
            "1:0": {
                "track_id": 1,
                "segment_index": 0,
                "start_frame": 0,
                "end_frame": 2,
                "quality": 0.9,
            },
        },
        "solution": {
            "identity_count": 1,
            "identities": [
                {
                    "canonical_id": 1,
                    "root_key": "1:0",
                    "member_keys": ["1:0"],
                },
            ],
            "junk_keys": [],
        },
    }

    rows = [
        {
            "frame_index": 0,
            "timestamp_s": 0.0,
            "tracks": [
                {
                    **raw_track(
                        1,
                        0.0,
                        kind="observed",
                    ),
                    "z": 34.8,
                },
            ],
        },
        {
            "frame_index": 1,
            "timestamp_s": 1 / 30,
            "tracks": [
                {
                    **raw_track(
                        1,
                        0.1,
                        kind="observed",
                    ),
                    "z": 35.0,
                },
            ],
        },
    ]

    result = build_canonical_preview(
        rows,
        payload,
        field_bounds=PreviewFieldBounds(
            field_length_m=105.0,
            field_width_m=68.0,
            continuation_margin_m=1.5,
        ),
    )

    assert result.rows == ()
    assert (
        result.report.canonical_ids_without_observed_inside
        == (1,)
    )
    assert result.report.final_track_count == 0
