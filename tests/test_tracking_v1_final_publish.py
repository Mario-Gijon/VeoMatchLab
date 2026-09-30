from __future__ import annotations

import copy

import pytest

from tracking.final.publish import (
    FinalPublishInputs,
    prepare_final_publication,
)


def _point(frame, time_s, segment):
    return {
        "identity_id": "identity-001",
        "segment_id": segment,
        "segment_index": (
            1 if segment.endswith("001")
            else 2
        ),
        "frame_index": frame,
        "timestamp_s": time_s,
        "measurement_x": float(frame),
        "measurement_z": 0.0,
        "smoothed_x": float(frame) + 0.1,
        "smoothed_z": 0.0,
        "measurement_confidence": 0.9,
        "measurement_sigma_m": 0.3,
        "origin": "observed",
        "fusion_mode": "single",
        "source_tracklet_ids": ["trk-a"],
        "source_detection_ids": [
            f"det-{frame}"
        ],
        "source_count": 1,
        "position_spread_m": 0.0,
        "position_adjustment_m": 0.1,
        "position_std_m": 0.15,
        "vx_mps": 1.0,
        "vz_mps": 0.0,
        "speed_mps": 1.0,
        "ax_mps2": 1.0,
        "az_mps2": 0.0,
        "acceleration_mps2": 1.0,
        "velocity_std_mps": 0.9,
        "kinematics_quality": "rts_cv",
        "metadata": {
            "smoothing_applied": True
        },
    }


def _inputs():
    trajectories = (
        {
            "trajectory_version": "rts-cv-v1",
            "identity_id": "identity-001",
            "tracklet_ids": ["trk-a"],
            "tracklet_count": 1,
            "point_count": 4,
            "segment_count": 2,
            "points": [
                _point(
                    0,
                    0.0,
                    "identity-001:segment-001",
                ),
                _point(
                    3,
                    0.1001,
                    "identity-001:segment-001",
                ),
                _point(
                    30,
                    1.001,
                    "identity-001:segment-002",
                ),
                _point(
                    33,
                    1.1011,
                    "identity-001:segment-002",
                ),
            ],
            "metadata": {},
        },
    )
    segment_breaks = (
        {
            "identity_id": "identity-001",
            "left_frame_index": 3,
            "right_frame_index": 30,
            "left_timestamp_s": 0.1001,
            "right_timestamp_s": 1.001,
            "duration_s": 0.9009,
            "reason": "preserved_gap_boundary",
        },
    )
    review_queue = {
        "diagnostics_version": (
            "final-trajectory-audit-v1"
        ),
        "provisional": True,
        "publishable": True,
        "speed_reviews": [],
        "acceleration_reviews": [],
        "adjustment_reviews": [],
        "uncertainty_reviews": [],
        "short_segments": [],
    }
    smoothing_summary = {
        "smoothing_version": "rts-cv-v1",
        "provisional": True,
        "final_stage_completed": False,
        "trajectory_count": 1,
        "output_point_count": 4,
        "segment_count": 2,
        "preserved_gap_break_count": 1,
    }
    diagnostics_summary = {
        "diagnostics_version": (
            "final-trajectory-audit-v1"
        ),
        "source_smoothing_version": (
            "rts-cv-v1"
        ),
        "publishable": True,
        "blocker_count": 0,
        "trajectory_count": 1,
        "point_count": 4,
        "segment_count": 2,
        "segment_break_count": 1,
        "review_point_count": 0,
        "review_segment_count": 0,
        "review_identity_count": 0,
        "short_segment_count": 0,
        "review_flag_counts": {},
        "point_origin_counts": {
            "observed": 4
        },
    }
    identities = (
        {
            "identity_id": "identity-001",
            "tracklet_ids": ["trk-a"],
            "canonical": True,
            "provisional": False,
        },
    )
    identity_summary = {
        "publisher_version": (
            "identity-publish-v1"
        ),
        "canonical": True,
        "identity_stage_completed": True,
        "identity_count": 1,
        "tracklet_count": 1,
    }

    return FinalPublishInputs(
        trajectories=trajectories,
        segment_breaks=segment_breaks,
        review_queue=review_queue,
        smoothing_summary=smoothing_summary,
        diagnostics_summary=diagnostics_summary,
        identities=identities,
        identity_summary=identity_summary,
    )


def _replace(inputs, **changes):
    values = {
        "trajectories": inputs.trajectories,
        "segment_breaks": inputs.segment_breaks,
        "review_queue": inputs.review_queue,
        "smoothing_summary": inputs.smoothing_summary,
        "diagnostics_summary": (
            inputs.diagnostics_summary
        ),
        "identities": inputs.identities,
        "identity_summary": (
            inputs.identity_summary
        ),
    }
    values.update(changes)
    return FinalPublishInputs(**values)


def test_prepare_final_publication_marks_outputs_canonical():
    result = prepare_final_publication(
        _inputs()
    )

    assert result.summary[
        "final_stage_completed"
    ] is True
    assert result.summary[
        "trajectory_count"
    ] == 1
    assert result.summary[
        "point_count"
    ] == 4
    assert result.trajectories[0][
        "canonical"
    ] is True
    assert result.trajectories[0][
        "provisional"
    ] is False
    assert result.review_queue[
        "final_stage_completed"
    ] is True


def test_review_flags_do_not_block_publication():
    inputs = _inputs()
    diagnostics = dict(
        inputs.diagnostics_summary
    )
    diagnostics.update(
        {
            "review_point_count": 2,
            "review_segment_count": 1,
            "review_identity_count": 1,
            "review_flag_counts": {
                "speed_gt_12mps": 2
            },
        }
    )
    queue = dict(inputs.review_queue)
    queue["speed_reviews"] = [
        {"identity_id": "identity-001"},
        {"identity_id": "identity-001"},
    ]

    result = prepare_final_publication(
        _replace(
            inputs,
            diagnostics_summary=diagnostics,
            review_queue=queue,
        )
    )
    assert result.summary[
        "review_point_count"
    ] == 2
    assert result.summary[
        "blocker_count"
    ] == 0


def test_diagnostics_blocker_rejects_publication():
    inputs = _inputs()
    diagnostics = dict(
        inputs.diagnostics_summary
    )
    diagnostics["publishable"] = False
    diagnostics["blocker_count"] = 1

    with pytest.raises(
        ValueError,
        match="publishable",
    ):
        prepare_final_publication(
            _replace(
                inputs,
                diagnostics_summary=(
                    diagnostics
                ),
            )
        )


def test_identity_population_mismatch_rejects_publication():
    inputs = _inputs()
    identities = (
        {
            "identity_id": "identity-001",
            "tracklet_ids": ["trk-other"],
            "canonical": True,
            "provisional": False,
        },
    )

    with pytest.raises(
        ValueError,
        match="population does not match",
    ):
        prepare_final_publication(
            _replace(
                inputs,
                identities=identities,
            )
        )


def test_segment_break_mismatch_rejects_publication():
    inputs = _inputs()

    with pytest.raises(
        ValueError,
        match="segment break artifact",
    ):
        prepare_final_publication(
            _replace(
                inputs,
                segment_breaks=(),
            )
        )


def test_smoothing_count_mismatch_rejects_publication():
    inputs = _inputs()
    smoothing = dict(
        inputs.smoothing_summary
    )
    smoothing[
        "output_point_count"
    ] = 99

    with pytest.raises(
        ValueError,
        match="smoothing output point count",
    ):
        prepare_final_publication(
            _replace(
                inputs,
                smoothing_summary=smoothing,
            )
        )
