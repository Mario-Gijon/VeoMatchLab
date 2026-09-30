from __future__ import annotations

import json

from experiments.tracking.run_v1_publish_final import (
    run_publish_final,
)
from tracking.artifacts import (
    TrackingArtifactStore,
)
from tracking.contracts import (
    StageStatus,
    TrackingStage,
)


def _write_jsonl(path, rows):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        "\n".join(
            json.dumps(row)
            for row in rows
        )
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_json(path, payload):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _complete_stage(
    store,
    stage,
    version,
):
    store.begin_stage(
        stage,
        version=version,
    )
    artifact = store.write_json(
        stage,
        "index.json",
        {
            "stage": stage.value,
        },
    )
    store.complete_stage(
        stage,
        artifact=artifact,
    )


def _prepare_manifest(store):
    store.ensure_manifest()
    _complete_stage(
        store,
        TrackingStage.DETECTIONS,
        "test-v1",
    )
    _complete_stage(
        store,
        TrackingStage.TRACKLETS,
        "test-v1",
    )
    _complete_stage(
        store,
        TrackingStage.APPEARANCE,
        "test-v1",
    )
    _complete_stage(
        store,
        TrackingStage.IDENTITIES,
        "identity-publish-v1",
    )


def _paths(tmp_path):
    source = tmp_path / "source"
    trajectories = source / "trajectories.jsonl"
    breaks = source / "segment_breaks.jsonl"
    review_queue = source / "review_queue.json"
    smoothing_summary = source / "smoothing_summary.json"
    diagnostics_summary = source / "diagnostics_summary.json"
    identities = source / "identities.jsonl"
    identity_summary = source / "identity_summary.json"

    def point(frame, time_s, segment):
        return {
            "identity_id": "identity-001",
            "segment_id": segment,
            "segment_index": (
                1
                if segment.endswith("001")
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

    _write_jsonl(
        trajectories,
        [
            {
                "trajectory_version": "rts-cv-v1",
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a"],
                "tracklet_count": 1,
                "point_count": 4,
                "segment_count": 2,
                "points": [
                    point(
                        0,
                        0.0,
                        "identity-001:segment-001",
                    ),
                    point(
                        3,
                        0.1001,
                        "identity-001:segment-001",
                    ),
                    point(
                        30,
                        1.001,
                        "identity-001:segment-002",
                    ),
                    point(
                        33,
                        1.1011,
                        "identity-001:segment-002",
                    ),
                ],
                "metadata": {},
            }
        ],
    )
    _write_jsonl(
        breaks,
        [
            {
                "identity_id": "identity-001",
                "left_frame_index": 3,
                "right_frame_index": 30,
                "left_timestamp_s": 0.1001,
                "right_timestamp_s": 1.001,
                "duration_s": 0.9009,
                "reason": "preserved_gap_boundary",
            }
        ],
    )
    _write_json(
        review_queue,
        {
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
        },
    )
    _write_json(
        smoothing_summary,
        {
            "smoothing_version": "rts-cv-v1",
            "provisional": True,
            "final_stage_completed": False,
            "trajectory_count": 1,
            "output_point_count": 4,
            "segment_count": 2,
            "preserved_gap_break_count": 1,
        },
    )
    _write_json(
        diagnostics_summary,
        {
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
        },
    )
    _write_jsonl(
        identities,
        [
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a"],
                "canonical": True,
                "provisional": False,
            }
        ],
    )
    _write_json(
        identity_summary,
        {
            "publisher_version": (
                "identity-publish-v1"
            ),
            "canonical": True,
            "identity_stage_completed": True,
            "identity_count": 1,
            "tracklet_count": 1,
        },
    )

    return {
        "trajectories": trajectories,
        "segment_breaks": breaks,
        "review_queue": review_queue,
        "smoothing_summary": smoothing_summary,
        "diagnostics_summary": diagnostics_summary,
        "identities": identities,
        "identity_summary": identity_summary,
    }


def test_runner_publishes_and_completes_final_stage(
    tmp_path,
):
    data_root = tmp_path / "data"
    clip_dir = (
        data_root
        / "matches"
        / "match_test"
        / "clips"
        / "clip_test"
    )
    clip_dir.mkdir(parents=True)

    store = TrackingArtifactStore.for_clip(
        data_root,
        "match_test",
        "clip_test",
    )
    _prepare_manifest(store)

    summary = run_publish_final(
        store=store,
        paths=_paths(tmp_path),
    )

    assert summary[
        "manifest_status"
    ] == "completed"
    assert summary[
        "manifest_version"
    ] == "final-publish-v1"
    assert summary[
        "trajectory_count"
    ] == 1
    assert summary["point_count"] == 4

    manifest = store.load_manifest()
    stage = manifest.stage(
        TrackingStage.FINAL
    )
    assert (
        stage.status
        is StageStatus.COMPLETED
    )
    assert (
        stage.version
        == "final-publish-v1"
    )

    for name in (
        "trajectories.jsonl",
        "segment_breaks.jsonl",
        "review_queue.json",
        "summary.json",
        "index.json",
    ):
        assert (
            store.root
            / "final"
            / name
        ).is_file()


def test_validation_failure_leaves_final_not_run(
    tmp_path,
):
    data_root = tmp_path / "data"
    clip_dir = (
        data_root
        / "matches"
        / "match_test"
        / "clips"
        / "clip_test"
    )
    clip_dir.mkdir(parents=True)

    store = TrackingArtifactStore.for_clip(
        data_root,
        "match_test",
        "clip_test",
    )
    _prepare_manifest(store)

    paths = _paths(tmp_path)
    diagnostics = json.loads(
        paths[
            "diagnostics_summary"
        ].read_text(
            encoding="utf-8"
        )
    )
    diagnostics["publishable"] = False
    diagnostics["blocker_count"] = 1
    _write_json(
        paths["diagnostics_summary"],
        diagnostics,
    )

    try:
        run_publish_final(
            store=store,
            paths=paths,
        )
        assert False, (
            "expected final publication to fail"
        )
    except ValueError:
        pass

    manifest = store.load_manifest()
    assert (
        manifest.stage(
            TrackingStage.FINAL
        ).status
        is StageStatus.NOT_RUN
    )
    assert not (
        store.root / "final"
    ).exists()


def test_wrong_identity_stage_version_blocks_before_mutation(
    tmp_path,
):
    data_root = tmp_path / "data"
    clip_dir = (
        data_root
        / "matches"
        / "match_test"
        / "clips"
        / "clip_test"
    )
    clip_dir.mkdir(parents=True)

    store = TrackingArtifactStore.for_clip(
        data_root,
        "match_test",
        "clip_test",
    )
    store.ensure_manifest()
    _complete_stage(
        store,
        TrackingStage.DETECTIONS,
        "test-v1",
    )
    _complete_stage(
        store,
        TrackingStage.TRACKLETS,
        "test-v1",
    )
    _complete_stage(
        store,
        TrackingStage.APPEARANCE,
        "test-v1",
    )
    _complete_stage(
        store,
        TrackingStage.IDENTITIES,
        "old-identities-v0",
    )

    try:
        run_publish_final(
            store=store,
            paths=_paths(tmp_path),
        )
        assert False, (
            "expected identity version guard"
        )
    except RuntimeError:
        pass

    assert (
        store.load_manifest()
        .stage(
            TrackingStage.FINAL
        )
        .status
        is StageStatus.NOT_RUN
    )
