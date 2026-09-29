import json
import math

import pytest

from tracking.contracts import StageManifest, StageStatus, TrackingManifest, TrackingStage
from tracking.detections import CanonicalDetection, CanonicalDetectionFrame


def _detection(**overrides) -> CanonicalDetection:
    payload = {
        "detection_id": "det-0007",
        "frame_index": 42,
        "timestamp_s": 1.4,
        "confidence": 0.87,
        "x": 34.5,
        "z": 17.25,
        "source_lens": "left",
        "source_bbox_xyxy": (10, 20, 30, 70),
        "source_footpoint_xy": (20, 70),
        "representative_view": "far_left",
        "virtual_bbox_xyxy": (101, 202, 303, 707),
        "contributing_views": ("far_left", "near_center", "far_left"),
        "metadata": {"owner": "primary", "score": 0.87},
    }
    payload.update(overrides)
    return CanonicalDetection(**payload)


def test_canonical_detection_normalizes_valid_payload():
    detection = _detection(
        detection_id="  det-0007  ",
        representative_view=" far_left ",
        contributing_views=("far_left", "near_center", "far_left"),
    )

    assert detection.detection_id == "det-0007"
    assert detection.frame_index == 42
    assert detection.timestamp_s == 1.4
    assert detection.confidence == 0.87
    assert (detection.x, detection.z) == (34.5, 17.25)
    assert detection.source_lens == "left"
    assert detection.source_bbox_xyxy == (10.0, 20.0, 30.0, 70.0)
    assert detection.source_footpoint_xy == (20.0, 70.0)
    assert detection.representative_view == "far_left"
    assert detection.virtual_bbox_xyxy == (101.0, 202.0, 303.0, 707.0)
    assert detection.contributing_views == ("far_left", "near_center")
    assert detection.metadata == {"owner": "primary", "score": 0.87}


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("detection_id", ""),
        ("frame_index", -1),
        ("timestamp_s", -0.01),
        ("confidence", -0.01),
        ("confidence", 1.01),
        ("timestamp_s", math.nan),
        ("x", math.inf),
        ("z", -math.inf),
        ("source_lens", "center"),
        ("representative_view", "  "),
    ],
)
def test_canonical_detection_rejects_invalid_scalar_values(field, value):
    with pytest.raises(ValueError):
        _detection(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_bbox_xyxy", (1, 2, 3)),
        ("virtual_bbox_xyxy", (4, 5, 3, 6)),
        ("source_bbox_xyxy", (1, 2, math.nan, 4)),
        ("source_footpoint_xy", (1,)),
        ("source_footpoint_xy", (1, math.inf)),
        ("contributing_views", ("far_left", "")),
        ("contributing_views", ("far_left", None)),
    ],
)
def test_canonical_detection_rejects_invalid_geometry_or_views(field, value):
    with pytest.raises(ValueError):
        _detection(**{field: value})


def test_canonical_detection_rejects_non_json_metadata():
    with pytest.raises(ValueError, match="metadata"):
        _detection(metadata={"unsupported": {1, 2}})


def test_canonical_detection_dict_and_json_round_trips():
    detection = _detection()

    from_dict = CanonicalDetection.from_dict(detection.to_dict())
    from_json = CanonicalDetection.from_json(detection.to_json())

    assert from_dict.to_dict() == detection.to_dict()
    assert from_json.to_dict() == detection.to_dict()
    assert json.loads(detection.to_json()) == detection.to_dict()


def test_canonical_detection_frame_accepts_multiple_or_no_detections():
    first = _detection(detection_id="det-a")
    second = _detection(detection_id="det-b", x=35.0, source_lens="right")
    frame = CanonicalDetectionFrame(42, 1.4, (first, second))
    empty = CanonicalDetectionFrame(43, 1.4 + 1 / 30)

    assert [item.detection_id for item in frame.detections] == ["det-a", "det-b"]
    assert empty.detections == ()


def test_canonical_detection_frame_dict_and_json_round_trips():
    frame = CanonicalDetectionFrame(42, 1.4, (_detection(),))

    assert CanonicalDetectionFrame.from_dict(frame.to_dict()).to_dict() == frame.to_dict()
    assert CanonicalDetectionFrame.from_json(frame.to_json()).to_dict() == frame.to_dict()


@pytest.mark.parametrize(
    ("frame_index", "timestamp_s", "detections"),
    [
        (43, 1.4, (_detection(),)),
        (42, 1.5, (_detection(),)),
        (42, 1.4, (_detection(), _detection())),
    ],
)
def test_canonical_detection_frame_rejects_mismatched_or_duplicate_detections(
    frame_index, timestamp_s, detections
):
    with pytest.raises(ValueError):
        CanonicalDetectionFrame(frame_index, timestamp_s, detections)


def test_tracking_stage_and_status_values_are_stable():
    assert [stage.value for stage in TrackingStage] == [
        "detections", "tracklets", "appearance", "identities", "final",
    ]
    assert [status.value for status in StageStatus] == [
        "not_run", "running", "completed", "failed",
    ]
    with pytest.raises(ValueError):
        TrackingStage("unknown")
    with pytest.raises(ValueError):
        StageStatus("unknown")


def test_stage_manifest_defaults_and_round_trip():
    manifest = StageManifest()

    assert manifest.status is StageStatus.NOT_RUN
    assert StageManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()


def test_stage_manifest_accepts_relative_artifacts_and_json_safe_data():
    manifest = StageManifest(
        status="completed",
        artifact=" detections/frames.jsonl ",
        summary="reports/summary.json",
        version=" v1 ",
        config={"threshold": 0.4},
        metadata={"count": 3},
    )

    assert manifest.status is StageStatus.COMPLETED
    assert manifest.artifact == "detections/frames.jsonl"
    assert manifest.summary == "reports/summary.json"
    assert manifest.version == "v1"
    assert manifest.config == {"threshold": 0.4}
    assert manifest.metadata == {"count": 3}


@pytest.mark.parametrize("field", ["artifact", "summary"])
@pytest.mark.parametrize("value", ["/tmp/result.json", "../result.json", "nested/../../result.json"])
def test_stage_manifest_rejects_absolute_or_parent_paths(field, value):
    with pytest.raises(ValueError):
        StageManifest(**{field: value})


@pytest.mark.parametrize(
    "kwargs",
    [
        {"status": "unknown"},
        {"version": False},
        {"version": "  "},
        {"config": {"nan": math.nan}},
        {"metadata": {"set": {1}}},
    ],
)
def test_stage_manifest_rejects_invalid_contract_values(kwargs):
    with pytest.raises(ValueError):
        StageManifest(**kwargs)


def test_tracking_manifest_defaults_include_all_known_stages():
    manifest = TrackingManifest(match_id="match-1", clip_id="clip-1")

    assert manifest.schema_version == 1
    assert set(manifest.stages) == {stage.value for stage in TrackingStage}
    assert all(manifest.stage(stage).status is StageStatus.NOT_RUN for stage in TrackingStage)


def test_tracking_manifest_stage_access_set_and_round_trips():
    manifest = TrackingManifest(match_id="match-1", clip_id="clip-1", metadata={"source": "dev"})
    completed = StageManifest(status=StageStatus.COMPLETED, artifact="detections/frames.jsonl")
    manifest.set_stage(TrackingStage.DETECTIONS, completed)

    assert manifest.stage("detections") is completed
    assert TrackingManifest.from_dict(manifest.to_dict()).to_dict() == manifest.to_dict()
    assert TrackingManifest.from_json(manifest.to_json()).to_dict() == manifest.to_dict()
    assert json.loads(manifest.to_json())["stages"]["detections"]["artifact"] == "detections/frames.jsonl"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"match_id": "", "clip_id": "clip-1"},
        {"match_id": "match-1", "clip_id": ""},
        {"match_id": "match-1", "clip_id": "clip-1", "schema_version": 0},
        {"match_id": "match-1", "clip_id": "clip-1", "schema_version": True},
    ],
)
def test_tracking_manifest_rejects_invalid_identity_or_schema_version(kwargs):
    with pytest.raises(ValueError):
        TrackingManifest(**kwargs)
