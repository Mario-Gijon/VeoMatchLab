from __future__ import annotations

from dataclasses import dataclass
import json

import pytest

from tracking.detections import CanonicalDetection, CanonicalDetectionFrame
from tracking.pipeline import (
    build_tracklets_stage,
    publish_detections_stage,
    run_detection_and_tracklet_stages,
)


@dataclass
class Artifact:
    ref: str
    record_count: int | None = None


@dataclass
class Stage:
    artifact: str | None = None


class Manifest:
    def __init__(self, stages):
        self.stages = stages

    def stage(self, stage):
        return self.stages.setdefault(stage, Stage())


class MemoryStore:
    def __init__(self):
        self.rows = {}
        self.json = {}
        self.stages = {}
        self.calls = []
        self.failed = []

    def begin_stage(self, stage, **kwargs):
        if stage == "tracklets" and not self.stages.get("detections", Stage()).artifact:
            raise RuntimeError("detections prerequisite missing")
        self.calls.append(("begin", stage))
        self.stages[stage] = Stage()
        return Manifest(self.stages)

    def write_jsonl(self, stage, relative_path, records):
        ref = f"{stage}/{relative_path}"
        materialized = [
            item.to_dict() if hasattr(item, "to_dict") else dict(item)
            for item in records
        ]
        self.rows[ref] = materialized
        self.calls.append(("jsonl", ref))
        return Artifact(ref, len(materialized))

    def write_json(self, stage, relative_path, payload, indent=2):
        ref = f"{stage}/{relative_path}"
        json.dumps(payload, allow_nan=False)
        self.json[ref] = payload
        self.calls.append(("json", ref))
        return Artifact(ref)

    def complete_stage(self, stage, *, artifact, summary=None, metadata=None):
        self.stages[stage] = Stage(artifact.ref)
        self.calls.append(("complete", stage))
        return Manifest(self.stages)

    def fail_stage(self, stage, error, metadata=None):
        self.failed.append((stage, str(error)))
        self.calls.append(("fail", stage))
        return Manifest(self.stages)

    def load_manifest(self):
        return Manifest(self.stages)

    def iter_jsonl(self, ref):
        yield from self.rows[ref]


def detection(frame, suffix, x):
    return CanonicalDetection(
        detection_id=f"{frame}-{suffix}",
        frame_index=frame,
        timestamp_s=frame / 10,
        confidence=0.9,
        x=x,
        z=20.0,
    )


def frames():
    return [
        CanonicalDetectionFrame(0, 0.0, (detection(0, "a", 10.0),)),
        CanonicalDetectionFrame(1, 0.1, (detection(1, "a", 10.2),)),
        CanonicalDetectionFrame(2, 0.2, (detection(2, "a", 10.4),)),
    ]


def test_publish_detections_stage_writes_frames_summary_and_completes():
    store = MemoryStore()

    summary = publish_detections_stage(store, frames())

    assert store.stages["detections"].artifact == "detections/frames.jsonl"
    assert len(store.rows["detections/frames.jsonl"]) == 3
    assert summary["frame_count"] == 3
    assert summary["detection_count"] == 3
    assert summary["empty_frame_count"] == 0
    assert summary["confidence_mean"] == pytest.approx(0.9)
    assert store.json["detections/summary.json"] == summary


def test_detection_summary_counts_empty_and_multiview_frames():
    store = MemoryStore()
    multi = detection(0, "a", 10.0)
    multi.contributing_views = ("far_c1", "far_c2")

    summary = publish_detections_stage(
        store,
        [
            CanonicalDetectionFrame(0, 0.0, (multi,)),
            CanonicalDetectionFrame(1, 0.1, ()),
        ],
    )

    assert summary["empty_frame_count"] == 1
    assert summary["multiview_detection_count"] == 1
    assert summary["detections_per_frame_mean"] == 0.5


def test_publish_detections_rejects_non_monotonic_input_and_marks_failed():
    store = MemoryStore()

    with pytest.raises(ValueError, match="increasing frame_index"):
        publish_detections_stage(
            store,
            [
                CanonicalDetectionFrame(1, 0.1, ()),
                CanonicalDetectionFrame(0, 0.2, ()),
            ],
        )

    assert store.failed
    assert store.failed[-1][0] == "detections"


def test_build_tracklets_stage_reloads_published_detection_artifact():
    store = MemoryStore()
    publish_detections_stage(store, frames())

    summary = build_tracklets_stage(store)

    assert store.stages["tracklets"].artifact == "tracklets/tracklets.jsonl"
    assert summary["tracklet_count"] == 1
    assert summary["observation_count"] == 3
    assert summary["singleton_count"] == 0
    assert len(store.rows["tracklets/tracklets.jsonl"]) == 1


def test_pipeline_runs_stage_boundary_in_order():
    store = MemoryStore()

    result = run_detection_and_tracklet_stages(store, frames())

    assert result["detections"]["detection_count"] == 3
    assert result["tracklets"]["tracklet_count"] == 1
    assert [call for call in store.calls if call[0] in ("begin", "complete")] == [
        ("begin", "detections"),
        ("complete", "detections"),
        ("begin", "tracklets"),
        ("complete", "tracklets"),
    ]
