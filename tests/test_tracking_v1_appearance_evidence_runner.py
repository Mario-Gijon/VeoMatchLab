from __future__ import annotations

import json
from pathlib import Path

from experiments.tracking.run_v1_appearance_evidence import publish_appearance_stage
from tracking.appearance.evidence import AppearanceEvidenceConfig
from tracking.artifacts import TrackingArtifactStore
from tracking.contracts import TrackingStage


def _embedding_row(sample_id, tracklet_id, vector, frame, quality=0.9):
    return {
        "sample_id": sample_id,
        "tracklet_id": tracklet_id,
        "detection_id": f"det-{sample_id}",
        "frame_index": frame,
        "timestamp_s": frame / 10.0,
        "quality": quality,
        "descriptors": [
            {
                "name": "reid",
                "model_id": "test-reid",
                "values": vector,
                "metric": "cosine",
            }
        ],
    }


def _prepare_completed_tracklets(tmp_path: Path) -> TrackingArtifactStore:
    data = tmp_path / "data"
    clip = data / "matches" / "match_test" / "clips" / "clip_test"
    clip.mkdir(parents=True)

    store = TrackingArtifactStore.for_clip(data, "match_test", "clip_test")

    store.begin_stage(TrackingStage.DETECTIONS, version="test")
    detections = store.write_jsonl(
        TrackingStage.DETECTIONS,
        "frames.jsonl",
        [{"frame_index": 0}],
    )
    store.complete_stage(TrackingStage.DETECTIONS, artifact=detections)

    store.begin_stage(TrackingStage.TRACKLETS, version="test")
    tracklets = store.write_jsonl(
        TrackingStage.TRACKLETS,
        "tracklets.jsonl",
        [
            {"tracklet_id": "trk-a"},
            {"tracklet_id": "trk-b"},
            {"tracklet_id": "trk-no-reid"},
        ],
    )
    store.complete_stage(TrackingStage.TRACKLETS, artifact=tracklets)
    return store


def test_runner_publishes_solver_facing_appearance_stage(tmp_path):
    store = _prepare_completed_tracklets(tmp_path)
    source = (
        store.root
        / "diagnostics"
        / "reid_benchmark"
        / "test-reid"
        / "embeddings.jsonl"
    )
    source.parent.mkdir(parents=True)
    rows = [
        _embedding_row("a1", "trk-a", [1.0, 0.0], 0),
        _embedding_row("a2", "trk-a", [0.98, 0.1], 10),
        _embedding_row("b1", "trk-b", [0.1, 1.0], 0),
        _embedding_row("b2", "trk-b", [0.2, 0.98], 10),
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    summary = publish_appearance_stage(
        store=store,
        embeddings_path=source,
        config=AppearanceEvidenceConfig(model_id="test-reid"),
    )

    assert summary["sample_count"] == 4
    assert summary["profile_count"] == 2
    assert summary["pair_count"] == 1
    assert summary["total_tracklet_count"] == 3
    assert summary["unprofiled_tracklet_count"] == 1
    assert summary["profile_coverage"] == 2 / 3

    manifest = store.load_manifest()
    stage = manifest.stage(TrackingStage.APPEARANCE)
    assert stage.status.value == "completed"
    assert stage.artifact == "appearance/index.json"
    assert stage.summary == "appearance/summary.json"

    index = store.read_json(stage.artifact)
    assert index["identity_decisions"] is False
    assert index["profiles"] == "appearance/profiles.jsonl"
    assert index["pairwise_evidence"] == "appearance/pairwise_evidence.jsonl"

    profiles = list(store.iter_jsonl(index["profiles"]))
    edges = list(store.iter_jsonl(index["pairwise_evidence"]))
    assert len(profiles) == 2
    assert len(edges) == 1
    assert "same_identity" not in edges[0]
