import hashlib
import json
import math
import os
from pathlib import Path

import pytest

from tracking.artifacts import ArtifactFile, TrackingArtifactStore
from tracking.contracts import StageStatus, TrackingManifest, TrackingStage
from tracking.detections import CanonicalDetection, CanonicalDetectionFrame


MATCH_ID = "match_test"
CLIP_ID = "clip_test"


def _clip_dir(data_root: Path) -> Path:
    clip_dir = data_root / "matches" / MATCH_ID / "clips" / CLIP_ID
    clip_dir.mkdir(parents=True)
    return clip_dir


def _store(tmp_path: Path, *, require_clip_exists: bool = True) -> TrackingArtifactStore:
    data_root = tmp_path / "data"
    if require_clip_exists:
        _clip_dir(data_root)
    return TrackingArtifactStore(data_root, MATCH_ID, CLIP_ID, require_clip_exists=require_clip_exists)


def _frame(index: int) -> CanonicalDetectionFrame:
    timestamp_s = index / 15
    detection = CanonicalDetection(
        detection_id=f"detection-{index}",
        frame_index=index,
        timestamp_s=timestamp_s,
        confidence=0.9,
        x=10.0 + index,
        z=20.0,
        source_lens="left",
        source_bbox_xyxy=(1, 2, 3, 4),
        source_footpoint_xy=(2, 4),
        representative_view="far_left",
        virtual_bbox_xyxy=(10, 20, 30, 40),
        contributing_views=("far_left",),
        metadata={"source": "synthetic"},
    )
    return CanonicalDetectionFrame(index, timestamp_s, (detection,))


def _complete_stage(store: TrackingArtifactStore, stage: TrackingStage) -> ArtifactFile:
    store.begin_stage(stage)
    artifact = store.write_json(stage, "artifact.json", {"stage": stage.value})
    store.complete_stage(stage, artifact=artifact)
    return artifact


def test_layout_is_exact_and_requires_an_existing_clip(tmp_path):
    data_root = tmp_path / "data"
    with pytest.raises(FileNotFoundError):
        TrackingArtifactStore(data_root, MATCH_ID, CLIP_ID)

    store = _store(tmp_path)
    assert store.root == data_root / "matches" / MATCH_ID / "clips" / CLIP_ID / "tracking" / "v1"
    assert store.manifest_path == store.root / "manifest.json"


@pytest.mark.parametrize("match_id", ["", "match", "match_test/../other", "../match_test"])
def test_store_rejects_invalid_match_identifiers(tmp_path, match_id):
    with pytest.raises(ValueError):
        TrackingArtifactStore(tmp_path / "data", match_id, CLIP_ID, require_clip_exists=False)


@pytest.mark.parametrize("clip_id", ["", "clip", "clip_test/../other", "../clip_test"])
def test_store_rejects_invalid_clip_identifiers(tmp_path, clip_id):
    with pytest.raises(ValueError):
        TrackingArtifactStore(tmp_path / "data", MATCH_ID, clip_id, require_clip_exists=False)


def test_store_can_be_used_without_a_preexisting_clip_for_isolated_tests(tmp_path):
    store = _store(tmp_path, require_clip_exists=False)

    assert not store.clip_dir.exists()
    store.ensure_manifest()
    assert store.manifest_path.is_file()


def test_store_rejects_managed_path_symlinks(tmp_path):
    data_root = tmp_path / "data"
    clips_dir = data_root / "matches" / MATCH_ID / "clips"
    clips_dir.mkdir(parents=True)
    target = tmp_path / "outside-clip"
    target.mkdir()
    try:
        os.symlink(target, clips_dir / CLIP_ID, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        TrackingArtifactStore(data_root, MATCH_ID, CLIP_ID)


def test_ensure_manifest_is_lazy_and_loadable(tmp_path):
    store = _store(tmp_path)
    manifest = store.ensure_manifest(metadata={"purpose": "test"})

    assert store.manifest_exists()
    assert manifest.match_id == MATCH_ID
    assert manifest.clip_id == CLIP_ID
    assert manifest.schema_version == 1
    assert manifest.metadata["purpose"] == "test"
    assert "created_at" in manifest.metadata
    assert all(manifest.stage(stage).status is StageStatus.NOT_RUN for stage in TrackingStage)
    assert store.load_manifest().to_dict() == manifest.to_dict()
    assert not any((store.root / stage.value).exists() for stage in TrackingStage)


def test_load_manifest_rejects_identifiers_for_another_store(tmp_path):
    store = _store(tmp_path)
    store.ensure_manifest()
    incorrect = TrackingManifest(match_id="match_other", clip_id=CLIP_ID)
    store.manifest_path.write_text(incorrect.to_json(), encoding="utf-8")

    with pytest.raises(ValueError, match="identifiers"):
        store.load_manifest()


def test_begin_stage_enforces_dependencies_and_running_restart_rules(tmp_path):
    store = _store(tmp_path)

    started = store.begin_stage("detections", version="v1", config={"threshold": 0.4})
    assert started.stage("detections").status is StageStatus.RUNNING
    assert started.stage("detections").version == "v1"
    assert started.stage("detections").config == {"threshold": 0.4}
    assert all(started.stage(stage).status is StageStatus.NOT_RUN for stage in tuple(TrackingStage)[1:])
    assert not any((store.root / stage.value).exists() for stage in TrackingStage)

    with pytest.raises(RuntimeError, match="already running"):
        store.begin_stage("detections")
    store.begin_stage("detections", restart=True)

    for stage in tuple(TrackingStage)[1:]:
        with pytest.raises(RuntimeError, match="requires completed"):
            store.begin_stage(stage)
    with pytest.raises(ValueError, match="unknown"):
        store.begin_stage("unknown")


def test_begin_stage_invalidates_only_downstream_stages(tmp_path):
    store = _store(tmp_path)
    for stage in TrackingStage:
        _complete_stage(store, stage)

    manifest = store.begin_stage(TrackingStage.TRACKLETS)
    assert manifest.stage("detections").status is StageStatus.COMPLETED
    assert manifest.stage("tracklets").status is StageStatus.RUNNING
    assert all(manifest.stage(stage).status is StageStatus.NOT_RUN for stage in tuple(TrackingStage)[2:])

    manifest = store.begin_stage(TrackingStage.DETECTIONS, restart=True)
    assert manifest.stage("detections").status is StageStatus.RUNNING
    assert all(manifest.stage(stage).status is StageStatus.NOT_RUN for stage in tuple(TrackingStage)[1:])


def test_write_json_is_atomic_replaces_content_and_checks_running_stage(tmp_path):
    store = _store(tmp_path)
    store.ensure_manifest()
    with pytest.raises(RuntimeError, match="must be running"):
        store.write_json("detections", "summary.json", {"count": 0})

    store.begin_stage("detections")
    first = store.write_json("detections", "summary.json", {"count": 1})
    second = store.write_json("detections", "summary.json", {"count": 2})

    assert isinstance(first, ArtifactFile)
    assert second.ref == "detections/summary.json"
    assert second.path == store.root / second.ref
    assert second.size_bytes == second.path.stat().st_size
    assert second.sha256 == hashlib.sha256(second.path.read_bytes()).hexdigest()
    assert json.loads(second.path.read_text(encoding="utf-8")) == {"count": 2}
    assert not list(second.path.parent.glob(".summary.json.*.tmp"))


@pytest.mark.parametrize("payload", [{"value": math.nan}, {"unsupported": {1}}])
def test_write_json_rejects_non_json_payloads(tmp_path, payload):
    store = _store(tmp_path)
    store.begin_stage("detections")

    with pytest.raises((TypeError, ValueError)):
        store.write_json("detections", "summary.json", payload)
    assert not (store.root / "detections" / "summary.json").exists()


def test_write_jsonl_streams_generator_and_publishes_one_line_per_frame(tmp_path):
    store = _store(tmp_path)
    store.begin_stage("detections")
    consumed: list[int] = []

    def frames():
        for index in range(3):
            consumed.append(index)
            yield _frame(index)

    artifact = store.write_jsonl("detections", "frames.jsonl", frames())
    lines = artifact.path.read_text(encoding="utf-8").splitlines()

    assert consumed == [0, 1, 2]
    assert artifact.ref == "detections/frames.jsonl"
    assert artifact.record_count == 3
    assert artifact.size_bytes == artifact.path.stat().st_size
    assert artifact.sha256 == hashlib.sha256(artifact.path.read_bytes()).hexdigest()
    assert [json.loads(line) for line in lines] == [frame.to_dict() for frame in (_frame(0), _frame(1), _frame(2))]
    assert list(store.iter_jsonl(artifact.ref)) == [frame.to_dict() for frame in (_frame(0), _frame(1), _frame(2))]
    assert not list(artifact.path.parent.glob(".frames.jsonl.*.tmp"))


@pytest.mark.parametrize("path", ["../foo.json", "../../etc/passwd", "/absolute/path.json", "..\\foo.json", "\\etc\\passwd", "."])
def test_writes_cannot_escape_the_selected_stage(tmp_path, path):
    store = _store(tmp_path)
    store.begin_stage("detections")

    with pytest.raises(ValueError):
        store.write_json("detections", path, {"safe": True})
    assert not (store.root.parent / "foo.json").exists()


def test_complete_stage_publishes_existing_same_stage_files_and_preserves_begin_data(tmp_path):
    store = _store(tmp_path)
    store.begin_stage("detections", version=7, config={"threshold": 0.4}, metadata={"attempt": 1})
    frames = store.write_json("detections", "frames.jsonl", {"rows": 2})
    summary = store.write_json("detections", "summary.json", {"count": 2})

    manifest = store.complete_stage("detections", artifact=frames, summary=summary, metadata={"count": 2})
    stage = manifest.stage("detections")
    assert stage.status is StageStatus.COMPLETED
    assert stage.artifact == "detections/frames.jsonl"
    assert stage.summary == "detections/summary.json"
    assert stage.version == 7
    assert stage.config == {"threshold": 0.4}
    assert stage.metadata == {"attempt": 1, "count": 2}

    with pytest.raises(RuntimeError, match="only complete from running"):
        store.complete_stage("detections", artifact=frames)


def test_complete_stage_rejects_invalid_or_missing_references(tmp_path):
    store = _store(tmp_path)
    store.ensure_manifest()
    with pytest.raises(RuntimeError, match="only complete from running"):
        store.complete_stage("detections", artifact="missing.json")

    store.begin_stage("detections")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    with pytest.raises((FileNotFoundError, ValueError)):
        store.complete_stage("detections", artifact="missing.json")
    with pytest.raises((FileNotFoundError, ValueError)):
        store.complete_stage("detections", artifact=outside)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Artifact references named after another known stage are prefixed into "
        "the selected stage instead of being rejected explicitly."
    ),
)
def test_complete_stage_rejects_references_for_another_stage(tmp_path):
    store = _store(tmp_path)
    store.begin_stage("detections")

    with pytest.raises(ValueError, match="inside stage"):
        store.complete_stage("detections", artifact="tracklets/other.json")


def test_fail_stage_keeps_partial_files_unpublished_and_requires_running(tmp_path):
    store = _store(tmp_path)
    store.ensure_manifest()
    with pytest.raises(RuntimeError, match="only fail from running"):
        store.fail_stage("detections", RuntimeError("boom"))

    store.begin_stage("detections", metadata={"attempt": 1})
    partial = store.write_json("detections", "partial.json", {"partial": True})
    manifest = store.fail_stage("detections", RuntimeError("boom"), metadata={"retry": False})
    stage = manifest.stage("detections")

    assert partial.path.is_file()
    assert stage.status is StageStatus.FAILED
    assert stage.artifact is None and stage.summary is None
    assert stage.metadata == {"attempt": 1, "retry": False, "error": "boom"}


def test_invalidate_stage_resets_targets_and_can_delete_their_files(tmp_path):
    store = _store(tmp_path)
    artifacts = {stage: _complete_stage(store, stage) for stage in TrackingStage}

    manifest = store.invalidate_stage("tracklets", include_downstream=True, delete_files=True, reason="rerun")
    assert manifest.stage("detections").status is StageStatus.COMPLETED
    assert artifacts[TrackingStage.DETECTIONS].path.is_file()
    for stage in tuple(TrackingStage)[1:]:
        assert manifest.stage(stage).status is StageStatus.NOT_RUN
        assert not (store.root / stage.value).exists()


def test_invalidate_stage_can_leave_downstream_completed(tmp_path):
    store = _store(tmp_path)
    for stage in TrackingStage:
        _complete_stage(store, stage)

    manifest = store.invalidate_stage("identities", include_downstream=False, delete_files=False)
    assert manifest.stage("identities").status is StageStatus.NOT_RUN
    assert manifest.stage("final").status is StageStatus.COMPLETED
    assert (store.root / "identities").is_dir()
