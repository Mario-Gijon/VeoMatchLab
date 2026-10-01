from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import tracking.clip_context as clip_context
from tracking.clip_context import resolve_tracking_clip_context


MATCH_ID = "match_context_test"
CLIP_ID = "clip_context_test"


def _materialized_clip(tmp_path: Path) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "data"
    match_dir = data_root / "matches" / MATCH_ID
    clip_dir = match_dir / "clips" / CLIP_ID
    clip_dir.mkdir(parents=True)
    calibration = match_dir / "calibration" / "camera_alignment.veo"
    calibration.parent.mkdir()
    calibration.write_text("calibration", encoding="utf-8")
    video = clip_dir / "source.mp4"
    video.write_bytes(b"fixture video")
    (match_dir / "match.json").write_text(
        json.dumps(
            {
                "id": MATCH_ID,
                "calibration": {"path": str(calibration)},
                "media": [{
                    "id": "media_clip",
                    "path": str(video),
                    "fps": 25.0,
                    "frame_count": 250,
                    "width": 1920,
                    "height": 1080,
                    "duration_s": 10.0,
                }],
                "clips": [{"id": CLIP_ID, "clip_media_id": "media_clip", "duration_s": 10.0}],
            }
        ),
        encoding="utf-8",
    )
    return data_root, video, calibration


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setattr(clip_context, "_probe_video", lambda _video: (25.0, 250, 1920, 1080))


def test_resolves_valid_materialized_match_and_clip(tmp_path, probe):
    data_root, video, calibration = _materialized_clip(tmp_path)

    context = resolve_tracking_clip_context(data_root, MATCH_ID, CLIP_ID)

    assert context.data_root == data_root.resolve()
    assert context.match_dir == data_root / "matches" / MATCH_ID
    assert context.clip_dir == data_root / "matches" / MATCH_ID / "clips" / CLIP_ID
    assert context.clip_video == video.resolve()
    assert context.calibration == calibration.resolve()
    assert context.tracking_root == context.clip_dir / "tracking" / "v1"
    assert (context.duration_s, context.fps, context.frame_count) == (10.0, 25.0, 250)
    assert (context.width, context.height) == (1920, 1080)


def test_rejects_missing_clip_video_before_probing(tmp_path, probe):
    data_root, video, _ = _materialized_clip(tmp_path)
    video.unlink()

    with pytest.raises(FileNotFoundError, match="clip video"):
        resolve_tracking_clip_context(data_root, MATCH_ID, CLIP_ID)


def test_rejects_missing_calibration(tmp_path, probe):
    data_root, _, calibration = _materialized_clip(tmp_path)
    calibration.unlink()

    with pytest.raises(FileNotFoundError, match="calibration"):
        resolve_tracking_clip_context(data_root, MATCH_ID, CLIP_ID)


def test_rejects_missing_clip_directory(tmp_path, probe):
    data_root, _, _ = _materialized_clip(tmp_path)

    with pytest.raises(FileNotFoundError, match="clip directory"):
        resolve_tracking_clip_context(data_root, MATCH_ID, "clip_missing")


@pytest.mark.parametrize(
    ("match_id", "clip_id"),
    [("match", CLIP_ID), (MATCH_ID, "clip")],
)
def test_rejects_invalid_identifiers(tmp_path, probe, match_id, clip_id):
    data_root, _, _ = _materialized_clip(tmp_path)

    with pytest.raises(ValueError, match="invalid"):
        resolve_tracking_clip_context(data_root, match_id, clip_id)


def test_rejects_invalid_canonical_metadata(tmp_path, probe):
    data_root, _, _ = _materialized_clip(tmp_path)
    metadata_path = data_root / "matches" / MATCH_ID / "match.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["media"][0]["fps"] = 0
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical fps"):
        resolve_tracking_clip_context(data_root, MATCH_ID, CLIP_ID)


def test_rejects_video_that_cannot_be_opened(tmp_path):
    data_root, _, _ = _materialized_clip(tmp_path)

    with pytest.raises(ValueError, match="could not be opened"):
        resolve_tracking_clip_context(data_root, MATCH_ID, CLIP_ID)


def test_rejects_symlinked_video_outside_the_managed_clip(tmp_path, probe):
    data_root, video, _ = _materialized_clip(tmp_path)
    external = tmp_path / "outside.mp4"
    external.write_bytes(b"external")
    video.unlink()
    try:
        os.symlink(external, video)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in this environment: {exc}")

    with pytest.raises(ValueError, match="symlink"):
        resolve_tracking_clip_context(data_root, MATCH_ID, CLIP_ID)


def test_runner_explicit_ids_do_not_read_legacy_dev_config(tmp_path, probe, monkeypatch):
    data_root, _, _ = _materialized_clip(tmp_path)
    runner = importlib.import_module("experiments.tracking.run_v1_pipeline")
    monkeypatch.setattr(runner, "load_dev_config", lambda _path: pytest.fail("legacy config was read"))
    args = SimpleNamespace(match_id=MATCH_ID, clip_id=CLIP_ID, data_root=data_root, dev_config=tmp_path / "missing.json")

    context, input_mode = runner.resolve_runner_clip_context(args)

    assert context.match_id == MATCH_ID
    assert context.clip_id == CLIP_ID
    assert input_mode == "explicit"


def test_runner_legacy_dev_config_still_resolves_clip_context(tmp_path, probe):
    data_root, video, calibration = _materialized_clip(tmp_path)
    config = tmp_path / "legacy.json"
    config.write_text(
        json.dumps({
            "match_id": MATCH_ID,
            "clip_id": CLIP_ID,
            "clip_video": str(video),
            "calibration": str(calibration),
            "duration_s": 10.0,
        }),
        encoding="utf-8",
    )
    runner = importlib.import_module("experiments.tracking.run_v1_pipeline")
    args = SimpleNamespace(match_id=None, clip_id=None, data_root=data_root, dev_config=config)

    context, input_mode = runner.resolve_runner_clip_context(args)

    assert context.clip_video == video.resolve()
    assert input_mode.startswith("legacy dev-config")
