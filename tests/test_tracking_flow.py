from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np
import pytest

from web_app.backend.app.importers.veo.classifier import classify_resource
from web_app.backend.app.importers.veo.models import importable
from web_app.backend.app.main import with_camera_directions
from web_app.backend.app.storage import WorkspaceStore
from web_app.backend.app import main
from web_app.backend.app.schemas import TrackingPointRequest
from geometry.tracking_ground_v3 import TRACKING_FPS, DEFAULT_TRACKING_TUNING, tracking_ground_for_pixel


def test_constrained_camera_directions_is_importable_asset():
    candidate = classify_resource("det", "https://cdn.example/detections_constrained.det")
    assert candidate.role == "camera_directions"
    assert importable(candidate)


def test_camera_directions_role_persists_in_match_store(tmp_path: Path):
    store = WorkspaceStore(tmp_path)
    match = store.create_match("Test", "", None)
    path = tmp_path / "detections_constrained.det"
    path.write_text("0,0,1,0\n", encoding="utf-8")
    store.attach_media(match["id"], {"id": "media_det", "name": path.name, "path": str(path), "role": "camera_directions", "kind": "resource"})
    saved = store.match(match["id"])
    assert saved and saved["media"][0]["role"] == "camera_directions"


def test_missing_camera_directions_is_explicit():
    with pytest.raises(Exception) as error:
        with_camera_directions({"media": []})
    assert "camera directions missing" in str(error.value).lower()


@pytest.fixture
def tracking_fixture(tmp_path, monkeypatch):
    store = WorkspaceStore(tmp_path / 'workspace')
    match = store.create_match('Tracking geometry', '', None)
    path = tmp_path / 'detections_constrained.det'
    path.write_text('0,0.7853981633974483,1,0\n0.3,0.7,1.1,0\n0,-0.5,1,0\n')
    store.attach_media(match['id'], {'id': 'directions', 'path': str(path), 'name': path.name,
                                    'role': 'camera_directions', 'kind': 'resource'})
    matrix = np.eye(4); matrix[1, 3] = 10
    metadata = SimpleNamespace(camera2world=matrix, field_length=105, field_width=68,
        raw_json={'calibration_version': 6.2, 'alignment': {'camera2logic': ','.join(map(str, np.eye(4).flatten()))}})
    monkeypatch.setattr(main, 'store', store)
    monkeypatch.setattr(main, 'with_calibration', lambda _match: metadata)
    return store.match(match['id']), metadata


@pytest.mark.parametrize('timestamp,row', [(0, 0), (1 / TRACKING_FPS, 0), (2 / TRACKING_FPS, 1)])
def test_tracking_endpoint_uses_imported_resource_exact_timestamp_and_zero_tuning(tracking_fixture, timestamp, row):
    match, metadata = tracking_fixture
    payload = TrackingPointRequest(timestamp_s=timestamp, x=960, y=540, width=1920, height=1080)
    result = main.tracking_point(match['id'], payload)
    expected = tracking_ground_for_pixel(metadata, with_camera_directions(match), timestamp, 960, 540, 1920, 1080,
                                         tuning=DEFAULT_TRACKING_TUNING)
    assert result['row_index'] == row
    assert result['timestamp_s'] == timestamp
    assert result['measured_x'] == expected.world_x
    assert result['measured_z'] == expected.world_z
    assert result['status'] == 'inside_field'
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize('timestamp,status', [(3 / TRACKING_FPS, 'no_ground_intersection'), (100, 'camera_state_unavailable')])
def test_invalid_tracking_projection_is_json_safe_with_no_point(tracking_fixture, timestamp, status):
    match, _ = tracking_fixture
    result = main.tracking_point(match['id'], TrackingPointRequest(timestamp_s=timestamp, x=960, y=540, width=1920, height=1080))
    assert result['status'] == status
    assert result['world_x'] is None and result['measured_z'] is None
    json.dumps(result, allow_nan=False)


def test_tracking_rejects_nonfinite_timestamp():
    for timestamp in [float('nan'), float('inf')]:
        with pytest.raises(ValueError):
            TrackingPointRequest(timestamp_s=timestamp, x=960, y=540, width=1920, height=1080)
