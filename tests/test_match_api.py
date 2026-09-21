"""HTTP contract checks with offline fixtures (no Veo calls)."""
from pathlib import Path
from io import BytesIO

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from web_app.backend.app import main
from web_app.backend.app.storage import WorkspaceStore
from web_app.backend.app.importers.veo.service import VeoImportService
from web_app.backend.app.importers.veo.models import VeoInspection
from web_app.backend.app.importers.veo.classifier import classify_resource

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "store", WorkspaceStore(tmp_path))
    monkeypatch.setattr(main, "veo_imports", VeoImportService(main.store, tmp_path))
    with TestClient(main.app) as value:
        yield value


def test_http_manual_match_and_view_geometry(client):
    assert client.get('/api/health').status_code == 200
    calibrations = client.get('/api/calibrations').json()
    match = client.post('/api/matches', json={"name": "HTTP fixture", "calibration_id": calibrations[0]['id']}).json()
    with (ROOT / 'tests/fixtures/images/stacked_frame.jpg').open('rb') as source:
        uploaded = client.post(f"/api/matches/{match['id']}/media/upload", files={'file': ('frame.jpg', source, 'image/jpeg')})
    assert uploaded.status_code == 201
    media = uploaded.json()
    assert client.patch(f"/api/media/{media['id']}/role", json={"role": "tracking"}).status_code == 422
    assert client.get(media['url'], headers={'Range': 'bytes=0-19'}).status_code == 206
    context = client.get(f"/api/matches/{match['id']}/view-context").json()
    assert context['tracking_geometry']['available'] is False
    assert len(context['landmarks']) == 15
    point = client.post('/api/view-point', json={"context_id": context['id'], "view": "panorama", "x": 800, "y": 600, "landmark_id": "center_mark"})
    assert point.status_code == 200
    assert point.json()['expected_x'] == 0
    assert client.post('/api/view-point', json={"context_id": context['id'], "view": "tracking", "x": 800, "y": 600}).status_code == 422
    assert client.post('/api/view-point', json={"context_id": context['id'], "view": "dual_lens", "width": 2048, "height": 2048, "x": 1024, "y": 700}).status_code == 200
    response = client.post('/api/panorama/render', json={"match_id": match['id'], "media_id": media['id'], "width": 320, "height": 180})
    assert response.status_code == 201
    assert client.get(response.json()['url']).status_code == 200


def test_http_invalid_calibration_and_upload(client):
    match = client.post('/api/matches', json={"name": "Errors"}).json()
    bad = client.post(f"/api/matches/{match['id']}/calibration/upload", files={'file': ('bad.veo', BytesIO(b'{}'), 'application/json')})
    assert bad.status_code == 422
    assert client.get(f"/api/matches/{match['id']}/view-context").status_code == 409
    bad_media = client.post(f"/api/matches/{match['id']}/media/upload", files={'file': ('invalid.mp4', b'not video', 'video/mp4')})
    assert bad_media.status_code == 422


def test_http_veo_mocked_inspection_and_job(client, monkeypatch):
    candidate = classify_resource('asset', 'https://cdn.example/camera_alignment.veo?secret=hidden', 'application/json')
    candidate.field_length, candidate.field_width = 105, 68
    tracking = classify_resource('tracking', 'https://cdn.example/video.mp4', discovery_phases=['initial'])
    dual = classify_resource('dual', 'https://cdn.example/transcode-id.mp4', discovery_phases=['after_interactive'])
    inspected = VeoInspection('inspection', 'https://app.veo.co/x', 'https://app.veo.co/x', candidates=[candidate, tracking, dual])
    monkeypatch.setattr('web_app.backend.app.importers.veo.service.inspect_match', lambda *args: inspected)
    response = client.post('/api/veo/inspect', json={'match_url': 'https://app.veo.co/x'})
    assert response.status_code == 200 and 'hidden' not in response.text
    assert client.post('/api/veo/import', json={'inspection_id': 'inspection', 'selected_assets': {}}).status_code == 422
    # Keep the background worker deterministic while exercising job HTTP serialization.
    monkeypatch.setattr(main.veo_imports, '_run_import', lambda job: None)
    assert client.post('/api/veo/import', json={'inspection_id': 'inspection', 'selected_assets': {'calibration': 'asset'}}).status_code == 422
    job = client.post('/api/veo/import', json={'inspection_id': 'inspection', 'selected_assets': {'calibration': 'asset', 'ball_tracking': 'tracking', 'panorama': 'dual'}})
    assert job.status_code == 202
    identifier = job.json()['identifier']
    assert client.get(f'/api/veo/import/{identifier}').json()['status'] == 'queued'
    assert client.delete(f'/api/veo/import/{identifier}').status_code == 200


def test_http_video_frame_extraction(client, tmp_path):
    path = tmp_path / 'fixture.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 32))
    for index in range(10):
        writer.write(np.full((32, 64, 3), index * 20, np.uint8))
    writer.release()
    match = client.post('/api/matches', json={'name': 'Video'}).json()
    media = client.post(f"/api/matches/{match['id']}/media/path", json={'path': str(path), 'role': 'tracking'}).json()
    frame = client.post('/api/media/frame', json={'media_id': media['id'], 'timestamp_s': .4})
    assert frame.status_code == 201
    assert frame.json()['kind'] == 'frame'
    assert frame.json()['timestamp_s'] == pytest.approx(.4)
