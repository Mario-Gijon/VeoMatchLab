from pathlib import Path

from fastapi.responses import FileResponse

from detection_lab.app import create_app
from detection_lab.library import MatchLibrary
from web_app.backend.app.storage import WorkspaceStore


def make_match(store: WorkspaceStore, root: Path, *, with_optional_assets: bool = True) -> dict:
    match = store.create_match("Partido de prueba", "", None)
    media_dir = store.match_dir(match["id"]) / "media"
    calibration_dir = store.match_dir(match["id"]) / "calibration"
    media_dir.mkdir()
    calibration_dir.mkdir()
    video = media_dir / "dual_lens.mp4"
    video.write_bytes(b"development test video")
    calibration = calibration_dir / "camera_alignment.veo"
    camera_directions = media_dir / "detections_constrained.det"
    if with_optional_assets:
        calibration.write_text("{}", encoding="utf-8")
        camera_directions.write_bytes(b"det")
    store.update_match(match["id"], lambda item: item.update(
        calibration={"path": str(root / "stale/camera_alignment.veo"), "ownership": "managed"},
        media=[
            {"id": "media_dual", "role": "dual_lens", "path": str(root / "stale/dual_lens.mp4"), "ownership": "managed"},
            {"id": "media_det", "role": "camera_directions", "path": str(root / "stale/detections_constrained.det"), "ownership": "managed"},
        ],
        structured_data={"schema_version": "1"},
    ))
    return store.match(match["id"])


def test_detection_lab_discovers_matches_via_workspace_store(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = make_match(store, tmp_path)

    assert MatchLibrary(store).matches() == [{"id": match["id"], "title": "Partido de prueba"}]


def test_preview_exposes_existing_dual_lens_media_and_assets(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = make_match(store, tmp_path)
    app = create_app(store)
    library = MatchLibrary(store)

    detail = library.detail(match["id"])
    assert detail["video_url"] == f"/api/matches/{match['id']}/dual-lens"
    assert all(asset["available"] for asset in detail["assets"])
    assert library.dual_lens_path(match["id"]).read_bytes() == b"development test video"
    preview_route = next(route for route in app.routes if route.path == "/partidos/{match_id}")
    video_route = next(route for route in app.routes if route.path == "/api/matches/{match_id}/dual-lens")
    assert match["id"].encode() in preview_route.endpoint(match["id"]).body
    response = video_route.endpoint(match["id"])
    assert isinstance(response, FileResponse)
    assert Path(response.path) == library.dual_lens_path(match["id"])


def test_missing_optional_assets_are_reported_without_crashing(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = make_match(store, tmp_path, with_optional_assets=False)

    detail = MatchLibrary(store).detail(match["id"])
    availability = {asset["label"]: asset["available"] for asset in detail["assets"]}
    assert availability["Vídeo de doble lente"] is True
    assert availability["Calibración (.veo)"] is False
    assert availability["Direcciones de cámara (.det)"] is False
