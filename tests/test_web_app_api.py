from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi import UploadFile

from web_app.backend.app import main
from web_app.backend.app.schemas import CreateMatchRequest, GroundPointRequest, PathImportRequest, RenderRequest
from web_app.backend.app.storage import WorkspaceStore


ROOT = Path(__file__).resolve().parents[1]


def test_local_api_workspace_render_and_landmark_flow(tmp_path, monkeypatch):
    """Exercise the FastAPI endpoint handlers without duplicating geometry."""
    monkeypatch.setattr(main, "RUNTIME_ROOT", tmp_path)
    monkeypatch.setattr(main, "store", WorkspaceStore(tmp_path))
    # Endpoint coverage does not retest the independently validated renderer.
    monkeypatch.setattr(main, "render", lambda _frame, _metadata, width, height: {"panorama": np.zeros((height, width, 3), dtype=np.uint8)})
    main.render_cache.clear()
    assert main.health() == {"status": "ok"}
    calibrations = main.calibrations()
    assert calibrations
    workspace = main.create_match(CreateMatchRequest(name="API sample", calibration_id=calibrations[0]["id"]))
    assert workspace["calibration"]["field_length"] > 0

    with (ROOT / "tests/fixtures/images/stacked_frame.jpg").open("rb") as source:
        uploaded = main.upload_media(
            workspace["id"], UploadFile(filename="uploaded_frame.jpg", file=BytesIO(source.read()))
        )
    assert uploaded["source"] == "upload"
    media = main.import_media_path(
        workspace["id"], PathImportRequest(path=str(ROOT / "tests/fixtures/images/stacked_frame.jpg"))
    )
    rendered = main.render_panorama(RenderRequest(match_id=workspace["id"], media_id=media["id"], width=320, height=180))
    assert main.render_image(rendered["id"]).status_code == 200
    assert rendered["landmarks"][0]["id"] == "center_mark"

    point = main.ground_point(GroundPointRequest(render_id=rendered["id"], x=160, y=90, landmark_id="center_mark"))
    assert point["landmark"] == "center_mark"
    assert point["status"] in {"inside_field", "outside_field", "no_ground_intersection"}
