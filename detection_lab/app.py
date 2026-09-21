"""Detection Lab: a deliberately small, read-only development application."""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from detection_lab.library import MatchLibrary
from web_app.backend.app.storage import WorkspaceStore


STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_DATA_ROOT = Path.home() / ".local" / "share" / "VeoMatchLab"


def workspace_store() -> WorkspaceStore:
    """Use the same authoritative root selection as the main web backend."""
    runtime_root = Path(os.environ.get("VEO_MATCH_DATA_ROOT", str(DEFAULT_DATA_ROOT))).expanduser().resolve()
    return WorkspaceStore(runtime_root)


def create_app(store: WorkspaceStore | None = None) -> FastAPI:
    app = FastAPI(title="Detection Lab", version="0.1.0")
    app.state.library = MatchLibrary(store or workspace_store())

    def library() -> MatchLibrary:
        return app.state.library

    @app.get("/", response_class=HTMLResponse)
    def match_list_page():
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8").replace("{{MATCH_ID}}", ""))

    @app.get("/partidos/{match_id}", response_class=HTMLResponse)
    def preview_page(match_id: str):
        if library().detail(match_id) is None:
            raise HTTPException(404, "Partido no encontrado.")
        return HTMLResponse((STATIC_DIR / "index.html").read_text(encoding="utf-8").replace("{{MATCH_ID}}", match_id))

    @app.get("/static/{asset_name}")
    def static_asset(asset_name: str):
        if asset_name not in {"app.js", "app.css"}:
            raise HTTPException(404, "Recurso no encontrado.")
        return FileResponse(STATIC_DIR / asset_name)

    @app.get("/api/matches")
    def matches():
        return library().matches()

    @app.get("/api/matches/{match_id}")
    def match_detail(match_id: str):
        detail = library().detail(match_id)
        if detail is None:
            raise HTTPException(404, "Partido no encontrado.")
        return detail

    @app.get("/api/matches/{match_id}/dual-lens")
    def dual_lens_video(match_id: str):
        path = library().dual_lens_path(match_id)
        if path is None:
            raise HTTPException(404, "El vídeo de doble lente no está disponible.")
        # FileResponse supports normal browser range requests, so the video element
        # can seek without copying or re-encoding the imported source file.
        return FileResponse(path, media_type="video/mp4", filename=path.name, content_disposition_type="inline")

    return app


app = create_app()
