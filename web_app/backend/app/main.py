"""FastAPI service for the local Veo Match Lab.

Python owns validated geometry and supplies the captured shader's uniforms.
Interactive picking also uses these uniforms locally, sharing its renderer's ray.
"""

from __future__ import annotations

import sys
import os
import math
import uuid
from pathlib import Path
from typing import Annotated, Any
from functools import wraps

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from .importers.veo.service import VeoImportService
from .schemas import CreateClipRequest, CreateMatchRequest, ExtractFrameRequest, GroundPointRequest, PathImportRequest, RenderRequest, StartClipAnalysisRequest, VeoImportRequest, VeoInspectRequest
from .services import (
    ROOT,
    extract_frame,
    load_stacked_frame,
    media_record,
    repository_calibrations,
    save_upload,
    validate_calibration_path,
    validate_media_role,
)
from .storage import WorkspaceStore
from .clip_jobs import ClipAnalysisService
from .data_api import data_router
from .schemas import MediaRoleRequest, ViewPointRequest, RenameMatchRequest, TrackingPointRequest
from .view_geometry import view_context, contexts
from geometry.raw_ground import raw_pixel_to_ground

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.panorama_ground import ground_intersection_for_pixel
from geometry.panorama_landmarks import field_landmarks, landmark_validation
from geometry.reproduce_veo_panorama import render
from geometry.veo_metadata import load_veo_metadata
from geometry.tracking_ground_v3 import DEFAULT_TRACKING_TUNING, LegacyCameraDirections, tracking_ground_for_pixel


RUNTIME_ROOT = Path(os.environ.get("VEO_MATCH_DATA_ROOT", str(Path.home() / ".local" / "share" / "VeoMatchLab"))).expanduser().resolve()
store = WorkspaceStore(RUNTIME_ROOT)
render_cache: dict[str, dict[str, Any]] = {}
tracking_cache: dict[tuple[str, int], LegacyCameraDirections] = {}
veo_imports = VeoImportService(store, RUNTIME_ROOT)
clip_jobs = ClipAnalysisService(store)

app = FastAPI(title="Veo Match Lab", version="2.0.0")
app.include_router(data_router(lambda: store))
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("VEO_FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(RequestValidationError)
async def invalid_request(_request, exc):
    # Do not echo submitted signed URLs or other input values in validation errors.
    fields = ", ".join(".".join(str(part) for part in error["loc"]) for error in exc.errors())
    return JSONResponse(status_code=422, content={"detail": {"message": f"Invalid request fields: {fields}", "code": "invalid_request"}})


def api_error(status: int, detail: str, code: str) -> HTTPException:
    return HTTPException(status_code=status, detail={"message": detail, "code": code})


def require_match(match_id: str) -> dict[str, Any]:
    match = store.match(match_id)
    if match is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return match


def require_media(media_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    result = store.media(media_id)
    if result is None:
        raise api_error(404, "Media item not found.", "media_not_found")
    return result


def with_calibration(match: dict[str, Any]):
    calibration = match.get("calibration")
    if not calibration:
        raise api_error(409, "Select a calibration before rendering.", "calibration_required")
    try:
        return load_veo_metadata(calibration["path"])
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise api_error(422, "Could not read the workspace calibration; select a valid .veo file.", "invalid_calibration") from exc


def with_camera_directions(match: dict[str, Any]) -> LegacyCameraDirections:
    resource = next((item for item in match.get("media", []) if item.get("role") == "camera_directions"), None)
    if not resource:
        raise api_error(409, "Tracking field projection unavailable: camera directions missing.", "camera_directions_missing")
    path = Path(resource["path"])
    try:
        key = (str(path.resolve()), path.stat().st_mtime_ns)
        if key not in tracking_cache:
            tracking_cache[key] = LegacyCameraDirections.load(path)
        return tracking_cache[key]
    except (OSError, ValueError) as exc:
        raise api_error(422, "Could not read the match camera directions resource.", "invalid_camera_directions") from exc


def match_summary(match: dict[str, Any]) -> dict[str, Any]:
    status = match.get("import_status")
    if status in {"queued", "downloading"} and not veo_imports.active_for_match(match["id"]):
        status = "interrupted"
    return {**{key: value for key, value in match.items() if key != "structured_data"},
            "media_count": len(match["media"]), "import_status": status}


def public_media(media: dict[str, Any]) -> dict[str, Any]:
    return {**media, "url": f"/api/media/{media['id']}/content"}


def public_clip(match_id: str, clip: dict[str, Any]) -> dict[str, Any]:
    result = dict(clip)
    analysis = result.get("analysis")
    if analysis and analysis.get("debug_video"):
        result["debug_video_url"] = f"/api/matches/{match_id}/clips/{clip['id']}/debug-video"
    return result


def active_clip_jobs() -> ClipAnalysisService:
    """Keep test/runtime store replacement from retaining a stale job service."""
    global clip_jobs
    if clip_jobs.store is not store:
        clip_jobs = ClipAnalysisService(store)
    return clip_jobs


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/calibrations")
def calibrations() -> list[dict[str, Any]]:
    return repository_calibrations()


@app.post("/api/veo/inspect")
def inspect_veo(payload: VeoInspectRequest) -> dict[str, Any]:
    try:
        return veo_imports.inspect(payload.match_url).public_dict()
    except ValueError as exc:
        raise api_error(422, str(exc), "veo_inspection_failed") from exc
    except Exception as exc:
        raise api_error(422, "Inspection failed. Check Playwright/Chromium installation and verify access to this match.", "veo_inspection_failed") from exc


@app.post("/api/veo/import", status_code=202)
def import_veo(payload: VeoImportRequest) -> dict[str, Any]:
    try:
        return veo_imports.start_import(payload.inspection_id, payload.selected_assets, payload.match_id).public_dict()
    except ValueError as exc:
        raise api_error(422, str(exc), "veo_import_invalid") from exc


@app.get("/api/veo/import/{job_id}")
def veo_import_status(job_id: str) -> dict[str, Any]:
    job = veo_imports.job(job_id)
    if job is None:
        raise api_error(404, "Import job not found.", "veo_import_not_found")
    return job.public_dict()


@app.delete("/api/veo/import/{job_id}")
def cancel_veo_import(job_id: str) -> dict[str, Any]:
    job = veo_imports.cancel(job_id)
    if job is None:
        raise api_error(404, "Import job not found.", "veo_import_not_found")
    return job.public_dict()


@app.get("/api/matches")
def list_matches() -> list[dict[str, Any]]:
    return [match_summary(match) for match in store.matches()]


@app.post("/api/matches", status_code=201)
def create_match(payload: CreateMatchRequest) -> dict[str, Any]:
    if not payload.name.strip():
        raise api_error(422, "Enter a match name.", "invalid_name")
    calibration = None
    if payload.calibration_id:
        try:
            calibration = validate_calibration_path(payload.calibration_id, "repository")
        except (OSError, ValueError) as exc:
            raise api_error(422, str(exc), "invalid_calibration") from exc
    return match_summary(store.create_match(payload.name.strip(), payload.description.strip(), calibration))


def require_editable(match_id: str) -> dict:
    match = require_match(match_id)
    if veo_imports.active_for_match(match_id):
        raise api_error(409, "Cancel or finish the import before changing match files.", "import_active")
    return match


def match_file_edit(function):
    """Serialize file changes against import creation and match deletion."""
    @wraps(function)
    def guarded(*args, **kwargs):
        with veo_imports._lock:
            return function(*args, **kwargs)
    return guarded


@app.patch("/api/matches/{match_id}")
def rename_match(match_id: str, payload: RenameMatchRequest) -> dict:
    require_match(match_id)
    if not payload.name.strip():
        raise api_error(422, "Enter a match name.", "invalid_name")
    return match_summary(store.update_match(match_id, lambda item: item.update(name=payload.name.strip())))


@app.delete("/api/matches/{match_id}")
def delete_match(match_id: str) -> dict:
    with veo_imports._lock:
        require_editable(match_id)
        store.delete_match(match_id)
        for identifier in [key for key, value in render_cache.items() if value["match_id"] == match_id]:
            render_cache.pop(identifier, None)
    return {"deleted": True}


@app.delete("/api/media/{media_id}")
def remove_media(media_id: str) -> dict:
    match, _ = require_media(media_id)
    with veo_imports._lock:
        require_editable(match["id"])
        store.remove_media(match["id"], media_id)
    return {"removed": True}


@app.delete("/api/matches/{match_id}/calibration")
def remove_calibration(match_id: str) -> dict:
    with veo_imports._lock:
        require_editable(match_id)
        return match_summary(store.set_calibration(match_id, None))


@app.get("/api/matches/{match_id}")
def get_match(match_id: str) -> dict[str, Any]:
    match = require_match(match_id)
    return {**match_summary(match), "media": [public_media(item) for item in match["media"]],
            "clips": [public_clip(match_id, item) for item in match.get("clips", [])]}


@app.get("/api/matches/{match_id}/clips")
def list_clips(match_id: str) -> list[dict[str, Any]]:
    require_match(match_id)
    return [public_clip(match_id, item) for item in store.clips(match_id)]


@app.post("/api/matches/{match_id}/clips", status_code=201)
def create_clip(match_id: str, payload: CreateClipRequest) -> dict[str, Any]:
    match = require_match(match_id)
    source = next((item for item in match["media"] if item["id"] == payload.source_media_id), None)
    if source is None or source.get("role") != "dual_lens" or source.get("kind") != "video":
        raise api_error(422, "Select the match dual_lens video as clip source.", "clip_source_invalid")
    if payload.end_s <= payload.start_s:
        raise api_error(422, "Clip end must be after its start.", "clip_interval_invalid")
    if source.get("duration_s") is not None and payload.end_s > float(source["duration_s"]):
        raise api_error(422, "Clip end lies outside the source video.", "clip_interval_invalid")
    clip = store.create_clip(match_id, source["id"], payload.start_s, payload.end_s, payload.name.strip())
    if clip is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return public_clip(match_id, clip)


@app.post("/api/matches/{match_id}/clips/{clip_id}/analysis", status_code=202)
def start_clip_analysis(match_id: str, clip_id: str, payload: StartClipAnalysisRequest) -> dict[str, Any]:
    match = require_match(match_id)
    clip = store.clip(match_id, clip_id)
    if clip is None:
        raise api_error(404, "Clip not found.", "clip_not_found")
    source = next((item for item in match["media"] if item["id"] == clip["source_media_id"]), None)
    if source is None or not match.get("calibration"):
        raise api_error(409, "Clip analysis requires dual_lens media and calibration.", "clip_analysis_unavailable")
    job = active_clip_jobs().start(match, clip, source, action=payload.action)
    return job.public_dict()


@app.get("/api/clip-jobs/{job_id}")
def clip_job_status(job_id: str) -> dict[str, Any]:
    job = active_clip_jobs().job(job_id)
    if job is None:
        raise api_error(404, "Clip analysis job not found.", "clip_job_not_found")
    return job.public_dict()


@app.delete("/api/clip-jobs/{job_id}")
def cancel_clip_job(job_id: str) -> dict[str, Any]:
    job = active_clip_jobs().cancel(job_id)
    if job is None:
        raise api_error(404, "Clip analysis job not found.", "clip_job_not_found")
    return job.public_dict()


@app.get("/api/matches/{match_id}/clips/{clip_id}/debug-video")
def clip_debug_video(match_id: str, clip_id: str):
    clip = store.clip(match_id, clip_id)
    analysis = clip.get("analysis") if clip else None
    relative = analysis.get("debug_video") if analysis else None
    if not relative:
        raise api_error(404, "No debug video is available for this clip.", "clip_debug_missing")
    root = store.match_dir(match_id).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise api_error(404, "Debug video is no longer available.", "clip_debug_missing")
    return FileResponse(path, media_type="video/mp4", filename="tracking_debug.mp4", content_disposition_type="inline")


@app.get("/api/matches/{match_id}/view-context")
def get_view_context(match_id: str) -> dict:
    match = require_match(match_id)
    with_calibration(match)
    return view_context(match["calibration"]["path"], any(item.get("role") == "camera_directions" for item in match.get("media", [])))


@app.get("/api/matches/{match_id}/import-job")
def match_import_job(match_id: str) -> dict | None:
    require_match(match_id)
    job = veo_imports.job_for_match(match_id)
    return job.public_dict() if job else None


@app.patch("/api/media/{media_id}/role")
@match_file_edit
def set_media_role(media_id: str, payload: MediaRoleRequest) -> dict:
    match, media = require_media(media_id)
    require_editable(match["id"])
    try:
        validate_media_role(media, payload.role)
    except ValueError as exc:
        raise api_error(422, str(exc), "invalid_role") from exc
    store.attach_media(match["id"], {**media, "role": payload.role})
    return public_media({**media, "role": payload.role})


@app.post("/api/view-point")
def view_point(payload: ViewPointRequest) -> dict:
    metadata = contexts.get(payload.context_id)
    if metadata is None:
        raise api_error(409, "Reload the match geometry context.", "context_expired")
    try:
        if payload.view == "dual_lens":
            if (payload.width, payload.height) != (2048, 2048):
                raise ValueError("RAW view coordinates must use the stacked 2048×2048 source")
            point = raw_pixel_to_ground(metadata, payload.x, payload.y)
        else:
            point = ground_intersection_for_pixel(metadata, payload.x, payload.y, payload.width, payload.height)
        response = point.to_dict()
        if payload.landmark_id:
            landmark = field_landmarks(metadata.field_length, metadata.field_width).get(payload.landmark_id)
            if landmark is None:
                raise ValueError("Unknown landmark")
            response.update(landmark_validation(landmark, point))
        else:
            response.update(landmark=None, landmark_name=None, expected_x=None, expected_z=None,
                            measured_x=point.world_x, measured_z=point.world_z, delta_x=None, delta_z=None, field_error_m=None)
        return response
    except ValueError as exc:
        raise api_error(422, str(exc), "invalid_view_point") from exc


@app.post("/api/matches/{match_id}/tracking-point")
def tracking_point(match_id: str, payload: TrackingPointRequest) -> dict[str, Any]:
    match = require_match(match_id)
    metadata = with_calibration(match)
    directions = with_camera_directions(match)
    try:
        if payload.x >= payload.width or payload.y >= payload.height:
            raise ValueError("Tracking pixel lies outside the native video.")
        point = tracking_ground_for_pixel(metadata, directions, payload.timestamp_s, payload.x, payload.y,
                                          payload.width, payload.height, tuning=DEFAULT_TRACKING_TUNING)
        # Missing .det samples legitimately contain NaN diagnostics in the
        # authoritative model. Keep the API JSON-safe without changing that model.
        result = {key: None if isinstance(value, float) and not math.isfinite(value) else value
                  for key, value in point.to_dict().items()}
        result.update({"pixel_x": payload.x, "pixel_y": payload.y,
                       "native_source_pixel": {"x": payload.x, "y": payload.y, "width": payload.width, "height": payload.height},
                       "timestamp_s": payload.timestamp_s, "distance": point.ground_distance_m,
                       "measured_x": point.world_x, "measured_z": point.world_z, "expected_x": None, "expected_z": None,
                       "landmark": None, "landmark_name": None, "delta_x": None, "delta_z": None, "field_error_m": None,
                       "view": "tracking"})
        return result
    except ValueError as exc:
        raise api_error(422, str(exc), "invalid_tracking_point") from exc


@app.post("/api/matches/{match_id}/calibration/path")
@match_file_edit
def import_calibration_path(match_id: str, payload: PathImportRequest) -> dict[str, Any]:
    require_editable(match_id)
    try:
        calibration = validate_calibration_path(payload.path)
    except (OSError, ValueError) as exc:
        raise api_error(422, str(exc), "invalid_calibration") from exc
    match = store.set_calibration(match_id, calibration)
    if match is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return match_summary(match)


@app.post("/api/matches/{match_id}/calibration/upload")
@match_file_edit
def upload_calibration(match_id: str, file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    require_editable(match_id)
    if not file.filename or Path(file.filename).suffix.lower() != ".veo":
        raise api_error(422, "Upload a .veo calibration file.", "invalid_calibration")
    destination = store.asset_dir(match_id, "calibration") / f"calibration_{uuid.uuid4().hex[:8]}.veo"
    save_upload(file, destination)
    try:
        calibration = validate_calibration_path(destination, "upload")
    except (OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise api_error(422, f"Could not read the uploaded calibration: {exc}", "invalid_calibration") from exc
    match = store.set_calibration(match_id, calibration)
    return match_summary(match) if match else require_match(match_id)


def add_media(match_id: str, record: dict[str, Any]) -> dict[str, Any]:
    record["id"] = store.new_id("media")
    match = store.attach_media(match_id, record)
    if match is None:
        raise api_error(404, "Workspace not found.", "match_not_found")
    return public_media(record)


@app.post("/api/matches/{match_id}/media/path", status_code=201)
@match_file_edit
def import_media_path(match_id: str, payload: PathImportRequest) -> dict[str, Any]:
    require_editable(match_id)
    try:
        record = media_record(Path(payload.path), source="path", kind="resource" if payload.role == "camera_directions" else None)
        validate_media_role(record, payload.role)
    except (OSError, ValueError) as exc:
        raise api_error(422, str(exc), "invalid_media") from exc
    if payload.role == "camera_directions":
        record = {**record, "kind": "resource", "width": 0, "height": 0, "fps": None, "duration_s": None, "name": Path(payload.path).name}
    record["role"] = payload.role if payload.role != "unknown" else record["role"]
    return add_media(match_id, record)


@app.post("/api/matches/{match_id}/media/upload", status_code=201)
@match_file_edit
def upload_media(match_id: str, file: Annotated[UploadFile, File(...)], role: Annotated[str, Form()] = "unknown") -> dict[str, Any]:
    require_editable(match_id)
    if not file.filename:
        raise api_error(422, "Choose a media file to upload.", "invalid_media")
    suffix = Path(file.filename).suffix.lower()
    destination = store.asset_dir(match_id, "media") / f"{uuid.uuid4().hex[:8]}{suffix}"
    save_upload(file, destination)
    try:
        record = media_record(destination, name=Path(file.filename).name, source="upload", kind="resource" if role == "camera_directions" else None)
        if role not in {"tracking", "dual_lens", "camera_directions", "veo_highlights", "clip", "image", "unknown"}:
            raise ValueError("Unknown media role")
        if role == "camera_directions":
            if suffix != ".det":
                raise ValueError("Camera Directions must be a .det resource.")
            record.update(kind="resource", width=0, height=0, fps=None, duration_s=None, name=Path(file.filename).name)
        validate_media_role(record, role)
        if role != "unknown":
            record["role"] = role
    except (OSError, ValueError) as exc:
        destination.unlink(missing_ok=True)
        raise api_error(422, str(exc), "invalid_media") from exc
    return add_media(match_id, record)


@app.get("/api/media/{media_id}/content")
def media_content(media_id: str):
    _, media = require_media(media_id)
    path = Path(media["path"])
    if not path.is_file():
        raise api_error(404, "The referenced media file is no longer available.", "media_missing")
    return FileResponse(path, filename=media["name"], content_disposition_type="inline")


@app.post("/api/media/frame", status_code=201)
@match_file_edit
def frame_from_video(payload: ExtractFrameRequest) -> dict[str, Any]:
    match, media = require_media(payload.media_id)
    require_editable(match["id"])
    if media["kind"] not in {"video", "clip"}:
        raise api_error(422, "Frame extraction requires video or clip media.", "not_video")
    destination = store.asset_dir(match["id"], "frames") / f"frame_{uuid.uuid4().hex[:8]}.jpg"
    try:
        details = extract_frame(Path(media["path"]), payload.timestamp_s, destination)
        record = media_record(destination, name=f"{media['name']} @ {details['actual_time_s']:.3f}s", kind="frame", source="extracted")
        record["extracted_from"] = media["id"]
        record["timestamp_s"] = details["actual_time_s"]
    except (OSError, RuntimeError, ValueError) as exc:
        raise api_error(422, str(exc), "frame_extraction_failed") from exc
    return add_media(match["id"], record)


@app.post("/api/panorama/render", status_code=201)
@match_file_edit
def render_panorama(payload: RenderRequest) -> dict[str, Any]:
    match = require_match(payload.match_id)
    media_match, media = require_media(payload.media_id)
    if media_match["id"] != match["id"]:
        raise api_error(422, "Selected media belongs to another workspace.", "media_workspace_mismatch")
    if media["kind"] not in {"image", "frame"}:
        raise api_error(422, "Select a stacked image or extracted frame to render.", "render_requires_image")
    metadata = with_calibration(match)
    try:
        stacked = load_stacked_frame(Path(media["path"]))
        result = render(stacked, metadata, payload.width, payload.height)
        encoded_ok, encoded = cv2.imencode(".png", result["panorama"])
        if not encoded_ok:
            raise RuntimeError("Could not encode the panorama.")
    except (OSError, RuntimeError, ValueError) as exc:
        raise api_error(422, str(exc), "render_failed") from exc
    render_id = f"render_{uuid.uuid4().hex[:12]}"
    image_path = store.asset_dir(match["id"], "cache") / f"{render_id}.png"
    image_path.write_bytes(encoded.tobytes())
    render_cache[render_id] = {
        "path": image_path,
        "metadata": metadata,
        "width": payload.width,
        "height": payload.height,
        "match_id": match["id"],
        "media_id": media["id"],
    }
    return {
        "id": render_id,
        "url": f"/api/renders/{render_id}/image",
        "width": payload.width,
        "height": payload.height,
        "field_length": metadata.field_length,
        "field_width": metadata.field_width,
        "landmarks": [value.to_dict() for value in field_landmarks(metadata.field_length, metadata.field_width).values()],
    }


@app.get("/api/renders/{render_id}/image")
def render_image(render_id: str):
    cached = render_cache.get(render_id)
    if cached is None or not cached["path"].is_file():
        raise api_error(404, "Rendered panorama is not available. Render it again.", "render_not_found")
    return FileResponse(cached["path"], media_type="image/png", filename="veo_panorama.png")


@app.post("/api/ground-point")
def ground_point(payload: GroundPointRequest) -> dict[str, Any]:
    cached = render_cache.get(payload.render_id)
    if cached is None:
        raise api_error(409, "Render a panorama before selecting points.", "render_required")
    if payload.x >= cached["width"] or payload.y >= cached["height"]:
        raise api_error(422, "Panorama pixel lies outside the active render.", "pixel_out_of_range")
    point = ground_intersection_for_pixel(cached["metadata"], payload.x, payload.y, cached["width"], cached["height"])
    response = point.to_dict()
    if payload.landmark_id:
        landmarks = field_landmarks(cached["metadata"].field_length, cached["metadata"].field_width)
        landmark = landmarks.get(payload.landmark_id)
        if landmark is None:
            raise api_error(422, "Unknown field landmark.", "unknown_landmark")
        response.update(landmark_validation(landmark, point))
    else:
        response.update({"landmark": None, "landmark_name": None, "expected_x": None, "expected_z": None,
                         "measured_x": point.world_x, "measured_z": point.world_z, "delta_x": None,
                         "delta_z": None, "field_error_m": None})
    return response
