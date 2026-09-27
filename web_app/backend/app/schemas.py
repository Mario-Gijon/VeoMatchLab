"""Pydantic request and response models for the local development API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class CreateMatchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(default="", max_length=500)
    calibration_id: str | None = None


class PathImportRequest(BaseModel):
    path: str = Field(min_length=1)
    role: Literal["tracking", "dual_lens", "camera_directions", "veo_highlights", "clip", "image", "unknown"] = "unknown"


class MediaRoleRequest(BaseModel):
    role: Literal["tracking", "dual_lens", "camera_directions", "veo_highlights", "clip", "image", "unknown"]


class ViewPointRequest(BaseModel):
    context_id: str
    view: Literal["panorama", "dual_lens"]
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(default=1920, gt=0, le=4096)
    height: int = Field(default=1080, gt=0, le=2160)
    landmark_id: str | None = None


class TrackingPointRequest(BaseModel):
    timestamp_s: float = Field(ge=0, allow_inf_nan=False)
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(gt=0, le=7680)
    height: int = Field(gt=0, le=4320)


class ExtractFrameRequest(BaseModel):
    media_id: str
    timestamp_s: float = Field(ge=0)


class RenderRequest(BaseModel):
    match_id: str
    media_id: str
    width: int = Field(default=1920, ge=320, le=3840)
    height: int = Field(default=1080, ge=180, le=2160)


class GroundPointRequest(BaseModel):
    render_id: str
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    landmark_id: str | None = None


class VeoInspectRequest(BaseModel):
    match_url: str = Field(min_length=8, max_length=4096)


class VeoImportRequest(BaseModel):
    inspection_id: str
    selected_assets: dict[str, str] = Field(default_factory=dict)
    match_id: str | None = None


class RenameMatchRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class CreateClipRequest(BaseModel):
    source_media_id: str
    start_s: float = Field(ge=0, allow_inf_nan=False)
    duration_s: float = Field(gt=0, allow_inf_nan=False)
    name: str = Field(default="", max_length=120)


class StartClipAnalysisRequest(BaseModel):
    action: Literal["all", "tracking"] = "all"


class ApiError(BaseModel):
    detail: str
    code: str


MediaKind = Literal["image", "frame", "video", "clip"]
