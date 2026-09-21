"""Contracts only: no inference, runner or scheduler. Times are match seconds.

Boxes use source pixels (x, y, width, height); field positions use canonical X,Z
metres. Implementations exchange artifacts, not algorithm-specific internals.
"""
from typing import Any, Literal
from pydantic import Field
from .match_data import Contract, DataSource, Provenance


class StageDefinition(Contract):
    stage_id: str
    version: str = "1"
    requires: list[str]
    produces: list[str]


class AnalysisArtifact(Contract):
    id: str
    match_id: str
    kind: str
    source: DataSource
    source_dataset: str | None = None
    run_id: str | None = None
    source_media_ids: list[str] = Field(default_factory=list)
    format: Literal["json", "jsonl_chunks"]
    record_count: int = Field(ge=0)
    start_s: float | None = None
    end_s: float | None = None
    created_at: str
    provenance: Provenance


class AnalysisRun(Contract):
    id: str
    match_id: str
    pipeline: str
    stage_id: str
    implementation: str
    version: str
    status: Literal["queued", "running", "completed", "failed", "cancelled"] = "queued"
    source_media_ids: list[str]
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: str
    started_at: str | None = None
    completed_at: str | None = None
    upstream_artifact_ids: list[str] = Field(default_factory=list)
    output_artifact_ids: list[str] = Field(default_factory=list)
    error: str | None = None


class Observation(Contract):
    id: str
    match_id: str
    timestamp_s: float = Field(ge=0)
    frame_index: int | None = Field(default=None, ge=0)
    source_media_id: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    provenance: Provenance


class Detection(Observation):
    bbox: tuple[float, float, float, float]
    category: str


class TrackObservation(Observation):
    track_id: str
    bbox: tuple[float, float, float, float] | None = None
    detection_id: str | None = None


class EntityClassification(Observation):
    entity_id: str
    category: str
    team: str | None = None


class FieldObservation(Observation):
    entity_id: str
    x: float
    z: float


class IdentityAssignment(Observation):
    entity_id: str
    jersey_number: str | None = None
    player_id: str | None = None


class ManualCorrection(Contract):
    """Append-only overlay; target raw artifacts are never edited in place."""
    id: str
    match_id: str
    source: Literal["manual"] = "manual"
    target_artifact_id: str
    entity_ids: list[str]
    kind: Literal["team", "role", "identity", "merge_tracks", "split_track"]
    values: dict[str, Any]
    start_s: float | None = None
    end_s: float | None = None
    created_at: str
    author: str
    supersedes_id: str | None = None


# Supported contracts are NOT match capabilities and do not imply execution.
STAGES = tuple(StageDefinition(stage_id=name, requires=inputs, produces=outputs) for name, inputs, outputs in (
    ("person_detection", ["media.video"], ["detections.person"]),
    ("person_tracking", ["detections.person"], ["tracks.person"]),
    ("team_classification", ["tracks.person", "media.video"], ["classifications.team"]),
    ("role_classification", ["tracks.person", "media.video"], ["classifications.role"]),
    ("ball_detection", ["media.video"], ["detections.ball"]),
    ("ball_tracking", ["detections.ball"], ["tracks.ball"]),
    ("identity", ["tracks.person", "media.video"], ["identities"]),
    ("field_positions", ["tracks.person", "tracks.ball", "calibration"], ["field_positions"]),
    ("events", ["field_positions", "classifications.team", "classifications.role"], ["match_events"]),
    ("metrics", ["field_positions"], ["metrics"]),
    ("heatmaps", ["field_positions"], ["heatmaps"]),
))
