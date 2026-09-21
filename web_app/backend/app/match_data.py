"""Versioned domain data, independent of Veo and model implementations."""
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DataSource = Literal["veo", "our_analysis", "manual"]
DatasetStatus = Literal["observed", "empty", "forbidden", "unavailable", "not_observed"]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    schema_version: str = "1"


class Provenance(Contract):
    source: DataSource
    source_dataset: str | None = None
    run_id: str | None = None
    parser_version: str | None = None
    upstream_artifact_ids: list[str] = Field(default_factory=list)


class MatchEvent(Contract):
    id: str
    match_id: str
    source: DataSource
    source_dataset: str | None = None
    run_id: str | None = None
    type: str
    label: str
    start_s: float = Field(ge=0)
    duration_s: float | None = Field(default=None, ge=0)
    end_s: float | None = Field(default=None, ge=0)
    team: str | None = None
    player: str | None = None
    x: float | None = None
    z: float | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)
    thumbnail_url: str | None = None
    clip_references: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance


class DatasetObservation(Contract):
    id: str
    source: Literal["veo"] = "veo"
    source_match: str
    dataset: str
    endpoint: str
    phase: str
    observed_at: str
    context: str
    status: DatasetStatus
    http_status: int
    payload_observed: bool
    parser_version: str = "veo-datasets/1"
    # Internal capture reference only, never returned to the product UI.
    payload_path: str | None = None
