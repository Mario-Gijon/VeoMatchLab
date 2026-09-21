"""Read-only dataset/analysis boundaries; no execution or filesystem-path APIs."""
from fastapi import APIRouter, HTTPException, Query

from .artifacts import LocalArtifactStore, read_json
from .importers.veo.discovery import DATASETS

CAPABILITY_KINDS = {
    "has_person_detections": "detections.person", "has_person_tracks": "tracks.person",
    "has_team_assignments": "classifications.team", "has_ball_track": "tracks.ball",
    "has_field_positions": "field_positions", "has_metrics": "metrics", "has_heatmaps": "heatmaps",
}


def data_router(store_provider) -> APIRouter:
    router = APIRouter(prefix="/api/matches/{match_id}")

    def context(match_id):
        store = store_provider()
        if not store.match(match_id):
            raise HTTPException(404, "Partido no encontrado.")
        index = read_json(store.asset_dir(match_id, "datasets") / "index.json", {"datasets": {}})
        return store, index["datasets"], LocalArtifactStore(store)

    @router.get("/datasets")
    def datasets(match_id: str):
        _, entries, artifacts = context(match_id)
        summaries = [{"id": f"veo:{name}", "source": "veo", "name": name,
            "status": entries.get(f"veo:{name}", {}).get("status", "not_observed"),
            "event_count": entries.get(f"veo:{name}", {}).get("event_count", 0)} for name in DATASETS]
        available = {item["name"] for item in summaries if item["status"] == "observed"}
        kinds = {a["kind"] for a in artifacts.list_artifacts(match_id) if a["record_count"] > 0 and a["source"] == "our_analysis"}
        capabilities = {flag: kind in kinds for flag, kind in CAPABILITY_KINDS.items()}
        capabilities.update(has_veo_highlights=entries.get("veo:highlights", {}).get("event_count", 0) > 0,
            has_veo_events="events" in available, has_veo_player_tracking="player_tracking" in available,
            has_veo_heatmaps="heatmaps" in available, has_veo_metrics="physical_metrics" in available)
        return {"datasets": summaries, "capabilities": capabilities}

    @router.get("/events")
    def events(match_id: str, offset: int = Query(0, ge=0), limit: int = Query(500, ge=1, le=1000),
               start_s: float = Query(0, ge=0, allow_inf_nan=False), end_s: float | None = Query(None, ge=0, allow_inf_nan=False),
               source: str | None = None, event_type: str | None = None):
        _, entries, artifacts = context(match_id)
        if end_s is not None and end_s < start_s:
            raise HTTPException(422, "El intervalo no es válido.")
        # Only current Veo snapshots; never combine stale versions. Other sources
        # retain run identity and are not deduplicated against Veo events.
        ids = {entry["event_artifact_id"] for entry in entries.values() if entry.get("event_artifact_id")}
        ids.update(a["id"] for a in artifacts.list_artifacts(match_id) if a["kind"] == "match_events" and a["source"] != "veo")
        rows = {}
        for identifier in sorted(ids):
            for event in artifacts.read_window(match_id, identifier, start_s, end_s if end_s is not None else float("inf")):
                if (source and event["source"] != source) or (event_type and event["type"] != event_type):
                    continue
                key = (event["source"], event.get("source_dataset"), event.get("run_id"), event["id"])
                rows[key] = {k: v for k, v in event.items() if k not in {"provenance", "metadata", "clip_references"}}
        ordered = sorted(rows.values(), key=lambda e: (e["start_s"], e["source"], e.get("run_id") or "", e["id"]))
        return {"items": ordered[offset:offset + limit], "total": len(ordered),
            "next_offset": offset + limit if offset + limit < len(ordered) else None}

    @router.get("/analysis")
    def analysis_metadata(match_id: str):
        _, _, artifacts = context(match_id)
        # Configurations/errors/provenance may contain technical paths. The public
        # boundary exposes only useful, typed metadata, not internal manifests.
        runs = [{k: v for k, v in run.items() if k not in {"configuration", "error"}} for run in artifacts.list_runs(match_id)]
        return {"runs": runs, "artifacts": [{k: v for k, v in artifact.items() if k != "provenance"}
            for artifact in artifacts.list_artifacts(match_id)]}

    return router
