"""Local asynchronous Clip analysis jobs; pipeline algorithms stay external."""

from __future__ import annotations

import json
import math
import threading
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis_layers import AnalysisRun
from .artifacts import LocalArtifactStore
from .match_data import Provenance
from .storage import WorkspaceStore


SAMPLE_PERIOD_S = 1.0 / 15.0


class JobCancelled(Exception):
    pass


@dataclass
class ClipJob:
    job_id: str
    match_id: str
    clip_id: str
    action: str
    total_frames: int
    status: str = "queued"
    stage: str = "detection"
    processed_frames: int = 0
    error: str | None = None
    cancel_requested: bool = False
    detection_artifact_id: str | None = None
    tracking_artifact_id: str | None = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def public_dict(self) -> dict[str, Any]:
        with self._lock:
            return {
                "job_id": self.job_id, "match_id": self.match_id,
                "clip_id": self.clip_id, "status": self.status, "stage": self.stage,
                "processed_frames": self.processed_frames, "total_frames": self.total_frames,
                "progress": self.processed_frames / self.total_frames if self.total_frames else 0.0,
                "error": self.error, "detection_artifact_id": self.detection_artifact_id,
                "tracking_artifact_id": self.tracking_artifact_id,
            }


class ClipAnalysisService:
    def __init__(self, store: WorkspaceStore) -> None:
        self.store = store
        self.artifacts = LocalArtifactStore(store)
        self._jobs: dict[str, ClipJob] = {}
        self._lock = threading.RLock()

    def job(self, job_id: str) -> ClipJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ClipJob | None:
        job = self.job(job_id)
        if job:
            with job._lock:
                job.cancel_requested = True
        return job

    def cancel_for_clip(self, match_id: str, clip_id: str) -> None:
        with self._lock:
            jobs = [item for item in self._jobs.values() if item.match_id == match_id and item.clip_id == clip_id]
        for job in jobs:
            self.cancel(job.job_id)

    def start(self, match: dict, clip: dict, source: dict, *, action: str) -> ClipJob:
        total = int(math.floor((clip["end_s"] - clip["start_s"]) / SAMPLE_PERIOD_S + 1e-9)) + 1
        job = ClipJob(self.store.new_id("job"), match["id"], clip["id"], action, total)
        with self._lock:
            self._jobs[job.job_id] = job
        threading.Thread(target=self._run, args=(job, match, clip, source), daemon=True).start()
        return job

    def _check_cancelled(self, job: ClipJob) -> None:
        if job.cancel_requested:
            raise JobCancelled()

    def _save_run(self, job: ClipJob, *, stage: str, status: str, source_media_id: str,
                  configuration: dict[str, Any], upstream: list[str] | None = None,
                  outputs: list[str] | None = None, error: str | None = None) -> None:
        run = AnalysisRun(
            id=f"run_{job.job_id}_{stage}", match_id=job.match_id,
            pipeline="clip_detection_tracking_v1", stage_id=stage,
            implementation="experiments.tracking", version="1", status=status,
            source_media_ids=[source_media_id], configuration=configuration,
            created_at=self.store.now(), started_at=self.store.now() if status != "queued" else None,
            completed_at=self.store.now() if status in {"completed", "failed", "cancelled"} else None,
            upstream_artifact_ids=upstream or [], output_artifact_ids=outputs or [], error=error,
        )
        self.artifacts.save_run(run)

    def _runner_args(self, source: dict, calibration: dict, output_dir: Path, clip: dict) -> Namespace:
        return Namespace(
            video=Path(source["path"]), calibration=Path(calibration["path"]),
            weights=Path("person_detect/model_weights/rfdetr_finetuned/veomatch_person_v1/c2_football_images_plus_veo/best.pth"),
            device=None, start_s=clip["start_s"], duration_s=clip["end_s"] - clip["start_s"],
            sample_period_s=SAMPLE_PERIOD_S, threshold=0.40, width=1280, height=720,
            split_fraction=0.50, guard_margin_m=0.50,
            grid_step_m=0.50, field_margin_m=0.60,
            rescue_confirm_threshold=0.40, rescue_confirm_max_world_distance_m=2.50,
            out_dir=output_dir,
        )

    @staticmethod
    def _measurement_row(frame_index: int, timestamp: float, measurements: tuple[TrackingMeasurement, ...]) -> dict:
        return {"timestamp_s": timestamp, "frame_index": frame_index, "measurements": [
            {"measurement_id": str(item.measurement_id), "frame_index": item.frame_index,
             "timestamp": item.timestamp, "x": item.x, "z": item.z,
             "confidence": item.confidence, "metadata": dict(item.metadata)}
            for item in measurements
        ]}

    def _run(self, job: ClipJob, match: dict, clip: dict, source: dict) -> None:
        try:
            with job._lock:
                job.status = "running"
            if job.action == "tracking":
                self._run_cached_tracking(job, match, clip, source)
            else:
                self._run_all(job, match, clip, source)
            with job._lock:
                job.status = "completed"
        except JobCancelled:
            with job._lock:
                job.status = "cancelled"
        except Exception as exc:
            with job._lock:
                job.status, job.error = "failed", str(exc)

    def _run_all(self, job: ClipJob, match: dict, clip: dict, source: dict) -> None:
        # Keep the web API importable in lightweight environments. Detection
        # dependencies are only needed by the worker that actually runs a clip.
        from experiments.tracking.render_tracking_debug_video import TrackingDebugVideo
        from experiments.tracking.run_global_tracker import run as run_tracker

        calibration = match["calibration"]
        output_dir = self.store.asset_dir(job.match_id, "clips") / clip["id"] / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        renderer = TrackingDebugVideo(output_dir / "tracking_debug.mp4", width=1280, height=720, fps=1.0 / SAMPLE_PERIOD_S)
        args = self._runner_args(source, calibration, output_dir, clip)

        def detected(**data: Any) -> None:
            self._check_cancelled(job)
            rows.append(self._measurement_row(data["frame_index"], data["timestamp"], data["measurements"]))
            with job._lock:
                job.stage, job.processed_frames = "detection", len(rows)

        def rendered(**data: Any) -> None:
            renderer(**data)

        try:
            run_tracker(args, frame_callback=rendered, measurement_callback=detected)
        finally:
            renderer.close()
        self._check_cancelled(job)
        provenance = Provenance(source="our_analysis", run_id=f"run_{job.job_id}_detection", parser_version="clip-v1")
        detection = self.artifacts.write_records(job.match_id, "detections.person_measurements", rows, provenance, source_media_ids=[source["id"]])
        job.detection_artifact_id = detection["id"]
        self._save_run(job, stage="person_detection", status="completed", source_media_id=source["id"],
                       configuration={"clip_id": clip["id"], "threshold": .40, "sample_period_s": SAMPLE_PERIOD_S}, outputs=[detection["id"]])
        with job._lock:
            job.stage, job.processed_frames = "tracking", 0
        track_rows = [json.loads(line) | {"timestamp_s": json.loads(line)["timestamp"]}
                      for line in (output_dir / "tracking_frames.jsonl").read_text(encoding="utf-8").splitlines()]
        tracks = self.artifacts.write_records(job.match_id, "tracks.person", track_rows,
            Provenance(source="our_analysis", run_id=f"run_{job.job_id}_tracking", parser_version="clip-v1", upstream_artifact_ids=[detection["id"]]), source_media_ids=[source["id"]])
        job.tracking_artifact_id = tracks["id"]
        self._save_run(job, stage="person_tracking", status="completed", source_media_id=source["id"],
                       configuration={"clip_id": clip["id"], "sample_period_s": SAMPLE_PERIOD_S}, upstream=[detection["id"]], outputs=[tracks["id"]])
        self.store.update_clip(job.match_id, clip["id"], lambda value: value.update(analysis={
            "status": "completed", "detection_artifact_id": detection["id"], "tracking_artifact_id": tracks["id"],
            "run_id": job.job_id, "debug_video": f"clips/{clip['id']}/{job.job_id}/tracking_debug.mp4",
        }))
        with job._lock:
            job.stage, job.processed_frames = "rendering", job.total_frames

    def _run_cached_tracking(self, job: ClipJob, match: dict, clip: dict, source: dict) -> None:
        from experiments.tracking.render_tracking_debug_video import TrackingDebugVideo
        from experiments.virtual_camera_coverage.coverage import DEFAULT_FAR_CAMERAS, DEFAULT_NEAR_CAMERAS, derive_specs, read_frame
        from experiments.virtual_camera_coverage.detect import ordered_tight_specs
        from geometry.veo_metadata import load_veo_metadata
        from tracking import GlobalTracker, TrackingMeasurement
        analysis = clip.get("analysis") or {}
        detection_id = analysis.get("detection_artifact_id")
        if not detection_id:
            raise ValueError("This clip has no cached detection artifact. Run all first.")
        output_dir = self.store.asset_dir(job.match_id, "clips") / clip["id"] / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        tracker = GlobalTracker()
        metadata = load_veo_metadata(match["calibration"]["path"])
        specs, _, _, _ = derive_specs(metadata, 1280, 720, split_fraction=.50, near_count=DEFAULT_NEAR_CAMERAS, far_count=DEFAULT_FAR_CAMERAS, guard_margin_m=.50, grid_step_m=.50)
        spec_by_name = {spec.name: spec for spec in ordered_tight_specs(specs)}
        track_rows = []
        render_rows: list[tuple[dict[str, Any], tuple[TrackingMeasurement, ...], Any]] = []
        try:
            for detection_row in self.artifacts.read_window(job.match_id, detection_id, clip["start_s"], clip["end_s"]):
                self._check_cancelled(job)
                measurements = tuple(TrackingMeasurement(**item) for item in detection_row["measurements"])
                frame = tracker.update(measurements, timestamp=detection_row["timestamp_s"], frame_index=detection_row["frame_index"])
                track_rows.append({"timestamp_s": frame.timestamp, "frame_index": frame.frame_index,
                               "timestamp": frame.timestamp, "measurement_count": len(measurements),
                               "tracks": [{"track_id": item.track_id, "x": item.x, "z": item.z, "vx": item.vx, "vz": item.vz,
                                           "lifecycle": item.lifecycle.value, "position_kind": item.position_kind.value,
                                           "measurement_id": item.measurement_id, "missed_seconds": item.missed_seconds, "hits": item.hits}
                                          for item in frame.tracks]})
                render_rows.append((detection_row, measurements, frame))
                with job._lock:
                    job.stage, job.processed_frames = "tracking", len(track_rows)
            renderer = TrackingDebugVideo(output_dir / "tracking_debug.mp4", width=1280, height=720, fps=1.0 / SAMPLE_PERIOD_S)
            try:
                for index, (detection_row, measurements, frame) in enumerate(render_rows, start=1):
                    self._check_cancelled(job)
                    raw_frame, _ = read_frame(Path(source["path"]), detection_row["timestamp_s"])
                    renderer(frame=raw_frame, metadata=metadata, spec_by_name=spec_by_name, tracking_frame=frame, measurements=measurements, record={"measurement_count": len(measurements)})
                    with job._lock:
                        job.stage, job.processed_frames = "rendering", index
            finally:
                renderer.close()
        finally:
            pass
        provenance = Provenance(source="our_analysis", run_id=f"run_{job.job_id}_tracking", parser_version="clip-v1", upstream_artifact_ids=[detection_id])
        tracks = self.artifacts.write_records(job.match_id, "tracks.person", track_rows, provenance, source_media_ids=[source["id"]])
        job.detection_artifact_id, job.tracking_artifact_id = detection_id, tracks["id"]
        self._save_run(job, stage="person_tracking", status="completed", source_media_id=source["id"],
                       configuration={"clip_id": clip["id"], "sample_period_s": SAMPLE_PERIOD_S}, upstream=[detection_id], outputs=[tracks["id"]])
        summary = {"frames_processed": len(track_rows), "measurements_processed": sum(row["measurement_count"] for row in track_rows),
                   "source_detection_artifact_id": detection_id}
        (output_dir / "tracking_frames.jsonl").write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in track_rows), encoding="utf-8")
        (output_dir / "tracking_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        self.store.update_clip(job.match_id, clip["id"], lambda value: value.update(analysis={
            "status": "completed", "detection_artifact_id": detection_id,
            "tracking_artifact_id": tracks["id"], "run_id": job.job_id,
            "debug_video": f"clips/{clip['id']}/{job.job_id}/tracking_debug.mp4",
        }))
