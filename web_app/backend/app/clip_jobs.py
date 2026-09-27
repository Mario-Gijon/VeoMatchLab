"""Local asynchronous Clip analysis jobs; pipeline algorithms stay external."""

from __future__ import annotations

import json
import logging
import math
import subprocess
import threading
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .analysis_layers import AnalysisRun
from .artifacts import LocalArtifactStore
from .match_data import Provenance
from .storage import WorkspaceStore
from .services import media_record


SAMPLE_PERIOD_S = 1.0 / 15.0
logger = logging.getLogger(__name__)


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
    process: subprocess.Popen[str] | None = field(default=None, repr=False)
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
        self._generation_jobs: dict[tuple[str, str], ClipJob] = {}
        self._analysis_jobs: dict[tuple[str, str], ClipJob] = {}
        self._lock = threading.RLock()

    def job(self, job_id: str) -> ClipJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> ClipJob | None:
        job = self.job(job_id)
        if job:
            with job._lock:
                job.cancel_requested = True
                process = job.process
            self._stop_process(process)
        return job

    @staticmethod
    def _stop_process(process: subprocess.Popen[str] | None) -> None:
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)

    def shutdown(self) -> None:
        """Reap only ffmpeg processes created by this service (including reload)."""
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            if job.action == "generate" and job.status in {"queued", "running"}:
                self.cancel(job.job_id)

    def cancel_for_clip(self, match_id: str, clip_id: str) -> None:
        with self._lock:
            jobs = [item for item in self._jobs.values() if item.match_id == match_id and item.clip_id == clip_id]
        for job in jobs:
            self.cancel(job.job_id)

    def start(self, match: dict, clip: dict, source: dict, *, action: str) -> ClipJob:
        from experiments.tracking.run_global_tracker import sample_timestamps
        total = len(sample_timestamps(0.0, float(clip["duration_s"]), SAMPLE_PERIOD_S))
        with self._lock:
            key = (match["id"], clip["id"])
            existing = self._analysis_jobs.get(key)
            if existing and existing.status in {"queued", "running"}:
                return existing
            job = ClipJob(self.store.new_id("job"), match["id"], clip["id"], action, total)
            self._jobs[job.job_id] = job
            self._analysis_jobs[key] = job
        threading.Thread(target=self._run, args=(job, match, clip, source), daemon=True).start()
        return job

    def start_generation(self, match: dict, clip: dict, source: dict) -> ClipJob:
        with self._lock:
            key = (match["id"], clip["id"])
            existing = self._generation_jobs.get(key)
            if existing and existing.status in {"queued", "running"}:
                return existing
            media_id = clip.get("clip_media_id")
            media = self.store.media(media_id)[1] if media_id and self.store.media(media_id) else None
            if media and Path(media["path"]).is_file():
                try:
                    media_record(Path(media["path"]), kind="clip")
                    job = ClipJob(self.store.new_id("job"), match["id"], clip["id"], "generate", 100, status="completed", stage="generating", processed_frames=100)
                    self._jobs[job.job_id] = job
                    return job
                except ValueError:
                    pass
            job = ClipJob(self.store.new_id("job"), match["id"], clip["id"], "generate", 100, stage="generating")
            self._jobs[job.job_id] = job
            self._generation_jobs[key] = job
        threading.Thread(target=self._run_generation, args=(job, match, clip, source), daemon=True).start()
        return job

    @staticmethod
    def _progress_seconds(key: str, value: str) -> float | None:
        try:
            if key in {"out_time_us", "out_time_ms"}:
                # This installed ffmpeg emits microseconds for both names.
                return float(value) / 1_000_000
            if key == "out_time":
                hours, minutes, seconds = value.split(":")
                return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        except (ValueError, OverflowError):
            return None
        return None

    def _run_generation(self, job: ClipJob, match: dict, clip: dict, source: dict) -> None:
        output = self.store.asset_dir(job.match_id, "clips") / clip["id"] / "source.mp4"
        temporary = output.with_name("source.part.mp4")
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            with job._lock:
                job.status, job.stage = "running", "generating"
            self.store.update_clip(job.match_id, clip["id"], lambda value: value.update(generation={"status": "generating", "job_id": job.job_id}))
            command = ["ffmpeg", "-y", "-v", "error", "-ss", str(clip["source_start_s"]), "-i", str(source["path"]),
                       "-t", str(clip["duration_s"]), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                       "-preset", "veryfast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                       "-progress", "pipe:1", "-nostats", str(temporary)]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            with job._lock:
                job.process = process
            assert process.stdout is not None
            for line in process.stdout:
                if job.cancel_requested:
                    self._stop_process(process)
                    raise JobCancelled()
                key, _, value = line.strip().partition("=")
                seconds = self._progress_seconds(key, value)
                if seconds is not None:
                    with job._lock:
                        job.processed_frames = min(99, max(0, int(seconds / clip["duration_s"] * 100)))
            stderr = process.stderr.read() if process.stderr else ""
            self._check_cancelled(job)
            if process.wait() != 0:
                raise RuntimeError(stderr.strip() or "ffmpeg could not generate the clip")
            self._check_cancelled(job)
            record = media_record(temporary, name="source.mp4", kind="clip", source="generated")
            if record["width"] != source["width"] or record["height"] != source["height"] or not record.get("fps") or not record.get("duration_s") or abs(record["duration_s"] - clip["duration_s"]) > max(.5, clip["duration_s"] * .1):
                raise ValueError("Generated clip failed media validation")
            temporary.replace(output)
            record["path"] = str(output.resolve())
            record.update(id=self.store.new_id("media"), role="clip", ownership="managed")
            self.store.attach_media(job.match_id, record)
            self.store.update_clip(job.match_id, clip["id"], lambda value: value.update(clip_media_id=record["id"], generation={"status": "completed", "job_id": job.job_id}))
            with job._lock:
                job.status, job.processed_frames = "completed", 100
        except JobCancelled:
            self._stop_process(job.process)
            temporary.unlink(missing_ok=True)
            self.store.update_clip(job.match_id, clip["id"], lambda value: value.update(generation={"status": "cancelled", "job_id": job.job_id}))
            with job._lock:
                job.status = "cancelled"
        except Exception as exc:
            self._stop_process(job.process)
            temporary.unlink(missing_ok=True)
            self.store.update_clip(job.match_id, clip["id"], lambda value: value.update(generation={"status": "failed", "job_id": job.job_id, "error": str(exc)}))
            with job._lock:
                job.status, job.error = "failed", str(exc)
        finally:
            with job._lock:
                job.process = None
            with self._lock:
                if self._generation_jobs.get((job.match_id, job.clip_id)) is job:
                    self._generation_jobs.pop((job.match_id, job.clip_id), None)

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
            device=None, start_s=0.0, duration_s=clip["duration_s"],
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
            logger.exception("[clip-analysis] failed clip=%s job=%s action=%s", job.clip_id, job.job_id, job.action)
            with job._lock:
                job.status, job.error = "failed", str(exc)
        finally:
            with self._lock:
                if self._analysis_jobs.get((job.match_id, job.clip_id)) is job:
                    self._analysis_jobs.pop((job.match_id, job.clip_id), None)

    def _run_all(self, job: ClipJob, match: dict, clip: dict, source: dict) -> None:
        # Keep the web API importable in lightweight environments. Detection
        # dependencies are only needed by the worker that actually runs a clip.
        from experiments.tracking.render_tracking_debug_video import TrackingDebugVideo
        from experiments.tracking.run_global_tracker import run as run_tracker
        from experiments.virtual_camera_coverage.ultra_panorama_overlay import render_ultra_panorama
        import cv2

        calibration = match["calibration"]
        output_dir = self.store.asset_dir(job.match_id, "clips") / clip["id"] / job.job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        association_rows: list[dict] = []
        renderer = TrackingDebugVideo(output_dir / "tracking_debug.mp4", width=1280, height=720, fps=1.0 / SAMPLE_PERIOD_S)
        ultra_path = output_dir / "ultra_base.mp4"
        ultra_writer: Any = None
        args = self._runner_args(source, calibration, output_dir, clip)

        def detected(**data: Any) -> None:
            self._check_cancelled(job)
            rows.append(self._measurement_row(data["frame_index"], data["timestamp"], data["measurements"]))
            with job._lock:
                job.stage, job.processed_frames = "processing", len(rows)

        def analysed(**data: Any) -> None:
            association_rows.append({"timestamp_s": data["timestamp"], "frame_index": data["frame_index"], **data["association"].to_dict()})

        def rendered(**data: Any) -> None:
            nonlocal ultra_writer
            renderer(**data)
            panorama = render_ultra_panorama(data["frame"], data["metadata"])
            if ultra_writer is None:
                ultra_writer = cv2.VideoWriter(str(ultra_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0 / SAMPLE_PERIOD_S, (panorama.shape[1], panorama.shape[0]))
                if not ultra_writer.isOpened():
                    raise RuntimeError("Could not open ultra panorama video writer")
            ultra_writer.write(panorama)

        try:
            print(f"[clip-analysis] clip={clip['id']} source={source['path']} duration={clip['duration_s']:.3f} sample_period={SAMPLE_PERIOD_S:.7f} expected_frames={job.total_frames}")
            run_tracker(args, frame_callback=rendered, measurement_callback=detected, analysis_callback=analysed)
        finally:
            renderer.close()
            if ultra_writer is not None:
                ultra_writer.release()
        self._check_cancelled(job)
        provenance = Provenance(source="our_analysis", run_id=f"run_{job.job_id}_detection", parser_version="clip-v1")
        detection = self.artifacts.write_records(job.match_id, "detections.person_measurements", rows, provenance, source_media_ids=[source["id"]])
        association_debug = self.artifacts.write_records(job.match_id, "association.person_frames", association_rows, provenance, source_media_ids=[source["id"]])
        job.detection_artifact_id = detection["id"]
        self._save_run(job, stage="person_detection", status="completed", source_media_id=source["id"],
                       configuration={"clip_id": clip["id"], "threshold": .40, "sample_period_s": SAMPLE_PERIOD_S}, outputs=[detection["id"], association_debug["id"]])
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
            "association_artifact_id": association_debug["id"],
            "run_id": job.job_id, "debug_video": f"clips/{clip['id']}/{job.job_id}/tracking_debug.mp4",
            "ultra_base": f"clips/{clip['id']}/{job.job_id}/ultra_base.mp4",
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
            for detection_row in self.artifacts.read_window(job.match_id, detection_id, 0.0, clip["duration_s"]):
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
            **({"association_artifact_id": analysis["association_artifact_id"]} if analysis.get("association_artifact_id") else {}),
            **({"ultra_base": analysis["ultra_base"]} if analysis.get("ultra_base") else {}),
        }))
