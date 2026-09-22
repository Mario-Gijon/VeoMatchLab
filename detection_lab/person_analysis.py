"""Small, local-only orchestration for Detection Lab person analyses.

This module deliberately contains no detector implementation.  The ML worker
imports the owned RF-DETR implementation from ``analysis.person.rfdetr_large``
in the separate detection environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import threading
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


CACHE_SCHEMA_VERSION = 1
DETECTOR_CODE_VERSION = "rfdetr-large-person-v3"
MODEL_ID = "rfdetr_large"
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    match_id: str
    start_s: float
    duration_s: float
    model_id: str = MODEL_ID
    sample_fps: float = 5.0
    inference_threshold: float = 0.10
    batch_size: int = 8
    inference_dtype: str = "float32"

    @classmethod
    def from_mapping(cls, match_id: str, value: dict[str, Any]) -> "AnalysisRequest":
        try:
            request = cls(
                match_id=match_id,
                start_s=float(value["start_s"]),
                duration_s=float(value["duration_s"]),
                model_id=str(value.get("model_id", MODEL_ID)),
                sample_fps=float(value.get("sample_fps", 5.0)),
                inference_threshold=float(value.get("inference_threshold", 0.10)),
                batch_size=int(value.get("batch_size", 8)),
                inference_dtype=str(value.get("inference_dtype", "float32")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Parámetros de análisis inválidos.") from exc
        request.validate()
        return request

    def validate(self) -> None:
        if not self.match_id or "/" in self.match_id or "\\" in self.match_id or ".." in self.match_id:
            raise ValueError("match_id inválido.")
        if self.model_id != MODEL_ID:
            raise ValueError("Por ahora solo está disponible RF-DETR Large.")
        if not math.isfinite(self.start_s) or self.start_s < 0:
            raise ValueError("El inicio debe ser un tiempo no negativo.")
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("La duración debe ser mayor que cero.")
        if not math.isfinite(self.sample_fps) or not 0 < self.sample_fps <= 30:
            raise ValueError("sample_fps debe estar entre 0 y 30.")
        if not math.isfinite(self.inference_threshold) or not 0 <= self.inference_threshold <= 1:
            raise ValueError("El umbral de inferencia debe estar entre 0 y 1.")
        if self.batch_size not in {1, 2, 4, 8}:
            raise ValueError("batch_size debe ser 1, 2, 4 u 8.")
        if self.inference_dtype != "float32":
            raise ValueError("Por ahora el tipo de inferencia debe ser float32.")

    def cache_payload(self) -> dict[str, Any]:
        return {**asdict(self), "cache_schema_version": CACHE_SCHEMA_VERSION, "detector_code_version": DETECTOR_CODE_VERSION}


def cache_key(request: AnalysisRequest) -> str:
    encoded = json.dumps(request.cache_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def cache_root() -> Path:
    configured = os.getenv("VEO_MATCH_DETECTION_CACHE_ROOT")
    return (Path(configured).expanduser() if configured else ROOT / ".detection-lab-cache").resolve()


def cache_path(request: AnalysisRequest, root: Path | None = None) -> Path:
    return (root or cache_root()) / "person" / f"{cache_key(request)}.json"


def normalize_result(value: dict[str, Any], request: AnalysisRequest) -> dict[str, Any]:
    """Validate the small JSON boundary shared by the ML runner and web app."""
    samples: list[dict[str, Any]] = []
    for sample in value.get("samples", []):
        timestamp_s = float(sample["timestamp_s"])
        frame_index = int(sample["frame_index"])
        detections: list[dict[str, Any]] = []
        for detection in sample.get("detections", []):
            lens = str(detection["lens"])
            bbox = [float(number) for number in detection["bbox"]]
            confidence = float(detection["confidence"])
            if lens not in {"left", "right"} or len(bbox) != 4 or not 0 <= confidence <= 1:
                raise ValueError("Resultado de detector inválido.")
            detections.append({"lens": lens, "bbox": bbox, "confidence": confidence})
        samples.append({"timestamp_s": timestamp_s, "frame_index": frame_index, "detections": detections})
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "analysis": request.cache_payload(),
        "video": {"fps": float(value.get("video", {}).get("fps", 0)), "width": int(value.get("video", {}).get("width", 2048)), "height": int(value.get("video", {}).get("height", 2048))},
        "samples": samples,
    }


def visible_detections(sample: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    """Pure display filtering; it never affects cached inference results."""
    if not 0 <= float(threshold) <= 1:
        raise ValueError("El umbral visible debe estar entre 0 y 1.")
    return [item for item in sample.get("detections", []) if float(item["confidence"]) >= float(threshold)]


def stacked_overlay_bbox(detection: dict[str, Any]) -> list[float]:
    """Map a native lens box into the 2048×2048 stacked-video coordinate space."""
    x1, y1, x2, y2 = (float(value) for value in detection["bbox"])
    offset = 1024.0 if detection["lens"] == "right" else 0.0
    return [x1, y1 + offset, x2, y2 + offset]


Runner = Callable[[AnalysisRequest, Path], dict[str, Any]]
ProgressReporter = Callable[[dict[str, int]], None]


def subprocess_runner(
    request: AnalysisRequest,
    video_path: Path,
    report_progress: ProgressReporter | None = None,
) -> dict[str, Any]:
    python = Path(os.getenv("VEO_MATCH_DETECTION_PYTHON", str(ROOT / ".venv-detection" / "bin" / "python"))).expanduser()
    if not python.is_file():
        raise RuntimeError(f"No se encontró el intérprete de detección: {python}")
    payload = {**asdict(request), "video_path": str(video_path)}
    process = subprocess.Popen(
        [str(python), "-m", "detection_lab.analysis.person.runner"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    process.stdin.write(json.dumps(payload))
    process.stdin.close()
    output: list[str] = []
    progress_marker = "__DETECTION_LAB_PROGRESS__"
    for line in process.stdout:
        if line.startswith(progress_marker):
            if report_progress is not None:
                try:
                    report_progress(json.loads(line[len(progress_marker):]))
                except json.JSONDecodeError:
                    pass
        else:
            output.append(line)
    stderr = process.stderr.read()
    returncode = process.wait()
    stdout = "".join(output)
    if returncode:
        detail = stderr.strip() or stdout.strip() or "falló el proceso de detección"
        raise RuntimeError(detail)
    marker = "__DETECTION_LAB_RESULT__"
    result_line = next((line[len(marker):] for line in reversed(stdout.splitlines()) if line.startswith(marker)), "")
    try:
        return json.loads(result_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError("El proceso de detección no devolvió JSON válido.") from exc


class PersonAnalysisService:
    """In-process job registry for the local development server only."""

    def __init__(self, *, root: Path | None = None, runner: Runner = subprocess_runner) -> None:
        self.root = root or cache_root()
        self.runner = runner
        self.jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(self, request: AnalysisRequest, video_path: Path) -> dict[str, Any]:
        path = cache_path(request, self.root)
        if path.is_file():
            return {"state": "cached", "cache_key": cache_key(request), "result": json.loads(path.read_text(encoding="utf-8"))}
        job_id = uuid.uuid4().hex
        with self._lock:
            self.jobs[job_id] = {"state": "processing", "cache_key": cache_key(request)}
        thread = threading.Thread(target=self._run, args=(job_id, request, video_path, path), daemon=True)
        thread.start()
        return {"state": "processing", "job_id": job_id, "cache_key": cache_key(request)}

    def _run(self, job_id: str, request: AnalysisRequest, video_path: Path, path: Path) -> None:
        try:
            def report_progress(progress: dict[str, int]) -> None:
                with self._lock:
                    current = self.jobs.get(job_id)
                    if current is not None:
                        current["progress"] = progress

            runner_result = (
                subprocess_runner(request, video_path, report_progress)
                if self.runner is subprocess_runner
                else self.runner(request, video_path)
            )
            result = normalize_result(runner_result, request)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            temporary.replace(path)
            state: dict[str, Any] = {"state": "complete", "cache_key": cache_key(request), "result": result}
        except Exception as exc:  # The API reports a clear local development error.
            message = str(exc)
            if "out of memory" in message.lower() or "cuda oom" in message.lower():
                suggestion = request.batch_size // 2
                message = (
                    f"GPU memory exhausted. Try reducing batch size from {request.batch_size} to {suggestion}."
                    if suggestion >= 1 else "GPU memory exhausted with batch size 1."
                )
            state = {"state": "error", "cache_key": cache_key(request), "error": message}
        with self._lock:
            self.jobs[job_id] = state

    def status(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            current = self.jobs.get(job_id)
            return dict(current) if current is not None else None
