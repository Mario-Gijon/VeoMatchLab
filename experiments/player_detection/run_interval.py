#!/usr/bin/env python3
"""Run RF-DETR person detection and validated RAW field projection on one interval.

This is intentionally an experiment. It neither changes match storage nor
integrates with the web application or analytics service.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.raw_ground import raw_pixel_to_ground
from geometry.veo_metadata import load_veo_metadata


STACKED_WIDTH = 2048
STACKED_HEIGHT = 2048
LENS_HEIGHT = 1024
MODEL_NAMES = {"large", "2xl"}


@dataclass(frozen=True)
class Sample:
    timestamp_s: float
    frame_index: int


def data_root() -> Path:
    """Resolve the authoritative data root, with the ignored local link as fallback."""
    configured = os.environ.get("VEO_MATCH_DATA_ROOT")
    root = Path(configured).expanduser() if configured else ROOT / "data"
    root = root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Match data root does not exist: {root}. Set VEO_MATCH_DATA_ROOT or create the local data symlink."
        )
    return root


def match_inputs(match_id: str) -> tuple[Path, Path, Path]:
    if not match_id.startswith("match_") or any(part in match_id for part in ("/", "\\", "..")):
        raise ValueError("--match-id must be a valid stored match identifier")
    match_dir = data_root() / "matches" / match_id
    video = match_dir / "media" / "dual_lens.mp4"
    calibration = match_dir / "calibration" / "camera_alignment.veo"
    for path in (video, calibration):
        if not path.is_file():
            raise FileNotFoundError(f"Required match input is missing: {path}")
    return match_dir, video, calibration


def build_sampling_plan(start: float, duration: float, sample_fps: float, source_fps: float) -> list[Sample]:
    """Build deterministic source-frame samples for every requested model."""
    if start < 0 or duration <= 0 or sample_fps <= 0 or source_fps <= 0:
        raise ValueError("start must be non-negative and duration/sample rates must be positive")
    count = math.ceil(duration * sample_fps)
    plan: list[Sample] = []
    seen: set[int] = set()
    for ordinal in range(count):
        requested_timestamp = start + ordinal / sample_fps
        frame_index = int(round(requested_timestamp * source_fps))
        if frame_index in seen:
            continue
        seen.add(frame_index)
        plan.append(Sample(timestamp_s=frame_index / source_fps, frame_index=frame_index))
    return plan


def split_dual_lens(stacked: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split the validated 2048×2048 stacked source into native lens images."""
    if stacked.ndim != 3 or stacked.shape[:2] != (STACKED_HEIGHT, STACKED_WIDTH):
        raise ValueError(f"Expected a {STACKED_WIDTH}×{STACKED_HEIGHT} stacked frame, got {stacked.shape!r}")
    return stacked[:LENS_HEIGHT, :, :], stacked[LENS_HEIGHT:, :, :]


def bbox_bottom_center(bbox_xyxy: Iterable[float]) -> tuple[float, float]:
    x1, _y1, x2, y2 = (float(value) for value in bbox_xyxy)
    return (x1 + x2) / 2.0, y2


def lens_footpoint_to_raw(lens: str, foot_x: float, foot_y: float) -> tuple[float, float]:
    if lens == "left":
        return foot_x, foot_y
    if lens == "right":
        return foot_x, foot_y + LENS_HEIGHT
    raise ValueError(f"Unknown lens: {lens!r}")


def raw_detection_record(*, timestamp_s: float, frame_index: int, lens: str, confidence: float, bbox_xyxy: Iterable[float], model: str, class_id: int | None = None, class_name: str | None = None) -> dict[str, Any]:
    bbox = [float(value) for value in bbox_xyxy]
    if len(bbox) != 4:
        raise ValueError("bbox_xyxy must contain four coordinates")
    foot_x, foot_y = bbox_bottom_center(bbox)
    raw_x, raw_y = lens_footpoint_to_raw(lens, foot_x, foot_y)
    return {
        "timestamp_s": timestamp_s,
        "frame_index": frame_index,
        "lens": lens,
        "confidence": float(confidence),
        "class_id": class_id,
        "class_name": class_name,
        "bbox_xyxy": bbox,
        "footpoint_native": [foot_x, foot_y],
        "raw_footpoint": [raw_x, raw_y],
        "model": model,
        "world_x": None,
        "world_z": None,
        "inside_field": None,
        "ground_status": None,
    }


def field_people(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only detections that authoritative RAW geometry places on the field."""
    return [record for record in records if record.get("inside_field") is True]


def debug_sample_indices(plan: list[Sample], limit: int) -> set[int]:
    if limit < 0:
        raise ValueError("--debug-frames must be non-negative")
    if not plan or limit == 0:
        return set()
    if limit >= len(plan):
        return set(range(len(plan)))
    return {round(index * (len(plan) - 1) / (limit - 1)) for index in range(limit)} if limit > 1 else {0}


def load_model(name: str):
    if name == "large":
        from rfdetr import RFDETRLarge
        return RFDETRLarge()
    if name == "2xl":
        from rfdetr_plus import RFDETR2XLarge
        return RFDETR2XLarge()
    raise ValueError(f"Unsupported model {name!r}; choose from {sorted(MODEL_NAMES)}")


def person_class_ids(model: Any) -> set[int]:
    """Find fallback person IDs from the model's declared class mapping."""
    names = getattr(model, "class_names", None)
    if isinstance(names, dict):
        found = {int(identifier) for identifier, label in names.items() if str(label).strip().lower() == "person"}
    elif isinstance(names, (list, tuple)):
        found = {index for index, label in enumerate(names) if str(label).strip().lower() == "person"}
    else:
        return set()
    return found


def person_predictions(model: Any, image_bgr: np.ndarray, threshold: float, person_ids: set[int]) -> list[tuple[list[float], float, int, str | None]]:
    """Run one image and prefer RF-DETR's returned class names over numeric IDs."""
    import cv2

    detections = model.predict(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), threshold=threshold)
    boxes = np.asarray(getattr(detections, "xyxy"))
    confidences = np.asarray(getattr(detections, "confidence"))
    class_ids = np.asarray(getattr(detections, "class_id"))
    data = getattr(detections, "data", None)
    class_names = data.get("class_name") if isinstance(data, dict) else None
    if class_names is not None:
        class_names = np.asarray(class_names, dtype=object)
        if len(class_names) != len(boxes):
            class_names = None
    if not (len(boxes) == len(confidences) == len(class_ids)):
        raise RuntimeError("RF-DETR returned inconsistent detection arrays")
    if class_names is None and not person_ids:
        raise RuntimeError("RF-DETR returned no class_name metadata and no class mapping fallback is available")
    predictions = []
    for index, (bbox, confidence, class_id) in enumerate(zip(boxes, confidences, class_ids, strict=True)):
        original_name = None if class_names is None else class_names[index]
        normalized_name = str(original_name).strip().lower() if original_name is not None else ""
        name_available = normalized_name not in {"", "none", "nan"}
        keep = normalized_name == "person" if name_available else int(class_id) in person_ids
        if keep:
            predictions.append((
                [float(value) for value in bbox],
                float(confidence),
                int(class_id),
                str(original_name) if name_available else None,
            ))
    return predictions


def map_detection(record: dict[str, Any], metadata: Any) -> None:
    """Attach authoritative RAW geometry results while retaining failed diagnostics."""
    try:
        ground = raw_pixel_to_ground(metadata, *record["raw_footpoint"])
    except (ValueError, np.linalg.LinAlgError) as exc:
        record["ground_status"] = "projection_error"
        record["ground_error"] = str(exc)
        return
    record.update(
        world_x=ground.world_x,
        world_z=ground.world_z,
        inside_field=ground.inside_field,
        ground_status=ground.status,
    )


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, sort_keys=True) + "\n")


def render_annotated_lens(path: Path, image: np.ndarray, records: Iterable[dict[str, Any]]) -> None:
    import cv2

    annotated = image.copy()
    for record in records:
        x1, y1, x2, y2 = (round(value) for value in record["bbox_xyxy"])
        inside = record.get("inside_field") is True
        color = (75, 210, 80) if inside else (65, 150, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        foot_x, foot_y = (round(value) for value in record["footpoint_native"])
        cv2.circle(annotated, (foot_x, foot_y), 4, color, -1)
        cv2.putText(annotated, f"{record['confidence']:.2f} {'field' if inside else record.get('ground_status')}", (x1, max(16, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    if not cv2.imwrite(str(path), annotated):
        raise RuntimeError(f"Could not write debug image {path}")


def pitch_image(metadata: Any, timestamp_s: float, records: list[dict[str, Any]], *, width: int = 1050, height: int = 680) -> np.ndarray:
    import cv2

    canvas = np.full((height, width, 3), (35, 120, 55), dtype=np.uint8)
    margin = 30
    cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (240, 240, 240), 2)
    cv2.line(canvas, (width // 2, margin), (width // 2, height - margin), (240, 240, 240), 2)
    cv2.circle(canvas, (width // 2, height // 2), round((height - 2 * margin) * 9.15 / metadata.field_width), (240, 240, 240), 2)
    for record in field_people(records):
        x = float(record["world_x"])
        z = float(record["world_z"])
        px = round(margin + (x + metadata.field_length / 2) / metadata.field_length * (width - 2 * margin))
        py = round(margin + (metadata.field_width / 2 - z) / metadata.field_width * (height - 2 * margin))
        color = (255, 210, 75) if record["lens"] == "left" else (255, 190, 130)
        cv2.circle(canvas, (px, py), 5, color, -1)
    cv2.putText(canvas, f"t={timestamp_s:.3f}s  field people={len(field_people(records))}", (35, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def gpu_stats() -> tuple[str | None, int | None]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0), int(torch.cuda.max_memory_allocated(0))
    except ImportError:
        pass
    return None, None


def clear_model(model: Any) -> None:
    del model
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def run_model(*, name: str, video_path: Path, metadata: Any, plan: list[Sample], threshold: float, debug_indices: set[int], output_dir: Path, sample_fps: float, match_id: str, interval: dict[str, float]) -> dict[str, Any]:
    import cv2

    output_dir.mkdir(parents=True, exist_ok=False)
    debug_dir = output_dir / "debug_frames"
    pitch_dir = output_dir / "pitch_frames"
    debug_dir.mkdir()
    pitch_dir.mkdir()
    model = load_model(name)
    person_ids = person_class_ids(model)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        clear_model(model)
        raise RuntimeError(f"Could not open Dual Lens video: {video_path}")
    raw_records: list[dict[str, Any]] = []
    inference_seconds = 0.0
    writer: cv2.VideoWriter | None = None
    try:
        for ordinal, sample in enumerate(plan):
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            ok, stacked = capture.read()
            if not ok or stacked is None:
                raise RuntimeError(f"Could not decode source frame {sample.frame_index}")
            left, right = split_dual_lens(stacked)
            frame_records: list[dict[str, Any]] = []
            for lens, image in (("left", left), ("right", right)):
                started = time.perf_counter()
                predictions = person_predictions(model, image, threshold, person_ids)
                inference_seconds += time.perf_counter() - started
                for bbox, confidence, class_id, class_name in predictions:
                    record = raw_detection_record(timestamp_s=sample.timestamp_s, frame_index=sample.frame_index, lens=lens, confidence=confidence, bbox_xyxy=bbox, model=name, class_id=class_id, class_name=class_name)
                    map_detection(record, metadata)
                    raw_records.append(record)
                    frame_records.append(record)
            pitch = pitch_image(metadata, sample.timestamp_s, frame_records)
            if writer is None:
                writer = cv2.VideoWriter(str(output_dir / "pitch.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), sample_fps, (pitch.shape[1], pitch.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError("Could not create diagnostic pitch.mp4")
            writer.write(pitch)
            if ordinal in debug_indices:
                stem = f"frame_{sample.frame_index:08d}_{sample.timestamp_s:010.3f}"
                render_annotated_lens(debug_dir / f"{stem}_left.jpg", left, (item for item in frame_records if item["lens"] == "left"))
                render_annotated_lens(debug_dir / f"{stem}_right.jpg", right, (item for item in frame_records if item["lens"] == "right"))
                if not cv2.imwrite(str(pitch_dir / f"{stem}.jpg"), pitch):
                    raise RuntimeError("Could not write pitch debug frame")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    inside = field_people(raw_records)
    write_jsonl(output_dir / "detections.jsonl", raw_records)
    write_jsonl(output_dir / "field_people.jsonl", inside)
    gpu_name, peak_memory = gpu_stats()
    summary = {
        "model": name,
        "match_id": match_id,
        "interval": interval,
        "sample_fps": sample_fps,
        "threshold": threshold,
        "sampled_frames": len(plan),
        "total_raw_person_detections": len(raw_records),
        "total_inside_field_detections": len(inside),
        "outside_field_detections": sum(item.get("inside_field") is False for item in raw_records),
        "failed_ground_intersections": sum(item.get("inside_field") is None for item in raw_records),
        "inference_elapsed_time_s": inference_seconds,
        "average_inference_ms_per_lens_image": inference_seconds * 1000 / (len(plan) * 2) if plan else 0.0,
        "gpu_name": gpu_name,
        "peak_cuda_memory_bytes": peak_memory,
        "cross_lens_duplicates_fused": False,
        "cross_lens_duplicates_note": "Cross-lens duplicates are intentionally preserved in this experiment.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    clear_model(model)
    return summary


def parse_models(value: str) -> list[str]:
    models = [name.strip().lower() for name in value.split(",") if name.strip()]
    if not models or any(name not in MODEL_NAMES for name in models):
        raise ValueError(f"--models must be a comma-separated subset of {sorted(MODEL_NAMES)}")
    return list(dict.fromkeys(models))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--start", required=True, type=float, help="Start time in source-video seconds")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.25)
    parser.add_argument("--models", default="large,2xl")
    parser.add_argument("--debug-frames", type=int, default=6, help="Diagnostic timestamps per model; zero disables still debug frames")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    models = parse_models(args.models)
    _match_dir, video_path, calibration_path = match_inputs(args.match_id)
    import cv2
    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    capture.release()
    plan = build_sampling_plan(args.start, args.duration, args.sample_fps, source_fps)
    if not plan:
        raise RuntimeError("Sampling plan is empty")
    metadata = load_veo_metadata(calibration_path)
    now = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{args.match_id}_start-{args.start:g}_duration-{args.duration:g}_fps-{args.sample_fps:g}_{now}"
    run_dir = data_root() / "experiments" / "player_detection" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    interval = {"start_s": args.start, "duration_s": args.duration, "source_fps": source_fps}
    debug_indices = debug_sample_indices(plan, args.debug_frames)
    manifest = {
        "run_id": run_id,
        "match_id": args.match_id,
        "video": str(video_path),
        "calibration": str(calibration_path),
        "interval": interval,
        "sample_fps": args.sample_fps,
        "threshold": args.threshold,
        "models": models,
        "sample_plan": [{"timestamp_s": sample.timestamp_s, "frame_index": sample.frame_index} for sample in plan],
        "cross_lens_duplicates_fused": False,
        "cross_lens_duplicates_note": "Cross-lens duplicates are intentionally preserved in this experiment.",
    }
    (run_dir / "run.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in models:
        summary = run_model(name=name, video_path=video_path, metadata=metadata, plan=plan, threshold=args.threshold, debug_indices=debug_indices, output_dir=run_dir / name, sample_fps=args.sample_fps, match_id=args.match_id, interval=interval)
        print(f"{name}: {summary['total_inside_field_detections']} inside-field detections")
    print(run_dir)


if __name__ == "__main__":
    main()
