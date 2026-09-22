"""ML-environment runner for Detection Lab; it consumes the owned detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from detection_lab.analysis.person.rfdetr_large import RFDETRLargePersonDetector
from detection_lab.person_analysis import AnalysisRequest
from experiments.player_detection.run_interval import build_sampling_plan, split_dual_lens


def report_progress(processed_frames: int, total_frames: int) -> None:
    percent = round((processed_frames / total_frames) * 100) if total_frames else 100
    print(
        "__DETECTION_LAB_PROGRESS__" + json.dumps({
            "processed_frames": processed_frames,
            "total_frames": total_frames,
            "percent": percent,
        }),
        flush=True,
    )


def run(value: dict[str, Any]) -> dict[str, Any]:
    request = AnalysisRequest.from_mapping(str(value["match_id"]), value)
    video_path = Path(str(value["video_path"])).resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"No se encuentra el vídeo: {video_path}")

    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    plan = build_sampling_plan(request.start_s, request.duration_s, request.sample_fps, source_fps)
    detector = RFDETRLargePersonDetector(threshold=request.inference_threshold)
    samples = [
        {"timestamp_s": sample.timestamp_s, "frame_index": sample.frame_index, "detections": []}
        for sample in plan
    ]
    pending: list[tuple[dict[str, Any], str, Any]] = []
    remaining_lenses = {id(sample): {"left", "right"} for sample in samples}
    completed_frames = 0

    def process_pending() -> None:
        nonlocal completed_frames
        if not pending:
            return
        images = [image for _sample, _lens, image in pending]
        result_sets = detector.detect_batch(images, color_space="bgr")
        if len(result_sets) != len(pending):
            raise RuntimeError("RF-DETR devolvió resultados de batch incompletos.")
        for (sample_record, lens, _image), persons in zip(pending, result_sets, strict=True):
            sample_record["detections"].extend(
                {"lens": lens, "bbox": list(person.bbox), "confidence": person.confidence}
                for person in persons
            )
            outstanding = remaining_lenses[id(sample_record)]
            outstanding.discard(lens)
            if not outstanding:
                completed_frames += 1
        report_progress(completed_frames, len(samples))
        pending.clear()
    try:
        detector.load()  # Exactly once for this job.
        detector.prepare_inference(batch_size=request.batch_size, dtype=request.inference_dtype, compile=False)
        report_progress(0, len(plan))
        for sample, sample_record in zip(plan, samples, strict=True):
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            ok, stacked = capture.read()
            if not ok:
                raise RuntimeError(f"No se pudo leer el frame {sample.frame_index}.")
            left, right = split_dual_lens(stacked)
            for lens, image in (("left", left), ("right", right)):
                pending.append((sample_record, lens, image))
            if len(pending) >= request.batch_size:
                process_pending()
        process_pending()
    finally:
        capture.release()
        detector.unload()
    return {"video": {"fps": source_fps, "width": width, "height": height}, "samples": samples}


def main() -> None:
    # Libraries may emit informational output.  The stable marker keeps the
    # generic web-process boundary unambiguous without touching detector code.
    print("__DETECTION_LAB_RESULT__" + json.dumps(run(json.load(sys.stdin)), separators=(",", ":")))


if __name__ == "__main__":
    main()
