"""ML-environment runner for Detection Lab; it consumes the owned detector."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from detection_lab.analysis.person.rfdetr_large import RFDETRLargePersonDetector
from detection_lab.person_analysis import AnalysisRequest
from experiments.player_detection.run_interval import build_sampling_plan, split_dual_lens


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
    samples: list[dict[str, Any]] = []
    try:
        detector.load()  # Exactly once for this job.
        for sample in plan:
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample.frame_index)
            ok, stacked = capture.read()
            if not ok:
                raise RuntimeError(f"No se pudo leer el frame {sample.frame_index}.")
            left, right = split_dual_lens(stacked)
            detections: list[dict[str, Any]] = []
            for lens, image in (("left", left), ("right", right)):
                for person in detector.detect(image, color_space="bgr"):
                    detections.append({"lens": lens, "bbox": list(person.bbox), "confidence": person.confidence})
            samples.append({"timestamp_s": sample.timestamp_s, "frame_index": sample.frame_index, "detections": detections})
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
