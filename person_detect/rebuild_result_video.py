#!/usr/bin/env python3
"""Rebuild a benchmark result video from stored detections, without inference."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect.run_detection import (
    COMBINED_H,
    COMBINED_W,
    PanoramaProjector,
    PanoramaRenderer,
    UltraPanoramaProjector,
    UltraPanoramaRenderer,
    make_combined_frame,
    open_ffmpeg_writer,
    draw_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rebuild(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    view = config["view"]
    if view not in {"panorama_view", "ultra_panorama_view"}:
        raise ValueError("rebuild_result_video currently supports panorama views only")

    video_path = Path(config["video"])
    calibration_path = Path(config["calibration"])
    panorama_size = config.get("panorama_size")
    if not panorama_size or len(panorama_size) != 2:
        raise ValueError("config.json does not contain panorama_size")
    width, height = map(int, panorama_size)

    if view == "panorama_view":
        renderer = PanoramaRenderer(calibration_path, width=width, height=height)
        projector = PanoramaProjector(
            far_touchline_margin_m=float(config.get("far_touchline_margin_m", 0.5)),
            calibration=calibration_path,
            width=width,
            height=height,
        )
    else:
        renderer = UltraPanoramaRenderer(calibration_path, width=width, height=height)
        projector = UltraPanoramaProjector(
            far_touchline_margin_m=float(config.get("far_touchline_margin_m", 0.5)),
            calibration=calibration_path,
            width=width,
            height=height,
        )

    rows = _load_rows(run_dir / "detections.jsonl")
    output = (output or (run_dir / "result_fixed.mp4")).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video reports invalid FPS")
    writer = open_ffmpeg_writer(output, width=COMBINED_W, height=COMBINED_H, fps=fps)
    model_name = config.get("model", {}).get("name", "detector")
    threshold = float(config.get("threshold", 0.0))

    try:
        for row in rows:
            if "frame" not in row:
                raise ValueError("detections.jsonl row has no frame identifier")
            frame_number = int(row["frame"])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"Could not read source frame {frame_number}")
            annotated = renderer.render(frame)
            accepted = []
            for stored in row.get("detections", []):
                if stored.get("status", "accepted") != "accepted":
                    continue
                detection = dict(stored)
                detection["xyxy"] = tuple(float(value) for value in stored["xyxy"])
                accepted.append(detection)
                draw_box(
                    annotated["panorama"],
                    detection["xyxy"],
                    float(detection["confidence"]),
                    float(detection["world_x"]),
                    float(detection["world_z"]),
                )
            combined = make_combined_frame(
                annotated,
                projector,
                accepted,
                model_name=model_name,
                threshold=threshold,
                frame_number=frame_number,
                time_s=float(row.get("time_s", frame_number / fps)),
                view_names=("panorama",),
                benchmark_view=view,
            )
            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin is unavailable")
            writer.stdin.write(combined.tobytes())
    finally:
        cap.release()
        if writer.stdin is not None:
            writer.stdin.close()
        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")
    return output


def main() -> None:
    args = parse_args()
    output = rebuild(args.run_dir, args.output)
    print(f"Rebuilt result video: {output}")


if __name__ == "__main__":
    main()
