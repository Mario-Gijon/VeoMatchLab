#!/usr/bin/env python3
"""Run one detector through the common Veo LEFT/CENTER/RIGHT geometry.

Common clean benchmark runner.

Current models:
    RF-DETR Large
    RF-DETR 2XL

Pipeline:
    stacked dual-lens clip
      -> LEFT / CENTER / RIGHT renderer
      -> detector (same three rendered views)
      -> person boxes only
      -> bbox bottom-center -> X/Z ground projection
      -> strict pitch filter (+0.50 m only on far touchline)
      -> non-overlapping logical ownership
      -> accepted player boxes + 2D pitch

Only accepted detections are drawn and written to detections.jsonl.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect.models.rfdetr_large import (  # noqa: E402
    RFDETRBackend as RFDETRLargeBackend,
)
from person_detect.models.rfdetr_2xl import (  # noqa: E402
    RFDETR2XLBackend,
)
from person_detect.models.dfine_x_coco import (  # noqa: E402
    DFineXCocoBackend,
)
from person_detect.models.dfine_x_obj2coco import (  # noqa: E402
    DFineXObj2CocoBackend,
)
from person_detect.models.yolo26x import (  # noqa: E402
    YOLO26XBackend,
)
from person_detect.inference_adapter import predict_views  # noqa: E402
from person_detect.views.interactive_view.renderer import (  # noqa: E402
    DEFAULT_CALIBRATION,
    DEFAULT_VIDEO,
    OUTPUT_H,
    OUTPUT_W,
    OVERLAP,
    InteractiveRenderer,
)
from person_detect.views.interactive_view.projection import (  # noqa: E402
    DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    FieldProjector,
)
from person_detect.views.two_lens_view.renderer import (  # noqa: E402
    TwoLensRenderer,
)
from person_detect.views.two_lens_view.projection import (  # noqa: E402
    TwoLensProjector,
)
from person_detect.views.panorama_view.renderer import (  # noqa: E402
    PanoramaRenderer,
)
from person_detect.views.panorama_view.projection import (  # noqa: E402
    PanoramaProjector,
)


DEFAULT_RFDETR_LARGE = (
    ROOT
    / "person_detect"
    / "model_weights"
    / "rfdetr"
    / "rf-detr-large-2026.pth"
)
DEFAULT_RFDETR_2XL = (
    ROOT
    / "person_detect"
    / "model_weights"
    / "rfdetr"
    / "rf-detr-xxlarge.pth"
)
DEFAULT_DFINE_X_COCO = (
    ROOT
    / "person_detect"
    / "model_weights"
    / "dfine"
    / "dfine_x_coco.pth"
)
DEFAULT_DFINE_X_OBJ2COCO = (
    ROOT
    / "person_detect"
    / "model_weights"
    / "dfine"
    / "dfine_x_obj2coco.pth"
)
DEFAULT_YOLO26X = (
    ROOT
    / "person_detect"
    / "model_weights"
    / "yolo"
    / "yolo26x.pt"
)
DEFAULT_CLIPS_ROOT = ROOT / "person_detect" / "clips"
DEFAULT_DATA_MATCHES_ROOT = ROOT / "data" / "matches"
DEFAULT_RUNS = ROOT / "person_detect" / "outputs"

COMBINED_W = 1920
COMBINED_H = 1080
TOP_H = 360
PANEL_W = 640


def open_ffmpeg_writer(
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
) -> subprocess.Popen:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s:v",
        f"{width}x{height}",
        "-r",
        f"{fps:.8f}",
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]

    try:
        return subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "ffmpeg was not found. Install ffmpeg or make sure it is in PATH."
        ) from exc


def draw_box(
    image: np.ndarray,
    xyxy: tuple[float, float, float, float],
    confidence: float,
    world_x: float,
    world_z: float,
) -> None:
    x1, y1, x2, y2 = xyxy
    p1 = (int(round(x1)), int(round(y1)))
    p2 = (int(round(x2)), int(round(y2)))

    cv2.rectangle(
        image,
        p1,
        p2,
        (0, 255, 0),
        3,
        cv2.LINE_AA,
    )

    foot = (
        int(round((x1 + x2) / 2.0)),
        int(round(y2)),
    )
    cv2.circle(
        image,
        foot,
        5,
        (0, 255, 255),
        -1,
        cv2.LINE_AA,
    )

    label = f"person {confidence:.2f}  ({world_x:.1f},{world_z:.1f})"
    label_y = max(24, int(round(y1)) - 8)
    cv2.putText(
        image,
        label,
        (int(round(x1)), label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def resize_panel(image: np.ndarray, name: str, *, panel_width: int = PANEL_W) -> np.ndarray:
    panel = cv2.resize(
        image,
        (panel_width, TOP_H),
        interpolation=cv2.INTER_AREA,
    )
    cv2.putText(
        panel,
        name.upper(),
        (18, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def pitch_to_pixel(
    world_x: float,
    world_z: float,
    *,
    field_length: float,
    field_width: float,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> tuple[int, int]:
    u = (world_x + field_length / 2.0) / field_length
    v = (field_width / 2.0 - world_z) / field_width

    px = x0 + int(round(u * width))
    py = y0 + int(round(v * height))
    return px, py


def draw_pitch(
    canvas: np.ndarray,
    projector: FieldProjector,
    detections: list[dict],
) -> None:
    available_h = COMBINED_H - TOP_H
    pitch_h = min(650, available_h - 34)
    pitch_w = int(round(
        pitch_h
        * float(projector.metadata.field_length)
        / float(projector.metadata.field_width)
    ))

    if pitch_w > COMBINED_W - 80:
        pitch_w = COMBINED_W - 80
        pitch_h = int(round(
            pitch_w
            * float(projector.metadata.field_width)
            / float(projector.metadata.field_length)
        ))

    x0 = (COMBINED_W - pitch_w) // 2
    y0 = TOP_H + (available_h - pitch_h) // 2

    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + pitch_w, y0 + pitch_h),
        (35, 90, 35),
        -1,
    )

    line = (235, 235, 235)
    cv2.rectangle(
        canvas,
        (x0, y0),
        (x0 + pitch_w, y0 + pitch_h),
        line,
        3,
        cv2.LINE_AA,
    )

    mid_x = x0 + pitch_w // 2
    cv2.line(
        canvas,
        (mid_x, y0),
        (mid_x, y0 + pitch_h),
        line,
        2,
        cv2.LINE_AA,
    )

    center = (mid_x, y0 + pitch_h // 2)
    radius = int(round(pitch_h * 9.15 / float(projector.metadata.field_width)))
    cv2.circle(
        canvas,
        center,
        radius,
        line,
        2,
        cv2.LINE_AA,
    )
    cv2.circle(
        canvas,
        center,
        4,
        line,
        -1,
        cv2.LINE_AA,
    )

    # Simple penalty areas using standard 16.5 m dimensions. This is only a
    # visual radar reference; detection filtering uses Veo's field dimensions.
    penalty_depth = int(round(
        pitch_w * 16.5 / float(projector.metadata.field_length)
    ))
    penalty_width = int(round(
        pitch_h * 40.32 / float(projector.metadata.field_width)
    ))
    penalty_y = y0 + (pitch_h - penalty_width) // 2

    cv2.rectangle(
        canvas,
        (x0, penalty_y),
        (x0 + penalty_depth, penalty_y + penalty_width),
        line,
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        canvas,
        (x0 + pitch_w - penalty_depth, penalty_y),
        (x0 + pitch_w, penalty_y + penalty_width),
        line,
        2,
        cv2.LINE_AA,
    )

    for detection in detections:
        px, py = pitch_to_pixel(
            detection["world_x"],
            detection["world_z"],
            field_length=float(projector.metadata.field_length),
            field_width=float(projector.metadata.field_width),
            x0=x0,
            y0=y0,
            width=pitch_w,
            height=pitch_h,
        )

        cv2.circle(
            canvas,
            (px, py),
            8,
            (0, 255, 255),
            -1,
            cv2.LINE_AA,
        )
        cv2.circle(
            canvas,
            (px, py),
            9,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        canvas,
        f"Accepted players: {len(detections)}",
        (x0, y0 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def make_combined_frame(
    annotated_views: dict[str, np.ndarray],
    projector: FieldProjector,
    accepted: list[dict],
    *,
    model_name: str,
    threshold: float,
    frame_number: int,
    time_s: float,
    view_names: tuple[str, ...],
) -> np.ndarray:
    canvas = np.zeros(
        (COMBINED_H, COMBINED_W, 3),
        dtype=np.uint8,
    )

    panel_width = COMBINED_W // len(view_names)
    top = np.concatenate(
        [resize_panel(annotated_views[name], name, panel_width=panel_width) for name in view_names],
        axis=1,
    )
    canvas[:TOP_H, :] = top

    draw_pitch(canvas, projector, accepted)

    text = (
        f"{model_name} | threshold={threshold:.2f} | "
        f"frame={frame_number} | t={time_s:.2f}s"
    )
    cv2.putText(
        canvas,
        text,
        (24, COMBINED_H - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (220, 220, 220),
        2,
        cv2.LINE_AA,
    )

    return canvas


def selected_weights(args) -> Path:
    if args.weights is not None:
        return args.weights

    if args.model == "rfdetr_large":
        return DEFAULT_RFDETR_LARGE

    if args.model == "rfdetr_2xl":
        return DEFAULT_RFDETR_2XL

    if args.model == "dfine_x_coco":
        return DEFAULT_DFINE_X_COCO

    if args.model == "dfine_x_obj2coco":
        return DEFAULT_DFINE_X_OBJ2COCO

    if args.model == "yolo26x":
        return DEFAULT_YOLO26X

    raise ValueError(f"Unsupported model: {args.model}")


def build_backend(args):
    weights = selected_weights(args)

    if args.model == "rfdetr_large":
        return RFDETRLargeBackend(
            weights,
            variant="large",
            device=args.device,
        )

    if args.model == "rfdetr_2xl":
        return RFDETR2XLBackend(
            weights,
            device=args.device,
        )

    if args.model == "dfine_x_coco":
        return DFineXCocoBackend(
            weights,
            device=args.device,
        )

    if args.model == "dfine_x_obj2coco":
        return DFineXObj2CocoBackend(
            weights,
            device=args.device,
        )

    if args.model == "yolo26x":
        return YOLO26XBackend(
            weights,
            device=args.device,
        )

    raise ValueError(f"Unsupported model: {args.model}")


def _safe_identifier(value: str) -> str:
    safe = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "_"
        for char in value.strip()
    )
    safe = safe.strip("._")
    return safe or "clip"


def clip_identifier(video_path: Path) -> str:
    """Stable run grouping for benchmark clips and arbitrary input videos.

    person_detect/clips/<match_id>/<clip>.mp4 -> <match_id>
    any other video path                         -> sanitized filename stem
    """
    resolved = video_path.expanduser().resolve()
    for known_root in (DEFAULT_CLIPS_ROOT, DEFAULT_DATA_MATCHES_ROOT):
        try:
            relative = resolved.relative_to(known_root.resolve())
        except (ValueError, OSError):
            continue
        if len(relative.parts) >= 2:
            return _safe_identifier(relative.parts[0])

    return _safe_identifier(resolved.stem)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a detector through the common Veo player pipeline."
    )

    parser.add_argument(
        "--model",
        choices=("rfdetr_large", "rfdetr_2xl", "dfine_x_coco", "dfine_x_obj2coco", "yolo26x"),
        default="rfdetr_large",
    )
    parser.add_argument(
        "--view",
        choices=("interactive_view", "two_lens_view", "panorama_view"),
        default="interactive_view",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help=(
            "Optional model checkpoint override. When omitted, each model "
            "uses its preserved checkpoint under person_detect/model_weights/."
        ),
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO,
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--start",
        type=float,
        default=10.0,
        help="Start time inside the 90-second benchmark clip.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Seconds to process. Start small while validating a new backend.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Example: cuda, cuda:0 or cpu. Default is automatic.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=OUTPUT_W,
    )
    parser.add_argument(
        "--height",
        type=int,
        default=OUTPUT_H,
    )
    parser.add_argument(
        "--panorama-width",
        type=int,
        default=None,
        help="Optional panorama width override; default is calibration-derived.",
    )
    parser.add_argument(
        "--panorama-height",
        type=int,
        default=None,
        help="Optional panorama height override; default is source-faithful 1024.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=OVERLAP,
    )
    parser.add_argument(
        "--far-margin",
        type=float,
        default=DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.duration <= 0.0:
        raise ValueError("--duration must be > 0")
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")
    if not args.video.exists():
        raise FileNotFoundError(args.video)
    if not args.calibration.exists():
        raise FileNotFoundError(args.calibration)
    weights = selected_weights(args)
    if not weights.exists():
        raise FileNotFoundError(weights)

    print("Loading common geometry...")
    if args.view == "interactive_view":
        renderer = InteractiveRenderer(
            args.calibration,
            width=args.width,
            height=args.height,
            overlap=args.overlap,
        )
        projector = FieldProjector(
            args.calibration,
            width=args.width,
            height=args.height,
            overlap=args.overlap,
            far_touchline_margin_m=args.far_margin,
        )
        view_names = ("left", "center", "right")
    elif args.view == "two_lens_view":
        renderer = TwoLensRenderer(args.calibration)
        projector = TwoLensProjector(
            args.calibration,
            far_touchline_margin_m=args.far_margin,
        )
        view_names = ("left", "right")
    else:
        renderer = PanoramaRenderer(
            args.calibration,
            width=args.panorama_width,
            height=args.panorama_height,
        )
        projector = PanoramaProjector(
            far_touchline_margin_m=args.far_margin,
            calibration=args.calibration,
            width=renderer.width,
            height=renderer.height,
        )
        view_names = ("panorama",)

    print(f"Loading model: {args.model}")
    backend = build_backend(args)
    print(json.dumps(backend.info(), indent=2))

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    clip_id = clip_identifier(args.video)
    run_dir = args.runs_dir / args.view / backend.name / clip_id / timestamp
    run_dir.mkdir(parents=True, exist_ok=False)

    result_video = run_dir / "result.mp4"
    detections_path = run_dir / "detections.jsonl"
    config_path = run_dir / "config.json"
    summary_path = run_dir / "summary.json"

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Video reports invalid FPS")

    start_frame = max(0, int(round(args.start * fps)))
    requested_frames = max(1, int(round(args.duration * fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = open_ffmpeg_writer(
        result_video,
        width=COMBINED_W,
        height=COMBINED_H,
        fps=fps,
    )

    config = {
        "model": backend.info(),
        "view": args.view,
        "clip_id": clip_id,
        "video": str(args.video),
        "calibration": str(args.calibration),
        "start_s": args.start,
        "duration_s": args.duration,
        "threshold": args.threshold,
        "render_size_per_view": [args.width, args.height],
        "panorama_size": (
            [renderer.width, renderer.height]
            if args.view == "panorama_view"
            else None
        ),
        "overlap": args.overlap,
        "far_touchline_margin_m": args.far_margin,
        "output_video": str(result_video),
        "pipeline": (
            f"dual_lens -> {args.view} -> person detector -> bbox footpoint -> "
            "ground X/Z -> field filter -> ownership -> video + 2D pitch"
        ),
    }

    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    processed = 0
    accepted_total = 0
    raw_person_total = 0
    rejected_outside = 0
    rejected_not_owner = 0
    rejected_other = 0
    started = time.perf_counter()

    try:
        with detections_path.open("w", encoding="utf-8") as detections_file:
            while processed < requested_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break

                absolute_frame = start_frame + processed
                time_s = absolute_frame / fps

                views = renderer.render(frame)
                predictions = predict_views(
                    backend,
                    views,
                    threshold=args.threshold,
                )

                annotated = {
                    name: image.copy()
                    for name, image in views.items()
                }

                accepted: list[dict] = []

                for view_name in view_names:
                    raw_person_total += len(predictions[view_name])

                    for detection in predictions[view_name]:
                        projection = projector.project_bbox(
                            view_name,
                            *detection.xyxy,
                        )

                        if projection.status != "accepted":
                            if projection.status == "rejected_outside_field":
                                rejected_outside += 1
                            elif projection.status == "rejected_not_owner":
                                rejected_not_owner += 1
                            else:
                                rejected_other += 1
                            continue

                        item = {
                            "view": view_name,
                            "confidence": detection.confidence,
                            "xyxy": list(detection.xyxy),
                            "world_x": projection.world_x,
                            "world_z": projection.world_z,
                            "inside_strict_field": projection.inside_strict_field,
                            "field_status": projection.field_status,
                            "owner_view": projection.owner_view,
                        }
                        accepted.append(item)

                        draw_box(
                            annotated[view_name],
                            detection.xyxy,
                            detection.confidence,
                            float(projection.world_x),
                            float(projection.world_z),
                        )

                accepted_total += len(accepted)

                detections_file.write(
                    json.dumps(
                        {
                            "frame": absolute_frame,
                            "time_s": time_s,
                            "detections": accepted,
                        }
                    )
                    + "\n"
                )

                combined = make_combined_frame(
                    annotated,
                    projector,
                    accepted,
                    model_name=backend.name,
                    threshold=args.threshold,
                    frame_number=absolute_frame,
                    time_s=time_s,
                    view_names=view_names,
                )

                if writer.stdin is None:
                    raise RuntimeError("ffmpeg stdin is unavailable")
                writer.stdin.write(combined.tobytes())

                processed += 1
                if processed == 1 or processed % max(1, int(round(fps))) == 0:
                    elapsed = time.perf_counter() - started
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    print(
                        f"{processed}/{requested_frames} frames | "
                        f"{rate:.2f} processed fps | "
                        f"accepted={len(accepted)}",
                        flush=True,
                    )

    finally:
        cap.release()

        if writer.stdin is not None:
            try:
                writer.stdin.close()
            except BrokenPipeError:
                pass

        return_code = writer.wait()
        if return_code != 0:
            raise RuntimeError(
                f"ffmpeg exited with status {return_code}"
            )

    elapsed = time.perf_counter() - started

    summary = {
        "model": backend.info(),
        "view": args.view,
        "panorama_size": (
            [renderer.width, renderer.height]
            if args.view == "panorama_view"
            else None
        ),
        "processed_frames": processed,
        "processed_duration_s": processed / fps if fps > 0 else None,
        "wall_time_s": elapsed,
        "processing_fps": processed / elapsed if elapsed > 0 else None,
        "raw_person_detections": raw_person_total,
        "accepted_detections": accepted_total,
        "rejected_outside_field": rejected_outside,
        "rejected_not_owner": rejected_not_owner,
        "rejected_other": rejected_other,
        "run_dir": str(run_dir),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("Finished.")
    print(json.dumps(summary, indent=2))
    print(f"Video: {result_video}")
    print(f"Detections: {detections_path}")


if __name__ == "__main__":
    main()
