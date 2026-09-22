#!/usr/bin/env python3
"""Estimate temporal lag between CENTER and RIGHT Veo virtual views.

V2 replaces the first visual-only lens test with a detector + world-geometry
measurement that directly targets the problem observed in the benchmark video.

Method
------
1. Render CENTER and RIGHT from sampled source frames.
2. Run the SAME RF-DETR Large model on those frames.
3. Keep raw person detections in the angular region visible by BOTH views.
   Ownership is intentionally ignored for this diagnostic.
4. Project each bbox bottom-center to world X/Z.
5. Search temporal offsets and find which CENTER/RIGHT frame pairing makes
   the two sets of world positions agree best.
6. Report the best lag and create two comparison videos:
      - zero_offset.mp4
      - best_offset.mp4

Offset convention
-----------------
best_offset_frames > 0:
    CENTER(frame N) best matches RIGHT(frame N + offset).
    Therefore RIGHT is visually delayed / lags CENTER by approximately
    best_offset_seconds.

best_offset_frames < 0:
    RIGHT leads CENTER by approximately abs(best_offset_seconds).

This is a diagnostic, not production tracking code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect.backends.rfdetr_large import RFDETRBackend  # noqa: E402
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


DEFAULT_WEIGHTS = (
    ROOT
    / "person_detect"
    / "models"
    / "rfdetr"
    / "rf-detr-large-2026.pth"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "person_detect"
    / "outputs"
    / "sync_debug"
)

PANEL_W = 960
PANEL_H = 540


@dataclass
class Sample:
    source_frame: int
    source_time_s: float
    center_points: list[tuple[float, float]]
    right_points: list[tuple[float, float]]


def open_writer(path: Path, fps: float) -> subprocess.Popen:
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
        f"{PANEL_W * 2}x{PANEL_H}",
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
        str(path),
    ]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def rendered_limits(spec) -> tuple[float, float]:
    """Compatibility with renderer versions before rendered_min/max existed."""
    if hasattr(spec, "rendered_min") and hasattr(spec, "rendered_max"):
        return float(spec.rendered_min), float(spec.rendered_max)

    half = float(spec.hfov) / 2.0
    return float(spec.longitude) - half, float(spec.longitude) + half


def common_center_right_longitude(projector: FieldProjector) -> tuple[float, float]:
    center = projector.spec_by_name["center"]
    right = projector.spec_by_name["right"]

    center_min, center_max = rendered_limits(center)
    right_min, right_max = rendered_limits(right)

    lo = max(center_min, right_min)
    hi = min(center_max, right_max)

    if hi <= lo:
        raise RuntimeError(
            "CENTER and RIGHT rendered views have no angular overlap."
        )

    return lo, hi


def raw_world_points(
    detections,
    view_name: str,
    projector: FieldProjector,
    common_lon: tuple[float, float],
) -> list[tuple[float, float]]:
    """Keep raw persons visible by both views; DO NOT apply ownership."""
    lo, hi = common_lon
    points: list[tuple[float, float]] = []

    for detection in detections:
        projection = projector.project_bbox(
            view_name,
            *detection.xyxy,
        )

        if projection.world_x is None or projection.world_z is None:
            continue
        if not projection.inside_allowed_field:
            continue
        if projection.presentation_longitude_rad is None:
            continue

        lon = float(projection.presentation_longitude_rad)
        if not (lo <= lon <= hi):
            continue

        points.append(
            (
                float(projection.world_x),
                float(projection.world_z),
            )
        )

    return points


def pair_score(
    a: list[tuple[float, float]],
    b: list[tuple[float, float]],
    *,
    max_pair_distance_m: float,
    unmatched_penalty_m: float,
) -> tuple[float | None, int, int]:
    """Greedy one-to-one world-space score.

    We intentionally avoid identity tracking. For every frame pair, the closest
    remaining CENTER/RIGHT world points are paired. Very distant pairs are not
    forced; unmatched detections receive a fixed penalty.
    """
    if not a and not b:
        return None, 0, 0

    if not a or not b:
        count = max(len(a), len(b))
        return unmatched_penalty_m, 0, count

    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)

    dist = np.linalg.norm(
        aa[:, None, :] - bb[None, :, :],
        axis=2,
    )

    used_a: set[int] = set()
    used_b: set[int] = set()
    matched_distances: list[float] = []

    while True:
        best = None
        best_distance = float("inf")

        for i in range(len(a)):
            if i in used_a:
                continue
            for j in range(len(b)):
                if j in used_b:
                    continue
                value = float(dist[i, j])
                if value < best_distance:
                    best_distance = value
                    best = (i, j)

        if best is None or best_distance > max_pair_distance_m:
            break

        i, j = best
        used_a.add(i)
        used_b.add(j)
        matched_distances.append(best_distance)

    unmatched = (
        len(a) - len(used_a)
        + len(b) - len(used_b)
    )

    total_slots = max(len(a), len(b), 1)

    # Sum matching error plus a penalty for unmatched players, normalized by
    # the larger set size so offsets with fewer detections do not win for free.
    score = (
        sum(matched_distances)
        + unmatched * unmatched_penalty_m
    ) / total_slots

    return score, len(matched_distances), unmatched


def evaluate_offset(
    samples: list[Sample],
    offset_samples: int,
    *,
    max_pair_distance_m: float,
    unmatched_penalty_m: float,
) -> dict:
    frame_scores: list[float] = []
    total_matches = 0
    total_unmatched = 0
    frame_pairs = 0

    for center_index in range(len(samples)):
        right_index = center_index + offset_samples
        if right_index < 0 or right_index >= len(samples):
            continue

        center_points = samples[center_index].center_points
        right_points = samples[right_index].right_points

        score, matched, unmatched = pair_score(
            center_points,
            right_points,
            max_pair_distance_m=max_pair_distance_m,
            unmatched_penalty_m=unmatched_penalty_m,
        )

        if score is None:
            continue

        frame_scores.append(score)
        total_matches += matched
        total_unmatched += unmatched
        frame_pairs += 1

    if not frame_scores:
        return {
            "offset_samples": offset_samples,
            "frame_pairs": 0,
            "median_score_m": None,
            "mean_score_m": None,
            "matches": 0,
            "unmatched": 0,
        }

    return {
        "offset_samples": offset_samples,
        "frame_pairs": frame_pairs,
        "median_score_m": float(np.median(frame_scores)),
        "mean_score_m": float(np.mean(frame_scores)),
        "matches": total_matches,
        "unmatched": total_unmatched,
    }


def label_panel(
    image: np.ndarray,
    *,
    title: str,
    source_frame: int,
    source_time_s: float,
) -> np.ndarray:
    panel = cv2.resize(
        image,
        (PANEL_W, PANEL_H),
        interpolation=cv2.INTER_AREA,
    )

    cv2.putText(
        panel,
        title,
        (22, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"frame={source_frame}  t={source_time_s:.3f}s",
        (22, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


def render_comparison(
    *,
    video_path: Path,
    renderer: InteractiveRenderer,
    start_frame: int,
    sample_count: int,
    stride: int,
    offset_frames: int,
    fps: float,
    output: Path,
    label: str,
) -> None:
    """Create a visual CENTER vs shifted-RIGHT comparison without inference."""
    center_cap = cv2.VideoCapture(str(video_path))
    right_cap = cv2.VideoCapture(str(video_path))

    if not center_cap.isOpened() or not right_cap.isOpened():
        center_cap.release()
        right_cap.release()
        raise RuntimeError("Could not reopen video for comparison rendering")

    output_fps = fps / stride
    writer = open_writer(output, output_fps)

    written = 0

    try:
        for sample_index in range(sample_count):
            center_frame_index = start_frame + sample_index * stride
            right_frame_index = center_frame_index + offset_frames

            if right_frame_index < 0:
                continue

            center_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                center_frame_index,
            )
            right_cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                right_frame_index,
            )

            ok_c, frame_c = center_cap.read()
            ok_r, frame_r = right_cap.read()

            if not ok_c or frame_c is None or not ok_r or frame_r is None:
                break

            center_view = renderer.render(frame_c)["center"]
            right_view = renderer.render(frame_r)["right"]

            center_panel = label_panel(
                center_view,
                title=f"CENTER | {label}",
                source_frame=center_frame_index,
                source_time_s=center_frame_index / fps,
            )
            right_panel = label_panel(
                right_view,
                title=f"RIGHT | offset={offset_frames:+d} frames",
                source_frame=right_frame_index,
                source_time_s=right_frame_index / fps,
            )

            combined = np.concatenate(
                [center_panel, right_panel],
                axis=1,
            )

            if writer.stdin is None:
                raise RuntimeError("ffmpeg stdin unavailable")
            writer.stdin.write(combined.tobytes())
            written += 1

    finally:
        center_cap.release()
        right_cap.release()
        if writer.stdin is not None:
            try:
                writer.stdin.close()
            except BrokenPipeError:
                pass
        rc = writer.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg exited with status {rc}")

    print(f"Saved comparison ({written} frames): {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure CENTER/RIGHT temporal lag using RF-DETR + X/Z."
    )

    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
    )
    parser.add_argument("--start", type=float, default=10.0)
    parser.add_argument(
        "--duration",
        type=float,
        default=6.0,
        help="Source seconds used for lag estimation.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=3,
        help="Run detector every N source frames. 3 ~= 10 diagnostic fps.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--max-lag-seconds",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--max-pair-distance-m",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--unmatched-penalty-m",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )
    parser.add_argument("--width", type=int, default=OUTPUT_W)
    parser.add_argument("--height", type=int, default=OUTPUT_H)
    parser.add_argument("--overlap", type=float, default=OVERLAP)
    parser.add_argument(
        "--far-margin",
        type=float,
        default=DEFAULT_FAR_TOUCHLINE_MARGIN_M,
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.duration <= 0:
        raise ValueError("--duration must be > 0")
    if args.stride <= 0:
        raise ValueError("--stride must be > 0")
    if args.max_lag_seconds <= 0:
        raise ValueError("--max-lag-seconds must be > 0")

    for path in (args.video, args.calibration, args.weights):
        if not path.exists():
            raise FileNotFoundError(path)

    args.out_dir.mkdir(parents=True, exist_ok=True)

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
    common_lon = common_center_right_longitude(projector)

    backend = RFDETRBackend(
        args.weights,
        variant="large",
        device=args.device,
    )

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid source FPS")

    start_frame = max(0, int(round(args.start * fps)))
    source_frames = max(1, int(round(args.duration * fps)))
    sample_count_requested = max(1, math.ceil(source_frames / args.stride))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    samples: list[Sample] = []
    read_index = 0
    began = time.perf_counter()

    try:
        while read_index < source_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if read_index % args.stride == 0:
                source_frame = start_frame + read_index
                views = renderer.render(frame)
                predictions = backend.predict_views(
                    views,
                    threshold=args.threshold,
                )

                center_points = raw_world_points(
                    predictions["center"],
                    "center",
                    projector,
                    common_lon,
                )
                right_points = raw_world_points(
                    predictions["right"],
                    "right",
                    projector,
                    common_lon,
                )

                samples.append(
                    Sample(
                        source_frame=source_frame,
                        source_time_s=source_frame / fps,
                        center_points=center_points,
                        right_points=right_points,
                    )
                )

                print(
                    f"sample {len(samples)}/{sample_count_requested} | "
                    f"frame={source_frame} | "
                    f"center={len(center_points)} | "
                    f"right={len(right_points)}",
                    flush=True,
                )

            read_index += 1

    finally:
        cap.release()

    if len(samples) < 5:
        raise RuntimeError(
            "Too few usable samples for temporal-lag estimation."
        )

    sample_period_s = args.stride / fps
    max_offset_samples = max(
        1,
        int(round(args.max_lag_seconds / sample_period_s)),
    )

    scores = []
    for offset_samples in range(
        -max_offset_samples,
        max_offset_samples + 1,
    ):
        result = evaluate_offset(
            samples,
            offset_samples,
            max_pair_distance_m=args.max_pair_distance_m,
            unmatched_penalty_m=args.unmatched_penalty_m,
        )

        result["offset_source_frames"] = offset_samples * args.stride
        result["offset_seconds"] = (
            offset_samples * args.stride / fps
        )
        scores.append(result)

    valid_scores = [
        row
        for row in scores
        if row["median_score_m"] is not None
    ]
    if not valid_scores:
        raise RuntimeError("Could not score any temporal offsets.")

    best = min(
        valid_scores,
        key=lambda row: (
            row["median_score_m"],
            row["mean_score_m"],
            abs(row["offset_source_frames"]),
        ),
    )

    best_offset_frames = int(best["offset_source_frames"])
    best_offset_seconds = float(best["offset_seconds"])

    elapsed = time.perf_counter() - began

    zero = next(
        row for row in valid_scores
        if row["offset_source_frames"] == 0
    )

    report = {
        "video": str(args.video),
        "model": backend.info(),
        "source_fps": fps,
        "start_frame": start_frame,
        "start_s": args.start,
        "duration_s": args.duration,
        "stride_frames": args.stride,
        "diagnostic_sample_fps": fps / args.stride,
        "samples": len(samples),
        "common_center_right_longitude_deg": [
            math.degrees(common_lon[0]),
            math.degrees(common_lon[1]),
        ],
        "zero_offset": zero,
        "best_offset": best,
        "interpretation": (
            (
                f"RIGHT lags CENTER by approximately "
                f"{best_offset_seconds:.3f} s "
                f"({best_offset_frames:+d} source frames)."
            )
            if best_offset_frames > 0
            else (
                f"RIGHT leads CENTER by approximately "
                f"{abs(best_offset_seconds):.3f} s "
                f"({best_offset_frames:+d} source frames)."
            )
            if best_offset_frames < 0
            else "No measurable CENTER/RIGHT temporal lag was preferred."
        ),
        "wall_time_s": elapsed,
    }

    report_path = args.out_dir / "sync_report.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    csv_path = args.out_dir / "lag_scores.csv"
    with csv_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "offset_samples",
                "offset_source_frames",
                "offset_seconds",
                "frame_pairs",
                "median_score_m",
                "mean_score_m",
                "matches",
                "unmatched",
            ],
        )
        writer.writeheader()
        writer.writerows(scores)

    comparison_count = min(
        len(samples),
        max(1, int(round(3.0 * fps / args.stride))),
    )

    render_comparison(
        video_path=args.video,
        renderer=renderer,
        start_frame=start_frame,
        sample_count=comparison_count,
        stride=args.stride,
        offset_frames=0,
        fps=fps,
        output=args.out_dir / "zero_offset.mp4",
        label="same source time",
    )

    render_comparison(
        video_path=args.video,
        renderer=renderer,
        start_frame=start_frame,
        sample_count=comparison_count,
        stride=args.stride,
        offset_frames=best_offset_frames,
        fps=fps,
        output=args.out_dir / "best_offset.mp4",
        label="estimated sync",
    )

    print()
    print(json.dumps(report, indent=2))
    print()
    print(f"Report: {report_path}")
    print(f"Scores: {csv_path}")
    print(f"Zero-offset video: {args.out_dir / 'zero_offset.mp4'}")
    print(f"Best-offset video: {args.out_dir / 'best_offset.mp4'}")


if __name__ == "__main__":
    main()
