#!/usr/bin/env python3
"""Run and summarize the complete person-detection benchmark matrix.

This module orchestrates the existing :mod:`person_detect.run_detection`
runner; it deliberately contains no detection or geometry logic.  Detection
JSONL files contain one row per processed frame and an accepted-only
``detections`` list.  Per-frame statistics therefore count that list and
preserve the real ``frame`` and ``time_s`` values.  Standard deviation is the
population standard deviation (``statistics.pstdev``).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect import run_detection  # noqa: E402


DEFAULT_MATCH_ID = "match_e652c1f15bde"
DEFAULT_VIDEO = ROOT / "data/matches/match_e652c1f15bde/media/dual_lens.mp4"
DEFAULT_CALIBRATION = ROOT / "data/matches/match_e652c1f15bde/calibration/camera_alignment.veo"
DEFAULT_START = 756.0
DEFAULT_DURATION = 3.0
DEFAULT_THRESHOLD = 0.25
SUITE_ROOT = ROOT / "person_detect/outputs/benchmark_suites"

STAT_FIELDS = (
    "frame_count", "accepted_min", "accepted_max", "accepted_range",
    "accepted_mean", "accepted_median", "accepted_std",
    "frames_at_min", "frames_at_max", "accepted_total",
)
SUMMARY_FIELDS = (
    "processed_frames", "processed_duration_s", "wall_time_s",
    "processing_fps", "raw_person_detections", "accepted_detections",
    "rejected_outside_field", "rejected_not_owner", "rejected_other",
)
CSV_FIELDS = ["view", "model", "status", "run_dir", *STAT_FIELDS,
              *SUMMARY_FIELDS, "error"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--match-id", default=DEFAULT_MATCH_ID)
    parser.add_argument("--start", type=float, default=DEFAULT_START)
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--views", nargs="+", choices=run_detection.VIEW_CHOICES)
    parser.add_argument("--models", nargs="+", choices=run_detection.MODEL_CHOICES)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def _runner_args(args: argparse.Namespace, view: str, model: str) -> argparse.Namespace:
    return argparse.Namespace(
        model=model, view=view, weights=None, video=_resolved(args.video),
        calibration=_resolved(args.calibration), start=args.start,
        duration=args.duration, threshold=args.threshold, device=None,
        width=run_detection.OUTPUT_W, height=run_detection.OUTPUT_H,
        panorama_width=None, panorama_height=None,
        overlap=run_detection.OVERLAP,
        far_margin=run_detection.DEFAULT_FAR_TOUCHLINE_MARGIN_M,
        runs_dir=run_detection.DEFAULT_RUNS,
    )


def validate_inputs(args: argparse.Namespace) -> None:
    video = _resolved(args.video)
    calibration = _resolved(args.calibration)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration not found: {calibration}")
    for model in (args.models or run_detection.MODEL_CHOICES):
        weights = run_detection.selected_weights(
            _runner_args(args, "interactive_view", model)
        )
        if not Path(weights).is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {weights}")


def aggregate_run(run_dir: Path) -> dict[str, Any]:
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    per_frame: list[dict[str, Any]] = []
    counts: list[int] = []
    with (run_dir / "detections.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            detections = row.get("detections", [])
            if isinstance(detections, list):
                accepted = sum(
                    1 for detection in detections
                    if not isinstance(detection, dict)
                    or detection.get("status", "accepted") == "accepted"
                )
            else:
                accepted = 0
            counts.append(accepted)
            item: dict[str, Any] = {"accepted": accepted}
            if "frame" in row:
                item["frame"] = row["frame"]
            if "time_s" in row:
                item["time_s"] = row["time_s"]
            per_frame.append(item)

    stats: dict[str, Any] = {field: None for field in STAT_FIELDS}
    stats["accepted_per_frame"] = per_frame
    stats["frame_count"] = len(counts)
    stats["accepted_total"] = sum(counts)
    if counts:
        minimum, maximum = min(counts), max(counts)
        stats.update({
            "accepted_min": minimum,
            "accepted_max": maximum,
            "accepted_range": maximum - minimum,
            "accepted_mean": statistics.mean(counts),
            "accepted_median": statistics.median(counts),
            "accepted_std": statistics.pstdev(counts),
            "frames_at_min": sum(value == minimum for value in counts),
            "frames_at_max": sum(value == maximum for value in counts),
        })
    stats.update({field: summary.get(field) for field in SUMMARY_FIELDS})
    return stats


def _record(view: str, model: str, status: str, run_dir: Path | None,
            metrics: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    record: dict[str, Any] = {"view": view, "model": model, "status": status,
                              "run_dir": str(run_dir) if run_dir else None,
                              "error": error}
    record.update(metrics or {field: None for field in (*STAT_FIELDS, *SUMMARY_FIELDS)})
    if metrics is None:
        record["accepted_per_frame"] = []
    return record


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_suite_files(suite_dir: Path, manifest: dict[str, Any],
                      records: list[dict[str, Any]]) -> None:
    _write_json(suite_dir / "benchmark_manifest.json", {**manifest, "runs": records})
    _write_json(suite_dir / "benchmark_summary.json", {
        "match_id": manifest["match_id"],
        "suite_timestamp": manifest["suite_timestamp"],
        "results": records,
    })
    with (suite_dir / "benchmark_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({field: record.get(field) for field in CSV_FIELDS})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.video = _resolved(args.video)
    args.calibration = _resolved(args.calibration)
    validate_inputs(args)
    views = args.views or list(run_detection.VIEW_CHOICES)
    models = args.models or list(run_detection.MODEL_CHOICES)
    matrix = [(view, model) for view in views for model in models]

    if args.dry_run:
        print(f"Dry run: {len(matrix)} planned runs")
        print(f"video={args.video}")
        print(f"calibration={args.calibration}")
        print(f"match_id={args.match_id} start={args.start} duration={args.duration} threshold={args.threshold}")
        for index, (view, model) in enumerate(matrix, 1):
            print(f"[{index:02d}/{len(matrix):02d}] {view} / {model}")
        return 0

    suite_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
    suite_dir = SUITE_ROOT / args.match_id / suite_timestamp
    suite_dir.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    for index, (view, model) in enumerate(matrix, 1):
        print(f"[{index:02d}/{len(matrix):02d}] {view} / {model}", flush=True)
        try:
            run_dir = run_detection.run_detection(_runner_args(args, view, model))
            metrics = aggregate_run(Path(run_dir))
            records.append(_record(view, model, "success", Path(run_dir), metrics))
            print(f"  accepted={metrics['accepted_total']} frames={metrics['frame_count']} "
                  f"mean={metrics['accepted_mean']} std={metrics['accepted_std']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - suite must continue by default
            records.append(_record(view, model, "failed", None, None,
                                   f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {type(exc).__name__}: {exc}", flush=True)
            if args.fail_fast:
                break
        finally:
            gc.collect()

    manifest = {
        "match_id": args.match_id,
        "video": str(args.video),
        "calibration": str(args.calibration),
        "start": args.start,
        "duration": args.duration,
        "threshold": args.threshold,
        "views": views,
        "models": models,
        "suite_timestamp": suite_timestamp,
        "suite_completion_timestamp": datetime.now().isoformat(),
    }
    write_suite_files(suite_dir, manifest, records)
    successful = [record for record in records if record["status"] == "success"]
    if successful:
        print("\nSuccessful combinations:")
        print("view\tmodel\tmin\tmax\trange\tmean\tstd\tfps")
        for record in successful:
            print("\t".join(str(record.get(field, "")) for field in (
                "view", "model", "accepted_min", "accepted_max",
                "accepted_range", "accepted_mean", "accepted_std",
                "processing_fps",
            )))
    print(f"Suite outputs: {suite_dir}")
    return 1 if any(record["status"] == "failed" for record in records) else 0


if __name__ == "__main__":
    raise SystemExit(main())
