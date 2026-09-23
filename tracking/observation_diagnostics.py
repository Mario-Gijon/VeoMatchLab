#!/usr/bin/env python3
"""Inspect tracker observation quality from an existing panorama detector run.

No model is loaded and no inference is executed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tracking.observations import load_run_observations


def _stats(values: list[float]) -> dict:
    if not values:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.quantile(array, 0.95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def build_summary(run) -> dict:
    observations = list(run.iter_observations())

    return {
        "run_dir": str(run.run_dir),
        "view": run.config.get("view"),
        "model": run.config.get("model", {}).get("name"),
        "frame_count": len(run.frames),
        "observation_count": len(observations),
        "empty_frames": sum(1 for frame in run.frames if not frame.observations),
        "reprojection_error_m": _stats(
            [obs.reprojection_error_m for obs in observations]
        ),
        "metres_per_pixel_u": _stats(
            [obs.quality.sensitivity.metres_per_pixel_u for obs in observations]
        ),
        "metres_per_pixel_v": _stats(
            [obs.quality.sensitivity.metres_per_pixel_v for obs in observations]
        ),
        "seam_score": _stats([obs.quality.seam_score for obs in observations]),
        "measurement_std_x_m": _stats(
            [obs.quality.std_x_m for obs in observations]
        ),
        "measurement_std_z_m": _stats(
            [obs.quality.std_z_m for obs in observations]
        ),
        "measurement_ellipse_major_m": _stats(
            [obs.quality.ellipse_std_major_m for obs in observations]
        ),
        "bbox_height_px": _stats([obs.bbox_height for obs in observations]),
        "confidence": _stats([obs.confidence for obs in observations]),
        "high_seam_observations": sum(
            1 for obs in observations if obs.quality.seam_score >= 0.5
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose metric observation quality from an existing run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON summary path.",
    )
    parser.add_argument(
        "--export-observations",
        type=Path,
        default=None,
        help="Optional enriched observation JSONL export.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run = load_run_observations(args.run_dir)
    summary = build_summary(run)

    print(json.dumps(summary, indent=2))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Saved summary: {args.out}")

    if args.export_observations is not None:
        args.export_observations.parent.mkdir(parents=True, exist_ok=True)
        with args.export_observations.open("w", encoding="utf-8") as handle:
            for frame in run.frames:
                handle.write(json.dumps(frame.to_dict()) + "\n")
        print(f"Saved observations: {args.export_observations}")


if __name__ == "__main__":
    main()
