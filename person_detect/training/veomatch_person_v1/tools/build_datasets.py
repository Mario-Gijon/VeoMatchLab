#!/usr/bin/env python3
"""Build the independent C1/C2/C3 cumulative COCO datasets."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .dataset_tools import convert_coco, merge_datasets


ROOT = Path(__file__).resolve().parents[4]
WORKSPACE = ROOT / "person_detect/training/veomatch_person_v1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--football-images", type=Path, required=True)
    parser.add_argument("--veo-ball-detection", type=Path, required=True)
    parser.add_argument("--football-eoakc", type=Path, required=True)
    args = parser.parse_args(argv)
    staging = WORKSPACE / ".staging"
    if staging.exists():
        shutil.rmtree(staging)
    converted = {}
    for name, source in (("football_images", args.football_images), ("veo_ball_detection", args.veo_ball_detection), ("football_eoakc", args.football_eoakc)):
        destination = staging / name
        convert_coco(source, destination, source_name=name)
        converted[name] = destination
    combinations = {
        "c1_football_images": ["football_images"],
        "c2_football_images_plus_veo": ["football_images", "veo_ball_detection"],
        "c3_football_images_plus_veo_plus_eoakc": ["football_images", "veo_ball_detection", "football_eoakc"],
    }
    c2_assignments = None
    for experiment, sources in combinations.items():
        output = WORKSPACE / "datasets" / experiment
        if output.exists():
            shutil.rmtree(output)
        result = merge_datasets(
            [converted[name] for name in sources],
            output,
            dataset_name=experiment,
            group_split_overrides=c2_assignments if experiment == "c3_football_images_plus_veo_plus_eoakc" else None,
        )
        if experiment == "c2_football_images_plus_veo":
            c2_assignments = result["sequence_assignments"]
        result.pop("sequence_assignments", None)
        print(result)
    shutil.rmtree(staging)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
