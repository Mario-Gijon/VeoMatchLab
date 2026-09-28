#!/usr/bin/env python3
"""Build the non-destructive V2.8B canonical tracking preview.

Example:
    PYTHONPATH=. .venv/bin/python \
      experiments/tracking/build_v28_preview.py \
      --tracking-frames \
      experiments/tracking/reid_crop_debug/_tracking_tmp/tracking_frames.jsonl \
      --identity-solution \
      experiments/tracking/reid_similarity_debug/identity_graph_solution_v28a_v5.json \
      --clip-id clip_63b0df129427

Outputs:
    experiments/tracking/reid_similarity_debug/previews/<clip-id>/
      tracking_frames_v28_preview.jsonl
      tracking_preview_report.json

Nothing in the current tracking artifact is overwritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.canonical_preview import (  # noqa: E402
    build_canonical_preview,
    json_safe,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        default=Path(
            "experiments/tracking/reid_crop_debug/"
            "_tracking_tmp/tracking_frames.jsonl"
        ),
    )
    parser.add_argument(
        "--identity-solution",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug/"
            "identity_graph_solution_v28a_v5.json"
        ),
    )
    parser.add_argument(
        "--clip-id",
        required=True,
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def load_jsonl(path: Path):
    rows = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as source:
        for line in source:
            if line.strip():
                rows.append(
                    json.loads(
                        line
                    )
                )

    return rows


def main():
    args = parse_args()

    raw_rows = load_jsonl(
        args.tracking_frames
    )
    identity_solution = json.loads(
        args.identity_solution.read_text(
            encoding="utf-8",
        )
    )

    result = build_canonical_preview(
        raw_rows,
        identity_solution,
    )

    out_dir = (
        args.out_dir
        if args.out_dir is not None
        else Path(
            "experiments/tracking/"
            "reid_similarity_debug/previews"
        )
        / args.clip_id
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_path = (
        out_dir
        / "tracking_frames_v28_preview.jsonl"
    )
    report_path = (
        out_dir
        / "tracking_preview_report.json"
    )

    frames_path.write_text(
        "".join(
            json.dumps(
                json_safe(
                    row
                ),
                allow_nan=False,
            )
            + "\n"
            for row in result.rows
        ),
        encoding="utf-8",
    )

    report_path.write_text(
        json.dumps(
            json_safe(
                result.report_dict()
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    summary = result.report

    print()
    print("V2.8B CANONICAL PREVIEW BUILT")
    print(
        f"canonical identities       : "
        f"{summary.canonical_identity_count}"
    )
    print(
        f"selected identity units    : "
        f"{summary.selected_unit_count}"
    )
    print(
        f"junk identity units        : "
        f"{summary.junk_unit_count}"
    )
    print(
        f"input frames               : "
        f"{summary.input_frame_count}"
    )
    print(
        f"mapped raw samples         : "
        f"{summary.mapped_sample_count}"
    )
    print(
        f"same-frame collisions      : "
        f"{summary.collision_frame_count}"
    )
    print(
        f"observed collisions        : "
        f"{summary.observed_collision_frame_count}"
    )
    print(
        f"max collision distance     : "
        f"{summary.max_collision_distance_m:.3f} m"
    )
    print(
        f"interpolated samples added : "
        f"{summary.inserted_sample_count}"
    )
    print(
        f"final tracks               : "
        f"{summary.final_track_count}"
    )
    print(
        f"final frames               : "
        f"{summary.final_frame_count}"
    )
    print(
        f"preview                    : "
        f"{frames_path.resolve()}"
    )
    print(
        f"report                     : "
        f"{report_path.resolve()}"
    )
    print(
        "Non-destructive: current tracks.person was not modified."
    )


if __name__ == "__main__":
    main()
