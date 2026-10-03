#!/usr/bin/env python3
"""Run Tracking V1 Hito 7I.7 actor-level trajectory reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.actor_trajectory_reconstruction import (  # noqa: E402
    ACTOR_TRAJECTORY_VERSION,
    ActorTrajectoryConfig,
    build_actor_trajectories,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


REASSIGNMENT_VERSION = "global-closed-roster-reassignment-v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain an object"
                )
            rows.append(dict(payload))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ) + "\n"
            )
            count += 1
    return count


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    reassignment_dir = (
        root
        / "diagnostics"
        / "global_closed_roster_reassignment"
        / REASSIGNMENT_VERSION
    )

    paths = {
        "tracklets": root / "tracklets" / "tracklets.jsonl",
        "detections": root / "detections" / "frames.jsonl",
        "actors": reassignment_dir / "actors.jsonl",
        "assignments": (
            reassignment_dir / "tracklet_assignments.jsonl"
        ),
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    result = build_actor_trajectories(
        tracklet_rows=_read_jsonl(paths["tracklets"]),
        actor_rows=_read_jsonl(paths["actors"]),
        assignment_rows=_read_jsonl(paths["assignments"]),
        detection_frame_rows=_read_jsonl(paths["detections"]),
        config=ActorTrajectoryConfig(
            expected_actor_slots=23,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "actor_trajectory_reconstruction"
        / ACTOR_TRAJECTORY_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "trajectories": output_dir / "trajectories.jsonl",
        "segments": output_dir / "segments.jsonl",
        "gap_events": output_dir / "gap_events.jsonl",
    }

    _write_jsonl(
        outputs["trajectories"],
        result.display_trajectories,
    )
    _write_jsonl(outputs["segments"], result.segment_rows)
    _write_jsonl(
        outputs["gap_events"],
        (event.to_dict() for event in result.gap_events),
    )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_reassignment": str(reassignment_dir),
        "source_tracklets": str(paths["tracklets"]),
        "source_detections": str(paths["detections"]),
        "outputs": {
            key: str(path)
            for key, path in outputs.items()
        },
        "provisional": True,
        "canonical": False,
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "trajectory_version": ACTOR_TRAJECTORY_VERSION,
            "summary": summary_path.name,
            **{
                key: path.name
                for key, path in outputs.items()
            },
            "actor_count": summary["actor_count"],
            "display_point_count": summary[
                "display_point_count"
            ],
            "segment_count": summary["segment_count"],
            "provisional": True,
            "canonical": False,
        },
    )

    cadence = summary["smoothing"].get(
        "sampling_cadence", {}
    )

    print(
        "[actor-trajectories-7I.7] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[actor-trajectories-7I.7] "
        f"actors={summary['actor_count']} "
        f"raw={summary['raw_observed_point_count']} "
        f"display={summary['display_point_count']} "
        f"interpolated={summary['interpolated_display_point_count']} "
        f"segments={summary['segment_count']}"
    )
    print(
        "[actor-trajectories-7I.7] "
        f"gaps={summary['gap_event_count']} "
        f"filled={summary['filled_gap_count']} "
        f"preserved={summary['preserved_gap_count']} "
        "reasons="
        + json.dumps(
            summary["gap_reason_counts"],
            sort_keys=True,
        )
    )
    print(
        "[actor-trajectories-7I.7] "
        f"visual_observed={summary['visual_observed_point_count']} "
        f"visual_interpolated="
        f"{summary['visual_interpolated_point_count']} "
        f"source_bbox_observed="
        f"{summary['source_bbox_observed_point_count']} "
        f"source_bbox_interpolated="
        f"{summary['source_bbox_interpolated_point_count']}"
    )
    print(
        "[actor-trajectories-7I.7] "
        f"frame_step={cadence.get('frame_step')} "
        f"time_step_s={cadence.get('time_step_s')} "
        f"estimated_fps={cadence.get('estimated_fps')}"
    )
    print(
        "[actor-trajectories-7I.7] "
        f"rts_segments={summary['smoothing'].get('rts_segment_count')} "
        f"preserved_gap_breaks="
        f"{summary['smoothing'].get('preserved_gap_break_count')} "
        f"adjustment_review="
        f"{summary['smoothing'].get('adjustment_review_count')}"
    )
    print(
        "[actor-trajectories-7I.7] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
