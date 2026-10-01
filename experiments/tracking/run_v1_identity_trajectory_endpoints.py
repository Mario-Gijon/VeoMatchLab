#!/usr/bin/env python3
"""Run Tracking V1 Hito 7G.1 trajectory segment/endpoint modelling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.identity_trajectory_endpoints import (  # noqa: E402
    IDENTITY_TRAJECTORY_ENDPOINT_VERSION,
    IdentityTrajectoryEndpointConfig,
    build_identity_trajectory_profiles,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _stats(values):
    values = sorted(float(v) for v in values)
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    def percentile(p):
        if len(values) == 1:
            return values[0]
        pos = (len(values) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(values) - 1)
        frac = pos - lo
        return values[lo] * (1.0 - frac) + values[hi] * frac

    return {
        "count": len(values),
        "min": values[0],
        "p10": percentile(0.10),
        "median": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "max": values[-1],
        "mean": sum(values) / len(values),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--identity-dir")
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

    identity_dir = (
        Path(args.identity_dir)
        if args.identity_dir
        else root / "diagnostics"
        / "global_identity_consolidation"
        / "global-identity-consolidation-v2-family-guard"
    )
    identities_path = identity_dir / "final_identities.jsonl"
    tracklets_path = root / "tracklets" / "tracklets.jsonl"

    for path in (identities_path, tracklets_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    identities = _iter_jsonl(identities_path)
    tracklets = _iter_jsonl(tracklets_path)

    config = IdentityTrajectoryEndpointConfig()
    profiles = build_identity_trajectory_profiles(
        identity_rows=identities,
        tracklet_rows=tracklets,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "identity_trajectory_endpoints"
        / IDENTITY_TRAJECTORY_ENDPOINT_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    profiles_path = output_dir / "profiles.jsonl"
    _write_jsonl(
        profiles_path,
        (row.to_dict() for row in profiles),
    )

    start_velocities = [
        row.start.velocity for row in profiles
        if row.start.velocity is not None
    ]
    end_velocities = [
        row.end.velocity for row in profiles
        if row.end.velocity is not None
    ]
    all_segments = [
        segment
        for profile in profiles
        for segment in profile.segments
    ]
    all_gaps = [
        gap
        for profile in profiles
        for gap in profile.internal_gaps
    ]
    segment_start_velocities = [
        row.start.velocity for row in all_segments
        if row.start.velocity is not None
    ]
    segment_end_velocities = [
        row.end.velocity for row in all_segments
        if row.end.velocity is not None
    ]

    summary = {
        "schema_version": 2,
        "trajectory_endpoint_version": IDENTITY_TRAJECTORY_ENDPOINT_VERSION,
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "identity_count": len(profiles),
        "source_tracklet_count": len(tracklets),
        "continuity_segment_count": len(all_segments),
        "identity_with_internal_gap_count": sum(
            bool(row.internal_gaps) for row in profiles
        ),
        "start_velocity_available_count": len(start_velocities),
        "end_velocity_available_count": len(end_velocities),
        "segment_start_velocity_available_count": len(
            segment_start_velocities
        ),
        "segment_end_velocity_available_count": len(
            segment_end_velocities
        ),
        "start_velocity_reliability": _stats([
            row.reliability for row in start_velocities
        ]),
        "end_velocity_reliability": _stats([
            row.reliability for row in end_velocities
        ]),
        "segment_start_velocity_reliability": _stats([
            row.reliability for row in segment_start_velocities
        ]),
        "segment_end_velocity_reliability": _stats([
            row.reliability for row in segment_end_velocities
        ]),
        "start_speed_mps": _stats([
            row.speed_mps for row in start_velocities
        ]),
        "end_speed_mps": _stats([
            row.speed_mps for row in end_velocities
        ]),
        "identity_observation_count": _stats([
            row.observation_count for row in profiles
        ]),
        "identity_tracklet_count": _stats([
            len(row.tracklet_ids) for row in profiles
        ]),
        "identity_duration_s": _stats([
            row.duration_s for row in profiles
        ]),
        "continuity_segments_per_identity": _stats([
            len(row.segments) for row in profiles
        ]),
        "internal_gap_count": _stats([
            row.internal_gap_count for row in profiles
        ]),
        "internal_gap_duration_s": _stats([
            row.gap_s for row in all_gaps
        ]),
        "internal_gap_endpoint_distance_m": _stats([
            row.endpoint_distance_m for row in all_gaps
        ]),
        "internal_gap_straight_line_speed_mps": _stats([
            row.straight_line_speed_mps
            for row in all_gaps
            if row.straight_line_speed_mps is not None
        ]),
        "max_internal_gap_s": _stats([
            row.max_internal_gap_s for row in profiles
        ]),
        "config": config.to_dict(),
        "source_identities": str(identities_path),
        "source_tracklets": str(tracklets_path),
        "field_coordinates": "x_z_meters_from_canonical_tracklets",
        "velocity_is_hard_constraint": False,
        "identity_decisions_changed": False,
        "provisional": True,
        "canonical": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 2,
            "trajectory_endpoint_version": IDENTITY_TRAJECTORY_ENDPOINT_VERSION,
            "profiles": profiles_path.name,
            "summary": summary_path.name,
            "identity_count": len(profiles),
            "continuity_segment_count": len(all_segments),
            "internal_gap_count": len(all_gaps),
            "identity_decisions_changed": False,
        },
    )

    print(
        "[identity-trajectory-endpoints-7G.1-v2] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[identity-trajectory-endpoints-7G.1-v2] "
        f"identities={len(profiles)} tracklets={len(tracklets)} "
        f"segments={len(all_segments)} gaps={len(all_gaps)} "
        f"identities_with_gaps="
        f"{summary['identity_with_internal_gap_count']}"
    )
    print(
        "[identity-trajectory-endpoints-7G.1-v2] "
        f"segment_start_velocity={len(segment_start_velocities)} "
        f"segment_end_velocity={len(segment_end_velocities)}"
    )
    print(
        "[identity-trajectory-endpoints-7G.1-v2] "
        "gap_duration="
        + json.dumps(summary["internal_gap_duration_s"], sort_keys=True)
    )
    print(
        "[identity-trajectory-endpoints-7G.1-v2] "
        "gap_distance="
        + json.dumps(
            summary["internal_gap_endpoint_distance_m"],
            sort_keys=True,
        )
    )
    print(
        "[identity-trajectory-endpoints-7G.1-v2] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
