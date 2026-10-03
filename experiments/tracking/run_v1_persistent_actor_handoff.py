#!/usr/bin/env python3
"""Run Tracking V1 Hito 7I.1 local persistent-actor handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.persistent_actor_handoff import (  # noqa: E402
    PERSISTENT_ACTOR_HANDOFF_VERSION,
    PersistentActorHandoffConfig,
    build_persistent_actor_handoffs,
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


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--family-dir")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--max-handoff-gap-s",
        type=float,
        default=1.50,
    )
    parser.add_argument(
        "--soft-live-actor-limit",
        type=int,
        default=24,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    tracklets_path = root / "tracklets" / "tracklets.jsonl"
    family_dir = (
        Path(args.family_dir)
        if args.family_dir
        else root
        / "diagnostics"
        / "kit_family_consensus"
        / "kit-family-consensus-v1"
    )
    family_path = family_dir / "assignments.jsonl"

    if not tracklets_path.is_file():
        raise FileNotFoundError(tracklets_path)
    if not family_path.is_file():
        raise FileNotFoundError(family_path)

    tracklets = _iter_jsonl(tracklets_path)
    family_rows = _iter_jsonl(family_path)

    config = PersistentActorHandoffConfig(
        max_handoff_gap_s=args.max_handoff_gap_s,
        soft_live_actor_limit=args.soft_live_actor_limit,
    )

    result = build_persistent_actor_handoffs(
        tracklet_rows=tracklets,
        family_rows=family_rows,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "persistent_actor_handoff"
        / PERSISTENT_ACTOR_HANDOFF_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    actors_path = output_dir / "actors.jsonl"
    assignments_path = output_dir / "tracklet_assignments.jsonl"
    handoffs_path = output_dir / "accepted_handoffs.jsonl"
    new_actors_path = output_dir / "new_actor_events.jsonl"

    _write_jsonl(actors_path, result.actors)
    _write_jsonl(assignments_path, result.tracklet_assignments)
    _write_jsonl(
        handoffs_path,
        (row.to_dict() for row in result.accepted_handoffs),
    )
    _write_jsonl(
        new_actors_path,
        (row.to_dict() for row in result.new_actor_events),
    )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_tracklets": str(tracklets_path),
        "source_tracklet_family_consensus": str(family_path),
        "outputs": {
            "actors": str(actors_path),
            "tracklet_assignments": str(assignments_path),
            "accepted_handoffs": str(handoffs_path),
            "new_actor_events": str(new_actors_path),
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
            "handoff_version": PERSISTENT_ACTOR_HANDOFF_VERSION,
            "summary": summary_path.name,
            "actors": actors_path.name,
            "tracklet_assignments": assignments_path.name,
            "accepted_handoffs": handoffs_path.name,
            "new_actor_events": new_actors_path.name,
            "actor_count": summary["actor_count"],
            "identity_decisions_changed": False,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[persistent-actor-handoff-7I.1] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[persistent-actor-handoff-7I.1] "
        f"tracklets={summary['tracklet_count']} "
        f"actors={summary['actor_count']} "
        f"handoffs={summary['accepted_handoff_count']}"
    )
    print(
        "[persistent-actor-handoff-7I.1] "
        f"peak_live={summary['peak_live_actor_count']} "
        f"soft_limit={summary['soft_live_actor_limit']} "
        f"pressure_handoffs="
        f"{summary['population_pressure_handoff_count']}"
    )
    print(
        "[persistent-actor-handoff-7I.1] "
        "rules="
        + json.dumps(
            summary["acceptance_rule_counts"],
            sort_keys=True,
        )
    )
    print(
        "[persistent-actor-handoff-7I.1] "
        "family="
        + json.dumps(
            summary["accepted_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[persistent-actor-handoff-7I.1] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
