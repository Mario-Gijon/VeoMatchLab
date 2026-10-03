#!/usr/bin/env python3
"""Run Tracking V1 Hito 7I.6 global closed-roster reassignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_closed_roster_reassignment import (  # noqa: E402
    GLOBAL_REASSIGNMENT_VERSION,
    GlobalClosedRosterConfig,
    optimize_global_closed_roster,
)
from tracking.appearance.evidence import (  # noqa: E402
    DEFAULT_MODEL_ID,
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


FIXED_ROSTER_VERSION = "fixed-actor-roster-v1"
CLOSED_ROSTER_VERSION = "closed-roster-resolution-v1"


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


def _appearance_rows(
    *,
    tracking_root: Path,
    model_id: str,
) -> tuple[list[dict[str, Any]], str, str]:
    published = tracking_root / "appearance" / "profiles.jsonl"
    if published.is_file():
        return _read_jsonl(published), str(published), "published_profiles"

    embeddings = (
        tracking_root
        / "diagnostics"
        / "reid_benchmark"
        / model_id
        / "embeddings.jsonl"
    )
    if not embeddings.is_file():
        raise FileNotFoundError(embeddings)

    config = AppearanceEvidenceConfig(
        descriptor_name="reid",
        model_id=model_id,
    )
    profiles = build_tracklet_profiles(
        list(iter_embedding_jsonl(embeddings, config)),
        config,
    )
    return (
        [profile.to_dict() for profile in profiles],
        str(embeddings),
        "rebuilt_from_embeddings_in_memory",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
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

    fixed_dir = (
        root
        / "diagnostics"
        / "fixed_actor_roster"
        / FIXED_ROSTER_VERSION
    )
    closed_dir = (
        root
        / "diagnostics"
        / "closed_roster_resolution"
        / CLOSED_ROSTER_VERSION
    )

    paths = {
        "tracklets": root / "tracklets" / "tracklets.jsonl",
        "fixed_actors": fixed_dir / "actors.jsonl",
        "fixed_assignments": fixed_dir / "tracklet_assignments.jsonl",
        "closed_assignments": closed_dir / "tracklet_assignments.jsonl",
        "closed_non_roster": closed_dir / "non_roster_tracklets.jsonl",
        "families": (
            root
            / "diagnostics"
            / "kit_family_consensus"
            / "kit-family-consensus-v1"
            / "assignments.jsonl"
        ),
        "hard_conflicts": (
            root
            / "diagnostics"
            / "actor_candidates"
            / "sparse-actor-candidates-v1"
            / "hard_conflicts.jsonl"
        ),
    }

    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    appearances, appearance_source, appearance_kind = _appearance_rows(
        tracking_root=root,
        model_id=args.model_id,
    )

    print(
        "[global-closed-roster-7I.6] "
        f"appearance_source={appearance_kind} "
        f"profiles={len(appearances)} "
        "(ReID weight=0.06)"
    )

    result = optimize_global_closed_roster(
        tracklet_rows=_read_jsonl(paths["tracklets"]),
        fixed_actor_rows=_read_jsonl(paths["fixed_actors"]),
        fixed_assignment_rows=_read_jsonl(paths["fixed_assignments"]),
        initial_assignment_rows=_read_jsonl(paths["closed_assignments"]),
        initial_non_roster_rows=_read_jsonl(paths["closed_non_roster"]),
        family_rows=_read_jsonl(paths["families"]),
        appearance_rows=appearances,
        hard_conflicts=_read_jsonl(paths["hard_conflicts"]),
        config=GlobalClosedRosterConfig(
            expected_actor_slots=23,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "global_closed_roster_reassignment"
        / GLOBAL_REASSIGNMENT_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "actors": output_dir / "actors.jsonl",
        "tracklet_assignments": output_dir / "tracklet_assignments.jsonl",
        "non_roster_tracklets": output_dir / "non_roster_tracklets.jsonl",
        "moves": output_dir / "moves.jsonl",
        "actor_trajectories": output_dir / "actor_trajectories.jsonl",
        "final_review": output_dir / "final_review.jsonl",
    }

    _write_jsonl(outputs["actors"], result.actors)
    _write_jsonl(
        outputs["tracklet_assignments"],
        result.tracklet_assignments,
    )
    _write_jsonl(
        outputs["non_roster_tracklets"],
        result.non_roster_tracklets,
    )
    _write_jsonl(
        outputs["moves"],
        (row.to_dict() for row in result.moves),
    )
    _write_jsonl(
        outputs["actor_trajectories"],
        result.actor_trajectories,
    )
    _write_jsonl(outputs["final_review"], result.final_review_rows)

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_fixed_roster": str(fixed_dir),
        "source_closed_roster": str(closed_dir),
        "source_tracklets": str(paths["tracklets"]),
        "source_appearance": appearance_source,
        "source_appearance_kind": appearance_kind,
        "appearance_profile_count": len(appearances),
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
            "reassignment_version": GLOBAL_REASSIGNMENT_VERSION,
            "summary": summary_path.name,
            **{
                key: path.name
                for key, path in outputs.items()
            },
            "actor_slot_count": summary["actor_slot_count"],
            "assigned_tracklet_count": summary[
                "assigned_tracklet_count"
            ],
            "non_roster_tracklet_count": summary[
                "non_roster_tracklet_count"
            ],
            "unresolved_tracklet_count": 0,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[global-closed-roster-7I.6] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[global-closed-roster-7I.6] "
        f"slots={summary['actor_slot_count']} "
        f"seeds={summary['seed_count']} "
        f"tracklets={summary['tracklet_count']} "
        f"assigned={summary['assigned_tracklet_count']} "
        f"non_roster={summary['non_roster_tracklet_count']} "
        f"unresolved={summary['unresolved_tracklet_count']}"
    )
    print(
        "[global-closed-roster-7I.6] "
        f"rounds={summary['rounds_executed']} "
        f"moves={summary['accepted_move_count']} "
        f"changed={summary['changed_tracklet_count']} "
        f"actor_to_actor={summary['actor_to_actor_reassignment_count']} "
        f"rescued={summary['rescued_from_non_roster_count']} "
        f"demoted={summary['demoted_to_non_roster_count']}"
    )
    print(
        "[global-closed-roster-7I.6] move_types="
        + json.dumps(
            summary["move_type_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-closed-roster-7I.6] "
        f"needs_review={summary['needs_review_count']} "
        "review_reasons="
        + json.dumps(
            summary["review_reason_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-closed-roster-7I.6] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
