#!/usr/bin/env python3
"""Run Tracking V1 Hito 7I.4 fixed persistent actor roster."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.fixed_actor_roster import (  # noqa: E402
    FIXED_ACTOR_ROSTER_VERSION,
    FixedActorRosterConfig,
    build_fixed_actor_roster,
)
from tracking.appearance.evidence import (  # noqa: E402
    DEFAULT_MODEL_ID,
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
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


def _resolve_appearance_rows(
    *,
    tracking_root: Path,
    appearance_dir: Path | None,
    embeddings_jsonl: Path | None,
    model_id: str,
) -> tuple[list[dict[str, Any]], str, str]:
    if appearance_dir is not None:
        path = appearance_dir / "profiles.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        return _iter_jsonl(path), str(path), "published_profiles_explicit"

    published = tracking_root / "appearance" / "profiles.jsonl"
    if published.is_file():
        return _iter_jsonl(published), str(published), "published_profiles"

    embeddings = (
        embeddings_jsonl
        if embeddings_jsonl is not None
        else tracking_root
        / "diagnostics"
        / "reid_benchmark"
        / model_id
        / "embeddings.jsonl"
    )
    if not embeddings.is_file():
        raise FileNotFoundError(
            "No appearance profiles or ReID embeddings available. "
            f"Tried {published} and {embeddings}"
        )

    config = AppearanceEvidenceConfig(
        descriptor_name="reid",
        model_id=model_id,
    )
    samples = list(iter_embedding_jsonl(embeddings, config))
    profiles = build_tracklet_profiles(samples, config)
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
    parser.add_argument("--actor-slots", type=int, default=23)
    parser.add_argument("--family-dir")
    parser.add_argument("--appearance-dir")
    parser.add_argument("--embeddings-jsonl")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--hard-conflicts-path")
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

    hard_conflicts_path = (
        Path(args.hard_conflicts_path)
        if args.hard_conflicts_path
        else root
        / "diagnostics"
        / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )

    for path in (tracklets_path, family_path, hard_conflicts_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    tracklets = _iter_jsonl(tracklets_path)
    family_rows = _iter_jsonl(family_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)

    appearance_rows, appearance_source, appearance_kind = (
        _resolve_appearance_rows(
            tracking_root=root,
            appearance_dir=(
                Path(args.appearance_dir)
                if args.appearance_dir
                else None
            ),
            embeddings_jsonl=(
                Path(args.embeddings_jsonl)
                if args.embeddings_jsonl
                else None
            ),
            model_id=args.model_id,
        )
    )

    print(
        "[fixed-actor-roster-7I.4] "
        f"appearance_source={appearance_kind} "
        f"profiles={len(appearance_rows)}"
    )

    config = FixedActorRosterConfig(
        actor_slots=args.actor_slots,
    )

    result = build_fixed_actor_roster(
        tracklet_rows=tracklets,
        family_rows=family_rows,
        appearance_rows=appearance_rows,
        hard_conflicts=hard_conflicts,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "fixed_actor_roster"
        / FIXED_ACTOR_ROSTER_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    actors_path = output_dir / "actors.jsonl"
    assignments_path = output_dir / "tracklet_assignments.jsonl"
    handoffs_path = output_dir / "accepted_handoffs.jsonl"
    unassigned_path = output_dir / "unassigned_tracklets.jsonl"

    _write_jsonl(actors_path, result.actors)
    _write_jsonl(assignments_path, result.tracklet_assignments)
    _write_jsonl(
        handoffs_path,
        (row.to_dict() for row in result.accepted_handoffs),
    )
    _write_jsonl(unassigned_path, result.unassigned_tracklets)

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_tracklets": str(tracklets_path),
        "source_tracklet_family_consensus": str(family_path),
        "source_appearance": appearance_source,
        "source_appearance_kind": appearance_kind,
        "appearance_profile_count": len(appearance_rows),
        "source_hard_conflicts": str(hard_conflicts_path),
        "outputs": {
            "actors": str(actors_path),
            "tracklet_assignments": str(assignments_path),
            "accepted_handoffs": str(handoffs_path),
            "unassigned_tracklets": str(unassigned_path),
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
            "roster_version": FIXED_ACTOR_ROSTER_VERSION,
            "summary": summary_path.name,
            "actors": actors_path.name,
            "tracklet_assignments": assignments_path.name,
            "accepted_handoffs": handoffs_path.name,
            "unassigned_tracklets": unassigned_path.name,
            "actor_slot_count": summary["actor_slot_count"],
            "assigned_tracklet_count": summary[
                "assigned_tracklet_count"
            ],
            "unassigned_tracklet_count": summary[
                "unassigned_tracklet_count"
            ],
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[fixed-actor-roster-7I.4] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[fixed-actor-roster-7I.4] "
        f"slots={summary['actor_slot_count']} "
        f"tracklets={summary['tracklet_count']} "
        f"assigned={summary['assigned_tracklet_count']} "
        f"unassigned={summary['unassigned_tracklet_count']} "
        f"coverage={summary['assignment_coverage']:.4f}"
    )
    print(
        "[fixed-actor-roster-7I.4] "
        f"bootstrap_frame={summary['bootstrap_frame']} "
        f"bootstrap_obs={summary['bootstrap_observation_count']} "
        f"seeds={summary['bootstrap_seed_count']}"
    )
    print(
        "[fixed-actor-roster-7I.4] tiers="
        + json.dumps(summary["tier_counts"], sort_keys=True)
    )
    print(
        "[fixed-actor-roster-7I.4] unassigned="
        + json.dumps(
            summary["unassigned_reason_counts"],
            sort_keys=True,
        )
    )
    print(
        "[fixed-actor-roster-7I.4] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
