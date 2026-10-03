#!/usr/bin/env python3
"""Run Tracking V1 Hito 7I.5 closed-roster offline resolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.closed_roster_resolution import (  # noqa: E402
    CLOSED_ROSTER_VERSION,
    ClosedRosterConfig,
    resolve_closed_roster,
)
from tracking.appearance.evidence import (  # noqa: E402
    DEFAULT_MODEL_ID,
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


FIXED_ROSTER_VERSION = "fixed-actor-roster-v1"


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
    parser.add_argument("--fixed-roster-dir")
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

    fixed_dir = (
        Path(args.fixed_roster_dir)
        if args.fixed_roster_dir
        else root
        / "diagnostics"
        / "fixed_actor_roster"
        / FIXED_ROSTER_VERSION
    )
    actors_path = fixed_dir / "actors.jsonl"
    assignments_path = fixed_dir / "tracklet_assignments.jsonl"

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

    for path in (
        tracklets_path,
        actors_path,
        assignments_path,
        family_path,
        hard_conflicts_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    tracklets = _iter_jsonl(tracklets_path)
    actors = _iter_jsonl(actors_path)
    assignments = _iter_jsonl(assignments_path)
    families = _iter_jsonl(family_path)
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
        "[closed-roster-7I.5] "
        f"appearance_source={appearance_kind} "
        f"profiles={len(appearance_rows)} "
        "(ReID is tie-break/confirmation only)"
    )

    config = ClosedRosterConfig(
        expected_actor_slots=23,
    )

    result = resolve_closed_roster(
        tracklet_rows=tracklets,
        actor_rows=actors,
        initial_assignment_rows=assignments,
        family_rows=families,
        appearance_rows=appearance_rows,
        hard_conflicts=hard_conflicts,
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "closed_roster_resolution"
        / CLOSED_ROSTER_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    actors_out = output_dir / "actors.jsonl"
    assignments_out = output_dir / "tracklet_assignments.jsonl"
    non_roster_out = output_dir / "non_roster_tracklets.jsonl"
    decisions_out = output_dir / "decisions.jsonl"
    trajectories_out = output_dir / "actor_trajectories.jsonl"

    _write_jsonl(actors_out, result.actors)
    _write_jsonl(assignments_out, result.tracklet_assignments)
    _write_jsonl(non_roster_out, result.non_roster_tracklets)
    _write_jsonl(
        decisions_out,
        (row.to_dict() for row in result.decisions),
    )
    _write_jsonl(trajectories_out, result.actor_trajectories)

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_fixed_roster": str(fixed_dir),
        "source_tracklets": str(tracklets_path),
        "source_family": str(family_path),
        "source_appearance": appearance_source,
        "source_appearance_kind": appearance_kind,
        "appearance_profile_count": len(appearance_rows),
        "source_hard_conflicts": str(hard_conflicts_path),
        "outputs": {
            "actors": str(actors_out),
            "tracklet_assignments": str(assignments_out),
            "non_roster_tracklets": str(non_roster_out),
            "decisions": str(decisions_out),
            "actor_trajectories": str(trajectories_out),
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
            "resolution_version": CLOSED_ROSTER_VERSION,
            "summary": summary_path.name,
            "actors": actors_out.name,
            "tracklet_assignments": assignments_out.name,
            "non_roster_tracklets": non_roster_out.name,
            "decisions": decisions_out.name,
            "actor_trajectories": trajectories_out.name,
            "actor_slot_count": summary["actor_slot_count"],
            "assigned_tracklet_count": summary["assigned_tracklet_count"],
            "non_roster_tracklet_count": (
                summary["non_roster_tracklet_count"]
            ),
            "unresolved_tracklet_count": 0,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[closed-roster-7I.5] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[closed-roster-7I.5] "
        f"slots={summary['actor_slot_count']} "
        f"tracklets={summary['input_tracklet_count']} "
        f"initial={summary['initial_assignment_count']} "
        f"recovered={summary['recovered_assignment_count']} "
        f"assigned={summary['assigned_tracklet_count']} "
        f"non_roster={summary['non_roster_tracklet_count']} "
        f"unresolved={summary['unresolved_tracklet_count']}"
    )
    print(
        "[closed-roster-7I.5] rules="
        + json.dumps(
            summary["decision_rule_counts"],
            sort_keys=True,
        )
    )
    print(
        "[closed-roster-7I.5] status="
        + json.dumps(
            summary["decision_status_counts"],
            sort_keys=True,
        )
    )
    print(
        "[closed-roster-7I.5] "
        f"needs_review={summary['needs_review_count']}"
    )
    print(
        "[closed-roster-7I.5] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
