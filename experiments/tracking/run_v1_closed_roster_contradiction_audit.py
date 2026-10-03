#!/usr/bin/env python3
"""Run Tracking V1 Hito 7I.5A closed-roster contradiction audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.closed_roster_contradiction_audit import (  # noqa: E402
    AUDIT_VERSION,
    ContradictionAuditConfig,
    build_contradiction_audit,
)
from tracking.appearance.evidence import (  # noqa: E402
    DEFAULT_MODEL_ID,
    AppearanceEvidenceConfig,
    build_tracklet_profiles,
    iter_embedding_jsonl,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


RESOLUTION_VERSION = "closed-roster-resolution-v1"


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

    resolution_dir = (
        root
        / "diagnostics"
        / "closed_roster_resolution"
        / RESOLUTION_VERSION
    )

    paths = {
        "tracklets": root / "tracklets" / "tracklets.jsonl",
        "actors": resolution_dir / "actors.jsonl",
        "assignments": resolution_dir / "tracklet_assignments.jsonl",
        "decisions": resolution_dir / "decisions.jsonl",
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

    targets, pairs, summary = build_contradiction_audit(
        tracklet_rows=_read_jsonl(paths["tracklets"]),
        actor_rows=_read_jsonl(paths["actors"]),
        assignment_rows=_read_jsonl(paths["assignments"]),
        decision_rows=_read_jsonl(paths["decisions"]),
        family_rows=_read_jsonl(paths["families"]),
        appearance_rows=appearances,
        hard_conflicts=_read_jsonl(paths["hard_conflicts"]),
        config=ContradictionAuditConfig(
            expected_actor_slots=23,
        ),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "closed_roster_contradiction_audit"
        / AUDIT_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    targets_path = output_dir / "target_audits.jsonl"
    pairs_path = output_dir / "actor_pair_audits.jsonl"
    summary_path = output_dir / "summary.json"

    _write_jsonl(targets_path, targets)
    _write_jsonl(pairs_path, pairs)

    summary = dict(summary)
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_resolution": str(resolution_dir),
        "source_appearance": appearance_source,
        "source_appearance_kind": appearance_kind,
        "appearance_profile_count": len(appearances),
        "outputs": {
            "target_audits": str(targets_path),
            "actor_pair_audits": str(pairs_path),
        },
        "provisional": True,
        "mutates_assignments": False,
    })
    _write_json(summary_path, summary)
    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "audit_version": AUDIT_VERSION,
            "summary": summary_path.name,
            "target_audits": targets_path.name,
            "actor_pair_audits": pairs_path.name,
            "target_count": summary["target_count"],
            "actor_pair_audit_count": summary["actor_pair_audit_count"],
            "provisional": True,
            "mutates_assignments": False,
        },
    )

    print(
        "[closed-roster-7I.5A] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[closed-roster-7I.5A] "
        f"targets={summary['target_count']} "
        f"no_viable={summary['no_physically_viable_actor_count']} "
        f"forced={summary['closed_world_best_viable_actor_count']} "
        f"pairs={summary['actor_pair_audit_count']}"
    )
    print(
        "[closed-roster-7I.5A] pair_blockers="
        + json.dumps(
            summary["pair_blocker_counts"],
            sort_keys=True,
        )
    )
    print(
        "[closed-roster-7I.5A] no_viable "
        f"strong_non_roster="
        f"{summary['no_viable_strong_non_roster_signal_count']} "
        f"initial_assignment_pressure="
        f"{summary['no_viable_initial_assignment_pressure_signal_count']} "
        f"now_has_viable_actor="
        f"{summary['no_viable_has_viable_actor_in_final_partition_count']}"
    )
    print(
        "[closed-roster-7I.5A] forced "
        f"chosen_still_viable="
        f"{summary['forced_chosen_actor_still_viable_count']} "
        f"chosen_now_blocked="
        f"{summary['forced_chosen_actor_now_blocked_count']} "
        f"score_lt_0.45="
        f"{summary['forced_low_score_below_0_45_count']} "
        f"margin_lt_0.035="
        f"{summary['forced_low_margin_below_0_035_count']}"
    )
    print(
        "[closed-roster-7I.5A] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
