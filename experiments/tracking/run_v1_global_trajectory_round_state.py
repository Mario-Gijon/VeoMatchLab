#!/usr/bin/env python3
"""Build the refreshed identity-evidence state after a 7G merge round."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_trajectory_round_state import (  # noqa: E402
    GLOBAL_TRAJECTORY_ROUND_STATE_VERSION,
    build_global_trajectory_round_state,
)
from tracking.actors.identity_expansion import (  # noqa: E402
    IdentityExpansionConfig,
    build_hard_identity_pairs,
)
from tracking.appearance.kit_entity_consensus import (  # noqa: E402
    KitEntityConsensusConfig,
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
    parser.add_argument("--source-identity-dir")
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

    source_dir = (
        Path(args.source_identity_dir)
        if args.source_identity_dir
        else root / "diagnostics"
        / "global_trajectory_application"
        / "global-trajectory-apply-v1"
    )

    identities_path = source_dir / "final_identities.jsonl"
    actor_assignments_path = (
        source_dir / "final_actor_assignments.jsonl"
    )
    tracklet_assignments_path = (
        source_dir / "final_tracklet_assignments.jsonl"
    )

    chain_profile_path = (
        root / "diagnostics" / "actor_chain_reid"
        / "actor-chain-reid-v1" / "profiles.jsonl"
    )
    tracklet_family_path = (
        root / "diagnostics" / "kit_family_consensus"
        / "kit-family-consensus-v1" / "assignments.jsonl"
    )
    hard_conflicts_path = (
        root / "diagnostics" / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "hard_conflicts.jsonl"
    )

    for path in (
        identities_path,
        actor_assignments_path,
        tracklet_assignments_path,
        chain_profile_path,
        tracklet_family_path,
        hard_conflicts_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    identities = _iter_jsonl(identities_path)
    actor_assignments = _iter_jsonl(actor_assignments_path)
    tracklet_assignments = _iter_jsonl(tracklet_assignments_path)
    chain_profiles = _iter_jsonl(chain_profile_path)
    tracklet_family = _iter_jsonl(tracklet_family_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)

    # This validates that no physical cannot-link ended up inside the current
    # applied identity partition.
    hard_pairs = build_hard_identity_pairs(
        tracklet_assignment_rows=tracklet_assignments,
        hard_conflicts=hard_conflicts,
    )

    state = build_global_trajectory_round_state(
        identity_rows=identities,
        tracklet_assignment_rows=tracklet_assignments,
        chain_profile_rows=chain_profiles,
        tracklet_family_rows=tracklet_family,
        profile_config=IdentityExpansionConfig(),
        family_config=KitEntityConsensusConfig(),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "global_trajectory_round_state"
        / "round-02-v1"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # A round-state directory intentionally mirrors the partition artifacts so
    # all later 7G runners can point at one self-contained identity directory.
    outputs = {
        "final_identities": output_dir / "final_identities.jsonl",
        "final_actor_assignments": (
            output_dir / "final_actor_assignments.jsonl"
        ),
        "final_tracklet_assignments": (
            output_dir / "final_tracklet_assignments.jsonl"
        ),
        "profiles": output_dir / "profiles.jsonl",
        "final_family_assignments": (
            output_dir / "final_family_assignments.jsonl"
        ),
    }

    _write_jsonl(outputs["final_identities"], identities)
    _write_jsonl(
        outputs["final_actor_assignments"],
        actor_assignments,
    )
    _write_jsonl(
        outputs["final_tracklet_assignments"],
        tracklet_assignments,
    )
    _write_jsonl(
        outputs["profiles"],
        (row.to_dict() for row in state.profiles),
    )
    _write_jsonl(
        outputs["final_family_assignments"],
        (row.to_dict() for row in state.family_rows),
    )

    summary = state.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "hard_identity_pair_count": len(hard_pairs),
        "source_identity_dir": str(source_dir),
        "source_chain_profiles": str(chain_profile_path),
        "source_tracklet_family_consensus": str(
            tracklet_family_path
        ),
        "source_hard_conflicts": str(hard_conflicts_path),
        "outputs": {
            key: str(value) for key, value in outputs.items()
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
            "round_state_version": (
                GLOBAL_TRAJECTORY_ROUND_STATE_VERSION
            ),
            "summary": summary_path.name,
            "final_identities": outputs["final_identities"].name,
            "final_actor_assignments": (
                outputs["final_actor_assignments"].name
            ),
            "final_tracklet_assignments": (
                outputs["final_tracklet_assignments"].name
            ),
            "profiles": outputs["profiles"].name,
            "final_family_assignments": (
                outputs["final_family_assignments"].name
            ),
            "identity_count": state.identity_count,
            "identity_decisions_changed": False,
        },
    )

    print(
        "[global-trajectory-round-state-7G.6] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[global-trajectory-round-state-7G.6] "
        f"identities={summary['identity_count']} "
        f"profiles={summary['profiled_identity_count']} "
        f"unprofiled={summary['unprofiled_identity_count']} "
        f"tracklets={summary['tracklet_count']}"
    )
    print(
        "[global-trajectory-round-state-7G.6] "
        f"family_usable={summary['family_usable_count']} "
        "family_states="
        + json.dumps(summary["family_state_counts"], sort_keys=True)
    )
    print(
        "[global-trajectory-round-state-7G.6] "
        f"hard_identity_pairs={len(hard_pairs)}"
    )
    print(
        "[global-trajectory-round-state-7G.6] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
