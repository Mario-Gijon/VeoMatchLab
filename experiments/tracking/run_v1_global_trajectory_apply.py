#!/usr/bin/env python3
"""Apply visually audited Tracking V1 Hito 7G.4 trajectory proposals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.global_trajectory_apply import (  # noqa: E402
    GLOBAL_TRAJECTORY_APPLY_VERSION,
    apply_global_trajectory_proposals,
)
from tracking.actors.identity_expansion import (  # noqa: E402
    build_hard_identity_pairs,
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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
    return payload


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
    parser.add_argument("--identity-dir")
    parser.add_argument("--solver-dir")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--accept-audited-proposals",
        action="store_true",
        help=(
            "Required explicit acknowledgement that the 7G.4A visual "
            "audit has been reviewed and these proposals are accepted."
        ),
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if not args.accept_audited_proposals:
        raise ValueError(
            "--accept-audited-proposals is required because this stage "
            "changes the identity partition"
        )

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
    solver_dir = (
        Path(args.solver_dir)
        if args.solver_dir
        else root / "diagnostics"
        / "global_trajectory_solver"
        / "global-trajectory-proposal-solver-v2-centered-geometry"
    )

    identities_path = identity_dir / "final_identities.jsonl"
    actor_assignments_path = (
        identity_dir / "final_actor_assignments.jsonl"
    )
    tracklet_assignments_path = (
        identity_dir / "final_tracklet_assignments.jsonl"
    )
    proposals_path = solver_dir / "selected_proposals.jsonl"
    audit_summary_path = (
        solver_dir / "audit"
        / "global-trajectory-proposal-audit-v1"
        / "summary.json"
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
        proposals_path,
        audit_summary_path,
        hard_conflicts_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    identities = _iter_jsonl(identities_path)
    actor_assignments = _iter_jsonl(actor_assignments_path)
    tracklet_assignments = _iter_jsonl(tracklet_assignments_path)
    proposals = _iter_jsonl(proposals_path)
    hard_conflicts = _iter_jsonl(hard_conflicts_path)
    audit = _load_json(audit_summary_path)

    if int(audit.get("audited_proposal_count", -1)) != len(proposals):
        raise ValueError(
            "audit proposal count does not match selected proposals"
        )
    if int(audit.get("missing_first_visual_count", -1)) != 0:
        raise ValueError("audit has missing first-identity visuals")
    if int(audit.get("missing_second_visual_count", -1)) != 0:
        raise ValueError("audit has missing second-identity visuals")
    if bool(audit.get("identity_decisions_changed", True)):
        raise ValueError(
            "audit artifact unexpectedly changed identity decisions"
        )

    hard_pairs = build_hard_identity_pairs(
        tracklet_assignment_rows=tracklet_assignments,
        hard_conflicts=hard_conflicts,
    )

    result = apply_global_trajectory_proposals(
        identity_rows=identities,
        actor_assignment_rows=actor_assignments,
        tracklet_assignment_rows=tracklet_assignments,
        proposal_rows=proposals,
        hard_identity_pairs=hard_pairs,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "global_trajectory_application"
        / GLOBAL_TRAJECTORY_APPLY_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "final_identities": output_dir / "final_identities.jsonl",
        "source_identity_assignments": (
            output_dir / "source_identity_assignments.jsonl"
        ),
        "final_actor_assignments": (
            output_dir / "final_actor_assignments.jsonl"
        ),
        "final_tracklet_assignments": (
            output_dir / "final_tracklet_assignments.jsonl"
        ),
        "lineage": output_dir / "lineage.jsonl",
        "accepted_proposals": (
            output_dir / "accepted_proposals.jsonl"
        ),
    }

    _write_jsonl(outputs["final_identities"], result.identities)
    _write_jsonl(
        outputs["source_identity_assignments"],
        result.source_identity_assignments,
    )
    _write_jsonl(
        outputs["final_actor_assignments"],
        result.actor_assignments,
    )
    _write_jsonl(
        outputs["final_tracklet_assignments"],
        result.tracklet_assignments,
    )
    _write_jsonl(outputs["lineage"], result.lineage)
    _write_jsonl(
        outputs["accepted_proposals"],
        result.accepted_proposals,
    )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "hard_identity_pair_count": len(hard_pairs),
        "visual_audit_declared_approved": True,
        "source_visual_audit": str(audit_summary_path),
        "source_identities": str(identities_path),
        "source_actor_assignments": str(
            actor_assignments_path
        ),
        "source_tracklet_assignments": str(
            tracklet_assignments_path
        ),
        "source_selected_proposals": str(proposals_path),
        "outputs": {
            key: str(value) for key, value in outputs.items()
        },
    })
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
            "summary": summary_path.name,
            "final_identities": outputs["final_identities"].name,
            "final_actor_assignments": (
                outputs["final_actor_assignments"].name
            ),
            "final_tracklet_assignments": (
                outputs["final_tracklet_assignments"].name
            ),
            "lineage": outputs["lineage"].name,
            "accepted_proposals": (
                outputs["accepted_proposals"].name
            ),
            "output_identity_count": len(result.identities),
            "identity_partition_changed": True,
            "provisional": True,
            "canonical": False,
        },
    )

    print(
        "[global-trajectory-apply-7G.5] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[global-trajectory-apply-7G.5] "
        f"source_identities={result.source_identity_count} "
        f"accepted={len(result.accepted_proposals)} "
        f"output_identities={len(result.identities)}"
    )
    print(
        "[global-trajectory-apply-7G.5] types="
        + json.dumps(summary["accepted_type_counts"], sort_keys=True)
    )
    print(
        "[global-trajectory-apply-7G.5] families="
        + json.dumps(
            summary["accepted_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[global-trajectory-apply-7G.5] conservation="
        f"actors:{summary['source_actor_count']}->"
        f"{summary['output_actor_assignment_count']} "
        f"tracklets:{summary['source_tracklet_count']}->"
        f"{summary['output_tracklet_assignment_count']}"
    )
    print(
        "[global-trajectory-apply-7G.5] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
