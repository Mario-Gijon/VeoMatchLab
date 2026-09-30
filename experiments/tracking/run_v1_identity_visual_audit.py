#!/usr/bin/env python3
"""Run Tracking V1 Hito 3D V2: backbone-partition visual identity audit.

Generates:
- top-down identity video in field coordinates;
- transition diagnostics between consecutive tracklets of each identity;
- summary and review queue.

This remains diagnostic/provisional and does not complete
TrackingStage.IDENTITIES.
"""

from __future__ import annotations

import argparse
import json
import os
import uuid
from pathlib import Path
from typing import Any, Iterable, Mapping

from tracking.artifacts import TrackingArtifactStore
from tracking.identities.visual_audit import (
    VISUAL_AUDIT_VERSION,
    VisualAuditConfig,
    build_transition_records,
    build_visual_summary,
    iter_jsonl,
    load_identities,
    load_identity_audits,
    load_scored_links,
    load_tracklets,
    render_topdown_video,
)


SOLVER_VERSION = "global-backbone-v1"
SCORING_VERSION = "soft-link-v1"
IDENTITY_AUDIT_VERSION = "identity-backbone-audit-v1"


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        with temp.open("wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def _write_json(path: Path, payload: Any) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    count = 0
    try:
        with temp.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(
                    json.dumps(
                        dict(row),
                        ensure_ascii=False,
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
                count += 1
            destination.flush()
            os.fsync(destination.fileno())
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)
    return count


def run_visual_identity_audit(
    *,
    identities_path: Path,
    tracklets_path: Path,
    identity_audits_path: Path,
    scored_links_path: Path,
    output_dir: Path,
    config: VisualAuditConfig,
    render_video: bool = True,
) -> dict[str, Any]:
    identities = load_identities(
        iter_jsonl(identities_path)
    )
    tracklets = load_tracklets(
        iter_jsonl(tracklets_path)
    )
    audits = load_identity_audits(
        iter_jsonl(identity_audits_path)
    )
    scored_links = load_scored_links(
        iter_jsonl(scored_links_path)
    )

    assigned = {
        tracklet_id
        for identity in identities
        for tracklet_id in identity.tracklet_ids
    }
    if assigned != set(tracklets):
        raise ValueError(
            "identity partition and tracklet set differ"
        )

    transitions = build_transition_records(
        identities,
        tracklets,
        audits,
        scored_links,
    )
    summary = build_visual_summary(
        identities,
        tracklets,
        audits,
        transitions,
        config,
    )

    transitions_path = output_dir / "transitions.jsonl"
    review_path = output_dir / "review_queue.json"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"
    video_path = output_dir / "identity_topdown.mp4"

    transition_count = _write_jsonl(
        transitions_path,
        transitions,
    )

    review_rows = sorted(
        transitions,
        key=lambda row: (
            len(row["review_flags"]),
            float(row["gap_s"]),
            float(row["endpoint_distance_m"]),
        ),
        reverse=True,
    )
    _write_json(
        review_path,
        {
            "schema_version": 1,
            "visual_audit_version": VISUAL_AUDIT_VERSION,
            "provisional": True,
            "flagged_transitions": [
                row
                for row in review_rows
                if row["review_flags"]
            ][:50],
            "flagged_identities": [
                {
                    "identity_id": identity_id,
                    "flags": audit.get("flags", []),
                    "timeline": audit.get("timeline"),
                    "backbone": audit.get("backbone"),
                }
                for identity_id, audit in sorted(audits.items())
                if audit.get("flags")
            ],
        },
    )

    video_metadata = None
    if render_video:
        video_metadata = render_topdown_video(
            output_path=video_path,
            identities=identities,
            tracklets=tracklets,
            audits=audits,
            config=config,
        )

    summary["video"] = (
        {
            "path": video_path.name,
            **video_metadata,
        }
        if video_metadata is not None
        else None
    )
    summary["source_identities"] = str(identities_path)
    summary["source_tracklets"] = str(tracklets_path)
    summary["source_identity_audits"] = str(
        identity_audits_path
    )
    summary["source_scored_links"] = str(
        scored_links_path
    )
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "visual_audit_version": VISUAL_AUDIT_VERSION,
            "provisional": True,
            "identity_stage_completed": False,
            "video": (
                video_path.name if render_video else None
            ),
            "transitions": transitions_path.name,
            "review_queue": review_path.name,
            "summary": summary_path.name,
            "transition_count": transition_count,
        },
    )
    return summary


def _load_dev_clip(path: Path) -> tuple[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["match_id"]), str(payload["clip_id"])


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="experiments/tracking/dev_clip.json",
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--identities")
    parser.add_argument("--tracklets")
    parser.add_argument("--identity-audits")
    parser.add_argument("--scored-links")
    parser.add_argument("--output-dir")
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument(
        "--trail-seconds",
        type=float,
        default=1.5,
    )
    parser.add_argument(
        "--show-tracklet-label",
        action="store_true",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="write diagnostics without rendering the MP4",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    match_id = args.match_id
    clip_id = args.clip_id
    if not match_id or not clip_id:
        cfg_match, cfg_clip = _load_dev_clip(
            Path(args.config)
        )
        match_id = match_id or cfg_match
        clip_id = clip_id or cfg_clip

    store = TrackingArtifactStore.for_clip(
        args.data_root,
        match_id,
        clip_id,
    )

    identities_path = (
        Path(args.identities)
        if args.identities
        else store.root
        / "diagnostics"
        / "identity_solver"
        / SOLVER_VERSION
        / "identities.jsonl"
    )
    tracklets_path = (
        Path(args.tracklets)
        if args.tracklets
        else store.root
        / "tracklets"
        / "tracklets.jsonl"
    )
    identity_audits_path = (
        Path(args.identity_audits)
        if args.identity_audits
        else store.root
        / "diagnostics"
        / "identity_diagnostics"
        / IDENTITY_AUDIT_VERSION
        / "identity_audits.jsonl"
    )
    scored_links_path = (
        Path(args.scored_links)
        if args.scored_links
        else store.root
        / "diagnostics"
        / "identity_link_scoring"
        / SCORING_VERSION
        / "scored_links.jsonl"
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else store.root
        / "diagnostics"
        / "identity_visual_audit"
        / VISUAL_AUDIT_VERSION
    )

    config = VisualAuditConfig(
        output_fps=args.fps,
        trail_seconds=args.trail_seconds,
        show_tracklet_label=args.show_tracklet_label,
    )

    print(
        "[identity-visual-audit] starting "
        f"match={match_id} clip={clip_id}"
    )
    summary = run_visual_identity_audit(
        identities_path=identities_path,
        tracklets_path=tracklets_path,
        identity_audits_path=identity_audits_path,
        scored_links_path=scored_links_path,
        output_dir=output_dir,
        config=config,
        render_video=not args.no_video,
    )
    print(
        "[identity-visual-audit] completed "
        f"identities={summary['identity_count']} "
        f"tracklets={summary['tracklet_count']} "
        f"transitions={summary['transition_count']}"
    )
    print(
        "[identity-visual-audit] review_flags="
        + json.dumps(
            summary["transition_review_flag_counts"],
            sort_keys=True,
        )
    )
    print(
        "[identity-visual-audit] artifact="
        f"{output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
