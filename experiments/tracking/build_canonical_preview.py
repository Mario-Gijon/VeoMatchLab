#!/usr/bin/env python3
"""Build an isolated V8 canonical tracking preview.

Inputs
------
* Raw tracker frames JSONL.
* ``match_participants_v8_report.json`` produced by the V8 audit.
* Optional manually validated duplicate evidence, e.g.:
      --duplicate 14:0=11:1

The duplicate syntax means:
    source segment 14:0 is a second observation source for the canonical
    identity that already owns target segment 11:1.

Duplicate evidence is intentionally explicit/audit-only for now.  The current
automatic duplicate score produced false positives on football players moving
together, so this builder does NOT auto-merge duplicate hypotheses.

Outputs
-------
* JSONL preview frames using stable canonical IDs under ``tracks.person``.
* JSON summary with coverage/source-selection diagnostics.

Nothing modifies production ``tracks.person`` or the current app preview.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.final_publication import (  # noqa: E402
    CanonicalIdentitySpec,
    DuplicateAttachment,
    SegmentWindow,
    build_publication_plan,
    publish_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--participants-report",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--duplicate",
        action="append",
        default=[],
        metavar="SOURCE=TARGET",
        help=(
            "Validated duplicate evidence, e.g. 14:0=11:1. "
            "May be passed multiple times. Never inferred automatically."
        ),
    )
    parser.add_argument(
        "--output-frames",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug_margin15/"
            "preview_v8_frames.jsonl"
        ),
    )
    parser.add_argument(
        "--output-summary",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug_margin15/"
            "preview_v8_summary.json"
        ),
    )
    parser.add_argument(
        "--expected-canonical-count",
        type=int,
        default=23,
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )


def parse_duplicate(
    value: str,
) -> DuplicateAttachment:
    if "=" not in value:
        raise ValueError(
            f"invalid --duplicate {value!r}; expected SOURCE=TARGET"
        )
    source, target = (
        item.strip()
        for item in value.split(
            "=",
            1,
        )
    )
    if not source or not target:
        raise ValueError(
            f"invalid --duplicate {value!r}; expected SOURCE=TARGET"
        )
    return DuplicateAttachment(
        source_segment_key=source,
        target_segment_key=target,
        reason="manual_visual_validation",
    )


def specs_from_report(
    report: dict[str, object],
) -> tuple[
    tuple[CanonicalIdentitySpec, ...],
    dict[str, SegmentWindow],
]:
    raw_segments = report.get(
        "segments",
        {},
    )
    if not isinstance(raw_segments, dict):
        raise ValueError(
            "participants report has no segments object"
        )

    segments: dict[str, SegmentWindow] = {}
    for payload in raw_segments.values():
        if not isinstance(payload, dict):
            continue

        key = str(payload["key"])
        segments[key] = SegmentWindow(
            key=key,
            raw_track_id=int(
                payload["raw_track_id"]
            ),
            start_frame=int(
                payload["start_frame"]
            ),
            end_frame=int(
                payload["end_frame"]
            ),
            identity_quality=str(
                payload.get(
                    "identity_quality",
                    "clean",
                )
            ),
            identity_quality_score=float(
                payload.get(
                    "identity_quality_score",
                    1.0,
                )
            ),
        )

    result = report.get(
        "physical_identity_result",
    )
    if not isinstance(result, dict):
        raise ValueError(
            "participants report has no physical_identity_result"
        )

    raw_identities = result.get(
        "identities",
        [],
    )
    if not isinstance(raw_identities, list):
        raise ValueError(
            "physical_identity_result.identities must be a list"
        )

    final_roles = report.get(
        "final_roles",
        {},
    )
    if not isinstance(final_roles, dict):
        final_roles = {}

    identities: list[
        CanonicalIdentitySpec
    ] = []

    for payload in raw_identities:
        if not isinstance(payload, dict):
            continue

        canonical_id = int(
            payload["identity_id"]
        )

        identities.append(
            CanonicalIdentitySpec(
                canonical_id=canonical_id,
                role=str(
                    final_roles.get(
                        str(canonical_id),
                        "player",
                    )
                ),
                team_label=str(
                    payload.get(
                        "team_label",
                        "unknown",
                    )
                ),
                primary_segment_keys=tuple(
                    str(key)
                    for key in payload.get(
                        "member_keys",
                        [],
                    )
                ),
            )
        )

    return (
        tuple(
            sorted(
                identities,
                key=lambda identity: identity.canonical_id,
            )
        ),
        segments,
    )


def iter_raw_frames(
    path: Path,
):
    with path.open(
        "r",
        encoding="utf-8",
    ) as stream:
        for line in stream:
            if not line.strip():
                continue
            yield json.loads(line)


def main() -> None:
    args = parse_args()

    report = load_json(
        args.participants_report
    )
    identities, segments = specs_from_report(
        report
    )

    if (
        len(identities)
        != args.expected_canonical_count
    ):
        raise RuntimeError(
            f"participants report contains {len(identities)} physical "
            f"identities; expected {args.expected_canonical_count}"
        )

    duplicates = tuple(
        parse_duplicate(value)
        for value in args.duplicate
    )

    plan = build_publication_plan(
        identities=identities,
        segments=segments,
        duplicate_attachments=duplicates,
    )

    args.output_frames.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    args.output_summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames_written = 0
    frames_with_people = 0
    max_people = 0
    total_people_samples = 0

    source_kind_counts: Counter[str] = Counter()
    position_kind_counts: Counter[str] = Counter()
    canonical_sample_counts: Counter[int] = Counter()
    role_sample_counts: Counter[str] = Counter()
    source_segment_counts: Counter[str] = Counter()

    first_frame_by_canonical: dict[
        int,
        int,
    ] = {}
    last_frame_by_canonical: dict[
        int,
        int,
    ] = {}

    with args.output_frames.open(
        "w",
        encoding="utf-8",
    ) as output:
        for raw_frame in iter_raw_frames(
            args.tracking_frames
        ):
            published = publish_frame(
                raw_frame,
                plan=plan,
                max_canonical_people=args.expected_canonical_count,
            )

            row = published.to_dict()
            output.write(
                json.dumps(
                    row,
                    separators=(",", ":"),
                )
                + "\n"
            )

            frames_written += 1

            count = len(
                published.people
            )
            if count > 0:
                frames_with_people += 1

            total_people_samples += count
            max_people = max(
                max_people,
                count,
            )

            for person in published.people:
                source_kind_counts[
                    person.source_kind
                ] += 1
                position_kind_counts[
                    person.position_kind.value
                ] += 1
                canonical_sample_counts[
                    person.canonical_id
                ] += 1
                role_sample_counts[
                    person.role
                ] += 1
                source_segment_counts[
                    person.source_segment_key
                ] += 1

                first_frame_by_canonical.setdefault(
                    person.canonical_id,
                    published.frame_index,
                )
                last_frame_by_canonical[
                    person.canonical_id
                ] = published.frame_index

    identities_summary = []

    spec_by_id = {
        identity.canonical_id: identity
        for identity in identities
    }

    for canonical_id in sorted(
        spec_by_id
    ):
        spec = spec_by_id[
            canonical_id
        ]
        identities_summary.append(
            {
                "canonical_id": canonical_id,
                "role": spec.role,
                "team_label": spec.team_label,
                "primary_segment_keys": list(
                    spec.primary_segment_keys
                ),
                "published_samples": int(
                    canonical_sample_counts[
                        canonical_id
                    ]
                ),
                "first_published_frame": (
                    first_frame_by_canonical.get(
                        canonical_id
                    )
                ),
                "last_published_frame": (
                    last_frame_by_canonical.get(
                        canonical_id
                    )
                ),
            }
        )

    summary = {
        "version": "canonical-preview-v8-publication-v1",
        "source_tracking_frames": str(
            args.tracking_frames
        ),
        "participants_report": str(
            args.participants_report
        ),
        "output_frames": str(
            args.output_frames
        ),
        "canonical_identity_count": len(
            identities
        ),
        "duplicate_attachments": [
            {
                "source_segment_key": row.source_segment_key,
                "target_segment_key": row.target_segment_key,
                "reason": row.reason,
            }
            for row in duplicates
        ],
        "frames_written": frames_written,
        "frames_with_people": frames_with_people,
        "max_people_in_frame": max_people,
        "total_published_people_samples": (
            total_people_samples
        ),
        "source_kind_counts": dict(
            source_kind_counts
        ),
        "position_kind_counts": dict(
            position_kind_counts
        ),
        "role_sample_counts": dict(
            role_sample_counts
        ),
        "source_segment_counts": dict(
            source_segment_counts
        ),
        "identities": identities_summary,
        "interpolation": {
            "performed_here": False,
            "endpoint_fill": False,
            "long_gap_fill": False,
            "note": (
                "Publisher preserves raw tracker state only. "
                "No new positions are invented."
            ),
        },
    }

    args.output_summary.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("V8 FINAL CANONICAL PREVIEW PUBLICATION")
    print(
        f"canonical identities      : {len(identities)}"
    )
    print(
        f"validated duplicates      : {len(duplicates)}"
    )
    print(
        f"frames written            : {frames_written}"
    )
    print(
        f"frames with people        : {frames_with_people}"
    )
    print(
        f"max people in one frame   : {max_people}"
    )
    print(
        f"published person samples  : {total_people_samples}"
    )
    print(
        "source kinds              : "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                source_kind_counts.items()
            )
        )
    )
    print(
        "position kinds            : "
        + ", ".join(
            f"{key}={value}"
            for key, value in sorted(
                position_kind_counts.items()
            )
        )
    )
    print(
        f"frames : {args.output_frames.resolve()}"
    )
    print(
        f"summary: {args.output_summary.resolve()}"
    )
    print(
        "No production tracking or frontend data was modified."
    )


if __name__ == "__main__":
    main()
