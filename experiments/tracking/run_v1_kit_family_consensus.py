#!/usr/bin/env python3
"""Run Hito 7C.1E whole-tracklet kit-family consensus."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.appearance.kit_family_consensus import (  # noqa: E402
    KIT_FAMILY_CONSENSUS_VERSION,
    KitFamilyConsensusConfig,
    build_kit_family_consensus,
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
        )
        + "\n",
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
                )
                + "\n"
            )
            count += 1
    return count


def _fit_panel(image, width=150, height=195):
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _sheet(items, columns=8, panel_width=150, panel_height=238):
    if not items:
        return None
    rows = (len(items) + columns - 1) // columns
    result = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 43
    for index, item in enumerate(items):
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:image_height] = _fit_panel(
            item["image"], panel_width, image_height
        )
        cv2.putText(
            panel,
            item["tracklet_id"],
            (3, image_height + 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"dom={item['dominance']:.2f} "
                f"known={item['known_fraction']:.2f}"
            ),
            (3, image_height + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"sup={item['support_samples']} "
                f"conf={item['conflicting_samples']} "
                f"unk={item['unknown_samples']}"
            ),
            (3, image_height + 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.27,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        row = index // columns
        col = index % columns
        result[
            row * panel_height:(row + 1) * panel_height,
            col * panel_width:(col + 1) * panel_width,
        ] = panel
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--sheet-limit", type=int, default=48)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    source_root = (
        root / "diagnostics" / "kit_families" / "kit-family-v1"
    )
    samples_path = source_root / "samples.jsonl"
    profiles_path = source_root / "profiles.jsonl"
    assignments_path = source_root / "assignments.jsonl"
    clusters_path = source_root / "clusters.jsonl"

    for path in (
        samples_path,
        profiles_path,
        assignments_path,
        clusters_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    samples = _iter_jsonl(samples_path)
    profiles = _iter_jsonl(profiles_path)
    profile_assignments = _iter_jsonl(assignments_path)
    clusters = _iter_jsonl(clusters_path)

    result = build_kit_family_consensus(
        sample_rows=samples,
        profile_rows=profiles,
        profile_assignment_rows=profile_assignments,
        cluster_rows=clusters,
        config=KitFamilyConsensusConfig(),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "kit_family_consensus"
        / KIT_FAMILY_CONSENSUS_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_jsonl(
        output_dir / "sample_evidence.jsonl",
        (row.to_dict() for row in result.sample_evidence),
    )
    _write_jsonl(
        output_dir / "assignments.jsonl",
        (row.to_dict() for row in result.tracklets),
    )
    _write_jsonl(
        output_dir / "calibrations.jsonl",
        (row.to_dict() for row in result.calibrations),
    )

    sample_rows_by_tracklet = defaultdict(list)
    for row in samples:
        sample_rows_by_tracklet[str(row["tracklet_id"])].append(row)

    evidence_by_tracklet = defaultdict(list)
    for row in result.sample_evidence:
        evidence_by_tracklet[row.tracklet_id].append(row)

    # Representative crop = strongest sample that agrees with final family.
    best_crop = {}
    for consensus in result.tracklets:
        rows = evidence_by_tracklet[consensus.tracklet_id]
        agreeing = [
            row for row in rows
            if (
                consensus.family_key is not None
                and row.winner_family_key == consensus.family_key
            )
        ]
        candidates = agreeing if agreeing else rows
        if not candidates:
            continue
        chosen = max(
            candidates,
            key=lambda row: (
                row.vote_weight,
                row.quality,
                row.best_cosine,
            ),
        )
        best_crop[consensus.tracklet_id] = chosen.crop_ref

    grouped = defaultdict(list)
    for consensus in result.tracklets:
        key = consensus.family_key or "unknown"
        grouped[key].append(consensus)

    contact_sheets = {}
    for key, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                -row.usable_for_tracking,
                -row.dominance,
                -row.known_fraction,
                -row.support_samples,
                row.tracklet_id,
            ),
        )
        items = []
        for consensus in ranked:
            crop_ref = best_crop.get(consensus.tracklet_id)
            if crop_ref is None:
                continue
            image = cv2.imread(
                str(root / crop_ref),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue
            items.append({
                "tracklet_id": consensus.tracklet_id,
                "dominance": consensus.dominance,
                "known_fraction": consensus.known_fraction,
                "support_samples": consensus.support_samples,
                "conflicting_samples": consensus.conflicting_samples,
                "unknown_samples": consensus.unknown_samples,
                "image": image,
            })
            if len(items) >= max(1, args.sheet_limit):
                break

        sheet = _sheet(items)
        if sheet is None:
            contact_sheets[key] = None
            continue

        safe_key = key.replace("-", "_")
        path = output_dir / f"{safe_key}.jpg"
        if not cv2.imwrite(
            str(path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"cannot write {path}")
        contact_sheets[key] = str(path)

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_samples": str(samples_path),
        "source_profiles": str(profiles_path),
        "source_profile_assignments": str(assignments_path),
        "source_clusters": str(clusters_path),
        "contact_sheets": contact_sheets,
    })
    _write_json(output_dir / "summary.json", summary)

    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "consensus_version": KIT_FAMILY_CONSENSUS_VERSION,
            "sample_evidence": "sample_evidence.jsonl",
            "assignments": "assignments.jsonl",
            "calibrations": "calibrations.jsonl",
            "summary": "summary.json",
            "tracklet_count": len(result.tracklets),
            "usable_for_tracking_count": (
                summary["usable_for_tracking_count"]
            ),
            "identity_decisions_changed": False,
            "identity_hard_constraints_created": False,
        },
    )

    print(
        "[kit-family-consensus-7C.1E] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[kit-family-consensus-7C.1E] "
        f"samples={len(result.sample_evidence)} "
        f"tracklets={len(result.tracklets)} "
        f"usable={summary['usable_for_tracking_count']}"
    )
    print(
        "[kit-family-consensus-7C.1E] states="
        + json.dumps(summary["state_counts"], sort_keys=True)
    )
    print(
        "[kit-family-consensus-7C.1E] families="
        + json.dumps(summary["family_counts"], sort_keys=True)
    )
    print(
        "[kit-family-consensus-7C.1E] calibrations="
        + json.dumps(summary["calibrations"], sort_keys=True)
    )
    print(
        "[kit-family-consensus-7C.1E] dominance="
        + json.dumps(summary["dominance"], sort_keys=True)
    )
    print(
        "[kit-family-consensus-7C.1E] known_fraction="
        + json.dumps(summary["known_fraction"], sort_keys=True)
    )
    for key, path in contact_sheets.items():
        if path:
            print(
                "[kit-family-consensus-7C.1E] "
                f"{key}_sheet={path}"
            )
    print(
        "[kit-family-consensus-7C.1E] "
        f"artifact={output_dir / 'index.json'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
