#!/usr/bin/env python3
"""Run Hito 7C.1F kit-family aggregation for actors and identities."""

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


from tracking.appearance.kit_entity_consensus import (  # noqa: E402
    KIT_ENTITY_CONSENSUS_VERSION,
    KitEntityConsensusConfig,
    aggregate_entity_family_consensus,
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


def _sheet(items, columns=8, panel_width=150, panel_height=244):
    if not items:
        return None

    rows = (len(items) + columns - 1) // columns
    result = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 49

    for index, item in enumerate(items):
        panel = np.zeros((panel_height, panel_width, 3), dtype=np.uint8)
        panel[:image_height] = _fit_panel(
            item["image"],
            panel_width,
            image_height,
        )
        cv2.putText(
            panel,
            item["entity_id"],
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
                f"conf={item['confidence']:.2f} "
                f"dom={item['dominance']:.2f}"
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
                f"cov={item['evidence_coverage']:.2f} "
                f"trk={item['support_tracklets']}/"
                f"{item['total_tracklets']}"
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


def _best_sample_by_entity(
    *,
    entities,
    sample_evidence_rows: list[dict[str, Any]],
    root: Path,
):
    samples_by_tracklet = defaultdict(list)
    for row in sample_evidence_rows:
        samples_by_tracklet[str(row["tracklet_id"])].append(row)

    result = {}
    for entity in entities:
        target_key = (
            entity.family_key
            or entity.candidate_family_key
        )
        candidates = []
        for tracklet_id in entity.member_tracklet_ids:
            for sample in samples_by_tracklet.get(tracklet_id, []):
                winner = sample.get("winner_family_key")
                match = int(
                    target_key is not None
                    and winner == target_key
                )
                candidates.append(
                    (
                        match,
                        float(sample.get("vote_weight", 0.0)),
                        float(sample.get("quality", 0.0)),
                        float(sample.get("best_cosine", 0.0)),
                        str(sample.get("crop_ref", "")),
                    )
                )

        if not candidates:
            continue

        candidates.sort(reverse=True)
        crop_ref = candidates[0][-1]
        if not crop_ref:
            continue
        image = cv2.imread(
            str(root / crop_ref),
            cv2.IMREAD_COLOR,
        )
        if image is None:
            continue
        result[entity.entity_id] = image

    return result


def _write_entity_sheets(
    *,
    prefix: str,
    result,
    sample_evidence_rows,
    root: Path,
    output_dir: Path,
    limit: int,
):
    images = _best_sample_by_entity(
        entities=result.entities,
        sample_evidence_rows=sample_evidence_rows,
        root=root,
    )

    grouped = defaultdict(list)
    for entity in result.entities:
        key = entity.family_key or "unknown"
        grouped[key].append(entity)

    paths = {}
    for key, rows in sorted(grouped.items()):
        ranked = sorted(
            rows,
            key=lambda row: (
                -row.usable_for_tracking,
                -row.confidence,
                -row.dominance,
                -row.evidence_coverage,
                row.entity_id,
            ),
        )

        items = []
        for entity in ranked:
            image = images.get(entity.entity_id)
            if image is None:
                continue
            items.append({
                "entity_id": entity.entity_id,
                "confidence": entity.confidence,
                "dominance": entity.dominance,
                "evidence_coverage": entity.evidence_coverage,
                "support_tracklets": entity.support_tracklets,
                "total_tracklets": entity.total_tracklets,
                "image": image,
            })
            if len(items) >= max(1, limit):
                break

        sheet = _sheet(items)
        if sheet is None:
            paths[key] = None
            continue

        safe_key = key.replace("-", "_")
        path = output_dir / f"{prefix}_{safe_key}.jpg"
        if not cv2.imwrite(
            str(path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"cannot write {path}")
        paths[key] = str(path)

    return paths


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--identity-dir",
        help=(
            "Identity-expansion directory. Defaults to the current "
            "score060 diagnostic."
        ),
    )
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

    consensus_root = (
        root / "diagnostics" / "kit_family_consensus"
        / "kit-family-consensus-v1"
    )
    tracklet_consensus_path = (
        consensus_root / "assignments.jsonl"
    )
    sample_evidence_path = (
        consensus_root / "sample_evidence.jsonl"
    )

    actor_mapping_path = (
        root / "diagnostics" / "actor_temporal_backbone"
        / "temporal-actor-backbone-v1" / "assignments.jsonl"
    )

    identity_dir = (
        Path(args.identity_dir)
        if args.identity_dir
        else root / "diagnostics" / "identity_expansion"
        / "seeded-identity-expansion-v1-score060"
    )
    identity_mapping_path = (
        identity_dir / "tracklet_assignments.jsonl"
    )

    for path in (
        tracklet_consensus_path,
        sample_evidence_path,
        actor_mapping_path,
        identity_mapping_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    tracklet_consensus = _iter_jsonl(tracklet_consensus_path)
    sample_evidence = _iter_jsonl(sample_evidence_path)
    actor_mapping = _iter_jsonl(actor_mapping_path)
    identity_mapping = _iter_jsonl(identity_mapping_path)

    config = KitEntityConsensusConfig()

    actor_result = aggregate_entity_family_consensus(
        tracklet_consensus_rows=tracklet_consensus,
        tracklet_entity_rows=actor_mapping,
        entity_id_field="actor_id",
        entity_level="actor",
        config=config,
    )
    identity_result = aggregate_entity_family_consensus(
        tracklet_consensus_rows=tracklet_consensus,
        tracklet_entity_rows=identity_mapping,
        entity_id_field="identity_id",
        entity_level="identity",
        config=config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "kit_entity_consensus"
        / KIT_ENTITY_CONSENSUS_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    actor_assignments_path = output_dir / "actor_assignments.jsonl"
    identity_assignments_path = (
        output_dir / "identity_assignments.jsonl"
    )

    _write_jsonl(
        actor_assignments_path,
        (row.to_dict() for row in actor_result.entities),
    )
    _write_jsonl(
        identity_assignments_path,
        (row.to_dict() for row in identity_result.entities),
    )

    actor_summary = actor_result.summary()
    identity_summary = identity_result.summary()

    actor_sheets = _write_entity_sheets(
        prefix="actor",
        result=actor_result,
        sample_evidence_rows=sample_evidence,
        root=root,
        output_dir=output_dir,
        limit=args.sheet_limit,
    )
    identity_sheets = _write_entity_sheets(
        prefix="identity",
        result=identity_result,
        sample_evidence_rows=sample_evidence,
        root=root,
        output_dir=output_dir,
        limit=args.sheet_limit,
    )

    summary = {
        "schema_version": 1,
        "entity_consensus_version": KIT_ENTITY_CONSENSUS_VERSION,
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "actor": actor_summary,
        "identity": identity_summary,
        "actor_contact_sheets": actor_sheets,
        "identity_contact_sheets": identity_sheets,
        "source_tracklet_consensus": str(tracklet_consensus_path),
        "source_sample_evidence": str(sample_evidence_path),
        "source_actor_mapping": str(actor_mapping_path),
        "source_identity_mapping": str(identity_mapping_path),
        "identity_source_dir": str(identity_dir),
        "identity_decisions_changed": False,
        "identity_hard_constraints_created": False,
    }
    _write_json(output_dir / "summary.json", summary)

    _write_json(
        output_dir / "index.json",
        {
            "schema_version": 1,
            "entity_consensus_version": KIT_ENTITY_CONSENSUS_VERSION,
            "actor_assignments": actor_assignments_path.name,
            "identity_assignments": identity_assignments_path.name,
            "summary": "summary.json",
            "actor_count": len(actor_result.entities),
            "identity_count": len(identity_result.entities),
            "identity_decisions_changed": False,
        },
    )

    print(
        "[kit-entity-consensus-7C.1F] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[kit-entity-consensus-7C.1F] actor="
        + json.dumps(actor_summary, sort_keys=True)
    )
    print(
        "[kit-entity-consensus-7C.1F] identity="
        + json.dumps(identity_summary, sort_keys=True)
    )
    for key, path in actor_sheets.items():
        if path:
            print(
                "[kit-entity-consensus-7C.1F] "
                f"actor_{key}_sheet={path}"
            )
    for key, path in identity_sheets.items():
        if path:
            print(
                "[kit-entity-consensus-7C.1F] "
                f"identity_{key}_sheet={path}"
            )
    print(
        "[kit-entity-consensus-7C.1F] "
        f"artifact={output_dir / 'index.json'}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
