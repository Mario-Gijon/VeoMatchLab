#!/usr/bin/env python3
"""Visual audit for Tracking V1 Hito 7E.3B selected persistent-chain links.

Creates paginated contact sheets for the provisional persistent links selected
by ``persistent-chain-path-v1``.  Each row shows representative appearance
crops from the predecessor actor and successor actor, together with score,
gap, cosine, confidence and anonymous-kit family.

This is diagnostic only.  It does not alter identities or Tracking V1 stages.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


AUDIT_VERSION = "persistent-chain-audit-v1"


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def _fit_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(
        width / max(1, image.shape[1]),
        height / max(1, image.shape[0]),
    )
    resized = cv2.resize(
        image,
        (
            max(1, int(round(image.shape[1] * scale))),
            max(1, int(round(image.shape[0] * scale))),
        ),
        interpolation=(
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        ),
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[
        y:y + resized.shape[0],
        x:x + resized.shape[1],
    ] = resized
    return canvas


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.42,
) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def _representative_crops(
    *,
    actor_id: str,
    actor_tracklets: Mapping[str, tuple[str, ...]],
    crops_by_tracklet: Mapping[str, list[Mapping[str, Any]]],
    tracking_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []

    for tracklet_id in actor_tracklets[actor_id]:
        rows = crops_by_tracklet.get(tracklet_id, [])
        if not rows:
            continue
        # Keep only the strongest crop for each contributing tracklet first.
        candidates.append(
            max(
                rows,
                key=lambda row: (
                    float(row.get("quality", 0.0)),
                    int(row.get("frame_index", 0)),
                ),
            )
        )

    candidates.sort(
        key=lambda row: (
            -float(row.get("quality", 0.0)),
            str(row["tracklet_id"]),
            int(row.get("frame_index", 0)),
        )
    )

    selected = candidates[: max(1, count)]
    result: list[dict[str, Any]] = []

    for row in selected:
        path = tracking_root / str(row["crop_ref"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        result.append({
            "tracklet_id": str(row["tracklet_id"]),
            "quality": float(row.get("quality", 0.0)),
            "timestamp_s": float(row.get("timestamp_s", 0.0)),
            "crop_ref": str(row["crop_ref"]),
            "image": image,
        })

    return result


def _risk_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    # Audit the weakest / longest accepted links first.
    return (
        float(row["rank_score"]),
        -float(row["gap_s"]),
        float(row["profile_confidence"]),
        str(row["predecessor_actor_id"]),
        str(row["successor_actor_id"]),
    )


def _make_page(
    rows: Sequence[Mapping[str, Any]],
    *,
    crops_per_actor: int,
    panel_width: int = 145,
    image_height: int = 180,
    info_width: int = 300,
    row_height: int = 235,
) -> np.ndarray:
    width = (
        crops_per_actor * panel_width * 2
        + info_width
    )
    page = np.zeros(
        (len(rows) * row_height, width, 3),
        dtype=np.uint8,
    )

    for row_index, row in enumerate(rows):
        y0 = row_index * row_height

        left = row["_predecessor_crops"]
        right = row["_successor_crops"]

        # Left actor representative crops.
        for crop_index in range(crops_per_actor):
            x0 = crop_index * panel_width
            if crop_index >= len(left):
                continue
            item = left[crop_index]
            page[
                y0:y0 + image_height,
                x0:x0 + panel_width,
            ] = _fit_panel(
                item["image"],
                width=panel_width,
                height=image_height,
            )
            _put_text(
                page,
                item["tracklet_id"],
                x0 + 3,
                y0 + image_height + 16,
                scale=0.34,
            )
            _put_text(
                page,
                f"q={item['quality']:.2f} t={item['timestamp_s']:.1f}",
                x0 + 3,
                y0 + image_height + 32,
                scale=0.31,
            )

        info_x = crops_per_actor * panel_width

        _put_text(
            page,
            (
                f"{row['predecessor_actor_id']}  ->  "
                f"{row['successor_actor_id']}"
            ),
            info_x + 8,
            y0 + 28,
            scale=0.43,
        )
        _put_text(
            page,
            f"score={float(row['rank_score']):.3f}",
            info_x + 8,
            y0 + 58,
            scale=0.43,
        )
        _put_text(
            page,
            f"gap={float(row['gap_s']):.2f}s",
            info_x + 8,
            y0 + 84,
            scale=0.43,
        )
        _put_text(
            page,
            f"cos={float(row['aggregate_cosine']):.3f}",
            info_x + 8,
            y0 + 110,
            scale=0.43,
        )
        _put_text(
            page,
            f"conf={float(row['profile_confidence']):.3f}",
            info_x + 8,
            y0 + 136,
            scale=0.43,
        )
        _put_text(
            page,
            f"kit={row['kit_family']}",
            info_x + 8,
            y0 + 162,
            scale=0.40,
        )
        _put_text(
            page,
            f"audit_rank={int(row['_audit_rank']):02d}",
            info_x + 8,
            y0 + 190,
            scale=0.38,
        )

        # Right actor representative crops.
        right_start = info_x + info_width
        for crop_index in range(crops_per_actor):
            x0 = right_start + crop_index * panel_width
            if crop_index >= len(right):
                continue
            item = right[crop_index]
            page[
                y0:y0 + image_height,
                x0:x0 + panel_width,
            ] = _fit_panel(
                item["image"],
                width=panel_width,
                height=image_height,
            )
            _put_text(
                page,
                item["tracklet_id"],
                x0 + 3,
                y0 + image_height + 16,
                scale=0.34,
            )
            _put_text(
                page,
                f"q={item['quality']:.2f} t={item['timestamp_s']:.1f}",
                x0 + 3,
                y0 + image_height + 32,
                scale=0.31,
            )

    return page


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--crops-per-actor",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--links-per-page",
        type=int,
        default=7,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.crops_per_actor < 1:
        raise ValueError("--crops-per-actor must be >= 1")
    if args.links_per_page < 1:
        raise ValueError("--links-per-page must be >= 1")

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    solver_root = (
        root / "diagnostics" / "persistent_chain_solver"
        / "persistent-chain-path-v1"
    )
    selected_links_path = solver_root / "selected_links.jsonl"

    backbone_root = (
        root / "diagnostics" / "actor_temporal_backbone"
        / "temporal-actor-backbone-v1"
    )
    actors_path = backbone_root / "actors.jsonl"

    crops_index_path = (
        root / "diagnostics" / "appearance_crops" / "index.json"
    )

    for path in (
        selected_links_path,
        actors_path,
        crops_index_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    selected_links = _iter_jsonl(selected_links_path)
    actor_rows = _iter_jsonl(actors_path)
    crop_index = _load_json(crops_index_path)

    actor_tracklets = {
        str(row["actor_id"]): tuple(
            str(tracklet_id)
            for tracklet_id in row["tracklet_ids"]
        )
        for row in actor_rows
    }

    crop_records = crop_index.get("records")
    if not isinstance(crop_records, list):
        raise ValueError("appearance crop index has no records list")

    crops_by_tracklet: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_records:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") != "saved":
            continue
        if not row.get("crop_ref"):
            continue
        crops_by_tracklet[str(row["tracklet_id"])].append(row)

    ranked = sorted(selected_links, key=_risk_key)

    audit_rows: list[dict[str, Any]] = []
    for audit_rank, row in enumerate(ranked, start=1):
        predecessor = str(row["predecessor_actor_id"])
        successor = str(row["successor_actor_id"])

        if predecessor not in actor_tracklets:
            raise ValueError(f"unknown predecessor actor {predecessor}")
        if successor not in actor_tracklets:
            raise ValueError(f"unknown successor actor {successor}")

        left = _representative_crops(
            actor_id=predecessor,
            actor_tracklets=actor_tracklets,
            crops_by_tracklet=crops_by_tracklet,
            tracking_root=root,
            count=args.crops_per_actor,
        )
        right = _representative_crops(
            actor_id=successor,
            actor_tracklets=actor_tracklets,
            crops_by_tracklet=crops_by_tracklet,
            tracking_root=root,
            count=args.crops_per_actor,
        )

        audit_rows.append({
            **row,
            "_audit_rank": audit_rank,
            "_predecessor_crops": left,
            "_successor_crops": right,
        })

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics" / "persistent_chain_audit"
        / AUDIT_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    public_rows = []
    for row in audit_rows:
        public_rows.append({
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        } | {
            "audit_rank": int(row["_audit_rank"]),
            "predecessor_crop_refs": [
                item["crop_ref"]
                for item in row["_predecessor_crops"]
            ],
            "successor_crop_refs": [
                item["crop_ref"]
                for item in row["_successor_crops"]
            ],
        })

    audit_jsonl = output_dir / "audit.jsonl"
    with audit_jsonl.open("w", encoding="utf-8") as handle:
        for row in public_rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )

    pages: list[str] = []
    per_page = args.links_per_page
    for start in range(0, len(audit_rows), per_page):
        page_rows = audit_rows[start:start + per_page]
        page = _make_page(
            page_rows,
            crops_per_actor=args.crops_per_actor,
        )
        page_number = start // per_page + 1
        page_path = output_dir / f"page_{page_number:02d}.jpg"
        if not cv2.imwrite(
            str(page_path),
            page,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"could not write {page_path}")
        pages.append(str(page_path))

    summary = {
        "schema_version": 1,
        "audit_version": AUDIT_VERSION,
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "selected_link_count": len(selected_links),
        "audited_link_count": len(audit_rows),
        "page_count": len(pages),
        "crops_per_actor": args.crops_per_actor,
        "links_per_page": args.links_per_page,
        "sort_order": (
            "lowest score first; ties prefer longer gap and lower confidence"
        ),
        "missing_predecessor_visual_count": sum(
            not row["_predecessor_crops"] for row in audit_rows
        ),
        "missing_successor_visual_count": sum(
            not row["_successor_crops"] for row in audit_rows
        ),
        "pages": pages,
        "audit_jsonl": str(audit_jsonl),
        "source_selected_links": str(selected_links_path),
        "source_backbone_actors": str(actors_path),
        "source_appearance_crops": str(crops_index_path),
        "identity_decisions_changed": False,
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[persistent-chain-audit-7E.3C] "
        f"links={len(selected_links)} "
        f"pages={len(pages)} "
        f"missing_left={summary['missing_predecessor_visual_count']} "
        f"missing_right={summary['missing_successor_visual_count']}"
    )
    for page in pages:
        print(
            "[persistent-chain-audit-7E.3C] "
            f"page={page}"
        )
    print(
        "[persistent-chain-audit-7E.3C] "
        f"artifact={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
