#!/usr/bin/env python3
"""Visual audit for Hito 7E.5A family-aware link changes.

Compares:
* baseline seeded identity expansion at score .60
* family-aware identity expansion

Only links whose selected/not-selected status changed are rendered:
NEW and LOST.  Complete candidate tables are used to show why a LOST edge
changed even though it no longer appears in selected_links.jsonl.

Diagnostic only: no tracking or identity decision is changed.
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


from tracking.actors.identity_family_delta_audit import (  # noqa: E402
    DELTA_AUDIT_VERSION,
    build_delta_records,
    summarize_delta,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x = (width - resized.shape[1]) // 2
    y = (height - resized.shape[0]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _put_text(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.36,
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
    identity_id: str,
    identity_tracklets: Mapping[str, tuple[str, ...]],
    crops_by_tracklet: Mapping[str, list[Mapping[str, Any]]],
    tracking_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []

    for tracklet_id in identity_tracklets.get(identity_id, ()):
        rows = crops_by_tracklet.get(tracklet_id, [])
        if not rows:
            continue
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

    result = []
    for row in candidates[:max(1, count)]:
        crop_path = tracking_root / str(row["crop_ref"])
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
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


def _family_text(value: Mapping[str, Any]) -> str:
    key = value.get("family_key") or value.get(
        "candidate_family_key"
    )
    if key is None:
        return "unknown"
    confidence = value.get("confidence")
    if confidence is None:
        return str(key)
    return f"{key} c={float(confidence):.2f}"


def _num(value: Any, fmt: str = ".3f") -> str:
    if value is None:
        return "-"
    return format(float(value), fmt)


def _make_page(
    rows: Sequence[Mapping[str, Any]],
    *,
    crops_per_identity: int,
    panel_width: int = 145,
    image_height: int = 180,
    info_width: int = 390,
    row_height: int = 250,
) -> np.ndarray:
    width = crops_per_identity * panel_width * 2 + info_width
    page = np.zeros(
        (len(rows) * row_height, width, 3),
        dtype=np.uint8,
    )

    for row_index, row in enumerate(rows):
        y0 = row_index * row_height
        left = row["_left_crops"]
        right = row["_right_crops"]

        for crop_index in range(crops_per_identity):
            x0 = crop_index * panel_width
            if crop_index < len(left):
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
                    y0 + image_height + 15,
                    scale=0.31,
                )
                _put_text(
                    page,
                    (
                        f"q={item['quality']:.2f} "
                        f"t={item['timestamp_s']:.1f}"
                    ),
                    x0 + 3,
                    y0 + image_height + 31,
                    scale=0.28,
                )

        info_x = crops_per_identity * panel_width
        status = str(row["status"]).upper()
        _put_text(
            page,
            (
                f"{status} #{int(row['audit_rank']):02d}  "
                f"{row['predecessor_identity_id']} -> "
                f"{row['successor_identity_id']}"
            ),
            info_x + 8,
            y0 + 24,
            scale=0.40,
        )
        _put_text(
            page,
            (
                f"gap={_num(row.get('gap_s'), '.2f')}s  "
                f"cos={_num(row.get('aggregate_cosine'))}  "
                f"conf={_num(row.get('profile_confidence'))}"
            ),
            info_x + 8,
            y0 + 50,
            scale=0.35,
        )
        _put_text(
            page,
            (
                f"baseline={_num(row.get('baseline_rank_score'))}  "
                f"base={_num(row.get('family_base_rank_score'))}  "
                f"family={_num(row.get('family_aware_rank_score'))}"
            ),
            info_x + 8,
            y0 + 76,
            scale=0.34,
        )
        _put_text(
            page,
            (
                f"relation={row.get('family_relation')}  "
                f"strength={_num(row.get('family_pair_strength'))}  "
                f"adj={_num(row.get('family_adjustment'))}"
            ),
            info_x + 8,
            y0 + 102,
            scale=0.33,
        )
        _put_text(
            page,
            "L: " + _family_text(row["predecessor_family"]),
            info_x + 8,
            y0 + 130,
            scale=0.33,
        )
        _put_text(
            page,
            "R: " + _family_text(row["successor_family"]),
            info_x + 8,
            y0 + 156,
            scale=0.33,
        )
        _put_text(
            page,
            (
                "ranks old="
                f"{row.get('baseline_outgoing_score_rank')}x"
                f"{row.get('baseline_incoming_score_rank')}  "
                "new="
                f"{row.get('family_outgoing_score_rank')}x"
                f"{row.get('family_incoming_score_rank')}"
            ),
            info_x + 8,
            y0 + 182,
            scale=0.32,
        )
        _put_text(
            page,
            (
                f"selected old={row['baseline_selected']} "
                f"new={row['family_aware_selected']}"
            ),
            info_x + 8,
            y0 + 208,
            scale=0.32,
        )

        right_start = info_x + info_width
        for crop_index in range(crops_per_identity):
            x0 = right_start + crop_index * panel_width
            if crop_index < len(right):
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
                    y0 + image_height + 15,
                    scale=0.31,
                )
                _put_text(
                    page,
                    (
                        f"q={item['quality']:.2f} "
                        f"t={item['timestamp_s']:.1f}"
                    ),
                    x0 + 3,
                    y0 + image_height + 31,
                    scale=0.28,
                )

    return page


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--baseline-dir")
    parser.add_argument("--family-aware-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--crops-per-identity", type=int, default=3)
    parser.add_argument("--links-per-page", type=int, default=6)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.crops_per_identity < 1:
        raise ValueError("--crops-per-identity must be >= 1")
    if args.links_per_page < 1:
        raise ValueError("--links-per-page must be >= 1")

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    baseline_dir = (
        Path(args.baseline_dir)
        if args.baseline_dir
        else root / "diagnostics" / "identity_expansion"
        / "seeded-identity-expansion-v1-score060"
    )
    family_dir = (
        Path(args.family_aware_dir)
        if args.family_aware_dir
        else root / "diagnostics" / "identity_expansion"
        / "seeded-identity-expansion-v2-family-aware"
    )

    baseline_selected_path = baseline_dir / "selected_links.jsonl"
    baseline_candidates_path = baseline_dir / "candidates.jsonl"
    family_selected_path = family_dir / "selected_links.jsonl"
    family_candidates_path = family_dir / "candidates.jsonl"
    source_family_path = (
        family_dir / "source_identity_family_assignments.jsonl"
    )

    persistent_root = (
        root / "diagnostics" / "persistent_chain_solver"
        / "persistent-chain-path-v1"
    )
    source_identities_path = persistent_root / "identities.jsonl"

    crops_index_path = (
        root / "diagnostics" / "appearance_crops" / "index.json"
    )

    required = (
        baseline_selected_path,
        baseline_candidates_path,
        family_selected_path,
        family_candidates_path,
        source_identities_path,
        crops_index_path,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    baseline_selected = _iter_jsonl(baseline_selected_path)
    baseline_candidates = _iter_jsonl(baseline_candidates_path)
    family_selected = _iter_jsonl(family_selected_path)
    family_candidates = _iter_jsonl(family_candidates_path)
    source_family = (
        _iter_jsonl(source_family_path)
        if source_family_path.is_file()
        else []
    )

    delta = list(build_delta_records(
        baseline_selected_rows=baseline_selected,
        family_selected_rows=family_selected,
        baseline_candidate_rows=baseline_candidates,
        family_candidate_rows=family_candidates,
        source_family_rows=source_family,
    ))

    identities = _iter_jsonl(source_identities_path)
    identity_tracklets = {
        str(row["identity_id"]): tuple(
            str(value) for value in row["tracklet_ids"]
        )
        for row in identities
    }

    crops_index = _load_json(crops_index_path)
    crop_records = crops_index.get("records")
    if not isinstance(crop_records, list):
        raise ValueError("appearance crop index has no records list")

    crops_by_tracklet: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in crop_records:
        if not isinstance(row, Mapping):
            continue
        if row.get("status") != "saved" or not row.get("crop_ref"):
            continue
        crops_by_tracklet[str(row["tracklet_id"])].append(row)

    audit_rows = []
    for row in delta:
        predecessor = str(row["predecessor_identity_id"])
        successor = str(row["successor_identity_id"])
        if predecessor not in identity_tracklets:
            raise ValueError(f"unknown identity {predecessor}")
        if successor not in identity_tracklets:
            raise ValueError(f"unknown identity {successor}")

        audit_rows.append({
            **row,
            "_left_crops": _representative_crops(
                identity_id=predecessor,
                identity_tracklets=identity_tracklets,
                crops_by_tracklet=crops_by_tracklet,
                tracking_root=root,
                count=args.crops_per_identity,
            ),
            "_right_crops": _representative_crops(
                identity_id=successor,
                identity_tracklets=identity_tracklets,
                crops_by_tracklet=crops_by_tracklet,
                tracking_root=root,
                count=args.crops_per_identity,
            ),
        })

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else family_dir / "delta_audit" / DELTA_AUDIT_VERSION
    )
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    public_rows = []
    for row in audit_rows:
        public = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        public["predecessor_crop_refs"] = [
            item["crop_ref"] for item in row["_left_crops"]
        ]
        public["successor_crop_refs"] = [
            item["crop_ref"] for item in row["_right_crops"]
        ]
        public_rows.append(public)

    delta_path = output_dir / "delta.jsonl"
    _write_jsonl(delta_path, public_rows)

    pages = []
    for start in range(0, len(audit_rows), args.links_per_page):
        page_rows = audit_rows[start:start + args.links_per_page]
        page = _make_page(
            page_rows,
            crops_per_identity=args.crops_per_identity,
        )
        page_number = start // args.links_per_page + 1
        page_path = output_dir / f"page_{page_number:02d}.jpg"
        if not cv2.imwrite(
            str(page_path),
            page,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"could not write {page_path}")
        pages.append(str(page_path))

    summary = summarize_delta(delta)
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "baseline_selected_link_count": len(baseline_selected),
        "family_aware_selected_link_count": len(family_selected),
        "common_link_count": (
            len(baseline_selected)
            + len(family_selected)
            - len(delta)
        ) // 2,
        "page_count": len(pages),
        "pages": pages,
        "missing_left_visual_count": sum(
            not row["_left_crops"] for row in audit_rows
        ),
        "missing_right_visual_count": sum(
            not row["_right_crops"] for row in audit_rows
        ),
        "source_baseline_dir": str(baseline_dir),
        "source_family_aware_dir": str(family_dir),
        "source_identity_family_assignments": (
            str(source_family_path)
            if source_family_path.is_file()
            else None
        ),
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    print(
        "[identity-family-delta-audit-7E.5A] "
        f"baseline={len(baseline_selected)} "
        f"family_aware={len(family_selected)} "
        f"delta={len(delta)} "
        f"lost={summary['status_counts'].get('lost', 0)} "
        f"new={summary['status_counts'].get('new', 0)}"
    )
    print(
        "[identity-family-delta-audit-7E.5A] relations="
        + json.dumps(
            summary["family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[identity-family-delta-audit-7E.5A] lost_relations="
        + json.dumps(
            summary["lost_family_relation_counts"],
            sort_keys=True,
        )
    )
    print(
        "[identity-family-delta-audit-7E.5A] new_relations="
        + json.dumps(
            summary["new_family_relation_counts"],
            sort_keys=True,
        )
    )
    for page in pages:
        print(
            "[identity-family-delta-audit-7E.5A] "
            f"page={page}"
        )
    print(
        "[identity-family-delta-audit-7E.5A] "
        f"artifact={summary_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
