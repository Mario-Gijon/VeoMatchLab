#!/usr/bin/env python3
"""Render Tracking V1 Hito 7G.4A trajectory proposal audit sheets."""

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


from tracking.actors.global_trajectory_solver_audit import (  # noqa: E402
    GLOBAL_TRAJECTORY_SOLVER_AUDIT_VERSION,
    enrich_proposal_audit_rows,
    proposal_identity_ids,
    summarize_proposal_audit,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain an object")
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


def _write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
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


def _representative_crops(
    *,
    tracklet_ids: Sequence[str],
    crops_by_tracklet: Mapping[str, list[Mapping[str, Any]]],
    tracking_root: Path,
    count: int,
) -> list[dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []

    for tracklet_id in tracklet_ids:
        rows = crops_by_tracklet.get(tracklet_id, [])
        if rows:
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

    result: list[dict[str, Any]] = []
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


def _put(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    scale: float = 0.33,
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


def _num(value: Any, fmt: str = ".3f") -> str:
    if value is None:
        return "-"
    return format(float(value), fmt)


def _make_page(
    rows: Sequence[Mapping[str, Any]],
    *,
    crops_per_identity: int,
    panel_width: int = 150,
    image_height: int = 185,
    info_width: int = 570,
    row_height: int = 275,
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
                _put(
                    page,
                    item["tracklet_id"],
                    x0 + 3,
                    y0 + image_height + 15,
                    scale=0.29,
                )
                _put(
                    page,
                    f"q={item['quality']:.2f} t={item['timestamp_s']:.1f}",
                    x0 + 3,
                    y0 + image_height + 31,
                    scale=0.26,
                )

        info_x = crops_per_identity * panel_width
        first, second = proposal_identity_ids(row)
        candidate_type = str(row["candidate_type"]).upper()

        _put(
            page,
            (
                f"AUDIT #{int(row['audit_rank']):02d} "
                f"solver#{int(row.get('proposal_rank', 0)):02d} "
                f"{candidate_type}: {first}  <->  {second}"
            ),
            info_x + 8,
            y0 + 22,
            scale=0.38,
        )
        _put(
            page,
            (
                f"score={_num(row.get('trajectory_score'))} "
                f"app={_num(row.get('appearance_score'))} "
                f"cos={_num(row.get('aggregate_cosine'))} "
                f"conf={_num(row.get('profile_confidence'))}"
            ),
            info_x + 8,
            y0 + 48,
        )
        _put(
            page,
            (
                f"geom={_num(row.get('geometry_support'))} "
                f"family={row.get('family_relation')} "
                f"strength={_num(row.get('family_pair_strength'))} "
                f"adj={_num(row.get('family_adjustment'))}"
            ),
            info_x + 8,
            y0 + 74,
        )

        if candidate_type == "APPEND":
            _put(
                page,
                (
                    f"gap={_num(row.get('gap_s'), '.2f')}s "
                    f"effort={_num(row.get('physical_effort_support'))} "
                    f"velocity={_num(row.get('velocity_support'))} "
                    f"time-info={_num(row.get('temporal_informativeness'))}"
                ),
                info_x + 8,
                y0 + 100,
            )
            _put(
                page,
                (
                    f"ReID rank out/in="
                    f"{row.get('outgoing_reid_rank')}/"
                    f"{row.get('incoming_reid_rank')}  "
                    f"score rank out/in="
                    f"{row.get('outgoing_score_rank')}/"
                    f"{row.get('incoming_score_rank')}"
                ),
                info_x + 8,
                y0 + 126,
            )
        else:
            _put(
                page,
                (
                    f"left/right gap="
                    f"{_num(row.get('left_gap_s'), '.2f')}/"
                    f"{_num(row.get('right_gap_s'), '.2f')}s "
                    f"bridge={_num(row.get('audit_gap_s'), '.2f')}s"
                ),
                info_x + 8,
                y0 + 100,
            )
            _put(
                page,
                (
                    f"left/right effort="
                    f"{_num(row.get('left_physical_effort_support'))}/"
                    f"{_num(row.get('right_physical_effort_support'))} "
                    f"time-info="
                    f"{_num(row.get('left_temporal_informativeness'))}/"
                    f"{_num(row.get('right_temporal_informativeness'))}"
                ),
                info_x + 8,
                y0 + 126,
            )

        reasons = ",".join(row.get("risk_reasons", ())) or "-"
        _put(
            page,
            "risk: " + reasons[:88],
            info_x + 8,
            y0 + 154,
            scale=0.29,
        )
        if len(reasons) > 88:
            _put(
                page,
                reasons[88:176],
                info_x + 8,
                y0 + 176,
                scale=0.29,
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
                _put(
                    page,
                    item["tracklet_id"],
                    x0 + 3,
                    y0 + image_height + 15,
                    scale=0.29,
                )
                _put(
                    page,
                    f"q={item['quality']:.2f} t={item['timestamp_s']:.1f}",
                    x0 + 3,
                    y0 + image_height + 31,
                    scale=0.26,
                )

    return page


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--solver-dir")
    parser.add_argument("--identity-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--crops-per-identity", type=int, default=3)
    parser.add_argument("--proposals-per-page", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    solver_dir = (
        Path(args.solver_dir)
        if args.solver_dir
        else root / "diagnostics"
        / "global_trajectory_solver"
        / "global-trajectory-proposal-solver-v2-centered-geometry"
    )
    identity_dir = (
        Path(args.identity_dir)
        if args.identity_dir
        else root / "diagnostics"
        / "global_identity_consolidation"
        / "global-identity-consolidation-v2-family-guard"
    )

    proposals_path = solver_dir / "selected_proposals.jsonl"
    solver_summary_path = solver_dir / "summary.json"
    identities_path = identity_dir / "final_identities.jsonl"
    crops_index_path = (
        root / "diagnostics" / "appearance_crops" / "index.json"
    )

    for path in (
        proposals_path,
        solver_summary_path,
        identities_path,
        crops_index_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    proposals = enrich_proposal_audit_rows(
        _iter_jsonl(proposals_path)
    )
    identity_tracklets = {
        str(row["identity_id"]): tuple(
            str(value) for value in row.get("tracklet_ids", ())
        )
        for row in _iter_jsonl(identities_path)
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

    render_rows: list[dict[str, Any]] = []
    for row in proposals:
        first, second = proposal_identity_ids(row)
        if first not in identity_tracklets:
            raise ValueError(f"missing identity {first}")
        if second not in identity_tracklets:
            raise ValueError(f"missing identity {second}")

        render_rows.append({
            **row,
            "_left_crops": _representative_crops(
                tracklet_ids=identity_tracklets[first],
                crops_by_tracklet=crops_by_tracklet,
                tracking_root=root,
                count=args.crops_per_identity,
            ),
            "_right_crops": _representative_crops(
                tracklet_ids=identity_tracklets[second],
                crops_by_tracklet=crops_by_tracklet,
                tracking_root=root,
                count=args.crops_per_identity,
            ),
        })

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else solver_dir / "audit"
        / GLOBAL_TRAJECTORY_SOLVER_AUDIT_VERSION
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    public_rows = []
    for row in render_rows:
        public = {
            key: value
            for key, value in row.items()
            if not key.startswith("_")
        }
        public["first_crop_refs"] = [
            item["crop_ref"] for item in row["_left_crops"]
        ]
        public["second_crop_refs"] = [
            item["crop_ref"] for item in row["_right_crops"]
        ]
        public_rows.append(public)

    _write_jsonl(output_dir / "audit.jsonl", public_rows)

    pages: list[str] = []
    for start in range(
        0,
        len(render_rows),
        args.proposals_per_page,
    ):
        rows = render_rows[
            start:start + args.proposals_per_page
        ]
        page = _make_page(
            rows,
            crops_per_identity=args.crops_per_identity,
        )
        page_number = start // args.proposals_per_page + 1
        page_path = output_dir / f"page_{page_number:02d}.jpg"
        if not cv2.imwrite(
            str(page_path),
            page,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"could not write {page_path}")
        pages.append(str(page_path))

    summary = summarize_proposal_audit(render_rows)
    solver_summary = _load_json(solver_summary_path)
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "solver_selected_proposal_count": solver_summary.get(
            "selected_proposal_count"
        ),
        "solver_projected_identity_count_if_all_accepted": (
            solver_summary.get(
                "projected_identity_count_if_all_accepted"
            )
        ),
        "page_count": len(pages),
        "pages": pages,
        "missing_first_visual_count": sum(
            not row["_left_crops"] for row in render_rows
        ),
        "missing_second_visual_count": sum(
            not row["_right_crops"] for row in render_rows
        ),
        "source_selected_proposals": str(proposals_path),
        "identity_decisions_changed": False,
        "audit_required_before_merge": True,
    })

    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)

    print(
        "[global-trajectory-solver-audit-7G.4A] "
        f"audited={len(render_rows)} pages={len(pages)}"
    )
    print(
        "[global-trajectory-solver-audit-7G.4A] types="
        + json.dumps(summary["candidate_type_counts"], sort_keys=True)
    )
    print(
        "[global-trajectory-solver-audit-7G.4A] families="
        + json.dumps(summary["family_relation_counts"], sort_keys=True)
    )
    print(
        "[global-trajectory-solver-audit-7G.4A] risks="
        + json.dumps(summary["risk_reason_counts"], sort_keys=True)
    )
    for page in pages:
        print(
            "[global-trajectory-solver-audit-7G.4A] "
            f"page={page}"
        )
    print(
        "[global-trajectory-solver-audit-7G.4A] "
        f"artifact={summary_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
