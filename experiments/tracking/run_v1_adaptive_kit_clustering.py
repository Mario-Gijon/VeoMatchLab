#!/usr/bin/env python3
"""Run Hito 7C.1C adaptive anonymous kit-group diagnostics.

This stage does not hard-code colours, team count, referee colour, goalkeeper
colour, or semantic roles.  It reuses existing Hito 7C.1 kit profiles and
selects a clip-specific anonymous visual-group model.

Outputs:
  diagnostics/kit_clustering_adaptive/kit-clustering-adaptive-v2/
      assignments.jsonl
      clusters.jsonl
      model_scores.jsonl
      selection_trace.jsonl
      cluster_XX.jpg
      ambiguous.jpg
      outliers.jpg
      low_reliability.jpg
      summary.json
      index.json
"""

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


from tracking.appearance.kit import KitTrackletProfile  # noqa: E402
from tracking.appearance.kit_clustering_adaptive import (  # noqa: E402
    ADAPTIVE_KIT_CLUSTERING_VERSION,
    AdaptiveKitClusteringConfig,
    cluster_kit_profiles_adaptive,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain an object"
                )
            yield payload


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


def _profile_from_row(row: Mapping[str, Any]) -> KitTrackletProfile:
    return KitTrackletProfile(
        tracklet_id=str(row["tracklet_id"]),
        aggregate_signature=tuple(
            float(value)
            for value in row["aggregate_signature"]
        ),
        sample_count=int(row["sample_count"]),
        effective_sample_count=float(row["effective_sample_count"]),
        reliability=float(row["reliability"]),
        mean_quality=float(row["mean_quality"]),
        internal_similarity=dict(row["internal_similarity"]),
        first_timestamp_s=float(row["first_timestamp_s"]),
        last_timestamp_s=float(row["last_timestamp_s"]),
    )


def _fit_panel(
    image: np.ndarray,
    *,
    width: int = 150,
    height: int = 195,
) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(
        width / image.shape[1],
        height / image.shape[0],
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
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    return canvas


def _contact_sheet(
    items: list[dict[str, Any]],
    *,
    columns: int = 8,
    panel_width: int = 150,
    panel_height: int = 230,
) -> np.ndarray | None:
    if not items:
        return None

    rows = (len(items) + columns - 1) // columns
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )
    image_height = panel_height - 35

    for index, item in enumerate(items):
        panel = np.zeros(
            (panel_height, panel_width, 3),
            dtype=np.uint8,
        )
        panel[:image_height] = _fit_panel(
            item["image"],
            width=panel_width,
            height=image_height,
        )
        cv2.putText(
            panel,
            str(item["tracklet_id"]),
            (3, image_height + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.30,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            (
                f"cos={float(item['best_cosine']):.3f} "
                f"m={float(item['margin']):.3f}"
            ),
            (3, image_height + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.29,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row = index // columns
        column = index % columns
        y = row * panel_height
        x = column * panel_width
        sheet[
            y:y + panel_height,
            x:x + panel_width,
        ] = panel

    return sheet


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-k", type=int, default=2)
    parser.add_argument("--max-k", type=int, default=8)
    parser.add_argument(
        "--contact-sheet-per-group",
        type=int,
        default=40,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    kit_root = (
        root / "diagnostics" / "kit_evidence"
        / "kit-appearance-v1"
    )
    profiles_path = kit_root / "profiles.jsonl"
    samples_path = kit_root / "samples.jsonl"

    for path in (profiles_path, samples_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    profiles = tuple(
        _profile_from_row(row)
        for row in _iter_jsonl(profiles_path)
    )

    config = AdaptiveKitClusteringConfig(
        min_k=args.min_k,
        max_k=args.max_k,
    )
    result = cluster_kit_profiles_adaptive(
        profiles,
        config,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root / "diagnostics"
        / "kit_clustering_adaptive"
        / ADAPTIVE_KIT_CLUSTERING_VERSION
    )

    assignments_path = output_dir / "assignments.jsonl"
    clusters_path = output_dir / "clusters.jsonl"
    scores_path = output_dir / "model_scores.jsonl"
    trace_path = output_dir / "selection_trace.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    assignment_count = _write_jsonl(
        assignments_path,
        (item.to_dict() for item in result.assignments),
    )
    cluster_count = _write_jsonl(
        clusters_path,
        (item.to_dict() for item in result.clusters),
    )
    score_count = _write_jsonl(
        scores_path,
        (item.to_dict() for item in result.model_scores),
    )
    trace_count = _write_jsonl(
        trace_path,
        (item.to_dict() for item in result.selection_trace),
    )

    best_sample_by_tracklet: dict[str, Mapping[str, Any]] = {}
    for sample in _iter_jsonl(samples_path):
        tracklet_id = str(sample["tracklet_id"])
        current = best_sample_by_tracklet.get(tracklet_id)
        if (
            current is None
            or float(sample["quality"])
            > float(current["quality"])
        ):
            best_sample_by_tracklet[tracklet_id] = sample

    assignments_by_cluster = defaultdict(list)
    assignments_by_state = defaultdict(list)
    for assignment in result.assignments:
        assignments_by_cluster[assignment.cluster_id].append(assignment)
        assignments_by_state[assignment.assignment_state].append(assignment)

    output_dir.mkdir(parents=True, exist_ok=True)
    limit = max(1, int(args.contact_sheet_per_group))
    contact_sheets: dict[str, str | None] = {}

    def write_sheet(
        key: str,
        assignments,
        *,
        confident_first: bool,
    ) -> None:
        ranked = sorted(
            assignments,
            key=(
                (
                    lambda item: (
                        -item.usable_for_identity,
                        -item.margin,
                        -item.best_cosine,
                        -item.profile_reliability,
                        item.tracklet_id,
                    )
                )
                if confident_first
                else (
                    lambda item: (
                        item.margin,
                        item.best_cosine,
                        item.profile_reliability,
                        item.tracklet_id,
                    )
                )
            ),
        )

        items = []
        for assignment in ranked:
            sample = best_sample_by_tracklet.get(
                assignment.tracklet_id
            )
            if sample is None:
                continue
            image = cv2.imread(
                str(root / str(sample["crop_ref"])),
                cv2.IMREAD_COLOR,
            )
            if image is None:
                continue
            items.append({
                "tracklet_id": assignment.tracklet_id,
                "best_cosine": assignment.best_cosine,
                "margin": assignment.margin,
                "image": image,
            })
            if len(items) >= limit:
                break

        sheet = _contact_sheet(items)
        if sheet is None:
            contact_sheets[key] = None
            return

        path = output_dir / f"{key}.jpg"
        if not cv2.imwrite(
            str(path),
            sheet,
            [int(cv2.IMWRITE_JPEG_QUALITY), 96],
        ):
            raise RuntimeError(f"could not write {path}")
        contact_sheets[key] = str(path)

    for cluster in result.clusters:
        write_sheet(
            f"cluster_{cluster.cluster_id:02d}",
            assignments_by_cluster[cluster.cluster_id],
            confident_first=True,
        )

    for state in (
        "ambiguous",
        "outlier",
        "low_reliability",
    ):
        write_sheet(
            state,
            assignments_by_state[state],
            confident_first=False,
        )

    summary = result.summary()
    summary.update({
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "source_profiles": str(profiles_path),
        "source_samples": str(samples_path),
        "contact_sheets": contact_sheets,
    })
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "adaptive_kit_clustering_version": (
                ADAPTIVE_KIT_CLUSTERING_VERSION
            ),
            "assignments": assignments_path.name,
            "clusters": clusters_path.name,
            "model_scores": scores_path.name,
            "selection_trace": trace_path.name,
            "summary": summary_path.name,
            "assignment_count": assignment_count,
            "cluster_count": cluster_count,
            "model_score_count": score_count,
            "selection_trace_count": trace_count,
            "selected_k": result.selected_k,
            "silhouette_best_k": result.silhouette_best_k,
            "semantic_labels_assigned": False,
            "identity_hard_constraints_created": False,
        },
    )

    print(
        "[adaptive-kit-7C.1C] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[adaptive-kit-7C.1C] "
        f"profiles={len(profiles)} "
        f"anchors={len(result.anchor_tracklet_ids)} "
        f"silhouette_best_k={result.silhouette_best_k} "
        f"selected_k={result.selected_k}"
    )
    print(
        "[adaptive-kit-7C.1C] model_scores="
        + json.dumps(
            [item.to_dict() for item in result.model_scores],
            sort_keys=True,
        )
    )
    print(
        "[adaptive-kit-7C.1C] selection_trace="
        + json.dumps(
            [item.to_dict() for item in result.selection_trace],
            sort_keys=True,
        )
    )
    print(
        "[adaptive-kit-7C.1C] states="
        + json.dumps(
            summary["assignment_state_counts"],
            sort_keys=True,
        )
    )
    print(
        "[adaptive-kit-7C.1C] clusters="
        + json.dumps(
            [
                {
                    "cluster_id": item.cluster_id,
                    "anchor_profiles": item.anchor_profile_count,
                    "assigned_profiles": item.assigned_profile_count,
                    "usable": item.usable_assignment_count,
                    "mass_fraction": item.anchor_mass_fraction,
                    "mean_cosine": item.mean_best_cosine,
                    "mean_margin": item.mean_margin,
                }
                for item in result.clusters
            ],
            sort_keys=True,
        )
    )
    for key, path in contact_sheets.items():
        if path is not None:
            print(
                "[adaptive-kit-7C.1C] "
                f"{key}_sheet={path}"
            )
    print(f"[adaptive-kit-7C.1C] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
