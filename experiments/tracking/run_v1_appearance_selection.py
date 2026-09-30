#!/usr/bin/env python3
"""Diagnose Tracking V1 appearance-moment selection on published artifacts.

This runner is deliberately CPU-only and model-free.  It does not reopen the
video, rerun RF-DETR, crop pixels or compute ReID embeddings.

It reads the already published V1 artifacts:

    detections/frames.jsonl
    tracklets/tracklets.jsonl

applies ``tracking.appearance.selection`` and writes a diagnostic JSON outside
the appearance stage itself:

    tracking/v1/diagnostics/appearance_selection.json

The appearance stage remains ``not_run`` until actual appearance descriptors
are produced in a later milestone.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_DEV_CONFIG = ROOT / "experiments" / "tracking" / "dev_clip.json"

from tracking.appearance.selection import (  # noqa: E402
    AppearanceSelectionConfig,
    AppearanceSelectionSet,
    select_appearance_moments,
)
from tracking.detections import CanonicalDetectionFrame  # noqa: E402
from tracking.tracklets import Tracklet, TrackletSet  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_dev_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dev config must encode a JSON object")

    required = {"match_id", "clip_id"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"dev config is missing keys: {missing}")

    return payload


def artifact_root(
    *,
    data_root: Path,
    match_id: str,
    clip_id: str,
) -> Path:
    if not isinstance(match_id, str) or not match_id.strip():
        raise ValueError("match_id must be a non-empty string")
    if not isinstance(clip_id, str) or not clip_id.strip():
        raise ValueError("clip_id must be a non-empty string")

    return (
        data_root
        / "matches"
        / match_id.strip()
        / "clips"
        / clip_id.strip()
        / "tracking"
        / "v1"
    )


def _iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path}: invalid JSON on line {line_number}"
                ) from exc

            if not isinstance(payload, dict):
                raise ValueError(
                    f"{path}: line {line_number} must encode a JSON object"
                )

            yield payload


def load_detection_frames(path: Path) -> tuple[CanonicalDetectionFrame, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"detections artifact not found: {path}")

    frames = tuple(
        CanonicalDetectionFrame.from_dict(payload)
        for payload in _iter_jsonl(path)
    )
    if not frames:
        raise ValueError(f"detections artifact is empty: {path}")

    return frames


def load_tracklets(path: Path) -> TrackletSet:
    if not path.is_file():
        raise FileNotFoundError(f"tracklets artifact not found: {path}")

    tracklets = tuple(
        Tracklet.from_dict(payload)
        for payload in _iter_jsonl(path)
    )
    if not tracklets:
        raise ValueError(f"tracklets artifact is empty: {path}")

    return TrackletSet(tracklets=tracklets)


def _quantile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")

    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]

    position = probability * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low

    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def build_diagnostics(
    selection: AppearanceSelectionSet,
    tracklets: TrackletSet,
    *,
    detail_limit: int = 12,
) -> dict[str, Any]:
    if detail_limit < 1:
        raise ValueError("detail_limit must be >= 1")

    tracklet_by_id = {
        tracklet.tracklet_id: tracklet
        for tracklet in tracklets.tracklets
    }

    selection_ids = {
        item.tracklet_id
        for item in selection.selections
    }
    tracklet_ids = set(tracklet_by_id)
    if selection_ids != tracklet_ids:
        missing = sorted(tracklet_ids - selection_ids)
        extra = sorted(selection_ids - tracklet_ids)
        raise ValueError(
            "selection/tracklet ids do not match: "
            f"missing={missing}, extra={extra}"
        )

    selected_counts = [
        item.selected_count
        for item in selection.selections
    ]
    eligible_counts = [
        item.eligible_count
        for item in selection.selections
    ]
    selected_qualities = [
        moment.quality
        for item in selection.selections
        for moment in item.moments
    ]

    selected_distribution = Counter(selected_counts)
    eligible_distribution = Counter(eligible_counts)

    details: list[dict[str, Any]] = []
    for item in selection.selections:
        tracklet = tracklet_by_id[item.tracklet_id]
        qualities = [
            moment.quality
            for moment in item.moments
        ]

        details.append(
            {
                "tracklet_id": item.tracklet_id,
                "observation_count": tracklet.observation_count,
                "duration_s": tracklet.duration_s,
                "candidate_count": item.candidate_count,
                "eligible_count": item.eligible_count,
                "selected_count": item.selected_count,
                "selected_quality_mean": (
                    statistics.fmean(qualities)
                    if qualities
                    else None
                ),
                "selected_quality_min": min(qualities) if qualities else None,
                "selected_quality_max": max(qualities) if qualities else None,
                "selected_frames": [
                    moment.frame_index
                    for moment in item.moments
                ],
                "selected_times_s": [
                    moment.timestamp_s
                    for moment in item.moments
                ],
            }
        )

    # Worst first: no/low evidence, then short tracklets.  Deterministic tie
    # breakers make repeated runs directly diffable.
    weakest = sorted(
        details,
        key=lambda item: (
            item["selected_count"],
            (
                item["selected_quality_mean"]
                if item["selected_quality_mean"] is not None
                else -1.0
            ),
            item["observation_count"],
            item["tracklet_id"],
        ),
    )[:detail_limit]

    # Best first: many selected moments, high quality, long support.
    strongest = sorted(
        details,
        key=lambda item: (
            -item["selected_count"],
            -(
                item["selected_quality_mean"]
                if item["selected_quality_mean"] is not None
                else -1.0
            ),
            -item["observation_count"],
            item["tracklet_id"],
        ),
    )[:detail_limit]

    return {
        "summary": selection.summary(),
        "selection_distribution": {
            str(count): selected_distribution[count]
            for count in sorted(selected_distribution)
        },
        "eligible_distribution": {
            str(count): eligible_distribution[count]
            for count in sorted(eligible_distribution)
        },
        "selected_quality": {
            "min": min(selected_qualities, default=None),
            "p10": _quantile(selected_qualities, 0.10),
            "median": _quantile(selected_qualities, 0.50),
            "p90": _quantile(selected_qualities, 0.90),
            "max": max(selected_qualities, default=None),
            "mean": (
                statistics.fmean(selected_qualities)
                if selected_qualities
                else None
            ),
        },
        "weakest_tracklets": weakest,
        "strongest_tracklets": strongest,
        "tracklets": details,
    }


def _selection_config_from_args(
    args: argparse.Namespace,
) -> AppearanceSelectionConfig:
    return AppearanceSelectionConfig(
        max_samples_per_tracklet=args.max_samples,
        min_samples_if_available=args.min_samples,
        min_quality=args.min_quality,
        min_temporal_spacing_s=args.min_spacing_s,
        temporal_diversity_weight=args.diversity_weight,
        virtual_width_px=args.virtual_width,
        virtual_height_px=args.virtual_height,
    )


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run CPU-only appearance-moment selection diagnostics over "
            "published Tracking V1 detections and tracklets."
        )
    )
    parser.add_argument(
        "--dev-config",
        default=str(DEFAULT_DEV_CONFIG),
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Optional output JSON. Default: "
            "<tracking-v1-root>/diagnostics/appearance_selection.json"
        ),
    )
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=0.42)
    parser.add_argument("--min-spacing-s", type=float, default=0.75)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--virtual-width", type=int, default=1280)
    parser.add_argument("--virtual-height", type=int, default=720)
    parser.add_argument("--detail-limit", type=int, default=12)

    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    dev_config_path = _repo_path(args.dev_config)
    data_root = _repo_path(args.data_root)
    dev = load_dev_config(dev_config_path)

    root = artifact_root(
        data_root=data_root,
        match_id=dev["match_id"],
        clip_id=dev["clip_id"],
    )
    detections_path = root / "detections" / "frames.jsonl"
    tracklets_path = root / "tracklets" / "tracklets.jsonl"

    print(f"[appearance-selection] detections={detections_path}")
    print(f"[appearance-selection] tracklets={tracklets_path}")

    frames = load_detection_frames(detections_path)
    tracklets = load_tracklets(tracklets_path)

    config = _selection_config_from_args(args)
    selection = select_appearance_moments(
        frames,
        tracklets,
        config=config,
    )

    diagnostics = build_diagnostics(
        selection,
        tracklets,
        detail_limit=args.detail_limit,
    )

    output = (
        _repo_path(args.output)
        if args.output is not None
        else root / "diagnostics" / "appearance_selection.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "match_id": dev["match_id"],
        "clip_id": dev["clip_id"],
        "detections_artifact": str(
            detections_path.relative_to(root)
        ),
        "tracklets_artifact": str(
            tracklets_path.relative_to(root)
        ),
        **diagnostics,
    }
    output.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("[appearance-selection] completed")
    print(json.dumps(diagnostics["summary"], indent=2))
    print(
        "[appearance-selection] selected_distribution="
        + json.dumps(
            diagnostics["selection_distribution"],
            sort_keys=True,
        )
    )
    print(
        "[appearance-selection] quality="
        + json.dumps(
            diagnostics["selected_quality"],
            sort_keys=True,
        )
    )

    print("[appearance-selection] weakest:")
    for item in diagnostics["weakest_tracklets"]:
        quality = item["selected_quality_mean"]
        quality_text = (
            "none"
            if quality is None
            else f"{quality:.3f}"
        )
        print(
            "  "
            f"{item['tracklet_id']} "
            f"obs={item['observation_count']} "
            f"duration={item['duration_s']:.3f}s "
            f"eligible={item['eligible_count']} "
            f"selected={item['selected_count']} "
            f"q_mean={quality_text}"
        )

    print("[appearance-selection] strongest:")
    for item in diagnostics["strongest_tracklets"]:
        quality = item["selected_quality_mean"]
        quality_text = (
            "none"
            if quality is None
            else f"{quality:.3f}"
        )
        print(
            "  "
            f"{item['tracklet_id']} "
            f"obs={item['observation_count']} "
            f"duration={item['duration_s']:.3f}s "
            f"eligible={item['eligible_count']} "
            f"selected={item['selected_count']} "
            f"q_mean={quality_text}"
        )

    print(f"[appearance-selection] output={output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
