#!/usr/bin/env python3
"""Run Hito 7C.1 kit appearance evidence on existing Tracking V1 crops.

This runner does not reopen the video and does not run RF-DETR or PRTReID.
It reuses:
  diagnostics/appearance_crops/index.json
  diagnostics/reid_quality/quality.json
  diagnostics/actor_candidates/sparse-actor-candidates-v1/candidates.jsonl

Outputs:
  diagnostics/kit_evidence/kit-appearance-v1/
      samples.jsonl
      rejected_samples.jsonl
      profiles.jsonl
      candidate_pairs.jsonl
      summary.json
      index.json

No team, role, identity or jersey-number labels are assigned in this hito.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.appearance.kit import (  # noqa: E402
    KIT_EVIDENCE_VERSION,
    KitEvidenceConfig,
    KitRejectedSample,
    KitSample,
    build_kit_tracklet_profiles,
    compare_kit_profiles,
    extract_kit_sample,
    kit_profile_summary,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402
from tracking.tracklets.models import Tracklet  # noqa: E402


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must encode a JSON object")
    return payload


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain a JSON object"
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


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _load_tracklets(path: Path) -> tuple[Tracklet, ...]:
    return tuple(
        Tracklet.from_dict(payload)
        for payload in _iter_jsonl(path)
    )


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    parser.add_argument("--min-crop-height", type=int, default=28)
    parser.add_argument("--min-crop-width", type=int, default=8)
    parser.add_argument(
        "--min-non-grass-pixels-per-roi",
        type=int,
        default=32,
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

    tracklets_path = root / "tracklets" / "tracklets.jsonl"
    crops_index_path = (
        root / "diagnostics" / "appearance_crops" / "index.json"
    )
    quality_path = (
        root / "diagnostics" / "reid_quality" / "quality.json"
    )
    candidates_path = (
        root
        / "diagnostics"
        / "actor_candidates"
        / "sparse-actor-candidates-v1"
        / "candidates.jsonl"
    )

    for required in (
        tracklets_path,
        crops_index_path,
        quality_path,
        candidates_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    tracklets = _load_tracklets(tracklets_path)
    known_tracklet_ids = {
        item.tracklet_id for item in tracklets
    }

    crop_payload = _load_json(crops_index_path)
    crop_records = crop_payload.get("records")
    if not isinstance(crop_records, list):
        raise ValueError("appearance crop index has no records list")

    quality_payload = _load_json(quality_path)
    quality_records = quality_payload.get("records")
    if not isinstance(quality_records, list):
        raise ValueError("reid quality artifact has no records list")

    quality_by_detection: dict[str, Mapping[str, Any]] = {}
    for record in quality_records:
        if not isinstance(record, Mapping):
            continue
        detection_id = str(record["detection_id"])
        quality_by_detection[detection_id] = record

    config = KitEvidenceConfig(
        min_crop_height_px=args.min_crop_height,
        min_crop_width_px=args.min_crop_width,
        min_non_grass_pixels_per_roi=(
            args.min_non_grass_pixels_per_roi
        ),
    )

    samples: list[KitSample] = []
    rejected: list[KitRejectedSample] = []

    for raw in crop_records:
        if not isinstance(raw, Mapping):
            raise ValueError("crop record must be an object")
        if raw.get("status") != "saved":
            continue

        tracklet_id = str(raw["tracklet_id"])
        if tracklet_id not in known_tracklet_ids:
            raise ValueError(
                f"crop references unknown tracklet {tracklet_id!r}"
            )

        detection_id = str(raw["detection_id"])
        crop_ref = str(raw["crop_ref"])
        crop_path = root / crop_ref
        image = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(
                f"could not read appearance crop: {crop_path}"
            )

        quality_record = quality_by_detection.get(detection_id)
        reid_quality = (
            float(quality_record["reid_quality"])
            if quality_record is not None
            and quality_record.get("reid_quality") is not None
            else None
        )
        selection_quality = float(
            raw.get(
                "selection_quality",
                raw.get("quality", 0.0),
            )
        )

        result = extract_kit_sample(
            image,
            tracklet_id=tracklet_id,
            detection_id=detection_id,
            frame_index=int(raw["frame_index"]),
            timestamp_s=float(raw["timestamp_s"]),
            crop_ref=crop_ref,
            selection_quality=selection_quality,
            reid_quality=reid_quality,
            config=config,
        )
        if isinstance(result, KitSample):
            samples.append(result)
        else:
            rejected.append(result)

    profiles = build_kit_tracklet_profiles(samples)
    profile_by_id = {
        profile.tracklet_id: profile
        for profile in profiles
    }

    pair_rows = []
    candidate_count = 0
    candidate_with_both_profiles = 0
    for candidate in _iter_jsonl(candidates_path):
        candidate_count += 1
        first_id = str(candidate["tracklet_a"])
        second_id = str(candidate["tracklet_b"])
        first = profile_by_id.get(first_id)
        second = profile_by_id.get(second_id)
        if first is None or second is None:
            continue
        candidate_with_both_profiles += 1
        pair_rows.append(
            compare_kit_profiles(first, second)
        )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else root
        / "diagnostics"
        / "kit_evidence"
        / KIT_EVIDENCE_VERSION
    )
    samples_path = output_dir / "samples.jsonl"
    rejected_path = output_dir / "rejected_samples.jsonl"
    profiles_path = output_dir / "profiles.jsonl"
    pairs_path = output_dir / "candidate_pairs.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    sample_count = _write_jsonl(
        samples_path,
        (item.to_dict() for item in samples),
    )
    rejected_count = _write_jsonl(
        rejected_path,
        (item.to_dict() for item in rejected),
    )
    profile_count = _write_jsonl(
        profiles_path,
        (item.to_dict() for item in profiles),
    )
    pair_count = _write_jsonl(
        pairs_path,
        (item.to_dict() for item in pair_rows),
    )

    rejection_reasons = Counter()
    for item in rejected:
        for reason in item.reasons:
            rejection_reasons[reason] += 1

    base_summary = kit_profile_summary(
        samples,
        profiles,
        total_tracklet_count=len(tracklets),
    )
    pair_cosines = [item.kit_cosine for item in pair_rows]
    torso_cosines = [item.torso_cosine for item in pair_rows]
    shorts_cosines = [item.shorts_cosine for item in pair_rows]

    summary = {
        **base_summary,
        "match_id": context.match_id,
        "clip_id": context.clip_id,
        "saved_crop_record_count": sum(
            1
            for item in crop_records
            if isinstance(item, Mapping)
            and item.get("status") == "saved"
        ),
        "rejected_sample_count": len(rejected),
        "rejection_reason_counts": dict(
            sorted(rejection_reasons.items())
        ),
        "candidate_graph_pair_count": candidate_count,
        "candidate_pair_with_both_kit_profiles_count": (
            candidate_with_both_profiles
        ),
        "candidate_pair_kit_evidence_count": len(pair_rows),
        "candidate_pair_kit_cosine": _stats(pair_cosines),
        "candidate_pair_torso_cosine": _stats(torso_cosines),
        "candidate_pair_shorts_cosine": _stats(shorts_cosines),
        "config": config.to_dict(),
        "source_crop_index": str(crops_index_path),
        "source_reid_quality": str(quality_path),
        "source_candidate_graph": str(candidates_path),
        "identity_decisions": False,
        "team_labels_assigned": False,
        "kit_clusters_assigned": False,
        "jersey_numbers_used": False,
    }
    _write_json(summary_path, summary)

    _write_json(
        index_path,
        {
            "schema_version": 1,
            "kit_evidence_version": KIT_EVIDENCE_VERSION,
            "samples": samples_path.name,
            "rejected_samples": rejected_path.name,
            "profiles": profiles_path.name,
            "candidate_pairs": pairs_path.name,
            "summary": summary_path.name,
            "sample_count": sample_count,
            "rejected_sample_count": rejected_count,
            "profile_count": profile_count,
            "candidate_pair_count": pair_count,
        },
    )

    print(
        "[kit-evidence-7C.1] "
        f"match={context.match_id} clip={context.clip_id}"
    )
    print(
        "[kit-evidence-7C.1] "
        f"saved_crops={summary['saved_crop_record_count']} "
        f"samples={sample_count} rejected={rejected_count} "
        f"profiles={profile_count}/{len(tracklets)} "
        f"coverage={summary['profile_coverage']:.6f}"
    )
    print(
        "[kit-evidence-7C.1] "
        f"candidate_pairs={candidate_count} "
        f"kit_evidence={pair_count} "
        f"coverage="
        f"{(pair_count/candidate_count if candidate_count else 0.0):.6f}"
    )
    print(
        "[kit-evidence-7C.1] "
        f"sample_quality="
        f"{json.dumps(summary['sample_quality'], sort_keys=True)}"
    )
    print(
        "[kit-evidence-7C.1] "
        f"profile_reliability="
        f"{json.dumps(summary['profile_reliability'], sort_keys=True)}"
    )
    print(
        "[kit-evidence-7C.1] "
        f"internal_similarity="
        f"{json.dumps(summary['profile_internal_similarity_mean'], sort_keys=True)}"
    )
    print(
        "[kit-evidence-7C.1] "
        f"candidate_kit_cosine="
        f"{json.dumps(summary['candidate_pair_kit_cosine'], sort_keys=True)}"
    )
    print(f"[kit-evidence-7C.1] artifact={index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
