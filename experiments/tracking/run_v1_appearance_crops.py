#!/usr/bin/env python3
"""Extract actual visual crops for selected Tracking V1 appearance moments.

This is still a diagnostic stage, not the published appearance stage:

* no ReID model is loaded;
* no embedding is produced;
* no identity/team/shirt-number decision is made;
* the Tracking V1 manifest's ``appearance`` stage remains ``not_run``.

The runner reuses the already published detections + tracklets, deterministically
recomputes the Hito 2B selection, re-renders ONLY the required representative
virtual views, and crops the exact ``virtual_bbox_xyxy`` associated with each
selected canonical detection.

Outputs:
    tracking/v1/diagnostics/appearance_crops/
        index.json
        weakest.jpg
        strongest.jpg
        trk-000001/*.jpg
        ...
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

DEFAULT_DEV_CONFIG = ROOT / "experiments" / "tracking" / "dev_clip.json"

from tracking.appearance.crops import (  # noqa: E402
    AppearanceCropConfig,
    crop_virtual_person,
    safe_filename_token,
)
from tracking.appearance.selection import (  # noqa: E402
    AppearanceSelectionConfig,
    AppearanceSelectionSet,
    select_appearance_moments,
)
from tracking.detections import (  # noqa: E402
    CanonicalDetection,
    CanonicalDetectionFrame,
)
from tracking.tracklets import Tracklet, TrackletSet  # noqa: E402


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def _iter_jsonl(path: Path):
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
                    f"{path}: line {line_number} must encode an object"
                )
            yield payload


def load_dev_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dev config must encode a JSON object")
    for key in ("match_id", "clip_id", "clip_video", "calibration"):
        if key not in payload:
            raise ValueError(f"dev config is missing {key!r}")
    return payload


def tracking_root(
    data_root: Path,
    *,
    match_id: str,
    clip_id: str,
) -> Path:
    return (
        data_root
        / "matches"
        / match_id
        / "clips"
        / clip_id
        / "tracking"
        / "v1"
    )


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest must encode a JSON object")
    return payload


def detection_render_size(
    manifest: Mapping[str, Any],
) -> tuple[int, int]:
    try:
        stage = manifest["stages"]["detections"]
        config = stage["config"]
        width = int(config["width"])
        height = int(config["height"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "manifest detections stage must expose config width/height"
        ) from exc

    if width <= 0 or height <= 0:
        raise ValueError("detection render size must be positive")
    return width, height


def load_frames(path: Path) -> tuple[CanonicalDetectionFrame, ...]:
    return tuple(
        CanonicalDetectionFrame.from_dict(payload)
        for payload in _iter_jsonl(path)
    )


def load_tracklets(path: Path) -> TrackletSet:
    return TrackletSet(
        tracklets=tuple(
            Tracklet.from_dict(payload)
            for payload in _iter_jsonl(path)
        )
    )


def detection_index(
    frames: Sequence[CanonicalDetectionFrame],
) -> dict[str, CanonicalDetection]:
    result: dict[str, CanonicalDetection] = {}
    for frame in frames:
        for detection in frame.detections:
            if detection.detection_id in result:
                raise ValueError(
                    f"duplicate detection id: {detection.detection_id!r}"
                )
            result[detection.detection_id] = detection
    return result


def ordered_tight_specs(specs: Sequence[Any]) -> list[Any]:
    tight = [spec for spec in specs if spec.mode == "tight"]
    far = sorted(
        [spec for spec in tight if spec.cell.band == "far"],
        key=lambda spec: spec.cell.index_in_band,
    )
    near = sorted(
        [spec for spec in tight if spec.cell.band == "near"],
        key=lambda spec: spec.cell.index_in_band,
    )
    if len(far) != 4 or len(near) != 2:
        raise RuntimeError(
            "Expected exactly 4 FAR + 2 NEAR TIGHT cameras, got "
            f"{len(far)} FAR + {len(near)} NEAR"
        )
    return [*far, *near]


def selected_records(
    selection: AppearanceSelectionSet,
    detections: Mapping[str, CanonicalDetection],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []

    for tracklet_selection in selection.selections:
        for moment in tracklet_selection.moments:
            detection = detections.get(moment.detection_id)
            if detection is None:
                raise ValueError(
                    f"selected detection is missing: {moment.detection_id}"
                )
            if detection.representative_view is None:
                records.append(
                    {
                        "tracklet_id": moment.tracklet_id,
                        "detection_id": moment.detection_id,
                        "frame_index": moment.frame_index,
                        "timestamp_s": moment.timestamp_s,
                        "quality": moment.quality,
                        "status": "skipped_no_representative_view",
                    }
                )
                continue
            if detection.virtual_bbox_xyxy is None:
                records.append(
                    {
                        "tracklet_id": moment.tracklet_id,
                        "detection_id": moment.detection_id,
                        "frame_index": moment.frame_index,
                        "timestamp_s": moment.timestamp_s,
                        "quality": moment.quality,
                        "representative_view": detection.representative_view,
                        "status": "skipped_no_virtual_bbox",
                    }
                )
                continue

            records.append(
                {
                    "tracklet_id": moment.tracklet_id,
                    "detection_id": moment.detection_id,
                    "frame_index": moment.frame_index,
                    "timestamp_s": moment.timestamp_s,
                    "quality": moment.quality,
                    "representative_view": detection.representative_view,
                    "virtual_bbox_xyxy": list(
                        detection.virtual_bbox_xyxy
                    ),
                    "status": "pending",
                }
            )

    return records


def _fit_panel(
    image: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    scale = min(
        width / image.shape[1],
        height / image.shape[0],
    )
    resized_w = max(1, int(round(image.shape[1] * scale)))
    resized_h = max(1, int(round(image.shape[0] * scale)))
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=(
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_LINEAR
        ),
    )
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def make_contact_sheet(
    records: Sequence[Mapping[str, Any]],
    *,
    columns: int = 8,
    panel_width: int = 160,
    panel_height: int = 230,
) -> np.ndarray | None:
    usable = [
        item
        for item in records
        if item.get("status") == "saved"
        and item.get("_image") is not None
    ]
    if not usable:
        return None

    rows = (len(usable) + columns - 1) // columns
    sheet = np.zeros(
        (rows * panel_height, columns * panel_width, 3),
        dtype=np.uint8,
    )

    image_height = panel_height - 42

    for index, item in enumerate(usable):
        row = index // columns
        column = index % columns
        panel = np.zeros(
            (panel_height, panel_width, 3),
            dtype=np.uint8,
        )
        fitted = _fit_panel(
            item["_image"],
            width=panel_width,
            height=image_height,
        )
        panel[:image_height] = fitted

        label = (
            f"{item['tracklet_id']} "
            f"q={item['quality']:.3f}"
        )
        cv2.putText(
            panel,
            label,
            (4, image_height + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"t={item['timestamp_s']:.2f}s",
            (4, image_height + 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        y1 = row * panel_height
        x1 = column * panel_width
        sheet[
            y1:y1 + panel_height,
            x1:x1 + panel_width,
        ] = panel

    return sheet


def _public_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if not key.startswith("_")
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dev-config",
        default=str(DEFAULT_DEV_CONFIG),
    )
    parser.add_argument(
        "--data-root",
        default=str(ROOT / "data"),
    )
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--min-samples", type=int, default=3)
    parser.add_argument("--min-quality", type=float, default=0.42)
    parser.add_argument("--min-spacing-s", type=float, default=0.75)
    parser.add_argument("--diversity-weight", type=float, default=0.25)
    parser.add_argument("--padding-x", type=float, default=0.08)
    parser.add_argument("--padding-y", type=float, default=0.06)
    parser.add_argument("--audit-count", type=int, default=40)
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Default: <tracking-v1>/diagnostics/appearance_crops"
        ),
    )
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    # Heavy Veo/OpenCV geometry dependencies are runtime-only.  Keeping them
    # out of module import makes helper tests cheap and prevents the diagnostic
    # contract from depending on the rendering stack.
    from geometry.veo_metadata import load_veo_metadata
    from experiments.virtual_camera_coverage.coverage import (
        SequentialFrameReader,
        build_map,
        derive_specs,
        render_map,
    )

    args = parse_args(argv)
    dev_path = _repo_path(args.dev_config)
    data_root = _repo_path(args.data_root)
    dev = load_dev_config(dev_path)

    root = tracking_root(
        data_root,
        match_id=dev["match_id"],
        clip_id=dev["clip_id"],
    )
    frames_path = root / "detections" / "frames.jsonl"
    tracklets_path = root / "tracklets" / "tracklets.jsonl"
    manifest_path = root / "manifest.json"

    video_path = _repo_path(dev["clip_video"])
    calibration_path = _repo_path(dev["calibration"])

    for required in (
        frames_path,
        tracklets_path,
        manifest_path,
        video_path,
        calibration_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    output_dir = (
        _repo_path(args.output_dir)
        if args.output_dir is not None
        else root / "diagnostics" / "appearance_crops"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = load_frames(frames_path)
    tracklets = load_tracklets(tracklets_path)
    detections = detection_index(frames)

    manifest = load_manifest(manifest_path)
    width, height = detection_render_size(manifest)

    selection_config = AppearanceSelectionConfig(
        max_samples_per_tracklet=args.max_samples,
        min_samples_if_available=args.min_samples,
        min_quality=args.min_quality,
        min_temporal_spacing_s=args.min_spacing_s,
        temporal_diversity_weight=args.diversity_weight,
        virtual_width_px=width,
        virtual_height_px=height,
    )
    crop_config = AppearanceCropConfig(
        padding_x_fraction=args.padding_x,
        padding_y_fraction=args.padding_y,
    )

    selection = select_appearance_moments(
        frames,
        tracklets,
        config=selection_config,
    )
    records = selected_records(selection, detections)

    pending = [
        record
        for record in records
        if record["status"] == "pending"
    ]

    print(
        "[appearance-crops] "
        f"selected={selection.selected_count} "
        f"renderable={len(pending)} "
        f"video={video_path}"
    )
    print(
        f"[appearance-crops] virtual_size={width}x{height}"
    )

    metadata = load_veo_metadata(calibration_path)
    specs, _, _, _ = derive_specs(
        metadata,
        width,
        height,
    )
    tight_specs = ordered_tight_specs(specs)
    spec_by_name = {
        spec.name: spec
        for spec in tight_specs
    }

    required_views = sorted(
        {
            record["representative_view"]
            for record in pending
        }
    )
    missing_views = [
        view
        for view in required_views
        if view not in spec_by_name
    ]
    if missing_views:
        raise ValueError(
            f"selected detections reference unknown views: {missing_views}"
        )

    camera_maps = {
        view: build_map(
            metadata,
            spec_by_name[view],
            width,
            height,
        )
        for view in required_views
    }

    by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in pending:
        by_frame[int(record["frame_index"])].append(record)

    ordered_frames = sorted(by_frame)

    with SequentialFrameReader(video_path) as reader:
        for frame_index in ordered_frames:
            frame_records = by_frame[frame_index]
            timestamp_s = float(frame_records[0]["timestamp_s"])

            source_frame, frame_info = reader.read(timestamp_s)
            actual_frame = int(frame_info["actual_frame"])
            if actual_frame != frame_index:
                raise RuntimeError(
                    "video/frame mismatch while extracting appearance crops: "
                    f"expected {frame_index}, got {actual_frame}"
                )

            rendered_cache: dict[str, np.ndarray] = {}

            for record in frame_records:
                view = str(record["representative_view"])
                if view not in rendered_cache:
                    rendered_cache[view] = render_map(
                        source_frame,
                        camera_maps[view],
                    )

                try:
                    crop_result = crop_virtual_person(
                        rendered_cache[view],
                        record["virtual_bbox_xyxy"],
                        config=crop_config,
                    )
                except ValueError as exc:
                    record["status"] = "skipped_invalid_crop"
                    record["error"] = str(exc)
                    continue

                tracklet_dir = (
                    output_dir
                    / safe_filename_token(record["tracklet_id"])
                )
                tracklet_dir.mkdir(parents=True, exist_ok=True)

                filename = (
                    f"f{frame_index:06d}_"
                    f"{safe_filename_token(record['detection_id'])}.jpg"
                )
                path = tracklet_dir / filename

                ok = cv2.imwrite(
                    str(path),
                    crop_result.crop,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95],
                )
                if not ok:
                    raise RuntimeError(f"could not write crop: {path}")

                record["status"] = "saved"
                record["crop_ref"] = str(
                    path.relative_to(root)
                )
                record["crop_width_px"] = crop_result.width
                record["crop_height_px"] = crop_result.height
                record["crop_bbox_xyxy"] = list(
                    crop_result.bbox_xyxy_int
                )
                record["crop_clipped"] = crop_result.clipped
                record["_image"] = crop_result.crop

    saved = [
        record
        for record in records
        if record["status"] == "saved"
    ]
    failed = [
        record
        for record in records
        if record["status"] != "saved"
    ]

    ranked = sorted(
        saved,
        key=lambda item: (
            float(item["quality"]),
            item["tracklet_id"],
            int(item["frame_index"]),
        ),
    )
    audit_count = max(1, int(args.audit_count))
    weakest = ranked[:audit_count]
    strongest = list(reversed(ranked[-audit_count:]))

    weakest_sheet = make_contact_sheet(weakest)
    strongest_sheet = make_contact_sheet(strongest)

    weakest_path = output_dir / "weakest.jpg"
    strongest_path = output_dir / "strongest.jpg"

    if weakest_sheet is not None:
        cv2.imwrite(str(weakest_path), weakest_sheet)
    if strongest_sheet is not None:
        cv2.imwrite(str(strongest_path), strongest_sheet)

    public_records = [_public_record(record) for record in records]
    payload = {
        "match_id": dev["match_id"],
        "clip_id": dev["clip_id"],
        "video": str(video_path),
        "calibration": str(calibration_path),
        "virtual_width": width,
        "virtual_height": height,
        "selection_summary": selection.summary(),
        "crop_config": crop_config.to_dict(),
        "selected_count": len(records),
        "saved_count": len(saved),
        "failed_count": len(failed),
        "weakest_contact_sheet": (
            str(weakest_path.relative_to(root))
            if weakest_sheet is not None
            else None
        ),
        "strongest_contact_sheet": (
            str(strongest_path.relative_to(root))
            if strongest_sheet is not None
            else None
        ),
        "records": public_records,
    }

    index_path = output_dir / "index.json"
    index_path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        "[appearance-crops] completed "
        f"saved={len(saved)} failed={len(failed)}"
    )
    if weakest_sheet is not None:
        print(f"[appearance-crops] weakest={weakest_path}")
    if strongest_sheet is not None:
        print(f"[appearance-crops] strongest={strongest_path}")
    print(f"[appearance-crops] index={index_path}")

    return payload


if __name__ == "__main__":
    run()
