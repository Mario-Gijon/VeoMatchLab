#!/usr/bin/env python3
"""Audit Sprint A rectified player-centred crops for future visual ReID.

This intentionally reruns the existing 15 Hz detector/tracker pipeline in an
isolated debug directory because tracking_frames.jsonl does not persist the
representative bbox/ray metadata needed to reconstruct visual crops later.

Nothing in this script modifies tracks.person, V2.7 identity resolution,
backend data, team labels, roles or jersey numbers.

Outputs:
    <out>/report.json
    <out>/track_XXX/best_01_....jpg
    <out>/track_XXX/contact_sheet.jpg
    <out>/_tracking_tmp/tracking_frames.jsonl   # isolated diagnostic rerun
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.tracking_v0 import run_global_tracker  # noqa: E402
from experiments.virtual_camera_coverage.detect import (  # noqa: E402
    DEFAULT_C2_WEIGHTS,
)
from tracking_v0 import PositionKind  # noqa: E402
from tracking_v0.reid_crops import (  # noqa: E402
    ReIDCrop,
    ReIDCropCandidate,
    ReIDCropConfig,
    candidate_from_detection,
    candidates_by_track,
    render_reid_crop,
    select_diverse_candidates,
    select_diverse_crops,
)


DEFAULT_OUT = Path("experiments/tracking_v0/reid_crop_debug")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_C2_WEIGHTS,
    )
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Default: from --start-s to end of video.",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=1.0 / 15.0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=run_global_tracker.DEFAULT_TRACKING_THRESHOLD,
    )

    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--split-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--guard-margin-m",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--grid-step-m",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--field-margin-m",
        type=float,
        default=0.60,
    )

    parser.add_argument(
        "--track-ids",
        nargs="*",
        type=int,
        default=None,
        help="Optional raw GlobalTracker IDs to audit; default: all.",
    )
    parser.add_argument(
        "--max-crops-per-track",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--min-time-gap-s",
        type=float,
        default=0.75,
    )
    parser.add_argument(
        "--min-proposal-quality",
        type=float,
        default=0.08,
    )
    parser.add_argument(
        "--min-final-quality",
        type=float,
        default=0.10,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
    )
    args = parser.parse_args()

    if not args.video.is_file():
        parser.error(f"video does not exist: {args.video}")
    if not args.calibration.is_file():
        parser.error(f"calibration does not exist: {args.calibration}")
    if not args.weights.is_file():
        parser.error(f"weights do not exist: {args.weights}")
    if args.start_s < 0.0:
        parser.error("--start-s must be >= 0")
    if args.sample_period_s <= 0.0:
        parser.error("--sample-period-s must be > 0")
    if args.max_crops_per_track < 1:
        parser.error("--max-crops-per-track must be >= 1")
    if args.min_time_gap_s < 0.0:
        parser.error("--min-time-gap-s must be >= 0")
    for name in ("min_proposal_quality", "min_final_quality", "threshold"):
        value = float(getattr(args, name))
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")

    return args


def video_duration_seconds(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0.0 or frames <= 0.0:
        raise RuntimeError("Video reports invalid FPS/frame count")
    return frames / fps


def _tracking_args(
    args: argparse.Namespace,
    *,
    duration_s: float,
    tracking_out: Path,
) -> argparse.Namespace:
    # Keep this explicit so the diagnostic exactly documents the tracking
    # settings that generated its raw IDs.
    return argparse.Namespace(
        video=args.video,
        calibration=args.calibration,
        weights=args.weights,
        device=args.device,
        start_s=args.start_s,
        duration_s=duration_s,
        sample_period_s=args.sample_period_s,
        threshold=args.threshold,
        width=args.width,
        height=args.height,
        split_fraction=args.split_fraction,
        guard_margin_m=args.guard_margin_m,
        grid_step_m=args.grid_step_m,
        field_margin_m=args.field_margin_m,
        rescue_confirm_threshold=args.threshold,
        rescue_confirm_max_world_distance_m=(
            run_global_tracker.DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M
            if hasattr(
                run_global_tracker,
                "DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M",
            )
            else 8.0
        ),
        out_dir=tracking_out,
    )


def _measurement_by_id(
    measurements: tuple[Any, ...],
) -> dict[str, Any]:
    return {
        str(item.measurement_id): item
        for item in measurements
    }


def collect_candidates(
    args: argparse.Namespace,
    *,
    duration_s: float,
    tracking_out: Path,
    config: ReIDCropConfig,
) -> tuple[
    list[ReIDCropCandidate],
    dict[str, Any],
    dict[str, Any],
]:
    candidates: list[ReIDCropCandidate] = []
    last_metadata: dict[str, Any] = {}
    last_specs: dict[str, Any] = {}
    requested_ids = (
        set(args.track_ids)
        if args.track_ids
        else None
    )

    def frame_callback(
        *,
        frame,
        metadata,
        spec_by_name,
        tracking_frame,
        measurements,
        record,
    ) -> None:
        del frame, record

        nonlocal last_metadata, last_specs
        last_metadata = {"metadata": metadata}
        last_specs = dict(spec_by_name)

        by_measurement = _measurement_by_id(
            tuple(measurements)
        )

        for snapshot in tracking_frame.tracks:
            if snapshot.position_kind is not PositionKind.OBSERVED:
                continue
            if requested_ids is not None and snapshot.track_id not in requested_ids:
                continue
            if snapshot.measurement_id is None:
                continue

            measurement = by_measurement.get(
                str(snapshot.measurement_id)
            )
            if measurement is None:
                continue

            visual = measurement.metadata
            camera = visual.get("representative_camera")
            bbox = visual.get("representative_bbox_xyxy")
            confidence = visual.get(
                "representative_detection_confidence"
            )

            if (
                camera is None
                or bbox is None
                or confidence is None
                or camera not in spec_by_name
            ):
                continue

            try:
                candidate = candidate_from_detection(
                    track_id=snapshot.track_id,
                    frame_index=snapshot.frame_index,
                    timestamp=snapshot.timestamp,
                    source_camera=camera,
                    source_spec=spec_by_name[camera],
                    source_bbox_xyxy=bbox,
                    detection_confidence=float(confidence),
                    source_width=args.width,
                    source_height=args.height,
                    config=config,
                )
            except (ValueError, FloatingPointError):
                continue

            if candidate.proposal_quality >= args.min_proposal_quality:
                candidates.append(candidate)

    tracking_args = _tracking_args(
        args,
        duration_s=duration_s,
        tracking_out=tracking_out,
    )

    summary = run_global_tracker.run(
        tracking_args,
        frame_callback=frame_callback,
    )

    if not last_metadata or not last_specs:
        raise RuntimeError("Tracker produced no usable frames/specifications")

    return (
        candidates,
        {
            "metadata": last_metadata["metadata"],
            "spec_by_name": last_specs,
        },
        summary,
    )


def select_proposals(
    candidates: list[ReIDCropCandidate],
    *,
    config: ReIDCropConfig,
) -> dict[int, tuple[ReIDCropCandidate, ...]]:
    grouped = candidates_by_track(candidates)
    return {
        track_id: select_diverse_candidates(
            rows,
            max_crops=config.max_crops_per_track,
            min_time_gap_seconds=config.min_time_gap_seconds,
            config=config,
        )
        for track_id, rows in grouped.items()
    }


def render_selected(
    *,
    video: Path,
    metadata: Any,
    spec_by_name: dict[str, Any],
    selected: dict[int, tuple[ReIDCropCandidate, ...]],
    source_width: int,
    source_height: int,
    config: ReIDCropConfig,
) -> dict[int, list[ReIDCrop]]:
    by_frame: dict[int, list[ReIDCropCandidate]] = defaultdict(list)

    for rows in selected.values():
        for candidate in rows:
            by_frame[candidate.frame_index].append(candidate)

    if not by_frame:
        return {}

    wanted_frames = set(by_frame)
    maximum_frame = max(wanted_frames)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video}")

    output: dict[int, list[ReIDCrop]] = defaultdict(list)
    frame_index = 0

    try:
        while frame_index <= maximum_frame:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if frame_index in wanted_frames:
                for candidate in by_frame[frame_index]:
                    source_spec = spec_by_name.get(
                        candidate.source_camera
                    )
                    if source_spec is None:
                        continue

                    try:
                        crop = render_reid_crop(
                            frame=frame,
                            metadata=metadata,
                            source_spec=source_spec,
                            candidate=candidate,
                            source_width=source_width,
                            source_height=source_height,
                            config=config,
                        )
                    except (ValueError, FloatingPointError):
                        continue

                    output[candidate.track_id].append(crop)

            frame_index += 1
    finally:
        cap.release()

    return dict(output)


def _captioned_tile(crop: ReIDCrop) -> np.ndarray:
    image = crop.image.copy()
    caption_h = 38
    tile = np.zeros(
        (
            image.shape[0] + caption_h,
            image.shape[1],
            3,
        ),
        dtype=np.uint8,
    )
    tile[:image.shape[0]] = image

    text = (
        f"f{crop.metrics.frame_index} "
        f"q={crop.metrics.quality:.2f}"
    )
    cv2.putText(
        tile,
        text,
        (4, image.shape[0] + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return tile


def contact_sheet(
    crops: tuple[ReIDCrop, ...],
    *,
    columns: int = 5,
) -> np.ndarray:
    if not crops:
        raise ValueError("contact_sheet requires at least one crop")

    tiles = [_captioned_tile(crop) for crop in crops]
    h, w = tiles[0].shape[:2]
    rows = int(math.ceil(len(tiles) / columns))

    sheet = np.zeros(
        (rows * h, columns * w, 3),
        dtype=np.uint8,
    )
    for index, tile in enumerate(tiles):
        row, col = divmod(index, columns)
        sheet[
            row * h:(row + 1) * h,
            col * w:(col + 1) * w,
        ] = tile

    return sheet


def persist(
    *,
    out_dir: Path,
    all_candidates: list[ReIDCropCandidate],
    rendered: dict[int, list[ReIDCrop]],
    config: ReIDCropConfig,
    args: argparse.Namespace,
    tracker_summary: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    grouped_candidates = candidates_by_track(all_candidates)

    report_tracks: dict[str, Any] = {}

    all_track_ids = sorted(
        set(grouped_candidates)
        | set(rendered)
    )

    for track_id in all_track_ids:
        track_dir = out_dir / f"track_{track_id:03d}"
        track_dir.mkdir(parents=True, exist_ok=True)

        final = tuple(
            crop
            for crop in rendered.get(track_id, [])
            if crop.metrics.quality >= args.min_final_quality
        )
        chosen = select_diverse_crops(
            final,
            max_crops=config.max_crops_per_track,
            min_time_gap_seconds=config.min_time_gap_seconds,
        ) if final else ()

        for index, crop in enumerate(
            sorted(
                chosen,
                key=lambda item: -item.metrics.quality,
            ),
            start=1,
        ):
            path = track_dir / (
                f"best_{index:02d}_"
                f"frame_{crop.metrics.frame_index:06d}_"
                f"q_{crop.metrics.quality:.3f}.jpg"
            )
            cv2.imwrite(str(path), crop.image)

        if chosen:
            cv2.imwrite(
                str(track_dir / "contact_sheet.jpg"),
                contact_sheet(chosen),
            )

        qualities = [
            crop.metrics.quality
            for crop in chosen
        ]

        report_tracks[str(track_id)] = {
            "candidate_crops": len(
                grouped_candidates.get(track_id, ())
            ),
            "rendered_crops": len(
                rendered.get(track_id, ())
            ),
            "accepted_crops": len(chosen),
            "best_quality": (
                max(qualities)
                if qualities
                else None
            ),
            "median_quality": (
                float(np.median(qualities))
                if qualities
                else None
            ),
            "selected": [
                crop.metrics.to_dict()
                for crop in chosen
            ],
        }

    payload = {
        "version": "tracking-reid-crops-sprint-a-v1",
        "video": str(args.video.resolve()),
        "calibration": str(args.calibration.resolve()),
        "weights": str(args.weights.resolve()),
        "config": {
            key: value
            for key, value in vars(config).items()
        } if hasattr(config, "__dict__") else {
            field: getattr(config, field)
            for field in config.__dataclass_fields__
        },
        "tracker_summary": tracker_summary,
        "tracks": report_tracks,
    }

    (out_dir / "report.json").write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    args = parse_args()

    full_duration = video_duration_seconds(
        args.video
    )
    duration_s = (
        max(
            1e-3,
            full_duration - args.start_s,
        )
        if args.duration_s is None
        else args.duration_s
    )
    if duration_s <= 0.0:
        raise SystemExit("duration must be > 0")

    config = ReIDCropConfig(
        max_crops_per_track=args.max_crops_per_track,
        min_time_gap_seconds=args.min_time_gap_s,
    )

    out_dir = args.out_dir.expanduser().resolve()
    tracking_out = out_dir / "_tracking_tmp"

    if tracking_out.exists():
        shutil.rmtree(tracking_out)
    tracking_out.mkdir(parents=True, exist_ok=True)

    print("Stage 1/3: rerun existing detector/tracker and collect visual proposals")
    candidates, runtime, tracker_summary = collect_candidates(
        args,
        duration_s=duration_s,
        tracking_out=tracking_out,
        config=config,
    )

    print(
        f"Collected {len(candidates)} proposal(s) "
        f"from {len(set(row.track_id for row in candidates))} track(s)"
    )

    print("Stage 2/3: choose temporally diverse proposals and render rectified crops")
    selected = select_proposals(
        candidates,
        config=config,
    )
    rendered = render_selected(
        video=args.video,
        metadata=runtime["metadata"],
        spec_by_name=runtime["spec_by_name"],
        selected=selected,
        source_width=args.width,
        source_height=args.height,
        config=config,
    )

    print("Stage 3/3: score, select, and write per-track contact sheets")
    payload = persist(
        out_dir=out_dir,
        all_candidates=candidates,
        rendered=rendered,
        config=config,
        args=args,
        tracker_summary=tracker_summary,
    )

    usable = [
        track_id
        for track_id, info in payload["tracks"].items()
        if info["accepted_crops"] > 0
    ]
    print()
    print("Sprint A ReID crop audit complete")
    print(f"tracks with usable crops : {len(usable)}")
    print(f"output                  : {out_dir}")
    print(f"report                  : {out_dir / 'report.json'}")
    print("No ReID/team/role classification was performed.")


if __name__ == "__main__":
    main()
