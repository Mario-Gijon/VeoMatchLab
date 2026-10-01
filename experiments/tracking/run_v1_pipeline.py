#!/usr/bin/env python3
"""Run the clean Tracking V1 detections -> tracklets pipeline on a Veo clip.

The primary input is a materialized ``--match-id`` + ``--clip-id`` resolved by
``tracking.clip_context``. ``--dev-config`` remains only as a legacy
development fallback. The runner uses the already-validated six-view same-frame
detection/association layer, converts the resulting physical-person hypotheses
to ``CanonicalDetectionFrame`` and then hands those frames to the pure V1
pipeline.

No Tracking V0 temporal tracker is imported or used.

The defaults are intentionally a *small smoke run*. Increase ``--duration-s``
and reduce ``--sample-period-s`` only after the smoke run is healthy.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_DEV_CONFIG = ROOT / "experiments" / "tracking" / "dev_clip.json"


def resolve_runner_clip_context(args: argparse.Namespace):
    """Resolve CLI clip input without importing models or running detection."""
    from tracking.clip_context import resolve_tracking_clip_context

    explicit_match = args.match_id is not None
    explicit_clip = args.clip_id is not None
    if explicit_match != explicit_clip:
        raise ValueError("--match-id and --clip-id must be provided together")

    data_root = _repo_path(args.data_root)
    if explicit_match:
        return resolve_tracking_clip_context(data_root, args.match_id, args.clip_id), "explicit"

    # Legacy/development fallback. Its IDs are resolved through the same
    # generic context; old paths and duration constants are not trusted.
    config_path = _repo_path(args.dev_config)
    dev = load_dev_config(config_path)
    return (
        resolve_tracking_clip_context(data_root, dev["match_id"], dev["clip_id"]),
        f"legacy dev-config {config_path}",
    )


def sample_times(
    start_s: float,
    duration_s: float,
    period_s: float,
) -> tuple[float, ...]:
    if start_s < 0:
        raise ValueError("start_s must be >= 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if period_s <= 0:
        raise ValueError("period_s must be > 0")

    count = int(math.ceil(duration_s / period_s - 1e-12))
    return tuple(start_s + index * period_s for index in range(count))


def _repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def load_dev_config(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "match_id",
        "clip_id",
        "clip_video",
        "calibration",
        "duration_s",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"dev config is missing keys: {missing}")
    return payload


def canonical_frames_from_veo(
    *,
    video: Path,
    calibration: Path,
    weights: Path,
    start_s: float,
    duration_s: float,
    sample_period_s: float,
    threshold: float,
    width: int,
    height: int,
    skip_rescue: bool,
) -> Iterable:
    """Yield canonical V1 frames from the existing same-frame vision layer."""
    # Heavy CV/model imports stay local so importing this runner remains cheap.
    from experiments.virtual_camera_coverage.association import (
        DEFAULT_FIELD_MARGIN_M,
        DEFAULT_MAX_CENTER_X_NORM,
        DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M,
        DEFAULT_MAX_SOURCE_ANCHOR_NORM,
        DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M,
        DEFAULT_MAX_WORLD_DISTANCE_M,
        DEFAULT_MIN_HORIZONTAL_IOM,
        DEFAULT_MIN_SOURCE_IOM,
        DEFAULT_MIN_SOURCE_IOU,
        DEFAULT_MIN_VERTICAL_IOM,
        Observation,
        associate_observations,
        build_observation,
    )
    from experiments.virtual_camera_coverage.coverage import (
        DEFAULT_FAR_CAMERAS,
        DEFAULT_GRID_STEP_M,
        DEFAULT_GUARD_M,
        DEFAULT_NEAR_CAMERAS,
        DEFAULT_SPLIT_FRACTION,
        SequentialFrameReader,
        build_map,
        derive_specs,
        render_map,
    )
    from experiments.virtual_camera_coverage.detect import (
        DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M,
        confirm_deferred_rescues_from_owner_images,
        ordered_tight_specs,
        predict_persons,
    )
    from geometry.veo_metadata import load_veo_metadata
    from person_detect.models.rfdetr_2xl import RFDETR2XLBackend
    from tracking.detections.adapter import canonical_frame_from_association

    metadata = load_veo_metadata(calibration)
    specs, layout, _, _ = derive_specs(
        metadata,
        width,
        height,
        split_fraction=DEFAULT_SPLIT_FRACTION,
        near_count=DEFAULT_NEAR_CAMERAS,
        far_count=DEFAULT_FAR_CAMERAS,
        guard_margin_m=DEFAULT_GUARD_M,
        grid_step_m=DEFAULT_GRID_STEP_M,
    )
    tight_specs = ordered_tight_specs(specs)
    spec_by_name = {spec.name: spec for spec in tight_specs}
    maps = {
        spec.name: build_map(metadata, spec, width, height)
        for spec in tight_specs
    }

    backend = RFDETR2XLBackend(weights)
    print(
        f"[tracking-v1] detector={backend.name} device={backend.device} "
        f"weights={weights}"
    )

    with SequentialFrameReader(video) as frame_reader:
        for requested_time in sample_times(
            start_s,
            duration_s,
            sample_period_s,
        ):
            frame, frame_info = frame_reader.read(requested_time)
            frame_index = int(frame_info["actual_frame"])
            timestamp_s = float(frame_info["actual_time_s"])

            rendered = {}
            observations: list[Observation] = []
            observation_id = 1

            for spec in tight_specs:
                image = render_map(frame, maps[spec.name])
                rendered[spec.name] = image

                for detection in predict_persons(
                    backend,
                    image,
                    threshold=threshold,
                ):
                    observations.append(
                        build_observation(
                            observation_id=observation_id,
                            frame_number=frame_index,
                            time_s=timestamp_s,
                            metadata=metadata,
                            layout=layout,
                            spec=spec,
                            camera_map=maps[spec.name],
                            xyxy=detection.xyxy,
                            confidence=detection.confidence,
                            field_margin_m=DEFAULT_FIELD_MARGIN_M,
                        )
                    )
                    observation_id += 1

            association = associate_observations(
                observations,
                layout,
                min_source_iou=DEFAULT_MIN_SOURCE_IOU,
                max_source_anchor_norm=DEFAULT_MAX_SOURCE_ANCHOR_NORM,
                max_world_distance_m=DEFAULT_MAX_WORLD_DISTANCE_M,
                min_source_iom=DEFAULT_MIN_SOURCE_IOM,
                min_horizontal_iom=DEFAULT_MIN_HORIZONTAL_IOM,
                min_vertical_iom=DEFAULT_MIN_VERTICAL_IOM,
                max_center_x_norm=DEFAULT_MAX_CENTER_X_NORM,
                max_containment_world_distance_m=(
                    DEFAULT_MAX_CONTAINMENT_WORLD_DISTANCE_M
                ),
                max_truncated_world_distance_m=(
                    DEFAULT_MAX_TRUNCATED_WORLD_DISTANCE_M
                ),
            )

            if not skip_rescue:
                association, _, _ = confirm_deferred_rescues_from_owner_images(
                    backend=backend,
                    metadata=metadata,
                    layout=layout,
                    rendered=rendered,
                    maps=maps,
                    spec_by_name=spec_by_name,
                    association=association,
                    frame_number=frame_index,
                    time_s=timestamp_s,
                    field_margin_m=DEFAULT_FIELD_MARGIN_M,
                    threshold=threshold,
                    max_world_distance_m=(
                        DEFAULT_RESCUE_CONFIRM_MAX_WORLD_DISTANCE_M
                    ),
                )

            canonical = canonical_frame_from_association(
                association,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
            )
            print(
                f"[tracking-v1] frame={frame_index} "
                f"t={timestamp_s:.3f}s "
                f"raw_obs={len(association.observations)} "
                f"people={len(canonical.detections)}"
            )
            yield canonical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", help="Materialized match identifier.")
    parser.add_argument("--clip-id", help="Materialized clip identifier.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Workspace data root containing matches/.",
    )
    parser.add_argument(
        "--dev-config",
        type=Path,
        default=DEFAULT_DEV_CONFIG,
        help="Legacy/development fallback when --match-id and --clip-id are omitted.",
    )
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=2.0,
        help="Smoke default is only 2 seconds.",
    )
    parser.add_argument(
        "--sample-period-s",
        type=float,
        default=0.20,
        help="Smoke default is 5 Hz; use 0.10 for a 10 Hz diagnostic.",
    )
    parser.add_argument("--threshold", type=float, default=0.40)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--skip-rescue",
        action="store_true",
        help="Skip the expensive same-frame owner-image rescue pass.",
    )
    parser.add_argument(
        "--restart-running",
        action="store_true",
        help="Reserved recovery flag; currently the artifact store should not "
             "normally be left RUNNING.",
    )
    return parser.parse_args()


def main() -> None:
    from experiments.virtual_camera_coverage.detect import DEFAULT_C2_WEIGHTS
    from tracking.artifacts import TrackingArtifactStore
    from tracking.pipeline import run_detection_and_tracklet_stages
    from tracking.tracklets.builder import TrackletBuilderConfig

    args = parse_args()
    context, input_mode = resolve_runner_clip_context(args)

    video = context.clip_video
    calibration = context.calibration
    weights = _repo_path(DEFAULT_C2_WEIGHTS)

    for path, label in (
        (video, "clip video"),
        (calibration, "calibration"),
        (weights, "RF-DETR weights"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    clip_duration_s = context.duration_s
    if args.start_s >= clip_duration_s:
        raise ValueError(
            f"start_s={args.start_s} is outside the {clip_duration_s}s clip"
        )

    duration_s = min(
        float(args.duration_s),
        clip_duration_s - float(args.start_s),
    )

    if args.sample_period_s > 0.30:
        raise ValueError(
            "sample_period_s > 0.30 is intentionally rejected for Hito 1D: "
            "the current local tracklet gates are designed for denser samples"
        )

    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if args.width <= 0 or args.height <= 0:
        raise ValueError("width and height must be positive")

    store = TrackingArtifactStore(context.data_root, context.match_id, context.clip_id)

    print(
        "[tracking-v1] "
        f"match={context.match_id} clip={context.clip_id} "
        f"video={context.clip_video} calibration={context.calibration} "
        f"duration={context.duration_s:.6f}s fps={context.fps:.6f} "
        f"frames={context.frame_count} size={context.width}x{context.height} "
        f"input={input_mode}"
    )

    frame_stream = canonical_frames_from_veo(
        video=video,
        calibration=calibration,
        weights=weights,
        start_s=float(args.start_s),
        duration_s=duration_s,
        sample_period_s=float(args.sample_period_s),
        threshold=float(args.threshold),
        width=int(args.width),
        height=int(args.height),
        skip_rescue=bool(args.skip_rescue),
    )

    tracklet_config = TrackletBuilderConfig()

    detection_metadata = {
        "input": input_mode,
        "video": str(video),
        "calibration": str(calibration),
    }
    if args.match_id is None:
        detection_metadata["dev_config"] = str(_repo_path(args.dev_config))

    result = run_detection_and_tracklet_stages(
        store,
        frame_stream,
        detection_config={
            "source": "six-view-same-frame-association",
            "threshold": float(args.threshold),
            "width": int(args.width),
            "height": int(args.height),
            "start_s": float(args.start_s),
            "duration_s": duration_s,
            "sample_period_s": float(args.sample_period_s),
            "rescue_enabled": not bool(args.skip_rescue),
        },
        detection_metadata=detection_metadata,
        tracklet_config=tracklet_config,
    )

    print("\n[tracking-v1] completed")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"[tracking-v1] artifacts={store.root}")


if __name__ == "__main__":
    main()
