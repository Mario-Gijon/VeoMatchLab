#!/usr/bin/env python3
"""CPU reproduction of Veo Standard/Follow (Tracking) view.

Purpose
-------
This is a validation tool, not an application feature yet. It reproduces the
critical geometry path observed in Veo's minified WebGL player:

    detections_constrained.det @ video time
      -> yaw / pitch / vertical FOV
      -> legacy virtualcam = Ry(yaw) @ Rx(pitch)
      -> virtualcam = legacy2logic @ legacy virtualcam
      -> viewDir(uv, fov)
      -> camera2world @ camera2logic @ virtualcam @ viewDir
      -> ground / projection bubble target
      -> left/right lens projection and Veo blend

The generated frame can be compared with Veo's pre-rendered Tracking video.
If they line up, Tracking pixel <-> field geometry is validated independently
of the inverse projection used by analysis.tracking_ground.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analysis.reproduce_veo_panorama import (  # noqa: E402
    extract_frame,
    project_world_target_to_lens,
)
from analysis.tracking_ground import (  # noqa: E402
    LegacyCameraDirections,
    TRACKING_ASPECT,
    camera2logic_matrix,
    camera_state_at_time,
)
from analysis.veo_metadata import load_veo_metadata  # noqa: E402


BLEND_DISTANCE = 0.3
PROJECTION_BUBBLE_RADIUS = 1000.0
BLEND_BUBBLE_RADIUS = 20.0
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Veo's Standard/Follow Tracking renderer on CPU."
    )
    parser.add_argument(
        "--dual-lens-video",
        type=Path,
        required=True,
        help="2048x2048 Veo panorama transcode (LEFT top, RIGHT bottom).",
    )
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--time", type=float, required=True)
    parser.add_argument(
        "--tracking-video",
        type=Path,
        default=None,
        help="Optional Veo pre-rendered Tracking video for direct comparison.",
    )
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument(
        "--row-chunk",
        type=int,
        default=120,
        help="Rows rendered per vectorized batch.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/reproduce_veo_tracking"),
    )
    return parser.parse_args()


def _tracking_target_geometry_chunk(
    metadata,
    state,
    y0: int,
    y1: int,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Literal Standard-shader target geometry for one output row chunk."""
    xx = np.arange(width, dtype=np.float64) + 0.5
    yy = np.arange(y0, y1, dtype=np.float64) + 0.5

    u = xx / float(width)
    # v_texcoord is bottom-left based; image rows are top-left based.
    v = 1.0 - yy / float(height)
    uu, vv = np.meshgrid(u, v)

    z = 0.5 / math.tan(state.fov / 2.0)
    view_dir = np.stack(
        [
            (uu - 0.5) * TRACKING_ASPECT,
            -(vv - 0.5),
            np.full_like(uu, z),
            np.zeros_like(uu),
        ],
        axis=-1,
    )

    count = view_dir.shape[0] * view_dir.shape[1]
    view_flat = view_dir.reshape(count, 4)

    camera2world = np.asarray(metadata.camera2world, dtype=np.float64)
    camera2logic = camera2logic_matrix(metadata)
    view_to_world = camera2world @ camera2logic @ state.virtualcam_logic

    # Row-array equivalent of GLSL M * column_vector.
    direction_world_h = view_flat @ view_to_world.T
    direction_world = direction_world_h[:, :3]

    camera_world = (
        camera2world
        @ np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    )

    dy = direction_world[:, 1]
    up_down = (np.sign(dy) + 1.0) / 2.0
    safe_dy = np.where(np.abs(dy) > EPSILON, dy, EPSILON)
    ground_distance = -camera_world[1] / safe_dy

    distance = (
        up_down * ground_distance
        + (1.0 - up_down) * PROJECTION_BUBBLE_RADIUS
    )

    projection_direction = direction_world * distance[:, None]
    target_xyz = camera_world[:3] + projection_direction
    target_world = np.concatenate(
        [target_xyz, np.ones((count, 1), dtype=np.float64)],
        axis=1,
    )

    projection_direction_h = np.concatenate(
        [
            projection_direction,
            np.zeros((count, 1), dtype=np.float64),
        ],
        axis=1,
    )

    world_to_logic = np.linalg.inv(camera2world @ camera2logic)
    blend_target = projection_direction_h @ world_to_logic.T
    # Shader explicitly sets w=1 before normalization.
    blend_target[:, 3] = 1.0

    projection_length = np.linalg.norm(projection_direction_h, axis=1)
    normalize_mask = (
        (projection_length > BLEND_BUBBLE_RADIUS)
        | (projection_direction[:, 1] < 0.0)
    )

    if np.any(normalize_mask):
        lengths = np.linalg.norm(blend_target[normalize_mask], axis=1)
        lengths = np.maximum(lengths, EPSILON)
        blend_target[normalize_mask] = (
            blend_target[normalize_mask]
            / lengths[:, None]
            * BLEND_BUBBLE_RADIUS
        )

    blend = np.clip(
        blend_target[:, 0] / BLEND_DISTANCE + 0.5,
        0.0,
        1.0,
    )

    return target_world, blend.reshape(y1 - y0, width)


def render_tracking(
    raw_frame: np.ndarray,
    metadata,
    state,
    width: int,
    height: int,
    *,
    row_chunk: int = 120,
) -> np.ndarray:
    """Render Standard/Follow geometry from the dual-lens frame."""
    raw_h, raw_w = raw_frame.shape[:2]
    result = np.zeros((height, width, 3), dtype=np.uint8)

    for y0 in range(0, height, row_chunk):
        y1 = min(height, y0 + row_chunk)
        target_world, blend = _tracking_target_geometry_chunk(
            metadata,
            state,
            y0,
            y1,
            width,
            height,
        )

        left_x, left_y, _ = project_world_target_to_lens(
            target_world,
            metadata.left,
            metadata.camera2world,
            raw_w,
            raw_h,
            True,
        )
        right_x, right_y, _ = project_world_target_to_lens(
            target_world,
            metadata.right,
            metadata.camera2world,
            raw_w,
            raw_h,
            False,
        )

        rows = y1 - y0
        left_x = left_x.reshape(rows, width)
        left_y = left_y.reshape(rows, width)
        right_x = right_x.reshape(rows, width)
        right_y = right_y.reshape(rows, width)

        left = cv2.remap(
            raw_frame,
            left_x,
            left_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        right = cv2.remap(
            raw_frame,
            right_x,
            right_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        blend3 = blend[..., None].astype(np.float32)
        out = (
            left.astype(np.float32) * (1.0 - blend3)
            + right.astype(np.float32) * blend3
        )
        result[y0:y1] = np.clip(out, 0, 255).astype(np.uint8)

        print(f"Rendered rows {y0:4d}-{y1 - 1:4d} / {height}")

    return result


def compare_frames(generated: np.ndarray, actual: np.ndarray) -> dict:
    """Estimate generated->actual image alignment using ORB + RANSAC."""
    if generated.shape[:2] != actual.shape[:2]:
        actual = cv2.resize(
            actual,
            (generated.shape[1], generated.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    gray_a = cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(actual, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=8000, fastThreshold=10)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    result = {
        "generated_keypoints": len(kp_a or []),
        "actual_keypoints": len(kp_b or []),
        "ratio_matches": 0,
        "homography": None,
        "inliers": 0,
        "inlier_fraction": None,
        "median_inlier_reprojection_px": None,
        "corner_displacement_px": None,
    }

    if des_a is None or des_b is None:
        return result

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for m, n in pairs if m.distance < 0.75 * n.distance]
    result["ratio_matches"] = len(good)

    if len(good) < 8:
        return result

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 3.0)
    if H is None or mask is None:
        return result

    inlier_mask = mask.ravel().astype(bool)
    result["homography"] = H.tolist()
    result["inliers"] = int(inlier_mask.sum())
    result["inlier_fraction"] = float(inlier_mask.mean())

    projected = cv2.perspectiveTransform(src, H)
    errors = np.linalg.norm(projected[:, 0] - dst[:, 0], axis=1)
    if np.any(inlier_mask):
        result["median_inlier_reprojection_px"] = float(
            np.median(errors[inlier_mask])
        )

    h, w = generated.shape[:2]
    corners = np.float32(
        [[[0, 0]], [[w - 1, 0]], [[w - 1, h - 1]], [[0, h - 1]]]
    )
    warped = cv2.perspectiveTransform(corners, H)
    disp = np.linalg.norm(warped[:, 0] - corners[:, 0], axis=1)
    result["corner_displacement_px"] = [float(x) for x in disp]

    return result


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_veo_metadata(args.calibration)
    directions = LegacyCameraDirections.load(args.detections)
    state = camera_state_at_time(metadata, directions, args.time)
    if state is None:
        raise RuntimeError(f"No camera state at t={args.time:.6f}s")

    print("Camera state:")
    print(json.dumps(state.to_dict(include_matrices=False), indent=2))

    print(f"\nExtracting dual-lens frame at t={args.time:.6f}s ...")
    raw_frame, raw_info = extract_frame(args.dual_lens_video, args.time)
    print(
        f"Dual-lens source: {raw_frame.shape[1]}x{raw_frame.shape[0]} "
        f"frame={raw_info['actual_frame']}"
    )

    print(f"\nRendering {args.width}x{args.height} Tracking reproduction ...")
    generated = render_tracking(
        raw_frame,
        metadata,
        state,
        args.width,
        args.height,
        row_chunk=max(1, args.row_chunk),
    )

    generated_path = args.out_dir / "tracking_reproduced.png"
    cv2.imwrite(str(generated_path), generated)

    summary = {
        "time_s": float(args.time),
        "dual_lens_video": str(args.dual_lens_video),
        "tracking_video": str(args.tracking_video) if args.tracking_video else None,
        "calibration": str(args.calibration),
        "detections": str(args.detections),
        "output_size": [args.width, args.height],
        "camera_state": state.to_dict(include_matrices=True),
        "dual_lens_frame": raw_info,
        "comparison": None,
    }

    if args.tracking_video is not None:
        print("\nExtracting Veo Tracking frame for comparison ...")
        actual, actual_info = extract_frame(args.tracking_video, args.time)
        actual = cv2.resize(
            actual,
            (args.width, args.height),
            interpolation=cv2.INTER_AREA,
        )

        actual_path = args.out_dir / "tracking_actual.png"
        side_path = args.out_dir / "tracking_side_by_side.png"
        diff_path = args.out_dir / "tracking_absdiff.png"
        cv2.imwrite(str(actual_path), actual)
        cv2.imwrite(str(side_path), np.hstack([actual, generated]))
        cv2.imwrite(str(diff_path), cv2.absdiff(actual, generated))

        comparison = compare_frames(generated, actual)
        summary["tracking_frame"] = actual_info
        summary["comparison"] = comparison

        print("\nFeature alignment generated -> actual:")
        print(json.dumps(comparison, indent=2))

        if comparison.get("homography") is not None:
            H = np.asarray(comparison["homography"], dtype=np.float64)
            aligned = cv2.warpPerspective(
                generated,
                H,
                (args.width, args.height),
            )
            cv2.imwrite(
                str(args.out_dir / "tracking_generated_aligned.png"),
                aligned,
            )
            cv2.imwrite(
                str(args.out_dir / "tracking_aligned_absdiff.png"),
                cv2.absdiff(actual, aligned),
            )

    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nSaved:")
    for path in sorted(args.out_dir.glob("tracking_*.png")):
        print(f"  {path}")
    print(f"  {summary_path}")


if __name__ == "__main__":
    main()
