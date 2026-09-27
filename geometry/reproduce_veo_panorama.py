#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geometry.veo_metadata import load_veo_metadata


CALIB_WIDTH = 3840.0
CALIB_HEIGHT = 2160.0

TOP_DEST_RATIO = 19.0 / 45.0
TOP_SOURCE_RATIO = 19.0 / 32.0
BOTTOM_DEST_RATIO = 1.0 - TOP_DEST_RATIO
BOTTOM_SOURCE_RATIO = 1.0 - TOP_SOURCE_RATIO

BLEND_DISTANCE = 0.3
PROJECTION_BUBBLE_RADIUS = 1000.0
BLEND_BUBBLE_RADIUS = 20.0


@dataclass(frozen=True)
class PanoramaRenderMaps:
    """Static panorama projection data for one calibration and image shape."""

    raw_width: int
    raw_height: int
    width: int
    height: int
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    left_valid: np.ndarray
    right_valid: np.ndarray
    blend: np.ndarray


def parse_args():
    p = argparse.ArgumentParser(
        description="CPU reproduction of Veo's panoramic WebGL shader."
    )

    p.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--calibration",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--time",
        type=float,
        required=True,
        help="Time in seconds in the hidden panorama video.",
    )

    p.add_argument(
        "--width",
        type=int,
        default=854,
    )

    p.add_argument(
        "--height",
        type=int,
        default=480,
    )

    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("output/reproduce_veo_panorama"),
    )

    return p.parse_args()


def extract_frame(video_path: Path, time_s: float):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    # Prefer frame-index seeking because our hidden panorama
    # source is a normal MP4 with a known frame rate.
    target_frame = int(round(time_s * fps))

    cap.set(
        cv2.CAP_PROP_POS_FRAMES,
        target_frame,
    )

    ok, frame = cap.read()

    actual_frame = int(
        cap.get(cv2.CAP_PROP_POS_FRAMES)
    ) - 1

    cap.release()

    if not ok or frame is None:
        raise RuntimeError(
            f"Could not read frame near t={time_s:.6f}s"
        )

    return frame, {
        "fps": float(fps),
        "frame_count": int(frame_count),
        "requested_time": float(time_s),
        "requested_frame": target_frame,
        "actual_frame": actual_frame,
        "actual_time_approx": (
            actual_frame / fps
            if fps > 0
            else None
        ),
    }


def distort_points(
    x: np.ndarray,
    y: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
):
    """
    Literal translation of the Veo GLSL distort() function.

    D OpenCV order:
    [k1,k2,p1,p2,k3,k4,k5,k6,...]
    """

    d = np.zeros(
        8,
        dtype=np.float64,
    )

    n = min(
        len(D),
        8,
    )

    d[:n] = np.asarray(
        D[:n],
        dtype=np.float64,
    )

    (
        k1,
        k2,
        p1,
        p2,
        k3,
        k4,
        k5,
        k6,
    ) = d

    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    radial = (
        1.0
        + k1 * r2
        + k2 * r4
        + k3 * r6
    ) / (
        1.0
        + k4 * r2
        + k5 * r4
        + k6 * r6
    )

    tangential_x = (
        p2 * (
            r2
            + 2.0 * x * x
        )
        + 2.0 * p1 * x * y
    )

    tangential_y = (
        p1 * (
            r2
            + 2.0 * y * y
        )
        + 2.0 * p2 * x * y
    )

    xd = (
        x * radial
        + tangential_x
    )

    yd = (
        y * radial
        + tangential_y
    )

    # GLSL:
    #
    # intrinsic * vec3(xd, yd, 1)
    #
    u = (
        K[0, 0] * xd
        + K[0, 1] * yd
        + K[0, 2]
    )

    v = (
        K[1, 0] * xd
        + K[1, 1] * yd
        + K[1, 2]
    )

    return u, v


def apply_source_compression(v):
    """
    Literal translation of Veo getImage() vertical mapping.

    Input v is the normalized coordinate in the original
    3840x2160 calibration image.

    Output remains 0..1 within one lens before stacking.
    """

    result = np.empty_like(
        v,
        dtype=np.float64,
    )

    upper = (
        v < TOP_DEST_RATIO
    )

    result[upper] = (
        v[upper]
        / TOP_DEST_RATIO
        * TOP_SOURCE_RATIO
    )

    result[~upper] = (
        (
            v[~upper]
            - TOP_DEST_RATIO
        )
        / BOTTOM_DEST_RATIO
        * BOTTOM_SOURCE_RATIO
        + TOP_SOURCE_RATIO
    )

    return result


def project_world_target_to_lens(
    target_world: np.ndarray,
    lens,
    camera2world: np.ndarray,
    raw_width: int,
    raw_height: int,
    is_left: bool,
):
    """
    Equivalent of:

      worldspace_projection_dir_left/right
      inverse(camera2world * lens2camera)
      getUV()
      getImage()
    """

    lens_transform_world = (
        camera2world
        @ lens.lens2camera
    )

    lens_position_world = (
        lens_transform_world
        @ np.array(
            [0.0, 0.0, 0.0, 1.0],
            dtype=np.float64,
        )
    )

    # GLSL subtraction of two positions:
    # resulting w = 0.
    direction_world = (
        target_world[:, :3]
        - lens_position_world[:3]
    )

    direction_world_h = np.concatenate(
        [
            direction_world,
            np.zeros(
                (
                    len(direction_world),
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    world_to_lens = np.linalg.inv(
        lens_transform_world
    )

    # row-array equivalent of M @ column_vector
    direction_lens = (
        direction_world_h
        @ world_to_lens.T
    )

    z = direction_lens[:, 2]

    valid_z = (
        np.abs(z) > 1e-12
    )

    x = np.zeros_like(z)
    y = np.zeros_like(z)

    x[valid_z] = (
        direction_lens[
            valid_z,
            0,
        ]
        / z[valid_z]
    )

    y[valid_z] = (
        direction_lens[
            valid_z,
            1,
        ]
        / z[valid_z]
    )

    u_4k, v_4k = distort_points(
        x,
        y,
        lens.intrinsic,
        lens.distortion,
    )

    # Shader getUV():
    #
    # coord.xy / vec2(3840,2160)
    #
    u = (
        u_4k
        / CALIB_WIDTH
    )

    v = (
        v_4k
        / CALIB_HEIGHT
    )

    # Shader getImage() compression mapping.
    v = apply_source_compression(
        v
    )

    valid = (
        valid_z
        & np.isfinite(u)
        & np.isfinite(v)
        & (u >= 0.0)
        & (u <= 1.0)
        & (v >= 0.0)
        & (v <= 1.0)
    )

    # Select upper/lower half of the 2048x2048 texture.
    stacked_v = (
        v * 0.5
    )

    if not is_left:
        stacked_v = (
            stacked_v
            + 0.5
        )

    # OpenGL normalized linear texture coordinates map to
    # texel-center coordinates approximately as:
    #
    # px = uv * size - 0.5
    #
    map_x = (
        u * raw_width
        - 0.5
    )

    map_y = (
        stacked_v
        * raw_height
        - 0.5
    )

    map_x = map_x.astype(
        np.float32
    )

    map_y = map_y.astype(
        np.float32
    )

    map_x[~valid] = -1.0
    map_y[~valid] = -1.0

    return map_x, map_y, valid


def build_output_world_geometry(
    metadata,
    width: int,
    height: int,
):
    """
    Literal translation of Panorama shader geometry:

      lon/lat
      camera2world
      camera2presentation
      ground intersection / 1000 m bubble
      blend target
    """

    va = metadata.va_presentation

    # Pixel-center texture coordinates.
    #
    # WebGL panorama:
    # uv.y = 0 at bottom
    # uv.y = 1 at top.
    #
    xx = (
        np.arange(
            width,
            dtype=np.float64,
        )
        + 0.5
    )

    yy = (
        np.arange(
            height,
            dtype=np.float64,
        )
        + 0.5
    )

    u = (
        xx
        / width
    )

    v = (
        1.0
        - yy
        / height
    )

    uu, vv = np.meshgrid(
        u,
        v,
    )

    lon = (
        uu
        * (
            va["right"]
            - va["left"]
        )
        + va["left"]
    )

    lat = (
        vv
        * (
            va["top"]
            - va["bottom"]
        )
        + va["bottom"]
    )

    # Exact result of:
    #
    # rotMatY(lon)
    # * rotMatX(lat)
    # * vec4(0,0,1,0)
    #
    # Important: GLSL mat constructors are column-major.
    #
    cos_lat = np.cos(lat)

    dir_presentation = np.stack(
        [
            np.sin(lon) * cos_lat,
            -np.sin(lat),
            np.cos(lon) * cos_lat,
        ],
        axis=-1,
    )

    count = (
        width
        * height
    )

    dir_presentation_h = np.concatenate(
        [
            dir_presentation.reshape(
                count,
                3,
            ),
            np.zeros(
                (
                    count,
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    effective_camera2presentation = np.linalg.inv(
    metadata.camera2presentation
    )

    camera_to_world_presentation = (
        metadata.camera2world
        @ effective_camera2presentation
    )

    direction_world_h = (
        dir_presentation_h
        @ camera_to_world_presentation.T
    )

    direction_world = (
        direction_world_h[:, :3]
    )

    camera_world = (
        metadata.camera2world
        @ np.array(
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
            dtype=np.float64,
        )
    )

    dy = (
        direction_world[:, 1]
    )

    # GLSL:
    #
    # up_down =
    # (sign(dir.y) + 1) / 2
    #
    up_down = (
        np.sign(dy)
        + 1.0
    ) / 2.0

    safe_dy = np.where(
        np.abs(dy) > 1e-12,
        dy,
        1e-12,
    )

    ground_distance = (
        -camera_world[1]
        / safe_dy
    )

    distance = (
        up_down
        * ground_distance
        + (
            1.0
            - up_down
        )
        * PROJECTION_BUBBLE_RADIUS
    )

    projection_direction = (
        direction_world
        * distance[:, None]
    )

    target_world_xyz = (
        camera_world[:3]
        + projection_direction
    )

    target_world = np.concatenate(
        [
            target_world_xyz,
            np.ones(
                (
                    count,
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    # Blend target:
    #
    # inverse(camera2world * camera2presentation)
    # * projection_direction
    #
    projection_direction_h = np.concatenate(
        [
            projection_direction,
            np.zeros(
                (
                    count,
                    1,
                ),
                dtype=np.float64,
            ),
        ],
        axis=1,
    )

    world_to_presentation = np.linalg.inv(
        camera_to_world_presentation
    )

    blend_target = (
        projection_direction_h
        @ world_to_presentation.T
    )

    # GLSL then explicitly sets w = 1.
    blend_target[:, 3] = 1.0

    projection_length = np.linalg.norm(
        projection_direction_h,
        axis=1,
    )

    normalize_mask = (
        (projection_length > BLEND_BUBBLE_RADIUS)
        | (
            projection_direction[:, 1]
            < 0.0
        )
    )

    if np.any(normalize_mask):
        lengths = np.linalg.norm(
            blend_target[
                normalize_mask
            ],
            axis=1,
        )

        lengths = np.maximum(
            lengths,
            1e-12,
        )

        blend_target[
            normalize_mask
        ] = (
            blend_target[
                normalize_mask
            ]
            / lengths[:, None]
            * BLEND_BUBBLE_RADIUS
        )

    blend = np.clip(
        (
            blend_target[:, 0]
            / BLEND_DISTANCE
        )
        + 0.5,
        0.0,
        1.0,
    )

    return (
        target_world,
        blend.reshape(
            height,
            width,
        ),
    )


def build_panorama_maps(metadata, raw_width: int, raw_height: int, width: int, height: int) -> PanoramaRenderMaps:
    """Precompute the unchanged CPU panorama geometry and lens remap tables."""
    if raw_width != 2048 or raw_height != 2048:
        print(
            "WARNING: Veo shader input was observed as "
            f"2048x2048, but input is {raw_width}x{raw_height}."
        )
    print(
        "Building world-space panorama geometry..."
    )

    target_world, blend = (
        build_output_world_geometry(
            metadata,
            width,
            height,
        )
    )

    print(
        "Projecting panorama into LEFT lens..."
    )

    left_x, left_y, left_valid = (
        project_world_target_to_lens(
            target_world,
            metadata.left,
            metadata.camera2world,
            raw_width,
            raw_height,
            True,
        )
    )

    print(
        "Projecting panorama into RIGHT lens..."
    )

    right_x, right_y, right_valid = (
        project_world_target_to_lens(
            target_world,
            metadata.right,
            metadata.camera2world,
            raw_width,
            raw_height,
            False,
        )
    )

    left_x = left_x.reshape(
        height,
        width,
    )

    left_y = left_y.reshape(
        height,
        width,
    )

    right_x = right_x.reshape(
        height,
        width,
    )

    right_y = right_y.reshape(
        height,
        width,
    )

    left_valid = left_valid.reshape(
        height,
        width,
    )

    right_valid = right_valid.reshape(
        height,
        width,
    )

    return PanoramaRenderMaps(
        raw_width=raw_width, raw_height=raw_height, width=width, height=height,
        left_x=left_x, left_y=left_y, right_x=right_x, right_y=right_y,
        left_valid=left_valid, right_valid=right_valid, blend=blend,
    )


def render_panorama_with_maps(frame: np.ndarray, maps: PanoramaRenderMaps):
    """Render one source frame with already-computed panorama maps."""
    raw_h, raw_w = frame.shape[:2]
    if (raw_w, raw_h) != (maps.raw_width, maps.raw_height):
        raise ValueError("Frame shape does not match precomputed panorama maps")
    left = cv2.remap(
        frame,
        maps.left_x,
        maps.left_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    right = cv2.remap(
        frame,
        maps.right_x,
        maps.right_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    blend3 = (
        maps.blend[..., None]
        .astype(np.float32)
    )

    panorama = (
        left.astype(np.float32)
        * (
            1.0
            - blend3
        )
        + right.astype(np.float32)
        * blend3
    )

    panorama = np.clip(
        panorama,
        0,
        255,
    ).astype(np.uint8)

    return {
        "panorama": panorama,
        "left": left,
        "right": right,
        "left_valid": maps.left_valid,
        "right_valid": maps.right_valid,
        "blend": maps.blend,
    }


def render(frame: np.ndarray, metadata, width: int, height: int):
    """Compatibility API for one-off renders."""
    raw_h, raw_w = frame.shape[:2]
    maps = build_panorama_maps(metadata, raw_w, raw_h, width, height)
    return render_panorama_with_maps(frame, maps)


def main():
    args = parse_args()

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Reading {args.video}"
    )

    frame, frame_info = extract_frame(
        args.video,
        args.time,
    )

    print(
        "Input frame: "
        f"{frame.shape[1]}x{frame.shape[0]}"
    )

    print(
        "Frame information:"
    )

    for key, value in (
        frame_info.items()
    ):
        print(
            f"  {key}: {value}"
        )

    metadata = load_veo_metadata(
        args.calibration
    )

    print()
    print(
        "Veo panorama angular range:"
    )

    va = metadata.va_presentation

    print(
        f"  left:   {np.degrees(va['left']):.3f}°"
    )

    print(
        f"  right:  {np.degrees(va['right']):.3f}°"
    )

    print(
        f"  bottom: {np.degrees(va['bottom']):.3f}°"
    )

    print(
        f"  top:    {np.degrees(va['top']):.3f}°"
    )

    print()

    result = render(
        frame,
        metadata,
        args.width,
        args.height,
    )

    panorama_path = (
        args.out_dir
        / "veo_panorama_reproduced.png"
    )

    left_path = (
        args.out_dir
        / "left_projected.png"
    )

    right_path = (
        args.out_dir
        / "right_projected.png"
    )

    blend_path = (
        args.out_dir
        / "blend_map.png"
    )

    frame_path = (
        args.out_dir
        / "source_frame.png"
    )

    cv2.imwrite(
        str(panorama_path),
        result["panorama"],
    )

    cv2.imwrite(
        str(left_path),
        result["left"],
    )

    cv2.imwrite(
        str(right_path),
        result["right"],
    )

    cv2.imwrite(
        str(frame_path),
        frame,
    )

    blend_vis = np.clip(
        result["blend"]
        * 255.0,
        0,
        255,
    ).astype(np.uint8)

    cv2.imwrite(
        str(blend_path),
        blend_vis,
    )

    summary = {
        "video": str(args.video),
        "calibration": str(
            args.calibration
        ),
        "output": [
            args.width,
            args.height,
        ],
        "frame": frame_info,
        "source_shape": [
            int(frame.shape[1]),
            int(frame.shape[0]),
        ],
        "left_valid_fraction": float(
            result[
                "left_valid"
            ].mean()
        ),
        "right_valid_fraction": float(
            result[
                "right_valid"
            ].mean()
        ),
        "shader_constants": {
            "calibration_size": [
                3840,
                2160,
            ],
            "topDestRatio": (
                TOP_DEST_RATIO
            ),
            "topSourceRatio": (
                TOP_SOURCE_RATIO
            ),
            "blend_distance": (
                BLEND_DISTANCE
            ),
            "projection_bubble_radius": (
                PROJECTION_BUBBLE_RADIUS
            ),
            "blend_bubble_radius": (
                BLEND_BUBBLE_RADIUS
            ),
        },
    }

    with (
        args.out_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("Saved:")
    print(
        f"  {panorama_path}"
    )
    print(
        f"  {left_path}"
    )
    print(
        f"  {right_path}"
    )
    print(
        f"  {blend_path}"
    )
    print(
        f"  {frame_path}"
    )


if __name__ == "__main__":
    main()
