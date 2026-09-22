#!/usr/bin/env python3
"""Synthetic seam stress-test for two-lens canonical fusion.

This is NOT a detector benchmark. It validates the fusion logic itself.

We generate synthetic players on/near the ownership seam, project every player
into both physical lenses, build detector-like bounding boxes around those
projected foot-points, add configurable pixel noise independently per lens,
and then ask `TwoLensFusion` to recover canonical entities.

The test includes:
  - single players exactly around the seam (duplicate-risk case)
  - pairs of nearby real players (false-merge-risk case)

A successful run should keep:
    canonical_entities == true_players
without creating cross-lens clones and without merging the nearby pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from person_detect.views.two_lens_view.renderer import DEFAULT_CALIBRATION  # noqa: E402
from person_detect.views.two_lens_view.projection import TwoLensProjector  # noqa: E402
from person_detect.views.two_lens_view.fusion import TwoLensFusion  # noqa: E402


@dataclass(frozen=True)
class FakeDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float


def seam_x_for_z(
    projector: TwoLensProjector,
    z: float,
) -> float:
    """Find the owner transition across X for one Z row."""
    xs = np.linspace(
        -projector.half_length,
        projector.half_length,
        2001,
    )

    previous_owner = None
    previous_x = None

    for x in xs:
        owner, _, _ = projector.owner_for_world_point(
            float(x),
            float(z),
        )

        if (
            previous_owner is not None
            and owner is not None
            and owner != previous_owner
            and previous_x is not None
        ):
            return 0.5 * (float(previous_x) + float(x))

        if owner is not None:
            previous_owner = owner
            previous_x = float(x)

    raise RuntimeError(f"No ownership seam found at z={z}")


def bbox_around_foot(
    foot_x: float,
    foot_y: float,
    *,
    height_px: float,
    confidence: float,
) -> FakeDetection:
    width_px = max(8.0, height_px * 0.34)

    x1 = foot_x - width_px / 2.0
    x2 = foot_x + width_px / 2.0
    y2 = foot_y + 0.5
    y1 = y2 - height_px

    return FakeDetection(
        xyxy=(x1, y1, x2, y2),
        confidence=confidence,
    )


def approximate_player_height_px(
    foot_y: float,
) -> float:
    # Synthetic only: far players near the horizon are small, near players
    # lower in the image are larger. Clamp to realistic detector-box sizes.
    return float(np.clip(18.0 + 0.085 * foot_y, 22.0, 120.0))


def generate_detection(
    projector: TwoLensProjector,
    lens_name: str,
    world_x: float,
    world_z: float,
    *,
    noise_px: float,
    rng: random.Random,
) -> FakeDetection | None:
    visibility = projector.visibility_for_world_point(
        lens_name,
        world_x,
        world_z,
    )

    if not visibility.visible:
        return None

    assert visibility.pixel_x is not None
    assert visibility.pixel_y is not None

    noisy_x = visibility.pixel_x + rng.gauss(0.0, noise_px)
    noisy_y = visibility.pixel_y + rng.gauss(0.0, noise_px)

    height = approximate_player_height_px(noisy_y)

    return bbox_around_foot(
        noisy_x,
        noisy_y,
        height_px=height,
        confidence=0.90,
    )


def build_true_players(
    projector: TwoLensProjector,
) -> list[tuple[float, float, str]]:
    """Players placed on both sides of seam + deliberately close real pairs."""
    players: list[tuple[float, float, str]] = []

    z_rows = np.linspace(
        -projector.half_width * 0.65,
        projector.half_width * 0.75,
        9,
    )

    for row_index, z in enumerate(z_rows):
        seam_x = seam_x_for_z(projector, float(z))

        # One player almost exactly on each side of the seam.
        offset = 0.18 if row_index % 2 == 0 else 0.30
        players.append(
            (seam_x - offset, float(z), f"seam_L_{row_index}")
        )
        players.append(
            (seam_x + offset, float(z), f"seam_R_{row_index}")
        )

    # Two close but real players: we want to make sure the fusion does not
    # collapse them merely because their world positions are near each other.
    for pair_index, z in enumerate((-8.0, 8.0, 20.0)):
        seam_x = seam_x_for_z(projector, z)
        players.append(
            (seam_x - 0.45, z, f"close_A_{pair_index}")
        )
        players.append(
            (seam_x + 0.45, z + 0.10, f"close_B_{pair_index}")
        )

    return players


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stress-test cross-lens seam fusion with synthetic players."
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=DEFAULT_CALIBRATION,
    )
    parser.add_argument(
        "--noise-px",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--max-world-distance",
        type=float,
        default=1.50,
    )
    parser.add_argument(
        "--max-reprojection-error",
        type=float,
        default=0.85,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.noise_px < 0:
        raise ValueError("--noise-px must be >= 0")

    if args.trials <= 0:
        raise ValueError("--trials must be > 0")

    projector = TwoLensProjector(args.calibration)
    fusion = TwoLensFusion(
        projector,
        max_world_distance_m=args.max_world_distance,
        max_reprojection_error=args.max_reprojection_error,
    )

    true_players = build_true_players(projector)

    exact_count_trials = 0
    overcount_trials = 0
    undercount_trials = 0

    entity_counts: list[int] = []
    matched_counts: list[int] = []

    for trial in range(args.trials):
        rng = random.Random(args.seed + trial)

        detections = {
            "left": [],
            "right": [],
        }

        visible_truth = 0

        for world_x, world_z, _ in true_players:
            any_visible = False

            for lens_name in ("left", "right"):
                detection = generate_detection(
                    projector,
                    lens_name,
                    world_x,
                    world_z,
                    noise_px=args.noise_px,
                    rng=rng,
                )

                if detection is not None:
                    detections[lens_name].append(detection)
                    any_visible = True

            if any_visible:
                visible_truth += 1

        result = fusion.fuse(detections)

        entity_count = len(result.entities)
        entity_counts.append(entity_count)
        matched_counts.append(len(result.matched_pairs))

        if entity_count == visible_truth:
            exact_count_trials += 1
        elif entity_count > visible_truth:
            overcount_trials += 1
        else:
            undercount_trials += 1

    summary = {
        "calibration": str(args.calibration),
        "synthetic_true_players": len(true_players),
        "noise_px_sigma": args.noise_px,
        "trials": args.trials,
        "gates": {
            "max_world_distance_m": args.max_world_distance,
            "max_reprojection_error_bbox_heights": args.max_reprojection_error,
        },
        "results": {
            "exact_count_trials": exact_count_trials,
            "overcount_trials": overcount_trials,
            "undercount_trials": undercount_trials,
            "exact_count_fraction": exact_count_trials / args.trials,
            "overcount_fraction": overcount_trials / args.trials,
            "undercount_fraction": undercount_trials / args.trials,
            "entity_count_min": min(entity_counts),
            "entity_count_max": max(entity_counts),
            "entity_count_mean": float(np.mean(entity_counts)),
            "matched_pairs_mean": float(np.mean(matched_counts)),
        },
        "important_limitation": (
            "This validates fusion under synthetic foot-point noise only. "
            "It is not an empirical 99.9% guarantee on real detector output. "
            "Real-video validation and temporal tracking are still required."
        ),
    }

    print(json.dumps(summary, indent=2))

    if overcount_trials:
        raise SystemExit(
            "FAIL: at least one synthetic trial produced cross-lens overcount."
        )


if __name__ == "__main__":
    main()
