"""Tests for the asymmetric 4 FAR + 2 NEAR coverage experiment.

CHATGPT DELIVERY MARKER: TEST_VIRTUAL_CAMERA_COVERAGE_V11
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.virtual_camera_coverage.coverage import (  # noqa: E402
    DEFAULT_FAR_CAMERAS,
    DEFAULT_NEAR_CAMERAS,
    build_ground_grid,
    derive_asymmetric_layout,
    derive_specs,
    depth_geometry,
    owner_ids,
    player_top_y,
    tight_physical_bounds,
)
from geometry.veo_metadata import load_veo_metadata  # noqa: E402
from person_detect.views.interactive_view.renderer import camera_world  # noqa: E402


CALIBRATION = sorted(
    (ROOT / "tests" / "fixtures" / "calibrations").glob("*.veo")
)[0]


def _metadata():
    return load_veo_metadata(CALIBRATION)


def test_default_layout_is_four_far_two_near():
    meta = _metadata()

    layout = derive_asymmetric_layout(
        meta,
        split_fraction=0.50,
        near_count=DEFAULT_NEAR_CAMERAS,
        far_count=DEFAULT_FAR_CAMERAS,
        max_step_m=1.0,
    )

    assert layout.near_count == 2
    assert layout.far_count == 4
    assert len(layout.cells()) == 6

    assert [cell.name for cell in layout.cells()] == [
        "near_c1",
        "near_c2",
        "far_c1",
        "far_c2",
        "far_c3",
        "far_c4",
    ]


def test_split_is_physical_half_depth_from_near_touchline():
    meta = _metadata()

    _, _, _, total_depth = depth_geometry(meta)

    layout = derive_asymmetric_layout(
        meta,
        split_fraction=0.50,
        near_count=2,
        far_count=4,
        max_step_m=1.0,
    )

    assert math.isclose(
        layout.split_depth_m,
        total_depth * 0.50,
    )


def test_every_tight_ground_sample_has_exactly_one_owner():
    meta = _metadata()

    layout = derive_asymmetric_layout(
        meta,
        split_fraction=0.50,
        near_count=2,
        far_count=4,
        max_step_m=1.0,
    )

    grid = build_ground_grid(
        meta,
        tight_physical_bounds(meta),
        layout,
        max_step_m=1.0,
    )

    assert grid.owner_id.shape == grid.longitude.shape
    assert grid.owner_id.shape == grid.depth_m.shape

    unique = set(
        int(value)
        for value in np.unique(grid.owner_id)
    )

    assert unique.issubset(set(range(6)))
    assert unique == set(range(6))

    counts = [
        int(np.sum(grid.owner_id == owner_id))
        for owner_id in range(6)
    ]

    assert sum(counts) == grid.owner_id.size
    assert all(count > 0 for count in counts)


def test_owner_ids_respect_near_far_band_boundary():
    meta = _metadata()

    layout = derive_asymmetric_layout(
        meta,
        split_fraction=0.50,
        near_count=2,
        far_count=4,
        max_step_m=1.0,
    )

    longitude = np.asarray(
        [
            np.mean(layout.near_lon_edges),
            np.mean(layout.far_lon_edges),
        ],
        dtype=np.float64,
    )

    depths = np.asarray(
        [
            layout.split_depth_m - 0.01,
            layout.split_depth_m,
        ],
        dtype=np.float64,
    )

    owners = owner_ids(
        longitude,
        depths,
        layout,
    )

    assert int(owners[0]) < layout.near_count
    assert int(owners[1]) >= layout.near_count


def test_player_height_uses_calibrated_veo_y_sign():
    meta = _metadata()

    camera_y = float(
        camera_world(meta)[1]
    )

    top_y = player_top_y(meta)

    assert camera_y != 0.0
    assert top_y != 0.0

    assert math.copysign(
        1.0,
        top_y,
    ) == math.copysign(
        1.0,
        camera_y,
    )


def test_twelve_specs_are_finite_and_deterministic():
    meta = _metadata()

    first, layout_a, tight_a, context_a = derive_specs(
        meta,
        320,
        180,
        split_fraction=0.50,
        near_count=2,
        far_count=4,
        guard_margin_m=0.50,
        near_context_m=1.0,
        outer_context_m=0.50,
        grid_step_m=1.0,
    )

    second, layout_b, tight_b, context_b = derive_specs(
        meta,
        320,
        180,
        split_fraction=0.50,
        near_count=2,
        far_count=4,
        guard_margin_m=0.50,
        near_context_m=1.0,
        outer_context_m=0.50,
        grid_step_m=1.0,
    )

    assert layout_a == layout_b
    assert len(first) == 12
    assert len(second) == 12

    assert [spec.name for spec in first] == [
        "tight_near_c1",
        "tight_near_c2",
        "tight_far_c1",
        "tight_far_c2",
        "tight_far_c3",
        "tight_far_c4",
        "context_near_c1",
        "context_near_c2",
        "context_far_c1",
        "context_far_c2",
        "context_far_c3",
        "context_far_c4",
    ]

    assert np.array_equal(
        tight_a.owner_id,
        tight_b.owner_id,
    )

    assert np.array_equal(
        context_a.owner_id,
        context_b.owner_id,
    )

    for a, b in zip(first, second):
        assert a.json() == b.json()
        assert math.isfinite(a.longitude)
        assert math.isfinite(a.latitude)
        assert 0.0 < a.hfov < math.pi
        assert 0.0 < a.vfov < math.pi
        assert a.required_ground_points > 0
        assert a.required_world_points == 2 * a.required_ground_points
        assert 0.0 <= a.required_point_valid_fraction <= 1.0


def test_cameras_share_pitch_within_each_depth_band_and_mode():
    meta = _metadata()

    specs, _, _, _ = derive_specs(
        meta,
        320,
        180,
        split_fraction=0.50,
        near_count=2,
        far_count=4,
        guard_margin_m=0.50,
        near_context_m=1.0,
        outer_context_m=0.50,
        grid_step_m=1.0,
    )

    for mode in ("tight", "context"):
        for band, expected_count in (("near", 2), ("far", 4)):
            group = [
                spec
                for spec in specs
                if spec.mode == mode
                and spec.cell.band == band
            ]

            assert len(group) == expected_count

            latitudes = {
                spec.latitude
                for spec in group
            }

            assert len(latitudes) == 1

            # Cameras in the same band must still keep independent pan.
            longitudes = {
                round(spec.longitude, 12)
                for spec in group
            }

            assert len(longitudes) > 1
