import math
from pathlib import Path

import numpy as np

from experiments.virtual_camera_coverage.coverage import (
    derive_specs,
    logical_cells,
)
from geometry.veo_metadata import load_veo_metadata


ROOT = Path(__file__).resolve().parents[1]
CALIBRATION = sorted((ROOT / "tests/fixtures/calibrations").glob("*.veo"))[0]


def test_six_logical_cells_cover_field_without_overlap():
    length, width = 105.0, 68.0
    cells = logical_cells(length, width)
    assert len(cells) == 6
    assert {(row, col) for _, col, row, _ in cells} == {(r, c) for r in range(2) for c in range(3)}
    xs = sorted({b.x_min for *_, b in cells} | {b.x_max for *_, b in cells})
    zs = sorted({b.z_min for *_, b in cells} | {b.z_max for *_, b in cells})
    assert xs == [-length / 2, -length / 6, length / 6, length / 2]
    assert zs == [-width / 2, 0.0, width / 2]
    for _, _, _, a in cells:
        for _, _, _, b in cells:
            if a is b:
                continue
            overlap_x = min(a.x_max, b.x_max) - max(a.x_min, b.x_min)
            overlap_z = min(a.z_max, b.z_max) - max(a.z_min, b.z_min)
            assert overlap_x <= 0 or overlap_z <= 0


def test_specs_use_calibration_dimensions_and_are_deterministic():
    metadata = load_veo_metadata(CALIBRATION)
    first = derive_specs(metadata, 320, 180, 0.75, 1.0)
    second = derive_specs(metadata, 320, 180, 0.75, 1.0)
    assert len(first) == 12
    assert all(s.ownership.x_min >= -metadata.field_length / 2 for s in first)
    assert all(s.ownership.x_max <= metadata.field_length / 2 for s in first)
    assert all(s.ownership.z_min >= -metadata.field_width / 2 for s in first)
    assert all(s.ownership.z_max <= metadata.field_width / 2 for s in first)
    for a, b in zip(first, second):
        assert a.json() == b.json()
        assert math.isfinite(a.hfov) and math.isfinite(a.vfov)
        assert 0 < a.hfov < math.pi and 0 < a.vfov < math.pi


def test_guard_changes_rendered_bounds_but_not_ownership():
    metadata = load_veo_metadata(CALIBRATION)
    no_guard = derive_specs(metadata, 320, 180, 0.0, 1.0)
    guard = derive_specs(metadata, 320, 180, 0.75, 1.0)
    for a, b in zip(no_guard, guard):
        assert a.ownership == b.ownership
    assert any(a.rendered != b.rendered for a, b in zip(no_guard, guard))
    for spec in guard:
        if spec.name.startswith("tight_"):
            assert spec.rendered.x_min <= spec.ownership.x_min
            assert spec.rendered.x_max >= spec.ownership.x_max
            assert spec.rendered.z_min <= spec.ownership.z_min
            assert spec.rendered.z_max >= spec.ownership.z_max
