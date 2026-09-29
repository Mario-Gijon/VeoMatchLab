import argparse
from pathlib import Path

import pytest

from experiments.tracking_v0 import audit_reid_crops


@pytest.mark.parametrize("field_margin_m", [0.60, 1.5])
def test_tracking_args_forwards_field_margin_to_raw_tracker(field_margin_m: float):
    source = argparse.Namespace(
        video=Path("source.mp4"),
        calibration=Path("camera_alignment.veo"),
        weights=Path("best.pth"),
        device=None,
        start_s=0.0,
        sample_period_s=1.0 / 15.0,
        threshold=0.40,
        width=1280,
        height=720,
        split_fraction=0.50,
        guard_margin_m=0.50,
        grid_step_m=0.50,
        field_margin_m=field_margin_m,
    )

    tracking = audit_reid_crops._tracking_args(
        source,
        duration_s=30.0,
        tracking_out=Path("_tracking_tmp"),
    )

    assert tracking.field_margin_m == field_margin_m
    assert tracking.threshold == 0.40
    assert tracking.sample_period_s == pytest.approx(1.0 / 15.0)
