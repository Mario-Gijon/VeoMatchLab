"""Shared test configuration for non-production reference fixtures."""

from __future__ import annotations

import os
from pathlib import Path


os.environ.setdefault(
    "VEO_REFERENCE_CALIBRATIONS_ROOT",
    str(Path(__file__).resolve().parent / "fixtures" / "calibrations"),
)
os.environ.setdefault("VEO_MATCH_DATA_ROOT", "/tmp/veomatchlab-pytest-runtime")
