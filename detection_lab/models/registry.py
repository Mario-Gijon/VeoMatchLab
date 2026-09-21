"""Declarative Detection Lab candidate-model registry; importing it loads no ML runtime."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_ROOT = ROOT / ".model-cache"


@dataclass(frozen=True)
class ModelSpec:
    """One detector candidate and its local checkpoint contract."""

    id: str
    name: str
    family: str
    checkpoint: Path
    upstream: str
    license_note: str
    dependency: str | None
    preparation: Literal["rfdetr", "rfdetr_plus", "ultralytics", "download"]
    requires_license_acceptance: bool = False
    auxiliary_files: tuple[Path, ...] = ()


MODEL_REGISTRY: tuple[ModelSpec, ...] = (
    ModelSpec(
        id="rfdetr_large",
        name="RF-DETR Large",
        family="RF-DETR",
        checkpoint=Path("rfdetr/rf-detr-large-2026.pth"),
        upstream="https://github.com/roboflow/rf-detr",
        license_note="Apache-2.0 (core RF-DETR model).",
        dependency="rfdetr",
        preparation="rfdetr",
    ),
    ModelSpec(
        id="rfdetr_xlarge",
        name="RF-DETR XLarge",
        family="RF-DETR+",
        checkpoint=Path("rfdetr/rf-detr-xlarge.pth"),
        upstream="https://github.com/roboflow/rf-detr_plus",
        license_note="Platform Model License 1.0; requires a valid Roboflow Platform Agreement.",
        dependency="rfdetr_plus",
        preparation="rfdetr_plus",
        requires_license_acceptance=True,
    ),
    ModelSpec(
        id="rfdetr_2xlarge",
        name="RF-DETR 2XLarge",
        family="RF-DETR+",
        checkpoint=Path("rfdetr/rf-detr-xxlarge.pth"),
        upstream="https://github.com/roboflow/rf-detr_plus",
        license_note="Platform Model License 1.0; requires a valid Roboflow Platform Agreement.",
        dependency="rfdetr_plus",
        preparation="rfdetr_plus",
        requires_license_acceptance=True,
    ),
    ModelSpec(
        id="yolo26x",
        name="YOLO26x",
        family="Ultralytics YOLO26",
        checkpoint=Path("yolo/yolo26x.pt"),
        upstream="https://docs.ultralytics.com/models/yolo26/",
        license_note="Ultralytics licensing applies; review AGPL-3.0 or an Enterprise license.",
        dependency="ultralytics",
        preparation="ultralytics",
    ),
    ModelSpec(
        id="dfine_x",
        name="D-FINE-X (COCO)",
        family="D-FINE",
        checkpoint=Path("dfine/dfine_x_coco.pth"),
        upstream="https://github.com/veritone/D-FINE",
        license_note="Apache-2.0 upstream; this entry intentionally uses the COCO checkpoint, not Objects365+COCO.",
        dependency=None,
        preparation="download",
        auxiliary_files=(Path("dfine/dfine_hgnetv2_x_coco.yml"),),
    ),
)


def model_root() -> Path:
    """Return the configured model root without creating it or importing ML packages."""
    configured = os.environ.get("VEO_MATCH_MODELS_ROOT")
    return (Path(configured).expanduser() if configured else DEFAULT_MODELS_ROOT).resolve()


def model_by_id(model_id: str) -> ModelSpec:
    """Return a registered model or raise a clear error for an unknown ID."""
    for model in MODEL_REGISTRY:
        if model.id == model_id:
            return model
    raise KeyError(f"Unknown model {model_id!r}")


def checkpoint_path(model: ModelSpec, root: Path | None = None) -> Path:
    """Return a model's expected local checkpoint path."""
    return (root or model_root()) / model.checkpoint


def dependency_available(model: ModelSpec) -> bool:
    """Check package availability without importing potentially heavy ML modules."""
    return model.dependency is None or importlib.util.find_spec(model.dependency) is not None
