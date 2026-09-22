"""D-FINE-X COCO backend for the clean Veo player-detection benchmark.

Model-specific responsibility only:
    LEFT/CENTER/RIGHT BGR images -> person boxes + confidence

This backend intentionally uses the official Peterande/D-FINE implementation
and the preserved local `dfine_x_coco.pth` checkpoint.

Geometry, field filtering, ownership, result rendering and run organization
remain in the common `person_detect/run_detection.py` pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DFINE_ROOT = PROJECT_ROOT / ".model-cache" / "D-FINE"
DEFAULT_CONFIG_RELATIVE = Path(
    "configs/dfine/dfine_hgnetv2_x_coco.yml"
)


@dataclass(frozen=True)
class ModelDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_name: str = "person"


class DFineXCocoBackend:
    """Official D-FINE-X COCO checkpoint inference wrapper.

    D-FINE's official Torch inference path uses:
      - the X COCO YAML config
      - 640x640 Resize + ToTensor preprocessing
      - cfg.model.deploy()
      - cfg.postprocessor.deploy()

    In deploy mode the COCO labels are contiguous 0-based IDs, so person=0.

    LEFT/CENTER/RIGHT are processed sequentially to keep GPU-memory behavior
    predictable across the benchmark.
    """

    PERSON_LABEL = 0
    INPUT_SIZE = 640

    def __init__(
        self,
        weights: str | Path,
        *,
        device: str | None = None,
        dfine_root: str | Path | None = None,
    ) -> None:
        self.weights = Path(weights).expanduser().resolve()
        if not self.weights.exists():
            raise FileNotFoundError(self.weights)

        self.dfine_root = self._resolve_dfine_root(dfine_root)
        self.config = (
            self.dfine_root / DEFAULT_CONFIG_RELATIVE
        ).resolve()

        if not self.config.exists():
            raise FileNotFoundError(
                "D-FINE X COCO config was not found at "
                f"{self.config}. The official D-FINE repository must be "
                "available under .model-cache/D-FINE or DFINE_ROOT."
            )

        # The upstream project imports its package as top-level `src`.
        # Put the repository root first so we use the pinned official checkout,
        # not an unrelated package named `src`.
        root_string = str(self.dfine_root)
        if root_string in sys.path:
            sys.path.remove(root_string)
        sys.path.insert(0, root_string)

        try:
            import torch
            import torch.nn as nn
            import torchvision.transforms as T
            from PIL import Image
            from src.core import YAMLConfig
        except ImportError as exc:
            raise RuntimeError(
                "Official D-FINE dependencies are unavailable. "
                "Install the requirements from the checked-out D-FINE repo."
            ) from exc

        self._torch = torch
        self._Image = Image
        self._transform = T.Compose(
            [
                T.Resize((self.INPUT_SIZE, self.INPUT_SIZE)),
                T.ToTensor(),
            ]
        )

        self.device = (
            device
            or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        cfg = YAMLConfig(str(self.config))

        # Matches the official tools/inference/torch_inf.py behavior: the
        # checkpoint already contains the backbone weights, so do not trigger
        # a separate pretrained-backbone load/download.
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        try:
            checkpoint = torch.load(
                str(self.weights),
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            # Compatibility with PyTorch releases before `weights_only`.
            checkpoint = torch.load(
                str(self.weights),
                map_location="cpu",
            )

        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
            self.checkpoint_state = "ema.module"
        elif "model" in checkpoint:
            state = checkpoint["model"]
            self.checkpoint_state = "model"
        else:
            raise RuntimeError(
                "Unsupported D-FINE checkpoint: expected `ema` or `model`."
            )

        cfg.model.load_state_dict(state)

        class DeployModel(nn.Module):
            def __init__(self, model, postprocessor):
                super().__init__()
                self.model = model.deploy()
                self.postprocessor = postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                return self.postprocessor(
                    outputs,
                    orig_target_sizes,
                )

        self.model = DeployModel(
            cfg.model,
            cfg.postprocessor,
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_dfine_root(
        explicit: str | Path | None,
    ) -> Path:
        candidates: list[Path] = []

        if explicit is not None:
            candidates.append(Path(explicit).expanduser())

        env_root = os.environ.get("DFINE_ROOT")
        if env_root:
            candidates.append(Path(env_root).expanduser())

        candidates.extend(
            [
                DEFAULT_DFINE_ROOT,
                PROJECT_ROOT / "person_detect" / ".model-cache" / "D-FINE",
                PROJECT_ROOT / "D-FINE",
            ]
        )

        for candidate in candidates:
            resolved = candidate.resolve()
            if (
                (resolved / "src" / "core").exists()
                and (
                    resolved
                    / DEFAULT_CONFIG_RELATIVE
                ).exists()
            ):
                return resolved

        checked = "\n".join(
            f"  - {candidate.resolve()}"
            for candidate in candidates
        )
        raise FileNotFoundError(
            "Official Peterande/D-FINE source checkout was not found.\n"
            "Checked:\n"
            f"{checked}\n"
            "Clone it into `.model-cache/D-FINE` or set DFINE_ROOT."
        )

    @property
    def name(self) -> str:
        return "dfine_x_coco"

    def info(self) -> dict:
        return {
            "backend": "dfine",
            "variant": "x_coco",
            "name": self.name,
            "weights": str(self.weights),
            "device": self.device,
            "native_resolution": self.INPUT_SIZE,
            "inference_mode": "sequential_views",
            "implementation": "Peterande/D-FINE official",
            "dfine_root": str(self.dfine_root),
            "config": str(self.config),
            "checkpoint_state": self.checkpoint_state,
        }

    def _predict_one(
        self,
        image_bgr: np.ndarray,
        *,
        threshold: float,
    ) -> list[ModelDetection]:
        if (
            image_bgr is None
            or image_bgr.ndim != 3
            or image_bgr.shape[2] != 3
        ):
            raise ValueError("Expected a valid BGR image")

        rgb = cv2.cvtColor(
            image_bgr,
            cv2.COLOR_BGR2RGB,
        )
        pil = self._Image.fromarray(rgb)

        width, height = pil.size

        image_tensor = (
            self._transform(pil)
            .unsqueeze(0)
            .to(self.device)
        )
        original_size = self._torch.tensor(
            [[width, height]],
            dtype=self._torch.int64,
            device=self.device,
        )

        with self._torch.inference_mode():
            labels, boxes, scores = self.model(
                image_tensor,
                original_size,
            )

        labels = labels[0].detach().cpu().numpy()
        boxes = boxes[0].detach().cpu().numpy()
        scores = scores[0].detach().cpu().numpy()

        persons: list[ModelDetection] = []

        for label, box, score in zip(
            labels,
            boxes,
            scores,
            strict=True,
        ):
            score = float(score)

            if score < threshold:
                continue

            # Official deploy postprocessor returns contiguous COCO labels.
            # COCO "person" is contiguous class 0.
            if int(label) != self.PERSON_LABEL:
                continue

            x1, y1, x2, y2 = [
                float(value)
                for value in box
            ]

            if not np.isfinite(
                [x1, y1, x2, y2, score]
            ).all():
                continue

            if x2 <= x1 or y2 <= y1:
                continue

            persons.append(
                ModelDetection(
                    xyxy=(x1, y1, x2, y2),
                    confidence=score,
                )
            )

        return persons

    def predict_views(
        self,
        views_bgr: dict[str, np.ndarray],
        *,
        threshold: float,
    ) -> dict[str, list[ModelDetection]]:
        """Run LEFT/CENTER/RIGHT sequentially through D-FINE-X COCO."""
        order = ("left", "center", "right")

        missing = [
            name
            for name in order
            if name not in views_bgr
        ]
        if missing:
            raise ValueError(
                f"Missing rendered views: {missing}"
            )

        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        return {
            view_name: self._predict_one(
                views_bgr[view_name],
                threshold=threshold,
            )
            for view_name in order
        }
