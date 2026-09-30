#!/usr/bin/env python3
"""Isolated PRTReID-SoccerNet embedding worker for Tracking V1.

Run this file with ``.venv-prtreid/bin/python``.

This worker intentionally does NOT try to register SoccerNet as a native
PRTReID dataset.  We only need inference on already-cropped player images.

The official SoccerNet checkpoint stores the BPBReID/PRTReID configuration
used for training.  Some config keys (notably ``masks.parts_names``) are
created dynamically during training and are therefore absent from PRTReID's
default config.  In the full SoccerNet pipeline those keys are created after a
custom SoccerNet dataset class is registered.  Our standalone worker does not
have that dataset registry, so it restores those dynamic keys directly from
the trusted checkpoint before performing YACS' strict merge.

The published SoccerNet baseline currently uses:
- test_embeddings = ['globl']
- dim_reduce_output = 256

The worker fails loudly if that contract changes.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

# PyTorch >= 2.6 defaults torch.load(..., weights_only=True).  The official
# SoccerNet PRTReID checkpoint also contains its saved YACS configuration, so
# it must be loaded in legacy/full-checkpoint mode.  The file is downloaded
# from the official SoccerNet Zenodo record and verified separately by MD5 in
# scripts/setup_prtreid_env.sh.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np


MODEL_ID = "prtreid-soccernet-baseline-global-v1"
EXPECTED_DIM = 256


def load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("request must encode a JSON object")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported request schema")
    if payload.get("model_id") != MODEL_ID:
        raise ValueError("request model_id does not match worker")
    if int(payload.get("expected_dim", -1)) != EXPECTED_DIM:
        raise ValueError("request expected_dim does not match worker")

    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("request items must be a list")

    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ValueError(f"request item {index} must be an object")
        if "crop_path" not in item:
            raise ValueError(f"request item {index} lacks crop_path")

    return payload


def _seed_missing_cfg_keys(destination, source) -> None:
    """Seed dynamic checkpoint keys before YACS' strict merge.

    PRTReID's default config intentionally lacks some runtime-computed values
    such as ``masks.parts_names``.  The saved SoccerNet checkpoint contains
    them.  YACS refuses to merge unknown keys, so we first create only the
    missing keys that are present in the trusted checkpoint and then let the
    normal strict merge validate/overwrite all values.

    This is recursive because future compatible checkpoints may carry another
    runtime-derived key nested under ``model.bpbreid``.
    """
    from yacs.config import CfgNode

    if not isinstance(destination, CfgNode):
        raise TypeError("destination must be a yacs CfgNode")
    if not isinstance(source, CfgNode):
        raise TypeError("source must be a yacs CfgNode")

    for key, source_value in source.items():
        if key not in destination:
            # clone() where available keeps CfgNode/list values independent
            # from the checkpoint object.
            if isinstance(source_value, CfgNode):
                destination[key] = source_value.clone()
            elif isinstance(source_value, list):
                destination[key] = list(source_value)
            elif isinstance(source_value, tuple):
                destination[key] = tuple(source_value)
            elif isinstance(source_value, dict):
                destination[key] = dict(source_value)
            else:
                destination[key] = source_value
            continue

        destination_value = destination[key]
        if (
            isinstance(destination_value, CfgNode)
            and isinstance(source_value, CfgNode)
        ):
            _seed_missing_cfg_keys(
                destination_value,
                source_value,
            )


def _merge_checkpoint_config(
    cfg,
    checkpoint_payload: Mapping[str, Any],
    *,
    hrnet_dir: Path,
):
    """Merge the saved SoccerNet BPBReID config into defaults.

    Local machine paths are preserved:
    - hrnet_pretrained_path -> our ``models/prtreid`` directory
    - masks.dir -> current local default, irrelevant for global-only inference

    Dynamic runtime keys from the checkpoint are seeded before strict merge.
    """
    saved_cfg = checkpoint_payload.get("config")
    if saved_cfg is None:
        raise ValueError(
            "SoccerNet checkpoint does not contain saved PRTReID config"
        )

    if not hasattr(saved_cfg, "model") or not hasattr(
        saved_cfg.model,
        "bpbreid",
    ):
        raise ValueError(
            "checkpoint config does not contain model.bpbreid"
        )

    bp_cfg = saved_cfg.model.bpbreid.clone()

    # Do not import paths from the machine that trained the checkpoint.
    bp_cfg.pop("hrnet_pretrained_path", None)
    if hasattr(bp_cfg, "masks"):
        bp_cfg.masks.pop("dir", None)

    # The checkpoint may contain dynamically generated keys such as
    # masks.parts_names.  Create those keys in our destination tree first,
    # then retain normal strict YACS merging for all actual values.
    _seed_missing_cfg_keys(
        cfg.model.bpbreid,
        bp_cfg,
    )
    cfg.model.bpbreid.merge_from_other_cfg(bp_cfg)

    # Local path must win after the checkpoint merge.
    cfg.model.bpbreid.hrnet_pretrained_path = str(hrnet_dir)

    return cfg


def build_extractor(
    *,
    checkpoint: Path,
    hrnet_dir: Path,
    device: str,
):
    import torch

    from prtreid.scripts.default_config import get_default_config
    from prtreid.tools.feature_extractor import FeatureExtractor
    from prtreid.utils import load_checkpoint

    cfg = get_default_config()

    use_cuda = (
        device.startswith("cuda")
        and torch.cuda.is_available()
    )
    cfg.use_gpu = use_cuda
    effective_device = device if use_cuda else "cpu"

    cfg.model.load_weights = str(checkpoint)
    cfg.model.load_config = True
    cfg.model.pretrained = True
    cfg.model.bpbreid.hrnet_pretrained_path = str(hrnet_dir)

    # IMPORTANT:
    # Do NOT set data.sources/targets to "soccernet" here.  Plain PRTReID does
    # not register SoccerNet as a built-in dataset.  The SoccerNet GameState
    # wrapper dynamically registers its own ReidDataset before build_config().
    # This standalone worker does not need a datamanager at all: FeatureExtractor
    # only needs the model config and image preprocessing settings.
    #
    # Keeping the default supported dataset avoids invoking an unavailable
    # SoccerNet dataset registry if other PRTReID code inspects cfg.data.
    cfg.data.sources = ["market1501"]
    cfg.data.targets = ["market1501"]

    # SoccerNet's inference crops are processed at 256x128.
    cfg.data.height = 256
    cfg.data.width = 128

    checkpoint_payload = load_checkpoint(
        str(checkpoint)
    )
    cfg = _merge_checkpoint_config(
        cfg,
        checkpoint_payload,
        hrnet_dir=hrnet_dir,
    )

    test_embeddings = list(
        cfg.model.bpbreid.test_embeddings
    )
    if test_embeddings != ["globl"]:
        raise RuntimeError(
            "This benchmark intentionally expects the published "
            "SoccerNet global embedding, got "
            f"{test_embeddings!r}"
        )

    output_dim = int(
        cfg.model.bpbreid.dim_reduce_output
    )
    if output_dim != EXPECTED_DIM:
        raise RuntimeError(
            "Unexpected SoccerNet embedding dimension in "
            f"checkpoint config: {output_dim}"
        )

    parts_names = getattr(
        cfg.model.bpbreid.masks,
        "parts_names",
        None,
    )
    parts_num = getattr(
        cfg.model.bpbreid.masks,
        "parts_num",
        None,
    )

    print(
        "[prtreid-worker] checkpoint-config "
        f"backbone={cfg.model.bpbreid.backbone} "
        f"test_embeddings={test_embeddings} "
        f"dim={output_dim} "
        f"parts_num={parts_num} "
        f"parts_names={'present' if parts_names is not None else 'missing'}"
    )

    extractor = FeatureExtractor(
        cfg,
        model_path=str(checkpoint),
        device=effective_device,
        image_size=(
            int(cfg.data.height),
            int(cfg.data.width),
        ),
        verbose=False,
    )

    return extractor, cfg, effective_device


def extract_batches(
    *,
    extractor,
    cfg,
    crop_paths: list[str],
    batch_size: int,
) -> np.ndarray:
    from prtreid.utils.tools import (
        extract_test_embeddings,
    )

    rows: list[np.ndarray] = []

    for start in range(
        0,
        len(crop_paths),
        batch_size,
    ):
        batch = crop_paths[
            start:start + batch_size
        ]

        # Passing file paths makes PRTReID open them as PIL RGB and avoids
        # OpenCV BGR/RGB ambiguity.
        result = extractor(batch)

        (
            embeddings,
            _visibility_scores,
            _body_masks,
            _pixels_cls_scores,
            _role_cls_scores,
        ) = extract_test_embeddings(
            result,
            cfg.model.bpbreid.test_embeddings,
        )

        # Global-only extraction can be returned either as N x D or N x 1 x D.
        if embeddings.ndim == 3:
            if embeddings.shape[1] != 1:
                raise RuntimeError(
                    "Expected exactly one global embedding branch, "
                    f"got shape {tuple(embeddings.shape)}"
                )
            embeddings = embeddings[:, 0, :]
        elif embeddings.ndim != 2:
            raise RuntimeError(
                "Unexpected PRTReID embedding shape "
                f"{tuple(embeddings.shape)}"
            )

        array = (
            embeddings
            .detach()
            .float()
            .cpu()
            .numpy()
            .astype(np.float32, copy=False)
        )

        if array.shape[1] != EXPECTED_DIM:
            raise RuntimeError(
                f"Expected {EXPECTED_DIM}-D embeddings, "
                f"got {array.shape}"
            )

        norms = np.linalg.norm(
            array,
            axis=1,
            keepdims=True,
        )
        if np.any(norms <= 1e-12):
            raise RuntimeError(
                "PRTReID returned a zero-norm embedding"
            )

        rows.append(array / norms)

        print(
            "[prtreid-worker] embedded "
            f"{min(start + len(batch), len(crop_paths))}/"
            f"{len(crop_paths)}"
        )

    if not rows:
        return np.empty(
            (0, EXPECTED_DIM),
            dtype=np.float32,
        )

    return np.concatenate(
        rows,
        axis=0,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__
    )
    parser.add_argument(
        "--request",
        required=True,
    )
    parser.add_argument(
        "--output",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )
    parser.add_argument(
        "--hrnet-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be > 0"
        )

    request_path = Path(
        args.request
    ).resolve()
    output_path = Path(
        args.output
    ).resolve()
    checkpoint = Path(
        args.checkpoint
    ).resolve()
    hrnet_dir = Path(
        args.hrnet_dir
    ).resolve()

    request = load_request(
        request_path
    )

    crop_paths: list[str] = []
    for item in request["items"]:
        crop_path = Path(
            item["crop_path"]
        ).resolve()

        if not crop_path.is_file():
            raise FileNotFoundError(
                crop_path
            )

        crop_paths.append(
            str(crop_path)
        )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            checkpoint
        )
    if not hrnet_dir.is_dir():
        raise FileNotFoundError(
            hrnet_dir
        )

    print(
        "[prtreid-worker] starting "
        f"crops={len(crop_paths)} "
        f"device_requested={args.device}"
    )

    extractor, cfg, effective_device = (
        build_extractor(
            checkpoint=checkpoint,
            hrnet_dir=hrnet_dir,
            device=args.device,
        )
    )

    print(
        "[prtreid-worker] model-ready "
        f"device={effective_device}"
    )

    embeddings = extract_batches(
        extractor=extractor,
        cfg=cfg,
        crop_paths=crop_paths,
        batch_size=args.batch_size,
    )

    expected_shape = (
        len(crop_paths),
        EXPECTED_DIM,
    )
    if embeddings.shape != expected_shape:
        raise RuntimeError(
            "worker output shape mismatch: "
            f"{embeddings.shape} != {expected_shape}"
        )

    metadata = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "embedding_dim": EXPECTED_DIM,
        "checkpoint_name": checkpoint.name,
        "test_embeddings": list(
            cfg.model.bpbreid.test_embeddings
        ),
        "image_size": [
            int(cfg.data.height),
            int(cfg.data.width),
        ],
        "backbone": str(
            cfg.model.bpbreid.backbone
        ),
        "device": effective_device,
        "crop_count": len(crop_paths),
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_path,
        embeddings=embeddings,
        metadata_json=np.asarray(
            json.dumps(
                metadata,
                separators=(",", ":"),
            )
        ),
    )

    print(
        "[prtreid-worker] completed "
        f"crops={len(crop_paths)} "
        f"shape={embeddings.shape} "
        f"device={effective_device}"
    )
    print(
        f"[prtreid-worker] output={output_path}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
