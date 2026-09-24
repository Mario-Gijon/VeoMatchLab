#!/usr/bin/env python3
"""Fine-tune RF-DETR 2XL independently for C1 or C2.

This entry point is intentionally lazy: importing it never imports RF-DETR or
loads a checkpoint.  Use ``--dry-run`` to validate paths and print the exact
independent experiment plan before starting a long training job.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = ROOT / "person_detect/training/veomatch_person_v1"
BASE_CHECKPOINT = ROOT / "person_detect/model_weights/rfdetr/rf-detr-xxlarge.pth"
EXPORT_ROOT = ROOT / "person_detect/model_weights/rfdetr_finetuned/veomatch_person_v1"
EXPERIMENTS = ("c1_football_images", "c2_football_images_plus_veo")
ALIASES = {"c1": EXPERIMENTS[0], "c2": EXPERIMENTS[1]}


def canonical_experiment(value: str) -> str:
    return ALIASES.get(value, value)


def resolve_num_classes(config: dict) -> int:
    class_names = config.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("training config must define a non-empty class_names list")
    return len(class_names)


def model_constructor_kwargs(config: dict, params: dict) -> dict:
    """Resolve the explicit task head while retaining baseline initialization."""
    return {
        "pretrain_weights": str(BASE_CHECKPOINT),
        "num_classes": resolve_num_classes(config),
        "gradient_checkpointing": bool(params["gradient_checkpointing"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*ALIASES, *EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty existing run directory")
    return parser.parse_args(argv)


def load_config(experiment: str) -> dict:
    experiment = canonical_experiment(experiment)
    path = WORKSPACE / "configs" / f"{experiment}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _find_best(run_dir: Path) -> Path:
    candidates = [
        run_dir / "checkpoint_best_total.pth",
        run_dir / "checkpoint_best_ema.pth",
        run_dir / "checkpoint_best_regular.pth",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"RF-DETR best checkpoint not found under {run_dir}; expected "
        "checkpoint_best_total.pth (or checkpoint_best_ema.pth / "
        "checkpoint_best_regular.pth)."
    )


def _validate_dataset(dataset_dir: Path) -> dict:
    split_reports = {}
    for split in ("train", "valid"):
        annotation_path = dataset_dir / split / "_annotations.coco.json"
        if not annotation_path.is_file():
            raise FileNotFoundError(
                f"Incomplete RF-DETR dataset: missing {annotation_path}. "
                "Rebuild C1/C2 with build_datasets.py."
            )
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = payload.get("categories", [])
        if categories != [{"id": 1, "name": "person", "supercategory": "person"}]:
            raise ValueError(f"{annotation_path} must contain exactly the person category")
        missing = []
        for image in payload.get("images", []):
            image_path = annotation_path.parent / image["file_name"]
            if not image_path.is_file():
                missing.append(str(image_path))
        if missing:
            raise FileNotFoundError(f"{annotation_path} references missing image files, e.g. {missing[0]}")
        split_reports[split] = {"images": len(payload.get("images", [])), "annotations": len(payload.get("annotations", []))}
    test_annotation = dataset_dir / "test" / "_annotations.coco.json"
    if test_annotation.is_file():
        payload = json.loads(test_annotation.read_text(encoding="utf-8"))
        split_reports["test"] = {"images": len(payload.get("images", [])), "annotations": len(payload.get("annotations", []))}
    return split_reports


def _training_metrics(run_dir: Path) -> tuple[int | None, dict]:
    for name in ("metrics.json", "results.json", "training_metrics.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            epoch = payload.get("best_epoch")
            return (int(epoch) if epoch is not None else None, payload.get("validation_metrics", payload))
    return None, {}


def _ensure_run_dir(run_dir: Path, *, overwrite: bool) -> None:
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_dir}. "
            "Use --overwrite explicitly or choose a new run directory."
        )
    run_dir.mkdir(parents=True, exist_ok=True)


def train(args) -> Path:
    experiment = canonical_experiment(args.experiment)
    config = load_config(experiment)
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    dataset_dir = ROOT / config["dataset_dir"]
    _validate_dataset(dataset_dir)
    run_dir = WORKSPACE / "runs" / experiment
    _ensure_run_dir(run_dir, overwrite=args.overwrite)
    params = dict(config["training"])
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.lr is not None:
        params["learning_rate"] = args.lr

    # RF-DETR is imported only when a real training run is requested.
    from rfdetr import RFDETR2XLarge

    model = RFDETR2XLarge(**model_constructor_kwargs(config, params))
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        grad_accum_steps=int(params["grad_accum_steps"]),
        lr=float(params["learning_rate"]),
        lr_encoder=float(params["learning_rate_encoder"]),
        resolution=int(params["resolution"]),
        use_ema=bool(params["use_ema"]),
        early_stopping=bool(params["early_stopping"]),
        early_stopping_patience=int(params["early_stopping_patience"]),
        skip_best_epochs=int(params["skip_best_epochs"]),
    )
    best = _find_best(run_dir)
    best_epoch, validation_metrics = _training_metrics(run_dir)
    export_dir = EXPORT_ROOT / experiment
    export_dir.mkdir(parents=True, exist_ok=True)
    exported = export_dir / "best.pth"
    shutil.copy2(best, exported)
    metadata = {
        "experiment_name": experiment,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "dataset_composition": config["composition"],
        "class_names": ["person"],
        "training_hyperparameters": params,
        "best_epoch": best_epoch if best_epoch is not None else config.get("best_epoch"),
        "validation_metrics": validation_metrics or config.get("validation_metrics", {}),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "checkpoint": str(exported),
    }
    (export_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return exported


def main(argv=None) -> int:
    args = parse_args(argv)
    experiment = canonical_experiment(args.experiment)
    config = load_config(experiment)
    dataset_dir = ROOT / config["dataset_dir"]
    split_reports = _validate_dataset(dataset_dir)
    params = dict(config["training"])
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.lr is not None:
        params["learning_rate"] = args.lr
    run_dir = WORKSPACE / "runs" / experiment
    plan = {
        "experiment": experiment,
        "base_checkpoint": str(BASE_CHECKPOINT),
        "dataset_directory": str(dataset_dir),
        "split_counts": split_reports,
        "class_names": config["class_names"],
        "num_classes": resolve_num_classes(config),
        "resolution": params["resolution"],
        "epochs": params["epochs"],
        "batch_size": params["batch_size"],
        "gradient_accumulation": params["grad_accum_steps"],
        "effective_batch_size": params["batch_size"] * params["grad_accum_steps"],
        "learning_rate": params["learning_rate"],
        "encoder_learning_rate": params["learning_rate_encoder"],
        "gradient_checkpointing": params["gradient_checkpointing"],
        "use_ema": params["use_ema"],
        "early_stopping": params["early_stopping"],
        "early_stopping_patience": params["early_stopping_patience"],
        "skip_best_epochs": params["skip_best_epochs"],
        "run_directory": str(run_dir),
        "final_export_directory": str(EXPORT_ROOT / experiment),
    }
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    print(f"Starting independent RF-DETR 2XL training for {args.experiment}...")
    print(f"Exported checkpoint: {train(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
