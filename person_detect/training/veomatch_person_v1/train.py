#!/usr/bin/env python3
"""Fine-tune RF-DETR 2XL independently for C1, C2, or C3.

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
EXPERIMENTS = ("c1_football_images", "c2_football_images_plus_veo", "c3_football_images_plus_veo_plus_eoakc")
ALIASES = {"c1": EXPERIMENTS[0], "c2": EXPERIMENTS[1], "c3": EXPERIMENTS[2]}


def canonical_experiment(value: str) -> str:
    return ALIASES.get(value, value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=(*ALIASES, *EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    return parser.parse_args(argv)


def load_config(experiment: str) -> dict:
    experiment = canonical_experiment(experiment)
    path = WORKSPACE / "configs" / f"{experiment}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _find_best(run_dir: Path) -> Path:
    candidates = [run_dir / "best.pth", run_dir / "checkpoint_best_total.pth", run_dir / "checkpoint_best.pth"]
    candidates.extend(sorted(run_dir.rglob("*.pth")))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No best checkpoint found under {run_dir}")


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


def train(args) -> Path:
    experiment = canonical_experiment(args.experiment)
    config = load_config(experiment)
    if not BASE_CHECKPOINT.is_file():
        raise FileNotFoundError(BASE_CHECKPOINT)
    dataset_dir = ROOT / config["dataset_dir"]
    if not (dataset_dir / "annotations.json").is_file():
        raise FileNotFoundError(dataset_dir / "annotations.json")
    run_dir = WORKSPACE / "runs" / experiment
    run_dir.mkdir(parents=True, exist_ok=True)
    params = dict(config["training"])
    if args.epochs is not None:
        params["epochs"] = args.epochs
    if args.batch_size is not None:
        params["batch_size"] = args.batch_size
    if args.lr is not None:
        params["learning_rate"] = args.lr

    # RF-DETR is imported only when a real training run is requested.
    from rfdetr import RFDETR2XLarge

    model = RFDETR2XLarge(pretrain_weights=str(BASE_CHECKPOINT))
    model.train(
        dataset_dir=str(dataset_dir),
        output_dir=str(run_dir),
        epochs=int(params["epochs"]),
        batch_size=int(params["batch_size"]),
        grad_accum_steps=int(params["grad_accum_steps"]),
        lr=float(params["learning_rate"]),
    )
    best = _find_best(run_dir)
    best_epoch, validation_metrics = _training_metrics(run_dir)
    export_dir = EXPORT_ROOT / args.experiment
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
    run_dir = WORKSPACE / "runs" / experiment
    print(json.dumps({"experiment": experiment, "base_checkpoint": str(BASE_CHECKPOINT), "dataset": config["dataset_dir"], "run_dir": str(run_dir)}, indent=2))
    if args.dry_run:
        return 0
    print(f"Starting independent RF-DETR 2XL training for {args.experiment}...")
    print(f"Exported checkpoint: {train(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
