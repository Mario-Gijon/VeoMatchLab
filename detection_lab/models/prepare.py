"""Prepare or verify Detection Lab pretrained-model cache entries without inference."""

from __future__ import annotations

import argparse
import gc
import os
import shutil
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .registry import MODEL_REGISTRY, ModelSpec, checkpoint_path, dependency_available, model_by_id, model_root


DFINE_CHECKPOINT_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_coco.pth"
DFINE_CONFIG_URL = "https://raw.githubusercontent.com/veritone/D-FINE/master/configs/dfine/dfine_hgnetv2_x_coco.yml"


@dataclass(frozen=True)
class Status:
    model: ModelSpec
    state: str
    detail: str


def status_for(model: ModelSpec, root: Path, *, accepted_rfdetr_plus_license: bool = False) -> Status:
    """Report cache/dependency state without constructing models or downloading files."""
    if model.requires_license_acceptance and not accepted_rfdetr_plus_license:
        return Status(model, "LICENSE REQUIRED", "Use --accept-rfdetr-plus-license only after accepting PML-1.0.")
    if not dependency_available(model):
        return Status(model, "DEPENDENCY MISSING", f"Install {model.dependency} in .venv-detection.")
    missing = [path for path in (checkpoint_path(model, root), *(root / item for item in model.auxiliary_files)) if not path.is_file() or path.stat().st_size == 0]
    if missing:
        return Status(model, "MISSING", ", ".join(str(path.relative_to(root)) for path in missing))
    return Status(model, "READY", str(checkpoint_path(model, root).relative_to(root)))


@contextmanager
def temporary_environment(name: str, value: str) -> Iterator[None]:
    """Set an upstream cache variable only while its official downloader runs."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def download_file(url: str, destination: Path) -> None:
    """Atomically download one official, explicitly declared asset."""
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    try:
        with urllib.request.urlopen(url) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def prepare_rfdetr(model: ModelSpec, root: Path) -> None:
    """Use RF-DETR's official downloader without constructing a detector."""
    from rfdetr.assets.model_weights import download_pretrain_weights

    target = checkpoint_path(model, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary_environment("RF_HOME", str(target.parent)):
        download_pretrain_weights(str(target), validate_md5=True)


def prepare_ultralytics(model: ModelSpec, root: Path) -> None:
    """Let the official Ultralytics loader cache its named checkpoint at our root."""
    target = checkpoint_path(model, root)
    target.parent.mkdir(parents=True, exist_ok=True)
    with temporary_environment("YOLO_CONFIG_DIR", str(target.parent / "config")):
        from ultralytics import YOLO

        instance = YOLO(str(target))
        del instance
        gc.collect()


def prepare_dfine(model: ModelSpec, root: Path) -> None:
    """Cache only official D-FINE-X COCO checkpoint/config; source remains separate."""
    download_file(DFINE_CHECKPOINT_URL, checkpoint_path(model, root))
    download_file(DFINE_CONFIG_URL, root / model.auxiliary_files[0])


def prepare_model(model: ModelSpec, root: Path, *, accepted_rfdetr_plus_license: bool) -> Status:
    """Prepare one model sequentially; model construction is avoided except YOLO's official cache loader."""
    before = status_for(model, root, accepted_rfdetr_plus_license=accepted_rfdetr_plus_license)
    if before.state in {"LICENSE REQUIRED", "DEPENDENCY MISSING", "READY"}:
        return before
    if model.preparation in {"rfdetr", "rfdetr_plus"}:
        prepare_rfdetr(model, root)
    elif model.preparation == "ultralytics":
        prepare_ultralytics(model, root)
    elif model.preparation == "download":
        prepare_dfine(model, root)
    else:  # pragma: no cover - registry construction is exhaustively typed
        raise ValueError(f"Unsupported preparation strategy: {model.preparation}")
    return status_for(model, root, accepted_rfdetr_plus_license=accepted_rfdetr_plus_license)


def print_statuses(statuses: list[Status], root: Path) -> None:
    """Print a compact human-readable preparation table."""
    print(f"Model cache: {root}")
    width = max(len(status.model.name) for status in statuses)
    for status in statuses:
        print(f"{status.model.name:<{width}}  {status.state:<18} {status.detail}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List model registry and current local status")
    parser.add_argument("--model", action="append", choices=[model.id for model in MODEL_REGISTRY], help="Prepare or verify one model; repeatable")
    parser.add_argument("--all", action="store_true", help="Prepare or verify every registered model")
    parser.add_argument("--verify", action="store_true", help="Only inspect local cache state; never download")
    parser.add_argument("--accept-rfdetr-plus-license", action="store_true", help="Confirm that you have separately accepted RF-DETR Plus PML-1.0")
    return parser.parse_args()


def selected_models(args: argparse.Namespace) -> list[ModelSpec]:
    """Resolve CLI selection, defaulting status-only commands to all candidates."""
    if args.all or args.list or args.verify:
        return list(MODEL_REGISTRY)
    if args.model:
        return [model_by_id(model_id) for model_id in dict.fromkeys(args.model)]
    raise ValueError("Choose --list, --verify, --model, or --all.")


def main() -> None:
    args = parse_args()
    try:
        selected = selected_models(args)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    root = model_root()
    if args.list or args.verify:
        print_statuses([status_for(model, root, accepted_rfdetr_plus_license=args.accept_rfdetr_plus_license) for model in selected], root)
        return
    statuses = [prepare_model(model, root, accepted_rfdetr_plus_license=args.accept_rfdetr_plus_license) for model in selected]
    print_statuses(statuses, root)


if __name__ == "__main__":
    main()
