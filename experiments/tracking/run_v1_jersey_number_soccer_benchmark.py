#!/usr/bin/env python3
"""Research-only SoccerNet jersey-number benchmark adapter for Tracking V1.

Pipeline:
    SoccerNet legibility ResNet34
      -> ViTPose torso crop
      -> SoccerNet fine-tuned PARSeq
      -> VeoMatchLab schema-v2 predictions.jsonl

The external reference repository/weights remain research-only and outside the
production/commercial pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile
from typing import Any, Mapping, Sequence

import cv2


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from experiments.tracking.run_v1_jersey_number_ocr import (  # noqa: E402
    _contact_sheet,
    _iter_jsonl,
    _select_rows,
    _write_jsonl,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


BACKEND_NAME = "soccer-reference-benchmark"
REFERENCE_REPO_URL = "https://github.com/mkoshkina/jersey-number-pipeline"
LEGIBILITY_FILENAME = "legibility_resnet34_soccer_20240215.pth"
STR_FILENAME = "parseq_epoch=24-step=2575-val_accuracy=95.6044-val_NED=96.3255.ckpt"
STRICT_NUMBER = re.compile(r"^[0-9]{1,2}$")


def _strict_number(text: Any) -> int | None:
    if text is None:
        return None
    value = str(text).strip()
    if STRICT_NUMBER.fullmatch(value) is None:
        return None
    number = int(value)
    return number if 1 <= number <= 99 else None


def _sequence_confidence(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(max(0.0, min(1.0, value)))
    try:
        values = [float(item) for item in value]
    except (TypeError, ValueError):
        return 0.0
    if not values:
        return 0.0
    result = 1.0
    for item in values:
        result *= max(0.0, min(1.0, item))
    return float(max(0.0, min(1.0, result)))


def _safe_filename(index: int, row: Mapping[str, Any]) -> str:
    tracklet = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(row["tracklet_id"]))
    sample = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(row["sample_id"]))
    return f"{index:04d}__{tracklet}__{sample}.jpg"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, allow_nan=False, indent=2)
        + "\n",
        encoding="utf-8",
    )


def _materialize_selected_views(
    *,
    root: Path,
    rows: Sequence[Mapping[str, Any]],
    input_dir: Path,
) -> list[dict[str, Any]]:
    input_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        source = root / str(row["number_roi_ref"])
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"could not read jersey ROI: {source}")

        filename = _safe_filename(index, row)
        destination = input_dir / filename
        if not cv2.imwrite(str(destination), image):
            raise RuntimeError(f"could not write {destination}")

        manifest.append(
            {
                "filename": filename,
                "tracklet_id": str(row["tracklet_id"]),
                "detection_id": str(row["detection_id"]),
                "sample_id": str(row["sample_id"]),
                "frame_index": int(row["frame_index"]),
                "timestamp_s": float(row["timestamp_s"]),
                "number_roi_ref": str(row["number_roi_ref"]),
                "readiness_score": float(row["readiness_score"]),
                "rank_within_tracklet": int(row["rank_within_tracklet"]),
            }
        )
    return manifest


def _normalise_state_dict(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model_state_dict"):
            candidate = payload.get(key)
            if isinstance(candidate, Mapping):
                payload = candidate
                break
    if not isinstance(payload, Mapping):
        raise ValueError("legibility checkpoint does not contain a state dict")

    result: dict[str, Any] = {}
    for key, value in payload.items():
        clean = str(key)
        if clean.startswith("module."):
            clean = clean[len("module.") :]
        result[clean] = value
    return result


def _load_legibility_model(weights: Path, device: str):
    try:
        import torch
        import torch.nn as nn
        from torchvision import models
    except ImportError as exc:
        raise RuntimeError("torch/torchvision are required") from exc

    class Legibility34(nn.Module):
        def __init__(self):
            super().__init__()
            self.model_ft = models.resnet34(weights=None)
            self.model_ft.fc = nn.Linear(self.model_ft.fc.in_features, 1)

        def forward(self, x):
            return torch.sigmoid(self.model_ft(x))

    model = Legibility34()
    state = _normalise_state_dict(torch.load(str(weights), map_location=device))
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return requested


def _run_legibility(
    *,
    manifest: Sequence[Mapping[str, Any]],
    input_dir: Path,
    weights: Path,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    import torch
    from torchvision import transforms

    resolved = _resolve_device(device)
    model = _load_legibility_model(weights, resolved)
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    scores: dict[str, float] = {}
    filenames = [str(row["filename"]) for row in manifest]

    for start in range(0, len(filenames), batch_size):
        names = filenames[start : start + batch_size]
        tensors = []
        for filename in names:
            image = cv2.imread(str(input_dir / filename), cv2.IMREAD_COLOR)
            if image is None:
                raise FileNotFoundError(input_dir / filename)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            tensors.append(transform(image))

        batch = torch.stack(tensors).to(resolved)
        with torch.inference_mode():
            values = model(batch).reshape(-1).detach().cpu().tolist()

        for filename, value in zip(names, values, strict=True):
            scores[filename] = float(value)

    return scores


def _pose_input_payload(
    manifest: Sequence[Mapping[str, Any]],
    *,
    input_dir: Path,
    scores: Mapping[str, float],
    threshold: float,
) -> tuple[dict[str, Any], set[str]]:
    images = []
    annotations = []
    legible: set[str] = set()

    for image_id, row in enumerate(
        item for item in manifest
        if float(scores.get(str(item["filename"]), 0.0)) >= threshold
    ):
        filename = str(row["filename"])
        image = cv2.imread(str(input_dir / filename), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(input_dir / filename)
        height, width = image.shape[:2]
        legible.add(filename)
        images.append(
            {
                "id": image_id,
                "file_name": str((input_dir / filename).resolve()),
                "width": width,
                "height": height,
            }
        )
        annotations.append(
            {
                "id": image_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [0, 0, width, height],
            }
        )

    return {"images": images, "annotations": annotations}, legible


def _pose_points(
    keypoints: Sequence[Sequence[float]],
    *,
    threshold: float,
) -> list[tuple[float, float]]:
    if len(keypoints) <= 12:
        return []
    selected = [keypoints[6], keypoints[5], keypoints[11], keypoints[12]]
    points = []
    for point in selected:
        if len(point) < 3 or float(point[2]) < threshold:
            return []
        points.append((float(point[0]), float(point[1])))
    return points


def _torso_box(
    image_shape: Sequence[int],
    keypoints: Sequence[Sequence[float]],
    *,
    confidence_threshold: float = 0.4,
    padding_px: int = 5,
) -> tuple[int, int, int, int] | None:
    points = _pose_points(keypoints, threshold=confidence_threshold)
    if not points:
        return None

    height, width = int(image_shape[0]), int(image_shape[1])
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x1 = max(0, int(math.floor(min(xs) - padding_px)))
    x2 = min(width, int(math.ceil(max(xs) + padding_px)))
    y1 = max(0, int(math.floor(min(ys) - padding_px)))
    y2 = min(height, int(math.ceil(max(ys))))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _load_pose(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("pose_results", [])
    return {Path(str(row["img_name"])).name: row for row in rows}


def _make_pose_crops(
    *,
    manifest: Sequence[Mapping[str, Any]],
    input_dir: Path,
    crop_dir: Path,
    legible: set[str],
    poses: Mapping[str, Mapping[str, Any]],
    confidence_threshold: float,
    padding_px: int,
) -> set[str]:
    crop_dir.mkdir(parents=True, exist_ok=True)
    saved: set[str] = set()

    for row in manifest:
        filename = str(row["filename"])
        if filename not in legible or filename not in poses:
            continue

        image = cv2.imread(str(input_dir / filename), cv2.IMREAD_COLOR)
        if image is None:
            continue

        box = _torso_box(
            image.shape,
            poses[filename].get("keypoints", []),
            confidence_threshold=confidence_threshold,
            padding_px=padding_px,
        )
        if box is None:
            continue

        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if cv2.imwrite(str(crop_dir / filename), crop):
            saved.add(filename)

    return saved


def _run(command: Sequence[str], *, cwd: Path) -> None:
    print("[soccer-benchmark] exec=" + " ".join(map(str, command)))
    subprocess.run(list(map(str, command)), cwd=str(cwd), check=True)



def _env_run_prefix(executable: str, env_name: str) -> list[str]:
    """Portable environment-run prefix for conda and micromamba.

    Do not use conda's ``--live-stream`` flag: micromamba 2.x does not expose
    that option. Plain ``run -n`` is supported by both tools.
    """
    return [executable, "run", "-n", env_name]


def _checkpoint_issue(
    path: Path,
    *,
    min_bytes: int,
    require_zip: bool = False,
) -> str | None:
    if not path.is_file():
        return f"missing: {path}"

    size = path.stat().st_size
    if size < min_bytes:
        return (
            f"too small ({size} bytes; expected >= {min_bytes}): {path}"
        )

    if require_zip and not zipfile.is_zipfile(path):
        return f"not a valid PyTorch ZIP checkpoint: {path}"

    return None


def _probe_environment(
    executable: str,
    env_name: str,
    code: str,
    *,
    cwd: Path,
) -> str | None:
    completed = subprocess.run(
        _env_run_prefix(executable, env_name)
        + ["python", "-c", code],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return None

    detail = (completed.stderr or completed.stdout or "").strip()
    if len(detail) > 600:
        detail = detail[-600:]
    return (
        f"environment {env_name!r} import probe failed"
        + (f": {detail}" if detail else "")
    )


def _run_reference_pose(args, pose_input: Path, pose_output: Path) -> None:
    reference_root = Path(args.reference_root).expanduser()
    command = _env_run_prefix(
        args.conda_exe,
        args.pose_env,
    ) + [
        "python3",
        str((reference_root / "pose.py").resolve()),
        str(Path(args.pose_config).expanduser().resolve()),
        str(Path(args.pose_checkpoint).expanduser().resolve()),
        "--img-root",
        "/",
        "--json-file",
        str(pose_input.resolve()),
        "--out-json",
        str(pose_output.resolve()),
        "--device",
        args.pose_device,
    ]
    _run(command, cwd=reference_root)


def _run_reference_str(args, str_root: Path, result_file: Path) -> None:
    reference_root = Path(args.reference_root).expanduser()
    command = _env_run_prefix(
        args.conda_exe,
        args.str_env,
    ) + [
        "python3",
        str((reference_root / "str.py").resolve()),
        str(Path(args.str_checkpoint).expanduser().resolve()),
        f"--data_root={str_root.resolve()}",
        "--batch_size=1",
        "--inference",
        f"--result_file={result_file.resolve()}",
        f"--device={args.str_device}",
    ]
    _run(command, cwd=reference_root)


def _prediction_row(
    source: Mapping[str, Any],
    *,
    legibility_score: float,
    legible: bool,
    pose_crop: bool,
    str_row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    audit_raw = None if str_row is None else str(str_row.get("label", ""))
    parsed = _strict_number(audit_raw)
    confidence = (
        _sequence_confidence(str_row.get("confidence"))
        if str_row is not None
        else 0.0
    )

    accepted = legible and pose_crop and parsed is not None
    if not legible:
        status = "illegible"
    elif not pose_crop:
        status = "pose_crop_missing"
    elif str_row is None:
        status = "str_missing"
    elif parsed is None:
        status = "str_invalid"
    else:
        status = "recognized"

    return {
        "schema_version": 2,
        "backend": BACKEND_NAME,
        "selection": "strongest",
        "preprocess_profile": "reference-soccernet",
        "tracklet_id": str(source["tracklet_id"]),
        "detection_id": str(source["detection_id"]),
        "sample_id": str(source["sample_id"]),
        "frame_index": int(source["frame_index"]),
        "timestamp_s": float(source["timestamp_s"]),
        "number_roi_ref": str(source["number_roi_ref"]),
        "readiness_score": float(source["readiness_score"]),
        "rank_within_tracklet": int(source["rank_within_tracklet"]),
        "raw_text": str(parsed) if accepted else None,
        "parsed_number": int(parsed) if accepted else None,
        "ocr_confidence": float(confidence if accepted else 0.0),
        "chosen_variant": "reference-pose-crop" if accepted else None,
        "variant_support_count": 1 if accepted else 0,
        "variant_consensus_score": float(confidence if accepted else 0.0),
        "variant_predictions": [],
        "benchmark_status": status,
        "legibility_score": float(legibility_score),
        "legibility_accepted": bool(legible),
        "pose_crop_available": bool(pose_crop),
        "audit_raw_text": audit_raw,
        "audit_str_confidence": float(confidence),
        "research_only_noncommercial_reference": True,
    }


def _conda_envs(conda_exe: str) -> set[str]:
    completed = subprocess.run(
        [conda_exe, "env", "list", "--json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return set()
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return set()
    return {Path(path).name for path in payload.get("envs", [])}


def check_prerequisites(args) -> list[str]:
    missing: list[str] = []
    reference_root = Path(args.reference_root).expanduser()

    if not reference_root.is_dir():
        return [
            f"reference repo missing: {reference_root} "
            f"(clone {REFERENCE_REPO_URL})"
        ]

    required_files = [
        ("reference pose.py", reference_root / "pose.py"),
        ("reference str.py", reference_root / "str.py"),
        ("ViTPose config", Path(args.pose_config).expanduser()),
    ]
    for label, path in required_files:
        if not path.is_file():
            missing.append(f"{label} missing: {path}")

    legibility_path = Path(args.legibility_weights).expanduser()
    issue = _checkpoint_issue(
        legibility_path,
        min_bytes=50_000_000,
    )
    if issue:
        missing.append(f"SoccerNet legibility weights {issue}")

    str_path = Path(args.str_checkpoint).expanduser()
    issue = _checkpoint_issue(
        str_path,
        min_bytes=100_000_000,
    )
    if issue:
        missing.append(f"SoccerNet PARSeq checkpoint {issue}")

    pose_path = Path(args.pose_checkpoint).expanduser()
    issue = _checkpoint_issue(
        pose_path,
        min_bytes=1_000_000_000,
        require_zip=True,
    )
    if issue:
        missing.append(f"ViTPose checkpoint {issue}")

    conda_exe = shutil.which(args.conda_exe)
    if conda_exe is None:
        missing.append(f"environment executable not found: {args.conda_exe}")
        return missing

    envs = _conda_envs(conda_exe)
    if args.pose_env not in envs:
        missing.append(f"environment missing: {args.pose_env}")
    else:
        pose_repo = reference_root / "pose" / "ViTPose"
        pose_probe = (
            "import sys, torch, mmcv; "
            f"sys.path.insert(0, {str(pose_repo)!r}); "
            "import mmpose; "
            "from mmpose.apis import init_pose_model, inference_top_down_pose_model; "
            "assert torch.cuda.is_available()"
        )
        issue = _probe_environment(
            conda_exe,
            args.pose_env,
            pose_probe,
            cwd=reference_root,
        )
        if issue:
            missing.append(issue)

    if args.str_env not in envs:
        missing.append(f"environment missing: {args.str_env}")
    else:
        str_probe = (
            "import torch, pytorch_lightning, timm, strhub; "
            "assert torch.cuda.is_available()"
        )
        issue = _probe_environment(
            conda_exe,
            args.str_env,
            str_probe,
            cwd=reference_root,
        )
        if issue:
            missing.append(issue)

    return missing


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id")
    parser.add_argument("--clip-id")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--reference-root",
        default=str(Path.home() / "Tools" / "jersey-number-pipeline"),
    )
    parser.add_argument("--conda-exe", default="conda")
    parser.add_argument("--pose-env", default="vitpose")
    parser.add_argument("--str-env", default="parseq2")
    parser.add_argument("--legibility-weights")
    parser.add_argument("--str-checkpoint")
    parser.add_argument("--pose-config")
    parser.add_argument("--pose-checkpoint")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pose-device", default="cuda:0")
    parser.add_argument("--str-device", default="cuda")
    parser.add_argument("--legibility-threshold", type=float, default=0.5)
    parser.add_argument("--legibility-batch-size", type=int, default=32)
    parser.add_argument("--pose-confidence", type=float, default=0.4)
    parser.add_argument("--pose-padding-px", type=int, default=5)
    parser.add_argument("--work-dir")
    parser.add_argument("--output")
    parser.add_argument("--audit-dir")
    parser.add_argument("--pose-json")
    parser.add_argument("--str-results")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--check-prereqs", action="store_true")
    args = parser.parse_args(argv)

    reference_root = Path(args.reference_root).expanduser()
    if args.legibility_weights is None:
        args.legibility_weights = str(reference_root / "models" / LEGIBILITY_FILENAME)
    if args.str_checkpoint is None:
        args.str_checkpoint = str(reference_root / "models" / STR_FILENAME)
    if args.pose_config is None:
        args.pose_config = str(
            reference_root
            / "pose"
            / "ViTPose"
            / "configs"
            / "body"
            / "2d_kpt_sview_rgb_img"
            / "topdown_heatmap"
            / "coco"
            / "ViTPose_huge_coco_256x192.py"
        )
    if args.pose_checkpoint is None:
        args.pose_checkpoint = str(
            reference_root
            / "pose"
            / "ViTPose"
            / "checkpoints"
            / "vitpose-h.pth"
        )
    return args


def main(argv=None) -> int:
    args = parse_args(argv)

    if args.check_prereqs:
        missing = check_prerequisites(args)
        if missing:
            print("[soccer-benchmark] prerequisites NOT ready")
            for item in missing:
                print(f"  - {item}")
            return 2
        print("[soccer-benchmark] prerequisites OK")
        return 0

    if not args.match_id or not args.clip_id:
        raise ValueError("--match-id and --clip-id are required")

    missing = check_prerequisites(args)
    if missing:
        raise RuntimeError(
            "benchmark prerequisites are not ready:\n"
            + "\n".join(f"- {item}" for item in missing)
        )

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root
    selected_path = (
        root
        / "diagnostics"
        / "jersey_view"
        / "jersey-view-v1"
        / "selected.jsonl"
    )
    selected_rows = _select_rows(
        list(_iter_jsonl(selected_path)),
        selection="strongest",
        limit=args.limit,
    )

    work_dir = (
        Path(args.work_dir)
        if args.work_dir
        else Path("/tmp") / "jersey_soccer_reference_benchmark"
    )
    input_dir = work_dir / "input_views"
    str_root = work_dir / "str_data"
    crop_dir = str_root / "imgs"
    work_dir.mkdir(parents=True, exist_ok=True)

    manifest = _materialize_selected_views(
        root=root,
        rows=selected_rows,
        input_dir=input_dir,
    )
    _write_json(work_dir / "manifest.json", {"rows": manifest})

    scores = _run_legibility(
        manifest=manifest,
        input_dir=input_dir,
        weights=Path(args.legibility_weights).expanduser(),
        device=args.device,
        batch_size=args.legibility_batch_size,
    )
    _write_json(work_dir / "legibility.json", {"scores": scores})

    pose_payload, legible = _pose_input_payload(
        manifest,
        input_dir=input_dir,
        scores=scores,
        threshold=args.legibility_threshold,
    )
    pose_input = work_dir / "pose_input.json"
    _write_json(pose_input, pose_payload)

    print(
        "[soccer-benchmark] "
        f"views={len(manifest)} legible={len(legible)}"
    )

    if args.prepare_only:
        print(f"[soccer-benchmark] prepared={work_dir}")
        return 0

    pose_output = (
        Path(args.pose_json)
        if args.pose_json
        else work_dir / "pose_results.json"
    )
    if not args.pose_json:
        _run_reference_pose(args, pose_input, pose_output)

    poses = _load_pose(pose_output)
    pose_crops = _make_pose_crops(
        manifest=manifest,
        input_dir=input_dir,
        crop_dir=crop_dir,
        legible=legible,
        poses=poses,
        confidence_threshold=args.pose_confidence,
        padding_px=args.pose_padding_px,
    )

    print(
        "[soccer-benchmark] "
        f"pose_crops={len(pose_crops)}/{len(legible)}"
    )

    str_results_path = (
        Path(args.str_results)
        if args.str_results
        else work_dir / "str_results.json"
    )
    if not args.str_results:
        _run_reference_str(args, str_root, str_results_path)

    str_results = json.loads(str_results_path.read_text(encoding="utf-8"))

    predictions = []
    audit_items = []
    recognized = 0

    for row in manifest:
        filename = str(row["filename"])
        prediction = _prediction_row(
            row,
            legibility_score=float(scores.get(filename, 0.0)),
            legible=filename in legible,
            pose_crop=filename in pose_crops,
            str_row=str_results.get(filename),
        )
        predictions.append(prediction)
        if prediction["parsed_number"] is not None:
            recognized += 1

        image = cv2.imread(str(input_dir / filename), cv2.IMREAD_COLOR)
        if image is not None:
            audit_items.append({**prediction, "image": image})

    output_path = (
        Path(args.output)
        if args.output
        else work_dir / "predictions.jsonl"
    )
    _write_jsonl(output_path, predictions)

    audit_dir = Path(args.audit_dir) if args.audit_dir else work_dir
    audit_dir.mkdir(parents=True, exist_ok=True)

    recognized_items = [
        item for item in audit_items
        if item["parsed_number"] is not None
    ]
    unknown_items = [
        item for item in audit_items
        if item["parsed_number"] is None
    ]
    recognized_items.sort(
        key=lambda item: (
            -float(item["ocr_confidence"]),
            -float(item["legibility_score"]),
        )
    )
    unknown_items.sort(
        key=lambda item: (
            -float(item["legibility_score"]),
            -float(item["readiness_score"]),
        )
    )

    recognized_sheet = _contact_sheet(recognized_items[:100])
    unknown_sheet = _contact_sheet(unknown_items[:100])

    if recognized_sheet is not None:
        cv2.imwrite(
            str(audit_dir / "recognized_contact_sheet.jpg"),
            recognized_sheet,
        )
    if unknown_sheet is not None:
        cv2.imwrite(
            str(audit_dir / "unknown_contact_sheet.jpg"),
            unknown_sheet,
        )

    statuses: dict[str, int] = {}
    for row in predictions:
        status = str(row["benchmark_status"])
        statuses[status] = statuses.get(status, 0) + 1

    summary = {
        "backend": BACKEND_NAME,
        "research_only_noncommercial_reference": True,
        "views": len(predictions),
        "legible_views": len(legible),
        "pose_crops": len(pose_crops),
        "recognized": recognized,
        "recognized_fraction": (
            recognized / len(predictions)
            if predictions
            else 0.0
        ),
        "status_counts": dict(sorted(statuses.items())),
    }
    _write_json(audit_dir / "summary.json", summary)

    print(
        "[soccer-benchmark] "
        f"views={len(predictions)} "
        f"legible={len(legible)} "
        f"pose_crops={len(pose_crops)} "
        f"recognized={recognized} "
        f"fraction={summary['recognized_fraction']:.6f}"
    )
    print(f"[soccer-benchmark] artifact={output_path}")
    print(f"[soccer-benchmark] summary={audit_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
