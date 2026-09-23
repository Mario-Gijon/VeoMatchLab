#!/usr/bin/env python3
"""Inspect, normalize, merge, and audit VeoMatch person datasets.

The supported interchange format is COCO detection JSON. All generated
datasets contain exactly one category: ``person``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLASS_MAP = {
    "player": "person", "Player": "person",
    "referee": "person", "Referee": "person",
    "goalkeeper": "person", "Goalkeeper": "person",
    "person": "person",
}
SPLITS = ("train", "valid", "test")


def normalize_class(name: str) -> str | None:
    """Return the allowed class name, or ``None`` for unknown/non-person."""
    return CLASS_MAP.get(str(name))


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    """Compute a small average hash for review-only near-duplicate reports."""
    try:
        import cv2
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"could not decode image: {path}")
        values = cv2.resize(image, (16, 16), interpolation=cv2.INTER_AREA)
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for near-duplicate checks") from exc
    threshold = float(values.mean())
    return "".join("1" if value >= threshold else "0" for value in values.ravel())


def _split(value: str | None, fallback: str = "train") -> str:
    value = (value or fallback).lower()
    if value in {"val", "validation"}:
        value = "valid"
    return value if value in SPLITS else fallback


def _find_coco(source: Path) -> Path:
    candidates = sorted(source.rglob("*.json"))
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and {"images", "annotations", "categories"} <= payload.keys():
            return candidate
    raise FileNotFoundError(f"No COCO annotation JSON found under {source}")


def _coco_records(source: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    annotation_path = _find_coco(source)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in payload.get("annotations", []):
        annotations_by_image[int(annotation["image_id"])].append(annotation)
    records = []
    unknown: list[str] = []
    for image in payload.get("images", []):
        source_image = (annotation_path.parent / image["file_name"]).resolve()
        if not source_image.exists():
            source_image = (source / image["file_name"]).resolve()
        split = _split(image.get("split"), Path(image["file_name"]).parts[0] if Path(image["file_name"]).parts else "train")
        mapped = []
        original_classes = set()
        original_counts = Counter()
        for annotation in annotations_by_image.get(int(image["id"]), []):
            original = categories.get(int(annotation.get("category_id", -1)), "")
            original_classes.add(original)
            original_counts[original] += 1
            normalized = normalize_class(original)
            if normalized is None:
                if original:
                    unknown.append(original)
                continue
            mapped.append({"bbox": annotation["bbox"], "category_id": 1, "area": annotation.get("area", 0), "iscrowd": annotation.get("iscrowd", 0)})
        records.append({"source_image": source_image, "source_split": split, "width": int(image.get("width", 0)), "height": int(image.get("height", 0)), "original_classes": sorted(original_classes), "original_counts": dict(original_counts), "annotations": mapped})
    return payload, records, sorted(set(unknown))


def inspect_dataset(source: str | Path) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    payload, records, unknown = _coco_records(source)
    original_counts = Counter()
    mapped_counts = Counter()
    split_counts = Counter()
    annotation_count = 0
    for record in records:
        split_counts[record["source_split"]] += 1
        original_counts.update(record["original_counts"])
        mapped_counts["person"] += len(record["annotations"])
        annotation_count += len(record["annotations"])
    return {
        "source": str(source),
        "image_count": len(records),
        "annotation_count": annotation_count,
        "original_classes": sorted({name for record in records for name in record["original_classes"]}),
        "annotations_per_class": dict(original_counts),
        "mapped_annotations_per_class": dict(mapped_counts),
        "split_sizes": {split: split_counts[split] for split in SPLITS},
        "unknown_classes": unknown,
    }


def convert_coco(source: str | Path, output: str | Path, *, source_name: str) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    _, records, unknown = _coco_records(source)
    output.mkdir(parents=True, exist_ok=True)
    images_root = output / "images"
    for split in SPLITS:
        (images_root / split).mkdir(parents=True, exist_ok=True)
    manifest = []
    coco_images = []
    coco_annotations = []
    next_image_id = 1
    next_annotation_id = 1
    for record in records:
        if not record["source_image"].exists():
            raise FileNotFoundError(record["source_image"])
        digest = file_hash(record["source_image"])
        name = f"{digest[:16]}_{record['source_image'].name}"
        destination = images_root / record["source_split"] / name
        shutil.copy2(record["source_image"], destination)
        coco_images.append({"id": next_image_id, "file_name": str(Path("images") / record["source_split"] / name), "width": record["width"], "height": record["height"], "split": record["source_split"]})
        for annotation in record["annotations"]:
            item = dict(annotation)
            item.update({"id": next_annotation_id, "image_id": next_image_id})
            coco_annotations.append(item)
            next_annotation_id += 1
        manifest.append({"source_dataset": source_name, "original_image_path": str(record["source_image"]), "generated_image_path": str(destination), "source_split": record["source_split"], "final_split": record["source_split"], "original_classes": record["original_classes"], "mapped_person_annotation_count": len(record["annotations"]), "file_hash": digest})
        next_image_id += 1
    dataset = {"images": coco_images, "annotations": coco_annotations, "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
    (output / "annotations.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"dataset": source_name, "unknown_classes": unknown, "images": manifest}, indent=2) + "\n", encoding="utf-8")
    if unknown:
        (output / "unknown_classes.json").write_text(json.dumps(unknown, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "images": len(manifest), "annotations": len(coco_annotations), "unknown_classes": unknown}


def merge_datasets(inputs: list[str | Path], output: str | Path, *, dataset_name: str) -> dict[str, Any]:
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
    image_rows = []
    annotations = []
    seen_hashes: dict[str, dict[str, Any]] = {}
    seen_phashes: dict[str, list[str]] = defaultdict(list)
    near_duplicates = []
    next_image_id = 1
    next_annotation_id = 1
    exact_duplicates_removed = 0
    split_conflicts = []
    for dataset_path in inputs:
        dataset_path = Path(dataset_path)
        payload = json.loads((dataset_path / "annotations.json").read_text(encoding="utf-8"))
        manifest = json.loads((dataset_path / "manifest.json").read_text(encoding="utf-8"))["images"]
        by_name = {row["generated_image_path"]: row for row in manifest}
        source_images = {item["id"]: item for item in payload["images"]}
        anns = defaultdict(list)
        for annotation in payload["annotations"]:
            anns[annotation["image_id"]].append(annotation)
        for image_id, image in source_images.items():
            row = by_name.get(str(dataset_path / image["file_name"])) or by_name.get(image["file_name"])
            if row is None:
                raise ValueError(f"Manifest missing image {image['file_name']}")
            source_image = Path(row["generated_image_path"])
            digest = row["file_hash"]
            if digest in seen_hashes:
                exact_duplicates_removed += 1
                if seen_hashes[digest]["split"] != _split(row.get("final_split"), "train"):
                    split_conflicts.append({"file_hash": digest, "existing": seen_hashes[digest], "duplicate_source": str(source_image), "duplicate_split": row.get("final_split")})
                continue
            phash = perceptual_hash(source_image)
            if phash in seen_phashes:
                near_duplicates.append({"file_hash": digest, "matches": seen_phashes[phash], "path": str(source_image)})
            seen_phashes[phash].append(str(source_image))
            split = _split(row.get("final_split"), "train")
            destination = output / "images" / split / f"{digest[:16]}_{source_image.name}"
            shutil.copy2(source_image, destination)
            seen_hashes[digest] = {"split": split, "path": str(destination)}
            new_id = next_image_id
            image_rows.append({"id": new_id, "file_name": str(Path("images") / split / destination.name), "width": image.get("width", 0), "height": image.get("height", 0), "split": split})
            for annotation in anns[image_id]:
                item = dict(annotation)
                item.update({"id": next_annotation_id, "image_id": new_id})
                annotations.append(item)
                next_annotation_id += 1
            row = dict(row)
            row.update({"generated_image_path": str(destination), "final_split": split})
            image_rows[-1]["_manifest"] = row
            next_image_id += 1
    final_manifest = [item.pop("_manifest") for item in image_rows]
    dataset = {"images": [{k: v for k, v in item.items() if k != "_manifest"} for item in image_rows], "annotations": annotations, "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
    (output / "annotations.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"dataset": dataset_name, "images": final_manifest, "near_duplicates_for_review": near_duplicates, "split_conflicts": split_conflicts}, indent=2) + "\n", encoding="utf-8")
    (output / "near_duplicates.json").write_text(json.dumps(near_duplicates, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "images": len(final_manifest), "exact_duplicates_removed": exact_duplicates_removed, "near_duplicates_for_review": len(near_duplicates), "split_conflicts": len(split_conflicts)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect_parser = sub.add_parser("inspect")
    inspect_parser.add_argument("source", type=Path)
    convert_parser = sub.add_parser("convert-coco")
    convert_parser.add_argument("source", type=Path)
    convert_parser.add_argument("output", type=Path)
    convert_parser.add_argument("--source-name", required=True)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        print(json.dumps(inspect_dataset(args.source), indent=2))
    else:
        print(json.dumps(convert_coco(args.source, args.output, source_name=args.source_name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
