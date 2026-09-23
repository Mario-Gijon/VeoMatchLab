#!/usr/bin/env python3
"""Inspect, normalize, merge, and audit VeoMatch person datasets.

The supported interchange format is COCO detection JSON. All generated
datasets contain exactly one category: ``person``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CLASS_MAP = {
    "player": "person", "referee": "person", "goalkeeper": "person", "person": "person",
}
IGNORED_CLASSES = {"ball", "football", "logo"}
SPLITS = ("train", "valid", "test")


def sequence_group_from_path(path: str | Path) -> str:
    """Normalize a source filename into a conservative temporal sequence key."""
    stem = Path(path).stem
    stem = re.sub(r"\.rf\.[A-Za-z0-9]+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?:_jpg|_jpeg|_png|_webp)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"(?:[_-]?frame|[_-]?image|[_-]?img)[_-]?\d+$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[_-]\d{3,}$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_").casefold()
    return stem or Path(path).stem.casefold()


def normalize_class(name: str) -> str | None:
    """Return the allowed class name, or ``None`` for unknown/non-person."""
    return CLASS_MAP.get(str(name).strip().casefold())


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


def _coco_paths(source: Path) -> list[tuple[str, Path]]:
    paths = []
    for split in SPLITS:
        split_dir = source / split
        if not split_dir.is_dir():
            continue
        preferred = split_dir / "_annotations.coco.json"
        candidates = [preferred] if preferred.is_file() else sorted(split_dir.glob("*.json"))
        for candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict) and {"images", "annotations", "categories"} <= payload.keys():
                paths.append((split, candidate))
                break
    if not paths:
        raise FileNotFoundError(f"No split COCO annotation JSON found under {source}")
    return paths


def _coco_records(source: Path) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records = []
    unknown: list[str] = []
    ignored: list[str] = []
    for split, annotation_path in _coco_paths(source):
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
        categories = {int(item["id"]): str(item["name"]) for item in payload.get("categories", [])}
        annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in payload.get("annotations", []):
            annotations_by_image[int(annotation["image_id"])].append(annotation)
        for image in payload.get("images", []):
            source_image = (annotation_path.parent / image["file_name"]).resolve()
            if not source_image.exists():
                source_image = (source / image["file_name"]).resolve()
            mapped = []
            original_classes = set()
            original_counts = Counter()
            for annotation in annotations_by_image.get(int(image["id"]), []):
                original = categories.get(int(annotation.get("category_id", -1)), "")
                original_classes.add(original)
                original_counts[original] += 1
                normalized = normalize_class(original)
                if normalized is None:
                    if original.casefold() in IGNORED_CLASSES:
                        ignored.append(original)
                    elif original:
                        unknown.append(original)
                    continue
                mapped.append({"bbox": annotation["bbox"], "category_id": 1, "area": annotation.get("area", 0), "iscrowd": annotation.get("iscrowd", 0)})
            records.append({"source_image": source_image, "source_split": split, "width": int(image.get("width", 0)), "height": int(image.get("height", 0)), "original_classes": sorted(original_classes), "original_counts": dict(original_counts), "sequence_group": sequence_group_from_path(image["file_name"]), "annotations": mapped})
    return records, sorted(set(unknown)), sorted(set(ignored))


def inspect_dataset(source: str | Path) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    records, unknown, ignored = _coco_records(source)
    original_counts = Counter()
    split_counts = Counter()
    original_annotation_count = 0
    mapped_person_annotation_count = 0
    ignored_annotation_count = 0
    for record in records:
        split_counts[record["source_split"]] += 1
        original_counts.update(record["original_counts"])
        mapped_person_annotation_count += len(record["annotations"])
        original_annotation_count += sum(record["original_counts"].values())
        ignored_annotation_count += sum(count for name, count in record["original_counts"].items() if name.casefold() in IGNORED_CLASSES)
    return {
        "source": str(source),
        "image_count": len(records),
        "original_annotation_count": original_annotation_count,
        "mapped_person_annotation_count": mapped_person_annotation_count,
        "ignored_annotation_count": ignored_annotation_count,
        "original_classes": sorted({name for record in records for name in record["original_classes"]}),
        "annotations_per_class": dict(original_counts),
        "mapped_annotations_per_class": {"person": mapped_person_annotation_count},
        "ignored_classes": ignored,
        "split_sizes": {split: split_counts[split] for split in SPLITS},
        "unknown_classes": unknown,
    }


def convert_coco(source: str | Path, output: str | Path, *, source_name: str) -> dict[str, Any]:
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    records, unknown, ignored = _coco_records(source)
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
        manifest.append({"source_dataset": source_name, "original_image_path": str(record["source_image"]), "generated_image_path": str(destination), "source_split": record["source_split"], "final_split": record["source_split"], "sequence_group": record["sequence_group"], "original_classes": record["original_classes"], "mapped_person_annotation_count": len(record["annotations"]), "file_hash": digest})
        next_image_id += 1
    dataset = {"images": coco_images, "annotations": coco_annotations, "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
    (output / "annotations.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({"dataset": source_name, "unknown_classes": unknown, "ignored_classes": ignored, "images": manifest}, indent=2) + "\n", encoding="utf-8")
    if unknown:
        (output / "unknown_classes.json").write_text(json.dumps(unknown, indent=2) + "\n", encoding="utf-8")
    (output / "ignored_classes.json").write_text(json.dumps(ignored, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "images": len(manifest), "annotations": len(coco_annotations), "unknown_classes": unknown}


def _assign_group_splits(rows: list[dict[str, Any]]) -> dict[str, str]:
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[row["sequence_group"]][row["source_split"]] += 1
    target_total = sum(sum(counts.values()) for counts in grouped.values())
    target = {split: target_total * sum(counts[split] for counts in grouped.values()) / target_total for split in SPLITS}
    current = Counter()
    assignments = {}
    group_rows = defaultdict(list)
    for row in rows:
        group_rows[row["sequence_group"]].append(row)
    ordered_groups = sorted(grouped, key=lambda group: (-sum(grouped[group].values()), group))
    for group in ordered_groups:
        counts = grouped[group]
        maximum = max(counts.values())
        preferred = [split for split in SPLITS if counts[split] == maximum]
        if all(row.get("source_dataset") == "football_eoakc" for row in group_rows[group]):
            choices = ("train",)
        else:
            choices = SPLITS
        size = sum(counts.values())
        def score(split: str) -> tuple[float, int]:
            deviation = sum(
                ((current[item] + (size if item == split else 0)) - target[item]) ** 2
                for item in SPLITS
            )
            return deviation, 0 if split in preferred else 1
        chosen = min(choices, key=score)
        assignments[group] = chosen
        current[chosen] += size
    return assignments


def merge_datasets(inputs: list[str | Path], output: str | Path, *, dataset_name: str, group_split_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    """Merge converted datasets, assigning each sequence group to one split."""
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
    candidates = []
    seen_hashes: dict[str, dict[str, Any]] = {}
    exact_duplicates_removed = 0
    split_conflicts = []
    for dataset_path_raw in inputs:
        dataset_path = Path(dataset_path_raw).expanduser().resolve()
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
                if seen_hashes[digest]["source_split"] != row.get("source_split"):
                    split_conflicts.append({"file_hash": digest, "existing_source": seen_hashes[digest].get("original_image_path"), "existing_split": seen_hashes[digest].get("source_split"), "duplicate_source": str(source_image), "duplicate_split": row.get("source_split")})
                continue
            row = dict(row)
            row["sequence_group"] = row.get("sequence_group") or sequence_group_from_path(row["original_image_path"])
            row["_source_image"] = source_image
            row["_image"] = image
            row["_annotations"] = anns[image_id]
            candidates.append(row)
            seen_hashes[digest] = row

    assignments = _assign_group_splits(candidates)
    if group_split_overrides:
        assignments.update({group: _split(split) for group, split in group_split_overrides.items()})
    image_rows = []
    annotations = []
    phash_buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    near_duplicates = []
    next_image_id = 1
    next_annotation_id = 1
    for row in candidates:
        source_image = row["_source_image"]
        split = assignments[row["sequence_group"]]
        destination = output / "images" / split / f"{row['file_hash'][:16]}_{source_image.name}"
        shutil.copy2(source_image, destination)
        phash = perceptual_hash(source_image)
        for previous in phash_buckets[phash]:
            near_duplicates.append({"a": previous, "b": {"file_hash": row["file_hash"], "path": str(destination), "split": split, "sequence_group": row["sequence_group"]}, "same_sequence_group": previous["sequence_group"] == row["sequence_group"]})
        phash_buckets[phash].append({"file_hash": row["file_hash"], "path": str(destination), "split": split, "sequence_group": row["sequence_group"]})
        new_id = next_image_id
        image_rows.append({"id": new_id, "file_name": str(Path("images") / split / destination.name), "width": row["_image"].get("width", 0), "height": row["_image"].get("height", 0), "split": split})
        for annotation in row["_annotations"]:
            item = dict(annotation)
            item.update({"id": next_annotation_id, "image_id": new_id})
            annotations.append(item)
            next_annotation_id += 1
        provenance = {key: value for key, value in row.items() if not key.startswith("_")}
        provenance.update({"generated_image_path": str(destination), "final_split": split})
        image_rows[-1]["_manifest"] = provenance
        next_image_id += 1
    final_manifest = [item.pop("_manifest") for item in image_rows]
    sequence_splits: dict[str, set[str]] = defaultdict(set)
    for row in final_manifest:
        sequence_splits[row["sequence_group"]].add(row["final_split"])
    leakage = {group: sorted(splits) for group, splits in sequence_splits.items() if len(splits) > 1}
    pair_summary = {f"{a}__{b}": 0 for a in SPLITS for b in SPLITS if SPLITS.index(a) <= SPLITS.index(b)}
    for pair in near_duplicates:
        a, b = pair["a"]["split"], pair["b"]["split"]
        key = f"{a}__{b}" if SPLITS.index(a) <= SPLITS.index(b) else f"{b}__{a}"
        pair_summary[key] += 1
    dataset = {"images": [{k: v for k, v in item.items() if k != "_manifest"} for item in image_rows], "annotations": annotations, "categories": [{"id": 1, "name": "person", "supercategory": "person"}]}
    (output / "annotations.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    audit = {"total_sequence_groups": len(sequence_splits), "groups_by_split": {split: sum(1 for values in sequence_splits.values() if values == {split}) for split in SPLITS}, "sequence_groups_spanning_splits": leakage, "sequence_group_leakage_count": len(leakage), "near_duplicate_pair_counts": pair_summary, "perceptual_cross_split_pairs": sum(value for key, value in pair_summary.items() if key.split("__")[0] != key.split("__")[1]), "near_duplicate_pairs_same_sequence_group": sum(item["same_sequence_group"] for item in near_duplicates), "near_duplicate_pairs_perceptual_only": sum(not item["same_sequence_group"] for item in near_duplicates), "exact_duplicates_removed": exact_duplicates_removed, "exact_duplicate_split_conflicts": split_conflicts}
    (output / "manifest.json").write_text(json.dumps({"dataset": dataset_name, "images": final_manifest, "sequence_audit": audit}, indent=2) + "\n", encoding="utf-8")
    (output / "near_duplicates.json").write_text(json.dumps({"pairs": near_duplicates, "summary": pair_summary, "sequence_group_leakage": leakage}, indent=2) + "\n", encoding="utf-8")
    return {"output": str(output), "images": len(final_manifest), "split_sizes": {split: sum(1 for row in final_manifest if row["final_split"] == split) for split in SPLITS}, "sequence_groups": len(sequence_splits), "sequence_group_leakage_count": len(leakage), "perceptual_cross_split_pairs": audit["perceptual_cross_split_pairs"], "exact_duplicates_removed": exact_duplicates_removed, "sequence_assignments": assignments}


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
