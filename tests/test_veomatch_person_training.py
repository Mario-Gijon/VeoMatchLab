import json
from pathlib import Path

from person_detect.training.veomatch_person_v1.tools.dataset_tools import (
    convert_coco,
    merge_datasets,
    normalize_class,
)


def _ppm(path: Path, value: int) -> None:
    path.write_bytes(f"P3\n2 2\n255\n{value} {value} {value} {value} {value} {value} {value} {value} {value} {value} {value} {value}\n".encode())


def _source(root: Path, name: str, *, split: str = "train") -> Path:
    images = root / name / "images" / split
    images.mkdir(parents=True)
    image = images / "frame.ppm"
    _ppm(image, 100)
    (root / name / "annotations.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": f"images/{split}/frame.ppm", "width": 2, "height": 2, "split": split}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        "categories": [{"id": 1, "name": "player"}],
    }))
    return root / name


def test_person_class_mapping_is_conservative():
    assert normalize_class("player") == "person"
    assert normalize_class("Referee") == "person"
    assert normalize_class("goalkeeper") == "person"
    assert normalize_class("ball") is None
    assert normalize_class("unknown") is None


def test_exact_duplicate_is_removed_and_split_conflict_reported(tmp_path):
    first = _source(tmp_path, "first", split="train")
    second = _source(tmp_path, "second", split="test")
    first_dataset = tmp_path / "converted_first"
    second_dataset = tmp_path / "converted_second"
    convert_coco(first, first_dataset, source_name="first")
    convert_coco(second, second_dataset, source_name="second")
    merged = tmp_path / "merged"
    result = merge_datasets([first_dataset, second_dataset], merged, dataset_name="merged")
    assert result["exact_duplicates_removed"] == 1
    manifest = json.loads((merged / "manifest.json").read_text())
    assert manifest["split_conflicts"]


def test_datasets_keep_one_copy_per_hash_across_splits(tmp_path):
    source = _source(tmp_path, "source", split="train")
    converted = tmp_path / "converted"
    convert_coco(source, converted, source_name="source")
    merged = tmp_path / "merged"
    merge_datasets([converted], merged, dataset_name="merged")
    payload = json.loads((merged / "annotations.json").read_text())
    files = [item["file_name"] for item in payload["images"]]
    assert len(files) == len(set(files)) == 1
