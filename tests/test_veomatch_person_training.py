import json
from pathlib import Path

from person_detect.training.veomatch_person_v1.tools.dataset_tools import (
    convert_coco,
    merge_datasets,
    normalize_class,
    sequence_group_from_path,
)
from person_detect.training.veomatch_person_v1.train import (
    _find_best,
    _ensure_run_dir,
    _validate_dataset,
    canonical_experiment,
    load_config,
    model_constructor_kwargs,
    resolve_num_classes,
)


def _ppm(path: Path, value: int) -> None:
    path.write_bytes(f"P3\n2 2\n255\n{value} {value} {value} {value} {value} {value} {value} {value} {value} {value} {value} {value}\n".encode())


def _source(root: Path, name: str, *, split: str = "train", category: str = "player") -> Path:
    images = root / name / "images" / split
    images.mkdir(parents=True)
    (root / name / split).mkdir(parents=True, exist_ok=True)
    image = images / "frame.ppm"
    _ppm(image, 100)
    (root / name / split / "_annotations.coco.json").write_text(json.dumps({
        "images": [{"id": 1, "file_name": "../images/%s/frame.ppm" % split, "width": 2, "height": 2}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 1, 1]}],
        "categories": [{"id": 1, "name": category}],
    }))
    return root / name


def test_person_class_mapping_is_conservative():
    assert normalize_class("player") == "person"
    assert normalize_class("Referee") == "person"
    assert normalize_class("Person") == "person"
    assert normalize_class("goalkeeper") == "person"
    assert normalize_class("ball") is None
    assert normalize_class("unknown") is None
    assert sequence_group_from_path("15-004519_Shot_on_goal_frame_0403.jpg") == "15_004519_shot_on_goal"
    assert sequence_group_from_path("15-004519_Shot_on_goal_frame_0414.jpg") == "15_004519_shot_on_goal"


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
    assert manifest["sequence_audit"]["exact_duplicate_split_conflicts"]


def test_datasets_keep_one_copy_per_hash_across_splits(tmp_path):
    source = _source(tmp_path, "source", split="train")
    converted = tmp_path / "converted"
    convert_coco(source, converted, source_name="source")
    merged = tmp_path / "merged"
    merge_datasets([converted], merged, dataset_name="merged")
    payload = json.loads((merged / "annotations.json").read_text())
    files = [item["file_name"] for item in payload["images"]]
    assert len(files) == len(set(files)) == 1


def test_inspection_distinguishes_ignored_and_mapped_annotations(tmp_path):
    from person_detect.training.veomatch_person_v1.tools.dataset_tools import inspect_dataset
    source = _source(tmp_path, "mixed", category="Football")
    report = inspect_dataset(source)
    assert report["original_annotation_count"] == 1
    assert report["mapped_person_annotation_count"] == 0
    assert report["ignored_annotation_count"] == 1
    assert report["ignored_classes"] == ["Football"]
    assert report["unknown_classes"] == []
    assert report["split_sizes"] == {"train": 1, "valid": 0, "test": 0}


def test_all_roboflow_splits_are_discovered(tmp_path):
    for split in ("train", "valid", "test"):
        _source(tmp_path, "multi", split=split)
    from person_detect.training.veomatch_person_v1.tools.dataset_tools import inspect_dataset
    report = inspect_dataset(tmp_path / "multi")
    assert report["image_count"] == 3
    assert report["split_sizes"] == {"train": 1, "valid": 1, "test": 1}
    assert report["mapped_person_annotation_count"] == 3
    converted = tmp_path / "converted_multi"
    convert_coco(tmp_path / "multi", converted, source_name="multi")
    payload = json.loads((converted / "annotations.json").read_text())
    assert {image["split"] for image in payload["images"]} == {"train", "valid", "test"}
    manifest = json.loads((converted / "manifest.json").read_text())
    assert len(manifest["images"]) == 3


def test_sequence_groups_are_single_split(tmp_path):
    first = _source(tmp_path, "sequence", split="train")
    second = _source(tmp_path, "sequence", split="test")
    # Give both source files the same normalized temporal-sequence stem.
    for split in ("train", "test"):
        image = tmp_path / "sequence" / "images" / split / "frame.ppm"
        renamed = image.with_name(f"Manchester_City_vs_Genk_frame_{'17947' if split == 'train' else '17948'}.ppm")
        image.rename(renamed)
        annotation = tmp_path / "sequence" / split / "_annotations.coco.json"
        payload = json.loads(annotation.read_text())
        payload["images"][0]["file_name"] = f"../images/{split}/{renamed.name}"
        annotation.write_text(json.dumps(payload))
    from person_detect.training.veomatch_person_v1.tools.dataset_tools import convert_coco
    train = tmp_path / "converted_sequence"
    convert_coco(tmp_path / "sequence", train, source_name="sequence")
    merged = tmp_path / "merged_sequence"
    merge_datasets([train], merged, dataset_name="merged_sequence")
    audit = json.loads((merged / "manifest.json").read_text())["sequence_audit"]
    assert audit["sequence_group_leakage_count"] == 0


def test_rfdetr_split_structure_and_single_category(tmp_path):
    source = _source(tmp_path, "rfdetr_source")
    converted = tmp_path / "converted"
    convert_coco(source, converted, source_name="source")
    report = _validate_dataset(converted)
    assert report["train"]["images"] == 1
    payload = json.loads((converted / "train" / "_annotations.coco.json").read_text())
    assert payload["categories"] == [{"id": 1, "name": "person", "supercategory": "person"}]
    assert (converted / "train" / payload["images"][0]["file_name"]).is_file()


def test_training_aliases_and_controlled_config():
    assert canonical_experiment("c1") == "c1_football_images"
    config = load_config("c1")
    assert config["training"] == {
        "epochs": 50, "batch_size": 1, "grad_accum_steps": 16,
        "learning_rate": 5e-5, "learning_rate_encoder": 7.5e-5,
        "resolution": 880, "gradient_checkpointing": True, "use_ema": True,
        "early_stopping": True, "early_stopping_patience": 10,
        "skip_best_epochs": 3,
    }


def test_all_experiments_resolve_one_class_and_preserve_baseline_constructor():
    for experiment in ("c1", "c2", "c3"):
        config = load_config(experiment)
        assert resolve_num_classes(config) == 1
        kwargs = model_constructor_kwargs(config, config["training"])
        assert kwargs["num_classes"] == 1
        assert kwargs["pretrain_weights"].endswith("rf-detr-xxlarge.pth")
        assert kwargs["gradient_checkpointing"] is True


def test_model_constructor_receives_one_class(monkeypatch):
    captured = {}

    class FakeRFDETR2XLarge:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    config = load_config("c1")
    import person_detect.training.veomatch_person_v1.train as train_module
    monkeypatch.setattr(train_module, "RFDETR2XLarge", FakeRFDETR2XLarge, raising=False)
    # This mirrors the real constructor call without importing RF-DETR.
    train_module.RFDETR2XLarge(**model_constructor_kwargs(config, config["training"]))
    assert captured["num_classes"] == 1


def test_checkpoint_selection_is_explicit(tmp_path):
    (tmp_path / "checkpoint_best_regular.pth").write_bytes(b"regular")
    assert _find_best(tmp_path).name == "checkpoint_best_regular.pth"
    (tmp_path / "unrelated.pth").write_bytes(b"unrelated")
    assert _find_best(tmp_path).name == "checkpoint_best_regular.pth"


def test_existing_run_requires_explicit_protection(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint_best_total.pth").write_bytes(b"checkpoint")
    import pytest
    with pytest.raises(FileExistsError):
        _ensure_run_dir(run_dir, overwrite=False)
    _ensure_run_dir(run_dir, overwrite=True)
