from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tracking_v2.experiment import _write_gallery, load_crop_records
from tracking_v2.metrics import evaluate_rows
from tracking_v2.team_discovery import (
    TEAM_A,
    TEAM_B,
    UNKNOWN,
    assign_relative_teams,
    extract_kit_feature,
    select_sample,
)


def _person_crop(path: Path, shirt: tuple[int, int, int], brightness: float = 1.0) -> None:
    image = Image.new("RGB", (30, 84), (44, 130, 45))
    draw = ImageDraw.Draw(image)
    color = tuple(round(channel * brightness) for channel in shirt)
    draw.rectangle((7, 15, 23, 53), fill=color)
    draw.rectangle((8, 54, 13, 80), fill=(35, 40, 48))
    draw.rectangle((17, 54, 22, 80), fill=(35, 40, 48))
    image.save(path)


def test_torso_feature_ignores_lower_background_and_is_stable_to_brightness(tmp_path: Path) -> None:
    bright = tmp_path / "bright.jpg"
    dim = tmp_path / "dim.jpg"
    blue = tmp_path / "blue.jpg"
    _person_crop(bright, (205, 42, 38), 1.0)
    _person_crop(dim, (205, 42, 38), 0.62)
    _person_crop(blue, (36, 74, 190), 1.0)

    bright_feature = extract_kit_feature(bright)
    dim_feature = extract_kit_feature(dim)
    blue_feature = extract_kit_feature(blue)

    assert float(np.dot(bright_feature, dim_feature)) > 0.9
    assert float(np.dot(bright_feature, blue_feature)) < 0.75


def test_relative_groups_are_anonymous_and_ambiguous_samples_can_be_unknown() -> None:
    group_a = [np.array([1.0, 0.0], dtype=np.float32) for _ in range(12)]
    group_b = [np.array([0.0, 1.0], dtype=np.float32) for _ in range(12)]
    ambiguous = [np.array([0.5, 0.5], dtype=np.float32)]
    predictions = assign_relative_teams(
        group_a + group_b + ambiguous,
        sample_ids=[f"S{i:02d}" for i in range(25)],
        seed=3,
    )

    a_teams = {row["team"] for row in predictions[:12]}
    b_teams = {row["team"] for row in predictions[12:24]}
    assert a_teams in ({TEAM_A}, {TEAM_B})
    assert b_teams in ({TEAM_A}, {TEAM_B})
    assert a_teams != b_teams
    assert predictions[-1]["team"] == UNKNOWN


def test_sample_keeps_outside_observations_and_does_not_depend_on_tracklet_ids() -> None:
    records = []
    for frame in range(80):
        records.append(
            {
                "frame_index": frame,
                "timestamp_s": frame * 0.1,
                "representative_view": f"view_{frame % 3}",
                "crop_width_px": 18 + frame % 12,
                "crop_height_px": 50 + frame % 30,
                "field_context": "outside" if frame in {20, 21, 22, 23, 24, 25} else "inside",
                "tracklet_id": f"legacy-{frame % 4}",
            }
        )
    selected = select_sample(records, sample_count=20, outside_count=5, seed=17)
    changed_legacy_ids = [{**row, "tracklet_id": f"changed-{i}"} for i, row in enumerate(records)]
    selected_after_change = select_sample(changed_legacy_ids, sample_count=20, outside_count=5, seed=17)

    signature = lambda rows: [
        (row["frame_index"], row["timestamp_s"], row["representative_view"])
        for row in rows
    ]
    assert len(selected) == 20
    assert sum(row["field_context"] == "outside" for row in selected) >= 5
    assert signature(selected) == signature(selected_after_change)


def test_metrics_align_anonymous_cluster_names_to_manual_reference() -> None:
    labels = [
        {"sample_id": "1", "manual_team": TEAM_A, "field_context": "inside"},
        {"sample_id": "2", "manual_team": TEAM_B, "field_context": "outside"},
        {"sample_id": "3", "manual_team": UNKNOWN, "field_context": "outside"},
    ]
    assignments = [
        {"sample_id": "1", "predicted_team": TEAM_B},
        {"sample_id": "2", "predicted_team": TEAM_A},
        {"sample_id": "3", "predicted_team": UNKNOWN},
    ]

    report = evaluate_rows(labels, assignments)
    assert report["precision_when_assigned"] == 1.0
    assert report["coverage"] == 2 / 3
    assert report["accuracy_including_unknown"] == 1.0
    assert report["known_team_confusions"] == {"TEAM_A_to_TEAM_B": 0, "TEAM_B_to_TEAM_A": 0}
    assert report["by_condition"]["field_context"]["outside"]["labeled_samples"] == 2



def test_crop_loader_resolves_index_relative_paths_and_keeps_outside_context(tmp_path: Path) -> None:
    tracking_root = tmp_path / "tracking" / "v1"
    crop = tracking_root / "diagnostics" / "appearance_crops" / "trk" / "crop.jpg"
    crop.parent.mkdir(parents=True)
    Image.new("RGB", (20, 50), (180, 30, 20)).save(crop)
    index_path = tracking_root / "diagnostics" / "appearance_crops" / "index.json"
    index_path.write_text(json.dumps({"records": [{
        "status": "saved", "crop_ref": "diagnostics/appearance_crops/trk/crop.jpg",
        "detection_id": "det-1", "timestamp_s": 2.0, "frame_index": 60,
        "representative_view": "view-a", "crop_width_px": 20, "crop_height_px": 50,
        "tracklet_id": "must-not-propagate"
    }]}), encoding="utf-8")
    detections_path = tmp_path / "detections.jsonl"
    detections_path.write_text(json.dumps({"detections": [{"detection_id": "det-1", "x": 0, "z": 33}]}) + "\n", encoding="utf-8")

    records, statuses = load_crop_records(
        index_path, detections_path=detections_path, field_length=105, field_width=64
    )

    assert statuses["saved"] == 1
    assert len(records) == 1
    assert records[0]["image_path"] == crop
    assert records[0]["field_context"] == "outside"
    assert "tracklet_id" not in records[0]


def test_gallery_csv_export_preserves_context_for_stratified_metrics(tmp_path: Path) -> None:
    gallery = tmp_path / "index.html"
    _write_gallery(gallery, [{
        "sample_id": "S001", "predicted_team": TEAM_A, "crop_file": "crops/S001.jpg",
        "timestamp_s": "1.25", "representative_view": "view-west",
        "crop_width_px": 20, "crop_height_px": 60, "scale_bin": "medium",
        "field_context": "outside",
    }])

    html = gallery.read_text(encoding="utf-8")
    assert 'data-field-context="outside"' in html
    assert 'data-view="view-west"' in html
    assert 'data-scale-bin="medium"' in html
    assert "'field_context','representative_view','scale_bin'" in html
    assert "c.dataset.fieldContext,c.dataset.view,c.dataset.scaleBin" in html
