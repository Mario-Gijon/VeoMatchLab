from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tracking.appearance.selection import (
    AppearanceSelectionConfig,
    select_appearance_moments,
)
from tracking.detections import CanonicalDetection, CanonicalDetectionFrame
from tracking.tracklets import Tracklet, TrackletObservation, TrackletSet


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "tracking"
    / "run_v1_appearance_selection.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_v1_appearance_selection",
    RUNNER_PATH,
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def detection(
    detection_id,
    frame_index,
    timestamp_s,
    *,
    x=0.0,
    z=0.0,
):
    return CanonicalDetection(
        detection_id=detection_id,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        confidence=0.95,
        x=x,
        z=z,
        representative_view="tight_far_c2",
        virtual_bbox_xyxy=(500.0, 200.0, 550.0, 320.0),
        contributing_views=("tight_far_c2", "tight_near_c1"),
        metadata={
            "owner_supported": True,
            "multi_view": True,
            "position_spread_m": 0.1,
            "source_pixels": [
                {
                    "lenses": [
                        {
                            "lens": "left",
                            "y_min": 100.0,
                            "y_max": 220.0,
                        }
                    ]
                }
            ],
        },
    )


def tracklet(tracklet_id, detections):
    return Tracklet(
        tracklet_id=tracklet_id,
        observations=tuple(
            TrackletObservation(
                detection_id=item.detection_id,
                frame_index=item.frame_index,
                timestamp_s=item.timestamp_s,
                x=item.x,
                z=item.z,
                confidence=item.confidence,
            )
            for item in detections
        ),
    )


def build_fixture():
    frames = []
    detections = []
    for index in range(5):
        item = detection(
            f"det-{index}",
            index * 3,
            float(index),
            x=float(index),
        )
        detections.append(item)
        frames.append(
            CanonicalDetectionFrame(
                frame_index=item.frame_index,
                timestamp_s=item.timestamp_s,
                detections=(item,),
            )
        )

    first = tracklet("trk-000001", detections[:4])
    second = tracklet("trk-000002", detections[4:])
    tracklets = TrackletSet(tracklets=(first, second))

    selection = select_appearance_moments(
        frames,
        tracklets,
        config=AppearanceSelectionConfig(
            max_samples_per_tracklet=3,
            min_samples_if_available=1,
            min_temporal_spacing_s=0.5,
        ),
    )
    return frames, tracklets, selection


def test_artifact_root_matches_tracking_v1_layout(tmp_path):
    result = runner.artifact_root(
        data_root=tmp_path,
        match_id="match-1",
        clip_id="clip-2",
    )
    assert result == (
        tmp_path
        / "matches"
        / "match-1"
        / "clips"
        / "clip-2"
        / "tracking"
        / "v1"
    )


@pytest.mark.parametrize(
    "match_id,clip_id",
    [
        ("", "clip"),
        ("match", ""),
        (" ", "clip"),
        ("match", " "),
    ],
)
def test_artifact_root_rejects_empty_ids(tmp_path, match_id, clip_id):
    with pytest.raises(ValueError):
        runner.artifact_root(
            data_root=tmp_path,
            match_id=match_id,
            clip_id=clip_id,
        )


def test_jsonl_loader_skips_blank_lines(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"a": 1}\n\n{"b": 2}\n',
        encoding="utf-8",
    )

    assert list(runner._iter_jsonl(path)) == [
        {"a": 1},
        {"b": 2},
    ]


def test_jsonl_loader_reports_line_number(tmp_path):
    path = tmp_path / "items.jsonl"
    path.write_text(
        '{"a": 1}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 2"):
        list(runner._iter_jsonl(path))


def test_quantile_handles_empty_and_singleton():
    assert runner._quantile([], 0.5) is None
    assert runner._quantile([7.0], 0.5) == 7.0


def test_quantile_interpolates():
    assert runner._quantile([0.0, 10.0], 0.25) == pytest.approx(2.5)


def test_build_diagnostics_counts_selection_distribution():
    _, tracklets, selection = build_fixture()

    result = runner.build_diagnostics(
        selection,
        tracklets,
        detail_limit=10,
    )

    assert result["summary"]["tracklet_count"] == 2
    assert result["summary"]["selected_count"] == 4
    assert result["selection_distribution"] == {
        "1": 1,
        "3": 1,
    }


def test_build_diagnostics_reports_quality_quantiles():
    _, tracklets, selection = build_fixture()
    result = runner.build_diagnostics(selection, tracklets)

    quality = result["selected_quality"]
    assert quality["min"] is not None
    assert quality["median"] is not None
    assert quality["p90"] is not None
    assert quality["max"] >= quality["min"]


def test_build_diagnostics_orders_weakest_and_strongest():
    _, tracklets, selection = build_fixture()
    result = runner.build_diagnostics(selection, tracklets)

    assert result["weakest_tracklets"][0]["tracklet_id"] == "trk-000002"
    assert result["strongest_tracklets"][0]["tracklet_id"] == "trk-000001"


def test_build_diagnostics_contains_selected_frames_and_times():
    _, tracklets, selection = build_fixture()
    result = runner.build_diagnostics(selection, tracklets)

    first = next(
        item
        for item in result["tracklets"]
        if item["tracklet_id"] == "trk-000001"
    )
    assert len(first["selected_frames"]) == first["selected_count"]
    assert len(first["selected_times_s"]) == first["selected_count"]


def test_build_diagnostics_rejects_mismatched_tracklet_ids():
    _, tracklets, selection = build_fixture()

    wrong = TrackletSet(
        tracklets=(tracklets.tracklets[0],)
    )
    with pytest.raises(ValueError, match="do not match"):
        runner.build_diagnostics(selection, wrong)


def test_build_diagnostics_rejects_zero_detail_limit():
    _, tracklets, selection = build_fixture()

    with pytest.raises(ValueError, match="detail_limit"):
        runner.build_diagnostics(
            selection,
            tracklets,
            detail_limit=0,
        )


def test_load_detection_frames_and_tracklets_roundtrip(tmp_path):
    frames, tracklets, _ = build_fixture()

    detections_path = tmp_path / "frames.jsonl"
    detections_path.write_text(
        "\n".join(json.dumps(frame.to_dict()) for frame in frames) + "\n",
        encoding="utf-8",
    )

    tracklets_path = tmp_path / "tracklets.jsonl"
    tracklets_path.write_text(
        "\n".join(
            item.to_json()
            for item in tracklets.tracklets
        )
        + "\n",
        encoding="utf-8",
    )

    loaded_frames = runner.load_detection_frames(detections_path)
    loaded_tracklets = runner.load_tracklets(tracklets_path)

    assert len(loaded_frames) == len(frames)
    assert loaded_tracklets.to_dict() == tracklets.to_dict()


def test_loaders_reject_empty_artifacts(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        runner.load_detection_frames(empty)

    with pytest.raises(ValueError, match="empty"):
        runner.load_tracklets(empty)


def test_load_dev_config_requires_match_and_clip(tmp_path):
    path = tmp_path / "dev.json"
    path.write_text(
        json.dumps({"match_id": "m"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="clip_id"):
        runner.load_dev_config(path)


def test_parse_args_defaults_are_expected():
    args = runner.parse_args([])

    assert args.max_samples == 8
    assert args.min_samples == 3
    assert args.min_quality == pytest.approx(0.42)
    assert args.min_spacing_s == pytest.approx(0.75)
    assert args.diversity_weight == pytest.approx(0.25)
    assert args.virtual_width == 1280
    assert args.virtual_height == 720


def test_diagnostics_payload_is_json_safe():
    _, tracklets, selection = build_fixture()
    payload = runner.build_diagnostics(selection, tracklets)

    json.dumps(payload, allow_nan=False)
