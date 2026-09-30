from __future__ import annotations

import numpy as np
import pytest

from tracking.identities.visual_audit import (
    VisualAuditConfig,
    build_timeline_index,
    build_transition_records,
    deterministic_identity_color,
    field_to_pixel,
    load_identities,
    load_tracklets,
    render_topdown_frame,
)


def _tracklet(tracklet_id, observations):
    return {
        "tracklet_id": tracklet_id,
        "observations": [
            {
                "frame_index": frame,
                "timestamp_s": time_s,
                "x": x,
                "z": z,
                "confidence": 0.9,
            }
            for frame, time_s, x, z in observations
        ],
    }


def test_field_center_maps_to_canvas_center():
    config = VisualAuditConfig()
    px, py = field_to_pixel(0.0, 0.0, config)
    assert abs(px - config.canvas_width_px // 2) <= 1
    assert abs(py - config.canvas_height_px // 2) <= 1


def test_identity_color_is_deterministic_and_distinct():
    assert deterministic_identity_color("identity-001") == (
        deterministic_identity_color("identity-001")
    )
    assert deterministic_identity_color("identity-001") != (
        deterministic_identity_color("identity-002")
    )


def test_transition_gap_and_backbone_membership():
    tracklets = load_tracklets(
        [
            _tracklet(
                "trk-a",
                [(0, 0.0, 0.0, 0.0), (1, 0.1, 1.0, 0.0)],
            ),
            _tracklet(
                "trk-b",
                [(3, 0.3, 2.0, 0.0), (4, 0.4, 3.0, 0.0)],
            ),
        ]
    )
    identities = load_identities(
        [
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a", "trk-b"],
            }
        ]
    )
    audits = {
        "identity-001": {
            "flags": [],
            "backbone": {
                "edges": [
                    {
                        "tracklet_a": "trk-a",
                        "tracklet_b": "trk-b",
                    }
                ]
            },
        }
    }
    scored = {
        ("trk-a", "trk-b"): {
            "rank_score": 0.9,
            "evidence_confidence": 0.8,
            "link_kind": "continuation",
        }
    }

    rows = build_transition_records(
        identities,
        tracklets,
        audits,
        scored,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["relation"] == "gap"
    assert row["gap_s"] == pytest.approx(0.2)
    assert row["endpoint_distance_m"] == pytest.approx(1.0)
    assert row["implicit_speed_mps"] == pytest.approx(5.0)
    assert row["is_backbone_edge"] is True


def test_overlap_transition_reports_shared_frame():
    tracklets = load_tracklets(
        [
            _tracklet(
                "trk-a",
                [(0, 0.0, 0.0, 0.0), (2, 0.2, 1.0, 0.0)],
            ),
            _tracklet(
                "trk-b",
                [(2, 0.2, 1.1, 0.0), (3, 0.3, 2.0, 0.0)],
            ),
        ]
    )
    identities = load_identities(
        [
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a", "trk-b"],
            }
        ]
    )
    rows = build_transition_records(
        identities,
        tracklets,
        {"identity-001": {"flags": [], "backbone": {"edges": []}}},
        {
            ("trk-a", "trk-b"): {
                "rank_score": 0.7,
                "evidence_confidence": 0.6,
                "link_kind": "overlap_duplicate",
            }
        },
    )
    assert rows[0]["shared_frame_count"] == 1
    assert "shared_observation_frame" in rows[0]["review_flags"]


def test_render_frame_has_expected_shape():
    config = VisualAuditConfig(
        canvas_width_px=640,
        canvas_height_px=420,
        margin_px=30,
    )
    tracklets = load_tracklets(
        [_tracklet("trk-a", [(0, 0.0, 0.0, 0.0)])]
    )
    identities = load_identities(
        [
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a"],
            }
        ]
    )
    timeline = build_timeline_index(
        identities,
        tracklets,
    )
    frame = render_topdown_frame(
        time_s=0.0,
        identities=identities,
        timeline_index=timeline,
        audits={},
        config=config,
    )
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (420, 640, 3)


def test_nested_overlap_does_not_create_fake_large_gap():
    # trk-b is a short duplicate nested inside trk-a. The true handoff to
    # trk-c is from the end of trk-a, not from the end of trk-b.
    tracklets = load_tracklets(
        [
            _tracklet(
                "trk-a",
                [
                    (0, 0.0, 0.0, 0.0),
                    (25, 2.5, 2.5, 0.0),
                ],
            ),
            _tracklet(
                "trk-b",
                [(3, 0.3, 0.3, 0.0)],
            ),
            _tracklet(
                "trk-c",
                [(37, 3.7, 3.7, 0.0)],
            ),
        ]
    )
    identities = load_identities(
        [
            {
                "identity_id": "identity-001",
                "tracklet_ids": ["trk-a", "trk-b", "trk-c"],
            }
        ]
    )
    audits = {
        "identity-001": {
            "flags": [],
            "backbone": {"edges": []},
        }
    }
    scored = {
        ("trk-a", "trk-b"): {
            "rank_score": 0.2,
            "evidence_confidence": 0.5,
            "link_kind": "overlap_duplicate",
        },
        ("trk-a", "trk-c"): {
            "rank_score": 0.8,
            "evidence_confidence": 0.7,
            "link_kind": "continuation",
        },
        ("trk-b", "trk-c"): {
            "rank_score": 0.5,
            "evidence_confidence": 0.5,
            "link_kind": "continuation",
        },
    }

    rows = build_transition_records(
        identities,
        tracklets,
        audits,
        scored,
    )
    gaps = [row for row in rows if row["relation"] == "gap"]
    overlaps = [row for row in rows if row["relation"] == "overlap"]

    assert len(gaps) == 1
    assert gaps[0]["tracklet_a"] == "trk-a"
    assert gaps[0]["tracklet_b"] == "trk-c"
    assert gaps[0]["gap_s"] == pytest.approx(1.2)
    assert len(overlaps) == 1
    assert {
        overlaps[0]["tracklet_a"],
        overlaps[0]["tracklet_b"],
    } == {"trk-a", "trk-b"}
