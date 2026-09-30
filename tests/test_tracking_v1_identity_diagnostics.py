from __future__ import annotations

from tracking.identities.diagnostics import (
    audit_backbone_partition,
    audit_partition,
    load_candidate_edges_for_diagnostics,
    load_hard_cannot_links,
    load_selected_backbone_links,
    load_tracklet_spans,
    load_utilities,
)


def _tracklet(tracklet_id, start_frame, end_frame):
    observations = []
    for frame in range(start_frame, end_frame + 1):
        observations.append(
            {
                "frame_index": frame,
                "timestamp_s": frame / 10.0,
            }
        )
    return {
        "tracklet_id": tracklet_id,
        "observations": observations,
    }


def _score(a, b, score, confidence=1.0):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "rank_score": score,
        "evidence_confidence": confidence,
        "link_kind": "continuation",
    }


def _edge(a, b, status):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": status,
    }


def test_exact_split_detects_group_split_missed_by_single_node_view():
    spans = load_tracklet_spans(
        [
            _tracklet("a", 0, 1),
            _tracklet("b", 2, 3),
            _tracklet("c", 4, 5),
            _tracklet("d", 6, 7),
        ]
    )
    utilities = load_utilities(
        [
            _score("a", "b", 0.95),
            _score("c", "d", 0.95),
            _score("a", "c", 0.40),
            _score("a", "d", 0.40),
            _score("b", "c", 0.40),
            _score("b", "d", 0.40),
        ]
    )
    identities = [
        {
            "identity_id": "identity-001",
            "tracklet_ids": ["a", "b", "c", "d"],
        }
    ]

    audits, _, summary = audit_partition(
        identities,
        spans,
        utilities,
        frozenset(),
    )
    split = audits[0]["best_exact_two_way_split"]

    assert split["evaluated"] is True
    assert split["profitable"] is True
    assert set(split["left"]) in (
        {"a", "b"},
        {"c", "d"},
    )
    assert summary[
        "profitable_exact_split_identity_count"
    ] == 1


def test_backbone_prefers_strong_chain_over_all_pairs():
    spans = load_tracklet_spans(
        [
            _tracklet("a", 0, 1),
            _tracklet("b", 2, 3),
            _tracklet("c", 4, 5),
        ]
    )
    utilities = load_utilities(
        [
            _score("a", "b", 0.90),
            _score("b", "c", 0.85),
            _score("a", "c", 0.55),
        ]
    )
    identities = [
        {
            "identity_id": "identity-001",
            "tracklet_ids": ["a", "b", "c"],
        }
    ]

    audits, _, _ = audit_partition(
        identities,
        spans,
        utilities,
        frozenset(),
    )
    backbone = audits[0]["backbone"]

    assert backbone["edge_count"] == 2
    pairs = {
        frozenset(
            (row["tracklet_a"], row["tracklet_b"])
        )
        for row in backbone["edges"]
    }
    assert pairs == {
        frozenset(("a", "b")),
        frozenset(("b", "c")),
    }


def test_temporal_audit_reports_gaps_and_exact_overlap():
    spans = load_tracklet_spans(
        [
            _tracklet("a", 0, 2),
            _tracklet("b", 2, 4),
            _tracklet("c", 10, 11),
        ]
    )
    utilities = load_utilities(
        [
            _score("a", "b", 0.8),
            _score("a", "c", 0.8),
            _score("b", "c", 0.8),
        ]
    )
    identities = [
        {
            "identity_id": "identity-001",
            "tracklet_ids": ["a", "b", "c"],
        }
    ]

    audits, _, _ = audit_partition(
        identities,
        spans,
        utilities,
        frozenset(),
    )
    timeline = audits[0]["timeline"]

    assert timeline["gap_count"] == 1
    assert timeline["max_exact_observation_concurrency"] == 2
    assert timeline[
        "shared_observation_frame_pair_count"
    ] == 1


def test_hard_incompatible_identity_is_not_offered_as_move_target():
    spans = load_tracklet_spans(
        [
            _tracklet("a", 0, 1),
            _tracklet("b", 2, 3),
            _tracklet("c", 4, 5),
        ]
    )
    utilities = load_utilities(
        [
            _score("a", "b", 0.9),
            _score("a", "c", 0.9),
        ]
    )
    hard = load_hard_cannot_links(
        [_edge("b", "c", "cannot_link")]
    )
    identities = [
        {
            "identity_id": "identity-001",
            "tracklet_ids": ["a", "b"],
        },
        {
            "identity_id": "identity-002",
            "tracklet_ids": ["c"],
        },
    ]

    _, tracklet_audits, _ = audit_partition(
        identities,
        spans,
        utilities,
        hard,
    )
    row_b = next(
        row
        for row in tracklet_audits
        if row["tracklet_id"] == "b"
    )
    assert row_b[
        "best_alternative_identity_id"
    ] is None



def test_backbone_audit_treats_expected_clique_split_as_non_error():
    spans = load_tracklet_spans(
        [
            _tracklet("a", 0, 10),
            _tracklet("b", 11, 20),
            _tracklet("c", 21, 30),
        ]
    )
    utilities = load_utilities(
        [
            _score("a", "b", 0.85),
            _score("a", "c", 0.00),
            _score("b", "c", 0.95),
        ]
    )
    candidates_raw = [
        {
            "tracklet_a": "a",
            "tracklet_b": "b",
            "status": "candidate",
            "reason": "transition_reachable",
            "physics": {
                "predecessor": "a",
                "successor": "b",
                "gap_s": 0.1,
                "endpoint_distance_m": 1.0,
                "required_speed_after_slack_mps": 0.0,
                "prediction_error_mean_m": 0.5,
                "velocity_direction_cosine": 0.9,
            },
        },
        {
            "tracklet_a": "a",
            "tracklet_b": "c",
            "status": "candidate",
            "reason": "transition_reachable",
            "physics": {
                "predecessor": "a",
                "successor": "c",
                "gap_s": 1.1,
                "endpoint_distance_m": 2.0,
            },
        },
        {
            "tracklet_a": "b",
            "tracklet_b": "c",
            "status": "candidate",
            "reason": "transition_reachable",
            "physics": {
                "predecessor": "b",
                "successor": "c",
                "gap_s": 0.1,
                "endpoint_distance_m": 1.0,
            },
        },
    ]
    candidate_edges = load_candidate_edges_for_diagnostics(
        candidates_raw
    )
    hard = load_hard_cannot_links(
        candidates_raw
    )
    selected = load_selected_backbone_links(
        [
            {
                "predecessor_tracklet_id": "a",
                "successor_tracklet_id": "b",
                "utility": 0.7,
                "rank_score": 0.85,
                "evidence_confidence": 1.0,
                "gap_s": 0.1,
                "endpoint_distance_m": 1.0,
                "source_tail_offset_s": 0.0,
                "target_head_offset_s": 0.0,
                "whole_clique_merge_delta": -0.3,
                "strong_utility_threshold": 0.5,
                "reciprocal_best": True,
            }
        ]
    )
    baseline = [
        {
            "identity_id": "identity-a",
            "tracklet_ids": ["a"],
        },
        {
            "identity_id": "identity-b",
            "tracklet_ids": ["b", "c"],
        },
    ]
    final = [
        {
            "identity_id": "identity-final",
            "tracklet_ids": ["a", "b", "c"],
        }
    ]

    (
        final_audits,
        bridge_audits,
        _,
        summary,
        _,
    ) = audit_backbone_partition(
        final_identities=final,
        baseline_identities=baseline,
        selected_backbone_links=selected,
        spans=spans,
        utilities=utilities,
        hard=hard,
        candidate_edges=candidate_edges,
        solver_summary={
            "strong_utility_threshold": 0.5,
        },
    )

    assert summary[
        "bridge_validation_failure_count"
    ] == 0
    assert summary[
        "unexpected_legacy_clique_split_pressure_count"
    ] == 0
    assert summary[
        "expected_legacy_clique_split_pressure_count"
    ] == 1
    assert bridge_audits[0]["bridge_valid"] is True
    assert final_audits[0][
        "legacy_clique_split_pressure"
    ]["treated_as_identity_error"] is False


def test_backbone_audit_rejects_hard_constraint_violation():
    spans = load_tracklet_spans(
        [
            _tracklet("a", 0, 10),
            _tracklet("b", 11, 20),
        ]
    )
    utilities = load_utilities(
        [_score("a", "b", 0.9)]
    )
    candidates_raw = [
        {
            "tracklet_a": "a",
            "tracklet_b": "b",
            "status": "cannot_link",
            "reason": "transition_unreachable",
            "physics": {},
        }
    ]
    hard = load_hard_cannot_links(
        candidates_raw
    )

    _, _, _, summary, _ = audit_backbone_partition(
        final_identities=[
            {
                "identity_id": "final",
                "tracklet_ids": ["a", "b"],
            }
        ],
        baseline_identities=[
            {
                "identity_id": "base-a",
                "tracklet_ids": ["a"],
            },
            {
                "identity_id": "base-b",
                "tracklet_ids": ["b"],
            },
        ],
        selected_backbone_links=[],
        spans=spans,
        utilities=utilities,
        hard=hard,
        candidate_edges=(
            load_candidate_edges_for_diagnostics(
                candidates_raw
            )
        ),
        solver_summary={
            "strong_utility_threshold": 0.5,
        },
    )
    assert summary[
        "final_hard_constraint_violation_count"
    ] == 1
