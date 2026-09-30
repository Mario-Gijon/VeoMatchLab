from __future__ import annotations

import pytest

from tracking.persistent.solver import (
    PersistentActorAssignmentConfig,
    solve_anchored_actor_assignment,
)


def tracklet(tracklet_id: str, observations: int = 1, start: float = 0.0):
    return {
        "tracklet_id": tracklet_id,
        "observation_count": observations,
        "start_time_s": start,
        "end_time_s": start + max(0, observations - 1) * 0.1,
        "mean_confidence": 0.9,
    }


def hard(a: str, b: str):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "status": "cannot_link",
    }


def identity_assignment(tracklet_id: str, identity_id: str):
    return {
        "tracklet_id": tracklet_id,
        "identity_id": identity_id,
    }


def identity_link(a: str, b: str):
    return {
        "tracklet_a": a,
        "tracklet_b": b,
        "selected_backbone_bridge": False,
    }


def soft(a: str, b: str, score: float, confidence: float = 1.0):
    return {
        "scoring_version": "soft-link-v1",
        "tracklet_a": a,
        "tracklet_b": b,
        "link_kind": "continuation",
        "rank_score": score,
        "evidence_confidence": confidence,
    }


def trajectory(identity_id: str, frame: int, timestamp: float, tracklet_id: str, std: float = 0.1):
    return {
        "identity_id": identity_id,
        "points": [
            {
                "identity_id": identity_id,
                "frame_index": frame,
                "timestamp_s": timestamp,
                "origin": "observed",
                "position_std_m": std,
                "source_tracklet_ids": [tracklet_id],
            }
        ],
    }


def population(count: int, confidence: str = "high"):
    return {
        "estimated_population": count,
        "confidence": confidence,
    }


def solve(
    *,
    tracklets,
    hard_pairs=(),
    trajectories,
    assignments,
    identity_links=(),
    soft_links=(),
    population_count=2,
):
    return solve_anchored_actor_assignment(
        tracklet_rows=tracklets,
        physical_candidate_rows=list(hard_pairs),
        trajectory_rows=trajectories,
        identity_assignment_rows=assignments,
        identity_link_rows=list(identity_links),
        scored_link_rows=list(soft_links),
        population=population(population_count),
        config=PersistentActorAssignmentConfig(
            max_anchor_configurations=12,
            time_limit_s=10.0,
        ),
    )


def actor_of(result, tracklet_id: str):
    row = next(row for row in result["assignments"] if row["tracklet_id"] == tracklet_id)
    return row["actor_id"]


def test_noise_state_makes_hard_incompatible_fragment_rejectable():
    tracklets = [tracklet("A", 5), tracklet("B", 5), tracklet("N", 1)]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B"), hard("A", "N"), hard("B", "N")],
        trajectories=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("N", "IN"),
        ],
    )
    assert result["summary"]["actor_count"] == 2
    assert result["summary"]["noise_tracklet_count"] == 1
    assert result["summary"]["dropped_observation_count"] == 1
    assert actor_of(result, "N") is None
    assert result["summary"]["hard_constraint_violation_count"] == 0


def test_minimises_dropped_observations_before_other_evidence():
    tracklets = [
        tracklet("A", 5),
        tracklet("B", 5),
        tracklet("LONG", 10),
        tracklet("SHORT", 1),
    ]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[
            hard("A", "B"),
            hard("A", "LONG"),
            hard("A", "SHORT"),
            hard("LONG", "SHORT"),
        ],
        trajectories=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("LONG", "IL"),
            identity_assignment("SHORT", "IS"),
        ],
    )
    noise = {row["tracklet_id"] for row in result["noise_tracklets"]}
    assert noise == {"SHORT"}
    assert result["summary"]["dropped_observation_count"] == 1


def test_v1_identity_links_are_stronger_soft_evidence_than_soft_link_score():
    tracklets = [tracklet("A", 5), tracklet("B", 5), tracklet("C", 2)]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B")],
        trajectories=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("C", "IB"),
        ],
        identity_links=[identity_link("B", "C")],
        soft_links=[soft("A", "C", 1.0), soft("B", "C", 0.05)],
    )
    assert actor_of(result, "C") == actor_of(result, "B")
    assert result["summary"]["preserved_v1_identity_link_count"] == 1


def test_soft_evidence_breaks_tie_after_v1_preservation():
    tracklets = [tracklet("A", 5), tracklet("B", 5), tracklet("C", 2)]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B")],
        trajectories=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("C", "IC"),
        ],
        soft_links=[soft("A", "C", 0.9), soft("B", "C", 0.2)],
    )
    assert actor_of(result, "C") == actor_of(result, "A")
    assert result["summary"]["soft_evidence_sum"] == pytest.approx(0.9)


def test_actor_count_comes_from_population_not_hardcoded_23():
    tracklets = [tracklet("A"), tracklet("B"), tracklet("C")]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B")],
        trajectories=[trajectory("IA", 3, 0.3, "A"), trajectory("IB", 3, 0.3, "B")],
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("C", "IC"),
        ],
        soft_links=[soft("A", "C", 0.8)],
        population_count=2,
    )
    assert result["summary"]["actor_count"] == 2
    assert {row["actor_id"] for row in result["actors"]} == {"actor-001", "actor-002"}


def test_interpolated_points_do_not_create_anchor_population():
    tracklets = [tracklet("A"), tracklet("B"), tracklet("C")]
    trajectories = [
        trajectory("IA", 10, 1.0, "A"),
        trajectory("IB", 10, 1.0, "B"),
        {
            "identity_id": "IC",
            "points": [
                {
                    "identity_id": "IC",
                    "frame_index": 10,
                    "timestamp_s": 1.0,
                    "origin": "interpolated",
                    "position_std_m": 0.2,
                    "source_tracklet_ids": ["C"],
                }
            ],
        },
    ]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B")],
        trajectories=trajectories,
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("C", "IC"),
        ],
        population_count=2,
    )
    assert result["summary"]["chosen_anchor_frame_index"] == 10
    assert result["summary"]["chosen_anchor_tracklet_count"] == 2


def test_anchor_tracklets_are_fixed_to_distinct_actor_slots():
    tracklets = [tracklet("A"), tracklet("B"), tracklet("C")]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B")],
        trajectories=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
        assignments=[
            identity_assignment("A", "IA"),
            identity_assignment("B", "IB"),
            identity_assignment("C", "IC"),
        ],
        population_count=2,
    )
    assert actor_of(result, "A") != actor_of(result, "B")
    assert actor_of(result, "A") == "actor-001"
    assert actor_of(result, "B") == "actor-002"


def test_identity_membership_reports_split_or_noise_instead_of_hiding_it():
    tracklets = [tracklet("A"), tracklet("B"), tracklet("C")]
    result = solve(
        tracklets=tracklets,
        hard_pairs=[hard("A", "B"), hard("A", "C"), hard("B", "C")],
        trajectories=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
        assignments=[
            identity_assignment("A", "I1"),
            identity_assignment("B", "I2"),
            identity_assignment("C", "I1"),
        ],
        population_count=2,
    )
    i1 = next(row for row in result["identity_membership"] if row["identity_id"] == "I1")
    assert i1["status"] == "partial_noise"
    assert i1["noise_tracklet_ids"] == ["C"]


def test_low_population_confidence_is_rejected():
    with pytest.raises(ValueError, match="population confidence"):
        solve_anchored_actor_assignment(
            tracklet_rows=[tracklet("A"), tracklet("B")],
            physical_candidate_rows=[hard("A", "B")],
            trajectory_rows=[trajectory("IA", 10, 1.0, "A"), trajectory("IB", 10, 1.0, "B")],
            identity_assignment_rows=[identity_assignment("A", "IA"), identity_assignment("B", "IB")],
            identity_link_rows=[],
            scored_link_rows=[],
            population=population(2, "medium"),
        )


def test_long_window_is_rejected_for_window_level_actor_slots():
    trajectories = [
        {
            "identity_id": "IA",
            "points": [
                {
                    "identity_id": "IA",
                    "frame_index": 0,
                    "timestamp_s": 0.0,
                    "origin": "observed",
                    "position_std_m": 0.1,
                    "source_tracklet_ids": ["A"],
                },
                {
                    "identity_id": "IA",
                    "frame_index": 1000,
                    "timestamp_s": 130.0,
                    "origin": "observed",
                    "position_std_m": 0.1,
                    "source_tracklet_ids": ["A"],
                },
            ],
        },
        trajectory("IB", 0, 0.0, "B"),
    ]
    with pytest.raises(ValueError, match="exceeds max_window_s"):
        solve_anchored_actor_assignment(
            tracklet_rows=[tracklet("A"), tracklet("B")],
            physical_candidate_rows=[hard("A", "B")],
            trajectory_rows=trajectories,
            identity_assignment_rows=[identity_assignment("A", "IA"), identity_assignment("B", "IB")],
            identity_link_rows=[],
            scored_link_rows=[],
            population=population(2),
            config=PersistentActorAssignmentConfig(max_window_s=120.0),
        )
