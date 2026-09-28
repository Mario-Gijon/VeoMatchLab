from __future__ import annotations

import pytest

import tracking.fixed_roster as fixed_roster
from tracking.fixed_roster import (
    FixedRosterConfig,
    FixedRosterInfeasibleError,
    IdentityPairEvidence,
    PairEvidenceMode,
    identity_pair_evidence,
    resolve_fixed_roster,
)
from tracking.models import PositionKind, TrackLifecycle, TrackSample


DT = 1.0 / 15.0


def sample(
    track_id: int,
    frame: int,
    *,
    x: float,
    z: float = 0.0,
    vx: float = 0.0,
    vz: float = 0.0,
    kind: PositionKind = PositionKind.OBSERVED,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame,
        timestamp=frame * DT,
        x=x,
        z=z,
        vx=vx,
        vz=vz,
        lifecycle=TrackLifecycle.ACTIVE,
        position_kind=kind,
        measurement_id=(
            f"{frame}:{track_id}"
            if kind is PositionKind.OBSERVED
            else None
        ),
        confidence=0.9 if kind is PositionKind.OBSERVED else None,
    )


def trajectory(
    track_id: int,
    frames: range,
    *,
    x0: float,
    vx: float = 0.0,
    z: float = 0.0,
    kind: PositionKind = PositionKind.OBSERVED,
) -> tuple[TrackSample, ...]:
    first = frames.start
    return tuple(
        sample(
            track_id,
            frame,
            x=x0 + vx * ((frame - first) * DT),
            z=z,
            vx=vx,
            kind=kind,
        )
        for frame in frames
    )


def fake_evidence(
    earlier_id: int,
    later_id: int,
    score: float,
    *,
    conflict: bool = False,
) -> IdentityPairEvidence:
    return IdentityPairEvidence(
        earlier_id=earlier_id,
        later_id=later_id,
        score=0.0 if conflict else score,
        mode=(
            PairEvidenceMode.NONE
            if conflict
            else PairEvidenceMode.HANDOVER
        ),
        hard_conflict=conflict,
        hard_conflict_reason="test_conflict" if conflict else None,
        simultaneous_observed_frames=10 if conflict else 0,
        simultaneous_observed_fraction=1.0 if conflict else 0.0,
        simultaneous_median_distance_m=20.0 if conflict else None,
        simultaneous_p90_distance_m=20.0 if conflict else None,
        simultaneous_median_velocity_difference_m_s=0.0 if conflict else None,
        handover_compared_frames=10 if not conflict else 0,
        handover_coverage=1.0 if not conflict else 0.0,
        handover_fraction=1.0 if not conflict else 0.0,
        handover_median_distance_m=0.5 if not conflict else None,
        handover_p90_distance_m=0.7 if not conflict else None,
        handover_median_velocity_difference_m_s=0.2 if not conflict else None,
        handover_score=0.0 if conflict else score,
        temporal_gap_seconds=None,
        temporal_endpoint_distance_m=None,
        temporal_implied_speed_m_s=None,
        temporal_forward_error_m=None,
        temporal_backward_error_m=None,
        temporal_velocity_difference_m_s=None,
        temporal_score=0.0,
        duplicate_score=0.0,
    )


def test_far_simultaneous_tracks_are_hard_conflict():
    one = trajectory(1, range(0, 30), x0=0.0)
    two = trajectory(2, range(0, 30), x0=15.0)

    evidence = identity_pair_evidence(one, two)

    assert evidence.hard_conflict
    assert evidence.score == 0.0
    assert evidence.mode is PairEvidenceMode.NONE


def test_close_simultaneous_tracks_are_hard_conflict_even_when_nearby():
    one = trajectory(1, range(0, 30), x0=0.0, vx=1.0)
    two = trajectory(2, range(0, 30), x0=0.3, vx=1.0)

    evidence = identity_pair_evidence(one, two)

    assert evidence.hard_conflict
    assert evidence.mode is PairEvidenceMode.NONE
    assert evidence.score == 0.0
    assert evidence.duplicate_score == 0.0


def test_absurd_handover_distance_cannot_score_like_old_additive_formula():
    parent = []
    for frame in range(0, 30):
        kind = (
            PositionKind.PREDICTED
            if 10 <= frame < 25
            else PositionKind.OBSERVED
        )
        parent.append(
            sample(
                1,
                frame,
                x=0.0,
                vx=0.0,
                kind=kind,
            )
        )

    child = trajectory(
        2,
        range(10, 25),
        x0=50.0,
        vx=0.0,
    )

    evidence = identity_pair_evidence(tuple(parent), child)

    assert not evidence.hard_conflict
    assert evidence.handover_score == 0.0
    assert evidence.score < 0.03


def test_good_handover_scores_high():
    parent = []
    for frame in range(0, 40):
        kind = (
            PositionKind.PREDICTED
            if 15 <= frame < 27
            else PositionKind.OBSERVED
        )
        parent.append(
            sample(
                1,
                frame,
                x=0.08 * frame,
                vx=1.2,
                kind=kind,
            )
        )

    child = tuple(
        sample(
            2,
            frame,
            x=0.08 * frame + 0.15,
            vx=1.2,
        )
        for frame in range(15, 27)
    )

    evidence = identity_pair_evidence(tuple(parent), child)

    assert not evidence.hard_conflict
    assert evidence.mode is PairEvidenceMode.HANDOVER
    assert evidence.handover_fraction == pytest.approx(1.0)
    assert evidence.score >= 0.75


def test_partial_handover_below_seventy_percent_is_rejected():
    parent = []
    for frame in range(0, 50):
        kind = (
            PositionKind.PREDICTED
            if 20 <= frame < 26
            else PositionKind.OBSERVED
        )
        parent.append(
            sample(
                1,
                frame,
                x=0.05 * frame,
                vx=0.75,
                kind=kind,
            )
        )

    # The child overlaps for 12 frames, but the old identity is predicted for
    # only 6 of them: handover_fraction=0.5, below the conservative 0.70 gate.
    child = tuple(
        sample(
            2,
            frame,
            x=0.05 * frame + 0.10,
            vx=0.75,
        )
        for frame in range(20, 32)
    )

    evidence = identity_pair_evidence(tuple(parent), child)

    assert evidence.handover_fraction == pytest.approx(0.5)
    assert evidence.handover_score == 0.0
    # Appreciable simultaneous OBSERVED support also makes these identities
    # distinct, so the final score must remain zero.
    assert evidence.score == 0.0


def test_temporal_continuation_still_works():
    one = trajectory(
        1,
        range(0, 20),
        x0=0.0,
        vx=2.0,
    )
    two = trajectory(
        2,
        range(21, 40),
        x0=2.0 * 21 * DT,
        vx=2.0,
    )

    evidence = identity_pair_evidence(one, two)

    assert not evidence.hard_conflict
    assert evidence.mode is PairEvidenceMode.TEMPORAL
    assert evidence.score >= 0.75


def test_exact_two_identity_solution_merges_fragment():
    trajectories = {
        1: trajectory(
            1,
            range(0, 20),
            x0=0.0,
            vx=2.0,
        ),
        2: trajectory(
            2,
            range(0, 40),
            x0=20.0,
            vx=0.2,
            z=10.0,
        ),
        3: trajectory(
            3,
            range(21, 40),
            x0=2.0 * 21 * DT,
            vx=2.0,
        ),
    }

    final, report = resolve_fixed_roster(
        trajectories,
        config=FixedRosterConfig(roster_size=2),
    )

    assert len(final) == 2
    assert report.final_identity_count == 2
    assert report.representative_by_track[1] == report.representative_by_track[3]
    assert report.representative_by_track[2] != report.representative_by_track[1]


def test_global_solver_avoids_greedy_trap(monkeypatch):
    # 3 and 4 are definitely different. Greedy 1->3=.90 would then make the
    # remaining assignment expensive. Global optimum is 1->4 + 2->3.
    table = {
        (1, 3): fake_evidence(1, 3, 0.90),
        (1, 4): fake_evidence(1, 4, 0.80),
        (2, 3): fake_evidence(2, 3, 0.79),
        (2, 4): fake_evidence(2, 4, 0.10),
        (3, 4): fake_evidence(3, 4, 0.0, conflict=True),
    }

    def patched(first, second, *, config=None):
        ids = tuple(sorted((first[0].track_id, second[0].track_id)))
        return table.get(
            ids,
            fake_evidence(ids[0], ids[1], 0.0),
        )

    monkeypatch.setattr(
        fixed_roster,
        "identity_pair_evidence",
        patched,
    )

    trajectories = {
        1: trajectory(1, range(0, 10), x0=0.0),
        2: trajectory(2, range(0, 10), x0=10.0),
        3: trajectory(3, range(20, 30), x0=0.0),
        4: trajectory(4, range(20, 30), x0=10.0),
    }

    _, report = resolve_fixed_roster(
        trajectories,
        config=FixedRosterConfig(
            roster_size=2,
            minimum_parent_score=0.03,
        ),
    )

    links = {
        (link.parent_id, link.child_id)
        for link in report.selected_parent_links
    }
    assert links == {(1, 4), (2, 3)}


def test_transitive_hard_conflict_blocks_same_component(monkeypatch):
    table = {
        (1, 2): fake_evidence(1, 2, 0.95),
        (2, 3): fake_evidence(2, 3, 0.94),
        (1, 3): fake_evidence(1, 3, 0.0, conflict=True),
    }

    def patched(first, second, *, config=None):
        ids = tuple(sorted((first[0].track_id, second[0].track_id)))
        return table[ids]

    monkeypatch.setattr(
        fixed_roster,
        "identity_pair_evidence",
        patched,
    )

    trajectories = {
        1: trajectory(1, range(0, 10), x0=0.0),
        2: trajectory(2, range(10, 20), x0=0.0),
        3: trajectory(3, range(20, 30), x0=0.0),
    }

    with pytest.raises(FixedRosterInfeasibleError):
        resolve_fixed_roster(
            trajectories,
            config=FixedRosterConfig(roster_size=1),
        )


def test_one_identity_can_have_multiple_close_fragments(monkeypatch):
    table = {
        (1, 2): fake_evidence(1, 2, 0.90),
        (1, 3): fake_evidence(1, 3, 0.85),
        (2, 3): fake_evidence(2, 3, 0.80),
    }

    def patched(first, second, *, config=None):
        ids = tuple(sorted((first[0].track_id, second[0].track_id)))
        return table[ids]

    monkeypatch.setattr(
        fixed_roster,
        "identity_pair_evidence",
        patched,
    )

    trajectories = {
        1: trajectory(1, range(0, 10), x0=0.0),
        2: trajectory(2, range(10, 20), x0=0.0),
        3: trajectory(3, range(20, 30), x0=0.0),
    }

    final, report = resolve_fixed_roster(
        trajectories,
        config=FixedRosterConfig(roster_size=1),
    )

    assert len(final) == 1
    assert report.final_identity_count == 1
    assert len(report.selected_parent_links) == 2


def test_exact_roster_preserves_distinct_tracks():
    trajectories = {
        1: trajectory(1, range(0, 20), x0=0.0),
        2: trajectory(2, range(0, 20), x0=15.0),
    }

    final, report = resolve_fixed_roster(
        trajectories,
        config=FixedRosterConfig(roster_size=2),
    )

    assert set(final) == {1, 2}
    assert report.selected_parent_links == ()
    assert report.final_identity_count == 2


def test_roster_larger_than_input_is_infeasible():
    with pytest.raises(FixedRosterInfeasibleError):
        resolve_fixed_roster(
            {
                1: trajectory(
                    1,
                    range(0, 10),
                    x0=0.0,
                ),
            },
            config=FixedRosterConfig(roster_size=2),
        )


def test_too_many_forced_roots_is_infeasible():
    trajectories = {
        1: trajectory(1, range(0, 20), x0=0.0),
        2: trajectory(2, range(0, 20), x0=15.0),
        3: trajectory(3, range(0, 20), x0=30.0),
    }

    with pytest.raises(
        FixedRosterInfeasibleError,
        match="forced_roots",
    ):
        resolve_fixed_roster(
            trajectories,
            config=FixedRosterConfig(roster_size=1),
        )


def test_report_scores_are_in_unit_interval():
    trajectories = {
        1: trajectory(
            1,
            range(0, 20),
            x0=0.0,
            vx=1.0,
        ),
        2: trajectory(
            2,
            range(21, 40),
            x0=21 * DT,
            vx=1.0,
        ),
    }

    _, report = resolve_fixed_roster(
        trajectories,
        config=FixedRosterConfig(roster_size=1),
    )

    payload = report.to_dict()
    assert payload["summary"]["final_identities"] == 1
    assert len(payload["selected_parent_links"]) == 1
    score = payload["selected_parent_links"][0]["score"]
    assert 0.0 <= score <= 1.0


def test_long_successor_is_not_penalized_for_outliving_predecessor_support():
    # Successor continues for a long time after the old ID has fully expired.
    # V2.7.1 incorrectly used the whole successor lifetime as the coverage
    # denominator, which killed otherwise excellent handovers like 1 -> 29.
    parent = []
    for frame in range(0, 80):
        kind = (
            PositionKind.PREDICTED
            if 35 <= frame < 80
            else PositionKind.OBSERVED
        )
        parent.append(
            sample(
                1,
                frame,
                x=0.05 * frame,
                vx=0.75,
                kind=kind,
            )
        )

    child = tuple(
        sample(
            2,
            frame,
            x=0.05 * frame + 0.10,
            vx=0.75,
        )
        for frame in range(40, 240)
    )

    evidence = identity_pair_evidence(tuple(parent), child)

    assert not evidence.hard_conflict
    assert evidence.mode is PairEvidenceMode.HANDOVER
    assert evidence.handover_coverage == pytest.approx(1.0)
    assert evidence.handover_fraction >= 0.70
    assert evidence.score >= 0.75


def test_strong_handover_can_override_moderate_simultaneous_observed_overlap():
    # Same-person ID switch can briefly give real observations to both IDs.
    # This mirrors the real 9 <-> 38 pattern: ~27% simultaneous observations
    # but >70% of the supported transition is old-PREDICTED/new-OBSERVED.
    parent = []
    for frame in range(0, 90):
        kind = (
            PositionKind.PREDICTED
            if 30 <= frame < 90
            else PositionKind.OBSERVED
        )
        parent.append(
            sample(
                1,
                frame,
                x=0.04 * frame,
                vx=0.60,
                kind=kind,
            )
        )

    child = tuple(
        sample(
            2,
            frame,
            x=0.04 * frame + 0.15,
            vx=0.60,
        )
        for frame in range(15, 71)
    )

    evidence = identity_pair_evidence(tuple(parent), child)

    assert evidence.simultaneous_observed_fraction > 0.20
    assert evidence.handover_fraction >= 0.70
    assert not evidence.hard_conflict
    assert evidence.mode is PairEvidenceMode.HANDOVER
    assert evidence.score >= 0.75
