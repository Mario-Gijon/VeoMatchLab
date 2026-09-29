from __future__ import annotations

import pytest

import tracking_v0.evidence_roster as evidence_roster
from tracking_v0.evidence_roster import (
    EvidenceRosterConfig,
    EvidenceRosterInfeasibleError,
    TrackAssignmentKind,
    analyze_track_evidence,
    resolve_evidence_roster,
    trim_unsupported_prediction_edges,
)
from tracking_v0.fixed_roster import (
    IdentityPairEvidence,
    PairEvidenceMode,
)
from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample


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


def ghost_trajectory(
    track_id: int,
    *,
    observed_frames: range,
    predicted_until: int,
    x0: float,
) -> tuple[TrackSample, ...]:
    rows = [
        sample(
            track_id,
            frame,
            x=x0,
            kind=PositionKind.OBSERVED,
        )
        for frame in observed_frames
    ]
    rows.extend(
        sample(
            track_id,
            frame,
            x=x0 + 0.15 * (frame - observed_frames.stop),
            vx=2.0,
            kind=PositionKind.PREDICTED,
        )
        for frame in range(
            observed_frames.stop,
            predicted_until,
        )
    )
    return tuple(rows)


def fake_pair(
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


def test_stable_track_scores_higher_than_short_ghost():
    stable = trajectory(
        1,
        range(0, 150),
        x0=0.0,
        vx=1.0,
    )
    ghost = ghost_trajectory(
        2,
        observed_frames=range(0, 5),
        predicted_until=50,
        x0=10.0,
    )

    evidence = analyze_track_evidence(
        {1: stable, 2: ghost},
    )

    assert evidence[1].real_score > 0.90
    assert evidence[2].real_score < evidence[1].real_score
    assert evidence[2].ghost_score > evidence[1].ghost_score


def test_ghost_metrics_expose_prediction_dominance_and_tail():
    ghost = ghost_trajectory(
        2,
        observed_frames=range(0, 5),
        predicted_until=50,
        x0=10.0,
    )

    metrics = analyze_track_evidence({2: ghost})[2]

    assert metrics.observed_count == 5
    assert metrics.predicted_count == 45
    assert metrics.observed_fraction == pytest.approx(0.10)
    assert metrics.trailing_prediction_seconds > 2.0
    assert metrics.longest_observed_run_seconds < 0.5


def test_trim_removes_long_dead_tail_but_keeps_short_grace():
    track = ghost_trajectory(
        1,
        observed_frames=range(0, 5),
        predicted_until=50,
        x0=0.0,
    )

    trimmed, report = trim_unsupported_prediction_edges(
        {1: track},
        grace_seconds=0.20,
    )

    assert trimmed[1][-1].timestamp <= track[4].timestamp + 0.20 + 1e-9
    assert report[1].removed_trailing_samples > 30


def test_trim_preserves_internal_predicted_gap():
    rows = []
    for frame in range(0, 20):
        kind = (
            PositionKind.PREDICTED
            if 5 <= frame <= 10
            else PositionKind.OBSERVED
        )
        rows.append(
            sample(
                1,
                frame,
                x=float(frame),
                kind=kind,
            )
        )

    trimmed, _ = trim_unsupported_prediction_edges(
        {1: tuple(rows)},
        grace_seconds=0.0,
    )

    assert any(
        item.position_kind is PositionKind.PREDICTED
        for item in trimmed[1]
    )
    assert trimmed[1][0].frame_index == 0
    assert trimmed[1][-1].frame_index == 19


def test_close_simultaneous_real_tracks_remain_distinct_roots():
    # Two real players may spend the whole clip only 30 cm apart in noisy X/Z.
    # V2.7.1 must never merge them on geometry alone.
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0, vx=1.0),
        2: trajectory(2, range(0, 100), x0=0.3, vx=1.0),
    }

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=2),
    )

    assert report.junk_track_ids == ()
    assert report.root_track_ids == (1, 2)
    assert report.representative_by_track[1] == 1
    assert report.representative_by_track[2] == 2


def test_solver_can_discard_ghost_instead_of_forcing_it_into_roster(monkeypatch):
    # Three good tracks represent the exact roster; track 4 is a low-evidence
    # ghost and should be JUNK instead of consuming/merging into a participant.
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=15.0),
        3: trajectory(3, range(0, 100), x0=30.0),
        4: ghost_trajectory(
            4,
            observed_frames=range(10, 15),
            predicted_until=60,
            x0=50.0,
        ),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    final, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=3),
    )

    assert len(final) == 3
    assert report.junk_track_ids == (4,)
    assert report.assignments[4].kind is TrackAssignmentKind.JUNK


def test_good_fragment_is_kept_and_merged(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=20.0),
        3: trajectory(3, range(0, 100), x0=40.0),
        4: trajectory(4, range(40, 80), x0=0.2),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        if (a, b) == (1, 4):
            return fake_pair(1, 4, 0.95)
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=3),
    )

    assert report.assignments[4].kind is TrackAssignmentKind.FRAGMENT
    assert report.assignments[4].representative_id == 1
    assert 4 not in report.junk_track_ids


def test_low_evidence_fragment_with_weak_match_prefers_junk(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=20.0),
        3: trajectory(3, range(0, 100), x0=40.0),
        4: ghost_trajectory(
            4,
            observed_frames=range(20, 25),
            predicted_until=70,
            x0=0.2,
        ),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        if (a, b) == (1, 4):
            return fake_pair(1, 4, 0.20)
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=3),
    )

    assert report.assignments[4].kind is TrackAssignmentKind.JUNK


def test_high_evidence_track_is_expensive_to_discard(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=20.0),
        3: trajectory(3, range(0, 100), x0=40.0),
        4: trajectory(4, range(40, 100), x0=0.1),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        if (a, b) == (1, 4):
            return fake_pair(1, 4, 0.70)
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=3),
    )

    assert report.assignments[4].kind is TrackAssignmentKind.FRAGMENT


def test_hard_conflict_cannot_be_bypassed_transitively(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(20, 80), x0=0.1),
        3: trajectory(3, range(40, 100), x0=0.2),
    }

    table = {
        (1, 2): fake_pair(1, 2, 0.95),
        (2, 3): fake_pair(2, 3, 0.95),
        (1, 3): fake_pair(1, 3, 0.0, conflict=True),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        return table[(a, b)]

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=2),
    )

    assert report.representative_by_track[1] != report.representative_by_track[3]
    assert report.junk_track_ids == ()
    assert len(report.components) == 2


def test_exact_number_of_roots_is_preserved(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=20.0),
        3: trajectory(3, range(0, 100), x0=40.0),
        4: trajectory(4, range(40, 80), x0=0.1),
        5: ghost_trajectory(
            5,
            observed_frames=range(10, 15),
            predicted_until=50,
            x0=60.0,
        ),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        if (a, b) == (1, 4):
            return fake_pair(1, 4, 0.90)
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=3),
    )

    assert len(report.root_track_ids) == 3
    assert report.junk_track_ids == (5,)


def test_roster_larger_than_available_tracks_is_infeasible():
    trajectories = {
        1: trajectory(1, range(0, 20), x0=0.0),
    }

    with pytest.raises(EvidenceRosterInfeasibleError):
        resolve_evidence_roster(
            trajectories,
            config=EvidenceRosterConfig(roster_size=2),
        )


def test_report_exposes_all_score_components(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=20.0),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=2),
    )
    payload = report.to_dict()

    row = payload["track_evidence"]["1"]
    assert "count_score" in row
    assert "span_score" in row
    assert "run_score" in row
    assert "ghost_score" in row
    assert "real_score" in row


def test_anchor_score_prefers_long_identity_over_short_clean_fragment():
    long_track = trajectory(
        1,
        range(0, 450),
        x0=0.0,
        vx=0.5,
    )
    short_clean = trajectory(
        2,
        range(200, 275),
        x0=5.0,
        vx=0.5,
    )

    evidence = analyze_track_evidence(
        {
            1: long_track,
            2: short_clean,
        }
    )

    assert evidence[2].real_score > 0.85
    assert evidence[1].anchor_score > evidence[2].anchor_score
    assert evidence[1].observed_span_coverage > 0.95
    assert evidence[2].observed_span_coverage < 0.25


def test_root_choice_uses_anchor_score_while_fragment_uses_real_score(monkeypatch):
    # Track 1 and 2 are long stable anchors. Track 3 is a short, very clean
    # real fragment of track 1. With roster_size=2 the short fragment must be
    # kept as FRAGMENT rather than displacing one of the long anchors.
    trajectories = {
        1: trajectory(1, range(0, 450), x0=0.0),
        2: trajectory(2, range(0, 450), x0=30.0),
        3: trajectory(3, range(200, 275), x0=0.1),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        if (a, b) == (1, 3):
            return fake_pair(1, 3, 0.95)
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=2),
    )

    assert report.assignments[1].kind is TrackAssignmentKind.ROOT
    assert report.assignments[2].kind is TrackAssignmentKind.ROOT
    assert report.assignments[3].kind is TrackAssignmentKind.FRAGMENT
    assert report.assignments[3].representative_id == 1


def test_assignment_report_exposes_anchor_score(monkeypatch):
    trajectories = {
        1: trajectory(1, range(0, 100), x0=0.0),
        2: trajectory(2, range(0, 100), x0=20.0),
    }

    def patched(first, second, *, config=None):
        a, b = sorted((first[0].track_id, second[0].track_id))
        return fake_pair(a, b, 0.0, conflict=True)

    monkeypatch.setattr(
        evidence_roster,
        "identity_pair_evidence",
        patched,
    )

    _, report = resolve_evidence_roster(
        trajectories,
        config=EvidenceRosterConfig(roster_size=2),
    )

    payload = report.to_dict()
    assert "anchor_score" in payload["track_evidence"]["1"]
    assert "anchor_score" in payload["assignments"]["1"]
