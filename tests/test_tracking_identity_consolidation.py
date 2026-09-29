from __future__ import annotations

import json
from tracking_v0.identity_consolidation import (
    IdentityConsolidationConfig,
    consolidate_identities,
    handover_candidate,
)
from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample
from tracking_v0.quality import analyze_trajectories


DT = 1.0 / 15.0
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def sample(
    track_id: int,
    frame_index: int,
    *,
    x: float,
    z: float,
    vx: float = 0.0,
    vz: float = 0.0,
    kind: PositionKind = PositionKind.OBSERVED,
    confidence: float | None = 0.95,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=frame_index * DT,
        x=x,
        z=z,
        vx=vx,
        vz=vz,
        lifecycle=TrackLifecycle.ACTIVE,
        position_kind=kind,
        measurement_id=(
            f"{frame_index}:{track_id}"
            if kind is PositionKind.OBSERVED
            else None
        ),
        confidence=(
            confidence
            if kind is PositionKind.OBSERVED
            else None
        ),
    )


def canonical_with_handover_gap(
    track_id: int,
    *,
    first_frame: int = 0,
    last_frame: int = 74,
    handover_start: int = 30,
    handover_end: int = 40,
    x0: float = 0.0,
    z: float = 0.0,
    vx: float = 1.0,
) -> list[TrackSample]:
    result: list[TrackSample] = []

    for frame in range(first_frame, last_frame + 1):
        timestamp = frame * DT
        x = x0 + vx * timestamp
        kind = (
            PositionKind.PREDICTED
            if handover_start <= frame <= handover_end
            else PositionKind.OBSERVED
        )
        result.append(
            sample(
                track_id,
                frame,
                x=x,
                z=z,
                vx=vx,
                kind=kind,
            )
        )

    return result


def fragment(
    track_id: int,
    *,
    start_frame: int,
    end_frame: int,
    x_offset: float,
    z: float = 0.0,
    vx: float = 1.0,
    confidence: float = 0.97,
) -> list[TrackSample]:
    result: list[TrackSample] = []

    for frame in range(start_frame, end_frame + 1):
        timestamp = frame * DT
        result.append(
            sample(
                track_id,
                frame,
                x=vx * timestamp + x_offset,
                z=z,
                vx=vx,
                kind=PositionKind.OBSERVED,
                confidence=confidence,
            )
        )

    return result


def quality_report_for(
    trajectories: dict[int, list[TrackSample] | tuple[TrackSample, ...]],
):
    return analyze_trajectories(
        trajectories,
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )


def test_handover_candidate_detects_observed_fragment_over_predicted_canonical() -> None:
    canonical = canonical_with_handover_gap(10)
    frag = fragment(
        27,
        start_frame=30,
        end_frame=40,
        x_offset=0.35,
    )

    result = handover_candidate(frag, canonical)

    assert result is not None
    assert result.fragment_id == 27
    assert result.candidate_id == 10
    assert result.temporal_coverage == 1.0
    assert result.handover_fraction == 1.0
    assert result.simultaneous_observed_fraction == 0.0
    assert result.median_distance_m < 0.5
    assert result.within_0_75m_fraction == 1.0
    assert result.candidate_prior_observed_samples >= 30
    assert result.candidate_later_observed_samples > 0


def test_clear_identity_handover_merges_fragment_into_original_id() -> None:
    trajectories = {
        10: canonical_with_handover_gap(10),
        27: fragment(
            27,
            start_frame=30,
            end_frame=40,
            x_offset=0.35,
        ),
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert tuple(consolidated) == (10,)
    assert report.canonical_id_by_track == {10: 10, 27: 10}
    assert len(report.merges) == 1

    merge = report.merges[0]
    assert merge.fragment_id == 27
    assert merge.canonical_id == 10
    assert merge.candidate.handover_fraction == 1.0

    by_frame = {
        item.frame_index: item
        for item in consolidated[10]
    }
    for frame in range(30, 41):
        assert by_frame[frame].track_id == 10
        assert by_frame[frame].position_kind is PositionKind.OBSERVED
        assert by_frame[frame].measurement_id == f"{frame}:27"


def test_simultaneously_observed_people_are_not_merged() -> None:
    canonical = canonical_with_handover_gap(
        10,
        handover_start=1000,
        handover_end=1001,
    )
    frag = fragment(
        27,
        start_frame=30,
        end_frame=40,
        x_offset=0.30,
    )

    trajectories = {
        10: canonical,
        27: frag,
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert set(consolidated) == {10, 27}
    assert report.canonical_id_by_track[27] == 27
    assert not report.merges
    assert report.rejected_proposals[27] == "not_a_temporal_handover"


def test_ambiguous_two_eligible_candidates_prevent_identity_merge() -> None:
    canonical_10 = canonical_with_handover_gap(
        10,
        x0=0.0,
        z=0.0,
    )
    canonical_11 = canonical_with_handover_gap(
        11,
        x0=0.15,
        z=0.0,
    )
    frag = fragment(
        27,
        start_frame=30,
        end_frame=40,
        x_offset=0.30,
    )

    trajectories = {
        10: canonical_10,
        11: canonical_11,
        27: frag,
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert set(consolidated) == {10, 11, 27}
    assert report.canonical_id_by_track[27] == 27
    assert len(report.candidate_rankings[27]) >= 2
    assert report.rejected_proposals[27] in {
        "best_candidate_not_unique_enough",
        "best_candidate_margin_too_small",
    }


def test_ineligible_runner_up_does_not_block_clear_handover() -> None:
    """
    Regression for the real-clip TRACK 35 pattern:

    - candidate 10 is a genuine handover: it is PREDICTED while fragment 27 is
      OBSERVED;
    - candidate 11 is spatially near enough to get a non-trivial soft score,
      but remains OBSERVED at the same time, so it is a different person.

    Candidate 11 must remain visible in diagnostics, but it must NOT count as
    the runner-up for the uniqueness ratio/margin.
    """
    canonical_10 = canonical_with_handover_gap(
        10,
        x0=0.0,
        z=0.0,
    )
    canonical_11 = canonical_with_handover_gap(
        11,
        x0=3.8,
        z=0.0,
        handover_start=1000,
        handover_end=1001,
    )
    frag = fragment(
        27,
        start_frame=30,
        end_frame=40,
        x_offset=3.9,
    )

    # Make candidate 10 plausible at ~3.9 m, while candidate 11 is even closer
    # spatially but is simultaneously OBSERVED and therefore hard-ineligible.
    trajectories = {
        10: canonical_10,
        11: canonical_11,
        27: frag,
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    ranking = report.candidate_rankings[27]
    assert {candidate.candidate_id for candidate in ranking[:2]} == {10, 11}

    assert report.canonical_id_by_track[27] == 10
    assert tuple(consolidated) == (10, 11)
    assert len(report.merges) == 1
    assert report.merges[0].fragment_id == 27
    assert report.merges[0].canonical_id == 10


def test_tiny_fragment_is_preserved_instead_of_force_merged() -> None:
    trajectories = {
        10: canonical_with_handover_gap(10),
        27: fragment(
            27,
            start_frame=34,
            end_frame=37,
            x_offset=0.20,
        ),
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert set(consolidated) == {10, 27}
    assert report.canonical_id_by_track[27] == 27
    assert report.rejected_proposals[27] == "too_few_fragment_observations"


def test_quality_rejected_track_never_participates_in_identity_consolidation() -> None:
    canonical = canonical_with_handover_gap(10)

    ghost: list[TrackSample] = []
    for frame in range(30, 60):
        kind = (
            PositionKind.OBSERVED
            if frame in {35, 36, 37}
            else PositionKind.PREDICTED
        )
        ghost.append(
            sample(
                24,
                frame,
                x=0.0,
                z=FIELD_WIDTH / 2.0 - 0.10,
                kind=kind,
            )
        )

    trajectories = {
        10: canonical,
        24: ghost,
    }
    quality = quality_report_for(trajectories)

    assert 24 in quality.rejected_track_ids

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert tuple(consolidated) == (10,)
    assert 24 not in report.canonical_id_by_track
    assert all(
        merge.fragment_id != 24
        for merge in report.merges
    )


def test_two_non_overlapping_fragments_can_return_to_same_identity() -> None:
    canonical: list[TrackSample] = []

    for frame in range(0, 100):
        timestamp = frame * DT
        kind = (
            PositionKind.PREDICTED
            if 30 <= frame <= 40 or 60 <= frame <= 70
            else PositionKind.OBSERVED
        )
        canonical.append(
            sample(
                10,
                frame,
                x=timestamp,
                z=0.0,
                vx=1.0,
                kind=kind,
            )
        )

    frag_27 = fragment(
        27,
        start_frame=30,
        end_frame=40,
        x_offset=0.30,
    )
    frag_35 = fragment(
        35,
        start_frame=60,
        end_frame=70,
        x_offset=0.25,
    )

    trajectories = {
        10: canonical,
        27: frag_27,
        35: frag_35,
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert tuple(consolidated) == (10,)
    assert report.canonical_id_by_track == {
        10: 10,
        27: 10,
        35: 10,
    }
    assert {
        (merge.fragment_id, merge.canonical_id)
        for merge in report.merges
    } == {
        (27, 10),
        (35, 10),
    }


def test_conflicting_fragments_cannot_both_replace_same_identity() -> None:
    canonical = canonical_with_handover_gap(10)

    strong = fragment(
        27,
        start_frame=30,
        end_frame=40,
        x_offset=0.20,
        confidence=0.99,
    )
    weaker = fragment(
        28,
        start_frame=30,
        end_frame=40,
        x_offset=1.80,
        confidence=0.90,
    )

    trajectories = {
        10: canonical,
        27: strong,
        28: weaker,
    }
    quality = quality_report_for(trajectories)

    consolidated, report = consolidate_identities(
        trajectories,
        quality_report=quality,
        config=IdentityConsolidationConfig(
            min_score_ratio=1.01,
            min_score_margin=0.001,
        ),
    )

    merged_fragments = {
        merge.fragment_id
        for merge in report.merges
    }

    assert 27 in merged_fragments
    assert 28 not in merged_fragments
    assert report.canonical_id_by_track[27] == 10
    assert report.canonical_id_by_track[28] == 28
    assert (
        report.rejected_proposals[28]
        == "conflicts_with_stronger_fragment"
    )
    assert set(consolidated) == {10, 28}


def test_report_to_dict_is_strict_json_safe_for_unique_candidate_ratio() -> None:
    trajectories = {
        10: canonical_with_handover_gap(10),
        27: fragment(
            27,
            start_frame=30,
            end_frame=40,
            x_offset=0.35,
        ),
    }
    quality = quality_report_for(trajectories)

    _, report = consolidate_identities(
        trajectories,
        quality_report=quality,
    )

    assert len(report.merges) == 1
    assert report.merges[0].score_ratio == float("inf")

    payload = report.to_dict()
    assert payload["merges"][0]["score_ratio"] is None

    # Must not raise: this is exactly the mode used by the audit scripts.
    encoded = json.dumps(payload, allow_nan=False)
    assert '"score_ratio": null' in encoded
