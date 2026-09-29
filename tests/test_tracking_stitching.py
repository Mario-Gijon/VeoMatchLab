from __future__ import annotations

from tracking_v0.models import PositionKind, TrackLifecycle, TrackSample
from tracking_v0.stitching import (
    StitchingConfig,
    candidate_between,
    stitch_trajectories,
)


DT = 1.0 / 15.0


def make_sample(
    track_id: int,
    frame_index: int,
    timestamp: float,
    *,
    x: float,
    z: float,
    vx: float,
    vz: float,
    kind: PositionKind = PositionKind.OBSERVED,
) -> TrackSample:
    return TrackSample(
        track_id=track_id,
        frame_index=frame_index,
        timestamp=timestamp,
        x=x,
        z=z,
        vx=vx,
        vz=vz,
        lifecycle=TrackLifecycle.ACTIVE,
        position_kind=kind,
        measurement_id=f"{frame_index}:1" if kind is PositionKind.OBSERVED else None,
        confidence=0.95 if kind is PositionKind.OBSERVED else None,
    )


def linear_tracklet(
    track_id: int,
    *,
    start_time: float,
    end_time: float,
    start_x: float,
    z: float,
    vx: float,
    frame_offset: int = 0,
) -> list[TrackSample]:
    """Generate observed samples exactly following constant X velocity."""
    count = int(round((end_time - start_time) / DT)) + 1
    samples: list[TrackSample] = []

    for index in range(count):
        timestamp = start_time + index * DT
        if timestamp > end_time + 1e-9:
            break
        samples.append(
            make_sample(
                track_id,
                frame_offset + index,
                timestamp,
                x=start_x + vx * (timestamp - start_time),
                z=z,
                vx=vx,
                vz=0.0,
            )
        )

    return samples


def test_candidate_accepts_physically_continuous_tracklets() -> None:
    predecessor = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=4.0,
        vx=2.0,
    )
    successor = linear_tracklet(
        27,
        start_time=1.2,
        end_time=2.0,
        start_x=2.4,
        z=4.0,
        vx=2.0,
        frame_offset=18,
    )

    candidate = candidate_between(predecessor, successor)

    assert candidate is not None
    assert candidate.predecessor_id == 5
    assert candidate.successor_id == 27
    assert abs(candidate.gap_seconds - 0.2) < 1e-9
    assert candidate.forward_prediction_error_m < 1e-9
    assert candidate.backward_prediction_error_m < 1e-9


def test_overlapping_observed_tracklets_are_never_stitched() -> None:
    predecessor = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=4.0,
        vx=1.0,
    )
    successor = linear_tracklet(
        27,
        start_time=0.8,
        end_time=1.8,
        start_x=0.8,
        z=4.0,
        vx=1.0,
        frame_offset=12,
    )

    assert candidate_between(predecessor, successor) is None


def test_physically_impossible_jump_is_rejected() -> None:
    predecessor = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=0.0,
        vx=0.0,
    )
    successor = linear_tracklet(
        27,
        start_time=1.5,
        end_time=2.0,
        start_x=20.0,
        z=0.0,
        vx=0.0,
        frame_offset=23,
    )

    assert candidate_between(predecessor, successor) is None


def test_stitching_preserves_original_identity_id() -> None:
    track_5 = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=-6.0,
        vx=2.0,
    )
    track_27 = linear_tracklet(
        27,
        start_time=1.2,
        end_time=2.0,
        start_x=2.4,
        z=-6.0,
        vx=2.0,
        frame_offset=18,
    )

    merged, report = stitch_trajectories({
        5: track_5,
        27: track_27,
    })

    assert tuple(merged) == (5,)
    assert report.chains == ((5, 27),)
    assert report.canonical_id_by_track == {5: 5, 27: 5}
    assert len(report.links) == 1

    merged_track = merged[5]
    assert all(sample.track_id == 5 for sample in merged_track)
    assert merged_track[0].timestamp == 0.0
    assert abs(merged_track[-1].timestamp - 2.0) < 1e-9


def test_stitching_builds_multi_fragment_chain() -> None:
    track_5 = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=2.0,
        vx=1.5,
    )
    track_27 = linear_tracklet(
        27,
        start_time=1.2,
        end_time=2.0,
        start_x=1.8,
        z=2.0,
        vx=1.5,
        frame_offset=18,
    )
    track_41 = linear_tracklet(
        41,
        start_time=2.2,
        end_time=3.0,
        start_x=3.3,
        z=2.0,
        vx=1.5,
        frame_offset=33,
    )

    merged, report = stitch_trajectories({
        5: track_5,
        27: track_27,
        41: track_41,
    })

    assert tuple(merged) == (5,)
    assert report.chains == ((5, 27, 41),)
    assert report.canonical_id_by_track == {
        5: 5,
        27: 5,
        41: 5,
    }
    assert {(link.predecessor_id, link.successor_id) for link in report.links} == {
        (5, 27),
        (27, 41),
    }


def test_global_assignment_chooses_best_predecessor_for_one_successor() -> None:
    # Both tracks can reach 30, but 8 is a nearly exact continuation while 5
    # requires a larger correction. Only one may claim successor 30.
    track_5 = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=0.0,
        vx=1.0,
    )
    track_8 = linear_tracklet(
        8,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=1.5,
        vx=1.0,
        frame_offset=100,
    )
    track_30 = linear_tracklet(
        30,
        start_time=1.2,
        end_time=2.0,
        start_x=1.2,
        z=1.5,
        vx=1.0,
        frame_offset=200,
    )

    merged, report = stitch_trajectories({
        5: track_5,
        8: track_8,
        30: track_30,
    })

    links = {(link.predecessor_id, link.successor_id) for link in report.links}

    assert (8, 30) in links
    assert (5, 30) not in links
    assert report.canonical_id_by_track[30] == 8
    assert set(merged) == {5, 8}


def test_rejected_track_can_be_excluded_before_stitching() -> None:
    track_5 = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=0.0,
        vx=1.0,
    )
    rejected_24 = linear_tracklet(
        24,
        start_time=1.2,
        end_time=1.8,
        start_x=1.2,
        z=0.0,
        vx=1.0,
        frame_offset=18,
    )

    merged, report = stitch_trajectories(
        {
            5: track_5,
            24: rejected_24,
        },
        allowed_track_ids=(5,),
    )

    assert tuple(merged) == (5,)
    assert report.links == ()
    assert report.chains == ((5,),)
    assert 24 not in report.canonical_id_by_track


def test_observed_sample_wins_over_predicted_tail_during_merge() -> None:
    predecessor = linear_tracklet(
        5,
        start_time=0.0,
        end_time=1.0,
        start_x=0.0,
        z=0.0,
        vx=1.0,
    )

    # Add a forward-predicted tail after the last real observation.
    predecessor.extend([
        make_sample(
            5,
            16,
            1.1,
            x=1.1,
            z=0.0,
            vx=1.0,
            vz=0.0,
            kind=PositionKind.PREDICTED,
        ),
        make_sample(
            5,
            17,
            1.2,
            x=1.2,
            z=0.0,
            vx=1.0,
            vz=0.0,
            kind=PositionKind.PREDICTED,
        ),
    ])

    successor = [
        make_sample(
            27,
            17,
            1.2,
            x=1.2,
            z=0.0,
            vx=1.0,
            vz=0.0,
            kind=PositionKind.OBSERVED,
        ),
        make_sample(
            27,
            18,
            1.3,
            x=1.3,
            z=0.0,
            vx=1.0,
            vz=0.0,
            kind=PositionKind.OBSERVED,
        ),
    ]

    config = StitchingConfig(min_gap_seconds=0.15)
    merged, report = stitch_trajectories(
        {
            5: predecessor,
            27: successor,
        },
        config=config,
    )

    assert report.chains == ((5, 27),)
    frame_17 = next(sample for sample in merged[5] if sample.frame_index == 17)

    assert frame_17.position_kind is PositionKind.OBSERVED
    assert frame_17.measurement_id is not None
    assert frame_17.track_id == 5
