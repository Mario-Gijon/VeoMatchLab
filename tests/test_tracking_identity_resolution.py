from __future__ import annotations

import json

from tracking.identity_resolution import (
    IdentityRelationKind,
    resolve_identities,
)
from tracking.models import PositionKind, TrackLifecycle, TrackSample


DT = 1.0 / 15.0
FIELD_LENGTH = 105.0
FIELD_WIDTH = 68.0


def sample(
    track_id: int,
    frame_index: int,
    *,
    x: float,
    z: float = 0.0,
    vx: float = 1.0,
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


def canonical(
    track_id: int,
    *,
    last_frame: int = 74,
    predicted_ranges: tuple[tuple[int, int], ...] = (),
    x_offset: float = 0.0,
) -> list[TrackSample]:
    result: list[TrackSample] = []

    for frame in range(last_frame + 1):
        kind = (
            PositionKind.PREDICTED
            if any(start <= frame <= end for start, end in predicted_ranges)
            else PositionKind.OBSERVED
        )
        result.append(
            sample(
                track_id,
                frame,
                x=frame * DT + x_offset,
                kind=kind,
            )
        )

    return result


def observed_fragment(
    track_id: int,
    *,
    start_frame: int,
    end_frame: int,
    x_offset: float,
    confidence: float = 0.97,
) -> list[TrackSample]:
    return [
        sample(
            track_id,
            frame,
            x=frame * DT + x_offset,
            kind=PositionKind.OBSERVED,
            confidence=confidence,
        )
        for frame in range(start_frame, end_frame + 1)
    ]


def test_transitive_stitch_then_handover_resolves_to_original_identity() -> None:
    """
    Regression for the real-clip topology:

        30 --stitch--> 34 --handover--> 2

    Track 30 is intentionally too small for handover consolidation on its own.
    Track 34 is a strong handover fragment of established ID 2.
    The global resolver must keep both raw relations and close the transitive
    component as one final identity with canonical ID 2.
    """
    track_2 = canonical(
        2,
        predicted_ranges=((20, 40),),
    )
    track_30 = observed_fragment(
        30,
        start_frame=20,
        end_frame=24,
        x_offset=0.15,
    )
    track_34 = observed_fragment(
        34,
        start_frame=26,
        end_frame=40,
        x_offset=0.20,
    )

    final, report = resolve_identities(
        {
            2: track_2,
            30: track_30,
            34: track_34,
        },
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert tuple(final) == (2,)
    assert report.canonical_id_by_track == {
        2: 2,
        30: 2,
        34: 2,
    }
    assert report.components == ((2, 30, 34),)

    relations = {
        (relation.kind, relation.left_id, relation.right_id)
        for relation in report.accepted_relations
    }

    assert (
        IdentityRelationKind.HANDOVER,
        34,
        2,
    ) in relations
    assert (
        IdentityRelationKind.STITCH,
        30,
        34,
    ) in relations


def test_transitive_component_uses_real_fragment_observations() -> None:
    track_2 = canonical(
        2,
        predicted_ranges=((20, 40),),
    )
    track_30 = observed_fragment(
        30,
        start_frame=20,
        end_frame=24,
        x_offset=0.15,
        confidence=0.98,
    )
    track_34 = observed_fragment(
        34,
        start_frame=26,
        end_frame=40,
        x_offset=0.20,
        confidence=0.99,
    )

    final, _ = resolve_identities(
        {
            2: track_2,
            30: track_30,
            34: track_34,
        },
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    by_frame = {
        item.frame_index: item
        for item in final[2]
    }

    for frame in range(20, 25):
        assert by_frame[frame].track_id == 2
        assert by_frame[frame].position_kind is PositionKind.OBSERVED
        assert by_frame[frame].measurement_id == f"{frame}:30"

    for frame in range(26, 41):
        assert by_frame[frame].track_id == 2
        assert by_frame[frame].position_kind is PositionKind.OBSERVED
        assert by_frame[frame].measurement_id == f"{frame}:34"

    # Frame 25 has no fragment observation, so the canonical prediction stays.
    assert by_frame[25].position_kind is PositionKind.PREDICTED


def test_component_conflict_blocks_bad_transitive_bridge() -> None:
    """
    A stitch can be locally plausible at its endpoint yet globally wrong.

    Track 30 coexists with ID 2 as a clearly different person for five frames,
    later converges near the position where 34 appears, and would therefore
    produce a locally valid 30->34 stitch. Because 34 is a handover fragment
    of ID 2, blindly closing the graph would merge 30 into player 2.

    The component-level simultaneous-observation guard must reject that stitch.
    """
    track_2 = canonical(
        2,
        predicted_ranges=((15, 40),),
    )

    track_30: list[TrackSample] = []
    for frame in range(10, 25):
        if frame <= 14:
            offset = 4.0
            vx = 1.0
        else:
            # Converge from 4.0 m offset to 0.15 m before the stitch endpoint.
            progress = (frame - 14) / 10.0
            offset = 4.0 * (1.0 - progress) + 0.15 * progress
            vx = 1.0 if frame == 24 else -4.0

        track_30.append(
            sample(
                30,
                frame,
                x=frame * DT + offset,
                vx=vx,
                kind=PositionKind.OBSERVED,
            )
        )

    track_34 = observed_fragment(
        34,
        start_frame=26,
        end_frame=40,
        x_offset=0.20,
    )

    final, report = resolve_identities(
        {
            2: track_2,
            30: track_30,
            34: track_34,
        },
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert set(final) == {2, 30}
    assert report.canonical_id_by_track[2] == 2
    assert report.canonical_id_by_track[34] == 2
    assert report.canonical_id_by_track[30] == 30

    assert any(
        item.relation.kind is IdentityRelationKind.STITCH
        and item.relation.left_id == 30
        and item.relation.right_id == 34
        and item.reason.startswith("component_observation_conflict")
        for item in report.rejected_relations
    )


def test_quality_rejected_sideline_ghost_never_reaches_identity_graph() -> None:
    track_2 = canonical(2)

    ghost: list[TrackSample] = []
    for frame in range(20, 50):
        kind = (
            PositionKind.OBSERVED
            if frame in {30, 31, 32}
            else PositionKind.PREDICTED
        )
        ghost.append(
            sample(
                24,
                frame,
                x=0.0,
                z=FIELD_WIDTH / 2.0 - 0.10,
                vx=0.0,
                kind=kind,
            )
        )

    final, report = resolve_identities(
        {
            2: track_2,
            24: ghost,
        },
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert 24 in report.quality.rejected_track_ids
    assert tuple(final) == (2,)
    assert 24 not in report.canonical_id_by_track
    assert all(
        relation.left_id != 24 and relation.right_id != 24
        for relation in report.accepted_relations
    )


def test_unrelated_persistent_people_remain_separate() -> None:
    track_2 = canonical(
        2,
        x_offset=0.0,
    )
    track_3 = canonical(
        3,
        x_offset=12.0,
    )

    final, report = resolve_identities(
        {
            2: track_2,
            3: track_3,
        },
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    assert set(final) == {2, 3}
    assert report.components == ((2,), (3,))
    assert report.canonical_id_by_track == {
        2: 2,
        3: 3,
    }


def test_resolution_report_is_strict_json_safe() -> None:
    track_2 = canonical(
        2,
        predicted_ranges=((20, 40),),
    )
    track_30 = observed_fragment(
        30,
        start_frame=20,
        end_frame=24,
        x_offset=0.15,
    )
    track_34 = observed_fragment(
        34,
        start_frame=26,
        end_frame=40,
        x_offset=0.20,
    )

    _, report = resolve_identities(
        {
            2: track_2,
            30: track_30,
            34: track_34,
        },
        field_length=FIELD_LENGTH,
        field_width=FIELD_WIDTH,
    )

    payload = report.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert '"final_identities": 1' in encoded
    assert '"score_ratio": null' in encoded
