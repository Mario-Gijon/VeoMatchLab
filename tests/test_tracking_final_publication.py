from tracking_v0.final_publication import (
    CanonicalIdentitySpec,
    DuplicateAttachment,
    SegmentWindow,
    build_publication_plan,
    publish_frame,
    publish_frames,
)


def frame(frame_index, tracks):
    return {
        "frame_index": frame_index,
        "timestamp_s": frame_index / 10.0,
        "tracks": tracks,
    }


def raw(
    raw_id,
    *,
    x,
    z=0.0,
    kind="observed",
    confidence=0.9,
):
    return {
        "track_id": raw_id,
        "x": x,
        "z": z,
        "position_kind": kind,
        "confidence": confidence,
    }


def base_segments():
    return {
        "11:1": SegmentWindow(
            key="11:1",
            raw_track_id=11,
            start_frame=100,
            end_frame=900,
            identity_quality="clean",
            identity_quality_score=0.92,
        ),
        "14:0": SegmentWindow(
            key="14:0",
            raw_track_id=14,
            start_frame=0,
            end_frame=302,
            identity_quality="clean",
            identity_quality_score=0.94,
        ),
        "27:0": SegmentWindow(
            key="27:0",
            raw_track_id=27,
            start_frame=0,
            end_frame=416,
            identity_quality="clean",
            identity_quality_score=0.87,
        ),
        "35:1": SegmentWindow(
            key="35:1",
            raw_track_id=35,
            start_frame=702,
            end_frame=900,
            identity_quality="clean",
            identity_quality_score=0.92,
        ),
    }


def test_two_sources_same_person_publish_exactly_one_observation():
    segments = base_segments()
    identities = [
        CanonicalIdentitySpec(
            canonical_id=7,
            role="player",
            team_label="team_b",
            primary_segment_keys=("11:1",),
        )
    ]
    plan = build_publication_plan(
        identities=identities,
        segments=segments,
        duplicate_attachments=(
            DuplicateAttachment(
                source_segment_key="14:0",
                target_segment_key="11:1",
            ),
        ),
    )

    published = publish_frame(
        frame(
            200,
            [
                raw(11, x=10.0, confidence=0.85),
                raw(14, x=10.2, confidence=0.99),
            ],
        ),
        plan=plan,
    )

    assert len(published.people) == 1
    person = published.people[0]
    assert person.canonical_id == 7
    # Both are OBSERVED, so primary wins rather than duplicate raw 14.
    assert person.source_segment_key == "11:1"
    assert person.source_kind == "primary"


def test_duplicate_evidence_fills_frame_missing_from_primary():
    segments = base_segments()
    plan = build_publication_plan(
        identities=(
            CanonicalIdentitySpec(
                canonical_id=7,
                role="player",
                team_label="team_b",
                primary_segment_keys=("11:1",),
            ),
        ),
        segments=segments,
        duplicate_attachments=(
            DuplicateAttachment("14:0", "11:1"),
        ),
    )

    published = publish_frame(
        frame(
            250,
            [
                raw(14, x=15.0, confidence=0.91),
            ],
        ),
        plan=plan,
    )

    assert len(published.people) == 1
    assert published.people[0].canonical_id == 7
    assert published.people[0].source_segment_key == "14:0"
    assert published.people[0].source_kind == "duplicate"


def test_observed_duplicate_beats_predicted_primary():
    segments = base_segments()
    plan = build_publication_plan(
        identities=(
            CanonicalIdentitySpec(
                canonical_id=7,
                role="player",
                team_label="team_b",
                primary_segment_keys=("11:1",),
            ),
        ),
        segments=segments,
        duplicate_attachments=(
            DuplicateAttachment("14:0", "11:1"),
        ),
    )

    published = publish_frame(
        frame(
            200,
            [
                raw(11, x=10.0, kind="predicted", confidence=0.99),
                raw(14, x=10.4, kind="observed", confidence=0.80),
            ],
        ),
        plan=plan,
    )

    assert published.people[0].source_segment_key == "14:0"
    assert published.people[0].position_kind.value == "observed"


def test_referee_long_gap_is_not_filled():
    segments = base_segments()
    plan = build_publication_plan(
        identities=(
            CanonicalIdentitySpec(
                canonical_id=21,
                role="referee",
                team_label="unknown",
                primary_segment_keys=("27:0", "35:1"),
            ),
        ),
        segments=segments,
    )

    frames = publish_frames(
        [
            frame(400, [raw(27, x=0.0)]),
            # 500 is between the two referee segments. Raw tracker may contain
            # other people, but publisher must not invent referee presence.
            frame(500, [raw(11, x=10.0), raw(14, x=12.0)]),
            frame(800, [raw(35, x=2.0)]),
        ],
        plan=plan,
    )

    assert [len(row.people) for row in frames] == [1, 0, 1]
    assert frames[0].people[0].canonical_id == 21
    assert frames[2].people[0].canonical_id == 21


def test_unassigned_outsider_is_never_published():
    segments = base_segments()
    plan = build_publication_plan(
        identities=(
            CanonicalIdentitySpec(
                canonical_id=1,
                role="player",
                team_label="team_a",
                primary_segment_keys=("11:1",),
            ),
        ),
        segments=segments,
    )

    published = publish_frame(
        frame(
            200,
            [
                raw(11, x=1.0),
                raw(24, x=50.0),  # outsider / coach / unassigned raw
            ],
        ),
        plan=plan,
    )

    assert [person.source_raw_track_id for person in published.people] == [11]


def test_no_more_than_23_canonical_ids_can_be_published():
    segments = {}
    identities = []

    for i in range(1, 24):
        key = f"{i}:0"
        segments[key] = SegmentWindow(
            key=key,
            raw_track_id=i,
            start_frame=0,
            end_frame=100,
        )
        identities.append(
            CanonicalIdentitySpec(
                canonical_id=i,
                role=("referee" if i == 23 else "player"),
                team_label="unknown",
                primary_segment_keys=(key,),
            )
        )

    plan = build_publication_plan(
        identities=identities,
        segments=segments,
    )

    published = publish_frame(
        frame(
            10,
            [
                raw(i, x=float(i))
                for i in range(1, 30)
            ],
        ),
        plan=plan,
    )

    assert len(published.people) == 23
    assert len(
        {
            person.canonical_id
            for person in published.people
        }
    ) == 23


def test_canonical_id_remains_stable_when_raw_source_changes():
    segments = base_segments()
    plan = build_publication_plan(
        identities=(
            CanonicalIdentitySpec(
                canonical_id=7,
                role="player",
                team_label="team_b",
                primary_segment_keys=("11:1",),
            ),
        ),
        segments=segments,
        duplicate_attachments=(
            DuplicateAttachment("14:0", "11:1"),
        ),
    )

    rows = publish_frames(
        [
            frame(200, [raw(14, x=1.0)]),
            frame(220, [raw(11, x=1.2)]),
        ],
        plan=plan,
    )

    assert rows[0].people[0].source_raw_track_id == 14
    assert rows[1].people[0].source_raw_track_id == 11
    assert rows[0].people[0].canonical_id == 7
    assert rows[1].people[0].canonical_id == 7


def test_contaminated_duplicate_attachment_is_rejected():
    segments = base_segments()
    segments["99:0"] = SegmentWindow(
        key="99:0",
        raw_track_id=99,
        start_frame=0,
        end_frame=100,
        identity_quality="contaminated",
        identity_quality_score=0.5,
    )

    try:
        build_publication_plan(
            identities=(
                CanonicalIdentitySpec(
                    canonical_id=7,
                    role="player",
                    team_label="team_b",
                    primary_segment_keys=("11:1",),
                ),
            ),
            segments=segments,
            duplicate_attachments=(
                DuplicateAttachment("99:0", "11:1"),
            ),
        )
    except ValueError as exc:
        assert "contaminated segment" in str(exc)
    else:
        raise AssertionError("contaminated duplicate attachment was accepted")
