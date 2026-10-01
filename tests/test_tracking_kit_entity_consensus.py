from __future__ import annotations

from tracking.appearance.kit_entity_consensus import (
    KitEntityConsensusConfig,
    aggregate_entity_family_consensus,
)


def _tracklet(
    tracklet_id,
    *,
    state,
    candidate_type=None,
    candidate_id=None,
    dominance=1.0,
    known_fraction=1.0,
    support_samples=3,
    total_samples=3,
    quality_mass=3.0,
):
    return {
        "tracklet_id": tracklet_id,
        "state": state,
        "candidate_family_type": candidate_type,
        "candidate_family_id": candidate_id,
        "dominance": dominance,
        "known_fraction": known_fraction,
        "support_samples": support_samples,
        "total_samples": total_samples,
        "total_quality_mass": quality_mass,
    }


def _map(tracklet_id, entity_id):
    return {
        "tracklet_id": tracklet_id,
        "entity_id": entity_id,
    }


def _run(rows, mapping, config=None):
    return aggregate_entity_family_consensus(
        tracklet_consensus_rows=rows,
        tracklet_entity_rows=mapping,
        entity_id_field="entity_id",
        entity_level="test",
        config=config,
    ).entities[0]


def test_many_same_primary_votes_survive_one_conflicting_tracklet():
    rows = [
        _tracklet(
            f"a{i}",
            state="confident_primary",
            candidate_type="primary",
            candidate_id=0,
        )
        for i in range(5)
    ]
    rows.append(
        _tracklet(
            "bad",
            state="confident_primary",
            candidate_type="primary",
            candidate_id=1,
        )
    )
    mapping = [_map(row["tracklet_id"], "E") for row in rows]

    entity = _run(rows, mapping)

    assert entity.family_key == "primary-0"
    assert entity.usable_for_tracking is True
    assert entity.support_tracklets == 5
    assert entity.conflicting_tracklets == 1
    assert entity.dominance > 0.80


def test_unknown_tracklets_are_not_counted_as_conflicts():
    rows = [
        _tracklet(
            "known",
            state="confident_primary",
            candidate_type="primary",
            candidate_id=0,
            total_samples=5,
            quality_mass=5.0,
        ),
    ] + [
        _tracklet(
            f"u{i}",
            state="unknown",
            candidate_type=None,
            candidate_id=None,
            dominance=0.0,
            known_fraction=0.0,
            support_samples=0,
            total_samples=1,
            quality_mass=1.0,
        )
        for i in range(4)
    ]
    mapping = [_map(row["tracklet_id"], "E") for row in rows]

    entity = _run(
        rows,
        mapping,
        KitEntityConsensusConfig(
            primary_min_confidence=0.20,
        ),
    )

    assert entity.candidate_family_key == "primary-0"
    assert entity.conflicting_tracklets == 0
    assert entity.unknown_tracklets == 4
    assert entity.dominance == 1.0


def test_balanced_known_primary_families_stay_weak():
    rows = []
    for index in range(3):
        rows.append(
            _tracklet(
                f"a{index}",
                state="confident_primary",
                candidate_type="primary",
                candidate_id=0,
            )
        )
        rows.append(
            _tracklet(
                f"b{index}",
                state="confident_primary",
                candidate_type="primary",
                candidate_id=1,
            )
        )
    mapping = [_map(row["tracklet_id"], "E") for row in rows]

    entity = _run(rows, mapping)

    assert entity.family_type == "unknown"
    assert entity.state == "weak_primary"
    assert 0.49 <= entity.dominance <= 0.51


def test_several_weak_primary_tracklets_can_collectively_become_useful():
    rows = [
        _tracklet(
            f"a{i}",
            state="weak_primary",
            candidate_type="primary",
            candidate_id=0,
            dominance=0.95,
            known_fraction=0.95,
            support_samples=2,
            total_samples=2,
            quality_mass=2.0,
        )
        for i in range(5)
    ]
    mapping = [_map(row["tracklet_id"], "E") for row in rows]

    entity = _run(
        rows,
        mapping,
        KitEntityConsensusConfig(
            primary_min_confidence=0.30,
        ),
    )

    assert entity.family_key == "primary-0"
    assert entity.support_tracklets == 5


def test_special_requires_multiple_supporting_tracklets():
    rows = [
        _tracklet(
            "special",
            state="confident_special",
            candidate_type="special",
            candidate_id=0,
            support_samples=6,
            total_samples=6,
            quality_mass=6.0,
        )
    ]
    mapping = [_map("special", "E")]

    entity = _run(rows, mapping)

    assert entity.family_type == "unknown"
    assert entity.state == "weak_special"


def test_missing_consensus_is_unknown_not_conflict():
    rows = [
        _tracklet(
            "known",
            state="confident_primary",
            candidate_type="primary",
            candidate_id=0,
            total_samples=4,
            quality_mass=4.0,
        )
    ]
    mapping = [
        _map("known", "E"),
        _map("missing", "E"),
    ]

    entity = _run(
        rows,
        mapping,
        KitEntityConsensusConfig(
            primary_min_confidence=0.20,
        ),
    )

    assert entity.missing_consensus_tracklets == 1
    assert entity.conflicting_tracklets == 0
    assert entity.candidate_family_key == "primary-0"


def test_entity_family_evidence_never_creates_hard_identity_constraint():
    rows = [
        _tracklet(
            "a",
            state="confident_primary",
            candidate_type="primary",
            candidate_id=0,
        )
    ]
    mapping = [_map("a", "E")]

    result = aggregate_entity_family_consensus(
        tracklet_consensus_rows=rows,
        tracklet_entity_rows=mapping,
        entity_id_field="entity_id",
        entity_level="identity",
    )
    summary = result.summary()

    assert summary["identity_hard_constraints_created"] is False
    assert summary["missing_evidence_is_negative"] is False
