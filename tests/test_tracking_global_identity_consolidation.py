from __future__ import annotations

from dataclasses import dataclass

import pytest

from tracking.actors.global_identity_consolidation import (
    GlobalIdentityConsolidationConfig,
    consolidate_until_convergence,
)


@dataclass
class _Candidate:
    predecessor_identity_id: str
    successor_identity_id: str
    rank_score: float = 0.65
    gap_s: float = 1.0
    family_relation: str = "same_primary"

    def to_dict(self):
        return {
            "predecessor_identity_id": self.predecessor_identity_id,
            "successor_identity_id": self.successor_identity_id,
            "rank_score": self.rank_score,
            "gap_s": self.gap_s,
            "family_relation": self.family_relation,
        }


@dataclass
class _Identity:
    identity_id: str
    source_identity_ids: tuple[str, ...]
    actor_ids: tuple[str, ...]
    tracklet_ids: tuple[str, ...]

    def to_dict(self):
        return {
            "identity_id": self.identity_id,
            "source_identity_ids": list(self.source_identity_ids),
            "actor_ids": list(self.actor_ids),
            "tracklet_ids": list(self.tracklet_ids),
        }


@dataclass
class _Expansion:
    profiles: tuple
    candidates: tuple
    selected_candidates: tuple
    expanded_identities: tuple
    actor_assignments: tuple
    tracklet_assignments: tuple


def _initial():
    identities = [
        {
            "identity_id": "I1",
            "actor_ids": ["A1"],
            "tracklet_ids": ["T1"],
        },
        {
            "identity_id": "I2",
            "actor_ids": ["A2"],
            "tracklet_ids": ["T2"],
        },
        {
            "identity_id": "I3",
            "actor_ids": ["A3"],
            "tracklet_ids": ["T3"],
        },
    ]
    actors = [
        {"actor_id": "A1", "identity_id": "I1"},
        {"actor_id": "A2", "identity_id": "I2"},
        {"actor_id": "A3", "identity_id": "I3"},
    ]
    tracklets = [
        {"tracklet_id": "T1", "identity_id": "I1"},
        {"tracklet_id": "T2", "identity_id": "I2"},
        {"tracklet_id": "T3", "identity_id": "I3"},
    ]
    return identities, actors, tracklets


def _family_step(tracklet_assignments):
    ids = sorted({
        row["identity_id"] for row in tracklet_assignments
    })
    return [
        {
            "identity_id": identity_id,
            "usable_for_tracking": True,
        }
        for identity_id in ids
    ]


def test_iterates_then_stops_on_zero_selected_links():
    identities, actors, tracklets = _initial()

    def expansion_step(
        identities,
        actor_assignments,
        tracklet_assignments,
        family_rows,
        round_index,
    ):
        if round_index == 1:
            return _Expansion(
                profiles=(1, 2, 3),
                candidates=(1,),
                selected_candidates=(
                    _Candidate("I1", "I2"),
                ),
                expanded_identities=(
                    _Identity(
                        "P1",
                        ("I1", "I2"),
                        ("A1", "A2"),
                        ("T1", "T2"),
                    ),
                    _Identity(
                        "P2",
                        ("I3",),
                        ("A3",),
                        ("T3",),
                    ),
                ),
                actor_assignments=(
                    {"actor_id": "A1", "identity_id": "P1"},
                    {"actor_id": "A2", "identity_id": "P1"},
                    {"actor_id": "A3", "identity_id": "P2"},
                ),
                tracklet_assignments=(
                    {"tracklet_id": "T1", "identity_id": "P1"},
                    {"tracklet_id": "T2", "identity_id": "P1"},
                    {"tracklet_id": "T3", "identity_id": "P2"},
                ),
            )

        return _Expansion(
            profiles=(1, 2),
            candidates=(),
            selected_candidates=(),
            expanded_identities=(
                _Identity(
                    "Q1",
                    ("P1",),
                    ("A1", "A2"),
                    ("T1", "T2"),
                ),
                _Identity(
                    "Q2",
                    ("P2",),
                    ("A3",),
                    ("T3",),
                ),
            ),
            actor_assignments=(
                {"actor_id": "A1", "identity_id": "Q1"},
                {"actor_id": "A2", "identity_id": "Q1"},
                {"actor_id": "A3", "identity_id": "Q2"},
            ),
            tracklet_assignments=(
                {"tracklet_id": "T1", "identity_id": "Q1"},
                {"tracklet_id": "T2", "identity_id": "Q1"},
                {"tracklet_id": "T3", "identity_id": "Q2"},
            ),
        )

    result = consolidate_until_convergence(
        initial_identities=identities,
        initial_actor_assignments=actors,
        initial_tracklet_assignments=tracklets,
        family_step=_family_step,
        expansion_step=expansion_step,
    )

    assert result.converged is True
    assert result.stopped_reason == "no_selected_links"
    assert len(result.rounds) == 2
    # No-op convergence round must not renumber the accepted state.
    assert {row["identity_id"] for row in result.final_identities} == {
        "P1", "P2"
    }


def test_lineage_is_composed_across_multiple_rounds():
    identities, actors, tracklets = _initial()

    def expansion_step(
        identities,
        actor_assignments,
        tracklet_assignments,
        family_rows,
        round_index,
    ):
        if round_index == 1:
            return _Expansion(
                profiles=(1, 2, 3),
                candidates=(1,),
                selected_candidates=(_Candidate("I1", "I2"),),
                expanded_identities=(
                    _Identity(
                        "P1", ("I1", "I2"),
                        ("A1", "A2"), ("T1", "T2")
                    ),
                    _Identity(
                        "P2", ("I3",),
                        ("A3",), ("T3",)
                    ),
                ),
                actor_assignments=(
                    {"actor_id": "A1", "identity_id": "P1"},
                    {"actor_id": "A2", "identity_id": "P1"},
                    {"actor_id": "A3", "identity_id": "P2"},
                ),
                tracklet_assignments=(
                    {"tracklet_id": "T1", "identity_id": "P1"},
                    {"tracklet_id": "T2", "identity_id": "P1"},
                    {"tracklet_id": "T3", "identity_id": "P2"},
                ),
            )
        if round_index == 2:
            return _Expansion(
                profiles=(1, 2),
                candidates=(1,),
                selected_candidates=(_Candidate("P1", "P2"),),
                expanded_identities=(
                    _Identity(
                        "Q1", ("P1", "P2"),
                        ("A1", "A2", "A3"),
                        ("T1", "T2", "T3")
                    ),
                ),
                actor_assignments=(
                    {"actor_id": "A1", "identity_id": "Q1"},
                    {"actor_id": "A2", "identity_id": "Q1"},
                    {"actor_id": "A3", "identity_id": "Q1"},
                ),
                tracklet_assignments=(
                    {"tracklet_id": "T1", "identity_id": "Q1"},
                    {"tracklet_id": "T2", "identity_id": "Q1"},
                    {"tracklet_id": "T3", "identity_id": "Q1"},
                ),
            )
        return _Expansion(
            profiles=(1,),
            candidates=(),
            selected_candidates=(),
            expanded_identities=(
                _Identity(
                    "R1", ("Q1",),
                    ("A1", "A2", "A3"),
                    ("T1", "T2", "T3")
                ),
            ),
            actor_assignments=(
                {"actor_id": "A1", "identity_id": "R1"},
                {"actor_id": "A2", "identity_id": "R1"},
                {"actor_id": "A3", "identity_id": "R1"},
            ),
            tracklet_assignments=(
                {"tracklet_id": "T1", "identity_id": "R1"},
                {"tracklet_id": "T2", "identity_id": "R1"},
                {"tracklet_id": "T3", "identity_id": "R1"},
            ),
        )

    result = consolidate_until_convergence(
        initial_identities=identities,
        initial_actor_assignments=actors,
        initial_tracklet_assignments=tracklets,
        family_step=_family_step,
        expansion_step=expansion_step,
    )

    assert len(result.final_identities) == 1
    assert result.lineage_rows[0]["initial_identity_ids"] == [
        "I1", "I2", "I3"
    ]


def test_selected_link_must_reduce_identity_count():
    identities, actors, tracklets = _initial()

    def broken_step(*args):
        return _Expansion(
            profiles=(1, 2, 3),
            candidates=(1,),
            selected_candidates=(_Candidate("I1", "I2"),),
            expanded_identities=tuple(
                _Identity(
                    row["identity_id"],
                    (row["identity_id"],),
                    tuple(row["actor_ids"]),
                    tuple(row["tracklet_ids"]),
                )
                for row in identities
            ),
            actor_assignments=tuple(actors),
            tracklet_assignments=tuple(tracklets),
        )

    with pytest.raises(RuntimeError):
        consolidate_until_convergence(
            initial_identities=identities,
            initial_actor_assignments=actors,
            initial_tracklet_assignments=tracklets,
            family_step=_family_step,
            expansion_step=broken_step,
        )


def test_duplicate_tracklet_assignment_is_rejected():
    identities, actors, tracklets = _initial()
    bad = list(tracklets) + [
        {"tracklet_id": "T1", "identity_id": "I1"}
    ]

    with pytest.raises(ValueError):
        consolidate_until_convergence(
            initial_identities=identities,
            initial_actor_assignments=actors,
            initial_tracklet_assignments=bad,
            family_step=_family_step,
            expansion_step=lambda *args: None,
        )


def test_max_rounds_stops_without_forcing_population():
    identities, actors, tracklets = _initial()

    def expansion_step(
        identities,
        actor_assignments,
        tracklet_assignments,
        family_rows,
        round_index,
    ):
        # Round 1 merges I1/I2. Round 2 would merge again, but max_rounds=1.
        return _Expansion(
            profiles=(1, 2, 3),
            candidates=(1,),
            selected_candidates=(_Candidate("I1", "I2"),),
            expanded_identities=(
                _Identity(
                    "P1", ("I1", "I2"),
                    ("A1", "A2"), ("T1", "T2")
                ),
                _Identity(
                    "P2", ("I3",),
                    ("A3",), ("T3",)
                ),
            ),
            actor_assignments=(
                {"actor_id": "A1", "identity_id": "P1"},
                {"actor_id": "A2", "identity_id": "P1"},
                {"actor_id": "A3", "identity_id": "P2"},
            ),
            tracklet_assignments=(
                {"tracklet_id": "T1", "identity_id": "P1"},
                {"tracklet_id": "T2", "identity_id": "P1"},
                {"tracklet_id": "T3", "identity_id": "P2"},
            ),
        )

    result = consolidate_until_convergence(
        initial_identities=identities,
        initial_actor_assignments=actors,
        initial_tracklet_assignments=tracklets,
        family_step=_family_step,
        expansion_step=expansion_step,
        config=GlobalIdentityConsolidationConfig(max_rounds=1),
    )

    assert result.converged is False
    assert result.stopped_reason == "max_rounds"
    assert len(result.final_identities) == 2
    assert result.summary()["population_forced"] is False


def test_summary_marks_output_as_canonical_candidate_not_canonical():
    identities, actors, tracklets = _initial()

    def no_op_step(
        identities,
        actor_assignments,
        tracklet_assignments,
        family_rows,
        round_index,
    ):
        return _Expansion(
            profiles=(1, 2, 3),
            candidates=(),
            selected_candidates=(),
            expanded_identities=tuple(
                _Identity(
                    row["identity_id"],
                    (row["identity_id"],),
                    tuple(row["actor_ids"]),
                    tuple(row["tracklet_ids"]),
                )
                for row in identities
            ),
            actor_assignments=tuple(actors),
            tracklet_assignments=tuple(tracklets),
        )

    result = consolidate_until_convergence(
        initial_identities=identities,
        initial_actor_assignments=actors,
        initial_tracklet_assignments=tracklets,
        family_step=_family_step,
        expansion_step=no_op_step,
    )
    summary = result.summary()

    assert summary["canonical_candidate"] is True
    assert summary["canonical"] is False
    assert summary["threshold_relaxed_between_rounds"] is False


def test_version_marks_family_guard_revision():
    from tracking.actors.global_identity_consolidation import (
        GLOBAL_IDENTITY_CONSOLIDATION_VERSION,
    )
    assert GLOBAL_IDENTITY_CONSOLIDATION_VERSION == (
        "global-identity-consolidation-v2-family-guard"
    )
