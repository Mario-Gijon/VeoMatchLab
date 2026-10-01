"""Tracking V1 Hito 7G.6: refresh evidence for the current trajectory identity state.

After an audited 7G merge round changes the identity partition, all evidence
that is aggregated at identity level must be rebuilt before another global
trajectory round:

* aggregate ReID profile from the stable actor-chain profiles;
* kit-family consensus from the stable per-tracklet family evidence.

This stage does NOT make any new identity decision.  It only refreshes evidence
for the supplied identity partition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from tracking.actors.identity_expansion import (
    IdentityExpansionConfig,
    IdentityProfile,
    build_identity_profiles,
)
from tracking.appearance.kit_entity_consensus import (
    EntityFamilyConsensus,
    KitEntityConsensusConfig,
    aggregate_entity_family_consensus,
)


GLOBAL_TRAJECTORY_ROUND_STATE_VERSION = "global-trajectory-round-state-v1"


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryRoundState:
    profiles: tuple[IdentityProfile, ...]
    family_rows: tuple[EntityFamilyConsensus, ...]
    identity_count: int
    tracklet_count: int

    def summary(self) -> dict[str, Any]:
        profiled_ids = {row.identity_id for row in self.profiles}
        family_by_state: dict[str, int] = {}
        family_by_key: dict[str, int] = {}

        for row in self.family_rows:
            family_by_state[row.state] = (
                family_by_state.get(row.state, 0) + 1
            )
            key = row.family_key or "unknown"
            family_by_key[key] = family_by_key.get(key, 0) + 1

        return {
            "schema_version": 1,
            "round_state_version": GLOBAL_TRAJECTORY_ROUND_STATE_VERSION,
            "identity_count": self.identity_count,
            "profiled_identity_count": len(self.profiles),
            "unprofiled_identity_count": (
                self.identity_count - len(profiled_ids)
            ),
            "tracklet_count": self.tracklet_count,
            "family_entity_count": len(self.family_rows),
            "family_usable_count": sum(
                row.usable_for_tracking
                for row in self.family_rows
            ),
            "family_state_counts": dict(sorted(family_by_state.items())),
            "family_counts": dict(sorted(family_by_key.items())),
            "identity_decisions_changed": False,
            "population_forced": False,
        }


def _validate_identity_partition(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    tracklet_assignment_rows: Sequence[Mapping[str, Any]],
) -> tuple[set[str], set[str]]:
    identity_ids: set[str] = set()
    expected_tracklets: dict[str, str] = {}

    for row in identity_rows:
        identity_id = str(row["identity_id"])
        if identity_id in identity_ids:
            raise ValueError(f"duplicate identity {identity_id}")
        identity_ids.add(identity_id)

        tracklet_ids = [
            str(value) for value in row.get("tracklet_ids", ())
        ]
        if not tracklet_ids:
            raise ValueError(
                f"identity {identity_id} has no tracklets"
            )

        for tracklet_id in tracklet_ids:
            previous = expected_tracklets.get(tracklet_id)
            if previous is not None:
                raise ValueError(
                    f"tracklet {tracklet_id} appears in both "
                    f"{previous} and {identity_id}"
                )
            expected_tracklets[tracklet_id] = identity_id

    assignment_owner: dict[str, str] = {}
    for row in tracklet_assignment_rows:
        tracklet_id = str(row["tracklet_id"])
        identity_id = str(row["identity_id"])

        if identity_id not in identity_ids:
            raise ValueError(
                f"tracklet assignment references unknown identity "
                f"{identity_id}"
            )
        if tracklet_id in assignment_owner:
            raise ValueError(
                f"duplicate tracklet assignment {tracklet_id}"
            )
        assignment_owner[tracklet_id] = identity_id

    if set(expected_tracklets) != set(assignment_owner):
        missing = sorted(set(expected_tracklets) - set(assignment_owner))
        extra = sorted(set(assignment_owner) - set(expected_tracklets))
        raise ValueError(
            "identity/tracklet assignment coverage mismatch "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    for tracklet_id, expected_identity in expected_tracklets.items():
        actual_identity = assignment_owner[tracklet_id]
        if actual_identity != expected_identity:
            raise ValueError(
                f"tracklet {tracklet_id} identity mismatch: "
                f"{expected_identity} != {actual_identity}"
            )

    return identity_ids, set(expected_tracklets)


def build_global_trajectory_round_state(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    tracklet_assignment_rows: Sequence[Mapping[str, Any]],
    chain_profile_rows: Sequence[Mapping[str, Any]],
    tracklet_family_rows: Sequence[Mapping[str, Any]],
    profile_config: IdentityExpansionConfig | None = None,
    family_config: KitEntityConsensusConfig | None = None,
) -> GlobalTrajectoryRoundState:
    identity_ids, tracklet_ids = _validate_identity_partition(
        identity_rows=identity_rows,
        tracklet_assignment_rows=tracklet_assignment_rows,
    )

    selected_profile_config = (
        profile_config or IdentityExpansionConfig()
    )
    selected_family_config = (
        family_config or KitEntityConsensusConfig()
    )

    profiles = build_identity_profiles(
        identity_rows=identity_rows,
        chain_profile_rows=chain_profile_rows,
        config=selected_profile_config,
    )
    profile_ids = {row.identity_id for row in profiles}
    if not profile_ids.issubset(identity_ids):
        raise AssertionError(
            "profile builder returned identity outside source partition"
        )

    family_result = aggregate_entity_family_consensus(
        tracklet_consensus_rows=tracklet_family_rows,
        tracklet_entity_rows=tracklet_assignment_rows,
        entity_id_field="identity_id",
        entity_level="global_trajectory_identity",
        config=selected_family_config,
    )
    family_rows = family_result.entities
    family_ids = {row.entity_id for row in family_rows}

    if family_ids != identity_ids:
        missing = sorted(identity_ids - family_ids)
        extra = sorted(family_ids - identity_ids)
        raise ValueError(
            "family consensus does not cover current identity partition "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

    return GlobalTrajectoryRoundState(
        profiles=profiles,
        family_rows=family_rows,
        identity_count=len(identity_ids),
        tracklet_count=len(tracklet_ids),
    )
