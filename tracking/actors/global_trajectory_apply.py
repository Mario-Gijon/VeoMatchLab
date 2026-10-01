"""Tracking V1 Hito 7G.5: apply visually approved trajectory proposals.

This module changes the identity partition for the first time after 7F.

Safety rules
------------
* only proposals already selected by 7G.4 are accepted;
* proposals must be identity-disjoint;
* every source identity, actor and tracklet must remain represented exactly once;
* hard physical identity conflicts remain inviolable after remapping;
* APPEND keeps the predecessor identity id as the stable anchor;
* INSERT keeps the host identity id as the stable anchor;
* unrelated identities keep their ids unchanged;
* no population target is used.

The stable-anchor rule intentionally avoids renumbering hundreds of unaffected
identities after every trajectory round.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


GLOBAL_TRAJECTORY_APPLY_VERSION = "global-trajectory-apply-v1"


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _proposal_pair(
    row: Mapping[str, Any],
) -> tuple[str, str, str]:
    candidate_type = str(row["candidate_type"])
    if candidate_type == "append":
        return (
            str(row["predecessor_identity_id"]),
            str(row["successor_identity_id"]),
            candidate_type,
        )
    if candidate_type == "insert":
        return (
            str(row["host_identity_id"]),
            str(row["inserted_identity_id"]),
            candidate_type,
        )
    raise ValueError(
        f"unsupported candidate_type {candidate_type!r}"
    )


@dataclass(frozen=True, slots=True)
class GlobalTrajectoryApplyResult:
    identities: tuple[Mapping[str, Any], ...]
    source_identity_assignments: tuple[Mapping[str, Any], ...]
    actor_assignments: tuple[Mapping[str, Any], ...]
    tracklet_assignments: tuple[Mapping[str, Any], ...]
    lineage: tuple[Mapping[str, Any], ...]
    accepted_proposals: tuple[Mapping[str, Any], ...]
    source_identity_count: int
    source_actor_count: int
    source_tracklet_count: int

    def summary(self) -> dict[str, Any]:
        type_counts = Counter(
            str(row["candidate_type"])
            for row in self.accepted_proposals
        )
        family_counts = Counter(
            str(row.get("family_relation", "neutral"))
            for row in self.accepted_proposals
        )
        scores = [
            float(row["trajectory_score"])
            for row in self.accepted_proposals
        ]
        return {
            "schema_version": 1,
            "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
            "source_identity_count": self.source_identity_count,
            "accepted_proposal_count": len(self.accepted_proposals),
            "output_identity_count": len(self.identities),
            "identity_count_reduction": (
                self.source_identity_count - len(self.identities)
            ),
            "source_actor_count": self.source_actor_count,
            "output_actor_assignment_count": len(
                self.actor_assignments
            ),
            "source_tracklet_count": self.source_tracklet_count,
            "output_tracklet_assignment_count": len(
                self.tracklet_assignments
            ),
            "accepted_type_counts": dict(sorted(type_counts.items())),
            "accepted_family_relation_counts": dict(
                sorted(family_counts.items())
            ),
            "accepted_score_min": min(scores) if scores else None,
            "accepted_score_max": max(scores) if scores else None,
            "hard_constraints_inviolable": True,
            "population_forced": False,
            "unaffected_identity_ids_preserved": True,
            "append_anchor_policy": "predecessor_identity_id",
            "insert_anchor_policy": "host_identity_id",
            "identity_partition_changed": bool(
                self.accepted_proposals
            ),
            "provisional": True,
            "canonical": False,
        }


def _flatten_source_ids(row: Mapping[str, Any]) -> tuple[str, ...]:
    raw = row.get("source_identity_ids")
    if isinstance(raw, (list, tuple)) and raw:
        return tuple(str(value) for value in raw)
    return (str(row["identity_id"]),)


def apply_global_trajectory_proposals(
    *,
    identity_rows: Sequence[Mapping[str, Any]],
    actor_assignment_rows: Sequence[Mapping[str, Any]],
    tracklet_assignment_rows: Sequence[Mapping[str, Any]],
    proposal_rows: Sequence[Mapping[str, Any]],
    hard_identity_pairs: set[tuple[str, str]],
) -> GlobalTrajectoryApplyResult:
    source = {
        str(row["identity_id"]): dict(row)
        for row in identity_rows
    }
    if len(source) != len(identity_rows):
        raise ValueError("duplicate source identity ids")

    source_ids = set(source)

    actor_owner: dict[str, str] = {}
    for row in actor_assignment_rows:
        actor_id = str(row["actor_id"])
        identity_id = str(row["identity_id"])
        if actor_id in actor_owner:
            raise ValueError(f"duplicate actor assignment {actor_id}")
        if identity_id not in source:
            raise ValueError(
                f"actor {actor_id} references unknown identity "
                f"{identity_id}"
            )
        actor_owner[actor_id] = identity_id

    tracklet_owner: dict[str, str] = {}
    for row in tracklet_assignment_rows:
        tracklet_id = str(row["tracklet_id"])
        identity_id = str(row["identity_id"])
        if tracklet_id in tracklet_owner:
            raise ValueError(
                f"duplicate tracklet assignment {tracklet_id}"
            )
        if identity_id not in source:
            raise ValueError(
                f"tracklet {tracklet_id} references unknown identity "
                f"{identity_id}"
            )
        tracklet_owner[tracklet_id] = identity_id

    # Cross-check the source identity rows against both assignment artifacts.
    for identity_id, row in source.items():
        for actor_id in row.get("actor_ids", ()):
            if actor_owner.get(str(actor_id)) != identity_id:
                raise ValueError(
                    f"actor assignment disagrees for {actor_id}"
                )
        for tracklet_id in row.get("tracklet_ids", ()):
            if tracklet_owner.get(str(tracklet_id)) != identity_id:
                raise ValueError(
                    f"tracklet assignment disagrees for {tracklet_id}"
                )

    used: set[str] = set()
    anchor_of: dict[str, str] = {
        identity_id: identity_id
        for identity_id in source_ids
    }
    proposal_by_anchor: dict[str, dict[str, Any]] = {}
    partner_by_anchor: dict[str, str] = {}

    normalized_proposals: list[dict[str, Any]] = []
    for raw in proposal_rows:
        row = dict(raw)
        first, second, candidate_type = _proposal_pair(row)

        if first not in source or second not in source:
            raise ValueError(
                f"proposal references unknown identity: {first}, {second}"
            )
        if first == second:
            raise ValueError("proposal cannot merge identity with itself")
        if first in used or second in used:
            raise ValueError(
                "trajectory proposals must be identity-disjoint"
            )
        if _canonical_pair(first, second) in hard_identity_pairs:
            raise ValueError(
                "proposal violates hard identity conflict: "
                f"{first}, {second}"
            )

        # APPEND first is predecessor; INSERT first is host.  Both are stable
        # semantic anchors for the merged identity.
        anchor = first
        partner = second
        anchor_of[partner] = anchor
        used.add(anchor)
        used.add(partner)
        proposal_by_anchor[anchor] = row
        partner_by_anchor[anchor] = partner
        normalized_proposals.append(row)

    groups: dict[str, list[str]] = defaultdict(list)
    for identity_id in sorted(source_ids):
        groups[anchor_of[identity_id]].append(identity_id)

    identities: list[dict[str, Any]] = []
    source_identity_assignments: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []

    for anchor in sorted(
        groups,
        key=lambda identity_id: (
            float(source[identity_id]["start_time_s"]),
            identity_id,
        ),
    ):
        immediate_sources = groups[anchor]
        if len(immediate_sources) not in (1, 2):
            raise AssertionError("unexpected trajectory apply group size")

        if len(immediate_sources) == 2:
            partner = partner_by_anchor[anchor]
            # Preserve semantic proposal order rather than lexicographic order.
            immediate_sources = [anchor, partner]

        source_rows = [source[value] for value in immediate_sources]

        actor_ids: list[str] = []
        tracklet_ids: list[str] = []
        flattened_source_ids: list[str] = []

        for row in source_rows:
            actor_ids.extend(
                str(value) for value in row.get("actor_ids", ())
            )
            tracklet_ids.extend(
                str(value) for value in row.get("tracklet_ids", ())
            )
            flattened_source_ids.extend(_flatten_source_ids(row))

        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError(
                f"duplicate actor inside output identity {anchor}"
            )
        if len(tracklet_ids) != len(set(tracklet_ids)):
            raise ValueError(
                f"duplicate tracklet inside output identity {anchor}"
            )

        proposal = proposal_by_anchor.get(anchor)
        merged = proposal is not None
        candidate_type = (
            str(proposal["candidate_type"])
            if proposal is not None
            else None
        )

        identity_payload = {
            "schema_version": 1,
            "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
            "identity_id": anchor,
            "source_identity_ids": list(
                dict.fromkeys(flattened_source_ids)
            ),
            "source_identity_count": len(
                dict.fromkeys(flattened_source_ids)
            ),
            "trajectory_immediate_source_identity_ids": list(
                immediate_sources
            ),
            "trajectory_immediate_source_identity_count": len(
                immediate_sources
            ),
            "actor_ids": actor_ids,
            "actor_count": len(actor_ids),
            "tracklet_ids": tracklet_ids,
            "tracklet_count": len(tracklet_ids),
            "start_time_s": min(
                float(row["start_time_s"]) for row in source_rows
            ),
            "end_time_s": max(
                float(row["end_time_s"]) for row in source_rows
            ),
            "seeded": any(
                bool(row.get("seeded", False))
                for row in source_rows
            ),
            "trajectory_merged": merged,
            "trajectory_candidate_type": candidate_type,
            "trajectory_score": (
                float(proposal["trajectory_score"])
                if proposal is not None
                else None
            ),
            "provisional": True,
        }
        identity_payload["duration_s"] = (
            identity_payload["end_time_s"]
            - identity_payload["start_time_s"]
        )
        identities.append(identity_payload)

        for source_identity_id in immediate_sources:
            source_identity_assignments.append({
                "schema_version": 1,
                "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
                "source_identity_id": source_identity_id,
                "identity_id": anchor,
                "merged": merged,
                "candidate_type": candidate_type,
                "provisional": True,
            })

        lineage.append({
            "schema_version": 1,
            "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
            "identity_id": anchor,
            "immediate_source_identity_ids": list(
                immediate_sources
            ),
            "flattened_source_identity_ids": list(
                dict.fromkeys(flattened_source_ids)
            ),
            "trajectory_merged": merged,
            "candidate_type": candidate_type,
            "trajectory_score": (
                float(proposal["trajectory_score"])
                if proposal is not None
                else None
            ),
            "family_relation": (
                str(proposal.get("family_relation", "neutral"))
                if proposal is not None
                else None
            ),
        })

    remap = {
        row["source_identity_id"]: row["identity_id"]
        for row in source_identity_assignments
    }
    if set(remap) != source_ids:
        raise AssertionError("source identity remap is incomplete")

    actor_assignments = []
    for row in actor_assignment_rows:
        actor_id = str(row["actor_id"])
        old_identity = str(row["identity_id"])
        actor_assignments.append({
            "schema_version": 1,
            "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
            "actor_id": actor_id,
            "source_identity_id": old_identity,
            "identity_id": remap[old_identity],
            "provisional": True,
        })

    tracklet_assignments = []
    for row in tracklet_assignment_rows:
        tracklet_id = str(row["tracklet_id"])
        old_identity = str(row["identity_id"])
        tracklet_assignments.append({
            "schema_version": 1,
            "application_version": GLOBAL_TRAJECTORY_APPLY_VERSION,
            "tracklet_id": tracklet_id,
            "source_identity_id": old_identity,
            "identity_id": remap[old_identity],
            "provisional": True,
        })

    # Final hard-constraint verification after remapping.
    for first, second in hard_identity_pairs:
        if first not in remap or second not in remap:
            raise ValueError(
                "hard pair references identity outside source partition"
            )
        if remap[first] == remap[second]:
            raise RuntimeError(
                "hard cannot-link survived inside applied trajectory "
                f"identity: {first}, {second}"
            )

    # Global conservation checks.
    output_actor_ids = {
        str(row["actor_id"]) for row in actor_assignments
    }
    output_tracklet_ids = {
        str(row["tracklet_id"]) for row in tracklet_assignments
    }
    if output_actor_ids != set(actor_owner):
        raise RuntimeError("actor conservation failed")
    if output_tracklet_ids != set(tracklet_owner):
        raise RuntimeError("tracklet conservation failed")

    expected_count = len(source) - len(normalized_proposals)
    if len(identities) != expected_count:
        raise RuntimeError(
            "identity count reduction does not match accepted proposals"
        )

    identities.sort(
        key=lambda row: (
            float(row["start_time_s"]),
            str(row["identity_id"]),
        )
    )
    source_identity_assignments.sort(
        key=lambda row: str(row["source_identity_id"])
    )
    actor_assignments.sort(key=lambda row: str(row["actor_id"]))
    tracklet_assignments.sort(
        key=lambda row: str(row["tracklet_id"])
    )
    lineage.sort(key=lambda row: str(row["identity_id"]))

    return GlobalTrajectoryApplyResult(
        identities=tuple(identities),
        source_identity_assignments=tuple(
            source_identity_assignments
        ),
        actor_assignments=tuple(actor_assignments),
        tracklet_assignments=tuple(tracklet_assignments),
        lineage=tuple(lineage),
        accepted_proposals=tuple(normalized_proposals),
        source_identity_count=len(source),
        source_actor_count=len(actor_owner),
        source_tracklet_count=len(tracklet_owner),
    )
