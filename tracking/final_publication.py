"""Final offline publication of canonical football identities.

This module does NOT solve identity.  It consumes the already-audited physical
identities and turns raw tracker rows into one canonical person per frame.

Design rules
------------
* Exactly one published observation per canonical identity and frame.
* Main selected segments define identity membership.
* A separately validated duplicate segment may be attached as SECONDARY
  evidence without creating another canonical identity.
* If main and duplicate sources coexist, only the best source is published.
* OBSERVED / INTERPOLATED / PREDICTED remain internal metadata; the frontend
  may render them identically.
* No endpoint filling and no interpolation is performed here.
* If nobody representing a canonical identity is present in a frame, that
  canonical identity is absent from that frame.  In particular, a long referee
  gap stays a real gap.

The module is intentionally independent from frontend/rendering code so the same
canonical publication can later feed pitch, panorama anchors, metrics, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, Mapping, Sequence


class PublishedPositionKind(str, Enum):
    OBSERVED = "observed"
    INTERPOLATED = "interpolated"
    PREDICTED = "predicted"
    UNKNOWN = "unknown"


_POSITION_RANK = {
    PublishedPositionKind.OBSERVED: 3,
    PublishedPositionKind.INTERPOLATED: 2,
    PublishedPositionKind.PREDICTED: 1,
    PublishedPositionKind.UNKNOWN: 0,
}


@dataclass(frozen=True)
class SegmentWindow:
    key: str
    raw_track_id: int
    start_frame: int
    end_frame: int
    identity_quality: str = "clean"
    identity_quality_score: float = 1.0

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True)
class CanonicalIdentitySpec:
    canonical_id: int
    role: str
    team_label: str
    primary_segment_keys: tuple[str, ...]


@dataclass(frozen=True)
class DuplicateAttachment:
    """A validated secondary source for an existing canonical identity.

    ``source_segment_key`` remains its own raw/segment identity.  It merely
    contributes observations to the canonical identity containing
    ``target_segment_key``.
    """

    source_segment_key: str
    target_segment_key: str
    reason: str = "validated_duplicate"


@dataclass(frozen=True)
class SourceBinding:
    canonical_id: int
    segment_key: str
    raw_track_id: int
    role: str
    team_label: str
    source_kind: str  # "primary" | "duplicate"
    segment_quality: str
    segment_quality_score: float
    start_frame: int
    end_frame: int

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True)
class PublicationPlan:
    canonical_identities: tuple[CanonicalIdentitySpec, ...]
    bindings_by_raw_track: Mapping[int, tuple[SourceBinding, ...]]
    duplicate_attachments: tuple[DuplicateAttachment, ...] = ()

    @property
    def canonical_ids(self) -> tuple[int, ...]:
        return tuple(
            identity.canonical_id
            for identity in self.canonical_identities
        )


@dataclass(frozen=True)
class PublicationCandidate:
    canonical_id: int
    role: str
    team_label: str
    source_segment_key: str
    source_raw_track_id: int
    source_kind: str

    frame_index: int
    timestamp_s: float
    x: float
    z: float

    position_kind: PublishedPositionKind
    confidence: float | None
    measurement_id: str | None
    lifecycle: str | None

    segment_quality: str
    segment_quality_score: float

    # Preserve original raw payload fields useful to later panorama/debug
    # adapters without coupling this module to a particular tracker schema.
    raw_metadata: Mapping[str, object] = field(default_factory=dict)

    def selection_key(self) -> tuple[float, ...]:
        """Higher tuple wins when several sources represent one person.

        Observation state dominates.  A directly selected physical segment is
        preferred to a duplicate when both have the same state.  Confidence
        and segment-quality score break remaining ties.
        """

        confidence = (
            float(self.confidence)
            if self.confidence is not None
            else 0.0
        )
        primary = 1.0 if self.source_kind == "primary" else 0.0

        return (
            float(_POSITION_RANK[self.position_kind]),
            primary,
            confidence,
            float(self.segment_quality_score),
            -float(self.source_raw_track_id),
        )

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "canonical_id": self.canonical_id,
            # Alias kept because existing pitch/debug surfaces historically
            # consume a numeric track identifier.
            "track_id": self.canonical_id,
            "role": self.role,
            "team_label": self.team_label,
            "x": self.x,
            "z": self.z,
            "position_kind": self.position_kind.value,
            "source_segment_key": self.source_segment_key,
            "source_raw_track_id": self.source_raw_track_id,
            "source_kind": self.source_kind,
            "segment_quality": self.segment_quality,
            "segment_quality_score": self.segment_quality_score,
        }
        if self.confidence is not None:
            payload["confidence"] = self.confidence
        if self.measurement_id is not None:
            payload["measurement_id"] = self.measurement_id
        if self.lifecycle is not None:
            payload["lifecycle"] = self.lifecycle
        if self.raw_metadata:
            payload["raw_metadata"] = dict(self.raw_metadata)
        return payload


@dataclass(frozen=True)
class PublishedFrame:
    frame_index: int
    timestamp_s: float
    people: tuple[PublicationCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        # `tracks.person` intentionally mirrors the existing preview-facing
        # shape without touching production tracks.person.
        return {
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "tracks": {
                "person": [
                    person.to_dict()
                    for person in self.people
                ]
            },
        }


def build_publication_plan(
    *,
    identities: Sequence[CanonicalIdentitySpec],
    segments: Mapping[str, SegmentWindow],
    duplicate_attachments: Sequence[DuplicateAttachment] = (),
) -> PublicationPlan:
    """Build and validate raw-segment -> canonical publication bindings."""

    identity_by_segment: dict[str, CanonicalIdentitySpec] = {}
    canonical_ids: set[int] = set()

    for identity in identities:
        if identity.canonical_id in canonical_ids:
            raise ValueError(
                f"duplicate canonical ID {identity.canonical_id}"
            )
        canonical_ids.add(identity.canonical_id)

        if not identity.primary_segment_keys:
            raise ValueError(
                f"canonical {identity.canonical_id} has no primary segments"
            )

        for key in identity.primary_segment_keys:
            if key not in segments:
                raise ValueError(
                    f"canonical {identity.canonical_id} references "
                    f"unknown segment {key}"
                )
            previous = identity_by_segment.get(key)
            if previous is not None:
                raise ValueError(
                    f"primary segment {key} belongs to canonical "
                    f"{previous.canonical_id} and {identity.canonical_id}"
                )
            identity_by_segment[key] = identity

    bindings: dict[int, list[SourceBinding]] = {}

    def add_binding(
        identity: CanonicalIdentitySpec,
        segment_key: str,
        source_kind: str,
    ) -> None:
        segment = segments[segment_key]
        binding = SourceBinding(
            canonical_id=identity.canonical_id,
            segment_key=segment.key,
            raw_track_id=segment.raw_track_id,
            role=identity.role,
            team_label=identity.team_label,
            source_kind=source_kind,
            segment_quality=segment.identity_quality,
            segment_quality_score=segment.identity_quality_score,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
        )
        bindings.setdefault(
            segment.raw_track_id,
            [],
        ).append(binding)

    for key, identity in identity_by_segment.items():
        add_binding(
            identity,
            key,
            "primary",
        )

    seen_duplicate_sources: set[str] = set()

    for attachment in duplicate_attachments:
        source = attachment.source_segment_key
        target = attachment.target_segment_key

        if source in seen_duplicate_sources:
            raise ValueError(
                f"duplicate attachment source repeated: {source}"
            )
        seen_duplicate_sources.add(source)

        if source not in segments:
            raise ValueError(
                f"duplicate attachment source does not exist: {source}"
            )
        if target not in identity_by_segment:
            raise ValueError(
                f"duplicate target is not a selected primary segment: {target}"
            )

        # Never silently steal a segment already selected as another canonical
        # identity.  Such a conflict must be resolved upstream.
        selected_owner = identity_by_segment.get(source)
        target_owner = identity_by_segment[target]
        if (
            selected_owner is not None
            and selected_owner.canonical_id
            != target_owner.canonical_id
        ):
            raise ValueError(
                f"duplicate source {source} is already primary for canonical "
                f"{selected_owner.canonical_id}; target canonical is "
                f"{target_owner.canonical_id}"
            )

        # A contaminated segment is not safe duplicate evidence.
        source_window = segments[source]
        if source_window.identity_quality == "contaminated":
            raise ValueError(
                f"contaminated segment cannot be duplicate evidence: {source}"
            )

        add_binding(
            target_owner,
            source,
            "duplicate",
        )

    frozen_bindings = {
        raw_id: tuple(
            sorted(
                rows,
                key=lambda row: (
                    row.start_frame,
                    row.end_frame,
                    row.canonical_id,
                    row.segment_key,
                    row.source_kind,
                ),
            )
        )
        for raw_id, rows in sorted(bindings.items())
    }

    return PublicationPlan(
        canonical_identities=tuple(
            sorted(
                identities,
                key=lambda identity: identity.canonical_id,
            )
        ),
        bindings_by_raw_track=frozen_bindings,
        duplicate_attachments=tuple(duplicate_attachments),
    )


def publish_frame(
    raw_frame: Mapping[str, object],
    *,
    plan: PublicationPlan,
    max_canonical_people: int = 23,
) -> PublishedFrame:
    """Publish one raw tracker frame into stable canonical identities."""

    frame_index = int(raw_frame["frame_index"])
    timestamp_s = float(
        raw_frame.get(
            "timestamp_s",
            raw_frame.get("timestamp", 0.0),
        )
    )

    candidates_by_canonical: dict[
        int,
        list[PublicationCandidate],
    ] = {}

    for raw_track in raw_frame.get("tracks", []):
        if not isinstance(raw_track, Mapping):
            continue

        raw_track_id_value = raw_track.get(
            "track_id",
            raw_track.get("raw_track_id"),
        )
        if raw_track_id_value is None:
            continue

        raw_track_id = int(raw_track_id_value)

        bindings = plan.bindings_by_raw_track.get(
            raw_track_id,
            (),
        )
        if not bindings:
            # Outsider / unassigned raw tracker ID.
            continue

        for binding in bindings:
            if not binding.contains(frame_index):
                continue

            candidate = _candidate_from_raw_track(
                raw_track,
                binding=binding,
                frame_index=frame_index,
                timestamp_s=timestamp_s,
            )
            if candidate is None:
                continue

            candidates_by_canonical.setdefault(
                binding.canonical_id,
                [],
            ).append(candidate)

    published: list[PublicationCandidate] = []

    for canonical_id, candidates in candidates_by_canonical.items():
        if not candidates:
            continue
        chosen = max(
            candidates,
            key=lambda row: row.selection_key(),
        )
        published.append(chosen)

    published.sort(
        key=lambda row: row.canonical_id
    )

    if len(published) > max_canonical_people:
        raise RuntimeError(
            f"published {len(published)} canonical people in frame "
            f"{frame_index}; maximum is {max_canonical_people}"
        )

    canonical_ids = [
        person.canonical_id
        for person in published
    ]
    if len(canonical_ids) != len(set(canonical_ids)):
        raise RuntimeError(
            f"duplicate canonical ID published in frame {frame_index}"
        )

    return PublishedFrame(
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        people=tuple(published),
    )


def publish_frames(
    raw_frames: Iterable[Mapping[str, object]],
    *,
    plan: PublicationPlan,
    max_canonical_people: int = 23,
) -> tuple[PublishedFrame, ...]:
    return tuple(
        publish_frame(
            raw_frame,
            plan=plan,
            max_canonical_people=max_canonical_people,
        )
        for raw_frame in raw_frames
    )


def _candidate_from_raw_track(
    raw_track: Mapping[str, object],
    *,
    binding: SourceBinding,
    frame_index: int,
    timestamp_s: float,
) -> PublicationCandidate | None:
    if raw_track.get("x") is None or raw_track.get("z") is None:
        return None

    try:
        x = float(raw_track["x"])
        z = float(raw_track["z"])
    except (TypeError, ValueError):
        return None

    raw_kind = str(
        raw_track.get(
            "position_kind",
            "unknown",
        )
    ).lower()
    try:
        position_kind = PublishedPositionKind(
            raw_kind
        )
    except ValueError:
        position_kind = PublishedPositionKind.UNKNOWN

    confidence = raw_track.get("confidence")
    confidence_out = (
        float(confidence)
        if confidence is not None
        else None
    )

    measurement_id = raw_track.get("measurement_id")
    lifecycle = raw_track.get("lifecycle")

    # Keep only small, useful metadata.  Full raw records remain in the source
    # JSONL and need not be duplicated into every preview frame.
    metadata_keys = (
        "hits",
        "missed",
        "age_s",
        "vx",
        "vz",
    )
    raw_metadata = {
        key: raw_track[key]
        for key in metadata_keys
        if key in raw_track
    }

    return PublicationCandidate(
        canonical_id=binding.canonical_id,
        role=binding.role,
        team_label=binding.team_label,
        source_segment_key=binding.segment_key,
        source_raw_track_id=binding.raw_track_id,
        source_kind=binding.source_kind,
        frame_index=frame_index,
        timestamp_s=timestamp_s,
        x=x,
        z=z,
        position_kind=position_kind,
        confidence=confidence_out,
        measurement_id=(
            str(measurement_id)
            if measurement_id is not None
            else None
        ),
        lifecycle=(
            str(lifecycle)
            if lifecycle is not None
            else None
        ),
        segment_quality=binding.segment_quality,
        segment_quality_score=binding.segment_quality_score,
        raw_metadata=raw_metadata,
    )
