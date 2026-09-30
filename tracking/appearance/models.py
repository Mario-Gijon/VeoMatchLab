"""Appearance contracts for the clean Tracking V1 pipeline.

The appearance stage describes *visual evidence about tracklets*.  It does not
assign physical-player identities, teams, shirt numbers or temporal links
between tracklets.  Those decisions belong to later stages.

The contract is deliberately model-agnostic:

    AppearanceDescriptor
        one named numeric descriptor produced by one model/algorithm

    AppearanceSample
        visual evidence extracted from one detection belonging to one tracklet

    TrackletAppearance
        all usable samples for one tracklet plus optional aggregate descriptors

    AppearanceSet
        clip-level collection of tracklet appearance profiles

Examples of descriptor names that may exist later are ``reid``,
``jersey_hsv_hist`` or ``number_logits``.  This module does not prescribe
which ones must be present.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence


DescriptorKey = tuple[str, str]


def _non_empty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_non_empty_string(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _non_empty_string(name, value)


def _non_negative_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _finite_float(name: str, value: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{name} must be finite")
    return parsed


def _unit_interval(name: str, value: float) -> float:
    parsed = _finite_float(name, value)
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return parsed


def _vector(name: str, value: Sequence[float]) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a numeric sequence")
    if len(value) == 0:
        raise ValueError(f"{name} must not be empty")
    return tuple(
        _finite_float(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    )


def _json_safe_dict(
    name: str,
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    result = dict(value or {})
    try:
        json.dumps(
            result,
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain JSON-serializable values"
        ) from exc
    return result


def _optional_relative_path(
    name: str,
    value: str | None,
) -> str | None:
    if value is None:
        return None

    normalized = _non_empty_string(name, value).replace("\\", "/")
    path = PurePosixPath(normalized)

    if path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ValueError(
            f"{name} must stay relative to the Tracking V1 artifact root"
        )

    return path.as_posix()


def _descriptor_key(
    descriptor: "AppearanceDescriptor",
) -> DescriptorKey:
    return descriptor.name, descriptor.model_id


def _normalize_descriptors(
    descriptors: Sequence[
        "AppearanceDescriptor | Mapping[str, Any]"
    ],
    *,
    owner: str,
) -> tuple["AppearanceDescriptor", ...]:
    normalized: list[AppearanceDescriptor] = []
    keys: set[DescriptorKey] = set()

    for raw in descriptors:
        descriptor = (
            raw
            if isinstance(raw, AppearanceDescriptor)
            else AppearanceDescriptor.from_dict(raw)
        )
        key = _descriptor_key(descriptor)
        if key in keys:
            raise ValueError(
                f"duplicate descriptor {key!r} in {owner}"
            )
        keys.add(key)
        normalized.append(descriptor)

    return tuple(normalized)


@dataclass(slots=True)
class AppearanceDescriptor:
    """One numeric visual descriptor produced by a named model/algorithm."""

    name: str
    model_id: str
    values: tuple[float, ...]
    metric: str = "cosine"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = _non_empty_string("name", self.name)
        self.model_id = _non_empty_string("model_id", self.model_id)
        self.metric = _non_empty_string("metric", self.metric)
        self.values = _vector("values", self.values)
        self.metadata = _json_safe_dict("metadata", self.metadata)

    @property
    def key(self) -> DescriptorKey:
        return _descriptor_key(self)

    @property
    def dimension(self) -> int:
        return len(self.values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "model_id": self.model_id,
            "values": list(self.values),
            "metric": self.metric,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "AppearanceDescriptor":
        return cls(
            name=payload["name"],
            model_id=payload["model_id"],
            values=payload["values"],
            metric=payload.get("metric", "cosine"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class AppearanceSample:
    """Usable visual evidence extracted from one tracklet observation.

    Multiple samples may point to the same ``detection_id`` when several source
    or virtual views provide useful crops.  ``sample_id`` is the unique unit.
    """

    sample_id: str
    tracklet_id: str
    detection_id: str
    frame_index: int
    timestamp_s: float
    quality: float
    descriptors: tuple[AppearanceDescriptor, ...]
    view_name: str | None = None
    source_lens: str | None = None
    crop_ref: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.sample_id = _non_empty_string("sample_id", self.sample_id)
        self.tracklet_id = _non_empty_string(
            "tracklet_id",
            self.tracklet_id,
        )
        self.detection_id = _non_empty_string(
            "detection_id",
            self.detection_id,
        )
        self.frame_index = _non_negative_int(
            "frame_index",
            self.frame_index,
        )
        self.timestamp_s = _finite_float(
            "timestamp_s",
            self.timestamp_s,
        )
        if self.timestamp_s < 0.0:
            raise ValueError("timestamp_s must be >= 0")

        self.quality = _unit_interval("quality", self.quality)

        self.view_name = _optional_non_empty_string(
            "view_name",
            self.view_name,
        )

        if self.source_lens not in (None, "left", "right"):
            raise ValueError(
                "source_lens must be 'left', 'right', or None"
            )

        self.crop_ref = _optional_relative_path(
            "crop_ref",
            self.crop_ref,
        )

        self.descriptors = _normalize_descriptors(
            self.descriptors,
            owner=f"sample {self.sample_id!r}",
        )
        if not self.descriptors:
            raise ValueError(
                "AppearanceSample must contain at least one descriptor"
            )

        self.metadata = _json_safe_dict("metadata", self.metadata)

    def descriptor(
        self,
        name: str,
        model_id: str,
    ) -> AppearanceDescriptor | None:
        key = (
            _non_empty_string("name", name),
            _non_empty_string("model_id", model_id),
        )
        for descriptor in self.descriptors:
            if descriptor.key == key:
                return descriptor
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "tracklet_id": self.tracklet_id,
            "detection_id": self.detection_id,
            "frame_index": self.frame_index,
            "timestamp_s": self.timestamp_s,
            "quality": self.quality,
            "descriptors": [
                descriptor.to_dict()
                for descriptor in self.descriptors
            ],
            "view_name": self.view_name,
            "source_lens": self.source_lens,
            "crop_ref": self.crop_ref,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "AppearanceSample":
        raw_descriptors = payload.get("descriptors", ())
        if not isinstance(raw_descriptors, (list, tuple)):
            raise ValueError("descriptors must be a list or tuple")

        return cls(
            sample_id=payload["sample_id"],
            tracklet_id=payload["tracklet_id"],
            detection_id=payload["detection_id"],
            frame_index=payload["frame_index"],
            timestamp_s=payload["timestamp_s"],
            quality=payload["quality"],
            descriptors=tuple(
                item
                if isinstance(item, AppearanceDescriptor)
                else AppearanceDescriptor.from_dict(item)
                for item in raw_descriptors
            ),
            view_name=payload.get("view_name"),
            source_lens=payload.get("source_lens"),
            crop_ref=payload.get("crop_ref"),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class TrackletAppearance:
    """All appearance evidence available for one short temporal tracklet.

    An empty profile is valid.  It explicitly means that the tracklet exists
    but no usable appearance sample could be extracted.  This is preferable to
    silently omitting that tracklet from the appearance artifact.
    """

    tracklet_id: str
    samples: tuple[AppearanceSample, ...] = ()
    aggregate_descriptors: tuple[AppearanceDescriptor, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.tracklet_id = _non_empty_string(
            "tracklet_id",
            self.tracklet_id,
        )

        normalized_samples: list[AppearanceSample] = []
        sample_ids: set[str] = set()

        for raw in self.samples:
            sample = (
                raw
                if isinstance(raw, AppearanceSample)
                else AppearanceSample.from_dict(raw)
            )
            if sample.tracklet_id != self.tracklet_id:
                raise ValueError(
                    f"sample {sample.sample_id!r} belongs to "
                    f"{sample.tracklet_id!r}, expected {self.tracklet_id!r}"
                )
            if sample.sample_id in sample_ids:
                raise ValueError(
                    f"duplicate sample_id in tracklet appearance: "
                    f"{sample.sample_id!r}"
                )
            sample_ids.add(sample.sample_id)
            normalized_samples.append(sample)

        # Appearance has no temporal linking semantics, but deterministic order
        # makes artifacts reproducible and easier to diff.
        normalized_samples.sort(
            key=lambda item: (
                item.frame_index,
                item.timestamp_s,
                item.sample_id,
            )
        )
        self.samples = tuple(normalized_samples)

        self.aggregate_descriptors = _normalize_descriptors(
            self.aggregate_descriptors,
            owner=f"tracklet {self.tracklet_id!r} aggregate",
        )

        dimensions: dict[DescriptorKey, int] = {}
        for sample in self.samples:
            for descriptor in sample.descriptors:
                previous = dimensions.setdefault(
                    descriptor.key,
                    descriptor.dimension,
                )
                if previous != descriptor.dimension:
                    raise ValueError(
                        "descriptor dimension changed inside tracklet "
                        f"{self.tracklet_id!r} for {descriptor.key!r}: "
                        f"{previous} vs {descriptor.dimension}"
                    )

        for descriptor in self.aggregate_descriptors:
            sample_dimension = dimensions.get(descriptor.key)
            if (
                sample_dimension is not None
                and sample_dimension != descriptor.dimension
            ):
                raise ValueError(
                    "aggregate descriptor dimension does not match samples "
                    f"for {descriptor.key!r}: "
                    f"{descriptor.dimension} vs {sample_dimension}"
                )

        self.metadata = _json_safe_dict("metadata", self.metadata)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def aggregate_descriptor_count(self) -> int:
        return len(self.aggregate_descriptors)

    @property
    def has_evidence(self) -> bool:
        return bool(self.samples or self.aggregate_descriptors)

    def aggregate_descriptor(
        self,
        name: str,
        model_id: str,
    ) -> AppearanceDescriptor | None:
        key = (
            _non_empty_string("name", name),
            _non_empty_string("model_id", model_id),
        )
        for descriptor in self.aggregate_descriptors:
            if descriptor.key == key:
                return descriptor
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tracklet_id": self.tracklet_id,
            "samples": [
                sample.to_dict()
                for sample in self.samples
            ],
            "aggregate_descriptors": [
                descriptor.to_dict()
                for descriptor in self.aggregate_descriptors
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "TrackletAppearance":
        raw_samples = payload.get("samples", ())
        raw_aggregates = payload.get("aggregate_descriptors", ())

        if not isinstance(raw_samples, (list, tuple)):
            raise ValueError("samples must be a list or tuple")
        if not isinstance(raw_aggregates, (list, tuple)):
            raise ValueError(
                "aggregate_descriptors must be a list or tuple"
            )

        return cls(
            tracklet_id=payload["tracklet_id"],
            samples=tuple(
                item
                if isinstance(item, AppearanceSample)
                else AppearanceSample.from_dict(item)
                for item in raw_samples
            ),
            aggregate_descriptors=tuple(
                item
                if isinstance(item, AppearanceDescriptor)
                else AppearanceDescriptor.from_dict(item)
                for item in raw_aggregates
            ),
            metadata=payload.get("metadata", {}),
        )


@dataclass(slots=True)
class AppearanceSet:
    """Clip-level appearance profiles keyed by tracklet."""

    profiles: tuple[TrackletAppearance, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized: list[TrackletAppearance] = []
        tracklet_ids: set[str] = set()
        sample_ids: set[str] = set()

        for raw in self.profiles:
            profile = (
                raw
                if isinstance(raw, TrackletAppearance)
                else TrackletAppearance.from_dict(raw)
            )

            if profile.tracklet_id in tracklet_ids:
                raise ValueError(
                    "duplicate tracklet_id in AppearanceSet: "
                    f"{profile.tracklet_id!r}"
                )
            tracklet_ids.add(profile.tracklet_id)

            for sample in profile.samples:
                if sample.sample_id in sample_ids:
                    raise ValueError(
                        "sample_id must be globally unique in AppearanceSet: "
                        f"{sample.sample_id!r}"
                    )
                sample_ids.add(sample.sample_id)

            normalized.append(profile)

        normalized.sort(key=lambda item: item.tracklet_id)
        self.profiles = tuple(normalized)
        self.metadata = _json_safe_dict("metadata", self.metadata)

    @property
    def tracklet_count(self) -> int:
        return len(self.profiles)

    @property
    def sample_count(self) -> int:
        return sum(
            profile.sample_count
            for profile in self.profiles
        )

    @property
    def empty_profile_count(self) -> int:
        return sum(
            not profile.has_evidence
            for profile in self.profiles
        )

    @property
    def aggregate_descriptor_count(self) -> int:
        return sum(
            profile.aggregate_descriptor_count
            for profile in self.profiles
        )

    def profile(
        self,
        tracklet_id: str,
    ) -> TrackletAppearance | None:
        selected = _non_empty_string("tracklet_id", tracklet_id)
        for profile in self.profiles:
            if profile.tracklet_id == selected:
                return profile
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [
                profile.to_dict()
                for profile in self.profiles
            ],
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
    ) -> "AppearanceSet":
        raw_profiles = payload.get("profiles", ())
        if not isinstance(raw_profiles, (list, tuple)):
            raise ValueError("profiles must be a list or tuple")

        return cls(
            profiles=tuple(
                item
                if isinstance(item, TrackletAppearance)
                else TrackletAppearance.from_dict(item)
                for item in raw_profiles
            ),
            metadata=payload.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str) -> "AppearanceSet":
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(
                "AppearanceSet JSON must encode an object"
            )
        return cls.from_dict(payload)

