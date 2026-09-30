"""Pure orchestration for the first executable Tracking V1 stages.

This module knows about canonical detections, short tracklets and the artifact
store interface. It deliberately knows nothing about Veo cameras, RF-DETR,
OpenCV, the web app or Tracking V0.

Stage boundary
--------------
detections/frames.jsonl
    -> canonical per-frame physical-person hypotheses

tracklets/tracklets.jsonl
    -> conservative short temporal fragments built only from the published
       detections artifact

The second stage reloads the first stage from the artifact store instead of
sharing in-memory detector objects. This keeps the pipeline reproducible and
makes every stage independently rerunnable.
"""

from __future__ import annotations

from dataclasses import asdict
import math
from typing import Any, Iterable, Iterator, Mapping, Protocol

from tracking.detections import CanonicalDetectionFrame
from tracking.tracklets import TrackletSet
from tracking.tracklets.builder import (
    ConservativeTrackletBuilder,
    TrackletBuilderConfig,
)


class _StageManifestLike(Protocol):
    artifact: str | None


class _ManifestLike(Protocol):
    def stage(self, stage: str) -> _StageManifestLike: ...


class _ArtifactLike(Protocol):
    ref: str
    record_count: int | None


class ArtifactStoreLike(Protocol):
    def begin_stage(
        self,
        stage: str,
        *,
        version: str | int | None = None,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        invalidate_downstream: bool = True,
        clean_outputs: bool = True,
        restart: bool = False,
    ) -> _ManifestLike: ...

    def complete_stage(
        self,
        stage: str,
        *,
        artifact: Any,
        summary: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> _ManifestLike: ...

    def fail_stage(
        self,
        stage: str,
        error: str | BaseException,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> _ManifestLike: ...

    def load_manifest(self) -> _ManifestLike: ...

    def write_jsonl(
        self,
        stage: str,
        relative_path: str,
        records: Iterable[Any],
    ) -> _ArtifactLike: ...

    def write_json(
        self,
        stage: str,
        relative_path: str,
        payload: Any,
        *,
        indent: int | None = 2,
    ) -> _ArtifactLike: ...

    def iter_jsonl(self, ref: str) -> Iterator[dict[str, Any]]: ...


def _builder_config_dict(config: TrackletBuilderConfig) -> dict[str, Any]:
    return asdict(config)


class _DetectionStats:
    def __init__(self) -> None:
        self.frame_count = 0
        self.detection_count = 0
        self.empty_frame_count = 0
        self.multiview_detection_count = 0
        self.confidence_sum = 0.0
        self.min_confidence: float | None = None
        self.max_confidence: float | None = None
        self.min_x: float | None = None
        self.max_x: float | None = None
        self.min_z: float | None = None
        self.max_z: float | None = None
        self.first_frame_index: int | None = None
        self.last_frame_index: int | None = None
        self.first_timestamp_s: float | None = None
        self.last_timestamp_s: float | None = None

    def observe(self, frame: CanonicalDetectionFrame) -> None:
        self.frame_count += 1
        if self.first_frame_index is None:
            self.first_frame_index = frame.frame_index
            self.first_timestamp_s = frame.timestamp_s
        self.last_frame_index = frame.frame_index
        self.last_timestamp_s = frame.timestamp_s

        if not frame.detections:
            self.empty_frame_count += 1

        for detection in frame.detections:
            self.detection_count += 1
            self.confidence_sum += detection.confidence
            if len(detection.contributing_views) > 1:
                self.multiview_detection_count += 1

            self.min_confidence = (
                detection.confidence
                if self.min_confidence is None
                else min(self.min_confidence, detection.confidence)
            )
            self.max_confidence = (
                detection.confidence
                if self.max_confidence is None
                else max(self.max_confidence, detection.confidence)
            )
            self.min_x = detection.x if self.min_x is None else min(self.min_x, detection.x)
            self.max_x = detection.x if self.max_x is None else max(self.max_x, detection.x)
            self.min_z = detection.z if self.min_z is None else min(self.min_z, detection.z)
            self.max_z = detection.z if self.max_z is None else max(self.max_z, detection.z)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_count": self.frame_count,
            "detection_count": self.detection_count,
            "empty_frame_count": self.empty_frame_count,
            "detections_per_frame_mean": (
                self.detection_count / self.frame_count
                if self.frame_count
                else 0.0
            ),
            "multiview_detection_count": self.multiview_detection_count,
            "confidence_mean": (
                self.confidence_sum / self.detection_count
                if self.detection_count
                else None
            ),
            "confidence_min": self.min_confidence,
            "confidence_max": self.max_confidence,
            "field_bounds": {
                "x_min": self.min_x,
                "x_max": self.max_x,
                "z_min": self.min_z,
                "z_max": self.max_z,
            },
            "first_frame_index": self.first_frame_index,
            "last_frame_index": self.last_frame_index,
            "first_timestamp_s": self.first_timestamp_s,
            "last_timestamp_s": self.last_timestamp_s,
        }


def _observed_frames(
    frames: Iterable[CanonicalDetectionFrame | Mapping[str, Any]],
    stats: _DetectionStats,
) -> Iterator[CanonicalDetectionFrame]:
    previous_frame_index: int | None = None
    previous_timestamp_s: float | None = None

    for raw in frames:
        frame = (
            raw
            if isinstance(raw, CanonicalDetectionFrame)
            else CanonicalDetectionFrame.from_dict(raw)
        )

        if (
            previous_frame_index is not None
            and frame.frame_index <= previous_frame_index
        ):
            raise ValueError("canonical frames must have increasing frame_index")
        if (
            previous_timestamp_s is not None
            and frame.timestamp_s <= previous_timestamp_s
        ):
            raise ValueError("canonical frames must have increasing timestamp_s")

        stats.observe(frame)
        previous_frame_index = frame.frame_index
        previous_timestamp_s = frame.timestamp_s
        yield frame


def publish_detections_stage(
    store: ArtifactStoreLike,
    frames: Iterable[CanonicalDetectionFrame | Mapping[str, Any]],
    *,
    version: str = "canonical-v1",
    config: Mapping[str, Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish canonical frames as the completed ``detections`` stage."""
    store.begin_stage(
        "detections",
        version=version,
        config=dict(config or {}),
        metadata=dict(metadata or {}),
    )

    stats = _DetectionStats()

    try:
        artifact = store.write_jsonl(
            "detections",
            "frames.jsonl",
            _observed_frames(frames, stats),
        )
        summary_payload = stats.to_dict()
        summary = store.write_json(
            "detections",
            "summary.json",
            summary_payload,
        )
        store.complete_stage(
            "detections",
            artifact=artifact,
            summary=summary,
            metadata={
                "frame_count": stats.frame_count,
                "detection_count": stats.detection_count,
            },
        )
        return summary_payload
    except Exception as exc:
        try:
            store.fail_stage("detections", exc)
        except Exception:
            pass
        raise


def _published_detection_frames(
    store: ArtifactStoreLike,
) -> Iterator[CanonicalDetectionFrame]:
    manifest = store.load_manifest()
    artifact_ref = manifest.stage("detections").artifact
    if not artifact_ref:
        raise RuntimeError("completed detections stage has no artifact reference")

    for payload in store.iter_jsonl(artifact_ref):
        yield CanonicalDetectionFrame.from_dict(payload)


def _tracklet_summary(tracklets: TrackletSet) -> dict[str, Any]:
    items = tracklets.tracklets
    observation_counts = [item.observation_count for item in items]
    durations = [item.duration_s for item in items]
    path_lengths = [item.path_length_m for item in items]
    max_frame_gaps = [item.max_frame_gap for item in items]

    singleton_count = sum(count == 1 for count in observation_counts)
    multi_observation_count = len(items) - singleton_count

    return {
        "tracklet_count": len(items),
        "observation_count": tracklets.observation_count,
        "singleton_count": singleton_count,
        "multi_observation_tracklet_count": multi_observation_count,
        "singleton_fraction": (
            singleton_count / len(items)
            if items
            else 0.0
        ),
        "observations_per_tracklet_mean": (
            sum(observation_counts) / len(items)
            if items
            else 0.0
        ),
        "observations_per_tracklet_max": max(observation_counts, default=0),
        "duration_s_mean": (
            sum(durations) / len(items)
            if items
            else 0.0
        ),
        "duration_s_max": max(durations, default=0.0),
        "path_length_m_mean": (
            sum(path_lengths) / len(items)
            if items
            else 0.0
        ),
        "path_length_m_max": max(path_lengths, default=0.0),
        "max_frame_gap": max(max_frame_gaps, default=0),
        "builder_metadata": dict(tracklets.metadata),
    }


def build_tracklets_stage(
    store: ArtifactStoreLike,
    *,
    config: TrackletBuilderConfig | None = None,
    version: str = "conservative-local-v1",
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reload published detections and build the completed tracklets stage."""
    selected_config = config or TrackletBuilderConfig()
    builder = ConservativeTrackletBuilder(selected_config)

    store.begin_stage(
        "tracklets",
        version=version,
        config=_builder_config_dict(selected_config),
        metadata=dict(metadata or {}),
    )

    try:
        tracklets = builder.build(_published_detection_frames(store))
        artifact = store.write_jsonl(
            "tracklets",
            "tracklets.jsonl",
            tracklets.tracklets,
        )
        summary_payload = _tracklet_summary(tracklets)
        summary = store.write_json(
            "tracklets",
            "summary.json",
            summary_payload,
        )
        store.complete_stage(
            "tracklets",
            artifact=artifact,
            summary=summary,
            metadata={
                "tracklet_count": tracklets.tracklet_count,
                "observation_count": tracklets.observation_count,
            },
        )
        return summary_payload
    except Exception as exc:
        try:
            store.fail_stage("tracklets", exc)
        except Exception:
            pass
        raise


def run_detection_and_tracklet_stages(
    store: ArtifactStoreLike,
    frames: Iterable[CanonicalDetectionFrame | Mapping[str, Any]],
    *,
    detection_config: Mapping[str, Any] | None = None,
    detection_metadata: Mapping[str, Any] | None = None,
    tracklet_config: TrackletBuilderConfig | None = None,
) -> dict[str, dict[str, Any]]:
    """Run the currently executable V1 pipeline from canonical frames."""
    detections = publish_detections_stage(
        store,
        frames,
        config=detection_config,
        metadata=detection_metadata,
    )
    tracklets = build_tracklets_stage(
        store,
        config=tracklet_config,
    )
    return {
        "detections": detections,
        "tracklets": tracklets,
    }
