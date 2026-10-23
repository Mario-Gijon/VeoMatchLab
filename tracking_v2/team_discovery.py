"""Relative team discovery from person-crop pixels.

This module has no dependency on historic tracking pipelines. Each crop is an
independent observation; track IDs are neither accepted nor consulted.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import hashlib
import math
from typing import Any, Sequence

import numpy as np
from PIL import Image


TEAM_A = "TEAM_A"
TEAM_B = "TEAM_B"
UNKNOWN = "UNKNOWN/SPECIAL"


class CropFeatureError(ValueError):
    """The crop does not contain enough pixels for a stable descriptor."""


def extract_kit_feature(image_path: str | Path) -> np.ndarray:
    """Build an illumination-tolerant color histogram from the upper torso."""
    with Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)

    height, width = rgb.shape[:2]
    if width < 6 or height < 14:
        raise CropFeatureError("crop is too small for a torso descriptor")

    x0 = max(0, int(round(width * 0.18)))
    x1 = min(width, max(x0 + 1, int(round(width * 0.82))))
    y0 = max(0, int(round(height * 0.16)))
    y1 = min(height, max(y0 + 1, int(round(height * 0.64))))
    roi = rgb[y0:y1, x0:x1]
    if roi.shape[0] * roi.shape[1] < 24:
        raise CropFeatureError("torso region has too few pixels")

    values = roi.astype(np.float32) / 255.0
    channel_sum = values.sum(axis=2)
    valid = channel_sum > 1e-5
    red_ratio = values[..., 0][valid] / channel_sum[valid]
    green_ratio = values[..., 1][valid] / channel_sum[valid]
    chroma_hist, _, _ = np.histogram2d(
        red_ratio, green_ratio, bins=(6, 6), range=((0.0, 1.0), (0.0, 1.0))
    )
    chroma_hist = chroma_hist.astype(np.float32)
    chroma_hist /= max(float(chroma_hist.sum()), 1.0)
    chroma_hist = np.sqrt(chroma_hist)

    hsv = np.asarray(Image.fromarray(roi).convert("HSV"), dtype=np.uint8).astype(np.float32) / 255.0
    hue, saturation, brightness = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    chromatic = (saturation >= 0.12) & (brightness >= 0.12)
    if np.any(chromatic):
        hue_hist, _ = np.histogram(
            hue[chromatic], bins=12, range=(0.0, 1.0), weights=saturation[chromatic]
        )
        hue_hist = hue_hist.astype(np.float32)
        hue_hist /= max(float(hue_hist.sum()), 1e-8)
        hue_hist = np.sqrt(hue_hist)
    else:
        hue_hist = np.zeros(12, dtype=np.float32)

    feature = np.concatenate((0.78 * chroma_hist.ravel(), 0.22 * hue_hist))
    norm = float(np.linalg.norm(feature))
    if norm <= 1e-8:
        raise CropFeatureError("crop has no measurable color information")
    return feature / norm


def _kmeans_two(features: np.ndarray, seed: int, restarts: int = 12, iterations: int = 60) -> tuple[np.ndarray, np.ndarray]:
    """Small deterministic two-means implementation for this experiment."""
    count = features.shape[0]
    if count < 2:
        raise ValueError("at least two valid crop features are required")

    best_labels: np.ndarray | None = None
    best_centers: np.ndarray | None = None
    best_inertia = math.inf
    rng = np.random.default_rng(seed)

    for _ in range(restarts):
        first = int(rng.integers(count))
        squared = np.sum((features - features[first]) ** 2, axis=1)
        squared[first] = 0.0
        total = float(squared.sum())
        if total <= 1e-12:
            second = (first + 1) % count
        else:
            second = int(rng.choice(count, p=squared / total))
            if second == first:
                second = int(np.argmax(squared))
        centers = np.stack((features[first], features[second])).astype(np.float32)

        for _iteration in range(iterations):
            distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
            labels = np.argmin(distances, axis=1)
            updated = centers.copy()
            nearest = distances.min(axis=1)
            for cluster in range(2):
                members = features[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    updated[cluster] = features[int(np.argmax(nearest))]
            if np.allclose(updated, centers, atol=1e-7, rtol=0.0):
                centers = updated
                break
            centers = updated

        distances = np.sum((features[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        labels = np.argmin(distances, axis=1)
        inertia = float(distances[np.arange(count), labels].sum())
        if inertia < best_inertia:
            best_labels, best_centers, best_inertia = labels.copy(), centers.copy(), inertia

    assert best_labels is not None and best_centers is not None
    return best_labels, best_centers


def assign_relative_teams(
    features: Sequence[np.ndarray | None],
    *,
    seed: int = 17,
    sample_ids: Sequence[str] | None = None,
    ambiguity_ratio: float = 0.90,
) -> list[dict[str, Any]]:
    """Discover two anonymous appearance groups and reject uncertain samples."""
    total = len(features)
    if sample_ids is not None and len(sample_ids) != total:
        raise ValueError("sample_ids and features must have equal length")
    valid_indices = [i for i, feature in enumerate(features) if feature is not None]
    output = [
        {
            "sample_id": sample_ids[i] if sample_ids is not None else str(i),
            "team": UNKNOWN,
            "confidence": 0.0,
            "cluster_id": None,
            "reason": "invalid_or_missing_crop",
        }
        for i in range(total)
    ]
    if len(valid_indices) < 4:
        for i in valid_indices:
            output[i]["reason"] = "insufficient_samples"
        return output

    matrix = np.stack([np.asarray(features[i], dtype=np.float32) for i in valid_indices])
    if matrix.ndim != 2 or not np.all(np.isfinite(matrix)):
        raise ValueError("features must be finite one-dimensional vectors")

    local_labels, centers = _kmeans_two(matrix, seed=seed)
    counts = np.bincount(local_labels, minlength=2)
    support_floor = max(2, int(math.ceil(len(valid_indices) * 0.04)))
    earliest = []
    for cluster in range(2):
        members = [valid_indices[j] for j, label in enumerate(local_labels) if label == cluster]
        earliest.append(min(members) if members else total + cluster)
    ordered_clusters = sorted(range(2), key=lambda cluster: (earliest[cluster], cluster))
    names = {ordered_clusters[0]: TEAM_A, ordered_clusters[1]: TEAM_B}

    distances = np.linalg.norm(matrix[:, None, :] - centers[None, :, :], axis=2)
    radii: dict[int, float] = {}
    for cluster in range(2):
        cluster_distances = distances[local_labels == cluster, cluster]
        radii[cluster] = (
            float(np.quantile(cluster_distances, 0.97))
            if len(cluster_distances) >= 5
            else float(cluster_distances.max(initial=0.0))
        )
    center_separation = float(np.linalg.norm(centers[0] - centers[1]))

    for local_index, original_index in enumerate(valid_indices):
        cluster = int(local_labels[local_index])
        own_distance = float(distances[local_index, cluster])
        other_distance = float(distances[local_index, 1 - cluster])
        ratio = own_distance / max(other_distance, 1e-8)
        confidence = max(0.0, min(1.0, 1.0 - ratio))
        reason = "assigned"
        team = names[cluster]
        if center_separation < 1e-4:
            team, reason = UNKNOWN, "no_separable_groups"
        elif counts[cluster] < support_floor:
            team, reason = UNKNOWN, "weak_cluster_support"
        elif own_distance > radii[cluster] + 1e-8:
            team, reason = UNKNOWN, "appearance_outlier"
        elif ratio >= ambiguity_ratio:
            team, reason = UNKNOWN, "ambiguous_between_groups"
        output[original_index] = {
            "sample_id": sample_ids[original_index] if sample_ids is not None else str(original_index),
            "team": team,
            "confidence": confidence if team != UNKNOWN else min(confidence, 0.49),
            "cluster_id": cluster,
            "reason": reason,
        }
    return output


def _scale_bin(record: dict[str, Any]) -> str:
    height = int(record.get("crop_height_px") or 0)
    if height < 40:
        return "small"
    if height < 80:
        return "medium"
    return "large"


def _stable_tie(seed: int, record: dict[str, Any]) -> str:
    token = "|".join((
        str(seed),
        str(record.get("frame_index", "")),
        f"{float(record.get('timestamp_s') or 0.0):.3f}",
        str(record.get("representative_view", "")),
        _scale_bin(record),
        str(record.get("crop_width_px", "")),
        str(record.get("crop_height_px", "")),
    ))
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _choose_from_frames(
    records: Sequence[dict[str, Any]],
    amount: int,
    *,
    seed: int,
    used_frames: set[str],
) -> list[dict[str, Any]]:
    by_frame: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        frame = str(record.get("frame_index", f"time:{float(record.get('timestamp_s') or 0):.3f}"))
        if frame not in used_frames:
            by_frame[frame].append(record)
    frames = sorted(
        by_frame,
        key=lambda frame: (
            min(float(item.get("timestamp_s") or 0.0) for item in by_frame[frame]),
            frame,
        ),
    )
    if amount <= 0 or not frames:
        return []
    chosen_count = min(amount, len(frames))
    positions = np.linspace(0, len(frames) - 1, chosen_count).round().astype(int)
    target_frames: list[str] = []
    for position in positions:
        frame = frames[int(position)]
        if frame not in target_frames:
            target_frames.append(frame)
    if len(target_frames) < chosen_count:
        target_frames.extend(frame for frame in frames if frame not in target_frames)
        target_frames = target_frames[:chosen_count]

    view_counts: dict[str, int] = defaultdict(int)
    scale_counts: dict[str, int] = defaultdict(int)
    chosen: list[dict[str, Any]] = []
    for frame in target_frames:
        options = sorted(
            by_frame[frame],
            key=lambda item: (
                view_counts[str(item.get("representative_view", "unknown"))],
                scale_counts[_scale_bin(item)],
                _stable_tie(seed, item),
            ),
        )
        pick = options[0]
        chosen.append(pick)
        used_frames.add(frame)
        view_counts[str(pick.get("representative_view", "unknown"))] += 1
        scale_counts[_scale_bin(pick)] += 1
    return chosen


def select_sample(
    records: Sequence[dict[str, Any]],
    *,
    sample_count: int = 96,
    outside_count: int = 12,
    seed: int = 17,
) -> list[dict[str, Any]]:
    """Select time-spread crops while explicitly retaining outside-field data.

    Field context affects sampling only. It is never used by the appearance
    descriptor or the team assignment.
    """
    if sample_count < 4:
        raise ValueError("sample_count must be at least four")
    outside = [r for r in records if r.get("field_context") == "outside"]
    remaining = [r for r in records if r.get("field_context") != "outside"]
    used_frames: set[str] = set()
    selected = _choose_from_frames(
        outside, min(max(outside_count, 0), sample_count), seed=seed, used_frames=used_frames
    )
    selected.extend(
        _choose_from_frames(
            remaining, sample_count - len(selected), seed=seed, used_frames=used_frames
        )
    )
    if len(selected) < sample_count:
        selected_frames = {
            str(r.get("frame_index", f"time:{float(r.get('timestamp_s') or 0):.3f}"))
            for r in selected
        }
        leftovers = [
            r for r in records
            if str(r.get("frame_index", f"time:{float(r.get('timestamp_s') or 0):.3f}"))
            not in selected_frames
        ]
        selected.extend(
            _choose_from_frames(
                leftovers, sample_count - len(selected), seed=seed + 1, used_frames=used_frames
            )
        )
    return sorted(
        selected[:sample_count],
        key=lambda r: (
            float(r.get("timestamp_s") or 0.0),
            str(r.get("representative_view", "")),
            str(r.get("frame_index", "")),
        ),
    )
