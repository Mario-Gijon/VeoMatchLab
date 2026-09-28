#!/usr/bin/env python3
"""V8 audit: reconstruct 23 physical people, then identify the referee separately.

Identity recovery stays role-agnostic.  The physical solver first reconstructs
23 different people using the full offline clip, segmentation quality,
continuity, team incompatibility and ReID.

Only AFTER those 23 physical identities exist does a dedicated referee
classifier select one identity as REFEREE.  The other 22 become PLAYER
identities.  The referee classifier never changes physical identity membership.

This keeps "who is the same person?" separate from "what role does that person
have?", which is exactly what we want for noisy football crops.

Nothing is written to production tracking or preview output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import re
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.identity_segmentation import (  # noqa: E402
    IdentityAppearanceSample,
    IdentitySegmentationConfig,
    SegmentIdentityQuality,
    TrackIdentitySegmentation,
    segment_all_tracks,
)
from tracking.physical_identity_solver import (  # noqa: E402
    PhysicalIdentitySolverConfig,
    PhysicalObservation,
    PhysicalSegmentQuality,
    PhysicalTrackSegment,
    solve_physical_identities,
)
from tracking.team_appearance import (  # noqa: E402
    AppearanceLabel,
    KitAppearanceConfig,
    extract_crop_appearance,
    fit_kit_appearance,
)
from tracking.team_similarity import RelativeTeamLabel  # noqa: E402
from tracking.referee_classifier import (  # noqa: E402
    RefereeClassifierConfig,
    classify_referee_identity,
)


BEST_RE = re.compile(
    r"best_\d+_frame_(?P<frame>\d+)_q_(?P<quality>[0-9.]+)\.jpg$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracking-frames", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--crop-dir", type=Path, required=True)
    parser.add_argument("--roster-size", type=int, default=23)
    parser.add_argument("--field-length", type=float, default=105.0)
    parser.add_argument("--field-width", type=float, default=68.0)
    parser.add_argument("--continuation-margin", type=float, default=1.5)
    parser.add_argument("--expected-team-size", type=int, default=11)
    parser.add_argument(
        "--expected-referee-segment",
        type=str,
        default=None,
        help=(
            "Optional diagnostic-only segment key such as 27:0. "
            "It never influences classification."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "experiments/tracking/reid_similarity_debug_margin15/"
            "match_participants_v8_report.json"
        ),
    )
    return parser.parse_args()


def load_raw_observations(
    path: Path,
) -> dict[int, tuple[PhysicalObservation, ...]]:
    grouped: dict[int, list[PhysicalObservation]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            frame_index = int(row["frame_index"])
            timestamp = float(row.get("timestamp_s", row.get("timestamp")))

            for track in row.get("tracks", []):
                if str(track.get("position_kind", "")) != "observed":
                    continue

                raw_id = int(track["track_id"])
                grouped[raw_id].append(
                    PhysicalObservation(
                        frame_index=frame_index,
                        timestamp=timestamp,
                        x=float(track["x"]),
                        z=float(track["z"]),
                        measurement_id=(
                            str(track["measurement_id"])
                            if track.get("measurement_id") is not None
                            else None
                        ),
                        confidence=(
                            float(track["confidence"])
                            if track.get("confidence") is not None
                            else None
                        ),
                    )
                )

    return {
        raw_id: tuple(
            sorted(rows, key=lambda item: (item.timestamp, item.frame_index))
        )
        for raw_id, rows in sorted(grouped.items())
    }


def load_embedding_samples(
    path: Path,
) -> dict[int, list[dict[str, object]]]:
    payload = np.load(path, allow_pickle=False)
    vectors = payload["vectors"]
    track_ids = payload["track_ids"]
    frames = payload["frame_indices"]
    qualities = payload["qualities"]

    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for index in range(len(vectors)):
        grouped[int(track_ids[index])].append(
            {
                "frame_index": int(frames[index]),
                "quality": float(qualities[index]),
                "vector": np.asarray(vectors[index], dtype=np.float32),
            }
        )

    for rows in grouped.values():
        rows.sort(key=lambda item: int(item["frame_index"]))
    return grouped


def load_crop_appearance_evidence(
    crop_dir: Path,
) -> tuple[
    dict[tuple[int, int], AppearanceLabel],
    dict[tuple[int, int], np.ndarray],
    dict[str, object],
]:
    """Return semantic labels AND raw kit descriptors for every saved crop.

    V1 segmentation only consumed the final TEAM_A/TEAM_B/SPECIAL cluster
    label. V2 also keeps the normalized descriptor itself so a persistent
    visual change can be detected even when global clustering labels both sides
    imperfectly.
    """
    config = KitAppearanceConfig()
    appearances = []
    keys: list[tuple[int, int]] = []

    for track_dir in sorted(crop_dir.glob("track_*")):
        if not track_dir.is_dir():
            continue
        try:
            raw_id = int(track_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue

        for path in sorted(track_dir.glob("best_*.jpg")):
            match = BEST_RE.match(path.name)
            if match is None:
                continue
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                continue

            frame = int(match.group("frame"))
            quality = float(match.group("quality"))
            try:
                appearance = extract_crop_appearance(
                    image,
                    track_id=raw_id,
                    frame_index=frame,
                    quality=quality,
                    config=config,
                )
            except ValueError:
                continue

            appearances.append(appearance)
            keys.append((raw_id, frame))

    if not appearances:
        raise RuntimeError(f"no best_*.jpg crops found under {crop_dir}")

    result = fit_kit_appearance(appearances, config=config)
    cluster_by_id = {
        cluster.cluster_id: cluster
        for cluster in result.clusters
    }

    labels: dict[tuple[int, int], AppearanceLabel] = {}
    descriptors: dict[tuple[int, int], np.ndarray] = {}

    for key, appearance, cluster_id in zip(
        keys,
        appearances,
        result.crop_cluster_ids,
    ):
        labels[key] = cluster_by_id[int(cluster_id)].semantic_label
        descriptors[key] = np.asarray(
            appearance.descriptor,
            dtype=np.float32,
        )

    return labels, descriptors, result.to_dict()



def build_segmentation_samples(
    embeddings: dict[int, list[dict[str, object]]],
    crop_labels: dict[tuple[int, int], AppearanceLabel],
    crop_descriptors: dict[tuple[int, int], np.ndarray],
) -> dict[int, list[IdentityAppearanceSample]]:
    result: dict[int, list[IdentityAppearanceSample]] = {}

    for raw_id, rows in sorted(embeddings.items()):
        samples = []
        for row in rows:
            frame = int(row["frame_index"])
            key = (raw_id, frame)
            samples.append(
                IdentityAppearanceSample(
                    raw_track_id=raw_id,
                    frame_index=frame,
                    quality=float(row["quality"]),
                    reid_vector=np.asarray(
                        row["vector"],
                        dtype=np.float32,
                    ),
                    appearance_label=crop_labels.get(
                        key,
                        AppearanceLabel.UNKNOWN,
                    ),
                    kit_vector=crop_descriptors.get(key),
                )
            )
        if samples:
            result[raw_id] = samples

    return result



def build_physical_segments(
    observations_by_raw: dict[int, tuple[PhysicalObservation, ...]],
    segmentations: dict[int, TrackIdentitySegmentation],
    *,
    field_length: float,
    field_width: float,
    continuation_margin: float,
) -> tuple[PhysicalTrackSegment, ...]:
    half_length = field_length / 2.0
    half_width = field_width / 2.0

    segments: list[PhysicalTrackSegment] = []
    next_segment_id = 1

    for raw_id, observations in sorted(observations_by_raw.items()):
        segmentation = segmentations.get(raw_id)
        if segmentation is None:
            boundaries: tuple[int, ...] = ()
            summaries = ()
        else:
            boundaries = segmentation.boundaries_after_frame
            summaries = segmentation.segments

        buckets: list[list[PhysicalObservation]] = [
            [] for _ in range(len(boundaries) + 1)
        ]
        for observation in observations:
            segment_index = sum(
                observation.frame_index > boundary
                for boundary in boundaries
            )
            buckets[segment_index].append(observation)

        for segment_index, bucket in enumerate(buckets):
            if not bucket:
                continue

            summary = (
                summaries[segment_index]
                if segment_index < len(summaries)
                else None
            )
            if summary is None:
                team_label = RelativeTeamLabel.UNKNOWN
                team_confidence = 0.0
                reid_prototype = None
                identity_quality = PhysicalSegmentQuality.AMBIGUOUS
                identity_quality_score = 0.0
            else:
                team_label = summary.team_label
                team_confidence = summary.team_confidence
                reid_prototype = summary.reid_prototype
                identity_quality = PhysicalSegmentQuality(
                    summary.identity_quality.value
                )
                identity_quality_score = summary.identity_quality_score

            strict = sum(
                -half_length <= row.x <= half_length
                and -half_width <= row.z <= half_width
                for row in bucket
            )
            allowed = sum(
                -half_length - continuation_margin <= row.x <= half_length + continuation_margin
                and -half_width - continuation_margin <= row.z <= half_width + continuation_margin
                for row in bucket
            )

            segments.append(
                PhysicalTrackSegment(
                    segment_id=next_segment_id,
                    raw_track_id=raw_id,
                    segment_index=segment_index,
                    observations=tuple(bucket),
                    team_label=team_label,
                    team_confidence=float(team_confidence),
                    reid_prototype=reid_prototype,
                    identity_quality=identity_quality,
                    identity_quality_score=float(identity_quality_score),
                    inside_observed_count=int(strict),
                    continuation_observed_count=int(allowed),
                )
            )
            next_segment_id += 1

    return tuple(segments)



def build_segment_kit_prototypes(
    segments: tuple[PhysicalTrackSegment, ...],
    segmentations: dict[int, TrackIdentitySegmentation],
    samples_by_track: dict[int, list[IdentityAppearanceSample]],
) -> dict[int, np.ndarray]:
    """Aggregate raw kit descriptors inside each final physical segment."""

    by_key: dict[tuple[int, int], np.ndarray] = {}

    for raw_id, samples in samples_by_track.items():
        segmentation = segmentations.get(raw_id)
        boundaries = (
            segmentation.boundaries_after_frame
            if segmentation is not None
            else ()
        )

        grouped: dict[int, list[IdentityAppearanceSample]] = defaultdict(list)

        for sample in samples:
            if sample.kit_vector is None:
                continue
            segment_index = sum(
                sample.frame_index > boundary
                for boundary in boundaries
            )
            grouped[segment_index].append(sample)

        for segment_index, rows in grouped.items():
            vectors = []
            weights = []

            for sample in rows:
                vector = np.asarray(
                    sample.kit_vector,
                    dtype=np.float64,
                )
                if vector.ndim != 1 or not np.isfinite(vector).all():
                    continue

                norm = float(np.linalg.norm(vector))
                if norm <= 1e-12:
                    continue

                vectors.append(vector / norm)
                weights.append(max(0.05, float(sample.quality)))

            if not vectors:
                continue

            prototype = np.average(
                np.stack(vectors, axis=0),
                axis=0,
                weights=np.asarray(weights, dtype=np.float64),
            )
            norm = float(np.linalg.norm(prototype))
            if norm <= 1e-12:
                continue

            by_key[(raw_id, segment_index)] = (
                prototype / norm
            ).astype(np.float32)

    return {
        segment.segment_id: by_key[
            (segment.raw_track_id, segment.segment_index)
        ]
        for segment in segments
        if (
            segment.raw_track_id,
            segment.segment_index,
        ) in by_key
    }



def main() -> None:
    args = parse_args()

    observations_by_raw = load_raw_observations(args.tracking_frames)
    embeddings = load_embedding_samples(args.embedding_cache)
    (
        crop_labels,
        crop_descriptors,
        kit_report,
    ) = load_crop_appearance_evidence(args.crop_dir)
    samples_by_track = build_segmentation_samples(
        embeddings,
        crop_labels,
        crop_descriptors,
    )

    segmentation_config = IdentitySegmentationConfig()
    segmentations = segment_all_tracks(
        samples_by_track,
        config=segmentation_config,
    )

    segments = build_physical_segments(
        observations_by_raw,
        segmentations,
        field_length=args.field_length,
        field_width=args.field_width,
        continuation_margin=args.continuation_margin,
    )

    solver_config = PhysicalIdentitySolverConfig(
        roster_size=args.roster_size,
    )
    result = solve_physical_identities(segments, config=solver_config)

    segment_by_id = {segment.segment_id: segment for segment in segments}

    kit_prototype_by_segment_id = build_segment_kit_prototypes(
        segments,
        segmentations,
        samples_by_track,
    )

    referee_config = RefereeClassifierConfig(
        expected_team_size=args.expected_team_size,
    )
    referee_result = classify_referee_identity(
        result.identities,
        segment_by_id=segment_by_id,
        kit_prototype_by_segment_id=kit_prototype_by_segment_id,
        field_length=args.field_length,
        field_width=args.field_width,
        config=referee_config,
    )

    referee_identity = next(
        identity
        for identity in result.identities
        if identity.identity_id == referee_result.referee_identity_id
    )
    player_identities = [
        identity
        for identity in result.identities
        if identity.identity_id in referee_result.player_identity_ids
    ]

    player_team_counts = {
        RelativeTeamLabel.TEAM_A: sum(
            identity.team_label is RelativeTeamLabel.TEAM_A
            for identity in player_identities
        ),
        RelativeTeamLabel.TEAM_B: sum(
            identity.team_label is RelativeTeamLabel.TEAM_B
            for identity in player_identities
        ),
        RelativeTeamLabel.UNKNOWN: sum(
            identity.team_label is RelativeTeamLabel.UNKNOWN
            for identity in player_identities
        ),
    }
    segmented_tracks = [
        segmentation
        for segmentation in segmentations.values()
        if segmentation.segmented
    ]

    quality_counts = {
        quality: sum(
            segment.identity_quality is quality
            for segment in segments
        )
        for quality in PhysicalSegmentQuality
    }

    total_observed = sum(len(rows) for rows in observations_by_raw.values())
    selected_observed = sum(
        segment_by_id[segment_id].observed_count
        for segment_id in result.segment_to_identity
    )
    total_inside = sum(segment.inside_observed_count for segment in segments)
    selected_inside = sum(
        segment_by_id[segment_id].inside_observed_count
        for segment_id in result.segment_to_identity
    )

    print()
    print("V8 OFFLINE MATCH PARTICIPANTS: 23 PEOPLE -> 22 PLAYERS + 1 REFEREE")
    print(f"raw tracker IDs             : {len(observations_by_raw)}")
    print(f"identity-consistent segments: {len(segments)}")
    print(f"raw IDs split internally    : {len(segmented_tracks)}")
    print(
        "split raw IDs               : "
        + (
            ",".join(str(row.raw_track_id) for row in segmented_tracks)
            if segmented_tracks
            else "(none)"
        )
    )
    print(f"physical identities         : {len(result.identities)}")
    print(
        "segment quality            : "
        f"clean={quality_counts[PhysicalSegmentQuality.CLEAN]}, "
        f"ambiguous={quality_counts[PhysicalSegmentQuality.AMBIGUOUS]}, "
        f"contaminated={quality_counts[PhysicalSegmentQuality.CONTAMINATED]}"
    )
    print(
        f"referee identity            : {referee_result.referee_identity_id} "
        f"({','.join(referee_identity.member_keys)})"
    )
    print(f"player identities           : {len(player_identities)}")
    print(
        "player team labels          : "
        f"team_a={player_team_counts[RelativeTeamLabel.TEAM_A]}, "
        f"team_b={player_team_counts[RelativeTeamLabel.TEAM_B]}, "
        f"unknown={player_team_counts[RelativeTeamLabel.UNKNOWN]}"
    )
    print("referee role constraint     : POST-HOC ONLY")
    print("physical solver roles       : NONE")
    print(f"solver status               : {result.solver_status}")
    print(f"objective                   : {result.objective:.3f}")
    print(f"selected segments           : {result.selected_segment_count}")
    print(f"unassigned segments         : {len(result.unassigned_segment_ids)}")
    print(f"continuation candidates     : {result.candidate_continuation_count}")
    print(f"hard opposite-team prunes   : {result.hard_team_prunes}")
    print(f"temporal/overlap prunes     : {result.temporal_prunes}")
    print(f"duplicate hypotheses        : {len(result.duplicate_candidates)}")
    print(
        f"observed coverage           : {selected_observed}/{total_observed} "
        f"({100.0 * selected_observed / max(total_observed, 1):.1f}%)"
    )
    print(
        f"strict-inside coverage      : {selected_inside}/{total_inside} "
        f"({100.0 * selected_inside / max(total_inside, 1):.1f}%)"
    )

    if segmented_tracks:
        print()
        print("INTERNAL RAW-ID SPLITS")
        print(
            "RAW  BOUNDARY   LEFT_APP  RIGHT_APP "
            "KIT_GAIN KIT_X BOUND_X REID_GAIN REID_X  REASON"
        )
        print("-" * 120)
        for segmentation in segmented_tracks:
            for split in segmentation.accepted_splits:
                kit_gain = (
                    f"{split.kit_split_gain:.3f}"
                    if split.kit_split_gain is not None
                    else "-"
                )
                kit_cross = (
                    f"{split.cross_kit_similarity:.3f}"
                    if split.cross_kit_similarity is not None
                    else "-"
                )
                boundary_cross = (
                    f"{split.boundary_kit_similarity:.3f}"
                    if split.boundary_kit_similarity is not None
                    else "-"
                )
                reid_gain = (
                    f"{split.reid_split_gain:.3f}"
                    if split.reid_split_gain is not None
                    else "-"
                )
                reid_cross = (
                    f"{split.cross_reid_similarity:.3f}"
                    if split.cross_reid_similarity is not None
                    else "-"
                )
                print(
                    f"{segmentation.raw_track_id:>3} "
                    f"{split.boundary_after_frame:>9} "
                    f"{split.left_appearance.value:>10} "
                    f"{split.right_appearance.value:>10} "
                    f"{kit_gain:>8} {kit_cross:>5} "
                    f"{boundary_cross:>7} "
                    f"{reid_gain:>9} {reid_cross:>6}  "
                    f"{split.reason}"
                )

    print()
    print("SEGMENT PURITY / IDENTITY QUALITY")
    print("SEG     QUALITY SCORE  TEAM CONF  APP_P KIT_C  REID_C  REASON")
    print("-" * 96)
    for raw_id, segmentation in sorted(segmentations.items()):
        for summary in segmentation.segments:
            kit_c = (
                f"{summary.kit_internal_similarity:.3f}"
                if summary.kit_internal_similarity is not None
                else "-"
            )
            reid_c = (
                f"{summary.reid_internal_similarity:.3f}"
                if summary.reid_internal_similarity is not None
                else "-"
            )
            print(
                f"{summary.key:>6} "
                f"{summary.identity_quality.value:>12} "
                f"{summary.identity_quality_score:>5.2f} "
                f"{summary.team_label.value:>7} "
                f"{summary.team_confidence:>4.2f} "
                f"{summary.appearance_confidence:>5.2f} "
                f"{kit_c:>5} "
                f"{reid_c:>7} "
                f"{summary.identity_quality_reason}"
            )

    print()
    print("VISUAL REGRESSION CHECKS (diagnostic only; never forced)")
    print("RAW EXPECT     RESULT     BOUNDARIES")
    print("-" * 52)
    expected = {
        6: "stable",
        8: "stable",
        9: "stable",
        10: "stable",
        11: "split",
        16: "stable",
        27: "split",
        35: "split",
    }
    for raw_id, expectation in expected.items():
        segmentation = segmentations.get(raw_id)
        if segmentation is None:
            result_text = "missing"
            boundaries = "-"
        else:
            result_text = "split" if segmentation.segmented else "stable"
            boundaries = (
                ",".join(str(value) for value in segmentation.boundaries_after_frame)
                if segmentation.boundaries_after_frame
                else "-"
            )
        marker = "OK" if result_text == expectation else "CHECK"
        print(
            f"{raw_id:>3} {expectation:>8} {result_text:>10} "
            f"{boundaries:>12}  {marker}"
        )

    print()
    print("PHYSICAL IDENTITIES + FINAL ROLE")
    print("ID      TEAM     ROLE MEMBERS")
    print("-" * 112)
    for identity in result.identities:
        members = []
        for segment_id in identity.member_segment_ids:
            segment = segment_by_id[segment_id]
            members.append(
                f"{segment.key}[{segment.identity_quality.value}]"
            )
        role = (
            "REFEREE"
            if identity.identity_id == referee_result.referee_identity_id
            else "PLAYER"
        )
        print(
            f"{identity.identity_id:>2} "
            f"{identity.team_label.value:>9} "
            f"{role:>8} "
            + ",".join(members)
        )

    print()
    print("REFEREE CANDIDATES (post-hoc; physical membership is unchanged)")
    print(
        "RANK ID    TEAM TOTAL STRUCT KIT_OUT NON_GK SUPPORT CENTRAL "
        "GOAL_FRAC KIT_AFF  AFTER(A/B/U) MEMBERS"
    )
    print("-" * 145)
    for rank, row in enumerate(referee_result.candidates, start=1):
        kit_aff = (
            f"{row.kit_team_affinity:.3f}"
            if row.kit_team_affinity is not None
            else "-"
        )
        marker = " <== REFEREE" if rank == 1 else ""
        print(
            f"{rank:>4} {row.identity_id:>2} "
            f"{row.team_label.value:>7} "
            f"{row.total_score:>5.3f} "
            f"{row.team_structure_score:>6.3f} "
            f"{row.kit_outlier_score:>7.3f} "
            f"{row.non_goalkeeper_score:>6.3f} "
            f"{row.support_score:>7.3f} "
            f"{row.central_presence_score:>7.3f} "
            f"{row.goal_zone_fraction:>8.3f} "
            f"{kit_aff:>7} "
            f"{row.team_a_after_removal:>2}/"
            f"{row.team_b_after_removal:>2}/"
            f"{row.unknown_after_removal:<2} "
            f"{','.join(row.member_keys)}"
            f"{marker}"
        )

    if args.expected_referee_segment is not None:
        selected_keys = set(referee_identity.member_keys)
        expected_ok = args.expected_referee_segment in selected_keys
        print()
        print("VISUAL REFEREE REGRESSION (diagnostic only; never forced)")
        print(
            f"expected segment            : {args.expected_referee_segment}"
        )
        print(
            f"selected referee identity   : "
            f"{referee_result.referee_identity_id} "
            f"({','.join(referee_identity.member_keys)})"
        )
        print(
            f"result                      : "
            f"{'OK' if expected_ok else 'CHECK'}"
        )

    print()
    print("SELECTED CONTINUATIONS")
    print("FROM       TO  SCORE  CONT  REID  TEAM    GAP")
    print("-" * 72)
    for edge in result.selected_continuations:
        left = segment_by_id[edge.left_segment_id]
        right = segment_by_id[edge.right_segment_id]
        print(
            f"{left.key:>7} {right.key:>7} "
            f"{edge.total_score:>6.3f} "
            f"{edge.continuity_score:>5.3f} "
            f"{edge.reid_score:>5.3f} "
            f"{edge.team_score:>5.2f} "
            f"{edge.gap_s:>6.2f}"
        )

    if result.duplicate_candidates:
        print()
        print("TOP DUPLICATE HYPOTHESES (diagnostic only; not merged)")
        print("LEFT   RIGHT SHARED  DIST_MED  MOVE_COS  REID  SCORE")
        print("-" * 72)
        for edge in sorted(
            result.duplicate_candidates,
            key=lambda row: -row.total_score,
        )[:20]:
            left = segment_by_id[edge.left_segment_id]
            right = segment_by_id[edge.right_segment_id]
            print(
                f"{left.key:>6} {right.key:>6} "
                f"{edge.shared_observed_frames:>6} "
                f"{(edge.overlap_median_distance_m or 0.0):>9.2f} "
                f"{(edge.movement_cosine_median or 0.0):>9.3f} "
                f"{edge.reid_score:>5.3f} "
                f"{edge.total_score:>5.3f}"
            )

    print()
    print("UNASSIGNED SEGMENTS")
    print("SEG       QUALITY     TEAM CONF  OBS    IN ALLOW   SPAN")
    print("-" * 82)
    for segment_id in sorted(
        result.unassigned_segment_ids,
        key=lambda sid: -segment_by_id[sid].inside_observed_count,
    ):
        segment = segment_by_id[segment_id]
        print(
            f"{segment.key:>6} "
            f"{segment.identity_quality.value:>12} "
            f"{segment.team_label.value:>8} "
            f"{segment.team_confidence:>4.2f} "
            f"{segment.observed_count:>4} "
            f"{segment.inside_observed_count:>5} "
            f"{segment.continuation_observed_count:>5} "
            f"{segment.observed_span_s:>6.2f}"
        )

    payload = {
        "version": "match-participants-v8-role-separated",
        "config": {
            "roster_size": args.roster_size,
            "field_length": args.field_length,
            "field_width": args.field_width,
            "continuation_margin": args.continuation_margin,
            "identity_segmentation": vars(segmentation_config),
            "physical_solver": vars(solver_config),
            "referee_classifier": vars(referee_config),
        },
        "kit_appearance": kit_report,
        "segmentations": {
            str(raw_id): segmentation.to_dict()
            for raw_id, segmentation in sorted(segmentations.items())
        },
        "segments": {
            str(segment.segment_id): {
                "segment_id": segment.segment_id,
                "key": segment.key,
                "raw_track_id": segment.raw_track_id,
                "segment_index": segment.segment_index,
                "team_label": segment.team_label.value,
                "team_confidence": segment.team_confidence,
                "identity_quality": segment.identity_quality.value,
                "identity_quality_score": segment.identity_quality_score,
                "start_frame": segment.start_frame,
                "end_frame": segment.end_frame,
                "start_timestamp": segment.start_timestamp,
                "end_timestamp": segment.end_timestamp,
                "observed_count": segment.observed_count,
                "inside_observed_count": segment.inside_observed_count,
                "continuation_observed_count": segment.continuation_observed_count,
            }
            for segment in segments
        },
        "physical_identity_result": result.to_dict(),
        "referee_classification": referee_result.to_dict(),
        "final_roles": {
            str(identity.identity_id): (
                "referee"
                if identity.identity_id == referee_result.referee_identity_id
                else "player"
            )
            for identity in result.identities
        },
        "player_team_labels": {
            "team_a": player_team_counts[RelativeTeamLabel.TEAM_A],
            "team_b": player_team_counts[RelativeTeamLabel.TEAM_B],
            "unknown": player_team_counts[RelativeTeamLabel.UNKNOWN],
        },
        "coverage": {
            "observed_selected": selected_observed,
            "observed_total": total_observed,
            "inside_selected": selected_inside,
            "inside_total": total_inside,
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print()
    print(f"report : {args.output.resolve()}")
    print("Audit only: production tracking and preview were not modified.")


if __name__ == "__main__":
    main()
