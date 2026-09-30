"""Hito 6B V1: anchored persistent-actor assignment with explicit noise.

Tracking V1 remains immutable evidence.  This solver works at the canonical
*tracklet* level because physical identities are conservative evidence groups,
not guaranteed indivisible people across an entire clip.

The model is deliberately asymmetric:

* original V1 tracklet ``cannot_link`` constraints stay HARD;
* every tracklet is assigned to one persistent actor slot or ``noise``;
* actor slots are inferred from a high-confidence directly-observed population
  frame (``observed``/``fused`` only; interpolated points never create actors);
* published V1 identity links are STRONG SOFT evidence;
* Tracking V1 ``soft-link-v1`` scores are secondary soft evidence;
* the optimization is lexicographic:
    1. minimise dropped observations;
    2. minimise the number of rejected tracklets;
    3. preserve as many published V1 identity links as possible;
    4. maximise soft association evidence;
    5. prefer the lower-uncertainty anchor frame.

No football-specific actor count is hard-coded.  The number of actor slots is
the high-confidence active-population estimate produced by Hito 6A.  This
solver is for short windows; full-match tracking will later stitch overlapping
windows so substitutions can create actor births/deaths without forcing the
whole match to contain a fixed number of unique people.

This stage is diagnostics only.  It does not overwrite tracking/v1/final.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix


SOLVER_VERSION = "anchored-actor-assignment-v1"
PHYSICAL_GRAPH_VERSION = "physical-compatibility-v1"
SOFT_LINK_VERSION = "soft-link-v1"
IDENTITY_PUBLISH_VERSION = "identity-publish-v1"
FINAL_PUBLISH_VERSION = "final-publish-v1"
PERSISTENT_CANDIDATE_VERSION = "persistent-actor-candidates-v1.1"


def _finite_float(name: str, value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise ValueError("pair requires two distinct tracklets")
    return (a, b) if a < b else (b, a)


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "p10": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _observation_count(row: Mapping[str, Any]) -> int:
    value = row.get("observation_count")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    observations = row.get("observations")
    if isinstance(observations, list) and observations:
        return len(observations)
    detections = row.get("detections")
    if isinstance(detections, list) and detections:
        return len(detections)
    return 1


def _tracklet_times(row: Mapping[str, Any]) -> tuple[float | None, float | None]:
    start = row.get("start_time_s")
    end = row.get("end_time_s")
    if start is not None and end is not None:
        return _finite_float("start_time_s", start), _finite_float("end_time_s", end)
    observations = row.get("observations")
    if isinstance(observations, list) and observations:
        times = [
            _finite_float("observation timestamp", item["timestamp_s"])
            for item in observations
            if isinstance(item, Mapping) and item.get("timestamp_s") is not None
        ]
        if times:
            return min(times), max(times)
    return None, None


def _mean_confidence(row: Mapping[str, Any]) -> float | None:
    value = row.get("mean_confidence")
    if value is not None:
        return _finite_float("mean_confidence", value)
    observations = row.get("observations")
    if isinstance(observations, list):
        values = [
            _finite_float("observation confidence", item["confidence"])
            for item in observations
            if isinstance(item, Mapping) and item.get("confidence") is not None
        ]
        if values:
            return float(np.mean(np.asarray(values, dtype=np.float64)))
    return None


@dataclass(frozen=True, slots=True)
class PersistentActorAssignmentConfig:
    max_anchor_configurations: int = 12
    allowed_population_confidences: tuple[str, ...] = ("high",)
    max_window_s: float = 120.0
    time_limit_s: float | None = 60.0
    mip_rel_gap: float = 0.0
    direct_origins: tuple[str, ...] = ("observed", "fused")

    def __post_init__(self) -> None:
        if self.max_anchor_configurations < 1:
            raise ValueError("max_anchor_configurations must be >= 1")
        if not self.allowed_population_confidences:
            raise ValueError("allowed_population_confidences must not be empty")
        if self.max_window_s <= 0:
            raise ValueError("max_window_s must be > 0")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be > 0 when supplied")
        if self.mip_rel_gap < 0:
            raise ValueError("mip_rel_gap must be >= 0")
        if not self.direct_origins:
            raise ValueError("direct_origins must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_anchor_configurations": self.max_anchor_configurations,
            "allowed_population_confidences": list(self.allowed_population_confidences),
            "max_window_s": self.max_window_s,
            "time_limit_s": self.time_limit_s,
            "mip_rel_gap": self.mip_rel_gap,
            "direct_origins": list(self.direct_origins),
        }


@dataclass(frozen=True, slots=True)
class AnchorSlot:
    actor_id: str
    anchor_identity_id: str
    tracklet_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AnchorConfiguration:
    frame_index: int
    timestamp_s: float
    uncertainty_sum_m: float
    slots: tuple[AnchorSlot, ...]

    @property
    def actor_count(self) -> int:
        return len(self.slots)

    @property
    def anchor_tracklet_ids(self) -> tuple[str, ...]:
        return tuple(
            tracklet_id
            for slot in self.slots
            for tracklet_id in slot.tracklet_ids
        )

    @property
    def signature(self) -> tuple[tuple[str, tuple[str, ...]], ...]:
        return tuple(
            (slot.anchor_identity_id, slot.tracklet_ids)
            for slot in self.slots
        )


@dataclass(frozen=True, slots=True)
class RewardPair:
    tracklet_a: str
    tracklet_b: str
    v1_identity_link: bool
    selected_backbone_bridge: bool
    soft_score: float
    soft_rank_score: float | None
    soft_evidence_confidence: float | None
    soft_link_kind: str | None

    @property
    def key(self) -> tuple[str, str]:
        return (self.tracklet_a, self.tracklet_b)


@dataclass(frozen=True, slots=True)
class AssignmentSolution:
    anchor: AnchorConfiguration
    assignment: Mapping[str, str | None]
    noise_tracklets: tuple[str, ...]
    dropped_observations: int
    preserved_v1_identity_links: int
    soft_evidence_sum: float
    optimization_status: int
    optimization_message: str


@dataclass(frozen=True, slots=True)
class _ModelLayout:
    tracklet_ids: tuple[str, ...]
    actor_ids: tuple[str, ...]
    x_count: int
    noise_offset: int
    same_offset: int | None = None
    reward_pair_count: int = 0

    @property
    def tracklet_count(self) -> int:
        return len(self.tracklet_ids)

    @property
    def actor_count(self) -> int:
        return len(self.actor_ids)

    @property
    def variable_count(self) -> int:
        return self.x_count + self.tracklet_count + self.reward_pair_count

    def xvar(self, tracklet_index: int, actor_index: int) -> int:
        return tracklet_index * self.actor_count + actor_index

    def nvar(self, tracklet_index: int) -> int:
        return self.noise_offset + tracklet_index

    def svar(self, reward_index: int) -> int:
        if self.same_offset is None:
            raise RuntimeError("layout has no reward variables")
        return self.same_offset + reward_index


def _milp_options(config: PersistentActorAssignmentConfig) -> dict[str, Any]:
    options: dict[str, Any] = {"mip_rel_gap": config.mip_rel_gap}
    if config.time_limit_s is not None:
        options["time_limit"] = config.time_limit_s
    return options


def _load_tracklets(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for raw in rows:
        row = dict(raw)
        tracklet_id = str(row["tracklet_id"])
        if not tracklet_id:
            raise ValueError("tracklet_id must not be empty")
        if tracklet_id in result:
            raise ValueError(f"duplicate tracklet: {tracklet_id}")
        result[tracklet_id] = row
    if not result:
        raise ValueError("persistent actor assignment received no tracklets")
    return result


def _load_identity_assignments(
    rows: Sequence[Mapping[str, Any]],
    tracklets: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for row in rows:
        tracklet_id = str(row["tracklet_id"])
        identity_id = str(row["identity_id"])
        if tracklet_id not in tracklets:
            raise ValueError(f"identity assignment references unknown tracklet {tracklet_id}")
        if tracklet_id in result:
            raise ValueError(f"duplicate identity assignment: {tracklet_id}")
        result[tracklet_id] = identity_id
    if set(result) != set(tracklets):
        missing = sorted(set(tracklets) - set(result))
        extra = sorted(set(result) - set(tracklets))
        raise ValueError(
            "identity assignments must cover tracklets exactly; "
            f"missing={missing[:10]} extra={extra[:10]}"
        )
    return result


def _load_hard_pairs(
    rows: Sequence[Mapping[str, Any]],
    tracklets: Mapping[str, Mapping[str, Any]],
) -> set[tuple[str, str]]:
    result: set[tuple[str, str]] = set()
    for row in rows:
        if row.get("status") != "cannot_link":
            continue
        pair = _canonical_pair(str(row["tracklet_a"]), str(row["tracklet_b"]))
        if pair[0] not in tracklets or pair[1] not in tracklets:
            raise ValueError(f"hard pair references unknown tracklet: {pair}")
        if pair in result:
            raise ValueError(f"duplicate hard cannot-link: {pair}")
        result.add(pair)
    return result


def _extract_anchor_configurations(
    trajectory_rows: Sequence[Mapping[str, Any]],
    *,
    population_count: int,
    direct_origins: Sequence[str],
    max_configurations: int,
    known_tracklets: set[str],
) -> tuple[AnchorConfiguration, ...]:
    if population_count < 1:
        raise ValueError("population_count must be >= 1")
    direct = set(direct_origins)
    by_frame: dict[int, list[dict[str, Any]]] = {}

    for trajectory in trajectory_rows:
        identity_id = str(trajectory["identity_id"])
        points = trajectory.get("points")
        if not isinstance(points, list):
            raise ValueError(f"{identity_id} trajectory points must be a list")
        for point in points:
            if not isinstance(point, Mapping):
                raise ValueError(f"{identity_id} contains non-object point")
            if str(point.get("origin")) not in direct:
                continue
            frame = int(point["frame_index"])
            source_tracklets = tuple(
                sorted(str(value) for value in point.get("source_tracklet_ids", []))
            )
            if not source_tracklets:
                raise ValueError(
                    f"direct point {identity_id} frame {frame} has no source tracklets"
                )
            unknown = [value for value in source_tracklets if value not in known_tracklets]
            if unknown:
                raise ValueError(
                    f"direct point {identity_id} frame {frame} references unknown tracklets {unknown}"
                )
            by_frame.setdefault(frame, []).append(
                {
                    "identity_id": identity_id,
                    "timestamp_s": _finite_float("timestamp_s", point["timestamp_s"]),
                    "tracklet_ids": source_tracklets,
                    "position_std_m": _finite_float(
                        "position_std_m", point.get("position_std_m", 999.0)
                    ),
                }
            )

    candidates: list[AnchorConfiguration] = []
    seen_signatures: set[tuple[tuple[str, tuple[str, ...]], ...]] = set()

    for frame, rows in by_frame.items():
        identities = [str(row["identity_id"]) for row in rows]
        if len(rows) != population_count or len(set(identities)) != population_count:
            continue
        ordered = sorted(rows, key=lambda row: str(row["identity_id"]))
        used_tracklets: dict[str, str] = {}
        slots: list[AnchorSlot] = []
        valid = True
        for actor_index, row in enumerate(ordered, start=1):
            identity_id = str(row["identity_id"])
            tracklet_ids = tuple(row["tracklet_ids"])
            for tracklet_id in tracklet_ids:
                previous = used_tracklets.get(tracklet_id)
                if previous is not None and previous != identity_id:
                    valid = False
                    break
                used_tracklets[tracklet_id] = identity_id
            if not valid:
                break
            slots.append(
                AnchorSlot(
                    actor_id=f"actor-{actor_index:03d}",
                    anchor_identity_id=identity_id,
                    tracklet_ids=tracklet_ids,
                )
            )
        if not valid:
            continue
        timestamps = [float(row["timestamp_s"]) for row in ordered]
        candidate = AnchorConfiguration(
            frame_index=frame,
            timestamp_s=float(np.median(np.asarray(timestamps, dtype=np.float64))),
            uncertainty_sum_m=float(
                sum(float(row["position_std_m"]) for row in ordered)
            ),
            slots=tuple(slots),
        )
        if candidate.signature in seen_signatures:
            continue
        seen_signatures.add(candidate.signature)
        candidates.append(candidate)

    candidates.sort(key=lambda item: (item.uncertainty_sum_m, item.frame_index))
    return tuple(candidates[:max_configurations])


def _build_reward_pairs(
    *,
    identity_link_rows: Sequence[Mapping[str, Any]],
    scored_link_rows: Sequence[Mapping[str, Any]],
    tracklets: Mapping[str, Mapping[str, Any]],
    hard_pairs: set[tuple[str, str]],
) -> tuple[RewardPair, ...]:
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for row in identity_link_rows:
        pair = _canonical_pair(str(row["tracklet_a"]), str(row["tracklet_b"]))
        if pair[0] not in tracklets or pair[1] not in tracklets:
            raise ValueError(f"identity link references unknown tracklet: {pair}")
        if pair in hard_pairs:
            raise ValueError(f"published identity link is also a hard cannot-link: {pair}")
        payload = merged.setdefault(
            pair,
            {
                "v1_identity_link": False,
                "selected_backbone_bridge": False,
                "soft_score": 0.0,
                "soft_rank_score": None,
                "soft_evidence_confidence": None,
                "soft_link_kind": None,
            },
        )
        payload["v1_identity_link"] = True
        payload["selected_backbone_bridge"] = bool(
            payload["selected_backbone_bridge"]
            or row.get("selected_backbone_bridge", False)
        )

    for row in scored_link_rows:
        if row.get("scoring_version") not in {None, SOFT_LINK_VERSION}:
            raise ValueError("scored link version mismatch")
        pair = _canonical_pair(str(row["tracklet_a"]), str(row["tracklet_b"]))
        if pair[0] not in tracklets or pair[1] not in tracklets:
            raise ValueError(f"soft link references unknown tracklet: {pair}")
        if pair in hard_pairs:
            raise ValueError(f"soft link is also a hard cannot-link: {pair}")
        rank_score = _clamp01(_finite_float("rank_score", row["rank_score"]))
        evidence_confidence = _clamp01(
            _finite_float("evidence_confidence", row["evidence_confidence"])
        )
        effective = rank_score * evidence_confidence
        payload = merged.setdefault(
            pair,
            {
                "v1_identity_link": False,
                "selected_backbone_bridge": False,
                "soft_score": 0.0,
                "soft_rank_score": None,
                "soft_evidence_confidence": None,
                "soft_link_kind": None,
            },
        )
        if effective >= float(payload["soft_score"]):
            payload["soft_score"] = effective
            payload["soft_rank_score"] = rank_score
            payload["soft_evidence_confidence"] = evidence_confidence
            payload["soft_link_kind"] = str(row.get("link_kind"))

    result = [
        RewardPair(
            tracklet_a=pair[0],
            tracklet_b=pair[1],
            v1_identity_link=bool(payload["v1_identity_link"]),
            selected_backbone_bridge=bool(payload["selected_backbone_bridge"]),
            soft_score=float(payload["soft_score"]),
            soft_rank_score=(
                None
                if payload["soft_rank_score"] is None
                else float(payload["soft_rank_score"])
            ),
            soft_evidence_confidence=(
                None
                if payload["soft_evidence_confidence"] is None
                else float(payload["soft_evidence_confidence"])
            ),
            soft_link_kind=(
                None
                if payload["soft_link_kind"] is None
                else str(payload["soft_link_kind"])
            ),
        )
        for pair, payload in sorted(merged.items())
    ]
    return tuple(result)


def _anchor_bounds(
    *,
    layout: _ModelLayout,
    anchor: AnchorConfiguration,
    tracklet_index: Mapping[str, int],
    actor_index: Mapping[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    lower = np.zeros(layout.variable_count, dtype=np.float64)
    upper = np.ones(layout.variable_count, dtype=np.float64)
    anchored: dict[str, str] = {}

    for slot in anchor.slots:
        actor = actor_index[slot.actor_id]
        for tracklet_id in slot.tracklet_ids:
            previous = anchored.get(tracklet_id)
            if previous is not None and previous != slot.actor_id:
                raise ValueError(f"{tracklet_id} anchors two actor slots")
            anchored[tracklet_id] = slot.actor_id
            t = tracklet_index[tracklet_id]
            lower[layout.nvar(t)] = 0.0
            upper[layout.nvar(t)] = 0.0
            for other_actor in range(layout.actor_count):
                idx = layout.xvar(t, other_actor)
                value = 1.0 if other_actor == actor else 0.0
                lower[idx] = value
                upper[idx] = value
    return lower, upper


def _base_constraints(
    *,
    layout: _ModelLayout,
    tracklet_index: Mapping[str, int],
    hard_pairs: Iterable[tuple[str, str]],
) -> tuple[list[int], list[int], list[float], list[float], list[float], int]:
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    lb: list[float] = []
    ub: list[float] = []
    r = 0

    for tracklet_id in layout.tracklet_ids:
        t = tracklet_index[tracklet_id]
        for actor in range(layout.actor_count):
            rows.append(r)
            cols.append(layout.xvar(t, actor))
            data.append(1.0)
        rows.append(r)
        cols.append(layout.nvar(t))
        data.append(1.0)
        lb.append(1.0)
        ub.append(1.0)
        r += 1

    for a, b in sorted(hard_pairs):
        ta = tracklet_index[a]
        tb = tracklet_index[b]
        for actor in range(layout.actor_count):
            rows.extend([r, r])
            cols.extend([layout.xvar(ta, actor), layout.xvar(tb, actor)])
            data.extend([1.0, 1.0])
            lb.append(-np.inf)
            ub.append(1.0)
            r += 1

    return rows, cols, data, lb, ub, r


def _decode_assignment(
    *,
    values: np.ndarray,
    layout: _ModelLayout,
) -> tuple[dict[str, str | None], tuple[str, ...]]:
    assignment: dict[str, str | None] = {}
    noise: list[str] = []
    for t, tracklet_id in enumerate(layout.tracklet_ids):
        if values[layout.nvar(t)] > 0.5:
            assignment[tracklet_id] = None
            noise.append(tracklet_id)
            continue
        actor_values = [values[layout.xvar(t, actor)] for actor in range(layout.actor_count)]
        actor = int(np.argmax(np.asarray(actor_values, dtype=np.float64)))
        if actor_values[actor] < 0.5:
            raise RuntimeError(f"solver left {tracklet_id} unassigned")
        assignment[tracklet_id] = layout.actor_ids[actor]
    return assignment, tuple(sorted(noise))


def _solve_min_noise(
    *,
    tracklets: Mapping[str, Mapping[str, Any]],
    hard_pairs: set[tuple[str, str]],
    anchor: AnchorConfiguration,
    config: PersistentActorAssignmentConfig,
) -> AssignmentSolution | None:
    tracklet_ids = tuple(sorted(tracklets))
    actor_ids = tuple(slot.actor_id for slot in anchor.slots)
    layout = _ModelLayout(
        tracklet_ids=tracklet_ids,
        actor_ids=actor_ids,
        x_count=len(tracklet_ids) * len(actor_ids),
        noise_offset=len(tracklet_ids) * len(actor_ids),
    )
    tracklet_index = {value: index for index, value in enumerate(tracklet_ids)}
    actor_index = {value: index for index, value in enumerate(actor_ids)}
    lower, upper = _anchor_bounds(
        layout=layout,
        anchor=anchor,
        tracklet_index=tracklet_index,
        actor_index=actor_index,
    )
    rows, cols, data, lb, ub, r = _base_constraints(
        layout=layout,
        tracklet_index=tracklet_index,
        hard_pairs=hard_pairs,
    )
    matrix = coo_matrix(
        (data, (rows, cols)), shape=(r, layout.variable_count)
    ).tocsr()

    # Exact lexicographic objective: one additional dropped observation costs
    # more than rejecting every tracklet in the clip.
    multiplier = len(tracklet_ids) + 1
    objective = np.zeros(layout.variable_count, dtype=np.float64)
    for t, tracklet_id in enumerate(tracklet_ids):
        objective[layout.nvar(t)] = (
            _observation_count(tracklets[tracklet_id]) * multiplier + 1
        )

    result = milp(
        c=objective,
        integrality=np.ones(layout.variable_count),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
        ),
        options=_milp_options(config),
    )
    if not result.success or result.x is None:
        return None
    assignment, noise = _decode_assignment(values=result.x, layout=layout)
    dropped = sum(_observation_count(tracklets[value]) for value in noise)
    return AssignmentSolution(
        anchor=anchor,
        assignment=assignment,
        noise_tracklets=noise,
        dropped_observations=dropped,
        preserved_v1_identity_links=0,
        soft_evidence_sum=0.0,
        optimization_status=int(result.status),
        optimization_message=str(result.message),
    )


def _solve_best_evidence(
    *,
    tracklets: Mapping[str, Mapping[str, Any]],
    hard_pairs: set[tuple[str, str]],
    reward_pairs: Sequence[RewardPair],
    anchor: AnchorConfiguration,
    fixed_dropped_observations: int,
    fixed_noise_count: int,
    config: PersistentActorAssignmentConfig,
) -> AssignmentSolution | None:
    tracklet_ids = tuple(sorted(tracklets))
    actor_ids = tuple(slot.actor_id for slot in anchor.slots)
    x_count = len(tracklet_ids) * len(actor_ids)
    noise_offset = x_count
    same_offset = noise_offset + len(tracklet_ids)
    layout = _ModelLayout(
        tracklet_ids=tracklet_ids,
        actor_ids=actor_ids,
        x_count=x_count,
        noise_offset=noise_offset,
        same_offset=same_offset,
        reward_pair_count=len(reward_pairs),
    )
    tracklet_index = {value: index for index, value in enumerate(tracklet_ids)}
    actor_index = {value: index for index, value in enumerate(actor_ids)}
    lower, upper = _anchor_bounds(
        layout=layout,
        anchor=anchor,
        tracklet_index=tracklet_index,
        actor_index=actor_index,
    )
    rows, cols, data, lb, ub, r = _base_constraints(
        layout=layout,
        tracklet_index=tracklet_index,
        hard_pairs=hard_pairs,
    )

    # Keep stage-1 optimum exactly fixed.
    for t, tracklet_id in enumerate(tracklet_ids):
        rows.append(r)
        cols.append(layout.nvar(t))
        data.append(float(_observation_count(tracklets[tracklet_id])))
    lb.append(float(fixed_dropped_observations))
    ub.append(float(fixed_dropped_observations))
    r += 1

    for t in range(len(tracklet_ids)):
        rows.append(r)
        cols.append(layout.nvar(t))
        data.append(1.0)
    lb.append(float(fixed_noise_count))
    ub.append(float(fixed_noise_count))
    r += 1

    # same[p] is rewarded only when both endpoints are assigned to the same
    # non-noise actor.  Because all rewards are non-negative, upper-bound
    # equivalence is enough: the objective will set same[p]=1 whenever allowed.
    for reward_index, pair in enumerate(reward_pairs):
        ta = tracklet_index[pair.tracklet_a]
        tb = tracklet_index[pair.tracklet_b]
        svar = layout.svar(reward_index)

        rows.extend([r, r])
        cols.extend([svar, layout.nvar(ta)])
        data.extend([1.0, 1.0])
        lb.append(-np.inf)
        ub.append(1.0)
        r += 1

        rows.extend([r, r])
        cols.extend([svar, layout.nvar(tb)])
        data.extend([1.0, 1.0])
        lb.append(-np.inf)
        ub.append(1.0)
        r += 1

        for actor in range(layout.actor_count):
            # same <= 1 - x_a + x_b
            rows.extend([r, r, r])
            cols.extend([svar, layout.xvar(ta, actor), layout.xvar(tb, actor)])
            data.extend([1.0, 1.0, -1.0])
            lb.append(-np.inf)
            ub.append(1.0)
            r += 1

            # same <= 1 + x_a - x_b
            rows.extend([r, r, r])
            cols.extend([svar, layout.xvar(ta, actor), layout.xvar(tb, actor)])
            data.extend([1.0, -1.0, 1.0])
            lb.append(-np.inf)
            ub.append(1.0)
            r += 1

    matrix = coo_matrix(
        (data, (rows, cols)), shape=(r, layout.variable_count)
    ).tocsr()

    total_soft_possible = sum(pair.soft_score for pair in reward_pairs)
    v1_multiplier = total_soft_possible + 1.0
    objective = np.zeros(layout.variable_count, dtype=np.float64)
    for reward_index, pair in enumerate(reward_pairs):
        reward = pair.soft_score
        if pair.v1_identity_link:
            reward += v1_multiplier
        objective[layout.svar(reward_index)] = -reward

    result = milp(
        c=objective,
        integrality=np.ones(layout.variable_count),
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(
            matrix,
            np.asarray(lb, dtype=np.float64),
            np.asarray(ub, dtype=np.float64),
        ),
        options=_milp_options(config),
    )
    if not result.success or result.x is None:
        return None

    assignment, noise = _decode_assignment(values=result.x, layout=layout)
    preserved = 0
    soft_sum = 0.0
    for reward_index, pair in enumerate(reward_pairs):
        if result.x[layout.svar(reward_index)] <= 0.5:
            continue
        if pair.v1_identity_link:
            preserved += 1
        soft_sum += pair.soft_score

    return AssignmentSolution(
        anchor=anchor,
        assignment=assignment,
        noise_tracklets=noise,
        dropped_observations=fixed_dropped_observations,
        preserved_v1_identity_links=preserved,
        soft_evidence_sum=soft_sum,
        optimization_status=int(result.status),
        optimization_message=str(result.message),
    )


def _hard_violation_count(
    assignment: Mapping[str, str | None],
    hard_pairs: Iterable[tuple[str, str]],
) -> int:
    return sum(
        assignment[a] is not None
        and assignment[a] == assignment[b]
        for a, b in hard_pairs
    )


def _window_duration_s(trajectory_rows: Sequence[Mapping[str, Any]]) -> float:
    timestamps = [
        _finite_float("timestamp_s", point["timestamp_s"])
        for trajectory in trajectory_rows
        for point in trajectory.get("points", [])
        if isinstance(point, Mapping) and point.get("timestamp_s") is not None
    ]
    if not timestamps:
        raise ValueError("final trajectories contain no timestamps")
    return max(timestamps) - min(timestamps)


def solve_anchored_actor_assignment(
    *,
    tracklet_rows: Sequence[Mapping[str, Any]],
    physical_candidate_rows: Sequence[Mapping[str, Any]],
    trajectory_rows: Sequence[Mapping[str, Any]],
    identity_assignment_rows: Sequence[Mapping[str, Any]],
    identity_link_rows: Sequence[Mapping[str, Any]],
    scored_link_rows: Sequence[Mapping[str, Any]],
    population: Mapping[str, Any],
    config: PersistentActorAssignmentConfig | None = None,
) -> dict[str, Any]:
    """Solve one short-window persistent-actor assignment.

    The returned payload is serialization-ready diagnostics.  No Tracking V1
    artifact is mutated here.
    """

    config = config or PersistentActorAssignmentConfig()
    estimate = int(population["estimated_population"])
    confidence = str(population["confidence"])
    if estimate < 1:
        raise ValueError("estimated_population must be >= 1")
    if confidence not in config.allowed_population_confidences:
        raise ValueError(
            "anchored actor assignment requires population confidence in "
            f"{config.allowed_population_confidences}, got {confidence!r}"
        )
    window_duration = _window_duration_s(trajectory_rows)
    if window_duration > config.max_window_s:
        raise ValueError(
            f"window {window_duration:.3f}s exceeds max_window_s={config.max_window_s}; "
            "use overlapping windows and later stitching"
        )

    tracklets = _load_tracklets(tracklet_rows)
    identity_assignment = _load_identity_assignments(
        identity_assignment_rows, tracklets
    )
    hard_pairs = _load_hard_pairs(physical_candidate_rows, tracklets)
    anchors = _extract_anchor_configurations(
        trajectory_rows,
        population_count=estimate,
        direct_origins=config.direct_origins,
        max_configurations=config.max_anchor_configurations,
        known_tracklets=set(tracklets),
    )
    if not anchors:
        raise ValueError(
            "no directly-supported anchor frame matches estimated_population"
        )
    reward_pairs = _build_reward_pairs(
        identity_link_rows=identity_link_rows,
        scored_link_rows=scored_link_rows,
        tracklets=tracklets,
        hard_pairs=hard_pairs,
    )

    stage1_rows: list[dict[str, Any]] = []
    stage1_solutions: list[AssignmentSolution] = []
    for anchor in anchors:
        solution = _solve_min_noise(
            tracklets=tracklets,
            hard_pairs=hard_pairs,
            anchor=anchor,
            config=config,
        )
        row = {
            "schema_version": 1,
            "solver_version": SOLVER_VERSION,
            "frame_index": anchor.frame_index,
            "timestamp_s": anchor.timestamp_s,
            "uncertainty_sum_m": anchor.uncertainty_sum_m,
            "actor_count": anchor.actor_count,
            "anchor_tracklet_count": len(anchor.anchor_tracklet_ids),
            "feasible": solution is not None,
            "dropped_observations": (
                None if solution is None else solution.dropped_observations
            ),
            "noise_tracklet_count": (
                None if solution is None else len(solution.noise_tracklets)
            ),
            "evidence_optimized": False,
            "preserved_v1_identity_link_count": None,
            "soft_evidence_sum": None,
            "selected": False,
        }
        stage1_rows.append(row)
        if solution is not None:
            stage1_solutions.append(solution)

    if not stage1_solutions:
        raise RuntimeError("all anchor configurations are infeasible even with noise")

    best_dropped = min(item.dropped_observations for item in stage1_solutions)
    best_noise_count = min(
        len(item.noise_tracklets)
        for item in stage1_solutions
        if item.dropped_observations == best_dropped
    )
    tied = [
        item
        for item in stage1_solutions
        if item.dropped_observations == best_dropped
        and len(item.noise_tracklets) == best_noise_count
    ]

    evidence_solutions: list[AssignmentSolution] = []
    for stage1 in tied:
        solution = _solve_best_evidence(
            tracklets=tracklets,
            hard_pairs=hard_pairs,
            reward_pairs=reward_pairs,
            anchor=stage1.anchor,
            fixed_dropped_observations=best_dropped,
            fixed_noise_count=best_noise_count,
            config=config,
        )
        if solution is not None:
            evidence_solutions.append(solution)
            for row in stage1_rows:
                if row["frame_index"] == solution.anchor.frame_index:
                    row["evidence_optimized"] = True
                    row["preserved_v1_identity_link_count"] = (
                        solution.preserved_v1_identity_links
                    )
                    row["soft_evidence_sum"] = solution.soft_evidence_sum
                    break

    if not evidence_solutions:
        raise RuntimeError("evidence optimization failed for every minimum-noise anchor")

    chosen = max(
        evidence_solutions,
        key=lambda item: (
            item.preserved_v1_identity_links,
            item.soft_evidence_sum,
            -item.anchor.uncertainty_sum_m,
            -item.anchor.frame_index,
        ),
    )
    for row in stage1_rows:
        if row["frame_index"] == chosen.anchor.frame_index:
            row["selected"] = True
            break

    hard_violations = _hard_violation_count(chosen.assignment, hard_pairs)
    if hard_violations:
        raise AssertionError(f"solver produced {hard_violations} hard violations")

    actor_rows: list[dict[str, Any]] = []
    for slot in chosen.anchor.slots:
        members = sorted(
            tracklet_id
            for tracklet_id, actor_id in chosen.assignment.items()
            if actor_id == slot.actor_id
        )
        source_identities = sorted({identity_assignment[value] for value in members})
        starts_ends = [_tracklet_times(tracklets[value]) for value in members]
        starts = [value[0] for value in starts_ends if value[0] is not None]
        ends = [value[1] for value in starts_ends if value[1] is not None]
        actor_rows.append(
            {
                "schema_version": 1,
                "solver_version": SOLVER_VERSION,
                "actor_id": slot.actor_id,
                "anchor_frame_index": chosen.anchor.frame_index,
                "anchor_identity_id": slot.anchor_identity_id,
                "anchor_tracklet_ids": list(slot.tracklet_ids),
                "tracklet_ids": members,
                "tracklet_count": len(members),
                "source_identity_ids": source_identities,
                "source_identity_count": len(source_identities),
                "observation_count": sum(
                    _observation_count(tracklets[value]) for value in members
                ),
                "start_timestamp_s": min(starts) if starts else None,
                "end_timestamp_s": max(ends) if ends else None,
                "provisional": True,
                "canonical": False,
            }
        )

    assignment_rows: list[dict[str, Any]] = []
    noise_rows: list[dict[str, Any]] = []
    anchor_tracklets = set(chosen.anchor.anchor_tracklet_ids)
    for tracklet_id in sorted(tracklets):
        actor_id = chosen.assignment[tracklet_id]
        row = tracklets[tracklet_id]
        start, end = _tracklet_times(row)
        payload = {
            "schema_version": 1,
            "solver_version": SOLVER_VERSION,
            "tracklet_id": tracklet_id,
            "actor_id": actor_id,
            "status": "noise" if actor_id is None else "assigned",
            "source_identity_id": identity_assignment[tracklet_id],
            "observation_count": _observation_count(row),
            "start_timestamp_s": start,
            "end_timestamp_s": end,
            "mean_confidence": _mean_confidence(row),
            "anchor_tracklet": tracklet_id in anchor_tracklets,
        }
        assignment_rows.append(payload)
        if actor_id is None:
            noise_rows.append(dict(payload))

    identity_rows: list[dict[str, Any]] = []
    grouped_identity_tracklets: dict[str, list[str]] = collections.defaultdict(list)
    for tracklet_id, identity_id in identity_assignment.items():
        grouped_identity_tracklets[identity_id].append(tracklet_id)
    identity_status_counts: collections.Counter[str] = collections.Counter()
    for identity_id in sorted(grouped_identity_tracklets):
        members = sorted(grouped_identity_tracklets[identity_id])
        actor_ids = sorted(
            {
                chosen.assignment[value]
                for value in members
                if chosen.assignment[value] is not None
            }
        )
        noise_members = [value for value in members if chosen.assignment[value] is None]
        if not actor_ids:
            status = "noise_only"
        elif len(actor_ids) > 1:
            status = "split"
        elif noise_members:
            status = "partial_noise"
        else:
            status = "preserved"
        identity_status_counts[status] += 1
        identity_rows.append(
            {
                "schema_version": 1,
                "solver_version": SOLVER_VERSION,
                "identity_id": identity_id,
                "tracklet_ids": members,
                "actor_ids": actor_ids,
                "noise_tracklet_ids": noise_members,
                "status": status,
            }
        )

    reward_rows: list[dict[str, Any]] = []
    for pair in reward_pairs:
        actor_a = chosen.assignment[pair.tracklet_a]
        actor_b = chosen.assignment[pair.tracklet_b]
        same = actor_a is not None and actor_a == actor_b
        reward_rows.append(
            {
                "schema_version": 1,
                "solver_version": SOLVER_VERSION,
                "tracklet_a": pair.tracklet_a,
                "tracklet_b": pair.tracklet_b,
                "v1_identity_link": pair.v1_identity_link,
                "selected_backbone_bridge": pair.selected_backbone_bridge,
                "soft_score": pair.soft_score,
                "soft_rank_score": pair.soft_rank_score,
                "soft_evidence_confidence": pair.soft_evidence_confidence,
                "soft_link_kind": pair.soft_link_kind,
                "same_actor": same,
                "actor_id": actor_a if same else None,
            }
        )

    total_observations = sum(_observation_count(row) for row in tracklets.values())
    v1_link_total = sum(pair.v1_identity_link for pair in reward_pairs)
    summary = {
        "schema_version": 1,
        "solver_version": SOLVER_VERSION,
        "canonical": False,
        "provisional": True,
        "tracking_v1_unchanged": True,
        "population_estimate": estimate,
        "population_confidence": confidence,
        "window_duration_s": window_duration,
        "actor_count": len(actor_rows),
        "tracklet_count": len(tracklets),
        "assigned_tracklet_count": len(tracklets) - len(chosen.noise_tracklets),
        "noise_tracklet_count": len(chosen.noise_tracklets),
        "total_observation_count": total_observations,
        "dropped_observation_count": chosen.dropped_observations,
        "dropped_observation_fraction": (
            chosen.dropped_observations / total_observations
            if total_observations
            else 0.0
        ),
        "hard_cannot_link_count": len(hard_pairs),
        "hard_constraint_violation_count": 0,
        "anchor_configuration_count": len(anchors),
        "minimum_noise_anchor_count": len(tied),
        "chosen_anchor_frame_index": chosen.anchor.frame_index,
        "chosen_anchor_timestamp_s": chosen.anchor.timestamp_s,
        "chosen_anchor_uncertainty_sum_m": chosen.anchor.uncertainty_sum_m,
        "chosen_anchor_tracklet_count": len(chosen.anchor.anchor_tracklet_ids),
        "reward_pair_count": len(reward_pairs),
        "v1_identity_link_count": v1_link_total,
        "preserved_v1_identity_link_count": chosen.preserved_v1_identity_links,
        "preserved_v1_identity_link_fraction": (
            chosen.preserved_v1_identity_links / v1_link_total
            if v1_link_total
            else None
        ),
        "soft_evidence_sum": chosen.soft_evidence_sum,
        "identity_status_counts": dict(sorted(identity_status_counts.items())),
        "actor_tracklet_count": _distribution(
            [float(row["tracklet_count"]) for row in actor_rows]
        ),
        "actor_source_identity_count": _distribution(
            [float(row["source_identity_count"]) for row in actor_rows]
        ),
        "noise_observation_count": _distribution(
            [float(row["observation_count"]) for row in noise_rows]
        ),
        "config": config.to_dict(),
        "optimization": {
            "lexicographic_order": [
                "min_dropped_observations",
                "min_noise_tracklets",
                "max_preserved_v1_identity_links",
                "max_soft_evidence",
                "min_anchor_uncertainty",
            ],
            "stage2_status": chosen.optimization_status,
            "stage2_message": chosen.optimization_message,
        },
    }

    return {
        "actors": actor_rows,
        "assignments": assignment_rows,
        "noise_tracklets": noise_rows,
        "identity_membership": identity_rows,
        "reward_pairs": reward_rows,
        "anchor_candidates": stage1_rows,
        "summary": summary,
    }
