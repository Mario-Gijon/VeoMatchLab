#!/usr/bin/env python3
"""Audit Sprint B.2 conservative role inference.

Inputs:
  * tracking_frames.jsonl
  * Sprint B.1 team_appearance_report.json

Outputs:
  * role_inference_report.json
  * console rankings for left GK, right GK, referee
  * per-track geometry + appearance + role scores

This audit does NOT force exactly 2 goalkeepers + 1 referee and does NOT modify
tracks.person, V2.7 identity resolution or frontend state.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tracking.models import (  # noqa: E402
    PositionKind,
    TrackLifecycle,
    TrackSample,
)
from tracking.role_inference import (  # noqa: E402
    FootballRole,
    RoleInferenceConfig,
    TrackAppearanceEvidence,
    infer_roles,
)


DEFAULT_FIELD_LENGTH_M = 105.0
DEFAULT_FIELD_WIDTH_M = 68.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tracking-frames",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--appearance-report",
        type=Path,
        default=Path(
            "experiments/tracking/team_appearance_debug/"
            "team_appearance_report.json"
        ),
    )
    parser.add_argument(
        "--field-length-m",
        type=float,
        default=DEFAULT_FIELD_LENGTH_M,
    )
    parser.add_argument(
        "--field-width-m",
        type=float,
        default=DEFAULT_FIELD_WIDTH_M,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def load_trajectories(
    path: Path,
) -> dict[int, tuple[TrackSample, ...]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    trajectories: dict[int, list[TrackSample]] = defaultdict(list)

    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line:
                continue

            try:
                row = json.loads(line)
                frame_index = int(row["frame_index"])
                timestamp = float(
                    row.get("timestamp_s", row.get("timestamp"))
                )
                raw_tracks = row["tracks"]
            except (
                json.JSONDecodeError,
                KeyError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"{path}:{line_number}: malformed tracking frame"
                ) from exc

            for raw in raw_tracks:
                try:
                    track_id = int(raw["track_id"])
                    trajectories[track_id].append(
                        TrackSample(
                            track_id=track_id,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            x=float(raw["x"]),
                            z=float(raw["z"]),
                            vx=float(raw.get("vx", 0.0)),
                            vz=float(raw.get("vz", 0.0)),
                            lifecycle=TrackLifecycle(
                                str(raw["lifecycle"])
                            ),
                            position_kind=PositionKind(
                                str(raw["position_kind"])
                            ),
                            measurement_id=raw.get("measurement_id"),
                            confidence=(
                                float(raw["confidence"])
                                if raw.get("confidence") is not None
                                else None
                            ),
                        )
                    )
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                ) as exc:
                    raise ValueError(
                        f"{path}:{line_number}: malformed track {raw!r}"
                    ) from exc

    return {
        track_id: tuple(
            sorted(
                samples,
                key=lambda sample: (
                    sample.frame_index,
                    sample.timestamp,
                ),
            )
        )
        for track_id, samples in sorted(trajectories.items())
    }


def load_appearance(
    path: Path,
) -> dict[int, TrackAppearanceEvidence]:
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = json.loads(
        path.read_text(
            encoding="utf-8",
        )
    )
    rows = payload.get(
        "track_decisions",
        {},
    )

    result: dict[int, TrackAppearanceEvidence] = {}
    for raw_track_id, row in rows.items():
        track_id = int(raw_track_id)
        result[track_id] = TrackAppearanceEvidence.from_report_row(
            track_id,
            row,
        )

    return result


def _score_for_role(
    decision,
    role: FootballRole,
) -> float:
    return float(
        decision.scores.as_mapping()[role]
    )


def print_ranking(
    title: str,
    result,
    role: FootballRole,
    ranking: tuple[int, ...],
    *,
    top: int,
) -> None:
    print()
    print(title)
    print(
        f"{'RANK':>4} {'ID':>4} {'SCORE':>7} "
        f"{'DECISION':>18} {'APP':>9} {'PURITY':>7} "
        f"{'GX':>7} {'GZ':>7} {'ZONE':>6} {'CENT':>6}"
    )
    print("-" * 90)

    for rank, track_id in enumerate(
        ranking[:top],
        start=1,
    ):
        decision = result.decisions[track_id]
        geometry = decision.geometry

        if role is FootballRole.GOALKEEPER_LEFT:
            zone = geometry.left_goal_zone_fraction
        elif role is FootballRole.GOALKEEPER_RIGHT:
            zone = geometry.right_goal_zone_fraction
        else:
            zone = max(
                geometry.left_goal_zone_fraction,
                geometry.right_goal_zone_fraction,
            )

        print(
            f"{rank:>4} "
            f"{track_id:>4} "
            f"{_score_for_role(decision, role):>7.3f} "
            f"{decision.role.value:>18} "
            f"{decision.appearance_label.value:>9} "
            f"{decision.appearance_purity:>7.2f} "
            f"{geometry.median_x:>7.2f} "
            f"{geometry.median_z:>7.2f} "
            f"{zone:>6.2f} "
            f"{geometry.central_fraction:>6.2f}"
        )


def print_decisions(
    result,
) -> None:
    print()
    print("TRACK ROLE DECISIONS")
    print(
        f"{'ID':>4} {'ROLE':>18} {'BEST':>7} "
        f"{'RUNNER':>18} {'Rscore':>7} {'MARGIN':>7} "
        f"{'APP':>9} {'SUPPORT':>7}"
    )
    print("-" * 98)

    for track_id, decision in sorted(
        result.decisions.items()
    ):
        geometry = decision.geometry
        print(
            f"{track_id:>4} "
            f"{decision.role.value:>18} "
            f"{max(decision.scores.as_mapping().values()):>7.3f} "
            f"{(decision.runner_up_role.value if decision.runner_up_role else '-'):>18} "
            f"{decision.runner_up_score:>7.3f} "
            f"{decision.margin:>7.3f} "
            f"{decision.appearance_label.value:>9} "
            f"{geometry.support_score:>7.3f}"
        )


def main() -> None:
    args = parse_args()

    tracking_path = (
        args.tracking_frames
        .expanduser()
        .resolve()
    )
    appearance_path = (
        args.appearance_report
        .expanduser()
        .resolve()
    )

    trajectories = load_trajectories(
        tracking_path
    )
    appearance = load_appearance(
        appearance_path
    )

    config = RoleInferenceConfig(
        field_length_m=args.field_length_m,
        field_width_m=args.field_width_m,
    )

    result = infer_roles(
        trajectories,
        appearance,
        config=config,
    )

    print()
    print("Sprint B.2 conservative role inference")
    print(f"tracking tracks    : {len(trajectories)}")
    print(f"appearance tracks  : {len(appearance)}")
    print(f"role decisions     : {len(result.decisions)}")
    print(
        "policy             : no exact-role-count forcing; "
        "UNKNOWN allowed"
    )

    print_ranking(
        "LEFT GOALKEEPER CANDIDATES",
        result,
        FootballRole.GOALKEEPER_LEFT,
        result.left_goalkeeper_ranking,
        top=args.top,
    )
    print_ranking(
        "RIGHT GOALKEEPER CANDIDATES",
        result,
        FootballRole.GOALKEEPER_RIGHT,
        result.right_goalkeeper_ranking,
        top=args.top,
    )
    print_ranking(
        "REFEREE CANDIDATES",
        result,
        FootballRole.REFEREE,
        result.referee_ranking,
        top=args.top,
    )
    print_decisions(
        result
    )

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else appearance_path.with_name(
            "role_inference_report.json"
        )
    )

    payload = {
        "version": "tracking-role-inference-sprint-b2-v1",
        "tracking_frames": str(tracking_path),
        "appearance_report": str(appearance_path),
        "config": {
            field: getattr(
                config,
                field,
            )
            for field in config.__dataclass_fields__
        },
        **result.to_dict(),
    }

    output.write_text(
        json.dumps(
            payload,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print(f"report : {output}")
    print(
        "No role counts were forced and nothing was published to tracks.person."
    )


if __name__ == "__main__":
    main()
