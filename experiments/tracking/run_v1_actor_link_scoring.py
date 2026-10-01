#!/usr/bin/env python3
"""Run Tracking V1 Hito 7E.1 conservative actor-link scoring.

Consumes:
* 7B.3 sparse actor candidates;
* 7C.1B anonymous kit assignments;
* 7D split-positive and simultaneous-negative ReID calibration.

Produces scores only.  No tracklet is merged and no identity is assigned.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from tracking.actors.link_scoring import (  # noqa: E402
    ACTOR_LINK_SCORING_VERSION,
    ActorLinkScoringConfig,
    build_actor_link_scores,
)
from tracking.clip_context import resolve_tracking_clip_context  # noqa: E402


def _iter_jsonl(path: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            payload = json.loads(raw)
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"{path}:{line_number} must contain an object"
                )
            result.append(dict(payload))
    return result


def _write_json(
    path: Path,
    payload: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(payload),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                )
                + "\n"
            )
            count += 1
    return count


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    context = resolve_tracking_clip_context(
        args.data_root,
        args.match_id,
        args.clip_id,
    )
    root = context.tracking_root

    candidate_root = (
        root
        / "diagnostics"
        / "actor_candidates"
        / "sparse-actor-candidates-v1"
    )
    kit_root = (
        root
        / "diagnostics"
        / "kit_clustering"
        / "kit-clustering-v1"
    )
    calibration_root = (
        root
        / "diagnostics"
        / "reid_calibration"
        / "reid-overlap-calibration-v1"
    )

    candidates_path = candidate_root / "candidates.jsonl"
    kit_path = kit_root / "assignments.jsonl"
    positive_path = calibration_root / "split_positive.jsonl"
    negative_path = (
        calibration_root / "simultaneous_negative.jsonl"
    )

    for required in (
        candidates_path,
        kit_path,
        positive_path,
        negative_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    candidates = _iter_jsonl(candidates_path)
    kit = _iter_jsonl(kit_path)
    positives = _iter_jsonl(positive_path)
    negatives = _iter_jsonl(negative_path)

    links, summary = build_actor_link_scores(
        candidates,
        kit_assignments=kit,
        split_positive_rows=positives,
        simultaneous_negative_rows=negatives,
        config=ActorLinkScoringConfig(),
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (
            root
            / "diagnostics"
            / "actor_link_scoring"
            / ACTOR_LINK_SCORING_VERSION
        )
    )
    links_path = output_dir / "scored_links.jsonl"
    summary_path = output_dir / "summary.json"
    index_path = output_dir / "index.json"

    count = _write_jsonl(
        links_path,
        (link.to_dict() for link in links),
    )

    # Extra counts useful for selecting the conservative 7E.2 backbone.
    score_bands = Counter()
    for link in links:
        for threshold in (0.70, 0.75, 0.80, 0.85, 0.90):
            if link.rank_score >= threshold:
                score_bands[f"ge_{threshold:.2f}"] += 1
                if link.mutual_best:
                    score_bands[
                        f"mutual_best_ge_{threshold:.2f}"
                    ] += 1
                if link.mutual_top3:
                    score_bands[
                        f"mutual_top3_ge_{threshold:.2f}"
                    ] += 1

    summary.update(
        {
            "match_id": context.match_id,
            "clip_id": context.clip_id,
            "score_band_counts": dict(
                sorted(score_bands.items())
            ),
            "source_candidates": str(candidates_path),
            "source_kit_assignments": str(kit_path),
            "source_split_positive": str(positive_path),
            "source_simultaneous_negative": str(
                negative_path
            ),
        }
    )
    _write_json(summary_path, summary)
    _write_json(
        index_path,
        {
            "schema_version": 1,
            "scoring_version": ACTOR_LINK_SCORING_VERSION,
            "scored_links": links_path.name,
            "summary": summary_path.name,
            "scored_link_count": count,
            "identity_decisions": False,
        },
    )

    print(
        "[actor-link-scoring-7E.1] "
        f"match={context.match_id} "
        f"clip={context.clip_id}"
    )
    print(
        "[actor-link-scoring-7E.1] "
        f"candidates={count} "
        f"families={json.dumps(summary['family_counts'], sort_keys=True)}"
    )
    print(
        "[actor-link-scoring-7E.1] "
        f"rank_score={json.dumps(summary['rank_score'], sort_keys=True)}"
    )
    print(
        "[actor-link-scoring-7E.1] "
        f"confidence={json.dumps(summary['evidence_confidence'], sort_keys=True)}"
    )
    print(
        "[actor-link-scoring-7E.1] "
        f"mutual_best={summary['mutual_best_count']} "
        f"mutual_top3={summary['mutual_top3_count']} "
        f"mutual_best_by_family="
        f"{json.dumps(summary['mutual_best_by_family'], sort_keys=True)}"
    )
    print(
        "[actor-link-scoring-7E.1] "
        f"bands={json.dumps(summary['score_band_counts'], sort_keys=True)}"
    )
    print(
        "[actor-link-scoring-7E.1] "
        f"artifact={index_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
