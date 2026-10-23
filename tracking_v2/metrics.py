"""Evaluation against a separately filled manual reference CSV."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .team_discovery import TEAM_A, TEAM_B, UNKNOWN


TRUTH_LABELS = (TEAM_A, TEAM_B, UNKNOWN)
PRED_LABELS = (TEAM_A, TEAM_B, UNKNOWN)


def _normalise_label(value: str | None) -> str | None:
    if value is None:
        return None
    label = value.strip().upper()
    if label in {TEAM_A, TEAM_B}:
        return label
    if label in {UNKNOWN, "UNKNOWN", "SPECIAL"}:
        return UNKNOWN
    return None


def _team_mapping(rows: list[dict[str, str]]) -> dict[str, str]:
    known = [
        row for row in rows
        if _normalise_label(row.get("manual_team")) in {TEAM_A, TEAM_B}
        and row.get("predicted_team") in {TEAM_A, TEAM_B}
    ]
    options = (
        {TEAM_A: TEAM_A, TEAM_B: TEAM_B},
        {TEAM_A: TEAM_B, TEAM_B: TEAM_A},
    )
    if not known:
        return options[0]
    return max(
        options,
        key=lambda mapping: sum(
            mapping[row["predicted_team"]] == _normalise_label(row.get("manual_team"))
            for row in known
        ),
    )


def _summarise(rows: list[dict[str, str]], mapping: dict[str, str]) -> dict[str, Any]:
    if not rows:
        return {
            "labeled_samples": 0,
            "precision_when_assigned": None,
            "accuracy_including_unknown": None,
            "coverage": None,
            "confusion": {truth: {pred: 0 for pred in PRED_LABELS} for truth in TRUTH_LABELS},
        }

    confusion = {truth: {pred: 0 for pred in PRED_LABELS} for truth in TRUTH_LABELS}
    assigned = 0
    correct = 0
    correct_assigned = 0
    for row in rows:
        truth = _normalise_label(row.get("manual_team")) or UNKNOWN
        predicted = row.get("predicted_team", UNKNOWN)
        if predicted in {TEAM_A, TEAM_B}:
            predicted = mapping[predicted]
            assigned += 1
        else:
            predicted = UNKNOWN
        confusion[truth][predicted] += 1
        is_correct = truth == predicted
        correct += int(is_correct)
        if predicted != UNKNOWN and is_correct:
            correct_assigned += 1

    return {
        "labeled_samples": len(rows),
        "precision_when_assigned": correct_assigned / assigned if assigned else None,
        "accuracy_including_unknown": correct / len(rows),
        "coverage": assigned / len(rows),
        "confusion": confusion,
    }


def evaluate_rows(
    labels: list[dict[str, str]],
    assignments: list[dict[str, str]],
) -> dict[str, Any]:
    """Align anonymous clusters to manual team labels, then report metrics."""
    pred_by_id = {row["sample_id"]: row for row in assignments}
    rows: list[dict[str, str]] = []
    for label in labels:
        manual = _normalise_label(label.get("manual_team"))
        if manual is None:
            continue
        predicted = pred_by_id.get(label.get("sample_id", ""), {}).get("predicted_team", UNKNOWN)
        rows.append({**label, "manual_team": manual, "predicted_team": predicted})

    mapping = _team_mapping(rows)
    by_condition: dict[str, dict[str, Any]] = {}
    for dimension in ("field_context", "lighting", "role", "representative_view", "scale_bin"):
        groups: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            value = (row.get(dimension) or "unknown").strip() or "unknown"
            groups[value].append(row)
        by_condition[dimension] = {
            value: _summarise(group, mapping)
            for value, group in sorted(groups.items())
        }

    result = _summarise(rows, mapping)
    result["cluster_to_manual_team"] = mapping
    result["by_condition"] = by_condition
    result["known_team_confusions"] = {
        "TEAM_A_to_TEAM_B": result["confusion"][TEAM_A][TEAM_B],
        "TEAM_B_to_TEAM_A": result["confusion"][TEAM_B][TEAM_A],
    }
    result["unknown_false_assignments"] = (
        result["confusion"][UNKNOWN][TEAM_A] + result["confusion"][UNKNOWN][TEAM_B]
    )
    return result


def evaluate_files(assignments_path: Path, labels_path: Path) -> dict[str, Any]:
    with assignments_path.open(newline="", encoding="utf-8") as stream:
        assignments = list(csv.DictReader(stream))
    with labels_path.open(newline="", encoding="utf-8") as stream:
        labels = list(csv.DictReader(stream))
    return evaluate_rows(labels, assignments)


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate metrics from manual sample labels.")
    parser.add_argument("--assignments", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate_files(args.assignments, args.labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
