"""
Runner for identity graph construction.

Consumes identity expansion outputs and generates graph diagnostics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tracking.actors.identity_graph import IdentityGraphBuilder


def _load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-id", required=True)
    parser.add_argument("--clip-id", required=True)
    parser.add_argument("--data-root", default="data")
    parser.add_argument(
        "--expansion-dir",
        default=None,
    )
    args = parser.parse_args()

    base = (
        Path(args.data_root)
        / "matches"
        / args.match_id
        / "clips"
        / args.clip_id
        / "tracking"
        / "v1"
        / "diagnostics"
        / "identity_expansion"
    )

    expansion_dir = (
        Path(args.expansion_dir)
        if args.expansion_dir
        else base / "seeded-identity-expansion-v2-family-aware"
    )

    output_dir = (
        Path(args.data_root)
        / "matches"
        / args.match_id
        / "clips"
        / args.clip_id
        / "tracking"
        / "v1"
        / "diagnostics"
        / "identity_graph"
        / "identity-graph-v1"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    builder = IdentityGraphBuilder()

    identities_path = expansion_dir / "identities.jsonl"
    links_path = expansion_dir / "selected_links.jsonl"

    nodes = []

    for row in _load_jsonl(identities_path):
        identity_id = (
            row.get("identity_id")
            or row.get("id")
            or row.get("identity")
        )

        if identity_id:
            builder.add_identity(identity_id)
            nodes.append(
                {
                    "identity_id": identity_id,
                }
            )

    edges = []

    for row in _load_jsonl(links_path):
        left = (
            row.get("left_identity")
            or row.get("source_identity")
            or row.get("source")
        )
        right = (
            row.get("right_identity")
            or row.get("target_identity")
            or row.get("target")
        )

        if not left or not right:
            continue

        builder.add_edge(
            left,
            right,
            float(row.get("score", 0.0)),
            row.get("family_relation", "neutral"),
        )

        edges.append(row)

    strong_components = builder.build_components(
        include_weak=False
    )

    recovery_components = builder.build_components(
        include_weak=True
    )

    with (output_dir / "nodes.jsonl").open("w") as f:
        for node in nodes:
            f.write(json.dumps(node) + "\n")

    with (output_dir / "edges.jsonl").open("w") as f:
        for edge in edges:
            f.write(json.dumps(edge) + "\n")

    with (output_dir / "strong_components.jsonl").open("w") as f:
        for component in strong_components:
            f.write(
                json.dumps(component.__dict__)
                + "\n"
            )

    with (output_dir / "recovery_components.jsonl").open("w") as f:
        for component in recovery_components:
            f.write(
                json.dumps(component.__dict__)
                + "\n"
            )

    index = {
        "schema_version": 1,
        "graph_version": "identity-graph-v1",
        "match_id": args.match_id,
        "clip_id": args.clip_id,
        "expansion_dir": str(expansion_dir),
        **builder.summary(),
        "strong_component_count": len(strong_components),
        "recovery_component_count": len(recovery_components),
        "outputs": {
            "nodes": "nodes.jsonl",
            "edges": "edges.jsonl",
            "strong_components": "strong_components.jsonl",
            "recovery_components": "recovery_components.jsonl",
        },
    }

    with (output_dir / "index.json").open("w") as f:
        json.dump(index, f, indent=2)

    print(
        "[identity-graph-7F.2] "
        f"nodes={index['node_count']} "
        f"edges={index['edge_count']} "
        f"strong={index['strong_edge_count']} "
        f"weak={index['weak_edge_count']}"
    )

    print(
        "[identity-graph-7F.2] "
        f"strong_components={len(strong_components)} "
        f"recovery_components={len(recovery_components)}"
    )

    print(
        "[identity-graph-7F.2] "
        f"artifact={output_dir / 'index.json'}"
    )


if __name__ == "__main__":
    main()
