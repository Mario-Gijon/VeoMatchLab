#!/usr/bin/env python3
"""Import a saved discovery capture's datasets only. Never downloads media."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from web_app.backend.app.artifacts import read_json
from web_app.backend.app.importers.veo.datasets import match_key, persist_datasets, replay_capture
from web_app.backend.app.storage import WorkspaceStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--runtime-root", type=Path, required=True,
                        help="External VEO_MATCH_DATA_ROOT directory; never a repository path.")
    parser.add_argument("--match-id")
    parser.add_argument("--create-match", action="store_true", help="Create a dataset-only workspace if no matching one exists")
    args = parser.parse_args()
    root = args.capture.resolve()
    capture = read_json(root / "index.json", {})
    source_url = capture.get("match_url", "")
    key = match_key(source_url)
    if not key:
        parser.error("Capture index must identify a Veo match URL")
    store = WorkspaceStore(args.runtime_root.resolve())
    if args.match_id:
        match = store.match(args.match_id)
        if not match or match_key(match.get("metadata", {}).get("veo_url", "")) != key:
            parser.error("Selected workspace belongs to a different match (or has no verified Veo URL)")
    else:
        matches = [m for m in store.matches() if match_key(m.get("metadata", {}).get("veo_url", "")) == key]
        if len(matches) > 1:
            parser.error("Multiple matching workspaces; select --match-id")
        match = matches[0] if matches else None
    if not match and not args.create_match:
        parser.error("No matching workspace; use --create-match for a dataset-only workspace")
    responses = read_json(root / "network/responses.json", [])
    bodies = read_json(root / "network/saved_bodies.json", {})
    discovery = replay_capture(root, responses, bodies, root / "datasets")
    if not match:
        title = key
        for endpoint, relative in bodies.items():
            path = (root / relative).resolve()
            if match_key(endpoint) == key and urlsplit(endpoint).path.rstrip("/").endswith(f"/matches/{key}") and path.is_relative_to(root):
                metadata = read_json(path, {})
                if isinstance(metadata, dict) and isinstance(metadata.get("title"), str) and metadata["title"].strip():
                    title = metadata["title"].strip()
                    break
        match = store.create_match(title, "Datos de Veo importados desde una captura local; vídeos aún no importados.", None)
        store.update_match(match["id"], lambda item: item.update(metadata={"veo_url": source_url}))
    index = persist_datasets(store, match["id"], source_url, discovery.observations)
    print(json.dumps({"match_id": match["id"], "source_match": key, "datasets": [
        {"name": d["name"], "status": d["status"], "events": d.get("event_count", 0)} for d in index["datasets"].values()
    ]}, indent=2))


if __name__ == "__main__":
    main()
