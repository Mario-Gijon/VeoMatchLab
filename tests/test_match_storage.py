import json
from concurrent.futures import ThreadPoolExecutor

from web_app.backend.app.storage import WorkspaceStore


def test_migration_is_idempotent_and_preserves_original_paths(tmp_path):
    legacy = {"matches": [{"id": "match_old", "name": "Original", "media": [{"id": "media_old", "path": "/original/input.mp4"}], "calibration": None}]}
    path = tmp_path / "workspaces.json"
    path.write_text(json.dumps(legacy))
    original = path.read_bytes()
    store = WorkspaceStore(tmp_path)
    store.update_match("match_old", lambda match: match.update(name="Updated"))
    loaded = WorkspaceStore(tmp_path).match("match_old")
    assert loaded["name"] == "Updated"
    assert loaded["media"][0]["path"] == "/original/input.mp4"
    assert path.read_bytes() == original
    assert (tmp_path / "matches/match_old/match.json").exists()


def test_concurrent_asset_updates_do_not_lose_media(tmp_path):
    store = WorkspaceStore(tmp_path)
    match = store.create_match("New", "", None)
    def update(index):
        store.update_match(match["id"], lambda item: item["media"].append({"id": str(index)}))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(update, range(20)))
    assert len(WorkspaceStore(tmp_path).match(match["id"])["media"]) == 20
