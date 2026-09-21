import io
import json
from urllib.request import Request

import pytest

from web_app.backend.app.importers.veo import downloader, service
from web_app.backend.app.importers.veo.models import VeoInspection
from web_app.backend.app.importers.veo.classifier import classify_resource, sanitize_url
from web_app.backend.app.importers.veo.inspector import normalized_match_url, resource_links
from web_app.backend.app.storage import WorkspaceStore


class Response(io.BytesIO):
    status = 200
    def __init__(self, payload, length):
        super().__init__(payload)
        self.headers = {"Content-Length": str(length)}


@pytest.mark.parametrize("cancel, length, succeeds", [(False, 7, True), (False, 9, False), (True, 7, False)])
def test_streaming_download_integrity_and_partial_cleanup(tmp_path, monkeypatch, cancel, length, succeeds):
    monkeypatch.setattr(downloader, "public_url", lambda _: None)
    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"fixture", length)
    monkeypatch.setattr(downloader, "authorized_opener", lambda _: Opener())
    path = tmp_path / "original.mp4"
    progress = []
    if succeeds:
        downloader.download_stream("https://cdn.example/x.mp4", path, lambda n, total: progress.append((n, total)), lambda: cancel)
        assert path.read_bytes() == b"fixture"
        assert progress == [(7, 7)]
    else:
        with pytest.raises((InterruptedError, ValueError)):
            downloader.download_stream("https://cdn.example/x.mp4", path, lambda *args: None, lambda: cancel)
        assert not path.exists()
    assert not path.with_name(path.name + ".part").exists()


def test_cookie_scope_and_url_sanitization():
    opener = downloader.authorized_opener([{"domain": "app.veo.co", "path": "/matches", "secure": True, "name": "session", "value": "private", "expires": -1}])
    jar = next(handler.cookiejar for handler in opener.handlers if hasattr(handler, "cookiejar"))
    for url in ("https://cdn.example/matches/x", "https://sub.app.veo.co/matches/x", "http://app.veo.co/matches/x", "https://app.veo.co/elsewhere"):
        request = Request(url); jar.add_cookie_header(request)
        assert not request.has_header("Cookie")
    request = Request("https://app.veo.co/matches/x"); jar.add_cookie_header(request)
    assert request.get_header("Cookie") == "session=private"
    assert "secret" not in sanitize_url("https://user:secret@app.veo.co/x?unfamiliar=secret&sig=secret#secret")


def test_preexisting_partial_is_never_deleted(tmp_path, monkeypatch):
    monkeypatch.setattr(downloader, "public_url", lambda _: None)
    class Opener:
        def open(self, *args, **kwargs):
            return Response(b"new", 3)
    monkeypatch.setattr(downloader, "authorized_opener", lambda _: Opener())
    destination = tmp_path / "video.mp4"
    partial = tmp_path / "video.mp4.part"
    partial.write_bytes(b"previous recoverable download")
    with pytest.raises(FileExistsError):
        downloader.download_stream("https://cdn.example/video.mp4", destination, lambda *args: None, lambda: False)
    assert partial.read_bytes() == b"previous recoverable download"
    assert not destination.exists()


def test_inspection_restrictions_and_nested_json_links():
    for url in ["http://app.veo.co/a", "https://localhost/a", "https://veo.co.evil.test/a", "https://u:p@app.veo.co/a"]:
        with pytest.raises(ValueError):
            normalized_match_url(url)
    assert normalized_match_url("https://app.veo.co/matches/x#t=5").endswith("/x")
    assert resource_links({"data": [{"resource": "https://cdn.example/transcode.mp4?sig=private"}]}) == ["https://cdn.example/transcode.mp4?sig=private"]


def test_job_serialization_failure_and_cancel(tmp_path, monkeypatch):
    manager = service.VeoImportService(WorkspaceStore(tmp_path), tmp_path)
    candidate = classify_resource("one", "https://cdn.example/transcode.mp4?token=secret", "video/mp4")
    manager._inspections["inspection"] = VeoInspection("inspection", "https://app.veo.co/x", "https://app.veo.co/x", candidates=[candidate])
    def fail(*args):
        raise RuntimeError("https://cdn.example/transcode.mp4?token=secret")
    monkeypatch.setattr(service, "download_stream", fail)
    job = service.ImportJob("job", "inspection", {"panorama": "one"})
    manager._run_import(job)
    public = job.public_dict()
    assert public["status"] == "failed" and "secret" not in json.dumps(public)
    assert public["match_id"] is not None
    job = service.ImportJob("cancel", "inspection", {"panorama": "one"})
    job.cancelled.set()
    manager._run_import(job)
    assert job.public_dict()["status"] == "cancelled"
    with pytest.raises(ValueError):
        manager.start_import("inspection", {})
    with pytest.raises(ValueError):
        manager.start_import("inspection", {"panorama": "one", "ball_tracking": "one"})
