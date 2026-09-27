"""Browser smoke check for physical clips: loading, play/pause and seeking."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
PORT = 8016


def request(url: str, method: str = "GET", payload: dict | None = None) -> dict:
    body = json.dumps(payload).encode() if payload is not None else None
    response = urlopen(Request(url, body, method=method, headers={"Content-Type": "application/json"}), timeout=10)
    return json.loads(response.read())


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="veo-clip-browser-") as directory:
        root = Path(directory)
        source = root / "dual.mp4"
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-loop", "1", "-i", str(ROOT / "tests/fixtures/images/stacked_frame.jpg"), "-t", "5", "-r", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source)], check=True)
        environment = {**os.environ, "VEO_MATCH_DATA_ROOT": str(root / "runtime")}
        server = subprocess.Popen([sys.executable, "-m", "uvicorn", "web_app.backend.app.main:app", "--host", "127.0.0.1", "--port", str(PORT)], cwd=ROOT, env=environment, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        base = f"http://127.0.0.1:{PORT}"
        try:
            for _ in range(100):
                try:
                    request(base + "/api/health")
                    break
                except OSError:
                    time.sleep(.05)
            match = request(base + "/api/matches", "POST", {"name": "Browser clip", "description": ""})
            source_media = request(f"{base}/api/matches/{match['id']}/media/path", "POST", {"path": str(source), "role": "dual_lens"})
            job = request(f"{base}/api/matches/{match['id']}/clips", "POST", {"source_media_id": source_media["id"], "start_s": 1, "duration_s": 3, "name": "Browser"})
            for _ in range(200):
                status = request(f"{base}/api/clip-jobs/{job['job_id']}")
                if status["status"] not in {"queued", "running"}:
                    break
                time.sleep(.05)
            assert status["status"] == "completed", status
            clip = request(f"{base}/api/matches/{match['id']}")["clips"][0]
            media = next(item for item in request(f"{base}/api/matches/{match['id']}")["media"] if item["id"] == clip["clip_media_id"])
            ranged = urlopen(Request(base + media["url"], headers={"Range": "bytes=0-99"}))
            assert ranged.status == 206 and ranged.headers.get_content_type() == "video/mp4"
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(base + "/api/health")
                result = page.evaluate("""async (src) => {
                  const video = document.createElement('video'); video.src = src; video.muted = true; document.body.append(video);
                  await new Promise((ok, bad) => { video.onloadedmetadata = ok; video.onerror = () => bad(new Error('video error')); });
                  const duration = video.duration; await video.play(); await new Promise(ok => setTimeout(ok, 350));
                  const advanced = video.currentTime; video.pause(); const paused = video.paused;
                  video.currentTime = 2; await new Promise(ok => video.onseeked = ok);
                  return {duration, advanced, paused, seek: video.currentTime};
                }""", base + media["url"])
                browser.close()
            assert 2.7 < result["duration"] < 3.3 and result["advanced"] > .05 and result["paused"] and abs(result["seek"] - 2) < .2, result
            request(f"{base}/api/matches/{match['id']}/clips/{clip['id']}", "DELETE")
            assert source.exists()
            print("physical clip browser play/pause/seek passed")
        finally:
            server.terminate()
            server.wait(timeout=10)


if __name__ == "__main__":
    main()
