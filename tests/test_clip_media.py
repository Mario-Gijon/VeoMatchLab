import subprocess
import time

import cv2

from web_app.backend.app.clip_jobs import ClipAnalysisService
from web_app.backend.app.services import media_record
from web_app.backend.app.storage import WorkspaceStore


def test_physical_clip_is_browser_compatible_and_scoped_to_interval(tmp_path):
    source_path = tmp_path / "dual.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=160x80:rate=10", "-t", "5", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(source_path)], check=True)
    store = WorkspaceStore(tmp_path / "workspace")
    match = store.create_match("Clip", "", None)
    source = media_record(source_path, source="path")
    source.update(id="dual", role="dual_lens")
    store.attach_media(match["id"], source)
    clip = store.create_clip(match["id"], source["id"], 1.0, 3.0, "Three seconds")
    service = ClipAnalysisService(store)
    job = service.start_generation(store.match(match["id"]), clip, source)
    for _ in range(100):
        if job.status not in {"queued", "running"}:
            break
        time.sleep(.05)
    assert job.status == "completed", job.error
    stored = store.clip(match["id"], clip["id"])
    generated = store.media(stored["clip_media_id"])[1]
    path = generated["path"]
    capture = cv2.VideoCapture(path)
    assert capture.isOpened()
    assert capture.get(cv2.CAP_PROP_FRAME_WIDTH) == source["width"]
    assert capture.get(cv2.CAP_PROP_FRAME_HEIGHT) == source["height"]
    assert capture.get(cv2.CAP_PROP_FPS) > 0
    assert abs(capture.get(cv2.CAP_PROP_FRAME_COUNT) / capture.get(cv2.CAP_PROP_FPS) - 3.0) < .25
    capture.release()
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "default=noprint_wrappers=1", path], check=True, capture_output=True, text=True).stdout
    assert "codec_name=h264" in probe and "pix_fmt=yuv420p" in probe
    assert source_path.exists()
    assert path.endswith("/source.mp4")
