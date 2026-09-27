import json
import subprocess

from web_app.backend.app.clip_jobs import _transcode_web_video


def test_ultra_video_transcode_is_web_compatible(tmp_path):
    intermediate = tmp_path / "ultra.intermediate.mp4"
    output = tmp_path / "ultra_base.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=32x16:rate=15", "-t", "0.2", "-c:v", "mpeg4", str(intermediate)], check=True)
    _transcode_web_video(intermediate, output)
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name,pix_fmt", "-of", "json", str(output)], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    assert stream["codec_name"] == "h264"
    assert stream["pix_fmt"] == "yuv420p"
