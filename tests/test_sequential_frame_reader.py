import cv2
import numpy as np
import pytest

from experiments.virtual_camera_coverage import coverage


@pytest.fixture
def tiny_video(tmp_path, monkeypatch):
    path = tmp_path / "frames.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (16, 8))
    assert writer.isOpened()
    for index in range(8):
        writer.write(np.full((8, 16, 3), index * 20, dtype=np.uint8))
    writer.release()
    monkeypatch.setattr(coverage, "SOURCE_W", 16)
    monkeypatch.setattr(coverage, "SOURCE_H", 8)
    return path


def test_sequential_reader_is_frame_exact_with_legacy_reader(tiny_video):
    timestamps = [0.0, 0.013, 0.051, 0.149, 0.251, 0.399]
    with coverage.SequentialFrameReader(tiny_video) as reader:
        for timestamp in timestamps:
            legacy_frame, legacy_info = coverage.read_frame(tiny_video, timestamp)
            sequential_frame, sequential_info = reader.read(timestamp)
            assert np.array_equal(legacy_frame, sequential_frame)
            assert sequential_info["fps"] == legacy_info["fps"]
            assert sequential_info["requested_frame"] == legacy_info["requested_frame"]
            assert sequential_info["actual_frame"] == legacy_info["actual_frame"]
            assert sequential_info["actual_time_s"] == legacy_info["actual_time_s"]


def test_sequential_reader_opens_capture_once(tiny_video, monkeypatch):
    original = coverage.cv2.VideoCapture
    opened = []

    def counted_video_capture(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(coverage.cv2, "VideoCapture", counted_video_capture)
    with coverage.SequentialFrameReader(tiny_video) as reader:
        for timestamp in [0.0, 0.1, 0.2, 0.3]:
            reader.read(timestamp)
    assert len(opened) == 1
