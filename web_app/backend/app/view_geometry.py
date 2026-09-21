"""Shader transport and cached Python geometry contexts for native video views."""
from __future__ import annotations

import uuid
from pathlib import Path

import numpy as np

from geometry.veo_metadata import load_veo_metadata
from geometry.panorama_landmarks import field_landmarks
from geometry.tracking_ground_v3 import TRACKING_ASPECT, TRACKING_FPS, logic_matrices

ROOT = Path(__file__).resolve().parents[3]
contexts: dict = {}
keys: dict = {}


def view_context(calibration_path: str, tracking_available: bool = False) -> dict:
    path = Path(calibration_path)
    key = (str(path.resolve()), path.stat().st_mtime_ns, tracking_available)
    if key in keys:
        return keys[key]
    metadata = load_veo_metadata(path)
    uniforms = {}
    for name, matrix in {
        "camera2world": metadata.camera2world,
        "camera2presentation": np.linalg.inv(metadata.camera2presentation),
        "lens_left2camera": metadata.left.lens2camera,
        "lens_right2camera": metadata.right.lens2camera,
        "intrinsic_left": metadata.left.intrinsic,
        "intrinsic_right": metadata.right.intrinsic,
    }.items():
        uniforms[name] = {"matrix": int(matrix.shape[0]), "value": matrix.flatten(order="F").tolist()}
    for name, value in metadata.va_presentation.items():
        uniforms[f"va_{name}"] = {"value": value}
    for lens in (metadata.left, metadata.right):
        distortion = np.zeros(8)
        distortion[:min(len(lens.distortion), 8)] = lens.distortion[:8]
        for name, value in zip(("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6"), distortion):
            uniforms[f"distortion_{lens.name}.{name}"] = {"value": float(value)}
    identifier = uuid.uuid4().hex
    contexts[identifier] = metadata
    tracking_geometry = {"available": tracking_available, "reason": "" if tracking_available else "Tracking field projection unavailable: camera directions missing."}
    if tracking_available:
        try:
            _camera2logic, logic2camera = logic_matrices(metadata)
            tracking_geometry.update({"logic2camera": logic2camera.flatten(order="F").tolist(), "fps": TRACKING_FPS, "aspect": TRACKING_ASPECT})
        except (ValueError, np.linalg.LinAlgError):
            tracking_geometry = {"available": False, "reason": "Tracking field projection unavailable: invalid calibration."}
    result = {
        "id": identifier, "uniforms": uniforms,
        "vertex_shader": (ROOT / "reference/veo_panoramic.vert.glsl").read_text(),
        "fragment_shader": (ROOT / "reference/veo_panoramic.frag.glsl").read_text(),
        "field_length": metadata.field_length, "field_width": metadata.field_width,
        "landmarks": [value.to_dict() for value in field_landmarks(metadata.field_length, metadata.field_width).values()],
        "tracking_geometry": tracking_geometry,
    }
    keys[key] = result
    return result
