"""Clip-scoped orchestration for the experimental V8 canonical pipeline.

V8 is deliberately limited to a short 11v11 clip with a fixed 22-player plus
one-referee roster.  It does not support substitutions or a full match.
"""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REID_WEIGHTS = ROOT / "person_detect/model_weights/reid/osnet_ain_x1_0_msmt17.pth"


def run_v8_pipeline(*, video: Path, calibration: Path, detector_weights: Path,
                    detection_threshold: float, output_dir: Path,
                    progress: Callable[[str], None] | None = None) -> dict:
    """Run existing V8 phases with all inputs and outputs owned by one clip.

    No duplicate attachment is supplied: overrides are empty for every new
    clip and remain audit-only.
    """
    if not DEFAULT_REID_WEIGHTS.is_file():
        raise FileNotFoundError(f"V8 ReID weights not found: {DEFAULT_REID_WEIGHTS}")
    output_dir.mkdir(parents=True, exist_ok=True)
    crops = output_dir / "crops"
    reid = output_dir / "reid"
    raw = output_dir / "raw_tracking.jsonl"
    participants = output_dir / "participants_report.json"
    preview = output_dir / "preview_frames.jsonl"
    summary = output_dir / "summary.json"

    def invoke(stage: str, script: str, *args: str) -> None:
        if progress:
            progress(stage)
        subprocess.run([sys.executable, str(ROOT / "experiments/tracking" / script), *args], cwd=ROOT, check=True)

    invoke("v8-crops", "audit_reid_crops.py", "--video", str(video), "--calibration", str(calibration),
           "--weights", str(detector_weights), "--threshold", str(detection_threshold),
           "--field-margin-m", "1.5", "--out-dir", str(crops))
    source_raw = crops / "_tracking_tmp/tracking_frames.jsonl"
    shutil.copy2(source_raw, raw)
    invoke("v8-reid", "audit_reid_similarity.py", "--crop-dir", str(crops), "--weights", str(DEFAULT_REID_WEIGHTS),
           "--output-dir", str(reid))
    invoke("v8-identities", "audit_match_participants.py", "--tracking-frames", str(raw),
           "--embedding-cache", str(reid / "reid_embeddings.npz"), "--crop-dir", str(crops), "--output", str(participants))
    invoke("v8-publication", "build_canonical_preview.py", "--tracking-frames", str(raw),
           "--participants-report", str(participants), "--output-frames", str(preview), "--output-summary", str(summary))
    metadata = {
        "tracking_mode": "v8_canonical", "detection_threshold": detection_threshold,
        "detector_cache_floor": None, "duplicate_overrides": [], "video": str(video),
        "calibration": str(calibration), "fixed_roster_limit": "short 11v11 clip; 22 players + 1 referee; no substitutions",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return {"preview_frames": preview, "summary": summary, "participants_report": participants}
