"""Conservative, explainable classification for player network resources."""

from __future__ import annotations

from pathlib import PurePosixPath
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .models import VeoCandidate


SENSITIVE_QUERY_KEYS = {"token", "signature", "sig", "expires", "expiry", "key", "credential", "policy", "x-amz-signature", "x-amz-credential", "x-amz-security-token"}


def sanitize_url(value: str) -> str:
    """Remove fragments and redact credentials before diagnostics or UI output."""
    parsed = urlsplit(value)
    # Query names evolve (CloudFront, Azure SAS, OAuth, etc.). Redact every
    # value rather than maintaining an inevitably incomplete credential list.
    query = [(key, "REDACTED") for key, _ in parse_qsl(parsed.query, keep_blank_values=True)]
    return urlunsplit((parsed.scheme, parsed.hostname or "", parsed.path, urlencode(query), ""))


def resource_filename(value: str) -> str:
    path = PurePosixPath(urlsplit(value).path)
    return path.name or "unnamed-resource"


def classify_resource(
    identifier: str,
    url: str,
    content_type: str | None = None,
    probe: dict | None = None,
    discovery_phases: list[str] | None = None,
) -> VeoCandidate:
    """Score roles from independent naming, MIME, and optional probe evidence."""
    probe = probe or {}
    name = resource_filename(url)
    lower = f"{urlsplit(url).path} {name}".lower()
    content = (content_type or "").lower()
    phases = list(dict.fromkeys(discovery_phases or []))
    reasons: list[str] = []
    score = {"panorama": 0.0, "ball_tracking": 0.0, "calibration": 0.0, "camera_directions": 0.0, "manifest": 0.0}
    if name.lower() == "detections_constrained.det" or name.lower().endswith("/detections_constrained.det"):
        score["camera_directions"] = 1.0
        reasons.append("Veo constrained camera-directions resource")
    if name.lower().endswith(".veo") or "camera_alignment" in lower:
        score["calibration"] += 0.95; reasons.append("calibration filename/path signal")
    if any(term in lower for term in ("panorama", "pano", "transcode", "dual_lens", "dual-lens")):
        score["panorama"] += 0.55; reasons.append("panorama/transcode path signal")
    if any(term in lower for term in ("ball_tracking", "ball-tracking", "follow_camera", "follow-camera", "followcam")):
        score["ball_tracking"] += 0.75; reasons.append("ball-tracking/follow-camera path signal")
    if "initial" in phases and name.lower() == "video.mp4":
        score["ball_tracking"] += 0.9; reasons.append("video.mp4 discovered in the initial/default player view")
    if "after_interactive" in phases and name.lower().startswith("transcode-") and name.lower().endswith(".mp4"):
        score["panorama"] += 0.9; reasons.append("transcode resource discovered after Interactive activation")
    if "after_interactive" in phases and (name.lower().endswith(".veo") or "camera_alignment" in lower):
        score["calibration"] += 0.1; reasons.append("calibration resource discovered after Interactive activation")
    if name.lower().endswith((".m3u8", ".mpd")) or "application/dash" in content or "mpegurl" in content:
        score["manifest"] += 0.7; reasons.append("streaming-manifest signal")
    if name.lower().endswith(".mp4") or content.startswith("video/"):
        score["panorama"] += 0.1; score["ball_tracking"] += 0.1; reasons.append("video resource signal")
    width, height = probe.get("width"), probe.get("height")
    if width == 2048 and height == 2048:
        score["panorama"] += 0.4; reasons.append("2048×2048 square-video signal")
    role, confidence = max(score.items(), key=lambda item: item[1])
    if confidence < 0.2:
        role = "unknown"
    if score["manifest"]:
        role, confidence = "manifest", score["manifest"]
    return VeoCandidate(
        identifier=identifier, url=urlunsplit((*urlsplit(url)[:4], "")), safe_url=sanitize_url(url), filename=name, content_type=content_type,
        role=role, confidence=min(confidence, 1.0), reasons=reasons or ["No reliable role signal"],
        discovery_phases=phases,
        width=width, height=height, fps=probe.get("fps"), duration_s=probe.get("duration_s"),
    )


def ranked_candidates(candidates: list[VeoCandidate], role: str) -> list[VeoCandidate]:
    return sorted((candidate for candidate in candidates if candidate.role == role), key=lambda candidate: candidate.confidence, reverse=True)
