"""Network-led, phase-aware Playwright inspection of an authorized Veo match."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request

from .classifier import classify_resource, sanitize_url
from .discovery import StructuredDiscovery, write_manifest
from .downloader import authorized_opener, public_url
from .models import VeoCandidate, VeoInspection


INITIAL_PHASE = "initial"
AFTER_INTERACTIVE_PHASE = "after_interactive"
MAX_RESOURCES = 100
MAX_JSON_RESPONSE_BYTES = 2_000_000
POST_INTERACTIVE_CAPTURE_MS = 30_000
ACTIVATION_OBSERVE_MS = 6_000
INTERACTIVE_LABEL = re.compile(r"^\s*(interactive|interactivo)\s*$", re.IGNORECASE)
PRIMARY_INTERACTIVE_SELECTOR = "button.interactive-player"
MEMBERSHIP_MODAL_SELECTOR = "[role='dialog'], dialog, [aria-modal='true']"
MEMBERSHIP_CLOSE_SELECTOR = 'button[aria-label="x"]'
LOGIN_MODAL_SELECTOR = ".modal.modal-login-prompt"
LOGIN_CLOSE_SELECTOR = 'button[aria-label="Close button"]'
MEMBERSHIP_TEXT = re.compile(r"become\s+a\s+member\s+of|request\s+membership", re.IGNORECASE)
MENU_LABEL = re.compile(
    r"interactive|interactivo|view|vista|camera|cámara|mode|modo|menu|more|más|options|opciones|"
    r"standard|estándar|panoramic|panorámica|tracking|seguimiento",
    re.IGNORECASE,
)


def resource_links(value, limit=200) -> list[str]:
    """Discover actual resource URLs supplied by player JSON, without guessing IDs."""
    found = []

    def visit(item, depth=0):
        if depth > 12 or len(found) >= limit:
            return
        if isinstance(item, str) and item.startswith("https://"):
            path = urlsplit(item).path.lower()
            if any(term in path for term in (".mp4", ".veo", ".det", ".m3u8", ".mpd", "camera_alignment")):
                found.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item[:200]:
                visit(child, depth + 1)

    visit(value)
    return found


def standard_video_camera_directions(value) -> list[tuple[str, str]]:
    """Return explicit camera-directions URLs paired with Standard/Follow videos."""
    pairs: list[tuple[str, str]] = []

    def visit(item, depth=0):
        if depth > 12:
            return
        if isinstance(item, dict):
            video_url = item.get("url")
            directions_url = item.get("camera_directions")
            render_type = str(item.get("render_type") or item.get("type") or "").strip().lower()
            if (
                isinstance(video_url, str)
                and video_url.startswith("https://")
                and isinstance(directions_url, str)
                and directions_url.startswith("https://")
                and render_type in {"standard", "follow", "follow_camera", "follow-camera"}
            ):
                pairs.append((video_url, directions_url))
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item[:500]:
                visit(child, depth + 1)

    visit(value)
    return list(dict.fromkeys(pairs))


def standard_follow_videos(value) -> list[tuple[str, str | None]]:
    """Explicit `/videos/` metadata is the authority for the Tracking source."""
    videos: list[tuple[str, str | None]] = []

    def visit(item, depth=0):
        if depth > 12:
            return
        if isinstance(item, dict):
            url = item.get("url")
            kind = str(item.get("render_type") or item.get("type") or "").strip().lower()
            directions = item.get("camera_directions")
            if isinstance(url, str) and url.startswith("https://") and kind in {"standard", "follow", "follow_camera", "follow-camera"}:
                videos.append((url, directions if isinstance(directions, str) and directions.startswith("https://") else None))
            for child in item.values():
                visit(child, depth + 1)
        elif isinstance(item, list):
            for child in item[:500]:
                visit(child, depth + 1)

    visit(value)
    return list(dict.fromkeys(videos))


def generated_highlights_videos(value) -> list[str]:
    """Accept only explicit compilation/reel evidence from the video or its parent."""
    videos: list[str] = []

    def visit(item, inherited_compilation=False, depth=0):
        if depth > 12:
            return
        if isinstance(item, dict):
            kind = str(item.get("render_type") or item.get("type") or item.get("kind") or "").lower()
            explicit_compilation = item.get("is_highlights_compilation") is True or item.get("is_compilation") is True or item.get("is_reel") is True or kind in {"highlights_compilation", "highlights_reel", "reel"}
            url = item.get("url")
            if (explicit_compilation or inherited_compilation) and isinstance(url, str) and url.startswith("https://") and urlsplit(url).path.lower().endswith(".mp4"):
                videos.append(url)
            for child in item.values():
                visit(child, inherited_compilation or explicit_compilation, depth + 1)
        elif isinstance(item, list):
            for child in item[:500]:
                visit(child, inherited_compilation, depth + 1)

    visit(value)
    return list(dict.fromkeys(videos))


def normalized_resource_url(value: str) -> str:
    """Compare resource identity without signed query strings or fragments."""
    return urlunsplit((*urlsplit(value)[:2], urlsplit(value).path, "", ""))


def selected_tracking_resource(resources: dict[str, dict[str, object]], metadata_videos: list[tuple[str, str | None]] = [], initial_player_url: str | None = None) -> str | None:
    """The visible pre-Interactive player is authoritative; metadata is association-only."""
    if initial_player_url and initial_player_url.startswith("https://"):
        return initial_player_url
    if metadata_videos:
        return metadata_videos[0][0]
    eligible = [
        resource for resource in resources.values()
        if urlsplit(str(resource["url"])).path.lower().endswith("/video.mp4")
        and INITIAL_PHASE in resource["phases"]
    ]
    return str(eligible[0]["url"]) if eligible else None


def camera_directions_for_tracking(resources: dict[str, dict[str, object]], metadata_pairs: list[tuple[str, str, str]], metadata_videos: list[tuple[str, str | None]] = [], initial_player_url: str | None = None) -> tuple[str, str] | None:
    """Resolve directions only from metadata attached to the selected Tracking URL."""
    tracking_url = selected_tracking_resource(resources, metadata_videos, initial_player_url)
    if not tracking_url:
        return None
    tracking_key = normalized_resource_url(tracking_url)
    return next(
        ((video_url, directions_url) for video_url, directions_url, _phase in metadata_pairs
         if normalized_resource_url(video_url) == tracking_key),
        None,
    )


def _is_relevant_resource(url: str) -> bool:
    path = urlsplit(url).path.lower()
    return any(token in path for token in ("transcode-", "video.mp4", "camera_alignment", "detections_constrained.det", ".veo", "/panorama/"))


def is_match_json_endpoint(url: str) -> bool:
    """Identify Veo match APIs safe to inspect when transfer size is unknown."""
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    return (host == "veo.co" or host.endswith(".veo.co")) and "/matches/" in path


def bounded_json_response(response) -> object | None:
    """Parse a JSON response within a hard body bound.

    Playwright exposes complete response bodies rather than a streaming reader.
    Unknown-length reads are therefore restricted to match-specific Veo APIs.
    """
    content = (response.headers.get("content-type") or "").lower()
    if "json" not in content:
        return None
    length_text = response.headers.get("content-length", "")
    if length_text.isdigit() and int(length_text) > MAX_JSON_RESPONSE_BYTES:
        return None
    if not length_text.isdigit() and not is_match_json_endpoint(response.url):
        return None
    try:
        body = response.body()
    except Exception:
        return None
    if len(body) > MAX_JSON_RESPONSE_BYTES:
        return None
    try:
        return json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None


def embedded_json_state(page) -> list[object]:
    """Read explicit JSON script tags only; never dump browser globals."""
    try:
        values = page.evaluate("""() => Array.from(document.querySelectorAll('script[type="application/json"], script#__NEXT_DATA__'))
            .map(node => node.textContent || '').filter(text => text.length > 1 && text.length <= 200000)""")
    except Exception:
        return []
    result = []
    for value in values[:20]:
        try:
            result.append(json.loads(value))
        except (TypeError, ValueError):
            continue
    return result


def normalized_match_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme != "https" or not parsed.hostname or not (parsed.hostname == "veo.co" or parsed.hostname.endswith(".veo.co")) or parsed.username or parsed.password:
        raise ValueError("Enter an authorized https://app.veo.co/ match URL.")
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, parsed.query, ""))


def probe_media(url: str, cookies: list[dict] | None = None) -> dict:
    executable = shutil.which("ffprobe")
    if executable is None:
        return {}
    try:
        public_url(url)
        # Bound inspection traffic to an 8 MiB prefix. Signed URLs never enter
        # ffprobe's command line; a trailing moov atom may make this inconclusive.
        with authorized_opener(cookies or []).open(Request(url, headers={"Range": "bytes=0-8388607"}), timeout=10) as response:
            data = response.read(8388608)
        with tempfile.NamedTemporaryFile(suffix=".mp4") as local:
            local.write(data)
            local.flush()
            command = [executable, "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate", "-of", "json", local.name]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=8, check=False)
        payload = json.loads(completed.stdout) if completed.returncode == 0 else {}
        stream = next((value for value in payload.get("streams", []) if value.get("codec_type") == "video"), {})
        rate = stream.get("avg_frame_rate", "0/0")
        numerator, denominator = (float(value) for value in rate.split("/"))
        return {"width": stream.get("width"), "height": stream.get("height"), "fps": numerator / denominator if denominator else None, "duration_s": float(payload.get("format", {}).get("duration")) if payload.get("format", {}).get("duration") else None}
    except (OSError, ValueError, json.JSONDecodeError, subprocess.TimeoutExpired):
        return {}


def wait_for_player(page) -> bool:
    """Wait for a decodable default player/video before revealing controls."""
    try:
        page.wait_for_function(
            """() => Array.from(document.querySelectorAll('video')).some((video) =>
                video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0)""",
            timeout=30_000,
        )
        return True
    except Exception:
        return False


def _physical_click(page, locator, description: str) -> bool:
    try:
        box = locator.bounding_box()
        if not box:
            return False
        x = box["x"] + box["width"] / 2
        y = box["y"] + box["height"] / 2
        page.mouse.move(x, y)
        page.wait_for_timeout(150)
        page.mouse.down()
        page.wait_for_timeout(100)
        page.mouse.up()
        return True
    except Exception:
        return False


def dismiss_login_modal(page) -> dict[str, bool]:
    """Close Veo's logged-out prompt using its stable, scoped close button."""
    result = {"login_modal_found": False, "login_modal_closed": False, "login_close_button_found": False}
    try:
        modals = page.locator(LOGIN_MODAL_SELECTOR)
        count = min(modals.count(), 10)
    except Exception:
        return result
    for index in range(count):
        modal = modals.nth(index)
        try:
            if not modal.is_visible(timeout=500):
                continue
            result["login_modal_found"] = True
            close = modal.locator(LOGIN_CLOSE_SELECTOR)
            if close.count() == 0 or not close.nth(0).is_visible(timeout=500):
                continue
            result["login_close_button_found"] = True
            if _physical_click(page, close.nth(0), "login modal close"):
                modal.wait_for(state="hidden", timeout=5_000)
                result["login_modal_closed"] = True
            return result
        except Exception:
            continue
    return result


def _surrounding_text(locator) -> str:
    try:
        return locator.evaluate("""el => {
            let node = el, fallback = "";
            for (let i = 0; node && i < 16; i++, node = node.parentElement) {
                const text = (node.innerText || '').replace(/\\s+/g, ' ').trim();
                if (!text) continue;
                fallback = text;
                if (/Become a member of|Request membership/i.test(text)) return text.slice(0, 1500);
            }
            return fallback.slice(0, 1500);
        }""") or ""
    except Exception:
        return ""


def dismiss_membership_modal(page) -> dict[str, bool]:
    """Close only the visible membership dialog, never a page-level X button."""
    result = {
        "membership_modal_found": False,
        "membership_modal_closed": False,
        "membership_close_button_found": False,
    }
    try:
        dialogs = page.locator(MEMBERSHIP_MODAL_SELECTOR)
        count = min(dialogs.count(), 20)
    except Exception:
        dialogs, count = None, 0
    for index in range(count):
        dialog = dialogs.nth(index)
        try:
            if not dialog.is_visible(timeout=500) or not MEMBERSHIP_TEXT.search(dialog.inner_text(timeout=800) or ""):
                continue
            result["membership_modal_found"] = True
            close_buttons = dialog.locator(MEMBERSHIP_CLOSE_SELECTOR)
            for close_index in range(min(close_buttons.count(), 12)):
                close_button = close_buttons.nth(close_index)
                if not close_button.is_visible(timeout=500):
                    continue
                result["membership_close_button_found"] = True
                close_button.click(timeout=2_000)
                dialog.wait_for(state="hidden", timeout=3_000)
                result["membership_modal_closed"] = True
                return result
        except Exception:
            continue

    # Some Veo overlays are not marked as dialogs; only accept an X whose
    # surrounding DOM contains the membership copy.
    try:
        buttons = page.locator(MEMBERSHIP_CLOSE_SELECTOR)
        count = min(buttons.count(), 20)
    except Exception:
        return result
    for index in range(count):
        close_button = buttons.nth(index)
        try:
            if not close_button.is_visible(timeout=500):
                continue
            if not MEMBERSHIP_TEXT.search(_surrounding_text(close_button)):
                continue
            dialog = close_button.locator("xpath=ancestor::*[self::div or self::section][1]")
        except Exception:
            continue
        result["membership_modal_found"] = True
        try:
            result["membership_close_button_found"] = True
            close_button.click(timeout=2_000)
            dialog.wait_for(state="hidden", timeout=3_000)
            result["membership_modal_closed"] = True
            return result
        except Exception:
            continue
    return result


def _hover_player(page) -> bool:
    """Reveal player controls without clicking unrelated page content."""
    for selector in ("video", "[data-testid*='player']", "[class*='player']"):
        try:
            locator = page.locator(selector)
            if locator.count() and locator.nth(0).is_visible(timeout=500):
                locator.nth(0).hover(timeout=1_500)
                page.wait_for_timeout(350)
                return True
        except Exception:
            continue
    return False


def _click_first_visible(locator, attempt: str, attempts: list[str], before_click, restore_after_failure) -> tuple[bool, bool]:
    """Click one visible, user-facing control; never manufacture a player URL."""
    try:
        count = min(locator.count(), 12)
    except Exception:
        return False, False
    found = False
    for index in range(count):
        candidate = locator.nth(index)
        previous_phase = None
        try:
            if not candidate.is_visible(timeout=500):
                continue
            found = True
            previous_phase = before_click()
            candidate.click(timeout=2_000)
            attempts.append(attempt)
            return True, True
        except Exception:
            if previous_phase is not None:
                restore_after_failure(previous_phase)
            continue
    return found, False


def _control_label(control) -> str:
    try:
        parts = [
            control.get_attribute("aria-label"),
            control.get_attribute("title"),
            control.get_attribute("data-testid"),
            control.inner_text(timeout=300),
        ]
        return " ".join(part.strip() for part in parts if part and part.strip())[:160]
    except Exception:
        return ""


def _snapshot_player(page) -> dict[str, object]:
    """Capture player state; _safe_snapshot strips private URLs before publication."""
    snapshot: dict[str, object] = {
        "button": {
            "found": False,
            "count": 0,
            "visible": False,
            "enabled": False,
            "disabled": None,
            "aria_disabled": None,
            "class_name": None,
            "bbox": None,
            "display": None,
            "visibility": None,
            "opacity": None,
            "pointer_events": None,
            "element_at_center": "not-found",
        },
        "videos": [],
        "canvases": [],
        "performance_resources": [],
        "relevant_resources": [],
        "dom": {},
    }
    try:
        locator = page.locator(PRIMARY_INTERACTIVE_SELECTOR)
        count = locator.count()
        snapshot["button"]["count"] = count
        snapshot["button"]["found"] = count > 0
        if count:
            button = locator.nth(0)
            box = button.bounding_box()
            computed = button.evaluate(
                """(element, point) => {
                    const style = getComputedStyle(element);
                    const top = point ? document.elementFromPoint(point.x, point.y) : null;
                    let hit = 'none';
                    if (top === element) hit = 'button';
                    else if (top && element.contains(top) && top.tagName === 'svg') hit = 'svg-child';
                    else if (top && element.contains(top)) hit = 'button-child';
                    else if (top) hit = 'overlaying-element';
                    const player = element.closest('[class*="player"], [data-testid*="player"]');
                    return {
                        disabled: element.disabled,
                        ariaDisabled: element.getAttribute('aria-disabled'),
                        className: String(element.className || ''),
                        display: style.display,
                        visibility: style.visibility,
                        opacity: style.opacity,
                        pointerEvents: style.pointerEvents,
                        elementAtCenter: hit,
                        playerClass: player ? String(player.className || '') : null,
                        playerChildCount: player ? player.children.length : null,
                    };
                }""",
                {"x": box["x"] + box["width"] / 2, "y": box["y"] + box["height"] / 2} if box else None,
            )
            visible = button.is_visible(timeout=500)
            disabled = bool(computed["disabled"])
            aria_disabled = computed["ariaDisabled"]
            snapshot["button"].update({
                "visible": visible,
                "enabled": visible and not disabled and aria_disabled != "true",
                "disabled": disabled,
                "aria_disabled": aria_disabled,
                "class_name": computed["className"],
                "bbox": box,
                "display": computed["display"],
                "visibility": computed["visibility"],
                "opacity": computed["opacity"],
                "pointer_events": computed["pointerEvents"],
                "element_at_center": computed["elementAtCenter"] if box else "no-bounding-box",
            })
            snapshot["dom"] = {"player_class": computed["playerClass"], "player_child_count": computed["playerChildCount"]}
    except Exception:
        pass
    try:
        snapshot["videos"] = page.evaluate(
            """() => Array.from(document.querySelectorAll('video')).map((video) => ({
                currentSrc: video.currentSrc || '', src: video.getAttribute('src') || '',
                videoWidth: video.videoWidth, videoHeight: video.videoHeight,
                readyState: video.readyState, paused: video.paused, currentTime: video.currentTime,
                visible: !!(video.offsetWidth || video.offsetHeight || video.getClientRects().length),
                area: video.clientWidth * video.clientHeight
            }))"""
        )
        snapshot["canvases"] = page.evaluate(
            """() => Array.from(document.querySelectorAll('canvas')).map((canvas) => ({
                width: canvas.width, height: canvas.height,
                clientWidth: canvas.clientWidth, clientHeight: canvas.clientHeight
            }))"""
        )
        performance = page.evaluate(
            """() => performance.getEntriesByType('resource').map((entry) => entry.name)
                .filter((name) => /transcode-|video\\.mp4|camera_alignment|\\.veo|\\/panorama\\//i.test(name))"""
        )
        snapshot["_performance_resources_raw"] = list(dict.fromkeys(performance))
        snapshot["performance_resources"] = [sanitize_url(value) for value in snapshot["_performance_resources_raw"]]
        snapshot["relevant_resources"] = snapshot["performance_resources"]
    except Exception:
        snapshot["_performance_resources_raw"] = []
    return snapshot


def initial_player_video_url(snapshot: dict[str, object]) -> str | None:
    """Pick the active, decodable default player video, never a hidden clip."""
    videos = [item for item in snapshot.get("videos", []) if isinstance(item, dict)
              and item.get("readyState", 0) >= 2 and item.get("videoWidth", 0) > 0
              and isinstance(item.get("currentSrc"), str) and item["currentSrc"].startswith("https://")]
    choices = [item for item in videos if item.get("visible")] or videos
    return str(max(choices, key=lambda item: (float(item.get("area") or 0), not bool(item.get("paused"))))["currentSrc"]) if choices else None


def _dom_change_summary(before: dict[str, object], after: dict[str, object]) -> dict[str, object]:
    changed: list[str] = []
    def video_structure(snapshot):
        return [{key: video.get(key) for key in ("currentSrc", "src", "videoWidth", "videoHeight", "readyState", "paused")} for video in snapshot.get("videos", [])]

    if video_structure(before) != video_structure(after):
        changed.append("videos")
    if before.get("canvases") != after.get("canvases"):
        changed.append("canvases")
    if before.get("dom") != after.get("dom"):
        changed.append("player_dom")
    before_button, after_button = before.get("button", {}), after.get("button", {})
    for key in ("class_name", "disabled", "aria_disabled", "visible", "pointer_events"):
        if before_button.get(key) != after_button.get(key):
            changed.append(f"button.{key}")
    before_resources = set(before.get("performance_resources", []))
    after_resources = set(after.get("performance_resources", []))
    new_resources = sorted(after_resources - before_resources)
    if new_resources:
        changed.append("performance_resources")
    return {"changed": bool(changed), "changed_components": changed, "new_performance_resources": new_resources}


def _safe_snapshot(snapshot: dict[str, object]) -> dict[str, object]:
    safe = {key: value for key, value in snapshot.items() if not key.startswith("_")}
    safe["videos"] = [
        {
            **video,
            "currentSrc": sanitize_url(video.get("currentSrc", "")) if video.get("currentSrc") else "",
            "src": sanitize_url(video.get("src", "")) if video.get("src") else "",
        }
        for video in snapshot.get("videos", [])
    ]
    return safe


def _activate_primary_interactive(page, before_click, restore_after_failure, screenshot_before: Path | None, screenshot_after: Path | None) -> dict[str, object]:
    """Use the observed button with a hit-tested physical pointer click."""
    result: dict[str, object] = {
        "primary_selector": PRIMARY_INTERACTIVE_SELECTOR,
        "primary_selector_found": False,
        "primary_selector_count": 0,
        "primary_selector_visible": False,
        "primary_selector_hover_required": False,
        "primary_selector_click_succeeded": False,
        "primary_selector_force_click_used": False,
        "click_method_attempts": [],
        "before": None,
        "after": None,
        "dom_change_summary": {"changed": False, "changed_components": [], "new_performance_resources": []},
        "activation_method": None,
    }
    try:
        locator = page.locator(PRIMARY_INTERACTIVE_SELECTOR)
        count = locator.count()
    except Exception:
        return result
    result["primary_selector_count"] = count
    result["primary_selector_found"] = count > 0
    if not count:
        return result

    def visible_buttons():
        visible = []
        for index in range(min(count, 12)):
            try:
                button = locator.nth(index)
                if button.is_visible(timeout=500):
                    visible.append(button)
            except Exception:
                continue
        return visible

    try:
        locator.nth(0).wait_for(state="visible", timeout=15_000)
    except Exception:
        pass
    buttons = visible_buttons()
    if not buttons:
        result["primary_selector_hover_required"] = True
        _hover_player(page)
        buttons = visible_buttons()
    result["primary_selector_visible"] = bool(buttons)
    if not buttons:
        return result
    before = _snapshot_player(page)
    result["before"] = _safe_snapshot(before)
    if screenshot_before:
        try:
            screenshot_before.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(screenshot_before), full_page=False)
        except Exception:
            pass
    button = buttons[0]
    hit = before["button"].get("element_at_center")
    result["primary_selector_center_hit"] = hit
    if before["button"].get("disabled") or before["button"].get("aria_disabled") == "true":
        result["click_method_attempts"].append({"method": "mouse", "attempted": False, "reason": "Interactive button is disabled"})
        return result
    if hit not in {"button", "svg-child", "button-child"}:
        result["click_method_attempts"].append({"method": "mouse", "attempted": False, "reason": f"button center hit {hit}; unrelated element may be covering it"})
    else:
        try:
            box = button.bounding_box()
        except Exception:
            box = None
        previous_phase = before_click()
        mouse_attempt = {"method": "mouse", "attempted": bool(box), "click_returned": False, "observable_change": False}
        if box:
            try:
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.wait_for_timeout(150)
                page.mouse.down()
                page.wait_for_timeout(100)
                page.mouse.up()
                mouse_attempt["click_returned"] = True
            except Exception as exc:
                mouse_attempt["error"] = type(exc).__name__
                restore_after_failure(previous_phase)
        else:
            restore_after_failure(previous_phase)
        page.wait_for_timeout(ACTIVATION_OBSERVE_MS)
        after = _snapshot_player(page)
        mouse_attempt["observable_change"] = _dom_change_summary(before, after)["changed"]
        result["click_method_attempts"].append(mouse_attempt)
        if mouse_attempt["click_returned"]:
            result["primary_selector_click_succeeded"] = True
            result["activation_method"] = "physical_mouse"
        result["after"] = _safe_snapshot(after)
        result["dom_change_summary"] = _dom_change_summary(before, after)
        result["_before_performance_resources_raw"] = before.get("_performance_resources_raw", [])
        result["_after_performance_resources_raw"] = after.get("_performance_resources_raw", [])
        if screenshot_after:
            try:
                page.screenshot(path=str(screenshot_after), full_page=False)
            except Exception:
                pass
    return result


def activate_interactive(page, before_click, restore_after_failure, screenshot_before: Path | None = None, screenshot_after: Path | None = None) -> dict[str, object]:
    """Open Interactive explicitly, including accessible menu-based controls.

    The observed semantic button is first; accessibility paths are fallbacks.
    The menu pass only opens
    visible controls whose own text indicates a view/menu; it does not click
    arbitrary page content or guess private player endpoints.
    """
    primary = _activate_primary_interactive(page, before_click, restore_after_failure, screenshot_before, screenshot_after)
    attempts: list[str] = ["button.interactive-player"] if primary["primary_selector_click_succeeded"] else []
    menu_controls: list[str] = []
    found = bool(primary["primary_selector_found"])
    if primary["primary_selector_click_succeeded"]:
        return {**primary, "interactive_control_found": True, "interactive_click_succeeded": True, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}

    # If the exact semantic button existed, its physical-click result is
    # authoritative. Do not fall through to a different text control after that.
    if primary["primary_selector_found"]:
        return {**primary, "interactive_control_found": True, "interactive_click_succeeded": False, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}

    def direct_control() -> tuple[bool, bool]:
        found = False
        locators = (
            (page.get_by_role("button", name=INTERACTIVE_LABEL), "interactive button"),
            (page.get_by_role("tab", name=INTERACTIVE_LABEL), "interactive tab"),
            (page.get_by_role("menuitem", name=INTERACTIVE_LABEL), "interactive menu item"),
            (page.get_by_role("radio", name=INTERACTIVE_LABEL), "interactive radio"),
            (page.get_by_text(INTERACTIVE_LABEL, exact=True), "interactive visible text"),
        )
        for locator, attempt in locators:
            visible, succeeded = _click_first_visible(locator, attempt, attempts, before_click, restore_after_failure)
            found = found or visible
            if succeeded:
                primary["activation_method"] = "locator_click"
                primary["click_method_attempts"].append({"method": "locator_click", "attempted": True, "click_returned": True, "observable_change": None})
                return found, True
        return found, False

    fallback_found, succeeded = direct_control()
    found = found or fallback_found
    if succeeded:
        return {**primary, "interactive_control_found": True, "interactive_click_succeeded": True, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}

    try:
        controls = page.locator("button, [role='button'], [aria-haspopup='menu'], [aria-expanded]")
        count = min(controls.count(), 40)
    except Exception:
        controls, count = None, 0
    for index in range(count):
        assert controls is not None
        control = controls.nth(index)
        try:
            if not control.is_visible(timeout=250):
                continue
        except Exception:
            continue
        label = _control_label(control)
        try:
            is_menu_trigger = control.get_attribute("aria-haspopup") == "menu"
        except Exception:
            is_menu_trigger = False
        if not is_menu_trigger and (not label or not MENU_LABEL.search(label)):
            continue
        menu_controls.append(label or "unnamed menu control")
        try:
            control.click(timeout=1_500)
            page.wait_for_timeout(300)
        except Exception:
            continue
        discovered, succeeded = direct_control()
        found = found or discovered
        if succeeded:
            return {**primary, "interactive_control_found": True, "interactive_click_succeeded": True, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}
    return {**primary, "interactive_control_found": found, "interactive_click_succeeded": False, "interactive_attempts": attempts, "menu_controls_inspected": menu_controls}


def _candidate_reference(candidate: VeoCandidate) -> dict[str, object]:
    return {"identifier": candidate.identifier, "filename": candidate.filename, "safe_url": candidate.safe_url, "discovery_phases": candidate.discovery_phases}


def phase_diagnostics(candidates: list[VeoCandidate], interaction: dict[str, object]) -> dict[str, object]:
    def in_phase(phase: str) -> list[dict[str, object]]:
        return [_candidate_reference(candidate) for candidate in candidates if phase in candidate.discovery_phases]

    def final(role: str) -> list[dict[str, object]]:
        return [_candidate_reference(candidate) for candidate in candidates if candidate.role == role]

    post_transcode = any(
        AFTER_INTERACTIVE_PHASE in candidate.discovery_phases
        and candidate.filename.lower().startswith("transcode-")
        and candidate.filename.lower().endswith(".mp4")
        for candidate in candidates
    )
    post_calibration = any(
        AFTER_INTERACTIVE_PHASE in candidate.discovery_phases
        and candidate.filename.lower().endswith(".veo")
        and candidate.field_length is not None
        and candidate.field_width is not None
        for candidate in candidates
    )
    confirmed = bool(interaction.get("interactive_click_succeeded") and post_transcode and post_calibration)
    confirmation_reason = (
        "post-Interactive transcode and valid calibration resources observed"
        if confirmed
        else "required post-Interactive transcode and valid calibration resources were not both observed"
    )
    return {
        "default_phase_resources": in_phase(INITIAL_PHASE),
        "interactive_control_found": interaction["interactive_control_found"],
        "interactive_click_succeeded": interaction["interactive_click_succeeded"],
        "interactive_activation_confirmed": confirmed,
        "activation": {"confirmed": confirmed, "confirmation_reason": confirmation_reason, "method": interaction.get("activation_method")},
        "player_ready": interaction.get("player_ready", False),
        "login_modal_found": interaction.get("login_modal_found", False),
        "login_modal_closed": interaction.get("login_modal_closed", False),
        "login_close_button_found": interaction.get("login_close_button_found", False),
        "membership_modal_found": interaction.get("membership_modal_found", False),
        "membership_modal_closed": interaction.get("membership_modal_closed", False),
        "close_button_found": interaction.get("membership_close_button_found", False),
        "interactive_button_found": interaction.get("primary_selector_found", False),
        "interactive_button_visible": interaction.get("primary_selector_visible", False),
        "interactive_center_hit": interaction.get("primary_selector_center_hit", "not-observed"),
        "physical_click_attempted": any(
            attempt.get("method") == "mouse" and attempt.get("attempted")
            for attempt in interaction.get("click_method_attempts", [])
        ),
        "physical_click_succeeded": any(
            attempt.get("method") == "mouse" and attempt.get("click_returned")
            for attempt in interaction.get("click_method_attempts", [])
        ),
        "post_interactive_transcode_found": post_transcode,
        "post_interactive_calibration_found": post_calibration,
        "interactive_button": {
            "selector": interaction.get("primary_selector", PRIMARY_INTERACTIVE_SELECTOR),
            "found": interaction.get("primary_selector_found", False),
            "count": interaction.get("primary_selector_count", 0),
            "visible": interaction.get("primary_selector_visible", False),
            "hover_required": interaction.get("primary_selector_hover_required", False),
            "center_hit": interaction.get("primary_selector_center_hit", "not-observed"),
            "click_method_attempts": interaction.get("click_method_attempts", []),
        },
        "before": interaction.get("before"),
        "after": interaction.get("after"),
        "dom_change_summary": interaction.get("dom_change_summary"),
        "network_events": interaction.get("network_events", []),
        "screenshot_before": interaction.get("screenshot_before"),
        "screenshot_after": interaction.get("screenshot_after"),
        "primary_selector": interaction.get("primary_selector", PRIMARY_INTERACTIVE_SELECTOR),
        "primary_selector_found": interaction.get("primary_selector_found", False),
        "primary_selector_count": interaction.get("primary_selector_count", 0),
        "primary_selector_visible": interaction.get("primary_selector_visible", False),
        "primary_selector_hover_required": interaction.get("primary_selector_hover_required", False),
        "primary_selector_click_succeeded": interaction.get("primary_selector_click_succeeded", False),
        "primary_selector_force_click_used": interaction.get("primary_selector_force_click_used", False),
        "interactive_attempts": interaction["interactive_attempts"],
        "menu_controls_inspected": interaction["menu_controls_inspected"],
        "post_interactive_resources": in_phase(AFTER_INTERACTIVE_PHASE),
        "final_classified_assets": {
            "tracking": final("ball_tracking"),
            "dual_lens": final("panorama"),
            "calibration": final("calibration"),
            "camera_directions": final("camera_directions"),
        },
    }


def inspect_match(match_url: str, diagnostics_dir: Path) -> VeoInspection:
    try:
        from playwright.sync_api import sync_playwright
    except ModuleNotFoundError as exc:
        raise RuntimeError("Playwright is unavailable. Install the backend requirements and run 'playwright install chromium'.") from exc
    target = normalized_match_url(match_url)
    public_url(target)
    seen: dict[str, dict[str, object]] = {}
    calibration_details: dict[str, tuple[float, float, object]] = {}
    metadata_camera_directions: list[tuple[str, str, str]] = []
    metadata_tracking_videos: list[tuple[str, str | None]] = []
    metadata_highlights_videos: list[str] = []
    initial_player_url: str | None = None
    metadata: dict[str, str] = {}
    phase = INITIAL_PHASE
    inspection_identifier = f"inspect_{uuid.uuid4().hex[:12]}"
    structured = StructuredDiscovery(diagnostics_dir / f"{inspection_identifier}-datasets")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    screenshot_before = diagnostics_dir / f"{inspection_identifier}-before-interactive.png"
    screenshot_after = diagnostics_dir / f"{inspection_identifier}-after-interactive.png"
    network_events: list[dict[str, object]] = []

    def record_resource(url: str, content_type: str | None, discovered_phase: str) -> None:
        normalized = urlunsplit((*urlsplit(url)[:4], ""))
        path = urlsplit(normalized).path.lower()
        if not (path.endswith((".mp4", ".m3u8", ".mpd", ".veo", ".det")) or (content_type and content_type.startswith("video/"))):
            return
        if normalized not in seen and len(seen) >= MAX_RESOURCES:
            return
        resource = seen.setdefault(normalized, {"url": normalized, "content_type": content_type, "phases": []})
        if content_type and not resource["content_type"]:
            resource["content_type"] = content_type
        if discovered_phase not in resource["phases"]:
            resource["phases"].append(discovered_phase)

    with sync_playwright() as playwright, tempfile.TemporaryDirectory(prefix="veo-inspection-") as profile_dir:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=profile_dir,
            headless=False,
            viewport={"width": 1600, "height": 1000},
        )
        page = context.pages[0] if context.pages else context.new_page()

        def remember(response):
            content = response.headers.get("content-type")
            if _is_relevant_resource(response.url):
                network_events.append({"kind": "response", "url": sanitize_url(response.url), "status": response.status, "phase": phase})
            if response.status not in {200, 206}:
                if "json" in (content or "") or response.status in {401, 403, 404}:
                    structured.observe_json(response.url, None, phase, status_code=response.status)
                return
            record_resource(response.url, content, phase)
            size = response.headers.get("content-length", "")
            resource = seen.get(urlunsplit((*urlsplit(response.url)[:4], "")))
            total = response.headers.get("content-range", "").rsplit("/", 1)[-1] if response.status == 206 else size
            if resource is not None and total.isdigit():
                resource["size_bytes"] = int(total)
            payload = bounded_json_response(response)
            if payload is None and "json" in (content or ""):
                structured.observe_json(response.url, None, phase, status_code=response.status)
            if payload is not None:
                structured.observe_json(response.url, payload, phase, status_code=response.status)
                for link in resource_links(payload, limit=MAX_RESOURCES):
                    record_resource(link, None, phase)
                if urlsplit(response.url).path.rstrip("/").lower().endswith("/videos"):
                    for video_url, directions_url in standard_video_camera_directions(payload):
                        metadata_camera_directions.append((video_url, directions_url, phase))
                    for video_url, directions_url in standard_follow_videos(payload):
                        metadata_tracking_videos.append((video_url, directions_url))
                        record_resource(video_url, None, phase)
                response_path = urlsplit(response.url).path.rstrip("/").lower()
                # Only match-owned video/highlight metadata can identify a Veo reel.
                # Arbitrary captured JSON and individual event clips are not enough.
                if response_path.endswith(("/videos", "/highlights")):
                    for video_url in generated_highlights_videos(payload):
                        metadata_highlights_videos.append(video_url)
                        record_resource(video_url, None, phase)
                if isinstance(payload, dict) and isinstance(payload.get("match"), dict):
                    for key in ("title", "date", "home_team", "away_team"):
                        if isinstance(payload["match"].get(key), str):
                            metadata[key] = payload["match"][key]

        def remember_request(request):
            if _is_relevant_resource(request.url):
                network_events.append({"kind": "request", "url": sanitize_url(request.url), "method": request.method, "phase": phase})
                record_resource(request.url, None, phase)

        context.on("response", remember)  # Attached before page/player initialization.
        context.on("request", remember_request)
        interaction: dict[str, object] = {
            "interactive_control_found": False,
            "interactive_click_succeeded": False,
            "interactive_attempts": [],
            "menu_controls_inspected": [],
            "membership_modal_found": False,
            "membership_modal_closed": False,
            "membership_close_button_found": False,
        }
        try:
            page.goto(target, wait_until="domcontentloaded", timeout=30_000)
            page.wait_for_timeout(5_000)
            interaction.update(dismiss_login_modal(page))
            page.wait_for_timeout(1_200)
            interaction.update(dismiss_membership_modal(page))
            player_ready = wait_for_player(page)
            _hover_player(page)
            interaction.update(dismiss_login_modal(page))
            interaction.update(dismiss_membership_modal(page))
            page.wait_for_timeout(500)
            _hover_player(page)
            metadata.setdefault("title", page.title())
            initial_snapshot = _snapshot_player(page)
            initial_player_url = initial_player_video_url(initial_snapshot)
            for embedded in embedded_json_state(page):
                structured.observe_json(target, embedded, "embedded_state", "embedded_json")
            for raw in initial_snapshot.get("_performance_resources_raw", []):
                record_resource(raw, None, INITIAL_PHASE)
            for video in initial_snapshot.get("videos", []):
                for key in ("currentSrc", "src"):
                    if video.get(key, "").startswith("https://"):
                        record_resource(video[key], None, INITIAL_PHASE)

            def start_interactive_phase() -> str:
                nonlocal phase
                previous_phase = phase
                phase = AFTER_INTERACTIVE_PHASE
                return previous_phase

            def restore_phase(previous_phase: str) -> None:
                nonlocal phase
                phase = previous_phase

            activation = activate_interactive(page, start_interactive_phase, restore_phase, screenshot_before, screenshot_after)
            interaction.update(activation)
            interaction["player_ready"] = player_ready
            interaction["network_events"] = network_events[:200]
            interaction["screenshot_before"] = str(screenshot_before) if screenshot_before.exists() else None
            interaction["screenshot_after"] = str(screenshot_after) if screenshot_after.exists() else None
            for raw in interaction.get("_before_performance_resources_raw", []):
                normalized = urlunsplit((*urlsplit(raw)[:4], ""))
                if normalized not in seen:
                    record_resource(raw, None, INITIAL_PHASE)
            for raw in interaction.get("_after_performance_resources_raw", []):
                normalized = urlunsplit((*urlsplit(raw)[:4], ""))
                if normalized not in seen:
                    record_resource(raw, None, AFTER_INTERACTIVE_PHASE)
            if phase == AFTER_INTERACTIVE_PHASE:
                deadline = time.monotonic() + POST_INTERACTIVE_CAPTURE_MS / 1000
                while time.monotonic() < deadline:
                    runtime_urls = page.evaluate("""() => [
                        ...performance.getEntriesByType('resource').map(entry => entry.name),
                        ...Array.from(document.querySelectorAll('video, source')).flatMap(node => [node.currentSrc || '', node.src || ''])
                    ].filter(Boolean)""")
                    for raw in runtime_urls:
                        record_resource(raw, None, AFTER_INTERACTIVE_PHASE)
                    has_transcode = any(
                        urlsplit(str(item["url"])).path.lower().split("/")[-1].startswith("transcode-")
                        and str(item["url"]).lower().split("?")[0].endswith(".mp4")
                        and AFTER_INTERACTIVE_PHASE in item["phases"]
                        for item in seen.values()
                    )
                    has_calibration = any(
                        urlsplit(str(item["url"])).path.lower().endswith(".veo")
                        and AFTER_INTERACTIVE_PHASE in item["phases"]
                        for item in seen.values()
                    )
                    if has_transcode and has_calibration:
                        break
                    page.wait_for_timeout(250)
            final_snapshot = _snapshot_player(page)
            for raw in final_snapshot.get("_performance_resources_raw", []):
                if raw not in initial_snapshot.get("_performance_resources_raw", []):
                    record_resource(raw, None, phase)
            for video in final_snapshot.get("videos", []):
                for key in ("currentSrc", "src"):
                    raw = video.get(key, "")
                    if raw.startswith("https://") and not any(raw == v.get(key) for v in initial_snapshot.get("videos", [])):
                        record_resource(raw, None, phase)
            interaction["after"] = _safe_snapshot(final_snapshot)
            try:
                page.screenshot(path=str(screenshot_after), full_page=False)
                interaction["screenshot_after"] = str(screenshot_after)
            except Exception:
                pass
            interaction["network_events"] = network_events[:200]
            interaction["interactive_click_succeeded"] = bool(interaction["interactive_click_succeeded"] or any(attempt.get("click_returned") for attempt in interaction.get("click_method_attempts", [])))

            for resource in list(seen.values()):
                url = str(resource["url"])
                path = urlsplit(url).path.lower()
                if ".veo" not in path and "camera_alignment" not in path:
                    continue
                try:
                    public_url(url)
                    cookies_for_url = context.cookies([url])
                    with authorized_opener(cookies_for_url).open(Request(url), timeout=8) as response:
                        body = response.read(2_000_001)
                    if len(body) > 2_000_000:
                        continue
                    from geometry.veo_metadata import load_veo_metadata
                    with tempfile.NamedTemporaryFile(suffix=".veo") as local:
                        local.write(body)
                        local.flush()
                        parsed = load_veo_metadata(local.name)
                    calibration_details[url] = (parsed.field_length, parsed.field_width, parsed.raw_json.get("calibration_version"))
                except Exception:
                    pass
            cookies = {str(resource["url"]): context.cookies([str(resource["url"])]) for resource in seen.values()}
            for _video_url, directions_url, _directions_phase in metadata_camera_directions:
                cookies.setdefault(urlunsplit((*urlsplit(directions_url)[:4], "")), context.cookies([directions_url]))
        finally:
            context.close()

    selected_tracking_url = selected_tracking_resource(seen, metadata_tracking_videos, initial_player_url)
    selected_pair = camera_directions_for_tracking(seen, metadata_camera_directions, metadata_tracking_videos, initial_player_url)
    selected_tracking_key = normalized_resource_url(selected_tracking_url) if selected_tracking_url else None
    associated_directions_key = None
    if selected_pair:
        video_url, directions_url = selected_pair
        discovered_phase = next((item_phase for item_video, item_url, item_phase in metadata_camera_directions
                                 if normalized_resource_url(item_video) == normalized_resource_url(video_url)
                                 and item_url == directions_url), INITIAL_PHASE)
        normalized = urlunsplit((*urlsplit(directions_url)[:4], ""))
        associated_directions_key = normalized_resource_url(normalized)
        resource = seen.setdefault(normalized, {"url": normalized, "content_type": None, "phases": []})
        if discovered_phase not in resource["phases"]:
            resource["phases"].append(discovered_phase)
        cookies.setdefault(normalized, [])
    if associated_directions_key:
        for key, resource in list(seen.items()):
            if urlsplit(str(resource["url"])).path.lower().endswith(".det") and normalized_resource_url(str(resource["url"])) != associated_directions_key:
                seen.pop(key)

    entries: list[tuple[dict[str, object], VeoCandidate]] = []
    for index, resource in enumerate(seen.values()):
        url = str(resource["url"])
        entries.append((resource, classify_resource(f"asset_{index + 1}", url, resource["content_type"], discovery_phases=list(resource["phases"]))))

    # Spend bounded media probes on the observed default/Interactive resources
    # before unrelated responses. Classification remains phase-aware either way.
    def probe_priority(entry: tuple[dict[str, object], VeoCandidate]) -> tuple[int, str]:
        _, candidate = entry
        initial_video = INITIAL_PHASE in candidate.discovery_phases and candidate.filename.lower() == "video.mp4"
        interactive_transcode = AFTER_INTERACTIVE_PHASE in candidate.discovery_phases and candidate.filename.lower().startswith("transcode-")
        return (0 if initial_video else 1 if interactive_transcode else 2, candidate.filename)

    indices = sorted((index for index, entry in enumerate(entries) if entry[1].filename.lower().endswith(".mp4")), key=lambda index: probe_priority(entries[index]))[:8]
    for index in indices:
        resource, candidate = entries[index]
        url = str(resource["url"])
        entries[index] = (resource, classify_resource(candidate.identifier, url, resource["content_type"], probe_media(url, cookies.get(url, [])), candidate.discovery_phases))

    candidates = []
    for resource, candidate in entries:
        url = str(resource["url"])
        candidate.cookies = cookies.get(url, [])
        candidate.size_bytes = resource.get("size_bytes")
        if url in calibration_details:
            candidate.field_length, candidate.field_width, version = calibration_details[url]
            candidate.calibration_version = str(version) if version is not None else None
        candidates.append(candidate)
    # A Standard/Follow relationship from `/videos/` outranks generic player
    # resources (including a highlights compilation called video.mp4).
    if selected_tracking_key:
        for candidate in candidates:
            if normalized_resource_url(candidate.url) == selected_tracking_key:
                candidate.role, candidate.confidence = "ball_tracking", 1.0
                candidate.reasons = [
                    "authoritative initial player video"
                    if initial_player_url and normalized_resource_url(initial_player_url) == selected_tracking_key
                    else "explicit Standard/Follow video metadata"
                ]
                setattr(candidate, "authoritative", True)
            elif candidate.role == "ball_tracking":
                candidate.role, candidate.confidence = "unknown", 0.0
                candidate.reasons = ["non-authoritative video resource; initial player source selected"]
    highlights_keys = {normalized_resource_url(url) for url in metadata_highlights_videos}
    for candidate in candidates:
        if normalized_resource_url(candidate.url) in highlights_keys and normalized_resource_url(candidate.url) != selected_tracking_key:
            candidate.role, candidate.confidence = "veo_highlights", 1.0
            candidate.reasons = ["explicit Veo-generated highlights metadata"]
    inspection = VeoInspection(
        identifier=inspection_identifier,
        match_url=target,
        safe_match_url=sanitize_url(target),
        metadata=metadata,
        dataset_observations=structured.observations,
        candidates=candidates,
        interaction=phase_diagnostics(candidates, interaction),
        access_error=(
            "This Veo match is not accessible without authentication."
            if not interaction.get("player_ready") and not any(candidate.role == "ball_tracking" for candidate in candidates)
            else None
        ),
    )
    interaction_confirmed = bool(inspection.interaction.get("interactive_activation_confirmed"))
    structured.record_interactive(interaction_confirmed, {
        "button_selector": PRIMARY_INTERACTIVE_SELECTOR,
        "physical_click_succeeded": inspection.interaction.get("physical_click_succeeded", False),
        "post_interactive_transcode_found": inspection.interaction.get("post_interactive_transcode_found", False),
        "post_interactive_calibration_found": inspection.interaction.get("post_interactive_calibration_found", False),
    })
    assets = {"tracking": None, "dual_lens": None, "calibration": None, "camera_directions": None}
    for candidate in candidates:
        role = {"ball_tracking": "tracking", "panorama": "dual_lens"}.get(candidate.role, candidate.role)
        if role in assets and assets[role] is None:
            assets[role] = candidate.public_dict()
    inspection.manifest = structured.manifest({**metadata, "source_url": inspection.safe_match_url}, assets, {
        "inspection_id": inspection_identifier,
        "browser_phases": [INITIAL_PHASE, AFTER_INTERACTIVE_PHASE],
        "interactive": inspection.interaction.get("activation"),
        "json_response_count": len(structured.responses),
        "referenced_resource_count": len(structured.references),
    })
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    diagnostic = diagnostics_dir / f"{inspection.identifier}.json"
    inspection.diagnostic_path = str(diagnostic)
    diagnostic.write_text(json.dumps(inspection.public_dict(), indent=2), encoding="utf-8")
    write_manifest(diagnostics_dir / f"{inspection.identifier}-manifest.json", inspection.manifest)
    return inspection
