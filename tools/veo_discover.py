#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(REPOSITORY_ROOT))

from web_app.backend.app.importers.veo.classifier import sanitize_url
from web_app.backend.app.importers.veo.discovery import (
    write_manifest,
)
from web_app.backend.app.importers.veo.datasets import replay_capture
from web_app.backend.app.importers.veo.inspector import (
    normalized_resource_url,
    standard_video_camera_directions,
)

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


MAX_TEXT_BODY = 5 * 1024 * 1024
MAX_WASM_BODY = 10 * 1024 * 1024
MAX_EVENTS = 30000
MAX_WS_FRAMES = 5000
MAX_WS_FRAME_SIZE = 512 * 1024

PRIMARY_INTERACTIVE_SELECTOR = "button.interactive-player"

LOGIN_MODAL_SELECTOR = ".modal.modal-login-prompt"
LOGIN_CLOSE_SELECTOR = 'button[aria-label="Close button"]'

MEMBERSHIP_CLOSE_SELECTOR = 'button[aria-label="x"]'
MEMBERSHIP_TEXT = re.compile(
    r"become\s+a\s+member\s+of|request\s+membership",
    re.IGNORECASE,
)

DISCOVERY_KEYWORDS = [
    "virtualcam",
    "virtualCam",
    "camera2logic",
    "camera2world",
    "camera2presentation",
    "camera_alignment",
    "cameraAlignment",
    "cameraState",
    "camera_state",
    "fov",
    "fieldOfView",
    "field_of_view",
    "playerTracking",
    "player_tracking",
    "ballTracking",
    "ball_tracking",
    "tracking",
    "panorama",
    "interactive-player",
    "lineup",
    "roster",
    "jersey",
    "shirtNumber",
    "shirt_number",
    "playerId",
    "player_id",
    "teamId",
    "team_id",
    "goalkeeper",
    "referee",
    "penalty",
    "penalties",
    "throwIn",
    "throw_in",
    "corner",
    "goal",
    "goals",
    "assist",
    "events",
    "timeline",
    "highlights",
    "annotations",
    "matchReport",
    "match_report",
    "heatmap",
    "position",
    "coordinates",
    "homography",
    "projection",
    "pitch",
]

SECRET_KEY_RE = re.compile(
    r"(token|authorization|cookie|password|passwd|secret|session|"
    r"access[_-]?key|api[_-]?key|signature|signed)",
    re.IGNORECASE,
)

JWT_RE = re.compile(
    r"\beyJ[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\.[a-zA-Z0-9_-]{8,}\b"
)

BEARER_RE = re.compile(
    r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{10,}"
)

URL_RE = re.compile(
    r"https?://[^\s\"'<>]+"
)


def mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def json_dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


def safe_url(url: str) -> str:
    if not isinstance(url, str):
        return str(url)

    try:
        parsed = urlsplit(url)

        if not parsed.scheme:
            return url

        query = parse_qsl(
            parsed.query,
            keep_blank_values=True,
        )

        safe_query = urlencode(
            [(key, "REDACTED") for key, _ in query],
            doseq=True,
        )

        return urlunsplit(
            (
                parsed.scheme,
                parsed.netloc,
                parsed.path,
                safe_query,
                "",
            )
        )

    except Exception:
        return url


def redact_text(text: str) -> str:
    if not text:
        return text

    text = JWT_RE.sub("<JWT_REDACTED>", text)
    text = BEARER_RE.sub("Bearer <REDACTED>", text)

    def replace_url(match):
        return safe_url(match.group(0))

    text = URL_RE.sub(replace_url, text)

    return text


def sanitize(value, key: str | None = None):
    if key and SECRET_KEY_RE.search(key):
        return "<REDACTED>"

    if isinstance(value, dict):
        return {
            str(k): sanitize(v, str(k))
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            sanitize(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return [
            sanitize(item)
            for item in value
        ]

    if isinstance(value, str):
        if value.startswith(("http://", "https://")):
            return safe_url(value)

        return redact_text(value)

    return value


def sanitize_headers(headers: dict) -> dict:
    result = {}

    for key, value in headers.items():
        if SECRET_KEY_RE.search(key):
            result[key] = "<REDACTED>"
        elif key.lower() == "location":
            result[key] = safe_url(value)
        else:
            result[key] = redact_text(value)

    return result


def stable_name(url: str, suffix: str = "") -> str:
    digest = hashlib.sha256(
        url.encode("utf-8", errors="ignore")
    ).hexdigest()[:16]

    basename = Path(
        urlsplit(url).path
    ).name

    basename = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "_",
        basename,
    )[:100]

    if not basename:
        basename = "resource"

    return f"{digest}_{basename}{suffix}"


def resource_role(url: str) -> str | None:
    try:
        name = Path(
            urlsplit(url).path
        ).name.lower()
    except Exception:
        return None

    if name == "video.mp4":
        return "tracking"

    if (
        name.startswith("transcode-")
        and name.endswith(".mp4")
    ):
        return "dual_lens"

    if name.endswith(".veo"):
        return "calibration"

    if name == "detections_constrained.det":
        return "camera_directions"

    if name.endswith(".m3u8"):
        return "hls_manifest"

    if name.endswith(".mpd"):
        return "dash_manifest"

    return None


def physical_click(page, locator, description: str):
    box = locator.bounding_box()

    if not box:
        raise RuntimeError(
            f"No bounding box for {description}"
        )

    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2

    print(
        f"[click] {description} "
        f"({x:.1f}, {y:.1f})"
    )

    page.mouse.move(x, y)
    page.wait_for_timeout(150)

    page.mouse.down()
    page.wait_for_timeout(100)
    page.mouse.up()


def dismiss_login_modal(page) -> bool:
    modals = page.locator(
        LOGIN_MODAL_SELECTOR
    )

    for index in range(
        min(modals.count(), 10)
    ):
        modal = modals.nth(index)

        try:
            if not modal.is_visible():
                continue

            close = modal.locator(
                LOGIN_CLOSE_SELECTOR
            )

            if (
                close.count() == 0
                or not close.first.is_visible()
            ):
                continue

            print(
                "[modal] Closing logged-out "
                "login prompt"
            )

            physical_click(
                page,
                close.first,
                "login modal close",
            )

            try:
                modal.wait_for(
                    state="hidden",
                    timeout=5000,
                )
            except PlaywrightTimeoutError:
                page.wait_for_timeout(1000)

            return True

        except Exception:
            continue

    return False


def surrounding_text(locator) -> str:
    try:
        return locator.evaluate(
            """
            el => {
                let node = el;
                let fallback = "";

                for (
                    let i = 0;
                    node && i < 16;
                    i++, node = node.parentElement
                ) {
                    const text = (
                        node.innerText || ""
                    )
                        .replace(/\\s+/g, " ")
                        .trim();

                    if (!text) continue;

                    fallback = text;

                    if (
                        /Become a member of/i.test(text) ||
                        /Request membership/i.test(text)
                    ) {
                        return text.slice(0, 2000);
                    }
                }

                return fallback.slice(0, 2000);
            }
            """
        ) or ""

    except Exception:
        return ""


def dismiss_membership_modal(page) -> bool:
    buttons = page.locator(
        MEMBERSHIP_CLOSE_SELECTOR
    )

    for index in range(
        min(buttons.count(), 20)
    ):
        button = buttons.nth(index)

        try:
            if not button.is_visible():
                continue

            text = surrounding_text(button)

            if not MEMBERSHIP_TEXT.search(text):
                continue

            print(
                "[modal] Closing membership prompt"
            )

            physical_click(
                page,
                button,
                "membership modal close",
            )

            page.wait_for_timeout(800)

            return True

        except Exception:
            continue

    return False


def dismiss_known_modals(page) -> None:
    dismissed = False

    if dismiss_login_modal(page):
        dismissed = True

    page.wait_for_timeout(250)

    if dismiss_membership_modal(page):
        dismissed = True

    if not dismissed:
        print("[modal] No known blocker found")


def wait_for_player(page) -> bool:
    print("[player] Waiting for video")

    try:
        page.wait_for_function(
            """
            () => [
                ...document.querySelectorAll("video")
            ].some(video =>
                video.readyState >= 2 &&
                video.videoWidth > 0 &&
                video.videoHeight > 0
            )
            """,
            timeout=30000,
        )

        print("[player] Ready")
        return True

    except PlaywrightTimeoutError:
        print("[player] WARNING: not ready")
        return False


def reveal_player_controls(page) -> None:
    try:
        video = page.locator("video").first

        if not video.is_visible():
            return

        box = video.bounding_box()

        if not box:
            return

        page.mouse.move(
            box["x"] + box["width"] / 2,
            box["y"] + box["height"] / 2,
        )

        page.wait_for_timeout(700)

    except Exception:
        pass


def interactive_hit_test(
    page,
    button,
):
    box = button.bounding_box()

    if not box:
        return {
            "ok": False,
            "reason": "no bounding box",
        }

    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2

    return page.evaluate(
        """
        ([x, y]) => {
            const button =
                document.querySelector(
                    "button.interactive-player"
                );

            const hit =
                document.elementFromPoint(x, y);

            if (!button || !hit) {
                return {
                    ok: false,
                    hit: null
                };
            }

            return {
                ok:
                    hit === button ||
                    button.contains(hit),

                hitTag: hit.tagName || null,

                hitClass:
                    typeof hit.className === "string"
                        ? hit.className
                        : null,

                hitAria:
                    hit.getAttribute
                        ? hit.getAttribute(
                            "aria-label"
                        )
                        : null,

                hitText:
                    (hit.innerText || "")
                        .slice(0, 500),

                hitHtml:
                    hit.outerHTML
                        ? hit.outerHTML.slice(
                            0,
                            2000
                        )
                        : ""
            };
        }
        """,
        [x, y],
    )


class DiscoveryRecorder:
    def __init__(
        self,
        root: Path,
    ):
        self.root = root

        self.network_dir = mkdir(
            root / "network"
        )

        self.json_dir = mkdir(
            root / "json"
        )

        self.scripts_dir = mkdir(
            root / "scripts"
        )

        self.manifests_dir = mkdir(
            root / "manifests"
        )

        self.media_dir = mkdir(
            root / "media_refs"
        )

        self.player_dir = mkdir(
            root / "player_state"
        )

        self.storage_dir = mkdir(
            root / "storage"
        )

        self.dom_dir = mkdir(
            root / "dom"
        )

        self.diagnostics_dir = mkdir(
            root / "diagnostics"
        )

        self.requests = []
        self.responses = []
        self.runtime_resources = []
        self.console = []
        self.page_errors = []
        self.websocket_events = []

        self.saved_bodies = {}
        self.seen_bodies = set()

        self.phase = "startup"

    def set_phase(
        self,
        phase: str,
    ):
        print(f"[phase] {phase}")
        self.phase = phase

    def on_request(
        self,
        request,
    ):
        if len(self.requests) >= MAX_EVENTS:
            return

        entry = {
            "time": time.time(),
            "phase": self.phase,
            "method": request.method,
            "resource_type": request.resource_type,
            "url": safe_url(request.url),
        }

        try:
            entry["headers"] = sanitize_headers(
                request.headers
            )
        except Exception:
            pass

        try:
            post_data = request.post_data

            if (
                post_data
                and len(post_data) <= 1_000_000
            ):
                try:
                    payload = json.loads(
                        post_data
                    )

                    entry["post_data"] = sanitize(
                        payload
                    )

                except Exception:
                    entry["post_data"] = redact_text(
                        post_data
                    )[:1_000_000]

        except Exception:
            pass

        self.requests.append(entry)

    def should_save_body(
        self,
        response,
    ) -> tuple[
        bool,
        str | None,
        int,
    ]:
        url = response.url
        path = urlsplit(url).path.lower()

        try:
            content_type = (
                response.headers.get(
                    "content-type",
                    ""
                )
                .split(";")[0]
                .strip()
                .lower()
            )
        except Exception:
            content_type = ""

        if (
            "video/" in content_type
            or path.endswith(
                (".mp4", ".mov", ".webm")
            )
        ):
            return False, None, 0

        if (
            "application/json" in content_type
            or path.endswith(".json")
        ):
            return True, "json", MAX_TEXT_BODY

        if (
            "javascript" in content_type
            or path.endswith(
                (".js", ".mjs")
            )
        ):
            return True, "script", MAX_TEXT_BODY

        if (
            path.endswith(
                (
                    ".m3u8",
                    ".mpd",
                    ".veo",
                    ".glsl",
                    ".vert",
                    ".frag",
                    ".txt",
                )
            )
        ):
            return True, "manifest", MAX_TEXT_BODY

        if content_type == "application/wasm":
            return True, "wasm", MAX_WASM_BODY

        return False, None, 0

    def save_response_body(
        self,
        response,
    ):
        url = response.url

        if url in self.seen_bodies:
            return

        should_save, kind, limit = (
            self.should_save_body(response)
        )

        if not should_save:
            return

        self.seen_bodies.add(url)

        try:
            body = response.body()
        except Exception:
            return

        if (
            not body
            or len(body) > limit
        ):
            return

        if kind == "json":
            try:
                payload = json.loads(
                    body.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

                payload = sanitize(payload)

                destination = (
                    self.json_dir
                    / stable_name(
                        url,
                        ".json",
                    )
                )

                json_dump(
                    destination,
                    payload,
                )

            except Exception:
                destination = (
                    self.json_dir
                    / stable_name(
                        url,
                        ".txt",
                    )
                )

                destination.write_text(
                    redact_text(
                        body.decode(
                            "utf-8",
                            errors="replace",
                        )
                    ),
                    encoding="utf-8",
                )

        elif kind == "script":
            destination = (
                self.scripts_dir
                / stable_name(
                    url,
                    ".js",
                )
            )

            destination.write_bytes(body)

        elif kind == "wasm":
            wasm_dir = mkdir(
                self.scripts_dir / "wasm"
            )

            destination = (
                wasm_dir
                / stable_name(
                    url,
                    ".wasm",
                )
            )

            destination.write_bytes(body)

        else:
            suffix = (
                Path(
                    urlsplit(url).path
                ).suffix
                or ".txt"
            )

            destination = (
                self.manifests_dir
                / stable_name(
                    url,
                    suffix,
                )
            )

            destination.write_text(
                redact_text(
                    body.decode(
                        "utf-8",
                        errors="replace",
                    )
                ),
                encoding="utf-8",
            )

        try:
            relative = str(
                destination.relative_to(
                    self.root
                )
            )

            self.saved_bodies[
                safe_url(url)
            ] = relative

        except Exception:
            pass

    def on_response(
        self,
        response,
    ):
        if len(self.responses) < MAX_EVENTS:
            entry = {
                "time": time.time(),
                "phase": self.phase,
                "status": response.status,
                "url": safe_url(
                    response.url
                ),
            }

            try:
                entry["headers"] = (
                    sanitize_headers(
                        response.headers
                    )
                )
            except Exception:
                pass

            self.responses.append(entry)

        try:
            self.save_response_body(
                response
            )
        except Exception as exc:
            self.console.append(
                {
                    "type": "recorder-error",
                    "text": (
                        f"body capture failed: "
                        f"{type(exc).__name__}"
                    ),
                }
            )

    def add_runtime_url(
        self,
        url: str,
        source: str,
    ):
        if not url:
            return

        safe = safe_url(url)

        key = (
            self.phase,
            source,
            safe,
        )

        for item in self.runtime_resources:
            if (
                item["phase"],
                item["source"],
                item["url"],
            ) == key:
                return

        self.runtime_resources.append(
            {
                "phase": self.phase,
                "source": source,
                "url": safe,
                "role": resource_role(url),
            }
        )

    def on_console(
        self,
        message,
    ):
        try:
            text = message.text

            if callable(text):
                text = text()

            msg_type = message.type

            if callable(msg_type):
                msg_type = msg_type()

            self.console.append(
                {
                    "time": time.time(),
                    "phase": self.phase,
                    "type": str(msg_type),
                    "text": redact_text(
                        str(text)
                    )[:10000],
                }
            )

        except Exception:
            pass

    def on_page_error(
        self,
        error,
    ):
        self.page_errors.append(
            {
                "time": time.time(),
                "phase": self.phase,
                "error": redact_text(
                    str(error)
                )[:20000],
            }
        )

    def on_websocket(
        self,
        websocket,
    ):
        try:
            websocket_url = safe_url(
                websocket.url
            )
        except Exception:
            websocket_url = "<unknown>"

        self.websocket_events.append(
            {
                "time": time.time(),
                "phase": self.phase,
                "event": "opened",
                "url": websocket_url,
            }
        )

        def record_frame(
            direction,
            payload,
        ):
            if (
                len(self.websocket_events)
                >= MAX_WS_FRAMES
            ):
                return

            if isinstance(
                payload,
                bytes,
            ):
                if (
                    len(payload)
                    > MAX_WS_FRAME_SIZE
                ):
                    value = (
                        f"<binary "
                        f"{len(payload)} bytes>"
                    )
                else:
                    value = (
                        payload.hex()
                    )

            else:
                value = str(payload)

                if (
                    len(value)
                    > MAX_WS_FRAME_SIZE
                ):
                    value = (
                        value[
                            :MAX_WS_FRAME_SIZE
                        ]
                        + "...<truncated>"
                    )

                value = redact_text(value)

                try:
                    decoded = json.loads(
                        value
                    )

                    value = sanitize(
                        decoded
                    )

                except Exception:
                    pass

            self.websocket_events.append(
                {
                    "time": time.time(),
                    "phase": self.phase,
                    "event": direction,
                    "url": websocket_url,
                    "payload": value,
                }
            )

        try:
            websocket.on(
                "framesent",
                lambda payload: record_frame(
                    "sent",
                    payload,
                ),
            )

            websocket.on(
                "framereceived",
                lambda payload: record_frame(
                    "received",
                    payload,
                ),
            )

            websocket.on(
                "close",
                lambda: self.websocket_events.append(
                    {
                        "time": time.time(),
                        "phase": self.phase,
                        "event": "closed",
                        "url": websocket_url,
                    }
                ),
            )

        except Exception:
            pass

    def attach_page(
        self,
        page,
    ):
        page.on(
            "console",
            self.on_console,
        )

        page.on(
            "pageerror",
            self.on_page_error,
        )

        page.on(
            "websocket",
            self.on_websocket,
        )

    def flush(
        self,
    ):
        json_dump(
            self.network_dir
            / "requests.json",
            self.requests,
        )

        json_dump(
            self.network_dir
            / "responses.json",
            self.responses,
        )

        json_dump(
            self.network_dir
            / "runtime_resources.json",
            self.runtime_resources,
        )

        json_dump(
            self.network_dir
            / "saved_bodies.json",
            self.saved_bodies,
        )

        json_dump(
            self.network_dir
            / "websockets.json",
            self.websocket_events,
        )

        json_dump(
            self.diagnostics_dir
            / "console.json",
            self.console,
        )

        json_dump(
            self.diagnostics_dir
            / "page_errors.json",
            self.page_errors,
        )


def collect_runtime_resources(
    page,
    recorder: DiscoveryRecorder,
):
    try:
        resources = page.evaluate(
            """
            () => performance
                .getEntriesByType("resource")
                .map(entry => entry.name)
            """
        )

        for url in resources:
            recorder.add_runtime_url(
                url,
                "performance",
            )

    except Exception:
        pass

    try:
        videos = page.evaluate(
            """
            () => [
                ...document.querySelectorAll(
                    "video"
                )
            ].flatMap(video => [
                video.currentSrc,
                video.src,
                ...[
                    ...video.querySelectorAll(
                        "source"
                    )
                ].map(source => source.src)
            ]).filter(Boolean)
            """
        )

        for url in videos:
            recorder.add_runtime_url(
                url,
                "video-element",
            )

    except Exception:
        pass


def capture_page_state(
    page,
    recorder: DiscoveryRecorder,
    label: str,
):
    print(f"[snapshot] {label}")

    collect_runtime_resources(
        page,
        recorder,
    )

    state = page.evaluate(
        """
        () => {
            const controls = [
                ...document.querySelectorAll(
                    'button, a, [role="button"], ' +
                    '[role="tab"], [role="menuitem"], ' +
                    '[role="radio"]'
                )
            ].slice(0, 1500).map((el) => ({
                tag: el.tagName,
                text: (
                    el.innerText || ""
                ).replace(/\\s+/g, " ")
                 .trim()
                 .slice(0, 500),

                ariaLabel:
                    el.getAttribute(
                        "aria-label"
                    ),

                title:
                    el.getAttribute(
                        "title"
                    ),

                role:
                    el.getAttribute(
                        "role"
                    ),

                href:
                    el.href || null,

                className:
                    typeof el.className ===
                    "string"
                        ? el.className
                        : null,

                visible:
                    Boolean(
                        el.offsetWidth ||
                        el.offsetHeight ||
                        el.getClientRects().length
                    )
            }));

            const videos = [
                ...document.querySelectorAll(
                    "video"
                )
            ].map((video, index) => ({
                index,
                currentSrc:
                    video.currentSrc || "",
                src:
                    video.getAttribute(
                        "src"
                    ) || "",
                videoWidth:
                    video.videoWidth,
                videoHeight:
                    video.videoHeight,
                readyState:
                    video.readyState,
                paused:
                    video.paused,
                currentTime:
                    video.currentTime,
                duration:
                    Number.isFinite(
                        video.duration
                    )
                        ? video.duration
                        : null,
                playbackRate:
                    video.playbackRate
            }));

            const canvases = [
                ...document.querySelectorAll(
                    "canvas"
                )
            ].map((canvas, index) => ({
                index,
                width:
                    canvas.width,
                height:
                    canvas.height,
                clientWidth:
                    canvas.clientWidth,
                clientHeight:
                    canvas.clientHeight,
                className:
                    typeof canvas.className ===
                    "string"
                        ? canvas.className
                        : null
            }));

            const scriptJson = [
                ...document.querySelectorAll(
                    'script[type="application/json"]'
                )
            ].map((script, index) => ({
                index,
                id:
                    script.id || null,
                className:
                    typeof script.className ===
                    "string"
                        ? script.className
                        : null,
                text:
                    (
                        script.textContent || ""
                    ).slice(
                        0,
                        2_000_000
                    )
            }));

            const scriptSources = [
                ...document.querySelectorAll(
                    "script[src]"
                )
            ].map(
                script => script.src
            );

            const globalNames =
                Object.keys(window)
                    .filter(name =>
                        /camera|virtual|fov|tracking|player|match|lineup|roster|event|timeline|pitch/i
                            .test(name)
                    )
                    .slice(0, 500)
                    .map(name => {
                        let value;

                        try {
                            value = window[name];
                        } catch {
                            return {
                                name,
                                type: "unreadable"
                            };
                        }

                        const result = {
                            name,
                            type: typeof value
                        };

                        if (
                            value === null ||
                            typeof value ===
                                "string" ||
                            typeof value ===
                                "number" ||
                            typeof value ===
                                "boolean"
                        ) {
                            result.value = value;
                        } else if (
                            Array.isArray(value)
                        ) {
                            result.length =
                                value.length;
                        } else if (
                            typeof value ===
                            "object"
                        ) {
                            try {
                                result.keys =
                                    Object.keys(
                                        value
                                    ).slice(
                                        0,
                                        100
                                    );
                            } catch {}
                        }

                        return result;
                    });

            return {
                url:
                    location.href,

                title:
                    document.title,

                bodyText:
                    (
                        document.body
                            ?.innerText || ""
                    ).slice(
                        0,
                        2_000_000
                    ),

                controls,
                videos,
                canvases,
                scriptJson,
                scriptSources,
                globalNames
            };
        }
        """
    )

    state = sanitize(state)

    json_dump(
        recorder.player_dir
        / f"{label}.json",
        {
            key: value
            for key, value
            in state.items()
            if key not in {
                "bodyText",
                "scriptJson",
            }
        },
    )

    (
        recorder.dom_dir
        / f"{label}_body.txt"
    ).write_text(
        redact_text(
            state.get(
                "bodyText",
                "",
            )
        ),
        encoding="utf-8",
    )

    json_dump(
        recorder.dom_dir
        / f"{label}_controls.json",
        state.get(
            "controls",
            [],
        ),
    )

    json_dump(
        recorder.dom_dir
        / f"{label}_inline_json.json",
        state.get(
            "scriptJson",
            [],
        ),
    )

    try:
        page.screenshot(
            path=str(
                recorder.diagnostics_dir
                / f"{label}.png"
            ),
            full_page=True,
        )
    except Exception:
        pass

    capture_storage(
        page,
        recorder,
        label,
    )


def capture_storage(
    page,
    recorder: DiscoveryRecorder,
    label: str,
):
    try:
        storage = page.evaluate(
            """
            () => {
                const read = storage => {
                    const result = {};

                    for (
                        let i = 0;
                        i < storage.length;
                        i++
                    ) {
                        const key =
                            storage.key(i);

                        result[key] =
                            storage.getItem(
                                key
                            );
                    }

                    return result;
                };

                return {
                    localStorage:
                        read(
                            localStorage
                        ),

                    sessionStorage:
                        read(
                            sessionStorage
                        )
                };
            }
            """
        )

        json_dump(
            recorder.storage_dir
            / f"{label}.json",
            sanitize(storage),
        )

    except Exception as exc:
        json_dump(
            recorder.storage_dir
            / f"{label}.json",
            {
                "error":
                    type(exc).__name__
            },
        )


def scan_scripts(
    root: Path,
):
    scripts_dir = root / "scripts"

    results = []

    for path in sorted(
        scripts_dir.rglob("*.js")
    ):
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            continue

        file_result = {
            "file": str(
                path.relative_to(root)
            ),
            "size_bytes": path.stat().st_size,
            "hits": {},
        }

        lower = text.lower()

        for keyword in DISCOVERY_KEYWORDS:
            needle = keyword.lower()

            positions = []

            start = 0

            while True:
                index = lower.find(
                    needle,
                    start,
                )

                if index < 0:
                    break

                positions.append(index)

                if len(positions) >= 25:
                    break

                start = index + len(
                    needle
                )

            if not positions:
                continue

            snippets = []

            for index in positions:
                left = max(
                    0,
                    index - 350,
                )

                right = min(
                    len(text),
                    index
                    + len(keyword)
                    + 500,
                )

                snippet = (
                    text[left:right]
                    .replace("\n", " ")
                    .replace("\r", " ")
                )

                snippets.append(
                    redact_text(
                        snippet
                    )
                )

            file_result[
                "hits"
            ][keyword] = snippets

        if file_result["hits"]:
            results.append(
                file_result
            )

    json_dump(
        root
        / "script_keyword_hits.json",
        results,
    )

    return results


def build_endpoint_inventory(
    recorder: DiscoveryRecorder,
):
    endpoints = {}

    for entry in (
        recorder.requests
        + recorder.responses
    ):
        url = entry.get("url")

        if not url:
            continue

        try:
            parsed = urlsplit(url)

            key = (
                f"{parsed.scheme}://"
                f"{parsed.netloc}"
                f"{parsed.path}"
            )
        except Exception:
            key = url

        info = endpoints.setdefault(
            key,
            {
                "url": key,
                "methods": set(),
                "resource_types": set(),
                "phases": set(),
                "statuses": set(),
            },
        )

        if entry.get("method"):
            info["methods"].add(
                entry["method"]
            )

        if entry.get("resource_type"):
            info["resource_types"].add(
                entry["resource_type"]
            )

        if entry.get("phase"):
            info["phases"].add(
                entry["phase"]
            )

        if entry.get("status") is not None:
            info["statuses"].add(
                entry["status"]
            )

    result = []

    for value in endpoints.values():
        result.append(
            {
                "url": value["url"],
                "methods": sorted(
                    value["methods"]
                ),
                "resource_types": sorted(
                    value[
                        "resource_types"
                    ]
                ),
                "phases": sorted(
                    value["phases"]
                ),
                "statuses": sorted(
                    value["statuses"]
                ),
            }
        )

    result.sort(
        key=lambda item: item["url"]
    )

    json_dump(
        recorder.network_dir
        / "endpoint_inventory.json",
        result,
    )

    return result


def build_media_inventory(
    recorder: DiscoveryRecorder,
):
    urls = {}

    def add(
        url,
        phase=None,
        source=None,
    ):
        if not url:
            return

        role = resource_role(url)

        path = urlsplit(url).path.lower()

        if not (
            role
            or path.endswith(
                (
                    ".mp4",
                    ".veo",
                    ".m3u8",
                    ".mpd",
                    ".mov",
                    ".webm",
                )
            )
        ):
            return

        safe = safe_url(url)

        item = urls.setdefault(
            safe,
            {
                "url": safe,
                "role": role,
                "phases": set(),
                "sources": set(),
            },
        )

        if phase:
            item["phases"].add(
                phase
            )

        if source:
            item["sources"].add(
                source
            )

    for item in recorder.runtime_resources:
        add(
            item["url"],
            item.get("phase"),
            item.get("source"),
        )

    for item in recorder.requests:
        add(
            item.get("url"),
            item.get("phase"),
            "request",
        )

    for item in recorder.responses:
        add(
            item.get("url"),
            item.get("phase"),
            "response",
        )

    result = []

    for item in urls.values():
        result.append(
            {
                "url": item["url"],
                "role": item["role"],
                "phases": sorted(
                    item["phases"]
                ),
                "sources": sorted(
                    item["sources"]
                ),
            }
        )

    result.sort(
        key=lambda item: (
            item.get("role") or "",
            item["url"],
        )
    )

    json_dump(
        recorder.media_dir
        / "index.json",
        result,
    )

    return result


def find_primary_assets(
    media_inventory,
):
    result = {
        "tracking": None,
        "dual_lens": None,
        "calibration": None,
        "camera_directions": None,
    }

    for item in media_inventory:
        role = item.get("role")

        if (
            role in result
            and result[role] is None
        ):
            result[role] = item["url"]

    return result


def build_structured_manifest(root: Path, match_url: str, primary_assets: dict, endpoints: list[dict], recorder) -> dict:
    """Interpret the tool's bounded JSON artifacts with the app's shared rules."""
    discovery = replay_capture(root, recorder.responses, recorder.saved_bodies, root / "datasets")
    assets = {"tracking": primary_assets.get("tracking"), "dual_lens": primary_assets.get("dual_lens"),
              "calibration": primary_assets.get("calibration"), "camera_directions": None}
    tracking_url = primary_assets.get("tracking")
    tracking_key = normalized_resource_url(tracking_url) if tracking_url else None
    for safe_url_value, relative_path in recorder.saved_bodies.items():
        json_path = root / relative_path
        if json_path.suffix != ".json" or not json_path.is_file():
            continue
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for video_url, directions_url in standard_video_camera_directions(payload):
            if tracking_key and normalized_resource_url(video_url) == tracking_key:
                assets["camera_directions"] = sanitize_url(directions_url)
                break
        if assets["camera_directions"]:
            break
    post_interactive_evidence = any(
        item.get("phase") == "interactive"
        for item in [*recorder.runtime_resources, *recorder.responses]
    )
    interactive_confirmed = bool(getattr(recorder, "interactive_activated", False) and post_interactive_evidence)
    discovery.record_interactive(interactive_confirmed, {
        "button_selector": PRIMARY_INTERACTIVE_SELECTOR,
        "physical_click_succeeded": bool(getattr(recorder, "interactive_activated", False)),
        "post_interactive_resources_observed": post_interactive_evidence,
    })
    return discovery.manifest({"source_url": safe_url(match_url)}, assets, {
        "capture_root": str(root), "capture_sources": ["requests", "responses", "runtime", "json", "dom/player"],
        "endpoint_count": len(endpoints), "saved_json_count": sum(1 for path in (root / "json").glob("*.json")),
    })


def write_readme(
    root: Path,
):
    text = """# Veo discovery capture

This directory was generated by tools/veo_discover.py.

It is a discovery snapshot, not an application runtime match.

Main files:

- index.json: summary and discovered primary assets
- manifest.json: structured match-discovery inventory shared with the app inspector
- network/requests.json: sanitized browser requests
- network/responses.json: sanitized browser responses
- network/endpoint_inventory.json: deduplicated endpoint list
- network/websockets.json: observed WebSocket frames when available
- json/: captured small JSON responses
- scripts/: JavaScript/WASM assets loaded by the page
- script_keyword_hits.json: automatic searches for camera, tracking, player, event and lineup terms
- manifests/: small manifests, .veo, shaders and related text resources
- media_refs/index.json: media URLs and logical roles without downloading large videos
- player_state/: video/canvas/control state snapshots
- dom/: visible text, controls and inline JSON snapshots
- storage/: sanitized localStorage/sessionStorage snapshots
- diagnostics/: screenshots, console messages and page errors

Passwords, cookies, authorization headers, tokens and signed query values are intentionally redacted.

Large media files are referenced but are not downloaded.
"""

    (
        root / "README.md"
    ).write_text(
        text,
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Authorized Veo web-player "
            "discovery capture"
        )
    )

    parser.add_argument(
        "--match-url",
        required=True,
    )

    parser.add_argument(
        "--out-root",
        default=(
            "the external VEO_MATCH_DATA_ROOT/"
            "veo-discovery"
        ),
    )

    parser.add_argument(
        "--post-seconds",
        type=float,
        default=20.0,
        help=(
            "Seconds to continue observing "
            "after Interactive loads"
        ),
    )

    args = parser.parse_args()

    parsed = urlsplit(
        args.match_url
    )

    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or not (
            parsed.hostname == "veo.co"
            or parsed.hostname.endswith(
                ".veo.co"
            )
        )
    ):
        raise SystemExit(
            "Expected an https://*.veo.co URL"
        )

    slug = (
        parsed.path.rstrip("/")
        .split("/")[-1]
        or "match"
    )

    slug = re.sub(
        r"[^a-zA-Z0-9._-]+",
        "-",
        slug,
    )[:100]

    timestamp = datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )

    capture_id = (
        f"{slug}-{timestamp}-"
        f"{uuid.uuid4().hex[:6]}"
    )

    root = mkdir(
        Path(args.out_root)
        / capture_id
    )

    recorder = DiscoveryRecorder(
        root
    )
    recorder.interactive_activated = False

    write_readme(root)

    with tempfile.TemporaryDirectory(
        prefix="veo-discovery-profile-"
    ) as profile_dir:

        with sync_playwright() as p:
            context = (
                p.chromium
                .launch_persistent_context(
                    user_data_dir=profile_dir,
                    headless=False,
                    viewport={
                        "width": 1600,
                        "height": 1000,
                    },
                )
            )

            context.on(
                "request",
                recorder.on_request,
            )

            context.on(
                "response",
                recorder.on_response,
            )

            def on_new_page(page):
                recorder.attach_page(
                    page
                )

            context.on(
                "page",
                on_new_page,
            )

            page = (
                context.pages[0]
                if context.pages
                else context.new_page()
            )

            recorder.attach_page(page)

            try:
                recorder.set_phase(
                    "initial_navigation"
                )

                print(
                    f"[nav] {args.match_url}"
                )

                try:
                    page.goto(
                        args.match_url,
                        wait_until=(
                            "domcontentloaded"
                        ),
                        timeout=90000,
                    )
                except PlaywrightTimeoutError:
                    print(
                        "[nav] Timeout; "
                        "continuing"
                    )

                page.wait_for_timeout(
                    5000
                )

                capture_page_state(
                    page,
                    recorder,
                    "01_initial",
                )

                recorder.set_phase(
                    "default_player"
                )

                dismiss_known_modals(
                    page
                )

                page.wait_for_timeout(
                    1200
                )

                wait_for_player(
                    page
                )

                reveal_player_controls(
                    page
                )

                dismiss_known_modals(
                    page
                )

                page.wait_for_timeout(
                    500
                )

                reveal_player_controls(
                    page
                )

                capture_page_state(
                    page,
                    recorder,
                    "02_default_ready",
                )

                interactive = page.locator(
                    PRIMARY_INTERACTIVE_SELECTOR
                ).first

                try:
                    interactive.wait_for(
                        state="visible",
                        timeout=15000,
                    )
                except PlaywrightTimeoutError:
                    raise RuntimeError(
                        "Interactive button "
                        "was not visible"
                    )

                hit = interactive_hit_test(
                    page,
                    interactive,
                )

                json_dump(
                    recorder.diagnostics_dir
                    / "interactive_hit_test.json",
                    sanitize(hit),
                )

                print(
                    "[interactive] hit test:"
                )

                print(
                    json.dumps(
                        hit,
                        indent=2,
                        ensure_ascii=False,
                    )
                )

                if not hit.get("ok"):
                    dismiss_known_modals(
                        page
                    )

                    page.wait_for_timeout(
                        1000
                    )

                    reveal_player_controls(
                        page
                    )

                    hit = (
                        interactive_hit_test(
                            page,
                            interactive,
                        )
                    )

                if not hit.get("ok"):
                    raise RuntimeError(
                        "Interactive is still "
                        "covered by another "
                        "element"
                    )

                recorder.set_phase(
                    "interactive_activation"
                )

                capture_page_state(
                    page,
                    recorder,
                    "03_before_interactive",
                )

                physical_click(
                    page,
                    interactive,
                    "Interactive",
                )
                recorder.interactive_activated = True

                recorder.set_phase(
                    "interactive"
                )

                print(
                    "[interactive] "
                    "Capturing resources..."
                )

                deadline = (
                    time.monotonic()
                    + max(
                        args.post_seconds,
                        5.0,
                    )
                )

                while (
                    time.monotonic()
                    < deadline
                ):
                    collect_runtime_resources(
                        page,
                        recorder,
                    )

                    page.wait_for_timeout(
                        250
                    )

                capture_page_state(
                    page,
                    recorder,
                    "04_interactive",
                )

                print(
                    "[interactive] "
                    "Capture complete"
                )

            except Exception as exc:
                print(
                    "[ERROR]",
                    type(exc).__name__,
                    str(exc),
                )

                json_dump(
                    recorder.diagnostics_dir
                    / "fatal_error.json",
                    {
                        "type":
                            type(exc).__name__,
                        "message":
                            str(exc),
                    },
                )

                try:
                    page.screenshot(
                        path=str(
                            recorder
                            .diagnostics_dir
                            / "fatal_error.png"
                        ),
                        full_page=True,
                    )
                except Exception:
                    pass

            finally:
                try:
                    collect_runtime_resources(
                        page,
                        recorder,
                    )
                except Exception:
                    pass

                recorder.flush()

                try:
                    context.close()
                except Exception:
                    pass

    endpoints = (
        build_endpoint_inventory(
            recorder
        )
    )

    media_inventory = (
        build_media_inventory(
            recorder
        )
    )

    primary_assets = (
        find_primary_assets(
            media_inventory
        )
    )

    manifest = build_structured_manifest(
        root,
        args.match_url,
        primary_assets,
        endpoints,
        recorder,
    )
    primary_assets["camera_directions"] = manifest["assets"].get("camera_directions")
    write_manifest(root / "manifest.json", manifest)

    keyword_hits = scan_scripts(
        root
    )

    summary = {
        "capture_id": capture_id,
        "match_url": safe_url(
            args.match_url
        ),
        "created_at": datetime.now().isoformat(),
        "primary_assets": primary_assets,
        "counts": {
            "requests":
                len(recorder.requests),
            "responses":
                len(recorder.responses),
            "runtime_resources":
                len(
                    recorder.runtime_resources
                ),
            "endpoints":
                len(endpoints),
            "saved_bodies":
                len(
                    recorder.saved_bodies
                ),
            "websocket_events":
                len(
                    recorder.websocket_events
                ),
            "script_files":
                len(
                    list(
                        (
                            root
                            / "scripts"
                        ).rglob("*")
                    )
                ),
            "script_files_with_keyword_hits":
                len(keyword_hits),
        },
    }

    json_dump(
        root / "index.json",
        summary,
    )

    archive_base = str(root)

    archive_path = shutil.make_archive(
        archive_base,
        "zip",
        root_dir=root,
    )

    print()
    print("=" * 70)
    print("VEO DISCOVERY COMPLETE")
    print("=" * 70)

    print()
    print(
        "Assets\n" + "\n".join(
            f"  {name.replace('_', ' ').title():<20} {'FOUND' if value else 'NOT OBSERVED'}"
            for name, value in primary_assets.items()
        )
    )

    print("\nMatch datasets")
    for name, dataset in manifest["datasets"].items():
        print(f"  {name.replace('_', ' ').title():<20} {dataset['status'].upper()}")

    print("\nManifest:")
    print(root / "manifest.json")

    print()
    print(
        f"Directory:\n{root}"
    )

    print()
    print(
        f"ZIP:\n{archive_path}"
    )

    print()
    print(
        "Upload the ZIP to ChatGPT "
        "and we can audit everything "
        "Veo exposed."
    )


if __name__ == "__main__":
    main()
