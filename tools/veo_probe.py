#!/usr/bin/env python3

import argparse
import json
import re
import time
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import (
    TimeoutError as PlaywrightTimeoutError,
    sync_playwright,
)


def classify_url(url: str):
    """Return the logical Veo asset role for a URL, if relevant."""

    try:
        path = urlparse(url).path
        filename = Path(path).name
    except Exception:
        return None

    filename_lower = filename.lower()

    if filename_lower == "video.mp4":
        return "tracking"

    if re.fullmatch(
        r"transcode-[^/]+\.mp4",
        filename,
        re.IGNORECASE,
    ):
        return "dual_lens"

    if filename_lower.endswith(".veo"):
        return "calibration"

    return None


def physical_click(page, locator, description: str):
    """Perform a real pointer-style click on a locator."""

    box = locator.bounding_box()

    if not box:
        raise RuntimeError(
            f"No bounding box available for {description}"
        )

    x = box["x"] + box["width"] / 2
    y = box["y"] + box["height"] / 2

    print(
        f"[click] {description} at "
        f"({x:.1f}, {y:.1f})"
    )

    page.mouse.move(x, y)
    page.wait_for_timeout(150)
    page.mouse.down()
    page.wait_for_timeout(100)
    page.mouse.up()


def surrounding_text(locator):
    """
    Walk up through the DOM and return useful surrounding text.

    This avoids depending on role="dialog", because Veo's overlays
    are not always semantically marked as dialogs.
    """

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
                    const text = (node.innerText || "")
                        .replace(/\\s+/g, " ")
                        .trim();

                    if (!text) {
                        continue;
                    }

                    fallback = text;

                    if (
                        /Become a member of/i.test(text) ||
                        /Request membership/i.test(text) ||
                        /Log in to access your powerful tools/i.test(text) ||
                        /Create account/i.test(text)
                    ) {
                        return text.slice(0, 1500);
                    }
                }

                return fallback.slice(0, 1500);
            }
            """
        )
    except Exception:
        return ""


def dismiss_login_modal(page):
    """
    Handle the logged-out Veo popup:

        .modal.modal-login-prompt
        button[aria-label="Close button"]
    """

    modals = page.locator(".modal.modal-login-prompt")

    count = modals.count()

    if count == 0:
        return False

    print(
        f"[modal] Found {count} login modal candidate(s)"
    )

    for i in range(count):
        modal = modals.nth(i)

        try:
            if not modal.is_visible():
                continue

            text = modal.inner_text().strip()

            print(
                "[modal] Visible login modal: "
                f"{text[:160]!r}"
            )

            close_buttons = modal.locator(
                'button[aria-label="Close button"]'
            )

            if close_buttons.count() == 0:
                print(
                    "[modal] Login modal found but "
                    "Close button was not found"
                )
                continue

            close_button = close_buttons.first

            if not close_button.is_visible():
                print(
                    "[modal] Login close button exists "
                    "but is not visible"
                )
                continue

            print("[modal] Login prompt detected")
            print("[modal] Closing login prompt")

            physical_click(
                page,
                close_button,
                "login prompt close button",
            )

            try:
                modal.wait_for(
                    state="hidden",
                    timeout=5000,
                )
            except PlaywrightTimeoutError:
                page.wait_for_timeout(1000)

            try:
                still_visible = modal.is_visible()
            except Exception:
                still_visible = False

            if not still_visible:
                print("[modal] Login prompt dismissed")
                return True

            print(
                "[modal] Login prompt is still visible"
            )

        except Exception as exc:
            print(
                "[modal] Login modal handling failed: "
                f"{exc}"
            )

    return False


def dismiss_membership_modal(page):
    """
    Handle the logged-in membership popup.

    Known close control:

        button[aria-label="x"]

    We deliberately inspect surrounding text so we do not blindly
    click arbitrary aria-label="x" buttons elsewhere in Veo.
    """

    buttons = page.locator(
        'button[aria-label="x"]'
    )

    count = buttons.count()

    if count == 0:
        return False

    print(
        f"[modal] Found {count} membership-X "
        "candidate(s)"
    )

    for i in range(count):
        button = buttons.nth(i)

        try:
            if not button.is_visible():
                continue

            text = surrounding_text(button)
            text_lower = text.lower()

            print(
                f"[modal] Visible X candidate {i}: "
                f"{text[:160]!r}"
            )

            if (
                "become a member of" not in text_lower
                and "request membership" not in text_lower
            ):
                continue

            print("[modal] Membership prompt detected")
            print("[modal] Closing membership prompt")

            physical_click(
                page,
                button,
                "membership prompt close button",
            )

            page.wait_for_timeout(800)

            try:
                if not button.is_visible():
                    print(
                        "[modal] Membership prompt dismissed"
                    )
                    return True
            except Exception:
                # Detached from DOM is also success.
                print(
                    "[modal] Membership prompt dismissed"
                )
                return True

        except Exception as exc:
            print(
                "[modal] Membership modal handling failed: "
                f"{exc}"
            )

    return False


def dismiss_known_blocking_modals(page):
    """
    Dismiss the known Veo overlays that can physically block the
    Interactive control.
    """

    dismissed_any = False

    if dismiss_login_modal(page):
        dismissed_any = True

    page.wait_for_timeout(300)

    if dismiss_membership_modal(page):
        dismissed_any = True

    if not dismissed_any:
        print(
            "[modal] No known blocking modal was dismissed"
        )

    return dismissed_any


def wait_for_player(page):
    """
    Wait until at least one HTMLVideoElement has media loaded.
    """

    print(
        "[player] Waiting for usable "
        "HTMLVideoElement..."
    )

    try:
        page.wait_for_function(
            """
            () => {
                const videos = [
                    ...document.querySelectorAll("video")
                ];

                return videos.some(
                    video =>
                        video.readyState >= 2 &&
                        video.videoWidth > 0 &&
                        video.videoHeight > 0
                );
            }
            """,
            timeout=30000,
        )

        print("[player] Video ready")
        return True

    except PlaywrightTimeoutError:
        print(
            "[player] WARNING: video did not reach "
            "ready state"
        )
        return False


def reveal_player_controls(page):
    """
    Hover the visible video/player area in case Veo hides its controls.
    """

    try:
        video = page.locator("video").first

        if video.count() and video.is_visible():
            box = video.bounding_box()

            if box:
                page.mouse.move(
                    box["x"] + box["width"] / 2,
                    box["y"] + box["height"] / 2,
                )

                page.wait_for_timeout(700)
                return

    except Exception as exc:
        print(
            f"[player] Could not hover video: {exc}"
        )


def interactive_hit_test(page, locator):
    """
    Verify that the actual Interactive button owns its center point.
    """

    box = locator.bounding_box()

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

            if (!button) {
                return {
                    ok: false,
                    reason: "interactive button missing"
                };
            }

            if (!hit) {
                return {
                    ok: false,
                    reason: "elementFromPoint returned null"
                };
            }

            return {
                ok:
                    hit === button ||
                    button.contains(hit),

                hitTag:
                    hit.tagName || null,

                hitClass:
                    typeof hit.className === "string"
                        ? hit.className
                        : null,

                hitAria:
                    hit.getAttribute
                        ? hit.getAttribute("aria-label")
                        : null,

                hitText:
                    (hit.innerText || "")
                        .slice(0, 400),

                hitHtml:
                    hit.outerHTML
                        ? hit.outerHTML.slice(0, 1800)
                        : ""
            };
        }
        """,
        [x, y],
    )


def scan_runtime_resources(page, record):
    """
    Scan sources that may contain resource URLs even when a request
    was served from cache or assigned dynamically.
    """

    try:
        resources = page.evaluate(
            """
            () =>
                performance
                    .getEntriesByType("resource")
                    .map(entry => entry.name)
            """
        )

        for url in resources:
            record(
                url,
                "performance",
            )

    except Exception:
        pass

    try:
        video_urls = page.evaluate(
            """
            () => [
                ...document.querySelectorAll("video")
            ].flatMap(video => [
                video.currentSrc,
                video.src,

                ...[
                    ...video.querySelectorAll("source")
                ].map(source => source.src)
            ]).filter(Boolean)
            """
        )

        for url in video_urls:
            record(
                url,
                "video-element",
            )

    except Exception:
        pass


def print_video_state(page):
    try:
        videos = page.evaluate(
            """
            () => [
                ...document.querySelectorAll("video")
            ].map((video, index) => ({
                index,
                currentSrc: video.currentSrc,
                videoWidth: video.videoWidth,
                videoHeight: video.videoHeight,
                readyState: video.readyState,
                paused: video.paused,
                currentTime: video.currentTime
            }))
            """
        )

        print(
            "[player] Videos:\n"
            + json.dumps(
                videos,
                indent=2,
                ensure_ascii=False,
            )
        )

    except Exception as exc:
        print(
            f"[player] Could not inspect videos: {exc}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Probe a Veo Match and discover its "
            "Tracking, Dual-lens and .veo resources."
        )
    )

    parser.add_argument(
        "--match-url",
        required=True,
        help="Authorized/public Veo Match URL",
    )

    parser.add_argument(
        "--profile-dir",
        default=(
            "the external VEO_MATCH_DATA_ROOT/"
            "veo-probe-profile"
        ),
        help=(
            "Persistent Chromium profile used by "
            "the probe."
        ),
    )

    parser.add_argument(
        "--output",
        default="veo_probe_result.json",
        help="Output JSON file",
    )

    args = parser.parse_args()

    found = {
        "tracking": None,
        "dual_lens": None,
        "calibration": None,
    }

    events = []
    seen_events = set()

    def record(url, source, status=None):
        if not url:
            return

        role = classify_url(url)

        if role is None:
            return

        key = (
            role,
            url,
            source,
            status,
        )

        if key in seen_events:
            return

        seen_events.add(key)

        event = {
            "role": role,
            "source": source,
            "status": status,
            "url": url,
        }

        events.append(event)

        if found[role] is None:
            found[role] = url

            print(
                f"\n[FOUND] {role}:"
                f"\n{url}\n"
            )

    profile = Path(args.profile_dir)
    profile.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = Path(args.output)

    with sync_playwright() as p:
        context = (
            p.chromium.launch_persistent_context(
                user_data_dir=str(profile),
                headless=False,
                viewport={
                    "width": 1600,
                    "height": 1000,
                },
            )
        )

        # Start observation BEFORE navigation.
        context.on(
            "request",
            lambda request: record(
                request.url,
                "request",
            ),
        )

        context.on(
            "response",
            lambda response: record(
                response.url,
                "response",
                response.status,
            ),
        )

        if context.pages:
            page = context.pages[0]
        else:
            page = context.new_page()

        print(
            f"[nav] Opening {args.match_url}"
        )

        try:
            page.goto(
                args.match_url,
                wait_until="domcontentloaded",
                timeout=90000,
            )

        except PlaywrightTimeoutError:
            print(
                "[nav] Navigation timeout, "
                "continuing with current page"
            )

        page.wait_for_timeout(5000)

        print(
            f"[nav] Current URL: {page.url}"
        )

        # Discover Tracking from the initial/default player.
        scan_runtime_resources(
            page,
            record,
        )

        print_video_state(page)

        # First attempt to clear known overlays.
        dismiss_known_blocking_modals(page)

        page.wait_for_timeout(1200)

        wait_for_player(page)

        reveal_player_controls(page)

        # Modal can appear after player initialization,
        # so check again.
        dismiss_known_blocking_modals(page)

        page.wait_for_timeout(500)

        reveal_player_controls(page)

        interactive = page.locator(
            "button.interactive-player"
        ).first

        try:
            interactive.wait_for(
                state="visible",
                timeout=15000,
            )

        except PlaywrightTimeoutError:
            print(
                "[interactive] ERROR: "
                "button.interactive-player "
                "is not visible"
            )

            page.screenshot(
                path=(
                    "veo_probe_"
                    "no_interactive.png"
                ),
                full_page=True,
            )

            context.close()
            raise SystemExit(2)

        print(
            "[interactive] Button found and visible"
        )

        hit = interactive_hit_test(
            page,
            interactive,
        )

        print(
            "[interactive] elementFromPoint result:"
        )

        print(
            json.dumps(
                hit,
                indent=2,
                ensure_ascii=False,
            )
        )

        # If still blocked, inspect and dismiss again.
        if not hit.get("ok"):
            print(
                "[interactive] Button is still covered."
            )

            if hit.get("hitClass"):
                print(
                    "[interactive] Blocking class: "
                    f"{hit['hitClass']}"
                )

            if hit.get("hitText"):
                print(
                    "[interactive] Blocking text: "
                    f"{hit['hitText'][:400]!r}"
                )

            print(
                "[interactive] Trying known "
                "modal dismissal once more..."
            )

            dismiss_known_blocking_modals(page)

            page.wait_for_timeout(1000)

            reveal_player_controls(page)

            hit = interactive_hit_test(
                page,
                interactive,
            )

            print(
                "[interactive] Second hit test:"
            )

            print(
                json.dumps(
                    hit,
                    indent=2,
                    ensure_ascii=False,
                )
            )

        if not hit.get("ok"):
            page.screenshot(
                path="veo_probe_blocked.png",
                full_page=True,
            )

            print(
                "\nERROR: Interactive is still "
                "physically covered."
            )

            print(
                "The covering element is shown "
                "in the diagnostic above."
            )

            print(
                "Screenshot: "
                "veo_probe_blocked.png"
            )

            context.close()
            raise SystemExit(3)

        page.screenshot(
            path=(
                "veo_probe_"
                "before_interactive.png"
            ),
            full_page=True,
        )

        # Take another resource snapshot before click.
        scan_runtime_resources(
            page,
            record,
        )

        print(
            "[interactive] Performing "
            "REAL mouse click"
        )

        physical_click(
            page,
            interactive,
            "Interactive",
        )

        # Observe for lazy-loaded resources.
        print(
            "[interactive] Waiting for "
            "transcode and calibration..."
        )

        deadline = time.time() + 30

        while time.time() < deadline:
            scan_runtime_resources(
                page,
                record,
            )

            if (
                found["dual_lens"]
                and found["calibration"]
            ):
                break

            page.wait_for_timeout(250)

        print_video_state(page)

        page.screenshot(
            path=(
                "veo_probe_"
                "after_interactive.png"
            ),
            full_page=True,
        )

        result = {
            "match_url": args.match_url,
            "assets": found,
            "success": all(
                found.values()
            ),
            "events": events,
        }

        output_path.write_text(
            json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        print()
        print("=" * 60)
        print("RESULT")
        print("=" * 60)

        print(
            json.dumps(
                found,
                indent=2,
                ensure_ascii=False,
            )
        )

        print()
        print(
            f"Saved JSON: {output_path}"
        )

        if all(found.values()):
            print()
            print(
                "SUCCESS: all three Veo "
                "assets were discovered."
            )

            print()
            print("Tracking:")
            print(found["tracking"])

            print()
            print("Dual lens:")
            print(found["dual_lens"])

            print()
            print("Calibration:")
            print(found["calibration"])

        else:
            print()
            print(
                "Probe completed, but some "
                "resources are still missing:"
            )

            for role, url in found.items():
                if url is None:
                    print(
                        f" - {role}"
                    )

        print()
        print(
            "Browser left open so you can "
            "inspect the final state."
        )

        input(
            "Press ENTER to close..."
        )

        context.close()


if __name__ == "__main__":
    main()
