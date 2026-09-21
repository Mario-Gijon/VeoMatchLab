from __future__ import annotations

from pathlib import Path
import shutil
import json

from web_app.backend.app import main
from web_app.backend.app.importers.veo.classifier import classify_resource, ranked_candidates, sanitize_url
from web_app.backend.app.importers.veo.inspector import AFTER_INTERACTIVE_PHASE, INITIAL_PHASE, activate_interactive, bounded_json_response, camera_directions_for_tracking, dismiss_membership_modal, generated_highlights_videos, initial_player_video_url, phase_diagnostics, resource_links, selected_tracking_resource, standard_follow_videos, standard_video_camera_directions
from web_app.backend.app.importers.veo.models import VeoCandidate, VeoInspection, importable
from web_app.backend.app.importers.veo.service import ImportJob, VeoImportService
from web_app.backend.app.schemas import VeoInspectRequest
from web_app.backend.app.storage import WorkspaceStore


ROOT = Path(__file__).resolve().parents[1]


def test_url_sanitization_and_conservative_resource_classification():
    safe = sanitize_url("https://cdn.example/video.mp4?token=secret&quality=high#t=15")
    assert "secret" not in safe and "token=REDACTED" in safe and "#" not in safe
    panorama = classify_resource("one", "https://cdn.example/panorama_transcode.mp4", "video/mp4", {"width": 2048, "height": 2048})
    ball = classify_resource("two", "https://cdn.example/follow-camera-ball_tracking.mp4", "video/mp4")
    calibration = classify_resource("three", "https://cdn.example/camera_alignment.veo", "application/json")
    assert panorama.role == "panorama" and panorama.confidence >= 0.9
    assert ball.role == "ball_tracking"
    assert calibration.role == "calibration"
    assert ranked_candidates([ball, panorama], "panorama") == [panorama]


def test_initial_panorama_transcode_requires_strong_panorama_path_evidence():
    preloaded = classify_resource(
        "preloaded",
        "https://cdn.veo.co/matches/abc/panorama/transcode-xxx.mp4",
        "video/mp4",
        discovery_phases=[INITIAL_PHASE],
    )
    outside_panorama = classify_resource(
        "outside",
        "https://cdn.veo.co/matches/abc/transcode-xxx.mp4",
        "video/mp4",
        discovery_phases=[INITIAL_PHASE],
    )
    post_interactive = classify_resource(
        "post",
        "https://cdn.veo.co/matches/abc/transcode-yyy.mp4",
        "video/mp4",
        discovery_phases=[AFTER_INTERACTIVE_PHASE],
    )
    generic = classify_resource(
        "generic",
        "https://cdn.veo.co/matches/abc/panorama/video.mp4",
        "video/mp4",
        discovery_phases=[INITIAL_PHASE],
    )
    generic.role = "panorama"
    assert preloaded.role == "panorama" and importable(preloaded)
    assert outside_panorama.role == "panorama" and not importable(outside_panorama)
    assert post_interactive.role == "panorama" and importable(post_interactive)
    assert not importable(generic)


def test_initial_panorama_transcode_can_make_inspection_ready_without_activation_evidence():
    tracking = classify_resource(
        "tracking",
        "https://cdn.veo.co/matches/abc/video.mp4",
        "video/mp4",
        discovery_phases=[INITIAL_PHASE],
    )
    panorama = classify_resource(
        "panorama",
        "https://cdn.veo.co/matches/abc/panorama/transcode-xxx.mp4",
        "video/mp4",
        discovery_phases=[INITIAL_PHASE],
    )
    calibration = classify_resource(
        "calibration",
        "https://cdn.veo.co/matches/abc/camera_alignment.veo",
        "application/json",
        discovery_phases=[INITIAL_PHASE],
    )
    calibration.field_length, calibration.field_width = 105, 68
    inspection = VeoInspection(
        "preloaded_inspection",
        "https://app.veo.co/matches/abc",
        "https://app.veo.co/matches/abc",
        candidates=[tracking, panorama, calibration],
        interaction={
            "interactive_activation_confirmed": False,
            "post_interactive_transcode_found": False,
            "post_interactive_calibration_found": False,
        },
    )
    public = inspection.public_dict()
    assert public["ready"] is True
    assert public["resolved_assets"]["panorama"] == "panorama"
    assert public["interaction"]["interactive_activation_confirmed"] is False


def test_default_and_interactive_discovery_phases_drive_classification_and_diagnostics():
    tracking = classify_resource("tracking", "https://cdn.example/video.mp4?token=private", "video/mp4", discovery_phases=[INITIAL_PHASE])
    dual_lens = classify_resource("dual", "https://cdn.example/transcode-123.mp4?token=private", "video/mp4", {"width": 2048, "height": 2048}, [AFTER_INTERACTIVE_PHASE])
    calibration = classify_resource("calibration", "https://cdn.example/camera_alignment-123.veo?token=private", "application/json", discovery_phases=[AFTER_INTERACTIVE_PHASE])
    calibration.field_length, calibration.field_width = 105, 68
    assert tracking.role == "ball_tracking" and tracking.confidence >= .9
    assert "initial/default" in " ".join(tracking.reasons)
    assert dual_lens.role == "panorama" and dual_lens.confidence >= .9
    assert "after Interactive" in " ".join(dual_lens.reasons)
    assert calibration.role == "calibration"
    diagnostics = phase_diagnostics([tracking, dual_lens, calibration], {
        "interactive_control_found": True,
        "interactive_click_succeeded": True,
        "dom_change_summary": {"changed": True},
        "interactive_attempts": ["interactive button"],
        "menu_controls_inspected": [],
    })
    assert [item["identifier"] for item in diagnostics["default_phase_resources"]] == ["tracking"]
    assert [item["identifier"] for item in diagnostics["post_interactive_resources"]] == ["dual", "calibration"]
    assert [item["identifier"] for item in diagnostics["final_classified_assets"]["tracking"]] == ["tracking"]
    assert [item["identifier"] for item in diagnostics["final_classified_assets"]["dual_lens"]] == ["dual"]
    assert [item["identifier"] for item in diagnostics["final_classified_assets"]["calibration"]] == ["calibration"]
    assert diagnostics["interactive_activation_confirmed"] is True
    assert "private" not in str(diagnostics)


def test_camera_directions_are_taken_from_the_selected_standard_tracking_video():
    tracking = "https://c.veocdn.com/match/standard/machine/45635d64/video.mp4"
    directions = "https://c.veocdn.com/match/standard/machine/45635d64/detections_constrained.det"
    videos = {"videos": [
        {"url": "https://c.veocdn.com/match/panoramic/video.mp4", "render_type": "panoramic", "camera_directions": "https://c.veocdn.com/other/detections_constrained.det"},
        {"url": tracking, "render_type": "standard", "camera_directions": directions},
        {"url": "https://c.veocdn.com/match/standard/machine/other/video.mp4", "render_type": "standard", "camera_directions": None},
    ]}
    pairs = standard_video_camera_directions(videos)
    assert pairs == [(tracking, directions)]
    resources = {tracking: {"url": tracking, "phases": [INITIAL_PHASE]}}
    assert camera_directions_for_tracking(resources, [(url, det, INITIAL_PHASE) for url, det in pairs]) == (tracking, directions)
    # URL availability follows the explicit field; no replacement or render guessing.
    assert directions in resource_links(videos)


def test_structured_standard_video_beats_incidental_video_mp4_and_keeps_its_own_det():
    highlights = "https://c.veocdn.com/match/highlights/video.mp4"
    tracking = "https://c.veocdn.com/match/standard/machine/45635d64/video.mp4"
    directions = "https://c.veocdn.com/match/standard/machine/45635d64/detections_constrained.det"
    videos = {"videos": [{"url": highlights, "render_type": "highlight", "camera_directions": None},
                           {"url": tracking, "render_type": "follow", "camera_directions": directions}]}
    structured = standard_follow_videos(videos)
    assert structured == [(tracking, directions)]
    resources = {highlights: {"url": highlights, "phases": [INITIAL_PHASE]}, tracking: {"url": tracking, "phases": [INITIAL_PHASE]}}
    assert selected_tracking_resource(resources, structured) == tracking
    assert camera_directions_for_tracking(resources, [(tracking, directions, INITIAL_PHASE)], structured) == (tracking, directions)
    highlight_candidate = classify_resource("highlights", highlights, "video/mp4", discovery_phases=[INITIAL_PHASE])
    assert highlight_candidate.role != "veo_highlights"


def test_initial_default_player_current_src_wins_over_nested_and_post_interactive_resources():
    initial = "https://c.veocdn.com/match/standard/machine/real/video.mp4?sig=initial"
    nested = "https://c.veocdn.com/match/highlights/standard/video.mp4?sig=clip"
    after = "https://c.veocdn.com/match/transcode-2048.mp4?sig=after"
    snapshot = {"videos": [
        {"currentSrc": nested, "readyState": 4, "videoWidth": 1920, "videoHeight": 1080, "visible": False, "area": 1920 * 1080},
        {"currentSrc": initial, "readyState": 4, "videoWidth": 1920, "videoHeight": 1080, "visible": True, "area": 1920 * 1080, "paused": False},
    ]}
    assert initial_player_video_url(snapshot) == initial
    resources = {initial: {"url": initial, "phases": [INITIAL_PHASE]}, nested: {"url": nested, "phases": [INITIAL_PHASE]}, after: {"url": after, "phases": [AFTER_INTERACTIVE_PHASE]}}
    metadata = [(nested, "https://c.veocdn.com/match/highlights/standard/clip.det"), (initial, "https://c.veocdn.com/match/standard/machine/real/detections_constrained.det")]
    assert selected_tracking_resource(resources, [(nested, None), (initial, None)], initial) == initial
    assert camera_directions_for_tracking(resources, [(url, det, INITIAL_PHASE) for url, det in metadata], [], initial) == (initial, metadata[1][1])


def test_generated_highlights_metadata_detects_a_generic_video_url():
    url = "https://c.veocdn.com/match/generated/video.mp4"
    assert generated_highlights_videos({"results": [{"is_ai_generated": True, "video": {"url": url}}]}) == []
    captured_individual_clips = [
        {"type": "goal", "is_ai_generated": True, "videos": [{"url": url}]},
        {"type": "shot_on_goal", "is_ai_generated": True, "videos": [{"url": url}]},
    ]
    assert generated_highlights_videos(captured_individual_clips) == []


def test_explicit_highlights_compilation_metadata_is_eligible_but_individual_clip_is_not():
    reel = "https://c.veocdn.com/match/reel/video.mp4"
    clip = "https://c.veocdn.com/match/clip/video.mp4"
    payload = {"videos": [{"type": "reel", "video": {"url": reel}}, {"url": clip, "render_type": "standard", "is_ai_generated": True}]}
    assert generated_highlights_videos(payload) == [reel]


def test_camera_directions_missing_for_selected_tracking_and_direct_network_fallback():
    tracking = "https://cdn.example/standard/video.mp4"
    resources = {tracking: {"url": tracking, "phases": [INITIAL_PHASE]}}
    assert standard_video_camera_directions({"videos": [
        {"url": tracking, "render_type": "standard", "camera_directions": None},
    ]}) == []
    assert camera_directions_for_tracking(resources, []) is None
    direct = resource_links({"asset": "https://cdn.example/detections_constrained.det?sig=private"})
    candidate = classify_resource("det", direct[0])
    assert candidate.role == "camera_directions" and importable(candidate)


class _JsonResponse:
    def __init__(self, url: str, payload: object, headers: dict[str, str]):
        self.url = url
        self.headers = headers
        self._body = json.dumps(payload).encode("utf-8")
        self.reads = 0

    def body(self):
        self.reads += 1
        return self._body


def test_bounded_json_capture_accepts_match_videos_with_and_without_content_length():
    payload = {"videos": [{"render_type": "standard", "url": "https://c.veocdn.com/m/standard/video.mp4",
                             "camera_directions": "https://c.veocdn.com/m/standard/detections_constrained.det"}]}
    endpoint = "https://api.veo.co/api/app/matches/match-1/videos/"
    with_length = _JsonResponse(endpoint, payload, {"content-type": "application/json", "content-length": str(len(json.dumps(payload)))})
    without_length = _JsonResponse(endpoint, payload, {"content-type": "application/json; charset=utf-8"})
    assert bounded_json_response(with_length) == payload
    assert bounded_json_response(without_length) == payload
    assert standard_video_camera_directions(bounded_json_response(without_length)) == [
        (payload["videos"][0]["url"], payload["videos"][0]["camera_directions"])
    ]


def test_bounded_json_capture_rejects_unknown_length_non_match_and_oversized_bodies():
    telemetry = _JsonResponse("https://api.amplitude.com/config", {"events": []}, {"content-type": "application/json"})
    assert bounded_json_response(telemetry) is None and telemetry.reads == 0
    oversized = _JsonResponse("https://api.veo.co/api/app/matches/match-1/videos/", {"blob": "x" * 2_100_000}, {"content-type": "application/json"})
    assert bounded_json_response(oversized) is None


def test_resolved_assets_include_associated_camera_directions():
    tracking = classify_resource("tracking", "https://cdn.example/video.mp4", discovery_phases=[INITIAL_PHASE])
    directions = classify_resource("directions", "https://cdn.example/detections_constrained.det", discovery_phases=[INITIAL_PHASE])
    inspection = VeoInspection("inspect", "https://app.veo.co/matches/x", "https://app.veo.co/matches/x", candidates=[tracking, directions])
    assert inspection.public_dict()["resolved_assets"]["camera_directions"] == "directions"


def test_import_selection_includes_discovered_camera_directions(tmp_path, monkeypatch):
    manager = VeoImportService(WorkspaceStore(tmp_path), tmp_path)
    calibration = classify_resource("calibration", "https://cdn.example/camera_alignment.veo", discovery_phases=[AFTER_INTERACTIVE_PHASE])
    calibration.field_length, calibration.field_width = 105, 68
    tracking = classify_resource("tracking", "https://cdn.example/video.mp4", discovery_phases=[INITIAL_PHASE])
    panorama = classify_resource("panorama", "https://cdn.example/transcode-1.mp4", discovery_phases=[AFTER_INTERACTIVE_PHASE])
    directions = classify_resource("directions", "https://cdn.example/detections_constrained.det", discovery_phases=[INITIAL_PHASE])
    inspection = VeoInspection("inspect", "https://app.veo.co/matches/x", "https://app.veo.co/matches/x", candidates=[calibration, tracking, panorama, directions])
    manager._inspections[inspection.identifier] = inspection
    monkeypatch.setattr(manager, "_run_import", lambda _job: None)
    job = manager.start_import(inspection.identifier, inspection.public_dict()["resolved_assets"])
    assert job.selected_assets["camera_directions"] == "directions"
    assert job.assets["camera_directions"]["filename"] == "detections_constrained.det"


def test_exact_interactive_player_button_is_preferred_over_text_fallback():
    state = {"activated": False}

    class Button:
        def is_visible(self, **_kwargs):
            return True
        def click(self, **_kwargs):
            raise AssertionError("locator click must not be used for the primary button")
        def bounding_box(self):
            return {"x": 10, "y": 20, "width": 40, "height": 20}
        def evaluate(self, script, *_args):
            return {"disabled": False, "ariaDisabled": None, "className": "btn interactive-player", "display": "block", "visibility": "visible", "opacity": "1", "pointerEvents": "auto", "elementAtCenter": "button", "playerClass": "player", "playerChildCount": 2}

    class Locator:
        def count(self):
            return 1
        def nth(self, _index):
            return Button()

    class Page:
        def locator(self, selector):
            selectors.append(selector)
            return Locator()
        class Mouse:
            def move(self, *_args): pass
            def down(self): pass
            def up(self): state["activated"] = True; clicks.append("mouse")
        mouse = Mouse()
        def wait_for_timeout(self, _milliseconds): pass
        def evaluate(self, script):
            if "querySelectorAll('video')" in script:
                return [{"currentSrc": "https://cdn.example/video.mp4", "src": "", "videoWidth": 1920, "videoHeight": 1080, "readyState": 4, "paused": False, "currentTime": 1.0}]
            if "querySelectorAll('canvas')" in script:
                return []
            if "performance.getEntriesByType" in script:
                return ["https://cdn.example/transcode-id.mp4"] if state["activated"] else []
            return []
        def get_by_role(self, *_args, **_kwargs):
            raise AssertionError("text/accessibility fallback must not run")
        def get_by_text(self, *_args, **_kwargs):
            raise AssertionError("text fallback must not run")

    selectors, clicks, phases = [], [], []
    result = activate_interactive(Page(), lambda: phases.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE, phases.append)
    assert selectors and all(selector == "button.interactive-player" for selector in selectors)
    assert clicks == ["mouse"] and phases == [AFTER_INTERACTIVE_PHASE]
    assert result["primary_selector_found"] is True
    assert result["primary_selector_count"] == 1
    assert result["primary_selector_visible"] is True
    assert result["primary_selector_click_succeeded"] is True
    assert result["primary_selector_center_hit"] == "button"


def test_primary_interactive_button_hidden_until_player_hover_is_retried():
    state = {"visible": False, "activated": False}

    class PrimaryButton:
        def is_visible(self, **_kwargs):
            return state["visible"]
        def click(self, **_kwargs):
            raise AssertionError("locator click must not be used")
        def bounding_box(self):
            return {"x": 5, "y": 5, "width": 30, "height": 20}
        def evaluate(self, script, *_args):
            return {"disabled": False, "ariaDisabled": None, "className": "btn interactive-player", "display": "block", "visibility": "visible", "opacity": "1", "pointerEvents": "auto", "elementAtCenter": "svg-child", "playerClass": "player", "playerChildCount": 2}

    class Video:
        def is_visible(self, **_kwargs):
            return True
        def hover(self, **_kwargs):
            state["visible"] = True

    class Locator:
        def __init__(self, element):
            self.element = element
        def count(self):
            return 1
        def nth(self, _index):
            return self.element

    class Page:
        def locator(self, selector):
            if selector == "button.interactive-player":
                return Locator(PrimaryButton())
            if selector == "video":
                return Locator(Video())
            raise AssertionError(f"Unexpected hover selector: {selector}")
        class Mouse:
            def move(self, *_args): pass
            def down(self): pass
            def up(self): state["activated"] = True; clicks.append("mouse")
        mouse = Mouse()
        def evaluate(self, script):
            if "querySelectorAll('video')" in script:
                return [{"currentSrc": "https://cdn.example/video.mp4", "src": "", "videoWidth": 1920, "videoHeight": 1080, "readyState": 4, "paused": False, "currentTime": 1.0}]
            if "querySelectorAll('canvas')" in script:
                return []
            if "performance.getEntriesByType" in script:
                return ["https://cdn.example/transcode-id.mp4"] if state["activated"] else []
            return []
        def wait_for_timeout(self, _milliseconds):
            pass
        def get_by_role(self, *_args, **_kwargs):
            raise AssertionError("fallback must not run")
        def get_by_text(self, *_args, **_kwargs):
            raise AssertionError("fallback must not run")

    clicks, phases = [], []
    result = activate_interactive(Page(), lambda: phases.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE, phases.append)
    assert clicks == ["mouse"]
    assert result["primary_selector_hover_required"] is True
    assert result["primary_selector_visible"] is True
    assert result["primary_selector_click_succeeded"] is True
    assert result["primary_selector_center_hit"] == "svg-child"


def test_primary_interactive_does_not_use_native_dom_click_after_mouse_attempt():
    state = {"activated": False}

    class Button:
        def is_visible(self, **_kwargs): return True
        def bounding_box(self): return {"x": 1, "y": 1, "width": 20, "height": 20}
        def click(self, **_kwargs): raise RuntimeError("physical fixture click produced no activation")
        def evaluate(self, script, *_args):
            assert "element.click" not in script
            return {"disabled": False, "ariaDisabled": None, "className": "btn interactive-player", "display": "block", "visibility": "visible", "opacity": "1", "pointerEvents": "auto", "elementAtCenter": "button", "playerClass": "player", "playerChildCount": 2}

    class Locator:
        def count(self): return 1
        def nth(self, _index): return Button()

    class Page:
        def locator(self, selector):
            assert selector == "button.interactive-player"
            return Locator()
        class Mouse:
            def move(self, *_args): pass
            def down(self): pass
            def up(self): pass
        mouse = Mouse()
        def wait_for_timeout(self, _milliseconds): pass
        def evaluate(self, script):
            if "querySelectorAll('video')" in script:
                return [{"currentSrc": "https://cdn.example/video.mp4", "src": "", "videoWidth": 1920, "videoHeight": 1080, "readyState": 4, "paused": False, "currentTime": 1.0}]
            if "querySelectorAll('canvas')" in script:
                return []
            if "performance.getEntriesByType" in script:
                return []
            return []

    phases = []
    result = activate_interactive(Page(), lambda: phases.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE, phases.append)
    assert result["activation_method"] == "physical_mouse"
    assert [attempt["method"] for attempt in result["click_method_attempts"]] == ["mouse"]
    assert result["primary_selector_click_succeeded"] is True


def test_membership_close_is_scoped_to_a_visible_membership_dialog():
    state = {"visible": True, "clicked": False}

    class CloseButton:
        def is_visible(self, **_kwargs):
            return True
        def click(self, **_kwargs):
            state["clicked"] = True
            state["visible"] = False

    class CloseLocator:
        def count(self):
            return 1
        def nth(self, _index):
            return CloseButton()

    class Dialog:
        def is_visible(self, **_kwargs):
            return state["visible"]
        def inner_text(self, **_kwargs):
            return "Become a member of REAL JAEN BASE! Request membership"
        def locator(self, selector):
            assert selector == 'button[aria-label="x"]'
            return CloseLocator()
        def wait_for(self, *, state: str, **_kwargs):
            assert state == "hidden" and not membership_state["visible"]

    class DialogLocator:
        def count(self):
            return 1
        def nth(self, _index):
            return Dialog()

    class Page:
        def locator(self, selector):
            assert selector == "[role='dialog'], dialog, [aria-modal='true']"
            return DialogLocator()

    membership_state = state
    result = dismiss_membership_modal(Page())
    assert result == {
        "membership_modal_found": True,
        "membership_modal_closed": True,
        "membership_close_button_found": True,
    }


def test_interactive_confirmation_requires_both_post_interactive_assets():
    tracking = classify_resource("tracking", "https://cdn.example/video.mp4", discovery_phases=[INITIAL_PHASE])
    transcode = classify_resource("dual", "https://cdn.example/transcode-123.mp4", discovery_phases=[AFTER_INTERACTIVE_PHASE])
    diagnostics = phase_diagnostics([tracking, transcode], {
        "interactive_control_found": True,
        "interactive_click_succeeded": True,
        "interactive_attempts": [],
        "menu_controls_inspected": [],
    })
    assert diagnostics["physical_click_succeeded"] is False
    assert diagnostics["post_interactive_transcode_found"] is True
    assert diagnostics["post_interactive_calibration_found"] is False
    assert diagnostics["interactive_activation_confirmed"] is False


def test_interactive_activation_prefers_accessibility_queries_and_supports_spanish_label():
    class Control:
        def is_visible(self, **_kwargs):
            return True
        def click(self, **_kwargs):
            clicks.append("clicked")

    class EmptyLocator:
        def count(self):
            return 0
        def nth(self, _index):
            raise AssertionError("No element should be selected")

    class OneLocator:
        def count(self):
            return 1
        def nth(self, _index):
            return Control()

    class Page:
        def get_by_role(self, role, **kwargs):
            calls.append((role, kwargs["name"]))
            return OneLocator() if role == "button" else EmptyLocator()
        def get_by_text(self, *_args, **_kwargs):
            return EmptyLocator()

    calls, clicks, phases = [], [], []
    result = activate_interactive(Page(), lambda: phases.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE, phases.append)
    assert result["interactive_control_found"] is True
    assert result["interactive_click_succeeded"] is True
    assert clicks == ["clicked"] and phases == [AFTER_INTERACTIVE_PHASE]
    assert calls[0][0] == "button" and calls[0][1].search("Interactivo")


def test_interactive_diagnostic_distinguishes_found_control_from_failed_click():
    class Control:
        def is_visible(self, **_kwargs):
            return True
        def click(self, **_kwargs):
            raise RuntimeError("disabled in fixture")

    class Locator:
        def count(self):
            return 1
        def nth(self, _index):
            return Control()

    class Page:
        def get_by_role(self, *_args, **_kwargs):
            return Locator()
        def get_by_text(self, *_args, **_kwargs):
            return Locator()
        def locator(self, *_args, **_kwargs):
            return Locator()
        def wait_for_timeout(self, _milliseconds):
            pass

    phase = [INITIAL_PHASE]
    result = activate_interactive(
        Page(),
        lambda: phase.append(AFTER_INTERACTIVE_PHASE) or INITIAL_PHASE,
        lambda prior: phase.__setitem__(0, prior),
    )
    assert result["interactive_control_found"] is True
    assert result["interactive_click_succeeded"] is False
    assert phase[0] == INITIAL_PHASE


def test_public_inspection_manifest_never_exposes_full_signed_url(tmp_path):
    candidate = classify_resource("asset_1", "https://cdn.example/panorama.mp4?X-Amz-Signature=secret", "video/mp4")
    inspection = VeoInspection("inspect_test", "https://app.veo.co/match/1", "https://app.veo.co/match/1", candidates=[candidate])
    manifest = inspection.public_dict()
    assert "url" not in manifest["candidates"][0]
    assert "secret" not in str(manifest)


def test_import_job_populates_normal_workspace_from_local_fixture(tmp_path, monkeypatch):
    service = VeoImportService(WorkspaceStore(tmp_path), tmp_path)
    calibration = ROOT / "tests/fixtures/calibrations/camera_alignment-67a3f0b3-002e-491c-a48c-298f6f0160d1.veo"
    candidate = VeoCandidate("cal", calibration.as_uri(), calibration.as_uri(), calibration.name, "application/json", "calibration", 1.0, ["fixture"])
    inspection = VeoInspection("inspect_fixture", "file://fixture", "file://fixture", metadata={"title": "Fixture match"}, candidates=[candidate])
    service._inspections[inspection.identifier] = inspection
    def fake_download(_url, destination, progress, cancelled, cookies):
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(calibration, destination)
        progress(destination.stat().st_size, destination.stat().st_size)
    monkeypatch.setattr("web_app.backend.app.importers.veo.service.download_stream", fake_download)
    job = ImportJob("job_fixture", inspection.identifier, {"calibration": "cal"})
    service._run_import(job)
    match = service.store.match(job.match_id or "")
    assert job.status == "completed"
    assert match is not None and match["name"] == "Fixture match"
    assert match["calibration"]["field_length"] > 0


def test_inspection_api_handler_uses_mocked_browser_result(monkeypatch):
    candidate = classify_resource("asset_mock", "https://cdn.example/camera_alignment.veo?token=secret", "application/json")
    fake = VeoInspection("inspect_mock", "https://app.veo.co/matches/x", "https://app.veo.co/matches/x", candidates=[candidate])
    monkeypatch.setattr(main.veo_imports, "inspect", lambda _url: fake)
    response = main.inspect_veo(VeoInspectRequest(match_url="https://app.veo.co/matches/x"))
    assert response["id"] == "inspect_mock"
    assert "secret" not in str(response)
