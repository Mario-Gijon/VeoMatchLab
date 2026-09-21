from web_app.backend.app.importers.veo.discovery import StructuredDiscovery, classify_endpoint, referenced_urls
from web_app.backend.app.importers.veo.inspector import INITIAL_PHASE, camera_directions_for_tracking, standard_video_camera_directions


def test_referenced_urls_and_camera_directions_are_discovered_from_json():
    tracking = "https://c.veocdn.com/m/standard/machine/a/video.mp4?sig=private"
    directions = "https://c.veocdn.com/m/standard/machine/a/detections_constrained.det?sig=private"
    payload = {"videos": [
        {"url": tracking, "render_type": "standard", "camera_directions": directions},
        {"url": "https://c.veocdn.com/m/other/video.mp4", "render_type": "standard", "camera_directions": None},
    ]}
    pairs = standard_video_camera_directions(payload)
    assert pairs == [(tracking, directions)]
    resources = {tracking.split("?")[0]: {"url": tracking, "phases": [INITIAL_PHASE]}}
    assert camera_directions_for_tracking(resources, [(tracking, directions, INITIAL_PHASE)]) == (tracking, directions)
    refs = referenced_urls(payload)
    assert any(item["field"].endswith("camera_directions") and "private" not in item["url"] for item in refs)


def test_match_dataset_evidence_stays_separate_from_platform_capabilities_and_unknowns():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://api.veo.co/matches/1/events?token=private", {
        "match": {"title": "Fixture"}, "events": [{"type": "goal"}], "analytics_available": True,
    }, "after_interactive")
    manifest = discovery.manifest({"title": "Fixture"}, {}, {"source": "test"})
    assert manifest["datasets"]["events"]["status"] == "observed"
    assert manifest["datasets"]["player_tracking"]["status"] == "not_observed"
    assert manifest["capabilities"]["analytics_available"]["status"] == "observed"
    assert "private" not in str(manifest)


def test_highlights_endpoint_is_highlights_and_preserves_records_not_events():
    discovery = StructuredDiscovery()
    payload = {"results": [{"id": "h1", "is_ai_generated": True, "annotations": [], "camera_directions": None,
                            "involved_players": [], "duration": 8, "tags": ["goal"]}]}
    url = "https://api.veo.co/api/app/matches/abc/highlights/"
    assert classify_endpoint(url) == "highlights"
    discovery.observe_json(url, payload, "after_interactive", status_code=200)
    manifest = discovery.manifest({}, {}, {})
    assert manifest["datasets"]["highlights"]["status"] == "observed"
    assert manifest["datasets"]["highlights"]["evidence"][0]["payload"]["results"][0]["id"] == "h1"
    assert manifest["datasets"]["events"]["status"] == "not_observed"


def test_known_dataset_statuses_and_unrequested_remain_distinct():
    discovery = StructuredDiscovery()
    root = "https://api.veo.co/api/app/matches/abc"
    discovery.observe_json(root + "/highlights/", [], "after_interactive", status_code=200)
    discovery.observe_json(root + "/events/", None, "after_interactive", status_code=403)
    discovery.observe_json(root + "/lineups/", None, "after_interactive", status_code=404)
    datasets = discovery.manifest({}, {}, {})["datasets"]
    assert datasets["highlights"]["status"] == "empty"
    assert datasets["events"]["status"] == "forbidden"
    assert datasets["lineup"]["status"] == "unavailable"
    assert datasets["roster"]["status"] == "not_observed"


def test_sdk_events_keys_are_filtered_but_match_event_endpoint_counts():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://api.amplitude.com/config", {"events": ["page_view"]}, "initial")
    discovery.observe_json("https://sentry.io/api/123", {"events": []}, "initial")
    assert discovery.manifest({}, {}, {})["datasets"]["events"]["status"] == "not_observed"
    discovery.observe_json("https://api.veo.co/api/app/matches/abc/events/", {"data": [{"type": "goal", "minute": 42}]}, "after_interactive")
    assert discovery.manifest({}, {}, {})["datasets"]["events"]["status"] == "observed"


def test_interactive_requires_confirmed_player_interaction_not_bundle_mentions():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://app.veo.co/matches/abc/", {"module": "interactive-player", "feature": "radar"}, "initial", "page_bootstrap")
    assert discovery.manifest({}, {}, {})["capabilities"]["interactive_available"]["status"] == "not_observed"
    discovery.record_interactive(True, {"button_selector": "button.interactive-player", "physical_click_succeeded": True})
    manifest = discovery.manifest({}, {}, {})
    assert manifest["capabilities"]["interactive_available"]["status"] == "observed"
    assert manifest["capabilities"]["interactive_available"]["evidence"][0]["source"] == "player_interaction"


def test_heatmap_permission_is_capability_evidence_not_dataset():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://api.veo.co/api/app/matches/abc/", {"analysis.read_match_heatmap": True}, "initial")
    manifest = discovery.manifest({}, {}, {})
    assert manifest["capabilities"]["heatmaps_available"]["status"] == "observed"
    assert manifest["datasets"]["heatmaps"]["status"] == "not_observed"


def test_unknown_match_json_is_bounded_and_sanitized():
    discovery = StructuredDiscovery()
    discovery.observe_json("https://api.veo.co/matches/1?signature=private", {"match": {"id": "m1"}, "custom_blob": "x" * 60_000}, "initial")
    manifest = discovery.manifest({}, {}, {})
    assert len(manifest["unknown_structured_data"]) == 1
    assert "private" not in str(manifest)
