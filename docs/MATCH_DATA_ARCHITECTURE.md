# Match-data architecture

## Storage boundary

Imported Veo assets and generated outputs live below `VEO_MATCH_DATA_ROOT`, not
in Git. This includes videos, `.det` camera directions, `.veo` calibration
files downloaded for a match, discovery captures, result artifacts, rendered
media, model weights, and future analyses.

`web_app` and the future `analytics_service` run on the same tower initially.
They share this local directory and communicate future work by match identifier;
whole media files are never transferred between them over HTTP.

## Current application ownership

The web application owns importing a Veo match URL, inspected resource
selection, downloads, match records, media serving, and browser playback. The
filesystem-backed store records whether each item is managed or external. A
managed file is only removed after its path has been validated under its match
directory; external files are never deleted by the application.

## Future analysis boundary

`analytics_service` has no API or implementation yet. Future results should be
written under the matching external data root and referenced by match ID. The
service may use `geometry` for projection, but it must not duplicate calibrated
camera behavior or import presentation code from `web_app/frontend`.
