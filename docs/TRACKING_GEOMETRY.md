# Tracking view geometry

The maintained Tracking/Follow reconstruction is
`geometry/tracking_ground_v3.py`. It preserves the observed Veo convention:

- frame rate is `30000 / 1001`;
- the camera row is `round(video_time * fps) - 1`, clamped at zero;
- calibration version 6.2 uses `alignment.camera2logic` and its inverse;
- the dynamic transform combines `camera2world`, `logic2camera`, yaw and pitch;
- field coordinates come from a forward Y=0 ray intersection.

The required dynamic yaw, pitch and FOV rows are imported per match as the
selected `detections_constrained.det` resource. They are runtime match data,
not a repository fixture. `TrackingTuning` defaults to zero, preserving the
reconstructed Veo behavior; it is a validation layer rather than a fitted
replacement camera model.

The detailed camera/lens/panorama foundation remains in
[VEO_GEOMETRY_FOUNDATION.md](VEO_GEOMETRY_FOUNDATION.md).
