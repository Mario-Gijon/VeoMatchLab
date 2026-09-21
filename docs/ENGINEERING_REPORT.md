# Engineering notes

## Maintained boundaries

The local FastAPI + React/Vite application is in `web_app/`. It stores its
small JSON metadata and imported assets in `VEO_MATCH_DATA_ROOT`, outside the
repository. The filesystem store uses atomic JSON replacement and validates
managed match paths before deletion; external input files are only unlinked
from metadata.

`geometry/` contains the validated calibrated camera and field math. The
backend transports shader uniforms and uses the same geometry for panorama,
RAW-lens, Interactive, and Tracking field-point operations. The browser keeps
WebGL rendering and interactions in `web_app/frontend`.

## Match layout

```text
$VEO_MATCH_DATA_ROOT/
  matches/<match-id>/
    match.json
    media/
    calibration/
    frames/
    cache/
    diagnostics/
    analysis/
    datasets/
  veo-inspections/
```

Veo import is local and uses the selected resource set: Tracking, Dual Lens,
calibration and camera directions. It preserves ownership metadata so managed
copies can be cleaned up without deleting externally referenced files.

## Deliberately absent

There is no database, queue, worker system, deployment stack, GPU framework,
or analysis model in this migration. `analytics_service/` documents the future
boundary only.
