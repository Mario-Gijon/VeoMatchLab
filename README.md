# VeoMatchLab

VeoMatchLab is a small, local-first foundation for importing Veo matches,
playing their existing views, and validating calibrated field geometry. It does
not yet perform player detection, tracking, classification, or other GPU-backed
analysis.

## Architecture

```text
geometry/          shared calibrated camera, lens, panorama, Tracking and field math
web_app/           current FastAPI backend and React/Vite browser application
analytics_service/ reserved boundary for future GPU-backed match analysis
```

`geometry` is independent of both services. `web_app` owns match import,
resource discovery and downloads, local match storage, media serving, playback,
the browser's WebGL panorama renderer, Interactive and Tracking views, pitch
overlays, and validation tools. `analytics_service` is intentionally only a
documented boundary today; when it exists it will use match identifiers and the
same local storage rather than receive complete video files over HTTP.

## Runtime model

One GPU-capable tower initially hosts the application, its external match-data
directory, and the future analytics service. A laptop on the same LAN uses the
application in a browser; remote development over SSH is supported naturally.

Real match assets are not repository content. Set `VEO_MATCH_DATA_ROOT` to a
directory outside this checkout (for example `/srv/RealJaenData`). The backend
uses `~/.local/share/VeoMatchLab` when it is not set.

```text
RealJaenData/
  matches/         imported media, calibration files, .det resources and metadata
  analyses/        future derived match outputs
  models/          future model weights
  veo-inspections/ importer diagnostics
```

`VEO_REFERENCE_CALIBRATIONS_ROOT` is optional and points to an operator-managed
directory of local reference `.veo` files; it is never assumed to be inside the
repository.

The Veo Chromium inspector runs headlessly by default, including when launched
from the web application. Set `VEO_BROWSER_HEADLESS=false` to show the browser
window while debugging an inspection.

## Run the web application

Create a Python environment and install the backend requirements:

```bash
.venv/bin/pip install -r web_app/backend/requirements-dev.txt
VEO_MATCH_DATA_ROOT=/srv/RealJaenData \
  .venv/bin/uvicorn web_app.backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd web_app/frontend
npm ci
npm run dev
```

Vite listens on `0.0.0.0:5173` and proxies `/api` to the localhost-only
FastAPI process. Find the tower's LAN address with:

```bash
hostname -I
```

On the laptop, open `http://<tower-ip>:5173`. The browser uses that single
origin for API calls and media playback; it does not need to know the backend
address. For a production frontend check, run `npm run build` in
`web_app/frontend`.

## Tests and reference material

The Python tests use small explicit calibration and image fixtures under
`tests/fixtures/`; these are only deterministic test inputs, not a match-data
store. The captured Veo panoramic shaders and uniforms live in `reference/`.

```bash
.venv/bin/python -m pytest tests
cd web_app/frontend && npm test && npm run build
```

The geometry foundation is documented in
[docs/VEO_GEOMETRY_FOUNDATION.md](docs/VEO_GEOMETRY_FOUNDATION.md). Historical
research experiments belong to the original research snapshot, not the
maintained VeoMatchLab architecture.
