# Detection Lab

Detection Lab is a development-only, read-only FastAPI utility. It uses the
existing `WorkspaceStore` and the same `VEO_MATCH_DATA_ROOT` selection as the
main VeoMatchLab backend. It neither imports media nor writes a new manifest or
database.

Start it from the repository root with the data root used by VeoMatchLab:

```bash
VEO_MATCH_DATA_ROOT="$PWD/data" \
  .venv/bin/uvicorn detection_lab.app:app --host 127.0.0.1 --port 8100
```

Open <http://127.0.0.1:8100>. The preview video is streamed with range support
from the imported match's existing `dual_lens` media record; it is never copied.
Each match has `Vista previa` and `Person` tabs. A working clip is stored only
in that browser's `localStorage`, keyed by match ID, as start and duration in
seconds; no clip file or server-side record is created.

The `Person` tab currently runs the owned `RFDETRLargePersonDetector` through
the existing `.venv-detection` interpreter, so the small web server does not
need ML dependencies. It samples the original `dual_lens.mp4` directly (top =
left lens, bottom = right lens) and draws canvas overlays over the logical
clip. Choose `RF-DETR Large`, use the default 5 FPS and 0.10 inference
threshold, then select **Analizar clip**. For a first check, choose a 5–10
second clip in `Vista previa`, save it, and open `Person`.

Results are JSON files in the ignored `.detection-lab-cache/person/` directory.
The cache key covers the match, logical clip, model, sample rate, inference
threshold and detector version. The low inference threshold determines what is
cached; the visible-confidence slider only filters/redraws those cached results
and never reruns RF-DETR. Set `VEO_MATCH_DETECTION_PYTHON` or
`VEO_MATCH_DETECTION_CACHE_ROOT` only when the default local paths need
overriding.

`analysis/` is deliberately only a placeholder for future development modules.
