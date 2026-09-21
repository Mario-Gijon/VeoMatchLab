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

`analysis/` is deliberately only a placeholder for future development modules.
