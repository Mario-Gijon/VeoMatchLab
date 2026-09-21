# Player detection experiment

This is the first end-to-end experimental person-on-field pipeline:

```text
Dual Lens video → RF-DETR persons → bbox footpoints → validated RAW geometry → field X/Z → diagnostic pitch
```

It is not part of `web_app` or `analytics_service`, and it does not modify
geometry, match records, or source media.

## Environment and models

Activate the separate detection environment before running it:

```bash
source .venv-detection/bin/activate
```

It requires `rfdetr`, `rfdetr_plus`, OpenCV, NumPy, and the project geometry
dependencies. `large` loads `rfdetr.RFDETRLarge`; `2xl` loads
`rfdetr_plus.RFDETR2XLarge` and is subject to its accepted Platform Model
License configuration. Models run sequentially, one lens image at a time.

## Run

`VEO_MATCH_DATA_ROOT` is authoritative. If it is unset, the script uses the
ignored repository-local `data` link as a development convenience.

```bash
python experiments/player_detection/run_interval.py \
  --match-id <match-id> \
  --start 1110 \
  --duration 120 \
  --sample-fps 1 \
  --threshold 0.25 \
  --models large,2xl
```

Each run is written outside Git under
`$VEO_MATCH_DATA_ROOT/experiments/player_detection/<run-id>/`, with a separate
directory per model. It contains raw and inside-field JSONL detections,
summaries, selected annotated lens/pitch frames, and a diagnostic `pitch.mp4`.

Bounding boxes remain in native 2048×1024 lens coordinates. Their bottom-center
is translated back to the stacked RAW source and passed directly to the existing
validated `raw_pixel_to_ground` implementation. Failed and outside-field
projections remain in `detections.jsonl`; only strict inside-field points enter
`field_people.jsonl`.

Cross-lens duplicates are intentionally **not fused**: a player visible in both
lenses can appear twice. This lets the experiment validate detector and geometry
behavior before any fusion heuristic. Team classification and tracking are not
part of this stage.
