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

## Player-on-pitch debug tracker

For a short visual check of the target match, run the separate diagnostic tool:

```bash
python experiments/player_detection/run_pitch_debug.py \
  --match match_fc5f185c7710 \
  --start 750 \
  --duration 5 \
  --threshold 0.25
```

It writes `pitch_debug.mp4`, composite still frames, raw detections, and merged
pitch observations under `$VEO_MATCH_DATA_ROOT/experiments/player_detection/`.
Only detections that existing RAW geometry places strictly inside the field are
eligible for tracking. Nearby left/right field detections may be merged
one-to-one (default 1.25 m); the two source boxes remain visible and receive the
same global ID. The pitch tracker uses only distance and a simple motion
prediction, so the IDs are diagnostic labels rather than identity or team
recognition. `--cross-camera-distance`, `--track-distance`, and `--max-missed`
make its deliberately simple association policy adjustable.
