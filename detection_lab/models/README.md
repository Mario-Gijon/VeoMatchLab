# Detection Lab model cache

The model cache is outside Git. `VEO_MATCH_MODELS_ROOT` selects its location;
when unset, Detection Lab uses the gitignored repository-local `.model-cache/`:

```text
.model-cache/
  rfdetr/
  yolo/
  dfine/
```

Use the existing isolated ML environment, not the web application's `.venv`:

```bash
.venv-detection/bin/pip install -r detection_lab/models/requirements.txt
VEO_MATCH_MODELS_ROOT="$PWD/.model-cache" \
  .venv-detection/bin/python -m detection_lab.models.prepare --all
```

RF-DETR XLarge and 2XLarge are intentionally skipped until the operator has
accepted the RF-DETR Plus Platform Model License and has a valid Roboflow
Platform Agreement. Then run:

```bash
VEO_MATCH_MODELS_ROOT="$PWD/.model-cache" \
  .venv-detection/bin/python -m detection_lab.models.prepare \
  --model rfdetr_xlarge --model rfdetr_2xlarge \
  --accept-rfdetr-plus-license
```

The D-FINE entry downloads only the upstream COCO checkpoint and its model
config. Its runtime source is deliberately not vendored or cloned by this tool;
when inference is implemented later, use a separately managed checkout of the
official D-FINE repository and point that runtime at this cache.
