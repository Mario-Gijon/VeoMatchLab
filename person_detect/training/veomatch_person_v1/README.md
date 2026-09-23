# VeoMatch person detector V1 fine-tuning

This workspace prepares a controlled one-class (`person`) RF-DETR 2XL
comparison.  Every experiment starts independently from the unchanged
baseline checkpoint:

`person_detect/model_weights/rfdetr/rf-detr-xxlarge.pth`

Experiments:

- C1: `football_images`
- C2: `football_images` + `veo_ball_detection`
- C3: `football_images` + `veo_ball_detection` + `football_eoakc`

Place imported source datasets under `sources/`.  No dataset downloader is
included; Roboflow imports require the user's own credentials and should be
downloaded into those directories manually.  Inspect and normalize a COCO
source with:

```bash
python -m person_detect.training.veomatch_person_v1.tools.dataset_tools inspect sources/football_images
python -m person_detect.training.veomatch_person_v1.tools.dataset_tools convert-coco \
  sources/football_images datasets/football_images --source-name football_images

After all three sources have been imported, build the independent cumulative
datasets with:

```bash
python -m person_detect.training.veomatch_person_v1.tools.build_datasets \
  --football-images sources/football_images \
  --veo-ball-detection sources/veo_ball_detection \
  --football-eoakc sources/football_eoakc
```
```

The current adapter accepts COCO detection JSON. Generated datasets use one COCO category (`person`) and retain a per-image
provenance manifest. Exact duplicates are removed during merge; perceptual
near-duplicates are reported in `near_duplicates.json` for review. Unknown
classes are reported and never silently mapped.

Before a long job, validate an independent experiment plan:

```bash
python -m person_detect.training.veomatch_person_v1.train c1 --dry-run
```

Training runs are written under `runs/<experiment>/`. After a successful run,
the selected checkpoint is exported independently to:

`person_detect/model_weights/rfdetr_finetuned/veomatch_person_v1/<experiment>/best.pth`

with adjacent `metadata.json`. This makes the intended comparison explicit:
baseline RF-DETR 2XL vs C1 vs C2 vs C3, without chaining C2 from C1 or C3
from C2.
