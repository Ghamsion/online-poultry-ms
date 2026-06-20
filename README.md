# Poultry YOLOv8 Training — Live / Dead / Egg Detection

This repository contains the training run for the AI camera-detection
subsystem described in **Chapter 5 (§5.4)** of *Online Poultry Management
System for Ghanaian Farmers* (Group 20). It fine-tunes a YOLOv8n object
detector to recognise three classes from poultry-house imagery: **live**
birds, **dead** birds, and **eggs**.

## Contents

```
.
├── poultry_yolov8_training.ipynb   # Colab notebook used to run the training
├── weights/
│   ├── best.pt                     # best checkpoint by validation mAP@0.5
│   └── last.pt                     # final-epoch checkpoint
├── results.csv                     # per-epoch loss/precision/recall/mAP log
├── args.yaml                       # exact hyperparameters used for this run
├── results.png                     # training curves (→ report Figure 5.1)
├── confusion_matrix.png            # (→ report Figure 5.2)
├── confusion_matrix_normalized.png
├── BoxP_curve.png  BoxR_curve.png  BoxF1_curve.png  BoxPR_curve.png
└── predictions/                    # sample annotated test images (→ Figure 5.3)
```

## What this model does

The detector is the AI component behind the system's live camera page. A
browser captures a frame from the user's webcam, sends it to the FastAPI
backend, and the backend runs this model to count live birds, dead birds,
and eggs in the frame. Full architectural context is in Chapter 4 (§4.2) of
the report; the implementation is in Chapter 5 (§5.3.2).

## Reproducing this run

1. Open `poultry_yolov8_training.ipynb` in Google Colab (free T4 GPU tier
   is sufficient).
2. Assemble a dataset in standard YOLO format — see **Dataset sources**
   below, since no single public dataset covers all three classes.
3. Run the notebook cells top to bottom. Training a YOLOv8n model for
   100 epochs on a small dataset takes roughly 1–5 hours on a free Colab T4.
4. The notebook saves everything in this list above into a single
   `poultry_yolov8n_run.zip`, ready to push to a repository like this one.

## Dataset sources

No public dataset already combines `live`, `dead`, and `egg` classes, so
this dataset was assembled from a combination of public sources and
farm-collected images:

| Class | Source |
|---|---|
| Live birds | Roboflow Universe chicken/broiler detection datasets, e.g. [chicken-detection-and-tracking](https://universe.roboflow.com/chickens/chicken-detection-and-tracking), [broiler-chicken-detection](https://universe.roboflow.com/innodatatics-cjikw/broiler-chicken-detection) |
| Dead birds | Farm-collected images (no usable public dataset of adequate size exists for this class at the time of writing) |
| Eggs | [egg-detection](https://universe.roboflow.com/sksu/egg-detection-ud1ys), [chicken-embryo-egg-detection](https://universe.roboflow.com/sliit-xsolh/chicken-embryo-egg-detection) |

Class imbalance on `dead` (the rarest class by a wide margin) was addressed
through stratified oversampling of training images containing that class,
rather than per-class loss weighting, since the Ultralytics YOLOv8 trainer
does not expose a simple per-class loss-weight parameter.

## Reading the results

- **`results.png`** — six-panel summary: box-regression loss, classification
  loss, precision/recall, mAP, F1-confidence curve, and precision-recall
  curve, each tracked across training epochs.
- **`confusion_matrix.png` / `confusion_matrix_normalized.png`** — how often
  each class was predicted correctly versus confused with another class or
  with background, computed on the held-out test split.
- **`predictions/`** — a handful of test images with the model's predicted
  bounding boxes and confidence scores drawn on top.
- **`results.csv`** — the full per-epoch log, in case you want to recompute
  or replot anything beyond what Ultralytics generates automatically.

## Honest note on the numbers

These results come from a real training run on a modest dataset (see
Chapter 3, Table 3.2, for the actual split sizes). They are not expected to
match — and should not be inflated to match — the larger, more heavily
resourced datasets reported in comparable published work (e.g. Subedi et
al., 2025; Ufitikirezi et al., 2024). Chapter 5 reports these numbers as
they came out of this run, including the lower scores on the `dead` class,
which is consistent with the difficulty that class presents across the
published literature as well.

## Environment

- Python 3.11
- `ultralytics` 8.3.x (YOLOv8)
- Trained on a Google Colab T4 GPU (free tier)
- Base weights: `yolov8n.pt` (COCO-pretrained, fine-tuned via transfer
  learning — only the final classification layer was replaced to match the
  three project classes; the backbone and neck were not architecturally
  modified)

## Citation

If you build on this dataset or training run, please cite the project
report:

> Group 20 (2026). *Online Poultry Management System for Ghanaian Farmers*
> [Final-year project report]. [Your institution name].

## License

Code in this repository is provided for academic purposes. Public datasets
referenced above retain their original licences — check each Roboflow
project page before redistributing its images.
