# 3D Detection Model Comparison Visualization (Text-Prompted)

Side-by-side comparison of **text-prompted 3D detectors** on the InTheWild v3 benchmark: **WildDet3D (Ours)** and **3D-MOOD**.

Each image shows a prompts banner (listing the GT category text prompts) followed by a 2-panel grid where each cell = **image overlay** (left) + **perspective BEV** (right).

## Files

| File | Description |
|---|---|
| `scripts/prepare_data.py` | Build `data/text/index.json` + per-image JSONs in `data/text/images/` |
| `scripts/patch_matched_counts.py` | Add score-filtered prediction counts to `index.json` |
| `index.html` / `image.html` | Interactive web viewer |
| `js/app.js` | Shared config (auto-detects local vs HuggingFace) and data loaders |
| `js/gallery.js` | Gallery page logic (index.html) |
| `js/detail.js` | Detail page logic (image.html), score filtering + cross-category NMS |
| `js/bev-renderer.js` | JS perspective BEV renderer |
| `js/overlay-renderer.js` | JS image overlay renderer |
| `js/scene-tree.js` | Scene category tree sidebar |

## Quick Start

```bash
conda activate opendet3d

# 1. (Re)build the data served to the viewer
export PYTHONPATH=/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet:$PYTHONPATH
python visualization_scripts/demo_comparison_v3/text/scripts/prepare_data.py
python visualization_scripts/demo_comparison_v3/text/scripts/patch_matched_counts.py

# 2. Serve locally
cd visualization_scripts/demo_comparison_v3
python3 -m http.server 8765
# Open http://localhost:8765/text/
```

## Data Layout

Data is output to `data/text/` locally, mirroring the HuggingFace repo structure:
```
model/
├── images/         # Shared images (coco_train, coco_val, obj365_val)
├── box/            # Box-prompted branch data (separate branch)
└── text/           # Text-prompted branch data (this branch)
    ├── index.json
    └── images/     # Per-image JSONs (0.json, 1.json, ...)
```

## Local vs HuggingFace Hosting

`app.js` auto-detects which data source to use:
- **Running on `localhost`, `127.0.0.1`, `file://`, or `*.trycloudflare.com`** -> reads from local `data/text/` and `images/`
- **Otherwise (production)** -> reads from the HuggingFace-hosted dataset at `model/text/`

## Prediction Filtering

Text-based models emit free-form text predictions, so there is no GT
matching / relabeling applied in the viewer. The only filter is a
**per-model confidence threshold**, configurable at runtime with a
slider on the detail page (range 0.05-0.95, step 0.05).

| Model | Default threshold | Slider id |
|---|---|---|
| WildDet3D (Ours) | 0.50 | `thr-WildDet3D` |
| 3D-MOOD | 0.10 | `thr-3D-MOOD` |

A **cross-category NMS** (IoU > 0.8) is also applied to suppress
near-duplicate boxes from similar text categories (e.g. "microwave"
vs "oven" on the same object). Same-category NMS is handled inside
the model itself.

Every prediction is displayed with its **original predicted category**
(from the model's training vocabulary). The gallery pills show counts
computed at the default thresholds with NMS, via `scripts/patch_matched_counts.py`.

## Text Prompting Strategy

The two models use different text prompting strategies during inference:

| Model | Strategy | Categories per image |
|---|---|---|
| WildDet3D | **Open-vocabulary** | All ~795 categories from the training vocab |
| 3D-MOOD | **Oracle text** (GT categories) | Only the GT categories present in each image |

WildDet3D receives the full category vocabulary on every image and must
identify which objects are present without any prior knowledge. 3D-MOOD
uses `per_image_categories=True` in the dataset config, which restricts
its text prompts to only the GT categories in each image (to avoid BERT
tokenizer truncation with 800+ categories). This gives 3D-MOOD an
advantage since it already knows which categories to look for.

The **prompts displayed** in the viewer are the unique GT annotation
categories for each image (from all 2D annotations, including those
without valid 3D boxes). These represent the categories that 3D-MOOD
receives as input prompts, and a subset of the full vocabulary that
WildDet3D searches over.

## Text Category Label Mapping

Predictions saved in the `detect_3D_results.json` files store `category_id` values from the **model's training vocabulary**, which is the InTheWild v3 category space that existed when the model was trained.

**Important:** the `InTheWild_v3_val.json` annotation file on disk has been **regenerated since the models were evaluated**, so its current category IDs no longer match the saved predictions. To fix this, `prepare_data.py` extracts the authoritative `cat_map` directly from the eval run's config YAML (parsing only within the `cat_map._fields` block) and uses that mapping for all predictions.

**Config path used:** `vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_v3/2026-03-23_00-57-13/config_2026-03-23_00-57-13.yaml`

This mapping covers 795 categories and includes all unique IDs present in the predictions. It is the correct mapping for both WildDet3D and 3D-MOOD (they share the same training vocab).

GT boxes continue to use the current `InTheWild_v3_val.json` category names.

## Prediction Sources

| Model | Prediction Path |
|---|---|
| WildDet3D | `vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_v3/2026-03-23_00-57-13/eval/detection_bbox/3D/detect_3D_results.json` |
| 3D-MOOD | `vis4d-workspace/gdino3d_swin-t_in_the_wild_v3/2026-03-23_00-51-40/eval/detection_bbox/3D/detect_3D_results.json` |

## Color Scheme

| Element | Hex |
|---|---|
| WildDet3D (Ours) | `#e74c3c` |
| 3D-MOOD | `#3b82f6` |
| GT | `#a855f7` |
