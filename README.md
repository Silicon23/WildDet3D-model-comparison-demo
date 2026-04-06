# 3D Detection Model Comparison - InTheWild Box Prompts

Side-by-side comparison of 3 models on InTheWild benchmark using **2D GT box prompts** to obtain 3D box predictions: **WildDet3D (Ours)**, **DetAny3D**, **OVMono3D**.

Each image shows a GT row (full width) + 1x3 model grid, where each cell = **image overlay** (left) + **perspective BEV** (right).

## Files

| File | Description |
|---|---|
| `prepare_demo_data.py` | Merge GT + 3 models' predictions into `demo_data.json` |
| `render_comparison.py` | Render static comparison images (image overlay + 3D BEV) |
| `demo_data.json` | Pre-merged data (~13 MB) |
| `comparison_renders/` | Output PNG images |
| `index.html` / `image.html` | Interactive web viewer |
| `js/bev-renderer.js` | JS perspective BEV renderer |
| `js/overlay-renderer.js` | JS image overlay renderer |

## Quick Start

```bash
conda activate opendet3d

# Step 1: Prepare data (re-run if predictions change)
export PYTHONPATH=/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet:$PYTHONPATH
python visualization_scripts/demo_comparison_v3/prepare_demo_data.py

# Step 2: Render comparison images
python visualization_scripts/demo_comparison_v3/render_comparison.py
```

Output goes to `comparison_renders/`.

## Prediction Filtering

- **WildDet3D**: Spurious predictions filtered by 2D IoU >= 0.5 with GT boxes (oracle mode should produce one output per prompt, but occasionally generates extras).
- **DetAny3D, OVMono3D**: No additional filtering (oracle/box-prompt mode, one output per GT prompt).
- **Category labels**: All box-prompted models use the annotation file's category mapping (`cat_id_to_name`), since the model's predicted category_id corresponds to the GT prompt's category in the annotation vocabulary.

## BEV Rendering

The BEV panel uses **manual 3D perspective projection** (no GPU required):

- **Coordinate system**: OpenCV camera coords (X=right, Y=down, Z=forward) converted to display coords (X=right, Y=up, Z=backward) via `(x, -y, -z)`.
- **Camera**: Configurable elevation (default 35 degrees), distance auto-scaled to fit all boxes.
- **Smart zoom**: Focal length auto-computed to tightly frame all boxes with minimal whitespace.
- **3D box corners**: Uses `bbox3D_cam` (8 rotated corner points) when available, falls back to axis-aligned boxes from `center_cam` + `dimensions`.
- **Rendering**: Painter's algorithm (depth-sorted), semi-transparent faces + wireframe edges + category labels.
- **Ground grid**: Fixed at 35-degree elevation regardless of box camera angle.

## Data Format

```json
{
  "image_id": 123,
  "file_path": "images/coco_val/000000000724.jpg",
  "width": 375, "height": 500,
  "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "gt": [
    {
      "category": "stop sign",
      "bbox2D": [x1, y1, x2, y2],
      "center_cam": [cx, cy, cz],
      "dimensions": [W, H, L],
      "bbox3D_cam": [[x,y,z], ...],
      "bbox3D_proj": [[u,v], ...]
    }
  ],
  "predictions": {
    "SAM3_3D": [...],
    "DetAny3D": [...],
    "OVMono3D": [...]
  }
}
```

## Prediction Sources

| Model | Prediction Path |
|---|---|
| WildDet3D | `vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_oracle_canonical/2026-04-06_14-04-43/eval/detection_bbox/3D/detect_3D_results.json` |
| DetAny3D | `DetAny3D/exps/in_the_wild_v3_eval/0405-121148/in_the_wild_in_the_wild_v3_predictions.json` |
| OVMono3D | `output/ovmono3d_itw_v3_oracle_predictions.json` |

## Color Scheme

| Element | Hex |
|---|---|
| WildDet3D | `#e74c3c` |
| DetAny3D | `#22c55e` |
| OVMono3D | `#f97316` |
| GT | `#a855f7` |

## Text-Prompted Comparison

A separate text-prompted comparison viewer lives under [`text/`](text/). It compares **WildDet3D** and **GDino3D** using text category prompts instead of 2D GT box prompts. See [`text/README.md`](text/README.md) for details.

Key differences from the box-prompted viewer at root:
- Models: WildDet3D + GDino3D (text-prompted only)
- Per-model confidence threshold sliders (no GT IoU matching)
- Cross-category NMS (IoU > 0.8) for near-duplicate suppression
- Category labels from the model's predicted text (not GT-relabeled)

## Hosting

- **Code**: GitHub Pages at [Silicon23/WildDet3D-model-comparison-demo](https://github.com/Silicon23/WildDet3D-model-comparison-demo)
  - Box-prompted viewer: `/` (root)
  - Text-prompted viewer: `/text/`
- **Data + Images**: HuggingFace at [Silicon23/WildDet3D-demo](https://huggingface.co/datasets/Silicon23/WildDet3D-demo) (under `model/`)
