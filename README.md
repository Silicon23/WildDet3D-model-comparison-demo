# 3D Detection Model Comparison Visualization

Side-by-side comparison of 4 models on InTheWild v3 benchmark: **SAM3_3D**, **GDino3D**, **DetAny3D**, **OVMono3D**.

Each image produces a 2x2 grid where each cell = **image overlay** (left) + **45-degree BEV** (right).

## Files

| File | Description |
|---|---|
| `prepare_demo_data.py` | Merge GT + 4 models' predictions into `demo_data.json` |
| `render_comparison.py` | Render static comparison images (image overlay + 3D BEV) |
| `demo_data.json` | Pre-merged data (200 images, ~13 MB) |
| `comparison_renders/` | Output PNG images |
| `index.html` / `image.html` | Interactive web viewer (separate from static renders) |
| `js/bev-renderer.js` | JS top-down BEV renderer (for web viewer) |
| `js/overlay-renderer.js` | JS image overlay renderer (for web viewer) |
| `js/three-viewer.js` | JS Three.js 3D perspective viewer (for web viewer) |

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

Predictions are matched to GT boxes using the same logic as the web interface (`detail.js: filterPredsByGTMatch`):

1. For each GT box, compute 2D IoU between GT `bbox2D` and each prediction's `bbox3D_proj` (projected to axis-aligned rect).
2. Keep the highest-scoring prediction with IoU >= 0.5.
3. Each GT keeps at most 1 matched prediction.

This ensures only meaningful predictions (those overlapping with a GT object) are shown, keeping the visualization clean.

## Image Selection Criteria

By default, `render_comparison.py` selects:
- **COCO images only** (not Objects365)
- **All 4 models must have >= 1 matched detection** on the image

This ensures every rendered comparison is informative (no empty panels).

## BEV Rendering

The BEV panel uses **manual 3D perspective projection** (no GPU required):

- **Coordinate system**: OpenCV camera coords (X=right, Y=down, Z=forward) converted to display coords (X=right, Y=up, Z=backward) via `(x, -y, -z)`.
- **Camera**: 35-degree elevation, looking straight forward (azimuth=0), distance auto-scaled to fit all boxes.
- **3D box corners**: Uses `bbox3D_cam` (8 rotated corner points from annotations) when available, falls back to axis-aligned boxes from `center_cam` + `dimensions`.
- **Rendering**: Painter's algorithm (depth-sorted), semi-transparent faces + wireframe edges.
- **Ground grid**: Auto-scaled to scene extent.

## Data Format (`demo_data.json`)

```json
[
  {
    "image_id": 123,
    "file_path": "images/coco_val/000000000724.jpg",
    "width": 375,
    "height": 500,
    "K": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    "gt": [
      {
        "category": "stop sign",
        "bbox2D": [x1, y1, x2, y2],
        "center_cam": [cx, cy, cz],
        "dimensions": [W, H, L],
        "bbox3D_cam": [[x,y,z], ...],  // 8 corners in camera coords
        "bbox3D_proj": [[u,v], ...]    // 8 corners projected to 2D
      }
    ],
    "predictions": {
      "SAM3_3D": [
        {
          "category": "stop sign",
          "score": 1.2,
          "bbox2D": [x, y, w, h],
          "center_cam": [cx, cy, cz],
          "dimensions": [W, H, L],
          "bbox3D_cam": [[x,y,z], ...],  // 8 corners (if available)
          "bbox3D_proj": [[u,v], ...]
        }
      ],
      "GDino3D": [...],
      "DetAny3D": [...],
      "OVMono3D": [...]
    }
  }
]
```

## Prediction Sources

| Model | Prediction Path |
|---|---|
| SAM3_3D | `vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_v3/2026-03-23_00-57-13/eval/detection_bbox/3D/detect_3D_results.json` |
| GDino3D | `vis4d-workspace/gdino3d_swin-t_in_the_wild_v3/2026-03-23_00-51-40/eval/detection_bbox/3D/detect_3D_results.json` |
| DetAny3D | `DetAny3D/exps/in_the_wild_v3_eval/0323-003209/in_the_wild_in_the_wild_v3_predictions.json` |
| OVMono3D | `output/ovmono3d_itw_v3_predictions.json` |

## Color Scheme

| Element | Color (BGR) | Hex |
|---|---|---|
| SAM3_3D | (55, 76, 231) | `#e74c3c` |
| GDino3D | (246, 130, 59) | `#3b82f6` |
| DetAny3D | (85, 197, 34) | `#22c55e` |
| OVMono3D | (22, 115, 249) | `#f97316` |
| GT | (247, 85, 168) | `#a855f7` |
