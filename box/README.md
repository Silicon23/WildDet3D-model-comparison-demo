# 3D Detection Model Comparison - Box Prompts

Comparison of 3 models on InTheWild benchmark using **2D GT box prompts** to obtain 3D box predictions: **WildDet3D (Ours)**, **DetAny3D**, **OVMono3D**.

Layout: 2D box prompts (top row) + 2x2 grid (GT 3D, WildDet3D, DetAny3D, OVMono3D), each cell showing image overlay + perspective BEV.

## Data Preparation

```bash
conda activate opendet3d
export PYTHONPATH=/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet:$PYTHONPATH

# Generate per-image JSONs into data/box/
python visualization_scripts/demo_comparison_v3/box/scripts/prepare_data.py

# Patch gallery matched counts
python visualization_scripts/demo_comparison_v3/box/scripts/patch_matched_counts.py
```

## Prediction Filtering

- **WildDet3D**: Spurious predictions filtered by 2D IoU >= 0.5 with GT boxes (oracle mode should produce one output per prompt, but occasionally generates extras).
- **DetAny3D, OVMono3D**: No additional filtering (oracle/box-prompt mode, one output per GT prompt).
- **Category labels**: All models use the annotation file's category mapping.

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
