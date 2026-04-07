# WildDet3D - 3D Detection Model Comparison

Interactive side-by-side comparison of 3D object detection models on the InTheWild benchmark. Each viewer shows image overlays with 3D bounding box wireframes and perspective BEV views.

## Viewers

| Viewer | Path | Models | Prompting |
|---|---|---|---|
| [Box Prompts](box/) | `/box/` | WildDet3D, DetAny3D, OVMono3D | GT 2D bounding boxes |
| [Text Prompts](text/) | `/text/` | WildDet3D, 3D-MOOD | Category text |

## Structure

```
index.html              Landing page
box/                    Box-prompted comparison viewer
  index.html / image.html
  css/ js/ scripts/
text/                   Text-prompted comparison viewer
  index.html / image.html
  css/ js/ scripts/
data/                   Generated data (gitignored, hosted on HuggingFace)
  box/                  Box-prompted per-image JSONs
  text/                 Text-prompted per-image JSONs
images/                 Symlink to extracted images (gitignored)
```

## Hosting

- **Code**: GitHub Pages at [Silicon23/WildDet3D-model-comparison-demo](https://github.com/Silicon23/WildDet3D-model-comparison-demo)
  - Landing page: `/`
  - Box-prompted viewer: `/box/`
  - Text-prompted viewer: `/text/`
- **Data + Images**: HuggingFace at [Silicon23/WildDet3D-demo](https://huggingface.co/datasets/Silicon23/WildDet3D-demo) (under `model/`)

## Local Development

```bash
cd visualization_scripts/demo_comparison_v3
python3 -m http.server 8765
# Open http://localhost:8765
```

Local mode auto-detects and reads from `data/` and `images/` instead of HuggingFace.

See [box/README.md](box/README.md) and [text/README.md](text/README.md) for viewer-specific details.
