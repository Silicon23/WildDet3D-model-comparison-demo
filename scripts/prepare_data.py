#!/usr/bin/env python3
"""Data preparation for the 4-model comparison visualization app.

Generates:
  data/index.json  -- gallery metadata with scene tree (2470 images)
  data/images/{image_id}.json  -- per-image predictions (lazy loaded)

Usage:
    conda activate opendet3d
    python visualization_scripts/demo_comparison_v3/scripts/prepare_data.py
"""

import json
import math
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet"
OUTPUT_DIR = Path(PROJECT_ROOT) / "visualization_scripts/demo_comparison_v3/data/box"

ANN_PATH = os.path.join(
    PROJECT_ROOT, "data/in_the_wild/annotations/InTheWild_v3_val.json"
)

CATMAP_CONFIG_PATH = os.path.join(
    PROJECT_ROOT,
    "vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_oracle_v3/"
    "2026-03-23_21-02-01/config_2026-03-23_21-02-01.yaml",
)

PRED_PATHS = {
    "SAM3_3D": os.path.join(
        PROJECT_ROOT,
        "vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_oracle_v3/"
        "2026-03-23_21-02-01/eval/detection_bbox/3D/detect_3D_results.json",
    ),
    "DetAny3D": os.path.join(
        PROJECT_ROOT,
        "DetAny3D/exps/in_the_wild_v3_eval/0405-121148/"
        "in_the_wild_in_the_wild_v3_predictions.json",
    ),
    "OVMono3D": os.path.join(
        PROJECT_ROOT,
        "output/ovmono3d_itw_v3_oracle_predictions.json",
    ),
}

SCENE_FILES = [
    (
        "coco",
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection/"
        "scene_background_diversity/coco/output/train/"
        "coco_train_classifications.jsonl",
    ),
    (
        "coco",
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection/"
        "scene_background_diversity/coco/output/val/"
        "coco_val_classifications.jsonl",
    ),
    (
        "obj365",
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection/"
        "scene_background_diversity/output_scene_tags/"
        "classifications_full.jsonl",
    ),
    (
        "obj365",
        "/weka/oe-training-default/weikaih/3d_boundingbox_detection/"
        "scene_background_diversity/val_output/"
        "obj365_val_classifications.jsonl",
    ),
]

TOP_K = 50
SCORE_THRESHOLD = 0.05
MODELS = ["SAM3_3D", "DetAny3D", "OVMono3D"]


# ============================================================================
# Utilities
# ============================================================================


def load_model_id_to_name(config_yaml_path):
    """Extract {category_id: name} from an eval config's embedded cat_map.

    The vis4d config YAML contains a cat_map block with name: id entries
    that covers the full ITW vocab used when predictions were generated.
    This is the authoritative mapping for saved category_id values.
    """
    with open(config_yaml_path) as f:
        content = f.read()

    name_to_id = {}
    plain_pat = re.compile(
        r"^(\s{18,})([A-Za-z][\w \-/,.()]+): (\d+)\s*$"
    )
    quoted_pat = re.compile(
        r"^(\s{18,})'([^']+)': (\d+)\s*$"
    )
    for line in content.split("\n"):
        m = quoted_pat.match(line) or plain_pat.match(line)
        if m:
            name = m.group(2)
            val = int(m.group(3))
            if val > 2000:
                continue
            if name not in name_to_id:
                name_to_id[name] = val

    return {v: k for k, v in name_to_id.items()}


def compute_iou_2d(box_a, box_b):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def filter_by_gt_iou(pred_boxes, gt_boxes, iou_thr=0.5):
    """Keep only predictions whose 2D box has IoU >= iou_thr with any GT box."""
    if not gt_boxes:
        return []
    gt_2d_list = [g["bbox2D"] for g in gt_boxes if g.get("bbox2D")]
    result = []
    for p in pred_boxes:
        p_2d = p.get("bbox2D")
        if not p_2d or len(p_2d) < 4:
            continue
        for gt_2d in gt_2d_list:
            if compute_iou_2d(p_2d, gt_2d) >= iou_thr:
                result.append(p)
                break
    return result


def project_3d_to_2d(bbox3D, K):
    """Project 8 corner points to 2D using camera intrinsics."""
    K = np.array(K)
    pts = np.array(bbox3D)
    if pts.shape != (8, 3):
        return None
    if not (pts[:, 2] > 0.01).any():
        return None
    proj = (K @ pts.T).T
    z = proj[:, 2:3]
    z[z < 0.01] = 0.01
    proj_2d = proj[:, :2] / z
    return [[round(float(x), 1), round(float(y), 1)] for x, y in proj_2d]


def yaw_from_R(R_cam):
    """Extract yaw (rotation around Y-axis) from 3x3 rotation matrix."""
    R = np.array(R_cam)
    if R.shape != (3, 3):
        return 0.0
    return float(math.atan2(R[0][2], R[0][0]))


def load_scene_classifications():
    """Load scene classifications from JSONL files."""
    lookup = {}
    for dataset_name, fpath in SCENE_FILES:
        if not os.path.exists(fpath):
            print(f"  WARNING: missing {fpath}")
            continue
        count = 0
        with open(fpath) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                img_id = entry.get("image_id")
                scene_path = entry.get("selected_path", "")
                if img_id is not None and scene_path:
                    lookup[(dataset_name, img_id)] = scene_path
                    count += 1
        print(f"  Loaded {count} scenes from {os.path.basename(fpath)}")
    return lookup


def image_source_and_id(file_path):
    """Extract (dataset_name, numeric_id, source_dir) from file_path."""
    parts = file_path.split("/")
    source_dir = parts[1]  # coco_val, coco_train, obj365_val
    filename = parts[2]    # 000000000724.jpg or obj365_val_000000000702.jpg
    name_no_ext = filename.split(".")[0]
    # obj365 filenames have prefix like "obj365_val_000000000702"
    # strip any non-digit prefix to get numeric id
    digits = ""
    for i in range(len(name_no_ext) - 1, -1, -1):
        if name_no_ext[i].isdigit():
            digits = name_no_ext[i] + digits
        else:
            break
    numeric_id = int(digits) if digits else 0
    if source_dir in ("coco_val", "coco_train"):
        dataset = "coco"
    elif source_dir.startswith("obj365"):
        dataset = "obj365"
    else:
        dataset = source_dir
    return dataset, numeric_id, source_dir


def build_scene_tree(scene_paths):
    """Build hierarchical scene tree from list of scene_path strings."""
    tree = {"name": "root", "path": "", "children": [], "image_count": 0}
    node_map = {"": tree}

    all_paths = sorted(set(scene_paths))
    for sp in all_paths:
        parts = sp.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if partial in node_map:
                continue
            parent_path = "/".join(parts[:i])
            parent = node_map[parent_path]
            node = {
                "name": parts[i],
                "path": partial,
                "children": [],
                "image_count": 0,
            }
            parent["children"].append(node)
            node_map[partial] = node

    # Count images per node (leaf counts propagated up)
    counts = Counter(scene_paths)
    for sp, cnt in counts.items():
        parts = sp.split("/")
        for i in range(len(parts)):
            partial = "/".join(parts[: i + 1])
            if partial in node_map:
                node_map[partial]["image_count"] += cnt
    tree["image_count"] = sum(counts.values())

    return tree


def bbox_xywh_to_xyxy(bbox):
    """Convert [x, y, w, h] to [x1, y1, x2, y2]."""
    return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]


# ============================================================================
# Main
# ============================================================================


def main():
    print("Loading annotations...")
    with open(ANN_PATH) as f:
        ann_data = json.load(f)

    img_id_to_info = {img["id"]: img for img in ann_data["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in ann_data["categories"]}

    # Load authoritative category mapping from eval config for predictions
    print("Loading model category mapping from eval config...")
    model_id_to_name = load_model_id_to_name(CATMAP_CONFIG_PATH)
    print(f"  Loaded {len(model_id_to_name)} categories from config")

    # GT by image
    gt_by_image = defaultdict(list)
    for ann in ann_data["annotations"]:
        if ann.get("valid3D", True) and not ann.get("behind_camera", False):
            gt_by_image[ann["image_id"]].append(ann)

    print(f"Images: {len(img_id_to_info)}, GT annotations: {sum(len(v) for v in gt_by_image.values())}")

    # Scene classifications
    print("Loading scene classifications...")
    scene_lookup = load_scene_classifications()
    print(f"  Total: {len(scene_lookup)} entries")

    # Load predictions
    model_preds_by_img = {}
    for model_name, pred_path in PRED_PATHS.items():
        print(f"Loading {model_name}...")
        with open(pred_path) as f:
            preds = json.load(f)
        # Normalize DetAny3D
        for p in preds:
            if "pose" in p and "R_cam" not in p:
                p["R_cam"] = p["pose"]
        by_img = defaultdict(list)
        for p in preds:
            by_img[p["image_id"]].append(p)
        # Sort by score, keep top-K
        for iid in by_img:
            by_img[iid].sort(key=lambda x: x.get("score", 0), reverse=True)
            if model_name == "SAM3_3D":
                # WildDet3D has built-in filtering, keep all predictions
                by_img[iid] = by_img[iid][:TOP_K]
            else:
                above = [p for p in by_img[iid] if p.get("score", 0) >= SCORE_THRESHOLD]
                by_img[iid] = above[:TOP_K]
        model_preds_by_img[model_name] = dict(by_img)
        total = sum(len(v) for v in by_img.values())
        print(f"  {model_name}: {total} preds across {len(by_img)} images")

    # Process each image
    print("Generating per-image JSONs...")
    os.makedirs(OUTPUT_DIR / "images", exist_ok=True)

    index_images = []
    images_by_scene = defaultdict(list)
    all_scene_paths = []

    for img_id in tqdm(sorted(img_id_to_info.keys())):
        img_info = img_id_to_info[img_id]
        K = img_info["K"]

        # Scene path
        dataset, numeric_id, source_dir = image_source_and_id(
            img_info["file_path"]
        )
        scene_path = scene_lookup.get(
            (dataset, numeric_id), "unclassified"
        )
        all_scene_paths.append(scene_path)

        # GT boxes
        gt_anns = gt_by_image.get(img_id, [])
        gt_boxes = []
        for ann in gt_anns:
            bbox3D_cam_raw = ann["bbox3D_cam"]
            proj = project_3d_to_2d(bbox3D_cam_raw, K)
            R_cam = ann.get("R_cam")
            # Round 3D corners for compactness
            bbox3D_cam_rounded = None
            if bbox3D_cam_raw and len(bbox3D_cam_raw) == 8:
                bbox3D_cam_rounded = [
                    [round(float(c), 3) for c in pt]
                    for pt in bbox3D_cam_raw
                ]
            gt_boxes.append({
                "category": ann.get(
                    "category_name",
                    cat_id_to_name.get(ann["category_id"], "?"),
                ),
                "bbox2D": ann["bbox2D_proj"],
                "bbox3D_proj": proj,
                "bbox3D_cam": bbox3D_cam_rounded,
                "center_cam": ann["center_cam"],
                "dimensions": ann["dimensions"],
                "yaw": yaw_from_R(R_cam) if R_cam else 0.0,
            })

        # Model predictions
        model_boxes = {}
        model_counts = {}
        for model_name in MODELS:
            preds = model_preds_by_img.get(model_name, {}).get(img_id, [])
            boxes = []
            for p in preds:
                bbox3D = p.get("bbox3D")
                proj = None
                if bbox3D and len(bbox3D) == 8:
                    proj = project_3d_to_2d(bbox3D, K)

                R_cam = p.get("R_cam")
                center = p.get("center_cam")
                depth = p.get("depth")
                if depth is None and center:
                    depth = center[2]

                bbox3D_cam_rounded = None
                if bbox3D and len(bbox3D) == 8:
                    bbox3D_cam_rounded = [
                        [round(float(c), 3) for c in pt]
                        for pt in bbox3D
                    ]

                # WildDet3D uses eval-config cat_map; others use annotation cat_map
                pred_cat_map = model_id_to_name if model_name == "SAM3_3D" else cat_id_to_name
                boxes.append({
                    "category": pred_cat_map.get(
                        p["category_id"], f"id_{p['category_id']}"
                    ),
                    "score": round(p.get("score", 0), 3),
                    "bbox2D": bbox_xywh_to_xyxy(p["bbox"]),
                    "bbox3D_proj": proj,
                    "bbox3D_cam": bbox3D_cam_rounded,
                    "center_cam": center,
                    "dimensions": p.get("dimensions"),
                    "yaw": yaw_from_R(R_cam) if R_cam else 0.0,
                    "depth": round(depth, 2) if depth else None,
                })
            # WildDet3D: filter out spurious boxes that don't match any GT
            if model_name == "SAM3_3D":
                boxes = filter_by_gt_iou(boxes, gt_boxes, iou_thr=0.5)
            model_boxes[model_name] = boxes
            model_counts[model_name] = len(boxes)

        # Write per-image JSON
        per_image = {
            "image_id": img_id,
            "file_path": img_info["file_path"],
            "width": img_info["width"],
            "height": img_info["height"],
            "K": K,
            "gt": gt_boxes,
            "predictions": model_boxes,
        }
        with open(OUTPUT_DIR / "images" / f"{img_id}.json", "w") as f:
            json.dump(per_image, f)

        # Index entry
        index_images.append({
            "id": img_id,
            "file_path": img_info["file_path"],
            "width": img_info["width"],
            "height": img_info["height"],
            "source": source_dir,
            "scene_path": scene_path,
            "formatted_id": img_info.get("formatted_id", str(img_id)),
            "gt_count": len(gt_boxes),
            "model_counts": model_counts,
        })
        images_by_scene[scene_path].append(img_id)

    # Build scene tree
    print("Building scene tree...")
    scene_tree = build_scene_tree(all_scene_paths)

    # Build index.json
    index = {
        "total_images": len(index_images),
        "scene_tree": scene_tree,
        "images": index_images,
        "images_by_scene": dict(images_by_scene),
        "categories": cat_id_to_name,
        "models": MODELS,
    }

    index_path = OUTPUT_DIR / "index.json"
    with open(index_path, "w") as f:
        json.dump(index, f)

    idx_size = os.path.getsize(index_path) / 1024
    per_img_size = sum(
        os.path.getsize(OUTPUT_DIR / "images" / f"{img['id']}.json")
        for img in index_images[:10]
    ) / 10 / 1024
    print(f"\nDone!")
    print(f"  index.json: {idx_size:.0f} KB")
    print(f"  Per-image JSON avg: {per_img_size:.1f} KB")
    print(f"  Total images: {len(index_images)}")
    print(f"  Scene categories: {len(set(all_scene_paths))}")

    # Scene stats
    scene_counts = Counter(all_scene_paths)
    print(f"\nScene distribution:")
    for sp, cnt in scene_counts.most_common():
        print(f"  {sp}: {cnt}")


if __name__ == "__main__":
    main()
