"""Prepare lightweight demo data for the 4-model comparison HTML viewer.

Merges predictions from SAM3_3D, GDino3D, DetAny3D, OVMono3D on
InTheWild_v3_val into a single JSON grouped by image_id.
Keeps only top-K predictions per model per image to keep file small.

Usage:
    conda activate opendet3d
    python visualization_scripts/demo_comparison_v3/prepare_demo_data.py
"""

import json
import os
from collections import defaultdict

import numpy as np

PROJECT_ROOT = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualization_scripts/demo_comparison_v3")

# Annotation
ANN_PATH = os.path.join(
    PROJECT_ROOT, "data/in_the_wild/annotations/InTheWild_v3_val.json"
)

# Prediction paths
PRED_PATHS = {
    "SAM3_3D": os.path.join(
        PROJECT_ROOT,
        "vis4d-workspace/sam3_3d_lingbot_depth_freeze21_in_the_wild_v3/"
        "2026-03-23_00-57-13/eval/detection_bbox/3D/detect_3D_results.json",
    ),
    "GDino3D": os.path.join(
        PROJECT_ROOT,
        "vis4d-workspace/gdino3d_swin-t_in_the_wild_v3/"
        "2026-03-23_00-51-40/eval/detection_bbox/3D/detect_3D_results.json",
    ),
    "DetAny3D": os.path.join(
        PROJECT_ROOT,
        "DetAny3D/exps/in_the_wild_v3_eval/0323-003209/"
        "in_the_wild_in_the_wild_v3_predictions.json",
    ),
    "OVMono3D": os.path.join(
        PROJECT_ROOT,
        "output/ovmono3d_itw_v3_predictions.json",
    ),
}

# Keep top K predictions per model per image
TOP_K = 30
SCORE_THRESHOLD = 0.1
# Only include images that have GT annotations
MAX_IMAGES = 200  # limit for demo


def project_3d_to_2d(bbox3D, K):
    """Project 8 corner points to 2D using camera intrinsics.

    Args:
        bbox3D: list of 8 points, each [x, y, z]
        K: 3x3 camera intrinsic matrix

    Returns:
        list of 8 [u, v] 2D pixel coordinates
    """
    K = np.array(K)
    pts = np.array(bbox3D)  # (8, 3)
    # Filter out points behind camera
    valid = pts[:, 2] > 0.01
    if not valid.any():
        return None
    proj = (K @ pts.T).T  # (8, 3)
    proj_2d = proj[:, :2] / proj[:, 2:3]
    return proj_2d.tolist()


def main():
    print("Loading annotations...")
    with open(ANN_PATH) as f:
        ann_data = json.load(f)

    img_id_to_info = {img["id"]: img for img in ann_data["images"]}
    cat_id_to_name = {cat["id"]: cat["name"] for cat in ann_data["categories"]}

    # Group GT annotations by image
    gt_by_image = defaultdict(list)
    for ann in ann_data["annotations"]:
        if ann.get("valid3D", True) and not ann.get("behind_camera", False):
            gt_by_image[ann["image_id"]].append(ann)

    # Only include images with GT
    valid_img_ids = sorted(
        [iid for iid in gt_by_image if len(gt_by_image[iid]) > 0]
    )
    # Sample subset for demo
    if len(valid_img_ids) > MAX_IMAGES:
        step = len(valid_img_ids) // MAX_IMAGES
        valid_img_ids = valid_img_ids[::step][:MAX_IMAGES]

    valid_img_set = set(valid_img_ids)
    print(f"Using {len(valid_img_ids)} images for demo")

    # Load and group predictions by model
    model_preds = {}
    for model_name, pred_path in PRED_PATHS.items():
        print(f"Loading {model_name} from {pred_path}...")
        with open(pred_path) as f:
            preds = json.load(f)

        # Normalize DetAny3D format
        for p in preds:
            if "pose" in p and "R_cam" not in p:
                p["R_cam"] = p["pose"]

        # Group by image_id, filter, keep top-K
        by_img = defaultdict(list)
        for p in preds:
            if p["image_id"] in valid_img_set:
                score = p.get("score", 0)
                if score >= SCORE_THRESHOLD:
                    by_img[p["image_id"]].append(p)

        # Sort by score and keep top-K per image
        for iid in by_img:
            by_img[iid].sort(key=lambda x: x.get("score", 0), reverse=True)
            by_img[iid] = by_img[iid][:TOP_K]

        model_preds[model_name] = dict(by_img)
        total = sum(len(v) for v in by_img.values())
        print(f"  {model_name}: {total} predictions across {len(by_img)} images")

    # Build demo data
    demo_images = []
    for img_id in valid_img_ids:
        img_info = img_id_to_info[img_id]
        K = img_info["K"]

        # GT boxes
        gt_anns = gt_by_image.get(img_id, [])
        gt_boxes = []
        for ann in gt_anns[:TOP_K]:
            proj = project_3d_to_2d(ann["bbox3D_cam"], K)
            gt_boxes.append({
                "category": ann.get("category_name", cat_id_to_name.get(ann["category_id"], "?")),
                "bbox2D": ann["bbox2D_proj"],
                "center_cam": ann["center_cam"],
                "dimensions": ann["dimensions"],
                "bbox3D_cam": ann["bbox3D_cam"],
                "bbox3D_proj": proj,
            })

        # Model predictions
        model_boxes = {}
        for model_name in PRED_PATHS:
            preds = model_preds[model_name].get(img_id, [])
            boxes = []
            for p in preds:
                bbox3D = p.get("bbox3D")
                proj = None
                if bbox3D and len(bbox3D) == 8:
                    proj = project_3d_to_2d(bbox3D, K)

                boxes.append({
                    "category": cat_id_to_name.get(p["category_id"], f"id_{p['category_id']}"),
                    "score": round(p.get("score", 0), 3),
                    "bbox2D": p["bbox"],
                    "center_cam": p.get("center_cam"),
                    "dimensions": p.get("dimensions"),
                    "bbox3D_cam": p.get("bbox3D"),
                    "depth": p.get("depth", p.get("center_cam", [0, 0, 0])[2] if p.get("center_cam") else 0),
                    "bbox3D_proj": proj,
                })
            model_boxes[model_name] = boxes

        demo_images.append({
            "image_id": img_id,
            "file_path": img_info["file_path"],
            "width": img_info["width"],
            "height": img_info["height"],
            "K": K,
            "gt": gt_boxes,
            "predictions": model_boxes,
        })

    # Save
    output_path = os.path.join(OUTPUT_DIR, "demo_data.json")
    with open(output_path, "w") as f:
        json.dump(demo_images, f)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nSaved {len(demo_images)} images to {output_path} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
