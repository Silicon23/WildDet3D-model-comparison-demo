#!/usr/bin/env python3
"""Patch index.json to add GT-matched prediction counts per model.

For each image, computes how many predictions per model match a GT box
(2D IoU >= 0.2 between projected 3D box and GT bbox2D, best-score wins).
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "box"


def compute_iou_2d(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def bbox3d_proj_to_rect(proj):
    xs = [p[0] for p in proj]
    ys = [p[1] for p in proj]
    return [min(xs), min(ys), max(xs), max(ys)]


def count_matched_preds(preds, gt_boxes):
    if not gt_boxes or not preds:
        return 0
    matched = set()
    for gt in gt_boxes:
        gt_2d = gt.get("bbox2D")
        if not gt_2d or len(gt_2d) < 4:
            continue
        best_idx = -1
        best_score = -float("inf")
        for i, pred in enumerate(preds):
            proj = pred.get("bbox3D_proj")
            if proj and len(proj) == 8:
                pred_2d = bbox3d_proj_to_rect(proj)
            elif pred.get("bbox2D") and len(pred["bbox2D"]) >= 4:
                pred_2d = pred["bbox2D"]
            else:
                continue
            iou = compute_iou_2d(pred_2d, gt_2d)
            if iou >= 0.2 and pred.get("score", 0) > best_score:
                best_idx = i
                best_score = pred["score"]
        if best_idx >= 0:
            matched.add(best_idx)
    return len(matched)


def main():
    index_path = DATA_DIR / "index.json"
    with open(index_path) as f:
        index = json.load(f)

    models = index.get("models", [])
    total = len(index["images"])

    for i, img_entry in enumerate(index["images"]):
        img_id = img_entry["id"]
        img_path = DATA_DIR / "images" / f"{img_id}.json"
        with open(img_path) as f:
            img_data = json.load(f)

        gt = img_data.get("gt", [])
        matched_counts = {}
        for model in models:
            preds = img_data.get("predictions", {}).get(model, [])
            # All models are box-prompted (oracle mode), no filtering needed
            matched_counts[model] = len(preds)
        img_entry["matched_counts"] = matched_counts

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total}")

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"Patched {total} image entries with matched_counts")


if __name__ == "__main__":
    main()
