#!/usr/bin/env python3
"""Patch index.json to add per-model prediction counts at default
confidence thresholds with cross-category NMS.

Text-based branch: predictions are filtered by confidence score then
cross-category NMS (IoU > 0.8), matching the detail page behavior.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "text"

# Must match DEFAULT_SCORE_THRESHOLDS in js/detail.js
DEFAULT_THRESHOLDS = {
    "SAM3_3D": 0.5,
    "GDino3D": 0.1,
}

CROSS_CAT_NMS_IOU = 0.8


def compute_iou_2d(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0


def filter_and_nms(preds, threshold):
    """Filter by score threshold, then suppress cross-category duplicates."""
    above = [p for p in preds if p.get("score", 0) >= threshold]
    above.sort(key=lambda p: p.get("score", 0), reverse=True)

    kept = []
    for p in above:
        p_box = p.get("bbox2D")
        if not p_box or len(p_box) < 4:
            kept.append(p)
            continue
        suppressed = False
        for q in kept:
            if q.get("category") == p.get("category"):
                continue
            q_box = q.get("bbox2D")
            if not q_box or len(q_box) < 4:
                continue
            if compute_iou_2d(p_box, q_box) > CROSS_CAT_NMS_IOU:
                suppressed = True
                break
        if not suppressed:
            kept.append(p)
    return len(kept)


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

        matched_counts = {}
        for model in models:
            preds = img_data.get("predictions", {}).get(model, [])
            thr = DEFAULT_THRESHOLDS.get(model, 0.0)
            matched_counts[model] = filter_and_nms(preds, thr)
        img_entry["matched_counts"] = matched_counts

        # Copy prompt categories from per-image JSON (includes all 2D annotations)
        img_entry["prompt_categories"] = img_data.get("prompt_categories", [])

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total}")

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"Patched {total} image entries with matched_counts")
    print(f"Thresholds: {DEFAULT_THRESHOLDS}, NMS IoU: {CROSS_CAT_NMS_IOU}")


if __name__ == "__main__":
    main()
