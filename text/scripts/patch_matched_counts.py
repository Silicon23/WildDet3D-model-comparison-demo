#!/usr/bin/env python3
"""Patch index.json to add per-model prediction counts at default
confidence thresholds.

Text-based branch: predictions are filtered purely by confidence score.
The gallery pills ("M:2 G:3") display counts computed at the same default
thresholds used by the detail-page sliders on first load.
"""

import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "text"

# Must match DEFAULT_SCORE_THRESHOLDS in js/detail.js
DEFAULT_THRESHOLDS = {
    "SAM3_3D": 0.5,
    "GDino3D": 0.1,
}


def count_above_threshold(preds, threshold):
    """Count predictions with score >= threshold."""
    return sum(1 for p in preds if p.get("score", 0) >= threshold)


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
            matched_counts[model] = count_above_threshold(preds, thr)
        img_entry["matched_counts"] = matched_counts

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{total}")

    with open(index_path, "w") as f:
        json.dump(index, f)

    print(f"Patched {total} image entries with matched_counts")
    print(f"Thresholds used: {DEFAULT_THRESHOLDS}")


if __name__ == "__main__":
    main()
