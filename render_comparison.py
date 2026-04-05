"""Render side-by-side 3D box comparison images for 4 models.

Each panel: image with 3D boxes (left) + 45-degree 3D perspective BEV (right).
BEV uses manual perspective projection + painter's algorithm rendering,
matching the Three.js ThreeViewerFocused style (cylinder-edge wireframe boxes
from elevated viewpoint with ground-plane grid).

Usage:
    conda activate opendet3d
    python visualization_scripts/demo_comparison_v3/render_comparison.py
"""

import json
import math
import os

import cv2
import h5py
import numpy as np

PROJECT_ROOT = "/weka/oe-training-default/weikaih/3d_boundingbox_detection/Foundation3DDet/sam3_da3/Foundation3DDet"
DEMO_DATA = os.path.join(
    PROJECT_ROOT,
    "visualization_scripts/demo_comparison_v3/demo_data.json",
)
ITW_DATA_ROOT = os.path.join(PROJECT_ROOT, "data/in_the_wild")
OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "visualization_scripts/demo_comparison_v3/comparison_renders",
)

MODEL_COLORS_BGR = {
    "SAM3_3D": (55, 76, 231),
    "GDino3D": (246, 130, 59),
    "DetAny3D": (85, 197, 34),
    "OVMono3D": (22, 115, 249),
}
GT_COLOR_BGR = (247, 85, 168)
SCORE_THR = 0.1
IOU_MATCH_THR = 0.5


# ============================================================================
# Image overlay (same as before)
# ============================================================================

def draw_3d_box_from_proj(img, bbox3D_proj, color, thickness=2, scale=1.0):
    if bbox3D_proj is None or len(bbox3D_proj) != 8:
        return
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    h, w = img.shape[:2]
    for i, j in edges:
        x1 = int(bbox3D_proj[i][0] * scale)
        y1 = int(bbox3D_proj[i][1] * scale)
        x2 = int(bbox3D_proj[j][0] * scale)
        y2 = int(bbox3D_proj[j][1] * scale)
        if abs(x1) > w * 3 or abs(y1) > h * 3 or abs(x2) > w * 3 or abs(y2) > h * 3:
            continue
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def load_image(file_path):
    extracted = os.path.join(ITW_DATA_ROOT, "extracted_images", file_path)
    if os.path.exists(extracted):
        return cv2.imread(extracted)
    hdf5_path = os.path.join(ITW_DATA_ROOT, file_path)
    hdf5_file = hdf5_path.rsplit(".", 1)[0] + ".hdf5"
    if os.path.exists(hdf5_file):
        with h5py.File(hdf5_file, "r") as f:
            data = np.array(f["data"])
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if os.path.exists(hdf5_path):
        return cv2.imread(hdf5_path)
    return None


def compute_iou_2d(boxA, boxB):
    """IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(boxA[0], boxB[0])
    y1 = max(boxA[1], boxB[1])
    x2 = min(boxA[2], boxB[2])
    y2 = min(boxA[3], boxB[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0


def bbox3d_proj_to_rect(proj):
    """Convert 8 projected corners to axis-aligned [x1,y1,x2,y2]."""
    xs = [p[0] for p in proj]
    ys = [p[1] for p in proj]
    return [min(xs), min(ys), max(xs), max(ys)]


def filter_preds_by_gt_match(preds, gt_boxes, model_name):
    """Match predictions to GT boxes (IoU>0.5), keep best per GT."""
    if not gt_boxes or not preds:
        return []

    matched_indices = set()
    for gt in gt_boxes:
        gt_2d = gt.get("bbox2D")
        if not gt_2d or len(gt_2d) < 4:
            continue

        best_idx = -1
        best_score = -float("inf")
        for i, p in enumerate(preds):
            if p.get("bbox3D_proj") and len(p["bbox3D_proj"]) == 8:
                pred_2d = bbox3d_proj_to_rect(p["bbox3D_proj"])
            elif p.get("bbox2D") and len(p["bbox2D"]) >= 4:
                pred_2d = p["bbox2D"]
            else:
                continue

            iou = compute_iou_2d(pred_2d, gt_2d)
            if iou >= IOU_MATCH_THR and p.get("score", 0) > best_score:
                best_idx = i
                best_score = p["score"]

        if best_idx >= 0:
            matched_indices.add(best_idx)

    return [preds[i] for i in matched_indices]


def render_single_model(img, entry, model_name, scale):
    canvas = img.copy()
    color = MODEL_COLORS_BGR.get(model_name, (0, 255, 0))

    all_preds = entry.get("predictions", {}).get(model_name, [])
    preds = filter_preds_by_gt_match(all_preds, entry.get("gt", []), model_name)
    drawn = 0
    for p in preds:
        if "bbox3D_proj" in p:
            draw_3d_box_from_proj(canvas, p["bbox3D_proj"], color, 2, scale)
            drawn += 1
    label = f"{model_name} ({drawn} dets)"
    cv2.rectangle(canvas, (0, 0), (len(label) * 12 + 10, 30), (0, 0, 0), -1)
    cv2.putText(canvas, label, (5, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return canvas


# ============================================================================
# 3D perspective BEV renderer (CPU, no GPU needed)
# ============================================================================

def look_at_matrix(eye, target, up=np.array([0.0, 1.0, 0.0])):
    """Compute 4x4 view matrix (world -> camera)."""
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    true_up = np.cross(right, fwd)

    R = np.eye(4)
    R[0, :3] = right
    R[1, :3] = true_up
    R[2, :3] = -fwd
    T = np.eye(4)
    T[:3, 3] = -eye
    return R @ T


def project_points(pts_3d, view_mat, K, w, h):
    """Project Nx3 world points to Nx2 pixel coords + depths.

    Returns (pts_2d, depths). pts_2d[:,0]=x, pts_2d[:,1]=y.
    """
    N = pts_3d.shape[0]
    ones = np.ones((N, 1))
    pts_h = np.hstack([pts_3d, ones])  # Nx4

    cam_pts = (view_mat @ pts_h.T).T  # Nx4
    depths = -cam_pts[:, 2]  # camera looks along -Z

    # Project (view matrix has Y-up, Z-backward; image needs Y-down)
    x = K[0, 0] * cam_pts[:, 0] / (depths + 1e-8) + K[0, 2]
    y = -K[1, 1] * cam_pts[:, 1] / (depths + 1e-8) + K[1, 2]

    pts_2d = np.stack([x, y], axis=1)
    return pts_2d, depths


def get_corners_display(box):
    """Get 8 corners in display coords (x, -y, -z) from OpenCV.

    Uses bbox3D_cam if available (rotated corners), otherwise axis-aligned.
    """
    if "bbox3D_cam" in box and box["bbox3D_cam"] is not None:
        corners = np.array(box["bbox3D_cam"])  # (8, 3) in OpenCV
        if corners.shape == (8, 3):
            corners_disp = corners.copy()
            corners_disp[:, 1] *= -1
            corners_disp[:, 2] *= -1
            return corners_disp

    # Fallback: axis-aligned from center + dims
    cx, cy, cz = box["center_cam"]
    w, h, l = box["dimensions"]
    hw, hh, hl = w / 2, h / 2, l / 2
    corners = np.array([
        [cx - hw, cy - hh, cz - hl],
        [cx + hw, cy - hh, cz - hl],
        [cx + hw, cy + hh, cz - hl],
        [cx - hw, cy + hh, cz - hl],
        [cx - hw, cy - hh, cz + hl],
        [cx + hw, cy - hh, cz + hl],
        [cx + hw, cy + hh, cz + hl],
        [cx - hw, cy + hh, cz + hl],
    ])
    corners[:, 1] *= -1
    corners[:, 2] *= -1
    return corners


EDGES_3D = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (4, 5), (5, 6), (6, 7), (7, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# 6 faces as quads (indices into corners)
FACES_3D = [
    (0, 1, 2, 3),  # back
    (4, 5, 6, 7),  # front
    (0, 1, 5, 4),  # bottom
    (2, 3, 7, 6),  # top
    (0, 3, 7, 4),  # left
    (1, 2, 6, 5),  # right
]


def draw_3d_box_perspective(canvas, corners_2d, depths, color, alpha=0.15, line_w=2):
    """Draw one 3D box with semi-transparent faces + wireframe edges."""
    h, w = canvas.shape[:2]

    # Check if any point is behind camera
    if np.any(depths <= 0):
        return

    pts = corners_2d.astype(np.int32)

    # Check if all points are wildly off screen
    if np.all(pts[:, 0] < -w) or np.all(pts[:, 0] > 2 * w):
        return
    if np.all(pts[:, 1] < -h) or np.all(pts[:, 1] > 2 * h):
        return

    # Sort faces by average depth (painter's algorithm: draw far first)
    face_depths = []
    for face in FACES_3D:
        avg_d = np.mean([depths[i] for i in face])
        face_depths.append(avg_d)
    face_order = np.argsort(face_depths)[::-1]  # far to near

    # Draw filled faces
    overlay = canvas.copy()
    for fi in face_order:
        face = FACES_3D[fi]
        poly = np.array([pts[i] for i in face], dtype=np.int32)
        cv2.fillConvexPoly(overlay, poly, color)
    cv2.addWeighted(overlay, alpha, canvas, 1 - alpha, 0, canvas)

    # Draw edges
    for i, j in EDGES_3D:
        cv2.line(canvas, tuple(pts[i]), tuple(pts[j]), color, line_w, cv2.LINE_AA)


def draw_ground_grid(canvas, view_mat, K, w, h, x_range, z_range, y_level,
                     spacing=2.0, color=(200, 200, 200)):
    """Draw grid on the ground plane."""
    x_min, x_max = x_range
    z_min, z_max = z_range

    lines_3d = []
    # Lines along X
    z = math.ceil(z_min / spacing) * spacing
    while z <= z_max:
        lines_3d.append(([x_min, y_level, z], [x_max, y_level, z]))
        z += spacing
    # Lines along Z
    x = math.ceil(x_min / spacing) * spacing
    while x <= x_max:
        lines_3d.append(([x, y_level, z_min], [x, y_level, z_max]))
        x += spacing

    for p1, p2 in lines_3d:
        pts = np.array([p1, p2])
        pts_2d, depths = project_points(pts, view_mat, K, w, h)
        if np.any(depths <= 0):
            continue
        pt1 = tuple(pts_2d[0].astype(int))
        pt2 = tuple(pts_2d[1].astype(int))
        cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)


def render_bev_perspective(entry, model_name, bev_w, bev_h):
    """Render a 45-degree perspective BEV view using manual projection."""
    all_preds = entry.get("predictions", {}).get(model_name, [])
    matched_preds = filter_preds_by_gt_match(all_preds, entry.get("gt", []), model_name)
    pred_boxes = [p for p in matched_preds
                  if "center_cam" in p and "dimensions" in p]
    gt_boxes = []  # Only show predictions, aligned with image panel

    pred_color = MODEL_COLORS_BGR.get(model_name, (0, 255, 0))

    if not gt_boxes and not pred_boxes:
        img = np.full((bev_h, bev_w, 3), 245, dtype=np.uint8)
        cv2.putText(img, "No 3D boxes", (bev_w // 4, bev_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        return img

    # Compute scene bounds
    all_centers_cv = []
    all_dims = []
    for b in gt_boxes + pred_boxes:
        all_centers_cv.append(b["center_cam"])
        all_dims.append(b["dimensions"])

    centers = np.array(all_centers_cv)
    dims_arr = np.array(all_dims)
    scene_center_cv = centers.mean(axis=0)
    max_dim = dims_arr.max()
    scene_range = max(
        centers[:, 0].ptp() + max_dim,
        centers[:, 2].ptp() + max_dim,
        max_dim * 3,
        2.0,
    )

    # Convert scene center to display coords (x, -y, -z)
    scene_center = np.array([
        scene_center_cv[0],
        -scene_center_cv[1],
        -scene_center_cv[2],
    ])

    # Camera: elevated front view, close to the boxes
    # Use median-based range to avoid outlier boxes pulling camera too far
    dists_from_center = np.linalg.norm(centers - scene_center_cv, axis=1)
    effective_range = max(np.percentile(dists_from_center, 80) * 2 + max_dim, max_dim * 3, 1.0)
    distance = effective_range * 1.0
    elev = math.radians(35)
    azim = math.radians(0)
    cam_offset = np.array([
        distance * math.cos(elev) * math.sin(azim),
        distance * math.sin(elev),
        distance * math.cos(elev) * math.cos(azim),
    ])
    eye = scene_center + cam_offset

    view_mat = look_at_matrix(eye, scene_center)

    # Intrinsics
    fx = fy = bev_w * 0.85
    cx_i = bev_w / 2
    cy_i = bev_h / 2
    K = np.array([
        [fx, 0, cx_i],
        [0, fy, cy_i],
        [0, 0, 1],
    ])

    # Light background
    canvas = np.full((bev_h, bev_w, 3), 248, dtype=np.uint8)

    # Ground grid
    ground_y_cv = centers[:, 1].max() + dims_arr[:, 1].max() / 2
    ground_y_disp = -ground_y_cv
    grid_half = effective_range * 0.6
    grid_spacing = max(0.5, effective_range / 5)
    draw_ground_grid(
        canvas, view_mat, K, bev_w, bev_h,
        x_range=[scene_center[0] - grid_half, scene_center[0] + grid_half],
        z_range=[scene_center[2] - grid_half, scene_center[2] + grid_half],
        y_level=ground_y_disp,
        spacing=grid_spacing,
        color=(210, 210, 210),
    )

    # Collect all boxes with depth for painter's algorithm sorting
    box_items = []
    for b in gt_boxes:
        corners = get_corners_display(b)
        center_disp = corners.mean(axis=0)
        cam_pt = (view_mat @ np.append(center_disp, 1))[:3]
        box_items.append((corners, GT_COLOR_BGR, -cam_pt[2], "gt"))
    for b in pred_boxes:
        corners = get_corners_display(b)
        center_disp = corners.mean(axis=0)
        cam_pt = (view_mat @ np.append(center_disp, 1))[:3]
        box_items.append((corners, pred_color, -cam_pt[2], "pred"))

    # Sort by depth: far first
    box_items.sort(key=lambda x: x[2], reverse=True)

    # Draw boxes
    for corners, color, _, tag in box_items:
        pts_2d, depths = project_points(corners, view_mat, K, bev_w, bev_h)
        draw_3d_box_perspective(canvas, pts_2d, depths, color,
                                alpha=0.12, line_w=2)

    return canvas


# ============================================================================
# Main composition
# ============================================================================

def render_comparison(entry, output_path):
    img = load_image(entry["file_path"])
    if img is None:
        print(f"Cannot load: {entry['file_path']}")
        return False

    img_panel_w, img_panel_h = 480, 360
    bev_panel_w, bev_panel_h = 360, 360

    orig_w, orig_h = entry["width"], entry["height"]
    h, w = img.shape[:2]
    scale_render = min(img_panel_w / w, img_panel_h / h)
    new_w, new_h = int(w * scale_render), int(h * scale_render)
    img_resized = cv2.resize(img, (new_w, new_h))
    proj_scale = new_w / orig_w

    models = ["SAM3_3D", "GDino3D", "DetAny3D", "OVMono3D"]
    panels = []
    for m in models:
        img_panel = render_single_model(img_resized, entry, m, proj_scale)
        pad_h = img_panel_h - new_h
        pad_w = img_panel_w - new_w
        img_panel = cv2.copyMakeBorder(
            img_panel, 0, max(pad_h, 0), 0, max(pad_w, 0),
            cv2.BORDER_CONSTANT, value=(30, 30, 30),
        )
        img_panel = img_panel[:img_panel_h, :img_panel_w]

        bev_panel = render_bev_perspective(entry, m, bev_panel_w, bev_panel_h)
        cell = np.hstack([img_panel, bev_panel])
        panels.append(cell)

    top = np.hstack([panels[0], panels[1]])
    bottom = np.hstack([panels[2], panels[3]])
    grid = np.vstack([top, bottom])

    gt_cats = set(g.get("category", "") for g in entry.get("gt", []))
    title = (
        f"Image: {entry.get('file_path', '').split('/')[-1]}  |  "
        f"GT: {len(entry.get('gt', []))} objects  |  "
        f"Categories: {', '.join(sorted(gt_cats)[:5])}"
    )
    title_bar = np.full((36, grid.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    cv2.putText(title_bar, title, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    grid = np.vstack([title_bar, grid])

    legend_bar = np.full((28, grid.shape[1], 3), (30, 30, 30), dtype=np.uint8)
    x = 10
    cv2.putText(legend_bar, "GT (purple)", (x, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, GT_COLOR_BGR, 1)
    x += 130
    for m in models:
        c = MODEL_COLORS_BGR[m]
        cv2.putText(legend_bar, m, (x, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, c, 1)
        x += 160
    grid = np.vstack([grid, legend_bar])

    cv2.imwrite(output_path, grid)
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(DEMO_DATA) as f:
        data = json.load(f)

    # Only COCO images where ALL 4 models have at least 1 matched detection
    selected_indices = []
    models = ["SAM3_3D", "GDino3D", "DetAny3D", "OVMono3D"]
    for i, e in enumerate(data):
        if "coco" not in e["file_path"]:
            continue
        gt = e.get("gt", [])
        if not gt:
            continue
        all_have_dets = True
        for m in models:
            all_preds = e.get("predictions", {}).get(m, [])
            matched = filter_preds_by_gt_match(all_preds, gt, m)
            if len(matched) == 0:
                all_have_dets = False
                break
        if all_have_dets:
            selected_indices.append(i)
    print(f"Found {len(selected_indices)} COCO images where all 4 models have detections")
    selected_indices = selected_indices[::max(1, len(selected_indices) // 20)]
    selected = [data[i] for i in selected_indices if i < len(data)]

    print(f"Rendering {len(selected)} comparison images...")
    for entry in selected:
        fname = entry["file_path"].split("/")[-1].replace(".jpg", ".png")
        out_path = os.path.join(OUTPUT_DIR, fname)
        gt_count = len(entry.get("gt", []))
        print(f"  {fname} (GT={gt_count})...", end=" ", flush=True)
        ok = render_comparison(entry, out_path)
        print("OK" if ok else "FAILED")

    print(f"\nSaved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
