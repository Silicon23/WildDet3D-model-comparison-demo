/**
 * BEVRenderer - 3D perspective BEV renderer.
 *
 * Renders 3D bounding boxes from a 35-degree elevated viewpoint using
 * manual perspective projection and painter's algorithm (no GPU needed).
 *
 * Coordinate system:
 *   OpenCV camera: X=right, Y=down, Z=forward
 *   Display:       X=right, Y=up,   Z=backward  (via x, -y, -z)
 */

var BEV_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
];

var BEV_FACES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 3, 7, 4],
    [1, 2, 6, 5]
];

class BEVRenderer {
    constructor(canvasId) {
        this.canvas = canvasId ? document.getElementById(canvasId) : null;
        this.ctx = this.canvas ? this.canvas.getContext('2d') : null;
        this.bgColor = '#f8f8f8';
    }

    /**
     * Render boxes for a single source (GT or one model).
     *
     * boxes: array of {bbox3D_cam, center_cam, dimensions, yaw, category}
     * color: hex color string
     * elevDeg: camera elevation in degrees (default 35)
     */
    render(boxes, color, elevDeg) {
        if (elevDeg === undefined) elevDeg = 35;
        this._resizeCanvas();
        var ctx = this.ctx;
        var w = this.canvas.width;
        var h = this.canvas.height;

        ctx.fillStyle = this.bgColor;
        ctx.fillRect(0, 0, w, h);

        var validBoxes = [];
        var labels = [];
        for (var i = 0; i < boxes.length; i++) {
            var corners = this._getCornersDisplay(boxes[i]);
            if (corners) {
                validBoxes.push(corners);
                labels.push(boxes[i].category || '');
            }
        }

        if (validBoxes.length === 0) {
            ctx.fillStyle = '#999';
            ctx.font = '12px Inter, Arial, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('No 3D boxes', w / 2, h / 2);
            return;
        }

        // Compute scene bounds from all corners
        var allPts = [];
        for (var i = 0; i < validBoxes.length; i++) {
            for (var j = 0; j < 8; j++) {
                allPts.push(validBoxes[i][j]);
            }
        }
        var sceneCenter = this._computeCenter(allPts);
        var sceneRange = this._computeRange(allPts, sceneCenter);

        // Camera for boxes: user-controlled elevation
        var distance = sceneRange * 1.0;
        var elev = elevDeg * Math.PI / 180;
        var eye = [
            sceneCenter[0],
            sceneCenter[1] + distance * Math.sin(elev),
            sceneCenter[2] + distance * Math.cos(elev)
        ];
        var viewMat = this._lookAt(eye, sceneCenter);

        // Smart zoom: find focal length that tightly frames all corners
        var K = this._computeSmartK(allPts, viewMat, w, h);

        // Camera for ground grid: fixed 35-degree elevation
        var gridElev = 35 * Math.PI / 180;
        var gridEye = [
            sceneCenter[0],
            sceneCenter[1] + distance * Math.sin(gridElev),
            sceneCenter[2] + distance * Math.cos(gridElev)
        ];
        var gridViewMat = this._lookAt(gridEye, sceneCenter);
        var gridK = this._computeSmartK(allPts, gridViewMat, w, h);

        // Draw ground grid with fixed camera
        var groundY = this._findGroundY(validBoxes);
        this._drawGrid(gridViewMat, gridK, w, h, sceneCenter, sceneRange, groundY);

        // Sort boxes by depth (painter's algorithm: far first)
        var boxItems = [];
        for (var i = 0; i < validBoxes.length; i++) {
            var center = this._computeCenter(validBoxes[i]);
            var camPt = this._transformPoint(center, viewMat);
            boxItems.push({ corners: validBoxes[i], depth: -camPt[2], label: labels[i] });
        }
        boxItems.sort(function (a, b) { return b.depth - a.depth; });

        // Draw boxes then labels using user-controlled camera
        for (var i = 0; i < boxItems.length; i++) {
            this._drawBox3D(boxItems[i].corners, viewMat, K, w, h, color);
        }
        for (var i = 0; i < boxItems.length; i++) {
            if (boxItems[i].label) {
                this._drawLabel(boxItems[i].corners, boxItems[i].label, viewMat, K, w, h, color);
            }
        }
    }

    /**
     * Compute intrinsics K that tightly frames all 3D points within the canvas,
     * leaving a small margin (10% on each side).
     */
    _computeSmartK(allPts, viewMat, w, h) {
        var margin = 0.10;

        // Transform all points to camera space
        var camXs = [];
        var camYs = [];
        var camDepths = [];
        for (var i = 0; i < allPts.length; i++) {
            var cam = this._transformPoint(allPts[i], viewMat);
            var depth = -cam[2];
            if (depth <= 0.01) continue;
            camXs.push(cam[0] / depth);
            camYs.push(-cam[1] / depth);
            camDepths.push(depth);
        }

        if (camXs.length === 0) {
            return [[w * 0.85, 0, w / 2], [0, w * 0.85, h / 2], [0, 0, 1]];
        }

        // Find the range of normalized image coords
        var minNx = Math.min.apply(null, camXs);
        var maxNx = Math.max.apply(null, camXs);
        var minNy = Math.min.apply(null, camYs);
        var maxNy = Math.max.apply(null, camYs);

        var rangeNx = maxNx - minNx;
        var rangeNy = maxNy - minNy;

        // Prevent division by zero for single-point scenes
        if (rangeNx < 1e-6) rangeNx = 1e-6;
        if (rangeNy < 1e-6) rangeNy = 1e-6;

        // Compute fx/fy so the range fills canvas with margin
        var usableW = w * (1 - 2 * margin);
        var usableH = h * (1 - 2 * margin);
        var fxCandidate = usableW / rangeNx;
        var fyCandidate = usableH / rangeNy;

        // Use the smaller to ensure everything fits
        var f = Math.min(fxCandidate, fyCandidate);

        // Center the projection on the midpoint of the scene
        var midNx = (minNx + maxNx) / 2;
        var midNy = (minNy + maxNy) / 2;
        var cx = w / 2 - f * midNx;
        var cy = h / 2 - f * midNy;

        return [[f, 0, cx], [0, f, cy], [0, 0, 1]];
    }

    _resizeCanvas() {
        var container = this.canvas.parentElement;
        var cw = container.clientWidth;
        var ch = container.clientHeight;
        if (cw <= 0) cw = 300;
        if (ch <= 0) ch = 300;
        this.canvas.width = cw;
        this.canvas.height = ch;
    }

    /**
     * Get 8 corners in display coords (x, -y, -z) from a box object.
     * Uses bbox3D_cam if available, falls back to axis-aligned from center+dims.
     */
    _getCornersDisplay(box) {
        if (box.bbox3D_cam && box.bbox3D_cam.length === 8) {
            var corners = [];
            for (var i = 0; i < 8; i++) {
                var p = box.bbox3D_cam[i];
                corners.push([p[0], -p[1], -p[2]]);
            }
            return corners;
        }

        if (!box.center_cam || !box.dimensions) return null;

        var cx = box.center_cam[0];
        var cy = box.center_cam[1];
        var cz = box.center_cam[2];
        var bw = box.dimensions[0] / 2;
        var bh = box.dimensions[1] / 2;
        var bl = box.dimensions[2] / 2;

        var corners = [
            [cx - bw, cy - bh, cz - bl],
            [cx + bw, cy - bh, cz - bl],
            [cx + bw, cy + bh, cz - bl],
            [cx - bw, cy + bh, cz - bl],
            [cx - bw, cy - bh, cz + bl],
            [cx + bw, cy - bh, cz + bl],
            [cx + bw, cy + bh, cz + bl],
            [cx - bw, cy + bh, cz + bl]
        ];
        for (var i = 0; i < 8; i++) {
            corners[i] = [corners[i][0], -corners[i][1], -corners[i][2]];
        }
        return corners;
    }

    _computeCenter(pts) {
        var sx = 0, sy = 0, sz = 0;
        for (var i = 0; i < pts.length; i++) {
            sx += pts[i][0]; sy += pts[i][1]; sz += pts[i][2];
        }
        var n = pts.length;
        return [sx / n, sy / n, sz / n];
    }

    _computeRange(pts, center) {
        var maxDist = 0;
        for (var i = 0; i < pts.length; i++) {
            var dx = pts[i][0] - center[0];
            var dy = pts[i][1] - center[1];
            var dz = pts[i][2] - center[2];
            var d = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (d > maxDist) maxDist = d;
        }
        return Math.max(maxDist * 2, 2.0);
    }

    _findGroundY(allCorners) {
        var minY = Infinity;
        for (var i = 0; i < allCorners.length; i++) {
            for (var j = 0; j < 8; j++) {
                if (allCorners[i][j][1] < minY) minY = allCorners[i][j][1];
            }
        }
        return minY;
    }

    /**
     * Compute 4x4 look-at view matrix.
     */
    _lookAt(eye, target) {
        var fwd = [target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]];
        var fLen = Math.sqrt(fwd[0] * fwd[0] + fwd[1] * fwd[1] + fwd[2] * fwd[2]);
        fwd = [fwd[0] / fLen, fwd[1] / fLen, fwd[2] / fLen];

        var up = [0, 1, 0];
        var right = this._cross(fwd, up);
        var rLen = Math.sqrt(right[0] * right[0] + right[1] * right[1] + right[2] * right[2]);
        right = [right[0] / rLen, right[1] / rLen, right[2] / rLen];

        var trueUp = this._cross(right, fwd);

        var m = new Float64Array(16);
        m[0] = right[0];   m[1] = right[1];   m[2] = right[2];
        m[3] = -(right[0] * eye[0] + right[1] * eye[1] + right[2] * eye[2]);
        m[4] = trueUp[0];  m[5] = trueUp[1];  m[6] = trueUp[2];
        m[7] = -(trueUp[0] * eye[0] + trueUp[1] * eye[1] + trueUp[2] * eye[2]);
        m[8] = -fwd[0];    m[9] = -fwd[1];    m[10] = -fwd[2];
        m[11] = -(-fwd[0] * eye[0] + -fwd[1] * eye[1] + -fwd[2] * eye[2]);
        m[12] = 0; m[13] = 0; m[14] = 0; m[15] = 1;
        return m;
    }

    _cross(a, b) {
        return [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        ];
    }

    _transformPoint(pt, mat) {
        return [
            mat[0] * pt[0] + mat[1] * pt[1] + mat[2] * pt[2] + mat[3],
            mat[4] * pt[0] + mat[5] * pt[1] + mat[6] * pt[2] + mat[7],
            mat[8] * pt[0] + mat[9] * pt[1] + mat[10] * pt[2] + mat[11]
        ];
    }

    /**
     * Project a 3D point (display coords) to 2D pixel + depth.
     */
    _project(pt, viewMat, K) {
        var cam = this._transformPoint(pt, viewMat);
        var depth = -cam[2];
        if (depth <= 0.01) return null;
        var x = K[0][0] * cam[0] / depth + K[0][2];
        var y = -K[1][1] * cam[1] / depth + K[1][2];
        return { x: x, y: y, depth: depth };
    }

    _drawGrid(viewMat, K, w, h, center, range, groundY) {
        var ctx = this.ctx;
        var half = range * 0.6;
        var spacing = Math.max(0.5, range / 5);

        ctx.save();
        ctx.strokeStyle = '#d0d0d0';
        ctx.lineWidth = 1;

        var zStart = Math.ceil((center[2] - half) / spacing) * spacing;
        for (var z = zStart; z <= center[2] + half; z += spacing) {
            var p1 = this._project([center[0] - half, groundY, z], viewMat, K);
            var p2 = this._project([center[0] + half, groundY, z], viewMat, K);
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }
        }
        var xStart = Math.ceil((center[0] - half) / spacing) * spacing;
        for (var x = xStart; x <= center[0] + half; x += spacing) {
            var p1 = this._project([x, groundY, center[2] - half], viewMat, K);
            var p2 = this._project([x, groundY, center[2] + half], viewMat, K);
            if (p1 && p2) {
                ctx.beginPath();
                ctx.moveTo(p1.x, p1.y);
                ctx.lineTo(p2.x, p2.y);
                ctx.stroke();
            }
        }
        ctx.restore();
    }

    /**
     * Draw one 3D box with semi-transparent faces + wireframe edges.
     */
    _drawBox3D(corners, viewMat, K, w, h, color) {
        var ctx = this.ctx;

        var pts2d = [];
        var depths = [];
        for (var i = 0; i < 8; i++) {
            var p = this._project(corners[i], viewMat, K);
            if (!p) return;
            pts2d.push(p);
            depths.push(p.depth);
        }

        // Sort faces by average depth (far first)
        var faceDepths = [];
        for (var f = 0; f < BEV_FACES.length; f++) {
            var face = BEV_FACES[f];
            var avg = 0;
            for (var j = 0; j < face.length; j++) avg += depths[face[j]];
            faceDepths.push({ idx: f, d: avg / face.length });
        }
        faceDepths.sort(function (a, b) { return b.d - a.d; });

        // Draw filled faces
        ctx.save();
        ctx.globalAlpha = 0.12;
        ctx.fillStyle = color;
        for (var fi = 0; fi < faceDepths.length; fi++) {
            var face = BEV_FACES[faceDepths[fi].idx];
            ctx.beginPath();
            ctx.moveTo(pts2d[face[0]].x, pts2d[face[0]].y);
            for (var j = 1; j < face.length; j++) {
                ctx.lineTo(pts2d[face[j]].x, pts2d[face[j]].y);
            }
            ctx.closePath();
            ctx.fill();
        }
        ctx.globalAlpha = 1.0;

        // Draw edges
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        for (var e = 0; e < BEV_EDGES.length; e++) {
            var i0 = BEV_EDGES[e][0];
            var i1 = BEV_EDGES[e][1];
            ctx.beginPath();
            ctx.moveTo(pts2d[i0].x, pts2d[i0].y);
            ctx.lineTo(pts2d[i1].x, pts2d[i1].y);
            ctx.stroke();
        }
        ctx.restore();
    }

    /**
     * Draw a category label above a box (at the topmost projected point).
     */
    _drawLabel(corners, label, viewMat, K, w, h, color) {
        var ctx = this.ctx;

        // Find topmost projected point
        var minY = Infinity;
        var labelX = 0;
        for (var i = 0; i < 8; i++) {
            var p = this._project(corners[i], viewMat, K);
            if (!p) return;
            if (p.y < minY) {
                minY = p.y;
                labelX = p.x;
            }
        }

        var fontSize = 10;
        var padH = 3;
        var padV = 2;

        ctx.save();
        ctx.font = fontSize + 'px Inter, Arial, sans-serif';
        var textW = ctx.measureText(label).width;
        var pillW = textW + padH * 2;
        var pillH = fontSize + padV * 2;
        var px = labelX - pillW / 2;
        var py = minY - pillH - 3;

        // Clamp to canvas
        if (px < 2) px = 2;
        if (px + pillW > w - 2) px = w - 2 - pillW;
        if (py < 2) py = 2;

        // Background pill
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = color;
        ctx.beginPath();
        var r = 2;
        ctx.moveTo(px + r, py);
        ctx.lineTo(px + pillW - r, py);
        ctx.arcTo(px + pillW, py, px + pillW, py + r, r);
        ctx.lineTo(px + pillW, py + pillH - r);
        ctx.arcTo(px + pillW, py + pillH, px + pillW - r, py + pillH, r);
        ctx.lineTo(px + r, py + pillH);
        ctx.arcTo(px, py + pillH, px, py + pillH - r, r);
        ctx.lineTo(px, py + r);
        ctx.arcTo(px, py, px + r, py, r);
        ctx.closePath();
        ctx.fill();

        // Text
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#ffffff';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(label, px + padH, py + padV);
        ctx.restore();
    }

    _hexToRgb(hex) {
        var r = parseInt(hex.slice(1, 3), 16);
        var g = parseInt(hex.slice(3, 5), 16);
        var b = parseInt(hex.slice(5, 7), 16);
        return [r, g, b];
    }
}
