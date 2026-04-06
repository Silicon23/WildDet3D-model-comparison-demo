/**
 * OverlayRenderer - Renders 3D bounding box wireframes and 2D boxes
 * on a canvas overlaid on an image.
 *
 * Expects pre-projected 2D coordinates (bbox3D_proj: array of 8 [u,v] points),
 * so no matrix math is needed.
 */

const MODEL_COLORS = {
    SAM3_3D: '#e74c3c',
    GDino3D: '#3b82f6',
    DetAny3D: '#22c55e',
    OVMono3D: '#f97316',
    GT: '#a855f7'
};

// 12 edges connecting the 8 corners of a 3D bounding box
const BOX3D_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],  // bottom face
    [4, 5], [5, 6], [6, 7], [7, 4],  // top face
    [0, 4], [1, 5], [2, 6], [3, 7]   // vertical edges
];

// 6 faces as quads (indices into 8 corners)
const BOX3D_FACES = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [2, 3, 7, 6],
    [0, 3, 7, 4],
    [1, 2, 6, 5]
];

class OverlayRenderer {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.image = null;
        this.scaleX = 1;
        this.scaleY = 1;
    }

    /**
     * Load image and resize canvas to fit container while maintaining
     * aspect ratio. Returns a Promise that resolves when the image is loaded.
     */
    loadImage(imagePath) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => {
                this.image = img;
                this._resizeCanvas();
                resolve();
            };
            img.onerror = () => {
                reject(new Error('Failed to load image: ' + imagePath));
            };
            img.src = imagePath;
        });
    }

    /**
     * Resize the canvas to fit its parent container while keeping
     * the image aspect ratio. Update scale factors accordingly.
     */
    _resizeCanvas() {
        if (!this.image) return;

        const container = this.canvas.parentElement;
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;

        const imgAspect = this.image.naturalWidth / this.image.naturalHeight;
        const containerAspect = containerWidth / containerHeight;

        let drawWidth, drawHeight;
        if (imgAspect > containerAspect) {
            // Image is wider than container -- fit to width
            drawWidth = containerWidth;
            drawHeight = containerWidth / imgAspect;
        } else {
            // Image is taller than container -- fit to height
            drawHeight = containerHeight;
            drawWidth = containerHeight * imgAspect;
        }

        this.canvas.width = drawWidth;
        this.canvas.height = drawHeight;

        this.scaleX = drawWidth / this.image.naturalWidth;
        this.scaleY = drawHeight / this.image.naturalHeight;
    }

    /**
     * Clear the canvas and redraw the background image.
     */
    clear() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        if (this.image) {
            this.ctx.drawImage(this.image, 0, 0, this.canvas.width, this.canvas.height);
        }
    }

    /**
     * Draw a pre-projected 3D bounding box wireframe (12 edges).
     * bbox3D_proj: array of 8 [u, v] points in original image coordinates.
     */
    draw3DBox(bbox3D_proj, color, lineWidth) {
        if (!bbox3D_proj || bbox3D_proj.length < 8) return;

        lineWidth = lineWidth || 3;
        var ctx = this.ctx;
        var sx = this.scaleX;
        var sy = this.scaleY;

        // Scaled points
        var pts = [];
        for (var i = 0; i < 8; i++) {
            var p = bbox3D_proj[i];
            if (!p) return;
            pts.push([p[0] * sx, p[1] * sy]);
        }

        ctx.save();

        // Draw semi-transparent filled faces
        ctx.globalAlpha = 0.18;
        ctx.fillStyle = color;
        for (var f = 0; f < BOX3D_FACES.length; f++) {
            var face = BOX3D_FACES[f];
            ctx.beginPath();
            ctx.moveTo(pts[face[0]][0], pts[face[0]][1]);
            for (var j = 1; j < face.length; j++) {
                ctx.lineTo(pts[face[j]][0], pts[face[j]][1]);
            }
            ctx.closePath();
            ctx.fill();
        }

        // Draw wireframe edges
        ctx.globalAlpha = 0.9;
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash([]);

        for (var i = 0; i < BOX3D_EDGES.length; i++) {
            var edge = BOX3D_EDGES[i];
            ctx.beginPath();
            ctx.moveTo(pts[edge[0]][0], pts[edge[0]][1]);
            ctx.lineTo(pts[edge[1]][0], pts[edge[1]][1]);
            ctx.stroke();
        }

        ctx.restore();
    }

    /**
     * Draw a 2D bounding box with dashed lines.
     * bbox2D: [x1, y1, x2, y2] in original image coordinates.
     */
    draw2DBox(bbox2D, color, lineWidth) {
        if (!bbox2D || bbox2D.length < 4) return;

        lineWidth = lineWidth || 1.5;
        var ctx = this.ctx;
        var sx = this.scaleX;
        var sy = this.scaleY;

        var x1 = bbox2D[0] * sx;
        var y1 = bbox2D[1] * sy;
        var x2 = bbox2D[2] * sx;
        var y2 = bbox2D[3] * sy;

        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = lineWidth;
        ctx.setLineDash([6, 4]);
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.restore();
    }

    /**
     * Draw a label with a background pill at (x, y) in canvas coordinates.
     */
    drawLabel(text, x, y, color) {
        var ctx = this.ctx;
        var fontSize = 11;
        var paddingH = 5;
        var paddingV = 3;

        ctx.save();
        ctx.font = fontSize + 'px Inter, Arial, sans-serif';

        var textMetrics = ctx.measureText(text);
        var textWidth = textMetrics.width;
        var pillWidth = textWidth + paddingH * 2;
        var pillHeight = fontSize + paddingV * 2;

        // Clamp to canvas bounds
        var pillX = Math.max(0, Math.min(x, this.canvas.width - pillWidth));
        var pillY = Math.max(0, Math.min(y, this.canvas.height - pillHeight));

        // Draw pill background
        var radius = 3;
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(pillX + radius, pillY);
        ctx.lineTo(pillX + pillWidth - radius, pillY);
        ctx.arcTo(pillX + pillWidth, pillY, pillX + pillWidth, pillY + radius, radius);
        ctx.lineTo(pillX + pillWidth, pillY + pillHeight - radius);
        ctx.arcTo(pillX + pillWidth, pillY + pillHeight, pillX + pillWidth - radius, pillY + pillHeight, radius);
        ctx.lineTo(pillX + radius, pillY + pillHeight);
        ctx.arcTo(pillX, pillY + pillHeight, pillX, pillY + pillHeight - radius, radius);
        ctx.lineTo(pillX, pillY + radius);
        ctx.arcTo(pillX, pillY, pillX + radius, pillY, radius);
        ctx.closePath();
        ctx.fill();

        // Draw text
        ctx.globalAlpha = 1.0;
        ctx.fillStyle = '#ffffff';
        ctx.textBaseline = 'top';
        ctx.fillText(text, pillX + paddingH, pillY + paddingV);

        ctx.restore();
    }

    /**
     * Render the full scene: background image + optional GT + model predictions.
     *
     * boxes: array of {bbox3D_proj, bbox2D, category, score}
     * color: string color for the boxes
     * options: {show3D, show2D, showLabels, showGT, scoreThreshold, isGT}
     */
    renderBoxes(boxes, color, options) {
        if (!boxes || boxes.length === 0) return;

        var show3D = options.show3D !== false;
        var show2D = options.show2D !== false;
        var showLabels = options.showLabels !== false;
        var scoreThreshold = options.scoreThreshold || 0;
        var isGT = options.isGT || false;

        for (var i = 0; i < boxes.length; i++) {
            var box = boxes[i];

            // Filter by score (GT boxes have no score -- always show)
            if (!isGT && box.score !== undefined && box.score !== null) {
                if (box.score < scoreThreshold) continue;
            }

            // Draw 3D wireframe
            if (show3D && box.bbox3D_proj) {
                this.draw3DBox(box.bbox3D_proj, color, isGT ? 4 : 5);
            }

            // Draw 2D box
            if (show2D && box.bbox2D) {
                this.draw2DBox(box.bbox2D, color, isGT ? 1 : 1.5);
            }

            // Draw label
            if (showLabels) {
                var labelText = box.category || 'unknown';

                // Position label at top-left of 2D box or first 3D corner
                var labelX = 0;
                var labelY = 0;
                if (box.bbox2D && box.bbox2D.length >= 4) {
                    labelX = box.bbox2D[0] * this.scaleX;
                    labelY = box.bbox2D[1] * this.scaleY - 18;
                } else if (box.bbox3D_proj && box.bbox3D_proj.length > 0) {
                    // Use topmost point from the projected 3D corners
                    var minY = Infinity;
                    var minYx = 0;
                    for (var j = 0; j < box.bbox3D_proj.length; j++) {
                        var pt = box.bbox3D_proj[j];
                        if (pt && pt[1] < minY) {
                            minY = pt[1];
                            minYx = pt[0];
                        }
                    }
                    labelX = minYx * this.scaleX;
                    labelY = minY * this.scaleY - 18;
                }

                this.drawLabel(labelText, labelX, labelY, color);
            }
        }
    }
}
