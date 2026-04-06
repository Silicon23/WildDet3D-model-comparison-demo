/**
 * Detail Page Logic (image.html)
 *
 * Handles per-image visualization with multi-model comparison panels,
 * score threshold filtering, display toggles, BEV rendering, and
 * keyboard navigation.
 *
 * Depends on: app.js (CONFIG, loadIndex, loadImageData, getImageUrl)
 *             overlay-renderer.js (OverlayRenderer, MODEL_COLORS)
 *             bev-renderer.js (BEVRenderer)
 */

// ============================================================================
// State
// ============================================================================

// Default per-model confidence thresholds. The UI sliders override these.
var DEFAULT_SCORE_THRESHOLDS = {
    SAM3_3D: 0.5,
    GDino3D: 0.1
};

var detailState = {
    imageId: null,
    imageData: null,
    imageList: null,
    currentIdx: -1,
    renderers: {},
    bevRenderers: {},
    loadedImage: null,
    showGT: false,
    showLabels: true,
    show3D: true,
    show2D: false,
    activeModels: new Set(['SAM3_3D', 'GDino3D']),
    filteredPreds: {},
    scoreThresholds: Object.assign({}, DEFAULT_SCORE_THRESHOLDS),
    bevElev: 35
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    initDetailPage();
});

async function initDetailPage() {
    var id = getQueryParam('id');
    if (!id) {
        showError('No image ID specified in URL');
        return;
    }

    detailState.imageId = parseInt(id);

    // Load index for navigation and per-image data in parallel
    var results = await Promise.all([
        loadIndex(),
        loadImageData(detailState.imageId)
    ]);
    var index = results[0];
    var imageData = results[1];

    detailState.imageList = index.images.filter(function (img) {
        if (img.gt_count <= 0) return false;
        var mc = img.matched_counts || img.model_counts || {};
        var total = 0;
        for (var k in mc) total += mc[k];
        return total > 0;
    });
    detailState.imageData = imageData;

    // Find current index in image list
    for (var i = 0; i < detailState.imageList.length; i++) {
        if (detailState.imageList[i].id === detailState.imageId) {
            detailState.currentIdx = i;
            break;
        }
    }

    // Find metadata from index
    var meta = detailState.imageList[detailState.currentIdx] || {};

    updateHeader(imageData, meta);
    updateNavigation();
    setupControls();
    setupKeyboardShortcuts();

    await renderAll();
}

// ============================================================================
// Error display
// ============================================================================

function showError(message) {
    var grid = document.getElementById('grid');
    if (grid) {
        grid.innerHTML =
            '<div class="card" style="grid-column:1/-1;padding:40px;text-align:center">' +
            message + '</div>';
    }
}

// ============================================================================
// Header
// ============================================================================

function updateHeader(imageData, meta) {
    var idEl = document.getElementById('image-id');
    if (idEl) {
        idEl.textContent = meta.formatted_id || imageData.image_id;
    }

    var badge = document.getElementById('source-badge');
    if (badge && meta.source) {
        badge.textContent = meta.source;
        badge.className = 'badge badge-' + meta.source;
        badge.style.display = '';
    }

    var sceneEl = document.getElementById('scene-path');
    if (sceneEl && meta.scene_path) {
        sceneEl.textContent = meta.scene_path.replace(/\//g, ' / ');
    }

    var dimEl = document.getElementById('dimensions');
    if (dimEl && imageData.width && imageData.height) {
        dimEl.textContent = imageData.width + ' x ' + imageData.height;
    }

    document.title = 'Image ' + (meta.formatted_id || imageData.image_id) +
        ' - 3D Detection Comparison';
}

// ============================================================================
// Navigation
// ============================================================================

function updateNavigation() {
    var total = detailState.imageList ? detailState.imageList.length : 0;
    var idx = detailState.currentIdx;

    var counter = document.getElementById('counter');
    if (counter) {
        counter.textContent = (idx + 1) + ' / ' + total + '  (id:' + detailState.imageId + ')';
    }

    var prevBtn = document.getElementById('btn-prev');
    var nextBtn = document.getElementById('btn-next');
    if (prevBtn) prevBtn.disabled = (idx <= 0);
    if (nextBtn) nextBtn.disabled = (idx >= total - 1);
}

function navigate(delta) {
    if (!detailState.imageList) return;

    var newIdx = detailState.currentIdx + delta;
    if (newIdx < 0 || newIdx >= detailState.imageList.length) return;

    var newImage = detailState.imageList[newIdx];
    window.location.href = 'image.html?id=' + newImage.id;
}

// ============================================================================
// Controls Setup
// ============================================================================

function setupControls() {
    // BEV elevation slider
    var elevSlider = document.getElementById('bev-elev');
    var elevDisplay = document.getElementById('elev-display');
    if (elevSlider) {
        elevSlider.value = detailState.bevElev;
        if (elevDisplay) elevDisplay.textContent = detailState.bevElev;
        elevSlider.addEventListener('input', function (e) {
            detailState.bevElev = parseInt(e.target.value);
            if (elevDisplay) elevDisplay.textContent = detailState.bevElev;
            rerenderAllCards();
        });
    }

    // Toggle checkboxes
    var gtCheck = document.getElementById('show-gt');
    var labelsCheck = document.getElementById('show-labels');
    var box3dCheck = document.getElementById('show-3d');
    var box2dCheck = document.getElementById('show-2d');

    if (gtCheck) {
        gtCheck.checked = detailState.showGT;
        gtCheck.addEventListener('change', function () {
            detailState.showGT = gtCheck.checked;
            rerenderAllCards();
        });
    }
    if (labelsCheck) {
        labelsCheck.checked = detailState.showLabels;
        labelsCheck.addEventListener('change', function () {
            detailState.showLabels = labelsCheck.checked;
            rerenderAllCards();
        });
    }
    if (box3dCheck) {
        box3dCheck.checked = detailState.show3D;
        box3dCheck.addEventListener('change', function () {
            detailState.show3D = box3dCheck.checked;
            rerenderAllCards();
        });
    }
    if (box2dCheck) {
        box2dCheck.checked = detailState.show2D;
        box2dCheck.addEventListener('change', function () {
            detailState.show2D = box2dCheck.checked;
            rerenderAllCards();
        });
    }

    // Per-model confidence-threshold sliders.
    // Slider HTML ids use display names (e.g. thr-WildDet3D) while
    // internal keys use CONFIG.MODELS (e.g. SAM3_3D).
    var SLIDER_IDS = { SAM3_3D: 'WildDet3D', GDino3D: 'GDino3D' };
    var models = CONFIG.MODELS;
    for (var mi = 0; mi < models.length; mi++) {
        (function (model) {
            var sliderId = SLIDER_IDS[model] || model;
            var slider = document.getElementById('thr-' + sliderId);
            var display = document.getElementById('thr-' + sliderId + '-display');
            if (!slider) return;
            slider.value = detailState.scoreThresholds[model];
            if (display) {
                display.textContent = Number(
                    detailState.scoreThresholds[model]
                ).toFixed(2);
            }
            slider.addEventListener('input', function (e) {
                var v = parseFloat(e.target.value);
                detailState.scoreThresholds[model] = v;
                if (display) display.textContent = v.toFixed(2);
                rerenderAllCards();
            });
        })(models[mi]);
    }

    // Model toggle buttons (if present)
    var modelToggles = document.querySelectorAll('.model-toggle');
    modelToggles.forEach(function (toggle) {
        var model = toggle.dataset.model;
        if (detailState.activeModels.has(model)) {
            toggle.classList.add('active');
        }
        toggle.addEventListener('click', function () {
            if (detailState.activeModels.has(model)) {
                detailState.activeModels.delete(model);
                toggle.classList.remove('active');
            } else {
                detailState.activeModels.add(model);
                toggle.classList.add('active');
            }
            rerenderAllCards();
        });
    });
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function (e) {
        // Skip if user is typing in an input field
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

        if (e.key === 'ArrowLeft') {
            navigate(-1);
        } else if (e.key === 'ArrowRight') {
            navigate(1);
        } else if (e.key === 'g' || e.key === 'G') {
            var gtCb = document.getElementById('show-gt');
            if (gtCb) {
                gtCb.checked = !gtCb.checked;
                gtCb.dispatchEvent(new Event('change'));
            }
        } else if (e.key === 'l' || e.key === 'L') {
            var lblCb = document.getElementById('show-labels');
            if (lblCb) {
                lblCb.checked = !lblCb.checked;
                lblCb.dispatchEvent(new Event('change'));
            }
        } else if (e.key === '3') {
            var box3Cb = document.getElementById('show-3d');
            if (box3Cb) {
                box3Cb.checked = !box3Cb.checked;
                box3Cb.dispatchEvent(new Event('change'));
            }
        } else if (e.key === '2') {
            var box2Cb = document.getElementById('show-2d');
            if (box2Cb) {
                box2Cb.checked = !box2Cb.checked;
                box2Cb.dispatchEvent(new Event('change'));
            }
        }
    });
}

// ============================================================================
// Image Loading
// ============================================================================

function loadSharedImage(filePath) {
    return new Promise(function (resolve, reject) {
        if (detailState.loadedImage) {
            resolve(detailState.loadedImage);
            return;
        }

        var url = getImageUrl(filePath);
        var img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = function () {
            detailState.loadedImage = img;
            resolve(img);
        };
        img.onerror = function () {
            reject(new Error('Failed to load image: ' + url));
        };
        img.src = url;
    });
}

// ============================================================================
// Score-Threshold Filter + Cross-Category NMS
// ============================================================================

// IoU threshold for suppressing near-duplicate boxes predicted under
// different text categories (e.g. "microwave" vs "oven" on the same
// object). Same-category NMS is already applied inside the model.
var CROSS_CAT_NMS_IOU = 0.8;

/**
 * Filter predictions by confidence score. Keeps their original predicted
 * category (no GT relabeling).
 */
function filterPredsByScore(preds, threshold) {
    if (!preds || preds.length === 0) return [];
    var out = [];
    for (var i = 0; i < preds.length; i++) {
        var s = preds[i].score;
        if (s === undefined || s === null) s = 0;
        if (s >= threshold) out.push(preds[i]);
    }
    return out;
}

/**
 * 2D IoU between two axis-aligned boxes [x1, y1, x2, y2].
 */
function iou2D(a, b) {
    var x1 = Math.max(a[0], b[0]);
    var y1 = Math.max(a[1], b[1]);
    var x2 = Math.min(a[2], b[2]);
    var y2 = Math.min(a[3], b[3]);
    var inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    var areaA = (a[2] - a[0]) * (a[3] - a[1]);
    var areaB = (b[2] - b[0]) * (b[3] - b[1]);
    var union = areaA + areaB - inter;
    return union > 0 ? inter / union : 0;
}

/**
 * Suppress near-duplicate boxes that belong to DIFFERENT text
 * categories. Sorts by score descending, then for each box removes
 * any lower-scoring box with a different category and IoU > threshold.
 */
function suppressCrossCategoryDuplicates(preds, iouThreshold) {
    if (!preds || preds.length <= 1) return preds || [];
    var sorted = preds.slice().sort(function (a, b) {
        var sa = a.score || 0, sb = b.score || 0;
        return sb - sa;
    });
    var kept = [];
    for (var i = 0; i < sorted.length; i++) {
        var p = sorted[i];
        var pBox = p.bbox2D;
        if (!pBox || pBox.length < 4) {
            kept.push(p);
            continue;
        }
        var suppressed = false;
        for (var k = 0; k < kept.length; k++) {
            var q = kept[k];
            if (q.category === p.category) continue;
            var qBox = q.bbox2D;
            if (!qBox || qBox.length < 4) continue;
            if (iou2D(pBox, qBox) > iouThreshold) {
                suppressed = true;
                break;
            }
        }
        if (!suppressed) kept.push(p);
    }
    return kept;
}

/**
 * Compute score-filtered predictions for all models using each model's
 * current threshold, then apply cross-category NMS. Stores result in
 * detailState.filteredPreds.
 */
function computeFilteredPreds(data) {
    detailState.filteredPreds = {};
    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        var model = models[i];
        var preds = (data.predictions && data.predictions[model]) || [];
        var thr = detailState.scoreThresholds[model];
        if (thr === undefined) thr = 0;
        var scoreFiltered = filterPredsByScore(preds, thr);
        detailState.filteredPreds[model] = suppressCrossCategoryDuplicates(
            scoreFiltered, CROSS_CAT_NMS_IOU
        );
    }
}

// ============================================================================
// Rendering
// ============================================================================

async function renderAll() {
    var data = detailState.imageData;
    if (!data) return;

    // Load the shared image
    var img = await loadSharedImage(data.file_path);

    // Initialize overlay renderer for GT panel
    var gtCanvas = document.getElementById('canvas-GT');
    if (gtCanvas) {
        var gtRenderer = new OverlayRenderer('canvas-GT');
        gtRenderer.image = img;
        gtRenderer._resizeCanvas();
        detailState.renderers['GT'] = gtRenderer;
    }

    // Initialize overlay renderers for each model panel
    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        var model = models[i];
        var canvasId = 'canvas-' + model;
        var canvas = document.getElementById(canvasId);
        if (canvas) {
            var renderer = new OverlayRenderer(canvasId);
            renderer.image = img;
            renderer._resizeCanvas();
            detailState.renderers[model] = renderer;
        }
    }

    // Initialize BEV renderers for GT and each model
    var bevIds = ['GT'].concat(CONFIG.MODELS);
    for (var i = 0; i < bevIds.length; i++) {
        var bevCanvasId = 'bev-' + bevIds[i];
        var bevCanvas = document.getElementById(bevCanvasId);
        if (bevCanvas && typeof BEVRenderer !== 'undefined') {
            detailState.bevRenderers[bevIds[i]] = new BEVRenderer(bevCanvasId);
        }
    }

    rerenderAllCards();
}

function rerenderAllCards() {
    var data = detailState.imageData;
    if (!data || !detailState.loadedImage) return;

    computeFilteredPreds(data);

    // Render GT-only panel
    renderGTCard(data);

    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        renderModelCard(models[i], data);
    }

    renderAllBEVs(data);
}

function renderGTCard(data) {
    var renderer = detailState.renderers['GT'];
    if (!renderer || !detailState.loadedImage) return;

    renderer.clear();

    if (data.gt && data.gt.length > 0) {
        renderer.renderBoxes(data.gt, MODEL_COLORS.GT, {
            show3D: detailState.show3D,
            show2D: detailState.show2D,
            showLabels: detailState.showLabels,
            isGT: true
        });
    }

    var countEl = document.getElementById('count-GT');
    if (countEl) {
        countEl.textContent = (data.gt ? data.gt.length : 0) + ' boxes';
    }
}

function renderModelCard(modelName, data) {
    var renderer = detailState.renderers[modelName];
    var canvas = document.getElementById('canvas-' + modelName);
    if (!canvas || !detailState.loadedImage) return;

    // If no dedicated renderer, fall back to direct canvas drawing
    if (!renderer) {
        renderCardDirect(modelName, data);
        return;
    }

    // Clear and redraw background image
    renderer.clear();

    var gtColor = MODEL_COLORS.GT;
    var modelColor = MODEL_COLORS[modelName];

    // Draw GT overlay if enabled
    if (detailState.showGT && data.gt) {
        renderer.renderBoxes(data.gt, gtColor, {
            show3D: detailState.show3D,
            show2D: detailState.show2D,
            showLabels: detailState.showLabels,
            isGT: true
        });
    }

    // Draw model predictions filtered by confidence threshold
    var filteredPreds = detailState.filteredPreds[modelName] || [];
    var visCount = filteredPreds.length;

    renderer.renderBoxes(filteredPreds, modelColor, {
        show3D: detailState.show3D,
        show2D: detailState.show2D,
        showLabels: detailState.showLabels,
        scoreThreshold: 0,
        isGT: false
    });

    // Update detection count display
    var countEl = document.getElementById('count-' + modelName);
    if (countEl) {
        countEl.textContent = visCount + ' dets';
    }
}

/**
 * Fallback rendering directly on canvas when the OverlayRenderer
 * was not initialized for a model panel (e.g. canvas not found initially).
 */
function renderCardDirect(modelName, data) {
    var canvas = document.getElementById('canvas-' + modelName);
    if (!canvas || !detailState.loadedImage) return;

    var ctx = canvas.getContext('2d');
    var img = detailState.loadedImage;

    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    ctx.drawImage(img, 0, 0);

    // Draw GT
    if (detailState.showGT && data.gt) {
        for (var g = 0; g < data.gt.length; g++) {
            var gtBox = data.gt[g];
            if (detailState.show3D && gtBox.bbox3D_proj) {
                drawBox3DRaw(ctx, gtBox.bbox3D_proj, MODEL_COLORS.GT, 4);
            }
            if (detailState.show2D && gtBox.bbox2D) {
                drawBox2DRaw(ctx, gtBox.bbox2D, MODEL_COLORS.GT, 2);
            }
            if (detailState.showLabels && gtBox.bbox2D) {
                var bx = gtBox.bbox2D[0] || 0;
                var by = gtBox.bbox2D[1] || 0;
                drawLabelRaw(ctx, gtBox.category, bx, by, MODEL_COLORS.GT);
            }
        }
    }

    // Draw score-filtered predictions
    var preds = detailState.filteredPreds[modelName] || [];
    var color = MODEL_COLORS[modelName];
    var visCount = preds.length;

    for (var p = 0; p < preds.length; p++) {
        var pred = preds[p];

        if (detailState.show3D && pred.bbox3D_proj) {
            drawBox3DRaw(ctx, pred.bbox3D_proj, color, 5);
        }
        if (detailState.show2D && pred.bbox2D) {
            drawBox2DRaw(ctx, pred.bbox2D, color, 2);
        }
        if (detailState.showLabels) {
            var lx, ly;
            if (pred.bbox3D_proj && pred.bbox3D_proj.length === 8) {
                lx = Math.min.apply(null, pred.bbox3D_proj.map(function (pt) { return pt[0]; }));
                ly = Math.min.apply(null, pred.bbox3D_proj.map(function (pt) { return pt[1]; }));
            } else if (pred.bbox2D) {
                lx = pred.bbox2D[0];
                ly = pred.bbox2D[1];
            }
            if (lx !== undefined) {
                var label = pred.category;
                drawLabelRaw(ctx, label, lx, ly, color);
            }
        }
    }

    var countEl = document.getElementById('count-' + modelName);
    if (countEl) {
        countEl.textContent = visCount + ' dets';
    }
}

// ============================================================================
// Raw Drawing Helpers (fallback when OverlayRenderer is not used)
// ============================================================================

var BOX3D_EDGES_RAW = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
];

var BOX3D_FACES_RAW = [
    [0, 1, 2, 3], [4, 5, 6, 7],
    [0, 1, 5, 4], [2, 3, 7, 6],
    [0, 3, 7, 4], [1, 2, 6, 5]
];

function drawBox3DRaw(ctx, proj, color, lineWidth) {
    if (!proj || proj.length !== 8) return;
    for (var i = 0; i < 8; i++) {
        if (!proj[i]) return;
    }
    ctx.save();

    // Filled faces
    ctx.globalAlpha = 0.18;
    ctx.fillStyle = color;
    for (var f = 0; f < BOX3D_FACES_RAW.length; f++) {
        var face = BOX3D_FACES_RAW[f];
        ctx.beginPath();
        ctx.moveTo(proj[face[0]][0], proj[face[0]][1]);
        for (var j = 1; j < face.length; j++) {
            ctx.lineTo(proj[face[j]][0], proj[face[j]][1]);
        }
        ctx.closePath();
        ctx.fill();
    }

    // Wireframe edges
    ctx.globalAlpha = 0.9;
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    for (var i = 0; i < BOX3D_EDGES_RAW.length; i++) {
        var edge = BOX3D_EDGES_RAW[i];
        ctx.beginPath();
        ctx.moveTo(proj[edge[0]][0], proj[edge[0]][1]);
        ctx.lineTo(proj[edge[1]][0], proj[edge[1]][1]);
        ctx.stroke();
    }
    ctx.globalAlpha = 1.0;
    ctx.restore();
}

function drawBox2DRaw(ctx, bbox, color, lineWidth) {
    if (!bbox || bbox.length < 4) return;
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.setLineDash([4, 3]);
    var x, y, w, h;
    if (bbox[2] > bbox[0] && bbox[3] > bbox[1] && bbox[2] > 10 && bbox[3] > 10) {
        x = bbox[0];
        y = bbox[1];
        w = bbox[2] - bbox[0];
        h = bbox[3] - bbox[1];
    } else {
        x = bbox[0];
        y = bbox[1];
        w = bbox[2];
        h = bbox[3];
    }
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
    ctx.restore();
}

function drawLabelRaw(ctx, text, x, y, color) {
    ctx.save();
    ctx.font = 'bold 11px Inter, sans-serif';
    var metrics = ctx.measureText(text);
    var pad = 3;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.8;
    ctx.fillRect(x, y - 13, metrics.width + pad * 2, 15);
    ctx.globalAlpha = 1.0;
    ctx.fillStyle = '#fff';
    ctx.fillText(text, x + pad, y - 1);
    ctx.restore();
}

// ============================================================================
// BEV Rendering
// ============================================================================

function renderAllBEVs(data) {
    var elev = detailState.bevElev;

    var showLabels = detailState.showLabels;

    // GT BEV
    var gtBev = detailState.bevRenderers['GT'];
    if (gtBev && data.gt) {
        gtBev.render(data.gt, CONFIG.MODEL_COLORS.GT, elev, showLabels);
    }

    // Model BEVs
    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        var model = models[i];
        var bev = detailState.bevRenderers[model];
        if (!bev) continue;
        var boxes = detailState.filteredPreds[model] || [];
        bev.render(boxes, CONFIG.MODEL_COLORS[model], elev, showLabels);
    }
}

// ============================================================================
// Download
// ============================================================================

var DL_SIZE = 600;

/**
 * Render overlay + BEV side-by-side and download.
 * Overlay: height=600, width preserves aspect ratio.
 * BEV: 600x600. Total: (overlayW + 600) x 600.
 */
function downloadCard(key) {
    var data = detailState.imageData;
    var img = detailState.loadedImage;
    if (!data || !img) return;

    // Overlay dimensions: height = DL_SIZE, width = aspect-matched
    var overlayH = DL_SIZE;
    var overlayW = Math.round(img.naturalWidth / img.naturalHeight * DL_SIZE);

    var outCanvas = document.createElement('canvas');
    outCanvas.width = overlayW + DL_SIZE;
    outCanvas.height = DL_SIZE;
    var outCtx = outCanvas.getContext('2d');

    // -- Left: overlay at native aspect ratio, height = 600 --
    outCtx.fillStyle = '#0f172a';
    outCtx.fillRect(0, 0, overlayW, overlayH);
    outCtx.drawImage(img, 0, 0, overlayW, overlayH);

    var drawW = overlayW;
    var drawH = overlayH;
    var offX = 0;
    var offY = 0;

    var sx = drawW / img.naturalWidth;
    var sy = drawH / img.naturalHeight;

    // Draw GT boxes if this is the GT card
    if (key === 'GT') {
        var gtColor = CONFIG.MODEL_COLORS.GT;
        var gtBoxes = data.gt || [];
        _drawBoxesOnCtx(outCtx, gtBoxes, gtColor, sx, sy, offX, offY, true);
    } else {
        // Draw GT overlay if enabled
        if (detailState.showGT && data.gt) {
            _drawBoxesOnCtx(outCtx, data.gt, CONFIG.MODEL_COLORS.GT, sx, sy, offX, offY, true);
        }
        // Draw model predictions
        var preds = detailState.filteredPreds[key] || [];
        var modelColor = CONFIG.MODEL_COLORS[key];
        _drawBoxesOnCtx(outCtx, preds, modelColor, sx, sy, offX, offY, false);
    }

    // -- Right half: BEV at 600x600 --
    // Create a temporary off-screen BEV renderer
    var bevCanvas = document.createElement('canvas');
    bevCanvas.width = DL_SIZE;
    bevCanvas.height = DL_SIZE;
    var tempBev = new BEVRenderer(null);
    tempBev.canvas = bevCanvas;
    tempBev.ctx = bevCanvas.getContext('2d');

    var boxes, color;
    if (key === 'GT') {
        boxes = data.gt || [];
        color = CONFIG.MODEL_COLORS.GT;
    } else {
        boxes = detailState.filteredPreds[key] || [];
        color = CONFIG.MODEL_COLORS[key];
    }

    // Override _resizeCanvas to use fixed size
    tempBev._resizeCanvas = function () {
        this.canvas.width = DL_SIZE;
        this.canvas.height = DL_SIZE;
    };
    tempBev.render(boxes, color, detailState.bevElev, detailState.showLabels);

    outCtx.drawImage(bevCanvas, overlayW, 0);

    // Download
    var dlNames = {
        GT: 'GT',
        SAM3_3D: 'WildDet3D',
        GDino3D: '3DMOOD',
        DetAny3D: 'DetAny3D',
        OVMono3D: 'OVMono3D'
    };
    var imageId = detailState.imageData.image_id;
    var filename = imageId + '_' + (dlNames[key] || key) + '.png';

    var link = document.createElement('a');
    link.download = filename;
    link.href = outCanvas.toDataURL('image/png');
    link.click();
}

/**
 * Draw 3D box wireframes + labels on a context at given scale/offset.
 */
function _drawBoxesOnCtx(ctx, boxes, color, sx, sy, offX, offY, isGT) {
    if (!boxes || boxes.length === 0) return;

    ctx.save();
    var lw = isGT ? 4 : 5;

    for (var b = 0; b < boxes.length; b++) {
        var box = boxes[b];

        // Draw 3D box with filled faces + wireframe
        if (detailState.show3D && box.bbox3D_proj && box.bbox3D_proj.length === 8) {
            var proj = box.bbox3D_proj;
            var valid = true;
            for (var ci = 0; ci < 8; ci++) {
                if (!proj[ci]) { valid = false; break; }
            }
            if (valid) {
                // Filled faces
                ctx.globalAlpha = 0.18;
                ctx.fillStyle = color;
                for (var f = 0; f < BOX3D_FACES_RAW.length; f++) {
                    var face = BOX3D_FACES_RAW[f];
                    ctx.beginPath();
                    ctx.moveTo(offX + proj[face[0]][0] * sx, offY + proj[face[0]][1] * sy);
                    for (var j = 1; j < face.length; j++) {
                        ctx.lineTo(offX + proj[face[j]][0] * sx, offY + proj[face[j]][1] * sy);
                    }
                    ctx.closePath();
                    ctx.fill();
                }
                // Wireframe
                ctx.globalAlpha = 0.9;
                ctx.strokeStyle = color;
                ctx.lineWidth = lw;
                for (var e = 0; e < BOX3D_EDGES_RAW.length; e++) {
                    var edge = BOX3D_EDGES_RAW[e];
                    ctx.beginPath();
                    ctx.moveTo(offX + proj[edge[0]][0] * sx, offY + proj[edge[0]][1] * sy);
                    ctx.lineTo(offX + proj[edge[1]][0] * sx, offY + proj[edge[1]][1] * sy);
                    ctx.stroke();
                }
                ctx.globalAlpha = 1.0;
            }
        }

        // Draw 2D box
        if (detailState.show2D && box.bbox2D && box.bbox2D.length >= 4) {
            ctx.setLineDash([6, 4]);
            var x1 = offX + box.bbox2D[0] * sx;
            var y1 = offY + box.bbox2D[1] * sy;
            var x2 = offX + box.bbox2D[2] * sx;
            var y2 = offY + box.bbox2D[3] * sy;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            ctx.setLineDash([]);
        }

        // Draw label
        if (detailState.showLabels) {
            var labelText = box.category || 'unknown';
            var lx, ly;
            if (box.bbox3D_proj && box.bbox3D_proj.length === 8) {
                var minPy = Infinity, minPx = 0;
                for (var j = 0; j < 8; j++) {
                    var pt = box.bbox3D_proj[j];
                    if (pt && pt[1] < minPy) {
                        minPy = pt[1];
                        minPx = pt[0];
                    }
                }
                lx = offX + minPx * sx;
                ly = offY + minPy * sy - 18;
            } else if (box.bbox2D) {
                lx = offX + box.bbox2D[0] * sx;
                ly = offY + box.bbox2D[1] * sy - 18;
            }
            if (lx !== undefined) {
                // Draw pill label
                ctx.font = '12px Inter, Arial, sans-serif';
                var tw = ctx.measureText(labelText).width;
                var pad = 4;
                ctx.globalAlpha = 0.85;
                ctx.fillStyle = color;
                ctx.fillRect(lx, ly, tw + pad * 2, 18);
                ctx.globalAlpha = 1.0;
                ctx.fillStyle = '#ffffff';
                ctx.textBaseline = 'top';
                ctx.fillText(labelText, lx + pad, ly + 3);
            }
        }
    }
    ctx.restore();
}
