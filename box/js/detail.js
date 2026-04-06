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

var detailState = {
    imageId: null,
    imageData: null,
    imageList: null,
    currentIdx: -1,
    renderers: {},
    bevRenderers: {},
    loadedImage: null,
    showGT: false,
    showLabels: false,
    show3D: true,
    show2D: false,
    activeModels: new Set(['SAM3_3D', 'DetAny3D', 'OVMono3D']),
    matchedPreds: {},
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
// GT-Matching Filter
// ============================================================================

/**
 * Compute 2D IoU between two axis-aligned boxes [x1, y1, x2, y2].
 */
function computeIoU2D(boxA, boxB) {
    var x1 = Math.max(boxA[0], boxB[0]);
    var y1 = Math.max(boxA[1], boxB[1]);
    var x2 = Math.min(boxA[2], boxB[2]);
    var y2 = Math.min(boxA[3], boxB[3]);

    var interArea = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    var areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    var areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    var union = areaA + areaB - interArea;

    return union > 0 ? interArea / union : 0;
}

/**
 * Convert 8 projected 3D corners to an axis-aligned 2D bounding box.
 */
function bbox3DProjToRect(proj) {
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var i = 0; i < proj.length; i++) {
        if (proj[i][0] < minX) minX = proj[i][0];
        if (proj[i][1] < minY) minY = proj[i][1];
        if (proj[i][0] > maxX) maxX = proj[i][0];
        if (proj[i][1] > maxY) maxY = proj[i][1];
    }
    return [minX, minY, maxX, maxY];
}

/**
 * Filter predictions by matching against GT boxes.
 *
 * For each GT box, find the prediction whose projected 3D box has
 * IoU > 0.5 with the GT 2D box and the highest confidence score.
 * Each GT keeps at most one prediction.
 */
function filterPredsByGTMatch(preds, gtBoxes) {
    if (!gtBoxes || gtBoxes.length === 0 || !preds || preds.length === 0) return [];

    // Map: pred index -> GT category (from best-matching GT)
    var matchedGtCategory = {};

    for (var g = 0; g < gtBoxes.length; g++) {
        var gtBox2D = gtBoxes[g].bbox2D;
        if (!gtBox2D || gtBox2D.length < 4) continue;

        var bestIdx = -1;
        var bestScore = -Infinity;

        for (var p = 0; p < preds.length; p++) {
            var pred2D;
            if (preds[p].bbox3D_proj && preds[p].bbox3D_proj.length === 8) {
                pred2D = bbox3DProjToRect(preds[p].bbox3D_proj);
            } else if (preds[p].bbox2D && preds[p].bbox2D.length >= 4) {
                pred2D = preds[p].bbox2D;
            } else {
                continue;
            }

            var iou = computeIoU2D(pred2D, gtBox2D);
            if (iou >= 0.2 && preds[p].score > bestScore) {
                bestIdx = p;
                bestScore = preds[p].score;
            }
        }

        if (bestIdx >= 0) {
            matchedGtCategory[bestIdx] = gtBoxes[g].category || 'unknown';
        }
    }

    var result = [];
    var indices = Object.keys(matchedGtCategory);
    for (var i = 0; i < indices.length; i++) {
        var idx = parseInt(indices[i]);
        var pred = {};
        for (var key in preds[idx]) pred[key] = preds[idx][key];
        pred.category = matchedGtCategory[idx];
        result.push(pred);
    }
    return result;
}

/**
 * Assign GT category names to predictions based on best IoU match.
 * Keeps all predictions but relabels those that match a GT box.
 */
function assignGTCategories(preds, gtBoxes) {
    if (!preds || preds.length === 0) return [];
    if (!gtBoxes || gtBoxes.length === 0) return preds;

    var result = [];
    for (var p = 0; p < preds.length; p++) {
        var pred = {};
        for (var key in preds[p]) pred[key] = preds[p][key];

        var pred2D;
        if (pred.bbox3D_proj && pred.bbox3D_proj.length === 8) {
            pred2D = bbox3DProjToRect(pred.bbox3D_proj);
        } else if (pred.bbox2D && pred.bbox2D.length >= 4) {
            pred2D = pred.bbox2D;
        }

        if (pred2D) {
            var bestIoU = 0;
            var bestCat = null;
            for (var g = 0; g < gtBoxes.length; g++) {
                var gtBox2D = gtBoxes[g].bbox2D;
                if (!gtBox2D || gtBox2D.length < 4) continue;
                var iou = computeIoU2D(pred2D, gtBox2D);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    bestCat = gtBoxes[g].category;
                }
            }
            if (bestCat && bestIoU >= 0.3) {
                pred.category = bestCat;
            }
        }
        result.push(pred);
    }
    return result;
}

/**
 * Compute matched predictions for all models against the GT.
 * Stores result in detailState.matchedPreds.
 */
function computeAllMatchedPreds(data) {
    detailState.matchedPreds = {};
    var gt = data.gt || [];
    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        var model = models[i];
        var preds = (data.predictions && data.predictions[model]) || [];
        // All models are box-prompted (oracle mode), no filtering needed
        // Category names are correct from data preparation
        detailState.matchedPreds[model] = preds;
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

    // Initialize overlay renderer for GT 2D prompt panel
    var gt2dCanvas = document.getElementById('canvas-GT-2D');
    if (gt2dCanvas) {
        var gt2dRenderer = new OverlayRenderer('canvas-GT-2D');
        gt2dRenderer.image = img;
        gt2dRenderer._resizeCanvas();
        detailState.renderers['GT-2D'] = gt2dRenderer;
    }

    // Initialize overlay renderer for GT 3D panel
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

    computeAllMatchedPreds(data);

    // Render GT 2D prompts panel
    renderGT2DCard(data);

    // Render GT 3D panel
    renderGTCard(data);

    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        renderModelCard(models[i], data);
    }

    renderAllBEVs(data);
}

function renderGT2DCard(data) {
    var renderer = detailState.renderers['GT-2D'];
    if (!renderer || !detailState.loadedImage) return;

    renderer.clear();

    if (data.gt && data.gt.length > 0) {
        renderer.renderBoxes(data.gt, MODEL_COLORS.GT, {
            show3D: false,
            show2D: true,
            showLabels: detailState.showLabels,
            isGT: true,
            solidBox: true
        });
    }

    var countEl = document.getElementById('count-GT-2D');
    if (countEl) {
        countEl.textContent = (data.gt ? data.gt.length : 0) + ' prompts';
    }
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

    // Draw model predictions filtered by GT matching
    var filteredPreds = detailState.matchedPreds[modelName] || [];
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

    // Draw GT-matched predictions
    var preds = detailState.matchedPreds[modelName] || [];
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
        var boxes = detailState.matchedPreds[model] || [];
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

    // GT-2D: overlay only (no BEV)
    if (key === 'GT-2D') {
        var outCanvas = document.createElement('canvas');
        outCanvas.width = overlayW;
        outCanvas.height = overlayH;
        var outCtx = outCanvas.getContext('2d');
        outCtx.fillStyle = '#0f172a';
        outCtx.fillRect(0, 0, overlayW, overlayH);
        outCtx.drawImage(img, 0, 0, overlayW, overlayH);

        var sx = overlayW / img.naturalWidth;
        var sy = overlayH / img.naturalHeight;
        var gtBoxes = data.gt || [];
        var color = CONFIG.MODEL_COLORS.GT;

        for (var b = 0; b < gtBoxes.length; b++) {
            var box = gtBoxes[b];
            if (!box.bbox2D || box.bbox2D.length < 4) continue;
            var x1 = box.bbox2D[0] * sx;
            var y1 = box.bbox2D[1] * sy;
            var x2 = box.bbox2D[2] * sx;
            var y2 = box.bbox2D[3] * sy;
            // Shaded fill
            outCtx.globalAlpha = 0.18;
            outCtx.fillStyle = color;
            outCtx.fillRect(x1, y1, x2 - x1, y2 - y1);
            // Solid stroke
            outCtx.globalAlpha = 0.9;
            outCtx.strokeStyle = color;
            outCtx.lineWidth = 4;
            outCtx.setLineDash([]);
            outCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);
            outCtx.globalAlpha = 1.0;
            // Label
            if (detailState.showLabels && box.category) {
                outCtx.font = '12px Inter, Arial, sans-serif';
                var tw = outCtx.measureText(box.category).width;
                var pad = 4;
                outCtx.globalAlpha = 0.85;
                outCtx.fillStyle = color;
                outCtx.fillRect(x1, y1 - 20, tw + pad * 2, 18);
                outCtx.globalAlpha = 1.0;
                outCtx.fillStyle = '#ffffff';
                outCtx.textBaseline = 'top';
                outCtx.fillText(box.category, x1 + pad, y1 - 17);
            }
        }

        var imageId = detailState.imageData.image_id;
        var link = document.createElement('a');
        link.download = imageId + '_2D_prompts.png';
        link.href = outCanvas.toDataURL('image/png');
        link.click();
        return;
    }

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
        var preds = detailState.matchedPreds[key] || [];
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
        boxes = detailState.matchedPreds[key] || [];
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
        'GT-2D': '2D_prompts',
        GT: 'GT',
        SAM3_3D: 'WildDet3D',
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
