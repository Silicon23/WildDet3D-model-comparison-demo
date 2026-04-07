/**
 * Gallery Page Logic (index.html)
 *
 * Handles image browsing with scene tree filtering, dataset filters,
 * search, pagination, and thumbnail grid rendering.
 *
 * Depends on: app.js (CONFIG, loadIndex, openImage)
 *             scene-tree.js (SceneTree)
 */

// ============================================================================
// State
// ============================================================================

var galleryState = {
    images: [],
    filteredImages: [],
    currentPage: 1,
    selectedScene: null,
    datasetFilter: 'all',
    requireAllModels: false,
    searchQuery: '',
    index: null,
    sceneTree: null
};

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', function () {
    initGallery();
});

async function initGallery() {
    galleryState.index = await loadIndex();
    galleryState.images = galleryState.index.images.filter(function (img) {
        if (img.gt_count <= 0) return false;
        var mc = img.matched_counts || img.model_counts || {};
        // Exclude images where WildDet3D has fewer detections than GT
        if (!mc.SAM3_3D || mc.SAM3_3D < img.gt_count) return false;
        var total = 0;
        for (var k in mc) total += mc[k];
        return total > 0;
    });

    // Shuffle using Fisher-Yates
    for (var i = galleryState.images.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var tmp = galleryState.images[i];
        galleryState.images[i] = galleryState.images[j];
        galleryState.images[j] = tmp;
    }

    galleryState.filteredImages = galleryState.images.slice();

    updateStats();
    initSceneTree();
    initDatasetFilter();
    initAllModelsFilter();
    initSearch();
    renderImageGrid();
}

// ============================================================================
// Stats
// ============================================================================

function updateStats() {
    var total = galleryState.index.total_images;
    var filtered = galleryState.filteredImages.length;
    var scenes = Object.keys(galleryState.index.images_by_scene || {}).length;

    var statTotal = document.getElementById('stat-total');
    var statFiltered = document.getElementById('stat-with-dets');
    var statScenes = document.getElementById('stat-categories');
    var statLabel = document.getElementById('stat-with-dets-label');

    if (statTotal) statTotal.textContent = total.toLocaleString();
    if (statFiltered) statFiltered.textContent = filtered.toLocaleString();
    if (statScenes) statScenes.textContent = scenes.toLocaleString();
    if (statLabel) {
        statLabel.innerHTML = galleryState.requireAllModels
            ? 'With Detection(s)<br>From All Models'
            : 'With GT 3D Box(es)<br><span style="visibility:hidden">From All Models</span>';
    }
}

// ============================================================================
// Scene Tree
// ============================================================================

function initSceneTree() {
    var container = document.getElementById('tree-container');
    if (!container) return;

    galleryState.sceneTree = new SceneTree('tree-container', onSceneSelect);
    galleryState.sceneTree.render(galleryState.index.scene_tree);

    var collapseBtn = document.getElementById('btn-collapse-all');
    if (collapseBtn) {
        collapseBtn.addEventListener('click', function () {
            galleryState.sceneTree.collapseAll();
        });
    }
}

function onSceneSelect(scenePath) {
    galleryState.selectedScene = scenePath;
    galleryState.currentPage = 1;

    var title = scenePath
        ? scenePath.split('/').pop().replace(/_/g, ' ')
        : 'All Images';
    var contentTitle = document.getElementById('content-title');
    if (contentTitle) contentTitle.textContent = title;

    filterImages();
}

// ============================================================================
// Dataset Filter
// ============================================================================

function initDatasetFilter() {
    var buttons = document.querySelectorAll('.filter-btn');
    buttons.forEach(function (btn) {
        btn.addEventListener('click', function () {
            buttons.forEach(function (b) { b.classList.remove('active'); });
            btn.classList.add('active');
            galleryState.datasetFilter = btn.dataset.filter;
            galleryState.currentPage = 1;
            filterImages();
        });
    });
}

// ============================================================================
// All Models Filter
// ============================================================================

function initAllModelsFilter() {
    var checkbox = document.getElementById('require-all-models');
    if (!checkbox) return;

    checkbox.addEventListener('change', function () {
        galleryState.requireAllModels = checkbox.checked;
        galleryState.currentPage = 1;
        filterImages();
    });
}

// ============================================================================
// Search
// ============================================================================

function initSearch() {
    var input = document.getElementById('search-input');
    if (!input) return;

    input.addEventListener('input', function (e) {
        galleryState.searchQuery = e.target.value.trim();
        galleryState.currentPage = 1;
        filterImages();
    });
}

// ============================================================================
// Filtering
// ============================================================================

function filterImages() {
    var filtered = galleryState.images.slice();

    // Filter by scene
    if (galleryState.selectedScene) {
        var sceneImages = new Set(
            galleryState.index.images_by_scene[galleryState.selectedScene] || []
        );
        // Include child scenes
        var prefix = galleryState.selectedScene + '/';
        var byScene = galleryState.index.images_by_scene;
        Object.keys(byScene).forEach(function (path) {
            if (path.indexOf(prefix) === 0) {
                var ids = byScene[path];
                for (var i = 0; i < ids.length; i++) {
                    sceneImages.add(ids[i]);
                }
            }
        });
        filtered = filtered.filter(function (img) {
            return sceneImages.has(img.id);
        });
    }

    // Filter by dataset
    if (galleryState.datasetFilter !== 'all') {
        var dsFilter = galleryState.datasetFilter;
        filtered = filtered.filter(function (img) {
            return img.source === dsFilter;
        });
    }

    // Filter by search query (substring match on formatted_id)
    if (galleryState.searchQuery) {
        var query = galleryState.searchQuery.toLowerCase();
        filtered = filtered.filter(function (img) {
            var fid = (img.formatted_id || String(img.id)).toLowerCase();
            return fid.indexOf(query) !== -1;
        });
    }

    // Filter: all models must have >= 1 detection
    if (galleryState.requireAllModels) {
        var models = CONFIG.MODELS;
        filtered = filtered.filter(function (img) {
            var mc = img.matched_counts || {};
            for (var i = 0; i < models.length; i++) {
                if (!mc[models[i]] || mc[models[i]] <= 0) return false;
            }
            return true;
        });
    }

    galleryState.filteredImages = filtered;
    galleryState.currentPage = 1;
    updateStats();
    renderImageGrid();
}

// ============================================================================
// Grid Rendering
// ============================================================================

function renderImageGrid() {
    var grid = document.getElementById('image-grid');
    var pagination = document.getElementById('pagination');
    if (!grid) return;

    var images = galleryState.filteredImages;
    var perPage = CONFIG.IMAGES_PER_PAGE;
    var totalPages = Math.ceil(images.length / perPage);
    var start = (galleryState.currentPage - 1) * perPage;
    var end = start + perPage;
    var pageImages = images.slice(start, end);

    if (pageImages.length === 0) {
        grid.innerHTML = '<div class="loading">No images found</div>';
        if (pagination) pagination.innerHTML = '';
        return;
    }

    grid.innerHTML = pageImages.map(function (img) {
        var modelPills = buildModelPills(img.matched_counts || img.model_counts || {});
        var source = img.source || 'unknown';
        var formattedId = img.formatted_id || String(img.id);

        return (
            '<div class="image-card" onclick="openImage(' + img.id + ')">' +
                '<div class="card-thumb">' +
                    '<canvas id="thumb-' + img.id + '"></canvas>' +
                '</div>' +
                '<div class="card-info">' +
                    '<span class="card-id">' + formattedId + '</span>' +
                    '<span class="badge badge-' + source + '">' + source + '</span>' +
                '</div>' +
                '<div class="card-models">' + modelPills + '</div>' +
            '</div>'
        );
    }).join('');

    renderPagination(totalPages);
    renderThumbnailOverlays(pageImages);
}

// ============================================================================
// Thumbnail GT Overlay Rendering
// ============================================================================

var THUMB_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7]
];

function renderThumbnailOverlays(pageImages) {
    for (var i = 0; i < pageImages.length; i++) {
        renderOneThumb(pageImages[i]);
    }
}

function renderOneThumb(imgMeta) {
    var canvas = document.getElementById('thumb-' + imgMeta.id);
    if (!canvas) return;

    var container = canvas.parentElement;
    var img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = function () {
        // Size canvas to container
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;

        var ctx = canvas.getContext('2d');

        // Draw image centered (contain)
        var scale = Math.min(canvas.width / img.naturalWidth, canvas.height / img.naturalHeight);
        var drawW = img.naturalWidth * scale;
        var drawH = img.naturalHeight * scale;
        var offX = (canvas.width - drawW) / 2;
        var offY = (canvas.height - drawH) / 2;
        ctx.drawImage(img, offX, offY, drawW, drawH);

        // Fetch per-image data for GT boxes
        loadImageData(imgMeta.id).then(function (data) {
            if (!data.gt || data.gt.length === 0) return;

            var sx = drawW / img.naturalWidth;
            var sy = drawH / img.naturalHeight;

            ctx.strokeStyle = CONFIG.MODEL_COLORS.GT;
            ctx.lineWidth = 2;
            ctx.globalAlpha = 0.85;

            for (var g = 0; g < data.gt.length; g++) {
                var proj = data.gt[g].bbox3D_proj;
                if (!proj || proj.length !== 8) continue;

                for (var e = 0; e < THUMB_EDGES.length; e++) {
                    var p0 = proj[THUMB_EDGES[e][0]];
                    var p1 = proj[THUMB_EDGES[e][1]];
                    if (!p0 || !p1) continue;
                    ctx.beginPath();
                    ctx.moveTo(offX + p0[0] * sx, offY + p0[1] * sy);
                    ctx.lineTo(offX + p1[0] * sx, offY + p1[1] * sy);
                    ctx.stroke();
                }
            }
            ctx.globalAlpha = 1.0;
        });
    };
    img.src = getImageUrl(imgMeta.file_path);
}

function buildModelPills(modelCounts) {
    var html = '';
    var models = CONFIG.MODELS;
    for (var i = 0; i < models.length; i++) {
        var model = models[i];
        var count = modelCounts[model] || 0;
        var short = CONFIG.MODEL_SHORT[model];
        var color = CONFIG.MODEL_COLORS[model];
        html += '<span class="model-pill" style="background:' + color + '">' +
                short + ':' + count + '</span>';
    }
    return html;
}

// ============================================================================
// Pagination
// ============================================================================

function renderPagination(totalPages) {
    var pagination = document.getElementById('pagination');
    if (!pagination || totalPages <= 1) {
        if (pagination) pagination.innerHTML = '';
        return;
    }

    var currentPage = galleryState.currentPage;
    var pages = [];

    // Always show first page
    pages.push(1);

    // Show pages around current
    for (var i = Math.max(2, currentPage - 2); i <= Math.min(totalPages - 1, currentPage + 2); i++) {
        if (pages[pages.length - 1] !== i - 1) {
            pages.push('...');
        }
        pages.push(i);
    }

    // Always show last page
    if (totalPages > 1) {
        if (pages[pages.length - 1] !== totalPages - 1) {
            pages.push('...');
        }
        pages.push(totalPages);
    }

    var html =
        '<button class="pagination-btn" ' +
            (currentPage === 1 ? 'disabled' : '') +
            ' data-page="prev">&larr;</button>';

    for (var j = 0; j < pages.length; j++) {
        var p = pages[j];
        if (p === '...') {
            html += '<span class="pagination-btn" style="cursor:default">...</span>';
        } else {
            html += '<button class="pagination-btn ' +
                (p === currentPage ? 'active' : '') +
                '" data-page="' + p + '">' + p + '</button>';
        }
    }

    html += '<button class="pagination-btn" ' +
        (currentPage === totalPages ? 'disabled' : '') +
        ' data-page="next">&rarr;</button>';

    pagination.innerHTML = html;

    // Attach click handlers
    pagination.querySelectorAll('.pagination-btn[data-page]').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var page = btn.dataset.page;
            if (page === 'prev') {
                galleryState.currentPage = Math.max(1, currentPage - 1);
            } else if (page === 'next') {
                galleryState.currentPage = Math.min(totalPages, currentPage + 1);
            } else if (!isNaN(parseInt(page))) {
                galleryState.currentPage = parseInt(page);
            }
            renderImageGrid();
            var gridEl = document.getElementById('image-grid');
            if (gridEl) gridEl.scrollTo(0, 0);
        });
    });
}
