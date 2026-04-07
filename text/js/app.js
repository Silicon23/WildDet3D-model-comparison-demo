/**
 * Shared Configuration and Routing for 3D Detection Comparison App
 *
 * Provides CONFIG, data loading utilities, and image URL helpers
 * used by both gallery.js (index page) and detail.js (image page).
 */

// ============================================================================
// Configuration
// ============================================================================

// Auto-detect local vs remote hosting: serve local data when running on
// localhost, file://, or through a cloudflared quick tunnel (which just
// forwards a local HTTP server). Otherwise use the HuggingFace dataset.
// NOTE: When hosted under a /text/ subfolder on GitHub Pages, local
// paths use ../ to reach the shared data/ and images/ at the repo root.
var _isLocal = window.location.hostname === 'localhost' ||
               window.location.hostname === '127.0.0.1' ||
               window.location.hostname === '' ||
               window.location.hostname.endsWith('.trycloudflare.com') ||
               window.location.protocol === 'file:';

const CONFIG = {
    DATA_BASE: _isLocal
        ? '../data/text'
        : 'https://huggingface.co/datasets/allenai/WildDet3D-visualization-source/resolve/main/model/text',
    IMAGE_BASE: _isLocal
        ? '../'
        : 'https://huggingface.co/datasets/allenai/WildDet3D-visualization-source/resolve/main/model/',
    IMAGES_PER_PAGE: 24,
    // Text-based branch: only text-prompted 3D detectors are compared.
    // DetAny3D and OVMono3D were excluded because they use 2D box prompts.
    MODELS: ['SAM3_3D', 'GDino3D'],
    MODEL_DISPLAY: {
        SAM3_3D: 'WildDet3D (Ours)',
        GDino3D: '3D-MOOD',
    },
    MODEL_COLORS: {
        SAM3_3D: '#e74c3c',
        GDino3D: '#3b82f6',
        GT: '#a855f7',
    },
    MODEL_SHORT: {
        SAM3_3D: 'WildDet3D',
        GDino3D: '3D-MOOD',
    },
    SOURCE_COLORS: {
        coco_val: '#3b82f6',
        coco_train: '#a855f7',
        obj365_val: '#f97316',
    },
    SOURCE_LABELS: {
        coco_val: 'COCO Val',
        coco_train: 'COCO Train',
        obj365_val: 'Obj365 Val',
    },
};

// ============================================================================
// State (shared between pages)
// ============================================================================

let indexData = null;

// ============================================================================
// Data Loading
// ============================================================================

/**
 * Load index.json and cache it. Returns the parsed JSON object.
 */
async function loadIndex() {
    if (indexData) return indexData;
    const resp = await fetch(CONFIG.DATA_BASE + '/index.json');
    indexData = await resp.json();
    return indexData;
}

/**
 * Load per-image JSON data for the detail page.
 * @param {string} imageId - The image identifier.
 * @returns {Promise<Object>} Parsed image data.
 */
async function loadImageData(imageId) {
    const resp = await fetch(CONFIG.DATA_BASE + '/images/' + imageId + '.json');
    return resp.json();
}

// ============================================================================
// URL Helpers
// ============================================================================

/**
 * Get the URL for an image given its file_path field.
 * @param {string} filePath - Relative path from the images directory.
 * @returns {string} Full URL to the image.
 */
function getImageUrl(filePath) {
    return CONFIG.IMAGE_BASE + filePath;
}

/**
 * Navigate to the detail page for a specific image.
 * @param {string} imageId - The image identifier.
 */
function openImage(imageId) {
    window.location.href = 'image.html?id=' + imageId;
}

/**
 * Get a URL query parameter by name.
 * @param {string} name - Parameter name.
 * @returns {string|null} Parameter value or null.
 */
function getQueryParam(name) {
    var params = new URLSearchParams(window.location.search);
    return params.get(name);
}
