/**
 * Scene Tree Component
 *
 * Renders a collapsible hierarchical tree of scene categories.
 * Used in the sidebar for dataset/scene navigation.
 */

class SceneTree {
    constructor(containerId, onSelect) {
        this.container = document.getElementById(containerId);
        this.onSelect = onSelect;
        this.selectedPath = null;
        this.expandedPaths = new Set();
    }

    render(treeData) {
        if (!this.container || !treeData) return;

        this.container.innerHTML = '';

        // Add root-level "All Images" option
        var allNode = this.createAllNode(treeData.image_count || 0);
        this.container.appendChild(allNode);

        // Render tree children
        if (treeData.children && treeData.children.length > 0) {
            for (var i = 0; i < treeData.children.length; i++) {
                var nodeEl = this.renderNode(treeData.children[i], 0);
                this.container.appendChild(nodeEl);
            }
        }
    }

    createAllNode(imageCount) {
        var node = document.createElement('div');
        node.className = 'tree-node';

        var header = document.createElement('div');
        header.className = 'tree-node-header' + (this.selectedPath === null ? ' selected' : '');
        header.innerHTML =
            '<span class="tree-toggle empty">' +
                '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
                    '<polyline points="9 18 15 12 9 6"></polyline>' +
                '</svg>' +
            '</span>' +
            '<span class="tree-node-name">All Images</span>' +
            '<span class="tree-node-count">' + imageCount + '</span>';

        var self = this;
        header.addEventListener('click', function () {
            self.selectPath(null);
        });

        node.appendChild(header);
        return node;
    }

    renderNode(nodeData, depth) {
        var node = document.createElement('div');
        node.className = 'tree-node';
        node.dataset.path = nodeData.path;

        var hasChildren = nodeData.children && nodeData.children.length > 0;
        var isExpanded = this.expandedPaths.has(nodeData.path);
        var isSelected = this.selectedPath === nodeData.path;

        // Header
        var header = document.createElement('div');
        header.className = 'tree-node-header' + (isSelected ? ' selected' : '');

        var toggleClass = hasChildren ? (isExpanded ? 'expanded' : '') : 'empty';

        header.innerHTML =
            '<span class="tree-toggle ' + toggleClass + '">' +
                '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">' +
                    '<polyline points="9 18 15 12 9 6"></polyline>' +
                '</svg>' +
            '</span>' +
            '<span class="tree-node-name">' + nodeData.name + '</span>' +
            '<span class="tree-node-count">' + nodeData.image_count + '</span>';

        // Click handler
        var self = this;
        header.addEventListener('click', function (e) {
            if (hasChildren && e.target.closest('.tree-toggle')) {
                self.toggleExpand(nodeData.path);
            } else {
                self.selectPath(nodeData.path);
            }
        });

        node.appendChild(header);

        // Children container
        if (hasChildren) {
            var children = document.createElement('div');
            children.className = 'tree-children' + (isExpanded ? ' expanded' : '');

            for (var i = 0; i < nodeData.children.length; i++) {
                var childNode = this.renderNode(nodeData.children[i], depth + 1);
                children.appendChild(childNode);
            }

            node.appendChild(children);
        }

        return node;
    }

    toggleExpand(path) {
        if (this.expandedPaths.has(path)) {
            this.expandedPaths.delete(path);
        } else {
            this.expandedPaths.add(path);
        }

        // Update DOM
        var node = this.container.querySelector('[data-path="' + path + '"]');
        if (node) {
            var toggle = node.querySelector(':scope > .tree-node-header .tree-toggle');
            var children = node.querySelector(':scope > .tree-children');

            if (this.expandedPaths.has(path)) {
                if (toggle) toggle.classList.add('expanded');
                if (children) children.classList.add('expanded');
            } else {
                if (toggle) toggle.classList.remove('expanded');
                if (children) children.classList.remove('expanded');
            }
        }
    }

    selectPath(path) {
        // Update selection state
        var previousSelected = this.container.querySelector('.tree-node-header.selected');
        if (previousSelected) {
            previousSelected.classList.remove('selected');
        }

        this.selectedPath = path;

        if (path === null) {
            // Select "All Images"
            var allHeader = this.container.querySelector('.tree-node:first-child .tree-node-header');
            if (allHeader) {
                allHeader.classList.add('selected');
            }
        } else {
            var node = this.container.querySelector('[data-path="' + path + '"]');
            if (node) {
                var header = node.querySelector(':scope > .tree-node-header');
                if (header) header.classList.add('selected');
            }
        }

        // Notify callback
        if (this.onSelect) {
            this.onSelect(path);
        }
    }

    collapseAll() {
        this.expandedPaths.clear();

        // Update DOM
        var toggles = this.container.querySelectorAll('.tree-toggle.expanded');
        var children = this.container.querySelectorAll('.tree-children.expanded');

        toggles.forEach(function (t) { t.classList.remove('expanded'); });
        children.forEach(function (c) { c.classList.remove('expanded'); });
    }

    expandPath(path) {
        if (!path) return;

        var parts = path.split('/');
        var currentPath = '';

        for (var i = 0; i < parts.length; i++) {
            currentPath = currentPath ? currentPath + '/' + parts[i] : parts[i];
            this.expandedPaths.add(currentPath);
        }

        // Note: caller should re-render or manually update DOM after expanding
    }
}

// Export for use in other modules
window.SceneTree = SceneTree;
