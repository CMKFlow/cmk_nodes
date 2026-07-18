import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKFaceProcessPipe";
const WATCH_INTERVAL_MS = 100;

const STANDARD_LABELS = {
    enable: "ENABLE",
    process_mode: "PROCESS MODE",
    select_face: "SELECT FACE",
    refine_mode: "REFINE MODE",
    detect_model: "DETECT MODEL",
    restore_model: "RESTORE MODEL",
    restore_facedetection: "FACE DETECTION",
    restore_visibility: "VISIBILITY",
    detail_guide_size: "GUIDE SIZE",
    detail_denoise: "DENOISE",
};

const ADVANCED_WIDGETS = new Set([
    "detect_bbox_threshold",
    "detect_bbox_dilation",
    "detect_crop_factor",
    "detect_drop_size",
    "restore_codeformer_weight",
    "detail_guide_size_for",
    "detail_max_size",
    "detail_noise_mask",
    "detail_force_inpaint",
    "detail_paste_feather",
]);

const COMMON_WIDGETS = new Set([
    "enable",
    "process_mode",
    "select_face",
    "refine_mode",
    "detect_model",
    "detect_bbox_threshold",
    "detect_bbox_dilation",
    "detect_crop_factor",
    "detect_drop_size",
]);

const RESTORE_WIDGETS = new Set([
    "restore_model",
    "restore_facedetection",
    "restore_visibility",
    "restore_codeformer_weight",
]);

const DETAILER_WIDGETS = new Set([
    "detail_guide_size",
    "detail_guide_size_for",
    "detail_max_size",
    "detail_denoise",
    "detail_noise_mask",
    "detail_force_inpaint",
    "detail_paste_feather",
]);

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function applyMetadata(widget) {
    if (!widget?.name) return;

    const advanced = ADVANCED_WIDGETS.has(widget.name);
    widget.advanced = advanced;
    widget.options ??= {};
    widget.options.advanced = advanced;

    const label = STANDARD_LABELS[widget.name];
    if (label) widget.label = label;
}

function ensureState(node) {
    node._cmkFaceUiV6 ??= {
        widgetsByName: new Map(),
        canonicalOrder: [],
        visibleMode: null,
        configured: false,
        rebuilding: false,
        watcher: null,
    };
    return node._cmkFaceUiV6;
}

function captureWidgets(node) {
    const state = ensureState(node);

    for (const widget of node.widgets ?? []) {
        if (!widget?.name) continue;
        applyMetadata(widget);

        if (!state.widgetsByName.has(widget.name)) {
            state.canonicalOrder.push(widget.name);
        }
        state.widgetsByName.set(widget.name, widget);
    }

    return state;
}

function fullWidgetList(node) {
    const state = captureWidgets(node);
    return state.canonicalOrder
        .map((name) => state.widgetsByName.get(name))
        .filter(Boolean);
}

function restoreFullWidgetList(node) {
    const full = fullWidgetList(node);
    node.widgets = full;
    return full;
}

function getWidget(node, name) {
    return captureWidgets(node).widgetsByName.get(name) ?? null;
}

function currentMode(node) {
    const value = String(getWidget(node, "process_mode")?.value ?? "restore")
        .trim()
        .toLowerCase();
    return value === "detailer" ? "detailer" : "restore";
}

function widgetBelongsToMode(name, mode) {
    if (COMMON_WIDGETS.has(name)) return true;
    if (mode === "detailer") return DETAILER_WIDGETS.has(name);
    return RESTORE_WIDGETS.has(name);
}

function resizeAfterRebuild(node) {
    try {
        const computed = node.computeSize?.();
        if (Array.isArray(computed)) {
            const width = Number(node.size?.[0]) || Number(computed[0]) || 320;
            const height = Math.max(Number(computed[1]) || 120, 120);
            node.setSize?.([width, height]);
        }
    } catch (_) {}

    node.setDirtyCanvas?.(true, true);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function rebuildVisibleWidgets(node, force = false) {
    if (!isTarget(node)) return;

    const state = captureWidgets(node);
    if (state.rebuilding) return;

    const mode = currentMode(node);
    if (!force && state.visibleMode === mode) return;

    state.rebuilding = true;
    try {
        const visible = state.canonicalOrder
            .filter((name) => widgetBelongsToMode(name, mode))
            .map((name) => state.widgetsByName.get(name))
            .filter(Boolean);

        // This is intentionally a real list projection, not a hidden flag.
        // It is respected by both LiteGraph and Vue Nodes 2.0. The complete
        // canonical list remains alive in state for switching and serialization.
        node.widgets = visible;
        state.visibleMode = mode;
        resizeAfterRebuild(node);
    } finally {
        state.rebuilding = false;
    }
}

function queueRebuild(node, force = false) {
    if (!isTarget(node)) return;

    const state = ensureState(node);
    state.pendingForce = Boolean(state.pendingForce || force);
    if (state.queued) return;

    state.queued = true;
    const run = () => {
        const doForce = Boolean(state.pendingForce);
        state.queued = false;
        state.pendingForce = false;
        rebuildVisibleWidgets(node, doForce);
    };

    if (typeof requestAnimationFrame === "function") {
        requestAnimationFrame(run);
    } else {
        setTimeout(run, 0);
    }
}

function installModeCallback(node) {
    const widget = getWidget(node, "process_mode");
    if (!widget || widget._cmkFaceV6CallbackInstalled) return;

    const original = widget.callback;
    widget.callback = function () {
        const result = original?.apply(this, arguments);
        queueRebuild(node, true);
        return result;
    };
    widget._cmkFaceV6CallbackInstalled = true;
}

function startWatcher(node) {
    const state = ensureState(node);
    if (state.watcher != null || typeof window === "undefined") return;

    state.watcher = window.setInterval(() => {
        if (!isTarget(node)) return;
        captureWidgets(node);
        installModeCallback(node);
        if (currentMode(node) !== state.visibleMode) {
            rebuildVisibleWidgets(node, true);
        }
    }, WATCH_INTERVAL_MS);
}

function stopWatcher(node) {
    const state = node?._cmkFaceUiV6;
    if (!state || state.watcher == null || typeof window === "undefined") return;
    window.clearInterval(state.watcher);
    state.watcher = null;
}

function serializableValues(node) {
    return fullWidgetList(node)
        .filter((widget) => widget?.serialize !== false)
        .map((widget) => widget?.value);
}

function initialise(node, { allowProjection = true } = {}) {
    if (!isTarget(node)) return;

    captureWidgets(node);
    installModeCallback(node);
    startWatcher(node);

    if (allowProjection) {
        queueRebuild(node, true);
        for (const delay of [30, 150, 500]) {
            setTimeout(() => queueRebuild(node, true), delay);
        }
    }
}

app.registerExtension({
    name: "cmk.faceprocess.pipe.dynamic_ui.v6",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            const configuring = Boolean(app.configuringGraph || window.app?.configuringGraph);
            initialise(this, { allowProjection: !configuring });
            return result;
        };

        const originalConfigure = nodeType.prototype.configure;
        nodeType.prototype.configure = function () {
            // LGraphNode.configure() assigns widgets_values before calling
            // onConfigure(). Therefore the complete canonical list must be
            // restored before the real configure method starts.
            captureWidgets(this);
            restoreFullWidgetList(this);
            const result = originalConfigure?.apply(this, arguments);
            captureWidgets(this);
            ensureState(this).configured = true;
            initialise(this, { allowProjection: true });
            return result;
        };

        const originalAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = originalAdded?.apply(this, arguments);
            const configuring = Boolean(app.configuringGraph || window.app?.configuringGraph);
            initialise(this, { allowProjection: !configuring });
            return result;
        };

        const originalWidgetChanged = nodeType.prototype.onWidgetChanged;
        nodeType.prototype.onWidgetChanged = function (name) {
            const result = originalWidgetChanged?.apply(this, arguments);
            if (name === "process_mode") queueRebuild(this, true);
            return result;
        };

        const originalSerialize = nodeType.prototype.onSerialize;
        nodeType.prototype.onSerialize = function (data) {
            const result = originalSerialize?.apply(this, arguments);
            if (data && typeof data === "object") {
                data.widgets_values = serializableValues(this);
            }
            return result;
        };

        const originalRemoved = nodeType.prototype.onRemoved;
        nodeType.prototype.onRemoved = function () {
            stopWatcher(this);
            return originalRemoved?.apply(this, arguments);
        };
    },

    nodeCreated(node) {
        if (!isTarget(node)) return;
        const configuring = Boolean(app.configuringGraph || window.app?.configuringGraph);
        initialise(node, { allowProjection: !configuring });
    },

    loadedGraphNode(node) {
        if (!isTarget(node)) return;
        captureWidgets(node);
        initialise(node, { allowProjection: true });
    },
});
