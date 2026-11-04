import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKFaceSwapImagePipe";
const PROFILE_WIDGET = "QUALITY PROFILE";
const PROFILE_CONTROLLED_WIDGETS = new Set([
    "FACE ENHANCER",
    "BLEND",
    "crop_factor",
    "feather",
    "IDENTITY STRENGTH",
]);

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function captureWidgets(node) {
    node._cmkFaceSwapProfileUi ??= {
        widgetsByName: new Map(),
        canonicalOrder: [],
        visibleProfile: null,
        rebuilding: false,
    };
    const state = node._cmkFaceSwapProfileUi;
    for (const widget of node.widgets ?? []) {
        if (!widget?.name) continue;
        if (!state.widgetsByName.has(widget.name)) {
            state.canonicalOrder.push(widget.name);
        }
        state.widgetsByName.set(widget.name, widget);
    }
    return state;
}

function installStableSerialization(node) {
    if (node._cmkFaceSwapStableSerializationInstalled) return;
    const originalOnSerialize = node.onSerialize;
    node.onSerialize = function (data) {
        const result = originalOnSerialize?.apply(this, arguments);
        const state = this._cmkFaceSwapProfileUi;
        const widgets = state
            ? state.canonicalOrder.map((name) => state.widgetsByName.get(name)).filter(Boolean)
            : this.widgets;
        if (this.serialize_widgets && Array.isArray(widgets)) {
            data.widgets_values = widgets
                .filter((widget) => widget?.serialize !== false)
                .map((widget) => {
                    const value = widget?.value;
                    if (value == null || typeof value !== "object") return value ?? null;
                    try { return JSON.parse(JSON.stringify(value)); } catch (_) { return value; }
                });
        }
        return result;
    };
    node._cmkFaceSwapStableSerializationInstalled = true;
}

function refresh(node, force = false) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    const state = captureWidgets(node);
    if (state.rebuilding) return;
    const profile = state.widgetsByName.get(PROFILE_WIDGET);
    if (!profile) return;
    const currentProfile = String(profile.value ?? "Automatic");
    if (!force && currentProfile === state.visibleProfile) return;

    state.rebuilding = true;
    try {
        const custom = currentProfile === "Custom";
        node.widgets = state.canonicalOrder
            .filter((name) => custom || !PROFILE_CONTROLLED_WIDGETS.has(name))
            .map((name) => state.widgetsByName.get(name))
            .filter(Boolean);
        state.visibleProfile = currentProfile;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    } finally {
        state.rebuilding = false;
    }
}

function install(node) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    installStableSerialization(node);
    const state = captureWidgets(node);
    const profile = state.widgetsByName.get(PROFILE_WIDGET);
    if (!profile) return;

    if (!profile._cmkProfileCallbackInstalled) {
        const originalCallback = profile.callback;
        profile.callback = function () {
            const result = originalCallback?.apply(this, arguments);
            queueMicrotask(() => refresh(node));
            return result;
        };
        profile._cmkProfileCallbackInstalled = true;
    }

    // Promoted subgraph widgets change the inner value without invoking its
    // callback. The watcher only rebuilds when that value actually changes.
    if (!node._cmkProfileValueWatcher) {
        node._cmkProfileValueWatcher = setInterval(() => refresh(node), 150);
        const originalOnRemoved = node.onRemoved;
        node.onRemoved = function () {
            clearInterval(this._cmkProfileValueWatcher);
            this._cmkProfileValueWatcher = null;
            return originalOnRemoved?.apply(this, arguments);
        };
    }
    refresh(node, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => install(node), delay);
}

app.registerExtension({
    name: "cmk.faceswap.quality_profile_ui.v4",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },

    nodeCreated(node) {
        if (isTarget(node)) schedule(node);
    },

    loadedGraphNode(node) {
        if (isTarget(node)) schedule(node);
    },
});
