import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKImageResize";
const METHOD_WIDGET = "scale_method";
const METHOD_PARAMETERS = {
    "Scale to Max Dimension": new Set(["largest_size"]),
    "Scale to Megapixels": new Set(["megapixels"]),
    "Resize and Pad": new Set(["target_width", "target_height", "padding_color"]),
};
const CONDITIONAL = new Set(["largest_size", "megapixels", "target_width", "target_height", "padding_color"]);

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS || node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function rememberWidgets(node) {
    node._cmkImageResizeUi ??= {
        byName: new Map(),
        order: [],
        visibleMethod: null,
        rebuilding: false,
    };
    const state = node._cmkImageResizeUi;
    for (const widget of node.widgets ?? []) {
        if (!widget?.name) continue;
        if (!state.byName.has(widget.name)) state.order.push(widget.name);
        state.byName.set(widget.name, widget);
    }
    return state;
}

function installStableSerialization(node) {
    if (node._cmkImageResizeSerialization) return;
    const original = node.onSerialize;
    node.onSerialize = function(data) {
        const result = original?.apply(this, arguments);
        const state = this._cmkImageResizeUi;
        if (this.serialize_widgets && state) {
            data.widgets_values = state.order
                .map((name) => state.byName.get(name))
                .filter((widget) => widget?.serialize !== false)
                .map((widget) => widget?.value ?? null);
        }
        return result;
    };
    node._cmkImageResizeSerialization = true;
}

function moveModelInputFirst(node) {
    if (!Array.isArray(node.inputs)) return;
    const index = node.inputs.findIndex((input) => input?.name === "MODEL (opt)");
    if (index > 0) {
        const [modelInput] = node.inputs.splice(index, 1);
        node.inputs.unshift(modelInput);
    }
}

function refresh(node, force = false) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    const state = rememberWidgets(node);
    if (state.rebuilding) return;
    const selector = state.byName.get(METHOD_WIDGET);
    if (!selector) return;
    const method = String(selector.value ?? "Scale to Max Dimension");
    if (!force && method === state.visibleMethod) return;

    state.rebuilding = true;
    try {
        const active = METHOD_PARAMETERS[method] ?? METHOD_PARAMETERS["Scale to Max Dimension"];
        node.widgets = state.order
            .filter((name) => !CONDITIONAL.has(name) || active.has(name))
            .map((name) => state.byName.get(name))
            .filter(Boolean);
        state.visibleMethod = method;
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
    } finally {
        state.rebuilding = false;
    }
}

function install(node) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    moveModelInputFirst(node);
    installStableSerialization(node);
    const state = rememberWidgets(node);
    const selector = state.byName.get(METHOD_WIDGET);
    if (!selector) return;
    if (!selector._cmkImageResizeCallback) {
        const original = selector.callback;
        selector.callback = function() {
            const result = original?.apply(this, arguments);
            queueMicrotask(() => refresh(node));
            return result;
        };
        selector._cmkImageResizeCallback = true;
    }
    refresh(node, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => install(node), delay);
}

app.registerExtension({
    name: "cmk.image_resize.ui.v1",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_CLASS) return;
        for (const hook of ["onNodeCreated", "onConfigure", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function() {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }
    },
    nodeCreated(node) { if (isTarget(node)) schedule(node); },
    loadedGraphNode(node) { if (isTarget(node)) schedule(node); },
});
