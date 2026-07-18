import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMK_SmartDetailerPipe";

// Only the standard UI layer is translated into concise uppercase labels.
// Advanced technical widget names intentionally remain unchanged.
const STANDARD_LABELS = {
    enable: "ENABLE",
    output_image_proceed: "OUTPUT IMAGE PROCEED",
    model_name: "DETECT MODEL",
    bbox_threshold: "BBOX THRESHOLD",
    crop_factor: "CROP FACTOR",
    guide_size: "GUIDE SIZE",
    denoise: "DENOISE",
};

function isTarget(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function configure(node) {
    if (!isTarget(node) || !Array.isArray(node.widgets)) return;
    for (const widget of node.widgets) {
        const label = STANDARD_LABELS[widget?.name];
        if (label) widget.label = label;
    }
    node.widgets = [...node.widgets];
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => configure(node), delay);
}

app.registerExtension({
    name: "cmk.smart.detailer.labels.v1",
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
    nodeCreated(node) { if (isTarget(node)) schedule(node); },
    loadedGraphNode(node) { if (isTarget(node)) schedule(node); },
});
