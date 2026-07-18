import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMK_SmartUpscalerPipe";
const LABELS = {
    enable: "ENABLE",
    limit_4x_mp: "LIMIT 4X MP",
    limit_2x_mp: "LIMIT 2X MP",
    model_4x: "MODEL 4X",
    model_2x: "MODEL 2X",
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
        const label = LABELS[widget?.name];
        if (label) widget.label = label;
    }
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function schedule(node) {
    for (const delay of [0, 50, 200]) setTimeout(() => configure(node), delay);
}

app.registerExtension({
    name: "cmk.smart.upscaler.pipe.labels.v1",
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
