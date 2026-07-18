import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKRefinerPrepareSDXLPipe";

// Standard widgets are visual anchors in the expanded node. Advanced widgets
// intentionally retain their technical lower-case/snake_case names.
const STANDARD_LABELS = {
    ckpt_name: "CHECKPOINT",
    use_prompt_lora_from_sampler: "USE PROMPT / LORA FROM SAMPLER",
    lora_name: "LORA",
    prompt_pos: "PROMPT POS",
    prompt_neg: "PROMPT NEG",
    steps: "STEPS",
    cfg: "CFG",
    sampler: "SAMPLER",
    scheduler: "SCHEDULER",
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

    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "cmk.refiner.prepare.ui.v2",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalOnNodeCreated?.apply(this, arguments);
            setTimeout(() => configure(this), 0);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function() {
            const result = originalOnConfigure?.apply(this, arguments);
            setTimeout(() => configure(this), 0);
            return result;
        };
    },

    nodeCreated(node) {
        setTimeout(() => configure(node), 0);
    },
});
