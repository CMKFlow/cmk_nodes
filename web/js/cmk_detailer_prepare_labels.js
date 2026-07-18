import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKDetailerPreparePipe";

const STANDARD_LABELS = {
    sam_model_name: "SAM MODEL",
    sam_device_mode: "SAM DEVICE",
    detailer_global_enable: "DETAILER GLOBAL ENABLE",
    use_prompt_lora_from_sampler: "USE PROMPT / LORA FROM SAMPLER",
    lora_name: "LORA",
    prompt_pos: "PROMPT POS",
    prompt_neg: "PROMPT NEG",
    steps: "STEPS",
    cfg: "CFG",
    sampler: "SAMPLER",
    scheduler: "SCHEDULER",
    freeu_enabled: "USE FREEU",
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
    name: "cmk.detailer.prepare.labels.v1",

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
