import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKDetailerPreparePipe";

const STANDARD_LABELS = {
    sam_model_name: "SAM MODEL",
    sam_device_mode: "SAM DEVICE",
    detailer_global_enable: "DETAILER GLOBAL ENABLE",
    use_prompt_lora_from_sampler: "USE PROMPT FROM 1ST PASS",
    use_lora_from_1st_pass: "USE LORA FROM 1ST PASS",
    use_1st_pass_sampling: "USE SAMPLING FROM 1ST PASS",
    lora_name: "LORA",
    prompt_pos: "PROMPT POS",
    prompt_neg: "PROMPT NEG",
    steps: "STEPS",
    cfg: "CFG",
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

function migrateLegacyWidgetValues(data) {
    const values = data?.widgets_values;
    if (!Array.isArray(values)) return data;

    // Legacy order had one combined prompt/LoRA switch at index 3. Preserve
    // its value as prompt inheritance, keep LoRA local, and make the newly
    // introduced sampling inheritance active by default.
    if (
        values.length >= 22
        && typeof values[2] === "boolean"
        && typeof values[3] === "boolean"
        && typeof values[4] === "string"
        && typeof values[7] === "string"
        && typeof values[8] === "string"
    ) {
        const migrated = [...values];
        migrated.splice(4, 0, false, true);
        return { ...data, widgets_values: migrated };
    }
    return data;
}

app.registerExtension({
    name: "cmk.detailer.prepare.ui.v2",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        for (const hook of ["onNodeCreated", "onAdded"]) {
            const original = nodeType.prototype[hook];
            nodeType.prototype[hook] = function () {
                const result = original?.apply(this, arguments);
                schedule(this);
                return result;
            };
        }


        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function(data) {
            const result = originalOnConfigure?.call(
                this,
                migrateLegacyWidgetValues(data)
            );
            schedule(this);
            return result;
        };
    },

    nodeCreated(node) {
        if (isTarget(node)) schedule(node);
    },

    loadedGraphNode(node) {
        if (isTarget(node)) schedule(node);
    },
});
