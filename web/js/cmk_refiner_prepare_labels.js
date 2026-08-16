import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKRefinerPrepareSDXLPipe";

// Standard widgets are visual anchors in the expanded node. Advanced widgets
// intentionally retain their technical lower-case/snake_case names.
const STANDARD_LABELS = {
    ckpt_name: "CHECKPOINT",
    use_prompt_lora_from_sampler: "USE PROMPT FROM 1ST PASS",
    use_lora_from_1st_pass: "USE LORA FROM 1ST PASS",
    lora_name: "LORA",
    steps: "STEPS",
    cfg: "CFG",
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

    for (const input of node.inputs ?? []) {
        if (input?.name === "prompt_pos_input") {
            input.label = "PROMPT POS";
            input.localized_name = "PROMPT POS";
        } else if (input?.name === "prompt_neg_input") {
            input.label = "PROMPT NEG";
            input.localized_name = "PROMPT NEG";
        }
    }

    node.widgets = [...node.widgets];
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function migrateLegacyWidgetValues(data) {
    const values = data?.widgets_values;
    if (!Array.isArray(values)) return data;
    const migrated = [...values];

    // Legacy order contained two multiline prompt widgets at positions 4/5.
    if (
        typeof migrated[4] === "string"
        && typeof migrated[5] === "string"
        && typeof migrated[6] === "number"
    ) {
        migrated.splice(4, 2);

        // The first split-version stored USE LORA at the very end. Older
        // workflows have no separate value and receive the safe OFF default.
        const hasSplitLoraValue = (
            typeof migrated.at(-1) === "boolean"
            && typeof migrated.at(-2) === "boolean"
        );
        const loraFromFirstPass = hasSplitLoraValue ? migrated.pop() : false;
        migrated.splice(1, 0, loraFromFirstPass);
        return { ...data, widgets_values: migrated };
    }
    return data;
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
        nodeType.prototype.onConfigure = function(data) {
            const migrated = migrateLegacyWidgetValues(data);
            const result = originalOnConfigure?.call(this, migrated);
            setTimeout(() => configure(this), 0);
            return result;
        };
    },

    nodeCreated(node) {
        setTimeout(() => configure(node), 0);
    },
});
