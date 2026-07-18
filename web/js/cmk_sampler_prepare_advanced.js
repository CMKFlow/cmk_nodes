import { app } from "../../../scripts/app.js";

const NODE_CLASS = "CMKSamplerPrepareSDXLPipe";

const ADVANCED_WIDGETS = new Set([
    "stop_at_clip_layer",
    "fooocus_patch",
    "fooocus_head",
    "strength_model",
    "strength_clip",
    "freeu_b1",
    "freeu_b2",
    "freeu_s1",
    "freeu_s2",
    "sampling",
    "zsnr",
    "pag_scale",
    "context_reference_expand",
    "context_reference_blur",
]);

const STANDARD_LABELS = {
    lora_name: "LORA",
    freeu_enabled: "USE FREEU",
    steps_1st_pass: "STEPS",
    cfg: "CFG",
    sampler: "SAMPLER",
    scheduler: "SCHEDULER",
    seed: "SEED",
    inpaint_noise_mask: "USE INPAINT NOISE MASK",
    context_reference_enabled: "USE CONTEXT REFERENCE LATENT MASK",
};

function isTargetNode(node) {
    return Boolean(node) && (
        node.comfyClass === NODE_CLASS ||
        node.type === NODE_CLASS ||
        node.constructor?.comfyClass === NODE_CLASS ||
        node.constructor?.nodeData?.name === NODE_CLASS
    );
}

function configureWidgets(node) {
    if (!isTargetNode(node) || !Array.isArray(node.widgets)) return;

    for (const widget of node.widgets) {
        if (!widget) continue;

        // Use ComfyUI's native advanced-input mechanism. The frontend itself
        // controls visibility, footer button, state and layout.
        widget.advanced = ADVANCED_WIDGETS.has(widget.name);

        const label = STANDARD_LABELS[widget.name];
        if (label) widget.label = label;
    }

    // Nodes 2.0 derives its widget view from this collection. Replacing the
    // array reference forces the Vue renderer to consume the updated metadata.
    node.widgets = [...node.widgets];
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}

function scheduleConfigure(node) {
    for (const delay of [0, 50, 200]) {
        setTimeout(() => configureWidgets(node), delay);
    }
}

app.registerExtension({
    name: "cmk.sampler_prepare.native_advanced_inputs.v3",

    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_CLASS) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            scheduleConfigure(this);
            return result;
        };

        const originalOnConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalOnConfigure?.apply(this, arguments);
            scheduleConfigure(this);
            return result;
        };

        const originalOnAdded = nodeType.prototype.onAdded;
        nodeType.prototype.onAdded = function () {
            const result = originalOnAdded?.apply(this, arguments);
            scheduleConfigure(this);
            return result;
        };
    },

    nodeCreated(node) {
        if (isTargetNode(node)) scheduleConfigure(node);
    },

    loadedGraphNode(node) {
        if (isTargetNode(node)) scheduleConfigure(node);
    },
});
